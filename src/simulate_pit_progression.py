#!/usr/bin/env python3
"""Continuous pit-pattern progression models (I -> IIIL -> IV_B -> IV_V).

Two finite-wavelength candidates share one CLI, one FFT grid, and one initial
noise.  External control is a single progression parameter p in [0, 1]; internal
coefficients follow fixed smooth maps theta(p).

Models
------
sh_inhibitor
    Swift-Hohenberg + slow local inhibitor H (primary candidate).
sh_memory
    Swift-Hohenberg + elongation memory field E (conserved-history variant).

Example:
    uv run python src/simulate_pit_progression.py sweep --model sh_inhibitor \\
        --output artifacts/pit_progression_sweep
    uv run python src/simulate_pit_progression.py ramp --model sh_inhibitor \\
        --output artifacts/pit_progression_ramp
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from evaluate_pit_steps import annotate, normalize_u8, write_metrics
from pit_goal_metrics import (
    classify_morphology,
    measure_goal_pattern,
    measure_pattern_timeseries,
    measure_temporal_stability,
)


class ProgressionModel(str, Enum):
    SH_INHIBITOR = "sh_inhibitor"
    SH_MEMORY = "sh_memory"


RunMode = Literal["sweep", "ramp", "reverse_ramp"]


@dataclass(frozen=True)
class ProgressionSettings:
    size: int = 128
    steps: int = 3000
    dt: float = 0.35
    seed: int = 7
    wavelength: float = 12.0
    snapshot_count: int = 6


@dataclass(frozen=True)
class InternalCoeffs:
    """Coefficients derived from progression p via fixed smooth maps."""

    p: float
    growth: float
    quadratic: float
    inhibitor_strength: float
    inhibitor_source: float
    inhibitor_decay: float
    memory_strength: float
    memory_source: float
    memory_decay: float
    anisotropy: float


def _sigmoid(x: float, center: float, width: float) -> float:
    width = max(width, 1.0e-6)
    return 1.0 / (1.0 + math.exp(-(x - center) / width))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def coeffs_from_progression(p: float, model: ProgressionModel) -> InternalCoeffs:
    """Map external p to internal coefficients (monotone, no per-Type tuning)."""
    p = float(np.clip(p, 0.0, 1.0))

    # Spot (high g) -> stripe/labyrinth (low g) as p increases (delayed transition).
    quadratic = _lerp(1.05, 0.0, _sigmoid(p, 0.42, 0.16))

    # Stay above buckling threshold at low p; moderate growth through branching band.
    growth = _lerp(0.26, 0.13, _sigmoid(p, 0.52, 0.20))

    # Inhibitor peaks in IV_B band, eases toward IV_V.
    inh_band = _sigmoid(p, 0.48, 0.12) * (1.0 - _sigmoid(p, 0.82, 0.10))
    inhibitor_strength = 0.40 * inh_band
    inhibitor_source = 0.28 * _sigmoid(p, 0.38, 0.14)
    inhibitor_decay = 0.10 + 0.06 * _sigmoid(p, 0.65, 0.12)

    # Memory field: builds with elongation; stronger at high p for sh_memory.
    mem_rise = _sigmoid(p, 0.55, 0.14)
    memory_strength = 0.35 * mem_rise if model == ProgressionModel.SH_MEMORY else 0.0
    memory_source = 0.24 * mem_rise
    memory_decay = 0.08 + 0.05 * _sigmoid(p, 0.72, 0.12)

    # IV_V is the isotropic labyrinth / gyrus phase — no directional bias.
    # (The former anisotropy term produced parallel stripes = false IV_V.)
    anisotropy = 0.0

    return InternalCoeffs(
        p=p,
        growth=growth,
        quadratic=quadratic,
        inhibitor_strength=inhibitor_strength,
        inhibitor_source=inhibitor_source,
        inhibitor_decay=inhibitor_decay,
        memory_strength=memory_strength,
        memory_source=memory_source,
        memory_decay=memory_decay,
        anisotropy=anisotropy,
    )


def smooth_unit_noise(
    rng: np.random.Generator, shape: tuple[int, int], sigma: float
) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    smooth -= float(smooth.mean())
    std = float(smooth.std())
    if std > 1.0e-8:
        smooth /= std
    return smooth.astype(np.float32)


def wavenumber_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.fft.fftfreq(size, d=1.0) * (2.0 * np.pi)
    kx = np.broadcast_to(freqs[None, :], (size, size)).astype(np.float32)
    ky = np.broadcast_to(freqs[:, None], (size, size)).astype(np.float32)
    k2 = (kx * kx + ky * ky).astype(np.float32)
    return kx, ky, k2


@dataclass
class SimulationResult:
    model: str
    mode: str
    p: float | None
    u: np.ndarray
    aux: np.ndarray
    coeffs: InternalCoeffs | None
    snapshots: list[tuple[int, np.ndarray]]
    timeseries: list[dict]


def _linear_operator_fft(
    u_hat: np.ndarray,
    k2: np.ndarray,
    k2_eff: np.ndarray,
    k0: float,
    growth: float,
    dt: float,
) -> np.ndarray:
    linear = growth - (1.0 - k2_eff / (k0**2)) ** 2
    denom = (1.0 - dt * linear).astype(np.float32)
    return np.clip(denom, 1.0e-4, None)


def _nonlinear_step(
    u: np.ndarray,
    aux: np.ndarray,
    coeffs: InternalCoeffs,
    model: ProgressionModel,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.clip(u, 0.0, None)
    nonlinear = coeffs.quadratic * (u**2) - (u**3)
    next_aux = aux

    if model == ProgressionModel.SH_INHIBITOR:
        next_aux = (
            aux
            + coeffs.inhibitor_source * (pos**2)
            - coeffs.inhibitor_decay * aux
        ).astype(np.float32)
        nonlinear = nonlinear - coeffs.inhibitor_strength * aux * pos
    else:
        next_aux = (
            aux
            + coeffs.memory_source * (pos**2)
            - coeffs.memory_decay * aux
        ).astype(np.float32)
        nonlinear = nonlinear - coeffs.memory_strength * aux * pos

    return nonlinear.astype(np.float32), np.clip(next_aux, 0.0, 5.0)


def simulate_at_p(
    p: float,
    *,
    model: ProgressionModel,
    settings: ProgressionSettings,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    coeffs = coeffs_from_progression(p, model)
    return _simulate_core(
        model=model,
        settings=settings,
        coeffs=coeffs,
        p_fixed=p,
        ramp_rate=0.0,
        rng=rng,
    )


def simulate_ramp(
    *,
    model: ProgressionModel,
    settings: ProgressionSettings,
    ramp_rate: float,
    reverse: bool = False,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    return _simulate_core(
        model=model,
        settings=settings,
        coeffs=None,
        p_fixed=None,
        ramp_rate=ramp_rate * (-1.0 if reverse else 1.0),
        rng=rng,
    )


def _simulate_core(
    *,
    model: ProgressionModel,
    settings: ProgressionSettings,
    coeffs: InternalCoeffs | None,
    p_fixed: float | None,
    ramp_rate: float,
    rng: np.random.Generator | None,
) -> SimulationResult:
    size = settings.size
    if rng is None:
        rng = np.random.default_rng(settings.seed)

    u = (0.04 * rng.normal(0.0, 1.0, (size, size))).astype(np.float32)
    aux = np.zeros((size, size), dtype=np.float32)
    kx, ky, k2 = wavenumber_grid(size)
    k0 = 2.0 * np.pi / settings.wavelength
    dt = settings.dt

    p_value = 0.0 if p_fixed is None else float(p_fixed)
    mode = "sweep" if p_fixed is not None else ("reverse_ramp" if ramp_rate < 0 else "ramp")

    snapshot_steps = set(
        np.linspace(0, settings.steps, settings.snapshot_count, dtype=int).tolist()
    )
    snapshots: list[tuple[int, np.ndarray]] = []
    timeseries: list[dict] = []

    if 0 in snapshot_steps:
        snapshots.append((0, u.copy()))
    timeseries.append({"step": 0, **measure_goal_pattern(u, field_kind="height")})

    for step in range(1, settings.steps + 1):
        if ramp_rate != 0.0:
            p_value = float(np.clip(p_value + ramp_rate, 0.0, 1.0))
            coeffs = coeffs_from_progression(p_value, model)
        elif coeffs is None:
            raise ValueError("coeffs must be set when not ramping")

        if coeffs.anisotropy > 0.0:
            scale = 1.0 + coeffs.anisotropy
            k2_eff = (kx * scale) ** 2 + (ky / scale) ** 2
        else:
            k2_eff = k2

        nonlinear, aux = _nonlinear_step(u, aux, coeffs, model)

        u_hat = np.fft.fft2(u)
        n_hat = np.fft.fft2(nonlinear)
        denom = _linear_operator_fft(u_hat, k2, k2_eff, k0, coeffs.growth, dt)
        u = np.real(np.fft.ifft2((u_hat + dt * n_hat) / denom)).astype(np.float32)
        u = np.clip(u, -2.5, 2.5)

        # Slow diffusion on auxiliary fields (real-space Laplacian).
        if model in (ProgressionModel.SH_INHIBITOR, ProgressionModel.SH_MEMORY):
            aux_lap = (
                np.roll(aux, 1, 0)
                + np.roll(aux, -1, 0)
                + np.roll(aux, 1, 1)
                + np.roll(aux, -1, 1)
                - 4.0 * aux
            )
            aux = np.clip(aux + 0.08 * dt * aux_lap, 0.0, 5.0)

        if step in snapshot_steps:
            snapshots.append((step, u.copy()))
        if step % max(settings.steps // settings.snapshot_count, 1) == 0:
            row = measure_goal_pattern(u, field_kind="height")
            row["step"] = step
            row["p"] = p_value
            timeseries.append(row)

    final_coeffs = coeffs_from_progression(p_value, model) if ramp_rate != 0.0 else coeffs
    return SimulationResult(
        model=model.value,
        mode=mode,
        p=p_fixed,
        u=u.astype(np.float32),
        aux=aux.astype(np.float32),
        coeffs=final_coeffs,
        snapshots=snapshots,
        timeseries=timeseries,
    )


def render_height(u: np.ndarray) -> np.ndarray:
    field = normalize_u8(u)
    return cv2.applyColorMap(field, cv2.COLORMAP_BONE)


def save_simulation(
    result: SimulationResult,
    output: Path,
    *,
    tag: str,
) -> dict:
    case_dir = output / tag
    case_dir.mkdir(parents=True, exist_ok=True)

    final_metrics = measure_goal_pattern(result.u, field_kind="height")
    stability = measure_temporal_stability(result.timeseries)
    label = classify_morphology(final_metrics)

    metrics = {
        "tag": tag,
        "model": result.model,
        "mode": result.mode,
        "p": result.p,
        "morphology": label,
        **final_metrics,
        **stability,
    }
    if result.coeffs is not None:
        metrics.update({f"coeff_{k}": v for k, v in asdict(result.coeffs).items()})

    cv2.imwrite(str(case_dir / "final_height.png"), render_height(result.u))
    annotated = annotate(
        render_height(result.u),
        [
            f"model={result.model} mode={result.mode} p={result.p}",
            f"morphology={label}",
            f"branch={final_metrics.get('branch_density', 0):.3f} "
            f"width={final_metrics.get('feature_width_median', 0):.1f} "
            f"labyrinth={final_metrics.get('labyrinth_score', 0):.1f}",
        ],
    )
    cv2.imwrite(str(case_dir / "final_annotated.png"), annotated)

    for step, field in result.snapshots:
        cv2.imwrite(str(case_dir / f"height_{step:06d}.png"), render_height(field))

    with (case_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    ts_path = case_dir / "timeseries.json"
    with ts_path.open("w", encoding="utf-8") as handle:
        json.dump(result.timeseries, handle, indent=2)

    return metrics


def run_sweep(
    *,
    model: ProgressionModel,
    settings: ProgressionSettings,
    output: Path,
    p_values: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for p in p_values:
        tag = f"p_{p:.3f}"
        result = simulate_at_p(float(p), model=model, settings=settings)
        metrics = save_simulation(result, output, tag=tag)
        stability = measure_temporal_stability(result.timeseries)
        row = {"p": float(p), **metrics, **stability}
        rows.append(row)
    return rows


def run_ramp_batch(
    *,
    model: ProgressionModel,
    settings: ProgressionSettings,
    output: Path,
    ramp_rates: list[float],
    reverse: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for rate in ramp_rates:
        suffix = "reverse" if reverse else "forward"
        tag = f"ramp_{suffix}_{rate:.5f}"
        result = simulate_ramp(
            model=model,
            settings=settings,
            ramp_rate=rate,
            reverse=reverse,
        )
        metrics = save_simulation(result, output, tag=tag)
        row = {"ramp_rate": rate, "reverse": reverse, **metrics}
        rows.append(row)
    return rows


def write_contact_sheet(rows: list[dict], image_paths: list[Path], output: Path) -> None:
    images = [cv2.imread(str(path)) for path in image_paths if path.exists()]
    images = [img for img in images if img is not None]
    if not images:
        return
    h = max(img.shape[0] for img in images)
    resized = []
    for img in images:
        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)
        resized.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate(resized, axis=1)
    cv2.imwrite(str(output), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["sweep", "ramp", "reverse_ramp", "compare"],
        help="Simulation mode",
    )
    parser.add_argument(
        "--model",
        choices=[m.value for m in ProgressionModel],
        default=ProgressionModel.SH_INHIBITOR.value,
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/pit_progression"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wavelength", type=float, default=12.0)
    parser.add_argument("--snapshot-count", type=int, default=6)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--p-max", type=float, default=1.0)
    parser.add_argument("--p-count", type=int, default=25)
    parser.add_argument(
        "--ramp-rates",
        type=float,
        nargs="+",
        default=[1.0 / 6000.0, 1.0 / 4000.0, 1.0 / 2500.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ProgressionSettings(
        size=args.size,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        wavelength=args.wavelength,
        snapshot_count=args.snapshot_count,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        p_values = np.linspace(args.p_min, args.p_max, args.p_count)
        all_rows: list[dict] = []
        for model in ProgressionModel:
            model_dir = output / model.value
            rows = run_sweep(
                model=model,
                settings=settings,
                output=model_dir,
                p_values=p_values,
            )
            for row in rows:
                row["model"] = model.value
            all_rows.extend(rows)
            images = [
                model_dir / f"p_{row['p']:.3f}" / "final_annotated.png" for row in rows
            ]
            write_contact_sheet(rows, images, model_dir / "contact_sheet.png")
        write_metrics(output, all_rows)
        return

    model = ProgressionModel(args.model)

    if args.mode == "sweep":
        p_values = np.linspace(args.p_min, args.p_max, args.p_count)
        rows = run_sweep(
            model=model,
            settings=settings,
            output=output,
            p_values=p_values,
        )
        images = [output / f"p_{row['p']:.3f}" / "final_annotated.png" for row in rows]
        write_contact_sheet(rows, images, output / "contact_sheet.png")
        write_metrics(output, rows)
        return

    reverse = args.mode == "reverse_ramp"
    rows = run_ramp_batch(
        model=model,
        settings=settings,
        output=output,
        ramp_rates=args.ramp_rates,
        reverse=reverse,
    )
    write_metrics(output, rows)


if __name__ == "__main__":
    main()
