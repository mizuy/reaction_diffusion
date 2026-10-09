#!/usr/bin/env python3
"""Independent-initialization activator--inhibitor pit-pattern candidate.

    u_t = Du Laplacian(u) + r*u - u**3 - g*v + b
    tau*v_t = Dv Laplacian(v) + u - v

All coefficients except ``bias`` are fixed during a progression sweep.  Every
bias value starts from its own noisy homogeneous equilibrium; no completed
pattern is continued into the next case.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pit_microphase_metrics import classify, measure


@dataclass(frozen=True)
class ActivatorInhibitorConfig:
    size: int = 128
    dx: float = 1.0
    dt: float = 0.08
    steps: int = 5000
    diffusion_u: float = 1.0
    diffusion_v: float = 20.0
    growth: float = 1.0
    coupling: float = 1.4
    tau: float = 0.5
    noise: float = 0.05
    seed: int = 7
    snapshot_count: int = 6


@dataclass
class SimulationResult:
    u: np.ndarray
    v: np.ndarray
    equilibrium: float
    diagnostics: dict


def homogeneous_equilibrium(bias: float, cfg: ActivatorInhibitorConfig) -> float:
    """Return the unique real equilibrium when coupling > growth."""
    roots = np.roots([1.0, 0.0, cfg.coupling - cfg.growth, -bias])
    real = roots[np.abs(roots.imag) < 1e-9].real
    if len(real) != 1:
        raise ValueError("configuration does not have a unique homogeneous equilibrium")
    return float(real[0])


def linear_growth_rates(bias: float, cfg: ActivatorInhibitorConfig) -> dict:
    """Evaluate the continuum Turing conditions at the homogeneous state."""
    u0 = homogeneous_equilibrium(bias, cfg)
    a = cfg.growth - 3.0 * u0 * u0
    trace0 = a - 1.0 / cfg.tau
    det0 = (cfg.coupling - a) / cfg.tau
    cross = a * cfg.diffusion_v - cfg.diffusion_u
    discriminant = cross * cross - 4.0 * cfg.diffusion_u * cfg.diffusion_v * (cfg.coupling - a)
    return {
        "equilibrium": u0,
        "a": a,
        "homogeneous_stable": bool(trace0 < 0 and det0 > 0),
        "turing_continuum": bool(trace0 < 0 and det0 > 0 and cross > 0 and discriminant > 0),
        "trace_q0": trace0,
        "determinant_q0": det0,
        "turing_discriminant": discriminant,
    }


def simulate(bias: float, cfg: ActivatorInhibitorConfig) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)
    u0 = homogeneous_equilibrium(bias, cfg)
    u = u0 + cfg.noise * rng.normal(size=(cfg.size, cfg.size))
    v = u0 + cfg.noise * rng.normal(size=(cfg.size, cfg.size))
    wave = 2.0 * np.pi * np.fft.fftfreq(cfg.size, d=cfg.dx)
    k2 = wave[:, None] ** 2 + wave[None, :] ** 2
    den_u = 1.0 + cfg.dt * cfg.diffusion_u * k2
    den_v = 1.0 + cfg.dt * (1.0 + cfg.diffusion_v * k2) / cfg.tau
    every = max(1, cfg.steps // max(cfg.snapshot_count - 1, 1))
    history = []
    previous = None

    for step in range(cfg.steps + 1):
        if step % every == 0 or step == cfg.steps:
            rhs_u = np.fft.ifft2(-cfg.diffusion_u * k2 * np.fft.fft2(u)).real + cfg.growth * u - u**3 - cfg.coupling * v + bias
            rhs_v = np.fft.ifft2(-cfg.diffusion_v * k2 * np.fft.fft2(v)).real / cfg.tau + (u - v) / cfg.tau
            history.append({
                "step": step,
                "rhs_rms": float(np.sqrt(np.mean(rhs_u**2 + rhs_v**2))),
                "u_mean": float(u.mean()),
                "u_std": float(u.std()),
            })
            if step >= int(0.8 * cfg.steps) and previous is None:
                previous = u.copy()
        if step == cfg.steps:
            break
        reaction_u = cfg.growth * u - u**3 - cfg.coupling * v + bias
        u_next = np.fft.ifft2(np.fft.fft2(u + cfg.dt * reaction_u) / den_u).real
        v_next = np.fft.ifft2(np.fft.fft2(v + cfg.dt * u / cfg.tau) / den_v).real
        u, v = u_next, v_next
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise FloatingPointError("non-finite activator-inhibitor state")

    threshold = 0.5 * (float(u.min()) + float(u.max()))
    diagnostics = {
        **linear_growth_rates(bias, cfg),
        "threshold": threshold,
        "late_rms": float(np.sqrt(np.mean((u - previous) ** 2))) if previous is not None else 0.0,
        "history": history,
    }
    return SimulationResult(u=u, v=v, equilibrium=u0, diagnostics=diagnostics)


def render(u: np.ndarray, threshold: float, path: Path) -> None:
    binary = u > threshold
    image = np.empty((*u.shape, 3), dtype=np.uint8)
    image[:] = (231, 222, 204)
    image[binary] = (74, 62, 32)
    cv2.imwrite(str(path), cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def run_sweep(biases: list[float], cfg: ActivatorInhibitorConfig, output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    fields = []
    cases = []
    for bias in biases:
        result = simulate(bias, cfg)
        threshold = result.diagnostics["threshold"]
        contrast = float(np.ptp(result.u))
        patterned = contrast >= 0.1
        metrics = measure(result.u, dx=cfg.dx, threshold=threshold) if patterned else measure(np.full_like(result.u, -1.0), dx=cfg.dx)
        cases.append((bias, result, threshold, contrast, patterned, metrics))
    widths = [case[-1]["width_median"] for case in cases if case[-2] and case[-1]["width_median"] > 0]
    reference_width = widths[0] if widths else 1.0

    for bias, result, threshold, contrast, patterned, metrics in cases:
        row = {
            "bias": bias,
            "seed": cfg.seed,
            "contrast": contrast,
            "patterned": patterned,
            **{k: v for k, v in result.diagnostics.items() if k != "history"},
            **metrics,
        }
        row["morphology"] = classify(metrics, reference_width, area=(cfg.size * cfg.dx) ** 2) if patterned else "uniform"
        rows.append(row)
        fields.append(result.u)
        tag = f"bias_{bias:+.4f}".replace("+", "p").replace("-", "m")
        render(result.u, threshold, output / f"{tag}.png")
        print(json.dumps(row), flush=True)

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "fields.npz", u=np.asarray(fields), biases=np.asarray(biases))
    (output / "report.json").write_text(json.dumps({"config": asdict(cfg), "rows": rows}, indent=2), encoding="utf-8")
    return rows


def run_seed_batch(biases: list[float], cfg: ActivatorInhibitorConfig, seeds: list[int], output: Path) -> list[dict]:
    """Run every parameter from independent noise for every requested seed."""
    all_rows = []
    for seed in seeds:
        case_cfg = ActivatorInhibitorConfig(**{**asdict(cfg), "seed": seed})
        rows = run_sweep(biases, case_cfg, output / f"seed_{seed}")
        all_rows.extend(rows)
    with (output / "all_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {}
    for bias in biases:
        labels = [row["morphology"] for row in all_rows if row["bias"] == bias]
        summary[str(bias)] = {label: labels.count(label) for label in sorted(set(labels))}
    (output / "seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pit_activator_inhibitor"))
    parser.add_argument("--biases", type=float, nargs="+", default=list(np.linspace(-0.3, 0.3, 13)))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    for name in ("size", "steps", "seed", "snapshot_count"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(ActivatorInhibitorConfig(), name))
    for name in ("dx", "dt", "diffusion_u", "diffusion_v", "growth", "coupling", "tau", "noise"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(ActivatorInhibitorConfig(), name))
    args = parser.parse_args()
    values = {name: getattr(args, name) for name in asdict(ActivatorInhibitorConfig())}
    cfg = ActivatorInhibitorConfig(**values)
    if args.seeds:
        run_seed_batch(args.biases, cfg, args.seeds, args.output)
    else:
        run_sweep(args.biases, cfg, args.output)


if __name__ == "__main__":
    main()
