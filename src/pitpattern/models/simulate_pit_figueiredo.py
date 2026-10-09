#!/usr/bin/env python3
"""Generate Kudo pit-like patterns with a Figueiredo-style coupled model.

Reduced 2D prototype of Figueiredo et al. (Math. Biosci. 2019): proliferative
cell density N obeys convection-diffusion-reaction with Darcy velocity
v = -xi * grad(p), pressure p solves an elliptic balance from cell excess,
and orifice height u deforms under pressure forcing on an elastic substrate.

Example:
    python3 src/pitpattern/models/simulate_pit_figueiredo.py --output artifacts/pit_figueiredo
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pitpattern.metrics.evaluate_pit_steps import (
    annotate,
    measure_pattern,
    normalize_u8,
    render_skeleton_overlay,
    write_metrics,
)
from pitpattern.models.pit_fission import apply_component_fission, local_resource


@dataclass(frozen=True)
class FigueiredoConfig:
    name: str
    description: str
    diffusion: float
    darcy: float
    carrying: float
    prolif_base: float
    prolif_boost: float
    pressure_gain: float
    orifice_gain: float
    wavelength: float
    focal_radius: float = 0.0
    focal_center: tuple[float, float] = (0.5, 0.5)
    intercryptal_band: float = 0.0
    multi_foci: bool = False
    pressure_saturation: float = 0.0
    resource_depletion: float = 0.0
    fission_strength: float = 0.0


@dataclass
class FigueiredoResult:
    config: FigueiredoConfig
    u: np.ndarray
    density: np.ndarray
    pressure: np.ndarray
    snapshots: list[tuple[int, np.ndarray]]


def wavenumber_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.fft.fftfreq(size, d=1.0) * (2.0 * np.pi)
    kx = freqs[None, :]
    ky = freqs[:, None]
    kx = np.broadcast_to(kx, (size, size)).astype(np.float32)
    ky = np.broadcast_to(ky, (size, size)).astype(np.float32)
    k2 = (kx**2 + ky**2).astype(np.float32)
    return kx, ky, k2


def smooth_unit_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    smooth -= float(smooth.mean())
    std = float(smooth.std())
    if std > 1.0e-8:
        smooth /= std
    return smooth.astype(np.float32)


def prolif_field(config: FigueiredoConfig, size: int, rng: np.random.Generator) -> np.ndarray:
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
    beta = np.full((size, size), config.prolif_base, dtype=np.float32)

    if config.focal_radius > 0.0:
        cx, cy = config.focal_center
        r2 = (xs - cx) ** 2 + (ys - cy) ** 2
        beta += config.prolif_boost * np.exp(-r2 / (config.focal_radius**2))

    if config.intercryptal_band > 0.0:
        band = np.abs(ys - 0.5)
        beta += config.prolif_boost * np.exp(-((band - config.intercryptal_band) ** 2) / 0.004)

    if config.multi_foci:
        foci = [(0.28, 0.32), (0.72, 0.35), (0.45, 0.72), (0.68, 0.68)]
        for cx, cy in foci:
            r2 = (xs - cx) ** 2 + (ys - cy) ** 2
            beta += 0.55 * config.prolif_boost * np.exp(-r2 / 0.012)

    beta += 0.08 * config.prolif_boost * smooth_unit_noise(rng, (size, size), sigma=max(4.0, size / 16))
    return np.clip(beta, 0.0, None).astype(np.float32)


def solve_pressure(density: np.ndarray, gain: float) -> np.ndarray:
    size = density.shape[0]
    _, _, k2 = wavenumber_grid(size)
    excess = density - float(density.mean())
    p_hat = np.fft.fft2(excess).astype(np.complex64)
    k2_safe = k2.copy()
    k2_safe[0, 0] = 1.0
    p_hat = -gain * p_hat / k2_safe
    p_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(p_hat)).astype(np.float32)


def gradients(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.gradient(field.astype(np.float32))


def laplacian(field: np.ndarray) -> np.ndarray:
    size = field.shape[0]
    _, _, k2 = wavenumber_grid(size)
    f_hat = np.fft.fft2(field.astype(np.float32))
    return np.real(np.fft.ifft2(-k2 * f_hat)).astype(np.float32)


def run_figueiredo(
    config: FigueiredoConfig,
    *,
    size: int,
    steps: int,
    dt: float,
    seed: int,
    snapshot_count: int,
) -> FigueiredoResult:
    rng = np.random.default_rng(seed)
    beta = prolif_field(config, size, rng)
    n = np.clip(0.15 + 0.03 * smooth_unit_noise(rng, (size, size), sigma=6.0), 0.05, 0.35).astype(
        np.float32
    )
    u = (0.04 * smooth_unit_noise(rng, (size, size), sigma=8.0)).astype(np.float32)

    _, _, k2 = wavenumber_grid(size)
    k0 = 2.0 * np.pi / config.wavelength
    linear = config.orifice_gain - (1.0 - k2 / (k0**2)) ** 2
    denom = (1.0 - dt * linear).astype(np.float32)

    snapshot_steps = set(np.linspace(0, steps, snapshot_count, dtype=int).tolist())
    snapshots: list[tuple[int, np.ndarray]] = []
    if 0 in snapshot_steps:
        snapshots.append((0, u.copy()))

    for step in range(1, steps + 1):
        p = solve_pressure(n, config.pressure_gain)
        pit = pit_field(u)
        resource = (
            local_resource(
                pit,
                sigma=max(2.0, config.wavelength / 2.0),
                depletion=config.resource_depletion,
            )
            if config.resource_depletion > 0.0
            else 1.0
        )
        if config.pressure_saturation > 0.0:
            load = 1.0 / resource if not np.isscalar(resource) else 1.0
            p = p / (1.0 + config.pressure_saturation * np.maximum(load - 1.0, 0.0))
        grad_px, grad_py = gradients(p)
        grad_nx, grad_ny = gradients(n)
        lap_n = laplacian(n)

        advection = config.darcy * (grad_px * grad_nx + grad_py * grad_ny + n * laplacian(p))
        growth = resource * beta * n * (1.0 - n / config.carrying)
        n = np.clip(n + dt * (config.diffusion * lap_n - advection + growth), 0.0, config.carrying)

        pressure_forcing = config.orifice_gain * (p - float(p.mean()))
        nonlinear = pressure_forcing * u - (u**3)
        u_hat = np.fft.fft2(u)
        n_hat = np.fft.fft2(nonlinear)
        u_hat = (u_hat + dt * n_hat) / denom
        u = np.real(np.fft.ifft2(u_hat)).astype(np.float32)
        u = np.clip(u, -3.0, 3.0)
        if config.fission_strength > 0.0 and step % 25 == 0:
            u = apply_component_fission(
                u,
                pit_field(u),
                strength=config.fission_strength,
                max_area_fraction=0.026,
                max_length_fraction=0.25,
                band_width=max(1.5, config.wavelength / 9.0),
            )

        if step in snapshot_steps:
            snapshots.append((step, u.copy()))

    p_final = solve_pressure(n, config.pressure_gain)
    return FigueiredoResult(config=config, u=u, density=n, pressure=p_final, snapshots=snapshots)


def pit_field(u: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(u, 1))
    hi = float(np.percentile(u, 99))
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.clip((u - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_suite() -> list[FigueiredoConfig]:
    return [
        FigueiredoConfig(
            name="figueiredo_pit_i_normal",
            description="Uniform low proliferation: round regular orifices (Type I).",
            diffusion=0.08,
            darcy=0.35,
            carrying=1.0,
            prolif_base=0.12,
            prolif_boost=0.0,
            pressure_gain=1.4,
            orifice_gain=0.28,
            wavelength=13.0,
        ),
        FigueiredoConfig(
            name="figueiredo_pit_iii_focal",
            description="Central hyperproliferation: enlarged focal orifice (Type IIIL).",
            diffusion=0.08,
            darcy=0.40,
            carrying=1.0,
            prolif_base=0.12,
            prolif_boost=0.55,
            pressure_gain=1.8,
            orifice_gain=0.24,
            wavelength=13.0,
            focal_radius=0.12,
            focal_center=(0.5, 0.5),
            pressure_saturation=0.6,
            resource_depletion=1.2,
            fission_strength=0.22,
        ),
        FigueiredoConfig(
            name="figueiredo_pit_iv_intercryptal",
            description="Inter-cryptal proliferation: branching deformed orifices (Type IV).",
            diffusion=0.10,
            darcy=0.55,
            carrying=1.0,
            prolif_base=0.10,
            prolif_boost=0.65,
            pressure_gain=2.2,
            orifice_gain=0.30,
            wavelength=11.0,
            intercryptal_band=0.18,
            pressure_saturation=0.9,
            resource_depletion=2.0,
            fission_strength=0.38,
        ),
        FigueiredoConfig(
            name="figueiredo_pit_vi_multifocal",
            description="Multiple abnormal foci: coexisting irregular orifices (Type VI).",
            diffusion=0.09,
            darcy=0.45,
            carrying=1.0,
            prolif_base=0.10,
            prolif_boost=0.70,
            pressure_gain=2.0,
            orifice_gain=0.28,
            wavelength=12.0,
            multi_foci=True,
            pressure_saturation=0.8,
            resource_depletion=1.6,
            fission_strength=0.32,
        ),
    ]


def render_height(u: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(normalize_u8(u), cv2.COLORMAP_BONE)


def render_density(n: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(normalize_u8(n), cv2.COLORMAP_VIRIDIS)


def save_result_images(result: FigueiredoResult, metrics: dict[str, float], output: Path) -> None:
    case_dir = output / result.config.name
    case_dir.mkdir(parents=True, exist_ok=True)

    for step, u in result.snapshots:
        cv2.imwrite(str(case_dir / f"height_{step:06d}.png"), render_height(u))

    cv2.imwrite(str(case_dir / "final_density.png"), render_density(result.density))
    cv2.imwrite(str(case_dir / "final_pressure.png"), render_density(result.pressure))

    pit = pit_field(result.u)
    final = annotate(
        render_height(result.u),
        [
            result.config.name,
            f"branch={metrics['branch_density']:.3f} long={metrics['long_line_score']:.2f}",
            f"irreg={metrics['vi_irregularity_score']:.2f} cov={metrics['binary_coverage']:.2f}",
        ],
    )
    overlay = annotate(
        render_skeleton_overlay(pit),
        [
            "skeleton overlay",
            "red=branch cyan=end",
            f"components={metrics['skeleton_components']:.0f}",
        ],
    )
    cv2.imwrite(str(case_dir / "final_height.png"), final)
    cv2.imwrite(str(case_dir / "final_skeleton.png"), overlay)


def write_contact_sheet(output: Path, results: list[tuple[FigueiredoResult, dict[str, float]]]) -> None:
    cells = []
    for result, metrics in results:
        height = annotate(
            render_height(result.u),
            [
                result.config.name,
                f"branch {metrics['branch_density']:.3f}",
                f"long {metrics['long_line_score']:.2f}",
            ],
        )
        skeleton = annotate(
            render_skeleton_overlay(pit_field(result.u)),
            [
                "skeleton",
                f"irreg {metrics['vi_irregularity_score']:.2f}",
                f"area_cv {metrics['component_area_cv']:.2f}",
            ],
        )
        cells.append(np.concatenate([height, skeleton], axis=1))

    if not cells:
        return

    width = max(cell.shape[1] for cell in cells)
    padded = []
    for cell in cells:
        if cell.shape[1] < width:
            pad = np.zeros((cell.shape[0], width - cell.shape[1], 3), dtype=np.uint8)
            cell = np.concatenate([cell, pad], axis=1)
        padded.append(cell)

    cv2.imwrite(str(output / "contact_sheet.png"), np.concatenate(padded, axis=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pit-pattern morphologies with a Figueiredo-style coupled model."
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/pit_figueiredo"))
    parser.add_argument("--size", type=int, default=160, help="Simulation width/height.")
    parser.add_argument("--steps", type=int, default=2500, help="Integration steps per case.")
    parser.add_argument("--dt", type=float, default=0.35, help="Time step.")
    parser.add_argument("--seed", type=int, default=7, help="Seed shared by all cases.")
    parser.add_argument("--snapshot-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    results: list[tuple[FigueiredoResult, dict[str, float]]] = []
    for config in build_suite():
        print(f"running {config.name} ...", flush=True)
        result = run_figueiredo(
            config,
            size=args.size,
            steps=args.steps,
            dt=args.dt,
            seed=args.seed,
            snapshot_count=args.snapshot_count,
        )
        b = pit_field(result.u)
        zeros = np.zeros_like(b)
        metrics = measure_pattern(zeros, b, zeros)
        save_result_images(result, metrics, args.output)

        rows.append(
            {
                "name": config.name,
                "description": config.description,
                "steps": args.steps,
                "size": args.size,
                **asdict(config),
                **metrics,
            }
        )
        results.append((result, metrics))

    write_metrics(args.output, rows)
    write_contact_sheet(args.output, results)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
