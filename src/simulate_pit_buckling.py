#!/usr/bin/env python3
"""Generate Kudo pit-like patterns with a NON reaction-diffusion model.

This script uses a mechanical buckling model instead of a Gray-Scott
reaction-diffusion system. The epithelial monolayer sitting on an elastic
stroma buckles under proliferation-induced compression. Hannezo, Prost and
Joanny (Phys. Rev. Lett. 107, 078104, 2011) showed that this buckling gives a
phase diagram of colon crypts, fingers, herringbone and labyrinth patterns,
and Shyer et al. (Science 342, 212, 2013) showed the same compressive folding
during gut villification.

Near the buckling threshold the surface height ``u(x, y)`` of such a growing
layer on a substrate obeys the Swift-Hohenberg equation, the canonical reduced
model for a wavelength-selecting elastic/convective instability:

    du/dt = r u - (k0^2 + laplacian)^2 u + g u^2 - u^3

- ``r``   : reduced growth / compressive stress (proliferation pressure).
- ``k0``  : preferred wavenumber; sets the crypt spacing (pit size).
- ``g``   : quadratic term; g large selects a hexagonal lattice of round pits,
            g ~ 0 selects stripes / labyrinth / branching.
- ``-u^3``: saturation.

The mechanism is elastic (mechanical), not chemical diffusion, yet it selects
a wavelength and reproduces the Type I -> IV -> VI morphological progression.

Peaks of ``u`` are treated as pit openings, so the same binary/skeleton
metrics used for the reaction-diffusion evaluation apply directly, allowing an
apples-to-apples morphological comparison.

Example:
    python3 src/simulate_pit_buckling.py --output artifacts/pit_buckling
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from evaluate_pit_steps import (
    annotate,
    measure_pattern,
    normalize_u8,
    render_skeleton_overlay,
    write_metrics,
)


@dataclass(frozen=True)
class BucklingConfig:
    """One mechanical-buckling variant mapped to a Kudo pit type."""

    name: str
    description: str
    wavelength: float
    growth: float
    quadratic: float
    growth_heterogeneity: float = 0.0
    wavelength_heterogeneity: float = 0.0
    quadratic_heterogeneity: float = 0.0
    anisotropy: float = 0.0


@dataclass
class BucklingResult:
    config: BucklingConfig
    u: np.ndarray
    snapshots: list[tuple[int, np.ndarray]]


def smooth_unit_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, shape).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    smooth -= float(smooth.mean())
    std = float(smooth.std())
    if std > 1.0e-8:
        smooth /= std
    return smooth.astype(np.float32)


def wavenumber_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.fft.fftfreq(size, d=1.0) * (2.0 * np.pi)
    kx = freqs[None, :]
    ky = freqs[:, None]
    kx = np.broadcast_to(kx, (size, size)).astype(np.float32)
    ky = np.broadcast_to(ky, (size, size)).astype(np.float32)
    k2 = kx**2 + ky**2
    return kx, ky, k2.astype(np.float32)


def run_buckling(
    config: BucklingConfig,
    *,
    size: int,
    steps: int,
    dt: float,
    seed: int,
    snapshot_count: int,
) -> BucklingResult:
    rng = np.random.default_rng(seed)
    u = (0.05 * rng.normal(0.0, 1.0, (size, size))).astype(np.float32)

    kx, ky, k2 = wavenumber_grid(size)
    k0 = 2.0 * np.pi / config.wavelength

    # Anisotropy stretches the preferred wavenumber along one axis, which biases
    # stripes into an oriented herringbone rather than an isotropic labyrinth.
    if config.anisotropy != 0.0:
        scale = 1.0 + config.anisotropy
        k2_eff = (kx * scale) ** 2 + (ky / scale) ** 2
    else:
        k2_eff = k2

    # Local wavelength heterogeneity: a smooth field that shifts the preferred
    # wavenumber region to region (mixed pit sizes -> Type VI style irregularity).
    if config.wavelength_heterogeneity > 0.0:
        wl_field = 1.0 + config.wavelength_heterogeneity * smooth_unit_noise(
            rng, (size, size), sigma=max(4.0, size / 12)
        )
        k0_field = (2.0 * np.pi / (config.wavelength * wl_field)).astype(np.float32)
    else:
        k0_field = None

    if k0_field is None:
        # Normalized Swift-Hohenberg operator (1 + laplacian/k0^2)^2 keeps the
        # uniform (k=0) mode damped for any pit size, selecting the k0 band.
        linear = config.growth - (1.0 - k2_eff / (k0**2)) ** 2
        denom = (1.0 - dt * linear).astype(np.float32)

    if config.growth_heterogeneity > 0.0:
        growth_field = config.growth_heterogeneity * smooth_unit_noise(
            rng, (size, size), sigma=max(4.0, size / 12)
        )
    else:
        growth_field = None

    # A smoothly varying quadratic bias lets some regions favor hexagonal spots
    # (round pits) while others favor stripes/branching, so several pit regimes
    # coexist in one field -- the hallmark of an irregular Type VI pattern.
    if config.quadratic_heterogeneity > 0.0:
        quadratic_field = config.quadratic + config.quadratic_heterogeneity * smooth_unit_noise(
            rng, (size, size), sigma=max(5.0, size / 8)
        )
        quadratic_field = np.clip(quadratic_field, 0.0, None).astype(np.float32)
    else:
        quadratic_field = None

    snapshot_steps = set(np.linspace(0, steps, snapshot_count, dtype=int).tolist())
    snapshots: list[tuple[int, np.ndarray]] = []
    if 0 in snapshot_steps:
        snapshots.append((0, u.copy()))

    for step in range(1, steps + 1):
        quad = config.quadratic if quadratic_field is None else quadratic_field
        nonlinear = quad * (u**2) - (u**3)
        if growth_field is not None:
            nonlinear = nonlinear + growth_field * u

        if k0_field is None:
            u_hat = np.fft.fft2(u)
            n_hat = np.fft.fft2(nonlinear)
            u_hat = (u_hat + dt * n_hat) / denom
            u = np.real(np.fft.ifft2(u_hat)).astype(np.float32)
        else:
            # Spatially varying wavelength cannot use a single Fourier symbol, so
            # apply the normalized buckling operator (1 + lap/k0^2)^2 explicitly.
            u_hat = np.fft.fft2(u)
            lap = np.real(np.fft.ifft2(-k2_eff * u_hat)).astype(np.float32)
            s = u + lap / (k0_field**2)
            lap_s = np.real(np.fft.ifft2(-k2_eff * np.fft.fft2(s))).astype(np.float32)
            operator = s + lap_s / (k0_field**2)
            du = config.growth * u - operator + nonlinear
            u = (u + dt * du).astype(np.float32)

        u = np.clip(u, -3.0, 3.0)
        if step in snapshot_steps:
            snapshots.append((step, u.copy()))

    return BucklingResult(config=config, u=u.astype(np.float32), snapshots=snapshots)


def pit_field(u: np.ndarray) -> np.ndarray:
    """Map buckling height to a [0, 1] pit-occupancy field (peaks = pits)."""

    lo = float(np.percentile(u, 1))
    hi = float(np.percentile(u, 99))
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.clip((u - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def render_height(u: np.ndarray) -> np.ndarray:
    field = normalize_u8(u)
    colored = cv2.applyColorMap(field, cv2.COLORMAP_BONE)
    return colored


def build_suite() -> list[BucklingConfig]:
    return [
        BucklingConfig(
            name="buckling_pit_i_round",
            description="Strong hexagon selection: regular round pits (Type I).",
            wavelength=13.0,
            growth=0.28,
            quadratic=1.1,
        ),
        BucklingConfig(
            name="buckling_pit_iii_s",
            description="Shorter wavelength hexagons: small dense round pits (Type IIIs).",
            wavelength=8.5,
            growth=0.30,
            quadratic=1.0,
        ),
        BucklingConfig(
            name="buckling_pit_iii_l",
            description="Longer wavelength hexagons: larger round/tubular pits (Type IIIL).",
            wavelength=18.0,
            growth=0.26,
            quadratic=0.9,
        ),
        BucklingConfig(
            name="buckling_pit_iv_labyrinth",
            description="No quadratic bias: stripes/labyrinth with branching (Type IV).",
            wavelength=13.0,
            growth=0.30,
            quadratic=0.0,
        ),
        BucklingConfig(
            name="buckling_pit_iv_herringbone",
            description="Anisotropic buckling: oriented branching herringbone (Type IV).",
            wavelength=13.0,
            growth=0.30,
            quadratic=0.15,
            anisotropy=0.35,
        ),
        BucklingConfig(
            name="buckling_pit_vi_irregular",
            description="Heterogeneous growth and spot/stripe bias: coexisting regimes, irregular (Type VI).",
            wavelength=12.0,
            growth=0.30,
            quadratic=1.0,
            growth_heterogeneity=0.15,
            quadratic_heterogeneity=1.4,
        ),
    ]


def save_result_images(result: BucklingResult, metrics: dict[str, float], output: Path) -> None:
    case_dir = output / result.config.name
    case_dir.mkdir(parents=True, exist_ok=True)

    for step, u in result.snapshots:
        cv2.imwrite(str(case_dir / f"height_{step:06d}.png"), render_height(u))

    final = annotate(
        render_height(result.u),
        [
            result.config.name,
            f"branch={metrics['branch_density']:.3f} long={metrics['long_line_score']:.2f}",
            f"irreg={metrics['vi_irregularity_score']:.2f} cov={metrics['binary_coverage']:.2f}",
        ],
    )
    overlay = annotate(
        render_skeleton_overlay(pit_field(result.u)),
        [
            "skeleton overlay",
            "red=branch cyan=end",
            f"components={metrics['skeleton_components']:.0f}",
        ],
    )
    cv2.imwrite(str(case_dir / "final_height.png"), final)
    cv2.imwrite(str(case_dir / "final_skeleton.png"), overlay)


def write_contact_sheet(output: Path, results: list[tuple[BucklingResult, dict[str, float]]]) -> None:
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
        description="Generate pit-pattern morphologies with a mechanical buckling model."
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/pit_buckling"))
    parser.add_argument("--size", type=int, default=160, help="Simulation width/height.")
    parser.add_argument("--steps", type=int, default=2000, help="Integration steps per case.")
    parser.add_argument("--dt", type=float, default=0.4, help="Time step.")
    parser.add_argument("--seed", type=int, default=7, help="Seed shared by all cases.")
    parser.add_argument("--snapshot-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    results: list[tuple[BucklingResult, dict[str, float]]] = []
    for config in build_suite():
        print(f"running {config.name} ...", flush=True)
        result = run_buckling(
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
