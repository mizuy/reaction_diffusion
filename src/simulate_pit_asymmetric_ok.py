#!/usr/bin/env python3
"""Asymmetric-well nonlocal phase separation for the pit-pattern goal.

    u_t = Laplacian( f'(u) - epsilon^2 Laplacian(u) ) - alpha (u - m),
    f'(u) = u (u - a) (u - 1),      <u> = m.

u=0 is the inter-pit background (stroma side), u=1 the pit / groove phase.
The barrier a is placed close to the background well, so a small pit fraction
m lies inside the spinodal and the pattern grows spontaneously from small
noise.  No heterogeneity field, no classifier, no cutting and no type-specific
switch takes part in the evolution.  m is the single control (pit material
fraction); epsilon, alpha and a are fixed for all four target morphologies.

Interfacial tension favours 120-degree triple junctions, while the long-range
term (alpha) penalises large units: with more pit material the cheapest units
go from disks to short rods to compact branched tripods and finally to a
connected labyrinth whose groove width is set by epsilon and alpha.

Example:
    uv run python src/simulate_pit_asymmetric_ok.py sweep \\
        --output artifacts/pit_asymmetric_ok/explore --seeds 1 7 42
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Config:
    size: int = 192
    dx: float = 1.0
    dt: float = 0.5
    steps: int = 6000
    epsilon: float = 1.0
    alpha: float = 0.006
    barrier: float = 0.1
    noise: float = 0.05
    stabilizer: float = 1.0
    threshold: float = 0.5


def fprime(u: np.ndarray, a: float) -> np.ndarray:
    return u * (u - a) * (u - 1.0)


class AsymmetricOK:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        k = 2 * np.pi * np.fft.fftfreq(cfg.size, d=cfg.dx)
        self.k2 = k[:, None] ** 2 + k[None, :] ** 2
        self.denominator = 1 + cfg.dt * (
            cfg.epsilon**2 * self.k2**2 + cfg.stabilizer * self.k2 + cfg.alpha
        )

    def initial(self, m: float, seed: int, noise: float | None = None) -> np.ndarray:
        amp = self.cfg.noise if noise is None else noise
        rng = np.random.default_rng(seed)
        u = m + amp * rng.normal(size=(self.cfg.size, self.cfg.size))
        return u + m - u.mean()

    def linear_growth(self, m: float) -> tuple[float, float]:
        """Max growth rate and its wavenumber of the uniform state u=m."""
        c = self.cfg
        a = c.barrier
        fpp = 3 * m * m - 2 * (1 + a) * m + a
        q2 = np.linspace(0, 4, 4001)
        sigma = -q2 * (fpp + c.epsilon**2 * q2) - c.alpha
        i = int(np.argmax(sigma))
        return float(sigma[i]), float(np.sqrt(q2[i]))

    def rhs_rms(self, u: np.ndarray, m: float) -> float:
        c = self.cfg
        hat = np.fft.fft2(u)
        mu = fprime(u, c.barrier) + c.epsilon**2 * np.fft.ifft2(self.k2 * hat).real
        rhs = np.fft.ifft2(-self.k2 * np.fft.fft2(mu)).real - c.alpha * (u - m)
        return float(np.sqrt(np.mean(rhs * rhs)))

    def evolve(self, u: np.ndarray, m: float, steps: int, snapshot_steps=()) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        c = self.cfg
        n2 = c.size * c.size
        snaps: dict[int, np.ndarray] = {}
        want = set(snapshot_steps)
        for i in range(1, steps + 1):
            nonlinear = fprime(u, c.barrier) - c.stabilizer * u
            hat = (np.fft.fft2(u) - c.dt * self.k2 * np.fft.fft2(nonlinear)) / self.denominator
            hat[0, 0] = m * n2
            u = np.fft.ifft2(hat).real
            if i in want:
                snaps[i] = u.copy()
        if not np.isfinite(u).all():
            raise FloatingPointError("nonfinite evolution")
        return u, snaps


def binary_of(u: np.ndarray, cfg: Config) -> np.ndarray:
    return u > cfg.threshold


def measure(u: np.ndarray, cfg: Config) -> dict[str, float]:
    from pit_goal_metrics import measure_goal_pattern

    b = binary_of(u, cfg).astype(np.float32)
    return measure_goal_pattern(b)


def render(binary: np.ndarray, scale: int = 2) -> Image.Image:
    rgb = np.empty((*binary.shape, 3), dtype=np.uint8)
    rgb[:] = (238, 226, 214)
    rgb[binary] = (52, 40, 92)
    img = Image.fromarray(rgb)
    return img.resize((binary.shape[1] * scale, binary.shape[0] * scale), Image.Resampling.NEAREST)


def contact_sheet(grid: list[list[np.ndarray]], col_labels: list[str], row_labels: list[str], path: Path, title: str, scale: int = 1) -> None:
    n = grid[0][0].shape[0] * scale
    pad, head, left = 4, 28, 70
    W = left + len(col_labels) * (n + pad)
    H = head + 16 + len(row_labels) * (n + pad)
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill="black")
    for j, lab in enumerate(col_labels):
        draw.text((left + j * (n + pad) + 4, head), lab, fill="black")
    for i, row in enumerate(grid):
        draw.text((4, head + 16 + i * (n + pad) + n // 2), row_labels[i], fill="black")
        for j, b in enumerate(row):
            sheet.paste(render(b, scale), (left + j * (n + pad), head + 16 + i * (n + pad)))
    sheet.save(path)


def sweep(cfg: Config, means: list[float], seeds: list[int], output: Path, noise: float | None = None, checkpoints=(1, 2)) -> list[dict]:
    """Independent initialisation for every (m, seed); nothing is carried over."""
    from pit_goal_metrics import classify_morphology

    output.mkdir(parents=True, exist_ok=True)
    model = AsymmetricOK(cfg)
    rows: list[dict] = []
    grid: list[list[np.ndarray]] = []
    fields = {}
    T = cfg.steps
    snap_steps = [T * c for c in checkpoints]
    for seed in seeds:
        row_imgs = []
        for m in means:
            u0 = model.initial(m, seed, noise)
            u, snaps = model.evolve(u0, m, max(snap_steps), snapshot_steps=snap_steps)
            for c, s in zip(checkpoints, snap_steps):
                met = measure(snaps[s], cfg)
                rec = dict(m=float(m), seed=int(seed), checkpoint=f"{c}T", step=int(s),
                           label=classify_morphology(met), coverage=float(binary_of(snaps[s], cfg).mean()),
                           rhs_rms=model.rhs_rms(snaps[s], m), **met)
                rows.append(rec)
            final = snaps[snap_steps[-1]]
            rows[-1]["binary_change_vs_first"] = float(np.mean(binary_of(final, cfg) != binary_of(snaps[snap_steps[0]], cfg)))
            fields[f"s{seed}_m{m:.4f}"] = final.astype(np.float32)
            row_imgs.append(binary_of(final, cfg))
            print(json.dumps({k: rows[-1][k] for k in ("m", "seed", "checkpoint", "label", "coverage",
                                                     "feature_width_median", "branch_points",
                                                     "longest_component_fraction")}), flush=True)
        grid.append(row_imgs)
    np.savez_compressed(output / "fields.npz", **fields)
    keys = sorted({k for r in rows for k in r})
    with (output / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (output / "config.json").write_text(json.dumps(dict(config=asdict(cfg), means=means, seeds=seeds,
                                                        noise=noise if noise is not None else cfg.noise,
                                                        checkpoints=list(checkpoints),
                                                        linear=[dict(m=m, sigma_q=model.linear_growth(m)) for m in means]), indent=2))
    contact_sheet(grid, [f"m={m:.3f}" for m in means], [f"seed {s}" for s in seeds], output / "sheet.png",
                  f"asymmetric OK  eps={cfg.epsilon} alpha={cfg.alpha} a={cfg.barrier}  t={max(snap_steps)*cfg.dt:g}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--means", type=float, nargs="+", default=[0.22, 0.25, 0.28, 0.30, 0.32, 0.34, 0.37, 0.40])
    s.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    s.add_argument("--noise", type=float, default=None)
    s.add_argument("--checkpoints", type=int, nargs="+", default=[1, 2])
    for name, typ in (("size", int), ("steps", int), ("dt", float), ("dx", float), ("epsilon", float),
                      ("alpha", float), ("barrier", float)):
        s.add_argument("--" + name, type=typ, default=getattr(Config(), name))
    a = p.parse_args()
    cfg = Config(size=a.size, steps=a.steps, dt=a.dt, dx=a.dx, epsilon=a.epsilon, alpha=a.alpha, barrier=a.barrier)
    sweep(cfg, a.means, a.seeds, a.output, a.noise, tuple(a.checkpoints))


if __name__ == "__main__":
    main()
