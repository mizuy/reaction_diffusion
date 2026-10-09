#!/usr/bin/env python3
"""Crest-groove model: pits open along the grooves between self-organised crests.

Two fields on a periodic domain, both started from small independent noise:

    v_t = r v - (1 + Laplacian/k0^2)^2 v + g v^2 - v^3                (crest relief)
    w_t = D Laplacian w + w - w^3 - beta (v - s)                       (pit opening)

v is a Swift-Hohenberg relief with a quadratic term, so it self-organises into
hexagonal crests (the intervening mucosa).  Its low set is a honeycomb of
grooves: deep wells at triple points and shallower saddles along each edge.
w is a bistable opening switch: a pit (w>0) opens where the local relief lies
below the invagination drive s.  The single control is s (equivalently p).

As s increases, wells open first (round pits, I); an edge opens when its
saddle drops below s, so two wells join (short tube, IIIL); a well with three
open edges is a Y, and finite clusters of open edges are branched trees
(IV_B); once open edges percolate the grooves form a mesh around crest
islands (IV_V).  Groove width is fixed by the crest spacing 2*pi/k0 and the
switch interface; it does not grow with s.  The order in which edges open is
decided by the natural irregularity (grain boundaries, dislocations, spacing
fluctuations) of the self-organised crests, not by an imposed random field.

r, g, k0, D, beta are fixed.  Nothing in the evolution depends on a classifier,
image cutting or a type-specific switch.  The observation threshold is w>0.

Example:
    uv run python src/simulate_pit_groove.py sweep \\
        --output artifacts/pit_groove/explore --seeds 1 7 42
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Config:
    size: int = 256
    dx: float = 1.0
    dt: float = 0.25
    steps: int = 4000
    wavelength: float = 24.0  # crest spacing 2*pi/k0 in grid units
    r: float = 0.1
    g: float = 1.0
    diffusion: float = 1.0
    beta: float = 4.0
    inhibition: float = 8.0  # long-range inhibition of opening by nearby open pit area
    inhibition_length: float = 120.0
    noise: float = 0.05


class GrooveModel:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        k = 2 * np.pi * np.fft.fftfreq(cfg.size, d=cfg.dx)
        self.k2 = k[:, None] ** 2 + k[None, :] ** 2
        k0sq = (2 * np.pi / cfg.wavelength) ** 2
        lin_v = cfg.r - (1 - self.k2 / k0sq) ** 2
        self.stab = 1.0
        self.den_v = 1 - cfg.dt * (lin_v - self.stab)
        self.stab_w = 4.0
        self.den_w = 1 + cfg.dt * (cfg.diffusion * self.k2 + self.stab_w)
        self.screen = 1.0 / (1.0 + cfg.inhibition_length**2 * self.k2)

    def initial(self, seed_v: int, seed_w: int, noise: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        amp = self.cfg.noise if noise is None else noise
        n = self.cfg.size
        v = amp * np.random.default_rng(seed_v).normal(size=(n, n))
        w = amp * np.random.default_rng(seed_w).normal(size=(n, n))
        return v, w

    def step(self, v: np.ndarray, w: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        nv = c.g * v * v - v**3 + self.stab * v
        v = np.fft.ifft2((np.fft.fft2(v) + c.dt * np.fft.fft2(nv)) / self.den_v).real
        nw = w - w**3 - c.beta * (v - s) + self.stab_w * w
        if c.inhibition:
            nw -= c.inhibition * self.inhibitor(w)
        w = np.fft.ifft2((np.fft.fft2(w) + c.dt * np.fft.fft2(nw)) / self.den_w).real
        return v, w

    def inhibitor(self, w: np.ndarray) -> np.ndarray:
        """Screened local average of open pit area, (1 - l^2 Laplacian) psi = (1 + w) / 2."""
        return np.fft.ifft2(np.fft.fft2(0.5 * (1.0 + w)) * self.screen).real

    def evolve(self, v, w, s, steps, snapshot_steps=()):
        want = set(snapshot_steps)
        snaps = {}
        for i in range(1, steps + 1):
            v, w = self.step(v, w, s)
            if i in want:
                snaps[i] = (v.copy(), w.copy())
        if not (np.isfinite(v).all() and np.isfinite(w).all()):
            raise FloatingPointError("nonfinite evolution")
        return v, w, snaps


def pit_binary(w: np.ndarray) -> np.ndarray:
    return w > 0.0


def measure(w: np.ndarray, wavelength: float) -> dict[str, float]:
    """Project goal metrics with the fork arm length tied to the crest spacing.

    The project default (size // 32) depends on the domain size, not on the
    pattern scale; pinning it to wavelength/3 keeps branch counts comparable
    across domain sizes and resolutions.
    """
    import evaluate_pit_steps
    import pit_goal_metrics

    arm = max(4, int(round(wavelength / 3.0)))
    saved = (evaluate_pit_steps.default_min_branch_arm_length, pit_goal_metrics.default_min_branch_arm_length)
    evaluate_pit_steps.default_min_branch_arm_length = lambda size: arm
    pit_goal_metrics.default_min_branch_arm_length = lambda size: arm
    try:
        met = pit_goal_metrics.measure_goal_pattern(pit_binary(w).astype(np.float32))
    finally:
        evaluate_pit_steps.default_min_branch_arm_length, pit_goal_metrics.default_min_branch_arm_length = saved
    from pit_fork_metrics import fork_metrics

    met.update(fork_metrics(pit_binary(w), arm_min=wavelength / 4.0))
    return met


def render(binary: np.ndarray, scale: int = 1) -> Image.Image:
    rgb = np.empty((*binary.shape, 3), dtype=np.uint8)
    rgb[:] = (238, 226, 214)
    rgb[binary] = (52, 40, 92)
    img = Image.fromarray(rgb)
    if scale != 1:
        img = img.resize((binary.shape[1] * scale, binary.shape[0] * scale), Image.Resampling.NEAREST)
    return img


def contact_sheet(grid, col_labels, row_labels, path: Path, title: str, scale: int = 1, crop: int | None = None) -> None:
    def prep(b):
        return b[:crop, :crop] if crop else b
    n = prep(grid[0][0]).shape[0] * scale
    pad, head, left = 4, 30, 70
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
            sheet.paste(render(prep(b), scale), (left + j * (n + pad), head + 16 + i * (n + pad)))
    sheet.save(path)


def sweep(cfg: Config, drives: list[float], seeds: list[int], output: Path, *, noise: float | None = None,
          checkpoints=(1, 2), seed_w_offset: int = 100000, crop: int | None = None, scale: int = 1) -> list[dict]:
    """Independent initialisation for every (s, seed): nothing is carried over."""
    from pit_goal_metrics import classify_morphology

    output.mkdir(parents=True, exist_ok=True)
    model = GrooveModel(cfg)
    rows: list[dict] = []
    grid = []
    fields = {}
    T = cfg.steps
    snap_steps = [T * c for c in checkpoints]
    for seed in seeds:
        row_imgs = []
        for s in drives:
            v, w = model.initial(seed, seed + seed_w_offset, noise)
            v, w, snaps = model.evolve(v, w, s, max(snap_steps), snapshot_steps=snap_steps)
            first = None
            for c, st in zip(checkpoints, snap_steps):
                sv, sw = snaps[st]
                met = measure(sw, cfg.wavelength / cfg.dx)
                b = pit_binary(sw)
                rec = dict(s=float(s), seed=int(seed), checkpoint=f"{c}T", step=int(st),
                           label=classify_morphology(met), coverage=float(b.mean()), **met)
                if first is None:
                    first = b
                    rec["binary_change_vs_1T"] = 0.0
                else:
                    rec["binary_change_vs_1T"] = float(np.mean(b != first))
                rows.append(rec)
            fields[f"seed{seed}_s{s:+.4f}_w"] = w.astype(np.float32)
            fields[f"seed{seed}_s{s:+.4f}_v"] = v.astype(np.float32)
            row_imgs.append(pit_binary(w))
            print(json.dumps({k: rows[-1][k] for k in ("s", "seed", "checkpoint", "label", "coverage",
                                                     "feature_width_median", "feature_width_p90", "branch_points",
                                                     "longest_component_fraction", "background_island_count",
                                                     "binary_change_vs_1T")}), flush=True)
        grid.append(row_imgs)
    np.savez_compressed(output / "fields.npz", **fields)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (output / "metrics.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    (output / "config.json").write_text(json.dumps(dict(config=asdict(cfg), drives=drives, seeds=seeds,
                                                        seed_w_offset=seed_w_offset,
                                                        noise=noise if noise is not None else cfg.noise,
                                                        checkpoints=list(checkpoints)), indent=2))
    contact_sheet(grid, [f"s={s:+.3f}" for s in drives], [f"seed {sd}" for sd in seeds], output / "sheet.png",
                  f"crest-groove  r={cfg.r} g={cfg.g} lambda={cfg.wavelength} D={cfg.diffusion} beta={cfg.beta}  "
                  f"t={max(snap_steps) * cfg.dt:g}", scale=scale, crop=crop)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--drives", type=float, nargs="+", default=[-0.42, -0.38, -0.34, -0.31, -0.28, -0.25, -0.22])
    s.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42])
    s.add_argument("--noise", type=float, default=None)
    s.add_argument("--checkpoints", type=int, nargs="+", default=[1, 2])
    s.add_argument("--crop", type=int, default=None)
    s.add_argument("--scale", type=int, default=1)
    for name in ("size", "steps"):
        s.add_argument("--" + name, type=int, default=getattr(Config(), name))
    for name in ("dt", "dx", "wavelength", "r", "g", "diffusion", "beta", "inhibition", "inhibition_length"):
        s.add_argument("--" + name, type=float, default=getattr(Config(), name))
    a = p.parse_args()
    cfg = replace(Config(), **{k: getattr(a, k) for k in ("size", "steps", "dt", "dx", "wavelength", "r", "g",
                                                          "diffusion", "beta", "inhibition",
                                                          "inhibition_length")})
    sweep(cfg, a.drives, a.seeds, a.output, noise=a.noise, checkpoints=tuple(a.checkpoints), crop=a.crop,
          scale=a.scale)


if __name__ == "__main__":
    main()
