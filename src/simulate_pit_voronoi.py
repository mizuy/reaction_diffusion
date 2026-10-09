#!/usr/bin/env python3
"""Generate Kudo pit-like patterns with a cell-based Voronoi/vertex model.

Epithelial cells are represented as discrete agents with area elasticity and
pairwise repulsion. Proliferation and division create a dynamic Voronoi packing;
pit openings are read from the edge-distance field (Voronoi ridges / triple
junctions), matching the honeycomb orifice layout described for colonic crypts.

Example:
    python3 src/simulate_pit_voronoi.py --output artifacts/pit_voronoi
"""

from __future__ import annotations

import argparse
import math
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
from pit_fission import apply_component_fission


@dataclass
class Cell:
    x: float
    y: float
    area: float
    target_area: float
    prolif: float
    alive: bool = True


@dataclass(frozen=True)
class VoronoiConfig:
    name: str
    description: str
    target_area: float
    repulsion: float
    prolif_mean: float
    spacing: float | None = None
    anisotropy: float = 0.0
    prolif_heterogeneity: float = 0.0
    max_cells: int = 900
    t1_threshold: float = 0.0
    t1_strength: float = 0.0


@dataclass
class VoronoiResult:
    config: VoronoiConfig
    pit: np.ndarray
    snapshots: list[tuple[int, np.ndarray]]


def hex_lattice(
    rng: np.random.Generator,
    *,
    width: float,
    height: float,
    spacing: float,
    target_area: float,
    prolif_mean: float,
    prolif_heterogeneity: float,
) -> list[Cell]:
    cells: list[Cell] = []
    row = 0
    y = spacing
    while y < height - spacing:
        offset = 0.5 * spacing if row % 2 else 0.0
        x = spacing + offset
        while x < width - spacing:
            jitter = 0.08 * spacing
            prolif = prolif_mean
            if prolif_heterogeneity > 0.0:
                prolif *= 1.0 + prolif_heterogeneity * rng.normal(0.0, 1.0)
            prolif = max(0.05, prolif)
            cells.append(
                Cell(
                    x=float(x + rng.uniform(-jitter, jitter)),
                    y=float(y + rng.uniform(-jitter, jitter)),
                    area=target_area,
                    target_area=target_area,
                    prolif=prolif,
                )
            )
            x += spacing
        y += spacing * math.sqrt(3.0) / 2.0
        row += 1
    return cells


def relax_cells(
    cells: list[Cell],
    *,
    repulsion: float,
    anisotropy: float,
    dt: float,
    margin: float,
    domain: float,
) -> None:
    cutoff = domain * 0.22
    cutoff2 = cutoff * cutoff
    for i, ci in enumerate(cells):
        if not ci.alive:
            continue
        fx = fy = 0.0
        ri = math.sqrt(max(ci.area, 1.0e-6) / math.pi)

        for wall_x in (margin, domain - margin):
            dx = ci.x - wall_x
            if abs(dx) < ri + 2.0:
                push = (ri + 2.0 - abs(dx)) * repulsion
                fx += -math.copysign(push, dx)
        for wall_y in (margin, domain - margin):
            dy = ci.y - wall_y
            if abs(dy) < ri + 2.0:
                push = (ri + 2.0 - abs(dy)) * repulsion
                fy += -math.copysign(push, dy)

        for j, cj in enumerate(cells):
            if i == j or not cj.alive:
                continue
            dx = ci.x - cj.x
            dy = ci.y - cj.y
            if anisotropy != 0.0:
                dx *= 1.0 + anisotropy
                dy /= 1.0 + anisotropy
            dist2 = dx * dx + dy * dy
            if dist2 > cutoff2 or dist2 < 1.0e-8:
                continue
            dist = math.sqrt(dist2)
            rj = math.sqrt(max(cj.area, 1.0e-6) / math.pi)
            overlap = (ri + rj) - dist
            if overlap > 0.0:
                scale = repulsion * overlap / dist
                fx += scale * dx
                fy += scale * dy

        ci.x += dt * fx
        ci.y += dt * fy
        ci.x = min(domain - margin, max(margin, ci.x))
        ci.y = min(domain - margin, max(margin, ci.y))


def divide_cells(
    rng: np.random.Generator,
    cells: list[Cell],
    *,
    dt: float,
    max_cells: int,
) -> None:
    new_cells: list[Cell] = []
    for cell in cells:
        if not cell.alive:
            continue
        if len(cells) + len(new_cells) >= max_cells:
            continue
        if cell.area <= 1.35 * cell.target_area:
            continue
        if rng.random() > cell.prolif * dt:
            continue
        angle = rng.uniform(0.0, 2.0 * math.pi)
        offset = 0.35 * math.sqrt(cell.area / math.pi)
        daughter = Cell(
            x=cell.x + offset * math.cos(angle),
            y=cell.y + offset * math.sin(angle),
            area=cell.target_area,
            target_area=cell.target_area,
            prolif=cell.prolif,
        )
        cell.area = cell.target_area
        cell.x -= 0.5 * offset * math.cos(angle)
        cell.y -= 0.5 * offset * math.sin(angle)
        new_cells.append(daughter)
    cells.extend(new_cells)


def t1_rearrange_cells(cells: list[Cell], *, threshold: float, strength: float, domain: float) -> None:
    """Approximate a vertex-model T1 transition in a cell-centre model.

    When two centers become too close, the collapsed contact is reopened in the
    perpendicular direction. This prevents long distorted corridors from
    remaining as a single stable neighbor chain.
    """

    if threshold <= 0.0 or strength <= 0.0:
        return
    active = [index for index, cell in enumerate(cells) if cell.alive]
    threshold2 = threshold * threshold
    margin = 4.0
    for a_pos, i in enumerate(active):
        ci = cells[i]
        for j in active[a_pos + 1 :]:
            cj = cells[j]
            dx = ci.x - cj.x
            dy = ci.y - cj.y
            dist2 = dx * dx + dy * dy
            if dist2 <= 1.0e-8 or dist2 > threshold2:
                continue
            dist = math.sqrt(dist2)
            nx = -dy / dist
            ny = dx / dist
            amount = 0.5 * strength * (threshold - dist)
            ci.x += amount * nx
            ci.y += amount * ny
            cj.x -= amount * nx
            cj.y -= amount * ny
            ci.x = min(domain - margin, max(margin, ci.x))
            ci.y = min(domain - margin, max(margin, ci.y))
            cj.x = min(domain - margin, max(margin, cj.x))
            cj.y = min(domain - margin, max(margin, cj.y))


def pit_field_from_cells(cells: list[Cell], size: int) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    alive = [c for c in cells if c.alive]
    if not alive:
        return np.zeros((size, size), dtype=np.float32)

    centers = np.array([[c.x, c.y] for c in alive], dtype=np.float32)
    d2 = (xs[..., None] - centers[None, None, :, 0]) ** 2 + (
        ys[..., None] - centers[None, None, :, 1]
    ) ** 2
    d2_sorted = np.sort(d2, axis=-1)
    d1 = np.sqrt(d2_sorted[..., 0] + 1.0e-8)
    d2n = np.sqrt(d2_sorted[..., 1] + 1.0e-8)
    d3n = np.sqrt(d2_sorted[..., min(2, len(alive) - 1)] + 1.0e-8)

    edge = d2n - d1
    vertex = np.minimum(edge, d3n - d2n)
    pit = 0.55 * edge + 0.45 * vertex

    # Suppress boundary artefacts where only one neighbour exists near the domain edge.
    valid = d2_sorted[..., 1] < (0.35 * size) ** 2
    pit = np.where(valid, pit, 0.0)
    pit /= float(np.percentile(pit, 99)) + 1.0e-6
    return np.clip(pit, 0.0, 1.0).astype(np.float32)


def run_voronoi(
    config: VoronoiConfig,
    *,
    size: int,
    steps: int,
    dt: float,
    seed: int,
    snapshot_count: int,
) -> VoronoiResult:
    rng = np.random.default_rng(seed)
    spacing = config.spacing or math.sqrt(config.target_area * 1.15)
    cells = hex_lattice(
        rng,
        width=float(size),
        height=float(size),
        spacing=spacing,
        target_area=config.target_area,
        prolif_mean=config.prolif_mean,
        prolif_heterogeneity=config.prolif_heterogeneity,
    )

    snapshot_steps = set(np.linspace(0, steps, snapshot_count, dtype=int).tolist())
    snapshots: list[tuple[int, np.ndarray]] = []
    if 0 in snapshot_steps:
        snapshots.append((0, pit_field_from_cells(cells, size)))

    for step in range(1, steps + 1):
        relax_cells(
            cells,
            repulsion=config.repulsion,
            anisotropy=config.anisotropy,
            dt=dt,
            margin=4.0,
            domain=float(size),
        )
        if config.t1_strength > 0.0 and step % 10 == 0:
            t1_rearrange_cells(
                cells,
                threshold=config.t1_threshold or 0.7 * spacing,
                strength=config.t1_strength,
                domain=float(size),
            )
        for cell in cells:
            if cell.alive:
                cell.area += 0.08 * config.prolif_mean * dt * cell.target_area
        divide_cells(rng, cells, dt=dt, max_cells=config.max_cells)
        if step in snapshot_steps:
            snapshots.append((step, pit_field_from_cells(cells, size)))

    pit = pit_field_from_cells(cells, size)
    if config.t1_strength > 0.0:
        pit = apply_component_fission(
            pit,
            pit,
            strength=0.35 * config.t1_strength,
            max_area_fraction=0.020,
            max_length_fraction=0.22,
            band_width=max(1.2, spacing / 4.0),
        )
    return VoronoiResult(config=config, pit=pit, snapshots=snapshots)


def build_suite() -> list[VoronoiConfig]:
    return [
        VoronoiConfig(
            name="voronoi_pit_i_regular",
            description="Uniform proliferation: regular honeycomb round pits (Type I).",
            target_area=28.0,
            repulsion=0.55,
            prolif_mean=0.35,
            spacing=6.2,
        ),
        VoronoiConfig(
            name="voronoi_pit_iii_s_dense",
            description="Tight packing and high proliferation: small dense pits (Type IIIs).",
            target_area=18.0,
            repulsion=0.65,
            prolif_mean=0.55,
            spacing=4.8,
        ),
        VoronoiConfig(
            name="voronoi_pit_iii_l_large",
            description="Loose packing: larger round/tubular pits (Type IIIL).",
            target_area=42.0,
            repulsion=0.45,
            prolif_mean=0.25,
            spacing=8.0,
        ),
        VoronoiConfig(
            name="voronoi_pit_iv_elongated",
            description="Anisotropic repulsion: elongated ridges and branching edges (Type IV).",
            target_area=28.0,
            repulsion=0.50,
            prolif_mean=0.45,
            spacing=6.0,
            anisotropy=0.55,
            t1_strength=0.9,
        ),
        VoronoiConfig(
            name="voronoi_pit_vi_irregular",
            description="Spatially heterogeneous proliferation: irregular pit sizes (Type VI).",
            target_area=28.0,
            repulsion=0.52,
            prolif_mean=0.40,
            spacing=6.0,
            prolif_heterogeneity=0.85,
            t1_strength=0.7,
        ),
    ]


def render_pit(pit: np.ndarray) -> np.ndarray:
    field = normalize_u8(pit)
    return cv2.applyColorMap(field, cv2.COLORMAP_BONE)


def save_result_images(result: VoronoiResult, metrics: dict[str, float], output: Path) -> None:
    case_dir = output / result.config.name
    case_dir.mkdir(parents=True, exist_ok=True)

    for step, pit in result.snapshots:
        cv2.imwrite(str(case_dir / f"pit_{step:06d}.png"), render_pit(pit))

    final = annotate(
        render_pit(result.pit),
        [
            result.config.name,
            f"branch={metrics['branch_density']:.3f} long={metrics['long_line_score']:.2f}",
            f"irreg={metrics['vi_irregularity_score']:.2f} cov={metrics['binary_coverage']:.2f}",
        ],
    )
    overlay = annotate(
        render_skeleton_overlay(result.pit),
        [
            "skeleton overlay",
            "red=branch cyan=end",
            f"components={metrics['skeleton_components']:.0f}",
        ],
    )
    cv2.imwrite(str(case_dir / "final_pit.png"), final)
    cv2.imwrite(str(case_dir / "final_skeleton.png"), overlay)


def write_contact_sheet(output: Path, results: list[tuple[VoronoiResult, dict[str, float]]]) -> None:
    cells = []
    for result, metrics in results:
        height = annotate(
            render_pit(result.pit),
            [
                result.config.name,
                f"branch {metrics['branch_density']:.3f}",
                f"long {metrics['long_line_score']:.2f}",
            ],
        )
        skeleton = annotate(
            render_skeleton_overlay(result.pit),
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
        description="Generate pit-pattern morphologies with a Voronoi/vertex cell model."
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/pit_voronoi"))
    parser.add_argument("--size", type=int, default=160, help="Simulation width/height.")
    parser.add_argument("--steps", type=int, default=400, help="Mechanical steps per case.")
    parser.add_argument("--dt", type=float, default=0.25, help="Relaxation time step.")
    parser.add_argument("--seed", type=int, default=7, help="Seed shared by all cases.")
    parser.add_argument("--snapshot-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    results: list[tuple[VoronoiResult, dict[str, float]]] = []
    for config in build_suite():
        print(f"running {config.name} ...", flush=True)
        result = run_voronoi(
            config,
            size=args.size,
            steps=args.steps,
            dt=args.dt,
            seed=args.seed,
            snapshot_count=args.snapshot_count,
        )
        b = result.pit
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
