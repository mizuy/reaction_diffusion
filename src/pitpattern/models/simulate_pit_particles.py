#!/usr/bin/env python3
"""Off-lattice E/S particle proof of concept for pit-pattern progression.

Epithelial (E) and stromal (S) particles interact through type-dependent Morse
potentials on a periodic box. A single progression parameter raises the E
division rate and the pressure-gated S removal rate. Positions are relaxed by
overdamped Euler steps. The morphology field used by the shared metrics is E occupancy,
so isolated disks remain comparable to the CPM PoC. The E–S interface
is still rasterized at a fixed width and recorded, but not used as the
classifier input: a ring around every disk would look like a labyrinth.

Polarity / buckling forces are deferred. This PoC tests whether proliferation
pressure plus volume exclusion plus stromal clearance can change topology.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pitpattern.metrics.pit_goal_metrics import classify_morphology, measure_goal_pattern

KIND_S = 0
KIND_E = 1


@dataclass(frozen=True)
class ParticleConfig:
    cluster_spacing: float = 14.0
    cluster_radius: float = 3.2
    particle_spacing: float = 2.2
    well_ee: float = 1.8
    well_ss: float = 0.9
    well_es: float = 0.22
    width_ee: float = 1.8
    width_ss: float = 1.8
    width_es: float = 2.2
    rest_ee: float = 2.2
    rest_ss: float = 2.2
    rest_es: float = 2.6
    cutoff: float = 5.5
    gamma: float = 1.0
    noise: float = 0.06
    max_step: float = 0.32
    k_div_max: float = 0.45
    k_death_max: float = 0.22
    pressure_threshold: float = 0.85
    max_epithelial_fraction: float = 0.55
    min_stroma_count: int = 48
    max_local_neighbors: int = 6
    raster_radius: float = 1.55
    interface_width: float = 1.8
    max_particles: int = 1600


@dataclass(frozen=True)
class ParticleRates:
    progression: float
    division: float
    stroma_death: float


@dataclass
class ParticleState:
    xy: np.ndarray
    kind: np.ndarray
    pressure: np.ndarray

    @property
    def epithelial_count(self) -> int:
        return int(np.count_nonzero(self.kind == KIND_E))

    @property
    def stroma_count(self) -> int:
        return int(np.count_nonzero(self.kind == KIND_S))


def smoothstep(value: float) -> float:
    p = float(np.clip(value, 0.0, 1.0))
    return p * p * (3.0 - 2.0 * p)


def wrap_delta(delta: np.ndarray | float, domain: float) -> np.ndarray | float:
    return delta - domain * np.round(np.asarray(delta) / domain)


def wrap_position(xy: np.ndarray, domain: float) -> np.ndarray:
    return np.mod(xy, domain)


def morse_energy(distance: np.ndarray | float, well: float, width: float, rest: float) -> np.ndarray | float:
    expo = np.exp(-width * (np.asarray(distance) - rest))
    return well * (1.0 - expo) ** 2 - well


def morse_force_magnitude(distance: np.ndarray | float, well: float, width: float, rest: float) -> np.ndarray | float:
    """Return dU/dr. Force on i is -(dU/dr) * r_hat_ij."""
    expo = np.exp(-width * (np.asarray(distance) - rest))
    return 2.0 * well * width * expo * (1.0 - expo)


def pair_morse(kind_a: int, kind_b: int, config: ParticleConfig) -> tuple[float, float, float]:
    if kind_a == kind_b == KIND_E:
        return config.well_ee, config.width_ee, config.rest_ee
    if kind_a == kind_b == KIND_S:
        return config.well_ss, config.width_ss, config.rest_ss
    return config.well_es, config.width_es, config.rest_es


def rates_from_progression(progression: float, config: ParticleConfig) -> ParticleRates:
    weight = smoothstep(progression)
    return ParticleRates(
        progression=float(np.clip(progression, 0.0, 1.0)),
        division=config.k_div_max * weight,
        stroma_death=config.k_death_max * weight,
    )


def hex_sites(domain: float, spacing: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    row = 0
    y = 0.5 * spacing
    while y < domain:
        x = 0.5 * spacing + (0.5 * spacing if row % 2 else 0.0)
        while x < domain:
            points.append((x, y))
            x += spacing
        y += spacing * math.sqrt(3.0) / 2.0
        row += 1
    return points


def initialize_state(domain: float, config: ParticleConfig, rng: np.random.Generator) -> ParticleState:
    """Place compact E disks on a hex lattice and fill the rest with S."""
    centers = hex_sites(domain, config.cluster_spacing)
    jitter = 0.12 * config.particle_spacing
    epithelial: list[tuple[float, float]] = []
    for cx, cy in centers:
        for x, y in hex_sites(domain, config.particle_spacing):
            dx = float(wrap_delta(x - cx, domain))
            dy = float(wrap_delta(y - cy, domain))
            if dx * dx + dy * dy <= config.cluster_radius * config.cluster_radius:
                epithelial.append(
                    (
                        float((x + rng.uniform(-jitter, jitter)) % domain),
                        float((y + rng.uniform(-jitter, jitter)) % domain),
                    )
                )

    keepout = config.cluster_radius + 0.45 * config.rest_es
    stroma: list[tuple[float, float]] = []
    for x, y in hex_sites(domain, config.particle_spacing):
        too_close = False
        for cx, cy in centers:
            dx = float(wrap_delta(x - cx, domain))
            dy = float(wrap_delta(y - cy, domain))
            if dx * dx + dy * dy <= keepout * keepout:
                too_close = True
                break
        if too_close:
            continue
        stroma.append(
            (
                float((x + rng.uniform(-jitter, jitter)) % domain),
                float((y + rng.uniform(-jitter, jitter)) % domain),
            )
        )

    xy = np.asarray(epithelial + stroma, dtype=np.float64)
    kind = np.concatenate(
        [
            np.full(len(epithelial), KIND_E, dtype=np.uint8),
            np.full(len(stroma), KIND_S, dtype=np.uint8),
        ]
    )
    if len(xy) > config.max_particles:
        raise RuntimeError(f"too many particles: {len(xy)} > {config.max_particles}")
    return ParticleState(
        xy=xy,
        kind=kind,
        pressure=np.zeros(len(xy), dtype=np.float64),
    )


def _cell_index(xy: np.ndarray, domain: float, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    cell = domain / n_cells
    ix = np.floor(xy[:, 0] / cell).astype(np.int32) % n_cells
    iy = np.floor(xy[:, 1] / cell).astype(np.int32) % n_cells
    return ix, iy


def compute_forces(
    state: ParticleState,
    *,
    domain: float,
    config: ParticleConfig,
) -> tuple[np.ndarray, float]:
    """Return per-particle forces and the total Morse energy."""
    n = len(state.xy)
    forces = np.zeros((n, 2), dtype=np.float64)
    pressure = np.zeros(n, dtype=np.float64)
    energy = 0.0
    n_cells = max(3, int(math.floor(domain / config.cutoff)))
    ix, iy = _cell_index(state.xy, domain, n_cells)
    buckets: list[list[int]] = [[] for _ in range(n_cells * n_cells)]
    for index, (cx, cy) in enumerate(zip(ix, iy, strict=True)):
        buckets[int(cy) * n_cells + int(cx)].append(index)

    cutoff2 = config.cutoff * config.cutoff
    offsets = (-1, 0, 1)
    for i in range(n):
        cx = int(ix[i])
        cy = int(iy[i])
        xi = state.xy[i, 0]
        yi = state.xy[i, 1]
        ki = int(state.kind[i])
        for oy in offsets:
            for ox in offsets:
                bucket = buckets[((cy + oy) % n_cells) * n_cells + ((cx + ox) % n_cells)]
                for j in bucket:
                    if j <= i:
                        continue
                    dx = float(wrap_delta(xi - state.xy[j, 0], domain))
                    dy = float(wrap_delta(yi - state.xy[j, 1], domain))
                    dist2 = dx * dx + dy * dy
                    if dist2 < 1.0e-12 or dist2 > cutoff2:
                        continue
                    dist = math.sqrt(dist2)
                    well, width, rest = pair_morse(ki, int(state.kind[j]), config)
                    slope = float(morse_force_magnitude(dist, well, width, rest))
                    pair_energy = float(morse_energy(dist, well, width, rest))
                    scale = -slope / dist
                    fx = scale * dx
                    fy = scale * dy
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    energy += pair_energy
                    if dist < rest:
                        overlap = well * (1.0 - math.exp(-width * (dist - rest))) ** 2
                        pressure[i] += overlap
                        pressure[j] += overlap

    state.pressure = pressure
    return forces, energy


def integrate(
    state: ParticleState,
    *,
    domain: float,
    config: ParticleConfig,
    dt: float,
    rng: np.random.Generator,
) -> float:
    forces, energy = compute_forces(state, domain=domain, config=config)
    step = (dt / config.gamma) * forces
    if config.noise > 0.0:
        step += config.noise * math.sqrt(dt) * rng.normal(0.0, 1.0, size=state.xy.shape)
    speed = np.linalg.norm(step, axis=1, keepdims=True)
    too_fast = speed > config.max_step
    if np.any(too_fast):
        step = np.where(too_fast, step * (config.max_step / np.maximum(speed, 1.0e-12)), step)
    state.xy = wrap_position(state.xy + step, domain)
    return energy


def neighbor_count(
    state: ParticleState,
    index: int,
    *,
    domain: float,
    radius: float,
) -> int:
    dx = wrap_delta(state.xy[index, 0] - state.xy[:, 0], domain)
    dy = wrap_delta(state.xy[index, 1] - state.xy[:, 1], domain)
    same = state.kind == state.kind[index]
    near = (dx * dx + dy * dy) < radius * radius
    return int(np.count_nonzero(same & near)) - 1


def divide_epithelial(
    state: ParticleState,
    *,
    domain: float,
    config: ParticleConfig,
    rates: ParticleRates,
    dt: float,
    rng: np.random.Generator,
) -> int:
    if rates.division <= 0.0:
        return 0
    if state.epithelial_count / max(1, len(state.kind)) >= config.max_epithelial_fraction:
        return 0
    if len(state.kind) >= config.max_particles:
        return 0

    parents = np.flatnonzero(state.kind == KIND_E)
    born_xy: list[list[float]] = []
    offset = 0.35 * config.rest_ee
    for index in parents:
        if len(state.kind) + len(born_xy) >= config.max_particles:
            break
        if rng.random() > rates.division * dt:
            continue
        if neighbor_count(state, int(index), domain=domain, radius=config.rest_ee * 1.15) >= config.max_local_neighbors:
            continue
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        shift = np.array([offset * math.cos(angle), offset * math.sin(angle)])
        state.xy[index] = wrap_position(state.xy[index] - 0.5 * shift, domain)
        born_xy.append(wrap_position(state.xy[index] + shift, domain).tolist())
        if (state.epithelial_count + len(born_xy)) / (len(state.kind) + len(born_xy)) >= config.max_epithelial_fraction:
            break

    if not born_xy:
        return 0
    added = np.asarray(born_xy, dtype=np.float64)
    state.xy = np.vstack([state.xy, added])
    state.kind = np.concatenate([state.kind, np.full(len(added), KIND_E, dtype=np.uint8)])
    state.pressure = np.concatenate([state.pressure, np.zeros(len(added), dtype=np.float64)])
    return len(added)


def remove_stroma(
    state: ParticleState,
    *,
    config: ParticleConfig,
    rates: ParticleRates,
    dt: float,
    rng: np.random.Generator,
) -> int:
    if rates.stroma_death <= 0.0:
        return 0
    if state.stroma_count <= config.min_stroma_count:
        return 0
    candidates = np.flatnonzero(
        (state.kind == KIND_S) & (state.pressure >= config.pressure_threshold)
    )
    if candidates.size == 0:
        return 0
    drop = candidates[rng.random(len(candidates)) < rates.stroma_death * dt]
    keep_limit = state.stroma_count - config.min_stroma_count
    if drop.size > keep_limit:
        drop = drop[:keep_limit]
    if drop.size == 0:
        return 0
    keep = np.ones(len(state.kind), dtype=bool)
    keep[drop] = False
    state.xy = state.xy[keep]
    state.kind = state.kind[keep]
    state.pressure = state.pressure[keep]
    return int(drop.size)


def _stamp_disks(
    positions: np.ndarray,
    *,
    size: int,
    domain: float,
    radius: float,
) -> np.ndarray:
    field = np.zeros((size, size), dtype=np.float32)
    if len(positions) == 0:
        return field
    scale = size / domain
    radius_px = max(1.0, radius * scale)
    kernel_r = int(math.ceil(2.6 * radius_px))
    yy, xx = np.mgrid[-kernel_r : kernel_r + 1, -kernel_r : kernel_r + 1]
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * radius_px * radius_px)).astype(np.float32)
    for x, y in positions:
        cx = int(round(float(x) * scale)) % size
        cy = int(round(float(y) * scale)) % size
        y0, x0 = cy - kernel_r, cx - kernel_r
        for dy in range(kernel.shape[0]):
            dest_y = (y0 + dy) % size
            dest_x0 = x0 % size
            if dest_x0 + kernel.shape[1] <= size:
                field[dest_y, dest_x0 : dest_x0 + kernel.shape[1]] += kernel[dy]
            else:
                first = size - dest_x0
                field[dest_y, dest_x0:] += kernel[dy, :first]
                field[dest_y, : kernel.shape[1] - first] += kernel[dy, first:]
    return np.clip(field, 0.0, 1.0)


def rasterize_occupancy(state: ParticleState, *, size: int, domain: float, config: ParticleConfig) -> np.ndarray:
    epithelial = state.xy[state.kind == KIND_E]
    return _stamp_disks(epithelial, size=size, domain=domain, radius=config.raster_radius)


def rasterize_interface(state: ParticleState, *, size: int, domain: float, config: ParticleConfig) -> np.ndarray:
    """Fixed-width E–S contact band. Width does not depend on p."""
    epithelial = _stamp_disks(
        state.xy[state.kind == KIND_E],
        size=size,
        domain=domain,
        radius=config.interface_width,
    )
    stroma = _stamp_disks(
        state.xy[state.kind == KIND_S],
        size=size,
        domain=domain,
        radius=config.interface_width,
    )
    return np.clip(epithelial * stroma * 4.0, 0.0, 1.0)


def render_state(
    state: ParticleState,
    *,
    size: int,
    domain: float,
    progression: float,
    label: str,
    config: ParticleConfig,
) -> np.ndarray:
    occupancy = rasterize_occupancy(state, size=size, domain=domain, config=config)
    image = np.full((size, size, 3), 242, dtype=np.uint8)
    image[occupancy > 0.35] = (69, 92, 184)
    boundary = (
        (occupancy > 0.35) != np.roll(occupancy > 0.35, 1, axis=0)
    ) | ((occupancy > 0.35) != np.roll(occupancy > 0.35, 1, axis=1))
    image[boundary] = (32, 32, 32)
    cv2.putText(
        image,
        f"p={progression:.2f}  {label}",
        (5, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return image


def run_progression(
    *,
    size: int,
    progressions: list[float],
    steps_per_stage: int,
    dt: float,
    seed: int,
    config: ParticleConfig,
) -> tuple[ParticleState, list[dict[str, object]], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    domain = float(size)
    state = initialize_state(domain, config, rng)
    rows: list[dict[str, object]] = []
    images: list[np.ndarray] = []

    for progression in progressions:
        rates = rates_from_progression(progression, config)
        energies: list[float] = []
        divided = 0
        removed = 0
        for _ in range(steps_per_stage):
            energies.append(integrate(state, domain=domain, config=config, dt=dt, rng=rng))
            divided += divide_epithelial(
                state,
                domain=domain,
                config=config,
                rates=rates,
                dt=dt,
                rng=rng,
            )
            removed += remove_stroma(state, config=config, rates=rates, dt=dt, rng=rng)

        occupancy = rasterize_occupancy(state, size=size, domain=domain, config=config)
        interface = rasterize_interface(state, size=size, domain=domain, config=config)
        metrics = measure_goal_pattern(occupancy)
        morphology = classify_morphology(metrics)
        rows.append(
            {
                "progression": progression,
                "morphology": morphology,
                "epithelial_count": state.epithelial_count,
                "stroma_count": state.stroma_count,
                "epithelial_fraction": float(state.epithelial_count / max(1, len(state.kind))),
                "occupancy_fraction": float(np.mean(occupancy > 0.35)),
                "interface_fraction": float(np.mean(interface)),
                "mean_epithelial_pressure": float(np.mean(state.pressure[state.kind == KIND_E]))
                if state.epithelial_count
                else 0.0,
                "mean_stromal_pressure": float(np.mean(state.pressure[state.kind == KIND_S]))
                if state.stroma_count
                else 0.0,
                "energy_mean": float(np.mean(energies)),
                "divided": divided,
                "removed_stroma": removed,
                "division_rate": rates.division,
                "stroma_death_rate": rates.stroma_death,
                **metrics,
            }
        )
        images.append(
            render_state(
                state,
                size=size,
                domain=domain,
                progression=progression,
                label=morphology,
                config=config,
            )
        )
    return state, rows, images


def parse_progressions(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("at least one progression value is required")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("progression values must be in [0, 1]")
    if values != sorted(values):
        raise ValueError("progression values must be nondecreasing")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal E/S Morse particle PoC.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pit_particles"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps-per-stage", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--progressions", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progressions = parse_progressions(args.progressions)
    config = ParticleConfig()
    args.output.mkdir(parents=True, exist_ok=True)

    state, rows, images = run_progression(
        size=args.size,
        progressions=progressions,
        steps_per_stage=args.steps_per_stage,
        dt=args.dt,
        seed=args.seed,
        config=config,
    )
    for progression, image in zip(progressions, images, strict=True):
        cv2.imwrite(str(args.output / f"particles_p{progression:.2f}.png"), image)
    cv2.imwrite(str(args.output / "contact_sheet.png"), np.concatenate(images, axis=1))
    np.savez(
        args.output / "particles.npz",
        xy=state.xy,
        kind=state.kind,
        pressure=state.pressure,
    )
    with (args.output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "size": args.size,
                "steps_per_stage": args.steps_per_stage,
                "dt": args.dt,
                "stages": rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"wrote {args.output}")
    for row in rows:
        print(
            f"p={row['progression']:.2f} {row['morphology']}: "
            f"E={row['epithelial_count']} S={row['stroma_count']} "
            f"occ={row['occupancy_fraction']:.3f} "
            f"div={row['divided']} death={row['removed_stroma']}"
        )


if __name__ == "__main__":
    main()
