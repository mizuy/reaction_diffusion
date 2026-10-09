#!/usr/bin/env python3
"""Cellular Potts proof of concept for pit-pattern progression.

The model contains individually labelled epithelial domains (E) in one stromal
medium (S). A single progression parameter increases the epithelial target
area and decreases the stromal target area. Metropolis copy attempts relax
contact and area energies without allowing sites to overlap.

This is deliberately a minimal CPM: cell division, explicit stromal cells and
polarity are deferred until the target-area-only hypothesis has been tested.
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


@dataclass(frozen=True)
class CPMConfig:
    spacing: int = 14
    initial_radius: int = 3
    epithelial_growth: float = 4.0
    max_epithelial_fraction: float = 0.58
    contact_ee: float = 3.0
    contact_es: float = 10.0
    lambda_epithelial: float = 1.0
    lambda_stroma: float = 0.01
    temperature: float = 8.0
    min_cell_area: int = 4


@dataclass(frozen=True)
class CPMTargets:
    progression: float
    epithelial: float
    stroma: float


@dataclass
class CPMState:
    labels: np.ndarray
    areas: np.ndarray
    initial_epithelial_area: float

    @property
    def cell_count(self) -> int:
        return len(self.areas) - 1


NEIGHBOR_OFFSETS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1.0 / math.sqrt(2.0)),
    (-1, 1, 1.0 / math.sqrt(2.0)),
    (1, -1, 1.0 / math.sqrt(2.0)),
    (1, 1, 1.0 / math.sqrt(2.0)),
)


def smoothstep(value: float) -> float:
    p = float(np.clip(value, 0.0, 1.0))
    return p * p * (3.0 - 2.0 * p)


def initialize_state(size: int, config: CPMConfig) -> CPMState:
    """Create hexagonally arranged round epithelial domains in stroma."""
    labels = np.zeros((size, size), dtype=np.int32)
    label = 1
    margin = config.spacing // 2
    row = 0
    y = margin
    while y < size - margin:
        offset = config.spacing // 2 if row % 2 else 0
        x = margin + offset
        while x < size - margin:
            cv2.circle(labels, (x, y), config.initial_radius, label, -1)
            label += 1
            x += config.spacing
        y += max(2, int(round(config.spacing * math.sqrt(3.0) / 2.0)))
        row += 1

    areas = np.bincount(labels.ravel(), minlength=label).astype(np.int64)
    mean_area = float(np.mean(areas[1:])) if label > 1 else 0.0
    return CPMState(labels=labels, areas=areas, initial_epithelial_area=mean_area)


def targets_from_progression(
    state: CPMState,
    config: CPMConfig,
    progression: float,
) -> CPMTargets:
    """Map the sole external control p to compatible E/S target areas."""
    requested = state.initial_epithelial_area * (
        1.0 + config.epithelial_growth * smoothstep(progression)
    )
    capacity = (
        config.max_epithelial_fraction * state.labels.size / max(1, state.cell_count)
    )
    epithelial = min(requested, capacity)
    stroma = state.labels.size - state.cell_count * epithelial
    return CPMTargets(
        progression=float(np.clip(progression, 0.0, 1.0)),
        epithelial=float(epithelial),
        stroma=float(stroma),
    )


def contact_coefficient(label_a: int, label_b: int, config: CPMConfig) -> float:
    if label_a == label_b:
        return 0.0
    if label_a == 0 or label_b == 0:
        return config.contact_es
    return config.contact_ee


def _area_delta(
    label: int,
    change: int,
    state: CPMState,
    targets: CPMTargets,
    config: CPMConfig,
) -> float:
    area = int(state.areas[label])
    if label == 0:
        target = targets.stroma
        strength = config.lambda_stroma
    else:
        target = targets.epithelial
        strength = config.lambda_epithelial
    return strength * ((area + change - target) ** 2 - (area - target) ** 2)


def copy_delta_energy(
    state: CPMState,
    *,
    y: int,
    x: int,
    proposed_label: int,
    targets: CPMTargets,
    config: CPMConfig,
) -> float:
    """Energy change if one site at (y, x) adopts proposed_label."""
    old_label = int(state.labels[y, x])
    if old_label == proposed_label:
        return 0.0

    height, width = state.labels.shape
    delta = 0.0
    for dy, dx, weight in NEIGHBOR_OFFSETS:
        neighbor = int(state.labels[(y + dy) % height, (x + dx) % width])
        delta += weight * (
            contact_coefficient(proposed_label, neighbor, config)
            - contact_coefficient(old_label, neighbor, config)
        )
    delta += _area_delta(old_label, -1, state, targets, config)
    delta += _area_delta(proposed_label, 1, state, targets, config)
    return float(delta)


def removal_preserves_local_connectivity(
    labels: np.ndarray,
    *,
    y: int,
    x: int,
    label: int,
) -> bool:
    """Conservatively reject removal of a local epithelial bridge."""
    if label == 0:
        return True
    height, width = labels.shape
    neighbors = {
        ((y + dy) % height, (x + dx) % width)
        for dy, dx, _ in NEIGHBOR_OFFSETS
        if int(labels[(y + dy) % height, (x + dx) % width]) == label
    }
    if len(neighbors) <= 1:
        return True

    pending = [next(iter(neighbors))]
    reached: set[tuple[int, int]] = set()
    while pending:
        cy, cx = pending.pop()
        if (cy, cx) in reached:
            continue
        reached.add((cy, cx))
        for dy, dx, _ in NEIGHBOR_OFFSETS:
            candidate = ((cy + dy) % height, (cx + dx) % width)
            if candidate in neighbors and candidate not in reached:
                pending.append(candidate)
    return reached == neighbors


def monte_carlo_sweep(
    state: CPMState,
    *,
    targets: CPMTargets,
    config: CPMConfig,
    rng: np.random.Generator,
) -> float:
    """Run one MCS and return the accepted-attempt fraction."""
    height, width = state.labels.shape
    accepted = 0
    for _ in range(state.labels.size):
        y = int(rng.integers(height))
        x = int(rng.integers(width))
        dy, dx, _ = NEIGHBOR_OFFSETS[int(rng.integers(len(NEIGHBOR_OFFSETS)))]
        proposed = int(state.labels[(y + dy) % height, (x + dx) % width])
        old = int(state.labels[y, x])
        if proposed == old:
            continue
        if old > 0 and state.areas[old] <= config.min_cell_area:
            continue
        if not removal_preserves_local_connectivity(
            state.labels,
            y=y,
            x=x,
            label=old,
        ):
            continue

        delta = copy_delta_energy(
            state,
            y=y,
            x=x,
            proposed_label=proposed,
            targets=targets,
            config=config,
        )
        accept = delta <= 0.0
        if not accept and config.temperature > 0.0:
            accept = rng.random() < math.exp(-delta / config.temperature)
        if not accept:
            continue

        state.labels[y, x] = proposed
        state.areas[old] -= 1
        state.areas[proposed] += 1
        accepted += 1
    return accepted / state.labels.size


def energy_terms(
    state: CPMState,
    targets: CPMTargets,
    config: CPMConfig,
) -> dict[str, float]:
    """Return contact and area terms using each lattice edge once."""
    labels = state.labels
    contact = 0.0
    for dy, dx, weight in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 1 / math.sqrt(2)), (1, -1, 1 / math.sqrt(2))):
        shifted = np.roll(labels, shift=(-dy, -dx), axis=(0, 1))
        different = labels != shifted
        es = different & ((labels == 0) | (shifted == 0))
        ee = different & ~es
        contact += weight * (
            config.contact_es * int(np.count_nonzero(es))
            + config.contact_ee * int(np.count_nonzero(ee))
        )

    epithelial_errors = state.areas[1:].astype(np.float64) - targets.epithelial
    epithelial_area = config.lambda_epithelial * float(
        np.dot(epithelial_errors, epithelial_errors)
    )
    stromal_error = float(state.areas[0]) - targets.stroma
    stromal_area = config.lambda_stroma * stromal_error * stromal_error
    return {
        "energy_contact": float(contact),
        "energy_epithelial_area": epithelial_area,
        "energy_stromal_area": stromal_area,
        "energy_total": float(contact + epithelial_area + stromal_area),
    }


def epithelial_field(state: CPMState) -> np.ndarray:
    return (state.labels > 0).astype(np.float32)


def interface_length(state: CPMState) -> int:
    epithelial = state.labels > 0
    horizontal = epithelial != np.roll(epithelial, -1, axis=1)
    vertical = epithelial != np.roll(epithelial, -1, axis=0)
    return int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))


def render_state(state: CPMState, progression: float, label: str) -> np.ndarray:
    field = epithelial_field(state)
    image = np.full((*field.shape, 3), 242, dtype=np.uint8)
    image[field > 0] = (69, 92, 184)
    boundary = (
        (field != np.roll(field, 1, axis=0))
        | (field != np.roll(field, -1, axis=0))
        | (field != np.roll(field, 1, axis=1))
        | (field != np.roll(field, -1, axis=1))
    )
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
    sweeps_per_stage: int,
    seed: int,
    config: CPMConfig,
) -> tuple[CPMState, list[dict[str, object]], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    state = initialize_state(size, config)
    rows: list[dict[str, object]] = []
    images: list[np.ndarray] = []

    for progression in progressions:
        targets = targets_from_progression(state, config, progression)
        acceptance: list[float] = []
        for _ in range(sweeps_per_stage):
            acceptance.append(
                monte_carlo_sweep(
                    state,
                    targets=targets,
                    config=config,
                    rng=rng,
                )
            )

        field = epithelial_field(state)
        metrics = measure_goal_pattern(field)
        morphology = classify_morphology(metrics)
        terms = energy_terms(state, targets, config)
        rows.append(
            {
                "progression": progression,
                "morphology": morphology,
                "target_epithelial_area": targets.epithelial,
                "target_stromal_area": targets.stroma,
                "epithelial_fraction": float(np.mean(field)),
                "mean_epithelial_area": float(np.mean(state.areas[1:])),
                "mean_absolute_area_error": float(
                    np.mean(np.abs(state.areas[1:] - targets.epithelial))
                ),
                "interface_length": interface_length(state),
                "acceptance_rate": float(np.mean(acceptance)),
                **terms,
                **metrics,
            }
        )
        images.append(render_state(state, progression, morphology))
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
    parser = argparse.ArgumentParser(description="Run the minimal E/S Cellular Potts PoC.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pit_cpm"))
    parser.add_argument("--size", type=int, default=84)
    parser.add_argument("--sweeps-per-stage", type=int, default=100)
    parser.add_argument("--progressions", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progressions = parse_progressions(args.progressions)
    config = CPMConfig()
    args.output.mkdir(parents=True, exist_ok=True)

    state, rows, images = run_progression(
        size=args.size,
        progressions=progressions,
        sweeps_per_stage=args.sweeps_per_stage,
        seed=args.seed,
        config=config,
    )
    for progression, image in zip(progressions, images, strict=True):
        cv2.imwrite(str(args.output / f"cpm_p{progression:.2f}.png"), image)
    cv2.imwrite(str(args.output / "contact_sheet.png"), np.concatenate(images, axis=1))
    np.save(args.output / "labels.npy", state.labels)
    with (args.output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": asdict(config), "size": args.size, "stages": rows},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"wrote {args.output}")
    for row in rows:
        print(
            f"p={row['progression']:.2f} {row['morphology']}: "
            f"E={row['epithelial_fraction']:.3f} "
            f"accept={row['acceptance_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
