#!/usr/bin/env python3
"""Evaluate reaction-diffusion changes for Kudo pit-like patterns.

This script keeps the pattern source in the equations. It does not draw pit
geometry directly; every case starts from the same random perturbation and then
changes only reaction terms or slow parameter fields.

Example:
    python src/evaluate_pit_steps.py --output artifacts/pit_step_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


KERNEL = np.array(
    [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ModelConfig:
    """One model variant in the evaluation suite."""

    name: str
    description: str
    feed: float
    k: float
    d_a: float = 1.0
    d_b: float = 0.5
    reaction_saturation: float = 0.0
    cubic_damping: float = 0.0
    static_env_scale: float = 0.0
    feed_env_sensitivity: float = 0.0
    k_env_sensitivity: float = 0.0
    dynamic_env: bool = False
    env_rate: float = 0.0
    env_diffusion: float = 0.0
    env_source: float = 0.0
    env_decay: float = 0.0


@dataclass
class StepResult:
    config: ModelConfig
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    snapshots: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]


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


def make_initial_state(
    size: int, seed: int, seed_density: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create a neutral stochastic perturbation shared by all variants."""

    rng = np.random.default_rng(seed)
    h = w = size
    a = np.ones((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)

    # This is only a disturbance of the homogeneous state, not a target shape.
    noise = smooth_unit_noise(rng, (h, w), sigma=max(1.5, size / 80))
    threshold = float(np.quantile(noise, 1.0 - seed_density))
    seeds = noise > threshold
    b[seeds] = 0.85
    a[seeds] = 0.25

    a += 0.02 * rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    b += 0.02 * rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    return np.clip(a, 0.0, 1.0), np.clip(b, 0.0, 1.0)


def build_static_environment(size: int, seed: int, scale: float) -> np.ndarray:
    if scale == 0.0:
        return np.zeros((size, size), dtype=np.float32)

    rng = np.random.default_rng(seed + 10_003)
    return scale * smooth_unit_noise(rng, (size, size), sigma=max(4.0, size / 16))


def effective_parameters(
    config: ModelConfig, c: np.ndarray
) -> tuple[np.ndarray | float, np.ndarray | float]:
    if (
        config.feed_env_sensitivity == 0.0
        and config.k_env_sensitivity == 0.0
        and config.static_env_scale == 0.0
        and not config.dynamic_env
    ):
        return config.feed, config.k

    feed = config.feed + config.feed_env_sensitivity * c
    k = config.k + config.k_env_sensitivity * c
    return np.clip(feed, 0.0, 0.12), np.clip(k, 0.0, 0.12)


def reaction_term(a: np.ndarray, b: np.ndarray, saturation: float) -> np.ndarray:
    ab2 = a * (b**2)
    if saturation <= 0.0:
        return ab2
    return ab2 / (1.0 + saturation * (b**2))


def calc_step(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, config: ModelConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feed, k = effective_parameters(config, c)
    reaction = reaction_term(a, b, config.reaction_saturation)

    lap_a = cv2.filter2D(a, -1, KERNEL)
    lap_b = cv2.filter2D(b, -1, KERNEL)

    next_a = a + config.d_a * lap_a - reaction + feed * (1.0 - a)
    next_b = (
        b
        + config.d_b * lap_b
        + reaction
        - (k + feed) * b
        - config.cubic_damping * (b**3)
    )

    if config.dynamic_env:
        lap_c = cv2.filter2D(c, -1, KERNEL)
        b_centered = b - float(b.mean())
        c = c + config.env_rate * (
            config.env_diffusion * lap_c
            + config.env_source * b_centered
            - config.env_decay * c
        )
        c = np.clip(c, -3.0, 3.0)

    return np.clip(next_a, 0.0, 1.0), np.clip(next_b, 0.0, 1.0), c.astype(np.float32)


def run_model(
    config: ModelConfig,
    *,
    size: int,
    steps: int,
    seed: int,
    seed_density: float,
    snapshot_count: int,
) -> StepResult:
    a, b = make_initial_state(size, seed, seed_density)
    c = build_static_environment(size, seed, config.static_env_scale)

    snapshot_steps = set(np.linspace(0, steps, snapshot_count, dtype=int).tolist())
    snapshots: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    if 0 in snapshot_steps:
        snapshots.append((0, a.copy(), b.copy(), c.copy()))

    for step in range(1, steps + 1):
        a, b, c = calc_step(a, b, c, config)
        if step in snapshot_steps:
            snapshots.append((step, a.copy(), b.copy(), c.copy()))

    return StepResult(config=config, a=a, b=b, c=c, snapshots=snapshots)


def normalize_u8(field: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(field, 1))
    hi = float(np.percentile(field, 99))
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.clip((field - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def pattern_binary(b: np.ndarray) -> np.ndarray:
    b_u8 = normalize_u8(b)
    otsu_threshold, _ = cv2.threshold(b_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    percentile_threshold = float(np.percentile(b_u8, 70))
    threshold = max(otsu_threshold, percentile_threshold, 20.0)
    binary = b_u8 >= threshold

    # Remove tiny numerical speckles before skeleton/region measurements.
    binary_u8 = binary.astype(np.uint8)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, 8)
    cleaned = np.zeros_like(binary_u8)
    for label in range(1, components):
        if stats[label, cv2.CC_STAT_AREA] >= 4:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def zhang_suen_thin(binary: np.ndarray, max_iterations: int = 120) -> np.ndarray:
    img = binary.astype(np.uint8).copy()
    img[[0, -1], :] = 0
    img[:, [0, -1]] = 0

    for _ in range(max_iterations):
        changed = False
        for sub_iteration in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            center = img[1:-1, 1:-1]

            neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if sub_iteration == 0:
                condition_a = (p2 * p4 * p6) == 0
                condition_b = (p4 * p6 * p8) == 0
            else:
                condition_a = (p2 * p4 * p8) == 0
                condition_b = (p2 * p6 * p8) == 0

            remove_inner = (
                (center == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & condition_a
                & condition_b
            )

            if bool(remove_inner.any()):
                marker = np.zeros_like(img, dtype=bool)
                marker[1:-1, 1:-1] = remove_inner
                img[marker] = 0
                changed = True

        if not changed:
            break

    return img.astype(bool)


def component_areas(binary: np.ndarray) -> np.ndarray:
    components, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if components <= 1:
        return np.array([], dtype=np.float32)
    return stats[1:, cv2.CC_STAT_AREA].astype(np.float32)


def local_density_stats(binary: np.ndarray, tiles: int = 8) -> tuple[float, float]:
    h, w = binary.shape
    densities = []
    for y0 in np.linspace(0, h, tiles + 1, dtype=int)[:-1]:
        y1 = min(h, y0 + math.ceil(h / tiles))
        for x0 in np.linspace(0, w, tiles + 1, dtype=int)[:-1]:
            x1 = min(w, x0 + math.ceil(w / tiles))
            tile = binary[y0:y1, x0:x1]
            if tile.size:
                densities.append(float(tile.mean()))

    values = np.array(densities, dtype=np.float32)
    hist, _ = np.histogram(values, bins=8, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float32)
    probs /= max(float(probs.sum()), 1.0)
    entropy = -float(np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
    return float(values.std()), entropy


def measure_pattern(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict[str, float]:
    binary = pattern_binary(b)
    skeleton = zhang_suen_thin(binary)

    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, neighbor_kernel)
    neighbors = neighbors.astype(np.int16) - skeleton.astype(np.int16)

    skeleton_pixels = int(skeleton.sum())
    branch_points = int((skeleton & (neighbors >= 3)).sum())
    endpoints = int((skeleton & (neighbors == 1)).sum())

    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton.astype(np.uint8), 8
    )
    if components > 1 and skeleton_pixels > 0:
        skeleton_areas = stats[1:, cv2.CC_STAT_AREA]
        longest_component_fraction = float(skeleton_areas.max() / skeleton_pixels)
        skeleton_components = int(components - 1)
    else:
        longest_component_fraction = 0.0
        skeleton_components = 0

    areas = component_areas(binary)
    if areas.size:
        area_mean = float(areas.mean())
        area_cv = float(areas.std() / max(area_mean, 1.0e-6))
    else:
        area_mean = 0.0
        area_cv = 0.0

    local_density_std, local_density_entropy = local_density_stats(binary)
    branch_density = branch_points / max(float(skeleton_pixels), 1.0)
    endpoint_density = endpoints / max(float(skeleton_pixels), 1.0)
    long_line_score = longest_component_fraction / max(branch_density + 0.01, 0.01)
    vi_irregularity_score = area_cv + local_density_std + 0.1 * local_density_entropy

    return {
        "a_mean": float(a.mean()),
        "b_mean": float(b.mean()),
        "b_std": float(b.std()),
        "c_std": float(c.std()),
        "binary_coverage": float(binary.mean()),
        "skeleton_pixels": float(skeleton_pixels),
        "skeleton_components": float(skeleton_components),
        "branch_points": float(branch_points),
        "endpoints": float(endpoints),
        "branch_density": float(branch_density),
        "endpoint_density": float(endpoint_density),
        "longest_component_fraction": float(longest_component_fraction),
        "long_line_score": float(long_line_score),
        "component_count": float(areas.size),
        "component_area_mean": float(area_mean),
        "component_area_cv": float(area_cv),
        "local_density_std": float(local_density_std),
        "local_density_entropy": float(local_density_entropy),
        "vi_irregularity_score": float(vi_irregularity_score),
    }


def render_fields(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    image = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    image[:, :, 2] = normalize_u8(a)
    image[:, :, 0] = normalize_u8(b)
    if float(np.abs(c).max()) > 1.0e-8:
        image[:, :, 1] = normalize_u8(c)
    else:
        image[:, :, 1] = 10
    return image


def render_skeleton_overlay(b: np.ndarray) -> np.ndarray:
    binary = pattern_binary(b)
    skeleton = zhang_suen_thin(binary)
    base = cv2.cvtColor(normalize_u8(b), cv2.COLOR_GRAY2BGR)
    base[binary] = (80, 80, 80)

    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, neighbor_kernel)
    neighbors = neighbors.astype(np.int16) - skeleton.astype(np.int16)
    branch_points = skeleton & (neighbors >= 3)
    endpoints = skeleton & (neighbors == 1)

    base[skeleton] = (255, 255, 255)
    base[endpoints] = (255, 255, 0)
    base[branch_points] = (0, 0, 255)
    return base


def annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    output = image.copy()
    pad = 6
    line_height = 14
    overlay_height = pad * 2 + line_height * len(lines)
    cv2.rectangle(output, (0, 0), (output.shape[1], overlay_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (pad, pad + 10 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def save_result_images(result: StepResult, metrics: dict[str, float], output: Path) -> None:
    case_dir = output / result.config.name
    case_dir.mkdir(parents=True, exist_ok=True)

    for step, a, b, c in result.snapshots:
        image = render_fields(a, b, c)
        cv2.imwrite(str(case_dir / f"fields_{step:06d}.png"), image)

    final = annotate(
        render_fields(result.a, result.b, result.c),
        [
            result.config.name,
            f"branch={metrics['branch_density']:.3f} long={metrics['long_line_score']:.2f}",
            f"irreg={metrics['vi_irregularity_score']:.2f} cov={metrics['binary_coverage']:.2f}",
        ],
    )
    overlay = annotate(
        render_skeleton_overlay(result.b),
        [
            "skeleton overlay",
            "red=branch cyan=end",
            f"components={metrics['skeleton_components']:.0f}",
        ],
    )
    cv2.imwrite(str(case_dir / "final_fields.png"), final)
    cv2.imwrite(str(case_dir / "final_skeleton.png"), overlay)


def write_metrics(output: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def write_contact_sheet(output: Path, results: list[tuple[StepResult, dict[str, float]]]) -> None:
    cells = []
    for result, metrics in results:
        fields = annotate(
            render_fields(result.a, result.b, result.c),
            [
                result.config.name,
                f"branch {metrics['branch_density']:.3f}",
                f"long {metrics['long_line_score']:.2f}",
            ],
        )
        skeleton = annotate(
            render_skeleton_overlay(result.b),
            [
                "skeleton",
                f"irreg {metrics['vi_irregularity_score']:.2f}",
                f"area_cv {metrics['component_area_cv']:.2f}",
            ],
        )
        cells.append(np.concatenate([fields, skeleton], axis=1))

    if not cells:
        return

    width = max(cell.shape[1] for cell in cells)
    padded = []
    for cell in cells:
        if cell.shape[1] < width:
            padding = np.zeros((cell.shape[0], width - cell.shape[1], 3), dtype=np.uint8)
            cell = np.concatenate([cell, padding], axis=1)
        padded.append(cell)

    sheet = np.concatenate(padded, axis=0)
    cv2.imwrite(str(output / "contact_sheet.png"), sheet)


def build_suite(suite: str) -> list[ModelConfig]:
    """Build a small ordered suite for comparing equation changes."""

    references = [
        ModelConfig(
            name="reference_pit_i",
            description="Original Gray-Scott reference near the current preset 1.",
            feed=0.055,
            k=0.062,
        ),
        ModelConfig(
            name="reference_pit_iii",
            description="Original Gray-Scott reference near the current preset 2.",
            feed=0.045,
            k=0.066,
        ),
    ]
    steps = [
        ModelConfig(
            name="step0_pit_iv_baseline",
            description="Original Gray-Scott Type IV candidate; tends to make long labyrinth lines.",
            feed=0.033,
            k=0.056,
        ),
        ModelConfig(
            name="step1_pit_iv_saturated",
            description="AB^2 is saturated to reduce over-stable long B-rich stripes.",
            feed=0.033,
            k=0.056,
            reaction_saturation=1.6,
        ),
        ModelConfig(
            name="step1b_pit_iv_saturated_cubic",
            description="Saturated reaction plus B^3 damping to penalize high-concentration stripe cores.",
            feed=0.033,
            k=0.056,
            reaction_saturation=1.6,
            cubic_damping=0.018,
        ),
        ModelConfig(
            name="step2_pit_iv_static_environment",
            description="Step 1b plus weak smooth f/k heterogeneity as tissue environment.",
            feed=0.033,
            k=0.056,
            reaction_saturation=1.6,
            cubic_damping=0.018,
            static_env_scale=1.0,
            feed_env_sensitivity=0.0030,
            k_env_sensitivity=-0.0015,
        ),
        ModelConfig(
            name="step3_pit_vi_dynamic_environment",
            description="Slow environment C is coupled to B, pushing local regions across pattern regimes.",
            feed=0.036,
            k=0.058,
            reaction_saturation=1.2,
            cubic_damping=0.012,
            static_env_scale=0.8,
            feed_env_sensitivity=0.0045,
            k_env_sensitivity=-0.0025,
            dynamic_env=True,
            env_rate=0.035,
            env_diffusion=0.20,
            env_source=0.70,
            env_decay=0.08,
        ),
    ]

    if suite == "references":
        return references
    if suite == "steps":
        return steps
    return references + steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run headless step-by-step evaluation of pit-pattern model changes."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pit_step_eval"),
        help="Directory for images and metrics.",
    )
    parser.add_argument("--size", type=int, default=160, help="Simulation width/height.")
    parser.add_argument("--steps", type=int, default=3500, help="Simulation steps per case.")
    parser.add_argument("--seed", type=int, default=7, help="Seed shared by all cases.")
    parser.add_argument(
        "--seed-density",
        type=float,
        default=0.035,
        help="Fraction of stochastic initial perturbation pixels.",
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=5,
        help="Number of time snapshots saved per case.",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "references", "steps"),
        default="all",
        help="Which model variants to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    results: list[tuple[StepResult, dict[str, float]]] = []
    for config in build_suite(args.suite):
        print(f"running {config.name} ...", flush=True)
        result = run_model(
            config,
            size=args.size,
            steps=args.steps,
            seed=args.seed,
            seed_density=args.seed_density,
            snapshot_count=args.snapshot_count,
        )
        metrics = measure_pattern(result.a, result.b, result.c)
        save_result_images(result, metrics, args.output)

        row: dict[str, object] = {
            "name": config.name,
            "description": config.description,
            "steps": args.steps,
            "size": args.size,
            **asdict(config),
            **metrics,
        }
        rows.append(row)
        results.append((result, metrics))

    write_metrics(args.output, rows)
    write_contact_sheet(args.output, results)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
