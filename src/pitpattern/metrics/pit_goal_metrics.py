#!/usr/bin/env python3
"""Goal-oriented metrics for I -> IIIL -> IV_B -> IV_V pit pattern progression.

Extends skeleton metrics from evaluate_pit_steps with feature width, branch arm
lengths, labyrinth / mesh scores, temporal stability, and morphology labels.

IV_V here means the constant-width labyrinth / gyrus mesh of pptx slide 6
(center), NOT parallel leaf-like stripes.  Parallel unbranched stripes are a
failure mode, not a phase.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Literal

import cv2
import numpy as np

from pitpattern.metrics.evaluate_pit_steps import (
    cluster_arm_lengths,
    component_areas,
    default_min_branch_arm_length,
    junction_cluster_labels,
    local_density_stats,
    measure_pattern,
    pattern_binary,
    prepare_skeleton_topology,
    skeleton_degree_map,
    skeleton_endpoints,
    zhang_suen_thin,
)

MorphologyLabel = Literal["I", "IIIL", "IV_B", "IV_V", "fail", "empty"]

GOAL_METRIC_KEYS = (
    "feature_width_median",
    "feature_width_p90",
    "feature_width_cv",
    "mean_branch_arm_length",
    "max_branch_arm_length",
    "p90_branch_arm_length",
    "branch_arm_count",
    "mean_component_elongation",
    "p90_component_elongation",
    "max_geodesic_length",
    "p90_geodesic_length",
    "orientation_coherence",
    "background_island_count",
    "background_gap_p90",
    "labyrinth_score",
    "branch_before_elongation_score",
)


def height_to_activator(u: np.ndarray) -> np.ndarray:
    """Map buckling height to a [0, 1] field compatible with pattern_binary."""
    lo = float(np.percentile(u, 1))
    hi = float(np.percentile(u, 99))
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.clip((u - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def field_to_binary(field: np.ndarray, *, field_kind: str = "activator") -> np.ndarray:
    if field_kind == "height":
        return pattern_binary(height_to_activator(field))
    return pattern_binary(field)


def measure_feature_width(binary: np.ndarray, skeleton: np.ndarray | None = None) -> dict[str, float]:
    """Local half-width via distance transform, sampled on skeleton pixels."""
    if not bool(binary.any()):
        return {
            "feature_width_median": 0.0,
            "feature_width_p90": 0.0,
            "feature_width_cv": 0.0,
        }

    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    if skeleton is None:
        skeleton = zhang_suen_thin(binary)
    sk = skeleton.astype(bool)
    if not bool(sk.any()):
        widths = dist[binary] * 2.0
    else:
        widths = dist[sk] * 2.0

    if widths.size == 0:
        return {
            "feature_width_median": 0.0,
            "feature_width_p90": 0.0,
            "feature_width_cv": 0.0,
        }

    median = float(np.median(widths))
    p90 = float(np.percentile(widths, 90))
    cv = float(np.std(widths) / max(median, 1.0e-6))
    return {
        "feature_width_median": median,
        "feature_width_p90": p90,
        "feature_width_cv": cv,
    }


def collect_branch_arm_lengths(
    skeleton: np.ndarray,
    *,
    min_arm_length: int | None = None,
) -> list[int]:
    if min_arm_length is None:
        min_arm_length = default_min_branch_arm_length(skeleton.shape[0])
    sk = skeleton.astype(bool)
    junction_labels, clusters = junction_cluster_labels(sk)
    arms: list[int] = []
    for label, cluster in enumerate(clusters, start=1):
        lengths = cluster_arm_lengths(sk, cluster, junction_labels, label)
        arms.extend(length for length in lengths if length >= min_arm_length)
    return arms


def measure_branch_arms(
    skeleton: np.ndarray,
    *,
    min_arm_length: int | None = None,
) -> dict[str, float]:
    arms = collect_branch_arm_lengths(skeleton, min_arm_length=min_arm_length)
    if not arms:
        return {
            "mean_branch_arm_length": 0.0,
            "max_branch_arm_length": 0.0,
            "p90_branch_arm_length": 0.0,
            "branch_arm_count": 0.0,
        }
    arr = np.array(arms, dtype=np.float32)
    return {
        "mean_branch_arm_length": float(arr.mean()),
        "max_branch_arm_length": float(arr.max()),
        "p90_branch_arm_length": float(np.percentile(arr, 90)),
        "branch_arm_count": float(arr.size),
    }


def _component_pca(binary: np.ndarray, label: int, labels: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(labels == label)
    if ys.size < 3:
        return 1.0, 1.0
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = coords.T @ coords / max(coords.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.clip(eigvals, 1.0e-6, None))
    major = float(math.sqrt(eigvals[-1]))
    minor = float(math.sqrt(eigvals[0]))
    return major, minor


def measure_component_elongation(binary: np.ndarray) -> dict[str, float]:
    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    if components <= 1:
        return {
            "mean_component_elongation": 0.0,
            "p90_component_elongation": 0.0,
        }

    ratios: list[float] = []
    for label in range(1, components):
        if stats[label, cv2.CC_STAT_AREA] < 4:
            continue
        major, minor = _component_pca(binary, label, labels)
        ratios.append(major / minor)

    if not ratios:
        return {
            "mean_component_elongation": 0.0,
            "p90_component_elongation": 0.0,
        }
    arr = np.array(ratios, dtype=np.float32)
    return {
        "mean_component_elongation": float(arr.mean()),
        "p90_component_elongation": float(np.percentile(arr, 90)),
    }


def _geodesic_from_seed(sk: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    h, w = sk.shape
    sy, sx = start
    if not sk[sy, sx]:
        return {}
    dist: dict[tuple[int, int], int] = {(sy, sx): 0}
    queue: deque[tuple[int, int]] = deque([(sy, sx)])
    offsets = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    while queue:
        y, x = queue.popleft()
        d = dist[(y, x)]
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] and (ny, nx) not in dist:
                dist[(ny, nx)] = d + 1
                queue.append((ny, nx))
    return dist


def measure_geodesic_lengths(skeleton: np.ndarray) -> dict[str, float]:
    sk = skeleton.astype(bool)
    if not bool(sk.any()):
        return {"max_geodesic_length": 0.0, "p90_geodesic_length": 0.0}

    endpoints = list(zip(*np.where(skeleton_endpoints(sk))))
    if not endpoints:
        ys, xs = np.where(sk)
        endpoints = [(int(ys[0]), int(xs[0]))]

    best = 0.0
    lengths: list[float] = []
    for start in endpoints[: min(len(endpoints), 32)]:
        dist_map = _geodesic_from_seed(sk, start)
        if not dist_map:
            continue
        local_max = float(max(dist_map.values()))
        best = max(best, local_max)
        lengths.append(local_max)

    if not lengths:
        return {"max_geodesic_length": 0.0, "p90_geodesic_length": 0.0}
    arr = np.array(lengths, dtype=np.float32)
    return {
        "max_geodesic_length": best,
        "p90_geodesic_length": float(np.percentile(arr, 90)),
    }


def measure_orientation_coherence(skeleton: np.ndarray) -> dict[str, float]:
    sk = skeleton.astype(bool)
    if not bool(sk.any()):
        return {"orientation_coherence": 0.0}

    gy, gx = np.gradient(sk.astype(np.float64))
    mag = np.sqrt(gx * gx + gy * gy)
    mask = sk & (mag > 1.0e-6)
    if not bool(mask.any()):
        return {"orientation_coherence": 0.0}

    angles = np.arctan2(gy[mask], gx[mask])
    vec_x = float(np.mean(np.cos(angles)))
    vec_y = float(np.mean(np.sin(angles)))
    coherence = float(np.sqrt(vec_x * vec_x + vec_y * vec_y))
    return {"orientation_coherence": coherence}


def measure_background_gaps(binary: np.ndarray) -> dict[str, float]:
    """Half-width of background gaps between foreground structures.

    A dense labyrinth / gyrus mesh (IV_V goal) fills the field with grooves at
    roughly one wavelength spacing, so background gaps stay small and uniform.
    Separated finite trees (IV_B) leave wide empty corridors between units.
    """
    mask = binary.astype(bool)
    if not bool(mask.any()) or bool(mask.all()):
        return {"background_gap_p90": 0.0, "background_gap_max": 0.0}
    bg = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(bg, cv2.DIST_L2, 5)
    vals = dist[bg.astype(bool)]
    return {
        "background_gap_p90": float(np.percentile(vals, 90)),
        "background_gap_max": float(vals.max()),
    }


def measure_background_islands(binary: np.ndarray) -> dict[str, float]:
    """Count enclosed background islands — signature of a connected mesh.

    A labyrinth / gyrus network traps the background into many small islands
    (loops); dots, dashes, trees, and parallel stripes leave the background as
    one border-touching region.
    """
    bg = (~binary.astype(bool)).astype(np.uint8)
    components, labels = cv2.connectedComponents(bg, connectivity=4)
    if components <= 1:
        return {"background_island_count": 0.0}

    border_labels = set(np.unique(labels[0, :]))
    border_labels |= set(np.unique(labels[-1, :]))
    border_labels |= set(np.unique(labels[:, 0]))
    border_labels |= set(np.unique(labels[:, -1]))

    islands = 0
    for label in range(1, components):
        if label in border_labels:
            continue
        islands += 1
    return {"background_island_count": float(islands)}


def measure_labyrinth_scores(metrics: dict[str, float]) -> dict[str, float]:
    """Score the IV_V target: connected winding mesh with constant width.

    High when the pattern is connected (longest fraction), long (geodesic),
    loop-rich (background islands), and isotropic (low orientation coherence).
    Parallel aligned stripes score low because they have no islands and high
    orientation coherence.
    """
    branch = metrics.get("branch_density", 0.0)
    geo = metrics.get("p90_geodesic_length", 0.0)
    longest = metrics.get("longest_component_fraction", 0.0)
    orient = min(metrics.get("orientation_coherence", 0.0), 1.0)
    islands = metrics.get("background_island_count", 0.0)
    width = metrics.get("feature_width_median", 1.0)

    mesh = longest * (1.0 + islands) * (1.5 - orient)
    labyrinth = (geo * mesh) / max(width, 1.0)
    branch_first = branch / max(labyrinth, 1.0e-6)
    return {
        "labyrinth_score": float(labyrinth),
        "branch_before_elongation_score": float(branch_first),
    }


def measure_goal_pattern(
    field: np.ndarray,
    *,
    a: np.ndarray | None = None,
    c: np.ndarray | None = None,
    field_kind: str = "activator",
) -> dict[str, float]:
    """Full goal metrics: base pattern metrics + width + labyrinth + topology."""
    if a is None:
        a = np.ones_like(field, dtype=np.float32)
    if c is None:
        c = np.zeros_like(field, dtype=np.float32)

    if field_kind == "height":
        b = height_to_activator(field)
    else:
        b = field.astype(np.float32)

    base = measure_pattern(a, b, c)
    binary = field_to_binary(field, field_kind=field_kind)
    skeleton, _ = prepare_skeleton_topology(zhang_suen_thin(binary))

    width = measure_feature_width(binary, skeleton)
    arms = measure_branch_arms(skeleton)
    elong = measure_component_elongation(binary)
    geo = measure_geodesic_lengths(skeleton)
    orient = measure_orientation_coherence(skeleton)
    islands = measure_background_islands(binary)
    gaps = measure_background_gaps(binary)

    merged = {**base, **width, **arms, **elong, **geo, **orient, **islands, **gaps}
    merged.update(measure_labyrinth_scores(merged))
    sk_pixels = max(merged.get("skeleton_pixels", 1.0), 1.0)
    merged["branch_arm_density"] = float(merged.get("branch_arm_count", 0.0) / sk_pixels)
    return merged


def measure_pattern_timeseries(
    snapshots: list[tuple[int, np.ndarray]],
    *,
    field_kind: str = "height",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, field in snapshots:
        metrics = measure_goal_pattern(field, field_kind=field_kind)
        rows.append({"step": int(step), **metrics})
    return rows


def _linear_slope(values: list[float], window_fraction: float = 0.25) -> float:
    if len(values) < 4:
        return 0.0
    n = max(2, int(len(values) * window_fraction))
    tail = np.array(values[-n:], dtype=np.float64)
    x = np.arange(tail.size, dtype=np.float64)
    x -= x.mean()
    y = tail - tail.mean()
    denom = float(np.dot(x, x))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def measure_temporal_stability(timeseries: list[dict[str, float]]) -> dict[str, float]:
    if len(timeseries) < 2:
        return {
            "longest_trend": 0.0,
            "branch_density_trend": 0.0,
            "feature_width_trend": 0.0,
            "labyrinth_score_trend": 0.0,
            "is_stationary": 1.0,
            "coarsening_detected": 0.0,
        }

    longest = [row.get("longest_component_fraction", 0.0) for row in timeseries]
    branch = [row.get("branch_density", 0.0) for row in timeseries]
    width = [row.get("feature_width_median", 0.0) for row in timeseries]
    labyrinth = [row.get("labyrinth_score", 0.0) for row in timeseries]

    longest_trend = _linear_slope(longest)
    branch_trend = _linear_slope(branch)
    width_trend = _linear_slope(width)
    labyrinth_trend = _linear_slope(labyrinth)

    stationary = (
        abs(longest_trend) < 0.02
        and abs(width_trend) < 0.15
        and abs(branch_trend) < 0.05
    )
    # Coarsening now means WIDTH growth (the true failure), not connectivity
    # growth: a connecting labyrinth mesh is the IV_V goal, not a defect.
    coarsening = width_trend > 0.25 and branch_trend <= 0.01

    return {
        "longest_trend": float(longest_trend),
        "branch_density_trend": float(branch_trend),
        "feature_width_trend": float(width_trend),
        "labyrinth_score_trend": float(labyrinth_trend),
        "is_stationary": 1.0 if stationary else 0.0,
        "coarsening_detected": 1.0 if coarsening else 0.0,
    }


def width_ratio_to_reference(metrics: dict[str, float], reference_width: float) -> float:
    width = metrics.get("feature_width_median", 0.0)
    if reference_width <= 1.0e-6:
        return 0.0
    return float(width / reference_width)


def measure_horizontal_runs(binary: np.ndarray) -> dict[str, float]:
    """Longest foreground run per row — useful for x-aligned stripes in narrow strips."""
    mask = binary.astype(bool)
    if not bool(mask.any()):
        return {
            "max_horizontal_run": 0.0,
            "mean_horizontal_run": 0.0,
            "horizontal_run_ratio": 0.0,
        }

    h, w = mask.shape
    row_max: list[float] = []
    for row in mask:
        if not bool(row.any()):
            continue
        padded = np.concatenate(([False], row, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        runs = changes[1::2] - changes[::2]
        row_max.append(float(runs.max()) if runs.size else 0.0)

    if not row_max:
        return {
            "max_horizontal_run": 0.0,
            "mean_horizontal_run": 0.0,
            "horizontal_run_ratio": 0.0,
        }

    max_run = float(max(row_max))
    mean_run = float(np.mean(row_max))
    return {
        "max_horizontal_run": max_run,
        "mean_horizontal_run": mean_run,
        "horizontal_run_ratio": max_run / max(w, 1),
    }


def classify_morphology(metrics: dict[str, float]) -> MorphologyLabel:
    """I / IIIL / IV_B / IV_V / fail.

    IV_B = branching present but units still finite (multiple trees).
    IV_V = branching present AND connected into a labyrinth / gyrus mesh
    (dominant component and/or enclosed background islands).
    Long unbranched parallel stripes are 'fail', not IV_V.
    """
    coverage = metrics.get("binary_coverage", 0.0)
    if coverage < 0.01:
        return "empty"

    branch = metrics.get("branch_density", 0.0)
    arm_count = metrics.get("branch_arm_count", 0.0)
    arm_density = metrics.get("branch_arm_density", 0.0)
    longest = metrics.get("longest_component_fraction", 0.0)
    elong = metrics.get("p90_component_elongation", 0.0)
    islands = metrics.get("background_island_count", 0.0)
    components = metrics.get("component_count", 0.0)
    sk_components = metrics.get("skeleton_components", 0.0)

    has_branching = branch >= 0.02 or arm_count >= 6 or arm_density >= 0.04
    is_mesh = longest >= 0.45 or islands >= 6.0

    if not has_branching and elong < 2.0 and components >= 8 and sk_components >= 20:
        return "I"
    if not has_branching and elong < 6.5 and longest < 0.55:
        return "IIIL"
    if has_branching and is_mesh:
        return "IV_V"
    if has_branching:
        return "IV_B"
    # No branching left: either short dashes (handled above as IIIL) or long
    # unbranched grooves / parallel stripes — both outside the goal sequence.
    return "fail"


def classify_spatial_morphology(metrics: dict[str, float]) -> MorphologyLabel:
    """Classify a vertical strip in a spatial p(x) field.

    Same phase definitions as classify_morphology; horizontal-run metrics are
    used only to DETECT (and reject) x-aligned parallel stripes, which narrow
    strips would otherwise mislabel.  A connected labyrinth mesh keeps
    branching / junctions even inside a narrow strip; unbranched full-width
    horizontal runs are the parallel-stripe failure.
    """
    coverage = metrics.get("binary_coverage", 0.0)
    if coverage < 0.01:
        return "empty"

    branch = metrics.get("branch_density", 0.0)
    arm_count = metrics.get("branch_arm_count", 0.0)
    arm_density = metrics.get("branch_arm_density", 0.0)
    longest = metrics.get("longest_component_fraction", 0.0)
    elong = metrics.get("p90_component_elongation", 0.0)
    islands = metrics.get("background_island_count", 0.0)
    components = metrics.get("component_count", 0.0)
    comp_density = metrics.get("component_density", 0.0)
    hr_ratio = metrics.get("horizontal_run_ratio", 0.0)
    mean_hr = metrics.get("mean_horizontal_run", 0.0)
    has_branching = branch >= 0.02 or arm_count >= 6 or (
        arm_density >= 0.025 and arm_count >= 4
    )
    # Windows cut the mesh into pieces, so connectivity is measured as FEW
    # components per area (units merged into large ones), a percolating giant
    # winding component (huge elongation), or the direct longest-fraction /
    # enclosed-island evidence.
    is_mesh = (
        longest >= 0.45
        or islands >= 2.0
        or (comp_density > 0.0 and comp_density <= 1.8)
        or elong >= 50.0
    )

    if not has_branching and elong < 4.5 and mean_hr < 9.5 and components >= 6:
        return "I"
    # Unbranched grooves crossing the whole strip = parallel stripes (fail).
    if not has_branching and hr_ratio >= 0.82:
        return "fail"
    if not has_branching:
        return "IIIL"
    # Light branching on many small elongated units = IIIL -> IV_B boundary.
    if comp_density >= 4.0 and arm_density < 0.035:
        return "IIIL"
    if is_mesh:
        return "IV_V"
    return "IV_B"


def evaluate_goal_sequence(
    sweep_rows: list[dict[str, Any]],
    *,
    reference_width: float | None = None,
) -> dict[str, Any]:
    """Score a p-sweep for ordered I -> IIIL -> IV_B -> IV_V progression."""
    if not sweep_rows:
        return {"passed": False, "reason": "empty sweep"}

    ordered = sorted(sweep_rows, key=lambda row: float(row["p"]))
    labels = [classify_morphology(row) for row in ordered]

    if reference_width is None:
        ref_rows = [row for row, label in zip(ordered, labels) if label == "I"]
        if ref_rows:
            reference_width = float(ref_rows[0].get("feature_width_median", 0.0))
        else:
            reference_width = float(ordered[0].get("feature_width_median", 1.0))

    width_ratios = [
        width_ratio_to_reference(row, reference_width) for row in ordered
    ]
    width_ok = all(0.5 <= ratio <= 2.5 for ratio in width_ratios if ratio > 0)

    phase_order = ["I", "IIIL", "IV_B", "IV_V"]
    seen: list[str] = []
    for label in labels:
        if label in ("fail", "empty"):
            continue
        if not seen or label != seen[-1]:
            seen.append(label)

    order_index = {name: idx for idx, name in enumerate(phase_order)}
    monotonic = True
    last_idx = -1
    for label in seen:
        idx = order_index.get(label, -1)
        if idx < 0:
            continue
        if idx < last_idx:
            monotonic = False
            break
        last_idx = max(last_idx, idx)

    has_iv_b = "IV_B" in seen
    has_iv_v = "IV_V" in seen
    skips_iv_b = has_iv_v and not has_iv_b

    branch_peak = max((row.get("branch_arm_density", 0.0) for row in ordered), default=0.0)
    peak_arm_p = max(
        ordered,
        key=lambda row: row.get("branch_arm_density", 0.0),
        default={"p": 0.0},
    ).get("p", 0.0)
    peak_labyrinth_p = max(
        ordered,
        key=lambda row: row.get("labyrinth_score", 0.0),
        default={"p": 1.0},
    ).get("p", 1.0)
    branch_before_elong = branch_peak >= 0.02 and float(peak_arm_p) <= float(peak_labyrinth_p) + 0.05

    stationary_ok = all(
        row.get("is_stationary", row.get("coarsening_detected", 1.0) == 0.0)
        for row in ordered
        if "is_stationary" in row
    ) or all(row.get("coarsening_detected", 0.0) == 0.0 for row in ordered)

    passed = (
        monotonic
        and width_ok
        and has_iv_b
        and has_iv_v
        and "IIIL" in seen
        and not skips_iv_b
        and branch_before_elong
        and stationary_ok
        and len(seen) >= 4
    )

    return {
        "passed": passed,
        "labels": labels,
        "seen_phases": seen,
        "monotonic": monotonic,
        "width_ok": width_ok,
        "width_ratios": width_ratios,
        "reference_width": reference_width,
        "has_iv_b": has_iv_b,
        "has_iv_v": has_iv_v,
        "branch_before_elongation": branch_before_elong,
        "stationary_ok": stationary_ok,
    }
