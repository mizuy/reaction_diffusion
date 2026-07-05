#!/usr/bin/env python3
"""Evaluate reaction-diffusion changes for Kudo pit-like patterns.

This script keeps the pattern source in the equations. It does not draw pit
geometry directly; every case starts from the same random perturbation and then
changes only reaction terms or slow parameter fields.

Example:
    python3 src/evaluate_pit_steps.py --output artifacts/pit_step_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np


KERNEL = np.array(
    [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class EvalRunSettings:
    """Initial-condition / run settings shared between batch eval and the UI."""

    size: int = 160
    steps: int = 3500
    seed: int = 7
    seed_density: float = 0.035


DEFAULT_EVAL_RUN = EvalRunSettings()


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
    # Spatial saturation: q=q_core on flat high-B cores, q=q_tip elsewhere (tips grow).
    # When reaction_saturation_tip <= 0, use uniform reaction_saturation only.
    reaction_saturation_tip: float = 0.0
    cubic_damping: float = 0.0
    static_env_scale: float = 0.0
    feed_env_sensitivity: float = 0.0
    k_env_sensitivity: float = 0.0
    dynamic_env: bool = False
    env_rate: float = 0.0
    env_diffusion: float = 0.0
    env_source: float = 0.0
    env_decay: float = 0.0
    # Extra A consumption inside B-rich regions (step1c_b experiment).
    a_depletion: float = 0.0
    # Decay applied to flat high-B stripe cores while keeping tips alive.
    uniform_core_decay: float = 0.0
    uniform_core_b_min: float = 0.55
    uniform_core_grad_max: float = 0.035
    # Refractory R: B residence memory; kills old corridors while tips stay active.
    #   R_t += refractory_rate * B - refractory_decay * R
    #   B   += ... - refractory_strength * R * B
    refractory_rate: float = 0.0
    refractory_decay: float = 0.0
    refractory_strength: float = 0.0
    # Proposal A: fast, far-diffusing local inhibitor H secreted by B.
    #   H_t = inhibitor_diffusion * lap(H) + inhibitor_source * B - inhibitor_decay * H
    #   B   += ... - inhibitor_strength * B * H
    inhibitor_strength: float = 0.0
    inhibitor_diffusion: float = 0.0
    inhibitor_source: float = 0.0
    inhibitor_decay: float = 0.0
    # Proposal W: bistable plateau pins stripe WIDTH/amplitude away from (f, k).
    #   B += bistable_strength * B * (1 - B) * (B - bistable_threshold)
    # Stable states B=0 / B=1, threshold in between; width ~ sqrt(d_b / strength).
    bistable_strength: float = 0.0
    bistable_threshold: float = 0.3
    # Proposal M: global feedback that pins COVERAGE (mean B) to a target.
    #   feed_eff = feed + coverage_feedback * (coverage_target - mean(B))
    coverage_feedback: float = 0.0
    coverage_target: float = 0.0
    # Polarity P (2D vector) + anisotropic B diffusion (Step 1e prototype).
    #   P_t = d_p Lap(P) + align * ∇B - decay * P  (then unit direction for D tensor)
    #   D = d_b n n^T + d_b_across (I - n n^T)  on B only; A stays isotropic.
    polarity_diffusion: float = 0.0
    polarity_align_rate: float = 0.0
    polarity_decay: float = 0.0
    # Lateral (across-P) B diffusion; 0 = isotropic d_b only.
    d_b_across: float = 0.0
    # Blend isotropic vs anisotropic B diffusion (0 = all isotropic, 1 = full aniso).
    anisotropic_strength: float = 0.0


@dataclass
class StepResult:
    config: ModelConfig
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    h: np.ndarray
    r: np.ndarray
    p_x: np.ndarray
    p_y: np.ndarray
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


def reaction_term_with_config(
    a: np.ndarray, b: np.ndarray, config: ModelConfig
) -> np.ndarray:
    """Reaction with optional spatial saturation (q_core on flat cores, q_tip on tips)."""
    ab2 = a * (b**2)
    if config.reaction_saturation_tip <= 0.0:
        return reaction_term(a, b, config.reaction_saturation)

    core = uniform_core_mask(
        b, config.uniform_core_b_min, config.uniform_core_grad_max
    )
    q = np.where(
        core,
        config.reaction_saturation,
        config.reaction_saturation_tip,
    ).astype(np.float32)
    return ab2 / (1.0 + q * (b**2))


def uniform_core_mask(b: np.ndarray, b_min: float, grad_max: float) -> np.ndarray:
    """Flat, high-B stripe interiors (low gradient) but not tips (high gradient)."""
    gy, gx = np.gradient(b)
    grad = np.sqrt(gx * gx + gy * gy)
    return (b > b_min) & (grad < grad_max)


def uses_polarity(config: ModelConfig) -> bool:
    """True when polarity field and/or anisotropic B diffusion is active."""
    return (
        config.polarity_align_rate > 0.0
        or config.polarity_diffusion > 0.0
        or config.d_b_across > 0.0
        or config.anisotropic_strength > 0.0
    )


def make_initial_polarity(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Weak random unit polarity so anisotropy is defined before B gradients exist."""
    rng = np.random.default_rng(seed + 20_011)
    angle = rng.uniform(0.0, 2.0 * math.pi, (size, size)).astype(np.float32)
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def advance_polarity(
    p_x: np.ndarray,
    p_y: np.ndarray,
    b: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not uses_polarity(config):
        return p_x, p_y

    lap_px = cv2.filter2D(p_x, -1, KERNEL)
    lap_py = cv2.filter2D(p_y, -1, KERNEL)
    gy, gx = np.gradient(b.astype(np.float64))
    next_px = (
        p_x
        + config.polarity_diffusion * lap_px
        + config.polarity_align_rate * gx.astype(np.float32)
        - config.polarity_decay * p_x
    )
    next_py = (
        p_y
        + config.polarity_diffusion * lap_py
        + config.polarity_align_rate * gy.astype(np.float32)
        - config.polarity_decay * p_y
    )
    return next_px.astype(np.float32), next_py.astype(np.float32)


def polarity_direction(
    p_x: np.ndarray,
    p_y: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit growth direction; falls back to ∇B where |P| is tiny."""
    gy, gx = np.gradient(b.astype(np.float64))
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    mag_p = np.sqrt(p_x * p_x + p_y * p_y)
    mag_g = np.sqrt(gx * gx + gy * gy)
    weak_p = mag_p < 0.02
    weak_g = mag_g < 1.0e-6
    nx = np.where(weak_p & ~weak_g, gx / (mag_g + 1.0e-8), p_x)
    ny = np.where(weak_p & ~weak_g, gy / (mag_g + 1.0e-8), p_y)
    nm = np.sqrt(nx * nx + ny * ny) + 1.0e-8
    return (nx / nm).astype(np.float32), (ny / nm).astype(np.float32)


def anisotropic_laplacian(
    field: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    d_parallel: float,
    d_perp: float,
) -> np.ndarray:
    """∇·(D∇f) with D = d_parallel n n^T + d_perp (I - n n^T), n = (nx, ny)."""
    f = field.astype(np.float64)
    gy, gx = np.gradient(f)
    gyy, gxy = np.gradient(gy)
    _, gxx = np.gradient(gx)
    d2_along = nx.astype(np.float64) ** 2 * gxx + 2.0 * nx * ny * gxy + ny.astype(
        np.float64
    ) ** 2 * gyy
    mx = -ny.astype(np.float64)
    my = nx.astype(np.float64)
    d2_across = mx * mx * gxx + 2.0 * mx * my * gxy + my * my * gyy
    return (d_parallel * d2_along + d_perp * d2_across).astype(np.float32)


def advance_reaction_diffusion(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
    p_x: np.ndarray,
    p_y: np.ndarray,
    config: ModelConfig,
    feed: np.ndarray | float,
    k: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One explicit Euler step given already-resolved feed / kill fields."""
    # Proposal M: pin coverage (mean B) by biasing the feed term globally.
    if config.coverage_feedback > 0.0:
        coverage_error = config.coverage_target - float(b.mean())
        feed = np.clip(feed + config.coverage_feedback * coverage_error, 0.0, 0.12)

    reaction = reaction_term_with_config(a, b, config)

    lap_a = cv2.filter2D(a, -1, KERNEL)
    next_p_x, next_p_y = advance_polarity(p_x, p_y, b, config)
    lap_iso_b = cv2.filter2D(b, -1, KERNEL)
    if uses_polarity(config) and config.anisotropic_strength > 0.0:
        nx, ny = polarity_direction(next_p_x, next_p_y, b)
        d_perp = (
            config.d_b_across
            if config.d_b_across > 0.0
            else config.d_b * 0.4
        )
        lap_aniso_b = anisotropic_laplacian(b, nx, ny, 1.0, d_perp / max(config.d_b, 1.0e-6))
        mix = min(max(config.anisotropic_strength, 0.0), 1.0)
        lap_b = config.d_b * ((1.0 - mix) * lap_iso_b + mix * lap_aniso_b)
    else:
        lap_b = config.d_b * lap_iso_b

    a_consumption = reaction
    if config.a_depletion > 0.0:
        # Extra substrate drain proportional to local A and B (step1c_b).
        a_consumption = a_consumption + config.a_depletion * a * b
    next_a = a + config.d_a * lap_a - a_consumption + feed * (1.0 - a)

    b_loss = (k + feed) * b + config.cubic_damping * (b**3)
    if config.uniform_core_decay > 0.0:
        core = uniform_core_mask(
            b, config.uniform_core_b_min, config.uniform_core_grad_max
        )
        b_loss = b_loss + config.uniform_core_decay * core.astype(np.float32) * b
    if config.inhibitor_strength > 0.0:
        b_loss = b_loss + config.inhibitor_strength * b * h
    if config.refractory_strength > 0.0:
        # Kill old flat cores using R memory; tips (high |∇B|) stay active.
        core = uniform_core_mask(
            b, config.uniform_core_b_min, config.uniform_core_grad_max
        )
        b_loss = (
            b_loss
            + config.refractory_strength * r * b * core.astype(np.float32)
        )
    next_b = b + lap_b + reaction - b_loss

    # Proposal W: bistable plateau pins the B "on" value (and thus stripe width)
    # at B=1 independent of (f, k). Threshold beta sets the basin boundary.
    if config.bistable_strength > 0.0:
        bistable = (
            config.bistable_strength
            * b
            * (1.0 - b)
            * (b - config.bistable_threshold)
        )
        next_b = next_b + bistable

    next_h = h
    if (
        config.inhibitor_diffusion > 0.0
        or config.inhibitor_source > 0.0
        or config.inhibitor_decay > 0.0
    ):
        lap_h = cv2.filter2D(h, -1, KERNEL)
        next_h = (
            h
            + config.inhibitor_diffusion * lap_h
            + config.inhibitor_source * b
            - config.inhibitor_decay * h
        )
        next_h = np.clip(next_h, 0.0, 10.0)

    next_c = c
    if config.dynamic_env:
        lap_c = cv2.filter2D(c, -1, KERNEL)
        b_centered = b - float(b.mean())
        next_c = c + config.env_rate * (
            config.env_diffusion * lap_c
            + config.env_source * b_centered
            - config.env_decay * c
        )
        next_c = np.clip(next_c, -3.0, 3.0)

    next_r = r
    if config.refractory_rate > 0.0 or config.refractory_decay > 0.0:
        next_r = (
            r + config.refractory_rate * b - config.refractory_decay * r
        ).astype(np.float32)
        next_r = np.clip(next_r, 0.0, 10.0)

    return (
        np.clip(next_a, 0.0, 1.0),
        np.clip(next_b, 0.0, 1.0),
        next_c.astype(np.float32),
        next_h.astype(np.float32),
        next_r.astype(np.float32),
        next_p_x,
        next_p_y,
    )


def calc_step(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
    p_x: np.ndarray,
    p_y: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feed, k = effective_parameters(config, c)
    return advance_reaction_diffusion(a, b, c, h, r, p_x, p_y, config, feed, k)


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
    h = np.zeros((size, size), dtype=np.float32)
    r = np.zeros((size, size), dtype=np.float32)
    if uses_polarity(config):
        p_x, p_y = make_initial_polarity(size, seed)
    else:
        p_x = np.zeros((size, size), dtype=np.float32)
        p_y = np.zeros((size, size), dtype=np.float32)

    snapshot_steps = set(np.linspace(0, steps, snapshot_count, dtype=int).tolist())
    snapshots: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    if 0 in snapshot_steps:
        snapshots.append((0, a.copy(), b.copy(), c.copy()))

    for step in range(1, steps + 1):
        a, b, c, h, r, p_x, p_y = calc_step(a, b, c, h, r, p_x, p_y, config)
        if step in snapshot_steps:
            snapshots.append((step, a.copy(), b.copy(), c.copy()))

    return StepResult(
        config=config, a=a, b=b, c=c, h=h, r=r, p_x=p_x, p_y=p_y, snapshots=snapshots
    )


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


# Clockwise 8-neighbor slots around a center pixel (N, NE, E, SE, S, SW, W, NW).
_RING_OFFSETS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


def skeleton_degree_map(skeleton: np.ndarray) -> np.ndarray:
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, neighbor_kernel)
    return neighbors.astype(np.int16) - skeleton.astype(np.int16)


def _skeleton_neighbors_at(sk: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    h, w = sk.shape
    return [
        (y + dy, x + dx)
        for dy, dx in _RING_OFFSETS
        if 0 <= y + dy < h and 0 <= x + dx < w and sk[y + dy, x + dx]
    ]


def default_min_branch_arm_length(size: int) -> int:
    """Minimum corridor length (in skeleton pixels) for each arm of a fork."""
    return max(4, size // 32)


def junction_cluster_labels(sk: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """8-connected clusters of skeleton pixels with degree >= 3."""
    degree = skeleton_degree_map(sk)
    junction = (sk & (degree >= 3)).astype(np.uint8)
    count, labels = cv2.connectedComponents(junction, connectivity=8)
    clusters: list[list[tuple[int, int]]] = []
    for label in range(1, count):
        coords = list(zip(*np.where(labels == label)))
        clusters.append(coords)
    return labels, clusters


def trace_arm_from_cluster_exit(
    sk: np.ndarray,
    start: tuple[int, int],
    cluster_set: set[tuple[int, int]],
    junction_labels: np.ndarray,
    own_label: int,
    *,
    max_steps: int = 512,
) -> tuple[int, tuple[int, int]]:
    """Follow one corridor leaving a junction cluster; return length and end pixel."""
    h, w = sk.shape
    cy, cx = start
    length = 1
    prev: tuple[int, int] | None = None

    for _ in range(max_steps - 1):
        degree = skeleton_degree_map(sk)[cy, cx]
        if degree == 1:
            break
        if degree >= 3:
            end_label = int(junction_labels[cy, cx])
            if end_label != 0 and end_label != own_label:
                break
            if (cy, cx) in cluster_set:
                break

        nbrs = [
            (cy + dy, cx + dx)
            for dy, dx in _RING_OFFSETS
            if 0 <= cy + dy < h
            and 0 <= cx + dx < w
            and sk[cy + dy, cx + dx]
            and (cy + dy, cx + dx) != prev
            and (cy + dy, cx + dx) not in cluster_set
        ]
        if not nbrs:
            nbrs = [
                (cy + dy, cx + dx)
                for dy, dx in _RING_OFFSETS
                if 0 <= cy + dy < h
                and 0 <= cx + dx < w
                and sk[cy + dy, cx + dx]
                and (cy + dy, cx + dx) != prev
            ]
            if not nbrs or (nbrs[0] in cluster_set and len(nbrs) == 1):
                break
        if len(nbrs) != 1:
            break
        prev = (cy, cx)
        cy, cx = nbrs[0]
        length += 1

    return length, (cy, cx)


def cluster_arm_lengths(
    sk: np.ndarray,
    cluster: list[tuple[int, int]],
    junction_labels: np.ndarray,
    own_label: int,
    *,
    max_steps: int = 512,
) -> list[int]:
    """Distinct arm lengths leaving a junction cluster along the skeleton."""
    cluster_set = set(cluster)
    seen_ends: list[tuple[int, int]] = []
    lengths: list[int] = []

    for y, x in cluster:
        for ny, nx in _skeleton_neighbors_at(sk, y, x):
            if (ny, nx) in cluster_set:
                continue
            arm_len, end = trace_arm_from_cluster_exit(
                sk,
                (ny, nx),
                cluster_set,
                junction_labels,
                own_label,
                max_steps=max_steps,
            )
            if arm_len <= 0:
                continue
            if any(abs(end[0] - ey) <= 1 and abs(end[1] - ex) <= 1 for ey, ex in seen_ends):
                continue
            seen_ends.append(end)
            lengths.append(arm_len)

    return lengths


def is_three_arm_junction_cluster(
    arm_lengths: list[int], *, min_arm_length: int
) -> bool:
    """Exactly three distinct arms, each at least min_arm_length along the skeleton."""
    if len(arm_lengths) != 3:
        return False
    return all(length >= min_arm_length for length in arm_lengths)


def prune_skeleton_spurs(skeleton: np.ndarray, min_length: int = 4) -> np.ndarray:
    """Delete short endpoint branches before junction counting."""
    sk = skeleton.astype(np.uint8).copy()
    h, w = sk.shape
    if min_length <= 1:
        return sk.astype(bool)

    changed = True
    while changed:
        changed = False
        degree = skeleton_degree_map(sk.astype(bool))
        for y, x in zip(*np.where((sk > 0) & (degree == 1))):
            path = [(y, x)]
            cy, cx = y, x
            prev: tuple[int, int] | None = None
            while True:
                nbrs = [
                    (cy + dy, cx + dx)
                    for dy, dx in _RING_OFFSETS
                    if 0 <= cy + dy < h
                    and 0 <= cx + dx < w
                    and sk[cy + dy, cx + dx]
                    and (cy + dy, cx + dx) != prev
                ]
                if not nbrs:
                    break
                if len(nbrs) > 1:
                    break
                ny, nx = nbrs[0]
                prev, cy, cx = (cy, cx), ny, nx
                path.append((ny, nx))
                if skeleton_degree_map(sk.astype(bool))[ny, nx] != 2:
                    break
            if len(path) < min_length:
                for py, px in path:
                    sk[py, px] = 0
                changed = True
    return sk.astype(bool)


def prepare_skeleton_topology(
    skeleton: np.ndarray,
    *,
    prune_spurs: int = 4,
    min_arm_length: int | None = None,
    max_arm_steps: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pruned_skeleton, three_way_branch_mask) for metrics and overlays.

    Junction pixels with degree >= 3 are merged into 8-connected clusters. From
    each cluster, corridors are traced along the skeleton until the next junction
    cluster or endpoint. A fork is counted only when there are exactly three
    distinct corridors and each spans at least ``min_arm_length`` pixels.
    """
    sk = (
        prune_skeleton_spurs(skeleton, min_length=prune_spurs)
        if prune_spurs > 1
        else skeleton.astype(bool)
    )
    if min_arm_length is None:
        min_arm_length = default_min_branch_arm_length(sk.shape[0])

    junction_labels, clusters = junction_cluster_labels(sk)
    branch = np.zeros_like(sk, dtype=bool)
    for label, cluster in enumerate(clusters, start=1):
        arms = cluster_arm_lengths(
            sk, cluster, junction_labels, label, max_steps=max_arm_steps
        )
        if not is_three_arm_junction_cluster(arms, min_arm_length=min_arm_length):
            continue
        for y, x in cluster:
            branch[y, x] = True
    return sk, branch


def three_way_branch_mask(
    skeleton: np.ndarray,
    *,
    prune_spurs: int = 4,
    min_arm_length: int | None = None,
) -> np.ndarray:
    """Mask of 3-way junction clusters whose three corridors extend along the skeleton."""
    _, branch = prepare_skeleton_topology(
        skeleton, prune_spurs=prune_spurs, min_arm_length=min_arm_length
    )
    return branch


def skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, neighbor_kernel)
    neighbors = neighbors.astype(np.int16) - skeleton.astype(np.int16)
    return skeleton & (neighbors == 1)


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
    skeleton, branch_mask = prepare_skeleton_topology(zhang_suen_thin(binary))

    skeleton_pixels = int(skeleton.sum())
    branch_points = int(branch_mask.sum())
    endpoints = int(skeleton_endpoints(skeleton).sum())

    # Mean length of skeleton runs between junctions/endpoints: this measures
    # directly "how far a line extends" before it branches or stops.
    segments = skeleton & ~branch_mask
    seg_components, _, seg_stats, _ = cv2.connectedComponentsWithStats(
        segments.astype(np.uint8), 8
    )
    if seg_components > 1:
        mean_segment_length = float(seg_stats[1:, cv2.CC_STAT_AREA].mean())
    else:
        mean_segment_length = 0.0

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
    # High when each component is a short tree (good Type IV), low for isolated
    # dashes (fragmented but unbranched) or one giant maze.
    branch_per_component = branch_points / max(float(skeleton_components), 1.0)
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
        "mean_segment_length": float(mean_segment_length),
        "branch_per_component": float(branch_per_component),
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
    skeleton, branch_points = prepare_skeleton_topology(zhang_suen_thin(binary))
    base = cv2.cvtColor(normalize_u8(b), cv2.COLOR_GRAY2BGR)
    base[binary] = (80, 80, 80)

    endpoints = skeleton_endpoints(skeleton)

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
            "red=3-arm fork cyan=end",
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
            name="step1c_a_strong_cubic",
            description="Step 1b with stronger B^3 damping to slow late-time stripe coarsening.",
            feed=0.033,
            k=0.056,
            reaction_saturation=1.6,
            cubic_damping=0.028,
        ),
        ModelConfig(
            name="step1c_b_a_depletion",
            description="Saturated reaction plus extra A consumption in B-rich regions.",
            feed=0.033,
            k=0.056,
            reaction_saturation=1.6,
            cubic_damping=0.018,
            a_depletion=0.25,
        ),
        ModelConfig(
            name="step1c_c_uniform_core",
            description="Saturated reaction plus decay on uniform high-B stripe cores (tips kept).",
            feed=0.033,
            k=0.056,
            reaction_saturation=1.6,
            cubic_damping=0.018,
            uniform_core_decay=0.12,
        ),
        ModelConfig(
            name="step1d_pit_iiil_spatial_sat",
            description=(
                "Spatial saturation: high q on flat B cores (IIIL body), low q on tips "
                "for wedge-like branching without global stripe coarsening."
            ),
            feed=0.033,
            k=0.056,
            reaction_saturation=1.5,
            reaction_saturation_tip=0.5,
            cubic_damping=0.018,
            uniform_core_decay=0.10,
        ),
        ModelConfig(
            name="step1d_b_pit_iiil_spatial_refractory",
            description=(
                "step1d spatial sat plus refractory R to cap corridor length and "
                "slow labyrinth coarsening."
            ),
            feed=0.033,
            k=0.056,
            reaction_saturation=1.5,
            reaction_saturation_tip=0.5,
            cubic_damping=0.018,
            uniform_core_decay=0.10,
            refractory_rate=0.035,
            refractory_decay=0.08,
            refractory_strength=0.15,
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
        ModelConfig(
            name="step4_pit_iv_inhibitor",
            description=(
                "Proposal A: fast far-diffusing local inhibitor H secreted by B. "
                "Imposes a length scale so tips stall/split into short branches."
            ),
            feed=0.033,
            k=0.056,
            inhibitor_strength=0.05,
            inhibitor_diffusion=0.60,
            inhibitor_source=0.05,
            inhibitor_decay=0.10,
        ),
        ModelConfig(
            name="step4b_pit_iv_inhibitor_cubic",
            description=(
                "Proposal A inhibitor combined with mild B^3 damping; aims for short "
                "branched trees rather than a space-filling maze."
            ),
            feed=0.033,
            k=0.056,
            cubic_damping=0.008,
            inhibitor_strength=0.06,
            inhibitor_diffusion=0.60,
            inhibitor_source=0.05,
            inhibitor_decay=0.10,
        ),
        ModelConfig(
            name="step5_pit_iv_bistable_width",
            description=(
                "Proposal W: bistable plateau pins stripe width away from (f, k) "
                "so variation flows into morphology instead of thickness."
            ),
            feed=0.033,
            k=0.056,
            bistable_strength=0.10,
            bistable_threshold=0.22,
        ),
        ModelConfig(
            name="step5b_pit_iv_bistable_inhibitor",
            description=(
                "Proposal W + A: pinned width (bistable) plus local inhibitor H "
                "(pinned spacing); aims for fixed-width short branched worms."
            ),
            feed=0.033,
            k=0.056,
            bistable_strength=0.10,
            bistable_threshold=0.22,
            inhibitor_strength=0.03,
            inhibitor_diffusion=0.60,
            inhibitor_source=0.04,
            inhibitor_decay=0.12,
        ),
        ModelConfig(
            name="step5c_pit_iv_three_channel",
            description=(
                "Proposal W + A + M: width (bistable), spacing (inhibitor) and "
                "coverage (feed feedback) all pinned; topology is the only free channel."
            ),
            feed=0.033,
            k=0.056,
            bistable_strength=0.10,
            bistable_threshold=0.22,
            inhibitor_strength=0.03,
            inhibitor_diffusion=0.60,
            inhibitor_source=0.04,
            inhibitor_decay=0.12,
            coverage_feedback=0.05,
            coverage_target=0.30,
        ),
        ModelConfig(
            name="step1e_pit_iv_polarity_aniso",
            description=(
                "Polarity P aligned to ∇B plus anisotropic B diffusion: strong along P, "
                "weak across P to limit lateral stripe coarsening."
            ),
            feed=0.033,
            k=0.056,
            polarity_diffusion=0.10,
            polarity_align_rate=0.06,
            polarity_decay=0.03,
            d_b_across=0.22,
            anisotropic_strength=0.45,
        ),
        ModelConfig(
            name="step1e_b_pit_iv_polarity_1c",
            description="step1c_c uniform core plus polarity/anisotropic B diffusion.",
            feed=0.033,
            k=0.056,
            reaction_saturation=1.6,
            cubic_damping=0.018,
            uniform_core_decay=0.12,
            polarity_diffusion=0.10,
            polarity_align_rate=0.06,
            polarity_decay=0.03,
            d_b_across=0.22,
            anisotropic_strength=0.0,
        ),
        ModelConfig(
            name="step1e_c_pit_iv_polarity_1d",
            description="step1d spatial saturation plus polarity/anisotropic B.",
            feed=0.033,
            k=0.056,
            reaction_saturation=1.5,
            reaction_saturation_tip=0.5,
            cubic_damping=0.018,
            uniform_core_decay=0.10,
            polarity_diffusion=0.10,
            polarity_align_rate=0.06,
            polarity_decay=0.03,
            d_b_across=0.22,
            anisotropic_strength=0.30,
        ),
    ]

    if suite == "references":
        return references
    if suite == "steps":
        return steps
    return references + steps


def load_eval_case(
    metrics_path: Path, case_name: str
) -> tuple[ModelConfig, EvalRunSettings]:
    """Rebuild a ModelConfig and its run settings from a metrics.json row."""
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    row = next((item for item in data if item.get("name") == case_name), None)
    if row is None:
        raise SystemExit(f"case {case_name!r} not found in {metrics_path}")

    valid = {f.name for f in fields(ModelConfig)}
    config = ModelConfig(**{key: row[key] for key in valid if key in row})
    settings = EvalRunSettings(
        size=int(row.get("size", DEFAULT_EVAL_RUN.size)),
        steps=int(row.get("steps", DEFAULT_EVAL_RUN.steps)),
        seed=int(row.get("seed", DEFAULT_EVAL_RUN.seed)),
        seed_density=float(row.get("seed_density", DEFAULT_EVAL_RUN.seed_density)),
    )
    return config, settings


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
    parser.add_argument(
        "--size", type=int, default=DEFAULT_EVAL_RUN.size, help="Simulation width/height."
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_EVAL_RUN.steps, help="Simulation steps per case."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_EVAL_RUN.seed, help="Seed shared by all cases."
    )
    parser.add_argument(
        "--seed-density",
        type=float,
        default=DEFAULT_EVAL_RUN.seed_density,
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
