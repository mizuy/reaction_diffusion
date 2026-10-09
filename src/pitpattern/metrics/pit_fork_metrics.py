#!/usr/bin/env python3
"""Fork (Y-junction) and unit-size metrics robust to skeleton staircases.

The project fork detector (evaluate_pit_steps.prepare_skeleton_topology) counts
8-neighbour degree, so diagonal staircases of a Zhang-Suen skeleton look like
junctions with 1-pixel arms and real forks are rejected.  Here a junction is a
skeleton pixel whose crossing number (0->1 transitions around its 8-ring) is
at least 3; nearby junction pixels are merged and a fork is a cluster from
which at least three skeleton segments of length >= arm_min leave.

All lengths are in pixels; pass arm_min tied to the pattern scale (not to the
domain size) so the counts are comparable across domain sizes.
"""

from __future__ import annotations

import cv2
import numpy as np

from pitpattern.metrics.evaluate_pit_steps import prune_skeleton_spurs, zhang_suen_thin


def crossing_number(sk: np.ndarray) -> np.ndarray:
    p = np.pad(sk.astype(np.int8), 1)
    ring = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    cn = np.zeros(sk.shape, dtype=np.int8)
    for i in range(8):
        cn += (ring[i] == 0) & (ring[(i + 1) % 8] == 1)
    return np.where(sk, cn, 0)


def fork_metrics(binary: np.ndarray, *, arm_min: float, spur: int | None = None) -> dict[str, float]:
    binary = binary.astype(bool)
    out = dict(fork_count=0.0, fork_components=0.0, unit_count=0.0, fork_component_fraction=0.0,
               forks_per_unit=0.0, unit_length_median=0.0, unit_length_p90=0.0, skeleton_length=0.0)
    if not binary.any():
        return out
    sk = zhang_suen_thin(binary)
    sk = prune_skeleton_spurs(sk, min_length=int(spur if spur is not None else max(2, arm_min // 2)))
    if not sk.any():
        return out
    cn = crossing_number(sk)
    junction = cn >= 3
    jzone = cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & sk
    n_cl, cl_lab = cv2.connectedComponents(jzone.astype(np.uint8), connectivity=8)
    seg = sk & ~jzone
    n_seg, seg_lab, seg_stats, _ = cv2.connectedComponentsWithStats(seg.astype(np.uint8), connectivity=8)
    seg_len = seg_stats[:, cv2.CC_STAT_AREA]
    # adjacency: dilate each cluster by one pixel and collect touching segment labels
    touch = cv2.dilate(cl_lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    # dilation of labels by max is ambiguous where clusters are within 2 px; they are merged by jzone already
    pairs = np.stack([touch[seg], seg_lab[seg]], axis=1)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    forks = np.zeros(n_cl, dtype=bool)
    if pairs.size:
        pairs = np.unique(pairs, axis=0)
        long_pairs = pairs[seg_len[pairs[:, 1]] >= arm_min]
        counts = np.bincount(long_pairs[:, 0], minlength=n_cl)
        forks = counts >= 3
    forks[0] = False

    n_unit, unit_lab, unit_stats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    lengths = unit_stats[1:, cv2.CC_STAT_AREA].astype(float)
    fork_units = set()
    if forks.any():
        for c in np.flatnonzero(forks):
            ys, xs = np.nonzero(cl_lab == c)
            fork_units.add(int(unit_lab[ys[0], xs[0]]))
    n_units = n_unit - 1
    out.update(
        fork_count=float(forks.sum()),
        fork_components=float(len(fork_units)),
        unit_count=float(n_units),
        fork_component_fraction=float(len(fork_units) / max(n_units, 1)),
        forks_per_unit=float(forks.sum() / max(n_units, 1)),
        unit_length_median=float(np.median(lengths)) if lengths.size else 0.0,
        unit_length_p90=float(np.percentile(lengths, 90)) if lengths.size else 0.0,
        skeleton_length=float(sk.sum()),
    )
    return out


def fork_overlay(binary: np.ndarray, *, arm_min: float) -> np.ndarray:
    """RGB image: pit (lavender), skeleton (black), counted forks (red disks)."""
    binary = binary.astype(bool)
    sk = prune_skeleton_spurs(zhang_suen_thin(binary), min_length=int(max(2, arm_min // 2)))
    rgb = np.full((*binary.shape, 3), 238, np.uint8)
    rgb[binary] = (160, 150, 200)
    rgb[sk] = (20, 20, 20)
    cn = crossing_number(sk)
    jzone = cv2.dilate((cn >= 3).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & sk
    n_cl, cl_lab = cv2.connectedComponents(jzone.astype(np.uint8), connectivity=8)
    seg = sk & ~jzone
    _, seg_lab, seg_stats, _ = cv2.connectedComponentsWithStats(seg.astype(np.uint8), connectivity=8)
    touch = cv2.dilate(cl_lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    for c in range(1, n_cl):
        labs = np.unique(seg_lab[(touch == c) & seg])
        labs = labs[labs > 0]
        if (seg_stats[labs, cv2.CC_STAT_AREA] >= arm_min).sum() >= 3:
            ys, xs = np.nonzero(cl_lab == c)
            cv2.circle(rgb, (int(xs.mean()), int(ys.mean())), 3, (220, 30, 30), -1)
    return rgb
