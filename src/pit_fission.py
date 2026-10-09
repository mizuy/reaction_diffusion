#!/usr/bin/env python3
"""Finite-pit constraints shared by the non-reaction-diffusion prototypes.

The reduced field models can stabilize stripes/labyrinths that extend across the
domain. Real crypt openings behave more like finite biological units: once one
opening becomes too long or too large it should neck and split. These helpers add
that component-level constraint without drawing target geometry directly.
"""

from __future__ import annotations

import cv2
import numpy as np

from evaluate_pit_steps import pattern_binary, zhang_suen_thin


def component_fission_mask(
    pit: np.ndarray,
    *,
    max_area_fraction: float = 0.030,
    max_length_fraction: float = 0.30,
    band_width: float = 2.0,
) -> np.ndarray:
    """Return a soft mask where overgrown pit components should constrict.

    Components are considered overgrown when either their binary area or their
    skeleton length exceeds the configured fraction of the image area/width. The
    constriction is placed through the component centroid, perpendicular to its
    principal axis, mimicking necking before crypt fission.
    """

    binary = pattern_binary(pit)
    skeleton = zhang_suen_thin(binary)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    if component_count <= 1:
        return np.zeros_like(pit, dtype=np.float32)

    size = float(min(pit.shape))
    max_area = max(8.0, max_area_fraction * pit.size)
    max_length = max(6.0, max_length_fraction * size)
    mask = np.zeros_like(pit, dtype=np.float32)

    for label in range(1, component_count):
        component = labels == label
        area = float(stats[label, cv2.CC_STAT_AREA])
        sk_len = float((skeleton & component).sum())
        if area <= max_area and sk_len <= max_length:
            continue

        ys, xs = np.where(component)
        if xs.size < 4:
            continue

        coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]

        along = centered @ major
        across = centered @ np.array([-major[1], major[0]], dtype=np.float32)
        half_width = max(band_width, 0.12 * float(np.sqrt(max(area, 1.0))))
        neck = (np.abs(along) <= band_width) & (np.abs(across) <= half_width)
        if not bool(neck.any()):
            continue
        mask[ys[neck], xs[neck]] = 1.0

    if not bool(mask.any()):
        return mask
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.2, sigmaY=1.2)
    max_value = float(mask.max())
    if max_value > 1.0e-8:
        mask /= max_value
    return mask.astype(np.float32)


def graded_fission_mask(
    binary: np.ndarray,
    p_map: np.ndarray,
    *,
    max_length_at_p,
    band_width: float = 2.0,
) -> np.ndarray:
    """Necking mask with a progression-dependent length limit.

    Each connected component is measured by skeleton length and compared to
    ``max_length_at_p(p_local)``, where ``p_local`` is the mean progression
    value over the component.  ``max_length_at_p`` may return ``np.inf`` to
    disable fission (IV_V mesh phase).  The neck is placed at the median of
    the along-axis coordinate (always on the body, unlike the centroid of a
    curved component), perpendicular to the principal axis.
    """
    mask = np.zeros(binary.shape, dtype=np.float32)
    if not bool(binary.any()):
        return mask

    skeleton = zhang_suen_thin(binary)
    component_count, labels = cv2.connectedComponents(binary.astype(np.uint8), 8)

    for label in range(1, component_count):
        component = labels == label
        sk_len = float((skeleton & component).sum())
        p_local = float(p_map[component].mean())
        limit = float(max_length_at_p(p_local))
        if not np.isfinite(limit) or sk_len <= limit:
            continue

        ys, xs = np.where(component)
        if xs.size < 8:
            continue
        coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, int(np.argmax(eigvals))]

        along = centered @ major
        neck_at = float(np.median(along))
        neck = np.abs(along - neck_at) <= band_width
        if not bool(neck.any()):
            continue
        mask[ys[neck], xs[neck]] = 1.0

    if not bool(mask.any()):
        return mask
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.2, sigmaY=1.2)
    max_value = float(mask.max())
    if max_value > 1.0e-8:
        mask /= max_value
    return mask.astype(np.float32)


def apply_component_fission(
    field: np.ndarray,
    pit: np.ndarray,
    *,
    strength: float = 0.45,
    max_area_fraction: float = 0.030,
    max_length_fraction: float = 0.30,
    band_width: float = 2.0,
) -> np.ndarray:
    """Suppress the middle of overgrown pit components in a scalar field."""

    mask = component_fission_mask(
        pit,
        max_area_fraction=max_area_fraction,
        max_length_fraction=max_length_fraction,
        band_width=band_width,
    )
    if not bool(mask.any()):
        return field

    dynamic_range = float(np.percentile(field, 99) - np.percentile(field, 1))
    if dynamic_range <= 1.0e-8:
        dynamic_range = 1.0
    return (field - strength * dynamic_range * mask).astype(np.float32)


def local_resource(pit: np.ndarray, *, sigma: float, depletion: float) -> np.ndarray:
    """Resource field that decreases around large/long occupied pit regions."""

    load = cv2.GaussianBlur(pit.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return (1.0 / (1.0 + depletion * load)).astype(np.float32)
