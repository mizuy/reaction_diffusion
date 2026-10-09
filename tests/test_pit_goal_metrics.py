#!/usr/bin/env python3
"""Synthetic pattern tests for pit_goal_metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitpattern.metrics.pit_goal_metrics import (  # noqa: E402
    classify_morphology,
    evaluate_goal_sequence,
    measure_goal_pattern,
    measure_temporal_stability,
)


def _blank(size: int = 96) -> np.ndarray:
    return np.zeros((size, size), dtype=np.float32)


def synthetic_dots(size: int = 96, radius: int = 3, spacing: int = 14) -> np.ndarray:
    field = _blank(size)
    for y in range(spacing // 2, size - spacing // 2, spacing):
        for x in range(spacing // 2, size - spacing // 2, spacing):
            cv2.circle(field, (x, y), radius, 1.0, -1)
    return field


def synthetic_short_tubes(size: int = 96) -> np.ndarray:
    field = _blank(size)
    for y in range(12, size - 12, 16):
        for x in range(10, size - 22, 18):
            cv2.ellipse(field, (x + 4, y), (6, 3), 20, 0, 360, 1.0, -1)
    return field


def synthetic_y_branches(size: int = 96) -> np.ndarray:
    field = _blank(size)
    centers = [(24, 24), (72, 28), (48, 68)]
    for cx, cy in centers:
        cv2.line(field, (cx, cy), (cx + 14, cy), 1.0, 3)
        cv2.line(field, (cx, cy), (cx - 8, cy + 12), 1.0, 3)
        cv2.line(field, (cx, cy), (cx + 8, cy + 12), 1.0, 3)
    return field


def synthetic_long_parallel_grooves(size: int = 96) -> np.ndarray:
    field = _blank(size)
    for x in range(8, size - 8, 10):
        cv2.line(field, (x, 6), (x + 2, size - 6), 1.0, 3)
    return field


def synthetic_labyrinth_mesh(size: int = 96) -> np.ndarray:
    """Constant-width winding mesh with cross links — the IV_V target."""
    field = _blank(size)
    rng = np.random.default_rng(3)
    rows: list[list[tuple[int, int]]] = []
    for i, y0 in enumerate(range(10, size - 6, 12)):
        pts = []
        for x in range(6, size - 6, 6):
            y = int(y0 + 2.5 * np.sin(0.2 * x + i))
            pts.append((x, y))
        for (x0, y0_), (x1, y1_) in zip(pts[:-1], pts[1:]):
            cv2.line(field, (x0, y0_), (x1, y1_), 1.0, 3)
        rows.append(pts)
    for r in range(len(rows) - 1):
        a, b = rows[r], rows[r + 1]
        n = min(len(a), len(b))
        for j in range(1 + (r % 2), n - 1, 4):
            if rng.random() < 0.2:
                continue
            cv2.line(field, a[j], b[j], 1.0, 3)
    return field


def synthetic_thick_labyrinth(size: int = 96) -> np.ndarray:
    field = _blank(size)
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    blur = cv2.GaussianBlur(noise, (0, 0), 3.0)
    field[blur > 0.15] = 1.0
    field = cv2.dilate(field, np.ones((9, 9), np.uint8), iterations=1)
    return np.clip(field, 0.0, 1.0)


class TestPitGoalMetrics(unittest.TestCase):
    def test_dots_have_low_branch_and_moderate_width(self) -> None:
        field = synthetic_dots()
        metrics = measure_goal_pattern(field)
        self.assertGreater(metrics["component_count"], 5)
        self.assertLess(metrics["branch_density"], 0.08)
        self.assertGreater(metrics["feature_width_median"], 2.0)
        self.assertLess(metrics["feature_width_median"], 12.0)

    def test_y_branches_have_higher_branch_than_dots(self) -> None:
        dots = measure_goal_pattern(synthetic_dots())
        branches = measure_goal_pattern(synthetic_y_branches())
        self.assertGreater(branches["branch_density"], dots["branch_density"])
        self.assertGreater(branches["branch_arm_count"], dots["branch_arm_count"])

    def test_long_grooves_have_higher_elongation_than_dots(self) -> None:
        dots = measure_goal_pattern(synthetic_dots())
        grooves = measure_goal_pattern(synthetic_long_parallel_grooves())
        self.assertGreater(
            grooves["p90_component_elongation"],
            dots["p90_component_elongation"],
        )
        self.assertGreater(grooves["p90_geodesic_length"], dots["p90_geodesic_length"])

    def test_thick_labyrinth_has_large_width(self) -> None:
        dots = measure_goal_pattern(synthetic_dots())
        maze = measure_goal_pattern(synthetic_thick_labyrinth())
        self.assertGreater(
            maze["feature_width_median"],
            dots["feature_width_median"] * 1.5,
        )

    def test_classify_morphology_ordering(self) -> None:
        self.assertEqual(classify_morphology(measure_goal_pattern(synthetic_dots())), "I")
        tubes = measure_goal_pattern(synthetic_short_tubes())
        self.assertIn(classify_morphology(tubes), {"IIIL", "I"})
        branches = measure_goal_pattern(synthetic_y_branches())
        self.assertIn(classify_morphology(branches), {"IV_B", "IIIL"})
        mesh = measure_goal_pattern(synthetic_labyrinth_mesh())
        self.assertEqual(classify_morphology(mesh), "IV_V")

    def test_parallel_stripes_are_not_iv_v(self) -> None:
        grooves = measure_goal_pattern(synthetic_long_parallel_grooves())
        self.assertNotEqual(classify_morphology(grooves), "IV_V")

    def test_labyrinth_mesh_has_islands_and_connectivity(self) -> None:
        mesh = measure_goal_pattern(synthetic_labyrinth_mesh())
        grooves = measure_goal_pattern(synthetic_long_parallel_grooves())
        self.assertGreaterEqual(mesh["background_island_count"], 6)
        self.assertLess(grooves["background_island_count"], mesh["background_island_count"])
        self.assertGreater(mesh["labyrinth_score"], grooves["labyrinth_score"])

    def test_temporal_stability_detects_width_coarsening(self) -> None:
        stable = [
            {"longest_component_fraction": 0.2, "branch_density": 0.1, "feature_width_median": 5.0, "labyrinth_score": 10.0},
            {"longest_component_fraction": 0.21, "branch_density": 0.1, "feature_width_median": 5.1, "labyrinth_score": 10.5},
            {"longest_component_fraction": 0.19, "branch_density": 0.09, "feature_width_median": 4.9, "labyrinth_score": 9.8},
            {"longest_component_fraction": 0.2, "branch_density": 0.1, "feature_width_median": 5.0, "labyrinth_score": 10.1},
        ]
        width_blowup = [
            {"longest_component_fraction": 0.2, "branch_density": 0.1, "feature_width_median": 5.0, "labyrinth_score": 10.0},
            {"longest_component_fraction": 0.3, "branch_density": 0.08, "feature_width_median": 6.0, "labyrinth_score": 12.0},
            {"longest_component_fraction": 0.4, "branch_density": 0.04, "feature_width_median": 7.5, "labyrinth_score": 20.0},
            {"longest_component_fraction": 0.5, "branch_density": 0.02, "feature_width_median": 9.5, "labyrinth_score": 35.0},
        ]
        # Connectivity growth alone (labyrinth forming) is NOT coarsening now.
        connecting = [
            {"longest_component_fraction": 0.2, "branch_density": 0.1, "feature_width_median": 5.0, "labyrinth_score": 10.0},
            {"longest_component_fraction": 0.4, "branch_density": 0.09, "feature_width_median": 5.0, "labyrinth_score": 20.0},
            {"longest_component_fraction": 0.6, "branch_density": 0.08, "feature_width_median": 5.1, "labyrinth_score": 40.0},
            {"longest_component_fraction": 0.8, "branch_density": 0.08, "feature_width_median": 5.0, "labyrinth_score": 60.0},
        ]
        stable_metrics = measure_temporal_stability(stable)
        blowup_metrics = measure_temporal_stability(width_blowup)
        connecting_metrics = measure_temporal_stability(connecting)
        self.assertEqual(stable_metrics["is_stationary"], 1.0)
        self.assertEqual(blowup_metrics["coarsening_detected"], 1.0)
        self.assertEqual(connecting_metrics["coarsening_detected"], 0.0)

    def test_evaluate_goal_sequence_ideal_order(self) -> None:
        rows = [
            {"p": 0.0, **measure_goal_pattern(synthetic_dots()), "is_stationary": 1.0, "coarsening_detected": 0.0},
            {"p": 0.33, **measure_goal_pattern(synthetic_short_tubes()), "is_stationary": 1.0, "coarsening_detected": 0.0},
            {"p": 0.55, **measure_goal_pattern(synthetic_y_branches()), "is_stationary": 1.0, "coarsening_detected": 0.0},
            {"p": 0.85, **measure_goal_pattern(synthetic_labyrinth_mesh()), "is_stationary": 1.0, "coarsening_detected": 0.0},
        ]
        report = evaluate_goal_sequence(rows)
        self.assertTrue(report["monotonic"])
        self.assertIn("IV_B", report["seen_phases"])
        self.assertIn("IV_V", report["seen_phases"])


if __name__ == "__main__":
    unittest.main()
