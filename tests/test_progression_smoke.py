#!/usr/bin/env python3
"""Smoke tests for progression simulation (no full sweep)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pit_goal_metrics import classify_morphology, measure_goal_pattern  # noqa: E402
from simulate_pit_progression import (  # noqa: E402
    ProgressionModel,
    ProgressionSettings,
    coeffs_from_progression,
    simulate_at_p,
)


class TestProgressionSmoke(unittest.TestCase):
    def test_coeffs_are_monotone_in_p(self) -> None:
        qs = [coeffs_from_progression(p, ProgressionModel.SH_INHIBITOR).quadratic for p in [0.0, 0.5, 1.0]]
        self.assertGreater(qs[0], qs[1])
        self.assertGreater(qs[1], qs[2])

    def test_low_p_produces_nonempty_pattern(self) -> None:
        settings = ProgressionSettings(size=96, steps=1200, seed=7, snapshot_count=3)
        result = simulate_at_p(0.0, model=ProgressionModel.SH_INHIBITOR, settings=settings)
        metrics = measure_goal_pattern(result.u, field_kind="height")
        self.assertGreater(metrics["binary_coverage"], 0.05)
        self.assertIn(classify_morphology(metrics), ("I", "IIIL"))

    def test_high_p_differs_from_low_p(self) -> None:
        settings = ProgressionSettings(size=96, steps=1500, seed=7, snapshot_count=3)
        low = measure_goal_pattern(
            simulate_at_p(0.0, model=ProgressionModel.SH_INHIBITOR, settings=settings).u,
            field_kind="height",
        )
        high = measure_goal_pattern(
            simulate_at_p(1.0, model=ProgressionModel.SH_INHIBITOR, settings=settings).u,
            field_kind="height",
        )
        self.assertNotEqual(classify_morphology(low), classify_morphology(high))


if __name__ == "__main__":
    unittest.main()
