#!/usr/bin/env python3
"""Tests for the minimal Cellular Potts pit model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitpattern.models.simulate_pit_cpm import (  # noqa: E402
    CPMConfig,
    copy_delta_energy,
    energy_terms,
    initialize_state,
    monte_carlo_sweep,
    parse_progressions,
    targets_from_progression,
)


class CPMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CPMConfig(spacing=12, initial_radius=2)
        self.state = initialize_state(48, self.config)

    def test_initial_labels_and_areas_are_consistent(self) -> None:
        self.assertGreater(self.state.cell_count, 0)
        observed = np.bincount(
            self.state.labels.ravel(), minlength=len(self.state.areas)
        )
        np.testing.assert_array_equal(self.state.areas, observed)
        self.assertTrue(np.all(self.state.areas[1:] > 0))

    def test_progression_exchanges_epithelial_and_stromal_targets(self) -> None:
        low = targets_from_progression(self.state, self.config, 0.0)
        high = targets_from_progression(self.state, self.config, 1.0)
        self.assertGreater(high.epithelial, low.epithelial)
        self.assertLess(high.stroma, low.stroma)
        self.assertAlmostEqual(
            high.stroma + self.state.cell_count * high.epithelial,
            self.state.labels.size,
        )

    def test_local_delta_matches_recomputed_total_energy(self) -> None:
        targets = targets_from_progression(self.state, self.config, 0.5)
        candidates = np.argwhere(
            self.state.labels != np.roll(self.state.labels, 1, axis=1)
        )
        y, x = map(int, candidates[0])
        proposed = int(self.state.labels[y, (x - 1) % self.state.labels.shape[1]])
        old = int(self.state.labels[y, x])
        if proposed == old:
            self.skipTest("selected boundary did not change the label")

        before = energy_terms(self.state, targets, self.config)["energy_total"]
        delta = copy_delta_energy(
            self.state,
            y=y,
            x=x,
            proposed_label=proposed,
            targets=targets,
            config=self.config,
        )
        self.state.labels[y, x] = proposed
        self.state.areas[old] -= 1
        self.state.areas[proposed] += 1
        after = energy_terms(self.state, targets, self.config)["energy_total"]
        self.assertAlmostEqual(after - before, delta, places=7)

    def test_sweep_preserves_site_and_area_counts(self) -> None:
        targets = targets_from_progression(self.state, self.config, 0.5)
        acceptance = monte_carlo_sweep(
            self.state,
            targets=targets,
            config=self.config,
            rng=np.random.default_rng(3),
        )
        self.assertGreaterEqual(acceptance, 0.0)
        self.assertLessEqual(acceptance, 1.0)
        self.assertEqual(int(np.sum(self.state.areas)), self.state.labels.size)
        observed = np.bincount(
            self.state.labels.ravel(), minlength=len(self.state.areas)
        )
        np.testing.assert_array_equal(self.state.areas, observed)
        self.assertTrue(np.all(self.state.areas[1:] >= self.config.min_cell_area))

    def test_progression_parser_requires_ordered_unit_interval(self) -> None:
        self.assertEqual(parse_progressions("0,0.5,1"), [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            parse_progressions("0.5,0.25")
        with self.assertRaises(ValueError):
            parse_progressions("0,1.1")


if __name__ == "__main__":
    unittest.main()
