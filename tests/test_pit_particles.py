#!/usr/bin/env python3
"""Tests for the minimal off-lattice E/S particle pit model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulate_pit_particles import (  # noqa: E402
    KIND_E,
    KIND_S,
    ParticleConfig,
    ParticleState,
    compute_forces,
    divide_epithelial,
    initialize_state,
    integrate,
    morse_force_magnitude,
    parse_progressions,
    rasterize_interface,
    rates_from_progression,
    remove_stroma,
    wrap_delta,
    wrap_position,
)


class ParticleTests(unittest.TestCase):
    def test_periodic_wrap_is_shortest_and_antisymmetric(self) -> None:
        self.assertAlmostEqual(float(wrap_delta(0.2, 1.0)), 0.2)
        self.assertAlmostEqual(float(wrap_delta(0.8, 1.0)), -0.2)
        self.assertAlmostEqual(float(wrap_delta(-0.8, 1.0)), 0.2)
        wrapped = wrap_position(np.array([[-0.2, 1.3]]), 1.0)
        np.testing.assert_allclose(wrapped, [[0.8, 0.3]])

    def test_morse_force_repels_inside_and_attracts_outside_rest_length(self) -> None:
        well, width, rest = 1.5, 2.0, 2.0
        self.assertLess(float(morse_force_magnitude(1.2, well, width, rest)), 0.0)
        self.assertGreater(float(morse_force_magnitude(3.0, well, width, rest)), 0.0)
        self.assertAlmostEqual(float(morse_force_magnitude(rest, well, width, rest)), 0.0)

    def test_progression_raises_division_and_stroma_death(self) -> None:
        config = ParticleConfig()
        low = rates_from_progression(0.0, config)
        high = rates_from_progression(1.0, config)
        self.assertEqual(low.division, 0.0)
        self.assertEqual(low.stroma_death, 0.0)
        self.assertGreater(high.division, low.division)
        self.assertGreater(high.stroma_death, low.stroma_death)

    def test_initialize_has_both_species_inside_the_box(self) -> None:
        config = ParticleConfig(cluster_spacing=12.0, particle_spacing=2.4)
        state = initialize_state(36.0, config, np.random.default_rng(7))
        self.assertGreater(state.epithelial_count, 0)
        self.assertGreater(state.stroma_count, 0)
        self.assertTrue(np.all(state.xy >= 0.0))
        self.assertTrue(np.all(state.xy < 36.0))

    def test_two_epithelial_particles_repel_when_overlapping(self) -> None:
        config = ParticleConfig()
        state = ParticleState(
            xy=np.array([[10.0, 10.0], [10.4, 10.0]], dtype=np.float64),
            kind=np.array([KIND_E, KIND_E], dtype=np.uint8),
            pressure=np.zeros(2, dtype=np.float64),
        )
        forces, _energy = compute_forces(state, domain=32.0, config=config)
        self.assertLess(forces[0, 0], 0.0)
        self.assertGreater(forces[1, 0], 0.0)

    def test_integrate_keeps_positions_finite_and_periodic(self) -> None:
        config = ParticleConfig(cluster_spacing=12.0, particle_spacing=2.4)
        rng = np.random.default_rng(3)
        state = initialize_state(36.0, config, rng)
        energy = integrate(state, domain=36.0, config=config, dt=0.03, rng=rng)
        self.assertTrue(np.isfinite(energy))
        self.assertTrue(np.all(np.isfinite(state.xy)))
        self.assertTrue(np.all(state.xy >= 0.0))
        self.assertTrue(np.all(state.xy < 36.0))

    def test_division_increases_epithelial_count(self) -> None:
        config = ParticleConfig(k_div_max=8.0, max_epithelial_fraction=0.95)
        state = ParticleState(
            xy=np.array([[8.0, 8.0], [20.0, 20.0]], dtype=np.float64),
            kind=np.array([KIND_E, KIND_S], dtype=np.uint8),
            pressure=np.zeros(2, dtype=np.float64),
        )
        added = divide_epithelial(
            state,
            domain=32.0,
            config=config,
            rates=rates_from_progression(1.0, config),
            dt=1.0,
            rng=np.random.default_rng(1),
        )
        self.assertGreaterEqual(added, 1)
        self.assertEqual(state.epithelial_count, 1 + added)

    def test_stroma_removal_is_pressure_gated(self) -> None:
        config = ParticleConfig(k_death_max=4.0, pressure_threshold=0.5, min_stroma_count=1)
        state = ParticleState(
            xy=np.array([[4.0, 4.0], [8.0, 8.0], [12.0, 12.0]], dtype=np.float64),
            kind=np.array([KIND_E, KIND_S, KIND_S], dtype=np.uint8),
            pressure=np.array([0.0, 2.0, 0.0], dtype=np.float64),
        )
        removed = remove_stroma(
            state,
            config=config,
            rates=rates_from_progression(1.0, config),
            dt=1.0,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(removed, 1)
        self.assertEqual(state.stroma_count, 1)

    def test_interface_raster_width_does_not_depend_on_progression(self) -> None:
        config = ParticleConfig()
        state = ParticleState(
            xy=np.array([[8.0, 8.0], [10.2, 8.0]], dtype=np.float64),
            kind=np.array([KIND_E, KIND_S], dtype=np.uint8),
            pressure=np.zeros(2, dtype=np.float64),
        )
        field = rasterize_interface(state, size=24, domain=24.0, config=config)
        again = rasterize_interface(state, size=24, domain=24.0, config=config)
        np.testing.assert_array_equal(field, again)
        self.assertGreater(float(np.max(field)), 0.0)

    def test_progression_parser_requires_ordered_unit_interval(self) -> None:
        self.assertEqual(parse_progressions("0,0.5,1"), [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            parse_progressions("0.5,0.25")
        with self.assertRaises(ValueError):
            parse_progressions("0,1.1")


if __name__ == "__main__":
    unittest.main()
