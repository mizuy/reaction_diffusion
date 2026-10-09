import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulate_pit_activator_inhibitor import (  # noqa: E402
    ActivatorInhibitorConfig,
    homogeneous_equilibrium,
    linear_growth_rates,
    simulate,
)


def test_homogeneous_equilibrium_satisfies_reaction() -> None:
    cfg = ActivatorInhibitorConfig()
    for bias in (-0.2, 0.0, 0.2):
        u = homogeneous_equilibrium(bias, cfg)
        assert abs(cfg.growth * u - u**3 - cfg.coupling * u + bias) < 1e-10


def test_default_has_turing_unstable_center_and_stable_edge() -> None:
    cfg = ActivatorInhibitorConfig()
    assert linear_growth_rates(0.0, cfg)["turing_continuum"]
    assert not linear_growth_rates(0.7, cfg)["turing_continuum"]


def test_simulation_is_finite_and_seed_reproducible() -> None:
    cfg = ActivatorInhibitorConfig(size=32, steps=40, seed=3)
    first = simulate(0.0, cfg)
    second = simulate(0.0, cfg)
    assert np.isfinite(first.u).all()
    np.testing.assert_array_equal(first.u, second.u)


def test_independent_seed_changes_initial_value_problem() -> None:
    first = simulate(-0.15, ActivatorInhibitorConfig(size=32, steps=2, seed=3))
    second = simulate(-0.15, ActivatorInhibitorConfig(size=32, steps=2, seed=4))
    assert not np.array_equal(first.u, second.u)
