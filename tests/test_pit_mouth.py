from dataclasses import replace

import numpy as np
import pytest

from pitpattern.evaluation.evaluate_pit_mouth import label, measure, periodic_label
from pitpattern.models.simulate_pit_mouth import Config, params, render, run

FAST = replace(Config(), width=275, height=275, relax_steps=800, steps=400)


def stage(p, seed=5, cfg=FAST):
    states, _ = run(p, seed, cfg)
    _, binary = render(states[-1][1], cfg)
    m = measure(binary, cfg)
    return label(m), m


@pytest.mark.parametrize("p,expected", [(0.2, "Round"), (1.0, "Elongated"), (2.0, "Branch"), (3.1, "Gyrus"), (3.8, "Villous")])
def test_each_stage_is_reached_from_a_random_start(p, expected):
    assert stage(p)[0] == expected


def test_groove_width_is_the_same_in_every_stage():
    widths = [stage(p)[1]["width_median"] for p in (0.2, 1.0, 2.0, 3.1, 3.8)]
    assert max(widths) / min(widths) < 1.15


def test_coefficients_are_single_valued_and_ordered_in_p():
    p = np.linspace(0, 4, 81)
    pr = params(p)
    assert np.all(np.diff(pr["kappa"]) <= 1e-12) and pr["kappa"][-1] == 0      # unit pinning only fades
    assert np.all(np.diff(pr["g"]) <= 1e-12) and pr["g"][0] > 0 > pr["g"][-1]   # ridges connected -> islands
    assert np.all(np.diff(pr["lam"]) >= -1e-12)                                 # pitch only widens
    assert np.all(np.diff(pr["area"]) >= -1e-9)                                 # crypts per area only fall
    assert params(2.0)["grow"] == 0 and params(4.0)["kappa_g"] == 0


def test_periodic_label_joins_across_the_border():
    m = np.zeros((20, 20), bool)
    m[5, :3] = m[5, -3:] = True      # one bar wrapped around x
    m[:2, 10] = m[-2:, 10] = True    # one bar wrapped around y
    lab, area = periodic_label(m)
    assert len(area) == 3 and sorted(area[1:]) == [4, 6]
