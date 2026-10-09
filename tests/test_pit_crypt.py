from dataclasses import replace

import numpy as np
import pytest

from pitpattern.models.simulate_pit_crypt import (
    Config, CryptModel, Lattice, composition, label, params, relax, render, run_one, wrap)

FAST = replace(Config(), sweeps=200, relax_steps=400)


def test_relaxation_gives_near_hexagonal_lattice():
    rng = np.random.default_rng(0)
    cfg = FAST
    x = relax(cfg.n_lobes, cfg.box, cfg.spacing, rng, cfg.relax_steps)
    r = np.sqrt((wrap(x[:, None] - x[None], cfg.box) ** 2).sum(-1)) + np.eye(len(x)) * 1e9
    nn = np.sort(r, 1)[:, :6]
    assert abs(nn[:, 0].mean() / cfg.spacing - 1) < 0.08
    assert (nn[:, 5] / nn[:, 0]).mean() < 1.2          # six nearly equal neighbours
    lat = Lattice(x, cfg.box, cfg.spacing)
    assert len(lat.tris) == 2 * cfg.n_lobes             # commensurate box: perfect triangulation


@pytest.mark.parametrize("p,expected", [(0.2, "I"), (0.9, "IIIL"), (2.3, "IV_B"), (3.3, "IV_V")])
def test_each_band_is_reached_from_a_random_start(p, expected):
    lat, model, _ = run_one(FAST, p, seed=5)
    assert label(composition(model)) == expected


def test_conflicting_units_never_coexist():
    lat, model, _ = run_one(FAST, 3.3, seed=6)
    U = model.U
    for u in np.flatnonzero(model.on):
        assert not any(model.on[w] for w in U.conf[u])


def test_params_are_single_valued_in_p():
    assert params(0.0)[3] == 0 and params(3.0)[2] == 0  # no unjoined-lobe cost in I, no fusion barrier in IV_V
    assert params(0.9)[0] > params(0.9)[1] / 1.5        # tubes beat Ys in IIIL
    assert params(2.3)[1] > 1.5 * params(2.3)[0]        # Ys beat tubes in IV_B


def test_render_widths_match():
    from pitpattern.metrics.pit_goal_metrics import measure_feature_width
    lat, model, _ = run_one(FAST, 0.9, seed=7)
    _, binary = render(lat, model.tubes, model.ys, FAST)
    tube_w = measure_feature_width(binary)["feature_width_median"]
    _, dots = render(lat, np.zeros_like(model.tubes), np.zeros_like(model.ys), FAST)
    import cv2
    n, _, st, _ = cv2.connectedComponentsWithStats(dots.astype(np.uint8), 8)
    dot_d = np.median(2 * np.sqrt(st[1:, cv2.CC_STAT_AREA] / np.pi))
    assert 0.8 < tube_w / dot_d < 1.25
