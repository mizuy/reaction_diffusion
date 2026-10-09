import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pitpattern.models.simulate_pit_ring import Parameters, Ring, minimize


def test_circle_geometry_and_bending():
    ring = Ring(Parameters())
    x = np.zeros(ring.b.shape[1])
    x[0] = np.log(1.3)
    _, _, terms = ring.energy(x, 0.2)
    np.testing.assert_allclose(terms['area'], np.pi*1.3**2, rtol=1e-12)
    np.testing.assert_allclose(terms['perimeter'], 2*np.pi*1.3, rtol=1e-12)
    np.testing.assert_allclose(terms['bending_energy'], np.pi*ring.p.bending/1.3, rtol=1e-12)


def test_analytic_gradient_including_contact():
    ring = Ring(Parameters(modes=6, points=128, wall_distance=0.4))
    x = ring.initial(4)*20
    x[0] = -1.5  # activates nonlocal contact without a contact threshold crossing
    _, analytic, terms = ring.energy(x, 0.3)
    assert terms['contact_energy'] > 0
    numerical = np.zeros_like(x)
    for j in range(len(x)):
        delta = np.zeros_like(x)
        delta[j] = 1e-6
        numerical[j] = (ring.energy(x+delta, 0.3)[0]-ring.energy(x-delta, 0.3)[0])/2e-6
    np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-7)


def test_relaxation_descends_and_round_state_recovers():
    ring = Ring(Parameters(modes=8, points=128))
    x, info = minimize(ring, 0.0, ring.initial(7)*10)
    assert info['converged']
    assert np.max(np.diff(info['trace'])) <= 1e-12
    assert np.max(np.abs(x[1:])) < 1e-5
