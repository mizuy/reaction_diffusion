from dataclasses import replace

import numpy as np

from pitpattern.models.pit_mouth_live import Live, LiveConfig

FAST = replace(LiveConfig(), size=160, margin=28, warmup=150, p_rate=1.0)     # 216 px grid


def run_to(live, p, settle=20):
    live.p_target = p
    while abs(live.p - p) > 1e-9:
        live.step()
    for _ in range(settle):
        live.step()


def test_crypt_number_is_conserved_and_the_surface_dilates_back():
    live = Live(FAST, seed=3, p=0.5)
    n, c0 = len(live.c), live.c.copy()
    run_to(live, 3.0)
    assert len(live.c) == n
    spread = np.abs(live.c - live.centre).mean() / np.abs(c0 - live.centre).mean()
    assert abs(spread / (live.scale(3.0) / live.scale(0.5)) - 1) < 0.1      # centres follow the surface scale
    assert live.n_visible() < 0.5 * ((c0 > FAST.margin) & (c0 < FAST.margin + FAST.size)).all(1).sum()


def test_arms_grow_beyond_branch_and_retract_completely():
    live = Live(FAST, seed=4, p=1.9)
    assert not live.branches
    run_to(live, 2.8, settle=40)
    grown = sum(b.length() for b in live.branches.values())
    assert grown > 200                                   # tips advanced
    assert max(b.length() for b in live.branches.values()) <= live.m.extra / 3 + 4
    run_to(live, 1.9)
    assert not live.branches                             # every grown groove is gone again


def test_the_field_follows_the_units_up_to_branch_and_forms_islands_at_villous():
    import cv2
    live = Live(FAST, seed=5, p=1.0)
    for _ in range(10):
        live.step()
    assert ((live.u > 0) == (live.T > 0)).mean() > 0.93                      # field = unit template (edges differ)
    run_to(live, 3.9, settle=60)
    red = (~live.binary()).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(red, connectivity=4)
    assert n - 1 >= 6 and live.binary().mean() > 0.3                         # ridges broke into several islands
