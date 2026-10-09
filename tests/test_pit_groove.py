import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_pit_groove import label, wilson
from pit_fork_metrics import fork_metrics
from simulate_pit_groove import Config, GrooveModel, pit_binary


def test_fork_counter_y_disk_and_diagonal_line():
    y = np.zeros((96, 96), np.uint8)
    for end in [(48, 12), (16, 76), (80, 76)]:
        cv2.line(y, (48, 48), end, 1, 5)
    assert fork_metrics(y, arm_min=6)["fork_count"] == 1
    disk = np.zeros_like(y)
    cv2.circle(disk, (48, 48), 10, 1, -1)
    assert fork_metrics(disk, arm_min=6)["fork_count"] == 0
    # a thick diagonal bar produces staircase skeletons; it must not count as a fork
    bar = np.zeros_like(y)
    cv2.line(bar, (10, 20), (85, 70), 1, 6)
    assert fork_metrics(bar, arm_min=6)["fork_count"] == 0


def test_initial_streams_are_independent():
    model = GrooveModel(Config(size=32))
    v1, w1 = model.initial(1, 2)
    v2, w2 = model.initial(1, 3)
    assert np.array_equal(v1, v2)
    assert not np.array_equal(w1, w2)


def test_short_run_is_finite_and_frozen_threshold():
    cfg = replace(Config(), size=64, steps=400)
    model = GrooveModel(cfg)
    v, w = model.initial(5, 6)
    v, w, snaps = model.evolve(v, w, 0.0, 400, snapshot_steps=(200,))
    assert np.isfinite(v).all() and np.isfinite(w).all()
    assert 200 in snaps
    assert pit_binary(w).dtype == bool


def test_labels_and_wilson_interval():
    base = dict(coverage=0.2, fork_count=0, fork_components=0, p90_component_elongation=1.3,
                unit_length_p90=3, longest_component_fraction=0.05, background_island_count=0)
    assert label(base) == "I"
    assert label({**base, "coverage": 0.001}) == "empty"
    assert label({**base, "p90_component_elongation": 3.0, "unit_length_p90": 15}) == "IIIL"
    assert label({**base, "fork_count": 8, "fork_components": 4, "longest_component_fraction": 0.2,
                  "p90_component_elongation": 3.0, "unit_length_p90": 80}) == "IV_B"
    assert label({**base, "fork_count": 80, "fork_components": 1, "longest_component_fraction": 0.9,
                  "background_island_count": 40, "p90_component_elongation": 3.0,
                  "unit_length_p90": 900}) == "IV_V"
    lo, hi = wilson(18, 20)
    assert 0.6 < lo < 0.9 < hi <= 1.0
