import math

import numpy as np

from benchmarks.zombihop_benchmark.metrics import best_y_so_far, compute_metrics, dup_fraction
from benchmarks.zombihop_benchmark.types import ObjectiveInfo


def test_best_y_so_far():
    assert np.allclose(best_y_so_far(np.array([0.1, 0.3, 0.2])), [0.1, 0.3, 0.3])


def test_duplicate_fraction_detects_repeated_points():
    X = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    assert dup_fraction(X, duplicate_radius_ilr=1e-6) == 1 / 3


def test_compute_metrics_with_needles():
    needle = np.array([[0.2, 0.3, 0.5]])
    X = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    info = ObjectiveInfo(name="obj", n_components=3, true_needles=needle, match_radius_ilr=0.01)
    row = compute_metrics(X, np.array([0.1, 0.4]), info, 0.001, 0.01, 1.2, step=2)
    assert row["step"] == 2
    assert row["best_y_so_far"] == 0.4
    assert row["pct_matched"] == 100.0
    assert row["pct_matched_ilr"] == 100.0
    assert row["pct_matched_comp"] == 100.0
    assert row["dist_to_needles"] == row["dist_to_needles_ilr"]
    assert row["dup_fraction"] == row["dup_fraction_ilr"]
    assert "dist_to_needles_comp" in row
    assert "dup_fraction_comp" in row
    assert row["dup_fraction_comp_all_points"] == row["dup_fraction_comp"]
    assert row["num_points"] == 2


def test_compute_metrics_without_needles_returns_nan():
    X = np.array([[0.2, 0.3, 0.5]])
    info = ObjectiveInfo(name="obj", n_components=3)
    row = compute_metrics(X, np.array([0.1]), info, 0.001, None, 0.0, step=0)
    assert math.isnan(row["dist_to_needles"])
    assert math.isnan(row["pct_matched"])
    assert math.isnan(row["dist_to_needles_ilr"])
    assert math.isnan(row["pct_matched_ilr"])
    assert math.isnan(row["dist_to_needles_comp"])
    assert math.isnan(row["pct_matched_comp"])


def test_line_aware_duplicate_fraction_ignores_same_line_neighbors():
    needle = np.array([[0.2, 0.3, 0.5]])
    X = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5],
            [0.201, 0.299, 0.5],
            [0.6, 0.2, 0.2],
        ]
    )
    info = ObjectiveInfo(name="obj", n_components=3, true_needles=needle, match_radius_ilr=0.01)
    row = compute_metrics(
        X,
        np.array([0.1, 0.2, 0.3, 0.4]),
        info,
        0.001,
        0.01,
        0.0,
        step=1,
        duplicate_radius_comp=0.01,
        line_group_ids=["init_0", "line_1", "line_1", "line_2"],
    )

    assert row["dup_fraction_comp_all_points"] > row["dup_fraction_comp_cross_line"]

