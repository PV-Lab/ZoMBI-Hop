"""Metric behaviour on hand-built cases where the right answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from zhbench import metrics as M


R = 0.05


def _toy():
    """Two true optima, far apart, on the 3-simplex."""
    T = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.2, 0.7]])
    V = np.array([1.0, 0.9])
    return T, V


def test_peak_ratio_none_half_all():
    T, V = _toy()
    far = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert M.solution_set_scores(far, T, V, r=R)["peak_ratio"] == 0.0

    one = T[:1] + 0.01
    one = one / one.sum()
    assert M.solution_set_scores(one, T, V, r=R)["peak_ratio"] == 0.5

    both = T.copy()
    assert M.solution_set_scores(both, T, V, r=R)["peak_ratio"] == 1.0


def test_precision_punishes_spam():
    T, V = _toy()
    clean = M.solution_set_scores(T, T, V, r=R)
    assert clean["precision"] == 1.0

    spam = np.vstack([T, np.repeat([[1 / 3, 1 / 3, 1 / 3]], 18, axis=0)])
    dirty = M.solution_set_scores(spam, T, V, r=R)
    assert dirty["peak_ratio"] == 1.0, "recall must not fall when junk is added"
    assert dirty["precision"] == pytest.approx(2 / 20)
    assert dirty["f1"] < clean["f1"]


def test_one_declared_point_cannot_claim_two_optima():
    """The reason matching is one-to-one: real reference sets contain optima
    closer together than 2r (the 3-D campaign GP has a pair 0.067 apart)."""
    a = np.array([0.50, 0.25, 0.25])
    b = a + np.array([0.03, -0.015, -0.015])       # 0.037 apart, < 2r
    T = np.vstack([a, b])
    mid = ((a + b) / 2)[None, :]
    # merge_true_optima collapses them, so there is only one optimum to find.
    T_m, _ = M.merge_true_optima(T, np.array([1.0, 0.9]))
    assert T_m.shape[0] == 1
    s = M.solution_set_scores(mid, T, np.array([1.0, 0.9]), r=R)
    assert s["n_true_optima"] == 1
    assert s["n_matched"] == 1
    assert s["peak_ratio"] == 1.0

    # And with two well-separated optima a single midpoint can satisfy at most one.
    T2 = np.array([[0.60, 0.20, 0.20], [0.50, 0.25, 0.25]])   # 0.122 apart, > 2r
    mid2 = T2.mean(axis=0)[None, :]
    s2 = M.solution_set_scores(mid2, T2, np.array([1.0, 0.9]), r=R)
    assert s2["n_matched"] <= 1


def test_reached_requires_value_not_just_proximity():
    """A point that lands next to an optimum but measures badly has not found it."""
    T, V = _toy()
    near_but_low = T[:1] + 0.001
    X = near_but_low / near_but_low.sum()
    X = np.vstack([X, X])
    y_low = np.array([0.0, 0.0])       # far below the peak value of 1.0
    y_high = np.array([1.0, 1.0])
    background = 0.0

    first_low = M.reached_flags(X, y_low, T, V, r=R, value_tol=0.25,
                                background=background)
    assert not np.isfinite(first_low[0]), "low-valued neighbour must not count"

    first_high = M.reached_flags(X, y_high, T, V, r=R, value_tol=0.25,
                                 background=background)
    assert first_high[0] == 0


def test_posthoc_set_respects_exclusion_radius():
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(3), size=500)
    y = rng.random(500)
    S = M.posthoc_solution_set(X, y, k=10, min_sep=2 * R)
    assert S.shape[0] <= 10
    if S.shape[0] > 1:
        d = M.pairwise(S, S)
        np.fill_diagonal(d, np.inf)
        assert d.min() >= 2 * R - 1e-12


def test_input_cost_line_is_cheaper_than_scatter():
    """The SnAKe cost is the whole point of measuring it: a printed line is a
    contiguous sweep, a scattered batch is a full tour."""
    t = np.linspace(0, 1, 24)[:, None]
    a, b = np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.1, 0.8])
    line = a[None, :] + t * (b - a)[None, :]
    scatter = np.random.default_rng(0).dirichlet(np.ones(3), size=24)
    assert M.input_cost(line) < M.input_cost(scatter) / 3


def test_peak_ratio_curve_is_monotone_in_k():
    T, V = _toy()
    rng = np.random.default_rng(1)
    X = np.vstack([T, rng.dirichlet(np.ones(3), size=200)])
    y = np.concatenate([[1.0, 0.9], rng.random(200) * 0.5])
    c = M.peak_ratio_curve(X, y, T, V, k_max=8, r=R)
    pr = c["peak_ratio"]
    assert all(pr[i] <= pr[i + 1] + 1e-12 for i in range(len(pr) - 1))
