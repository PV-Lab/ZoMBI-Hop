"""Axial trace initial design (src/utils/axial_init.py).

Checks the design against the reference doc ("Axial trace initialization for
simplex-constrained search spaces"): exact endpoints on the bare simplex, the
L-pseudocomponent form under lower bounds, the even trace count, and the doc's
coverage table. Then checks the box generalisation on the configurations the
hardware actually runs and on random boxes: every trace stays in the box, ends on
its boundary at both ends, sweeps its component over the full feasible range, and
all traces cross at one point.
"""

import numpy as np
import pytest

from src.utils.axial_init import (
    axial_design,
    feasible_box,
    n_traces_for,
    pair_traces,
    reference_point,
    sample_trace,
    segment_coverage,
    uniform_feasible,
)


def _assert_valid_design(des, lo_in, hi_in, atol=1e-9):
    lo, hi = feasible_box(lo_in, hi_in)
    for t in des.traces:
        for x in (t.start, t.end):
            assert abs(x.sum() - 1.0) < atol
            assert (x >= lo - atol).all() and (x <= hi + atol).all()
        i = t.component
        # full feasible range of the swept component
        assert t.start[i] == pytest.approx(hi[i], abs=1e-9)
        assert t.end[i] == pytest.approx(lo[i], abs=1e-9)
        # both ends on the boundary of the search space (some bound active)
        for x in (t.start, t.end):
            active = np.isclose(x, lo, atol=1e-9) | np.isclose(x, hi, atol=1e-9)
            assert active.any()
        # the trace passes through the common reference point
        d = t.end - t.start
        s = float((des.reference - t.start) @ d / (d @ d))
        assert 0.0 <= s <= 1.0
        assert np.linalg.norm(t.start + s * d - des.reference) < 1e-9


@pytest.mark.parametrize("n", range(2, 11))
def test_bare_simplex_matches_doc_endpoints(n):
    des = axial_design(n, drop=None)
    assert len(des.traces) == n
    np.testing.assert_allclose(des.reference, np.full(n, 1.0 / n))
    for t in des.traces:
        e = np.zeros(n); e[t.component] = 1.0
        np.testing.assert_allclose(t.start, e, atol=1e-12)                   # A_i = e_i
        np.testing.assert_allclose(t.end, (1.0 - e) / (n - 1), atol=1e-12)    # B_i


@pytest.mark.parametrize("n", range(2, 11))
def test_trace_count_rounds_down_to_even(n):
    des = axial_design(n)
    assert len(des.traces) == n_traces_for(n) == (n if n % 2 == 0 else n - 1)
    assert len(des.traces) % 2 == 0
    if n % 2:
        assert des.dropped == [n - 1]       # ties broken toward the last component
    else:
        assert des.dropped == []


def test_auto_drop_picks_narrowest_range():
    # 5 components, component 1 capped at 0.2 -> it loses the trace
    des = axial_design(5, upper=[1, 0.2, 1, 1, 1])
    assert des.dropped == [1]
    assert sorted(t.component for t in des.traces) == [0, 2, 3, 4]
    des = axial_design(5, upper=[1, 0.2, 1, 1, 1], drop=3)
    assert des.dropped == [3]


@pytest.mark.parametrize("n,k", [(3, 7), (5, 11), (10, 11), (4, 9), (6, 13)])
def test_doc_unique_composition_count(n, k):
    """k = m*n + 1 puts the centroid on every trace: n(k-1)+1 unique compositions."""
    des = axial_design(n, drop=None)
    X = np.vstack([sample_trace(t, k) for t in des.traces])
    assert np.allclose(X.sum(axis=1), 1.0) and (X > -1e-12).all()
    assert len(np.unique(X.round(12), axis=0)) == n * (k - 1) + 1


# doc table: (n, mean/p95 with n traces, mean/p95 with n-1 traces)
DOC_COVERAGE = [
    (3, (0.078, 0.183), (0.144, 0.435)),
    (4, (0.141, 0.274), (0.179, 0.396)),
    (5, (0.164, 0.278), (0.191, 0.368)),
    (6, (0.174, 0.274), (0.194, 0.337)),
    (7, (0.179, 0.268), (0.193, 0.314)),
    (8, (0.180, 0.260), (0.191, 0.295)),
    (9, (0.180, 0.253), (0.189, 0.281)),
    (10, (0.178, 0.247), (0.185, 0.268)),
]


@pytest.mark.parametrize("n,full,floor", DOC_COVERAGE)
def test_coverage_reproduces_doc_table(n, full, floor):
    P = uniform_feasible(n, 100_000, seed=1)
    traces = axial_design(n, drop=None).traces
    m, p = segment_coverage([(t.start, t.end) for t in traces], P)
    assert m == pytest.approx(full[0], abs=0.003) and p == pytest.approx(full[1], abs=0.006)
    m, p = segment_coverage([(t.start, t.end) for t in traces[:-1]], P)   # n - 1 floor
    assert m == pytest.approx(floor[0], abs=0.003) and p == pytest.approx(floor[1], abs=0.006)


def test_lower_bounds_give_doc_pseudocomponent_design():
    lower = np.array([0.1, 0.05, 0.2, 0.0])
    scale = 1.0 - lower.sum()
    des = axial_design(4, lower=lower)
    assert des.tau == 0.0
    np.testing.assert_allclose(des.reference, lower + scale / 4)
    for t in des.traces:
        e = np.zeros(4); e[t.component] = 1.0
        np.testing.assert_allclose(t.start, lower + scale * e, atol=1e-12)
        np.testing.assert_allclose(t.end, lower + scale * (1 - e) / 3, atol=1e-12)


def test_hardware_6d_box_with_cap():
    """The configuration the 6-dim campaigns run: dim index 3 capped at 0.3."""
    lo, hi = [0] * 6, [1, 1, 1, 0.3, 1, 1]
    des = axial_design(6, lower=lo, upper=hi)
    assert len(des.traces) == 6 and des.dropped == []
    # the centroid is feasible and every trace keeps its full range through it,
    # so this is exactly the doc's design with only the capped trace clipped
    assert des.tau == 0.0
    np.testing.assert_allclose(des.reference, np.full(6, 1 / 6))
    _assert_valid_design(des, lo, hi)
    capped = [t for t in des.traces if t.component == 3][0]
    np.testing.assert_allclose(capped.start, [0.14, 0.14, 0.14, 0.3, 0.14, 0.14], atol=1e-12)
    np.testing.assert_allclose(capped.end, [0.2, 0.2, 0.2, 0.0, 0.2, 0.2], atol=1e-12)


def test_tight_additive_cap_does_not_cut_other_traces():
    """Doc's failure case for clipping: 10D with C10 <= 0.05 cut every other trace at
    t = 0.45. Shifting the crossing point keeps every trace boundary-to-boundary."""
    hi = [1] * 9 + [0.05]
    des = axial_design(10, upper=hi)
    assert len(des.traces) == 10
    _assert_valid_design(des, [0] * 10, hi)
    for t in des.traces[:9]:
        assert t.start[t.component] == pytest.approx(1.0)   # still reaches the pure end


def test_random_boxes_property():
    rng = np.random.default_rng(0)
    n_checked = 0
    for _ in range(400):
        n = int(rng.integers(3, 11))
        lo = np.where(rng.random(n) < 0.3, rng.uniform(0, 0.15, n), 0.0)
        hi = np.where(rng.random(n) < 0.4, rng.uniform(0.05, 0.6, n), 1.0)
        hi = np.maximum(hi, lo + 0.01)
        if lo.sum() >= 0.95 or hi.sum() <= 1.05:
            continue
        des = axial_design(n, lower=lo, upper=hi)
        assert len(des.traces) % 2 == 0 and len(des.traces) >= n - 1
        _assert_valid_design(des, lo, hi)
        n_checked += 1
    assert n_checked > 200


def test_reference_point_is_inside_and_admissible():
    lo, hi = feasible_box([0.1, 0.2, 0, 0], [0.6, 1, 0.4, 0.15])
    s, tau = reference_point(lo, hi)
    assert 0.0 < tau < 1.0
    assert abs(s.sum() - 1) < 1e-12 and (s > lo).all() and (s < hi).all()


def test_infeasible_or_degenerate_boxes_raise():
    with pytest.raises(ValueError):
        axial_design(3, lower=[0.5, 0.4, 0.3])          # sum(lo) > 1
    with pytest.raises(ValueError):
        axial_design(3, upper=[0.3, 0.3, 0.3])          # sum(hi) < 1
    with pytest.raises(ValueError):
        axial_design(3, lower=[1, 0, 0])                # a single point


def test_pinned_component_gets_no_trace():
    # component 2 pinned at 0.2 by its box; the other three are free -> 2 traces
    des = axial_design(4, lower=[0, 0, 0.2, 0], upper=[1, 1, 0.2, 1])
    assert des.fixed == [2]
    assert len(des.traces) == 2
    for t in des.traces:
        assert t.start[2] == pytest.approx(0.2) and t.end[2] == pytest.approx(0.2)


def test_pairing_is_seeded_and_complete():
    des = axial_design(10)
    p1 = pair_traces(des.traces, seed=7)
    p2 = pair_traces(des.traces, seed=7)
    assert [(a.component, b.component) for a, b in p1] == \
           [(a.component, b.component) for a, b in p2]
    comps = sorted(c for a, b in p1 for c in (a.component, b.component))
    assert comps == list(range(10))
    with pytest.raises(ValueError):
        pair_traces(axial_design(3, drop=None).traces)
