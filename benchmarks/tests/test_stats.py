"""Acceptance tests for the paired-statistics layer.

The house rule (DESIGN.md 20) is to assert on artifacts rather than on what the
code intends, so the golden test below reads the committed s1_real numbers back out
of a real run directory and checks the published values reproduce. The unit tests
above it pin the two behaviours that would silently corrupt a claim: ties must never
be counted as evidence, and `resolved` must require both tests to agree.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from zhbench import stats

RUN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "s1_real_20260824_221242")
_has_run = os.path.exists(os.path.join(RUN_DIR, "curves.json"))


def _d(vals):
    return {i: v for i, v in enumerate(vals)}


def test_paired_compare_counts_wins_ties_losses():
    a = _d([1.0, 1.0, 1.0, 0.0, 0.5])
    b = _d([0.0, 0.0, 1.0, 1.0, 0.0])
    c = stats.paired_compare(a, b)
    assert (c["wins"], c["ties"], c["losses"]) == (3, 1, 1)
    assert c["n"] == 5
    assert c["mean_diff"] == pytest.approx((1 + 1 + 0 - 1 + 0.5) / 5)


def test_sign_test_drops_ties_rather_than_counting_them():
    # 4 wins, 0 losses, 6 ties. The sign test must see n=4 (p=0.125), NOT n=10.
    # Counting ties as evidence would report p=0.002 and invent a finding.
    a = _d([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    b = _d([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    c = stats.paired_compare(a, b)
    assert (c["wins"], c["ties"], c["losses"]) == (4, 6, 0)
    assert c["p_sign"] == pytest.approx(0.125, abs=1e-9)


def test_all_ties_yield_nan_not_a_pvalue():
    # real4d zombihop_nc5 vs random is exactly 0.0000. That is a real result, but
    # no test has anything to work with, and emitting p=1.0 would read as "tested
    # and found equal" rather than "untestable".
    a = _d([0.2, 0.3, 0.4])
    c = stats.paired_compare(a, dict(a))
    assert c["mean_diff"] == 0.0
    assert c["ties"] == 3 and c["wins"] == 0 and c["losses"] == 0
    assert not np.isfinite(c["p_t"])
    assert not np.isfinite(c["p_sign"])
    assert not stats.resolved(c)


def test_resolved_requires_both_tests():
    strong = {"p_t": 0.01, "p_sign": 0.02}
    t_only = {"p_t": 0.01, "p_sign": 0.20}
    sign_only = {"p_t": 0.30, "p_sign": 0.03}
    assert stats.resolved(strong)
    assert not stats.resolved(t_only)
    assert not stats.resolved(sign_only)


def test_paired_compare_uses_only_shared_seeds():
    a = _d([1.0, 1.0, 1.0])
    b = {0: 0.0, 1: 0.0}          # seed 2 missing
    c = stats.paired_compare(a, b)
    assert c["n"] == 2


def test_seed_parsing():
    assert stats._seed_of("real_gp_dim3__zombihop__s7") == 7
    assert stats._seed_of("ensemble_dim6_n20__gp_ts__s12") == 12


def test_load_rejects_a_results_bundle_without_curves(tmp_path):
    # benchmarks/results/s1_real ships without curves.json; report.py degrades
    # silently there, which is a documented trap. stats must fail loudly instead.
    (tmp_path / "aggregate.csv").write_text("objective,optimizer,seed\nreal3d,random,0\n",
                                            encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="curves.json"):
        stats.load(str(tmp_path))


# ---------------------------------------------------------------- golden tests

@pytest.mark.skipif(not _has_run, reason="s1_real run directory not present")
def test_matched_k_reproduces_published_values():
    rows, _ = stats.load(RUN_DIR)
    assert stats.matched_k(rows, "real3d") == 6
    assert stats.matched_k(rows, "real4d") == 15
    assert stats.matched_k(rows, "real6d") == 15


@pytest.mark.skipif(not _has_run, reason="s1_real run directory not present")
def test_matched_s_recall_reproduces_results_md():
    """The published real3d matched-|S| column, to 3 decimals."""
    rows, curves = stats.load(RUN_DIR)
    k = stats.matched_k(rows, "real3d")
    expected = {"random": 0.257, "gp_qucb": 0.286, "gp_qlogei": 0.314,
                "gp_ts": 0.321, "zombihop": 0.343, "zombihop_nc5": 0.307}
    for opt, want in expected.items():
        per_seed = stats.per_seed_matched(curves, "real3d", opt, k)
        assert len(per_seed) == 10, opt
        assert np.mean(list(per_seed.values())) == pytest.approx(want, abs=5e-4), opt


@pytest.mark.skipif(not _has_run, reason="s1_real run directory not present")
def test_the_one_resolved_headline_claim():
    """zombihop > random at matched |S| on real3d: 8/1/1, both tests clear 0.05.

    This is the only method-vs-method comparison in s1_real that resolves. If a
    change to the metric or the extractor moves it, that must be a deliberate,
    visible decision -- not something noticed after the claim is in a draft.
    """
    rows, curves = stats.load(RUN_DIR)
    k = stats.matched_k(rows, "real3d")
    c = stats.paired_compare(stats.per_seed_matched(curves, "real3d", "zombihop", k),
                             stats.per_seed_matched(curves, "real3d", "random", k))
    assert (c["wins"], c["ties"], c["losses"]) == (8, 1, 1)
    assert c["mean_diff"] == pytest.approx(0.0857, abs=5e-4)
    assert stats.resolved(c)

    # ... and the same comparison against the strongest baseline does NOT resolve.
    c_ts = stats.paired_compare(stats.per_seed_matched(curves, "real3d", "zombihop", k),
                                stats.per_seed_matched(curves, "real3d", "gp_ts", k))
    assert not stats.resolved(c_ts)


@pytest.mark.slow
@pytest.mark.skipif(not _has_run, reason="s1_real run directory not present")
def test_matched_curves_removes_the_declaration_size_artifact(tmp_path):
    """At matched |S| the distance gap between ZoMBI-Hop and random must collapse.

    Straight from curves.json, real3d zombihop sits near 0.30 against random's 0.08
    -- but only because `metric_dist_to_needles` charges 0.5 for each of the 8 true
    optima its 6 declarations never claimed. Applying one extractor at one |S| is
    what makes the two numbers comparable, and here it must put them within 0.05.
    """
    mc = stats.matched_curves(RUN_DIR)
    assert len(mc) == 180

    def final(opt):
        v = [c["by_n"]["2000"]["dist_to_needles"] for c in mc.values()
             if c["objective"] == "real3d" and c["optimizer"] == opt]
        assert len(v) == 10, opt
        return float(np.mean(v))

    zh, rnd = final("zombihop"), final("random")
    assert abs(zh - rnd) < 0.05
    # and every method is scored on the same number of declarations
    assert {c["matched_k"] for c in mc.values() if c["objective"] == "real3d"} == {6}


@pytest.mark.slow
@pytest.mark.skipif(not _has_run, reason="s1_real run directory not present")
def test_build_writes_stats_md(tmp_path):
    import shutil
    dst = tmp_path / "run"
    dst.mkdir()
    for f in ("aggregate.csv", "curves.json"):
        shutil.copy(os.path.join(RUN_DIR, f), dst / f)
    path = stats.build(str(dst))
    text = open(path, encoding="utf-8").read()
    assert "Matched-|S| recall" in text
    assert "real3d" in text and "real6d" in text
    # the resolved claim must be marked as such, and the gp_ts one must not be
    assert "**yes**" in text
    assert json.loads(json.dumps({"ok": True}))["ok"]
