"""End-to-end acceptance tests. Small budgets so the file stays runnable on CPU."""

from __future__ import annotations

import numpy as np
import pytest

from zhbench import metrics as M
from zhbench import objectives as O
from zhbench.protocol import Protocol
from zhbench.runner import run_one

_ENS3 = {"kind": "ensemble", "dim": 3, "n_optima": 5, "landscape": 0, "seed": 0}


def _p(n=96):
    return Protocol(n_samples=n, batch_size=24, noise="hardware")


def test_every_optimizer_spends_exactly_the_budget(tmp_path):
    for spec in ({"name": "random"},
                 {"name": "gp_qucb", "pool_size": 128},
                 {"name": "zombihop", "hparams": "smoke"}):
        res = run_one(_ENS3, spec, seed=0, protocol=_p(),
                      out_dir=str(tmp_path / spec["name"]))
        assert res["n_samples"] == 96, (spec["name"], res["n_samples"])
        assert res["error"] == ""


def test_all_methods_share_the_initial_design(tmp_path):
    firsts = {}
    for spec in ({"name": "random"}, {"name": "gp_qucb", "pool_size": 128}):
        run_one(_ENS3, spec, seed=3, protocol=_p(), out_dir=str(tmp_path / spec["name"]))
        import csv
        rows = list(csv.DictReader(open(tmp_path / spec["name"] / "points.csv")))
        firsts[spec["name"]] = np.array(
            [[float(r[f"x_act_{i}"]) for i in range(3)] for r in rows[:48]])
    a, b = firsts.values()
    assert np.array_equal(a, b)


def test_zombihop_declares_and_stops_in_budget(tmp_path):
    res = run_one(_ENS3, {"name": "zombihop", "hparams": "smoke"}, seed=0,
                  protocol=Protocol(n_samples=240, batch_size=24, noise="none"),
                  out_dir=str(tmp_path / "zh"))
    assert res["n_samples"] == 240
    assert res["declared_source"] == "method"
    assert res["n_declared"] >= 1


def test_zombihop_pays_a_far_lower_input_cost_than_a_scattered_batch(tmp_path):
    """ZoMBI-Hop prints contiguous lines; a batch baseline scatters. This is the
    physical advantage the benchmark deliberately does NOT charge the baselines
    for, so it has to be measured instead."""
    p = Protocol(n_samples=240, batch_size=24, noise="none")
    zh = run_one(_ENS3, {"name": "zombihop", "hparams": "smoke"}, seed=0, protocol=p,
                 out_dir=str(tmp_path / "zh"))
    rd = run_one(_ENS3, {"name": "random"}, seed=0, protocol=p,
                 out_dir=str(tmp_path / "rd"))
    assert zh["input_cost"] < rd["input_cost"] / 3


@pytest.mark.slow
def test_random_does_not_saturate_the_metric():
    """Sanity that peak_ratio is not trivially 1.0: with 20 needles and 1000
    samples, uniform random must still fail to identify some of them."""
    spec = {"kind": "ensemble", "dim": 3, "n_optima": 20, "landscape": 0, "seed": 0}
    res = run_one(spec, {"name": "random"}, seed=0,
                  protocol=Protocol(n_samples=1000, batch_size=24,
                                    noise="hardware"))
    assert res["peak_ratio"] < 1.0


def test_objective_reports_its_own_contrast():
    """Objectives must be able to say how discriminative they are, so a high
    peak_ratio on a shallow landscape is not mistaken for a strong result."""
    obj = O.make_ensemble(dim=3, n_optima=5, landscape=0, seed=0)
    c = M.landscape_contrast(obj.fn, obj.true_optima, obj.true_values,
                             dim=obj.dim, n_probe=500)
    assert 0.0 <= c["peak_rarity_median"] <= 1.0
    assert 0.0 <= c["frac_peaks_in_top_1pct"] <= 1.0
    assert c["random_p99"] >= c["random_median"]


def test_landscape_contrast_is_stable_under_probe_count():
    """The saturation bug: ensemble tops out at 1.0, so once >1% of the domain sits
    at the ceiling the 99th percentile IS the ceiling, and a strict peak > p99 test
    flips with the probe count. A rarity is a rank, so it does not."""
    obj = O.make_ensemble(dim=3, n_optima=20, landscape=0, seed=0)
    vals = [M.landscape_contrast(obj.fn, obj.true_optima, obj.true_values,
                                 dim=3, n_probe=n)["peak_rarity_median"]
            for n in (800, 3000)]
    assert abs(vals[0] - vals[1]) < 0.05, vals
