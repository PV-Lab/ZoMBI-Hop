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


def test_budget_is_spent_exactly_even_when_not_divisible_by_q():
    """N - n_init is not generally divisible by q. A fixed decision count would stop
    the baselines short (2000 -> 1992) while ZoMBI-Hop, which runs until
    BudgetExhausted, spent the full budget -- so the two would be compared at
    different sample counts."""
    p = Protocol(n_samples=200, batch_size=24, n_init_lines=2, noise="none")
    assert (p.n_samples - p.n_init_points) % p.batch_size != 0, "pick a ragged N"
    for spec in ({"name": "random"}, {"name": "zombihop", "hparams": "smoke"}):
        res = run_one(_ENS3, spec, seed=0, protocol=p)
        assert res["n_samples"] == 200, (spec["name"], res["n_samples"])


def test_every_declared_needle_is_logged_even_the_last(tmp_path):
    """The needle declared by the budget-exhausting line must reach the curves.

    ``obj_wrapper`` logs a needle only on the NEXT objective call, so a needle
    declared as the budget runs out was never recorded: ``inner()`` raises
    ``BudgetExhausted`` and the append never runs. The final metrics were still
    right (they read ``dh.needles`` directly), but the ``@N`` prefix curves were
    short -- and fig1, fig3 and fig4 are all built from those. In the published
    s1_real bundle this hit 6 of 60 cells (569 declared, 563 logged) and moved two
    headline numbers: real4d/zombihop/s6 peak_ratio@2000 0.185 vs 0.222 final.

    Asserting on the written artifacts, per DESIGN.md 20: the three places a needle
    count appears must agree, and the last checkpoint must equal the final value
    when the budget was spent exactly.
    """
    import csv
    import json
    import os

    out = tmp_path / "zh"
    res = run_one(_ENS3, {"name": "zombihop", "hparams": "smoke"}, seed=0,
                  protocol=Protocol(n_samples=240, batch_size=24, noise="hardware",
                                    eval_at=[120, 240]),
                  out_dir=str(out))
    assert res["error"] == ""
    assert res["n_samples"] == 240

    with open(os.path.join(out, "config_resolved.json"), encoding="utf-8") as fh:
        n_state = int(json.load(fh)["optimizer_state"]["n_needles"])
    n_log = len(list(csv.DictReader(open(os.path.join(out, "needles.csv"),
                                        encoding="utf-8"))))
    n_dec = len(list(csv.DictReader(open(os.path.join(out, "declared_optima.csv"),
                                        encoding="utf-8"))))
    assert n_state == n_log == n_dec, (
        f"needle count disagrees across artifacts: optimizer_state={n_state}, "
        f"needles.csv={n_log}, declared_optima.csv={n_dec}. A needle declared "
        f"after the final objective call was not drained into needle_log.")

    # The budget was spent exactly, so the last checkpoint IS the end of the run.
    if n_state:
        assert res["n_declared@240"] == res["n_declared"]
        assert res["peak_ratio@240"] == pytest.approx(res["peak_ratio"])


def test_per_cell_timeout_is_configurable_and_per_objective(tmp_path, monkeypatch):
    """The 7200 s limit must be reachable from config, and overridable per objective.

    It was hard-coded and exposed nowhere. Fine at N=2000; dangerous above it. The
    slowest observed cell (real6d / zombihop) took 6374 s = 89% of the limit, so the
    planned 6-D run at N=6000 would have started recording timeouts as failures --
    and a timeout is deliberately NOT retried, so those cells would simply be absent
    from the bundle with only an `error` string to show for it.

    Captures the jobs the suite builds rather than running them: what matters is the
    number that reaches ``subprocess.run(timeout=...)``.
    """
    from zhbench import suite as S

    seen = []

    def _fake(job):
        seen.append((job["cell"], job["timeout_s"]))
        return job["cell"], {"objective": S._obj_label(job["objective"]),
                             "optimizer": job["optimizer"]["name"],
                             "seed": job["seed"], "error": "stubbed"}

    monkeypatch.setattr(S, "_run_cell", _fake)
    monkeypatch.setattr(S, "_run_cell_inprocess", _fake)

    cfg = {"name": "t", "protocol": {"n_samples": 48, "batch_size": 24},
           "seeds": [0], "value_tol": 0.25,
           "timeout_s": 111.0,
           "objectives": [
               {"kind": "ensemble", "dim": 3, "n_optima": 5, "landscape": 0},
               {"kind": "ensemble", "dim": 4, "n_optima": 5, "landscape": 0,
                "timeout_s": 222.0},
           ],
           "optimizers": [{"name": "random"}]}
    S.run_suite(cfg, str(tmp_path), workers=1)

    got = dict(seen)
    assert len(got) == 2
    # cell names strip "=" (_obj_label -> replace("=", "")), so it is "dim3".
    assert any(t == 111.0 for c, t in got.items() if "dim3" in c), got
    assert any(t == 222.0 for c, t in got.items() if "dim4" in c), got

    # ...and timeout_s must NOT have leaked into the objective spec, or it would
    # become an Ensemble config key and change the cell name.
    assert all("timeout_s" not in c for c in got), got
