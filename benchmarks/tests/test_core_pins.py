"""Pins on core values this benchmark silently inherits.

The harness was built so that nothing here edits the ZoMBI-Hop core -- but importing
it means the core can still change what our numbers mean without changing a line of
our code. DESIGN.md worried about that channel in ``optimize/eval_metrics.py``. It
did not materialise there (``MATCH_RADIUS`` and ``metric_dist_to_needles`` are
byte-identical from 77054a9 through ``origin/brianna-v2``); it materialised in
``optimize/evaluate._force_zoom_floors``, which reads ZoMBIHop's ``__init__``
SIGNATURE by reflection and then RAISES our tuned hyperparameters to that floor
(``zhbench/zombihop_runner.py:172-175``).

That is not hypothetical. On ``origin/brianna``, ``min_iters_per_zoom`` moved 2 -> 3,
which raises the ``max_iterations`` floor from 2 to 3 -- and both
``zhbench/data/hparams/4d.json`` and ``optimize/hparams/6d_ensemble.json`` carry
``max_iterations: 2``, i.e. exactly at the old floor. Merging that branch would
re-tune the real4d and real6d ZoMBI-Hop arms with no diff in ``benchmarks/`` at all.

These tests are meant to FAIL when the core moves. A failure is not a bug in the
benchmark; it is the signal to decide, deliberately, whether to re-run the affected
arms and re-stamp the provenance block in
``benchmarks/results/s1_real/RESULTS.md``.
"""

from __future__ import annotations

import inspect
import json
import os

import pytest

from zhbench._repo import eval_metrics, evaluate, run_mobo

# The core state the published s1_real bundle was produced against.
PUBLISHED = {
    "min_zoom_for_needle": 1,
    "min_iters_per_zoom": 2,
    "zoom_floor": 2,          # min_zoom_for_needle + 1
    "iter_floor": 2,          # min_iters_per_zoom
    "MATCH_RADIUS": 0.05,
    "NOISE_LEVEL": 0.128,
    "OUTPUT_NOISE_FRAC": 0.045,
}

_WHY = (
    "\n\nThe published benchmarks/results/s1_real numbers were produced with "
    "{k} = {want!r}. If this change is intended, re-run the 60 zombihop / "
    "zombihop_nc5 cells and update the Provenance section of "
    "benchmarks/results/s1_real/RESULTS.md. Do not silently republish."
)


def _sig_default(name):
    return inspect.signature(run_mobo().ZoMBIHop.__init__).parameters[name].default


@pytest.mark.parametrize("key", ["min_zoom_for_needle", "min_iters_per_zoom"])
def test_core_search_discipline_defaults_are_pinned(key):
    want = PUBLISHED[key]
    got = _sig_default(key)
    assert got == want, f"ZoMBIHop.__init__ {key}: {got} != {want}" + _WHY.format(
        k=key, want=want)


def test_force_zoom_floors_are_pinned():
    """The exact values our runner raises tuned hyperparameters to."""
    zoom_floor, iter_floor = evaluate()._force_zoom_floors()
    assert (zoom_floor, iter_floor) == (PUBLISHED["zoom_floor"], PUBLISHED["iter_floor"]), (
        f"_force_zoom_floors() = {(zoom_floor, iter_floor)} != "
        f"{(PUBLISHED['zoom_floor'], PUBLISHED['iter_floor'])}"
        + _WHY.format(k="_force_zoom_floors()",
                      want=(PUBLISHED["zoom_floor"], PUBLISHED["iter_floor"])))


def test_tuned_hparams_are_not_silently_raised_by_the_floors():
    """The tuned JSONs must sit at or above the floors, or they are being rewritten.

    ``4d.json`` and ``6d_ensemble.json`` both carry ``max_iterations: 2``, which is
    exactly the current floor. Any increase to ``min_iters_per_zoom`` upstream turns
    the tuned value into a value nobody chose.
    """
    _zoom_floor, iter_floor = evaluate()._force_zoom_floors()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(here)
    paths = [os.path.join(here, "zhbench", "data", "hparams", "3d.json"),
             os.path.join(here, "zhbench", "data", "hparams", "4d.json"),
             os.path.join(root, "optimize", "hparams", "6d_ensemble.json")]
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            hp = json.load(fh)
        mi = hp.get("max_iterations")
        if mi is None:
            continue
        assert mi >= iter_floor, (
            f"{os.path.basename(p)} max_iterations={mi} < floor {iter_floor}: the "
            f"runner will silently raise it, so this arm is no longer running its "
            f"tuned configuration." + _WHY.format(k="max_iterations", want=mi))


def test_scoring_constants_have_not_drifted():
    em = eval_metrics()
    assert float(em.MATCH_RADIUS) == PUBLISHED["MATCH_RADIUS"]
    rm = run_mobo()
    assert float(rm.NOISE_LEVEL) == PUBLISHED["NOISE_LEVEL"]
    assert float(rm.OUTPUT_NOISE_FRAC) == PUBLISHED["OUTPUT_NOISE_FRAC"]


def test_nc5_arm_still_means_what_it_says():
    """``zombihop_nc5`` exists to mirror src/default_hparams.py's stale value.

    If upstream ever syncs default_hparams.py to the JSON (UPSTREAM_REQUESTS item 2),
    the arm stops being a 'what production actually ships' sensitivity and becomes an
    arbitrary hyperparameter probe -- still worth running, but it must be relabelled.
    """
    import importlib

    from zhbench.optimizers import _SPECS

    assert _SPECS["zombihop_nc5"]["n_consecutive_converged"] == 5
    dh = importlib.import_module("src.default_hparams").DEFAULT_HPARAMS
    assert dh["n_consecutive_converged"] == 5, (
        "src/default_hparams.py no longer carries n_consecutive_converged=5. "
        "The zombihop_nc5 arm's justification in DESIGN.md 13 is now stale -- "
        "relabel it before the next bundle.")


@pytest.mark.slow
def test_a_run_records_the_full_zombihop_configuration(tmp_path):
    """Every ``ZoMBIHop.__init__`` argument must land in ``config_resolved.json``.

    The drift that motivated this file lived entirely in parameters we never pass:
    ``min_zoom_for_needle`` and ``min_iters_per_zoom`` are in none of our
    hyperparameter JSONs, so their values came from the core's signature and were
    recorded nowhere. The published bundle stored ``n_needles`` and a provenance
    path, and could not answer "what was ``min_iters_per_zoom`` on that run?" at all.
    """
    from zhbench.protocol import Protocol
    from zhbench.runner import run_one

    out = tmp_path / "mz0"
    res = run_one({"kind": "ensemble", "dim": 3, "n_optima": 8, "landscape": 0,
                   "seed": 0},
                  {"name": "zombihop_mz0", "hparams": "smoke"}, seed=0,
                  protocol=Protocol(n_samples=120, batch_size=24, noise="hardware",
                                    eval_at=[120]),
                  out_dir=str(out))
    assert res["error"] == ""

    with open(os.path.join(out, "config_resolved.json"), encoding="utf-8") as fh:
        st = json.load(fh)["optimizer_state"]
    rh = st["resolved_hparams"]

    for key in ("min_zoom_for_needle", "min_iters_per_zoom", "max_iterations",
                "max_zooms", "n_consecutive_converged", "input_noise"):
        assert key in rh, f"{key} missing from resolved_hparams"

    # The arm's whole purpose: the gate is open, and it is recorded as open.
    assert rh["min_zoom_for_needle"] == 0
    # ... while the parameter that silently moved upstream is now written down.
    assert rh["min_iters_per_zoom"] == PUBLISHED["min_iters_per_zoom"]
    # The floors are always recorded, whether or not they changed anything.
    assert any(a["key"] == "_force_zoom_floors" for a in st["hparam_adjustments"])
