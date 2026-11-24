"""Lanes hyperparameters, part 2 (DESIGN.md section 5 row test_lanes_hparams, 2.1; T3): the engine side.

Part 1 (registry, options, conflicts) is T1's ``tests/test_lanes_hparams.py``. Here:
* ``engine_hparams(d)`` (= the hardware convention ``apply_dim_rules(defaults_for_dim(d), d)``) and the sandbox
  recipe (``benchmarks/zoom_fix/lanes/lane_zombihop.py`` LZ 178-182: ``apply_dim_rules(load_hparams_file(
  sim.HPARAMS), d)`` + the ``input_noise`` default, every other knob at the ZoMBIHop constructor default) give
  the same ``hparams_in_force()`` for d = 6 and 10 (two ZoMBIHop builds with a dummy objective, 8 init rows);
* ``build_lane_engine(hparams=None)`` uses ``engine_hparams(d)``; ``resolve_options`` presets / overrides;
* lanes conflicts are refused before anything is built (Fix 4 never installed); core knobs outside the
  validated set are logged, not refused.
The sandbox recipe is re-stated here from ``src.hparams`` (no sandbox import); when the sandbox is present its
source is checked to still be that recipe. Writes nothing (ZoMBIHop without a checkpoint dir).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import src.core.linebo as _linebo
import src.core.zombihop as _zh
import src.utils.domain as _domain
from src.core.zombihop import ZoMBIHop
from src.hparams import ALL_DEFAULTS, HyperparameterConflict, apply_dim_rules, load_hparams_file
from src.lanes import PRESETS, LaneOptions, build_lane_engine
from src.lanes.engine import engine_hparams, resolve_options

REPO = Path(__file__).resolve().parents[1]
SANDBOX_HPARAMS = REPO / "optimize" / "hparams" / "dup1c_custom.json"      # sim.HPARAMS (sim.py 103)
_MODS = (_domain, _zh, _linebo)
DEV = torch.device("cuda")
DT = torch.float64


# ── REVIEW F9: restore the global RNG streams and the three random_simplex attributes ──
@pytest.fixture(autouse=True)
def _restore_global_state():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    saved = {m: m.__dict__.get("random_simplex") for m in _MODS}
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for m, f in saved.items():
            if f is not None:
                setattr(m, "random_simplex", f)


def sandbox_recipe(d: int) -> dict:
    """LZ 178-182 (default_hparams) re-stated: the hparams file through the dim rules + input_noise default."""
    hp = apply_dim_rules(load_hparams_file(str(SANDBOX_HPARAMS), log=None), d)
    hp.setdefault("input_noise", ALL_DEFAULTS["input_noise"])
    return hp


def _init(d: int, n: int = 8, seed: int = 0):
    rng = np.random.default_rng([seed, d])
    X = rng.dirichlet(np.ones(d), n)
    Y = -((X - 1.0 / d) ** 2).sum(axis=1)
    t = lambda a: torch.as_tensor(a, device=DEV, dtype=DT)  # noqa: E731
    return t(X), t(X), t(Y).reshape(-1, 1)


def _in_force(hp: dict, d: int) -> dict:
    X, Xe, Y = _init(d)

    def _obj(*a, **k):
        raise RuntimeError("never called")
    z = ZoMBIHop(objective=_obj, X_init_actual=X, X_init_expected=Xe, Y_init=Y, device=str(DEV), dtype=DT,
                 checkpoint_dir=None, verbose=False, **hp)
    return z.hparams_in_force()


def test_sandbox_recipe_is_still_the_sandbox_code():
    lz, sim = REPO / "benchmarks/zoom_fix/lanes/lane_zombihop.py", REPO / "benchmarks/zoom_fix/sim.py"
    if not (lz.exists() and sim.exists()):
        pytest.skip("sandbox not present")
    src_lz, src_sim = lz.read_text(encoding="utf-8"), sim.read_text(encoding="utf-8")
    assert 'hp = apply_dim_rules(load_hparams_file(str(HPARAMS)), d)' in src_lz
    assert 'hp.setdefault("input_noise", ALL_DEFAULTS["input_noise"])' in src_lz
    assert re.search(r'^HPARAMS = REPO / "optimize" / "hparams" / "dup1c_custom.json"$', src_sim, re.M)


@pytest.mark.parametrize("d", [6, 10])
def test_engine_hparams_equal_sandbox_recipe_in_force(d):
    eng, sbx = engine_hparams(d), sandbox_recipe(d)
    params = set(inspect.signature(ZoMBIHop.__init__).parameters)
    assert set(eng) <= params and set(sbx) <= params            # every key is a constructor argument
    assert eng["top_m_points"] == sbx["top_m_points"] >= d + 1        # apply_dim_rules: 8 at 6-D, 11 at 10-D
    assert eng["top_m_points"] == {6: 8, 10: 11}[d]
    a, b = _in_force(eng, d), _in_force(sbx, d)
    assert set(a) == set(b)
    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    assert diff == {}


def test_engine_hparams_is_a_fresh_copy():
    a = engine_hparams(6)
    a["raw"] = -1
    assert engine_hparams(6)["raw"] == ALL_DEFAULTS["raw"]


def test_build_defaults_to_engine_hparams_and_leader():
    X, Xe, Y = _init(4, n=16)
    e = build_lane_engine(X, Xe, Y, np.stack([np.zeros(4), np.ones(4)]), snapshots=False)
    try:
        assert e.hp == engine_hparams(4)
        assert e.opts == LaneOptions(snapshots=False) and e.opts.replace(snapshots=True).is_leader()
        assert e.z.hparams_in_force() == _in_force(engine_hparams(4), 4)
        assert e.build_info["warnings"] == [] and e.build_info["preset"] == "leader"
        assert e.build_info["hp_stock"] == dict(max_zooms=3, max_iter=5, min_iters=2, n_consec=2, min_zoom_needle=2)
        assert [e.build_info["hp_z"][f"z{k}"]["min_iters"] for k in range(3)] == [2, 2, 3]
        assert e.build_info["activation_budget"] == 24 and e.build_info["catchup_top_m"] == 8
    finally:
        e.close()


def test_resolve_options():
    assert PRESETS == {"leader": {}}
    o = resolve_options("leader")
    assert o == LaneOptions() and o.is_leader()
    o2 = resolve_options("leader", {"lanes_z2_min": 4, "lanes_lead_s": 900.0}, snapshots=False)
    assert o2.z2_min == 4 and o2.snapshots is False and not o2.is_leader()
    with pytest.raises(ValueError):
        resolve_options("fastest")
    with pytest.raises(ValueError):
        resolve_options("leader", {"lanes_bridge": True})          # not a lanes knob (not ported)


@pytest.mark.parametrize("bad", [{"lanes_z2_min": 1}, {"lanes_z2_min": 9}, {"lanes_catchup_zooms": "3"},
                                 {"lanes_unknown": 1}])
def test_conflicting_overrides_refused_before_build(bad):
    X, Xe, Y = _init(4)
    before = {m: m.__dict__["random_simplex"] for m in _MODS}
    with pytest.raises((HyperparameterConflict, ValueError)):
        build_lane_engine(X, Xe, Y, np.stack([np.zeros(4), np.ones(4)]), overrides=bad, snapshots=False)
    assert all(m.__dict__["random_simplex"] is before[m] for m in _MODS)


def test_core_knob_outside_validated_set_is_logged_not_refused(capsys):
    X, Xe, Y = _init(4, n=16)
    hp = engine_hparams(4)
    hp.update(raw=64, max_iterations=6)
    e = build_lane_engine(X, Xe, Y, np.stack([np.zeros(4), np.ones(4)]), hparams=hp, snapshots=False)
    try:
        w = e.build_info["warnings"]
        assert any("raw=64" in m for m in w) and any("max_iterations" in m for m in w)
        assert e.build_info["activation_budget"] == 3 * (6 + 3)
        assert "outside the validated leader set" in capsys.readouterr().err
    finally:
        e.close()
