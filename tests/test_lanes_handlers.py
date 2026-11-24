"""Lane subclasses of ZoMBIHop / DataHandler / GPSimplex (src/lanes/handlers.py) and isolation part 1.

DESIGN.md 1.3.4 / section 5 (T2): the instance patches of the sandbox (lane mask, own-best, exact fit
memo, timers) are subclass code; ``attach`` swaps only the two instances; importing the modules has
no side effect (REVIEW F12) and the base classes stay untouched. Engine-level isolation (build and
close an engine, Fix 4 scope) is T3's ``tests/test_lanes_isolation.py``.

Small problem (d = 4, 12 init rows), a stub engine, CUDA. Writes only under ``tmp_path``. The optional
sandbox cross-check compares against ``benchmarks/zoom_fix/proposed_fixes`` (skipped without it).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import src.core.linebo as _linebo
import src.core.zombihop as _zh
import src.utils.domain as _domain
from src.core.linebo import LineBO
from src.core.zombihop import ZoMBIHop
from src.utils.datahandler import DataHandler
from src.utils.gp_simplex import GPSimplex
from src.lanes.handlers import (FIT_STAT_KEYS, LaneDataHandler, LaneGPSimplex, LaneZoMBIHop,
                                attach)

REPO = Path(__file__).resolve().parents[1]
D = 4
DT = torch.float64
DEV = torch.device("cuda")
_MODS = (_domain, _zh, _linebo)
_BASES = (DataHandler, GPSimplex, ZoMBIHop, LineBO)


# ── F9: restore the global RNG streams and the three random_simplex attributes ──
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


class _StubTimers:
    """Same contract as src.lanes.instrument.Timers (LZ 429-449 logic); used only while T1's module
    does not exist."""

    def __init__(self):
        from collections import defaultdict
        self.comp = defaultdict(float)
        self._stack = []

    def timed(self, key):
        import contextlib
        import time

        @contextlib.contextmanager
        def cm():
            self._stack.append([0.0])
            t = time.perf_counter()
            try:
                yield
            finally:
                dt = time.perf_counter() - t
                child = self._stack.pop()[0]
                if self._stack:
                    self._stack[-1][0] += dt
                self.comp[key] += max(dt - child, 0.0)
                self.comp[key + "_n"] += 1
        return cm()


def _timers():
    try:
        from src.lanes.instrument import Timers
    except ImportError:                       # T1 module not there yet
        Timers = _StubTimers
    return Timers()


class _Engine:
    """What attach() and the subclasses read from the engine: the lane mask (LZ 409-414 is the
    engine's), ``timers``, ``fit_stats``."""

    def __init__(self, timers: bool = True):
        if timers:
            self.timers = _timers()
        self.mask = None
        self.n_mask = 0

    def _current_activation_mask(self):
        self.n_mask += 1
        return self.mask


def _init_rows(n: int = 12, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = -torch.log(torch.rand(n, D, generator=g, dtype=DT, device="cpu").clamp(min=1e-300))
    X = (E / E.sum(dim=1, keepdim=True)).to(DEV)
    Y = torch.rand(n, 1, generator=g, dtype=DT, device="cpu").to(DEV)
    return X, Y


def _bounds():
    return torch.stack([torch.zeros(D, dtype=DT, device=DEV), torch.ones(D, dtype=DT, device=DEV)])


def _kw(checkpoint_dir=None, seed=0):
    X, Y = _init_rows(seed=seed)
    return dict(X_init_actual=X, X_init_expected=X.clone(), Y_init=Y, bounds=_bounds(), device="cuda",
                dtype=DT, checkpoint_dir=checkpoint_dir, run_uuid=None, resume=False, verbose=False)


def _build(engine=None, **kw):
    engine = engine if engine is not None else _Engine()
    return LaneZoMBIHop(engine=engine, **_kw(**kw)), engine


def _class_state():
    return {c.__name__: {k: id(v) for k, v in vars(c).items()} for c in _BASES}


def _simplex_attrs():
    return {m.__name__: m.__dict__.get("random_simplex") for m in _MODS}


def _rng_states():
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


# ── isolation part 1 ──

def test_import_has_no_side_effects(tmp_path):
    """A fresh interpreter: importing src.lanes.handlers / src.lanes.linebo changes no class, no
    random_simplex attribute, no RNG state and imports nothing from benchmarks / scripts."""
    code = r"""
import json, sys
import torch
import src.core.zombihop as zh, src.core.linebo as lb, src.utils.domain as dom
from src.utils.datahandler import DataHandler
from src.utils.gp_simplex import GPSimplex
from src.core.zombihop import ZoMBIHop
from src.core.linebo import LineBO
def snap():
    out = {m.__name__: id(m.__dict__.get("random_simplex")) for m in (dom, zh, lb)}
    for c in (DataHandler, GPSimplex, ZoMBIHop, LineBO):
        out[c.__name__] = sorted((k, id(v)) for k, v in vars(c).items())
    out["linebo_mod"] = sorted((k, id(v)) for k, v in vars(lb).items())
    return out
a = snap()
cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
mods0 = set(sys.modules)
import src.lanes.handlers, src.lanes.linebo
b = snap()
new = sorted(set(sys.modules) - mods0)
print(json.dumps(dict(
    same=a == b,
    same_rng=bool(torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())),
    foreign=[m for m in new if m.split(".")[0] in ("benchmarks", "scripts", "interface", "optimize")],
    new=new)))
"""
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["same"], "importing src.lanes.handlers / linebo changed a class or random_simplex"
    assert out["same_rng"]
    assert out["foreign"] == [], out["foreign"]
    assert "src.lanes.handlers" in out["new"] and "src.lanes.linebo" in out["new"]


def test_attach_swaps_only_the_two_instances():
    before, simplex0 = _class_state(), _simplex_attrs()
    z, eng = _build()
    assert type(z) is LaneZoMBIHop
    assert type(z.data_handler) is LaneDataHandler and type(z.gp_handler) is LaneGPSimplex
    assert z.data_handler._lanes_engine is eng and z.gp_handler._lanes_engine is eng
    assert z._lanes_engine is eng
    # no instance patches: every lane behaviour is class code
    for name in ("current_activation_mask", "get_best_unpenalized", "get_best_in_bounds",
                 "add_all_points", "take_snapshot", "_relabel_pared_with_medians"):
        assert name not in vars(z.data_handler)
    for name in ("fit", "get_candidate", "determine_penalty_ellipsoid"):
        assert name not in vars(z.gp_handler)
    for name in ("_check_convergence_to_needle", "_declare_needle_from_point", "_penalize_capped_zone"):
        assert name not in vars(z)
    assert _class_state() == before, "a base class was modified"
    assert _simplex_attrs() == simplex0, "handlers must not touch random_simplex (Fix 4 is rng.py)"
    # a plain ZoMBIHop built afterwards is today's object
    X, Y = _init_rows(seed=1)
    plain = ZoMBIHop(objective=lambda *a, **k: None, X_init_actual=X, X_init_expected=X.clone(),
                     Y_init=Y, bounds=_bounds(), device="cuda", dtype=DT, checkpoint_dir=None,
                     verbose=False)
    assert type(plain.data_handler) is DataHandler and type(plain.gp_handler) is GPSimplex
    assert bool(plain.data_handler.current_activation_mask().all())      # stock: no start index
    assert not hasattr(plain.data_handler, "_lanes_engine") or plain.data_handler._lanes_engine is None


def test_objective_raises_and_is_fixed():
    z, _ = _build()
    with pytest.raises(RuntimeError, match="never calls the blocking objective"):
        z.objective(torch.zeros(1, 2, D))
    with pytest.raises(TypeError):
        LaneZoMBIHop(engine=_Engine(), objective=lambda *a: None, **_kw())


def test_attach_refuses_twice_and_an_engine_without_mask():
    z, eng = _build()
    with pytest.raises(TypeError, match="already attached"):
        attach(z, eng)
    X, Y = _init_rows(seed=2)
    plain = ZoMBIHop(objective=lambda *a, **k: None, X_init_actual=X, X_init_expected=X.clone(),
                     Y_init=Y, bounds=_bounds(), device="cuda", dtype=DT, checkpoint_dir=None,
                     verbose=False)
    with pytest.raises(TypeError, match="_current_activation_mask"):
        attach(plain, object())
    assert type(plain.data_handler) is DataHandler and type(plain.gp_handler) is GPSimplex


def test_engine_gets_timers_and_fit_stats_when_absent():
    pytest.importorskip("src.lanes.instrument")
    eng = _Engine(timers=False)
    assert not hasattr(eng, "timers") and not hasattr(eng, "fit_stats")
    _build(eng)
    from src.lanes.instrument import Timers
    assert isinstance(eng.timers, Timers)
    assert eng.fit_stats == dict(FIT_STAT_KEYS)


# ── lane mask and own-best (LZ 322 / 409-414, PF 1004-1038) ──

def test_current_activation_mask_routes_to_the_engine():
    z, eng = _build()
    dh = z.data_handler
    n = int(dh.X_all_actual.shape[0])
    eng.mask = torch.zeros(n, dtype=torch.bool, device=DEV)
    eng.mask[[1, 5]] = True
    assert dh.current_activation_mask() is eng.mask and eng.n_mask == 1


def _own_best_ref(X, Y, own_unpen, bounds=None):
    idx = torch.where(own_unpen)[0]
    if idx.numel() == 0:
        return None
    if bounds is not None:
        inb = ((X[idx] >= bounds[0]) & (X[idx] <= bounds[1])).all(dim=1)
        if bool(inb.any()):
            idx = idx[inb]
    return int(idx[int(Y[idx].reshape(-1).argmax())])


def test_own_best_reads_only_own_unpenalised_rows():
    z, eng = _build()
    dh = z.data_handler
    X, Y = dh.X_all_actual, dh.Y_all
    n = int(X.shape[0])
    own = torch.zeros(n, dtype=torch.bool, device=DEV)
    own[[1, 3, 4, 6, 9]] = True
    pen = torch.ones(n, dtype=torch.bool, device=DEV)
    pen[3] = False
    dh._penalty_mask = pen
    eng.mask = own
    # make the global best a foreign row and the own best a penalised row
    Y[0, 0] = 10.0
    Y[3, 0] = 9.0
    ou = own & pen
    xb, yb, j = dh.get_best_unpenalized()
    assert j == _own_best_ref(X, Y, ou) and j not in (0, 3)
    assert torch.equal(xb, X[j]) and torch.equal(yb, Y[j])
    # in-box first: a box around rows 4 and 9 only
    sel = X[[4, 9]]
    box = torch.stack([sel.min(dim=0).values, sel.max(dim=0).values])
    want = _own_best_ref(X, Y, ou, box)
    _, _, j = dh.get_best_in_bounds(box)
    assert j == want
    inb = ((X[ou] >= box[0]) & (X[ou] <= box[1])).all(dim=1)
    assert bool(inb.any())
    # a box containing no own row: own anywhere
    far = torch.stack([torch.full((D,), 0.999, dtype=DT, device=DEV), torch.ones(D, dtype=DT, device=DEV)])
    _, _, j = dh.get_best_in_bounds(far)
    assert j == _own_best_ref(X, Y, ou)
    # no own row: no reference at all
    eng.mask = torch.zeros(n, dtype=torch.bool, device=DEV)
    assert dh.get_best_unpenalized() == (None, None, None)
    assert dh.get_best_in_bounds(box) == (None, None, None)


# ── exact fit memo (PF 716-866, mode "all") and the timer around it ──

def test_fit_memo_skips_exact_repeats_only(monkeypatch):
    z, eng = _build()
    dh, gp = z.data_handler, z.gp_handler
    st = eng.fit_stats
    assert gp._lanes_memo == {"X": None} and st == dict(FIT_STAT_KEYS)
    X, Y = dh.get_gp_data()
    gp.fit(X, Y)
    assert (st["fit_calls"], st["fit_reused"]) == (1, 0) and st["fit_s"] > 0.0
    g0, m0, c0 = gp.gp, gp.mll, gp.comp_std
    noise = []
    monkeypatch.setattr(dh, "update_gp_noise", lambda s: noise.append(s))
    rng0 = _rng_states()
    gp.fit(X.clone(), Y.clone())                       # identical data: skipped
    assert (st["fit_calls"], st["fit_reused"]) == (2, 1)
    assert gp.gp is g0 and gp.mll is m0 and gp.comp_std is c0
    assert noise == [gp.get_output_noise()]            # update_gp_noise re-applied on a skip
    assert all(torch.equal(a, b) for a, b in zip(rng0, _rng_states()))
    gp.fit_from_data_handler()                         # GPSimplex's internal self.fit hits the memo
    assert (st["fit_calls"], st["fit_reused"]) == (3, 2)
    gp.fit(X, Y + 1e-3)                                # other data: real fit
    assert (st["fit_calls"], st["fit_reused"]) == (4, 2) and gp.gp is not g0
    g1 = gp.gp
    gp.gp = None                                       # another model object (recompute_all_ellipsoids)
    gp.fit(X, Y + 1e-3)
    assert (st["fit_calls"], st["fit_reused"]) == (5, 2) and gp.gp is not None and gp.gp is not g1
    noise0 = dh.input_noise
    monkeypatch.setattr(dh, "input_noise", float(noise0) * 1.5)   # other lengthscale floor input
    gp.fit(X, Y + 1e-3)
    assert (st["fit_calls"], st["fit_reused"]) == (6, 2)
    gp.fit(X, Y + 1e-3)
    assert (st["fit_calls"], st["fit_reused"]) == (7, 3)
    # the timer sits outside the memo: it counts every call, skips included
    assert eng.timers.comp["fit_n"] == st["fit_calls"]
    assert st["fit_skip_after_rng"] <= st["fit_rng_consumed"]


def test_failed_fit_leaves_the_memo_empty(monkeypatch):
    z, eng = _build()
    dh, gp = z.data_handler, z.gp_handler
    X, Y = dh.get_gp_data()
    gp.fit(X, Y)

    def boom(self, X, Y):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(GPSimplex, "fit", boom)
    with pytest.raises(RuntimeError, match="fit failed"):
        gp.fit(X, Y + 1.0)
    assert gp._lanes_memo["X"] is None
    monkeypatch.undo()
    gp.fit(X, Y)                                       # memo empty: real fit, not a skip
    assert eng.fit_stats["fit_reused"] == 0 and eng.fit_stats["fit_calls"] == 3
    assert eng.timers.comp["fit_n"] == 3               # the failed call was timed too


# ── timed pass-throughs (LZ 451-459) ──

def test_timed_methods_pass_through(monkeypatch):
    z, eng = _build()
    dh, gp = z.data_handler, z.gp_handler
    comp = eng.timers.comp
    seen = []

    def fake(name, ret):
        def f(self, *a, **k):
            seen.append((name, self, a, k))
            return ret
        return f

    cases = [
        (ZoMBIHop, z, "_check_convergence_to_needle", "convergence"),
        (ZoMBIHop, z, "_declare_needle_from_point", "declare"),
        (ZoMBIHop, z, "_penalize_capped_zone", "cap_zone"),
        (DataHandler, dh, "add_all_points", "add_points"),
        (DataHandler, dh, "take_snapshot", "snapshot"),
        (DataHandler, dh, "_relabel_pared_with_medians", "relabel"),
        (GPSimplex, gp, "get_candidate", "candidate"),
        (GPSimplex, gp, "determine_penalty_ellipsoid", "ellipsoid"),
    ]
    for base, obj, name, key in cases:
        sentinel = object()
        monkeypatch.setattr(base, name, fake(name, sentinel))
        n0 = comp[key + "_n"]
        assert getattr(obj, name)(1, "two", three=3) is sentinel
        assert seen[-1] == (name, obj, (1, "two"), {"three": 3})
        assert comp[key + "_n"] == n0 + 1


def test_nested_timers_are_exclusive(monkeypatch):
    z, eng = _build()
    gp = z.gp_handler
    import time as _time

    def ellipsoid(self, *a, **k):
        _time.sleep(0.05)
        return "E"

    def cap_zone(self, dh, bounds):
        return self.gp_handler.determine_penalty_ellipsoid(bounds[0])

    monkeypatch.setattr(GPSimplex, "determine_penalty_ellipsoid", ellipsoid)
    monkeypatch.setattr(ZoMBIHop, "_penalize_capped_zone", cap_zone)
    assert z._penalize_capped_zone(z.data_handler, _bounds()) == "E"
    comp = eng.timers.comp
    assert comp["ellipsoid"] >= 0.04 and comp["cap_zone"] < 0.04
    assert comp["ellipsoid_n"] == 1 and comp["cap_zone_n"] == 1


def test_real_snapshot_and_add_points_under_tmp(tmp_path):
    z, eng = _build(checkpoint_dir=str(tmp_path))
    dh = z.data_handler
    n0 = int(dh.X_all_actual.shape[0])
    Xn, Yn = _init_rows(n=3, seed=5)
    dh.add_all_points(Xn, Xn.clone(), Yn)
    assert int(dh.X_all_actual.shape[0]) == n0 + 3
    dh.take_snapshot("act0_z0_F_p0")
    comp = eng.timers.comp
    assert comp["add_points_n"] == 1 and comp["snapshot_n"] == 1
    snaps = sorted(p.name for p in (dh.run_dir / "snapshots").iterdir())
    assert any(s.endswith("_act0_z0_F_p0") for s in snaps)
    assert str(dh.run_dir).startswith(str(tmp_path))


def test_second_engine_is_independent():
    z1, e1 = _build()
    z2, e2 = _build(seed=3)
    assert z1.gp_handler._lanes_memo is not z2.gp_handler._lanes_memo
    X, Y = z1.data_handler.get_gp_data()
    z1.gp_handler.fit(X, Y)
    assert e1.fit_stats["fit_calls"] == 1 and e2.fit_stats["fit_calls"] == 0
    n = int(z2.data_handler.X_all_actual.shape[0])
    e1.mask = torch.ones(n, dtype=torch.bool, device=DEV)
    e2.mask = torch.zeros(n, dtype=torch.bool, device=DEV)
    assert bool(z1.data_handler.current_activation_mask().all())
    assert not bool(z2.data_handler.current_activation_mask().any())
    assert e1.timers is not e2.timers and e2.timers.comp["fit_n"] == 0


# ── sandbox cross-check (skipped when the sandbox is absent) ──

def test_matches_sandbox_patches():
    """Same answers and the same memo hit pattern as PF.install_own_best + install_fit_reuse(mode="all")
    on a plain ZoMBIHop with an instance-patched mask (LZ 322-324)."""
    PF = pytest.importorskip("benchmarks.zoom_fix.proposed_fixes")
    z, eng = _build()
    X0, Y0 = _init_rows()
    ref = ZoMBIHop(objective=lambda *a, **k: None, X_init_actual=X0, X_init_expected=X0.clone(),
                   Y_init=Y0, bounds=_bounds(), device="cuda", dtype=DT, checkpoint_dir=None, verbose=False)
    n = int(X0.shape[0])
    own = torch.zeros(n, dtype=torch.bool, device=DEV)
    own[[0, 2, 5, 7, 8, 11]] = True
    pen = torch.ones(n, dtype=torch.bool, device=DEV)
    pen[[2, 8]] = False
    eng.mask = own
    ref.data_handler.current_activation_mask = lambda: own
    for h in (z.data_handler, ref.data_handler):
        h._penalty_mask = pen.clone()
    PF.install_own_best(ref.data_handler)
    ref_stats: dict = {}
    PF.install_fit_reuse(ref.gp_handler, ref_stats, mode="all")
    box = torch.stack([X0[[0, 5]].min(dim=0).values, X0[[0, 5]].max(dim=0).values])
    for name, args in (("get_best_unpenalized", ()), ("get_best_in_bounds", (box,)),
                       ("get_best_in_bounds", (_bounds(),))):
        a = getattr(z.data_handler, name)(*args)
        b = getattr(ref.data_handler, name)(*args)
        assert a[2] == b[2] and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def seq(zz):
        torch.manual_seed(0)
        X, Y = zz.data_handler.get_gp_data()
        zz.gp_handler.fit(X, Y)
        zz.gp_handler.fit(X, Y)
        zz.gp_handler.fit_from_data_handler()
        zz.gp_handler.fit(X, Y * 0.5)
        zz.gp_handler.fit(X, Y * 0.5)
        Q = X[:5]
        return zz.gp_handler.predict(Q)[0]

    mu_lane = seq(z)
    mu_ref = seq(ref)
    for k, _ in FIT_STAT_KEYS:
        if k != "fit_s":
            assert eng.fit_stats[k] == ref_stats[k], k
    assert (eng.fit_stats["fit_calls"], eng.fit_stats["fit_reused"]) == (5, 3)
    assert torch.equal(mu_lane, mu_ref)
