"""Isolation of the lanes engine from everything else in the process (DESIGN.md section 5, N2, N7; REVIEW F3,
F12; T3). Part 1 (handlers / linebo imports, ``attach``) is T2's ``tests/test_lanes_handlers.py``.

* Importing ``src.lanes`` and every engine module has no side effect (F12): ``random_simplex`` in the three
  modules, the class dicts of DataHandler / GPSimplex / ZoMBIHop / LineBO, the CPU and CUDA RNG states and the
  default device are unchanged, no ``benchmarks`` / ``scripts`` / ``interface`` / ``optimize`` /
  ``visualization`` module is imported, and ``import src.lanes`` does not load the engine (fresh interpreter).
* Building an engine installs Fix 4 only for its lifetime: ``close()`` (or garbage collection, or a failed
  build) restores the original ``random_simplex`` objects; the base classes are never modified; two engines
  ref-count; a second (and a third) engine works.
* ``build_lane_engine`` never reseeds the global RNG (F3).
* The existing ``src/`` files are unchanged (``git diff HEAD`` empty; the only new path is ``src/lanes/``).
Small engines (d = 4, 24 init rows, snapshots off: nothing is written). One decide per engine.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import src.core.linebo as _linebo
import src.core.zombihop as _zh
import src.utils.domain as _domain
from src.core.linebo import LineBO
from src.core.zombihop import ZoMBIHop
from src.lanes import build_lane_engine
from src.lanes.engine import engine_hparams
from src.lanes.handlers import LaneDataHandler, LaneGPSimplex
from src.lanes.rng import SeededSimplex, is_seeded
from src.utils.datahandler import DataHandler
from src.utils.gp_simplex import GPSimplex

REPO = Path(__file__).resolve().parents[1]
_MODS = (_domain, _zh, _linebo)
_BASES = (DataHandler, GPSimplex, ZoMBIHop, LineBO)
D = 4
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


def _originals():
    return {m: m.__dict__["random_simplex"] for m in _MODS}


def _class_dicts():
    return {c.__name__: {k: id(v) for k, v in vars(c).items()} for c in _BASES}


def _small_hp():
    hp = engine_hparams(D)
    hp.update(raw=64, n_restarts=8, nat_grad_max_steps=60)
    return hp


def _engine(seed=0, **kw):
    rng = np.random.default_rng([seed, 9])
    X = rng.dirichlet(np.ones(D), 24)
    Y = np.exp(-((X - np.array([0.5, 0.3, 0.15, 0.05])) ** 2).sum(axis=1) / 0.05)
    t = lambda a: torch.as_tensor(a, device=DEV, dtype=DT)  # noqa: E731
    kw.setdefault("hparams", _small_hp())
    kw.setdefault("snapshots", False)
    return build_lane_engine(t(X), t(X), t(Y).reshape(-1, 1), np.stack([np.zeros(D), np.ones(D)]), seed=seed, **kw)


# ── imports ──────────────────────────────────────────────────────────────────────────────────────

_PROBE = r"""
import hashlib, json, sys
import torch
import src                                                   # baseline: src/__init__ imports the core
import src.core.linebo as lb, src.core.zombihop as zh, src.utils.domain as dm
from src.core.linebo import LineBO
from src.core.zombihop import ZoMBIHop
from src.utils.datahandler import DataHandler
from src.utils.gp_simplex import GPSimplex
H = lambda t: hashlib.sha1(t.cpu().numpy().tobytes()).hexdigest()
def snap():
    return dict(rs=[id(m.random_simplex) for m in (dm, zh, lb)],
                cls={c.__name__: sorted((k, id(v)) for k, v in vars(c).items())
                     for c in (DataHandler, GPSimplex, ZoMBIHop, LineBO)},
                cpu=H(torch.get_rng_state()), cuda=H(torch.cuda.get_rng_state()),
                dev=str(torch.get_default_device()), dtype=str(torch.get_default_dtype()))
mods0 = set(sys.modules)
b = snap()
import src.lanes
engine_after_pkg = "src.lanes.engine" in sys.modules
import src.lanes.engine, src.lanes.store, src.lanes.lane, src.lanes.catchup, src.lanes.audit
import src.lanes.handlers, src.lanes.linebo, src.lanes.rng, src.lanes.labels, src.lanes.budget
a = snap()
new = sorted(set(sys.modules) - mods0)
roots = ("benchmarks", "scripts", "interface", "optimize", "visualization", "sim")
print(json.dumps(dict(same={k: a[k] == b[k] for k in a}, engine_after_pkg=engine_after_pkg,
                      foreign=[m for m in new if m.split(".")[0] in roots],
                      seeded=bool(getattr(dm.random_simplex, "_lanes_seeded", False)
                                  or getattr(dm.random_simplex, "_zoomfix", False)))))
"""


def test_import_has_no_side_effects():
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", _PROBE], cwd=str(REPO), env=env, capture_output=True,
                       text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["same"] == {k: True for k in out["same"]}, out["same"]
    assert out["engine_after_pkg"] is False                    # import src.lanes stays cheap (lazy engine)
    assert out["foreign"] == []
    assert out["seeded"] is False                              # Fix 4 is not installed by any import


# ── engine lifetime ──────────────────────────────────────────────────────────────────────────────

def test_build_close_restores_random_simplex_and_leaves_classes_untouched():
    orig, cls0 = _originals(), _class_dicts()
    mask0 = DataHandler.current_activation_mask
    e = _engine()
    try:
        assert all(is_seeded(m.__dict__["random_simplex"]) for m in _MODS)
        wrapper = _domain.random_simplex
        assert _zh.random_simplex is wrapper and _linebo.random_simplex is wrapper
        assert type(e.dh) is LaneDataHandler and type(e.gp) is LaneGPSimplex
        assert type(e.z).__mro__[1] is ZoMBIHop
        assert _class_dicts() == cls0 and DataHandler.current_activation_mask is mask0
        main, backup, tags = e.lanes["front"].decide(0.0)
        assert tags["main"]["lane"] == "front" and main.ends.shape == (2, D)
    finally:
        e.close()
    assert _originals() == orig and all(m.__dict__["random_simplex"] is orig[m] for m in _MODS)
    assert _class_dicts() == cls0
    e.close()                                                  # idempotent
    assert _originals() == orig
    # a plain ZoMBIHop built afterwards is stock
    X = torch.as_tensor(np.random.default_rng(1).dirichlet(np.ones(D), 8), device=DEV, dtype=DT)
    z = ZoMBIHop(objective=lambda *a, **k: None, X_init_actual=X, X_init_expected=X,
                 Y_init=X[:, :1].clone(), device=str(DEV), dtype=DT, checkpoint_dir=None, verbose=False)
    assert type(z.data_handler) is DataHandler and type(z.gp_handler) is GPSimplex


def test_two_engines_refcount_and_a_second_engine_works():
    orig = _originals()
    a = _engine(seed=0)
    b = _engine(seed=1)
    try:
        assert SeededSimplex.active_count() >= 2
        a.close()
        assert is_seeded(_domain.random_simplex)               # b still alive: Fix 4 stays
        b.lanes["front"].decide(0.0)
        b.lanes["back"].decide(0.0)
        assert [ln.act for ln in b.lanes.values()] == [0, 1]
    finally:
        a.close()
        b.close()
    assert _originals() == orig
    c = _engine(seed=2)                                        # a third engine after both closed
    try:
        assert is_seeded(_domain.random_simplex)
        c.lanes["back"].decide(0.0)
    finally:
        c.close()
    assert _originals() == orig


def test_failed_build_releases_fix4():
    orig = _originals()
    hp = _small_hp()
    hp["not_a_core_knob"] = 1                                  # ZoMBIHop(**hp) raises TypeError
    with pytest.raises(TypeError):
        _engine(hparams=hp)
    assert _originals() == orig


def test_unclosed_engine_releases_fix4_when_collected():
    orig = _originals()
    e = _engine()
    assert is_seeded(_domain.random_simplex)
    del e
    gc.collect()
    assert _originals() == orig


def test_build_never_reseeds_the_global_rng(monkeypatch):
    # REVIEW F3: only the hardware driver may call torch.manual_seed
    def boom(*a, **k):
        raise AssertionError("build_lane_engine reseeded the global RNG")
    for owner, name in ((torch, "manual_seed"), (torch.random, "manual_seed"), (torch.cuda, "manual_seed"),
                        (torch.cuda, "manual_seed_all"), (np.random, "seed")):
        monkeypatch.setattr(owner, name, boom)
    e = _engine()
    e.close()


# ── existing src/ files ──────────────────────────────────────────────────────────────────────────

def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=60)


def test_existing_src_files_unchanged():
    if shutil.which("git") is None or _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("git not available")
    paths = ["src/core", "src/utils", "src/hparams.py", "src/default_hparams.py", "src/__init__.py",
             "optimize/hparams"]
    r = _git("diff", "--stat", "HEAD", "--", *paths)
    assert r.returncode == 0 and r.stdout.strip() == "", r.stdout
    st = _git("status", "--porcelain", "--untracked-files=all", "--", "src")
    assert st.returncode == 0
    lines = [ln for ln in st.stdout.splitlines() if ln.strip()]
    outside = [ln for ln in lines if not ln[3:].strip().strip('"').startswith("src/lanes/")]
    assert outside == [], outside
    assert all(ln.startswith("??") for ln in lines)            # src/lanes is new, never a modified tracked file
