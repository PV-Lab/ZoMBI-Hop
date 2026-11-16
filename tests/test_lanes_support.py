"""The small T1 modules: src/lanes/instrument.py (Timers = the sandbox's exclusive-time stack, LZ 429-460),
src/lanes/lines.py (LineRec / LaneLine moved verbatim, LZ 187-233: AST identity with the sandbox when it is
present), src/lanes/events.py (frozen sandbox event kinds, DESIGN N5 / REVIEW F1) and the package's public
names (lazy ``build_lane_engine``).

Pure / CPU; fast. REVIEW F9 autouse fixture as in every lanes test module; writes nothing.
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import src.lanes as lanes_pkg
from src.lanes import events as E
from src.lanes.instrument import TIMED_METHODS, TIMER_KEYS, Timers
from src.lanes.lines import LaneLine, LineRec

REPO = Path(__file__).resolve().parents[1]
LANES_DIR = REPO / "benchmarks" / "zoom_fix" / "lanes"
LZ_PATH = LANES_DIR / "lane_zombihop.py"


@pytest.fixture(autouse=True)
def _restore_rng_and_simplex():
    """REVIEW F9: restore CPU / CUDA RNG states and the three random_simplex attributes."""
    from src.core import linebo as _lb
    from src.core import zombihop as _zh
    from src.utils import domain as _dom
    saved = {m: m.__dict__.get("random_simplex") for m in (_dom, _zh, _lb)}
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for m, f in saved.items():
            setattr(m, "random_simplex", f)


# ── Timers ───────────────────────────────────────────────────────────────────────────────────────

def _busy(seconds: float) -> None:
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        pass


def test_timers_exclusive_nested():
    tm = Timers()
    t0 = time.perf_counter()
    with tm.timed("outer"):
        _busy(0.02)
        with tm.timed("inner"):
            _busy(0.03)
        with tm.timed("inner"):
            _busy(0.01)
    total = time.perf_counter() - t0
    c = tm.comp
    assert c["outer_n"] == 1 and c["inner_n"] == 2
    assert c["inner"] >= 0.04 - 1e-4 and c["outer"] >= 0.02 - 1e-4
    # exclusive times partition the outer call: the inner seconds are not counted in outer
    assert c["outer"] + c["inner"] <= total + 1e-6
    assert c["outer"] + c["inner"] == pytest.approx(total, abs=5e-3)
    assert tm.depth == 0
    assert set(tm.snapshot()) == {"outer", "outer_n", "inner", "inner_n"}
    assert tm.rounded(3)["inner_n"] == 2


def test_timers_count_on_exception_and_shared_comp():
    from collections import defaultdict
    comp = defaultdict(float)
    tm = Timers(comp)
    assert tm.comp is comp
    comp["ingest"] += 0.5                                   # the engine adds its own key directly (LZ 539)
    with pytest.raises(ValueError):
        with tm.timed("fit"):
            raise ValueError("x")
    assert comp["fit_n"] == 1 and comp["fit"] >= 0.0 and tm.depth == 0
    f = tm.wrap(lambda a, b=1: a + b, "lines")
    assert f(2, b=3) == 5 and comp["lines_n"] == 1


def test_timer_keys_are_the_sandbox_keys():
    assert [k for _, _, k in TIMED_METHODS] == ["fit", "candidate", "ellipsoid", "lines", "add_points", "snapshot",
                                                "relabel", "convergence", "declare", "cap_zone"]
    assert TIMER_KEYS[("gp", "fit")] == "fit" and TIMER_KEYS[("z", "_penalize_capped_zone")] == "cap_zone"


# ── LineRec / LaneLine ───────────────────────────────────────────────────────────────────────────

def _rec(**kw):
    base = dict(line_id=3, rid="F3", lane="front", act=0, zoom=1, eff_zoom=1, bridge=False, rank=0, decision=2,
                kind="regular", ends=np.arange(8, dtype=float).reshape(2, 4), box=np.zeros((2, 4)),
                line_box=np.zeros((2, 4)), candidate=None, t_decide=12.5)
    base.update(kw)
    return LineRec(**base)


def test_linerec_and_laneline():
    r = _rec()
    assert (r.cand_src, r.cand_fail, r.status, r.t_pick, r.t_result, r.pkt) == ("gp", None, "queued", None, None, None)
    assert r.tags() == dict(lane="front", act=0, zoom=1, line=3, rid="F3", bridge=False, rank=0, decision=2,
                            kind="regular")
    ll = LaneLine(r)
    assert ll.rid == "F3" and ll.tags == r.tags()
    assert np.array_equal(np.asarray(ll), r.ends) and ll.ends is not r.ends
    r.ends[0, 0] = 99.0
    assert ll.ends[0, 0] == 0.0                              # a copy
    assert repr(ll) == "LaneLine(F3 a0 z1)"


def _classes(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.ClassDef)}


@pytest.mark.skipif(not LZ_PATH.exists(), reason="sandbox lane_zombihop.py not present")
def test_lines_are_the_sandbox_code():
    import src.lanes.lines as M
    sand, prod = _classes(LZ_PATH), _classes(Path(M.__file__))
    differ = [n for n in ("LineRec", "LaneLine") if sand.get(n) != prod.get(n)]
    assert not differ, f"not token-identical to lane_zombihop.py: {differ}"


# ── events ───────────────────────────────────────────────────────────────────────────────────────

def test_event_kinds_are_the_sandbox_strings():
    assert (E.EV_ZB_BIND, E.EV_ZB_FALLBACK, E.EV_CU_HOLD, E.EV_CU_RELEASE) == \
        ("x_zbudget_bind", "x_zbudget_fallback", "x_catchup_hold", "x_catchup_release")
    assert E.method_event("zbudget", "bind") == E.EV_ZB_BIND
    assert E.method_event("catchup", "release") == E.EV_CU_RELEASE
    assert {"extend", "needle_hold", "zoom_refused", "box_overlaps_other", "overpen_shrink", "activation",
            "zoom_in", "needle", "zone", "dive_end"} <= E.STORE_EVENT_KINDS
    assert not (E.STORE_EVENT_KINDS & E.NOT_PORTED_KINDS)
    consts = {k: v for k, v in vars(E).items() if k.startswith("EV_")}
    assert len(set(consts.values())) == len(consts) and set(consts.values()) == E.STORE_EVENT_KINDS


@pytest.mark.skipif(not LZ_PATH.exists(), reason="sandbox lanes code not present")
def test_event_kinds_occur_in_the_sandbox_sources():
    """Every kind the port may emit is a kind the sandbox emitted (no new kind on the emulator path, F1)."""
    text = LZ_PATH.read_text(encoding="utf-8")
    ex = LANES_DIR / "explore"
    for f in ("methods_zbudget.py", "methods_catchup.py"):
        if (ex / f).exists():
            text += (ex / f).read_text(encoding="utf-8")
    for kind in E.STORE_EVENT_KINDS:
        if kind.startswith("x_"):
            _, method, sub = kind.split("_", 2)
            assert f'ctx.event(self.s, "{sub}"' in text, kind
        else:
            assert f'kind="{kind}"' in text, kind


# ── package ──────────────────────────────────────────────────────────────────────────────────────

def test_package_public_names():
    for n in ("PRESETS", "LANES_HPARAMS", "lanes_defaults", "lanes_conflicts", "build_lane_engine", "LaneOptions"):
        assert hasattr(lanes_pkg, n), n
    assert callable(lanes_pkg.build_lane_engine)
    assert lanes_pkg.PRESETS == {"leader": {}}
