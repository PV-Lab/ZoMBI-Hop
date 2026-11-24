"""src/lanes/budget.py: the leader's zoom-2 floor (z2_min = 3), the budget actions at zoom 2, the
activation budget, zooms 0/1 untouched, and all-zero parameters == the sandbox's stock lane rule
(lane_zombihop.py LZ 1297-1317 budget transition, LZ 920 verdict) on 600 random states (the recipe of
explore/check_zbudget.py T1). Also an AST identity check against the sandbox source
(explore/methods_zbudget.py ZB 77-155; skipped when the sandbox is absent).

Pure; no GPU work. REVIEW F9 autouse fixture as in every lanes test module.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from src.lanes import budget as B
from src.lanes.options import LaneOptions

REPO = Path(__file__).resolve().parents[1]
ZB_PATH = REPO / "benchmarks" / "zoom_fix" / "lanes" / "explore" / "methods_zbudget.py"

STOCK = dict(max_zooms=3, max_iter=5, min_iters=2, n_consec=2, min_zoom_needle=2)   # the core defaults
CAP = 3                                                                               # lanes_ext_cap
LEADER_P = LaneOptions().zbudget_params()                                             # z2_min = 3


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


def _act_budget(P, stock=STOCK, cap=CAP):
    return B.activation_budget([B.zbudget_hp(stock, j, P) for j in range(stock["max_zooms"])], cap)


def _action(P, *, k, sends, results, ext, lines_act=0, stock=STOCK, cap=CAP):
    """What the ported lane's _budget_transition (ZB 313-322) decides."""
    return B.budget_action(k=k, hk=B.zbudget_hp(stock, k, P), max_zooms=stock["max_zooms"], ext_cap=cap,
                           act_budget=_act_budget(P, stock, cap), lines_act=lines_act, sends=sends,
                           results=results, ext=ext, stock_min=int(stock["min_iters"]))


# ── the leader ───────────────────────────────────────────────────────────────────────────────────

def test_leader_params():
    assert LEADER_P == dict(B.PARAMS, z2_min=3)
    B.validate_zbudget_params(LEADER_P)
    B.validate_against_stock(LEADER_P, STOCK)


def test_zoom2_floor_verdict():
    hz2 = B.zbudget_hp(STOCK, 2, LEADER_P)
    assert hz2 == dict(STOCK, min_iters=3)
    assert B.verdict_fires(hz2, consec=2, results=2) is False
    assert B.verdict_fires(hz2, consec=2, results=3) is True
    assert B.verdict_fires(hz2, consec=1, results=3) is False
    # the stock rule would have tried the needle with 2 results
    assert B.verdict_fires(STOCK, consec=2, results=2) is True


def test_zoom2_budget_actions():
    # sends ran out with 2 results < 3: extend while ext < ext_cap
    assert _action(LEADER_P, k=2, sends=5, results=2, ext=0) == "extend"
    assert _action(LEADER_P, k=2, sends=6, results=2, ext=1) == "extend"
    assert _action(LEADER_P, k=2, sends=7, results=2, ext=2) == "extend"
    # not yet at the (extended) budget: keep sending
    assert _action(LEADER_P, k=2, sends=5, results=2, ext=1) == "send"
    # extensions exhausted with 2 (= the stock min) results: the stock zone, never a new no_results zone
    assert _action(LEADER_P, k=2, sends=8, results=2, ext=3) == "fallback_zooms_exhausted"
    # extensions exhausted with 1 result: no_results (as the stock rule)
    assert _action(LEADER_P, k=2, sends=8, results=1, ext=3) == "no_results"
    # 3 results: the regular budget end of the last zoom
    assert _action(LEADER_P, k=2, sends=5, results=3, ext=0) == "zooms_exhausted"
    assert _action(LEADER_P, k=2, sends=7, results=3, ext=2) == "zooms_exhausted"
    # the activation budget binds first
    assert _action(LEADER_P, k=2, sends=3, results=1, ext=0, lines_act=24) == "activation_budget"


def test_activation_budget_is_24():
    assert _act_budget(LEADER_P) == 24 == 3 * (5 + CAP)
    from src.lanes.hparams import lanes_lines_per_activation
    assert lanes_lines_per_activation() == 24


def test_zooms_0_and_1_untouched():
    assert B.zbudget_hp(STOCK, 0, LEADER_P) == STOCK
    assert B.zbudget_hp(STOCK, 1, LEADER_P) == STOCK
    for k in (0, 1):
        for sends in range(0, 9):
            for results in range(0, 6):
                for ext in range(0, 4):
                    for lines_act in (0, 10, 23, 24):
                        a = _action(LEADER_P, k=k, sends=sends, results=results, ext=ext, lines_act=lines_act)
                        s = _action(B.PARAMS, k=k, sends=sends, results=results, ext=ext, lines_act=lines_act)
                        assert a == s, (k, sends, results, ext, lines_act, a, s)


def test_other_params():
    P = dict(B.PARAMS, z2_max=7, z1_max=7, z0_max=3)
    assert _act_budget(P) == (3 + 3) + (7 + 3) + (7 + 3)
    assert B.zbudget_hp(STOCK, 2, dict(B.PARAMS, z2_nc=1)) == dict(STOCK, n_consec=1)
    assert B.zbudget_hp(STOCK, 3, LEADER_P) == STOCK                      # zooms outside 0..2: stock
    assert _action(dict(B.PARAMS, z1_min=3), k=1, sends=8, results=2, ext=3) == "fallback_zoom_in"
    with pytest.raises(ValueError):
        B.validate_zbudget_params(dict(B.PARAMS, z2_min=1))
    with pytest.raises(ValueError):
        B.validate_zbudget_params(dict(B.PARAMS, z0_max=1))
    with pytest.raises(ValueError):
        B.validate_zbudget_params(dict(B.PARAMS, z1_nc=-1))
    with pytest.raises(ValueError):
        B.validate_against_stock(dict(B.PARAMS, z2_min=6), STOCK)           # 6 results from 5 sends
    with pytest.raises(ValueError):
        B.validate_against_stock(dict(B.PARAMS, z2_min=2), dict(STOCK, min_iters=3))   # below stock (A1)
    assert B.TRANSITION_LOG_CAP == 2000


# ── all-zero params == the sandbox stock rule ────────────────────────────────────────────────────

def _stock_transition(hp, ext_cap, *, k, lines_act, sends, results, ext):
    """LZ 1297-1317 transcribed (action names of budget_action)."""
    if lines_act >= hp["max_zooms"] * (hp["max_iter"] + ext_cap):
        return "activation_budget"
    if sends < hp["max_iter"] + ext:
        return "send"
    if results >= hp["min_iters"]:
        if k < hp["max_zooms"] - 1:
            return "zoom_in"
        return "zooms_exhausted"
    if ext < ext_cap:
        return "extend"
    return "no_results"


def _stock_verdict(hp, consec, results):
    """LZ 920."""
    return consec >= hp["n_consec"] and results >= hp["min_iters"]


def test_default_params_equal_stock_rule_600_states():
    rng = np.random.default_rng(0)
    nt = nv = 0
    n = 600
    for _ in range(n):
        k = int(rng.integers(0, 3))
        results = [int(v) for v in rng.integers(0, 6, 3)]
        sends = [int(v) for v in rng.integers(0, 11, 3)]
        ext = [int(v) for v in rng.integers(0, 4, 3)]
        consec = int(rng.integers(0, 4))
        lines_act = int(rng.integers(0, 27))
        ours = _action(B.PARAMS, k=k, sends=sends[k], results=results[k], ext=ext[k], lines_act=lines_act)
        stock = _stock_transition(STOCK, CAP, k=k, lines_act=lines_act, sends=sends[k], results=results[k],
                                  ext=ext[k])
        nt += int(ours != stock)
        nv += int(B.verdict_fires(B.zbudget_hp(STOCK, k, B.PARAMS), consec, results[k])
                  != _stock_verdict(STOCK, consec, results[k]))
    assert nt == 0, f"{nt}/{n} budget transitions differ"
    assert nv == 0, f"{nv}/{n} verdicts differ"


# ── verbatim move ────────────────────────────────────────────────────────────────────────────────

def _top_level(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node)
    return out


@pytest.mark.skipif(not ZB_PATH.exists(), reason="sandbox methods_zbudget.py not present")
def test_budget_is_the_sandbox_code():
    sand, prod = _top_level(ZB_PATH), _top_level(Path(B.__file__))
    names = ("ZOOMS", "KNOBS", "PARAMS", "TRANSITION_LOG_CAP", "zbudget_hp", "validate_zbudget_params",
             "validate_against_stock", "activation_budget", "budget_action", "verdict_fires")
    differ = [n for n in names if sand.get(n) != prod.get(n)]
    assert not differ, f"not token-identical to methods_zbudget.py: {differ}"
