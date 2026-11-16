"""Per-zoom budget rules of the lanes engine (pure; no torch).

Moved verbatim from the sandbox explore method ``benchmarks/zoom_fix/lanes/explore/methods_zbudget.py``
(ZB) lines 77-155: ``ZOOMS``, ``KNOBS``, ``PARAMS``, ``TRANSITION_LOG_CAP``, ``zbudget_hp``,
``validate_zbudget_params``, ``validate_against_stock``, ``activation_budget``, ``budget_action``,
``verdict_fires`` (DESIGN.md section 1.2). The leader's zoom-2 floor (``z2_min=3``) is applied through
these functions by ``src.lanes.lane.DiveLane`` (``_hp_z`` / ``_verdict`` / ``_budget_transition``,
ZB 220-351).

NOT moved (DESIGN 1.2, N4): ``bridge_rule`` (ZB 158-167, bridges are not ported), ``ZB_FUNCS`` /
``REFUSED`` / ``STOCK_AST_SHA`` / ``stock_source_ok`` / ``check_exclusive`` / ``refuse_combinations``
(ZB 80-81, 86-87, 170-209: the explore framework's composition guards; there is no stock copy left to
drift from).

Parameters (0 = the stock value; stock = the core ``max_iterations`` / ``min_iters_per_zoom`` /
``n_consecutive_converged``):
    z<k>_max  zoom-k sends before the budget transition (zoom-in at k < 2, 'zooms_exhausted' zone at 2)
    z<k>_min  own counted zoom-k results needed before the zoom-in / the needle try
    z<k>_nc   consecutive converged checks needed at zoom k
Rules: z*_max 0 or >= 2; z*_min 0 or >= max(2, stock min_iters); z*_nc 0 or >= 1; effective max_k >= min_k.
In production these knobs are the ``lanes_z{k}_{max,min,nc}`` entries of ``src/lanes/hparams.py``.
"""
from __future__ import annotations

from typing import Mapping, Optional

# ZB 77-79, 82
ZOOMS = (0, 1, 2)
KNOBS = (("max", "max_iter"), ("min", "min_iters"), ("nc", "n_consec"))
PARAMS = {f"z{k}_{p}": 0 for p, _ in KNOBS for k in ZOOMS}
TRANSITION_LOG_CAP = 2000


# ── pure helpers (ZB 92-155 verbatim) ───────────────────────────────────────────────────────────

def zbudget_hp(stock: Mapping, k: int, P: Mapping) -> dict:
    """The lane's stock ``_hp()`` with the zoom-k overrides of ``P`` (0 = keep the stock value)."""
    hp = dict(stock)
    if int(k) in ZOOMS:
        for p, key in KNOBS:
            v = int(P.get(f"z{int(k)}_{p}", 0) or 0)
            if v > 0:
                hp[key] = v
    return hp


def validate_zbudget_params(P: Mapping) -> None:
    for name in PARAMS:
        if int(P[name]) < 0:
            raise ValueError(f"zbudget: {name} must be >= 0 (0 = stock), got {P[name]}")
    for k in ZOOMS:
        if 0 < int(P[f"z{k}_max"]) < 2:
            raise ValueError(f"zbudget: z{k}_max must be 0 (stock) or >= 2")
        if 0 < int(P[f"z{k}_min"]) < 2:
            raise ValueError(f"zbudget: z{k}_min must be 0 (stock) or >= 2 (every zoom-in / needle needs >= 2 "
                             f"own counted results)")


def validate_against_stock(P: Mapping, stock: Mapping) -> None:
    """Checks that need the lane's stock values (run when the store exists)."""
    for k in ZOOMS:
        m = int(P[f"z{k}_min"])
        if 0 < m < int(stock["min_iters"]):
            raise ValueError(f"zbudget: z{k}_min={m} is below the stock min_iters {stock['min_iters']} (auditor A1)")
        hk = zbudget_hp(stock, k, P)
        if hk["max_iter"] < hk["min_iters"]:
            raise ValueError(f"zbudget: zoom {k} would need {hk['min_iters']} results from {hk['max_iter']} sends "
                             f"(z{k}_max < z{k}_min); raise z{k}_max")


def activation_budget(hps, ext_cap: int) -> int:
    """sum_k (max_iter_k + ext_cap) over the zooms (stock: max_zooms x (max_iter + ext_cap))."""
    return int(sum(int(h["max_iter"]) + int(ext_cap) for h in hps))


def budget_action(*, k: int, hk: Mapping, max_zooms: int, ext_cap: int, act_budget: int, lines_act: int,
                  sends: int, results: int, ext: int, stock_min: Optional[int] = None) -> str:
    """What the budget transition does on a lane state (pure). With the stock hp, the stock activation
    budget and ``stock_min=None`` it is ZoMBILane._budget_transition (LZ 1297-1317) exactly:
        'activation_budget' | 'send' | 'zoom_in' | 'zooms_exhausted' | 'extend' | 'no_results'
    ``stock_min`` (zbudget): extensions exhausted below the zoom's min results but with >= stock_min
    -> 'fallback_zoom_in' (k < last zoom) / 'fallback_zooms_exhausted' (last zoom): what the stock
    rule did at its own min, never a 'no_results' zone the stock rule would not make."""
    if lines_act >= act_budget:
        return "activation_budget"
    if sends < int(hk["max_iter"]) + ext:
        return "send"
    if results >= int(hk["min_iters"]):
        return "zoom_in" if k < max_zooms - 1 else "zooms_exhausted"
    if ext < ext_cap:
        return "extend"
    if stock_min is not None and results >= int(stock_min):
        return "fallback_zoom_in" if k < max_zooms - 1 else "fallback_zooms_exhausted"
    return "no_results"


def verdict_fires(hk: Mapping, consec: int, results: int) -> bool:
    """LZ 920: the convergence test fires (zoom-in at k < min_zoom_needle, needle try at k >= it)."""
    return consec >= int(hk["n_consec"]) and results >= int(hk["min_iters"])


__all__ = ["ZOOMS", "KNOBS", "PARAMS", "TRANSITION_LOG_CAP", "zbudget_hp", "validate_zbudget_params",
           "validate_against_stock", "activation_budget", "budget_action", "verdict_fires"]
