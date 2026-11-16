"""Lanes geometry helpers (pure tensor functions) and the lanes device / dtype.

Moved verbatim from the sandbox prototype ``benchmarks/zoom_fix/lanes/lane_zombihop.py`` (LZ):
``DEV`` / ``DT`` and the lane constants are LZ 48-53, the helpers LZ 58-175. The function bodies,
signatures and docstrings are token-identical to the sandbox (tests/test_lanes_geometry.py compares
their ASTs with the sandbox file when it is present): the lanes engine's decisions depend on them, and
the port must stay decision-identical (DESIGN.md section 1.2, N1).

``x_requested`` is the sandbox's ``X_expected`` formula (principal-direction line through one packet's
printed points, projected onto the simplex). It differs from ``scripts/run_zombi_main.expected_from_actual``
on purpose (critic C5): the lanes engine must reproduce the validated sandbox.

No import-time side effects (REVIEW F12): ``torch.cuda.is_available()`` is only a query.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

# LZ 48-53
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
LANES = ("front", "back")
LANE_CODE = {"front": 0, "back": 1}
LANE_TAG = {"front": "F", "back": "B"}
COUNTED = ("own_current", "own_current_bridge")


# ── geometry helpers (pure; LZ 58-175 verbatim) ───────────────────────────────────────────────

def in_box(X: torch.Tensor, box: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    return ((X >= box[0] - tol) & (X <= box[1] + tol)).all(dim=-1)


def zoom_floor(min_w: float, rho: float, to_zoom: int, max_zooms: int, rule: str = "graded") -> float:
    """Input-noise width floor of the zoom-``to_zoom`` box (``LaneOptions.floor_rule``).

    'volume': ``min_w`` (= input_noise_threshold_mult x input_noise) at every zoom.
    'graded': ``min_w / rho^(max_zooms-1-to_zoom)``: ``min_w`` at the last zoom, ``min_w / rho`` one
    zoom earlier, ... A box at its floor can then still shrink by rho at the next zoom-in, so the
    user's cap (every axis <= rho x parent) holds on every axis unless the parent itself is narrower
    than the floor."""
    if rule == "graded" and 0.0 < rho < 1.0:
        return float(min_w) / (float(rho) ** max(0, int(max_zooms) - 1 - int(to_zoom)))
    return float(min_w)


def width_ratios(child: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
    return (child[1] - child[0]) / (parent[1] - parent[0]).clamp(min=1e-12)


def nested_box(top_X: torch.Tensor, best_x: torch.Tensor, parent: torch.Tensor, rho: float,
               min_w: float) -> Tuple[Optional[torch.Tensor], dict]:
    """The zoom-(k+1) box from the top rows ``top_X`` of zoom k (``best_x`` = the top-ranked one).
    ``min_w``: the floor of the NEW box (``zoom_floor``: graded by zoom in the lanes core).

    AABB of the rows, clipped inside ``parent``; every axis wider than rho x parent width is cut to
    that cap by a window that keeps ``best_x``; every axis narrower than the input-noise floor
    (min(min_w, parent width): never wider than the parent) is widened about its centre; the box
    is slid back inside the parent. An axis whose floor exceeds the cap keeps the floor width
    (<= parent width; recorded in info['floor_over_cap']). Returns (None, info) when the box would
    not be strictly smaller than the parent (every axis at the parent width: the floor needs the
    whole parent) -- do not zoom further, never widen."""
    P_lo, P_hi = parent[0], parent[1]
    W = (P_hi - P_lo).clamp(min=0.0)
    lo = torch.maximum(top_X.min(dim=0).values, P_lo)
    hi = torch.minimum(top_X.max(dim=0).values, P_hi)
    best = torch.minimum(torch.maximum(best_x, P_lo), P_hi)
    cap = rho * W
    floor = torch.minimum(torch.full_like(W, float(min_w)), W)
    w = hi - lo
    over = w > cap
    c = torch.minimum(torch.maximum(best, lo + cap / 2), hi - cap / 2)
    lo = torch.where(over, c - cap / 2, lo)
    hi = torch.where(over, c + cap / 2, hi)
    w = hi - lo
    under = w < floor
    mid = (lo + hi) / 2
    lo = torch.where(under, mid - floor / 2, lo)
    hi = torch.where(under, mid + floor / 2, hi)
    up = (P_lo - lo).clamp(min=0.0)
    lo, hi = lo + up, hi + up
    dn = (hi - P_hi).clamp(min=0.0)
    lo, hi = lo - dn, hi - dn
    lo, hi = torch.maximum(lo, P_lo), torch.minimum(hi, P_hi)
    box = torch.stack([lo, hi])
    wc = hi - lo
    info = dict(n_top=int(top_X.shape[0]), capped=int(over.sum()), floored=int(under.sum()),
                floor_over_cap=int((floor > cap + 1e-12).sum()), floor=round(float(min_w), 6),
                over_rho=int((wc > cap + 1e-9).sum()),
                width_ratio=[round(float(a), 4) for a in (wc / W.clamp(min=1e-12)).tolist()])
    vol_ratio = float(torch.exp(torch.log((wc / W.clamp(min=1e-12)).clamp(min=1e-12)).sum()))
    info["vol_ratio"] = round(vol_ratio, 6)
    if not vol_ratio < 1.0 - 1e-6:
        info["refused"] = "floor_blocks_zoom"
        return None, info
    return box, info


def check_nested(child: torch.Tensor, parent: torch.Tensor, rho: float, min_w: float,
                 tol: float = 1e-9) -> dict:
    """Assert the user rule: child inside parent, no axis wider than the parent, strictly smaller
    volume, and every axis within rho x parent unless the noise floor (<= parent width) binds."""
    W, w = parent[1] - parent[0], child[1] - child[0]
    inside = bool((child[0] >= parent[0] - tol).all() and (child[1] <= parent[1] + tol).all())
    narrower = bool((w <= W + tol).all())
    floor = torch.minimum(torch.full_like(W, float(min_w)), W)
    cap_ok = bool(((w <= rho * W + tol) | (w <= floor + tol)).all())
    vol = float(torch.prod((w / W.clamp(min=1e-12)).clamp(min=0.0)))
    out = dict(inside=inside, narrower=narrower, cap_ok=cap_ok, vol_ratio=vol,
               over_rho=int((w > rho * W + tol).sum()), eq_parent=int((w >= W - tol).sum()))
    assert inside, f"zoom box not inside its parent: {child.tolist()} vs {parent.tolist()}"
    assert narrower, f"zoom box wider than its parent on some axis: {w.tolist()} vs {W.tolist()}"
    assert cap_ok, f"zoom box axis above rho x parent without a binding floor: {w.tolist()} vs {W.tolist()}"
    assert vol < 1.0 - 1e-6, f"zoom box not strictly smaller than its parent (vol ratio {vol})"
    return out


def inner_box(box: torch.Tensor, margin: float, frac: float) -> torch.Tensor:
    """Box shrunk by an absolute margin per side (<= frac x the axis width), kept simplex-feasible
    (sum(lo) <= 1 <= sum(hi)); the margin is halved until it is."""
    w = (box[1] - box[0]).clamp(min=0.0)
    m = torch.minimum(torch.full_like(w, float(margin)), frac * w)
    for f in (1.0, 0.5, 0.25, 0.125, 0.0):
        lo, hi = box[0] + f * m, box[1] - f * m
        if float(lo.sum()) <= 1.0 + 1e-12 and float(hi.sum()) >= 1.0 - 1e-12:
            return torch.stack([lo, hi])
    return box.clone()


def overlap_frac(a: torch.Tensor, b: torch.Tensor) -> float:
    """Volume of (a ∩ b) / volume of a (axis-aligned)."""
    lo, hi = torch.maximum(a[0], b[0]), torch.minimum(a[1], b[1])
    inter = (hi - lo).clamp(min=0.0)
    wa = (a[1] - a[0]).clamp(min=1e-12)
    return float(torch.prod(inter / wa))


def x_requested(pts: torch.Tensor) -> torch.Tensor:
    """Principal-direction line through one packet's printed points (= sim._x_requested)."""
    from src.utils.simplex import proj_simplex
    if pts.shape[0] < 2:
        return pts.clone()
    xc = pts - pts.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
    p = xc @ Vt[0]
    tv = torch.linspace(p.min().item(), p.max().item(), pts.shape[0], device=pts.device, dtype=pts.dtype)
    return proj_simplex(pts.mean(0, keepdim=True) + tv[:, None] * Vt[0][None])


__all__ = ["DEV", "DT", "LANES", "LANE_CODE", "LANE_TAG", "COUNTED", "in_box", "zoom_floor", "width_ratios",
           "nested_box", "check_nested", "inner_box", "overlap_frac", "x_requested"]
