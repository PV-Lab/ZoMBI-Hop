"""Lane line generator: Fix 8 incumbent-mean lines plus the line gate (lanes engine, Phase 1).

Moved from the sandbox without changing a decision (DESIGN.md 1.2, 1.3.4). Abbreviations:
PF = benchmarks/zoom_fix/proposed_fixes.py, XR = benchmarks/zoom_fix/lanes/explore/xruntime.py,
LZ = benchmarks/zoom_fix/lanes/lane_zombihop.py.

    sandbox                                                    here
    ---------------------------------------------------------  -----------------------------------
    ZoomAwareLineBO.__init__ / _incumbent        PF 221-246    LaneLineBO.__init__ / _incumbent
    IncumbentMeanLineBO.__init__ / ranked_...    PF 311-351    LaneLineBO._ranked (incumbent=True)
    stock LineBO (incmean off)                   LZ 331-332    LaneLineBO._ranked (incumbent=False)
    no_face_view                                 PF 635-655    no_face_view
    valid_lines                                  XR 63-73      valid_lines
    line gate around ranked_line_endpoints       XR 87-134     LaneLineBO.ranked_line_endpoints
    "lines" timer around the gate                LZ 453        LaneLineBO.ranked_line_endpoints
NOT moved: ``ZoomAwareLineBO.ranked_line_endpoints`` (PF 248-292; Fix 8 never calls it).

The gate in the leader. The sandbox installed it because the arm had explore methods (XR 171-172),
with no ``line_box`` / ``rank_acquisition`` / ``line_generator`` hooks: it passed ``bounds`` and
the acquisition through unchanged, ran the generator, dropped rows whose endpoints are not finite,
not on the simplex (1e-6) or outside the TARGET box (the lane's zoom box: bridges are not ported,
so ``_x_pred`` is always None), and re-ran the same generator with the same arguments when no row
survived (XR 100-131). It never acted in the 32 leader campaigns. Stats: ``engine.stats
["line_gate"]`` = ``calls, invalid_rows, fallback, degenerate_main, rows_returned``
(``lbox_changed`` dropped: there are no line-box hooks). No import-time side effects.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Optional

import torch

from src.core.linebo import (
    LineBO,
    batch_line_bounds_segments,
    random_chords_through_simplex,
)

__all__ = ["LaneLineBO", "no_face_view", "valid_lines", "LINE_TOL", "GATE_STAT_KEYS"]

LINE_TOL = 1e-6          # XR 36: endpoints >= -tol, |sum - 1| <= tol, inside the target box +- tol
GATE_STAT_KEYS = ("calls", "invalid_rows", "fallback", "degenerate_main", "rows_returned")


def valid_lines(xl: torch.Tensor, xr: torch.Tensor, box: torch.Tensor, tol: float = LINE_TOL) -> torch.Tensor:
    """Rows whose two endpoints are finite simplex points inside ``box`` (then the whole straight
    segment is: the box and the simplex are convex). XR 63-73 verbatim."""
    xl = torch.as_tensor(xl).reshape(-1, box.shape[-1]).to(box)
    xr = torch.as_tensor(xr).reshape(-1, box.shape[-1]).to(box)
    ok = torch.ones(xl.shape[0], dtype=torch.bool, device=box.device)
    for E in (xl, xr):
        ok &= torch.isfinite(E).all(dim=1)
        ok &= (E >= -tol).all(dim=1) & ((E.sum(dim=1) - 1.0).abs() <= tol)
        ok &= (E >= box[0] - tol).all(dim=1) & (E <= box[1] + tol).all(dim=1)
    return ok


def no_face_view(acquisition_function):
    """Fix 7. The acquisition LineBO ranks lines with, minus the simplex-face term (PF 635-655).

    RepulsiveAcquisition = UCB - lambda * (zone terms + sum_i exp(-x_i / eps)), with
    eps = input_noise and lambda = max(10 * median|UCB|, 100). This view sets
    boundary_epsilon = 0 (face term off) and keeps everything else: UCB, the needle-zone
    repulsion, and the penalty_mask_fn. Fix 8 uses it to isolate the zone term
    (view - view.base); get_candidate is untouched.
    """
    if acquisition_function is None or not hasattr(acquisition_function, "boundary_epsilon"):
        return acquisition_function
    view = copy.copy(acquisition_function)
    view.boundary_epsilon = 0.0
    return view


class LaneLineBO(LineBO):
    """LineBO for the lanes engine (DESIGN.md 1.3.1 item 5)::

        LaneLineBO(None, d, num_points_per_line=100, num_lines=opts.num_lines, device=str(DEV),
                   zombi_ref=[engine.z], engine=engine, incumbent=opts.incmean)

    ``incumbent=True`` (the leader, Fix 8): zoom 0 (global bounds) is stock
    ``LineBO.ranked_line_endpoints``; zoom >= 1 ranks 100 chords through the denoised incumbent
    (argmax of the GP posterior mean over this activation's unpenalised rows inside the line box)
    by the chord mean of GP mean + needle-zone repulsion (no face term, no UCB bonus); falls back
    to 100 random chords through the box (``n_fallback``) when there is no incumbent or fewer than
    10 non-degenerate chords through it. ``incumbent=False``: stock LineBO at every zoom.
    ``ranked_line_endpoints`` = "lines" timer -> line gate -> ``_ranked``.
    """

    def __init__(self, *args, zombi_ref: list, engine=None, incumbent: bool = True,
                 n_candidates: int = 100, incumbent_frac: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # PF 221-228 (ZoomAwareLineBO.__init__) with the IncumbentMeanLineBO defaults (PF 311-314)
        self.zombi_ref = zombi_ref
        self.n_candidates = int(n_candidates)
        self.incumbent_frac = float(incumbent_frac)     # kept for parity; Fix 8 does not read it
        self.last_incumbent: Optional[torch.Tensor] = None
        self.n_fallback = 0
        self.incumbent = bool(incumbent)
        self._lanes_engine = engine
        self.gate_stats = {k: 0 for k in GATE_STAT_KEYS}

    # ── Fix 8 ──
    # incumbent = argmax of the GP posterior mean over this activation's
    # unpenalised points inside the box (a denoised "best so far"). PF 232-246 verbatim.
    def _incumbent(self, bounds: torch.Tensor) -> Optional[torch.Tensor]:
        z = self.zombi_ref[0]
        dh = z.data_handler
        X = dh.X_all_actual
        lo, hi = bounds[0], bounds[1]
        keep = dh._penalty_mask & dh.current_activation_mask()
        keep = keep & ((X >= lo - 1e-9) & (X <= hi + 1e-9)).all(dim=1)
        if not keep.any():
            return None
        Xk = X[keep]
        try:
            mu = z.gp_handler.predict(Xk)[0].reshape(-1)
        except Exception:
            mu = dh.Y_all[keep].reshape(-1)
        return Xk[int(mu.argmax())].clone()

    def _ranked(self, x_tell, bounds=None, acquisition_function=None):
        """The line generator without gate or timer: Fix 8 (PF 316-351 verbatim) when
        ``incumbent``, else stock ``LineBO.ranked_line_endpoints``."""
        if not self.incumbent:
            return LineBO.ranked_line_endpoints(self, x_tell, bounds, acquisition_function)
        z = self.zombi_ref[0] if self.zombi_ref else None
        if (z is None or bounds is None or acquisition_function is None
                or z._is_global_bounds(bounds)):
            return LineBO.ranked_line_endpoints(self, x_tell, bounds, acquisition_function)
        dev, dt, d = self.device, self.dtype, self.d
        b = bounds.to(device=dev, dtype=dt)
        x_star = self._incumbent(b)
        self.last_incumbent = x_star
        xl = xr = None
        if x_star is not None:
            D = self.domain.directions(3 * self.n_candidates, d, device=str(dev), dtype=dt)
            l, r, _, _, _ = batch_line_bounds_segments(x_star.to(dev, dt), D, b)   # valid chords only
            good = (r - l).norm(dim=1) > 1e-9
            if int(good.sum()) >= 10:
                xl, xr = l[good][: self.n_candidates], r[good][: self.n_candidates]
        if xl is None:
            self.n_fallback += 1
            xl, xr = random_chords_through_simplex(self.n_candidates, b, dev, dt, domain=self.domain)
            if xl.shape[0] == 0:
                return LineBO.ranked_line_endpoints(self, x_tell, bounds, acquisition_function)
        points = self._line_points(xl, xr)                                        # (k, P, d)
        k, P, _ = points.shape
        mu = z.gp_handler.predict(points.reshape(-1, d))[0].reshape(k, P).to(device=dev, dtype=dt)
        nf = no_face_view(acquisition_function)
        base = getattr(nf, "base", None)
        if base is not None:
            zone = (self._acquisition_at_points(points, nf)
                    - self._acquisition_at_points(points, base))                  # <= 0
        else:
            zone = torch.zeros_like(mu)
        score = (mu + zone).mean(dim=1)
        order = torch.argsort(score, descending=True)[: self.num_lines]
        if order.numel() < min(2, self.num_lines):
            order = torch.cat([order, order[:1].repeat(min(2, self.num_lines) - order.numel())])
        return xl[order], xr[order]

    # ── gate + timer ──
    def _timed(self, key: str):
        eng = self._lanes_engine
        t = getattr(eng, "timers", None) if eng is not None else None
        return t.timed(key) if t is not None else nullcontext()

    def _gate_stats(self) -> dict:
        """``engine.stats["line_gate"]``: this generator's own counters, linked into the engine's
        ``stats`` dict (created when the engine has none)."""
        eng = self._lanes_engine
        stats = getattr(eng, "stats", None)
        if stats is None:
            stats = {}
            eng.stats = stats
        if stats.get("line_gate") is not self.gate_stats:
            stats["line_gate"] = self.gate_stats
        return self.gate_stats

    def ranked_line_endpoints(self, x_tell, bounds=None, acquisition_function=None):
        """"lines" timer (LZ 453) around the line gate (XR 100-131) around ``_ranked``.

        The target box is the computing lane's zoom box (``engine.active.box``). Without an engine
        or outside a lane context (no ``engine.active``) the gate is skipped: the lanes never do
        that (``LaneEngine.context`` sets ``active`` around every decision)."""
        with self._timed("lines"):
            eng = self._lanes_engine
            lane = getattr(eng, "active", None) if eng is not None else None
            if lane is None:
                return self._ranked(x_tell, bounds, acquisition_function)
            st = self._gate_stats()
            st["calls"] += 1
            target = lane.box
            xl, xr = self._ranked(x_tell, bounds, acquisition_function)
            ok = valid_lines(xl, xr, target)
            if not bool(ok.all()):
                st["invalid_rows"] += int((~ok).sum())
                if bool(ok.any()):
                    xl, xr = xl[ok], xr[ok]
                else:
                    st["fallback"] += 1
                    xl, xr = self._ranked(x_tell, bounds, acquisition_function)
            st["rows_returned"] += int(xl.shape[0])
            if float(torch.norm(xr[0] - xl[0])) < 1e-9:
                st["degenerate_main"] += 1
            return xl, xr
