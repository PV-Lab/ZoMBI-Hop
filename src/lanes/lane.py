"""One dive lane of the lanes engine (DESIGN.md 1.2, 1.3.2).

``DiveLane`` is the sandbox's ``ZoMBILane`` (``benchmarks/zoom_fix/lanes/lane_zombihop.py`` = LZ 597-1424)
with the zbudget explore method folded in (``benchmarks/zoom_fix/lanes/explore/methods_zbudget.py`` = ZB
220-351): its per-zoom ``_verdict`` / ``_budget_transition`` ARE the production rule (the leader runs
``z2_min=3``: three own counted zoom-2 results before the needle is tried). The catch-up layer is a mixin
above it (``src/lanes/catchup.py``; ``LeaderLane(CatchupMixin, DiveLane)`` in ``src/lanes/engine.py``), so
the method dispatch is the sandbox MRO's ``FrameworkLaneMixin -> ZBudgetMixin -> CatchupMixin -> ZoMBILane``:
``_verdict`` / ``_budget_transition`` call ``self._zoom_in`` / ``self._try_needle``, which resolve to the
catch-up hold first.

Changes against the sandbox (DESIGN 1.3.2; every other line is LZ / ZB verbatim):
  1. ``_process_packet``: the confirm-and-revise / needle-refinement hooks (LZ 819-831) and the snapshot
     (LZ 832-834; now at ingest, N6) are gone; the bridge branch (LZ 846-858) is the non-bridge one.
  2. ``_verdict`` = ZB 266-309 (per-zoom ``hk = self._hp_z(k)``, bind log, ``_zb_record``).
  3. ``_budget_transition`` = ZB 313-351 (incl. the ``fallback`` events).
  4. ``_hp_z``, ``_act_budget`` (ZB ``_x_act_budget``), ``_zb_state``, ``_zb_bind``, ``_zb_record`` (ZB 220-263);
     counters ``engine.count("zbudget.<key>")``, transitions in ``engine.zb_transitions``; the events
     ``x_zbudget_bind`` / ``x_zbudget_fallback`` carry ``lane / act / zoom`` of this (the computing) lane,
     after the payload, exactly where the explore framework's ``ctx.event`` put them (FW 619-624, REVIEW F1).
  5. Deleted (switches not ported, N4): ``_is_bridge_send``, ``_predict_next_box``, ``_gate``,
     ``_confirm_box`` / ``_finalize_confirm`` / ``_refine_needle``.
  6. ``_box_rows(k)`` without the ``sent`` path (confirm-and-revise only).
  7. ``_rank``: measured Y (the gp_box branch is gone).
  8. ``_box_blocked``: the gate branch is gone; the overlap and cover rules are unchanged (the cover rule
     keeps computing coverage under ``fork_rng``; with ``cross_cover_frac=1.01`` it never fires).
  9. ``_zoom_in``: the ``inflight`` / ``last`` computation is kept (logged); no bridge move, no
     ``pending_confirm``; ``bridge_moved=False``, ``provisional=False``.
 10. ``_try_needle``: no provisional needle; ``provisional=False``.
 11. ``_choose``: ``bridge = False``, ``target = box``; ``LineRec(bridge=False, eff_zoom=k)``; no noface view,
     no bridge event.
 12. ``_start_activation`` keeps ``pending_confirm = None`` (read by ``state()`` and logged).
 13. ``_finish`` commits the decision (one zero-row snapshot, ``labels.decision_label``) unless the
     engine defers decision commits (Phase-2 controller, DESIGN 6.5).

Every ``bridge`` / ``eff_zoom`` / ``pending_confirm`` field stays so the logs and the auditor keep their shape
(``bridge`` is always False, ``eff_zoom == zoom``). No import-time side effects.
"""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.lanes import labels
from src.lanes.budget import (
    TRANSITION_LOG_CAP,
    activation_budget,
    budget_action,
    verdict_fires,
    zbudget_hp,
)
from src.lanes.events import (
    EV_ACTIVATION,
    EV_BOX_OVERLAPS_OTHER,
    EV_DIVE_END,
    EV_EXTEND,
    EV_NEEDLE,
    EV_NEEDLE_HOLD,
    EV_OVERPEN_SHRINK,
    EV_ZB_BIND,
    EV_ZB_FALLBACK,
    EV_ZONE,
    EV_ZOOM_IN,
    EV_ZOOM_REFUSED,
)
from src.lanes.geometry import (
    DEV,
    DT,
    LANE_CODE,
    LANE_TAG,
    check_nested,
    in_box,
    inner_box,
    nested_box,
    overlap_frac,
    zoom_floor,
)
from src.lanes.lines import LaneLine, LineRec

__all__ = ["DiveLane"]


class DiveLane:
    """One lane: its own activation / zoom / box state, an event-driven copy of ``ZoMBIHop.run``'s loop body
    (LZ 597-1424) with the per-zoom budget rule (ZB 220-351).

        lane.decide(now)            -> (main_line, backup_line, tags)   lines for the lane's 2 slots
        lane.on_result(packet, tags) -> packet class for this lane      every packet, any lane
        lane.refresh(now)           -> None | (main_line, backup_line, tags)
        lane.on_pickup(rid, now)    optional: the rail read this line
        lane.compute_seconds        compute of the last decide/refresh (incl. on_result work before it)
    """

    def __init__(self, store, name: str):
        self.s = store
        self.name = name
        self.code = LANE_CODE[name]
        self.opts = store.opts
        self._zb_P = self.opts.zbudget_params()     # ZB 215: P = the method's params (z0_max .. z2_nc)
        self.lines: Dict[int, LineRec] = {}
        self.order: List[int] = []
        self.queued: Dict[str, Optional[LineRec]] = {"main": None, "backup": None}
        self.pending: List[tuple] = []
        self.decisions: List[dict] = []
        self.compute_seconds = 0.0
        self.compute_total = 0.0
        self._acc = 0.0
        self.n_dec = 0
        self.n_refresh_skipped = 0
        self.seen = (store.version, store.penalty_version)
        self.act = -1
        self._start_activation("start", log=False)

    # ── state ──
    @property
    def box(self) -> torch.Tensor:
        return self.boxes[-1]

    @property
    def dh(self):
        return self.s.dh

    def _hp(self):
        z, dh = self.s.z, self.s.dh
        return dict(max_zooms=int(dh.max_zooms), max_iter=int(dh.max_iterations),
                    min_iters=int(z.min_iters_per_zoom), n_consec=int(dh.n_consecutive_converged),
                    min_zoom_needle=int(z.min_zoom_for_needle))

    def floor(self, to_zoom: int) -> float:
        """Width floor of a zoom-``to_zoom`` box (``zoom_floor`` with the lane options)."""
        return zoom_floor(float(self.s.dh.min_box_width()), self.opts.rho, to_zoom, int(self.s.dh.max_zooms),
                          self.opts.floor_rule)

    def _start_activation(self, reason: str, now=None, log=True):
        old = self.act
        self.act = self.s.new_act(self.name)
        mz = int(self.s.dh.max_zooms)
        self.zoom = 0
        self.boxes = [self.s.full.clone()]
        self.box_info: List[dict] = [dict(kind="full")]
        self.results = [0] * (mz + 1)
        self.sends = [0] * (mz + 1)
        self.ext = [0] * (mz + 1)
        self.consec = 0
        self.lines_act = 0
        self.pending_confirm: Optional[dict] = None       # confirm-and-revise is not ported: always None
        self.zone_seq0 = self.s.zone_seq
        self._cover_cache: Optional[tuple] = None
        if log:
            self.s.event(kind=EV_ACTIVATION, lane=self.name, act=self.act, prev_act=old, reason=reason, t=now)
        if self.s.active is self:
            self._sync_dh()

    def _sync_dh(self):
        dh, z = self.s.dh, self.s.z
        dh.bounds = self.box.clone()
        dh.current_zoom_bounds = self.box.clone()
        z.bounds = self.box.clone()
        dh.current_activation = self.act
        dh.current_zoom = self.zoom

    def state(self) -> dict:
        """The lane's state, for harness logs."""
        return dict(lane=self.name, act=self.act, zoom=self.zoom, box=self.box.tolist(),
                    results=list(self.results), sends=list(self.sends), consec=self.consec,
                    lines_act=self.lines_act, pending=len(self.pending),
                    queued=[r.rid if r is not None else None for r in (self.queued["main"], self.queued["backup"])],
                    provisional_box=bool(self.pending_confirm))

    def own_act_mask(self, act: Optional[int] = None) -> torch.Tensor:
        R = self.s.rows
        return (R.t("lane") == self.code) & (R.t("act") == (self.act if act is None else act))

    def _unpen(self) -> torch.Tensor:
        return self.s.dh._penalty_mask.reshape(-1)

    # ── per-zoom budget (ZB 220-263) ──
    # per-zoom hyperparameters (bare _hp() stays stock: the auditor reads lanes['front']._hp())
    def _hp_z(self, k: int) -> dict:
        return zbudget_hp(self._hp(), int(k), self._zb_P)

    def _act_budget(self) -> int:
        mz = int(self._hp()["max_zooms"])
        return activation_budget([self._hp_z(j) for j in range(mz)], self.opts.ext_cap)

    # logging (no RNG, no store state the decisions read)
    def _zb_state(self) -> dict:
        mz = int(self._hp()["max_zooms"])
        d = dict(consec=int(self.consec), lines_act=int(self.lines_act))
        for j in range(mz):
            d[f"s{j}"], d[f"r{j}"], d[f"e{j}"] = int(self.sends[j]), int(self.results[j]), int(self.ext[j])
        return d

    def _zb_bind(self, path: str, stock: str, zb: str, now) -> None:
        k = int(self.zoom)
        self.s.count(f"zbudget.bind_{path}_z{k}_{stock}_to_{zb}")
        # ZB 238 ctx.event(self.s, "bind", ...): lane / act / zoom appended after the payload (FW 621-624)
        self.s.event(kind=EV_ZB_BIND, path=path, stock=str(stock), zb=str(zb), t=now, **self._zb_state(),
                     lane=self.name, act=self.act, zoom=self.zoom)

    def _zb_record(self, path: str, what: str, now, call, stock_would: Optional[bool] = None):
        """Run ``call()`` (a transition) and record it with the per-zoom state before it (ZB 240-263)."""
        k, act0, z0 = int(self.zoom), int(self.act), int(self.zoom)
        hk = self._hp_z(k)
        rec = dict(t=(round(float(now), 1) if now is not None else None), lane=self.name, act=act0, k=k,
                   path=path, what=what, max=int(hk["max_iter"]), min=int(hk["min_iters"]),
                   nc=int(hk["n_consec"]), stock_would=stock_would, **self._zb_state())
        out = call()
        if what == "needle":                       # ok / not_repeatable / no_points / catchup_hold
            outcome = rec["status"] = str(out)
        elif int(self.act) != act0:
            outcome = "ended"                      # a zone, or the zoom-in ended the dive (blocked, floor)
        elif int(self.zoom) != z0:
            outcome = "zoomed"
        else:
            outcome = "held"                       # catch-up kept the lane at zoom k
        rec["outcome"] = outcome
        log = self.s.zb_transitions
        if len(log) < TRANSITION_LOG_CAP:
            log.append(rec)
        else:
            self.s.count("zbudget.transition_log_dropped")
        self.s.count(f"zbudget.tr_{path}_{what}_z{k}_{outcome}")
        return out

    # ── API ──
    def decide(self, now=0.0):
        """Decide the lane's next line (main) and a backup in the same zoom box."""
        t0 = time.perf_counter()
        c0 = dict(self.s.comp)
        self._assume_pickup()
        with self.s.context(self):
            self._process_pending(now)
            self._cross_lane_check(now)
            main, backup = self._choose(now, "regular")
        return self._finish(t0, c0, "decide", main, backup, now)

    def _assume_pickup(self):
        """One line ahead: when the lane decides again, its rail has its previous main; the new
        write replaces the previous backup (unless the harness reported otherwise)."""
        q, b = self.queued["main"], self.queued["backup"]
        if q is not None and q.status == "queued":
            q.status = "picked"
        if b is not None and b.status == "queued":
            b.status = "replaced"

    def refresh(self, now=0.0):
        """Re-decide the queued (not yet picked-up) lines with the data that arrived since."""
        q = self.queued["main"]
        if q is None or q.status != "queued":
            return None
        if not self._refresh_relevant():
            self.n_refresh_skipped += 1
            return None
        t0 = time.perf_counter()
        c0 = dict(self.s.comp)
        with self.s.context(self):
            self._process_pending(now)
            self._cross_lane_check(now)
            reuse = None
            if (not self.opts.refresh_full and q.status == "queued" and q.act == self.act
                    and q.zoom == self.zoom and q.candidate is not None):
                reuse = (q.candidate, q.act, q.zoom, q.cand_src, q.cand_fail)
            self._unqueue()
            main, backup = self._choose(now, "refresh", reuse_candidate=reuse)
        return self._finish(t0, c0, "refresh", main, backup, now)

    def on_result(self, packet, tags: Optional[dict] = None, now=None) -> str:
        """Every packet, any lane: added to the shared data once; returns its class for this lane
        (preliminary 'pending' for own packets; the final class is set at the next decide)."""
        t0 = time.perf_counter()
        rec, _ = self.s.ingest(packet, tags, now)
        self._acc += time.perf_counter() - t0
        if rec["lane"] == self.name:
            return rec["cls"]
        if rec["lane"] is None:
            return rec["cls"]
        return "other_lane"

    def on_pickup(self, rid, now=None):
        rec = self.s.line_by_rid.get(rid) if isinstance(rid, str) else self.lines.get(int(rid))
        if rec is None or rec.lane != self.name:
            return
        if rec.status == "replaced" and rec.rank == 0 and rec.kind is not None and rec.act == self.act:
            # a refreshed-away main that the rail had read after all (refresh too late): it is a
            # send of its zoom again
            self.sends[rec.zoom] += 1
            self.lines_act += 1
        if rec.status in ("queued", "replaced"):
            rec.status, rec.t_pick = "picked", now
        if rec.rank == 1:                      # a backup is read only after its main
            for r in self.lines.values():
                if r.decision == rec.decision and r.rank == 0 and r.status in ("queued", "replaced"):
                    r.status, r.t_pick = "picked", now

    # ── bookkeeping around decisions ──
    def _finish(self, t0, c0, kind, main: LineRec, backup: LineRec, now):
        dt = time.perf_counter() - t0
        self.compute_seconds = self._acc + dt
        self._acc = 0.0
        self.compute_total += self.compute_seconds
        comp = {k: round(v - c0.get(k, 0.0), 4) for k, v in self.s.comp.items()
                if not k.endswith("_n") and v - c0.get(k, 0.0) > 1e-6}
        self.decisions.append(dict(n=self.n_dec, kind=kind, t=now, seconds=round(self.compute_seconds, 4),
                                   comp=comp, act=main.act, zoom=main.zoom, rid=main.rid,
                                   backup=backup.rid, bridge=main.bridge, rows=int(self.s.dh.X_all_actual.shape[0]),
                                   cand_reused=bool(getattr(self, "_cand_reused", False)),
                                   cand_src=main.cand_src, cand_fail=main.cand_fail))
        self.n_dec += 1
        self.seen = (self.s.version, self.s.penalty_version)
        # N6 / DESIGN 1.3.1 item 8: one zero-row snapshot per committed decision (the lane's act / zoom
        # AFTER the decision, its decision index); the Phase-2 controller commits itself after the wire ids
        if not getattr(self.s, "defer_decision_commit", False):
            rec = self.decisions[-1]
            self.s.commit("decision", labels.decision_label(self.act, self.zoom, self.name, rec["n"]), rec)
        return LaneLine(main), LaneLine(backup), {"main": main.tags(), "backup": backup.tags()}

    def _refresh_relevant(self) -> bool:
        v, pv = self.s.version, self.s.penalty_version
        if (v, pv) == self.seen and not self.pending:
            return False
        if not self.opts.refresh_relevance or pv != self.seen[1] or self.pending or self.zoom == 0:
            return True
        r0 = self.s.rows_at_version[self.seen[0]]
        X = self.s.dh.X_all_actual[r0:]
        return bool(in_box(X, self.box).any()) if X.shape[0] else False

    def _unqueue(self):
        for slot in ("main", "backup"):
            r = self.queued[slot]
            if r is not None and r.status == "queued":
                r.status = "replaced"
                if slot == "main" and r.act == self.act:
                    self.sends[r.zoom] -= 1
                    self.lines_act -= 1

    # ── packets of this lane ──
    def _on_own_packet(self, rec: dict, line: LineRec):
        if line.status == "measured":
            rec["cls"] = "duplicate"
            return
        for lid in self.order:                 # rail FIFO: earlier picked lines never returned
            if lid == line.line_id:
                break
            L = self.lines[lid]
            if L.status == "picked":
                L.status = "lost"
        if line.rank == 1:                     # a backup was printed: its main was read before
            for r in self.lines.values():
                if r.decision == line.decision and r.rank == 0 and r.status in ("queued", "replaced"):
                    r.status = "picked"
            if line.act == self.act:
                self.sends[line.zoom] += 1     # it took a slot of its zoom like any send
                self.lines_act += 1
        line.status, line.pkt, line.t_result = "measured", rec["pid"], rec["t"]
        rec["cls"] = "pending"
        self.pending.append((rec, line))

    def _process_pending(self, now):
        while self.pending:
            rec, line = self.pending.pop(0)
            self._process_packet(rec, line, now)

    def _rows(self, rec) -> np.ndarray:
        return np.arange(rec["row0"], rec["row0"] + rec["n"], dtype=np.int64)

    def _process_packet(self, rec: dict, line: LineRec, now):
        # LZ 816-864 minus LZ 819-834 (confirm-and-revise, needle refinement, the snapshot: N6) and the
        # bridge branch of LZ 846-858 (DESIGN 1.3.2 item 1)
        rec["processed_seq"] = len(self.s.events)
        rows = self._rows(rec)
        if line.act != self.act:
            rec["cls"] = "closed_act"
            return
        if line.eff_zoom != self.zoom:
            rec["cls"] = "own_earlier"
            return
        k = self.zoom
        rows_t = torch.as_tensor(rows, device=DEV)
        Xr = self.s.dh.X_all_actual[rows_t] if rows.size else self.s.dh.X_all_actual[:0]
        inb = in_box(Xr, self.box) if k >= 1 else torch.ones(rows.size, dtype=torch.bool, device=DEV)
        frac = float(inb.double().mean()) if rows.size else 0.0
        if rows.size == 0 or (k >= 1 and frac < self.opts.count_frac):
            rec["cls"] = "own_outside"
            rec["in_box_frac"] = frac
            return
        rec["cls"] = "own_current"
        rec["counted_zoom"] = k
        rec["in_box_frac"] = frac
        self.results[k] += 1
        self._verdict(rec, line, rows[inb.cpu().numpy()], now)

    # ── the loop body of ZoMBIHop.run, per own result ──
    def _fit_box(self, box: torch.Tensor):
        z, dh = self.s.z, self.s.dh
        glob = z._is_global_bounds(box)
        X, Y = dh.get_gp_data() if glob else dh.get_zoom_gp_data(box)
        self.s.gp.fit(X, Y)
        return X, Y

    def _own_best_before(self, row0: int):
        dh = self.s.dh
        m = self.own_act_mask() & self._unpen()
        m[row0:] = False
        if not bool(m.any()):
            return None, None, None
        idx = torch.where(m)[0]
        if self.zoom >= 1:
            ib = in_box(dh.X_all_actual[idx], self.box)
            if bool(ib.any()):
                idx = idx[ib]
        j = idx[int(dh.Y_all[idx].reshape(-1).argmax())]
        return dh.X_all_actual[j], dh.Y_all[j], int(j)

    # = ZoMBILane._verdict (LZ 888-926) with the zoom-k min_iters / n_consec (zoom 2 = the needle): ZB 266-309
    def _verdict(self, rec, line: LineRec, rows: np.ndarray, now):
        z, dh, hp = self.s.z, self.s.dh, self._hp()
        k = self.zoom
        hk = self._hp_z(k)
        X, Y = self._fit_box(self.box)                 # post-result refit (run(): post-obj GP refit)
        best_f_local = float(Y.max()) if Y.numel() else float("-inf")
        rows_t = torch.as_tensor(rows, device=DEV, dtype=torch.long)
        unpen = self._unpen()[rows_t] if rows.size else torch.zeros(0, dtype=torch.bool, device=DEV)
        used = rows_t[unpen]
        pbX, pbY, pbi = self._own_best_before(rec["row0"])
        self.s.audit_rows("verdict_rows", self, used, pid=rec["pid"],
                          box=(self.box.tolist() if k >= 1 else None))
        if pbi is not None:
            self.s.audit_rows("prev_best", self, [pbi], pid=rec["pid"])
        ei = None
        skipped = False
        if used.numel() == 0:
            converged, why = False, "all_penalised"
        elif line.candidate is None:
            # get_candidate failed AND no unpenalised GP-mean fallback existed when the line was
            # chosen: no point to judge EI at. The verdict is skipped (the count neither grows nor
            # resets), so a floored box whose candidate search fails cannot block its needle.
            converged, why, skipped = False, "no_candidate", True
        else:
            converged, ei, _ = z._check_convergence_to_needle(
                line.candidate, dh.X_all_actual[used], dh.Y_all[used].reshape(-1), pbX, pbY,
                best_f_ref=best_f_local)
            why = "ei" if line.cand_src == "gp" else "ei_fallback"
        if not skipped:
            self.consec = self.consec + 1 if converged else 0
        rec["verdict"] = dict(converged=bool(converged), why=why, ei=ei, consec=self.consec,
                              results=self.results[k], zoom=k, skipped=skipped,
                              cand_fail=line.cand_fail)
        ours = verdict_fires(hk, self.consec, self.results[k])
        stock = verdict_fires(hp, self.consec, self.results[k])
        what = "needle" if k >= hp["min_zoom_needle"] else "zoom_in"
        if ours != stock:
            self._zb_bind("verdict", what if stock else "wait", what if ours else "wait", now)
        if ours:
            if k >= hp["min_zoom_needle"]:
                status = self._zb_record("verdict", "needle", now, lambda: self._try_needle(now), stock)
                if status == "no_points":
                    self._end_with_zone(self.box, "needle_no_points", now)
            else:
                self._zb_record("verdict", "zoom_in", now, lambda: self._zoom_in(now, "converged"), stock)

    def _box_rows(self, k: int) -> torch.Tensor:
        """Own activation rows of zoom k inside box_k, unpenalised, not in the other lane's active
        box (LZ 928-939 without the confirm-and-revise ``sent`` path)."""
        R = self.s.rows
        zcol = R.t("eff")
        m = self.own_act_mask() & (zcol == k) & self._unpen()
        X = self.s.dh.X_all_actual
        m &= in_box(X, self.boxes[k])
        o = self.s.other(self)
        if o is not None and o.zoom >= 1:
            m &= ~in_box(X, o.box)
        return torch.where(m)[0]

    def _rank(self, idx: torch.Tensor) -> torch.Tensor:
        dh = self.s.dh
        return dh.Y_all[idx].reshape(-1)

    def _make_box(self, idx: torch.Tensor, parent: torch.Tensor, to_zoom: int):
        dh = self.s.dh
        score = self._rank(idx)
        kk = min(int(dh.top_m_points), int(idx.numel()))
        top = idx[torch.topk(score, kk).indices]
        box, info = nested_box(dh.X_all_actual[top], dh.X_all_actual[top[0]], parent,
                               self.opts.rho, self.floor(to_zoom))
        return box, info, top

    def _cover_other(self, box: torch.Tensor) -> Tuple[float, List[int]]:
        """Share of ``box`` covered by the OTHER lane's needles / zones (all of them, whenever they
        were made); cached per (box, penalty state)."""
        zones = [zr for zr in self.s.zones if zr["lane"] != self.name]
        if not zones:
            return 0.0, []
        key = (tuple(round(float(v), 12) for v in box.reshape(-1).tolist()), self.s.penalty_version,
               len(self.s.zones))
        if self._cover_cache is not None and self._cover_cache[0] == key:
            return self._cover_cache[1], self._cover_cache[2]
        frac = self.s.coverage(box, zones)
        seqs = [zr["seq"] for zr in zones]
        self._cover_cache = (key, frac, seqs)
        return frac, seqs

    def _box_blocked(self, box: torch.Tensor, k: int, now, where: str) -> bool:
        """The cross-lane rules for a box about to be committed (a zoom-in): the other lane's active box
        overlapping it by >= the overlap share, or the other lane's needles / zones covering >= the
        cover share -> end the dive (no zone). True when the dive ended. (LZ 974-995 without the gate.)"""
        o = self.s.other(self)
        if o is not None and o.zoom >= 1:
            ov = overlap_frac(box, o.box)
            if ov >= self.opts.cross_overlap_frac:
                self.s.event(kind=EV_BOX_OVERLAPS_OTHER, lane=self.name, act=self.act, zoom=k, overlap=ov,
                             where=where, t=now)
                self._end_dive("overlaps_other_box", now, where=where)
                return True
        cov, seqs = self._cover_other(box)
        if cov >= self.opts.cross_cover_frac:
            self._end_dive("covered_by_other", now, cover=cov, zones=seqs, where=where)
            return True
        return False

    def _zoom_in(self, now, reason: str):
        k = self.zoom
        parent = self.box
        self._fit_box(parent)                          # the zoom-k model (memo: usually a no-op)
        idx = self._box_rows(k)
        src = "own_zoom_rows"
        if idx.numel() == 0:                           # fallback: own rows of any zoom in the box
            m = self.own_act_mask() & self._unpen() & in_box(self.s.dh.X_all_actual, parent)
            o = self.s.other(self)
            if o is not None and o.zoom >= 1:
                m &= ~in_box(self.s.dh.X_all_actual, o.box)
            idx = torch.where(m)[0]
            src = "own_any_zoom_rows"
        if idx.numel() == 0:
            self._end_dive("no_box_rows", now)
            return
        box, info, top = self._make_box(idx, parent, k + 1)
        self.s.audit_rows("box_rows", self, top, to_zoom=k + 1)
        if box is None:
            self.s.event(kind=EV_ZOOM_REFUSED, lane=self.name, act=self.act, zoom=k, info=info, t=now)
            self._end_with_zone(parent, "floor_blocks_zoom", now)
            return
        chk = check_nested(box, parent, self.opts.rho, self.floor(k + 1))
        if self._box_blocked(box, k, now, "zoom_in"):
            return
        cover_other = self._cover_other(box)[0]          # cached by _box_blocked
        self.boxes.append(box)
        self.box_info.append(dict(info, src=src, reason=reason, provisional=False))
        self.zoom = k + 1
        self.consec = 0
        inflight = [L for L in self.lines.values()
                    if L.act == self.act and L.status == "picked" and L.zoom == k]
        last = max(inflight, key=lambda L: L.decision) if inflight else None
        self.s.event(kind=EV_ZOOM_IN, lane=self.name, act=self.act, from_zoom=k, to_zoom=k + 1, reason=reason,
                     results=self.results[k], sends=self.sends[k], box=box.tolist(), parent=parent.tolist(),
                     info=info, nested=chk, inflight=(last.rid if last else None), bridge_moved=False,
                     provisional=False, cover_other=cover_other, t=now)
        self._sync_dh()

    def _try_needle(self, now) -> str:
        z, dh, hp = self.s.z, self.s.dh, self._hp()
        k = self.zoom
        R = self.s.rows
        X, Yall = dh.X_all_actual, dh.Y_all.reshape(-1)
        own2 = self.own_act_mask() & (R.t("eff") == k) & self._unpen()
        # candidate, repeats, pick and local median: own zoom-2 rows INSIDE the zoom-2 box only (the
        # lanes rule; stock _select_needle_point also takes repeats just outside the box, which let
        # the declared point land up to 0.07 outside a 0.03-0.06 box on a worse composition)
        cand = torch.where(own2 & in_box(X, self.box))[0]
        if cand.numel() == 0:
            self.s.event(kind=EV_NEEDLE_HOLD, lane=self.name, act=self.act, status="no_points", t=now)
            return "no_points"
        best = cand[int(Yall[cand].argmax())]
        pool = cand
        near = pool[torch.norm(X[pool] - X[best].unsqueeze(0), dim=1) <= z._repeat_radius(self.box)]
        n_others = int(near.numel()) - 1
        if n_others < int(z.needle_min_repeats):
            self.s.event(kind=EV_NEEDLE_HOLD, lane=self.name, act=self.act, status="not_repeatable",
                         n_others=n_others, t=now)
            return "not_repeatable"
        Yn = Yall[near]
        med = Yn.median()
        pick = int(near[int((Yn - med).abs().argmin())])
        own_med, med_rows = self._own_median(pick, self.act, self.box)
        bl = self.box.tolist()
        self.s.audit_rows("needle_cand", self, cand, box=bl)
        self.s.audit_rows("needle_repeats", self, near, box=bl)
        self.s.audit_rows("needle_pick", self, [pick], box=bl)
        self.s.audit_rows("needle_median_rows", self, med_rows, box=bl)
        n_before = len(dh.needles_results)
        z._declare_needle_from_point(dh, X[pick], dh.Y_all[pick], pick, zoom=k,
                                     iteration=self.lines_act, reason="EI convergence")
        idx = n_before
        dh.needles_results[idx]["median_value"] = own_med if own_med is not None else float(med)
        dh.needles_results[idx]["lane"] = self.name
        zrec = self.s.add_zone("needle", self, idx, pick=pick)  # noqa: F841  (LZ 1109)
        self.s.event(kind=EV_NEEDLE, lane=self.name, act=self.act, zoom=k, needle=idx, pick=pick,
                     cand=int(best), radius=float(z._repeat_radius(self.box)), box=self.box.tolist(),
                     value=float(dh.Y_all[pick]), median=dh.needles_results[idx]["median_value"],
                     n_others=n_others, provisional=False, results=self.results[k],
                     y_true=(float(self.s.predict(X[pick].detach().cpu().numpy().reshape(1, -1))[0])
                             if self.s.predict is not None else None), t=now)
        self._overpen_check(now)
        self._new_activation("needle", now)
        return "ok"

    def _own_median(self, pick: int, act: int, box: torch.Tensor):
        """Median of this activation's own zoom-2 rows inside its zoom-2 ``box`` within
        paring_spatial_halfnoise x input_noise of the pick (the needle's local median, own rows only)."""
        dh, R = self.s.dh, self.s.rows
        thr = float(dh.paring_spatial_halfnoise) * float(dh.input_noise)
        m = (R.t("lane") == self.code) & (R.t("act") == act) & (R.t("eff") == 2)
        m &= in_box(dh.X_all_actual, box)
        m &= torch.norm(dh.X_all_actual - dh.X_all_actual[pick].unsqueeze(0), dim=1) <= thr
        rows = torch.where(m)[0]
        if rows.numel() == 0:
            return None, rows
        return float(dh.Y_all[rows].reshape(-1).median()), rows

    # ── activation ends ──
    def _overpen_check(self, now):
        z, dh = self.s.z, self.s.dh
        test = z.random_sampler(dh.raw, self.s.full[0], self.s.full[1], device=str(DEV), torch_dtype=DT)
        pct = 1.0 - float(dh.get_penalty_mask(test).float().mean())
        if pct > 0.90:
            dh.shrink_all_needle_radii(0.30)
            self.s.penalty_version += 1
            self.s.event(kind=EV_OVERPEN_SHRINK, lane=self.name, act=self.act, pct=pct, t=now)

    def _new_activation(self, reason: str, now):
        self.s.dh._relabel_pared_with_medians()
        self._start_activation(reason, now)

    def _end_with_zone(self, box: torch.Tensor, reason: str, now):
        dh = self.s.dh
        glob = self.s.z._is_global_bounds(box)
        c, _, ci = dh.get_best_unpenalized() if glob else dh.get_best_in_bounds(box)   # own rows
        if ci is not None:
            self.s.audit_rows("zone_centre", self, [ci])
        n0 = int(dh.exclusions.shape[0]) if dh.exclusions is not None else 0
        self.s.z._penalize_capped_zone(dh, box)
        n1 = int(dh.exclusions.shape[0]) if dh.exclusions is not None else 0
        if n1 > n0:
            self.s.add_zone("zone", self, n0, centre=ci, reason=reason)
        self.s.event(kind=EV_ZONE, lane=self.name, act=self.act, zoom=self.zoom, reason=reason, centre=ci,
                     added=n1 > n0, box=box.tolist(), t=now)
        self._overpen_check(now)
        self._new_activation(reason, now)

    def _end_dive(self, reason: str, now, **extra):
        self.s.event(kind=EV_DIVE_END, lane=self.name, act=self.act, zoom=self.zoom, reason=reason, t=now, **extra)
        self._overpen_check(now)                       # (other lanes' zones may have saturated the box)
        self._new_activation(f"dive_end:{reason}", now)

    def _cross_lane_check(self, now):
        if self.zoom == 0:
            return
        # all of the other lane's needles / zones (also those made before this activation started);
        # the same rule already ran before the box was committed (_box_blocked)
        frac, seqs = self._cover_other(self.box)
        if frac >= self.opts.cross_cover_frac:
            self._end_dive("covered_by_other", now, cover=frac, zones=seqs, where="decision")
            return
        m = self.own_act_mask() & self._unpen() & in_box(self.s.dh.X_all_actual, self.box)
        if not bool(m.any()):
            self._end_dive("no_own_unpen", now)

    # ── sending ──
    # = ZoMBILane._budget_transition (LZ 1297-1317) with the zoom-k max_iter / min_iters, the per-zoom
    # activation budget and the no-new-dive-end fallback (budget_action): ZB 313-351
    def _budget_transition(self, now) -> bool:
        hp = self._hp()
        k = self.zoom
        hk = self._hp_z(k)
        mz, cap = int(hp["max_zooms"]), int(self.opts.ext_cap)
        st = dict(k=k, max_zooms=mz, ext_cap=cap, lines_act=int(self.lines_act), sends=int(self.sends[k]),
                  results=int(self.results[k]), ext=int(self.ext[k]))
        ours = budget_action(hk=hk, act_budget=self._act_budget(), stock_min=int(hp["min_iters"]), **st)
        stock = budget_action(hk=hp, act_budget=mz * (int(hp["max_iter"]) + cap), **st)
        if ours != stock:
            self._zb_bind("budget", stock, ours, now)
        if ours == "send":
            return False
        if ours == "extend":
            self.ext[k] += 1
            self.s.event(kind=EV_EXTEND, lane=self.name, act=self.act, zoom=k, ext=self.ext[k],
                         results=self.results[k], t=now)
            return False
        sw = ours == stock
        if ours == "activation_budget":
            self._zb_record("budget", "zone", now, lambda: self._end_with_zone(self.box, "activation_budget", now), sw)
        elif ours == "zoom_in":
            self._zb_record("budget", "zoom_in", now, lambda: self._zoom_in(now, "iterations"), sw)
        elif ours == "zooms_exhausted":
            self._zb_record("budget", "zone", now, lambda: self._end_with_zone(self.box, "zooms_exhausted", now), sw)
        elif ours == "fallback_zoom_in":
            self.s.count(f"zbudget.fallback_zoom_in_z{k}")
            self.s.event(kind=EV_ZB_FALLBACK, action="zoom_in", need=int(hk["min_iters"]), t=now,
                         **self._zb_state(), lane=self.name, act=self.act, zoom=self.zoom)
            self._zb_record("fallback", "zoom_in", now, lambda: self._zoom_in(now, "iterations"), False)
        elif ours == "fallback_zooms_exhausted":
            self.s.count(f"zbudget.fallback_zooms_exhausted_z{k}")
            self.s.event(kind=EV_ZB_FALLBACK, action="zooms_exhausted", need=int(hk["min_iters"]), t=now,
                         **self._zb_state(), lane=self.name, act=self.act, zoom=self.zoom)
            self._zb_record("fallback", "zone", now,
                            lambda: self._end_with_zone(self.box, "zooms_exhausted", now), False)
        else:                                                    # 'no_results' (as stock)
            self._zb_record("budget", "zone", now, lambda: self._end_with_zone(self.box, "no_results", now), sw)
        return True

    def _fallback_candidate(self, box: torch.Tensor):
        """get_candidate failed: the GP posterior-mean argmax over unpenalised samples of the box (the
        point the line's convergence verdict is judged at). None when the box has none."""
        z, dh, gp = self.s.z, self.s.dh, self.s.gp
        try:
            S = z.random_sampler(1024, box[0], box[1], device=str(DEV), torch_dtype=DT).to(DEV, DT)
            S = S[dh.get_penalty_mask(S).reshape(-1)]
            if S.shape[0] == 0:
                return None
            mu, _ = gp.predict(S)
            return S[int(mu.reshape(-1).argmax())]
        except Exception:
            return None

    def _choose(self, now, kind: str, reuse_candidate=None) -> Tuple[LineRec, LineRec]:
        # LZ 1353-1424 without bridges and noface (DESIGN 1.3.2 item 11)
        s, z, dh, gp = self.s, self.s.z, self.s.dh, self.s.gp
        for _ in range(8):
            if not self._budget_transition(now):
                break
        k = self.zoom
        box = self.box
        X, Y = self._fit_box(box)
        best_f = float(Y.max()) if Y.numel() else None
        cand_src, cand_fail = "gp", None
        if reuse_candidate is not None and (self.act, k) == tuple(reuse_candidate[1:3]):
            cand = reuse_candidate[0]                 # refresh: re-rank only (see refresh_full)
            cand_src, cand_fail = reuse_candidate[3], reuse_candidate[4]
            acq = gp.create_acquisition(best_f=best_f)
            self._cand_reused = True
        else:
            self._cand_reused = False
            cand = gp.get_candidate(box, best_f=best_f)
            acq = gp.acq_fn if gp.acq_fn is not None else gp.create_acquisition(best_f=best_f)
            if cand is None:
                cand_fail = getattr(gp, "last_candidate_failure", None) or "unknown"
                cand = self._fallback_candidate(box)
                cand_src = "fallback_mean" if cand is not None else "none"
        bridge = False
        target = box
        if k == 0:
            lbox = box.clone()
        else:
            lbox = inner_box(target, self.opts.inner_margin, self.opts.inner_margin_frac)
        # LineBO needs only a valid simplex anchor (the stock and Fix-8 rankings do not use it);
        # the candidate itself is kept on the line for its convergence verdict.
        if cand is not None:
            x_tell = cand.to(DEV, DT)
        else:
            x_tell = z.random_sampler(1, lbox[0], lbox[1], device=str(DEV), torch_dtype=DT).to(DEV, DT)[0]
        x_tell = x_tell / x_tell.sum()
        acq_rank = copy.copy(acq)
        acq_rank.penalty_mask_fn = None               # core: no penalty-zone trimming
        if hasattr(s.linebo, "last_incumbent"):
            s.linebo.last_incumbent = None
        xl, xr = s.linebo.ranked_line_endpoints(x_tell, lbox, acq_rank)
        inc = getattr(s.linebo, "last_incumbent", None)
        if inc is not None:
            hit = torch.where((dh.X_all_actual == inc.to(dh.X_all_actual).reshape(1, -1)).all(dim=1))[0]
            if hit.numel():
                s.audit_rows("incumbent", self, [int(hit[0])])
        j2 = 1 if xl.shape[0] > 1 else 0
        recs = []
        for rank, j in ((0, 0), (1, j2)):
            lid = s.new_line_id()
            rec = LineRec(line_id=lid, rid=f"{LANE_TAG[self.name]}{lid}", lane=self.name, act=self.act,
                          zoom=k, eff_zoom=k, bridge=bridge, rank=rank, decision=self.n_dec, kind=kind,
                          ends=np.stack([xl[j].detach().cpu().numpy(), xr[j].detach().cpu().numpy()]),
                          box=target.detach().cpu().numpy(), line_box=lbox.detach().cpu().numpy(),
                          candidate=(cand.detach().clone() if cand is not None else None), t_decide=now,
                          cand_src=cand_src, cand_fail=cand_fail)
            self.lines[lid] = rec
            self.order.append(lid)
            s.line_by_rid[rec.rid] = rec
            recs.append(rec)
        self.sends[k] += 1
        self.lines_act += 1
        self.queued = {"main": recs[0], "backup": recs[1]}
        return recs[0], recs[1]
