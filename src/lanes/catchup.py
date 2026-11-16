"""Catch-up: hold a zoom transition until the zoom's MEASURED top level catches up with the previous zoom's.

Moved from the sandbox explore method ``benchmarks/zoom_fix/lanes/explore/methods_catchup.py`` (CU) without
changing a decision (DESIGN.md 1.2, 1.3.3). Line geometry is never touched: every line is still the base
generator's straight chord of the CURRENT zoom box; only WHEN the lane leaves a zoom changes.

The user's standard (report/metrics2.py, 'nodip_obs'): per dive (lane + activation) and per SENT zoom k,
T_k = mean of the best m MEASURED droplets of the dive's own packets sent at zoom k (m = 8 in 6D, 11 in
10D); the dive has no dip when T1 >= T0 - eps and T2 >= T1 - eps (eps 0.01). Catch-up applies that test
online, at the moment the lane would LEAVE zoom k in ``catchup_zooms``:

    leaving zoom k      = a zoom-in k -> k+1 (verdict 'converged' or budget 'iterations'), or a needle
                          declaration at zoom k (zoom 2), but only when the stock repeat rule would declare it
    T_k, T_{k-1}        = the top-m level over the dive's own packets SENT at zoom k / k-1 that have ARRIVED
                          (``engine.packets``, measured y, in or out of the box). Never noise-free values,
                          never model predictions.
    hold                when T_k < T_{k-1} - eps, fewer than ``catchup_max`` extra lines of this activation
                        and zoom were sent, and the activation budget keeps room (lines_act + 1 + reserve <
                        activation budget; reserve = the next zoom's max_iter at k < last zoom, else 0).

While held the transition is PENDING; at every decision of the lane (``_choose``) and at every re-trigger
(another converged verdict) the test is re-run on the packets arrived so far:
    caught up (T_k >= T_{k-1} - eps)   -> release ('caught_up'): the held transition is executed now
    ``catchup_max`` extra lines sent   -> release ('max'): executed now
    activation budget would bind       -> release ('budget_guard'): executed now
    no packet at one of the two levels -> release ('no_data')
    else                               -> keep holding: one more REGULAR zoom-k line in the same box, ext[k]
                                          raised just enough that the budget path does not fire.
A held transition is executed with the reason it was held with on the decision-tick release (CU 256-262);
on a re-trigger release it runs with the re-triggering call's reason (CU 311-313). Catch-up never ends a
dive itself: no zone, no dive end, no new activation.

Changes against the sandbox (DESIGN 1.3.3):
  * parameters from ``self.opts`` (``catchup_zooms`` parsed once per lane, ``catchup_eps``, ``catchup_max``)
    instead of the closure over ``ctx.params`` (CU 127-129);
  * ``ctx.count(key)`` -> ``engine.count("catchup." + key)``; ``ctx.event(self.s, "hold"|"release", ...)``
    -> ``engine.event(kind="x_catchup_hold"|"x_catchup_release", ...)`` with the identical payload (the
    explicit lane / act / zoom of CU 214, 249); ``ctx.LZ.in_box`` -> ``src.lanes.geometry.in_box``;
  * ``_cu_max_iter`` / ``_cu_act_budget`` call the lane's ``_hp_z(k)`` / ``_act_budget()`` directly (the
    per-zoom budget is always present; the ``getattr`` fallbacks of CU 163-171 go);
  * ``_cu_m()`` = ``opts.catchup_top_m`` or ``TOP_M.get(d, TOP_M_DEFAULT)`` (CU 88-89, 137);
  * the ``_predict_next_box`` override (CU 329-334) is gone (bridges are not ported);
  * attribute / helper names ``_x_cu*`` -> ``_cu*`` (``_cu_held``, ``_cu_extra``, ``_cu_bypass``). The lane
    method names ``_choose``, ``_zoom_in``, ``_try_needle`` and their ``super()`` calls are unchanged, so
    ``LeaderLane(CatchupMixin, DiveLane)`` dispatches exactly like the sandbox MRO.

No import-time side effects.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.lanes.events import EV_CU_HOLD, EV_CU_RELEASE
from src.lanes.geometry import in_box

__all__ = ["TOP_M", "TOP_M_DEFAULT", "WHATS", "parse_zooms", "top_level", "dive_tops", "CatchupMixin",
           "catchup_report", "catchup_top_m"]

TOP_M = {6: 8, 10: 11}          # report/metrics2.TOP_M (the box rule's top_m_points by dimension)
TOP_M_DEFAULT = 8               # metrics2: TOP_M.get(dim, 8) for any other dimension
WHATS = ("zoom_in", "needle")


# ── pure helpers (CU 93-123 verbatim) ───────────────────────────────────────────────────────────

def parse_zooms(s) -> Tuple[int, ...]:
    z = str(s).strip()
    if not z or any(c not in "12" for c in z):
        raise ValueError(f"catchup: zooms must be digits from '12' (k >= 1: T_k vs T_(k-1)), got {s!r}")
    return tuple(sorted({int(c) for c in z}))


def top_level(ys, m: int) -> float:
    """metrics2.level(.., 'topm', m): mean of the m largest values (all of them when fewer); nan if none."""
    v = sorted((float(y) for y in ys), reverse=True)[:max(1, int(m))]
    return float(sum(v) / len(v)) if v else float("nan")


def dive_tops(packets, lane: str, act: int, m: int) -> Dict[int, Tuple[float, int, int]]:
    """{sent zoom: (T, droplets, packets)} over the ARRIVED packets of one dive (engine.packets: the
    measured y of each packet, attributed lane / act / sent zoom; oracle attribution = metrics2's view)."""
    by: Dict[int, list] = {}
    for p in packets:
        if p.get("lane") != lane or p.get("act") != act:
            continue
        z = p.get("zoom")
        if z is None or int(z) < 0:
            continue
        d = by.setdefault(int(z), [[], 0])
        d[0].extend(p.get("y") or [])
        d[1] += 1
    return {z: (top_level(ys, m), len(ys), n) for z, (ys, n) in by.items()}


def _r(v, nd=5):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


def catchup_top_m(opts, d: int) -> int:
    """The top-m of the catch-up level: ``opts.catchup_top_m`` or 8 (6-D) / 11 (10-D) / 8 (CU 88-89, 137)."""
    return int(getattr(opts, "catchup_top_m", 0) or TOP_M.get(int(d), TOP_M_DEFAULT))


# ── the lane mixin (CU 131-327) ─────────────────────────────────────────────────────────────────

class CatchupMixin:
    """Catch-up layer above ``DiveLane`` (see the module docstring). Needs on the lane: ``s`` (the engine:
    ``packets``, ``d``, ``count``, ``event``, ``z``, ``dh``, ``rows``), ``name``, ``act``, ``zoom``, ``sends``,
    ``results``, ``ext``, ``lines_act``, ``lines``, ``n_dec``, ``opts``, ``box``, ``_hp()``, ``_hp_z(k)``,
    ``_act_budget()``, ``own_act_mask()``, ``_unpen()``, ``_end_with_zone``, and the next class's ``_choose``,
    ``_zoom_in``, ``_try_needle``."""

    _cu_held = None              # the held transition of this lane: dict(act, k, what, reason, ...)
    _cu_bypass = False           # True while a released transition is being executed

    # ── parameters (parsed once per lane) ──
    def _cu_cfg(self) -> tuple:
        c = self.__dict__.get("_cu_cfg_cache")
        if c is None:
            o = self.opts
            c = (parse_zooms(o.catchup_zooms), float(o.catchup_eps), int(o.catchup_max))
            self.__dict__["_cu_cfg_cache"] = c
        return c

    # ── measured levels ──
    def _cu_m(self) -> int:
        return catchup_top_m(self.opts, int(self.s.d))

    def _cu_gap(self, k: int) -> Optional[tuple]:
        """(T_k, T_(k-1), droplets_k, droplets_(k-1)) of this dive, or None when either is missing."""
        tops = dive_tops(self.s.packets, self.name, int(self.act), self._cu_m())
        a, b = tops.get(int(k)), tops.get(int(k) - 1)
        if a is None or b is None or math.isnan(a[0]) or math.isnan(b[0]):
            return None
        return a[0], b[0], a[1], b[1]

    def _cu_deficit(self, g) -> bool:
        eps = self._cu_cfg()[1]
        return g is not None and g[0] < g[1] - eps

    # ── extra lines (main lines of this activation sent at zoom k while held; refreshed-away ones do
    #    not count) ──
    def _cu_ids(self, k: int, act: Optional[int] = None) -> List[int]:
        book = self.__dict__.setdefault("_cu_extra", {})
        return book.setdefault((int(self.act if act is None else act), int(k)), [])

    def _cu_n_extra(self, k: int, act: Optional[int] = None) -> int:
        return sum(1 for lid in self._cu_ids(k, act)
                   if lid in self.lines and self.lines[lid].status != "replaced")

    # ── budgets (the lane's per-zoom values) ──
    def _cu_max_iter(self, k: int) -> int:
        return int(self._hp_z(k)["max_iter"])

    def _cu_act_budget(self) -> int:
        return int(self._act_budget())

    def _cu_budget_ok(self, k: int) -> bool:
        reserve = self._cu_max_iter(k + 1) if k < int(self._hp()["max_zooms"]) - 1 else 0
        return self.lines_act + 1 + reserve < self._cu_act_budget()

    def _cu_allow_send(self, k: int) -> None:
        """Raise ext[k] just enough that one more zoom-k send fits (the budget path stays quiet)."""
        need = self.sends[k] + 1 - (self._cu_max_iter(k) + self.ext[k])
        if need > 0:
            self.ext[k] += need
            if self._cu_held is not None:
                self._cu_held["ext_added"] += need
            self.s.count("catchup.ext_added", need)

    # ── hold / keep / release ──
    def _cu_enter(self, now, what: str, reason: str) -> bool:
        """At an exit of zoom k: True = hold (the caller must not execute the transition)."""
        zooms, _eps, max_x = self._cu_cfg()
        k = int(self.zoom)
        if k not in zooms or max_x <= 0:
            return False
        h = self._cu_held
        if h is not None:
            if h["act"] == self.act and h["k"] == k:
                return self._cu_keep(now)               # a re-trigger while held (e.g. another verdict)
            self._cu_release(now, "moot")               # state of an earlier dive (never executed)
        g = self._cu_gap(k)
        if g is None:
            self.s.count(f"catchup.pass_no_data_z{k}")
            return False
        if not self._cu_deficit(g):
            self.s.count(f"catchup.pass_ok_z{k}")
            return False
        n_x = self._cu_n_extra(k)
        if n_x >= max_x:
            self.s.count(f"catchup.pass_max_z{k}")
            return False
        if not self._cu_budget_ok(k):
            self.s.count(f"catchup.pass_budget_z{k}")
            return False
        self._cu_held = dict(act=int(self.act), k=k, what=what, reason=str(reason), t=now, ext_added=0,
                             decision=int(self.n_dec), T_k=g[0], T_prev=g[1])
        self.s.count(f"catchup.hold_{what}_z{k}")
        # CU 214: ctx.event(self.s, "hold", lane=..., act=..., zoom=..., ...) (explicit lane / act / zoom)
        self.s.event(kind=EV_CU_HOLD, lane=self.name, act=int(self.act), zoom=k, what=what, reason=str(reason),
                     T_k=_r(g[0]), T_prev=_r(g[1]), gap=_r(g[0] - g[1]), n_k=int(g[2]), n_prev=int(g[3]),
                     n_extra=int(n_x), sends=int(self.sends[k]), results=int(self.results[k]),
                     ext=int(self.ext[k]), lines_act=int(self.lines_act), t=now)
        self._cu_allow_send(k)
        return True

    def _cu_keep(self, now) -> bool:
        """Held: True = keep holding (one more zoom-k line allowed); False = released (state cleared,
        the caller executes the transition)."""
        max_x = self._cu_cfg()[2]
        h = self._cu_held
        k = h["k"]
        g = self._cu_gap(k)
        why = None
        if g is None:
            why = "no_data"
        elif not self._cu_deficit(g):
            why = "caught_up"
        elif self._cu_n_extra(k) >= max_x:
            why = "max"
        elif not self._cu_budget_ok(k):
            why = "budget_guard"
        if why is None:
            self._cu_allow_send(k)
            return True
        self._cu_release(now, why, g)
        return False

    def _cu_release(self, now, why: str, g=None) -> None:
        h, self._cu_held = self._cu_held, None
        if h is None:
            return
        if g is None and h["act"] == self.act:
            g = self._cu_gap(h["k"])
        self.s.count(f"catchup.release_{why}")
        # CU 249: ctx.event(self.s, "release", lane=..., act=..., zoom=..., ...)
        self.s.event(kind=EV_CU_RELEASE, lane=self.name, act=int(h["act"]), zoom=int(h["k"]), why=why,
                     what=h["what"], reason=h["reason"],
                     T_k=_r(g[0]) if g else None, T_prev=_r(g[1]) if g else None,
                     gap=_r(g[0] - g[1]) if g else None, T_k_at_hold=_r(h["T_k"]),
                     n_extra=int(self._cu_n_extra(h["k"], h["act"])), ext_added=int(h["ext_added"]),
                     held_decisions=int(self.n_dec - h["decision"]), t=now)

    def _cu_execute(self, h: dict, now) -> None:
        """Execute a released transition (through the whole MRO, without re-holding)."""
        self._cu_bypass = True
        try:
            if h["what"] == "zoom_in":
                self._zoom_in(now, h["reason"])
            else:
                st = self._try_needle(now)
                self.s.count(f"catchup.needle_after_release_{st}")
                if st == "no_points":                     # as the verdict does with this status
                    self._end_with_zone(self.box, "needle_no_points", now)
        finally:
            self._cu_bypass = False

    def _cu_tick(self, now) -> None:
        h = self._cu_held
        if h is None:
            return
        if h["act"] != self.act or int(self.zoom) != h["k"]:
            self._cu_release(now, "moot")               # the dive ended / moved on for another reason
            return
        if self._cu_keep(now):
            self.s.count("catchup.held_decisions")
            return
        self._cu_execute(h, now)

    def _cu_would_declare(self) -> bool:
        """Side-effect-free copy of the stock repeat test of DiveLane._try_needle (LZ 1073-1093): own
        unpenalised rows counted at this zoom inside the box, >= needle_min_repeats others within the
        repeat radius of their best row."""
        import torch
        z, dh, R = self.s.z, self.s.dh, self.s.rows
        k = int(self.zoom)
        X, Yall = dh.X_all_actual, dh.Y_all.reshape(-1)
        own = self.own_act_mask() & (R.t("eff") == k) & self._unpen()
        cand = torch.where(own & in_box(X, self.box))[0]
        if cand.numel() == 0:
            return False
        best = cand[int(Yall[cand].argmax())]
        near = cand[torch.norm(X[cand] - X[best].unsqueeze(0), dim=1) <= z._repeat_radius(self.box)]
        return int(near.numel()) - 1 >= int(z.needle_min_repeats)

    # ── the lane functions ──
    def _choose(self, now, kind, reuse_candidate=None):
        self._cu_tick(now)
        main, backup = super()._choose(now, kind, reuse_candidate=reuse_candidate)
        h = self._cu_held
        if (h is not None and main.act == self.act == h["act"] and int(main.zoom) == h["k"]
                and not main.bridge):
            self._cu_ids(h["k"]).append(int(main.line_id))
            self.s.count(f"catchup.extra_lines_z{h['k']}")
        return main, backup

    def _zoom_in(self, now, reason: str):
        if not self._cu_bypass and self._cu_enter(now, "zoom_in", reason):
            return None
        return super()._zoom_in(now, reason)

    def _try_needle(self, now) -> str:
        zooms = self._cu_cfg()[0]
        if not self._cu_bypass and int(self.zoom) in zooms:
            h = self._cu_held
            held_here = h is not None and h["act"] == self.act and h["k"] == int(self.zoom)
            # a new hold only where the stock repeat rule would declare (= the lane would leave);
            # a re-trigger while held re-tests the held transition
            if (held_here or self._cu_would_declare()) and self._cu_enter(now, "needle", "needle"):
                return "catchup_hold"                    # the verdict ends nothing on this status
            if held_here and self._cu_held is None:       # released by this re-trigger: the held needle
                st = super()._try_needle(now)             # (counted as on the decision-tick release path)
                self.s.count(f"catchup.needle_after_release_{st}")
                return st
        return super()._try_needle(now)


# ── report (CU 339-373) ─────────────────────────────────────────────────────────────────────────

def catchup_report(store) -> dict:
    """Holds / releases and, per dive, the measured T0 / T1 / T2 at the end of the campaign (the sandbox's
    ``summary.explore.finish.catchup``, same shape)."""
    m = catchup_top_m(store.opts, int(store.d))
    holds = [e for e in store.events if e.get("kind") == EV_CU_HOLD]
    rels = [e for e in store.events if e.get("kind") == EV_CU_RELEASE]
    why: Dict[str, int] = {}
    for e in rels:
        why[str(e.get("why"))] = why.get(str(e.get("why")), 0) + 1
    held = {}
    for e in holds:
        held.setdefault((e.get("lane"), int(e.get("act"))), set()).add(int(e.get("zoom")))
    extra: Dict[tuple, int] = {}
    for ln in store.lanes.values():
        for (act, k), ids in (ln.__dict__.get("_cu_extra") or {}).items():
            n = sum(1 for lid in ids if lid in ln.lines and ln.lines[lid].status != "replaced")
            if n:
                extra[(ln.name, int(act), int(k))] = n
    dives = []
    keys = sorted({(p.get("lane"), int(p.get("act"))) for p in store.packets
                   if p.get("lane") is not None and p.get("act") is not None and int(p.get("act")) >= 0},
                  key=lambda t: (str(t[0]), t[1]))
    n3 = ok = 0
    for lane, act in keys:
        tops = dive_tops(store.packets, lane, act, m)
        T = [tops.get(z, (float("nan"), 0, 0))[0] for z in (0, 1, 2)]
        rec = dict(dive=f"{lane}:{act}", T=[_r(t) for t in T], lines=[tops.get(z, (0, 0, 0))[2] for z in (0, 1, 2)],
                   held=sorted(held.get((lane, act), ())),
                   extra=[extra.get((lane, act, z), 0) for z in (0, 1, 2)])
        if not any(math.isnan(t) for t in T):
            n3 += 1
            rec["nodip"] = bool(T[1] >= T[0] - 0.01 - 1e-12 and T[2] >= T[1] - 0.01 - 1e-12)
            ok += int(rec["nodip"])
        dives.append(rec)
    return dict(m=m, holds=len(holds), held_dives=len(held), releases=why, extra_lines=int(sum(extra.values())),
                nodip_3zoom=dict(n=n3, ok=ok, pct=(_r(100.0 * ok / n3, 2) if n3 else None)), dives=dives)
