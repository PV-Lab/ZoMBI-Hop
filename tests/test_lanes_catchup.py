"""Catch-up layer of the lanes engine (src/lanes/catchup.py) on stub lanes (DESIGN.md section 5, T3).

``StubLane(CatchupMixin, StubBase)``: StubBase records the ``_zoom_in`` / ``_try_needle`` / ``_choose`` calls the
mixin lets through and carries exactly what the mixin reads (the engine's ``packets / d / count / event``, the
lane's ``act / zoom / sends / results / ext / lines_act / lines / n_dec / opts`` and ``_hp / _hp_z /
_act_budget``). No GPU work: the repeat test of the needle hold is stubbed (``_cu_would_declare``).

Measured levels: m = 8 in 6-D (CU 88-89). Levels are set by packets whose y are all one value, so T_k is
that value. Hold rule: T_k < T_(k-1) - eps (eps 0.01), max 2 extra lines, the activation budget guard
lines_act + 1 + reserve < 24 (reserve = the next zoom's max_iter, 5).
"""
from __future__ import annotations

import math

import pytest
import torch

import src.core.linebo as _linebo
import src.core.zombihop as _zh
import src.utils.domain as _domain
from src.lanes.budget import activation_budget, zbudget_hp
from src.lanes.catchup import (TOP_M, TOP_M_DEFAULT, CatchupMixin, catchup_report, catchup_top_m, dive_tops,
                               parse_zooms, top_level)
from src.lanes.events import EV_CU_HOLD, EV_CU_RELEASE
from src.lanes.options import LaneOptions

_MODS = (_domain, _zh, _linebo)


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


# ── stubs ────────────────────────────────────────────────────────────────────────────────────────

class StubEngine:
    def __init__(self, d=6, opts=None):
        self.packets = []
        self.d = d
        self.stats = {}
        self.events = []
        self.opts = opts or LaneOptions()
        self.lanes = {}

    def count(self, key, n=1):
        self.stats[key] = self.stats.get(key, 0) + n

    def event(self, **ev):
        self.events.append(ev)
        return ev


class StubLine:
    def __init__(self, line_id, act, zoom):
        self.line_id, self.act, self.zoom = line_id, act, zoom
        self.bridge = False
        self.status = "queued"


class StubBase:
    """What DiveLane offers the mixin, with recorded transitions."""
    STOCK = dict(max_zooms=3, max_iter=5, min_iters=2, n_consec=2, min_zoom_needle=2)

    def __init__(self, s, name="front", act=0, zoom=1, sends=5, lines_act=10):
        self.s = s
        self.name = name
        self.opts = s.opts
        self.act, self.zoom = act, zoom
        self.sends = [0, 0, 0, 0]
        self.results = [0, 0, 0, 0]
        self.ext = [0, 0, 0, 0]
        self.sends[zoom] = sends
        self.results[zoom] = 2
        self.lines_act = lines_act
        self.lines = {}
        self.n_dec = 10
        self.box = "BOX"
        self.calls = []
        self.needle_status = "ok"
        self.would_declare = True
        self._lid = 100
        s.lanes[name] = self

    def _hp(self):
        return dict(self.STOCK)

    def _hp_z(self, k):
        return zbudget_hp(self._hp(), int(k), self.opts.zbudget_params())

    def _act_budget(self):
        return activation_budget([self._hp_z(j) for j in range(3)], self.opts.ext_cap)

    # the base transitions (DiveLane's) the mixin calls through super()
    def _zoom_in(self, now, reason):
        self.calls.append(("zoom_in", now, reason))
        self.zoom += 1

    def _try_needle(self, now):
        self.calls.append(("try_needle", now))
        return self.needle_status

    def _choose(self, now, kind, reuse_candidate=None):
        self.calls.append(("choose", now, kind))
        self._lid += 1
        main = StubLine(self._lid, self.act, self.zoom)
        self._lid += 1
        backup = StubLine(self._lid, self.act, self.zoom)
        self.lines[main.line_id] = main
        self.lines[backup.line_id] = backup
        self.sends[self.zoom] += 1
        self.lines_act += 1
        self.n_dec += 1
        return main, backup

    def _end_with_zone(self, box, reason, now):
        self.calls.append(("zone", now, reason))


class StubLane(CatchupMixin, StubBase):
    def _cu_would_declare(self):                  # the repeat rule of _try_needle, stubbed
        return self.would_declare


def _pkt(s, lane="front", act=0, zoom=0, y=1.0, n=8):
    s.packets.append(dict(lane=lane, act=act, zoom=zoom, y=[float(y)] * n))


def _kinds(s, kind):
    return [e for e in s.events if e["kind"] == kind]


def _transitions(lane):
    return [c for c in lane.calls if c[0] != "choose"]


def _deficit_lane(zoom=1, **kw):
    """A lane at ``zoom`` whose level there (0.90) is below the level one zoom up (1.00)."""
    s = StubEngine()
    ln = StubLane(s, zoom=zoom, **kw)
    _pkt(s, zoom=zoom - 1, y=1.00)
    _pkt(s, zoom=zoom, y=0.90)
    return s, ln


# ── pure helpers ─────────────────────────────────────────────────────────────────────────────────

def test_top_level():
    assert top_level([1.0, 3.0, 2.0], 2) == pytest.approx(2.5)
    assert top_level([1.0, 3.0, 2.0], 10) == pytest.approx(2.0)          # fewer than m: all of them
    assert top_level([1.0, 3.0], 0) == pytest.approx(3.0)                # m >= 1
    assert math.isnan(top_level([], 8))


def test_dive_tops_filters_lane_act_and_sent_zoom():
    P = [dict(lane="front", act=0, zoom=0, y=[1.0, 2.0]),
         dict(lane="front", act=0, zoom=1, y=[5.0]),
         dict(lane="front", act=0, zoom=1, y=[3.0, 4.0]),
         dict(lane="back", act=0, zoom=1, y=[99.0]),                     # other lane
         dict(lane="front", act=1, zoom=1, y=[99.0]),                    # other activation
         dict(lane="front", act=0, zoom=-1, y=[99.0]),                   # unattributed zoom
         dict(lane="front", act=0, zoom=None, y=[99.0]),
         dict(lane="front", act=0, zoom=2, y=[])]                        # a packet without novel rows
    t = dive_tops(P, "front", 0, 2)
    assert set(t) == {0, 1, 2}
    assert t[0] == (pytest.approx(1.5), 2, 1)
    assert t[1] == (pytest.approx(4.5), 3, 2)
    assert math.isnan(t[2][0]) and t[2][1:] == (0, 1)
    assert dive_tops(P, "back", 7, 2) == {}


def test_parse_zooms_and_top_m():
    assert parse_zooms("21") == (1, 2) and parse_zooms("1") == (1,)
    for bad in ("", "0", "3", "1a"):
        with pytest.raises(ValueError):
            parse_zooms(bad)
    o = LaneOptions()
    assert catchup_top_m(o, 6) == 8 == TOP_M[6] and catchup_top_m(o, 10) == 11 and catchup_top_m(o, 8) == TOP_M_DEFAULT
    assert catchup_top_m(LaneOptions.from_hparams({"lanes_catchup_top_m": 3}), 6) == 3
    s = StubEngine(d=10)
    assert StubLane(s)._cu_m() == 11


# ── holds and releases ───────────────────────────────────────────────────────────────────────────

def test_hold_at_1_to_2_raises_ext_by_the_missing_send():
    s, ln = _deficit_lane(zoom=1, sends=5)
    out = ln._zoom_in(100.0, "converged")
    assert out is None
    assert _transitions(ln) == []                             # the zoom-in is NOT executed
    assert ln.zoom == 1
    assert ln.ext[1] == 1                                     # sends 5 + 1 - (max_iter 5 + ext 0)
    assert s.stats["catchup.hold_zoom_in_z1"] == 1
    assert s.stats["catchup.ext_added"] == 1
    [ev] = _kinds(s, EV_CU_HOLD)
    assert (ev["lane"], ev["act"], ev["zoom"], ev["what"], ev["reason"]) == ("front", 0, 1, "zoom_in", "converged")
    assert ev["T_k"] == pytest.approx(0.9) and ev["T_prev"] == pytest.approx(1.0)
    assert ev["gap"] == pytest.approx(-0.1) and ev["n_k"] == 8 and ev["n_prev"] == 8
    assert (ev["n_extra"], ev["sends"], ev["results"], ev["ext"], ev["lines_act"], ev["t"]) == (0, 5, 2, 0, 10, 100.0)
    # payload key order = the sandbox's ctx.event call (CU 214) with the explicit lane / act / zoom first
    assert list(ev)[:4] == ["kind", "lane", "act", "zoom"]
    assert ln._cu_held["ext_added"] == 1


def test_no_ext_when_sends_still_fit():
    s, ln = _deficit_lane(zoom=1, sends=2)
    assert ln._zoom_in(1.0, "converged") is None
    assert ln.ext[1] == 0 and "catchup.ext_added" not in s.stats


def test_extra_line_recorded_on_choose():
    s, ln = _deficit_lane(zoom=1)
    ln._zoom_in(1.0, "converged")
    main, _ = ln._choose(2.0, "regular")
    assert ln._cu_ids(1) == [main.line_id]
    assert ln._cu_n_extra(1) == 1
    assert s.stats["catchup.extra_lines_z1"] == 1
    assert s.stats["catchup.held_decisions"] == 1
    assert _transitions(ln) == []


def test_release_caught_up_executes_held_zoom_in_once_with_held_reason():
    s, ln = _deficit_lane(zoom=1)
    ln._zoom_in(1.0, "iterations")
    ln._choose(2.0, "regular")
    _pkt(s, zoom=1, y=1.5)                                   # zoom 1 catches up
    ln._choose(3.0, "regular")
    assert _transitions(ln) == [("zoom_in", 3.0, "iterations")]
    [rel] = _kinds(s, EV_CU_RELEASE)
    assert (rel["why"], rel["what"], rel["reason"], rel["zoom"], rel["t"]) == ("caught_up", "zoom_in", "iterations", 1, 3.0)
    assert rel["n_extra"] == 1 and rel["held_decisions"] == 1 and rel["ext_added"] == 1
    assert ln._cu_held is None and ln._cu_bypass is False
    assert s.stats["catchup.release_caught_up"] == 1
    ln._choose(4.0, "regular")                               # nothing held any more
    assert len(_transitions(ln)) == 1


def test_release_max_after_two_extra_mains_refreshed_away_not_counted():
    s, ln = _deficit_lane(zoom=1)
    ln._zoom_in(1.0, "converged")
    m1, _ = ln._choose(2.0, "regular")
    m1.status = "replaced"                                   # refreshed away: not an extra line
    assert ln._cu_n_extra(1) == 0
    ln._choose(3.0, "regular")
    ln._choose(4.0, "regular")
    assert ln._cu_n_extra(1) == 2 and _transitions(ln) == []
    ln._choose(5.0, "regular")
    assert _transitions(ln) == [("zoom_in", 5.0, "converged")]
    assert [e["why"] for e in _kinds(s, EV_CU_RELEASE)] == ["max"]


def test_release_budget_guard():
    # lines_act 17: hold allowed (17 + 1 + 5 < 24); after one extra line 18 + 1 + 5 = 24 -> release
    s, ln = _deficit_lane(zoom=1, lines_act=17)
    assert ln._zoom_in(1.0, "converged") is None
    ln._choose(2.0, "regular")
    ln._choose(3.0, "regular")
    assert _transitions(ln) == [("zoom_in", 3.0, "converged")]
    assert [e["why"] for e in _kinds(s, EV_CU_RELEASE)] == ["budget_guard"]


def test_no_hold_when_budget_has_no_room():
    s, ln = _deficit_lane(zoom=1, lines_act=18)               # 18 + 1 + 5 = 24: no room
    ln._zoom_in(1.0, "converged")
    assert _transitions(ln) == [("zoom_in", 1.0, "converged")]
    assert s.stats == {"catchup.pass_budget_z1": 1} and not s.events


def test_moot_when_act_or_zoom_changed():
    s, ln = _deficit_lane(zoom=1)
    ln._zoom_in(1.0, "converged")
    ln.act = 1                                               # the dive ended for another reason
    ln._choose(2.0, "regular")
    assert _transitions(ln) == []                            # the held transition is never executed
    [rel] = _kinds(s, EV_CU_RELEASE)
    assert rel["why"] == "moot" and rel["act"] == 0
    assert ln._cu_held is None


def test_never_a_hold_at_zoom_0():
    s, ln = _deficit_lane(zoom=1)
    ln.zoom = 0
    s.packets.clear()
    _pkt(s, zoom=0, y=0.5)
    ln._zoom_in(1.0, "converged")
    assert _transitions(ln) == [("zoom_in", 1.0, "converged")]
    assert s.stats == {} and s.events == []


def test_no_data_pass_when_a_level_has_no_packet():
    s = StubEngine()
    ln = StubLane(s, zoom=1)
    _pkt(s, zoom=1, y=0.5)                                   # no zoom-0 packet of this dive
    _pkt(s, lane="back", zoom=0, y=9.0)                      # another lane's does not count
    ln._zoom_in(1.0, "converged")
    assert _transitions(ln) == [("zoom_in", 1.0, "converged")]
    assert s.stats == {"catchup.pass_no_data_z1": 1}


def test_pass_ok_when_no_deficit():
    s = StubEngine()
    ln = StubLane(s, zoom=1)
    _pkt(s, zoom=0, y=1.00)
    _pkt(s, zoom=1, y=0.995)                                 # within eps
    ln._zoom_in(1.0, "converged")
    assert _transitions(ln) == [("zoom_in", 1.0, "converged")]
    assert s.stats == {"catchup.pass_ok_z1": 1}


def test_catchup_max_0_never_holds():
    s = StubEngine(opts=LaneOptions.from_hparams({"lanes_catchup_max": 0}))
    ln = StubLane(s, zoom=1)
    _pkt(s, zoom=0, y=1.0)
    _pkt(s, zoom=1, y=0.5)
    ln._zoom_in(1.0, "converged")
    assert _transitions(ln) == [("zoom_in", 1.0, "converged")] and s.stats == {}


def test_needle_hold_only_when_the_repeat_rule_would_declare():
    s, ln = _deficit_lane(zoom=2)
    ln.would_declare = False
    assert ln._try_needle(1.0) == "ok"                       # no hold: the base needle try runs
    assert _transitions(ln) == [("try_needle", 1.0)] and not s.events
    ln.calls.clear()
    ln.would_declare = True
    assert ln._try_needle(2.0) == "catchup_hold"
    assert _transitions(ln) == []
    [ev] = _kinds(s, EV_CU_HOLD)
    assert (ev["what"], ev["reason"], ev["zoom"]) == ("needle", "needle", 2)
    assert s.stats["catchup.hold_needle_z2"] == 1


def test_held_needle_released_on_decision_tick():
    s, ln = _deficit_lane(zoom=2)
    assert ln._try_needle(1.0) == "catchup_hold"
    ln._choose(2.0, "regular")
    _pkt(s, zoom=2, y=1.2)
    ln._choose(3.0, "regular")
    assert _transitions(ln) == [("try_needle", 3.0)]
    assert s.stats["catchup.needle_after_release_ok"] == 1
    # no_points on a release: the zone the verdict would make (CU 265-266)
    s2, ln2 = _deficit_lane(zoom=2)
    ln2.needle_status = "no_points"
    ln2._try_needle(1.0)
    _pkt(s2, zoom=2, y=1.2)
    ln2._choose(2.0, "regular")
    assert _transitions(ln2) == [("try_needle", 2.0), ("zone", 2.0, "needle_no_points")]


def test_retrigger_while_held_retests_and_releases_with_the_new_reason():
    # CU 311-313: a re-trigger release runs with the re-triggering call's reason (sandbox.md 7.2)
    s, ln = _deficit_lane(zoom=1)
    ln._zoom_in(1.0, "iterations")
    assert ln._zoom_in(2.0, "converged") is None             # still behind: keep holding
    assert _transitions(ln) == [] and len(_kinds(s, EV_CU_HOLD)) == 1
    _pkt(s, zoom=1, y=2.0)
    ln._zoom_in(3.0, "converged")                            # caught up: released by the re-trigger
    assert _transitions(ln) == [("zoom_in", 3.0, "converged")]
    assert [e["why"] for e in _kinds(s, EV_CU_RELEASE)] == ["caught_up"]


def test_bypass_while_executing_does_not_rehold():
    s, ln = _deficit_lane(zoom=1)
    h = dict(act=0, k=1, what="zoom_in", reason="converged")
    ln._cu_execute(h, 5.0)                                   # deficit still there, but bypassed
    assert _transitions(ln) == [("zoom_in", 5.0, "converged")] and not s.events


def test_catchup_report_shape():
    s, ln = _deficit_lane(zoom=1)
    ln._zoom_in(1.0, "converged")
    ln._choose(2.0, "regular")
    _pkt(s, zoom=2, y=1.1)
    rep = catchup_report(s)
    assert rep["m"] == 8 and rep["holds"] == 1 and rep["held_dives"] == 1 and rep["releases"] == {}
    assert rep["extra_lines"] == 1
    [dv] = rep["dives"]
    assert dv["dive"] == "front:0" and dv["held"] == [1] and dv["extra"] == [0, 1, 0]
    assert dv["T"] == [1.0, 0.9, 1.1] and dv["nodip"] is False
    assert rep["nodip_3zoom"] == dict(n=1, ok=0, pct=0.0)
