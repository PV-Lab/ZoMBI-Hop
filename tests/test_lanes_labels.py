"""src/lanes/labels.py: snapshot labels, the sidecar schema and the snapshot-count warning (DESIGN.md 6.6,
critic C4, REVIEW F10).

* ``packet_label(act=7, zoom=1, lane="front", pid=12) == "act7_z1_F_p12"``; from an engine packet record too;
* packet and decision labels match the GUI's / plot's ``_ACT_RE`` / ``_ACT_ZOOM_RE`` (``search`` on the
  snapshot directory name; the regexes are read from interface/app.py and visualization/plot_convergence.py
  without importing those modules) and retro's ``_ACT_LABEL`` (``match`` on the label);
* ``shared_p12`` / ``init_p3`` match none of them;
* ``LABEL_RE`` / ``parse_label`` / ``format_label`` round-trip;
* sidecar JSON round-trip and validation;
* the F10 warning thresholds.

Pure; writes nothing. REVIEW F9 autouse fixture as in every lanes test module.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from src.lanes import labels as L

REPO = Path(__file__).resolve().parents[1]


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


def _regex_from_source(path: Path, name: str) -> re.Pattern:
    """``<name> = re.compile(r"...")`` read from a source file (no import of GUI / plotting modules)."""
    text = path.read_text(encoding="utf-8")
    m = re.search(rf'^{name}\s*=\s*re\.compile\(r"([^"]+)"\)', text, re.M)
    assert m, f"{name} not found in {path}"
    return re.compile(m.group(1))


def _readers():
    from src.core.retro import _ACT_LABEL
    out = dict(retro=_ACT_LABEL)
    for rel in ("interface/app.py", "visualization/plot_convergence.py"):
        p = REPO / rel
        if p.exists():
            out[f"{rel}:_ACT_RE"] = _regex_from_source(p, "_ACT_RE")
            out[f"{rel}:_ACT_ZOOM_RE"] = _regex_from_source(p, "_ACT_ZOOM_RE")
    return out


# ── labels ───────────────────────────────────────────────────────────────────────────────────────

def test_packet_label_basic():
    assert L.packet_label(act=7, zoom=1, lane="front", pid=12) == "act7_z1_F_p12"
    assert L.packet_label(act=0, zoom=2, lane="back", pid=0) == "act0_z2_B_p0"
    assert L.packet_label(act=3, zoom=0, lane="B", pid=4) == "act3_z0_B_p4"


def test_packet_label_from_engine_record():
    # an attributed packet: the LINE's act and SENT zoom (not the lane's current state)
    rec = dict(pid=12, lane="front", act=7, zoom=1, kind="regular", cls="closed_act")
    assert L.packet_label(rec) == "act7_z1_F_p12"
    # numpy / torch scalars as the engine may hold them
    rec_np = dict(pid=np.int64(5), lane="back", act=np.int64(2), zoom=torch.tensor(2), kind="refresh")
    assert L.packet_label(rec_np) == "act2_z2_B_p5"
    # unattributed (no owner) and unknown line (owner but act / zoom -1)
    assert L.packet_label(dict(pid=12, lane=None, act=-1, zoom=-1, kind="unattributed")) == "shared_p12"
    assert L.packet_label(dict(pid=9, lane="front", act=-1, zoom=-1, kind="regular", cls="unknown_line")) \
        == "shared_p9"
    # init packets
    assert L.packet_label(dict(pid=3, lane=None, act=-1, zoom=-1, kind="init", cls="init")) == "init_p3"
    assert L.packet_label(dict(pid=3, lane=None, act=-1, zoom=-1, kind=None, cls="init")) == "init_p3"
    # keywords win over the record
    assert L.packet_label(rec, pid=13) == "act7_z1_F_p13"
    with pytest.raises(ValueError):
        L.packet_label(dict(lane="front", act=1, zoom=0))          # no pid


def test_decision_label():
    assert L.decision_label(7, 1, "front", 4) == "act7_z1_F_d4"
    assert L.decision_label(act=2, zoom=0, lane="back", n=0) == "act2_z0_B_d0"
    for bad in (dict(act=-1, zoom=0, lane="front", n=0), dict(act=1, zoom=0, lane="side", n=0),
                dict(act=1, zoom=0, lane="front", n=None)):
        with pytest.raises(ValueError):
            L.decision_label(**bad)


def test_labels_parse_with_existing_readers():
    readers = _readers()
    assert len(readers) == 5, sorted(readers)       # retro + two regexes in each of app.py / plot_convergence.py
    for label, act, zoom in (("act7_z1_F_p12", 7, 1), ("act7_z1_F_d4", 7, 1), ("act12_z2_B_p305", 12, 2)):
        sdir = L.snapshot_dir_name(42, label)
        assert sdir == f"0042_{label}"
        m = readers["retro"].match(label)
        assert m and int(m.group(1)) == act
        for key, rx in readers.items():
            if key == "retro":
                continue
            m = rx.search(sdir)
            assert m, (key, sdir)
            assert int(m.group(1)) == act
            if key.endswith("_ACT_ZOOM_RE"):
                assert int(m.group(2)) == zoom
    for label in ("shared_p12", "init_p3"):
        assert readers["retro"].match(label) is None
        for key, rx in readers.items():
            if key != "retro":
                assert rx.search(L.snapshot_dir_name(7, label)) is None, (key, label)


def test_label_re_round_trip():
    cases = {
        "act7_z1_F_p12": dict(kind="packet", act=7, zoom=1, lane="front", pid=12),
        "act0_z2_B_d3": dict(kind="decision", act=0, zoom=2, lane="back", n=3),
        "shared_p12": dict(kind="shared", pid=12),
        "init_p3": dict(kind="init", pid=3),
    }
    for label, parsed in cases.items():
        assert L.LABEL_RE.match(label)
        assert L.parse_label(label) == parsed
        assert L.format_label(parsed) == label
        assert L.parse_label(L.snapshot_dir_name(17, label)) == parsed      # with the NNNN_ prefix
    for bad in ("act7_z1_X_p12", "F_act3_z1_p2", "act7_z1_F_q1", "shared_d1", "", "0005"):
        assert L.parse_label(bad) is None, bad
        assert L.LABEL_RE.match(bad) is None
    assert L.LABEL_RE.pattern == r"^(?:act(\d+)_z(\d+)_([FB])_([pd])(\d+)|(shared|init)_p(\d+))$"


def test_snapshot_dir_name_matches_datahandler():
    assert L.snapshot_dir_name(1, "act0_z0_F_p0") == "0001_act0_z0_F_p0"
    assert L.snapshot_dir_name(12) == "0012"
    assert L.snapshot_dir_name(10000, "x") == "10000_x"


# ── sidecars ─────────────────────────────────────────────────────────────────────────────────────

def test_packet_sidecar_round_trip():
    rec = dict(pid=np.int64(12), key="row345", t=np.float64(3600.5), lane="front", act=7, zoom=1, line=31,
               rid="F31", row0=np.int64(340), n=5, kind="regular", cls="own_current", y=[0.1] * 5)
    side = L.packet_sidecar(rec, wire_id=17, router_cls="matched")
    assert tuple(side) == L.PACKET_SIDECAR_KEYS
    assert side["schema"] == L.SIDECAR_SCHEMA == 1 and side["kind"] == "packet"
    assert side["pid"] == 12 and type(side["pid"]) is int and side["row0"] == 340 and side["t"] == 3600.5
    text = L.sidecar_to_json(side)
    back = L.sidecar_from_json(text)
    assert back == side
    assert json.loads(text)["wire_id"] == 17
    # unattributed packet: lane None, act / zoom -1, no wire id
    side2 = L.packet_sidecar(dict(pid=1, key="k", t=None, lane=None, act=-1, zoom=-1, line=None, rid=None,
                                  row0=0, n=2))
    assert L.sidecar_from_json(L.sidecar_to_json(side2)) == side2
    assert side2["wire_id"] is None and side2["router_cls"] is None


def test_decision_sidecar_round_trip():
    side = L.decision_sidecar(lane="back", act=3, zoom=2, n=9, decision_kind="regular",
                              events=["zoom_in", "x_catchup_hold"])
    assert tuple(side) == L.DECISION_SIDECAR_KEYS
    assert L.sidecar_from_json(L.sidecar_to_json(side)) == side
    assert side["events"] == ["zoom_in", "x_catchup_hold"]


def test_sidecar_validation():
    with pytest.raises(ValueError):
        L.sidecar_from_json(json.dumps(dict(schema=2, kind="packet")))
    with pytest.raises(ValueError):
        L.sidecar_from_json(json.dumps(dict(schema=1, kind="other")))
    with pytest.raises(ValueError):
        L.sidecar_from_json(json.dumps(dict(schema=1, kind="decision", lane="front")))
    with pytest.raises(ValueError):
        L.sidecar_from_json("[1, 2]")
    assert L.SIDE_PACKET == "lanes.json" and L.SIDE_STATE == "lanes_state.json"


# ── REVIEW F10 ───────────────────────────────────────────────────────────────────────────────────

def test_snapshot_limit_warning():
    assert L.SNAPSHOT_WARN_AT == 9000 and L.SNAPSHOT_SORT_LIMIT == 9999
    assert L.snapshot_limit_warning(0) is None
    assert L.snapshot_limit_warning(8999) is None
    w = L.snapshot_limit_warning(9000)
    assert w and w.startswith("WARNING") and "999 left" in w
    assert L.snapshot_limit_warning(9999).startswith("WARNING")
    e = L.snapshot_limit_warning(10000)
    assert e and e.startswith("ERROR")
    # the reason: name sort puts 10000_* before 2000_*
    assert sorted([L.snapshot_dir_name(2000, "a"), L.snapshot_dir_name(10000, "b")])[0].startswith("10000")
