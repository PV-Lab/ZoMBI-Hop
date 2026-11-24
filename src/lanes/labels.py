"""Snapshot labels and per-snapshot sidecar schema of a lanes run (DESIGN.md section 6.6, N6; critic C4).

A lanes engine takes one DataHandler snapshot per ingested packet (any class) and one zero-row snapshot
per committed decision. DataHandler names the directory ``f"{count:04d}_{label}"`` (datahandler.py 372).

    snapshot                                     label                         rows
    attributed packet (any class)                act{A}_z{Z}_{F|B}_p{pid}      the packet's novel rows
        A / Z = the LINE's activation and SENT zoom, F/B = the owner lane
    unattributed / unknown line / legacy / id    shared_p{pid}                 rows
    late init packet                             init_p{pid}                   rows
    decision commit                              act{A}_z{Z}_{F|B}_d{n}        0
        A / Z = the lane's state AFTER the decision, n = the lane's decision index

Every attributed label STARTS with ``act{A}_z{Z}`` so the readers that already exist parse it unchanged:
retro's ``^act(\\d+)`` with ``match`` on the summary label (src/core/retro.py 53), the GUI's and the plot's
``_act(\\d+)`` / ``_act(\\d+)_z(\\d+)`` with ``search`` on the directory name (interface/app.py 662-663,
visualization/plot_convergence.py 53-54). ``shared_*`` / ``init_*`` labels match none of them.

Sidecars (written next to the snapshot by the Phase-2 commit protocol, 6.5):
    lanes.json        packet:   {schema, kind:"packet", pid, row0, n, lane, act, zoom, line, rid, wire_id,
                                 router_cls, t, key}
                      decision: {schema, kind:"decision", lane, act, zoom, n, decision_kind, events:[kinds]}
    lanes_state.json  full engine state (src/lanes/state.py, Phase 2)
Readers rebuild per-row lane / act / sent zoom by replaying the snapshot directories in order.

Snapshot-count limit (REVIEW F10): DataHandler, the GUI and the plot sort snapshot directories by NAME
(datahandler.py 649, 729; app.py 722; plot_convergence.py 184), so ``10000_*`` sorts before ``2000_*``.
That is a pre-existing limit of 9999 snapshots per run; a lanes run takes about two per result (~100-130
a day), i.e. it is reached after ~80-100 days. ``src/`` outside ``src/lanes`` stays frozen (DESIGN N2), so
the driver warns from ``SNAPSHOT_WARN_AT`` snapshots on (``snapshot_limit_warning``).

Pure module: json / re only, no torch use, no side effects.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

# ── lanes ────────────────────────────────────────────────────────────────────────────────────────
LANE_TAG = {"front": "F", "back": "B"}          # = src.lanes.geometry.LANE_TAG (LZ 51)
TAG_LANE = {v: k for k, v in LANE_TAG.items()}

# ── labels ───────────────────────────────────────────────────────────────────────────────────────
LABEL_PATTERN = r"^(?:act(\d+)_z(\d+)_([FB])_([pd])(\d+)|(shared|init)_p(\d+))$"
LABEL_RE = re.compile(LABEL_PATTERN)

# ── sidecars ─────────────────────────────────────────────────────────────────────────────────────
SIDE_PACKET = "lanes.json"
SIDE_STATE = "lanes_state.json"
SIDECAR_SCHEMA = 1
PACKET_SIDECAR_KEYS = ("schema", "kind", "pid", "row0", "n", "lane", "act", "zoom", "line", "rid", "wire_id",
                       "router_cls", "t", "key")
DECISION_SIDECAR_KEYS = ("schema", "kind", "lane", "act", "zoom", "n", "decision_kind", "events")

# ── snapshot naming limit (REVIEW F10) ───────────────────────────────────────────────────────────
SNAPSHOT_NAME_WIDTH = 4         # datahandler.py 372: f"{count:04d}_{label}"
SNAPSHOT_SORT_LIMIT = 9999      # name sort of snapshot dirs breaks at 10000
SNAPSHOT_WARN_AT = 9000


def lane_tag(lane: Any) -> Optional[str]:
    """'F' / 'B' for a lane name ('front' / 'back') or tag ('F' / 'B'); None for anything else."""
    if lane is None:
        return None
    s = str(lane)
    if s in LANE_TAG:
        return LANE_TAG[s]
    if s in TAG_LANE:
        return s
    return None


def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def shared_label(pid: int) -> str:
    """Label of an unattributed / unknown-line / legacy / unknown-id packet."""
    return f"shared_p{int(pid)}"


def init_label(pid: int) -> str:
    """Label of a (late) init packet."""
    return f"init_p{int(pid)}"


def packet_label(rec: Optional[Mapping] = None, *, act: Any = None, zoom: Any = None, lane: Any = None,
                 pid: Any = None, kind: Any = None) -> str:
    """Snapshot label of one ingested packet.

    Called as ``packet_label(rec)`` with the engine's packet record (keys ``pid``, ``lane``, ``act``,
    ``zoom`` = the line's SENT zoom, ``kind``, ``cls``) or with keywords (keywords win over ``rec``).
    ``init_p{pid}`` for a packet tagged ``kind="init"`` (or classed ``init``); ``act{A}_z{Z}_{F|B}_p{pid}``
    when the packet belongs to a known line of a lane (act >= 0, zoom >= 0); ``shared_p{pid}`` otherwise
    (unattributed, unknown line)."""
    if rec is not None:
        act = rec.get("act") if act is None else act
        zoom = rec.get("zoom") if zoom is None else zoom
        lane = rec.get("lane") if lane is None else lane
        pid = rec.get("pid") if pid is None else pid
        if kind is None:
            kind = "init" if (rec.get("kind") == "init" or rec.get("cls") == "init") else rec.get("kind")
    p = _int_or_none(pid)
    if p is None or p < 0:
        raise ValueError(f"packet_label: pid must be an int >= 0, got {pid!r}")
    if kind == "init":
        return init_label(p)
    tag, a, z = lane_tag(lane), _int_or_none(act), _int_or_none(zoom)
    if tag is None or a is None or z is None or a < 0 or z < 0:
        return shared_label(p)
    return f"act{a}_z{z}_{tag}_p{p}"


def decision_label(act: Any, zoom: Any, lane: Any, n: Any) -> str:
    """Snapshot label of a committed decision: the lane's act / zoom AFTER the decision, its decision
    index ``n``."""
    tag, a, z, k = lane_tag(lane), _int_or_none(act), _int_or_none(zoom), _int_or_none(n)
    if tag is None or a is None or z is None or k is None or min(a, z, k) < 0:
        raise ValueError(f"decision_label: need lane front/back and act, zoom, n >= 0; got "
                         f"lane={lane!r} act={act!r} zoom={zoom!r} n={n!r}")
    return f"act{a}_z{z}_{tag}_d{k}"


def parse_label(label: str) -> Optional[dict]:
    """``{"kind": "packet"|"decision"|"shared"|"init", ...}`` for a lanes label (with or without the
    ``NNNN_`` snapshot-directory prefix), else None.
    packet: act, zoom, lane, pid; decision: act, zoom, lane, n; shared / init: pid."""
    s = str(label)
    m = LABEL_RE.match(s)
    if m is None:
        pre = re.match(r"^\d+_(.+)$", s)
        m = LABEL_RE.match(pre.group(1)) if pre else None
        if m is None:
            return None
    if m.group(1) is not None:
        out = dict(kind="packet" if m.group(4) == "p" else "decision", act=int(m.group(1)), zoom=int(m.group(2)),
                   lane=TAG_LANE[m.group(3)])
        out["pid" if m.group(4) == "p" else "n"] = int(m.group(5))
        return out
    return dict(kind=m.group(6), pid=int(m.group(7)))


def format_label(parsed: Mapping) -> str:
    """Inverse of ``parse_label``."""
    k = parsed["kind"]
    if k == "packet":
        return packet_label(act=parsed["act"], zoom=parsed["zoom"], lane=parsed["lane"], pid=parsed["pid"])
    if k == "decision":
        return decision_label(parsed["act"], parsed["zoom"], parsed["lane"], parsed["n"])
    if k == "shared":
        return shared_label(parsed["pid"])
    if k == "init":
        return init_label(parsed["pid"])
    raise ValueError(f"format_label: unknown kind {k!r}")


def snapshot_dir_name(count: int, label: str = "") -> str:
    """The directory name DataHandler gives snapshot number ``count`` (datahandler.py 372)."""
    return f"{int(count):04d}_{label}" if label else f"{int(count):04d}"


def snapshot_limit_warning(count: int) -> Optional[str]:
    """None below ``SNAPSHOT_WARN_AT``; else a message for the run log / GUI (REVIEW F10). ``count`` is the
    number of snapshots the run has (``dh._snapshot_count``)."""
    n = int(count)
    if n < SNAPSHOT_WARN_AT:
        return None
    if n <= SNAPSHOT_SORT_LIMIT:
        return (f"WARNING: {n} snapshots in this run; snapshot directories are sorted by name and the "
                f"{SNAPSHOT_NAME_WIDTH}-digit numbering ends at {SNAPSHOT_SORT_LIMIT} ({SNAPSHOT_SORT_LIMIT - n} "
                f"left): finish or archive the run before then")
    return (f"ERROR: {n} snapshots in this run (> {SNAPSHOT_SORT_LIMIT}); snapshot directories numbered "
            f">= 10000 sort before the earlier ones, so resume, the GUI and the plots read them out of order")


# ── sidecars ─────────────────────────────────────────────────────────────────────────────────────

def _jsonable(v: Any) -> Any:
    """Plain JSON value (numpy / torch scalars -> Python; NaN / inf -> None)."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v) if math.isfinite(v) else None
    item = getattr(v, "item", None)
    if callable(item) and getattr(v, "ndim", 0) == 0:
        return _jsonable(item())
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def packet_sidecar(rec: Mapping, *, wire_id: Any = None, router_cls: Optional[str] = None) -> dict:
    """The ``lanes.json`` sidecar of one packet snapshot, from the engine's packet record (row0, n, lane,
    act, zoom = SENT zoom, line, rid, t, key) plus the wire id / router class (hardware; None in the
    emulator)."""
    out = dict(schema=SIDECAR_SCHEMA, kind="packet", pid=rec["pid"], row0=rec["row0"], n=rec["n"],
               lane=rec.get("lane"), act=rec.get("act"), zoom=rec.get("zoom"), line=rec.get("line"),
               rid=rec.get("rid"), wire_id=wire_id, router_cls=router_cls, t=rec.get("t"), key=rec.get("key"))
    return {k: _jsonable(out[k]) for k in PACKET_SIDECAR_KEYS}


def decision_sidecar(*, lane: str, act: int, zoom: int, n: int, decision_kind: str,
                     events: Sequence[str] = ()) -> dict:
    """The ``lanes.json`` sidecar of one decision snapshot (``events``: the kinds of the store events the
    decision emitted)."""
    out = dict(schema=SIDECAR_SCHEMA, kind="decision", lane=lane, act=act, zoom=zoom, n=n,
               decision_kind=decision_kind, events=[str(e) for e in events])
    return {k: _jsonable(out[k]) for k in DECISION_SIDECAR_KEYS}


def validate_sidecar(obj: Mapping) -> dict:
    """Check schema, kind and keys of a sidecar dict; returns it as a plain dict. Raises ValueError."""
    if not isinstance(obj, Mapping):
        raise ValueError("lanes sidecar must be a JSON object")
    if obj.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"lanes sidecar schema {obj.get('schema')!r} != {SIDECAR_SCHEMA}")
    keys = {"packet": PACKET_SIDECAR_KEYS, "decision": DECISION_SIDECAR_KEYS}.get(obj.get("kind"))
    if keys is None:
        raise ValueError(f"lanes sidecar kind {obj.get('kind')!r} is not 'packet' or 'decision'")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"lanes {obj['kind']} sidecar misses {missing}")
    return dict(obj)


def sidecar_to_json(obj: Mapping) -> str:
    return json.dumps(validate_sidecar(obj), allow_nan=False)


def sidecar_from_json(text: str) -> dict:
    return validate_sidecar(json.loads(text))


__all__ = ["LANE_TAG", "TAG_LANE", "LABEL_PATTERN", "LABEL_RE", "SIDE_PACKET", "SIDE_STATE", "SIDECAR_SCHEMA",
           "PACKET_SIDECAR_KEYS", "DECISION_SIDECAR_KEYS", "SNAPSHOT_NAME_WIDTH", "SNAPSHOT_SORT_LIMIT",
           "SNAPSHOT_WARN_AT", "lane_tag", "shared_label", "init_label", "packet_label", "decision_label",
           "parse_label", "format_label", "snapshot_dir_name", "snapshot_limit_warning", "packet_sidecar",
           "decision_sidecar", "validate_sidecar", "sidecar_to_json", "sidecar_from_json"]
