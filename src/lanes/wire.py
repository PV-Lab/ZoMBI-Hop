"""
src/lanes/wire.py - result-id wire codec and ZoMBI-side recipe bookkeeping for the lanes protocol.

Production copy of the sandbox reference ``benchmarks/zoom_fix/lanes/result_id/wire.py`` (cited below as
``W <line>``), moved per DESIGN.md section 6.2 with the reviewer corrections R2 and R4 (DESIGN.md
"## Reviewer corrections") and REVIEW.md F14. The codec functions are copied verbatim, so the bytes on the
cable are the ones the sandbox tests and the hardware cable test (``result_id/hw_test``, shared block)
produce for the same inputs. Tests: ``tests/test_lanes_wire.py`` and the sandbox's own ``test_wire.py`` /
``test_router.py`` run against this module by ``benchmarks/zoom_fix/lanes/port/test_prod_wire.py``.

Import hygiene (REVIEW F12): this file imports only the standard library and has no import-time side
effects. ``numpy`` is imported lazily, inside ``grid24`` only. Importing it as ``src.lanes.wire`` runs
``src/__init__.py`` (torch + ZoMBIHop); a process that must stay torch-free (the serial child) can load
this file by path with ``importlib.util.spec_from_file_location``.

Framing (unchanged from today, W 10-16)
---------------------------------------
* One JSON object per line, ``json.dumps(pkt) + "\\n"``, UTF-8, Python's default separators
  (", " and ": "), floats in Python repr (shortest round-trip, up to 17 significant digits).
* 115200 baud, 8N1 (pyserial defaults): 10 bit-times per byte -> 11 520 bytes/s.
* No checksum, no per-packet ACK. Both senders re-send their current packet every second
  (1 Hz); the receivers de-duplicate by the packet's ``ts``.

Packets (all new keys OPTIONAL and appended last, W 18-39)
---------------------------------------------------------
DiSCO -> ZoMBI, objective packet::

    {"type": "objective", "values": [...], "comps": [[...]], "ts": T, "r": 417, "k": [418, 352]}

    r : recipe id of THIS slide (the id ZoMBI gave the recipe the rail read), int 0..999.
    k : read acknowledgement, [last id read by the front rail, last id read by the back rail]
        (null for a rail that has not read since DiSCO started).

ZoMBI -> DiSCO, four-slot lanes packet (full state, no grids)::

    {"type": "composition", "ts": T, "s": [[slot, id, start10, end10], ...]}

    slot: 0 front main, 1 front backup, 2 back main, 3 back backup. A slot that is absent is EMPTY
    (withdraw an unread recipe). DiSCO replaces a slot only when its id changed. Optional "m": 1 =
    start/end are integer thousandths.

Id ranges (R2, DESIGN 6.2 / 6.10 / 7.5)
---------------------------------------
* On the wire an id is any integer 0..999 (``check_id``).
* A run issues 0..899 (``RecipeIds()`` default: ``base=0, modulus=RUN_ID_MODULUS=900``).
* 900..999 are reserved for Manual Control (``RecipeIds.for_manual()``). A run's ``ResultRouter`` classes a
  result whose ``r`` is in that range as ``manual`` BEFORE looking the id up: logged, never ingested, never
  routed to a line. A router refuses an allocator whose range overlaps its manual range.

Persistence (R4)
----------------
``RecipeIds.on_change`` / ``ResultRouter.on_change`` are called synchronously after every state change
(``try_issue`` BEFORE it returns the new id, i.e. before the caller writes the slot carrying it; acks;
routes that change state: matched, late, null, reprint, FIFO / timeout closures, supersede, DiSCO restart).
``ResultRouter.persist_to(path, saver)`` installs a hook that saves ``to_json(now)`` (e.g. ``lanes/ids.json``
with ``persist.save_json``). ``ResultRouter.pick_newer`` chooses between ``ids.json`` and the copy embedded in
an engine commit; ``RecipeIds.recover`` rebuilds a counter from ``sent.jsonl`` when ``ids.json`` is missing.

Not moved / changed relative to W
---------------------------------
* ``save_json`` (W 335-346) is kept only as the DEFAULT saver of ``ResultRouter.save``; the Phase-2 driver
  passes ``persist.save_json`` (DESIGN 6.2 / 6.5).
* ``RecipeIds.issue`` still raises ``RuntimeError`` when no id is free (sandbox semantics, its tests rely
  on it); drivers call ``try_issue``, which never raises (DESIGN 6.12).
* ``added_bytes_objective`` (W 617-619) is kept for the size tables.
"""
from __future__ import annotations

import json
import logging
import math  # noqa: F401  (W 44; kept so the module's public surface matches the sandbox)
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

log = logging.getLogger(__name__)

BAUD = 115200
BITS_PER_BYTE = 10            # 8N1: start + 8 data + stop
ID_MODULUS = 1000             # recipe ids on the wire: 0..999 (<= 3 ASCII digits)
REUSE_HORIZON_S = 24 * 3600   # an id is never re-issued within 24 h of its last use
LOST_AFTER_READ_S = 90 * 60   # read -> result was <= 78 min on the hardware (p90 61-63)
MIN_READ_TO_RESULT_S = 30 * 60  # print + anneal + HS + probe + Prism: a real result cannot come sooner

# R2 / DESIGN 6.2, 6.10, 7.5: a run issues 0..899; 900..999 belong to Manual Control.
RUN_ID_BASE = 0
RUN_ID_MODULUS = 900
MANUAL_ID_BASE = 900
MANUAL_ID_MODULUS = ID_MODULUS - MANUAL_ID_BASE      # 100
RUN_ID_RANGE = (RUN_ID_BASE, RUN_ID_BASE + RUN_ID_MODULUS)          # [0, 900)
MANUAL_ID_RANGE = (MANUAL_ID_BASE, MANUAL_ID_BASE + MANUAL_ID_MODULUS)   # [900, 1000)

# DESIGN 6.12: exhaustion fallback. First a normal retry after prune, then the reuse horizon is shortened to
# 12 h, and superseded recipes that were never acked as read for 12 h are closed as unprinted.
FALLBACK_HORIZON_S = 12 * 3600
EVENTS_KEEP = 5000            # router events kept in memory
EVENTS_PERSIST = 200          # router events tail saved with the state (R4: "the events tail")

SLOT_NAMES = ("front_main", "front_backup", "back_main", "back_backup")
SLOT_RAIL = ("front", "front", "back", "back")
RAILS = ("front", "back")
MAIN_SLOT = {"front": 0, "back": 2}
BACKUP_SLOT = {"front": 1, "back": 3}
# The controller's slot keys (protocols.py `_put4`: "<lane>-main" / "<lane>-cache") -> wire slot index.
SLOT_OF_KEY = {"front-main": 0, "front-cache": 1, "back-main": 2, "back-cache": 3}

# Recipe.line (DESIGN 6.2): the int on the wire -> the line it stands for.
LINE_KINDS = ("lane", "init")
LINE_FIELDS = ("kind", "lane", "rid", "hrid", "role", "act", "zoom", "trace", "send")


# ── framing (W 60-95, verbatim) ─────────────────────────────────────────────────────
def dumps_line(pkt: dict) -> bytes:
    """Exactly how both programs frame a packet today."""
    return (json.dumps(pkt) + "\n").encode("utf-8")


def loads_line(line) -> Optional[dict]:
    if isinstance(line, (bytes, bytearray)):
        line = bytes(line).decode("utf-8", errors="ignore")
    line = line.strip()
    if not line:
        return None
    try:
        pkt = json.loads(line)
    except json.JSONDecodeError:
        return None
    return pkt if isinstance(pkt, dict) else None


def wire_seconds(nbytes: int, baud: int = BAUD) -> float:
    return nbytes * BITS_PER_BYTE / float(baud)


def _int_or_none(x):
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (isinstance(x, float) and not x.is_integer()):
        raise ValueError(f"id must be an integer, got {x!r}")
    return int(x)


def check_id(rid) -> int:
    rid = _int_or_none(rid)
    if rid is None or not (0 <= rid < ID_MODULUS):
        raise ValueError(f"recipe id must be 0..{ID_MODULUS - 1}, got {rid!r}")
    return rid


def is_manual_id(rid, manual_ids: Optional[Tuple[int, int]] = MANUAL_ID_RANGE) -> bool:
    """R2: an id in Manual Control's range (900..999 by default)."""
    if rid is None or manual_ids is None:
        return False
    return int(manual_ids[0]) <= int(rid) < int(manual_ids[1])


# ── DiSCO -> ZoMBI: objective packet (W 98-144, verbatim) ───────────────────────────
def encode_objective(values: Sequence[float], comps: Sequence[Sequence[float]], ts,
                     rid: Optional[int] = None,
                     last_read: Optional[Tuple[Optional[int], Optional[int]]] = None) -> bytes:
    """DiSCO objective_sender's packet (communication.py:2936) plus the optional r / k keys.
    With rid=None and last_read=None the bytes are identical to today's packet."""
    pkt = {"type": "objective", "values": list(values), "comps": [list(r) for r in comps], "ts": ts}
    if rid is not None:
        pkt["r"] = check_id(rid)
    if last_read is not None:
        f, b = last_read
        pkt["k"] = [None if f is None else check_id(f), None if b is None else check_id(b)]
    return dumps_line(pkt)


def decode_objective(line) -> Optional[dict]:
    """New receiver. Returns None for a non-objective line. A packet without "values" or with an
    empty "values" list is returned too (an optional NULL-slide notice carries only r)."""
    pkt = loads_line(line)
    if pkt is None or pkt.get("type") != "objective" or "values" not in pkt:
        return None
    rid = pkt.get("r")
    k = pkt.get("k")
    try:
        rid = None if rid is None else check_id(rid)
    except ValueError:
        rid = None                                   # a malformed id never blocks the data
    last_read = None
    if isinstance(k, list) and len(k) == 2:
        try:
            last_read = tuple(None if x is None else check_id(x) for x in k)
        except ValueError:
            last_read = None
    return dict(values=pkt.get("values"), comps=pkt.get("comps"), ts=pkt.get("ts"),
                rid=rid, last_read=last_read)


def legacy_objective_view(line) -> Optional[dict]:
    """What TODAY's ZoMBI objective_receiver (scripts/communication.py:562-818) reads from a line:
    type, values, comps, ts - nothing else. Used to prove the new keys are ignored."""
    pkt = loads_line(line)
    if pkt is None or pkt.get("type") != "objective" or "values" not in pkt:
        return None
    vals = pkt["values"]
    if not isinstance(vals, list) or len(vals) == 0:
        return None                                   # today: size 0 -> "Unexpected values shape", ignored
    return dict(values=vals, comps=pkt.get("comps", None), ts=pkt.get("ts", None))


# ── ZoMBI -> DiSCO: two-slot transitional packet (W 147-190, verbatim; reference only) ──────
# Not used by any production path in v1 (legacy mode sends today's packet unchanged; lanes4 sends the
# four-slot packet). Kept because the compatibility checks of the sandbox's test_wire.py exercise it.
_LEGACY_KEYS = ("type", "ts", "start", "end", "comps", "start_cache", "end_cache", "compositions_cache")


def encode_composition_legacy(ts, start, end, comps, start_cache, end_cache, comps_cache,
                              ids: Optional[Tuple[int, int]] = None) -> bytes:
    """ZoMBI composition_sender's packet (scripts/communication.py:888-897), key order unchanged,
    plus the optional "r": [id_main, id_cache]."""
    pkt = {"type": "composition", "ts": ts, "start": list(start), "end": list(end),
           "comps": [list(r) for r in comps], "start_cache": list(start_cache), "end_cache": list(end_cache),
           "compositions_cache": [list(r) for r in comps_cache]}
    if ids is not None:
        pkt["r"] = [check_id(ids[0]), check_id(ids[1])]
    return dumps_line(pkt)


def decode_composition_legacy(line) -> Optional[dict]:
    pkt = loads_line(line)
    if pkt is None or pkt.get("type") != "composition" or "start" not in pkt:
        return None
    ids = pkt.get("r")
    try:
        ids = None if ids is None else (check_id(ids[0]), check_id(ids[1]))
    except (ValueError, TypeError, IndexError):
        ids = None
    out = {k: pkt.get(k) for k in _LEGACY_KEYS}
    out["ids"] = ids
    return out


def legacy_disco_accepts(line) -> bool:
    """TODAY's DiSCO composition_receiver shape check (communication.py:2457-2466): it reads only
    the named keys, so an extra "r" key passes untouched."""
    pkt = loads_line(line)
    if pkt is None or pkt.get("type") != "composition":
        return False
    ts, s, e, c = pkt.get("ts"), pkt.get("start"), pkt.get("end"), pkt.get("comps")
    sc, ec, cc = pkt.get("start_cache"), pkt.get("end_cache"), pkt.get("compositions_cache")
    return bool(
        isinstance(ts, (int, float)) and
        isinstance(s, list) and len(s) == 10 and isinstance(e, list) and len(e) == 10 and
        isinstance(c, list) and all(isinstance(r, list) and len(r) == 10 for r in c) and
        isinstance(sc, list) and len(sc) == 10 and isinstance(ec, list) and len(ec) == 10 and
        isinstance(cc, list) and all(isinstance(r, list) and len(r) == 10 for r in cc))


# ── ZoMBI -> DiSCO: four-slot lean packet (W 193-242) ──────────────────────────────
def milli_vector(x: Sequence[float]) -> List[int]:
    """ZoMBI's np.round(x, 3) as integer thousandths (np.round(x, 3) == rint(x*1000)/1000)."""
    return [int(round(float(v) * 1000.0)) for v in x]


def from_milli(m: Sequence[int]) -> List[float]:
    """Canonical normalisation of integer thousandths. Both sides must use THIS function
    (ZoMBI for its record of the printed endpoints, DiSCO before writing archerfish.db).
    The total is an exact integer sum, so the result does not depend on the Python version
    (hw_test plain_sum note: sum() of FLOATS changed in Python 3.12)."""
    tot = sum(int(v) for v in m)
    if tot <= 0:
        return [1.0 / len(m)] * len(m)
    return [int(v) / tot for v in m]


def encode_slots(ts, slots: Dict[int, Tuple[int, Sequence, Sequence]], milli: bool = False,
                 full: bool = True) -> bytes:
    """slots: {slot_index: (id, start10, end10)}. full=True: absent slot = EMPTY (recommended).
    full=False: a delta, absent slot = unchanged ("d": 1). (W 208-228, verbatim.)

    Byte-identical to the hw_test shared block's ``encode_slots(ts, slots4, milli)`` for the same
    entries (full=True): keys "type", "ts", "s" (sorted by slot) and "m": 1 when milli."""
    s = []
    for k in sorted(slots):
        if k not in (0, 1, 2, 3):
            raise ValueError(f"slot must be 0..3, got {k}")
        rid, a, b = slots[k]
        a, b = list(a), list(b)
        if len(a) != 10 or len(b) != 10:
            raise ValueError("start/end must have 10 components")
        if milli:
            a, b = [int(v) for v in a], [int(v) for v in b]
        s.append([k, check_id(rid), a, b])
    pkt = {"type": "composition", "ts": ts, "s": s}
    if milli:
        pkt["m"] = 1
    if not full:
        pkt["d"] = 1
    return dumps_line(pkt)


def _check_slot_values(vals, milli: bool) -> None:
    """The value checks of the hw_test shared block's decode_slots (what the cable-tested DiSCO side
    applies): 10 numbers, never bool, never negative, integers when milli."""
    if not (isinstance(vals, list) and len(vals) == 10):
        raise ValueError("start/end must have 10 values")
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or (milli and not isinstance(v, int)):
            raise ValueError(f"bad composition value {v!r}")


def decode_slots(line) -> Optional[dict]:
    """-> dict(ts, slots={slot: (id, start10, end10)}, full, milli), or None for a line that is not a
    four-slot composition packet. Raises ValueError on a malformed four-slot packet.

    W 231-242 plus the structural checks of the hw_test shared block's decode_slots (a bad ts, an entry
    that is not [slot, id, start, end], a slot outside 0..3 or repeated, bad values). The output for a
    well-formed packet is the sandbox's, unchanged."""
    pkt = loads_line(line)
    if pkt is None or pkt.get("type") != "composition" or "s" not in pkt:
        return None
    ts = pkt.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        raise ValueError("missing or bad ts")
    ents = pkt["s"]
    if not isinstance(ents, list):
        raise ValueError("missing slot list 's'")
    milli = pkt.get("m") == 1
    out = {}
    for ent in ents:
        if not (isinstance(ent, list) and len(ent) == 4):
            raise ValueError("slot entry is not [slot, id, start, end]")
        k, rid, a, b = ent
        if isinstance(k, bool) or k not in (0, 1, 2, 3) or int(k) in out:
            raise ValueError(f"bad slot number {k!r}")
        _check_slot_values(a, milli)
        _check_slot_values(b, milli)
        if milli:
            a, b = from_milli(a), from_milli(b)
        out[int(k)] = (check_id(rid), [float(v) for v in a], [float(v) for v in b])
    return dict(ts=ts, slots=out, full=pkt.get("d") != 1, milli=milli)


def grid24(start: Sequence[float], end: Sequence[float], n: int = 24) -> List[List[float]]:
    """Reference per-point array for DiSCO (DESIGN 7.3): ``n`` points ``normalize(round(start + t*(end-start),
    3))`` with ``t = linspace(0, 1, n)``; the arithmetic of ``_write_line_pair`` (RZM 858-866:
    ``np.linspace``, ``np.round(., 3)``, ``normalize_last_axis``), applied to the wire's 10-d endpoints."""
    import numpy as np                                # lazy: keeps the module stdlib-only at import
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    g = np.array([a + t * (b - a) for t in np.linspace(0, 1, n)])
    g = np.round(g, 3)
    sums = g.sum(axis=-1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    out = g / sums
    bad = ~np.isfinite(out).all(axis=-1, keepdims=True)
    if np.any(bad):
        out = np.where(bad, 1.0 / g.shape[-1], out)
    return out.tolist()


# ── DiSCO-side reference model of the four slots (W 245-311, verbatim) ─────────────
@dataclass
class SlotState:
    rid: Optional[int] = None
    start: Optional[list] = None
    end: Optional[list] = None
    state: str = "EMPTY"          # EMPTY | UNREAD | READ | CLEARED
    pending: Optional[tuple] = None   # replacement that arrived while the slot was READ (printing)


class DiscoSlots:
    """compositions.db per-slot rule: replace a slot only if its id changed; a READ slot is never
    replaced (it is printing); a CLEARED slot keeps its id so a re-send cannot resurrect it.
    Reference model for tests (FakeDiSCO4, R1); not used on the ZoMBI side at run time."""

    def __init__(self):
        self.slots = {k: SlotState() for k in range(4)}
        self.last_ts = None

    def apply(self, decoded: dict) -> Dict[int, str]:
        acts = {}
        if self.last_ts is not None and decoded["ts"] == self.last_ts:
            return {k: "same_ts" for k in range(4)}
        self.last_ts = decoded["ts"]
        for k in range(4):
            cur = self.slots[k]
            new = decoded["slots"].get(k)
            if new is None:
                if not decoded["full"]:
                    acts[k] = "unchanged"
                elif cur.state == "UNREAD":
                    self.slots[k] = SlotState(rid=cur.rid, state="EMPTY")
                    acts[k] = "withdrawn"
                else:
                    acts[k] = "keep"
                continue
            rid, a, b = new
            if cur.rid == rid:
                acts[k] = "keep"                       # same recipe: rows, read flag, start/end untouched
            elif cur.state == "READ":
                acts[k] = "keep_read"                  # printing: the replacement waits for the clear
                cur.pending = (rid, a, b)
            else:
                self.slots[k] = SlotState(rid=rid, start=a, end=b, state="UNREAD")
                acts[k] = "replace"
        return acts

    def read(self, rail: str):
        """The rail reads its main, else its backup. Returns (slot, rid, start, end) or None."""
        base = 0 if rail == "front" else 2
        for k in (base, base + 1):
            s = self.slots[k]
            if s.state == "UNREAD":
                s.state = "READ"
                return k, s.rid, s.start, s.end
        return None

    def clear(self, rail: str):
        """archerfish.py:234 after the print: READ -> CLEARED (the id is kept)."""
        base = 0 if rail == "front" else 2
        for k in (base, base + 1):
            s = self.slots[k]
            if s.state == "READ":
                pend = s.pending
                if pend is not None:
                    self.slots[k] = SlotState(rid=pend[0], start=pend[1], end=pend[2], state="UNREAD")
                else:
                    s.state = "CLEARED"


# ── persistence helpers ─────────────────────────────────────────────────────────────
def _jsonable(x):
    """Recursively convert numpy arrays / scalars / tuples to JSON types (lines are persisted). (W 324-332)"""
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def save_json(path, obj, tries: int = 8) -> bool:
    """In-place write with retries (the run directory lives in Dropbox: never os.replace). (W 335-346)

    Only the DEFAULT saver of ``ResultRouter.save`` / ``persist_to``. The Phase-2 driver passes
    ``src.lanes.persist.save_json`` (side file + retried replace, never raises; DESIGN 6.5) instead."""
    import time as _t
    txt = json.dumps(_jsonable(obj))
    for attempt in range(tries):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(txt)
            return True
        except OSError:
            _t.sleep(0.25 * (attempt + 1))
    return False


def make_line(kind: str, *, lane: Optional[str] = None, rid=None, hrid=None, role: Optional[str] = None,
              act: Optional[int] = None, zoom: Optional[int] = None, trace=None, send=None) -> dict:
    """``Recipe.line`` for production (DESIGN 6.2): ``{"kind": "lane"|"init", "lane", "rid" (engine LineRec rid,
    e.g. "F12"), "hrid" (controller rid, e.g. "F5" / "F5b"), "role", "act", "zoom", "trace", "send"}``.
    The wire carries only the int id; this dict is what the id stands for (persisted with the recipe)."""
    if kind not in LINE_KINDS:
        raise ValueError(f"line kind must be one of {LINE_KINDS}, got {kind!r}")
    if lane is not None and lane not in RAILS:
        raise ValueError(f"line lane must be one of {RAILS} or None, got {lane!r}")
    return dict(kind=kind, lane=lane, rid=rid, hrid=hrid, role=role, act=act, zoom=zoom, trace=trace, send=send)


def line_kind(line) -> Optional[str]:
    """The ``kind`` of a ``Recipe.line`` dict ("lane" / "init"), or None for anything else."""
    if isinstance(line, dict):
        return line.get("kind")
    return None


def _call_hook(owner, hook, reason: str, now) -> None:
    """Run a persistence hook. A hook never raises into the driver (DESIGN 6.12): the error is logged
    and counted on ``owner.hook_errors``."""
    if hook is None:
        return
    try:
        hook(reason, now)
    except Exception:                                   # noqa: BLE001 - persistence must not stop routing
        owner.hook_errors = getattr(owner, "hook_errors", 0) + 1
        log.exception("lanes wire: persistence hook failed (reason=%s)", reason)


# ── ZoMBI side: id allocation (W 314-443) ───────────────────────────────────────────
# Revised 2026-11-12 after VERIFY.md M1/M2 (RESULT_ID.md, "Revision after verification"):
#  * every issue / read ack / supersede gets a number from ONE monotone event counter
#    (``RecipeIds.clock``). The order of reads on a rail comes only from evidence (ack order, or "acked
#    before the other recipe was even issued"), never from the order in which ZoMBI learned of a read;
#  * the 30-min stale-id guard is keyed to the ISSUE time (a lower bound on the read time). The ack
#    time is only an upper bound on the read time, so the guard no longer uses it;
#  * the full state (counter, history, every recipe of the last 24 h with its line, status and row
#    hashes, the router's last acks) is persisted (``to_json`` / ``save``) and reloaded
#    (``from_json`` / ``load``).
@dataclass
class Recipe:
    rid: int
    slot: int
    rail: str                     # the rail that read it (from "k"); before any ack: the slot's rail
    line: object
    t_issue: float
    seq_issue: int = 0
    t_read: Optional[float] = None        # when ZoMBI first SAW the read ack: an UPPER bound on the read time
    seq_ack: Optional[int] = None         # its place in the event order (ack order = read order, per rail)
    t_superseded: Optional[float] = None
    seq_superseded: Optional[int] = None
    t_result: Optional[float] = None
    status: str = "issued"        # issued | read | resulted | lost | unprinted | null
    payload_hashes: list = field(default_factory=list)

    FIELDS = ("rid", "slot", "rail", "line", "t_issue", "seq_issue", "t_read", "seq_ack", "t_superseded",
              "seq_superseded", "t_result", "status", "payload_hashes")

    def to_json(self) -> dict:
        return {k: _jsonable(getattr(self, k)) for k in self.FIELDS}

    @classmethod
    def from_json(cls, d: dict) -> "Recipe":
        return cls(**{k: d[k] for k in cls.FIELDS if k in d})


OPEN = ("issued", "read")


class RecipeIds:
    """Rolling ids ``base .. base+modulus-1``. An id is never re-issued while its recipe is open or within
    the reuse horizon of its last use. ``to_json`` / ``from_json`` persist the whole state, so a ZoMBI
    restart resumes the counter, the open recipes WITH their lines, and the finished recipes of the last
    24 h with their row hashes (duplicate detection).

    Production defaults (R2): a run allocator issues 0..899 (``base=0, modulus=900``); Manual Control uses
    ``RecipeIds.for_manual()`` (900..999). With ``base=0`` the arithmetic is the sandbox's (W 385-412):
    ``next = start % modulus``.

    ``on_change(reason, now)`` (R4) is called after every issue, before the new id is returned."""

    def __init__(self, start: int = RUN_ID_BASE, modulus: int = RUN_ID_MODULUS,
                 horizon_s: float = REUSE_HORIZON_S, base: int = RUN_ID_BASE):
        base, modulus = int(base), int(modulus)
        if base < 0 or modulus < 1 or base + modulus > ID_MODULUS:
            raise ValueError(f"id range {base}..{base + modulus - 1} must lie within 0..{ID_MODULUS - 1}")
        self.base = base
        self.modulus, self.horizon = modulus, horizon_s
        self.next = base + (int(start) - base) % modulus
        self.by_id: Dict[int, Recipe] = {}
        self.history: Dict[int, float] = {}       # rid -> last time it was issued/used
        self.clock = 0                            # monotone event counter (issues, acks, supersedes)
        self.on_change: Optional[Callable[[str, float], None]] = None
        self.hook_errors = 0
        self.n_fallback = 0                       # ids issued only thanks to the 12 h fallback
        self.n_exhausted = 0                      # try_issue calls that returned None
        self.n_stale_closed = 0                   # superseded, never-read recipes closed by the fallback

    @classmethod
    def for_run(cls, start: int = RUN_ID_BASE, horizon_s: float = REUSE_HORIZON_S) -> "RecipeIds":
        """A run's allocator: ids 0..899 (R2)."""
        return cls(start=start, modulus=RUN_ID_MODULUS, horizon_s=horizon_s, base=RUN_ID_BASE)

    @classmethod
    def for_manual(cls, start: int = MANUAL_ID_BASE, horizon_s: float = REUSE_HORIZON_S) -> "RecipeIds":
        """Manual Control's allocator: ids 900..999 (DESIGN 6.10, persisted in sql/manual_ids.json)."""
        return cls(start=start, modulus=MANUAL_ID_MODULUS, horizon_s=horizon_s, base=MANUAL_ID_BASE)

    @property
    def id_range(self) -> Tuple[int, int]:
        """[lo, hi) of the ids this allocator issues."""
        return self.base, self.base + self.modulus

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def _changed(self, reason: str, now) -> None:
        _call_hook(self, self.on_change, reason, now)

    def _free(self, rid, now, horizon: Optional[float] = None):
        horizon = self.horizon if horizon is None else horizon
        r = self.by_id.get(rid)
        if r is not None and r.status in OPEN:
            return False
        last = self.history.get(rid)
        return last is None or now - last >= horizon

    def _issue(self, slot: int, line, now: float, horizon: Optional[float] = None) -> Optional[int]:
        """W 403-411 (the loop of ``issue``) with the horizon as a parameter; None when no id is free."""
        for _ in range(self.modulus):
            rid = self.next
            self.next = self.base + (self.next - self.base + 1) % self.modulus
            if self._free(rid, now, horizon):
                self.by_id[rid] = Recipe(rid=rid, slot=slot, rail=SLOT_RAIL[slot], line=line, t_issue=now,
                                         seq_issue=self.tick())
                self.history[rid] = now
                return rid
        return None

    def issue(self, slot: int, line, now: float) -> int:
        """Sandbox semantics (W 403-412): raises RuntimeError when no id is free. Drivers use ``try_issue``."""
        rid = self._issue(slot, line, now)
        if rid is None:
            raise RuntimeError("no free recipe id: more than ID_MODULUS ids used within the reuse horizon")
        self._changed("issue", now)
        return rid

    def _close_stale_superseded(self, now: float, age_s: float) -> int:
        """Exhaustion fallback only: a recipe that was superseded at least ``age_s`` ago and never acked as
        read is closed as ``unprinted`` (what the router's ``_close_superseded`` concludes once the rail's
        next read is acked; without acks such recipes would stay open for ever). A result that still
        arrives for it is routed as ``late`` while its id has not been re-issued."""
        n = 0
        for r in self.by_id.values():
            if (r.status == "issued" and r.seq_ack is None and r.t_superseded is not None
                    and now - r.t_superseded >= age_s):
                r.status = "unprinted"
                n += 1
        self.n_stale_closed += n
        return n

    def try_issue(self, slot: int, line, now: float) -> Optional[int]:
        """Never raises into the driver (DESIGN 6.2 / 6.12): issue; else prune and retry; else retry with the
        12 h horizon after closing superseded recipes never read for 12 h (WARNING); else None (CRITICAL).
        On success ``on_change("issue", now)`` runs BEFORE the id is returned (R4: saved before the slot that
        carries it is written)."""
        rid = self._issue(slot, line, now)
        if rid is None:
            self.prune(now)
            rid = self._issue(slot, line, now)
        changed = False
        if rid is None:
            fb = min(float(self.horizon), float(FALLBACK_HORIZON_S))
            changed = self._close_stale_superseded(now, fb) > 0
            rid = self._issue(slot, line, now, horizon=fb)
            if rid is not None:
                self.n_fallback += 1
                log.warning("lanes wire: recipe ids %d..%d exhausted within %.0f h; issued %d under the %.0f h "
                            "fallback horizon", self.base, self.base + self.modulus - 1, self.horizon / 3600.0,
                            rid, fb / 3600.0)
        if rid is None:
            self.n_exhausted += 1
            log.critical("lanes wire: NO free recipe id in %d..%d (all open or used within %.0f h); the slot "
                         "keeps its previous recipe", self.base, self.base + self.modulus - 1,
                         min(float(self.horizon), float(FALLBACK_HORIZON_S)) / 3600.0)
            if changed:
                self._changed("stale_closed", now)
            return None
        self._changed("issue", now)
        return rid

    def prune(self, now: float) -> int:
        """Forget closed recipes whose last use is older than the horizon (their history stays). (W 414-420)"""
        drop = [rid for rid, r in self.by_id.items()
                if r.status not in OPEN and now - self.history.get(rid, r.t_issue) >= self.horizon]
        for rid in drop:
            del self.by_id[rid]
        return len(drop)

    def to_json(self, now: Optional[float] = None) -> dict:
        if now is not None:
            self.prune(now)
        return dict(version=3, next=self.next, modulus=self.modulus, base=self.base, horizon=self.horizon,
                    clock=self.clock, history={str(k): v for k, v in self.history.items()},
                    recipes=[r.to_json() for r in self.by_id.values()],
                    n_fallback=self.n_fallback, n_exhausted=self.n_exhausted, n_stale_closed=self.n_stale_closed)

    @classmethod
    def from_json(cls, d: dict) -> "RecipeIds":
        """Loads version 1 (sandbox: open recipes only), 2 (sandbox) and 3 (this module: + base, counters).
        Files without ``base`` were written with base 0 (W 422-443)."""
        ids = cls(start=int(d.get("next", 0)), modulus=int(d.get("modulus", ID_MODULUS)),
                  horizon_s=float(d.get("horizon", REUSE_HORIZON_S)), base=int(d.get("base", 0)))
        ids.clock = int(d.get("clock", 0))
        ids.history = {int(k): float(v) for k, v in (d.get("history") or {}).items()}
        for rd in d.get("recipes") or []:
            r = Recipe.from_json(rd)
            ids.by_id[int(r.rid)] = r
        if int(d.get("version", 1)) < 2:          # a version-1 file: open recipes only, without lines
            for k, v in (d.get("open") or {}).items():
                ids.by_id[int(k)] = Recipe(rid=int(k), slot=int(v["slot"]), rail=SLOT_RAIL[int(v["slot"])],
                                           line=None, t_issue=float(v["t_issue"]), t_read=v.get("t_read"),
                                           status=v.get("status", "issued"))
        ids.n_fallback = int(d.get("n_fallback", 0))
        ids.n_exhausted = int(d.get("n_exhausted", 0))
        ids.n_stale_closed = int(d.get("n_stale_closed", 0))
        return ids

    @classmethod
    def recover(cls, used: Union[Dict[int, float], Iterable[int]], now: float, *, last: Optional[int] = None,
                base: int = RUN_ID_BASE, modulus: int = RUN_ID_MODULUS,
                horizon_s: float = REUSE_HORIZON_S) -> "RecipeIds":
        """R4 item 4: ``ids.json`` is missing on a resume. Rebuild a counter from the ids already sent
        (``sent.jsonl``): ``used`` maps id -> time of its last write, or is a plain iterable of ids (then each
        is taken as used ``now``, so none is re-issued within the horizon). ``next`` = 1 + ``last`` (the id
        written last), else 1 + max(used). Ids outside this allocator's range are ignored. The caller
        re-issues every queued recipe under a NEW id and logs it."""
        ids = cls(start=base, modulus=modulus, horizon_s=horizon_s, base=base)
        lo, hi = ids.id_range
        items = used.items() if isinstance(used, dict) else ((r, now) for r in used)
        seen = []
        for rid, t in items:
            rid = check_id(rid)
            if not (lo <= rid < hi):
                continue
            seen.append(rid)
            t = float(now if t is None else t)
            ids.history[rid] = max(ids.history.get(rid, t), t)
        seed = None
        if last is not None and lo <= check_id(last) < hi:
            seed = int(last)
        elif seen:
            seed = max(seen)
        if seed is not None:
            ids.next = base + (seed - base + 1) % modulus
        log.warning("lanes wire: recipe-id state rebuilt from %d sent ids; next id %d", len(seen), ids.next)
        return ids


def _payload_hash(values, comps) -> str:
    """Stable across processes (sha1 of the canonical JSON of the rows), so it survives a restart. (W 446-450)"""
    import hashlib
    txt = json.dumps([[float(v) for v in values], [[float(x) for x in r] for r in (comps or [])]])
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:16]


def _ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


_DEFAULT = object()


class ResultRouter:
    """ZoMBI-side routing of objective packets by recipe id, and lost-line detection (W 453-613).

    on_objective() classes:
      legacy     no id (old DiSCO, objective_injector without ids): unattributed
      manual     (R2) id in Manual Control's range 900..999, checked before the id lookup: logged only,
                 never ingested, never routed to a line
      unknown    id never issued (stale packet, other run), or arriving < 30 min after its recipe was
                 ISSUED (an old packet whose id was re-issued since; counted as ``too_soon``, F14): unattributed
      matched    first result for an open id -> route to its lane/line
      late       first result for a recipe already closed as lost / unprinted / null: still routed
      duplicate  same id and same rows again (re-delivery, also after a ZoMBI restart): already ingested
      reprint    same id, different rows (DiSCO re-printed a recipe, e.g. after its restart): a second
                 measurement of the same line, ingested and attributed
      null       NULL-slide notice (empty values, id only): the line is closed with no data
    What a class means for the engine in lanes4 (R2 item 5, F5, F13) is ``disposition()``.

    Rail order (lost-line rule): a rail is serial, so its coupons are measured in read order. When
    recipe r of rail X returns, a rail-X recipe q that was READ and has no result is lost only if q
    was certainly read before r: both acked and q's ack came first ("k" carries the LAST read, so ack
    order is read order), or q was acked before r was even issued. A read first seen in the same
    packet as a result, or reads that happened between two acks, are never closed by inference; the
    90-min timeout (counted from the ack, an upper bound on the read time) is their backstop.

    Persistence (R4): ``on_change(reason, now)`` runs once at the end of every public call that changed
    the state, and on every issue of the bound ``RecipeIds``. ``rev`` counts those changes."""

    def __init__(self, ids: RecipeIds, lost_after_read_s: float = LOST_AFTER_READ_S,
                 manual_ids: Optional[Tuple[int, int]] = MANUAL_ID_RANGE,
                 on_change: Optional[Callable[[str, float], None]] = None):
        if manual_ids is not None:
            manual_ids = (int(manual_ids[0]), int(manual_ids[1]))
            if _ranges_overlap(ids.id_range, manual_ids):
                raise ValueError(
                    f"recipe ids {ids.id_range[0]}..{ids.id_range[1] - 1} overlap the Manual Control range "
                    f"{manual_ids[0]}..{manual_ids[1] - 1}: results for them would be classed 'manual' and never "
                    f"routed (use RecipeIds.for_run(), or manual_ids=None for Manual Control's own router)")
        self.ids = ids
        self.lost_after = lost_after_read_s
        self.manual_ids = manual_ids
        self.last_read = {"front": None, "back": None}
        self.events: List[tuple] = []
        self.counts: Dict[str, int] = {}
        self.rev = 0
        self.hook_errors = 0
        self.on_change = on_change
        self._dirty = False
        ids.on_change = self._ids_changed

    # persistence ---------------------------------------------------------------------------------
    def _ids_changed(self, reason: str, now) -> None:
        self._dirty = True
        self._flush(reason, now)

    def _flush(self, reason: str, now) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self.rev += 1
        _call_hook(self, self.on_change, reason, now)

    def _event(self, *ev) -> None:
        self.events.append(tuple(ev))
        if len(self.events) > EVENTS_KEEP:
            del self.events[:len(self.events) - EVENTS_KEEP]

    def to_json(self, now: Optional[float] = None) -> dict:
        return dict(version=3, ids=self.ids.to_json(now), last_read=dict(self.last_read),
                    lost_after=self.lost_after,
                    manual_ids=None if self.manual_ids is None else list(self.manual_ids),
                    rev=self.rev, counts=dict(self.counts),
                    events=[_jsonable(e) for e in self.events[-EVENTS_PERSIST:]])

    @classmethod
    def from_json(cls, d: dict) -> "ResultRouter":
        """Loads version 2 (sandbox) and 3. A file without ``manual_ids`` gets the default range; a sandbox
        allocator that overlaps it (modulus 1000) is clamped to the run range with a WARNING instead of
        refusing the resume."""
        ids = RecipeIds.from_json(d["ids"])
        manual = tuple(d["manual_ids"]) if d.get("manual_ids") is not None else (
            None if "manual_ids" in d else MANUAL_ID_RANGE)
        if manual is not None and _ranges_overlap(ids.id_range, manual):
            if ids.base < manual[0]:
                log.warning("lanes wire: loaded recipe ids %d..%d overlap Manual Control's %d..%d; clamping the "
                            "allocator to %d..%d", ids.base, ids.base + ids.modulus - 1, manual[0], manual[1] - 1,
                            ids.base, manual[0] - 1)
                ids.modulus = manual[0] - ids.base
                ids.next = ids.base + (ids.next - ids.base) % ids.modulus
            else:
                manual = None
        R = cls(ids, float(d.get("lost_after", LOST_AFTER_READ_S)), manual_ids=manual)
        lr = d.get("last_read") or {}
        R.last_read = {rail: (None if lr.get(rail) is None else int(lr[rail])) for rail in RAILS}
        R.rev = int(d.get("rev", 0))
        R.counts = {str(k): int(v) for k, v in (d.get("counts") or {}).items()}
        R.events = [tuple(e) for e in (d.get("events") or [])]
        return R

    def save(self, path, now: Optional[float] = None, saver: Optional[Callable[[object, dict], bool]] = None) -> bool:
        """Write ``to_json(now)`` to ``path`` with ``saver`` (default: this module's in-place ``save_json``)."""
        return (saver or save_json)(path, self.to_json(now))

    @classmethod
    def load(cls, path) -> "ResultRouter":
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(json.load(fh))

    def persist_to(self, path, saver: Optional[Callable[[object, dict], bool]] = None) -> None:
        """R4: save the whole state to ``path`` (``lanes/ids.json``) after every change, synchronously."""
        def _hook(reason, now):
            if not self.save(path, now, saver):
                log.warning("lanes wire: could not write %s (reason=%s); retried at the next change", path, reason)
        self.on_change = _hook

    @staticmethod
    def state_key(d: Optional[dict]) -> Tuple[int, int]:
        """Order of two saved router states: (rev, ids.clock). ``rev`` counts every state change and refines
        R4's "the newer by RecipeIds.clock" (the clock does not tick on results)."""
        if not d:
            return (-1, -1)
        return int(d.get("rev", 0)), int((d.get("ids") or {}).get("clock", 0))

    @classmethod
    def pick_newer(cls, *states: Optional[dict]) -> Optional[dict]:
        """R4 item 2: of ``ids.json`` and the copy embedded in the last engine commit, the newer wins
        (ties: the first argument). None entries are skipped; None if all are None."""
        best = None
        for d in states:
            if d is None:
                continue
            if best is None or cls.state_key(d) > cls.state_key(best):
                best = d
        return best

    # issuing (passthrough, so the driver needs only the router) ------------------------------------
    def try_issue(self, slot: int, line, now: float) -> Optional[int]:
        return self.ids.try_issue(slot, line, now)

    # read acknowledgements ("k") ------------------------------------------------------------------
    def _read_ack(self, last_read, now: float):
        """W 502-519 (``on_read_ack``), marking the state dirty when it changes."""
        if last_read is None:
            return
        for rail, rid in zip(RAILS, last_read):
            if rid is None or rid == self.last_read[rail]:
                continue
            self.last_read[rail] = rid
            self._dirty = True
            r = self.ids.by_id.get(rid)
            if r is None:
                self._event("ack_unknown", rail, rid, now)
                continue
            if r.seq_ack is None:
                r.t_read, r.seq_ack = now, self.ids.tick()
                r.rail = rail                     # "k" says which rail really read it (shared slots too)
                if r.status in ("issued", "unprinted"):
                    r.status = "read"
                self._event("read", rail, rid, now)
                self._close_superseded(rail, r, now)

    def on_read_ack(self, last_read, now: float):
        self._read_ack(last_read, now)
        self._flush("ack", now)

    def supersede(self, rid: int, now: float):
        """ZoMBI overwrote this slot with another recipe. (W 521-525)"""
        r = self.ids.by_id.get(rid)
        if r is not None and r.t_superseded is None:
            r.t_superseded, r.seq_superseded = now, self.ids.tick()
            self._dirty = True
        self._flush("supersede", now)

    def _close_superseded(self, rail, latest: Recipe, now):
        # a recipe overwritten before any ack showed it read, and the rail has since read something issued
        # later -> presumed never printed (an inference: a result that still arrives is routed as 'late')
        for r in self.ids.by_id.values():
            if (r.rail == rail and r.status == "issued" and r.seq_superseded is not None
                    and latest.seq_issue > r.seq_issue and r.seq_superseded < latest.seq_ack):
                r.status = "unprinted"
                self._dirty = True
                self._event("unprinted", rail, r.rid, now)

    @staticmethod
    def _before(q: Recipe, r: Recipe) -> bool:
        """True only when q was CERTAINLY read before r (same rail)."""
        if q.seq_ack is None:
            return False
        if r.seq_ack is not None:
            return q.seq_ack < r.seq_ack
        return q.seq_ack < r.seq_issue

    # results ---------------------------------------------------------------------------------------
    def _route(self, decoded: dict, now: float) -> dict:
        """W 546-575 (``on_objective``) plus the R2 ``manual`` class, marking the state dirty when it changes."""
        self._read_ack(decoded.get("last_read"), now)
        rid = decoded.get("rid")
        vals = decoded.get("values") or []
        if rid is None:
            return dict(cls="legacy", rid=None, lost=[])
        if is_manual_id(rid, self.manual_ids):           # R2 item 4: before the id lookup
            return dict(cls="manual", rid=rid, lost=[])
        r = self.ids.by_id.get(rid)
        if r is None:
            return dict(cls="unknown", rid=rid, lost=[])
        if len(vals) == 0:
            if r.status in OPEN:
                r.status, r.t_result = "null", now
                self.ids.history[rid] = now
                self._dirty = True
            return dict(cls="null", rid=rid, lost=self._fifo_close(r, now))
        h = _payload_hash(vals, decoded.get("comps"))
        if h in r.payload_hashes:
            return dict(cls="duplicate", rid=rid, lost=[])
        if r.payload_hashes:
            r.payload_hashes.append(h)
            self._dirty = True
            return dict(cls="reprint", rid=rid, line=r.line, slot=r.slot, rail=r.rail, lost=[])
        if now - r.t_issue < MIN_READ_TO_RESULT_S:
            # physically impossible for this recipe (issue -> read -> print -> HS -> Prism took 50-78 min
            # from the READ alone on run_96a9): a stale packet whose id was re-issued since
            return dict(cls="unknown", rid=rid, lost=[], why="too soon after issue")
        late = r.status in ("lost", "unprinted", "null")   # a timeout / inference was wrong: still routed
        r.status, r.t_result = "resulted", now
        r.payload_hashes.append(h)
        self.ids.history[rid] = now
        self._dirty = True
        return dict(cls="late" if late else "matched", rid=rid, line=r.line, slot=r.slot, rail=r.rail,
                    lost=self._fifo_close(r, now))

    def on_objective(self, decoded: dict, now: float) -> dict:
        res = self._route(decoded, now)
        cls = res["cls"]
        self.counts[cls] = self.counts.get(cls, 0) + 1
        if res.get("why") == "too soon after issue":       # F14: count and alarm
            self.counts["too_soon"] = self.counts.get("too_soon", 0) + 1
            log.warning("lanes wire: result for id %s arrived %.0f s after the id was issued (< %.0f s): classed "
                        "unknown; if a real cycle is this short, own results lose their credit",
                        res.get("rid"), now - self.ids.by_id[res["rid"]].t_issue, MIN_READ_TO_RESULT_S)
        self._flush(cls, now)
        return res

    def _fifo_close(self, r: Recipe, now: float) -> List[int]:
        """Rail order: every recipe of r's rail CERTAINLY read before r (``_before``) and still without a
        result will never get one. (W 577-588)"""
        lost = []
        for q in self.ids.by_id.values():
            if q is r or q.rail != r.rail or q.status != "read":
                continue
            if self._before(q, r):
                q.status = "lost"
                lost.append(q)
                self._dirty = True
                self._event("lost_fifo", r.rail, q.rid, now)
        return [q.rid for q in sorted(lost, key=lambda q: q.seq_ack)]

    def expire(self, now: float) -> List[int]:
        """Timeout backstop: a read without result for longer than ``lost_after`` since its ack (the ack
        time is an upper bound on the read time, so this never fires early). (W 590-599)"""
        lost = []
        for q in self.ids.by_id.values():
            if q.status == "read" and q.t_read is not None and now - q.t_read > self.lost_after:
                q.status = "lost"
                lost.append(q.rid)
                self._dirty = True
                self._event("lost_timeout", q.rail, q.rid, now)
        self._flush("expire", now)
        return lost

    def on_disco_restart(self, now: float, issued_before: Optional[float] = None) -> List[int]:
        """DiSCO restarted (its selftest pings reached ZoMBI): in-flight coupons are not measured
        (stability.db / prism.db are re-initialised) and compositions.db is wiped. Close every read
        recipe as lost; ZoMBI then re-issues its queued lines under NEW ids. (W 601-613)

        issued_before (G2, scripts/lanes_hw.py): close only recipes ISSUED before this time (the
        restart's first self-test ping); a recipe issued after it belongs to the new DiSCO session and
        is left open. None (default): every recipe, the sandbox behaviour. A result that still arrives
        for a closed recipe routes by its id as 'late' (its id is not re-issued within the horizon)."""
        lost = []
        for q in self.ids.by_id.values():
            if issued_before is not None and not float(q.t_issue) < float(issued_before):
                continue
            if q.status == "read":
                q.status = "lost"
                lost.append(q.rid)
            elif q.status == "issued":
                q.status = "unprinted"
        self.last_read = {"front": None, "back": None}
        self._dirty = True
        self._event("disco_restart", None, None, now)
        self._flush("disco_restart", now)
        return lost


# ── what a routed packet means for the engine in lanes4 (R2 item 5, F5, F13) ───────────────────────
LANE, INIT, SHARED, LOG = "lane", "init", "shared", "log"


def disposition(routed: dict, *, ingest_unattributed: bool = False, t_recv: Optional[float] = None,
                ts=None, run_start: Optional[float] = None, first_write_ts=None) -> str:
    """-> ``"lane"`` (ingest as the line's own result), ``"init"`` (late init packet: shared data, label
    ``init_p{pid}``), ``"shared"`` (unattributed data, only with ``lanes_ingest_unattributed``) or ``"log"``
    (log in received.jsonl only: never ingested, no controller call).

    * matched / late / reprint: by ``Recipe.line["kind"]`` ("init" -> init, anything else -> lane);
    * duplicate (F13), null (F5), manual (R2 item 4): log;
    * unknown / legacy: log, unless ``ingest_unattributed`` AND ``t_recv >= run_start`` AND
      ``ts > first_write_ts`` (each condition checked only when its reference is given) -> shared.
    An all-dropped packet (hygiene) is a NULL result for its id and never reaches this function as data."""
    cls = routed.get("cls")
    if cls in ("matched", "late", "reprint"):
        return INIT if line_kind(routed.get("line")) == "init" else LANE
    if cls in ("unknown", "legacy") and ingest_unattributed:
        if run_start is not None and (t_recv is None or t_recv < run_start):
            return LOG
        if first_write_ts is not None:
            try:
                newer = ts is not None and float(ts) > float(first_write_ts)
            except (TypeError, ValueError):
                newer = False
            if not newer:
                return LOG
        return SHARED
    return LOG


# ── size helpers (W 616-619) ─────────────────────────────────────────────────────────
def added_bytes_objective(rid: Optional[int] = 417, last_read=(418, 352)) -> int:
    base = len(encode_objective([0.5], [[0.0] * 10], 1793817831.5912442))
    return len(encode_objective([0.5], [[0.0] * 10], 1793817831.5912442, rid, last_read)) - base
