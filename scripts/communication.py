import serial
import sqlite3
import time
from datetime import datetime
import numpy as np
import json
import threading
import queue
import atexit
import sys
import os
import signal
from typing import Any, Optional

# Global flag for graceful shutdown
_shutdown_flag = threading.Event()

# one global Lock for all serial I/O
_serial_lock = threading.Lock()

# Global lock for objective database access
_objective_db_lock = threading.Lock()

# Flag to indicate when objective data is being written
_objective_writing = False

# Last objective packet (dedup: only process/print once per distinct packet)
_last_objective_vals = None
_last_objective_comps = None
# Last objective packet timestamp. DiSCO stamps objective.db with a fresh ts on every
# write and ships it in the packet; we only ingest a packet whose ts is new, so the
# ~1 Hz re-sends of identical content don't keep overwriting the DBs. Mirrors the
# composition path (composition_receiver only writes when compositions.db's ts changes).
_last_objective_ts = None


def _serial_process_signal_handler(signum, frame):
    """Only used in the dedicated serial I/O process (not at import: avoids every child
    that `import communication` from stealing SIGINT; ZoMBI does not read _shutdown_flag)."""
    print(f"[SerialIO] Signal {signum} — closing serial path…")
    _shutdown_flag.set()


def _is_access_denied(exc: BaseException) -> bool:
    """Did opening the port fail because another process already holds it?

    pyserial wraps the Win32 error in a SerialException whose text is
    ``could not open port 'COM5': PermissionError(13, 'Access is denied.', None, 5)``,
    so the errno is not directly reachable — match on the text as well as on a
    chained PermissionError.
    """
    if isinstance(exc, PermissionError) or isinstance(getattr(exc, "__cause__", None), PermissionError):
        return True
    text = str(exc).lower()
    return "access is denied" in text or "permissionerror" in text


def _register_signal_handlers_in_serial_process_only() -> None:
    """Call once from `start_serial_dual_io_shared_port` in the serial subprocess."""
    try:
        signal.signal(signal.SIGINT, _serial_process_signal_handler)
        signal.signal(signal.SIGTERM, _serial_process_signal_handler)
        # Windows Ctrl+Break (how the GUI asks a hardware run to stop). Handling it
        # here lets the serial child close the COM port cleanly instead of being
        # killed by the default action with the port still open.
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _serial_process_signal_handler)
    except (OSError, ValueError):
        pass


def reset_objective(db_path="./sql/objective.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='objective'"
    )
    if cursor.fetchone():
        cursor.execute("DELETE FROM objective;")
        conn.commit()
    conn.close()


def clear_in_flight_objective_state(db_path: str = "./sql/objective.db") -> None:
    """
    After an interrupted / unclean run: clear the objective handshake so the
    next launch does not treat a partial/stale result as a completed read.
    Ensures the next `objective()` call can send a new line and
    `get_y_measurements(..., ready_for_objectives=True)` waits for fresh data.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS handshake (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                new_objective_available INTEGER DEFAULT 0
            )"""
        )
        cur.execute("INSERT OR IGNORE INTO handshake (id, new_objective_available) VALUES (1, 0)")
        cur.execute("UPDATE handshake SET new_objective_available = 0 WHERE id = 1")
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='objective'")
        if cur.fetchone():
            cur.execute("DELETE FROM objective")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[communication] clear_in_flight_objective_state: {e!r}")


def write_compositions(start,
                       end,
                       array,
                       timestamp,
                       start_cache,
                       end_cache,
                       array_cache,
                       db_path="./sql/compositions.db"):
    """
    Overwrite the following tables in compositions.db:
      - compositions       : the main array (n×10)
      - start              : the main 1×10 start vector
      - end                : the main 1×10 end   vector
      - compositions_cache : the cached array (n×10)
      - start_cache        : the cached 1×10 start vector
      - end_cache          : the cached 1×10 end   vector
      - timestamp          : a single REAL ts
    """
    # -- shape validations --

    # start = np.array([1,0,0,0,0,0,0,0,0,0])
    # start_cache = start
    # end = np.array([1,0,0,0,0,0,0,0,0,0])
    # end_cache = end
    # array = np.linspace(start,end,24)
    # array_cache =array

    for name, arr in [("array", array), ("array_cache", array_cache)]:
        if arr.ndim != 2 or arr.shape[1] != 10:
            raise ValueError(f"`{name}` must be 2D with exactly 10 columns")

    for name, vec in [
        ("start", start), ("end", end),
        ("start_cache", start_cache), ("end_cache", end_cache)
    ]:
        if vec.ndim != 1 or vec.shape[0] != 10:
            raise ValueError(f"`{name}` must be 1D with length 10")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # reusable column defs and placeholders
    col_defs    = ", ".join(f"col_{i} REAL" for i in range(10))
    placeholders = ", ".join("?" for _ in range(10))

    # 1) main compositions
    cur.execute("DROP TABLE IF EXISTS compositions")
    cur.execute(f"CREATE TABLE compositions ({col_defs})")
    cur.executemany(
        f"INSERT INTO compositions VALUES ({placeholders})",
        array.tolist()
    )

    # 2) main start
    cur.execute("DROP TABLE IF EXISTS start")
    cur.execute(f"CREATE TABLE start ({col_defs})")
    cur.execute(
        f"INSERT INTO start VALUES ({placeholders})",
        start.tolist()
    )

    # 3) main end
    cur.execute("DROP TABLE IF EXISTS end")
    cur.execute(f"CREATE TABLE end ({col_defs})")
    cur.execute(
        f"INSERT INTO end VALUES ({placeholders})",
        end.tolist()
    )

    # 4) cache compositions
    cur.execute("DROP TABLE IF EXISTS compositions_cache")
    cur.execute(f"CREATE TABLE compositions_cache ({col_defs})")
    cur.executemany(
        f"INSERT INTO compositions_cache VALUES ({placeholders})",
        array_cache.tolist()
    )

    # 5) cache start
    cur.execute("DROP TABLE IF EXISTS start_cache")
    cur.execute(f"CREATE TABLE start_cache ({col_defs})")
    cur.execute(
        f"INSERT INTO start_cache VALUES ({placeholders})",
        start_cache.tolist()
    )

    # 6) cache end
    cur.execute("DROP TABLE IF EXISTS end_cache")
    cur.execute(f"CREATE TABLE end_cache ({col_defs})")
    cur.execute(
        f"INSERT INTO end_cache VALUES ({placeholders})",
        end_cache.tolist()
    )

    # 7) timestamp
    cur.execute("DROP TABLE IF EXISTS timestamp")
    cur.execute("CREATE TABLE timestamp (ts REAL)")
    cur.execute("INSERT INTO timestamp VALUES (?)", (float(timestamp),))

    conn.commit()
    conn.close()
    

# # — 3) reset_compositions drops all four tables —
def reset_compositions(db_path="./sql/compositions.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for tbl in ("compositions", "start", "end", "timestamp"):
        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.commit()
    conn.close()


# ─── Helper to drop & recreate the 'results' table with one row ─────────────
def add_vector_to_first_row(x, db_path):
    x = np.asarray(x).ravel()
    N = x.shape[0]
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS objective")
    cols = ", ".join(f"v{i} REAL" for i in range(N))
    cur.execute(f"CREATE TABLE objective ({cols})")
    ph   = ", ".join("?" for _ in range(N))
    cur.execute(f"INSERT INTO objective VALUES ({ph})", tuple(x.tolist()))
    conn.commit()
    conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# LANES (four-slot protocol) — additive. Nothing below changes a legacy byte:
#   * write_slots4 / clear_slots4 own two NEW tables in compositions.db
#     (slots4, slots4_meta). The legacy writers DROP+CREATE only their own
#     tables, so they never touch these; initialize_db() wipes the whole file
#     at a new trial, which also turns lanes sending off (active row gone).
#   * composition_sender sends the four-slot packet ONLY while
#     slots4_meta.active == 1; otherwise it runs today's code path unchanged.
#     A tick whose slots4 read fails is skipped (nothing sent) unless the last
#     readable read showed four-slot off; then today's path runs as before.
#   * objective_receiver additionally appends every NEW objective packet to
#     sql/objective_queue.db (table q); its legacy behaviour is unchanged.
# ═════════════════════════════════════════════════════════════════════════════
COMPOSITIONS_DB = "./sql/compositions.db"
OBJECTIVE_QUEUE_DB = "./sql/objective_queue.db"
SLOTS4_LOCK_DEADLINE_S = 30.0      # write_slots4 / clear_slots4 never raise on a lock before this
QUEUE_APPEND_DEADLINE_S = 5.0      # receiver thread: retry this long, then leave it to the 1 Hz re-send
QUEUE_READ_DEADLINE_S = 5.0        # read_objective_queue: retry this long, then return [] (next poll retries)

_wire_mod = None                   # src/lanes/wire.py, loaded by path (keeps the serial child torch-free)
_last_queued_ts = None             # queue de-dup (independent of the legacy re-arm of _last_objective_ts)
_last_queued_raw = None            # queue de-dup for a packet without ts
_last_queued_has_r = False         # the queued frame of _last_queued_ts carried a recipe id "r"


def _lanes_wire():
    """src/lanes/wire.py loaded BY PATH, as its docstring prescribes: importing it as
    ``src.lanes.wire`` would run src/__init__.py (torch + ZoMBIHop) inside the serial child."""
    global _wire_mod
    if _wire_mod is None:
        mod = sys.modules.get("src.lanes.wire")
        if mod is None:
            import importlib.util
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "lanes", "wire.py")
            spec = importlib.util.spec_from_file_location("_zombi_lanes_wire", os.path.normpath(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod          # @dataclass needs the module registered while it runs
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                sys.modules.pop(spec.name, None)
                raise
        _wire_mod = mod
    return _wire_mod


def _is_lock_error(exc: BaseException) -> bool:
    """sqlite errors that mean 'someone else (or Dropbox) holds the file right now'."""
    text = str(exc).lower()
    return isinstance(exc, sqlite3.OperationalError) and (
        "locked" in text or "busy" in text or "unable to open" in text or "disk i/o" in text)


def _sqlite_retry(path, fn, deadline_s: float, what: str, timeout: float = 2.0):
    """Run fn(conn) in its own short transaction, retrying lock errors until `deadline_s`.
    Re-raises the last lock error after the deadline, and any other error at once."""
    t_end = time.monotonic() + deadline_s
    delay = 0.05
    while True:
        conn = None
        try:
            conn = sqlite3.connect(os.fspath(path), timeout=timeout)
            conn.execute("PRAGMA busy_timeout = %d" % int(timeout * 1000))
            out = fn(conn)
            conn.commit()
            return out
        except sqlite3.OperationalError as e:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            if not _is_lock_error(e) or time.monotonic() >= t_end:
                raise
            print(f"[{what}] sqlite busy ({e}); retrying…")
            time.sleep(delay)
            delay = min(delay * 2.0, 1.0)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


_SLOTS4_DDL = (
    "CREATE TABLE IF NOT EXISTS slots4 (slot INTEGER PRIMARY KEY, rid INTEGER, start TEXT, end TEXT)",
    "CREATE TABLE IF NOT EXISTS slots4_meta (id INTEGER PRIMARY KEY CHECK (id = 0), ts REAL, active INTEGER)",
)


def _check_vec10(v, name):
    out = [float(x) for x in list(v)]
    if len(out) != 10:
        raise ValueError(f"{name} must have 10 components, got {len(out)}")
    for x in out:
        if not np.isfinite(x) or x < 0:
            raise ValueError(f"{name} has a bad component {x!r}")
    return out


def write_slots4(slots, ts=None, path=COMPOSITIONS_DB):
    """Write the FULL four-slot state (DESIGN R1: never a per-lane delta) into compositions.db.

    slots: up to 4 tuples (slot:int 0..3, rid:int 0..999, start:10 floats, end:10 floats); a slot that
    is absent is EMPTY (withdrawn). One transaction: slots4 replaced, slots4_meta = (ts, active=1).
    ts: time.time() when None; always strictly newer than the stored ts (DiSCO de-dups by ts).
    Retries 'database is locked' (Dropbox) for SLOTS4_LOCK_DEADLINE_S, then raises. Returns the ts written.
    """
    rows, seen = [], set()
    for ent in slots:
        slot, rid, a, b = ent
        if isinstance(slot, bool) or int(slot) != slot or int(slot) not in (0, 1, 2, 3) or int(slot) in seen:
            raise ValueError(f"bad or repeated slot {slot!r}")
        slot = int(slot)
        seen.add(slot)
        if isinstance(rid, bool) or int(rid) != rid or not (0 <= int(rid) <= 999):
            raise ValueError(f"recipe id must be an integer 0..999, got {rid!r}")
        rows.append((slot, int(rid), json.dumps(_check_vec10(a, "start")), json.dumps(_check_vec10(b, "end"))))
    rows.sort()
    want_ts = None if ts is None else float(ts)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _tx(conn):
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        for ddl in _SLOTS4_DDL:
            cur.execute(ddl)
        cur.execute("SELECT ts FROM slots4_meta WHERE id = 0")
        prev = cur.fetchone()
        new_ts = time.time() if want_ts is None else want_ts
        if want_ts is None and prev is not None and prev[0] is not None and new_ts <= float(prev[0]):
            new_ts = float(prev[0]) + 0.001
        cur.execute("DELETE FROM slots4")
        cur.executemany("INSERT INTO slots4 (slot, rid, start, end) VALUES (?, ?, ?, ?)", rows)
        cur.execute("INSERT OR REPLACE INTO slots4_meta (id, ts, active) VALUES (0, ?, 1)", (new_ts,))
        return new_ts

    return _sqlite_retry(path, _tx, SLOTS4_LOCK_DEADLINE_S, "write_slots4")


def clear_slots4(path=COMPOSITIONS_DB):
    """Turn four-slot sending off (active=0): composition_sender goes back to today's packet."""
    if not os.path.exists(path):
        return

    def _tx(conn):
        cur = conn.cursor()
        for ddl in _SLOTS4_DDL:
            cur.execute(ddl)
        cur.execute("UPDATE slots4_meta SET active = 0 WHERE id = 0")

    _sqlite_retry(path, _tx, SLOTS4_LOCK_DEADLINE_S, "clear_slots4")


SLOTS4_ACTIVE = "active"          # _slots4_state: slots4_meta.active == 1, map read
SLOTS4_INACTIVE = "inactive"      # no compositions.db / no slots4_meta table or row / active != 1
SLOTS4_UNREADABLE = "unreadable"  # the read failed (lock, Dropbox 'unable to open', a bad row)


def _slots4_state(comp_db_path):
    """-> (state, payload). Never raises.
    (SLOTS4_ACTIVE, (ts, {slot: (rid, start, end)})) while slots4_meta.active == 1;
    (SLOTS4_INACTIVE, None) when four-slot sending is off (a legacy session never has slots4_meta);
    (SLOTS4_UNREADABLE, meta_active) when the read failed: meta_active is True when
    slots4_meta.active == 1 was read before the failure, None when the mode could not be read.
    composition_sender needs 'inactive' and 'unreadable' apart (review B1): an unreadable read in
    lanes mode must not fall back to today's two-slot packet."""
    meta_active = None
    try:
        if not os.path.exists(comp_db_path):
            return SLOTS4_INACTIVE, None
        conn = sqlite3.connect(comp_db_path, timeout=10.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='slots4_meta'")
            if not cur.fetchone():
                return SLOTS4_INACTIVE, None
            cur.execute("SELECT ts, active FROM slots4_meta WHERE id = 0")
            meta = cur.fetchone()
            if not meta or meta[1] != 1:
                return SLOTS4_INACTIVE, None
            meta_active = True
            cur.execute("SELECT slot, rid, start, end FROM slots4 ORDER BY slot")
            slots = {int(s): (int(r), json.loads(a), json.loads(b)) for s, r, a, b in cur.fetchall()}
        finally:
            conn.close()
        return SLOTS4_ACTIVE, (meta[0], slots)
    except Exception as e:
        print(f"[composition_sender] slots4 read error: {e!r}")
        return SLOTS4_UNREADABLE, meta_active


def _read_slots4(comp_db_path):
    """-> (ts, {slot: (rid, start, end)}) while slots4_meta.active == 1, else None. Never raises.
    (None also on a read error; composition_sender uses _slots4_state to tell the two apart.)"""
    state, payload = _slots4_state(comp_db_path)
    return payload if state == SLOTS4_ACTIVE else None


def _queue_objective_packet(line, pkt, obj_db_path, queue_db_path=None):
    """Append one NEW objective packet (raw JSON line) to the lanes queue. De-dup by ts (by the raw
    line when there is no ts), independent of the legacy receiver. A write that keeps failing is NOT
    marked as queued, so the next 1 Hz re-send of the same packet retries it. Never raises."""
    global _last_queued_ts, _last_queued_raw, _last_queued_has_r
    try:
        if not isinstance(pkt, dict) or pkt.get("type") != "objective":
            return False
        ts = pkt.get("ts", None)
        if isinstance(ts, bool) or not isinstance(ts, (int, float)):
            ts = None
        has_r = pkt.get("r", None) is not None
        if ts is not None:
            # Same table re-sent: skip it, UNLESS the first copy came without a recipe id and this
            # one carries it (DiSCO adds "r" once its ZoMBI link is four-slot again after a restart).
            # The r-less copy was journaled as legacy and never ingested, so queueing the r-bearing
            # copy once lets the measurement reach its lane instead of being lost.
            if _last_queued_ts is not None and ts == _last_queued_ts and (_last_queued_has_r or not has_r):
                return False
        elif _last_queued_raw is not None and line == _last_queued_raw:
            return False
        qpath = queue_db_path or os.path.join(os.path.dirname(obj_db_path) or ".", "objective_queue.db")
        t_recv = time.time()

        def _tx(conn):
            conn.execute("CREATE TABLE IF NOT EXISTS q (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                         "t_recv REAL, ts REAL, raw TEXT)")
            conn.execute("INSERT INTO q (t_recv, ts, raw) VALUES (?, ?, ?)",
                         (t_recv, None if ts is None else float(ts), line))

        _sqlite_retry(qpath, _tx, QUEUE_APPEND_DEADLINE_S, "objective_queue")
        if ts is not None and ts == _last_queued_ts and has_r and not _last_queued_has_r:
            print(f"[objective_receiver] result ts={ts} re-sent WITH its recipe id r={pkt.get('r')}: queued again")
        _last_queued_ts = ts
        _last_queued_raw = line
        _last_queued_has_r = has_r
        return True
    except Exception as e:
        print(f"[objective_receiver] queue append failed (will retry on re-send): {e!r}")
        return False


def read_objective_queue(after_id: int = 0, path=OBJECTIVE_QUEUE_DB):
    """-> [(id, packet_dict), ...] for queue rows with id > after_id, in id order. Missing file/table
    -> []. A lock that outlasts QUEUE_READ_DEADLINE_S also returns [] (the caller polls again)."""
    if not os.path.exists(path):
        return []

    def _tx(conn):
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='q'")
        if not cur.fetchone():
            return []
        cur.execute("SELECT id, raw FROM q WHERE id > ? ORDER BY id", (int(after_id),))
        return cur.fetchall()

    try:
        rows = _sqlite_retry(path, _tx, QUEUE_READ_DEADLINE_S, "read_objective_queue")
    except Exception as e:
        print(f"[read_objective_queue] {e!r}; returning nothing this poll")
        return []
    out = []
    for qid, raw in rows:
        try:
            pkt = json.loads(raw)
        except Exception:
            print(f"[read_objective_queue] row {qid}: unparsable raw, skipped")
            continue
        if isinstance(pkt, dict):
            out.append((int(qid), pkt))
    return out


# ── DiSCO restart detection (lanes, gap G2) ───────────────────────────────────
# THE SIGNAL. DiSCO sends {"type": "selftest_ping", "from": "disco", "seq": n, "pad": "x"*3500}
# ONLY from its startup self-test (DiSCO communication.py serial_link_selftest, run by
# dual_serial_zombi before its live threads start), i.e. once per DiSCO process start, right
# after DiSCO's startup wiped compositions.db (slots, four-slot mode marker, read acks). DiSCO's
# live link never pings: its ZombiLink only ANSWERS ZoMBI's pings. So a DiSCO ping reaching
# ZoMBI's live threads means DiSCO (re)started.
#
# Before this, objective_receiver dropped that ping as "Not an objective packet" and no live
# ZoMBI thread answered it, so DiSCO's gated self-test failed after 120 s and DiSCO never
# started its ZoMBI link unless ZoMBI was restarted too. Now (lanes only; legacy unchanged):
#   * objective_receiver hands each DiSCO ping to a DiscoRestartLink. The first ping of an
#     episode (no DiSCO ping for DISCO_PING_EPISODE_GAP_S) appends ONE 'disco_restart' row to
#     the link_events table of objective_queue.db (table q is untouched); read_link_events
#     returns it to the lanes driver, which re-issues every slot under NEW ids;
#   * composition_sender, while slots4 is active, answers the newest ping with the padded
#     selftest_pong the startup self-test sends, so DiSCO's self-test passes against this
#     running ZoMBI, and HOLDS the four-slot map from the restart on: a map goes out again only
#     when its ts is newer than the restart AND it carries no recipe id of the last map sent
#     before the restart. So the fresh DiSCO never sees an id its previous session may already
#     have printed, whatever the timing of the driver's re-issue.
DISCO_PING_EPISODE_GAP_S = 10.0    # = DiSCO's ZOMBI_PING_EPISODE_GAP: a longer silence ends a self-test
DISCO_PONG_MAX_AGE_S = 5.0         # an unanswered ping older than this is not answered any more
DISCO_HOLD_WARN_S = 30.0           # console line when the map is still held this long after the restart
DISCO_HOLD_REMIND_S = 300.0        # ... repeated this often while it stays held
SELFTEST_PAD = 3500                # pong padding, identical to serial_link_selftest's
LINK_EVENTS_TABLE = "link_events"
_QUEUE_DDL = ("CREATE TABLE IF NOT EXISTS q (id INTEGER PRIMARY KEY AUTOINCREMENT, "
              "t_recv REAL, ts REAL, raw TEXT)")          # identical to _queue_objective_packet's
_LINK_EVENTS_DDL = (f"CREATE TABLE IF NOT EXISTS {LINK_EVENTS_TABLE} (id INTEGER PRIMARY KEY "
                    "AUTOINCREMENT, t_recv REAL, after_qid INTEGER, kind TEXT, raw TEXT)")


def append_link_event(ev, path=OBJECTIVE_QUEUE_DB):
    """Append one link event (a JSON-able dict with "type") to objective_queue.db's link_events
    table. after_qid = the newest q row at that moment, so the lanes driver replays events and
    objective packets in arrival order. Returns the event id; raises after
    QUEUE_APPEND_DEADLINE_S of locks (DiscoRestartLink keeps the row and retries)."""
    raw = json.dumps(ev)
    t_recv = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _tx(conn):
        conn.execute(_QUEUE_DDL)
        conn.execute(_LINK_EVENTS_DDL)
        row = conn.execute("SELECT MAX(id) FROM q").fetchone()
        after = int(row[0]) if row and row[0] is not None else 0
        cur = conn.execute(f"INSERT INTO {LINK_EVENTS_TABLE} (t_recv, after_qid, kind, raw) "
                           "VALUES (?, ?, ?, ?)", (t_recv, after, str(ev.get("type")), raw))
        return int(cur.lastrowid)

    return _sqlite_retry(path, _tx, QUEUE_APPEND_DEADLINE_S, "link_events")


def read_link_events(after_id: int = 0, path=OBJECTIVE_QUEUE_DB):
    """-> [(id, after_qid, event_dict), ...] for link_events rows with id > after_id, in id order.
    Missing file/table -> []; a lock that outlasts QUEUE_READ_DEADLINE_S -> [] (poll again)."""
    if not os.path.exists(path):
        return []

    def _tx(conn):
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (LINK_EVENTS_TABLE,))
        if not cur.fetchone():
            return []
        cur.execute(f"SELECT id, after_qid, raw FROM {LINK_EVENTS_TABLE} WHERE id > ? ORDER BY id",
                    (int(after_id),))
        return cur.fetchall()

    try:
        rows = _sqlite_retry(path, _tx, QUEUE_READ_DEADLINE_S, "read_link_events")
    except Exception as e:
        print(f"[read_link_events] {e!r}; returning nothing this poll")
        return []
    out = []
    for eid, after, raw in rows:
        try:
            ev = json.loads(raw)
        except Exception:
            print(f"[read_link_events] row {eid}: unparsable raw, skipped")
            continue
        if isinstance(ev, dict):
            out.append((int(eid), int(after or 0), ev))
    return out


def max_link_event_id(path=OBJECTIVE_QUEUE_DB) -> int:
    """Newest link_events id (0 when there is none or the file cannot be read)."""
    if not os.path.exists(path):
        return 0

    def _tx(conn):
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (LINK_EVENTS_TABLE,))
        if not cur.fetchone():
            return 0
        row = cur.execute(f"SELECT MAX(id) FROM {LINK_EVENTS_TABLE}").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    try:
        return int(_sqlite_retry(path, _tx, QUEUE_READ_DEADLINE_S, "max_link_event_id"))
    except Exception:
        return 0


def selftest_pong_frame(ack, role="zombi"):
    """The padded selftest_pong frame, byte-identical to serial_link_selftest's reply."""
    return (json.dumps({"type": "selftest_pong", "from": role, "ack": ack, "pad": "x" * SELFTEST_PAD})
            + "\n").encode()


class DiscoRestartLink:
    """ZoMBI-side state of the DiSCO link across a DiSCO restart (block comment above). One per
    serial session, shared by objective_receiver (note_ping), composition_sender (take_pong,
    map_allowed, note_map_sent; lanes branch only) and the startup self-test (note_ping with
    defer=True). Thread-safe; never raises into those threads."""

    def __init__(self, queue_path=OBJECTIVE_QUEUE_DB, clock=time.time):
        self.queue_path = queue_path
        self.clock = clock
        self._lock = threading.Lock()
        self.last_ping_t = None            # clock() of the last DiSCO selftest_ping (live or startup)
        self.episode = 0                   # DiSCO self-test episodes seen (one per DiSCO start)
        self.episode_t = None              # clock() of the first ping of the current episode
        self.unsaved = []                  # event rows not appended yet (deferred, or a lock)
        self.n_events = 0                  # rows appended
        self._pong = None                  # (seq, clock()) of the newest unanswered LIVE ping
        self._hold_episode = 0             # newest episode the sender has turned into a hold
        self.hold = None                   # dict while the four-slot map is held
        self.last_sent_rids = frozenset()  # recipe ids of the last four-slot frame on the wire

    # -- receiver / startup self-test --------------------------------------------------
    def note_ping(self, pkt, src="live", defer=False):
        """A DiSCO selftest_ping. The first ping after a quiet gap starts a new episode and queues
        ONE 'disco_restart' event; a live ping also asks the sender for a pong (the startup
        self-test answers its own). defer=True keeps the row in memory until flush() (no sqlite
        inside the self-test loop). Returns the event dict when this ping started an episode."""
        try:
            now = float(self.clock())
            seq = pkt.get("seq") if isinstance(pkt, dict) else None
            ev = None
            with self._lock:
                if self.last_ping_t is None or now - self.last_ping_t >= DISCO_PING_EPISODE_GAP_S:
                    self.episode += 1
                    self.episode_t = now
                    ev = {"type": "disco_restart", "src": src, "t_ping": now, "seq": seq,
                          "episode": self.episode, "pid": os.getpid()}
                    self.unsaved.append(ev)
                self.last_ping_t = now
                if src == "live":
                    self._pong = (seq, now)
            if not defer:
                self.flush()
            return ev
        except Exception as e:                          # pragma: no cover - never kill the receiver
            print(f"[link] DiSCO ping bookkeeping failed: {e!r}")
            return None

    def flush(self) -> int:
        """Append the queued event rows. A row that cannot be written stays queued and is retried
        at the next ping or sender tick. Returns the number of rows appended."""
        with self._lock:
            todo, self.unsaved = self.unsaved, []
        done = 0
        for i, ev in enumerate(todo):
            try:
                append_link_event(ev, self.queue_path)
                done += 1
            except Exception as e:
                with self._lock:
                    self.unsaved = todo[i:] + self.unsaved
                print(f"[link] could not record the DiSCO restart event yet (retrying): {e!r}")
                break
        with self._lock:
            self.n_events += done
        return done

    # -- sender, lanes branch only ---------------------------------------------------------
    def take_pong(self):
        """(seq, t) of the newest unanswered live DiSCO ping, once; None when nothing is due."""
        with self._lock:
            p, self._pong = self._pong, None
        if p is None or float(self.clock()) - p[1] > DISCO_PONG_MAX_AGE_S:
            return None
        return p

    def map_allowed(self, ts4, slots4) -> bool:
        """The hold: after a DiSCO restart a four-slot map is sent only once its ts is newer than the
        restart's first ping AND it carries none of the ids of the last map sent before it."""
        now = float(self.clock())
        msg = None
        with self._lock:
            if self.episode > self._hold_episode:
                self._hold_episode = self.episode
                self.hold = dict(t=float(self.episode_t), blocked=self.last_sent_rids,
                                 episode=self.episode, n=0, warned=None)
            h = self.hold
            if h is None:
                return True
            rids = set()
            for v in (slots4 or {}).values():
                try:
                    rids.add(int(v[0]))
                except Exception:
                    pass
            try:
                newer = ts4 is not None and float(ts4) > h["t"]
            except (TypeError, ValueError):
                newer = False
            if newer and not (rids & h["blocked"]):
                self.hold = None
                if h["warned"] is not None:
                    msg = (f"[lanes] four-slot map released {now - h['t']:.0f} s after the DiSCO restart "
                           f"(new ids {sorted(rids)})")
                ok = True
            else:
                h["n"] += 1
                age = now - h["t"]
                if age >= DISCO_HOLD_WARN_S and (h["warned"] is None or now - h["warned"] >= DISCO_HOLD_REMIND_S):
                    h["warned"] = now
                    msg = (f"[lanes] four-slot map HELD {age:.0f} s after the DiSCO restart: the lanes driver "
                           f"has not re-issued the slots yet (old ids {sorted(h['blocked'])} are never re-sent)")
                ok = False
        if msg:
            print(msg, flush=True)
        return ok

    def note_map_sent(self, slots4) -> None:
        with self._lock:
            self.last_sent_rids = frozenset(int(v[0]) for v in (slots4 or {}).values())


# # — THREAD 2: read compositions.db and emit JSON packets —
# def composition_sender(ser, hz, comp_db_path):
#     pause = 1.0/hz

#     def read_all():
#         conn = sqlite3.connect(comp_db_path); cur = conn.cursor()
#         cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='timestamp'")
#         if not cur.fetchone():
#             conn.close(); return None
#         cur.execute("SELECT ts FROM timestamp LIMIT 1"); row = cur.fetchone()
#         if not row:
#             conn.close(); return None
#         ts = row[0]
#         def tbl(n): 
#             cur.execute(f"SELECT * FROM {n}"); return cur.fetchall()
#         start = tbl("start"); end = tbl("end"); comps = tbl("compositions")
#         conn.close()
#         if not (start and end and comps):
#             return None
#         return ts, start[0], end[0], comps

#     while True:
#         data = read_all()
#         if data:
#             ts, start, end, comps = data
#             pkt = {
#               "type":"composition",
#               "ts":ts,
#               "start": list(start),
#               "end":   list(end),
#               "comps": [list(r) for r in comps]
#             }
#             ser.write((json.dumps(pkt)+"\n").encode())
#             print(f"[composition_sender] sent ts={ts}")
#         time.sleep(pause)


# # ─── THREAD: read compositions.db (including caches) and emit JSON packets ──
# def composition_sender(ser, hz, comp_db_path, chaos=False, verbose=True):
#     """
#     Every 1/hz seconds, polls compositions.db and, once all six tables are
#     non‐empty, emits one JSON packet over `ser` with the six pieces.
#     """
#     ser.timeout = 1.0 / hz
#     pause = 1.0 / hz
#     next_send = time.monotonic()

#     def read_all():
#         try:
#             conn = sqlite3.connect(comp_db_path)
#             cur  = conn.cursor()
#             # timestamp
#             cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='timestamp'")
#             if not cur.fetchone():
#                 return None
#             cur.execute("SELECT ts FROM timestamp LIMIT 1")
#             trow = cur.fetchone()
#             if not trow:
#                 return None
#             ts = trow[0]

#             # helper
#             def fetch(tbl):
#                 cur.execute(f"SELECT * FROM {tbl}")
#                 return cur.fetchall()

#             start = fetch("start")
#             end   = fetch("end")
#             comps = fetch("compositions")
#             sc    = fetch("start_cache")
#             ec    = fetch("end_cache")
#             cc    = fetch("compositions_cache")
#             conn.close()

#             if not (start and end and comps and sc and ec and cc):
#                 return None

#             return ts, start[0], end[0], comps, sc[0], ec[0], cc
#         except:
#             return None

#     while True:
#         # throttle to exactly hz
#         now = time.monotonic()
#         if now < next_send:
#             time.sleep(next_send - now)
#         next_send += pause

#         data = read_all()
#         if not data:
#             continue

#         ts, start, end, comps, sc, ec, cc = data
#         if chaos:
#             ts = np.random.randint(1,10)

#         pkt = {
#             "type":               "composition",
#             "ts":                  ts,
#             "start":              list(start),
#             "end":                list(end),
#             "comps":              [list(r) for r in comps],
#             "start_cache":        list(sc),
#             "end_cache":          list(ec),
#             "compositions_cache": [list(r) for r in cc]
#         }
#         raw = (json.dumps(pkt) + "\n").encode()

#         # write under the same lock
#         try:
#             with _serial_lock:
#                 ser.write(raw)
#             if verbose:
#                 print(f"[composition_sender] sent ts={ts}")
#         except serial.SerialException as e:
#             if verbose:
#                 print(f"[composition_sender] SerialException on write: {e!r}, resetting output buffer.")
#             try:
#                 with _serial_lock:
#                     ser.reset_output_buffer()
#             except:
#                 pass
#             # retry next cycle


# ─── Helper to write a 1D objective row ─────────────────────────────────────
def _write_objective_row(vals, db_path, log_skip: bool = False):
    """
    Write objective values to the `objective` table.
    If the table already has data, check if the new data is different from memory DB.
    Only allow overwrite if data is actually new.

    NOTE: this deliberately does NOT raise the handshake flag. The objective values
    and the compositions matrix are two separate writes; raising the flag here let
    `get_y_measurements` read new y values against the *previous* batch's
    compositions table, which then got zero-padded (or silently truncated) to
    length. The caller must raise the flag via `_set_objective_ready()` only after
    every table for the packet has landed.

    Returns True if rows were written, False if the write was skipped as a duplicate.
    """
    global _objective_writing
    
    with _objective_db_lock:  # Acquire lock to prevent race conditions
        _objective_writing = True  # Set flag to indicate writing
        try:
            conn = sqlite3.connect(db_path, timeout=30.0)
            cur  = conn.cursor()

            try:
                # Check if table exists and has the right structure
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='objective'")
                table_exists = cur.fetchone() is not None
                
                if not table_exists:
                    # Create table if it doesn't exist
                    cur.execute("CREATE TABLE objective (val REAL)")
                else:
                    # Check if we need to alter the table structure
                    cur.execute("PRAGMA table_info(objective)")
                    columns = cur.fetchall()
                    if len(columns) != 1 or columns[0][1] != 'val':
                        # Table structure is wrong, recreate it
                        cur.execute("DROP TABLE objective")
                        cur.execute("CREATE TABLE objective (val REAL)")
                
                # Check if table already has data
                cur.execute("SELECT COUNT(*) FROM objective")
                count = cur.fetchone()[0]
                
                if count > 0:
                    # Table has data - check if new data is different from memory DB
                    try:
                        mem_db_path = "./sql/objective_memory.db"
                        if os.path.exists(mem_db_path):
                            mem_conn = sqlite3.connect(mem_db_path, timeout=10.0)
                            mem_cur = mem_conn.cursor()
                            mem_cur.execute("SELECT * FROM objective")
                            mem_rows = mem_cur.fetchall()
                            
                            if mem_rows:
                                # Convert memory data to comparable format
                                if len(mem_rows) == 1 and len(mem_rows[0]) > 1:
                                    mem_flat = np.array(mem_rows[0], dtype=float)
                                elif len(mem_rows) > 1 and len(mem_rows[0]) == 1:
                                    mem_flat = np.array([r[0] for r in mem_rows], dtype=float)
                                else:
                                    mem_flat = np.array([], dtype=float)
                                
                                # Convert incoming data to comparable format
                                incoming_flat = np.array(vals, dtype=float)
                                
                                # Check if data is the same
                                if mem_flat.shape == incoming_flat.shape and np.allclose(mem_flat, incoming_flat, rtol=1e-6, atol=1e-8):
                                    # Data is identical to memory, skip write to prevent race condition
                                    if log_skip:
                                        print(f"[_write_objective_row] Data identical to memory DB, skipping write to prevent race condition")
                                    mem_conn.close()
                                    conn.close()
                                    return False
                                else:
                                    # Data is different, allow overwrite
                                    if log_skip:
                                        print(f"[_write_objective_row] Data different from memory DB, allowing overwrite")
                                    mem_conn.close()
                            else:
                                # Memory DB is empty, allow overwrite
                                if log_skip:
                                    print(f"[_write_objective_row] Memory DB empty, allowing overwrite")
                                mem_conn.close()
                        else:
                            # Memory DB doesn't exist, allow overwrite
                            if log_skip:
                                print(f"[_write_objective_row] Memory DB doesn't exist, allowing overwrite")
                    except Exception as e:
                        # If there's an error checking memory DB, allow overwrite to be safe
                        if log_skip:
                            print(f"[_write_objective_row] Error checking memory DB: {e}, allowing overwrite")
                
                # Clear existing data and insert new data
                cur.execute("DELETE FROM objective")
                rows = [(None if v is None else float(v),) for v in vals]
                cur.executemany("INSERT INTO objective (val) VALUES (?)", rows)

                conn.commit()
                if log_skip:
                    print(f"[_write_objective_row] Successfully wrote {len(rows)} values to objective table")
                return True

            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
        finally:
            _objective_writing = False  # Clear flag when done writing

# ─── Helper to write an n×M compositions matrix ────────────────────────────
def _write_compositions_matrix(mat, db_path):
    """
    Replace the `compositions` table with `mat`.

    Holds `_objective_db_lock` / `_objective_writing` for the same reason
    `_write_objective_row` does: the table is dropped and rebuilt, so a reader that
    lands mid-write sees either an absent table or a partial one. Combined with the
    handshake flag now being raised only after this returns, `get_y_measurements`
    can no longer pair a new y vector with a stale compositions table.
    """
    global _objective_writing

    mat = np.asarray(mat, dtype=float)
    n, M = mat.shape
    with _objective_db_lock:
        _objective_writing = True
        try:
            conn = sqlite3.connect(db_path, timeout=30.0)
            cur  = conn.cursor()
            try:
                cur.execute("DROP TABLE IF EXISTS compositions")
                cols = ", ".join(f"c{i} REAL" for i in range(M))
                cur.execute(f"CREATE TABLE compositions ({cols})")
                ph = ", ".join("?" for _ in range(M))
                cur.executemany(f"INSERT INTO compositions VALUES ({ph})", mat.tolist())
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        finally:
            _objective_writing = False


# ─── Helper to drop a stale compositions table ─────────────────────────────
def _drop_compositions_table(db_path):
    """Remove the `compositions` table so a reader cannot pair fresh y values with
    a previous packet's compositions."""
    global _objective_writing

    with _objective_db_lock:
        _objective_writing = True
        try:
            conn = sqlite3.connect(db_path, timeout=30.0)
            cur  = conn.cursor()
            try:
                cur.execute("DROP TABLE IF EXISTS compositions")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        finally:
            _objective_writing = False


# ─── Helper to raise the "new objective ready" handshake ───────────────────
def _set_objective_ready(db_path):
    """
    Raise `handshake.new_objective_available` for a fully-written packet.

    Must be called only after BOTH the objective row and (when the packet carried
    one) the compositions matrix have committed — this is the barrier that makes
    the two-table write appear atomic to `get_y_measurements`.
    """
    with _objective_db_lock:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cur  = conn.cursor()
        try:
            cur.execute("""CREATE TABLE IF NOT EXISTS handshake (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                new_objective_available INTEGER DEFAULT 0
            )""")
            cur.execute("INSERT OR IGNORE INTO handshake (id, new_objective_available) VALUES (1, 0)")
            cur.execute("UPDATE handshake SET new_objective_available = 1 WHERE id = 1")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ─── THREAD: receive JSON objective+comps & write to DB ─────────────────────
def objective_receiver(hz, obj_db_path, mem_db_path, verbose=False, super_verbose=False,
                       queue_db_path=None, link=None):
    # queue_db_path (lanes, additive): where every NEW objective packet is also appended
    # (default: objective_queue.db next to obj_db_path, i.e. ./sql/objective_queue.db).
    # link (lanes, additive): a DiscoRestartLink; DiSCO selftest_ping packets are recorded in
    # it (one 'disco_restart' link event per DiSCO start). None: today's behaviour.
    pause = 1.0 / hz
    consecutive_errors = 0
    max_consecutive_errors = 10
    last_activity_time = time.time()
    max_idle_time = 1800  # 30 minutes without any activity

    print(f"[objective_receiver] ENTERED FUNCTION (hz={hz}, obj_db_path={obj_db_path}, mem_db_path={mem_db_path}, verbose={verbose})")
    if verbose:
        print(f"[objective_receiver] Started - waiting for objective data...")
        print(f"[objective_receiver] Will timeout after {max_idle_time} seconds of inactivity")

    while not _shutdown_flag.is_set():
        if super_verbose:
            print(f"[objective_receiver] LOOP TOP - running, _shutdown_flag={_shutdown_flag.is_set()}, time={time.time()}")
        try:
            # Use a shorter timeout to check shutdown flag more frequently
            try:
                if super_verbose:
                    print(f"[objective_receiver] Waiting for data in _read_queue (size={_read_queue.qsize()}) with timeout={min(pause, 0.5)}")
                raw = _read_queue.get(timeout=min(pause, 0.5))
                last_activity_time = time.time()  # Reset activity timer
                if super_verbose:
                    print(f"[objective_receiver] Got data from _read_queue: {len(raw)} bytes")
            except queue.Empty:
                if super_verbose:
                    print(f"[objective_receiver] _read_queue EMPTY after timeout, checking idle time...")
                # Check if we've been idle too long
                if time.time() - last_activity_time > max_idle_time:
                    print(f"[objective_receiver] ⚠️  No data received for {max_idle_time} seconds")
                    print(f"[objective_receiver] This might indicate:")
                    print(f"   - External device is not sending data")
                    print(f"   - Serial connection issue")
                    print(f"   - Wrong baud rate or port")
                    print(f"[objective_receiver] Continuing to wait...")
                    last_activity_time = time.time()  # Reset timer
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
                if super_verbose:
                    print(f"[objective_receiver] Decoded line: {line!r}")
            except Exception as e:
                print(f"[objective_receiver] Decode error: {e}")
                consecutive_errors += 1
                continue

            # parse JSON
            try:
                pkt = json.loads(line)
                if super_verbose:
                    print(f"[objective_receiver] Parsed JSON: {pkt.get('type', 'unknown')}, keys={list(pkt.keys())}")
            except json.JSONDecodeError as e:
                print(f"[objective_receiver] ⚠ JSON decode failed: {e}")
                consecutive_errors += 1
                continue

            # Lanes (additive): log every NEW objective packet, raw, to the queue the lanes
            # driver polls. Own de-dup and own transaction; never raises, never skips the
            # legacy handling below.
            _queue_objective_packet(line, pkt, obj_db_path, queue_db_path)

            # Lanes (additive, G2): a DiSCO selftest_ping = DiSCO (re)started (DiscoRestartLink).
            # Recorded only; the packet is still skipped just below, as before.
            if (link is not None and isinstance(pkt, dict) and pkt.get("type") == "selftest_ping"
                    and pkt.get("from") == "disco"):
                link.note_ping(pkt, "live")

            if pkt.get("type") != "objective" or "values" not in pkt:
                if super_verbose:
                    print(f"[objective_receiver] ⚠ Not an objective packet, skipping (type={pkt.get('type')}, keys={list(pkt.keys())})")
                continue

            # ── validate incoming values ──────────────────────────────────
            try:
                vals = np.asarray(pkt["values"], dtype=float)
            except (ValueError, TypeError) as e:
                if verbose:
                    print(f"[objective_receiver] ⚠ Could not convert values to float array, ignoring: {e}")
                continue

            if vals.ndim != 1 or vals.size == 0:
                if verbose:
                    print(f"[objective_receiver] ⚠ Unexpected values shape {vals.shape}, ignoring")
                continue

            if not np.isfinite(vals).any():
                if verbose:
                    print(f"[objective_receiver] ⚠ All-NaN/inf packet ({vals.size} values), ignoring")
                continue
            # ─────────────────────────────────────────────────────────────

            comps = pkt.get("comps", None)
            ts = pkt.get("ts", None)

            # Dedup so we only process/write once per distinct packet.
            global _last_objective_vals, _last_objective_comps, _last_objective_ts
            if ts is not None:
                # Timestamp handshake (mirrors composition_receiver): DiSCO bumps
                # objective.db's ts on every write, so identical ~1 Hz re-sends carry
                # the SAME ts. Ingest only when the ts is new; this stops the receiver
                # from continually overwriting objective.db / objective_memory.db with
                # the same data. New data (new ts) is always processed, so the loop
                # never stalls on genuinely-new results.
                if _last_objective_ts is not None and ts == _last_objective_ts:
                    continue
            else:
                # Legacy fallback (packet has no ts): value-based dedup as before.
                try:
                    comps_match = (comps is None and _last_objective_comps is None) or (
                        comps is not None and _last_objective_comps is not None
                        and len(comps) == len(_last_objective_comps)
                        and all(np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(comps, _last_objective_comps))
                    )
                except Exception:
                    comps_match = False
                vals_match = (
                    _last_objective_vals is not None
                    and vals.shape == _last_objective_vals.shape
                    and np.allclose(vals, _last_objective_vals, rtol=1e-6, atol=1e-8)
                )
                if vals_match and comps_match:
                    continue
            _last_objective_ts = ts
            _last_objective_vals = vals.copy()
            _last_objective_comps = comps if comps is None else [np.asarray(c).copy() for c in comps]

            if verbose:
                print(f"[objective_receiver] ▶ Received raw data: {len(raw)} bytes")
            if super_verbose:
                print(f"[objective_receiver] Processing objective values: {vals}, comps: {type(comps)}")

            # memory‑diff with better error handling
            memory_diff = True  # Default to True to ensure processing
            try:
                if os.path.exists(mem_db_path):
                    if super_verbose:
                        print(f"[objective_receiver] Checking memory DB at {mem_db_path}")
                    conn = sqlite3.connect(mem_db_path, timeout=10.0)
                    c = conn.cursor()
                    c.execute("SELECT * FROM objective")
                    mem = c.fetchall()
                    conn.close()
                    if super_verbose:
                        print(f"[objective_receiver] Memory DB fetchall: {mem}")
                    # Convert mem to a flat array if not empty
                    if mem and len(mem) > 0:
                        if len(mem) == 1 and len(mem[0]) > 1:
                            mem_flat = np.array(mem[0], dtype=float)
                        elif len(mem) > 1 and len(mem[0]) == 1:
                            mem_flat = np.array([r[0] for r in mem], dtype=float)
                        else:
                            mem_flat = np.array([], dtype=float)
                    else:
                        mem_flat = np.array([], dtype=float)
                    if super_verbose and mem_flat.shape == vals.shape and mem_flat.size > 0:
                        memory_diff = not np.allclose(mem_flat, vals, rtol=1e-6, atol=1e-8)
                        if super_verbose:
                            print(f"[objective_receiver] memory_diff result: {memory_diff}")
            except Exception as e:
                print(f"[objective_receiver] memory DB error: {e!r}")
                memory_diff = True

            # objective‑empty with better error handling
            obj_empty = True  # Default to True to ensure processing
            try:
                if os.path.exists(obj_db_path):
                    if super_verbose:
                        print(f"[objective_receiver] Checking objective DB at {obj_db_path}")
                    conn = sqlite3.connect(obj_db_path, timeout=10.0)
                    c = conn.cursor()
                    c.execute("SELECT * FROM objective")
                    obj = c.fetchall()
                    conn.close()
                    if super_verbose:
                        print(f"[objective_receiver] Objective DB fetchall: {obj}")
                        print(f"[objective_receiver] obj_empty: {obj_empty}")
            except Exception as e:
                print(f"[objective_receiver] objective DB error: {e!r}")
                obj_empty = True
            if super_verbose:
                print(f"[objective_receiver] memory_diff={memory_diff}, obj_empty={obj_empty}")
            if memory_diff and obj_empty:
                # Write objective + compositions, THEN raise the handshake flag.
                # The flag is the barrier that tells get_y_measurements the packet is
                # complete, so nothing may raise it until every table has landed.
                success = False
                for attempt in range(3):
                    try:
                        if super_verbose:
                            print(f"[objective_receiver] Writing objective data (attempt {attempt+1}): {vals}")
                        _write_objective_row(vals.tolist(), obj_db_path, log_skip=False)
                        _write_objective_row(vals.tolist(), mem_db_path, log_skip=False)
                        if verbose:
                            print("[objective_receiver] ✔ wrote objective", vals)
                        success = True
                        consecutive_errors = 0  # Reset error counter on success
                        break
                    except Exception as e:
                        print(f"[objective_receiver] ERROR writing objective (attempt {attempt+1}): {e!r}")
                        time.sleep(0.1)  # Brief pause before retry
                if not success:
                    print(f"[objective_receiver] Failed to write objective after 3 attempts.")
                    consecutive_errors += 1

                # write compositions if present
                comps_ok = True
                has_comps = isinstance(comps, list) and comps and isinstance(comps[0], list)
                if has_comps:
                    try:
                        if super_verbose:
                            print(f"[objective_receiver] Writing compositions matrix: {len(comps)}x{len(comps[0]) if comps else 0}")
                        _write_compositions_matrix(comps, obj_db_path)
                        if verbose: print(f"[objective_receiver] ✔ wrote comps ({len(comps)}×{len(comps[0])})")
                    except Exception as e:
                        print(f"[objective_receiver] ERROR writing comps: {e!r}")
                        comps_ok = False
                        consecutive_errors += 1
                else:
                    # No compositions in this packet. Drop any table left over from the
                    # previous packet so the reader cannot silently pair these y values
                    # against the last batch's compositions.
                    try:
                        _drop_compositions_table(obj_db_path)
                        print("[objective_receiver] WARNING packet carried no comps; dropped stale compositions table")
                    except Exception as e:
                        print(f"[objective_receiver] ERROR dropping stale comps: {e!r}")
                        comps_ok = False

                if success and comps_ok:
                    try:
                        _set_objective_ready(obj_db_path)
                        if super_verbose:
                            print("[objective_receiver] handshake raised (packet complete)")
                    except Exception as e:
                        print(f"[objective_receiver] ERROR raising handshake: {e!r}")
                        consecutive_errors += 1
                        # Flag not raised -> allow the next ~1 Hz re-send of this same
                        # packet through the ts dedup so the write can be retried,
                        # rather than waiting forever for genuinely new data.
                        _last_objective_ts = None
                else:
                    # Partial write: leave the flag down and re-arm the dedup so the
                    # re-send retries. A consumer must never see half a packet.
                    print("[objective_receiver] WARNING partial packet write; handshake NOT raised, will retry on re-send")
                    _last_objective_ts = None
            else:
                if super_verbose:
                    print(f"[objective_receiver] skip (mem_diff={memory_diff}, empty={obj_empty})")
                    if not obj_empty:
                        print(f"[objective_receiver] Table not empty, waiting for get_y_measurements to clear it")

            # Check for too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                print(f"[objective_receiver] Too many consecutive errors ({consecutive_errors}), pausing...")
                time.sleep(5.0)  # Longer pause to let system recover
                consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            try:
                print(f"[objective_receiver] Unexpected error: {e!r}")
            except UnicodeEncodeError:
                print(f"[objective_receiver] Unexpected error: {ascii(repr(e))}")
            time.sleep(1.0)
    if super_verbose:
        print("[objective_receiver] Exiting main loop (shutdown flag set)")
        print("[objective_receiver] Shutdown complete")


# ─── THREAD: read compositions.db & emit JSON ────────────────────────────────
def composition_sender(ser, hz, comp_db_path, chaos=False, verbose=True, super_verbose=False,
                       link=None):
    # link (lanes, additive): a DiscoRestartLink. Used ONLY while slots4 is active: answer a
    # restarted DiSCO's self-test ping and hold the four-slot map until the lanes driver has
    # re-issued it under new ids. The legacy branch never touches it.
    pause = 1.0 / hz
    next_send = time.monotonic()
    consecutive_errors = 0
    max_consecutive_errors = 10
    lanes_write_failed = False   # lanes four-slot path only
    four_slot_on = None          # four-slot active at the last readable _slots4_state (None: no such read yet)

    def answer_disco_ping():
        # Lanes only (G2): a restarted DiSCO is in its gated self-test -> answer its newest ping.
        nonlocal lanes_write_failed
        link.flush()
        pong = link.take_pong()
        if pong is None:
            return
        frame = selftest_pong_frame(pong[0])
        if lanes_write_failed:
            frame = b"\n" + frame
        try:
            with _serial_lock:
                ser.write(frame)
                ser.flush()
            lanes_write_failed = False
        except serial.SerialException as e:
            lanes_write_failed = True
            print(f"[composition_sender] self-test reply write err: {e!r}")
            try:
                with _serial_lock:
                    ser.reset_output_buffer()
            except Exception:
                pass

    def read_all():
        try:
            if not os.path.exists(comp_db_path):
                if super_verbose:
                    print(f"[composition_sender] DB {comp_db_path} does not exist.")
                return None
            conn = sqlite3.connect(comp_db_path, timeout=10.0)
            cur  = conn.cursor()
            # timestamp
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='timestamp'")
            if not cur.fetchone(): 
                if super_verbose:
                    print(f"[composition_sender] No timestamp table found.")
                conn.close()
                return None
            cur.execute("SELECT ts FROM timestamp LIMIT 1")
            row = cur.fetchone()
            if not row: 
                if super_verbose:
                    print(f"[composition_sender] No timestamp row found.")
                conn.close()
                return None
            ts = row[0]
            # fetch all six
            def f(t): 
                cur.execute(f"SELECT * FROM {t}")
                return cur.fetchall()
            start, end    = f("start"), f("end")
            comps         = f("compositions")
            sc, ec, cc    = f("start_cache"), f("end_cache"), f("compositions_cache")
            conn.close()
            if not (start and end and comps and sc and ec and cc): 
                if super_verbose:
                    print(f"[composition_sender] Not all required tables have data.")
                return None
            if super_verbose:
                print(f"[composition_sender] Read all tables successfully.")
            return ts, start[0], end[0], comps, sc[0], ec[0], cc
        except Exception as e:
            print(f"[composition_sender] Database read error: {e}")
            return None

    while not _shutdown_flag.is_set():
        try:
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.1))  # Check shutdown more frequently
                continue
            next_send += pause

            # Lanes (additive): while slots4_meta.active == 1 send the four-slot full-state
            # packet instead of today's; otherwise fall through to today's path unchanged.
            state4, lanes = _slots4_state(comp_db_path)
            if state4 == SLOTS4_UNREADABLE:
                if lanes or four_slot_on is not False:
                    # Review B1: compositions.db unreadable (lock, Dropbox) while four-slot is active,
                    # was active at the last readable read, or is not known to be off yet -> skip
                    # this tick. Falling through would put the stale two-slot (axial-init) proposal
                    # on the wire: it bypasses the G2 hold, DiSCO could print it, and it flips
                    # DiSCO's link back to legacy slots. The 1 Hz loop retries at the next tick.
                    if lanes:
                        four_slot_on = True
                    if link is not None and four_slot_on:
                        answer_disco_ping()
                    continue
                lanes = None               # four-slot known to be off: today's path, unchanged
            else:
                four_slot_on = state4 == SLOTS4_ACTIVE
            if lanes is not None:
                ts4, slots4 = lanes
                if link is not None:
                    # G2: a restarted DiSCO is in its gated self-test -> answer it; and never put an
                    # id from before the restart back on the wire (map_allowed).
                    answer_disco_ping()
                    if not link.map_allowed(ts4, slots4):
                        continue
                raw = _lanes_wire().encode_slots(ts4, slots4, full=True)
                if lanes_write_failed:
                    raw = b"\n" + raw      # resync DiSCO after a torn frame (lanes only)
                try:
                    with _serial_lock:
                        ser.write(raw)
                        ser.flush()
                    lanes_write_failed = False
                    consecutive_errors = 0
                    if link is not None:
                        link.note_map_sent(slots4)
                    if super_verbose:
                        print(f"[composition_sender] sent four-slot ts={ts4} slots={sorted(slots4)}")
                except serial.SerialException as e:
                    lanes_write_failed = True
                    consecutive_errors += 1
                    print(f"[composition_sender] four-slot write err: {e!r}, resetting output buffer")
                    try:
                        with _serial_lock:
                            ser.reset_output_buffer()
                    except Exception:
                        pass
                continue

            data = read_all()
            if not data:
                if super_verbose:
                    print(f"[composition_sender] No data to send this cycle.")
                continue

            ts, start, end, comps, sc, ec, cc = data
            if chaos:
                ts = np.random.randint(1,10)

            pkt = {
                "type":               "composition",
                "ts":                  ts,
                "start":             list(start),
                "end":               list(end),
                "comps":             [list(r) for r in comps],
                "start_cache":       list(sc),
                "end_cache":         list(ec),
                "compositions_cache":[list(r) for r in cc]
            }
            raw = (json.dumps(pkt) + "\n").encode()
            if super_verbose:
                print(f"[composition_sender] Sending packet: {pkt}")
            try:
                with _serial_lock:
                    ser.write(raw)
                    ser.flush()  # Ensure data is sent immediately
                if super_verbose:
                    print(f"[composition_sender] Packet sent successfully. Queue size: {_read_queue.qsize()}")
                    print(f"[composition_sender] sent ts={ts}")
                    print(f"[composition_sender] start={start}, end={end}, comps={comps}, sc={sc}, ec={ec}, cc={cc}")
                consecutive_errors = 0  # Reset error counter on success
            except serial.SerialException as e:
                consecutive_errors += 1
                print(f"[composition_sender] write err: {e!r}, resetting output buffer")
                try: 
                    with _serial_lock:
                        ser.reset_output_buffer()
                except: 
                    pass
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[composition_sender] Too many consecutive errors ({consecutive_errors}), pausing...")
                    time.sleep(5.0)  # Longer pause to let system recover
                    consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"[composition_sender] Unexpected error: {e!r}")
            time.sleep(1.0)
    if super_verbose:
        print("[composition_sender] Shutdown complete")


# a single queue for all incoming bytes
_read_queue = queue.Queue()


def serial_link_selftest(ser, role, peer, timeout=120.0, ping_period=0.5, grace=2.0,
                         on_peer_ping=None):
    """
    Bidirectional serial connectivity self-test, run BEFORE the normal reader/sender
    threads so it has the port to itself. Both peers run the SAME routine (differing
    role/peer): each repeatedly sends a small JSON 'selftest_ping' and echoes any ping
    it receives from the peer as a 'selftest_pong'. This side PASSES once it receives a
    pong — which proves BOTH directions (our ping reached the peer AND the peer's reply
    reached us). All non-selftest traffic is ignored, and the input buffer is flushed
    on exit so leftover test bytes never reach the real receivers.

    role/peer: 'zombi'/'disco' (this side / the other). Returns True on success.
    Both programs must be launched within `timeout` seconds of each other.

    on_peer_ping (lanes, additive): called with each ping packet received from the peer (a
    DiSCO ping here means DiSCO is starting too; DiscoRestartLink records it). Exceptions in it
    are ignored; None: today's behaviour.
    """
    print(f"[selftest:{role}] bidirectional link test (peer={peer}, timeout={timeout:.0f}s)…")
    saved_timeout = ser.timeout
    try:
        ser.timeout = 0.2
    except Exception:
        pass
    deadline = time.time() + timeout
    buf = bytearray()
    got_ping = False   # received peer's ping  -> our RX works
    got_pong = False   # received peer's pong  -> full round trip (our TX + RX)
    pong_time = None
    seq = 0
    last_send = 0.0
    try:
        while time.time() < deadline:
            if got_pong and pong_time is None:
                pong_time = time.time()
            if pong_time is not None and time.time() - pong_time > grace:
                break
            now = time.time()
            if now - last_send >= ping_period:
                seq += 1
                try:
                    # Pad to ~3.5 KB so the test exercises the SAME multi-chunk reassembly
                    # path the real ~3 KB objective packets use (a tiny ping would not
                    # reproduce a large-packet framing/buffering failure).
                    ser.write((json.dumps({"type": "selftest_ping", "from": role, "seq": seq, "pad": "x" * 3500}) + "\n").encode())
                    ser.flush()
                except Exception as e:
                    print(f"[selftest:{role}] write error: {e!r}")
                last_send = now
            try:
                chunk = ser.read(256)
            except Exception as e:
                print(f"[selftest:{role}] read error: {e!r}")
                chunk = b""
            if chunk:
                buf.extend(chunk)
                if len(buf) > 65536:            # resync guard against a garbled stream
                    buf = buf[-4096:]
                while True:
                    nl = buf.find(b"\n")
                    if nl == -1:
                        break
                    line = bytes(buf[:nl]); buf = buf[nl + 1:]
                    try:
                        pkt = json.loads(line.decode("utf-8", "ignore").strip())
                    except Exception:
                        continue
                    t, frm = pkt.get("type"), pkt.get("from")
                    if t == "selftest_ping" and frm == peer:
                        got_ping = True
                        if on_peer_ping is not None:
                            try:
                                on_peer_ping(pkt)
                            except Exception:
                                pass
                        try:
                            ser.write((json.dumps({"type": "selftest_pong", "from": role, "ack": pkt.get("seq"), "pad": "x" * 3500}) + "\n").encode())
                            ser.flush()
                        except Exception:
                            pass
                    elif t == "selftest_pong" and frm == peer:
                        got_pong = True
    finally:
        try:
            ser.timeout = saved_timeout
        except Exception:
            pass
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
    if got_pong:
        print(f"[selftest:{role}] ✅ PASS — bidirectional link OK (rx_ping={got_ping}, roundtrip={got_pong})")
    else:
        print(f"[selftest:{role}] ❌ FAIL — no round trip in {timeout:.0f}s "
              f"(rx_ping={got_ping}, roundtrip={got_pong}). Check baud/port and that the "
              f"{peer} process is running; both sides must start within the timeout window.")
    return got_pong


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────
def start_serial_dual_io_shared_port(COM, baud,
                                     obj_hz=5.0,
                                     comp_hz=1.0,
                                     obj_db ="./sql/objective.db",
                                     mem_db ="./sql/objective_memory.db",
                                     comp_db="./sql/compositions.db",
                                     chaos=False,
                                     verbose=True,
                                     super_verbose=False,
                                     selftest=True,
                                     parent_shutdown: Optional[Any] = None):
    """
    parent_shutdown: optional multiprocessing.Event; when set (by the parent
    process), this routine shuts down threads and releases the serial port
    so the next run can open COM again (Windows / spawn-safe).
    """

    # Ensure SQL directory exists
    os.makedirs(os.path.dirname(obj_db), exist_ok=True)

    # ── cleanup infrastructure FIRST ──────────────────────────────────────────
    # Everything that closes the port is installed before the port is opened, and
    # therefore before the (up to 120 s) link self-test. Registering it after the
    # self-test — as this used to — left a long window in which the port was open
    # with no signal handler and no atexit hook, so a Stop / Ctrl-Break during the
    # link test killed the process with COM still open and orphaned the port.
    ser = None
    _threads: list[tuple[threading.Thread, str]] = []
    _cleanup_done = False

    def _cleanup():
        _shutdown_flag.set()
        for t, label in _threads:
            if t is not None and t.is_alive():
                t.join(timeout=4.0)
                if t.is_alive():
                    print(f"[Machine2] [{label}] still alive after join timeout")
        time.sleep(0.15)
        try:
            if ser and ser.is_open and hasattr(ser, "cancel_read"):
                ser.cancel_read()
        except Exception:
            pass
        # Drop any half-received frame so the next session does not inherit a
        # partial packet from this one.
        for buf_reset in ("reset_input_buffer", "reset_output_buffer"):
            try:
                if ser and ser.is_open:
                    getattr(ser, buf_reset)()
            except Exception:
                pass
        try:
            if ser and ser.is_open:
                ser.close()
        except Exception as e:
            print(f"[Machine2] Error closing serial: {e!r}")
        print(f"[Machine2] PORT_CLOSED {COM}", flush=True)
        print("[Machine2] Serial port released; cleanup complete", flush=True)

    def _cleanup_once():
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        _cleanup()

    atexit.register(_cleanup_once)
    _register_signal_handlers_in_serial_process_only()

    # 1) open the port with retry logic
    max_port_retries = 5
    released_stale = False
    for attempt in range(max_port_retries):
        if _shutdown_flag.is_set():
            print(f"[Machine2] Stop requested while waiting for {COM}; giving up.")
            break
        try:
            ser = serial.Serial(COM, baud, timeout=0.1, write_timeout=5.0)
            print(f"[Machine2] Opened {COM}@{baud}")
            print(f"[Machine2] PORT_OPEN {COM}@{baud}", flush=True)
            break
        except serial.SerialException as e:
            # "Access is denied" means some other process still holds the port —
            # in practice a ZoMBI serial process that outlived its parent. Kill it
            # instead of making the user unplug the adapter. Tried once, early, so
            # the remaining retries can succeed.
            if not released_stale and _is_access_denied(e):
                released_stale = True
                print(f"[Machine2] {COM} is held by another process — attempting to "
                      f"release it (this replaces unplugging the adapter)…", flush=True)
                try:
                    from scripts.com_port import release_port
                    release_port(COM, log=lambda m: print(m, flush=True))
                except Exception as rel_err:
                    print(f"[Machine2] Could not auto-release {COM}: {rel_err!r}")
                continue
            if attempt < max_port_retries - 1:
                print(f"[Machine2] Failed to open {COM} (attempt {attempt+1}): {e}")
                time.sleep(2.0)
            else:
                print(f"[Machine2] Failed to open {COM} after {max_port_retries} attempts: {e}")

    if ser is None:
        print(f"[Machine2] PORT_FAIL {COM}", flush=True)
        print("[Machine2] Could not open serial port")
        atexit.unregister(_cleanup_once)
        return

    def _parent_requests_stop() -> bool:
        if parent_shutdown is None:
            return False
        try:
            return bool(parent_shutdown.is_set())
        except Exception:
            return False

    # 3) launch a single reader thread that does nothing but read_until('\n') and push into our queue
    def _serial_reader():
        buf = bytearray()
        consecutive_read_errors = 0
        max_read_errors = 10
        if super_verbose:
            print("[serial_reader] Thread started.")
        while not _shutdown_flag.is_set():
            try:
                if super_verbose:
                    print("[serial_reader] Reading from serial port...")
                chunk = ser.read(256)   # up to 256 bytes, blocks until timeout
                if chunk:
                    if super_verbose:
                        print(f"[serial_reader] Read chunk: {chunk!r}")
                    consecutive_read_errors = 0  # Reset error counter on successful read
                    buf.extend(chunk)
                    # Safety: if a corrupted stream loses a newline, buf would grow
                    # forever and the reader would silently stall (never emitting a
                    # line). Cap it: on overflow with no newline, drop the stale bytes
                    # and resync on the next packet boundary rather than hang.
                    if len(buf) > 262144 and buf.find(b'\n') == -1:
                        print(f"[serial_reader] ⚠️ {len(buf)} bytes buffered with no newline; "
                              f"resyncing (dropping stale bytes)")
                        buf = bytearray()
                    # extract all complete lines
                    while True:
                        nl = buf.find(b'\n')
                        if nl == -1:
                            break
                        line = bytes(buf[:nl+1])  # include the newline
                        if super_verbose:
                            print(f"[serial_reader] Got line: {line!r}")
                        _read_queue.put(line)
                        if super_verbose:
                            print(f"[serial_reader] Put line into _read_queue (size now: {_read_queue.qsize()})")
                        buf = buf[nl+1:]
                else:
                    # No data read, but not an error
                    if super_verbose:
                        print("[serial_reader] No data read from serial port.")
                    time.sleep(0.01)
            except (serial.SerialException, OSError) as e:
                consecutive_read_errors += 1
                print(f"[serial_reader] Read error: {e}")
                if consecutive_read_errors >= max_read_errors:
                    print(f"[serial_reader] Too many read errors, stopping")
                    break
                time.sleep(0.1)
            except Exception as e:
                consecutive_read_errors += 1
                print(f"[serial_reader] Unexpected error: {e}")
                time.sleep(0.1)
        if super_verbose:
            print("[serial_reader] Shutdown complete")
    
    # Everything from here on runs with the port open, so it all lives under one
    # try/finally: no exit path — self-test failure, a dead thread, an unhandled
    # exception, a signal — can leave COM held.
    try:
        # 2) Verify the link works BOTH ways before starting the real reader/sender
        # threads; a silent one-directional failure otherwise deadlocks the loop
        # (ZoMBI blocks in get_y_measurements waiting for an objective it never
        # received). Gated: exit if it fails, so the failure is loud rather than a
        # silent stall. The port is released by the finally below.
        # Lanes (G2): DiSCO-restart bookkeeping for this serial session. A DiSCO ping seen by the
        # startup self-test (DiSCO starting at the same time) is recorded too, written after LINK_OK.
        disco_link = DiscoRestartLink(
            queue_path=os.path.join(os.path.dirname(obj_db) or ".", "objective_queue.db"))
        if selftest and not serial_link_selftest(
                ser, role="zombi", peer="disco",
                on_peer_ping=lambda p: disco_link.note_ping(p, "startup_selftest", defer=True)):
            print("[Machine2] LINK_FAIL", flush=True)
            print("[Machine2] ❌ connectivity self-test FAILED — releasing port and exiting "
                  "(fix the link, or pass selftest=False to bypass).")
            return
        print("[Machine2] LINK_OK", flush=True)
        disco_link.flush()

        reader_thread = threading.Thread(target=_serial_reader, daemon=True)
        reader_thread.start()
        _threads.append((reader_thread, "serial_reader"))

        # 4) launch your existing two workers
        t_o = threading.Thread(
            target=objective_receiver,
            args=(obj_hz, obj_db, mem_db, verbose),
            kwargs={"link": disco_link},
            daemon=True
        )
        t_c = threading.Thread(
            target=composition_sender,
            args=(ser, comp_hz, comp_db, chaos, verbose),
            kwargs={"link": disco_link},
            daemon=True
        )

        t_o.start()
        t_c.start()
        _threads.append((t_o, "objective_receiver"))
        _threads.append((t_c, "composition_sender"))

        # 5) block until shutdown or error
        while not _shutdown_flag.is_set() and not _parent_requests_stop():
            # Check if threads are still alive
            if not reader_thread.is_alive():
                print("[Machine2] Serial reader thread died")
                break
            if not t_o.is_alive():
                print("[Machine2] Objective receiver thread died")
                break
            if not t_c.is_alive():
                print("[Machine2] Composition sender thread died")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Machine2] KeyboardInterrupt, shutting down…")
    except Exception as e:
        print(f"[Machine2] Unexpected error: {e}")
    finally:
        if _parent_requests_stop() and not _shutdown_flag.is_set():
            print("[Machine2] Parent requested shutdown; closing serial port…")
        _cleanup_once()
        sys.exit(0)