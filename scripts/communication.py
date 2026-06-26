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


def _serial_process_signal_handler(signum, frame):
    """Only used in the dedicated serial I/O process (not at import: avoids every child
    that `import communication` from stealing SIGINT; ZoMBI does not read _shutdown_flag)."""
    print(f"[SerialIO] Signal {signum} — closing serial path…")
    _shutdown_flag.set()


def _register_signal_handlers_in_serial_process_only() -> None:
    """Call once from `start_serial_dual_io_shared_port` in the serial subprocess."""
    try:
        signal.signal(signal.SIGINT, _serial_process_signal_handler)
        signal.signal(signal.SIGTERM, _serial_process_signal_handler)
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
    Only allow overwrite if data is actually new. Set handshake flag when new data is written.
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
                                    return
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
                
                # Set handshake flag
                cur.execute('''CREATE TABLE IF NOT EXISTS handshake (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    new_objective_available INTEGER DEFAULT 0
                )''')
                cur.execute('INSERT OR IGNORE INTO handshake (id, new_objective_available) VALUES (1, 0)')
                cur.execute('UPDATE handshake SET new_objective_available = 1 WHERE id = 1')
                
                conn.commit()
                if log_skip:
                    print(f"[_write_objective_row] Successfully wrote {len(rows)} values to objective table and set handshake flag")
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
        finally:
            _objective_writing = False  # Clear flag when done writing

# ─── Helper to write an n×M compositions matrix ────────────────────────────
def _write_compositions_matrix(mat, db_path):
    mat = np.asarray(mat, dtype=float)
    n, M = mat.shape
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS compositions")
    cols = ", ".join(f"c{i} REAL" for i in range(M))
    cur.execute(f"CREATE TABLE compositions ({cols})")
    ph = ", ".join("?" for _ in range(M))
    cur.executemany(f"INSERT INTO compositions VALUES ({ph})", mat.tolist())
    conn.commit()
    conn.close()


# ─── THREAD: receive JSON objective+comps & write to DB ─────────────────────
def objective_receiver(hz, obj_db_path, mem_db_path, verbose=False, super_verbose=False):
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

            # Dedup: only process and print once per distinct packet (avoid filling terminal)
            global _last_objective_vals, _last_objective_comps
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
                # write to both DBs with retry logic
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
                if isinstance(comps, list) and comps and isinstance(comps[0], list):
                    try:
                        if super_verbose:
                            print(f"[objective_receiver] Writing compositions matrix: {len(comps)}x{len(comps[0]) if comps else 0}")
                        _write_compositions_matrix(comps, obj_db_path)
                        if verbose: print(f"[objective_receiver] ✔ wrote comps ({len(comps)}×{len(comps[0])})")
                    except Exception as e:
                        print(f"[objective_receiver] ERROR writing comps: {e!r}")
                        consecutive_errors += 1
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
def composition_sender(ser, hz, comp_db_path, chaos=False, verbose=True, super_verbose=False):
    pause = 1.0 / hz
    next_send = time.monotonic()
    consecutive_errors = 0
    max_consecutive_errors = 10
    last_sent_ts = None  # only transmit when the composition actually changes

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

            data = read_all()
            if not data:
                if super_verbose:
                    print(f"[composition_sender] No data to send this cycle.")
                continue

            ts, start, end, comps, sc, ec, cc = data
            if chaos:
                ts = np.random.randint(1,10)

            # Skip if this composition was already sent. Re-transmitting the
            # same multi-KB packet every cycle overruns the serial link (at
            # 9600 baud a single packet can take seconds to drain), which fills
            # the OS write buffer and triggers SerialTimeoutException('Write
            # timeout'). A new ts is written by the GUI on every "Update", so
            # this still sends immediately whenever the composition changes.
            if ts == last_sent_ts:
                if super_verbose:
                    print(f"[composition_sender] ts={ts} unchanged, skipping send.")
                continue

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
                last_sent_ts = ts  # mark sent only after a successful write
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
                                     parent_shutdown: Optional[Any] = None):
    """
    parent_shutdown: optional multiprocessing.Event; when set (by the parent
    process), this routine shuts down threads and releases the serial port
    so the next run can open COM again (Windows / spawn-safe).
    """

    # Ensure SQL directory exists
    os.makedirs(os.path.dirname(obj_db), exist_ok=True)

    # 1) open the port with retry logic
    ser = None
    max_port_retries = 5
    for attempt in range(max_port_retries):
        try:
            ser = serial.Serial(COM, baud, timeout=0.1, write_timeout=5.0)
            print(f"[Machine2] Opened {COM}@{baud}")
            break
        except serial.SerialException as e:
            if attempt < max_port_retries - 1:
                print(f"[Machine2] Failed to open {COM} (attempt {attempt+1}): {e}")
                time.sleep(2.0)
            else:
                print(f"[Machine2] Failed to open {COM} after {max_port_retries} attempts: {e}")
                return

    if ser is None:
        print("[Machine2] Could not open serial port")
        return

    _register_signal_handlers_in_serial_process_only()

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
    
    reader_thread = threading.Thread(target=_serial_reader, daemon=True)
    reader_thread.start()

    # 4) launch your existing two workers
    t_o = threading.Thread(
        target=objective_receiver,
        args=(obj_hz, obj_db, mem_db, verbose),
        daemon=True
    )
    t_c = threading.Thread(
        target=composition_sender,
        args=(ser, comp_hz, comp_db, chaos, verbose),
        daemon=True
    )
    
    t_o.start()
    t_c.start()

    def _cleanup():
        _shutdown_flag.set()
        for t, label in ((reader_thread, "serial_reader"), (t_o, "objective_receiver"), (t_c, "composition_sender")):
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
        try:
            if ser and ser.is_open:
                ser.close()
        except Exception as e:
            print(f"[Machine2] Error closing serial: {e!r}")
        print("[Machine2] Serial port released; cleanup complete")

    _cleanup_done = False

    def _cleanup_once():
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        _cleanup()

    atexit.register(_cleanup_once)

    # 5) block until shutdown or error
    try:
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