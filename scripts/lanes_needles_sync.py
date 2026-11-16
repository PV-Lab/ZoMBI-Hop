"""Mirror a LANES run's needles into sql/needles.db (what DiSCO's visualization.py draws as stars).

The lanes driver (scripts/lanes_hw.py) declares needles in its engine's DataHandler, whose snapshots
(<run>/lanes/engine/run_*/snapshots/<latest>/needles.json) hold every needle; the legacy optimizer mirrored its
needles into sql/needles.db, the lanes driver did not until 2026-11-15. This tool fills that gap for a running
lanes process that predates the fix: it re-reads the newest engine snapshot and rewrites sql/needles.db (sorted
best-first, src/utils/needles_db.write_needles) whenever the needle set changes.

    python scripts/lanes_needles_sync.py --run runs/run_1d1f            # loop (every 60 s)
    python scripts/lanes_needles_sync.py --run runs/run_1d1f --once     # one sync

Read-only on the run; writes only sql/needles.db (in place). Never raises inside the loop.

It compares the needle set with what the table actually holds (so an external wipe, e.g. main.py's reset on a
start, heals at the next tick) and never replaces it with FEWER needles than it has already seen (a restarted
lanes process points latest.txt at its 0001_init snapshot, 0 needles, for its first seconds) or than the table
holds (a lanes driver with the built-in mirror may have just added one its snapshot does not show yet). Once a
lanes process with the built-in mirror (needles_db=True, 2026-11-15) runs, stop this tool: the driver keeps the
table itself.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import needles_db as ndb  # noqa: E402
from src.utils.needles_db import read_run_dir_needles, write_needles  # noqa: E402


def engine_dir(run: Path) -> Path | None:
    cands = [Path(p) for p in glob.glob(str(run / "lanes" / "engine" / "run_*")) if (Path(p) / "latest.txt").exists()]
    return max(cands, key=lambda p: (p / "latest.txt").stat().st_mtime) if cands else None


def run_dims(run: Path) -> list:
    meta = run / "lanes" / "meta.json"
    dims = json.loads(meta.read_text(encoding="utf-8")).get("opt_dims") if meta.exists() else None
    if not dims:
        raise RuntimeError(f"no opt_dims in {meta}")
    return [int(d) for d in dims]


def signature(recs: list) -> tuple:
    return tuple((round(float(r.get("value") or 0), 9), tuple(round(float(v), 9) for v in (r.get("point") or [])))
                 for r in recs if isinstance(r, dict))


def expected_rows(recs: list, dims: list, n_cols: int = ndb.N_COMP_COLS) -> list:
    """The (10 compositions, objective) rows write_needles would store for ``recs``, best first."""
    rows = []
    for r in recs:
        pt = ndb._needle_point(r)
        if pt is None:
            continue
        full = [0.0] * n_cols
        if len(pt) == n_cols:
            full = [float(v) for v in pt]
        else:
            for j, d in enumerate(dims):
                if j < len(pt) and 0 <= d < n_cols:
                    full[d] = float(pt[j])
        val = ndb._finite_or_none(r.get("value")) if isinstance(r, dict) else None
        rows.append((tuple(round(v, 9) for v in full), None if val is None else round(val, 9)))
    rows.sort(key=lambda x: (x[1] is not None, x[1] if x[1] is not None else 0.0), reverse=True)
    return rows


def db_rows(db: str | None, n_cols: int = ndb.N_COMP_COLS):
    """What the table holds now, as expected_rows() shapes it; None when it cannot be read."""
    p = Path(db) if db else ndb.DEFAULT_NEEDLES_DB_PATH
    if not p.exists():
        return None
    try:
        con = sqlite3.connect(str(p), timeout=10.0)
        try:
            cols = ", ".join(f"col_{i}" for i in range(n_cols))
            got = con.execute(f"SELECT {cols}, objective_value FROM needles ORDER BY rank").fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return [(tuple(round(float(v or 0.0), 9) for v in r[:n_cols]),
             None if r[n_cols] is None else round(float(r[n_cols]), 9)) for r in got]


def sync_once(run: Path, db: str | None = None, last: list | None = None):
    """One sync. ``last``: the richest needle list seen so far (None at the start). Returns (state, n written
    or None when nothing was written: the table already holds the set, or it holds MORE rows than the set - a
    live mirror (the lanes driver's own, needles_db=True) may have just added a needle the snapshot does not
    show yet, so the table is never downgraded)."""
    ed = engine_dir(run)
    best = list(last or [])
    if ed is not None:
        recs = read_run_dir_needles(ed)
        if len(recs) >= len(best):
            best = recs
    if not best:
        return best, None
    dims = run_dims(run)
    want = expected_rows(best, dims)
    held = db_rows(db)
    if held == want:
        return best, None
    if held is not None and len(held) > len(want):
        return best, None
    n = write_needles(best, dims, path=db)
    return best, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--db", default=None, help="needles.db path (default: <repo>/sql/needles.db)")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--hours", type=float, default=100.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    last, end = None, time.time() + a.hours * 3600
    while True:
        try:
            last, n = sync_once(run, a.db, last)
            if n is not None:
                print(f"{time.strftime('%H:%M:%S')} needles.db <- {n} needle(s) from {run.name}", flush=True)
        except Exception as e:  # noqa: BLE001 - a mirror must never stop
            print(f"{time.strftime('%H:%M:%S')} needles sync failed (will retry): {e!r}", flush=True)
        if a.once or time.time() > end:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
