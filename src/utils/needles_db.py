"""
src/utils/needles_db.py
=======================
``sql/needles.db`` — a small SQLite table mirroring the needles ZoMBI-Hop has
discovered in the current hardware run, so downstream tools/operators can read
the best compositions found so far.

Each row is one needle expressed in the *full* hardware composition space
(``col_0 … col_9``; components outside the run's optimizing dims are 0), plus:

  * ``objective_value``          — the needle's peak objective value.
  * ``radial_objective_value``   — the needle's radial (basin) median objective:
                                   the median objective over all raw measurements
                                   within the needle's paring radius. May be NULL.

Rows are always ordered by ``objective_value``, highest first (``rank`` 0 = best).

The table is fully rewritten from the current needle list on every change, so it
is always correctly sorted and consistent no matter whether a needle is added,
preloaded on resume, or (should the optimizer ever gain the ability) removed.
Because a campaign only ever holds a handful of needles, the full rewrite is
cheap and far less error-prone than incremental in-place edits.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Sequence

# Full hardware composition width. Matches the ``col_0 … col_9`` layout of
# compositions.db / objective.db (see scripts/initialize_databases.py).
N_COMP_COLS = 10

# Anchor the DB under <repo>/sql/needles.db regardless of the caller's CWD, so
# the parent launcher (scripts/main.py) and the ZoMBI subprocess agree on one
# file — alongside compositions.db / objective.db.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEEDLES_DB_PATH = _REPO_ROOT / "sql" / "needles.db"


def _resolve_path(path: Optional[str | Path]) -> Path:
    return Path(path) if path is not None else DEFAULT_NEEDLES_DB_PATH


def _col_names(n_cols: int = N_COMP_COLS) -> List[str]:
    return [f"col_{i}" for i in range(n_cols)]


def _create_table(cur: sqlite3.Cursor, n_cols: int = N_COMP_COLS) -> None:
    col_defs = ", ".join(f"{c} REAL" for c in _col_names(n_cols))
    cur.execute(
        "CREATE TABLE IF NOT EXISTS needles ("
        f"rank INTEGER PRIMARY KEY, {col_defs}, "
        "objective_value REAL, radial_objective_value REAL)"
    )


def _finite_or_none(x: Any) -> Optional[float]:
    """Coerce to float, mapping None / NaN / non-numeric to None (SQL NULL)."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(xf) else xf


def _needle_point(rec: Any) -> Optional[List[float]]:
    """Extract a needle's composition vector (optimizing-dim space) as a list.

    Accepts a dict with a ``point`` key (torch tensor, ndarray, or list) or a
    bare vector. Returns None if no usable point is present.
    """
    pt = rec.get("point") if isinstance(rec, dict) else rec
    if pt is None:
        return None
    if hasattr(pt, "detach"):          # torch.Tensor
        pt = pt.detach().cpu().numpy()
    try:
        return [float(v) for v in list(pt)]
    except (TypeError, ValueError):
        return None


def reset_needles_db(path: Optional[str | Path] = None,
                     n_cols: int = N_COMP_COLS) -> None:
    """Drop and recreate an empty ``needles`` table (a hard reset)."""
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS needles")
        _create_table(cur, n_cols)
        conn.commit()
    finally:
        conn.close()


def write_needles(needles: Sequence[Any],
                  optimizing_dims: Sequence[int],
                  path: Optional[str | Path] = None,
                  n_cols: int = N_COMP_COLS) -> int:
    """Rewrite the whole ``needles`` table from ``needles``, sorted best-first.

    ``needles``        — sequence of records, each a dict with keys ``point``
                         (composition over ``optimizing_dims``), ``value`` and
                         ``median_value`` (e.g. ``DataHandler.needles_results`` or
                         a checkpoint ``needles.json``).
    ``optimizing_dims``— hardware column indices the run optimises (e.g. ``[0,8,9]``);
                         a needle's point is scattered into these columns and the
                         rest of the 10-wide row is left at 0.

    Rows are ordered by ``objective_value`` descending (NULL/NaN last), so the
    first row (``rank`` 0) is always the highest-objective needle. Returns the
    number of rows written.
    """
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dims = [int(d) for d in optimizing_dims]

    rows: List[tuple] = []   # (full_comp: list[float], value, radial)
    for rec in (needles or []):
        pt = _needle_point(rec)
        if pt is None:
            continue
        full = [0.0] * n_cols
        if len(pt) == n_cols:
            full = [float(v) for v in pt]
        else:
            for j, d in enumerate(dims):
                if j < len(pt) and 0 <= d < n_cols:
                    full[d] = float(pt[j])
        val = _finite_or_none(rec.get("value")) if isinstance(rec, dict) else None
        rad = _finite_or_none(rec.get("median_value")) if isinstance(rec, dict) else None
        rows.append((full, val, rad))

    # Highest objective value first; needles with no value (NULL) sort last.
    rows.sort(key=lambda r: (r[1] is not None, r[1] if r[1] is not None else 0.0),
              reverse=True)

    col_names = _col_names(n_cols)
    placeholders = ", ".join(["?"] * (1 + n_cols + 2))
    insert_sql = (
        f"INSERT INTO needles (rank, {', '.join(col_names)}, "
        f"objective_value, radial_objective_value) VALUES ({placeholders})"
    )

    conn = sqlite3.connect(str(p), timeout=30.0)
    try:
        cur = conn.cursor()
        _create_table(cur, n_cols)
        cur.execute("DELETE FROM needles")          # clear rows, keep schema
        for rank, (full, val, rad) in enumerate(rows):
            cur.execute(insert_sql, [rank] + full + [val, rad])
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def read_run_dir_needles(run_dir: Optional[str | Path],
                         snapshot: Optional[str] = None) -> List[dict]:
    """Read needle records from a run directory's snapshots (latest by default).

    Returns the list parsed from ``snapshots/<snap>/needles.json`` (records with
    ``point``, ``value``, ``median_value``, …), or ``[]`` when the run/snapshot/
    file is absent or unreadable.
    """
    if run_dir is None:
        return []
    run_dir = Path(run_dir)
    snap_root = run_dir / "snapshots"
    if not snap_root.exists():
        return []

    snap_name = snapshot
    if snap_name is None:
        latest = run_dir / "latest.txt"
        if latest.exists():
            try:
                snap_name = latest.read_text().strip()
            except OSError:
                snap_name = None
    if snap_name is None or not (snap_root / snap_name).exists():
        snaps = sorted(d.name for d in snap_root.iterdir() if d.is_dir())
        snap_name = snaps[-1] if snaps else None
    if snap_name is None:
        return []

    nj = snap_root / snap_name / "needles.json"
    if not nj.exists():
        return []
    try:
        data = json.loads(nj.read_text())
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def load_checkpoint_needles(checkpoint_dir: Optional[str | Path],
                            run_uuid: Optional[str],
                            snapshot: Optional[str] = None) -> List[dict]:
    """Read a run's needle records from its checkpoint (latest snapshot by default).

    Convenience wrapper over :func:`read_run_dir_needles` that locates the run by
    ``<checkpoint_dir>/run_<run_uuid>``.
    """
    if checkpoint_dir is None or run_uuid is None:
        return []
    return read_run_dir_needles(Path(checkpoint_dir) / f"run_{run_uuid}", snapshot)


def read_run_dims(checkpoint_dir: Optional[str | Path],
                  run_uuid: Optional[str]) -> Optional[List[int]]:
    """Read a run's optimizing-dim indices from hw_config.json / config.json.

    Mirrors the GUI's dim lookup; returns e.g. ``[0, 8, 9]`` or None if unknown.
    Used as a fallback when the launcher was not given explicit ``--dims``.
    """
    if checkpoint_dir is None or run_uuid is None:
        return None
    run_dir = Path(checkpoint_dir) / f"run_{run_uuid}"
    for fname in ("hw_config.json", "config.json"):
        p = run_dir / fname
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        dims_raw = data.get("dims", "")
        parsed = [int(x.strip()) for x in str(dims_raw).split(",")
                  if x.strip().lstrip("-").isdigit()]
        if parsed:
            return parsed
    return None
