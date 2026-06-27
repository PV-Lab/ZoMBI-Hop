"""
scripts/db_to_csv.py
====================
Export a ZoMBI-Hop results database (e.g. ``data/second_run.db``) to a
campaign-style CSV (same layout as ``data/campaign1a.csv``).

The database is a SQLite file with a single ``results`` table whose columns
are one row per measured sample: the composition columns (FAPbI3, MAPbI3,
MAPbBr3, ...), derived properties (Bandgap, Photoconductance, Stability,
Objective), environmental sensors, and the full reflectance / PL spectra.
This script simply dumps that table to a CSV.

Usage
-----
  python scripts/db_to_csv.py data/second_run.db
      -> writes data/second_run.csv

  python scripts/db_to_csv.py runs/whatever/foo.db --out data/my_campaign.csv
  python scripts/db_to_csv.py data/second_run.db --table results
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
DATA_DIR = ROOT / "data"


def _pick_table(conn: sqlite3.Connection, requested: str | None) -> str:
    """Resolve which table to export, defaulting to 'results'."""
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    if not tables:
        raise ValueError("Database contains no tables.")
    if requested:
        if requested not in tables:
            raise ValueError(
                f"Table {requested!r} not found. Available: {', '.join(tables)}"
            )
        return requested
    if "results" in tables:
        return "results"
    if len(tables) == 1:
        return tables[0]
    raise ValueError(
        f"No 'results' table; specify one with --table. Available: {', '.join(tables)}"
    )


def db_to_csv(db_path: Path, out_path: Path, table: str | None = None) -> Path:
    """Export ``table`` from ``db_path`` to ``out_path`` as CSV. Returns out_path."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        table_name = _pick_table(conn, table)
        # Quote the table name in case it is non-identifier-safe.
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {out_path}")
    return out_path


def _default_out(db_path: Path) -> Path:
    """Default CSV destination: data/<db-stem>.csv."""
    return DATA_DIR / f"{db_path.stem}.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, help="Path to the .db file to export.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/<db-stem>.csv).",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Table to export (default: 'results', or the only table present).",
    )
    args = parser.parse_args(argv)

    out_path = args.out if args.out is not None else _default_out(args.db)
    try:
        db_to_csv(args.db, out_path, args.table)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
