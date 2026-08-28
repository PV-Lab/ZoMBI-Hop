"""
visualization/db_histogram.py
=============================
Histogram the value distributions of a ZoMBI-Hop results database.

Point it at a ``.db`` (or a campaign ``.csv``) and it draws one histogram panel
per column -- by default every objective-side column the file actually populated
(``Objective``, ``Bandgap``, ``Photoconductance``, ``Stability``), optionally the
composition columns too -- and prints the matching summary statistics.

Run it with no ``--db`` and it opens a file picker in ``data/``, so "plot the
distribution of that run" is two clicks.

Why it does not import ``plot_run``
-----------------------------------
``plot_run`` already has the column-detection logic below, but importing it pulls
in sklearn and ``src.utils.datahandler`` -> torch: ~14 s, and torch's global
defaults get mutated on the way in. That is a poor trade for a script needing
only sqlite and matplotlib, so the detection is duplicated here deliberately. It
mirrors ``plot_run.detect_comp_columns`` / ``plot_run.db_value_columns`` and
should be kept in step with them.

Per-column null handling
------------------------
Each column is loaded independently, dropping only *its own* nulls. A run where
the stability assay failed on half the samples therefore still shows the full
``Objective`` histogram, rather than every panel being truncated to the rows that
survived in all of them. The dropped count is reported per column.

Usage
-----
  uv run python visualization/db_histogram.py
  uv run python visualization/db_histogram.py --db data/6d.db
  uv run python visualization/db_histogram.py --db data/6d.db --list
  uv run python visualization/db_histogram.py --db data/3d.db --col Objective
  uv run python visualization/db_histogram.py --db data/6d.db --comp
  uv run python visualization/db_histogram.py --db data/4d.db --col Bandgap,Stability --bins 60
  uv run python visualization/db_histogram.py --db data/6d.db --out hist.png

Flags
-----
  --db PATH      Data file (.db results table or .csv campaign). Omit for a picker.
  --col COLS     Comma-separated columns to plot (repeatable). Default: the
                 populated value columns.
  --comp         Also plot the detected composition columns.
  --all          Plot every populated numeric column (careful: spectra).
  --list         Print the available columns and exit.
  --bins N       Bin count, or "auto" for numpy's rule (default: 40).
  --log          Log-scale the counts axis.
  --out PATH     Save to PATH instead of opening a window.
  --dpi N        Output resolution (default: 150).
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
DATA_DIR = ROOT / "data"

# Mirrors plot_run.DB_VALUE_COLS / _COMP_SPAN / _MODULE_RE.
DB_VALUE_COLS: list[str] = ["Objective", "Bandgap", "Photoconductance", "Stability"]
_COMP_SPAN = ("Iteration", "X")
_MODULE_RE = re.compile(r"module\d+$", re.IGNORECASE)


# -- file resolution ----------------------------------------------------------

def _is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def _resolve_db_path(db_arg: str) -> Path:
    """Accept a full path, a name under ``data/``, or a bare stem."""
    candidate = Path(db_arg)
    if candidate.is_file():
        return candidate.resolve()
    for stem in (db_arg, f"{db_arg}.db", f"{db_arg}.csv"):
        candidate = DATA_DIR / stem
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(f"Data file not found: {db_arg}")


def _pick_db_path() -> Path:
    """Open a file dialog in ``data/``; fall back to listing it on the console."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        found = sorted(DATA_DIR.glob("*.db")) + sorted(DATA_DIR.glob("*.csv"))
        listing = "\n".join(f"  {p.relative_to(ROOT)}" for p in found) or "  (none)"
        raise SystemExit(
            f"No file picker available (tkinter missing). Pass --db.\n"
            f"Found in data/:\n{listing}")
    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askopenfilename(
        title="Select a ZoMBI-Hop results database",
        initialdir=str(DATA_DIR if DATA_DIR.is_dir() else ROOT),
        filetypes=[("Results database", "*.db"), ("Campaign CSV", "*.csv"),
                   ("All files", "*.*")])
    root.destroy()
    if not chosen:
        raise SystemExit("No file selected.")
    return Path(chosen).resolve()


# -- reading ------------------------------------------------------------------

def table_columns(path: Path) -> list[str]:
    """Column names of the file's table, in declaration order."""
    if _is_csv(path):
        import pandas as pd
        return list(pd.read_csv(path, nrows=0).columns)
    con = sqlite3.connect(str(path))
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(results)")]
    finally:
        con.close()
    if not cols:
        raise SystemExit(f"{path} has no 'results' table -- is this a ZoMBI-Hop db?")
    return cols


def load_column(path: Path, col: str) -> np.ndarray:
    """Finite values of one column, nulls and non-numeric entries dropped."""
    if _is_csv(path):
        import pandas as pd
        v = pd.to_numeric(pd.read_csv(path, usecols=[col]).iloc[:, 0],
                          errors="coerce").to_numpy(dtype=float)
        return v[np.isfinite(v)]
    con = sqlite3.connect(str(path))
    try:
        rows = con.execute(
            f'SELECT "{col}" FROM results WHERE "{col}" IS NOT NULL').fetchall()
    finally:
        con.close()
    out = []
    for (x,) in rows:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=float)


def n_rows(path: Path) -> int:
    if _is_csv(path):
        import pandas as pd
        return int(len(pd.read_csv(path, usecols=[0])))
    con = sqlite3.connect(str(path))
    try:
        return int(con.execute("SELECT COUNT(*) FROM results").fetchone()[0])
    finally:
        con.close()


# -- column detection ---------------------------------------------------------

def detect_comp_columns(path: Path) -> list[str]:
    """The composition columns the file actually varied, in table order.

    The candidates are the hardware module slots, which sit contiguously between
    ``Iteration`` and ``X``. A slot that kept its ``ModuleN`` placeholder name, or
    that is all zeros, was never loaded with a precursor. This is what makes
    3d.db, 4d.db and 6d.db report d = 3, 4 and 6 respectively rather than a fixed
    width.
    """
    cols = table_columns(path)
    try:
        lo = cols.index(_COMP_SPAN[0]) + 1
        hi = cols.index(_COMP_SPAN[1])
    except ValueError:
        return []
    keep = []
    for c in cols[lo:hi]:
        if _MODULE_RE.match(c):
            continue
        v = load_column(path, c)
        if v.size and np.abs(v).max() > 0:
            keep.append(c)
    return keep


def detect_value_columns(path: Path) -> list[str]:
    """Those of :data:`DB_VALUE_COLS` present in the file and holding data."""
    have = set(table_columns(path))
    present = [c for c in DB_VALUE_COLS if c in have]
    populated = [c for c in present if load_column(path, c).size]
    return populated or present


def numeric_columns(path: Path) -> list[str]:
    """Every column that parses as numeric and holds at least one value."""
    return [c for c in table_columns(path) if load_column(path, c).size]


# -- statistics and output ----------------------------------------------------

def summarise(values: np.ndarray, total: int) -> dict:
    nan = float("nan")
    if values.size == 0:
        return {"n": 0, "dropped": total, "mean": nan, "std": nan, "min": nan,
                "p25": nan, "median": nan, "p75": nan, "max": nan}
    return {
        "n": int(values.size),
        "dropped": int(total - values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else nan,
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def print_report(path: Path, columns: list[str], stats: dict[str, dict],
                 total: int) -> None:
    print(f"{path}  ({total} rows)")
    head = (f"  {'column':<20}{'n':>6}{'dropped':>9}{'mean':>12}{'std':>12}"
            f"{'min':>12}{'median':>12}{'max':>12}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for col in columns:
        s = stats[col]
        print(f"  {col:<20}{s['n']:>6}{s['dropped']:>9}{s['mean']:>12.4g}"
              f"{s['std']:>12.4g}{s['min']:>12.4g}{s['median']:>12.4g}"
              f"{s['max']:>12.4g}")


def plot_histograms(path: Path, columns: list[str], data: dict[str, np.ndarray],
                    stats: dict[str, dict], *, bins, log: bool,
                    out: Path | None, dpi: int) -> None:
    import matplotlib
    if out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(columns)
    ncols = 1 if n == 1 else (2 if n <= 4 else 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, col in zip(flat, columns):
        s = stats[col]
        ax.hist(data[col], bins=bins, color="#4878CF", edgecolor="black",
                linewidth=0.4)
        ax.axvline(s["mean"], color="crimson", linestyle="--", linewidth=1.4,
                   label=f"mean = {s['mean']:.4g}")
        ax.axvline(s["median"], color="darkorange", linestyle=":", linewidth=1.4,
                   label=f"median = {s['median']:.4g}")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("samples")
        ax.set_title(f"{col}  (n = {s['n']}, {s['dropped']} dropped)", fontsize=10)
        ax.legend(fontsize=8)
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{path.name} - value distributions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"\nSaved -> {out}")
    else:
        plt.show()


# -- entry point --------------------------------------------------------------

def _select_columns(path: Path, args: argparse.Namespace) -> list[str]:
    if args.col:
        columns = [c.strip() for spec in args.col
                   for c in spec.split(",") if c.strip()]
        have = set(table_columns(path))
        missing = [c for c in columns if c not in have]
        if missing:
            raise SystemExit(f"Columns not in {path.name}: {', '.join(missing)}\n"
                             f"Run with --list to see what is available.")
    elif args.all_cols:
        columns = numeric_columns(path)
    else:
        columns = detect_value_columns(path)
    if args.comp:
        comp = [c for c in detect_comp_columns(path) if c not in set(columns)]
        columns = comp + columns
    return columns


def main() -> None:
    # Shared code in this repo prints non-ASCII; cp1252 stdio turns that into a
    # mid-run UnicodeEncodeError on Windows.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Histogram the value distributions of a ZoMBI-Hop results db.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None,
                        help="Data file (.db or .csv). Omit to open a file picker.")
    parser.add_argument("--col", action="append", default=None,
                        help="Comma-separated columns to plot (repeatable).")
    parser.add_argument("--comp", action="store_true",
                        help="Also plot the detected composition columns.")
    parser.add_argument("--all", action="store_true", dest="all_cols",
                        help="Plot every populated numeric column.")
    parser.add_argument("--list", action="store_true", dest="list_cols",
                        help="Print the available columns and exit.")
    parser.add_argument("--bins", default="40",
                        help='Bin count, or "auto" (default: 40).')
    parser.add_argument("--log", action="store_true", help="Log-scale the counts.")
    parser.add_argument("--out", default=None,
                        help="Save to PATH instead of showing a window.")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output resolution (default: 150).")
    args = parser.parse_args()

    path = _resolve_db_path(args.db) if args.db else _pick_db_path()
    total = n_rows(path)

    if args.list_cols:
        comp = detect_comp_columns(path)
        value = detect_value_columns(path)
        known = set(comp) | set(value)
        other = [c for c in numeric_columns(path) if c not in known]
        print(f"{path}  ({total} rows)")
        print(f"  composition (d={len(comp)}): {', '.join(comp) or '(none detected)'}")
        print(f"  value:                       {', '.join(value) or '(none)'}")
        print(f"  other populated numeric:     {len(other)} columns"
              + (f", e.g. {', '.join(other[:8])}" if other else ""))
        return

    columns = _select_columns(path, args)
    if not columns:
        raise SystemExit(f"Nothing to plot in {path.name}; try --list or --all.")

    data = {c: load_column(path, c) for c in columns}
    empty = [c for c in columns if data[c].size == 0]
    if empty:
        print(f"Skipping all-null columns: {', '.join(empty)}", file=sys.stderr)
        columns = [c for c in columns if c not in set(empty)]
    if not columns:
        raise SystemExit("Every requested column is empty.")

    stats = {c: summarise(data[c], total) for c in columns}
    print_report(path, columns, stats, total)

    bins = "auto" if str(args.bins).lower() == "auto" else int(args.bins)
    plot_histograms(path, columns, data, stats, bins=bins, log=args.log,
                    out=Path(args.out).resolve() if args.out else None, dpi=args.dpi)


if __name__ == "__main__":
    main()
