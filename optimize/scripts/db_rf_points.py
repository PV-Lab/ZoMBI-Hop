"""One property of a hardware campaign DB as RF training points, for ``--dataset runrf``.

A run directory only stores the scalar objective. The campaign DB (e.g. data/6d.db
for run_69af) also stores the per-property scores (Bandgap, Photoconductance,
Stability). This reads ``--value`` from the DB's ``results`` table, puts the
composition columns in the run's ``optimizing_dims`` order (the order the run's own
X uses), drops rows where that value is missing, and writes one ``.npz`` (``X``,
``Y``). ``evaluate.py --dataset runrf --gp-run <file>.npz`` then fits its 500-tree
forest to those points exactly as it does to a run directory.

    uv run python optimize/scripts/db_rf_points.py --db data/6d.db --run runs/run_69af \\
        --value Photoconductance --out analysis/run69af_photoconductance_points
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# The DB's ten hardware module slots sit contiguously between these two columns;
# slot k is hardware dim k (visualization/plot_run._COMP_SPAN).
_COMP_SPAN = ("Iteration", "X")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="campaign SQLite DB (results table)")
    ap.add_argument("--run", required=True, metavar="RUN_DIR",
                    help="run dir whose optimizing_dims fix the column order")
    ap.add_argument("--value", required=True, help="DB column to use as Y, e.g. Photoconductance")
    ap.add_argument("--out", required=True, help="output directory for the .npz")
    a = ap.parse_args()

    run = Path(a.run).resolve()
    dims = json.loads((run / "live_plot_state.json").read_text())["optimizing_dims"]

    with sqlite3.connect(f"file:{Path(a.db).resolve()}?mode=ro", uri=True) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(results)")]
        slots = cols[cols.index(_COMP_SPAN[0]) + 1: cols.index(_COMP_SPAN[1])]
        if len(slots) <= max(dims):
            sys.exit(f"{a.db} has {len(slots)} module slots; run dims are {dims}")
        if a.value not in cols:
            sys.exit(f"{a.value!r} is not a column of {a.db}")
        comp = [slots[d] for d in dims]
        q = ", ".join(f'"{c}"' for c in comp + [a.value])
        rows = list(conn.execute(f"SELECT {q} FROM results ORDER BY rowid"))

    arr = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
    X, Y = arr[:, :-1], arr[:, -1]
    ok = np.isfinite(Y) & np.isfinite(X).all(axis=1) & (X.sum(axis=1) > 0)
    X, Y = X[ok], Y[ok]
    X = X / X.sum(axis=1, keepdims=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run.name}_{a.value.lower()}_N{len(X)}.npz"
    np.savez(path, X=X, Y=Y, db=str(a.db), run=str(run), value=a.value,
             columns=np.array(comp), dims=np.array(dims))
    print(f"  {a.db}: {len(rows)} rows, {len(X)} with {a.value}  "
          f"(columns {comp} = run dims {dims})")
    print(f"  Y: min {Y.min():.4g}  median {np.median(Y):.4g}  max {Y.max():.4g}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
