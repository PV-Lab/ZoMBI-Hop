"""
visualization/ilr_vs_comp_oob.py
==================================
Head-to-head OOB Random Forest comparison on ``2nd_real_run.db``:

  * **Compositional** — std-normalized compositions (``X / comp_std``), matching
    the current ``GPSimplex`` input space.
  * **ILR** — Helmert ILR coordinates std-normalized (``ILR(X) / ilr_std``),
    matching the legacy GP input space.

Reports OOB R² and residual RMSE per property (Bandgap, Photoconductance,
Stability, Objective).

Usage
-----
  python visualization/ilr_vs_comp_oob.py
  python visualization/ilr_vs_comp_oob.py --db data/2nd_real_run.db \\
      --out-dir data/2nd_real_run_ilr_vs_comp
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT))

DB_COMP_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
DB_PROPERTIES = ["Bandgap", "Photoconductance", "Stability", "Objective"]
DEFAULT_DB = "2nd_real_run.db"
RF_N_ESTIMATORS = 500
_EPS = 1e-10
_STD_FLOOR = 1e-3


def _resolve_db_path(db_arg: str) -> Path:
    p = Path(db_arg)
    if p.is_file():
        return p
    candidate = ROOT / "data" / db_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Data file not found: {db_arg}")


def _load_db_rows(db_path: Path, cols: list[str]) -> np.ndarray:
    con = sqlite3.connect(str(db_path))
    try:
        sel = ", ".join(f'"{c}"' for c in cols)
        where = " AND ".join(f'"{c}" IS NOT NULL' for c in cols)
        rows = con.execute(f"SELECT {sel} FROM results WHERE {where}").fetchall()
    finally:
        con.close()
    return np.asarray(rows, dtype=float)


def load_db_all_properties(
    db_path: Path, properties: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cols = DB_COMP_COLS + properties
    arr = _load_db_rows(db_path, cols)
    if arr.shape[0] == 0:
        raise RuntimeError(f"No complete rows for {cols} in {db_path}")
    X = arr[:, :3]
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    ys = {prop: arr[:, 3 + i] for i, prop in enumerate(properties)}
    return X, ys


def composition_to_ilr(x: np.ndarray) -> np.ndarray:
    """Helmert ILR; ``x`` shape (n, d) → (n, d-1). Matches ``src.utils.simplex``."""
    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    log_x = np.log(x + _EPS)
    coords = []
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        term1 = log_x[:, : i + 1].sum(axis=1) / (i + 1)
        term2 = log_x[:, i + 1]
        coords.append(coef * (term1 - term2))
    return np.column_stack(coords)


def std_normalize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = np.nan_to_num(features.std(axis=0), nan=1.0)
    std = np.maximum(std, _STD_FLOOR)
    return features / std, std


def fit_rf_oob(features: np.ndarray, y: np.ndarray, n_estimators: int) -> dict:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=42,
        oob_score=True,
        bootstrap=True,
    )
    rf.fit(features, y)
    y_oob = rf.oob_prediction_
    resid = y - y_oob
    return {
        "oob_r2": float(rf.oob_score_),
        "residual_var": float(np.var(resid)),
        "residual_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "n_features": int(features.shape[1]),
    }


def compare_spaces(
    X: np.ndarray, ys: dict[str, np.ndarray], n_estimators: int,
) -> dict[str, dict]:
    X_comp, comp_std = std_normalize(X)
    X_ilr_raw = composition_to_ilr(X)
    X_ilr, ilr_std = std_normalize(X_ilr_raw)

    results: dict[str, dict] = {
        "_spaces": {
            "compositional": {
                "description": "X / comp_std (d=3 features)",
                "comp_std": comp_std.tolist(),
            },
            "ilr": {
                "description": "ILR(X) / ilr_std (d-1=2 features)",
                "ilr_std": ilr_std.tolist(),
            },
        },
    }

    for prop, y in ys.items():
        comp = fit_rf_oob(X_comp, y, n_estimators)
        ilr = fit_rf_oob(X_ilr, y, n_estimators)
        delta_r2 = comp["oob_r2"] - ilr["oob_r2"]
        winner = "compositional" if delta_r2 >= 0 else "ilr"
        if abs(delta_r2) < 0.005:
            winner = "tie"
        results[prop] = {
            "compositional": comp,
            "ilr": ilr,
            "delta_oob_r2_comp_minus_ilr": float(delta_r2),
            "winner": winner,
        }
    return results


def _print_table(results: dict[str, dict], properties: list[str]) -> None:
    w = max(len(p) for p in properties)
    header = (
        f"{'Property':<{w}}  {'Comp OOB R²':>11}  {'ILR OOB R²':>10}  "
        f"{'Δ R²':>8}  {'Comp RMSE':>10}  {'ILR RMSE':>9}  {'Winner':>8}"
    )
    print(header)
    print("-" * len(header))
    for prop in properties:
        r = results[prop]
        c, i = r["compositional"], r["ilr"]
        print(
            f"{prop:<{w}}  {c['oob_r2']:11.4f}  {i['oob_r2']:10.4f}  "
            f"{r['delta_oob_r2_comp_minus_ilr']:+8.4f}  "
            f"{c['residual_rmse']:10.5f}  {i['residual_rmse']:9.5f}  "
            f"{r['winner']:>8}"
        )
    print()
    print("Δ R² = compositional OOB R² − ILR OOB R²  (positive → compositional better)")


def plot_oob_comparison(
    results: dict[str, dict],
    properties: list[str],
    out_path: Path,
) -> None:
    comp_r2 = [results[p]["compositional"]["oob_r2"] for p in properties]
    ilr_r2 = [results[p]["ilr"]["oob_r2"] for p in properties]

    x = np.arange(len(properties))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, comp_r2, width, label="Compositional (X / comp_std)", color="#4477aa")
    ax.bar(x + width / 2, ilr_r2, width, label="ILR (ILR(X) / ilr_std)", color="#cc6677")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.set_ylabel("OOB R²")
    ax.set_ylim(min(0.0, min(comp_r2 + ilr_r2) - 0.05), 1.0)
    ax.set_title("RF surrogate quality: compositional vs ILR input space")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOB RF comparison: compositional vs ILR features on real data.",
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--properties", default=",".join(DB_PROPERTIES))
    parser.add_argument("--n-estimators", type=int, default=RF_N_ESTIMATORS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db)
    properties = [p.strip() for p in args.properties.split(",") if p.strip()]
    out_dir = args.out_dir or (db_path.parent / f"{db_path.stem}_ilr_vs_comp")

    X, ys = load_db_all_properties(db_path, properties)
    results = compare_spaces(X, ys, args.n_estimators)

    summary = {
        "database": str(db_path),
        "n_points": int(X.shape[0]),
        "n_estimators": args.n_estimators,
        "properties": {p: results[p] for p in properties},
        "spaces": results["_spaces"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ilr_vs_comp_oob.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Database: {db_path}")
    print(f"Points:   {X.shape[0]}  (all properties non-null)")
    print(f"RF trees: {args.n_estimators}")
    print()
    _print_table(results, properties)

    wins = {p: results[p]["winner"] for p in properties}
    n_comp = sum(1 for v in wins.values() if v == "compositional")
    n_ilr = sum(1 for v in wins.values() if v == "ilr")
    n_tie = sum(1 for v in wins.values() if v == "tie")
    print(f"Winners: compositional={n_comp}  ilr={n_ilr}  tie={n_tie}")
    print()

    plot_path = out_dir / "oob_r2_comparison.png"
    plot_oob_comparison(results, properties, plot_path)
    print(f"Wrote {json_path.name} and {plot_path.name} -> {out_dir}/")


if __name__ == "__main__":
    main()
