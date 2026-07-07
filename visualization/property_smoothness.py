"""
visualization/property_smoothness.py
====================================
Quantify how smooth each measured property is over the 3-component simplex by
training a Random Forest per target and reporting:

  * OOB R²
  * Residual variance (OOB prediction errors)
  * Local Lipschitz estimates — how fast Y changes over composition distances
    comparable to instrument input noise (default 0.064 L₂)

Also exports ternary RF-background plots per property via ``plot_run.py --export``.

Usage
-----
  conda activate zombi-hop
  python visualization/property_smoothness.py
  python visualization/property_smoothness.py --db data/2nd_real_run.db --out-dir data/2nd_real_run_smoothness
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT))

DB_COMP_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
DB_VALUE_COLS = ["Bandgap", "Photoconductance", "Stability", "Objective"]
DEFAULT_DB = "2nd_real_run.db"
DEFAULT_INPUT_NOISE = 0.064
RF_N_ESTIMATORS = 500
TERNARY_GRID_N = 120
DIST_BAND = (0.5, 1.5)
_SQRT3_2 = np.sqrt(3) / 2


def _is_csv(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def ternary_grid(n: int) -> np.ndarray:
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


def fit_rf_background(
    X: np.ndarray, Y: np.ndarray, grid_n: int, n_estimators: int,
) -> tuple[np.ndarray, np.ndarray]:
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(X, Y)
    grid_pts = ternary_grid(grid_n)
    return grid_pts, rf.predict(grid_pts)


def _resolve_db_path(db_arg: str) -> Path:
    p = Path(db_arg)
    if p.is_file():
        return p
    candidate = ROOT / "data" / db_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Data file not found: {db_arg}")


def _load_csv_rows(csv_path: Path, cols: list[str]) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Columns {missing} not found in {csv_path}")
    return df[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)


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
    """Load compositions and all property columns, keeping only complete rows."""
    cols = DB_COMP_COLS + properties
    arr = _load_csv_rows(db_path, cols) if _is_csv(db_path) else _load_db_rows(db_path, cols)
    if arr.shape[0] == 0:
        raise RuntimeError(f"No complete rows for {cols} in {db_path}")
    X = arr[:, :3]
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    ys = {prop: arr[:, 3 + i] for i, prop in enumerate(properties)}
    return X, ys


def _load_all_properties(db_path: Path, properties: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return load_db_all_properties(db_path, properties)


def _fit_rf_oob(X: np.ndarray, y: np.ndarray, n_estimators: int) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=42,
        oob_score=True,
        bootstrap=True,
    )
    rf.fit(X, y)
    return rf


def _local_lipschitz(
    X: np.ndarray,
    y: np.ndarray,
    input_noise: float,
) -> dict[str, float | int]:
    n = X.shape[0]
    if n < 2:
        nan = float("nan")
        return {
            "n_pairs": 0,
            "median_slope": nan,
            "p90_slope": nan,
            "max_slope": nan,
            "median_abs_dy_at_noise": nan,
            "p90_abs_dy_at_noise": nan,
            "median_abs_dy_in_band": nan,
            "p90_abs_dy_in_band": nan,
        }

    nn = NearestNeighbors(radius=input_noise, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    neighbors = nn.radius_neighbors(X, return_distance=True)

    slopes: list[float] = []
    abs_dy_band: list[float] = []
    lo = DIST_BAND[0] * input_noise
    hi = DIST_BAND[1] * input_noise

    for i in range(n):
        idx = neighbors[1][i]
        dists = neighbors[0][i]
        for j, dist in zip(idx, dists):
            if j <= i or dist <= 1e-12:
                continue
            dy = abs(float(y[i] - y[j]))
            slopes.append(dy / float(dist))
            if lo <= dist <= hi:
                abs_dy_band.append(dy)

    if not slopes:
        nan = float("nan")
        return {
            "n_pairs": 0,
            "median_slope": nan,
            "p90_slope": nan,
            "max_slope": nan,
            "median_abs_dy_at_noise": nan,
            "p90_abs_dy_at_noise": nan,
            "median_abs_dy_in_band": nan,
            "p90_abs_dy_in_band": nan,
        }

    slopes_arr = np.asarray(slopes, dtype=float)
    med_slope = float(np.median(slopes_arr))
    return {
        "n_pairs": int(len(slopes)),
        "median_slope": med_slope,
        "p90_slope": float(np.percentile(slopes_arr, 90)),
        "max_slope": float(np.max(slopes_arr)),
        "median_abs_dy_at_noise": med_slope * input_noise,
        "p90_abs_dy_at_noise": float(np.percentile(slopes_arr, 90)) * input_noise,
        "median_abs_dy_in_band": float(np.median(abs_dy_band)) if abs_dy_band else float("nan"),
        "p90_abs_dy_in_band": float(np.percentile(abs_dy_band, 90)) if abs_dy_band else float("nan"),
    }


def analyze_property(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int,
    input_noise: float,
) -> dict:
    rf = _fit_rf_oob(X, y, n_estimators)
    y_oob = rf.oob_prediction_
    resid = y - y_oob
    lipschitz = _local_lipschitz(X, y, input_noise)
    y_range = float(y.max() - y.min())
    med_dy = lipschitz["median_abs_dy_at_noise"]
    return {
        "n_points": int(X.shape[0]),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_range": y_range,
        "y_std": float(y.std()),
        "oob_r2": float(rf.oob_score_),
        "residual_var": float(np.var(resid)),
        "residual_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "lipschitz": lipschitz,
        "median_dy_over_range_at_noise": (
            float(med_dy / y_range) if y_range > 1e-12 and np.isfinite(med_dy) else float("nan")
        ),
    }


def _print_table(results: dict[str, dict], input_noise: float) -> None:
    props = list(results.keys())
    w = max(len(p) for p in props)
    header = (
        f"{'Property':<{w}}  {'N':>5}  {'OOB R²':>8}  {'resid σ':>9}  "
        f"{'|ΔY|@ℓ med':>11}  {'|ΔY|/range':>10}  {'Lip med':>9}  {'pairs':>7}"
    )
    print(header)
    print("-" * len(header))
    for prop in props:
        r = results[prop]
        lip = r["lipschitz"]
        print(
            f"{prop:<{w}}  {r['n_points']:5d}  {r['oob_r2']:8.4f}  "
            f"{r['residual_rmse']:9.5f}  {lip['median_abs_dy_at_noise']:11.5f}  "
            f"{r['median_dy_over_range_at_noise']:10.4f}  "
            f"{lip['median_slope']:9.4f}  {lip['n_pairs']:7d}"
        )
    print()
    print(f"Local Lipschitz: pairs with ||Δcomp|| ≤ {input_noise:.4f} (composition L₂).")
    print(f"|ΔY|@ℓ med ≈ median slope × {input_noise:.4f}.")
    print("|ΔY|/range = median |ΔY| at noise scale divided by property range.")


def _export_ternary_png(
    X: np.ndarray,
    y: np.ndarray,
    out_png: Path,
    *,
    value_name: str,
    db_name: str,
    grid_n: int,
    n_estimators: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = tuple(DB_COMP_COLS)
    title = f"{db_name} — {X.shape[0]} measured points  ({value_name})"
    grid_pts, grid_vals = fit_rf_background(X, y, grid_n, n_estimators)
    vmin = float(min(grid_vals.min(), y.min()))
    vmax = float(max(grid_vals.max(), y.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-0.04, -0.04, labels[0], ha="right", va="top", fontsize=9)
    ax.text(1.04, -0.04, labels[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + 0.04, labels[2], ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)

    gxy = comp_to_xy(grid_pts)
    ax.scatter(gxy[:, 0], gxy[:, 1], c=grid_vals, cmap="viridis",
               vmin=vmin, vmax=vmax, s=8, alpha=0.80, zorder=2, rasterized=True)
    pxy = comp_to_xy(X)
    sc = ax.scatter(pxy[:, 0], pxy[:, 1], c=y, cmap="viridis", vmin=vmin, vmax=vmax,
                    s=30, alpha=0.95, zorder=4, edgecolors="black", linewidths=0.9)
    fig.colorbar(sc, ax=ax, label=value_name, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_png}")


def _export_ternary_plots(
    db_path: Path,
    out_dir: Path,
    X: np.ndarray,
    ys: dict[str, np.ndarray],
    properties: list[str],
    *,
    grid_n: int,
    n_estimators: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for prop in properties:
        out_png = out_dir / f"{db_path.stem}_{prop.lower()}.png"
        _export_ternary_png(
            X, ys[prop], out_png,
            value_name=prop,
            db_name=db_path.name,
            grid_n=grid_n,
            n_estimators=n_estimators,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RF smoothness metrics and ternary plots per measured property."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Data file (.db or .csv).")
    parser.add_argument(
        "--properties",
        default=",".join(DB_VALUE_COLS),
        help=f"Comma-separated value columns (default: {','.join(DB_VALUE_COLS)}).",
    )
    parser.add_argument("--input-noise", type=float, default=DEFAULT_INPUT_NOISE,
                        help="Composition L₂ step for Lipschitz scale (default: 0.064).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directory for PNGs and summary JSON.")
    parser.add_argument("--no-plots", action="store_true", help="Skip ternary PNG export.")
    parser.add_argument("--grid-n", type=int, default=TERNARY_GRID_N)
    parser.add_argument("--n-estimators", type=int, default=RF_N_ESTIMATORS)
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db)
    properties = [p.strip() for p in args.properties.split(",") if p.strip()]
    out_dir = args.out_dir or (db_path.parent / f"{db_path.stem}_smoothness")

    print(f"Database   : {db_path}")
    print(f"Properties : {properties}")
    print(f"Input noise: {args.input_noise:.4f} (composition L₂)")
    print()

    X, ys = _load_all_properties(db_path, properties)

    results: dict[str, dict] = {}
    for prop in properties:
        print(f"Analyzing {prop} …")
        results[prop] = analyze_property(
            X, ys[prop],
            n_estimators=args.n_estimators,
            input_noise=args.input_noise,
        )

    print()
    _print_table(results, args.input_noise)

    ranked = sorted(
        properties,
        key=lambda p: (
            -results[p]["oob_r2"],
            results[p]["median_dy_over_range_at_noise"],
        ),
    )
    print("Smoothness ranking (higher OOB R², lower |ΔY|/range at noise scale):")
    for i, prop in enumerate(ranked, 1):
        r = results[prop]
        print(
            f"  {i}. {prop}: OOB R²={r['oob_r2']:.4f}, "
            f"|ΔY|/range@ℓ={r['median_dy_over_range_at_noise']:.4f}"
        )
    print()

    summary = {
        "database": str(db_path),
        "composition_columns": DB_COMP_COLS,
        "input_noise": args.input_noise,
        "n_estimators": args.n_estimators,
        "properties": results,
        "smoothness_ranking": ranked,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "smoothness_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {json_path}")

    if not args.no_plots:
        print(f"Exporting ternary plots -> {out_dir}/")
        _export_ternary_plots(
            db_path, out_dir, X, ys, properties,
            grid_n=args.grid_n, n_estimators=args.n_estimators,
        )


if __name__ == "__main__":
    main()
