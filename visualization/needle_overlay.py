"""
visualization/needle_overlay.py
===============================
Plot a run's *background dataset* (the RF-surrogate objective landscape the run
optimized against) on a high-resolution ternary, with that run's **needles**
overlaid on top — and nothing else (no true optima, no collected points).

Point it at a run directory containing a ``needles.csv`` (e.g.
``optimize/runs/rerun_.../trial_0/run_1``).  The surrogate CSV is discovered from
the nearest ``rerun_config.json`` / ``run_config.json`` walking up from that
directory (or pass ``--csv`` to override), then the RF surrogate is rebuilt
exactly as ``run_mobo.build_rf_and_grid`` does (deterministic: fixed tree count /
random_state) and evaluated over a dense ternary grid for the background.

Usage
-----
  conda activate zombi-hop
  python visualization/needle_overlay.py optimize/runs/rerun_.../trial_0/run_1
  python visualization/needle_overlay.py RUN_DIR --grid-n 300 --out overlay.png
  python visualization/needle_overlay.py RUN_DIR --csv interactive_testing/campaign1a.csv

Flags
-----
  --csv PATH     Surrogate CSV override (default: discovered from run config).
  --value COL    Objective column in the CSV (default: Objective).
  --grid-n N     Ternary render resolution for the background (default: 240).
  --out PATH     Output PNG (default: RUN_DIR/needle_overlay.png).
  --no-show      Only save the PNG; don't open an interactive window
                 (the figure is displayed by default).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ── project root on sys.path so `optimize` imports resolve ─────────────────────
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT))

from optimize.mobo_landscapes import (  # noqa: E402
    infer_composition_columns,
    resolve_surrogate_csv_path,
)

# ── constants (mirror optimize/run_mobo.py) ────────────────────────────────────
RF_N_ESTIMATORS = 500           # matches run_mobo.build_rf_and_grid
CORNER_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")  # [bottom-left, bottom-right, top]
# needles.csv composition columns vary by run: short ("FA","MA","Br") or full names.
NEEDLE_COMP_COL_SETS = [["FA", "MA", "Br"], ["FAPbI3", "MAPbI3", "MAPbBr3"]]
_SQRT3_2 = np.sqrt(3) / 2


# ── ternary utilities (mirror run_mobo.comp_to_xy / ternary_grid) ──────────────

def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """(N,3) simplex compositions → (N,2) Cartesian ternary coordinates.

    col0 → (0,0) bottom-left,  col1 → (1,0) bottom-right,  col2 → (0.5, √3/2) top.
    """
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def ternary_grid(n: int) -> np.ndarray:
    """(M,3) uniform grid on the probability simplex at resolution n."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


# ── surrogate CSV discovery ────────────────────────────────────────────────────

def _discover_csv_path(run_dir: Path) -> str:
    """Find the surrogate CSV from the nearest run/rerun config above run_dir."""
    for parent in [run_dir, *run_dir.parents]:
        for name in ("run_config.json", "rerun_config.json"):
            cfg_file = parent / name
            if cfg_file.is_file():
                try:
                    cfg = json.loads(cfg_file.read_text())
                except Exception:
                    continue
                if cfg.get("csv_path"):
                    return resolve_surrogate_csv_path(cfg["csv_path"], str(ROOT))
        # don't escape the repo root
        if parent == ROOT:
            break
    # last resort: let the resolver scan data/ + interactive_testing/
    return resolve_surrogate_csv_path(None, str(ROOT))


# ── RF surrogate ───────────────────────────────────────────────────────────────
# Two surrogate sources, matching how the run was driven:
#   *.csv  → campaign CSV (RF dataset; mirrors run_mobo.build_rf_and_grid)
#   *.db   → live ``results`` table (newRF dataset; mirrors evaluate._newrf_load_db)
# Both train the same deterministic RF (500 trees, random_state=42).
DB_COMP_COLS = list(CORNER_LABELS)   # [FAPbI3, MAPbI3, MAPbBr3], matches plot_run


def _load_xy(source_path: str, objective_column: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read training ``(X (N,3), Y (N,), comp_cols)`` from a CSV or results DB."""
    if source_path.lower().endswith(".db"):
        from visualization.plot_run import load_db_dataset
        X, Y, _labels, _title = load_db_dataset(Path(source_path), objective_column)
        return X, Y, list(DB_COMP_COLS)
    df = pd.read_csv(source_path)
    comp_cols = infer_composition_columns(df)
    df = df.dropna(subset=comp_cols + [objective_column])
    X = df[comp_cols].values.astype(float)
    X /= X.sum(axis=1, keepdims=True)
    Y = df[objective_column].values.astype(float)
    return X, Y, comp_cols


def build_rf(source_path: str, objective_column: str) -> tuple[RandomForestRegressor, list[str]]:
    """Train the deterministic RF surrogate on the run's surrogate source."""
    X, y, comp_cols = _load_xy(source_path, objective_column)
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    return rf, comp_cols


# ── needles ────────────────────────────────────────────────────────────────────

def load_needles(run_dir: Path) -> np.ndarray:
    """Read needles.csv → (K,3) compositions ordered [bottom-left, bottom-right, top]."""
    nf = run_dir / "needles.csv"
    if not nf.is_file():
        raise FileNotFoundError(f"No needles.csv in {run_dir}")
    df = pd.read_csv(nf)
    cols = next((c for c in NEEDLE_COMP_COL_SETS if all(x in df.columns for x in c)), None)
    if cols is None:
        raise ValueError(
            f"needles.csv has none of {NEEDLE_COMP_COL_SETS} (has {list(df.columns)})")
    comp = df[cols].values.astype(float)
    s = comp.sum(axis=1, keepdims=True)
    return comp / np.where(s == 0, 1.0, s)


# ── plot ───────────────────────────────────────────────────────────────────────

def _draw_frame(ax, pad: float = 0.04) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, CORNER_LABELS[0], ha="right", va="top", fontsize=10)
    ax.text(1 + pad, -pad, CORNER_LABELS[1], ha="left", va="top", fontsize=10)
    ax.text(0.5, _SQRT3_2 + pad, CORNER_LABELS[2], ha="center", va="bottom", fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a run's RF-surrogate background landscape with its "
                    "needles overlaid on a high-resolution ternary."
    )
    parser.add_argument("run_dir", help="Run directory containing needles.csv.")
    parser.add_argument("--csv", default=None,
                        help="Surrogate CSV override (default: from run config).")
    parser.add_argument("--value", default="Objective",
                        help="Objective column in the CSV (default: Objective).")
    parser.add_argument("--grid-n", type=int, default=240,
                        help="Ternary render resolution for the background.")
    parser.add_argument("--out", default=None,
                        help="Output PNG (default: RUN_DIR/needle_overlay.png).")
    parser.add_argument("--no-show", action="store_true",
                        help="Only save the PNG; don't open an interactive window.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")

    csv_path = (resolve_surrogate_csv_path(args.csv, str(ROOT))
                if args.csv else _discover_csv_path(run_dir))
    print(f"Run dir   : {run_dir}")
    print(f"Surrogate : {csv_path}")

    rf, comp_cols = build_rf(csv_path, args.value)
    needles = load_needles(run_dir)
    print(f"Comp cols : {comp_cols}")
    print(f"Needles   : {needles.shape[0]}")

    # Background: RF predicted over a dense ternary grid.
    grid_pts = ternary_grid(args.grid_n)
    grid_vals = rf.predict(grid_pts)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    _draw_frame(ax)
    ax.set_title(f"{run_dir.name} — RF background + needles", fontsize=11)

    gxy = comp_to_xy(grid_pts)
    bg = ax.scatter(gxy[:, 0], gxy[:, 1], c=grid_vals, cmap="viridis",
                    s=6, alpha=0.9, zorder=2, rasterized=True, linewidths=0)
    fig.colorbar(bg, ax=ax, label=args.value, fraction=0.046, pad=0.04)

    nxy = comp_to_xy(needles)
    ax.scatter(nxy[:, 0], nxy[:, 1], marker="*", s=240, c="red",
               edgecolors="white", linewidths=1.0, zorder=5, label="needles")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    out = Path(args.out) if args.out else run_dir / "needle_overlay.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved -> {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
