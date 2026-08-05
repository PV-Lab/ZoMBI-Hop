#!/usr/bin/env python3
"""Sweep edge_min for greedy optima on the 2nd_real_run RF surrogate and plot."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))

from visualization.plot_run import load_db_dataset  # noqa: E402
from warm_start.greedy_optima import (  # noqa: E402
    CANDIDATE_MULTIPLIER,
    DEFAULT_DESIGN,
    TOP_FRACTION,
    find_optima,
    n_design_for_optima,
)
from warm_start.run_greedy_optima_ela_rf import comp_to_xy, ternary_grid  # noqa: E402

DB_PATH = REPO / "data" / "2nd_real_run.db"
COMP_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
OBJECTIVE_COL = "Objective"
RF_N_ESTIMATORS = 500
RF_SEED = 42

EDGE_MINS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
N_OPTIMA = 20
SEED = 0
GRID_N = 100
_SQRT3_2 = np.sqrt(3) / 2
OUT = REPO / "warm_start" / "optima_finder_2nd_real_run_rf_edge_sweep"


def build_2nd_real_run_rf_objective(
    db_path: Path = DB_PATH,
) -> tuple[object, dict]:
    X, y, labels, title = load_db_dataset(db_path, OBJECTIVE_COL)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    X = X / X.sum(axis=1, keepdims=True)

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        n_jobs=-1,
        random_state=RF_SEED,
    )
    rf.fit(X, y)

    def objective(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.asarray(rf.predict(x), dtype=float).ravel()

    meta = {
        "run": "2nd_real_run",
        "db_path": str(db_path.relative_to(REPO)),
        "composition_columns": list(labels) if labels else list(COMP_COLS),
        "objective_column": OBJECTIVE_COL,
        "n_train": int(X.shape[0]),
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_seed": RF_SEED,
        "objective": "RF(2nd_real_run)",
        "maximize": True,
        "dataset_title": title,
    }
    return objective, meta


def run_one(
    objective,
    meta: dict,
    *,
    n_optima: int,
    seed: int,
    edge_min: float,
    n_design: int,
    out_dir: Path,
) -> dict:
    t0 = time.perf_counter()
    X, y = find_optima(
        objective,
        dim=3,
        n=n_optima,
        seed=seed,
        design=DEFAULT_DESIGN,
        n_design=n_design,
        edge_min=edge_min,
    )
    elapsed = time.perf_counter() - t0

    dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    nn = dmat.min(axis=1)

    payload = {
        **meta,
        "n_optima": int(len(X)),
        "seed": seed,
        "design": DEFAULT_DESIGN,
        "n_design": int(n_design),
        "edge_min": float(edge_min),
        "top_frac": TOP_FRACTION,
        "candidate_multiplier": CANDIDATE_MULTIPLIER,
        "true_optima": X.tolist(),
        "y_optima": y.tolist(),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_mean": float(y.mean()),
        "min_sep": float(nn.min()),
        "med_sep": float(np.median(nn)),
        "elapsed_s": round(elapsed, 3),
        "ok": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "2nd_real_run_optima.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload


def draw_panel(ax, objective, X, y, *, title: str) -> None:
    grid = ternary_grid(GRID_N)
    vals = objective(grid)
    xy = comp_to_xy(grid)
    found_xy = comp_to_xy(X)
    ax.tripcolor(xy[:, 0], xy[:, 1], vals, shading="gouraud", cmap="viridis")
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.0)
    ax.scatter(
        found_xy[:, 0], found_xy[:, 1],
        c="red", marker="x", s=36, linewidths=1.4, zorder=5,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.10, _SQRT3_2 + 0.12)
    ax.text(-0.02, -0.02, "FA", ha="right", va="top", fontsize=7)
    ax.text(1.02, -0.02, "MA", ha="left", va="top", fontsize=7)
    ax.text(0.5, _SQRT3_2 + 0.02, "Br", ha="center", va="bottom", fontsize=7)
    ax.set_title(title, fontsize=8)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n_design = n_design_for_optima(N_OPTIMA)
    print(
        f"2nd_real_run RF edge_min sweep: n_optima={N_OPTIMA} seed={SEED} "
        f"design={DEFAULT_DESIGN} n_design={n_design} → {OUT}"
    )
    print(f"Training RF on {DB_PATH.relative_to(REPO)} …")
    objective, meta = build_2nd_real_run_rf_objective()
    print(f"  n_train={meta['n_train']}  trees={RF_N_ESTIMATORS}")

    results: dict[float, dict] = {}
    summary: dict = {
        "n_optima": N_OPTIMA,
        "seed": SEED,
        "design": DEFAULT_DESIGN,
        "n_design": n_design,
        "top_frac": TOP_FRACTION,
        "candidate_multiplier": CANDIDATE_MULTIPLIER,
        "edge_mins": EDGE_MINS,
        **{k: meta[k] for k in (
            "db_path", "composition_columns", "objective_column",
            "n_train", "rf_n_estimators", "rf_seed", "objective", "maximize",
        )},
        "edges": {},
    }

    for edge in EDGE_MINS:
        sub = OUT / f"edge_{edge:.2f}"
        row = run_one(
            objective, meta,
            n_optima=N_OPTIMA,
            seed=SEED,
            edge_min=edge,
            n_design=n_design,
            out_dir=sub,
        )
        results[edge] = row
        summary["edges"][f"{edge:.2f}"] = {
            "y_min": row["y_min"],
            "y_max": row["y_max"],
            "y_mean": row["y_mean"],
            "min_sep": row["min_sep"],
            "med_sep": row["med_sep"],
            "elapsed_s": row["elapsed_s"],
        }
        print(
            f"  edge_min={edge:.2f}  y∈[{row['y_min']:.4f},{row['y_max']:.4f}]  "
            f"min_sep={row['min_sep']:.4f}  {row['elapsed_s']:.2f}s"
        )

    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    fig, axes = plt.subplots(1, len(EDGE_MINS), figsize=(18.0, 3.4))
    for ax, edge in zip(axes, EDGE_MINS):
        row = results[edge]
        X = np.asarray(row["true_optima"], float)
        y = np.asarray(row["y_optima"], float)
        draw_panel(
            ax, objective, X, y,
            title=(
                f"edge={edge:.2f}\n"
                f"y∈[{y.min():.3f},{y.max():.3f}] sep={row['min_sep']:.3f}"
            ),
        )
    fig.suptitle(
        f"2nd_real_run RF edge_min sweep (n={N_OPTIMA}, Sobol={n_design}, seed={SEED})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = OUT / "2nd_real_run_edge_sweep.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    metrics = [("y_max", "max RF"), ("y_mean", "mean RF"), ("min_sep", "min separation")]
    for ax, (key, label) in zip(axes, metrics):
        ys = [results[e][key] for e in EDGE_MINS]
        ax.plot(EDGE_MINS, ys, marker="o", color="C0")
        ax.set_xlabel("edge_min")
        ax.set_ylabel(label)
        ax.set_xticks(EDGE_MINS)
        ax.grid(True, alpha=0.3)
    fig.suptitle("2nd_real_run RF edge_min sweep metrics", fontsize=11)
    fig.tight_layout()
    metrics_png = OUT / "edge_sweep_metrics.png"
    fig.savefig(metrics_png, dpi=140)
    plt.close(fig)
    print(f"wrote {metrics_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
