#!/usr/bin/env python3
"""Sweep edge_min for RF(g) greedy optima on the two ELA twins and plot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))

from warm_start.run_greedy_optima_ela_rf import (  # noqa: E402
    build_rf_g_objective,
    comp_to_xy,
    run_one,
    ternary_grid,
)

RUNS = ["ela_3d_18535497", "ela_3d_18503666"]
EDGE_MINS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
N_OPTIMA = 20
SEED = 0
GRID_N = 100
_SQRT3_2 = np.sqrt(3) / 2
CORNERS = ("FA", "MA", "Br")
OUT = REPO / "warm_start" / "optima_finder_ela_rf_edge_sweep"


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
    ax.text(-0.02, -0.02, CORNERS[0], ha="right", va="top", fontsize=7)
    ax.text(1.02, -0.02, CORNERS[1], ha="left", va="top", fontsize=7)
    ax.text(0.5, _SQRT3_2 + 0.02, CORNERS[2], ha="center", va="bottom", fontsize=7)
    ax.set_title(title, fontsize=8)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"n_optima": N_OPTIMA, "seed": SEED, "edge_mins": EDGE_MINS, "runs": {}}

    objectives: dict[str, object] = {}
    results: dict[str, dict[float, dict]] = {run: {} for run in RUNS}

    for run in RUNS:
        run_dir = REPO / "ela" / "runs" / run
        print(f"=== {run} ===")
        objectives[run], _ = build_rf_g_objective(run_dir)
        for edge in EDGE_MINS:
            sub = OUT / f"edge_{edge:.2f}"
            row = run_one(
                run_dir,
                n_optima=N_OPTIMA,
                seed=SEED,
                plot=False,
                edge_min=edge,
                out_dir=sub,
            )
            # rename default json to include edge for clarity (run_one writes <run>_optima.json)
            results[run][edge] = row
            print(
                f"  edge_min={edge:.2f}  y∈[{row['y_min']:.4f},{row['y_max']:.4f}]  "
                f"min_sep={row['min_sep']:.4f}"
            )
        summary["runs"][run] = {
            f"{edge:.2f}": {
                "y_min": results[run][edge]["y_min"],
                "y_max": results[run][edge]["y_max"],
                "y_mean": results[run][edge]["y_mean"],
                "min_sep": results[run][edge]["min_sep"],
                "med_sep": results[run][edge]["med_sep"],
            }
            for edge in EDGE_MINS
        }

    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    # Per-landscape 1x6 row
    for run in RUNS:
        fig, axes = plt.subplots(1, len(EDGE_MINS), figsize=(18.0, 3.4))
        for ax, edge in zip(axes, EDGE_MINS):
            row = results[run][edge]
            X = np.asarray(row["true_optima"], float)
            y = np.asarray(row["y_optima"], float)
            draw_panel(
                ax, objectives[run], X, y,
                title=(
                    f"edge={edge:.2f}\n"
                    f"y∈[{y.min():.3f},{y.max():.3f}] sep={row['min_sep']:.3f}"
                ),
            )
        fig.suptitle(
            f"{run}  RF(g) edge_min sweep (n={N_OPTIMA}, Sobol=8192, seed={SEED})",
            fontsize=11,
        )
        fig.tight_layout()
        out_png = OUT / f"{run}_edge_sweep.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"wrote {out_png}")

    # Combined 2x6
    fig, axes = plt.subplots(2, len(EDGE_MINS), figsize=(18.0, 6.6))
    for r, run in enumerate(RUNS):
        for c, edge in enumerate(EDGE_MINS):
            row = results[run][edge]
            X = np.asarray(row["true_optima"], float)
            y = np.asarray(row["y_optima"], float)
            title = (
                f"{run.split('_')[-1]} | edge={edge:.2f}\n"
                f"y∈[{y.min():.3f},{y.max():.3f}] sep={row['min_sep']:.3f}"
            )
            draw_panel(axes[r, c], objectives[run], X, y, title=title)
    fig.suptitle(
        "RF(g) greedy optima edge_min sweep 0.00→0.05 (step 0.01)",
        fontsize=12,
    )
    fig.tight_layout()
    combo = OUT / "pair_edge_sweep_0.00_to_0.05.png"
    fig.savefig(combo, dpi=140)
    plt.close(fig)
    print(f"wrote {combo}")

    # Metric curves
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    metrics = [("y_max", "max RF(g)"), ("y_mean", "mean RF(g)"), ("min_sep", "min separation")]
    for ax, (key, label) in zip(axes, metrics):
        for run in RUNS:
            xs = EDGE_MINS
            ys = [results[run][e][key] for e in EDGE_MINS]
            ax.plot(xs, ys, marker="o", label=run.replace("ela_3d_", ""))
        ax.set_xlabel("edge_min")
        ax.set_ylabel(label)
        ax.set_xticks(EDGE_MINS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("edge_min sweep metrics", fontsize=11)
    fig.tight_layout()
    metrics_png = OUT / "edge_sweep_metrics.png"
    fig.savefig(metrics_png, dpi=140)
    plt.close(fig)
    print(f"wrote {metrics_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
