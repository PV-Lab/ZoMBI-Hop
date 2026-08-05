#!/usr/bin/env python3
"""Side-by-side edge_min=0.00 vs 0.03 RF(g) optima for two ELA twins."""
from __future__ import annotations

import json
from pathlib import Path

import sys

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
    ternary_grid,
)
RUNS = ["ela_3d_18535497", "ela_3d_18503666"]
EDGE_DIRS = {
    0.00: REPO / "warm_start" / "optima_finder_ela_rf",
    0.03: REPO / "warm_start" / "optima_finder_ela_rf_interior",
}
OUT = REPO / "warm_start" / "optima_finder_ela_rf_edge_compare"
_SQRT3_2 = np.sqrt(3) / 2
CORNERS = ("FA", "MA", "Br")
GRID_N = 120


def draw_panel(ax, objective, X, y, *, title: str) -> None:
    grid = ternary_grid(GRID_N)
    vals = objective(grid)
    xy = comp_to_xy(grid)
    found_xy = comp_to_xy(X)
    ax.tripcolor(xy[:, 0], xy[:, 1], vals, shading="gouraud", cmap="viridis")
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.scatter(
        found_xy[:, 0], found_xy[:, 1],
        c="red", marker="x", s=56, linewidths=1.7, zorder=5,
        label=f"n={len(X)}",
    )
    for i, (u, v) in enumerate(found_xy):
        ax.text(
            u, v, str(i), color="white", fontsize=6,
            ha="center", va="bottom", zorder=6,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.text(-0.03, -0.03, CORNERS[0], ha="right", va="top", fontsize=9)
    ax.text(1.03, -0.03, CORNERS[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + 0.04, CORNERS[2], ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    for run in RUNS:
        objective, _ = build_rf_g_objective(REPO / "ela" / "runs" / run)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
        for ax, edge in zip(axes, (0.00, 0.03)):
            path = EDGE_DIRS[edge] / f"{run}_optima.json"
            data = json.loads(path.read_text())
            X = np.asarray(data["true_optima"], float)
            y = np.asarray(data["y_optima"], float)
            dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            min_sep = float(dmat.min())
            draw_panel(
                ax, objective, X, y,
                title=(
                    f"edge_min={edge:.2f}  y∈[{y.min():.3f},{y.max():.3f}]  "
                    f"min_sep={min_sep:.3f}"
                ),
            )
        fig.suptitle(
            f"{run}  RF(g) greedy optima (Sobol=8192, n=20, seed=0)",
            fontsize=12,
        )
        fig.tight_layout()
        out_png = OUT / f"{run}_edge00_vs_03.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"wrote {out_png}")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2))
    for r, run in enumerate(RUNS):
        objective, _ = build_rf_g_objective(REPO / "ela" / "runs" / run)
        for c, edge in enumerate((0.00, 0.03)):
            path = EDGE_DIRS[edge] / f"{run}_optima.json"
            data = json.loads(path.read_text())
            X = np.asarray(data["true_optima"], float)
            y = np.asarray(data["y_optima"], float)
            dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            min_sep = float(dmat.min())
            draw_panel(
                axes[r, c], objective, X, y,
                title=(
                    f"{run} | edge_min={edge:.2f}\n"
                    f"y∈[{y.min():.3f},{y.max():.3f}]  min_sep={min_sep:.3f}"
                ),
            )
    fig.suptitle("RF(g) greedy optima: edge_min 0.00 vs 0.03", fontsize=13)
    fig.tight_layout()
    combo = OUT / "pair_18535497_18503666_edge00_vs_03.png"
    fig.savefig(combo, dpi=150)
    plt.close(fig)
    print(f"wrote {combo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
