"""
warm_start/test_greedy_optima.py
================================
Visual check of :func:`warm_start.greedy_optima.find_optima` against ground
truth on 3d :class:`~synthetic_data.ensemble.Ensemble` landscapes.

Draws 5 **random** ensemble landscapes (a fresh, unseeded draw every run, so
repeated runs exercise different feature mixes), runs the greedy optima finder
on each, and saves one ternary plot per landscape showing

    * the objective as a filled contour,
    * the landscape's own true optima (``Ensemble.known_maxima``) in **blue**,
    * the optima the finder returned in **red**.

Images land in ``warm_start/optima_finder/`` as ``landscape_<i>_seed<s>.png``.

Run:
  uv run python warm_start/test_greedy_optima.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, so synthetic_data is importable

from synthetic_data.ensemble import Ensemble, random_ensemble_config  # noqa: E402
from warm_start.greedy_optima import find_optima, n_optima  # noqa: E402

OUT_DIR = _HERE / "optima_finder"
N_LANDSCAPES = 5
DIM = 3
GRID = 220                     # contour resolution along each barycentric axis

_SQRT3_2 = np.sqrt(3) / 2


# ── ternary helpers (match visualization/plot_run.py) ─────────────────────────

def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """(N,3) simplex compositions → (N,2) ternary Cartesian.

    col0 → (0,0) bottom-left, col1 → (1,0) bottom-right, col2 → (0.5,√3/2) top.
    """
    p = np.asarray(comp, float)
    s = p.sum(-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def _simplex_grid(steps: int) -> np.ndarray:
    """Regular lattice of 3-component compositions (rows sum to 1)."""
    a, b = np.meshgrid(np.arange(steps + 1), np.arange(steps + 1), indexing="ij")
    a, b = a.ravel(), b.ravel()
    keep = a + b <= steps
    a, b = a[keep], b[keep]
    return np.column_stack([a, b, steps - a - b]) / float(steps)


def _draw_triangle(ax, labels=("A", "B", "C")) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.text(-0.03, -0.03, labels[0], ha="right", va="top", fontsize=10)
    ax.text(1.03, -0.03, labels[1], ha="left", va="top", fontsize=10)
    ax.text(0.5, _SQRT3_2 + 0.04, labels[2], ha="center", va="bottom", fontsize=10)


# ── one landscape ─────────────────────────────────────────────────────────────

def plot_landscape(fn: Ensemble, X_found: np.ndarray, y_found: np.ndarray,
                   out_png: Path, subtitle: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp = _simplex_grid(GRID)
    z = fn.predict(comp)
    xy = comp_to_xy(comp)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    tc = ax.tricontourf(xy[:, 0], xy[:, 1], z, levels=60, cmap="viridis")
    _draw_triangle(ax, ("FAPbI3", "MAPbI3", "MAPbBr3"))
    fig.colorbar(tc, ax=ax, shrink=0.78, label="objective")

    true_xy = comp_to_xy(fn.centers) if len(fn.centers) else np.empty((0, 2))
    if len(true_xy):
        ax.scatter(true_xy[:, 0], true_xy[:, 1], s=52, marker="o",
                   facecolors="none", edgecolors="#1f6feb", linewidths=1.6,
                   label=f"true optima (n={len(true_xy)})", zorder=3)
    found_xy = comp_to_xy(X_found)
    ax.scatter(found_xy[:, 0], found_xy[:, 1], s=42, marker="X", c="#e01e37",
               edgecolors="k", linewidths=0.4,
               label=f"found optima (n={len(found_xy)})", zorder=4)
    ax.legend(loc="upper left", fontsize=8, frameon=False,
              bbox_to_anchor=(-0.08, 1.02))

    ax.set_title(subtitle, fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def nearest_true_distance(X_found: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """For each found optimum, the L2 distance to the closest true optimum."""
    if not len(centers):
        return np.full(len(X_found), np.nan)
    return np.linalg.norm(X_found[:, None, :] - centers[None, :, :], axis=2).min(1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Unseeded: a different set of 5 landscapes every run.
    rng = np.random.default_rng()
    n_opt = n_optima(DIM)

    print(f"{'#':>2} {'seed':>7} {'index':>6} {'layout':>9} {'true':>5} "
          f"{'best y':>8} {'mean d(found→true)':>20}")
    print("-" * 66)
    for i in range(N_LANDSCAPES):
        seed = int(rng.integers(1_000_000))
        index = int(rng.integers(1_000))
        cfg = random_ensemble_config(DIM, index=index, seed=seed)
        fn = Ensemble(**cfg)

        X_found, y_found = find_optima(fn.predict, DIM, n_opt,
                                       seed=int(rng.integers(1_000_000)))
        d = nearest_true_distance(X_found, fn.centers)

        out_png = OUT_DIR / f"landscape_{i + 1}_seed{seed}.png"
        plot_landscape(
            fn, X_found, y_found, out_png,
            subtitle=(f"Ensemble(dim={DIM}, layout={cfg['optima_layout']}, "
                      f"seed={seed}, index={index})\n"
                      f"blue = true optima ({len(fn.centers)})   "
                      f"red = greedy_optima ({n_opt})   "
                      f"mean dist to nearest true = {np.nanmean(d):.3f}"),
        )
        print(f"{i + 1:>2} {seed:>7} {index:>6} {cfg['optima_layout']:>9} "
              f"{len(fn.centers):>5} {y_found.max():>8.4f} {np.nanmean(d):>20.4f}")

    print(f"\nSaved {N_LANDSCAPES} figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
