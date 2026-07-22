"""
warm_start/test_greedy_optima_gp.py
===================================
Visual check of :func:`warm_start.greedy_optima.find_optima` against a GP
landscape fit to a *real* ZoMBI-Hop run — the data-driven analogue of
``test_greedy_optima.py`` (which checks the finder against synthetic
:class:`~synthetic_data.ensemble.Ensemble` landscapes).

For each run:

    1. Reconstruct the measured dataset ``(X, Y)`` from the run's snapshots
       (``visualization.plot_run.load_run_source``).
    2. Fit a Gaussian-Process surrogate — Matern(nu=2.5) with a **fixed
       length-scale of 0.05** — and treat its ``predict`` as the objective
       landscape.
    3. Run ``find_optima`` on that landscape.
    4. Plot the landscape with
         * the GP landscape itself (filled contour for d=3, translucent 3D
           volume for d=4),
         * the landscape's own peaks (dense-grid local maxima) in **blue** —
           the "true optima" reference, since a real run has no known centers,
         * the optima the finder returned in **red**.

Two runs are visualised:
    * ``runs/run_7eb9``  — d=3, drawn as a ternary contour.
    * ``runs/run_9dfe``  — d=4, drawn as a 3D tetrahedron.

Images land in ``warm_start/optima_finder_gp/``.

Run:
  uv run python warm_start/test_greedy_optima_gp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root

from visualization.plot_run import (  # noqa: E402
    TETRA_EDGES,
    TETRA_VERTS,
    comp_to_xy,
    comp_to_xyz,
    load_run_source,
    simplex_grid,
)
from warm_start.greedy_optima import find_optima, n_optima  # noqa: E402

OUT_DIR = _HERE / "optima_finder_gp"
GP_LENGTH_SCALE = 0.05
GRID_3D = 220          # ternary contour resolution
GRID_4D = 42           # 3-simplex lattice resolution (O(n^3) points)
SEED = 0

_SQRT3_2 = np.sqrt(3) / 2

# Which runs to visualise, with the composition-corner labels reported by
# load_run_source (kept here so the titles/axes are self-documenting).
RUNS = [
    ("runs/run_7eb9", 3),
    ("runs/run_9dfe", 4),
]


# ── GP landscape ──────────────────────────────────────────────────────────────

def build_gp_landscape(X: np.ndarray, Y: np.ndarray, length_scale: float):
    """Fit a fixed-length-scale Matern GP to ``(X, Y)`` and return its predictor.

    Mirrors ``plot_run.fit_gp_background``'s kernel, but returns a callable
    ``predict(comp) -> values`` (on the original Y scale) so it can be handed
    straight to :func:`find_optima` as the objective landscape.
    """
    y_mean = float(Y.mean())
    y_std = float(Y.std()) or 1.0
    y = (Y - y_mean) / y_std

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=False, n_restarts_optimizer=2, random_state=42
    )
    gp.fit(X, y)

    def predict(comp: np.ndarray) -> np.ndarray:
        comp = np.atleast_2d(np.asarray(comp, dtype=float))
        return gp.predict(comp) * y_std + y_mean

    return predict


def landscape_peaks(
    predict, grid: np.ndarray, z: np.ndarray, n: int, min_sep: float
) -> np.ndarray:
    """Dense-grid local maxima of ``z`` over ``grid``, thinned to ``n`` peaks.

    A grid point is a local maximum if no lattice neighbour within
    ``3 * spacing`` is higher. Local maxima are then taken in descending value,
    each accepted only if it is at least ``min_sep`` from every peak already
    kept — so tall basins contribute one marker instead of a cluster.
    """
    tree = cKDTree(grid)
    # Grid spacing on the simplex: adjacent lattice points differ by 1/steps in
    # two coordinates, so the L2 step is sqrt(2)/steps. Estimate it from the data.
    spacing = float(np.median(tree.query(grid[:200], k=2)[0][:, 1]))
    radius = 3.0 * spacing

    neighbours = tree.query_ball_point(grid, r=radius)
    is_max = np.array([z[i] >= z[nb].max() for i, nb in enumerate(neighbours)])
    cand = np.where(is_max)[0]
    cand = cand[np.argsort(z[cand])[::-1]]     # descending value

    kept: list[int] = []
    for idx in cand:
        p = grid[idx]
        if all(np.linalg.norm(p - grid[k]) >= min_sep for k in kept):
            kept.append(idx)
        if len(kept) >= n:
            break
    return grid[kept]


def nearest_true_distance(X_found: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """For each found optimum, the L2 distance to the closest landscape peak."""
    if not len(peaks):
        return np.full(len(X_found), np.nan)
    return np.linalg.norm(X_found[:, None, :] - peaks[None, :, :], axis=2).min(1)


# ── d=3 ternary plot ──────────────────────────────────────────────────────────

def _draw_triangle(ax, labels) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.text(-0.03, -0.03, labels[0], ha="right", va="top", fontsize=10)
    ax.text(1.03, -0.03, labels[1], ha="left", va="top", fontsize=10)
    ax.text(0.5, _SQRT3_2 + 0.04, labels[2], ha="center", va="bottom", fontsize=10)


def plot_ternary(grid, z, peaks, X_found, labels, out_png, subtitle, plt) -> None:
    xy = comp_to_xy(grid)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    tc = ax.tricontourf(xy[:, 0], xy[:, 1], z, levels=60, cmap="viridis")
    _draw_triangle(ax, labels)
    fig.colorbar(tc, ax=ax, shrink=0.78, label="GP objective")

    if len(peaks):
        pxy = comp_to_xy(peaks)
        ax.scatter(pxy[:, 0], pxy[:, 1], s=58, marker="o", facecolors="none",
                   edgecolors="#1f6feb", linewidths=1.8,
                   label=f"landscape peaks (n={len(peaks)})", zorder=3)
    fxy = comp_to_xy(X_found)
    ax.scatter(fxy[:, 0], fxy[:, 1], s=46, marker="X", c="#e01e37",
               edgecolors="k", linewidths=0.4,
               label=f"found optima (n={len(X_found)})", zorder=4)
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(-0.08, 1.02))
    ax.set_title(subtitle, fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── d=4 tetrahedron plot ──────────────────────────────────────────────────────

def plot_tetra(grid, z, peaks, X_found, labels, out_png, subtitle, plt) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    fig = plt.figure(figsize=(8.4, 7.6))
    ax = fig.add_subplot(111, projection="3d")

    for i, j in TETRA_EDGES:
        ax.plot(*zip(TETRA_VERTS[i], TETRA_VERTS[j]), color="black", lw=1.1)
    centroid = TETRA_VERTS.mean(axis=0)
    for v, name in zip(TETRA_VERTS, labels[:4]):
        p = v + 0.12 * (v - centroid)
        ax.text(p[0], p[1], p[2], name, ha="center", va="center", fontsize=10)

    # Translucent GP volume: show only the top values so the high-objective
    # basins read as a coloured cloud instead of a solid fog of low values.
    gxyz = comp_to_xyz(grid)
    thr = np.percentile(z, 75)
    keep = z >= thr
    mappable = ax.scatter(
        gxyz[keep, 0], gxyz[keep, 1], gxyz[keep, 2], c=z[keep], cmap="viridis",
        s=8, alpha=0.22, linewidths=0, depthshade=False)
    fig.colorbar(mappable, ax=ax, label="GP objective", fraction=0.03, pad=0.02)

    if len(peaks):
        qxyz = comp_to_xyz(peaks)
        ax.scatter(qxyz[:, 0], qxyz[:, 1], qxyz[:, 2], s=120, marker="o",
                   facecolors="none", edgecolors="#1f6feb", linewidths=2.2,
                   label=f"landscape peaks (n={len(peaks)})", depthshade=False)
    fxyz = comp_to_xyz(X_found)
    ax.scatter(fxyz[:, 0], fxyz[:, 1], fxyz[:, 2], s=90, marker="X", c="#e01e37",
               edgecolors="k", linewidths=0.5,
               label=f"found optima (n={len(X_found)})", depthshade=False)

    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(subtitle, fontsize=9)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=35)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'run':>14} {'dim':>3} {'pts':>5} {'peaks':>5} {'found':>5} "
          f"{'best y':>8} {'mean d(found->peak)':>20}")
    print("-" * 70)

    for run_arg, dim in RUNS:
        X, Y, labels, title = load_run_source(Path(run_arg), None)
        predict = build_gp_landscape(X, Y, GP_LENGTH_SCALE)

        n_opt = n_optima(dim)
        X_found, y_found = find_optima(predict, dim, n_opt, seed=SEED)

        grid = simplex_grid(GRID_3D if dim == 3 else GRID_4D, dim)
        z = predict(grid)
        # Merge peaks closer than ~2x the finder's own spread scale.
        peaks = landscape_peaks(predict, grid, z, n_opt, min_sep=0.12)
        d = nearest_true_distance(X_found, peaks)

        name = Path(run_arg).name
        out_png = OUT_DIR / f"{name}_gp_ls{GP_LENGTH_SCALE}_d{dim}.png"
        subtitle = (
            f"{name}  (d={dim}, {X.shape[0]} measured pts)   "
            f"GP Matern length_scale={GP_LENGTH_SCALE}\n"
            f"blue = landscape peaks ({len(peaks)})   "
            f"red = greedy_optima ({len(X_found)})   "
            f"mean dist found->nearest peak = {np.nanmean(d):.3f}"
        )
        if dim == 3:
            plot_ternary(grid, z, peaks, X_found, labels, out_png, subtitle, plt)
        else:
            plot_tetra(grid, z, peaks, X_found, labels, out_png, subtitle, plt)

        print(f"{name:>14} {dim:>3} {X.shape[0]:>5} {len(peaks):>5} "
              f"{len(X_found):>5} {y_found.max():>8.4f} {np.nanmean(d):>20.4f}")

    print(f"\nSaved figures -> {OUT_DIR}")


if __name__ == "__main__":
    main()
