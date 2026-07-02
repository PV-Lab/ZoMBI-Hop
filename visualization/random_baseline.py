"""
visualization/random_baseline.py
================================
Random-sampling baseline on the ``run_7eb9`` Random-Forest landscape.

This is the "dumb" comparator for a ZoMBI-Hop run: instead of the intelligent
LineBO/zoom sampling, it just throws ``N_POINTS`` uniformly-random compositions
at the same surrogate and looks at what they hit.

Pipeline
--------
  1. Reconstruct ``(X, Y)`` from a real run directory (default ``run_7eb9``) and
     fit the SAME Random-Forest surrogate that ``visualization/plot_run.py``
     draws — this surrogate is treated as the ground-truth oracle.
  2. Draw ``N_POINTS`` compositions uniformly on the 3-simplex (Dirichlet(1,1,1))
     and "measure" each by evaluating the RF surrogate. This is the random
     sampling campaign.
  3. Render two diagnostics, mirroring the ones ``optimize/run_mobo.py`` writes
     for a ZoMBI-Hop trial:
       * ``convergence.png`` — sampled Y in draw order + running best
         (cf. ``run_mobo.plot_convergence``, minus the penalty/needle structure
         random sampling doesn't have).
       * ``coverage.png``    — ternary RF background + the sampled points overlaid
         (cf. ``optimize/coverage_plot``).

Usage
-----
  conda activate zombi-hop
  python visualization/random_baseline.py
  python visualization/random_baseline.py --n-points 200 --seed 0
  python visualization/random_baseline.py --run runs/run_7eb9 --minimize
  python visualization/random_baseline.py --out-dir runs/run_7eb9/random_baseline

Flags
-----
  --run PATH        Run directory or bare run name (default: run_7eb9).
  --snapshot NAME   Snapshot to reconstruct up to (default: latest.txt).
  --n-points N      Number of random samples to draw (default: 192, == a typical
                    ZoMBI-Hop trial: NUM_LINES*NUM_EXPERIMENTS-ish budget).
  --seed S          RNG seed for reproducible sampling (default: 0).
  --minimize        Treat lower Y as better (running best = running min).
                    Default is maximize, matching run_mobo.plot_convergence.
  --grid-n N        Ternary grid resolution for the RF background (default: 120).
  --n-estimators N  Number of trees in the RF surrogate (default: 500).
  --out-dir DIR     Where to save convergence.png / coverage.png
                    (default: <run_dir>/random_baseline).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project root on sys.path so sibling imports resolve ────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from visualization.plot_run import (  # noqa: E402
    _resolve_run_dir,
    comp_to_xy,
    fit_rf_background,
    load_run_source,
)

_SQRT3_2 = np.sqrt(3) / 2


# ── random sampling on the simplex ─────────────────────────────────────────────

def sample_simplex(n_points: int, dim: int = 3, seed: int = 0) -> np.ndarray:
    """Draw ``n_points`` compositions uniformly on the (dim-1)-simplex.

    A Dirichlet(1, …, 1) draw is exactly uniform over the probability simplex.
    """
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(dim), size=n_points)


# ── plots (mirror run_mobo.plot_convergence / optimize.coverage_plot) ──────────

def plot_convergence(path: Path, Y: np.ndarray, *, maximize: bool, title: str) -> None:
    """Sampled Y in draw order + running best (mirror of run_mobo.plot_convergence).

    Random sampling has no penalty mask or needles, so this is the plain version:
    every point is a valid observation and the running best is a monotone envelope.
    """
    idx = np.arange(len(Y))
    running_best = (np.maximum.accumulate(Y) if maximize
                    else np.minimum.accumulate(Y))
    best_label = "running best (max)" if maximize else "running best (min)"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(idx, Y, s=10, alpha=0.65, color="steelblue", label="random sample",
               zorder=3)
    ax.plot(idx, running_best, color="darkorange", lw=1.8, label=best_label,
            zorder=4)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Objective Y (RF surrogate)")
    ax.set_title(f"{title}  ({len(Y)} random points)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right" if maximize else "upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(
    path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    grid_pts: np.ndarray,
    grid_vals: np.ndarray,
    labels: tuple[str, str, str],
    *,
    title: str,
) -> None:
    """Ternary RF background + random samples overlaid (mirror of coverage_plot).

    Corner mapping matches comp_to_xy: col0 → bottom-left, col1 → bottom-right,
    col2 → top.
    """
    vmin = float(min(grid_vals.min(), Y.min()))
    vmax = float(max(grid_vals.max(), Y.max()))
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
    sc = ax.scatter(pxy[:, 0], pxy[:, 1], c=Y, cmap="viridis", vmin=vmin, vmax=vmax,
                    s=30, alpha=0.95, zorder=4, edgecolors="black", linewidths=0.9)
    fig.colorbar(sc, ax=ax, label="Objective", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-sampling baseline (convergence + coverage) on a run's "
                    "RF-interpolated landscape."
    )
    parser.add_argument("--run", default="run_7eb9",
                        help="Run directory or bare run name (default: run_7eb9).")
    parser.add_argument("--snapshot", default=None,
                        help="Snapshot to reconstruct up to (default: latest.txt).")
    parser.add_argument("--n-points", type=int, default=192,
                        help="Number of random samples to draw (default: 192).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible sampling (default: 0).")
    parser.add_argument("--minimize", action="store_true",
                        help="Treat lower Y as better (default: maximize).")
    parser.add_argument("--grid-n", type=int, default=120,
                        help="Ternary grid resolution for the RF background.")
    parser.add_argument("--n-estimators", type=int, default=500,
                        help="Number of trees in the RF surrogate.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <run_dir>/random_baseline).")
    args = parser.parse_args()

    maximize = not args.minimize

    # 1. Rebuild the run_7eb9 RF surrogate (same as plot_run.py).
    run_dir = _resolve_run_dir(args.run)
    X_run, Y_run, labels, _run_title = load_run_source(run_dir, args.snapshot)
    print(f"Run    : {run_dir.name}")
    print(f"Loaded : {X_run.shape[0]} collected points  "
          f"Y range [{Y_run.min():.4f}, {Y_run.max():.4f}]")

    grid_pts, grid_vals = fit_rf_background(
        X_run, Y_run, args.grid_n, args.n_estimators)

    # The oracle is the same RF, evaluated at arbitrary compositions.
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators, n_jobs=-1, random_state=42)
    rf.fit(X_run, Y_run)

    # 2. Random sampling campaign on the surrogate.
    X_rand = sample_simplex(args.n_points, dim=X_run.shape[1], seed=args.seed)
    Y_rand = rf.predict(X_rand)
    best = Y_rand.max() if maximize else Y_rand.min()
    print(f"Random : {args.n_points} uniform samples (seed {args.seed})  "
          f"best {'max' if maximize else 'min'} Y = {best:.4f}")

    # 3. Plots.
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "random_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    conv_path = out_dir / "convergence.png"
    cov_path = out_dir / "coverage.png"
    plot_convergence(
        conv_path, Y_rand, maximize=maximize,
        title=f"Random baseline convergence — {run_dir.name}")
    plot_coverage(
        cov_path, X_rand, Y_rand, grid_pts, grid_vals, labels,
        title=f"Random baseline coverage — {run_dir.name}  "
              f"({args.n_points} random points)")

    print(f"Saved  : {conv_path}")
    print(f"Saved  : {cov_path}")


if __name__ == "__main__":
    main()
