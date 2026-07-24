#!/usr/bin/env python3
"""Overlay a single run's convergence curve on a uniform-random-search baseline.

Usage
-----
    python plot_vs_random_baseline.py <rep_dir> [out.png]

``<rep_dir>`` is one run directory, e.g.

    llm/results/sweep_catastrophic_20260709_223225/baseline/rep1

It must contain:

  * ``points.csv``               — the run trajectory (``Y`` column is the objective).
  * ``coverage_ground_truth.npz`` — the dense ground-truth grid (``grid_vals``) that
    uniformly covers the feasible domain; used to score uniformly-drawn samples.

The random baseline draws points uniformly at random from the same domain (the
ground-truth grid) and tracks its running-best objective, averaged over many
independent random searches. This mirrors ``run_mobo.plot_convergence``'s axes
(sample index vs. objective, running best) but shows ONLY the running-best LINES
for the run and the baseline — no scatter of individual samples.

Output defaults to ``<rep_dir>/convergence_vs_random.png``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# How many independent uniform-random searches to average for the baseline line.
N_RANDOM_SEARCHES = 500
RNG_SEED = 0


def _read_run_Y(points_csv: Path) -> np.ndarray:
    """Full objective trajectory from a rep's points.csv ``Y`` column (in order)."""
    ys: list[float] = []
    with open(points_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                ys.append(float(row["Y"]))
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(ys, dtype=float)


def _find_maximize(rep_dir: Path) -> bool:
    """Read the ``maximize`` flag from the sweep's run_config.json (walk up)."""
    for parent in [rep_dir, *rep_dir.parents]:
        cfg = parent / "run_config.json"
        if cfg.exists():
            try:
                return bool(json.loads(cfg.read_text()).get("maximize", True))
            except Exception:
                break
    return True  # sweeps in this repo maximize the objective by default


def _running_best(y: np.ndarray, maximize: bool, axis: int = -1) -> np.ndarray:
    accum = np.maximum.accumulate if maximize else np.minimum.accumulate
    return accum(y, axis=axis)


def plot_vs_random(rep_dir: Path, out_png: Path | None = None) -> Path:
    points_csv = rep_dir / "points.csv"
    gt_npz = rep_dir / "coverage_ground_truth.npz"
    if not points_csv.exists():
        raise FileNotFoundError(f"no points.csv under {rep_dir}")
    if not gt_npz.exists():
        raise FileNotFoundError(f"no coverage_ground_truth.npz under {rep_dir}")

    maximize = _find_maximize(rep_dir)

    y_run = _read_run_Y(points_csv)
    if y_run.size == 0:
        raise ValueError(f"points.csv under {rep_dir} has no usable Y values")
    n = y_run.size
    run_best = _running_best(y_run, maximize)

    grid_vals = np.load(gt_npz)["grid_vals"].ravel()
    if grid_vals.size == 0:
        raise ValueError(f"coverage_ground_truth.npz under {rep_dir} has empty grid_vals")

    # Uniform random search: for each of N_RANDOM_SEARCHES independent searches,
    # draw n points uniformly at random from the domain (the ground-truth grid)
    # and track running best. The baseline curve is the mean across searches.
    rng = np.random.default_rng(RNG_SEED)
    draws = grid_vals[rng.integers(0, grid_vals.size, size=(N_RANDOM_SEARCHES, n))]
    rand_curves = _running_best(draws, maximize, axis=1)
    rand_mean = rand_curves.mean(axis=0)
    rand_lo = np.percentile(rand_curves, 5, axis=0)
    rand_hi = np.percentile(rand_curves, 95, axis=0)

    idx = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(idx, run_best, color="darkorange", lw=1.8,
            label="run (running best)", zorder=3)
    ax.fill_between(idx, rand_lo, rand_hi, color="slategray", alpha=0.2,
                    lw=0, label="uniform random (5–95%)", zorder=1)
    ax.plot(idx, rand_mean, color="slategray", lw=1.6, ls="--",
            label=f"uniform random (mean of {N_RANDOM_SEARCHES})", zorder=2)

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Objective Y")
    ax.set_title(f"Convergence vs. uniform-random baseline  "
                 f"({n} pts, {'max' if maximize else 'min'})",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()

    if out_png is None:
        out_png = rep_dir / "convergence_vs_random.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        sys.exit(1)
    rep_dir = Path(argv[0]).expanduser()
    out_png = Path(argv[1]).expanduser() if len(argv) > 1 else None
    written = plot_vs_random(rep_dir, out_png)
    print(f"wrote {written}")


if __name__ == "__main__":
    main(sys.argv[1:])
