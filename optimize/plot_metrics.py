"""
Plot metrics_over_time.csv from a ZoMBI-Hop MOBO run.

Usage:
    python optimize/plot_metrics.py <csv_path> [--log-x] [--log-y]
    python optimize/plot_metrics.py --needle-values <run_dir> [--out PNG]

Examples:
    python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv
    python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv --log-y
    python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv --log-x --log-y
    python optimize/plot_metrics.py --needle-values optimize/runs/.../trial_0/run_1

``--needle-values`` writes the standalone needle-value-vs-iteration plot
(``needle_values.png``) from a finished run's CSVs, so it can be backfilled onto
runs that predate it without re-running the optimizer.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import MATCH_RADIUS


def needle_discoveries(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """``(iterations, values)`` of each needle, in discovery order.

    ``needles.csv`` records the iteration each needle was *declared* on, which is
    the only exact source: ``metrics_over_time.csv``'s ``recent_needle_value`` is
    sampled one iteration late (the payload is captured after the objective call
    that declared the needle), and reading discovery times off where that column
    changes also misses needles whose value repeats.
    """
    path = os.path.join(run_dir, "needles.csv")
    if not os.path.isfile(path):
        return np.empty(0), np.empty(0)
    df = pd.read_csv(path)
    if df.empty or "iteration" not in df.columns or "value" not in df.columns:
        return np.empty(0), np.empty(0)
    df = df.dropna(subset=["iteration", "value"]).sort_values("iteration")
    return df["iteration"].to_numpy(dtype=float), df["value"].to_numpy(dtype=float)


def ensemble_true_best(run_dir: str) -> float | None:
    """Noiseless objective value at the landscape's true optima, or None.

    Rebuilt from the run's ``ensemble_config.json`` (which fully determines the
    landscape). Every true optimum of an ``Ensemble`` sits at the same analytic
    peak height, so the max over them is the best value the surface can return.
    Observed Y can still exceed it: ``make_sim_obj`` adds multiplicative output
    noise to each measurement.
    """
    cfg_path = os.path.join(run_dir, "ensemble_config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from synthetic_data.ensemble import Ensemble
        with open(cfg_path) as f:
            cfg = json.load(f)
        vals = [v for _, v in Ensemble(**cfg).known_maxima]
        return float(max(vals)) if vals else None
    except Exception:
        return None


def best_found_objective(run_dir: str) -> float | None:
    """Best objective value actually observed in the run (max Y over all points)."""
    path = os.path.join(run_dir, "points.csv")
    if not os.path.isfile(path):
        return None
    try:
        y = pd.read_csv(path, usecols=["Y"])["Y"].to_numpy(dtype=float)
    except Exception:
        return None
    return float(np.nanmax(y)) if y.size else None


def plot_needle_values(run_dir: str, save_path: str | None = None, *,
                       true_best: float | None = None,
                       best_found: float | None = None,
                       title: str | None = None) -> str | None:
    """Standalone "most recent needle value" plot with true-best / best-found lines.

    Each needle is drawn at the iteration it was declared on (from ``needles.csv``),
    with a steps-post line carrying its value forward until the next needle — so the
    horizontal run-lengths are the real gaps between discoveries rather than one
    marker per iteration, which made every needle look uniformly spaced.

    ``true_best`` and ``best_found`` default to the ensemble landscape's analytic
    optimum and the run's max observed Y; either can be passed in by a caller that
    already knows them. Returns the written path, or None if there is nothing to
    plot.
    """
    run_dir = os.path.abspath(run_dir)
    save_path = save_path or os.path.join(run_dir, "needle_values.png")
    iters, vals = needle_discoveries(run_dir)

    n_iters = None
    mot = os.path.join(run_dir, "metrics_over_time.csv")
    if os.path.isfile(mot):
        try:
            n_iters = float(pd.read_csv(mot, usecols=["iteration"])["iteration"].max())
        except Exception:
            n_iters = None
    if n_iters is None and iters.size:
        n_iters = float(iters[-1])
    if n_iters is None:
        return None

    if true_best is None:
        true_best = ensemble_true_best(run_dir)
    if best_found is None:
        best_found = best_found_objective(run_dir)

    fig, ax = plt.subplots(figsize=(8, 4))
    if iters.size:
        # Carry each needle's value forward to the next discovery, and out to the
        # last iteration, so the step widths read as time-between-needles.
        step_x = np.concatenate([iters, [n_iters]])
        step_y = np.concatenate([vals, [vals[-1]]])
        ax.plot(step_x, step_y, color="darkorange", lw=1.6, drawstyle="steps-post",
                zorder=3, label="most recent needle value")
        ax.plot(iters, vals, "o", ms=4, color="darkorange", zorder=4,
                label=f"needle found ({len(iters)})")
    else:
        ax.text(0.5, 0.5, "no needles found", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888888")

    if true_best is not None:
        ax.axhline(true_best, color="seagreen", ls="--", lw=1.4, zorder=2,
                   label=f"true best ({true_best:.4f})")
    if best_found is not None:
        ax.axhline(best_found, color="steelblue", ls=":", lw=1.4, zorder=2,
                   label=f"best found ({best_found:.4f})")

    ax.set_xlim(0, n_iters)
    ax.set_xlabel("Iteration (measured line)")
    ax.set_ylabel("Objective Y")
    ax.set_title(title or "Most Recent Needle Value", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_metrics(csv_path: str, log_x: bool = False, log_y: bool = False,
                 save_path: str | None = None):
    """Plot the time-series metrics from ``metrics_over_time.csv``.

    When ``save_path`` is given the figure is written there (PNG) and closed —
    no interactive window — so callers like ``run_mobo.py`` can generate the plot
    automatically at the end of a trial. Otherwise the figure is shown.
    """
    df = pd.read_csv(csv_path)

    metrics = [
        ("dist_to_needles", "Distance to True Needles", "steelblue", False),
        ("dup_fraction", "Duplicate Sample Fraction", "tomato", False),
        ("avg_pairwise_dist", "Avg Pairwise Needle Distance", "mediumpurple", False),
        ("recent_needle_value", "Most Recent Needle Value", "darkorange", True),
    ]
    metrics = [m for m in metrics if m[0] in df.columns]

    # Iterations on which a needle was actually declared. Marking every row (the
    # old ``marker="o"`` on the whole series) put a dot on all ~600 iterations, so
    # needles read as evenly spaced no matter when they were really found. Prefer
    # needles.csv's recorded discovery iteration; fall back to the iteration
    # n_needles increments on, and only then to a change in the value itself.
    disc_iters, disc_vals = needle_discoveries(str(Path(csv_path).resolve().parent))
    if disc_iters.size == 0:
        if "n_needles" in df.columns:
            grew = df["n_needles"].diff().fillna(df["n_needles"]) > 0
        else:
            rv = df.get("recent_needle_value")
            grew = rv.notna() & (rv != rv.shift()) if rv is not None else None
        if grew is not None and grew.any():
            disc_iters = df.loc[grew, "iteration"].to_numpy(dtype=float)
            disc_vals = df.loc[grew, "recent_needle_value"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Hparam Opt Time Series Metrics")

    used = 0
    for ax, (col, label, color, steps) in zip(axes.flat, metrics):
        used += 1
        if steps:
            # Step the value between discoveries; dot ONLY the discovery iterations.
            if disc_iters.size:
                # Drive the step off the recorded discovery iterations too, so the
                # markers sit exactly on their own risers (the CSV column lags them
                # by one iteration, see needle_discoveries).
                last_it = float(df["iteration"].max())
                ax.plot(np.concatenate([disc_iters, [last_it]]),
                        np.concatenate([disc_vals, [disc_vals[-1]]]),
                        color=color, drawstyle="steps-post")
                ax.plot(disc_iters, disc_vals, "o", ms=4, color=color,
                        label=f"needle found ({disc_iters.size})")
                ax.legend(fontsize=7, loc="lower right")
            else:
                ax.plot(df["iteration"], df[col], color=color, drawstyle="steps-post")
        else:
            ax.plot(df["iteration"], df[col], color=color)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    has_comp = "pct_matched_comp" in df.columns or "pct_matched" in df.columns
    if has_comp and used < len(axes.flat):
        pct_axes = axes.flat[used]
        if "pct_matched_comp" in df.columns:
            pct_axes.plot(df["iteration"], df["pct_matched_comp"],
                          color="seagreen", label=f"comp ≤ {MATCH_RADIUS}")
        elif "pct_matched" in df.columns:
            pct_axes.plot(df["iteration"], df["pct_matched"],
                          color="seagreen", label=f"comp ≤ {MATCH_RADIUS} (legacy)")
        pct_axes.set_xlabel("Iteration")
        pct_axes.set_ylabel("Pct matched")
        pct_axes.set_title("Pct Needles Matching True Optimum")
        pct_axes.legend(fontsize=8)
        pct_axes.grid(True, alpha=0.3)
        if log_x:
            pct_axes.set_xscale("log")
        if log_y:
            pct_axes.set_yscale("log")
        used += 1

    for ax in axes.flat[used:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics_over_time.csv")
    parser.add_argument("csv_path", help="Path to metrics_over_time.csv, or a run "
                                         "directory with --needle-values")
    parser.add_argument("--log-x", action="store_true", help="Log scale on x-axis")
    parser.add_argument("--log-y", action="store_true", help="Log scale on y-axis")
    parser.add_argument("--needle-values", action="store_true",
                        help="write needle_values.png for the given run directory")
    parser.add_argument("--out", default=None, help="output PNG (with --needle-values)")
    args = parser.parse_args()
    if args.needle_values:
        out = plot_needle_values(args.csv_path, args.out)
        print(out or "nothing to plot")
    else:
        plot_metrics(args.csv_path, log_x=args.log_x, log_y=args.log_y)
