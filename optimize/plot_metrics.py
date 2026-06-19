"""
Plot metrics_over_time.csv from a ZoMBI-Hop MOBO run.

Usage:
    python optimize/plot_metrics.py <csv_path> [--log-x] [--log-y]

Examples:
    python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv
    python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv --log-y
    python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv --log-x --log-y
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import MATCH_RADIUS


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

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Hparam Opt Time Series Metrics")

    used = 0
    for ax, (col, label, color, steps) in zip(axes.flat, metrics):
        used += 1
        if steps:
            ax.plot(df["iteration"], df[col], color=color, drawstyle="steps-post", marker="o", ms=3)
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
    has_ilr = "pct_matched_ilr" in df.columns
    if (has_comp or has_ilr) and used < len(axes.flat):
        pct_axes = axes.flat[used]
        if "pct_matched_comp" in df.columns:
            pct_axes.plot(df["iteration"], df["pct_matched_comp"],
                          color="seagreen", label=f"comp ≤ {MATCH_RADIUS}")
        elif "pct_matched" in df.columns:
            pct_axes.plot(df["iteration"], df["pct_matched"],
                          color="seagreen", label=f"comp ≤ {MATCH_RADIUS} (legacy)")
        if has_ilr:
            pct_axes.plot(df["iteration"], df["pct_matched_ilr"],
                          color="darkcyan", label="ILR (scaled)")
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
    parser.add_argument("csv_path", help="Path to metrics_over_time.csv")
    parser.add_argument("--log-x", action="store_true", help="Log scale on x-axis")
    parser.add_argument("--log-y", action="store_true", help="Log scale on y-axis")
    args = parser.parse_args()
    plot_metrics(args.csv_path, log_x=args.log_x, log_y=args.log_y)
