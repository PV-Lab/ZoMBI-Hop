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
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path: str, log_x: bool = False, log_y: bool = False):
    df = pd.read_csv(csv_path)

    metrics = [
        ("dist_to_needles", "Distance to True Needles", "steelblue"),
        ("dup_fraction", "Duplicate Sample Fraction", "tomato"),
        ("pct_matched", "Pct Needles Matching True Needle", "seagreen"),
        ("avg_pairwise_dist", "Avg Pairwise Needle Distance", "mediumpurple"),
        ("recent_needle_value", "Most Recent Needle Value", "darkorange"),
    ]
    # Tolerate older CSVs written before a column existed.
    metrics = [m for m in metrics if m[0] in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Hparam Opt Time Series Metrics")

    for ax, (col, label, color) in zip(axes.flat, metrics):
        # recent_needle_value is a step function (holds until a new needle is
        # found) and is NaN until the first needle — steps-post + NaN gaps.
        if col == "recent_needle_value":
            ax.plot(df["iteration"], df[col], color=color,
                    drawstyle="steps-post", marker="o", ms=3)
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

    # Hide any unused panels (5 metrics in a 2x3 grid leaves one blank).
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics_over_time.csv")
    parser.add_argument("csv_path", help="Path to metrics_over_time.csv")
    parser.add_argument("--log-x", action="store_true", help="Log scale on x-axis")
    parser.add_argument("--log-y", action="store_true", help="Log scale on y-axis")
    args = parser.parse_args()
    plot_metrics(args.csv_path, log_x=args.log_x, log_y=args.log_y)
