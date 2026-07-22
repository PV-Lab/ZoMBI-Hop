"""
Summarise a warm-start vs cold-start comparison run.

Reads ``summary.csv`` written by :mod:`warm_start.compare` and reports the
comparison **paired by repetition**.  Pairing matters more than the headline
averages here: the Ensemble landscape is re-randomized per repetition and varies a
great deal in difficulty, so the spread of ``dist_to_needles`` *across* landscapes
is much larger than the warm-start effect *within* one.  Comparing arm means would
mostly measure which landscapes happened to land in which arm — except that both
arms see the same landscape, so the per-repetition difference cancels that
variation entirely and is the quantity worth reading.

Writes ``comparison.png`` and ``comparison.md`` next to the summary, and prints the
table.

Usage
-----
    uv run python -m warm_start.analyze --run-dir warm_start/runs/compare_3d
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

#: Metrics to compare, as ``(column, human label, lower_is_better)``.
METRICS = [
    ("dist", "dist_to_needles", True),
    ("best_y", "best objective found", False),
    ("n_needles", "needles found", False),
    ("dup", "duplicate fraction", True),
]


def load_summary(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no summary.csv in {run_dir}")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = dict(r)
            row["rep"] = int(float(r["rep"]))
            for k in ("dist", "best_y", "dup", "n_needles", "n_points", "n_iters",
                      "n_init", "runtime"):
                try:
                    row[k] = float(r[k])
                except (TypeError, ValueError):
                    row[k] = float("nan")
            rows.append(row)
    return rows


def paired(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Per-repetition ``(cold, warm)`` values for `key`, over complete pairs only."""
    by = {}
    for r in rows:
        by.setdefault(r["rep"], {})[r["arm"]] = r
    reps = sorted(k for k, v in by.items() if "cold" in v and "warm" in v)
    cold = np.array([by[k]["cold"][key] for k in reps], dtype=float)
    warm = np.array([by[k]["warm"][key] for k in reps], dtype=float)
    return cold, warm, reps


def _fmt(v: float, nd: int = 4) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def build_report(rows: list[dict]) -> str:
    out: list[str] = []
    _, _, reps = paired(rows, "dist")
    n = len(reps)
    out.append(f"# Warm start vs cold start — {n} paired repetition(s)\n")
    if n == 0:
        out.append("No complete repetitions yet.\n")
        return "\n".join(out)

    for key, label, lower_better in METRICS:
        cold, warm, _ = paired(rows, key)
        diff = warm - cold
        better = diff < 0 if lower_better else diff > 0
        direction = "lower is better" if lower_better else "higher is better"
        out.append(f"\n## {label}  ({direction})\n")
        out.append("| rep | cold | warm | warm − cold | warm better |")
        out.append("|----:|-----:|-----:|------------:|:-----------:|")
        for i, r in enumerate(reps):
            out.append(f"| {r} | {_fmt(cold[i])} | {_fmt(warm[i])} | "
                       f"{diff[i]:+.4f} | {'yes' if better[i] else 'no'} |")
        out.append(f"| **mean** | **{_fmt(np.nanmean(cold))}** | "
                   f"**{_fmt(np.nanmean(warm))}** | "
                   f"**{np.nanmean(diff):+.4f}** | "
                   f"**{int(np.sum(better))}/{n}** |")
        sd = np.nanstd(diff, ddof=1) if n > 1 else float("nan")
        out.append(f"\nPaired difference: mean {np.nanmean(diff):+.4f}, sd {_fmt(sd)}.")

    out.append(
        "\n\n## Reading this\n\n"
        f"With {n} repetitions the per-metric win count and the sign of the mean "
        "paired difference are the honest summary; a p-value on 5 pairs would not "
        "be meaningful, so none is quoted. Treat a result as suggestive unless the "
        "warm arm wins in most repetitions *and* the mean difference is large "
        "relative to its spread. If the two arms look close, that is itself the "
        "finding: the warm start cost 96 of the 600-point budget, so parity means "
        "the seeds bought back exactly what they cost."
    )
    return "\n".join(out)


def plot(rows: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [m for m in METRICS]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, label, lower_better) in zip(axes, metrics):
        cold, warm, reps = paired(rows, key)
        # Slope chart: one line per repetition. Because the arms share a landscape,
        # the *slope* is the effect and the vertical spread between lines is the
        # landscape-to-landscape variation the pairing removes.
        for i, r in enumerate(reps):
            improved = (warm[i] < cold[i]) if lower_better else (warm[i] > cold[i])
            ax.plot([0, 1], [cold[i], warm[i]], "-o", ms=5, lw=1.6,
                    color="tab:green" if improved else "tab:red", alpha=0.75,
                    label="warm better" if improved else "cold better")
        if len(reps):
            ax.plot([0, 1], [np.nanmean(cold), np.nanmean(warm)], "-s", ms=9, lw=3,
                    color="black", zorder=5, label="mean")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["cold", "warm"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_title(f"{label}\n({'lower' if lower_better else 'higher'} is better)",
                     fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        # De-duplicate the per-line legend labels.
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=7, loc="best")

    fig.suptitle("Warm start vs cold start (paired by landscape), 3d, "
                 "equal 600-point budget", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-dir", default="warm_start/runs/compare_3d")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    rows = load_summary(run_dir)
    report = build_report(rows)
    print(report)

    # encoding pinned: the report uses non-ASCII glyphs (the U+2212 minus in the
    # "warm − cold" headers), and open()'s default is the locale codec — cp1252 on
    # Windows, which cannot encode them and would abort the write.
    with open(os.path.join(run_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    try:
        plot(rows, os.path.join(run_dir, "comparison.png"))
        print(f"\nwrote comparison.md and comparison.png in {run_dir}")
    except Exception as exc:
        print(f"\nplot failed: {exc}")


if __name__ == "__main__":
    main()
