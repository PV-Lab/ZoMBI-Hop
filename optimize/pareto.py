"""
pareto.py
=========
Collect every MOBO trial across ``runs/mobo_*/mobo_progress.json``, determine the
Pareto-optimal set of hyperparameter configurations, write it to ``pareto.json``,
and render a Pareto-front figure.

This replaces the old on-the-fly ``pareto`` flag that ``run_mobo.py`` used to
stamp into each ``mobo_progress.json`` (which was computed per-run and therefore
inconsistent). Pareto membership is a global property of all trials, so it is
determined here, after the fact, over the union of every run.

Objectives (all MINIMISED):
    dist_to_needles   – distance from discovered needles to the reference optima
    dup_fraction      – fraction of duplicated samples
    runtime_s         – wall-clock seconds per trial

Each run's ``mobo_progress.json`` records only its own trials, so the union over
all runs never double-counts (a resumed run seeds the GP with prior history but
writes only its own new trials).  ``IGNORE_mobo_*`` directories are excluded by
the ``mobo_*`` glob.

Usage
-----
  conda activate zombi-hop
  python optimize/pareto.py                 # crawl optimize/runs, write there
  python optimize/pareto.py <runs_dir>      # crawl a specific runs directory
  python optimize/pareto.py --out <dir>     # write pareto.json / .png elsewhere
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")   # headless: write PNG, never open a window
import matplotlib.pyplot as plt

OBJECTIVES = ("dist_to_needles", "dup_fraction", "runtime_s")


# ─── Collection ────────────────────────────────────────────────────────────────

def collect_trials(runs_dir: str) -> list[dict]:
    """Crawl ``runs_dir/mobo_*/mobo_progress.json`` → list of trial records.

    Each record: {source_run, trial, metrics{...}, hparams{...}}. Trials missing
    any of the three objective metrics are skipped.
    """
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "mobo_*", "mobo_progress.json"))):
        run_name = os.path.basename(os.path.dirname(path))
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  [collect] {run_name}: unreadable ({exc}); skipping.")
            continue
        used = 0
        for t in data.get("trials", []):
            m = t.get("metrics", {})
            if not all(k in m for k in OBJECTIVES):
                continue
            try:
                metrics = {k: float(m[k]) for k in OBJECTIVES}
            except (TypeError, ValueError):
                continue
            records.append({
                "source_run": run_name,
                "trial":      t.get("trial"),
                "metrics":    metrics,
                "hparams":    t.get("hparams", {}),
            })
            used += 1
        if used:
            print(f"  [collect] {run_name}: {used} trial(s)")
    return records


# ─── Pareto front (minimisation on all objectives) ─────────────────────────────

def pareto_mask_min(M: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of ``M`` (all columns minimised).

    Row j dominates row i iff ``M[j] <= M[i]`` elementwise and ``M[j] < M[i]`` in
    at least one objective. A row kept iff no other row dominates it (so points
    with identical objective vectors are all kept).
    """
    n = len(M)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(M <= M[i], axis=1) & np.any(M < M[i], axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


# ─── Visualisation ─────────────────────────────────────────────────────────────

def plot_pareto(M: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    """Pairwise objective scatter; Pareto-optimal points starred."""
    pairs = [
        (0, 2, OBJECTIVES[0], OBJECTIVES[2]),
        (0, 1, OBJECTIVES[0], OBJECTIVES[1]),
        (1, 2, OBJECTIVES[1], OBJECTIVES[2]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"MOBO Pareto front across all runs  "
                 f"(★ = Pareto-optimal, {int(mask.sum())}/{len(mask)})", fontsize=12)
    for ax, (ix, iy, xl, yl) in zip(axes, pairs):
        ax.scatter(M[~mask, ix], M[~mask, iy], c="steelblue", alpha=0.6,
                   edgecolors="k", linewidths=0.3, label="dominated")
        ax.scatter(M[mask, ix], M[mask, iy], marker="*", s=220, c="gold",
                   zorder=5, edgecolors="k", linewidths=0.5, label="Pareto")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto plot -> {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the global Pareto-optimal MOBO trials and write pareto.json.")
    parser.add_argument("runs_dir", nargs="?", default=None,
                        help="Directory containing runs/mobo_* (default: optimize/runs).")
    parser.add_argument("--out", default=None,
                        help="Output directory for pareto.json / pareto_front.png "
                             "(default: the runs directory).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.abspath(args.runs_dir) if args.runs_dir else os.path.join(script_dir, "runs")
    out_dir = os.path.abspath(args.out) if args.out else runs_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"MOBO Pareto collection  |  runs: {runs_dir}")
    print("=" * 70)

    records = collect_trials(runs_dir)
    if not records:
        sys.exit(f"No usable trials found under {runs_dir}/mobo_*/mobo_progress.json.")

    M = np.array([[r["metrics"][k] for k in OBJECTIVES] for r in records], dtype=float)
    mask = pareto_mask_min(M)
    n_total, n_pareto = len(records), int(mask.sum())
    print(f"\n  {n_total} trial(s) total -> {n_pareto} Pareto-optimal.")

    # Pareto records, best dist_to_needles first.
    pareto = [records[i] for i in np.where(mask)[0]]
    pareto.sort(key=lambda r: r["metrics"]["dist_to_needles"])

    out = {
        "generated":      datetime.datetime.now().isoformat(timespec="seconds"),
        "runs_dir":       runs_dir,
        "objectives":     {k: "minimize" for k in OBJECTIVES},
        "n_trials_total": n_total,
        "n_pareto":       n_pareto,
        "pareto":         pareto,
    }
    json_path = os.path.join(out_dir, "pareto.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  pareto.json -> {json_path}")

    plot_pareto(M, mask, os.path.join(out_dir, "pareto_front.png"))

    print("\n  Pareto-optimal configurations (best dist first):")
    for r in pareto:
        m = r["metrics"]
        print(f"    {r['source_run']} trial {r['trial']}:  "
              f"dist={m['dist_to_needles']:.4f}  dup={m['dup_fraction']:.4f}  "
              f"runtime={m['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
