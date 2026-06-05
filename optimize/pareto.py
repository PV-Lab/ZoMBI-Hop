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
  python optimize/pareto.py --no-interactive # save static PNG instead of live window
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import datetime

import subprocess
import platform

import numpy as np

import matplotlib
# Backend is set later: "Agg" for static PNG, system default for interactive.
import matplotlib.pyplot as plt

OBJECTIVES = ("dist_to_needles", "dup_fraction", "runtime_s")


# ─── Collection ────────────────────────────────────────────────────────────────

def collect_trials(runs_dir: str, *, exclude_old: bool = False) -> list[dict]:
    """Crawl ``runs_dir/mobo_*/mobo_progress.json`` → list of trial records.

    Each record: {source_run, trial, metrics{...}, hparams{...}}. Trials missing
    any of the three objective metrics are skipped.
    """
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "mobo_*", "mobo_progress.json"))):
        run_name = os.path.basename(os.path.dirname(path))
        if exclude_old and run_name == "mobo_old_jackson":
            continue
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

_PAIRS = [
    (0, 2, OBJECTIVES[0], OBJECTIVES[2]),
    (0, 1, OBJECTIVES[0], OBJECTIVES[1]),
    (1, 2, OBJECTIVES[1], OBJECTIVES[2]),
]


def plot_pareto(M: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    """Pairwise objective scatter; Pareto-optimal points starred (static PNG)."""
    matplotlib.use("Agg")
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"MOBO Pareto front across all runs  "
                 f"(★ = Pareto-optimal, {int(mask.sum())}/{len(mask)})", fontsize=12)
    for ax, (ix, iy, xl, yl) in zip(axes, _PAIRS):
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


def _open_file(path: str) -> None:
    """Open a file with the OS default viewer."""
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _final_plot_for_trial(runs_dir: str, source_run: str, trial: int) -> str | None:
    """Return the path of the last iter_*.png for a trial, or None."""
    plots_dir = os.path.join(runs_dir, source_run, f"trial_{trial}", "plots")
    pngs = sorted(glob.glob(os.path.join(plots_dir, "iter_*.png")))
    return pngs[-1] if pngs else None


def plot_pareto_interactive(
    M: np.ndarray,
    mask: np.ndarray,
    records: list[dict],
    runs_dir: str,
) -> None:
    """Interactive Pareto plot: hover highlights across all subplots, click opens trial image."""
    pareto_idx = np.where(mask)[0]
    pareto_M = M[pareto_idx]
    n_pareto = len(pareto_idx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    fig.suptitle(
        f"MOBO Pareto front across all runs  "
        f"(★ = Pareto-optimal, {n_pareto}/{len(mask)})  —  hover/click stars",
        fontsize=12,
    )

    for ax, (ix, iy, xl, yl) in zip(axes, _PAIRS):
        ax.scatter(
            M[~mask, ix], M[~mask, iy],
            c="steelblue", alpha=0.6, edgecolors="k", linewidths=0.3, label="dominated",
        )
        ax.scatter(
            pareto_M[:, ix], pareto_M[:, iy],
            marker="*", s=220, c="gold", zorder=5,
            edgecolors="k", linewidths=0.5, label="Pareto",
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)

    highlight_artists = []
    for ax, (ix, iy, _, _) in zip(axes, _PAIRS):
        hl = ax.scatter(
            [], [], marker="*", s=400, c="red", zorder=10,
            edgecolors="k", linewidths=1.0,
        )
        highlight_artists.append(hl)

    tooltip = fig.text(0.5, 0.01, "", ha="center", fontsize=9, color="gray")

    active_idx = [None]

    def _nearest_pareto(event) -> int | None:
        if event.inaxes is None:
            return None
        ax = event.inaxes
        try:
            panel = list(axes).index(ax)
        except ValueError:
            return None
        ix, iy = _PAIRS[panel][0], _PAIRS[panel][1]
        dx = pareto_M[:, ix] - event.xdata
        dy = pareto_M[:, iy] - event.ydata
        sx = ax.get_xlim()
        sy = ax.get_ylim()
        x_range = sx[1] - sx[0]
        y_range = sy[1] - sy[0]
        if x_range == 0 or y_range == 0:
            return None
        dist = np.sqrt((dx / x_range) ** 2 + (dy / y_range) ** 2)
        best = int(np.argmin(dist))
        if dist[best] < 0.05:
            return best
        return None

    def _on_motion(event):
        idx = _nearest_pareto(event)
        if idx == active_idx[0]:
            return
        active_idx[0] = idx
        if idx is None:
            for hl in highlight_artists:
                hl.set_offsets(np.empty((0, 2)))
            tooltip.set_text("")
        else:
            for hl, (ix, iy, _, _) in zip(highlight_artists, _PAIRS):
                hl.set_offsets([[pareto_M[idx, ix], pareto_M[idx, iy]]])
            rec = records[pareto_idx[idx]]
            m = rec["metrics"]
            tooltip.set_text(
                f"{rec['source_run']} trial {rec['trial']}  |  "
                f"dist={m['dist_to_needles']:.4f}  dup={m['dup_fraction']:.4f}  "
                f"runtime={m['runtime_s']:.1f}s"
            )
        fig.canvas.draw_idle()

    def _on_click(event):
        idx = _nearest_pareto(event)
        if idx is None:
            return
        rec = records[pareto_idx[idx]]
        img = _final_plot_for_trial(runs_dir, rec["source_run"], rec["trial"])
        if img:
            print(f"  Opening: {img}")
            _open_file(img)
        else:
            print(f"  No plots found for {rec['source_run']}/trial_{rec['trial']}")

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    print("  Interactive Pareto plot open. Hover stars to highlight, click to open trial image.")
    plt.show()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the global Pareto-optimal MOBO trials and write pareto.json.")
    parser.add_argument("runs_dir", nargs="?", default=None,
                        help="Directory containing runs/mobo_* (default: optimize/runs).")
    parser.add_argument("--out", default=None,
                        help="Output directory for pareto.json / pareto_front.png "
                             "(default: the runs directory).")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Save a static PNG instead of opening the interactive window.")
    parser.add_argument("--no-old", action="store_true",
                        help="Exclude trials from mobo_old_jackson.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.abspath(args.runs_dir) if args.runs_dir else os.path.join(script_dir, "runs")
    out_dir = os.path.abspath(args.out) if args.out else runs_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"MOBO Pareto collection  |  runs: {runs_dir}")
    print("=" * 70)

    records = collect_trials(runs_dir, exclude_old=args.no_old)
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

    if args.no_interactive:
        plot_pareto(M, mask, os.path.join(out_dir, "pareto_front.png"))
    else:
        plot_pareto_interactive(M, mask, records, runs_dir)

    print("\n  Pareto-optimal configurations (best dist first):")
    for r in pareto:
        m = r["metrics"]
        print(f"    {r['source_run']} trial {r['trial']}:  "
              f"dist={m['dist_to_needles']:.4f}  dup={m['dup_fraction']:.4f}  "
              f"runtime={m['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
