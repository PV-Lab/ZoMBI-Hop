"""
plot_variance.py
================
Re-run the best hyperparameter configurations several times to quantify the
run-to-run variance of the three MOBO objectives, then visualise the resulting
clusters on the same pairwise-objective layout as ``optimize/pareto.py`` (but
without marking the Pareto frontier).

Selection
---------
"Best" = the ``N_SETS`` (default 10) trials with the LOWEST ``dist_to_needles``
among trials whose ``dup_fraction`` is below ``DUP_MAX`` (default 0.4), taken
from a single hparam-optimisation run's ``mobo_progress.json`` (default
``runs/mobo_05_06_15_32``).

Re-runs
-------
Each selected configuration is re-evaluated ``RERUNS`` (default 2) more times on
the SAME Random-Forest surrogate / reference optima as the source run (read from
its ``run_config.json``), each with a ``TIME_LIMIT_MIN`` (default 10) wall-clock
budget.  Together with the ORIGINAL stored point, every configuration ends up
with ``RERUNS + 1`` (default 3) data points.

The re-runs are the expensive part (~``N_SETS * RERUNS * TIME_LIMIT_MIN`` minutes
of compute).  Results are cached to ``variance_results.json`` next to the output
figure, so re-running this script only redraws the plot.  Use ``--force`` to
discard the cache and recompute, or ``--plot-only`` to fail if no cache exists.

Only the three objective metrics are computed per re-run; the per-iteration
frames / videos / CSVs that ``run_mobo.py`` writes are intentionally skipped
(they don't affect the metrics and would dominate the wall-clock cost).

Plot
----
Three pairwise-objective scatter panels, identical axes/layout to ``pareto.py``:
    (dist_to_needles, runtime_s) | (dist_to_needles, dup_fraction) | (dup_fraction, runtime_s)
The Pareto frontier is NOT drawn.  Instead each configuration ("set") gets its
own colour; its ``RERUNS + 1`` points share that colour (the original point is
drawn as a star, re-runs as circles).  The runtime axis is fixed to 0–600 s
(0–10 min), as requested.

Usage
-----
  conda activate zombi-hop
  python optimize/plot_variance.py                      # default run, 10 sets x 2 reruns
  python optimize/plot_variance.py <mobo_run_dir>       # a different source run
  python optimize/plot_variance.py --force              # ignore cache, recompute reruns
  python optimize/plot_variance.py --plot-only          # only plot a cached result
  python optimize/plot_variance.py --time-limit-min 10 --reruns 2 --n-sets 10
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import datetime

import numpy as np

# run_mobo sets matplotlib to TkAgg on import; the static plot switches to Agg.
import matplotlib
import matplotlib.pyplot as plt

# Reuse the surrogate, metrics and ZoMBI trial machinery from run_mobo.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_mobo as rm

OBJECTIVES = ("dist_to_needles", "dup_fraction", "runtime_s")

# Same pairwise panels (and order) as pareto.py.
_PAIRS = [
    (0, 2, OBJECTIVES[0], OBJECTIVES[2]),
    (0, 1, OBJECTIVES[0], OBJECTIVES[1]),
    (1, 2, OBJECTIVES[1], OBJECTIVES[2]),
]

RUNTIME_AXIS_MAX = 600.0   # runtime axis spans 0–600 s (0–10 min), per request


# ─── Selection ──────────────────────────────────────────────────────────────────

def select_best_trials(progress_path: str, n_sets: int, dup_max: float) -> list[dict]:
    """Return the ``n_sets`` trials with lowest dist_to_needles and dup < dup_max.

    Each returned record: {"trial", "hparams", "metrics"{dist/dup/runtime}}.
    """
    with open(progress_path) as f:
        data = json.load(f)
    eligible = []
    for t in data.get("trials", []):
        m = t.get("metrics", {})
        if not all(k in m for k in OBJECTIVES):
            continue
        try:
            metrics = {k: float(m[k]) for k in OBJECTIVES}
        except (TypeError, ValueError):
            continue
        if metrics["dup_fraction"] >= dup_max:
            continue
        eligible.append({
            "trial":   t.get("trial"),
            "hparams": t.get("hparams", {}),
            "metrics": metrics,
        })
    eligible.sort(key=lambda r: r["metrics"]["dist_to_needles"])
    return eligible[:n_sets]


# ─── A single lightweight re-run (metrics only, no frames) ───────────────────────

def run_trial_metrics(hparams: dict, rf_fn, true_optima, maximize: bool,
                      time_limit_hours: float) -> dict:
    """Run one time-limited ZoMBI trial on the RF surrogate; return its metrics.

    Mirrors the *timed* core of ``run_mobo.run_single_trial`` but skips every
    per-iteration payload / frame / CSV artifact, since the variance plot only
    needs the three objective values.
    """
    plot_state: dict = {"line_0": None, "line_1": None}
    sim_obj = rm.make_sim_obj(rf_fn, rm.DEVICE, rm.DTYPE, maximize=maximize)
    inner   = rm.make_linebo_wrapper(sim_obj, 3, rm.NUM_LINES, rm.DEVICE, rm.DTYPE, plot_state)

    def obj_wrapper(x_tell, bounds, acq_fn):
        return inner(x_tell, bounds, acq_fn)

    X_a, X_e, Y = rm._gen_init_data(rf_fn, maximize)
    optimizer = rm.ZoMBIHop(
        objective=obj_wrapper,
        X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
        **rm.ZOMBI_FIXED, **hparams,
        device=str(rm.DEVICE), dtype=rm.DTYPE,
        run_uuid=None, checkpoint_dir=None,
    )
    dh = optimizer.data_handler

    t0 = time.time()
    try:
        optimizer.run(max_activations=float("inf"), time_limit_hours=time_limit_hours)
    except Exception as exc:
        print(f"      [rerun] ZoMBI crashed: {exc}")
    runtime = time.time() - t0

    needle_t   = dh.get_all_needle_locations()
    discovered = needle_t.detach().cpu().numpy() if needle_t.numel() > 0 else np.empty((0, 3))
    X_all_np   = (dh.X_all_actual.detach().cpu().numpy()
                  if dh.X_all_actual is not None else np.empty((0, 3)))
    dist = rm.metric_dist_to_needles(discovered, true_optima)
    dup  = rm.metric_dup_fraction(X_all_np, rm.NOISE_LEVEL / 2.0)
    print(f"      [rerun]  dist={dist:.4f}  dup={dup:.4f}  t={runtime:.1f}s  "
          f"needles={len(discovered)}/{len(true_optima)}")
    return {"dist_to_needles": dist, "dup_fraction": dup, "runtime_s": runtime}


# ─── Compute (or load cached) variance data ─────────────────────────────────────

def compute_variance(mobo_run_dir: str, *, n_sets: int, dup_max: float,
                     reruns: int, time_limit_min: float) -> dict:
    """Select the best trials and re-evaluate each ``reruns`` times on the RF."""
    progress_path = os.path.join(mobo_run_dir, "mobo_progress.json")
    config_path   = os.path.join(mobo_run_dir, "run_config.json")
    if not os.path.exists(progress_path):
        sys.exit(f"No mobo_progress.json under {mobo_run_dir}.")
    if not os.path.exists(config_path):
        sys.exit(f"No run_config.json under {mobo_run_dir} (need RF / optima config).")

    with open(config_path) as f:
        cfg = json.load(f)
    maximize    = bool(cfg["maximize"])
    csv_path    = cfg["csv_path"]
    true_optima = [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
    if not os.path.exists(csv_path):
        sys.exit(f"Surrogate CSV no longer exists: {csv_path}")

    best = select_best_trials(progress_path, n_sets, dup_max)
    if not best:
        sys.exit(f"No trials with dup_fraction < {dup_max} found in {progress_path}.")
    print(f"  Selected {len(best)} configuration(s) "
          f"(lowest dist_to_needles with dup_fraction < {dup_max}):")
    for r in best:
        m = r["metrics"]
        print(f"    trial {r['trial']}:  dist={m['dist_to_needles']:.4f}  "
              f"dup={m['dup_fraction']:.4f}  runtime={m['runtime_s']:.1f}s")

    print(f"\n  Rebuilding RF surrogate from {csv_path} "
          f"({'maximize' if maximize else 'minimize'}, "
          f"{len(true_optima)} reference optima) …")
    _, rf_fn, _grid_pts, _grid_vals = rm.build_rf_and_grid(csv_path)
    print("  RF ready.")

    time_limit_hours = time_limit_min / 60.0
    sets = []
    total = len(best) * reruns
    done = 0
    for r in best:
        print(f"\n  === Configuration trial {r['trial']}  "
              f"({reruns} re-run(s) @ {time_limit_min} min) ===")
        rerun_metrics = []
        for k in range(reruns):
            done += 1
            print(f"    [re-run {k + 1}/{reruns}]  (overall {done}/{total})")
            rerun_metrics.append(
                run_trial_metrics(r["hparams"], rf_fn, true_optima, maximize,
                                  time_limit_hours)
            )
        sets.append({
            "trial":    r["trial"],
            "hparams":  r["hparams"],
            "original": r["metrics"],
            "reruns":   rerun_metrics,
        })

    return {
        "generated":      datetime.datetime.now().isoformat(timespec="seconds"),
        "mobo_run_dir":   os.path.abspath(mobo_run_dir),
        "n_sets":         n_sets,
        "dup_max":        dup_max,
        "reruns":         reruns,
        "time_limit_min": time_limit_min,
        "maximize":       maximize,
        "sets":           sets,
    }


# ─── Plot ────────────────────────────────────────────────────────────────────────

def plot_variance(result: dict, out_path: str) -> None:
    """Pairwise objective scatter (pareto.py layout); one colour per config set."""
    matplotlib.use("Agg")
    plt.switch_backend("Agg")

    sets = result["sets"]
    n_sets = len(sets)
    cmap = plt.get_cmap("tab10" if n_sets <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(n_sets)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"MOBO re-run variance  ({n_sets} configs x "
        f"{result['reruns'] + 1} pts; ★ = original, ● = re-run @ "
        f"{result['time_limit_min']} min)",
        fontsize=12,
    )

    for ax, (ix, iy, xl, yl) in zip(axes, _PAIRS):
        for s, color in zip(sets, colors):
            pts = [s["original"]] + s["reruns"]
            P = np.array([[p[OBJECTIVES[ix]], p[OBJECTIVES[iy]]] for p in pts])
            # re-runs (circles)
            ax.scatter(P[1:, 0], P[1:, 1], color=color, s=55, alpha=0.85,
                       edgecolors="k", linewidths=0.3, zorder=3)
            # original (star)
            ax.scatter(P[0, 0], P[0, 1], color=color, marker="*", s=240,
                       edgecolors="k", linewidths=0.5, zorder=4,
                       label=f"trial {s['trial']}")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        # Fix the runtime axis to 0–600 s (0–10 min) wherever it appears.
        if OBJECTIVES[ix] == "runtime_s":
            ax.set_xlim(0, RUNTIME_AXIS_MAX)
        if OBJECTIVES[iy] == "runtime_s":
            ax.set_ylim(0, RUNTIME_AXIS_MAX)

    # Single shared legend (sets are identical across panels).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8,
               title="config set", framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  variance plot -> {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_run = os.path.join(script_dir, "runs", "mobo_05_06_15_32")

    parser = argparse.ArgumentParser(
        description="Re-run best MOBO configs to assess variance and plot the clusters.")
    parser.add_argument("mobo_run_dir", nargs="?", default=default_run,
                        help="Source hparam run dir (default: runs/mobo_05_06_15_32).")
    parser.add_argument("--n-sets", type=int, default=10,
                        help="Number of best configurations to re-run (default: 10).")
    parser.add_argument("--dup-max", type=float, default=0.4,
                        help="Only consider trials with dup_fraction below this (default: 0.4).")
    parser.add_argument("--reruns", type=int, default=2,
                        help="Extra re-runs per configuration (default: 2; +1 original = 3 pts).")
    parser.add_argument("--time-limit-min", type=float, default=10.0,
                        help="Wall-clock budget per re-run, minutes (default: 10).")
    parser.add_argument("--out", default=None,
                        help="Output directory for variance.png / variance_results.json "
                             "(default: the source run dir).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore any cached variance_results.json and recompute.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Plot from cached variance_results.json only; never recompute.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.mobo_run_dir)
    out_dir = os.path.abspath(args.out) if args.out else run_dir
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "variance_results.json")
    plot_path  = os.path.join(out_dir, "variance.png")

    print("=" * 70)
    print(f"MOBO re-run variance  |  source: {run_dir}")
    print("=" * 70)

    result = None
    if (args.plot_only or not args.force) and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                result = json.load(f)
            print(f"  Loaded cached results from {cache_path} "
                  f"({len(result.get('sets', []))} set(s)).")
        except Exception as exc:
            print(f"  Cache unreadable ({exc}); will recompute.")
            result = None

    if result is None:
        if args.plot_only:
            sys.exit(f"--plot-only set but no usable cache at {cache_path}.")
        result = compute_variance(
            run_dir, n_sets=args.n_sets, dup_max=args.dup_max,
            reruns=args.reruns, time_limit_min=args.time_limit_min,
        )
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Cached results -> {cache_path}")

    plot_variance(result, plot_path)


if __name__ == "__main__":
    main()
