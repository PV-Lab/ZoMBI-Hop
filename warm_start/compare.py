"""
Side-by-side comparison: warm-started vs cold-started ZoMBI-Hop in 3d.

Runs ``--reps`` repetitions of a paired experiment.  Within a repetition both arms
face the **same** randomized Ensemble landscape and the same hyperparameters, and
each spends the same total measurement budget; across repetitions the landscape and
the initial designs are re-randomized.  Pairing is what makes 5 repetitions
informative — landscape-to-landscape variation is large compared with the effect we
are looking for, and it cancels in the per-repetition difference.

The two arms
------------
``cold``  stock init: 2 random simplex lines (48 points), then adaptive sampling
          until the budget is spent.

``warm``  a line-constrained space-filling design (4 lines, 96 points) scored on a
          *partial* objective and registered with the GP as high-noise observations
          (see :mod:`warm_start.seed_prior`), then adaptive sampling until the same
          budget is spent.

The warm arm pays for its seeds out of the shared budget, so this answers "is the
warm start worth what it costs?" rather than the easier question of whether extra
data helps.

Artifacts
---------
Per trial, the full ``run_mobo`` set (points.csv, needles.csv,
metrics_over_time.csv, convergence.png, dist_from_centre.png, line_length_hist.png,
plots/, conet) under ``<run-dir>/rep<k>/<arm>/``, plus ``summary.csv`` and
``summary.json`` at the run root.  Completed trials are skipped on resume, so a
requeued job continues rather than restarting.

Usage
-----
    uv run python -m warm_start.compare --run-dir warm_start/runs/compare_3d
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "optimize")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optimize import run_mobo as rm

from warm_start.trial import ARMS, MEASUREMENT_BUDGET, run_trial

DEFAULT_REPS = 5
DEFAULT_DIM = 3
OPTIMA_MARGIN = 0.2

SUMMARY_FIELDS = [
    "rep", "arm", "dist", "dup", "best_y", "n_needles", "n_true_optima",
    "n_points", "n_init", "n_iters", "runtime", "avg_time_per_iter",
    "budget", "budget_hit", "y_std", "seed_var", "real_var",
]


def landscape_for_rep(dim: int, rep: int, base_seed: int):
    """Build the (landscape, ensemble_config) pair for repetition `rep`.

    Both arms of a repetition are handed this identical pair, which is what makes
    the comparison paired.  ``base_seed`` shifts the whole family so a rerun can be
    made independent of a previous one.
    """
    seed = base_seed + rep
    landscape = rm.build_ensemble_landscape(
        dim, optima_margin=OPTIMA_MARGIN, seed=seed, time_limit_hours=None)
    config = rm.random_ensemble_config(
        dim, index=0, total=1, seed=seed, optima_margin=OPTIMA_MARGIN)
    return landscape, config


def _write_summary(run_dir: str, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["rep"], r["arm"]))
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)


def _load_completed(run_dir: str, reps: int) -> dict:
    """Metrics of trials that already finished, keyed ``(rep, arm)``."""
    done = {}
    for rep in range(reps):
        for arm in ARMS:
            path = os.path.join(run_dir, f"rep{rep}", arm, "metrics.json")
            if os.path.isfile(path):
                try:
                    with open(path) as f:
                        done[(rep, arm)] = json.load(f)["metrics"]
                except Exception:
                    pass
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--run-dir", default="warm_start/runs/compare_3d")
    p.add_argument("--dim", type=int, default=DEFAULT_DIM)
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--budget", type=int, default=MEASUREMENT_BUDGET,
                   help="measured compositions per trial, both arms")
    p.add_argument("--base-seed", type=int, default=0,
                   help="shifts the landscape family; change for an independent rerun")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "compare_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    done = _load_completed(run_dir, args.reps)
    if done:
        print(f"[compare] resuming — {len(done)} trial(s) already complete")

    results = list(done.values())
    total = args.reps * len(args.arms)
    n = 0
    for rep in range(args.reps):
        landscape, ens_cfg = landscape_for_rep(args.dim, rep, args.base_seed)
        for arm in args.arms:
            n += 1
            if (rep, arm) in done:
                print(f"[compare] ({n}/{total}) rep{rep} {arm} — already done, skipping")
                continue
            print(f"\n[compare] ({n}/{total}) rep{rep} {arm}", flush=True)
            trial_dir = os.path.join(run_dir, f"rep{rep}", arm)
            try:
                metrics = run_trial(arm, rep, landscape, trial_dir,
                                    ensemble_config=ens_cfg, budget=args.budget)
                results.append(metrics)
            except Exception as exc:
                # One arm dying must not take the paired experiment with it; the
                # rest still runs and the trial can be resumed later.
                print(f"[compare] rep{rep} {arm} FAILED: {exc}")
                traceback.print_exc()
            _write_summary(run_dir, results)

    _write_summary(run_dir, results)
    print(f"\n[compare] done — {len(results)}/{total} trials in {run_dir}")


if __name__ == "__main__":
    main()
