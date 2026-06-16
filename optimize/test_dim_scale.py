"""
optimize/test_dim_scale.py
==========================
Controlled experiment: **does the dimensionality scaling actually help?**

Runs one fixed hyperparameter set (mobo ``trial_112``) across five conditions,
three runs each, on the ``realistic`` Ackley benchmark.  Each individual run is
executed by ``evaluate.run_single_eval`` so the per-run artifacts and logging
frequency are *identical* to ``evaluate.py`` (metrics.json, points.csv,
needles.csv, metrics_over_time.csv, static plots, and the dimension-appropriate
landscape view).  Nothing here re-implements the run loop — it only orchestrates.

The "WITHOUT scaling" conditions wrap the whole run in
``src.utils.scaling.dimension_scaling_disabled()``, so the *same* code paths run
with both scaling factors collapsed to 1.0 — an honest A/B with no separate
code branch.

Conditions
----------
  1. ackley3d  realistic, scaling on*  20 min/run   (baseline; *no-op at d=3)
  2. ackley4d  realistic, scaling ON   30 min/run
  3. ackley4d  realistic, scaling OFF  30 min/run
  4. ackley10d realistic, scaling ON  120 min/run
  5. ackley10d realistic, scaling OFF 120 min/run

Output
------
``optimize/runs/dimscale_test_DD_MM_HH_MM/`` (or ``--out`` to resume one):

  test_config.json                     static config (trial, conditions, budgets)
  results.json                         master rollup, REWRITTEN AFTER EVERY RUN
  <condition>/run_<k>/                  full evaluate.run_single_eval artifact set

A master ``results.json`` is rewritten after every single run, and any run whose
``metrics.json`` already exists is skipped — so the ~16 h sweep can be killed and
re-launched (point ``--out`` at the existing folder) without losing work.

Usage
-----
  conda activate zombi-hop
  python optimize/test_dim_scale.py
  python optimize/test_dim_scale.py --num-runs 3 --quick          # short smoke test
  python optimize/test_dim_scale.py --out optimize/runs/dimscale_test_11_06_14_00   # resume
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

# Reuse the entire evaluation machinery (importing it also wires sys.path to the
# repo root, so ``src`` becomes importable for the scaling toggle below).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev
import run_mobo as rm

import torch

from src.utils.scaling import dimension_scaling_disabled

# ── Fixed experiment configuration ──────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SOURCE_RUN     = os.path.join(SCRIPT_DIR, "runs", "mobo_05_06_15_32")
TRIAL          = 112
ACKLEY_VARIANT = "realistic"

# (name, dataset, scaling_on, minutes_per_run).  Scaling is a no-op at d=3, so the
# baseline's flag is irrelevant; it is kept True so condition 1 is "native" 3-D.
CONDITIONS = [
    ("cond1_ackley3d_baseline",  "ackley3d",  True,   20),
    ("cond2_ackley4d_scaled",    "ackley4d",  True,   30),
    ("cond3_ackley4d_unscaled",  "ackley4d",  False,  30),
    ("cond4_ackley10d_scaled",   "ackley10d", True,  120),
    ("cond5_ackley10d_unscaled", "ackley10d", False, 120),
]


def _write_results(path: str, header: dict, per_cond: dict) -> None:
    """Rewrite the master results.json (cheap; called after every run)."""
    out = dict(header)
    out["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    conditions = []
    for name, dataset, scaling, minutes in CONDITIONS:
        runs = per_cond.get(name, [])
        entry = {"name": name, "dataset": dataset, "scaling": scaling,
                 "time_limit_min": minutes, "n_runs": len(runs), "runs": runs}
        if runs:
            for key in ("dist_to_needles", "dup_fraction", "runtime_s"):
                entry[key] = ev._agg([r[key] for r in runs])
        conditions.append(entry)
    out["conditions"] = conditions
    # Atomic-ish write so a crash mid-dump can't corrupt the rollup.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-runs", type=int, default=2,
                    help="Repeats per condition (default: 2).")
    ap.add_argument("--out", default=None,
                    help="Existing dimscale_test_* dir to resume into "
                         "(default: a fresh timestamped folder under optimize/runs).")
    ap.add_argument("--quick", action="store_true",
                    help="Override every budget to 1 min/run for a fast smoke test.")
    args = ap.parse_args()

    if args.num_runs < 1:
        sys.exit("--num-runs must be >= 1.")
    if not os.path.isdir(SOURCE_RUN):
        sys.exit(f"Source mobo run not found: {SOURCE_RUN}")

    # Hyperparameters: mobo trial 112 (reuses evaluate's validated loader).
    hparams = ev.load_trial_hparams(SOURCE_RUN, [TRIAL])[TRIAL]

    if args.out:
        candidate = os.path.abspath(args.out)
        if not os.path.isdir(candidate):
            # Allow passing just the directory name, e.g. --out dimscale_test_11_06_18_10
            under_runs = os.path.join(SCRIPT_DIR, "runs", args.out)
            if os.path.isdir(under_runs):
                candidate = under_runs
        out_dir = candidate
    else:
        out_dir = os.path.join(SCRIPT_DIR, "runs",
                               datetime.datetime.now().strftime("dimscale_test_%d_%m_%H_%M"))
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.json")

    header = {
        "generated":      datetime.datetime.now().isoformat(timespec="seconds"),
        "trial":          TRIAL,
        "source_run":     SOURCE_RUN,
        "ackley_variant": ACKLEY_VARIANT,
        "num_runs":       args.num_runs,
        "quick":          args.quick,
        "hparams":        hparams,
    }
    with open(os.path.join(out_dir, "test_config.json"), "w") as f:
        json.dump({**header, "conditions": [
            {"name": n, "dataset": d, "scaling": s, "time_limit_min": m}
            for n, d, s, m in CONDITIONS]}, f, indent=2)

    total = len(CONDITIONS) * args.num_runs
    budget_min = sum((1 if args.quick else m) * args.num_runs for _, _, _, m in CONDITIONS)
    print("=" * 72)
    print(f"ZoMBI-Hop dimension-scaling A/B  |  trial {TRIAL}  |  {args.num_runs} run(s)/cond")
    print(f"total runs: {total}   approx wall-clock budget: "
          f"{budget_min} min (~{budget_min / 60:.1f} h)")
    print(f"output: {out_dir}")
    print("=" * 72)

    per_cond: dict = {}
    done = 0
    for name, dataset, scaling, minutes in CONDITIONS:
        # Resolve the objective + reference optima once per condition (cheap).
        ds = ev.resolve_dataset(dataset, SOURCE_RUN, ACKLEY_VARIANT)
        limit = 1 if args.quick else minutes
        per_cond[name] = []
        cond_dir = os.path.join(out_dir, name)
        os.makedirs(cond_dir, exist_ok=True)

        print(f"\n{'#' * 72}\n# {name}  ({dataset}, scaling={'ON' if scaling else 'OFF'}, "
              f"{limit} min/run)\n{'#' * 72}")

        for k in range(1, args.num_runs + 1):
            done += 1
            run_dir = os.path.join(cond_dir, f"run_{k}")
            tag = f"[{name} run {k}/{args.num_runs}]  (overall {done}/{total})"

            # Resume: a finished run already wrote metrics.json — reuse it.
            metrics_path = os.path.join(run_dir, "metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path) as f:
                        m = json.load(f)
                    per_cond[name].append({"run": k,
                                           "dist_to_needles": m["dist_to_needles"],
                                           "dup_fraction":    m["dup_fraction"],
                                           "runtime_s":       m["runtime_s"]})
                    print(f"  {tag} — already complete, skipping.")
                    _write_results(results_path, header, per_cond)
                    continue
                except Exception as exc:
                    print(f"  {tag} — stale metrics.json ({exc}); re-running.")

            print(f"  {tag} — running …", flush=True)
            t0 = time.time()
            try:
                if scaling:
                    res = ev.run_single_eval(hparams, ds, dataset, run_dir, limit)
                else:
                    with dimension_scaling_disabled():
                        res = ev.run_single_eval(hparams, ds, dataset, run_dir, limit)
                per_cond[name].append({"run": k,
                                       "dist_to_needles": round(res["dist"], 6),
                                       "dup_fraction":    round(res["dup"], 6),
                                       "runtime_s":       round(res["runtime"], 3)})
            except KeyboardInterrupt:
                print("\n[!] Interrupted — writing results so far and exiting.")
                _write_results(results_path, header, per_cond)
                raise
            except Exception as exc:
                print(f"  {tag} FAILED after {time.time() - t0:.1f}s: {exc}")

            # Persist after EVERY run so a crash never loses completed work.
            _write_results(results_path, header, per_cond)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. {done} run(s). Master results: {results_path}")


if __name__ == "__main__":
    main()
