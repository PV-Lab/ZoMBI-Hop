"""
optimize/parametric_sweep.py
============================
Basin-width parametric sweep on the 10-simplex Ackley benchmark.

Determines whether ZoMBI-Hop can get *any* signal in a 10-dimensional space by
sweeping from an embarrassingly trivial objective (one broad basin, no noise)
through progressively harder configurations, ending at the current production
defaults (many optima, narrow basins, background noise).

Each condition constructs its own ``Ackley("realistic", dim=10, ...)`` instance
with explicit overrides, so the config-file defaults are bypassed entirely.

``basin_width`` is the Ackley sharpness coefficient ``b`` (see ``ackley.py``):
*larger* ``b`` means a *sharper, narrower* peak, not a wider one.  The sweep
therefore goes from a small ``b`` (broad, easy-to-find basin) up to the
production ``b=20`` (sharp peak), adding optima and noise along the way.

Conditions (easiest → hardest)
------------------------------
  1. 1 optimum,  basin_width=2,   no noise   — broad single basin
  2. 1 optimum,  basin_width=5,   no noise   — narrower single basin
  3. 5 optima,   basin_width=5,   no noise   — a few well-separated peaks
  4. 5 optima,   basin_width=20,  no noise   — sharp peaks
  5. 20 optima,  basin_width=20,  no noise   — many sharp peaks, still clean
  6. 20 optima,  basin_width=20,  noise=20   — add background noise
  7. 90 optima,  basin_width=20,  noise=20   — full production config (scaled)

Every run uses the same hyperparameter set (mobo trial 112) with dimension
scaling OFF, matching the ``evaluate.py`` default transfer mode.

Output
------
``optimize/runs/parametric_DD_MM_HH_MM/`` (or ``--out`` to resume):

  sweep_config.json                  static config (trial, conditions, budgets)
  results.json                       master rollup, rewritten after every run
  <condition>/run_<k>/               full evaluate.run_single_eval artifact set

A ``results.json`` is rewritten after every single run, and any run whose
``metrics.json`` already exists is skipped on resume.

Usage
-----
  conda activate zombi-hop
  python optimize/parametric_sweep.py
  python optimize/parametric_sweep.py --num-runs 3 --quick
  python optimize/parametric_sweep.py --out optimize/runs/parametric_17_06_12_00
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev
import run_mobo as rm

from src.utils.scaling import dimension_scaling_disabled
from synthetic_data.ackley import Ackley

# ── Fixed experiment configuration ─────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SOURCE_RUN     = os.path.join(SCRIPT_DIR, "runs", "mobo_05_06_15_32")
TRIAL          = 112
DIM            = 10
TIME_LIMIT_MIN = 39.0   # 0.65 hours
DATASET_LABEL  = "ackley10d"

# (name, n_optima, basin_width, noise_amp)
# basin_width is the Ackley sharpness b: larger => sharper/narrower peak.
# Ordered easiest -> hardest, i.e. broad (small b) -> sharp (production b=20).
CONDITIONS = [
    ("cond1_1peak_bw2_nonoise",     1,    2,  0.0),
    ("cond2_1peak_bw5_nonoise",     1,    5,  0.0),
    ("cond3_5peak_bw5_nonoise",     5,    5,  0.0),
    ("cond4_5peak_bw20_nonoise",    5,   20,  0.0),
    ("cond5_20peak_bw20_nonoise",  20,   20,  0.0),
    ("cond6_20peak_bw20_noise20",  20,   20, 20.0),
    ("cond7_90peak_bw20_noise20",  90,   20, 20.0),
]


def _build_dataset(n_optima: int, basin_width: float, noise_amp: float) -> dict:
    """Build a 10-simplex Ackley dataset with explicit overrides."""
    fn = Ackley("realistic", dim=DIM,
                n_optima=n_optima, basin_width=basin_width,
                noise_amp=noise_amp)
    true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
    print(f"  [dataset] ackley10d: n_optima={n_optima}, basin_width={basin_width}, "
          f"noise_amp={noise_amp} — {len(true_optima)} peak(s)")
    return dict(dim=DIM, maximize=True, fn=fn, true_optima=true_optima,
                grid_pts=None, grid_vals=None, ackley_fn=None,
                label=DATASET_LABEL, csv_path=None)


def _write_results(path: str, header: dict, per_cond: dict) -> None:
    """Rewrite the master results.json (called after every run)."""
    out = dict(header)
    out["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    conditions = []
    for name, n_optima, basin_width, noise_amp in CONDITIONS:
        runs = per_cond.get(name, [])
        entry = {"name": name, "n_optima": n_optima, "basin_width": basin_width,
                 "noise_amp": noise_amp, "time_limit_min": TIME_LIMIT_MIN,
                 "n_runs": len(runs), "runs": runs}
        if runs:
            for key in ("dist_to_needles", "dup_fraction", "runtime_s"):
                entry[key] = ev._agg([r[key] for r in runs])
        conditions.append(entry)
    out["conditions"] = conditions
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
                    help="Existing parametric_* dir to resume into "
                         "(default: a fresh timestamped folder under optimize/runs).")
    ap.add_argument("--quick", action="store_true",
                    help="Override every budget to 1 min/run for a fast smoke test.")
    args = ap.parse_args()

    if args.num_runs < 1:
        sys.exit("--num-runs must be >= 1.")
    if not os.path.isdir(SOURCE_RUN):
        sys.exit(f"Source mobo run not found: {SOURCE_RUN}")

    hparams = ev.load_trial_hparams(SOURCE_RUN, [TRIAL])[TRIAL]

    if args.out:
        candidate = os.path.abspath(args.out)
        if not os.path.isdir(candidate):
            under_runs = os.path.join(SCRIPT_DIR, "runs", args.out)
            if os.path.isdir(under_runs):
                candidate = under_runs
        out_dir = candidate
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = rm.unique_run_dir(os.path.join(SCRIPT_DIR, "runs"), "parametric")
    results_path = os.path.join(out_dir, "results.json")

    limit = 1.0 if args.quick else TIME_LIMIT_MIN

    header = {
        "generated":      datetime.datetime.now().isoformat(timespec="seconds"),
        "trial":          TRIAL,
        "source_run":     SOURCE_RUN,
        "dim":            DIM,
        "time_limit_min": limit,
        "num_runs":       args.num_runs,
        "quick":          args.quick,
        "hparams":        hparams,
    }
    with open(os.path.join(out_dir, "sweep_config.json"), "w") as f:
        json.dump({**header, "conditions": [
            {"name": n, "n_optima": o, "basin_width": b, "noise_amp": a}
            for n, o, b, a in CONDITIONS]}, f, indent=2)

    total = len(CONDITIONS) * args.num_runs
    budget_min = total * limit
    print("=" * 72)
    print(f"ZoMBI-Hop parametric sweep  |  trial {TRIAL}  |  dim {DIM}  |  "
          f"{args.num_runs} run(s)/cond")
    print(f"total runs: {total}   approx wall-clock budget: "
          f"{budget_min:.0f} min (~{budget_min / 60:.1f} h)")
    print(f"output: {out_dir}")
    print("=" * 72)

    per_cond: dict = {}
    done = 0
    for name, n_optima, basin_width, noise_amp in CONDITIONS:
        ds = _build_dataset(n_optima, basin_width, noise_amp)
        per_cond[name] = []
        cond_dir = os.path.join(out_dir, name)
        os.makedirs(cond_dir, exist_ok=True)

        print(f"\n{'#' * 72}\n# {name}  (n_optima={n_optima}, bw={basin_width}, "
              f"noise={noise_amp}, {limit:.0f} min/run)\n{'#' * 72}")

        for k in range(1, args.num_runs + 1):
            done += 1
            run_dir = os.path.join(cond_dir, f"run_{k}")
            tag = f"[{name} run {k}/{args.num_runs}]  (overall {done}/{total})"

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
                with dimension_scaling_disabled():
                    res = ev.run_single_eval(hparams, ds, DATASET_LABEL,
                                             run_dir, limit)
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

            _write_results(results_path, header, per_cond)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. {done} run(s). Master results: {results_path}")


if __name__ == "__main__":
    main()
