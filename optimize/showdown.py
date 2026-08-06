"""
optimize/showdown.py
====================
Head-to-head comparison of a handful of hyperparameter configurations on ONE
shared set of ensemble landscapes.

Why a separate script
---------------------
``pareto.py`` ranks trials that were each scored on their *own* randomized
landscapes, so a trial can top the front partly because it drew easy landscapes.
A showdown removes that confound: every selected configuration is re-run on the
*same* landscapes, so the differences that remain are the configurations'.

What it does
------------
1. **Collect** every trial exactly as ``pareto.py`` does — including its
   single-run-dir shared-history rule, where naming one run dir pools all of its
   config-signature-matching siblings. So ``--runs optimize/runs/mobo_ensemble_6d_job19202380``
   aggregates the whole 6d pool, not just that one run.
2. **Select** four configurations off the Pareto front: the 2 best by
   ``dist_to_needles`` and the 2 best by ``dup_fraction``. A trial that wins on
   both axes is only taken once and the next-best on the contested axis is taken
   instead, so four DISTINCT configurations always come out (see ``select_configs``).
3. **Pick** ``--n-landscapes`` random ensemble landscapes (Sobol indices from a
   recorded ``--landscape-seed``, so the selection is reproducible).
4. **Emit** a plan: one hyperparameter JSON per configuration plus a SLURM array
   script that evaluates all of them on those same landscapes. With 4 configs and
   5 landscapes that is 20 runs — the 20 rows of the summary table.

Every task in the array is one (config, landscape) pair, so the array's ``%N``
concurrency limit is exactly the number of SLURM jobs in flight.

Usage
-----
  conda activate zombi-hop

  # Plan a 6d showdown off the whole 6d pool, then submit it.
  python optimize/showdown.py --runs optimize/runs/mobo_ensemble_6d_job19202380 \\
      --dim 6 --time-limit 0.5 --out optimize/runs/showdown_6d
  sbatch optimize/runs/showdown_6d/showdown.sbatch

  # Same, but from explicit hyperparameter JSONs (skips Pareto selection) —
  # this is how the 10d good-region test reuses the harness.
  python optimize/showdown.py --configs a.json,b.json,c.json,d.json \\
      --dim 10 --time-limit 0.7 --out optimize/runs/showdown_10d

  # Summarise a finished showdown (also runs automatically via --summary-only).
  python optimize/showdown.py --summary-only --out optimize/runs/showdown_6d
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from pareto import (  # noqa: E402
    DIST_KEY,
    DUP_KEY,
    collect_trials,
    pareto_mask_min,
    _load_run_signature,
    _dedup_realpath,
)
from summary_table import write_landscape_summary  # noqa: E402


DEFAULT_N_LANDSCAPES = 5
# Sobol landscape indices are drawn from [0, LANDSCAPE_INDEX_MAX). The sequence is
# low-discrepancy over the whole ensemble configuration space, so any window works;
# this one is wide enough that a 5-draw collision is negligible and small enough to
# stay well clear of the int32 limits downstream.
LANDSCAPE_INDEX_MAX = 100_000


# ─── Trial collection (mirrors pareto.py's single-run-dir shared-history rule) ────

def collect_pool(runs_arg: str, *, shared_history: bool = True) -> list[dict]:
    """Every comparable trial for *runs_arg*, using pareto.py's own collection rules.

    A runs *parent* directory is crawled wholesale. A single run directory pools its
    config-signature-matching siblings (same dataset / dim / time budget / direction /
    variant / ensemble settings) — the same pooling ``python optimize/pareto.py <run_dir>``
    performs, so a showdown selects from exactly the front the Pareto plot shows.
    """
    runs_dir = os.path.abspath(runs_arg)
    single_run = (
        not glob.glob(os.path.join(runs_dir, "mobo_*", "mobo_progress.json"))
        and os.path.isfile(os.path.join(runs_dir, "mobo_progress.json"))
    )
    only_signature = None
    if single_run and shared_history:
        only_signature = _load_run_signature(runs_dir)
        if only_signature is None:
            print(f"  [pool] {os.path.basename(runs_dir)} has no run_config.json; "
                  "using this run's trials only.")
        else:
            print(f"  [pool] pooling siblings matching {only_signature}")
            runs_dir = os.path.dirname(runs_dir)

    records: list[dict] = []
    for d in _dedup_realpath([runs_dir]):
        records += collect_trials(d, exclude_old=True, only_signature=only_signature)
    return records


# ─── Configuration selection ─────────────────────────────────────────────────────

def select_configs(records: list[dict], *, n_per_objective: int = 2) -> list[dict]:
    """The n best by ``dist_to_needles`` and n best by ``dup_fraction``, off the Pareto front.

    Both objectives are minimised, so "best" is smallest. The two rankings can name
    the same trial — a genuinely dominant configuration tops both — and the spec is
    to keep walking down the contested ranking until *n* distinct trials have been
    taken for it. dist is resolved first, so a trial that wins both axes is credited
    to dist and dup falls through to its next-best.

    Each returned record carries a ``selected_for`` tag ('dist' / 'dup') and its
    ``rank`` on that objective, which the summary table reports.
    """
    if not records:
        raise SystemExit("no trials collected — nothing to select from")
    M = np.array([[r["metrics"][DIST_KEY], r["metrics"][DUP_KEY], r["time_value"]]
                  for r in records], dtype=float)
    mask = pareto_mask_min(M)
    front = [records[i] for i in np.where(mask)[0]]
    print(f"  [select] {len(records)} trial(s) -> {len(front)} Pareto-optimal")
    if len(front) < 2 * n_per_objective:
        raise SystemExit(
            f"Pareto front has {len(front)} trial(s); need at least "
            f"{2 * n_per_objective} to pick {n_per_objective} per objective")

    chosen: list[dict] = []
    taken: set[tuple] = set()

    def _key(r: dict) -> tuple:
        return (r["source_run"], r["trial"])

    for objective, metric in (("dist", DIST_KEY), ("dup", DUP_KEY)):
        ranked = sorted(front, key=lambda r: r["metrics"][metric])
        rank = 0
        for rec in ranked:
            if len(chosen) >= n_per_objective * (1 if objective == "dist" else 2):
                break
            rank += 1
            if _key(rec) in taken:
                # Already claimed by the earlier objective — skip to the next best
                # so the two objectives contribute distinct configurations.
                continue
            taken.add(_key(rec))
            out = dict(rec)
            out["selected_for"] = objective
            out["rank"] = rank
            chosen.append(out)
    for c in chosen:
        m = c["metrics"]
        print(f"  [select] {c['selected_for']:4s} #{c['rank']}  "
              f"{c['source_run']} trial {c['trial']}:  "
              f"dist={m[DIST_KEY]:.4f}  dup={m[DUP_KEY]:.4f}")
    return chosen


def pick_landscapes(n: int, seed: int) -> list[int]:
    """*n* distinct random Sobol landscape indices, reproducible from *seed*."""
    rng = np.random.default_rng(seed)
    return sorted(int(v) for v in
                  rng.choice(LANDSCAPE_INDEX_MAX, size=n, replace=False))


# ─── Plan emission ───────────────────────────────────────────────────────────────

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={out_dir}/logs/%x_%A_%a.out
#SBATCH --error={out_dir}/logs/%x_%A_%a.err
#SBATCH --time={walltime}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --array=0-{last_task}%{concurrency}

# One array task = one (config, landscape) pair, so the %{concurrency} limit above is
# literally the number of SLURM jobs this campaign ever has in flight.
#
# Every task passes the SAME --ensemble-landscape-indices list and selects its own
# entry with --num-runs 1 plus the landscape for its row, so all {n_configs} configs are
# scored on an identical landscape set (that is the whole point of a showdown).

cd {repo}

CONFIGS=({config_list})
LANDSCAPES=({landscape_list})

CFG_I=$(( SLURM_ARRAY_TASK_ID / {n_landscapes} ))
LS_I=$((  SLURM_ARRAY_TASK_ID % {n_landscapes} ))
CFG="${{CONFIGS[$CFG_I]}}"
LS="${{LANDSCAPES[$LS_I]}}"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID: config=$CFG landscape=$LS"

uv run python optimize/evaluate.py \\
    --hparams-json "{out_dir}/configs/$CFG.json" \\
    --dataset ensemble --dim {dim} \\
    --num-runs 1 \\
    --ensemble-landscape-indices "$LS" \\
    --ensemble-seed {landscape_seed} \\
    --time-limit-min {time_limit_min} \\
    --device cuda \\
    --no-video \\
    --out-dir "{out_dir}/runs/${{CFG}}__ls${{LS}}"
"""


def write_plan(chosen: list[dict], landscapes: list[int], args) -> str:
    """Write configs/, the manifest and the SLURM array script. Returns the sbatch path."""
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, "configs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)

    names: list[str] = []
    for c in chosen:
        name = c.get("config_name") or f"{c['selected_for']}{c['rank']}"
        names.append(name)
        # Write the WHOLE record, not just the bare hyperparameters: evaluate.py's
        # --hparams-json accepts a trial.json-style blob with an "hparams" key, and
        # keeping the provenance means re-planning from these files (the --configs
        # path) does not silently strip the selection metadata off them.
        with open(os.path.join(out_dir, "configs", f"{name}.json"), "w") as f:
            json.dump({k: v for k, v in c.items() if k != "config_name"},
                      f, indent=2, default=str)

    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dim": args.dim,
        "time_limit_hours": args.time_limit,
        "landscape_seed": args.landscape_seed,
        "landscapes": landscapes,
        "n_rows": len(names) * len(landscapes),
        "configs": [
            {
                "name": n,
                "selected_for": c.get("selected_for"),
                "rank": c.get("rank"),
                "source_run": c.get("source_run"),
                "trial": c.get("trial"),
                "source_metrics": c.get("metrics"),
                "hparams": c["hparams"],
            }
            for n, c in zip(names, chosen)
        ],
    }
    with open(os.path.join(out_dir, "showdown_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    time_limit_min = args.time_limit * 60.0
    # Wall-time per array task: the per-run budget plus a margin for landscape
    # construction, metrics and the CoNet renders. The renders are the slow tail —
    # UMAP on an N×N co-occurrence matrix, which for a ~15k-sample run is minutes to
    # tens of minutes (run_mobo caps each at a 1800 s timeout) — so the margin is
    # sized for them, not for the ZoMBI budget.
    walltime_h = max(1, int(np.ceil(args.time_limit + args.walltime_margin)))
    sbatch = SBATCH_TEMPLATE.format(
        job_name=args.job_name or f"showdown_{args.dim}d",
        out_dir=out_dir,
        repo=_REPO,
        walltime=f"{walltime_h}:00:00",
        last_task=len(names) * len(landscapes) - 1,
        concurrency=args.concurrency,
        n_configs=len(names),
        n_landscapes=len(landscapes),
        config_list=" ".join(names),
        landscape_list=" ".join(str(v) for v in landscapes),
        dim=args.dim,
        time_limit_min=f"{time_limit_min:g}",
        landscape_seed=args.landscape_seed,
    )
    sbatch_path = os.path.join(out_dir, "showdown.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch)
    os.chmod(sbatch_path, 0o755)

    print(f"\n  plan -> {out_dir}")
    print(f"    {len(names)} config(s) x {len(landscapes)} landscape(s) = "
          f"{len(names) * len(landscapes)} run(s)")
    print(f"    landscapes: {landscapes}")
    print(f"    submit with:  sbatch {sbatch_path}")
    return sbatch_path


# ─── Explicit-config mode ────────────────────────────────────────────────────────

def load_explicit_configs(paths: list[str]) -> list[dict]:
    """Trial-like records from hyperparameter JSON files (skips Pareto selection).

    Each file is either a flat hyperparameter dict or a trial.json-style blob with a
    ``hparams`` key. The file stem names the configuration in the outputs.
    """
    out: list[dict] = []
    for p in paths:
        p = p.strip()
        if not p:
            continue
        with open(p) as f:
            blob = json.load(f)
        hp = blob.get("hparams", blob)
        name = os.path.splitext(os.path.basename(p))[0]
        out.append({
            "config_name": name,
            "selected_for": blob.get("selected_for", "explicit"),
            "rank": blob.get("rank"),
            "source_run": blob.get("source_run", os.path.dirname(p)),
            "trial": blob.get("trial"),
            "metrics": blob.get("metrics", {}),
            "hparams": hp,
        })
    if not out:
        raise SystemExit("--configs matched no readable files")
    return out


# ─── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None,
                    help="runs dir or single run dir to aggregate (pareto.py rules)")
    ap.add_argument("--configs", default=None,
                    help="comma-separated hyperparameter JSONs to use instead of "
                         "selecting off a Pareto front")
    ap.add_argument("--out", required=True, help="output directory for the plan/results")
    ap.add_argument("--dim", type=int, default=6, help="ensemble simplex dimension")
    ap.add_argument("--time-limit", type=float, default=0.5,
                    help="per-run wall-clock budget in HOURS (6d: 0.5, 10d: 0.7)")
    ap.add_argument("--n-landscapes", type=int, default=DEFAULT_N_LANDSCAPES,
                    help="how many shared landscapes every config is run on")
    ap.add_argument("--n-per-objective", type=int, default=2,
                    help="configs taken from each of dist_to_needles / dup_fraction")
    ap.add_argument("--landscape-seed", type=int, default=0,
                    help="seed for the random landscape choice AND the ensemble "
                         "construction, so the whole showdown is reproducible")
    ap.add_argument("--walltime-margin", type=float, default=2.5,
                    help="hours added to the per-run budget for each array task's "
                         "SLURM wall-time, covering the CoNet renders (default: %(default)s)")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="max simultaneous SLURM array tasks (== max jobs in flight)")
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--no-shared-history", action="store_true",
                    help="when --runs is a single run dir, do NOT pool its siblings")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip planning; just write the summary for a finished showdown")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)

    if args.summary_only:
        write_landscape_summary(out_dir)
        return

    if args.configs:
        chosen = load_explicit_configs(args.configs.split(","))
        print(f"  [select] {len(chosen)} explicit config(s): "
              + ", ".join(c["config_name"] for c in chosen))
    else:
        if not args.runs:
            ap.error("one of --runs or --configs is required")
        print("=" * 70)
        print(f"Showdown  |  runs: {args.runs}")
        print("=" * 70)
        records = collect_pool(args.runs, shared_history=not args.no_shared_history)
        chosen = select_configs(records, n_per_objective=args.n_per_objective)

    landscapes = pick_landscapes(args.n_landscapes, args.landscape_seed)
    write_plan(chosen, landscapes, args)


if __name__ == "__main__":
    main()
