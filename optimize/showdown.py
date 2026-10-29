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
4. **Emit** a plan: one hyperparameter JSON per configuration, a task queue
   (``tasks.tsv``) and a SLURM script. With 4 configs and 5 landscapes that is 20
   runs — the 20 rows of the summary table.

Budget
------
Every run gets the same SAMPLING budget — ``--budget-points`` measured points
(default 3000), converted to evaluate.py's ``--max-lines``. A showdown compares what
configurations do with an equal number of experiments; a wall-clock budget instead
compares them at whatever point count each happened to reach, which drifts with node
speed and with how expensive a config's own acquisition step is. ``--time-limit`` is
still passed, but only as a safety cap, so runs normally end on the point budget.
Pass ``--budget-points 0`` for the old wall-clock-only behaviour.

Repeats
-------
``--n-repeats N`` runs every (config, landscape) cell N times. One run of a cell is
a single draw from a stochastic optimizer, so a difference between two configs on
one landscape can easily be noise; repeats turn each cell into a distribution. The
landscape is held fixed across repeats (it depends only on the Sobol index and
``--ensemble-seed``), so the spread is the optimizer's, not the landscape's.

Only repeat 1 renders plots — the rest run with ``evaluate.py --no-plots`` and write
CSVs plus metrics.json. The summary Markdown shows one plot column per cell (from
repeat 1) and the MEAN of each metric; median/variance/p5/p95 and every raw value go
to CSVs beside it, since a table with five statistics per cell is unreadable.

Execution model
---------------
The SLURM array is a pool of PERSISTENT WORKERS, not one task per run. Each worker
claims tasks off ``tasks.tsv`` (atomic ``mkdir``) until the queue drains or its
wall-time runs out. A 200-run campaign therefore queues ``--n-workers`` times rather
than 200 times: finishing a run hands the GPU to the next run inside the same
allocation instead of returning it to the scheduler. ``done/`` markers make the
campaign resumable — re-submitting skips completed work.

Each worker scans a ROTATED view of the queue, starting ``k/n_workers`` of the way
down and wrapping, so the pool spreads across configurations instead of grinding
through them one at a time — a campaign cut short by wall-time then has partial
repeats for every config rather than full repeats for the first few and none for the
rest. Workers rescan until a full pass claims nothing, so none exits while unclaimed
work remains. Throughput is capped by ``--n-workers`` (concurrent runs), NOT by the
task count: 200 tasks across 5 workers is ~40 sequential tasks per worker.

Usage
-----
  conda activate zombi-hop

  # Plan a 6d showdown off the whole 6d pool, then submit it.
  python optimize/showdown.py --runs optimize/runs/mobo_ensemble_6d_job19202380 \\
      --dim 6 --time-limit 0.5 --out optimize/runs/showdown_6d
  sbatch optimize/runs/showdown_6d/showdown.sbatch

  # 10 repeats per cell across 5 persistent workers.
  python optimize/showdown.py --configs a.json,b.json,c.json,d.json \\
      --dim 6 --time-limit 0.5 --n-repeats 10 --n-workers 5 \\
      --out optimize/runs/showdown_6d_reps

  # Progress, and resuming after workers were killed.
  python optimize/showdown.py --status      --out optimize/runs/showdown_6d_reps
  python optimize/showdown.py --reset-stale --out optimize/runs/showdown_6d_reps

  # Same, but from explicit hyperparameter JSONs (skips Pareto selection) —
  # this is how the 10d good-region test reuses the harness.
  python optimize/showdown.py --configs a.json,b.json,c.json,d.json \\
      --dim 10 --time-limit 0.7 --out optimize/runs/showdown_10d

  # Summarise a finished showdown (also runs automatically via --summary-only).
  python optimize/showdown.py --summary-only --out optimize/runs/showdown_6d

  # Same, one results row per repeat instead of the per-cell mean. The --no-plots
  # repeats need their CoNet pair rendered first (a UMAP fit, so via SLURM).
  python optimize/showdown.py --backfill-conet --out optimize/runs/showdown_6d_reps
  sbatch optimize/runs/showdown_6d_reps/conet_backfill.sbatch
  python optimize/showdown.py --summary-only --unfold-repeats --out optimize/runs/showdown_6d_reps
"""

from __future__ import annotations

import os
import sys
import json
import glob
import shutil
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

sys.path.insert(0, _REPO)
from src.hparams import conflicts, extract_hparams  # noqa: E402


DEFAULT_N_LANDSCAPES = 5
# Sampling budget per run, in MEASURED POINTS. Every run of a showdown gets the same
# number of experiments, so the comparison is "who does more with the same budget"
# rather than "who ran on the faster node". The wall-clock --time-limit stays on as a
# secondary safety cap only.
DEFAULT_BUDGET_POINTS = 3000
# One LineBO line = NUM_EXPERIMENTS measured points, and every run starts with a fixed
# preamble of init lines before the optimizer's first objective call (see
# n_init_lines). Mirrors run_mobo.NUM_EXPERIMENTS / N_INIT_LINES; duplicated here so
# planning stays importable without pulling in torch.
POINTS_PER_LINE = 24
N_INIT_LINES = 2
INIT_DESIGNS = ("random", "axial")
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


def n_init_lines(dim: int, init_design: str = "random") -> int:
    """Init lines measured before the optimizer's first call (run_mobo.n_init_lines).

    Axial: one trace per component rounded down to even (axial_init.n_traces_for),
    restated here so planning does not import torch through src.utils.
    """
    if init_design == "axial":
        return int(dim) - int(dim) % 2 if dim >= 2 else 0
    return N_INIT_LINES


def budget_to_max_lines(budget_points: int, n_init: int = N_INIT_LINES) -> int | None:
    """Measured-point budget -> the ``--max-lines`` cap evaluate.py takes.

    ``--max-lines`` counts objective calls, i.e. lines the OPTIMIZER asked for; the
    *n_init* init lines are a deterministic preamble that runs before the first such
    call, so they come off the budget rather than counting against the cap — an axial
    init (more lines) leaves fewer optimizer lines, and every run still measures the
    same number of points. Returns None when the budget is <= 0, meaning "no point
    cap, time budget only".
    """
    if budget_points is None or budget_points <= 0:
        return None
    n_lines = int(budget_points // POINTS_PER_LINE) - n_init
    if n_lines < 1:
        raise SystemExit(
            f"--budget-points {budget_points} leaves no optimizer lines after the "
            f"{n_init} init line(s) of {POINTS_PER_LINE} point(s) each")
    return n_lines


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
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --array=0-{last_worker}

# PERSISTENT WORKER POOL — not one SLURM task per run.
#
# The array is {n_workers} long-lived WORKERS, not {n_tasks} runs. Each worker starts once,
# then pulls task after task off the shared queue in tasks.tsv until the queue is empty
# or its wall-clock runs out. So this campaign queues {n_workers} times, not {n_tasks} times:
# a run that finishes hands its GPU straight to the next run inside the same allocation
# instead of going back to the scheduler and waiting for a fresh one.
#
# Claiming is one atomic `mkdir` per task, which is the only primitive that is reliably
# atomic on a shared filesystem (unlike flock over NFS). Exactly one worker can create
# claims/<id>, and that worker owns the task. A separate done/<id> marker is what makes
# the campaign RESUMABLE: re-submitting this script skips finished tasks. Clear the
# claims left behind by killed workers first, or they are never retried:
#
#     python optimize/showdown.py --reset-stale --out {out_dir}
#     sbatch {out_dir}/showdown.sbatch

cd {repo}

QUEUE="{out_dir}/tasks.tsv"
CLAIMS="{out_dir}/claims"
DONE="{out_dir}/done"
WORKQ="{out_dir}/logs/workq_${{SLURM_ARRAY_TASK_ID}}.tsv"
mkdir -p "$CLAIMS" "$DONE"

# ROTATED QUEUE VIEW — worker k starts k/{n_workers} of the way down tasks.tsv and wraps.
# Reading top-down in lockstep, every worker would race for the same first task and lose
# {n_workers}-1 mkdir races before finding work, and the whole pool would grind through
# one config before touching the next — so a campaign killed halfway leaves the last
# configs with zero repeats. Rotating starts each worker on a different config, so the
# pool covers all of them at once and a partial campaign is still a fair comparison.
NTASKS=$(wc -l < "$QUEUE")
OFFSET=$(( SLURM_ARRAY_TASK_ID * NTASKS / {n_workers} ))
{{ tail -n +$(( OFFSET + 1 )) "$QUEUE"; head -n "$OFFSET" "$QUEUE"; }} > "$WORKQ"

# Stop claiming new work when too little wall-time is left to finish a task; a task
# killed mid-flight leaves a claim with no result, which is pure waste.
DEADLINE=$(( $(date +%s) + {worker_seconds} - {task_seconds} ))

echo "[$(date)] worker $SLURM_ARRAY_TASK_ID up on $(hostname), gpu=${{CUDA_VISIBLE_DEVICES:-?}}, queue=$WORKQ (offset $OFFSET/$NTASKS)"

# A broken GPU makes every run die in seconds with "CUDA unknown error"; check it
# BEFORE claiming, so a bad node takes no work from the healthy workers.
if ! uv run python -c "import torch; torch.zeros(1, device='cuda')" < /dev/null; then
    echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: CUDA unusable on $(hostname) — exiting without claiming"
    exit 1
fi

# Each claim records its owner (array job + task). A preempted worker is REQUEUED under
# the same id and its earlier incarnation is dead, so any unfinished claim it owns is
# abandoned work — release it, or the requeued worker would skip its own tasks.
ME="${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
for c in "$CLAIMS"/*/; do
    [ -d "$c" ] || continue
    TID_C=$(basename "$c")
    if [ ! -d "$DONE/$TID_C" ] && [ "$(cat "$c/owner" 2>/dev/null)" = "$ME" ]; then
        echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: releasing own pre-requeue claim $TID_C"
        rm -rf "$c"
    fi
done

n_ran=0
out_of_time=0
pass_no=0

# Outer rescan: one pass over the queue can walk past a task that was claimed at the
# time but never completed (a transient mkdir failure, or a claim released by
# --reset-stale while this worker was running). A pass that claims nothing means the
# queue is genuinely drained, so this terminates — but a worker never exits while
# unclaimed work is still sitting in the file.
while [ "$out_of_time" -eq 0 ]; do
    pass_no=$(( pass_no + 1 ))
    claimed_this_pass=0

    while read -r TID CFG LS REP NOPLOTS; do
        [ -z "$TID" ] && continue
        [ -d "$DONE/$TID" ] && continue
        if [ "$(date +%s)" -gt "$DEADLINE" ]; then
            echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: out of wall-time, stopping (ran $n_ran)"
            out_of_time=1
            break
        fi
        # Atomic claim: mkdir fails for every worker but the first.
        mkdir "$CLAIMS/$TID" 2>/dev/null || continue
        echo "$ME" > "$CLAIMS/$TID/owner"
        claimed_this_pass=$(( claimed_this_pass + 1 ))

        RUN_OUT="{out_dir}/runs/${{CFG}}__ls${{LS}}__r${{REP}}"
        # A retried task must not inherit a killed attempt's partial files.
        rm -rf "$RUN_OUT"
        echo "[$(date)] worker $SLURM_ARRAY_TASK_ID -> task $TID: config=$CFG landscape=$LS repeat=$REP"

        # Repeat 1 renders the full artifact set; the rest pass --no-plots and produce only
        # CSVs + metrics.json. The repeats exist to be counted, not looked at, and the CoNet
        # UMAP render costs as much as the run itself.
        PLOTFLAG=""
        [ "$NOPLOTS" = "1" ] && PLOTFLAG="--no-plots"
{landscape_case}
        uv run python optimize/evaluate.py \\
            --hparams-json "{out_dir}/configs/$CFG.json" \\
            {dataset_flags}--num-runs 1 \\
            {extra_flags}--time-limit-min {time_limit_min} \\
            {max_lines_flag}--device cuda \\
            --no-video $PLOTFLAG \\
            --out-dir "$RUN_OUT" < /dev/null
        # `< /dev/null` matters: without it the child inherits this loop's stdin and can
        # swallow the rest of the queue, silently truncating it.

        # evaluate.py catches a failed run and still exits 0, so success is the run's
        # metrics.json existing — not the exit code alone.
        RC=$?
        if [ "$RC" -eq 0 ] && compgen -G "$RUN_OUT/trial_*/run_*/metrics.json" > /dev/null; then
            mkdir -p "$DONE/$TID"
            n_ran=$(( n_ran + 1 ))
        else
            echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: task $TID FAILED rc=$RC or no metrics.json (claim kept; --reset-stale requeues it)"
        fi
    done < "$WORKQ"

    [ "$claimed_this_pass" -eq 0 ] && break
    echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: pass $pass_no claimed $claimed_this_pass task(s); rescanning"
done

echo "[$(date)] worker $SLURM_ARRAY_TASK_ID done, ran $n_ran task(s) over $pass_no pass(es)"
"""


def write_plan(chosen: list[dict], landscapes: list[int], args) -> str:
    """Write configs/, the manifest and the SLURM array script. Returns the sbatch path."""
    # Refuse to plan a configuration ZoMBIHop cannot run, before writing anything.
    # Checked as evaluate.py will read it (legacy records upgraded first), so a
    # config that passes here is the config the workers run — nothing is repaired.
    bad = {}
    for c in chosen:
        name = c.get("config_name") or f"{c['selected_for']}{c['rank']}"
        problems = conflicts(extract_hparams(c, source=name, log=None))
        if problems:
            bad[name] = problems
    if bad:
        sys.exit("Hyperparameter conflict — no showdown planned:\n" + "\n".join(
            f"  {n}:\n" + "\n".join(f"    - {p}" for p in probs)
            for n, probs in bad.items()))

    # Reject a mistyped override key now rather than in every worker.
    if args.dataset == "mixed":
        if args.ensemble_overrides:
            sys.exit("--ensemble-override does not apply with --mix; put the overrides "
                     "in each --mix spec")
    elif args.dataset in ("rungp", "runrf") and args.ensemble_overrides:
        sys.exit("--ensemble-override only applies to --dataset ensemble / needles")
    if args.dataset in ("rungp", "runrf") and not args.gp_run:
        sys.exit(f"--dataset {args.dataset} needs --gp-run RUN_DIR")
    if args.dataset == "needles":
        keys = set(args.ensemble_overrides)
        if not {"n_optima", "basin_width"} <= keys <= {"n_optima", "basin_width",
                                                        "placement_width", "unit_range"}:
            sys.exit("--dataset needles takes --ensemble-override "
                     "'n_optima=N,basin_width=B[,placement_width=W,unit_range=true]'")
    elif args.ensemble_overrides:
        from synthetic_data.ensemble import random_ensemble_config
        valid = random_ensemble_config(args.dim, landscapes[0], seed=args.landscape_seed)
        unknown = sorted(set(args.ensemble_overrides) - set(valid))
        if unknown:
            sys.exit(f"--ensemble-override: unknown ensemble key(s) {unknown}; "
                     f"valid: {sorted(valid)}")

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

    n_repeats = max(1, int(args.n_repeats))
    n_init = n_init_lines(args.dim, args.init)
    max_lines = budget_to_max_lines(args.budget_points, n_init)
    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dim": args.dim,
        "dataset": args.dataset,
        "gp_run": os.path.abspath(args.gp_run) if args.dataset in ("rungp", "runrf") else None,
        "gp_length_scale": args.gp_length_scale if args.dataset == "rungp" else None,
        "rf_n_estimators": args.rf_n_estimators if args.dataset == "runrf" else None,
        "time_limit_hours": args.time_limit,
        "budget_points": args.budget_points if max_lines is not None else None,
        "max_lines": max_lines,
        "init_design": args.init,
        "n_init_lines": n_init,
        "ensemble_overrides": args.ensemble_overrides or None,
        "landscape_seed": args.landscape_seed,
        "landscapes": landscapes,
        # --mix: landscape k of `landscapes` is built from landscape_specs[k].
        "landscape_specs": args.mix_specs or None,
        "n_repeats": n_repeats,
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

    # The work queue, one line per (config, landscape, repeat). Written as a file rather
    # than derived from $SLURM_ARRAY_TASK_ID arithmetic because workers no longer map
    # 1:1 onto tasks — and because a queue you can read is a queue you can debug.
    tasks: list[tuple[int, str, int, int, int]] = []
    for name in names:
        for ls in landscapes:
            for rep in range(1, n_repeats + 1):
                # Repeat 1 is the one the summary table shows plots for.
                tasks.append((len(tasks), name, ls, rep, 0 if rep == 1 else 1))
    queue_path = os.path.join(out_dir, "tasks.tsv")
    with open(queue_path, "w") as f:
        for tid, name, ls, rep, noplots in tasks:
            f.write(f"{tid:05d}\t{name}\t{ls}\t{rep}\t{noplots}\n")

    time_limit_min = args.time_limit * 60.0
    # Per-TASK time: the per-run budget plus a margin for landscape construction,
    # metrics and the CoNet renders. The renders are the slow tail — UMAP on an N×N
    # co-occurrence matrix, which for a ~15k-sample run is minutes to tens of minutes
    # (run_mobo caps each at a 1800 s timeout) — so the margin is sized for them.
    # A worker uses this to decide whether it still has room for one more task.
    task_h = args.time_limit + args.walltime_margin
    walltime_h = max(1, int(args.worker_hours))
    # ``walltime_margin`` is a SAFETY RESERVE — what a worker holds back before
    # claiming one more task — NOT a per-task cost. Estimating the drain from it
    # overstates the campaign by ~2.5x and cries wolf about needing a resubmit:
    # the 200-run showdown_6d_clamped_reps10 was measured at 0.507 h/task against
    # a 0.5 h optimizer budget (5 workers x 40 tasks, drained in 20.3 h of a 24 h
    # wall-time, one submission, zero out-of-time events). Per-task overhead —
    # landscape construction, metrics, and the repeat-1 CoNet renders — is minutes,
    # not the reserve. So the expected drain is the budget plus that overhead, and
    # the reserve only sets the worst case.
    PER_TASK_OVERHEAD_H = 0.05
    n_plotted = sum(1 for t in tasks if t[4] == 0)
    exp_h = len(tasks) * (args.time_limit + PER_TASK_OVERHEAD_H) / max(1, args.n_workers)
    worst_h = len(tasks) * task_h / max(1, args.n_workers)
    # A worker stops claiming once less than one task-reserve of wall-time is left,
    # so this — not the raw wall-time — is the budget the campaign has to drain in.
    claim_h = walltime_h - task_h
    landscape_case = ""
    if args.dataset == "mixed":
        # Each landscape id picks its own evaluate.py dataset flags.
        arms = "\n".join(
            f"            {ls}) DSFLAGS=({_mix_flags(sp, args.dim, args.landscape_seed)}) ;;"
            for ls, sp in zip(landscapes, args.mix_specs))
        landscape_case = (
            "        # Mixed showdown (--mix): the landscape id selects the dataset.\n"
            "        case \"$LS\" in\n" + arms + "\n"
            "            *) echo \"unknown landscape $LS\"; continue ;;\n"
            "        esac\n")
        dataset_flags = "\"${DSFLAGS[@]}\" \\\n            "
    elif args.dataset == "rungp":
        dataset_flags = (f"--dataset rungp --gp-run '{os.path.abspath(args.gp_run)}' "
                         f"--gp-length-scale {args.gp_length_scale:g} \\\n            ")
    elif args.dataset == "runrf":
        dataset_flags = (f"--dataset runrf --gp-run '{os.path.abspath(args.gp_run)}' "
                         f"--rf-n-estimators {args.rf_n_estimators} \\\n            ")
    else:
        dataset_flags = (f"--dataset {args.dataset} --dim {args.dim} \\\n            "
                         f"--ensemble-landscape-indices \"$LS\" \\\n            "
                         f"--ensemble-seed {args.landscape_seed} \\\n            ")
    extra_flags = ""
    if args.init != "random":
        extra_flags += f"--init {args.init} \\\n            "
    if args.ensemble_overrides:
        spec = ",".join(f"{k}={json.dumps(v)}" for k, v in args.ensemble_overrides.items())
        extra_flags += f"--ensemble-override '{spec}' \\\n            "
    sbatch = SBATCH_TEMPLATE.format(
        job_name=args.job_name or f"showdown_{args.dim}d",
        out_dir=out_dir,
        repo=_REPO,
        walltime=f"{walltime_h}:00:00",
        worker_seconds=int(walltime_h * 3600),
        task_seconds=int(task_h * 3600),
        last_worker=args.n_workers - 1,
        n_workers=args.n_workers,
        n_tasks=len(tasks),
        mem=args.mem,
        partition=args.partition,
        dataset_flags=dataset_flags,
        landscape_case=landscape_case,
        time_limit_min=f"{time_limit_min:g}",
        max_lines_flag=(f"--max-lines {max_lines} \\\n            "
                        if max_lines is not None else ""),
        extra_flags=extra_flags,
    )
    sbatch_path = os.path.join(out_dir, "showdown.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch)
    os.chmod(sbatch_path, 0o755)

    print(f"\n  plan -> {out_dir}")
    print(f"    {len(names)} config(s) x {len(landscapes)} landscape(s) x "
          f"{n_repeats} repeat(s) = {len(tasks)} run(s)")
    print(f"    landscapes: {landscapes}")
    for ls, sp in zip(landscapes, args.mix_specs or []):
        print(f"      {ls}: {mix_label(sp)}")
    if args.ensemble_overrides:
        print(f"    landscape overrides: {args.ensemble_overrides}")
    if max_lines is not None:
        print(f"    budget: {args.budget_points} point(s)/run = {n_init} {args.init} init + "
              f"{max_lines} optimizer line(s) @ {POINTS_PER_LINE} point(s); "
              f"--time-limit {args.time_limit:g} h is only a safety cap, so runs "
              "usually finish well inside the estimates below")
    else:
        print(f"    budget: wall-clock only ({args.time_limit:g} h/run, no point cap)")
    print(f"    queue -> {queue_path}")
    print(f"    {args.n_workers} persistent worker(s) @ {walltime_h} h; "
          f"~{exp_h:.1f} h to drain (worst case {worst_h:.1f} h) "
          f"({n_plotted} plotted, {len(tasks) - n_plotted} --no-plots)")
    print(f"    workers stop claiming with {task_h:.2g} h left, so the drain budget "
          f"is {claim_h:.1f} h")
    if exp_h > claim_h:
        over = exp_h / max(1e-9, claim_h)
        print(f"    NOTE: the expected drain exceeds that budget by {over:.1f}x — this "
              "campaign will NOT finish in one submission.")
        print(f"          Either raise --n-workers to >= "
              f"{int(-(-len(tasks) * (args.time_limit + PER_TASK_OVERHEAD_H) // claim_h))}, "
              "or re-submit (after --reset-stale) to finish the remainder.")
    print(f"    submit with:  sbatch {sbatch_path}")
    return sbatch_path


CONET_BACKFILL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=conet_backfill
#SBATCH --output={out_dir}/logs/%x_%j.out
#SBATCH --error={out_dir}/logs/%x_%j.err
#SBATCH --time={walltime}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8,mit_preemptable,mit_normal

# Render the CoNet pair for every finished showdown run that has none — the repeats
# that ran with --no-plots. paired_conet is CPU-only (UMAP), so no GPU is requested;
# the job starts on whichever partition frees first. sched_mit_sloan_batch is left out
# on purpose: its nodes are EL7, where the project .venv cannot load.
# Already-rendered runs are skipped, so re-submitting only redoes what is missing.

cd {repo}
for d in {run_dirs}; do
    [ -f "$d/conet.png" ] && [ -f "$d/conet_uniform.png" ] && continue
    echo "[$(date)] conet -> $d"
    MPLBACKEND=Agg timeout 1800 uv run python visualization/paired_conet.py \\
        --run "$d" --save "$d/conet.png" --save-uniform "$d/conet_uniform.png" < /dev/null \\
        || echo "[$(date)] FAILED rc=$? $d"
done
echo "[$(date)] done"
"""


def write_conet_backfill(out_dir: str) -> str | None:
    """Write ``conet_backfill.sbatch`` for the finished runs that lack a CoNet pair.

    The CoNet render is a UMAP fit, so it goes through SLURM rather than running on the
    login node alongside --summary-only. Returns the sbatch path, or None when every
    finished run already has both plots.
    """
    todo = sorted(
        os.path.dirname(m) for m in glob.glob(os.path.join(
            out_dir, "runs", "*", "trial_*", "run_*", "metrics.json"))
        if not all(os.path.isfile(os.path.join(os.path.dirname(m), f))
                   for f in ("conet.png", "conet_uniform.png")))
    if not todo:
        print("  every finished run already has its CoNet pair — nothing to backfill")
        return None
    # Each render is capped at 30 min, as in run_mobo; 12 h fits every partition.
    hours = min(12, 1 + len(todo) // 2)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    path = os.path.join(out_dir, "conet_backfill.sbatch")
    with open(path, "w") as f:
        f.write(CONET_BACKFILL_TEMPLATE.format(
            out_dir=out_dir, repo=_REPO, walltime=f"{hours}:00:00",
            run_dirs=" ".join(f'"{d}"' for d in todo)))
    os.chmod(path, 0o755)
    print(f"  {len(todo)} run(s) need a CoNet pair -> {path}")
    print(f"  submit with:  sbatch {path}")
    print("  then re-run --summary-only --unfold-repeats to pick the plots up")
    return path


def print_status(out_dir: str) -> None:
    """Queue progress: how many tasks are done, in flight, and still waiting."""
    queue = os.path.join(out_dir, "tasks.tsv")
    if not os.path.isfile(queue):
        raise SystemExit(f"no tasks.tsv in {out_dir}")
    with open(queue) as f:
        tids = [ln.split("\t")[0] for ln in f if ln.strip()]
    done = os.path.join(out_dir, "done")
    claims = os.path.join(out_dir, "claims")
    n_done = sum(1 for t in tids if os.path.isdir(os.path.join(done, t)))
    n_claim = sum(1 for t in tids
                  if os.path.isdir(os.path.join(claims, t))
                  and not os.path.isdir(os.path.join(done, t)))
    print(f"  {os.path.basename(out_dir)}: {len(tids)} task(s) — "
          f"{n_done} done, {n_claim} running, {len(tids) - n_done - n_claim} pending")


def reset_stale(out_dir: str) -> int:
    """Drop claims that never produced a done marker, so those tasks are retried.

    A worker that is killed (wall-time, node failure, OOM) leaves its claim behind with
    no result. Nothing else ever clears it, so without this the task is invisible to
    every later submission — the campaign would look "drained" while silently missing
    runs. Safe to run only when no workers are live; a claim on a RUNNING task looks
    exactly like an abandoned one.

    Also drops a done marker whose run has no ``metrics.json``: workers before
    2026-10-18 marked a task done on evaluate.py's exit code alone, which is 0 even
    when the run inside it failed.
    """
    claims = os.path.join(out_dir, "claims")
    done = os.path.join(out_dir, "done")
    if not os.path.isdir(claims):
        print(f"  no claims dir in {out_dir} — nothing to reset")
        return 0
    queue = os.path.join(out_dir, "tasks.tsv")
    if os.path.isfile(queue):
        with open(queue) as f:
            for ln in f:
                if not ln.strip():
                    continue
                tid, cfg, ls, rep = ln.split("\t")[:4]
                run_out = os.path.join(out_dir, "runs", f"{cfg}__ls{ls}__r{rep}")
                if (os.path.isdir(os.path.join(done, tid))
                        and not glob.glob(os.path.join(run_out, "trial_*", "run_*",
                                                       "metrics.json"))):
                    os.rmdir(os.path.join(done, tid))
                    print(f"  task {tid}: done marker without metrics.json — requeued")
    n = 0
    for tid in sorted(os.listdir(claims)):
        if not os.path.isdir(os.path.join(done, tid)):
            shutil.rmtree(os.path.join(claims, tid))
            n += 1
    n_done = len(os.listdir(done)) if os.path.isdir(done) else 0
    print(f"  reset {n} stale claim(s); {n_done} task(s) already complete")
    return n


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


def _parse_overrides(spec: str | None) -> dict:
    """'basin_width=2.2,n_optima=15' -> dict (same rules as evaluate.parse_ensemble_overrides)."""
    out: dict = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"--ensemble-override: {item!r} is not KEY=VALUE")
        k, v = (t.strip() for t in item.split("=", 1))
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


# Keys each --mix dataset accepts beyond its landscape overrides.
_MIX_KEYS = {
    "runrf": {"gp_run", "rf_n_estimators"},
    "rungp": {"gp_run", "gp_length_scale"},
    "needles": {"index", "n_optima", "basin_width", "placement_width", "unit_range"},
    "ensemble": {"index"},
}


def _parse_mix(spec: str, dim: int, seed: int) -> dict:
    """'needles:n_optima=2,basin_width=2.2' -> one landscape spec of a mixed showdown.

    ``runrf`` / ``rungp`` need ``gp_run``; ``needles`` needs ``n_optima`` and
    ``basin_width`` (``index`` = the sweep's draw, default 0); ``ensemble`` needs
    ``index`` and takes any ensemble key as an override. Every landscape shares
    ``--landscape-seed``.
    """
    dataset, _, rest = spec.partition(":")
    dataset = dataset.strip()
    if dataset not in _MIX_KEYS:
        raise SystemExit(f"--mix {spec!r}: dataset must be one of {sorted(_MIX_KEYS)}")
    kv = _parse_overrides(rest)
    out: dict = {"dataset": dataset}
    if dataset in ("runrf", "rungp"):
        unknown = sorted(set(kv) - _MIX_KEYS[dataset])
        if unknown or "gp_run" not in kv:
            raise SystemExit(f"--mix {spec!r}: {dataset} takes gp_run=RUN_DIR"
                             f"{',rf_n_estimators=N' if dataset == 'runrf' else ',gp_length_scale=L'}"
                             + (f"; unknown key(s) {unknown}" if unknown else ""))
        out["gp_run"] = os.path.abspath(str(kv["gp_run"]))
        if dataset == "runrf":
            out["rf_n_estimators"] = int(kv.get("rf_n_estimators", 500))
        else:
            out["gp_length_scale"] = float(kv.get("gp_length_scale", 0.05))
        return out
    out["index"] = int(kv.pop("index", 0 if dataset == "needles" else -1))
    if out["index"] < 0:
        raise SystemExit(f"--mix {spec!r}: ensemble needs index=IDX")
    if dataset == "needles":
        keys = set(kv)
        if not {"n_optima", "basin_width"} <= keys <= _MIX_KEYS["needles"]:
            raise SystemExit(f"--mix {spec!r}: needles takes n_optima=N,basin_width=B"
                             "[,placement_width=W,unit_range=true,index=I]")
    elif kv:
        from synthetic_data.ensemble import random_ensemble_config
        unknown = sorted(set(kv) - set(random_ensemble_config(dim, out["index"], seed=seed)))
        if unknown:
            raise SystemExit(f"--mix {spec!r}: unknown ensemble key(s) {unknown}")
    out["overrides"] = kv
    return out


def mix_label(spec: dict) -> str:
    """One-line description of a mixed-showdown landscape, for the summary."""
    if spec["dataset"] == "runrf":
        return f"RF of {os.path.basename(spec['gp_run'])} ({spec['rf_n_estimators']} trees)"
    if spec["dataset"] == "rungp":
        return f"GP of {os.path.basename(spec['gp_run'])} (length scale {spec['gp_length_scale']:g})"
    knobs = ", ".join(f"{k}={v}" for k, v in spec.get("overrides", {}).items())
    return f"{spec['dataset']} #{spec['index']}" + (f" ({knobs})" if knobs else "")


def _mix_flags(spec: dict, dim: int, seed: int) -> str:
    """evaluate.py dataset flags for one mixed-showdown landscape, as one shell line."""
    if spec["dataset"] == "runrf":
        return (f"--dataset runrf --gp-run '{spec['gp_run']}' "
                f"--rf-n-estimators {spec['rf_n_estimators']}")
    if spec["dataset"] == "rungp":
        return (f"--dataset rungp --gp-run '{spec['gp_run']}' "
                f"--gp-length-scale {spec['gp_length_scale']:g}")
    flags = (f"--dataset {spec['dataset']} --dim {dim} "
             f"--ensemble-landscape-indices {spec['index']} --ensemble-seed {seed}")
    if spec.get("overrides"):
        ov = ",".join(f"{k}={json.dumps(v)}" for k, v in spec["overrides"].items())
        flags += f" --ensemble-override '{ov}'"
    return flags


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
    ap.add_argument("--budget-points", type=int, default=DEFAULT_BUDGET_POINTS,
                    help="measured-point budget per run, converted to evaluate.py's "
                         "--max-lines so every config spends the same number of "
                         "experiments. --time-limit then acts only as a safety cap. "
                         "0 disables the cap and reverts to a pure wall-clock budget "
                         "(default: %(default)s)")
    ap.add_argument("--init", choices=INIT_DESIGNS, default="random",
                    help="initial design for every run (evaluate.py --init). 'axial' "
                         "measures more init lines, so it gets fewer optimizer lines "
                         "under the same --budget-points (default: %(default)s)")
    ap.add_argument("--ensemble-override", default=None, metavar="KEY=VAL[,...]",
                    help="pin ensemble knobs on every landscape (evaluate.py "
                         "--ensemble-override), e.g. 'basin_width=2.2,n_optima=15'; the "
                         "rest of each Sobol draw is kept")
    ap.add_argument("--n-landscapes", type=int, default=DEFAULT_N_LANDSCAPES,
                    help="how many shared landscapes every config is run on")
    ap.add_argument("--landscapes", default=None, metavar="IDX[,IDX...]",
                    help="explicit ensemble landscape indices instead of a random "
                         "--n-landscapes draw, e.g. '9182,22358' to reuse an earlier "
                         "showdown's landscapes")
    ap.add_argument("--dataset", choices=("ensemble", "needles", "rungp", "runrf"), default="ensemble",
                    help="'needles': the benchmarks/sweeps bumps-only landscape (flat "
                         "plain + Ackley bumps; needs --ensemble-override "
                         "'n_optima=N,basin_width=B'; same index = same centers at every "
                         "basin width). 'rungp': ONE fixed landscape, the GP fit to --gp-run's measured "
                         "points (evaluate.py --dataset rungp); its dimension comes from "
                         "the run, and the landscape list is just [0]")
    ap.add_argument("--mix", action="append", default=None, metavar="DATASET:KEY=VAL[,...]",
                    help="mixed showdown: one landscape per --mix (repeatable), numbered "
                         "0, 1, ... in order, all in one queue and one summary. "
                         "'runrf:gp_run=runs/run_69af' (gp_run may be a points .npz), "
                         "'rungp:gp_run=DIR,gp_length_scale=L', "
                         "'needles:n_optima=N,basin_width=B[,index=I]', "
                         "'ensemble:index=IDX[,KEY=VAL...]'. Replaces --dataset / "
                         "--landscapes; --dim and --landscape-seed apply to all")
    ap.add_argument("--gp-run", default=None, metavar="RUN_DIR",
                    help="rungp / runrf: run directory to fit the GP / RF to (e.g. runs/run_69af)")
    ap.add_argument("--gp-length-scale", type=float, default=0.05,
                    help="rungp: fixed Matern length scale (default: %(default)s)")
    ap.add_argument("--rf-n-estimators", type=int, default=500,
                    help="runrf: trees in the random forest fit to --gp-run's points, the "
                         "rungp landscape with an RF instead of the GP (default: %(default)s, "
                         "as visualization/plot_run.py)")
    ap.add_argument("--n-per-objective", type=int, default=2,
                    help="configs taken from each of dist_to_needles / dup_fraction")
    ap.add_argument("--landscape-seed", type=int, default=0,
                    help="seed for the random landscape choice AND the ensemble "
                         "construction, so the whole showdown is reproducible")
    ap.add_argument("--walltime-margin", type=float, default=0.75,
                    help="hours added to the per-run budget to bound ONE task, covering "
                         "landscape construction, metrics and the CoNet renders (whose "
                         "own timeout is 0.5 h). A worker stops claiming when less than "
                         "this much time is left, so over-reserving costs idle wall-time "
                         "at the tail of every worker (default: %(default)s)")
    ap.add_argument("--n-repeats", type=int, default=1,
                    help="repeats of every (config, landscape) cell. >1 turns the "
                         "showdown into a distribution rather than a single draw: the "
                         "summary reports the mean and the CSVs carry median/variance/"
                         "p5/p95 plus every raw value. Only repeat 1 renders plots.")
    ap.add_argument("--n-workers", type=int, default=5,
                    help="persistent SLURM workers that drain the task queue. This is "
                         "the number of jobs the campaign ever queues, regardless of "
                         "how many runs it contains (default: %(default)s)")
    ap.add_argument("--worker-hours", type=float, default=24,
                    help="wall-time per worker; workers stop claiming when a task no "
                         "longer fits, so this can be shorter than the whole campaign "
                         "(default: %(default)s, the partition maximum)")
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--partition", default="sched_mit_sloan_gpu_r8",
                    help="SLURM partition(s), comma-separated (default: %(default)s)")
    ap.add_argument("--mem", default="128G",
                    help="SLURM memory per worker; workers peak at ~6 GB "
                         "(default: %(default)s)")
    ap.add_argument("--no-shared-history", action="store_true",
                    help="when --runs is a single run dir, do NOT pool its siblings")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip planning; just write the summary for a finished showdown")
    ap.add_argument("--unfold-repeats", action="store_true",
                    help="with --summary-only: give every repeat its own row in the "
                         "results table instead of one mean row per (config, landscape) "
                         "cell. Repeats that ran --no-plots get their convergence and "
                         "%% space penalised plots rebuilt from CSVs; their CoNet pair "
                         "comes from --backfill-conet")
    ap.add_argument("--backfill-conet", action="store_true",
                    help="write conet_backfill.sbatch, which renders the CoNet pair for "
                         "every finished run lacking one (the --no-plots repeats), then "
                         "exit. Submit it, then re-run --summary-only --unfold-repeats")
    ap.add_argument("--reset-stale", action="store_true",
                    help="clear claims left by killed workers so those tasks are retried, "
                         "then exit. Run before RE-submitting; never while workers are up.")
    ap.add_argument("--status", action="store_true",
                    help="print queue progress (done / claimed / pending) and exit")
    args = ap.parse_args()
    args.ensemble_overrides = _parse_overrides(args.ensemble_override)
    args.mix_specs = [_parse_mix(m, args.dim, args.landscape_seed) for m in args.mix or []]
    if args.mix_specs:
        args.dataset = "mixed"

    out_dir = os.path.abspath(args.out)

    if args.reset_stale:
        reset_stale(out_dir)
        return

    if args.status:
        print_status(out_dir)
        return

    if args.backfill_conet:
        write_conet_backfill(out_dir)
        return

    if args.unfold_repeats and not args.summary_only:
        ap.error("--unfold-repeats only applies with --summary-only")

    if args.summary_only:
        write_landscape_summary(out_dir, unfold_repeats=args.unfold_repeats)
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

    if args.dataset == "mixed":
        landscapes = list(range(len(args.mix_specs)))
    elif args.dataset in ("rungp", "runrf"):
        landscapes = [0]
    elif args.landscapes:
        landscapes = sorted(int(t) for t in args.landscapes.split(",") if t.strip())
    else:
        landscapes = pick_landscapes(args.n_landscapes, args.landscape_seed)
    write_plan(chosen, landscapes, args)


if __name__ == "__main__":
    main()
