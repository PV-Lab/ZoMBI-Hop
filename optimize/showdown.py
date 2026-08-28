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
(default 4000), converted to evaluate.py's ``--max-lines``. A showdown compares what
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
# Sampling budget per run, in MEASURED POINTS. Every run of a showdown gets the same
# number of experiments, so the comparison is "who does more with the same budget"
# rather than "who ran on the faster node". The wall-clock --time-limit stays on as a
# secondary safety cap only.
DEFAULT_BUDGET_POINTS = 4000
# One LineBO line = NUM_EXPERIMENTS measured points, and every run starts with a fixed
# preamble of N_INIT_LINES init lines before the optimizer's first objective call.
# Mirrors run_mobo.NUM_EXPERIMENTS / N_INIT_LINES (run_mobo.py:291); duplicated here so
# planning stays importable without pulling in torch.
POINTS_PER_LINE = 24
N_INIT_LINES = 2
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


def budget_to_max_lines(budget_points: int) -> int | None:
    """Measured-point budget -> the ``--max-lines`` cap evaluate.py takes.

    ``--max-lines`` counts objective calls, i.e. lines the OPTIMIZER asked for; the
    ``N_INIT_LINES`` init lines are a deterministic preamble that runs before the first
    such call, so they come off the budget rather than counting against the cap.
    Returns None when the budget is <= 0, meaning "no point cap, time budget only".
    """
    if budget_points is None or budget_points <= 0:
        return None
    n_lines = int(budget_points // POINTS_PER_LINE) - N_INIT_LINES
    if n_lines < 1:
        raise SystemExit(
            f"--budget-points {budget_points} leaves no optimizer lines after the "
            f"{N_INIT_LINES} init line(s) of {POINTS_PER_LINE} point(s) each")
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
#SBATCH --mem=128G
#SBATCH --partition=sched_mit_sloan_gpu_r8
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
        claimed_this_pass=$(( claimed_this_pass + 1 ))

        RUN_OUT="{out_dir}/runs/${{CFG}}__ls${{LS}}__r${{REP}}"
        echo "[$(date)] worker $SLURM_ARRAY_TASK_ID -> task $TID: config=$CFG landscape=$LS repeat=$REP"

        # Repeat 1 renders the full artifact set; the rest pass --no-plots and produce only
        # CSVs + metrics.json. The repeats exist to be counted, not looked at, and the CoNet
        # UMAP render costs as much as the run itself.
        PLOTFLAG=""
        [ "$NOPLOTS" = "1" ] && PLOTFLAG="--no-plots"

        uv run python optimize/evaluate.py \\
            --hparams-json "{out_dir}/configs/$CFG.json" \\
            --dataset ensemble --dim {dim} \\
            --num-runs 1 \\
            --ensemble-landscape-indices "$LS" \\
            --ensemble-seed {landscape_seed} \\
            --time-limit-min {time_limit_min} \\
            {max_lines_flag}--device cuda \\
            --no-video $PLOTFLAG \\
            --out-dir "$RUN_OUT" < /dev/null
        # `< /dev/null` matters: without it the child inherits this loop's stdin and can
        # swallow the rest of the queue, silently truncating it.

        if [ $? -eq 0 ]; then
            mkdir -p "$DONE/$TID"
            n_ran=$(( n_ran + 1 ))
        else
            echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: task $TID FAILED (claim kept; --reset-stale requeues it)"
        fi
    done < "$WORKQ"

    [ "$claimed_this_pass" -eq 0 ] && break
    echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: pass $pass_no claimed $claimed_this_pass task(s); rescanning"
done

echo "[$(date)] worker $SLURM_ARRAY_TASK_ID done, ran $n_ran task(s) over $pass_no pass(es)"
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

    n_repeats = max(1, int(args.n_repeats))
    max_lines = budget_to_max_lines(args.budget_points)
    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dim": args.dim,
        "time_limit_hours": args.time_limit,
        "budget_points": args.budget_points if max_lines is not None else None,
        "max_lines": max_lines,
        "landscape_seed": args.landscape_seed,
        "landscapes": landscapes,
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
        dim=args.dim,
        time_limit_min=f"{time_limit_min:g}",
        max_lines_flag=(f"--max-lines {max_lines} \\\n            "
                        if max_lines is not None else ""),
        landscape_seed=args.landscape_seed,
    )
    sbatch_path = os.path.join(out_dir, "showdown.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch)
    os.chmod(sbatch_path, 0o755)

    print(f"\n  plan -> {out_dir}")
    print(f"    {len(names)} config(s) x {len(landscapes)} landscape(s) x "
          f"{n_repeats} repeat(s) = {len(tasks)} run(s)")
    print(f"    landscapes: {landscapes}")
    if max_lines is not None:
        print(f"    budget: {args.budget_points} point(s)/run = {N_INIT_LINES} init + "
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
    """
    claims = os.path.join(out_dir, "claims")
    done = os.path.join(out_dir, "done")
    if not os.path.isdir(claims):
        print(f"  no claims dir in {out_dir} — nothing to reset")
        return 0
    n = 0
    for tid in sorted(os.listdir(claims)):
        if not os.path.isdir(os.path.join(done, tid)):
            os.rmdir(os.path.join(claims, tid))
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
    ap.add_argument("--n-landscapes", type=int, default=DEFAULT_N_LANDSCAPES,
                    help="how many shared landscapes every config is run on")
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
    ap.add_argument("--no-shared-history", action="store_true",
                    help="when --runs is a single run dir, do NOT pool its siblings")
    ap.add_argument("--summary-only", action="store_true",
                    help="skip planning; just write the summary for a finished showdown")
    ap.add_argument("--reset-stale", action="store_true",
                    help="clear claims left by killed workers so those tasks are retried, "
                         "then exit. Run before RE-submitting; never while workers are up.")
    ap.add_argument("--status", action="store_true",
                    help="print queue progress (done / claimed / pending) and exit")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)

    if args.reset_stale:
        reset_stale(out_dir)
        return

    if args.status:
        print_status(out_dir)
        return

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
