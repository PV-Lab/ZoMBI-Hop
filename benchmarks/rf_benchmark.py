"""
benchmarks/rf_benchmark.py
==========================
Tuned vs. LLM-chosen hyperparameters on the REAL campaigns' GP landscapes, at
d = 3, 4 and 6.

What this answers
-----------------
``optimize/showdown.py`` compares configurations on *synthetic ensemble*
landscapes: an unbounded supply of statistically comparable surfaces, which is the
right instrument for ranking many configurations against each other. It cannot say
whether the winner is better **on the material system the lab actually ran**.

This suite asks that question instead, and only that question. Each dimension's
landscape is the length-scale-0.05 Matern GP fit to the entire measured campaign at
that dimension (``data/2nd_real_run.db``, ``data/3rd_real_run.db``, ``data/6d.db``
— ``warm_start.warm_gp_landscape.fullgp_objective``, the repo's evaluation ground
truth), and exactly two arms are run on it:

    tuned   the hyperparameters tuned for THAT dimension
    llm     optimize/hparams/3d_llm_chosen.json, the same file at every dimension

so each dimension is a clean paired comparison on one fixed, real surface.

Differences from showdown.py
----------------------------
* **The landscape is fixed, not sampled.** A campaign's GP surface is
  deterministic — one dimension, one landscape — so there is no ``--n-landscapes``
  and no landscape seed. All spread within a cell is the optimizer's own, which is
  what ``--n-repeats`` measures.
* **Three dimensions in one campaign.** A task carries its dimension, and every
  worker can run any of them.
* **The summary is built in and deliberately thin.** ``summary_table.py`` renders
  four tables, two chart sections and a plot column per cell. Here the question has
  one shape — is tuned better than llm, on which axis, by how much — so the summary
  is three tables: per-dimension means with confidence intervals, the per-dimension
  percent change, and one overall number per metric. Everything else (every raw
  value, every per-cell statistic) goes to the CSVs beside it.
* **There is a second, truncated view of the same runs.** The last section of the
  summary re-reads every run at an earlier, per-dimension line budget
  (``CUT_LINES``) out of its ``metrics_over_time.csv``, because the full budget is
  past the point where the higher dimensions still separate. It costs no compute —
  see ``cut_metrics`` — and it carries one caveat about dup fraction, documented
  there.
* **``dup_fraction`` is reported, not ``median_nn_spacing``.** The showdown tables
  swapped in spacing because they span campaigns scored on either side of the
  ``NOISE_LEVEL`` change (0.064 -> 0.128), which makes old and new dup fractions
  answers to different questions. This campaign is scored in one pass under one
  ``NOISE_LEVEL``, so dup fraction is directly comparable across every cell of it
  and needs no threshold-free stand-in.

Both reported metrics are LOWER IS BETTER, so a negative percent change is the
tuned configuration winning.

Execution model
---------------
Identical to showdown.py's, and for the same reasons: the SLURM array is a pool of
PERSISTENT WORKERS draining a shared ``tasks.tsv`` by atomic ``mkdir`` claim, with
``done/`` markers making the campaign resumable and a rotated queue view spreading
the pool across arms and dimensions instead of grinding through one at a time. See
that module's header for the full rationale.

Usage
-----
  conda activate zombi-hop

  # Plan the default campaign (3 dims x 2 arms x 10 repeats = 60 runs) and submit.
  python benchmarks/rf_benchmark.py plan --out optimize/runs/rf_benchmark
  sbatch optimize/runs/rf_benchmark/rf_benchmark.sbatch

  # Progress, and resuming after workers were killed.
  python benchmarks/rf_benchmark.py status      --out optimize/runs/rf_benchmark
  python benchmarks/rf_benchmark.py reset-stale --out optimize/runs/rf_benchmark

  # The summary (also written automatically by the last worker to finish).
  python benchmarks/rf_benchmark.py summary --out optimize/runs/rf_benchmark
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DIMS = (3, 4, 6)

# The tuned arm, per dimension. These are the configurations this repo arrived at
# for each dimension by its own tuning route, which is why they come from three
# different places rather than one directory.
TUNED_HPARAMS: dict[int, str] = {
    3: "optimize/hparams/trial_112_composition.json",
    4: "optimize/runs/mobo_ensemble_4d_job17147232/trial_10/trial.json",
    6: "optimize/hparams/clamped_6d/dist1c.json",
}
# The control arm: ONE file, used unchanged at every dimension. That is the point
# of it — an LLM asked for reasonable hyperparameters without seeing the landscape,
# so it does not get a per-dimension version.
LLM_HPARAMS = "optimize/hparams/3d_llm_chosen.json"

ARMS = ("tuned", "llm")

# Sampling budget per run, in MEASURED POINTS: every run of every arm spends the
# same number of experiments, so the comparison is "who does more with the same
# budget" rather than "who ran on the faster node". Mirrors showdown.py.
DEFAULT_BUDGET_POINTS = 3000
POINTS_PER_LINE = 24       # run_mobo.NUM_EXPERIMENTS
N_INIT_LINES = 2           # run_mobo.N_INIT_LINES

DEFAULT_N_REPEATS = 10
DEFAULT_N_WORKERS = 6
DEFAULT_TIME_LIMIT_H = 0.5   # safety cap only; 3000 points measured at 6-17 min

# Metrics carried through every table and CSV. Both are minimised.
METRICS = ("dist_to_needles", "dup_fraction")

# EARLY-BUDGET CUT, in TOTAL measured lines (init lines included), per dimension.
# The full budget is 125 lines at every dimension, which is past the point where the
# higher dimensions are still moving: the question this view asks is who is ahead
# while the campaign is still being decided, not who ends level. The cuts are
# per-dimension because the dimensions converge at different rates.
CUT_LINES: dict[int, int] = {3: 40, 4: 60, 6: 110}
CUT_METRICS = ("dist_to_needles_cut", "dup_fraction_cut")

METRIC_LABEL = {"dist_to_needles": "dist to needles", "dup_fraction": "dup fraction",
                "dist_to_needles_cut": "dist to needles",
                "dup_fraction_cut": "dup fraction (unscaled radius)"}


# ─── small statistics (stdlib only, like summary_table.py) ───────────────────────

# 95% two-sided Student-t quantiles by degrees of freedom, so a 10-repeat cell gets
# an honest interval rather than the normal approximation's 1.96 (which is 13% too
# narrow at n=10). Values past the table converge on 1.96.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000}


def _t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    smaller = [k for k in _T95 if k < df]
    return _T95[max(smaller)] if smaller else 1.96


def mean_ci(values) -> tuple[float | None, float | None, int]:
    """``(mean, half-width of the 95% CI, n)`` over the finite values.

    The interval is Student-t on n-1 degrees of freedom. A single value has a mean
    but no interval — reported as None rather than 0, which would claim one run
    pins the mean exactly.
    """
    xs = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x:
            xs.append(x)
    if not xs:
        return None, None, 0
    mu = statistics.fmean(xs)
    if len(xs) < 2:
        return mu, None, len(xs)
    se = statistics.stdev(xs) / (len(xs) ** 0.5)
    return mu, _t95(len(xs) - 1) * se, len(xs)


def pct_change(new: float | None, base: float | None) -> float | None:
    """Percent change of *new* against *base*, the llm arm being the base.

    Both metrics are minimised, so a NEGATIVE result is the tuned arm winning.
    """
    if new is None or base is None or base == 0:
        return None
    return (new - base) / abs(base) * 100.0


# ─── planning ────────────────────────────────────────────────────────────────────

def budget_to_max_lines(budget_points: int) -> int | None:
    """Measured-point budget -> evaluate.py's ``--max-lines``.

    ``--max-lines`` counts objective calls, i.e. lines the OPTIMIZER asked for;
    the ``N_INIT_LINES`` init lines are a deterministic preamble before the first
    such call, so they come off the budget rather than counting against the cap.
    """
    if budget_points is None or budget_points <= 0:
        return None
    n = int(budget_points // POINTS_PER_LINE) - N_INIT_LINES
    if n < 1:
        raise SystemExit(f"--budget-points {budget_points} leaves no optimizer lines")
    return n


def _load_hparams(path: str) -> dict:
    """The hyperparameter dict from a flat JSON or a ``trial.json``-style blob."""
    full = path if os.path.isabs(path) else os.path.join(_REPO, path)
    if not os.path.isfile(full):
        raise SystemExit(f"hyperparameter file not found: {full}")
    with open(full) as f:
        blob = json.load(f)
    hp = blob.get("hparams", blob)
    if not isinstance(hp, dict) or not hp:
        raise SystemExit(f"{full} carries no hyperparameters")
    return hp


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

# PERSISTENT WORKER POOL — not one SLURM task per run. The array is {n_workers}
# long-lived workers, not {n_tasks} runs: each claims task after task off tasks.tsv
# until the queue drains or its wall-time runs out, so a finished run hands its GPU
# to the next run inside the same allocation instead of returning it to the
# scheduler. Claiming is one atomic `mkdir` per task, the only primitive reliably
# atomic on a shared filesystem; done/<id> markers make the campaign resumable.
# Clear claims left by killed workers before re-submitting, or they are never
# retried:
#
#     python benchmarks/rf_benchmark.py reset-stale --out {out_dir}
#     sbatch {out_dir}/rf_benchmark.sbatch

cd {repo}

QUEUE="{out_dir}/tasks.tsv"
CLAIMS="{out_dir}/claims"
DONE="{out_dir}/done"
WORKQ="{out_dir}/logs/workq_${{SLURM_ARRAY_TASK_ID}}.tsv"
mkdir -p "$CLAIMS" "$DONE"

# ROTATED QUEUE VIEW — worker k starts k/{n_workers} of the way down and wraps, so
# the pool spreads across dimensions and arms at once. Read top-down in lockstep,
# every worker would race for the same first task, and a campaign cut short by
# wall-time would hold full repeats of the first arm and none of the last.
NTASKS=$(wc -l < "$QUEUE")
OFFSET=$(( SLURM_ARRAY_TASK_ID * NTASKS / {n_workers} ))
{{ tail -n +$(( OFFSET + 1 )) "$QUEUE"; head -n "$OFFSET" "$QUEUE"; }} > "$WORKQ"

# Stop claiming when too little wall-time is left to finish a task; a task killed
# mid-flight leaves a claim with no result, which is pure waste.
DEADLINE=$(( $(date +%s) + {worker_seconds} - {task_seconds} ))

echo "[$(date)] worker $SLURM_ARRAY_TASK_ID up on $(hostname), gpu=${{CUDA_VISIBLE_DEVICES:-?}}, queue=$WORKQ (offset $OFFSET/$NTASKS)"

n_ran=0
out_of_time=0
pass_no=0

# Outer rescan: one pass can walk past a task that was claimed at the time but
# never completed. A pass that claims nothing means the queue is genuinely
# drained, so this terminates — but no worker exits while unclaimed work remains.
while [ "$out_of_time" -eq 0 ]; do
    pass_no=$(( pass_no + 1 ))
    claimed_this_pass=0

    while read -r TID DIM CFG REP NOPLOTS; do
        [ -z "$TID" ] && continue
        [ -d "$DONE/$TID" ] && continue
        if [ "$(date +%s)" -gt "$DEADLINE" ]; then
            echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: out of wall-time, stopping (ran $n_ran)"
            out_of_time=1
            break
        fi
        mkdir "$CLAIMS/$TID" 2>/dev/null || continue
        claimed_this_pass=$(( claimed_this_pass + 1 ))

        RUN_OUT="{out_dir}/runs/${{CFG}}__r${{REP}}"
        echo "[$(date)] worker $SLURM_ARRAY_TASK_ID -> task $TID: dim=$DIM config=$CFG repeat=$REP"

        # Repeat 1 renders the full artifact set; the rest pass --no-plots and write
        # only CSVs + metrics.json. The repeats exist to be counted, not looked at,
        # and the CoNet UMAP render can cost as much as the run itself.
        PLOTFLAG=""
        [ "$NOPLOTS" = "1" ] && PLOTFLAG="--no-plots"

        uv run python optimize/evaluate.py \\
            --hparams-json "{out_dir}/configs/$CFG.json" \\
            --dataset fullgp --dim "$DIM" \\
            --num-runs 1 \\
            --time-limit-min {time_limit_min} \\
            {max_lines_flag}--device cuda \\
            --no-video $PLOTFLAG \\
            --out-dir "$RUN_OUT" < /dev/null
        # `< /dev/null` matters: without it the child inherits this loop's stdin and
        # can swallow the rest of the queue, silently truncating it.

        if [ $? -eq 0 ]; then
            mkdir -p "$DONE/$TID"
            n_ran=$(( n_ran + 1 ))
        else
            echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: task $TID FAILED (claim kept; reset-stale requeues it)"
        fi
    done < "$WORKQ"

    [ "$claimed_this_pass" -eq 0 ] && break
    echo "[$(date)] worker $SLURM_ARRAY_TASK_ID: pass $pass_no claimed $claimed_this_pass task(s); rescanning"
done

echo "[$(date)] worker $SLURM_ARRAY_TASK_ID done, ran $n_ran task(s) over $pass_no pass(es)"

# Last one out writes the summary. Every worker tries; the ones that are not last
# simply report incomplete cells, and the file is rewritten by whoever follows.
uv run python benchmarks/rf_benchmark.py summary --out {out_dir} || true
"""


def config_name(dim: int, arm: str) -> str:
    return f"{dim}d_{arm}"


def precompute_peaks(dims) -> None:
    """Build each dimension's GP surface once, before any worker starts.

    The d>=5 peak detector caches to disk, and without this every worker would
    race to compute the same peaks on first use. It also fails LOUDLY here, at plan
    time, rather than inside 60 queued jobs — a missing campaign DB or an
    unimportable scientific stack is a planning error, not a runtime one.
    """
    from warm_start.warm_gp_landscape import fullgp_objective

    for d in dims:
        o = fullgp_objective(int(d))
        print(f"  [landscape] d={d}: GP on {o['n_points']} pt(s) / {o['n_lines']} "
              f"line(s) -> {len(o['peaks'])} reference peak(s)")


def plan(args) -> str:
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, "configs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    n_repeats = max(1, int(args.n_repeats))
    max_lines = budget_to_max_lines(args.budget_points)

    sources = {"llm": LLM_HPARAMS, **{f"tuned{d}": TUNED_HPARAMS[d] for d in dims}}
    configs: list[dict] = []
    for d in dims:
        for arm in ARMS:
            src = LLM_HPARAMS if arm == "llm" else TUNED_HPARAMS[d]
            name = config_name(d, arm)
            hp = _load_hparams(src)
            with open(os.path.join(out_dir, "configs", f"{name}.json"), "w") as f:
                json.dump({"config_name": name, "dim": d, "arm": arm,
                           "source": src, "hparams": hp}, f, indent=2)
            configs.append({"name": name, "dim": d, "arm": arm,
                            "source": src, "hparams": hp})

    if args.precompute_peaks:
        precompute_peaks(dims)

    tasks: list[tuple[int, int, str, int, int]] = []
    for c in configs:
        for rep in range(1, n_repeats + 1):
            tasks.append((len(tasks), c["dim"], c["name"], rep, 0 if rep == 1 else 1))
    with open(os.path.join(out_dir, "tasks.tsv"), "w") as f:
        for tid, dim, name, rep, noplots in tasks:
            f.write(f"{tid:05d}\t{dim}\t{name}\t{rep}\t{noplots}\n")

    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "suite": "rf_benchmark",
        "landscape": "fullgp",
        "landscape_note": ("length-scale-0.05 Matern GP fit to the entire measured "
                           "campaign at each dimension; deterministic, one landscape "
                           "per dimension"),
        "dims": dims,
        "arms": list(ARMS),
        "hparam_sources": sources,
        "budget_points": args.budget_points if max_lines is not None else None,
        "max_lines": max_lines,
        "time_limit_hours": args.time_limit,
        "n_repeats": n_repeats,
        "n_tasks": len(tasks),
        "configs": configs,
    }
    with open(os.path.join(out_dir, "rf_benchmark_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    task_h = args.time_limit + args.walltime_margin
    walltime_h = max(1, int(args.worker_hours))
    # Expected drain uses the measured per-task overhead (landscape build, metrics,
    # the repeat-1 renders), NOT the claim reserve; sizing off the reserve overstates
    # a campaign by ~2.5x. See showdown.py's note on the same constant.
    PER_TASK_OVERHEAD_H = 0.05
    exp_h = len(tasks) * (args.time_limit + PER_TASK_OVERHEAD_H) / max(1, args.n_workers)
    claim_h = walltime_h - task_h

    sbatch = SBATCH_TEMPLATE.format(
        job_name=args.job_name or "rf_benchmark",
        out_dir=out_dir,
        repo=_REPO,
        walltime=f"{walltime_h}:00:00",
        worker_seconds=int(walltime_h * 3600),
        task_seconds=int(task_h * 3600),
        last_worker=args.n_workers - 1,
        n_workers=args.n_workers,
        n_tasks=len(tasks),
        time_limit_min=f"{args.time_limit * 60.0:g}",
        max_lines_flag=(f"--max-lines {max_lines} \\\n            "
                        if max_lines is not None else ""),
    )
    sbatch_path = os.path.join(out_dir, "rf_benchmark.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch)
    os.chmod(sbatch_path, 0o755)

    print(f"\n  plan -> {out_dir}")
    print(f"    {len(dims)} dim(s) x {len(ARMS)} arm(s) x {n_repeats} repeat(s) "
          f"= {len(tasks)} run(s)")
    for d in dims:
        print(f"    d={d}: tuned <- {TUNED_HPARAMS[d]}")
    print(f"    all dims: llm <- {LLM_HPARAMS}")
    if max_lines is not None:
        print(f"    budget: {args.budget_points} point(s)/run = {N_INIT_LINES} init + "
              f"{max_lines} optimizer line(s) @ {POINTS_PER_LINE} point(s); "
              f"--time-limit {args.time_limit:g} h is a safety cap only")
    print(f"    {args.n_workers} persistent worker(s) @ {walltime_h} h; "
          f"~{exp_h:.1f} h to drain against a {claim_h:.1f} h claim budget")
    if exp_h > claim_h:
        print("    NOTE: expected drain exceeds the claim budget — this campaign "
              "will NOT finish in one submission.")
    print(f"    submit with:  sbatch {sbatch_path}")
    return sbatch_path


# ─── queue bookkeeping ───────────────────────────────────────────────────────────

def _task_ids(out_dir: str) -> list[str]:
    queue = os.path.join(out_dir, "tasks.tsv")
    if not os.path.isfile(queue):
        raise SystemExit(f"no tasks.tsv in {out_dir}")
    with open(queue) as f:
        return [ln.split("\t")[0] for ln in f if ln.strip()]


def print_status(out_dir: str) -> None:
    tids = _task_ids(out_dir)
    done, claims = os.path.join(out_dir, "done"), os.path.join(out_dir, "claims")
    n_done = sum(1 for t in tids if os.path.isdir(os.path.join(done, t)))
    n_claim = sum(1 for t in tids if os.path.isdir(os.path.join(claims, t))
                  and not os.path.isdir(os.path.join(done, t)))
    print(f"  {os.path.basename(out_dir)}: {len(tids)} task(s) — {n_done} done, "
          f"{n_claim} running, {len(tids) - n_done - n_claim} pending")


def reset_stale(out_dir: str) -> int:
    """Drop claims with no done marker so those tasks are retried.

    A killed worker (wall-time, node failure, OOM) leaves its claim behind with no
    result, and nothing else clears it, so without this the campaign looks drained
    while silently missing runs. Safe ONLY when no workers are live: a claim on a
    running task is indistinguishable from an abandoned one.
    """
    claims, done = os.path.join(out_dir, "claims"), os.path.join(out_dir, "done")
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


# ─── summary ─────────────────────────────────────────────────────────────────────

def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _run_dir_under(base: str) -> str | None:
    """The ``trial_<n>/run_<k>`` artifact dir under one evaluate.py output dir."""
    hits = sorted(glob.glob(os.path.join(base, "trial_*", "run_*")))
    for h in hits:
        if os.path.isfile(os.path.join(h, "metrics.json")):
            return h
    return hits[0] if hits else None


def collect_runs(out_dir: str, name: str, cut_lines: int | None = None) -> list[dict]:
    """Every finished repeat of one configuration, as ``{repeat, run_dir, metrics}``.

    ``dist_to_needles`` and ``dup_fraction`` are read straight from ``metrics.json``,
    not recomputed. ``summary_table.recomputed_dist`` exists to put runs scored
    before 2026-08-11 (greedy matching, penalty 10.0) on the same axis as later ones;
    every run here is scored in one pass by the current ``eval_metrics``, which
    already does the optimal assignment at a 0.5 cutoff, so there is nothing to
    reconcile. It also could not run if it wanted to — it rebuilds the landscape
    from ``ensemble_config.json``, which a fullgp run does not write.
    """
    rows: list[dict] = []
    for base in sorted(glob.glob(os.path.join(out_dir, "runs", f"{name}__r*"))):
        try:
            rep = int(base.rsplit("__r", 1)[1])
        except (IndexError, ValueError):
            continue
        rd = _run_dir_under(base)
        if not rd:
            continue
        met = _read_json(os.path.join(rd, "metrics.json"))
        if not met:
            continue
        row = {
            "repeat": rep,
            "run_dir": rd,
            "dist_to_needles": met.get("dist_to_needles"),
            "dup_fraction": met.get("dup_fraction"),
            "runtime_s": met.get("runtime_s"),
            "n_needles": _n_needles(rd),
        }
        row.update(cut_metrics(rd, cut_lines) if cut_lines
                   else {m: None for m in CUT_METRICS})
        rows.append(row)
    return sorted(rows, key=lambda r: r["repeat"])


def cut_metrics(run_dir: str, cut_lines: int) -> dict[str, float | None]:
    """``dist_to_needles``/``dup_fraction`` as of the first *cut_lines* measured lines.

    Back-calculated, not re-simulated: ``metrics_over_time.csv`` already carries both
    metrics evaluated at every optimizer line, so truncating the campaign is a row
    lookup rather than a rerun. Row ``iteration = k`` is written when the optimizer
    asks for its k-th line, BEFORE that line is measured, so the state it describes
    holds ``k + N_INIT_LINES - 1`` measured lines; this takes the last row at or
    under the cut, which lands exactly on ``cut_lines * POINTS_PER_LINE`` points.

    ONE CAVEAT, and it is why the cut dup fraction is labelled separately:
    ``metrics_over_time.csv`` computes dup fraction at ONE GLOBAL radius, while
    ``metrics.json`` — the full-budget column in the tables above — scales the radius
    per point by the zoom-zone size it was sampled in. The zoom SIZES are not
    persisted (``points.csv`` stores the zoom index only), so the zoom-scaled version
    cannot be reconstructed at a cut, and the unscaled one reads systematically
    higher. It is exact and comparable ACROSS ARMS, which is what the comparison
    needs, but it is not comparable to the full-budget dup fraction column.
    ``dist_to_needles`` has no such split — both files make the identical call.
    """
    out: dict[str, float | None] = {m: None for m in CUT_METRICS}
    path = os.path.join(run_dir, "metrics_over_time.csv")
    if not os.path.isfile(path):
        return out
    try:
        with open(path, newline="") as f:
            best = None
            for row in csv.DictReader(f):
                try:
                    it = int(row["iteration"])
                except (KeyError, TypeError, ValueError):
                    continue
                if it + N_INIT_LINES - 1 <= cut_lines:
                    best = row
    except OSError:
        return out
    if best is None:
        return out
    for m, col in zip(CUT_METRICS, METRICS):
        try:
            out[m] = float(best[col])
        except (KeyError, TypeError, ValueError):
            out[m] = None
    return out


def _n_needles(run_dir: str) -> int | None:
    """How many needles a run declared, counted from ``needles.csv``.

    Not in ``metrics.json`` — but it is the cardinality half of
    ``dist_to_needles`` (which averages over ``max(n_declared, n_true)``), so a
    surprising dist is usually explained by this column, and the ledger is the
    place to be able to check that without reopening 60 run directories.
    """
    path = os.path.join(run_dir, "needles.csv")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="") as f:
            return max(0, sum(1 for _ in f) - 1)      # minus the header
    except OSError:
        return None


def _fmt(v, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ci(mu, half, nd: int = 4) -> str:
    """``mean ± half-width``, or just the mean when one run gives no interval."""
    if mu is None:
        return "—"
    return f"{mu:.{nd}f}" + (f" ± {half:.{nd}f}" if half is not None else "")


def _fmt_pct(v) -> str:
    """Signed percent change, with the direction spelled out.

    Both metrics are minimised, so the sign alone is ambiguous to a reader skimming
    the table — the word is what makes the row unambiguous.
    """
    if v is None:
        return "—"
    word = "better" if v < 0 else ("worse" if v > 0 else "no change")
    return f"{v:+.1f}% ({word})"


def write_summary(out_dir: str) -> str:
    out_dir = os.path.abspath(out_dir)
    manifest = _read_json(os.path.join(out_dir, "rf_benchmark_manifest.json"))
    if not manifest:
        raise SystemExit(f"no rf_benchmark_manifest.json in {out_dir}")
    dims = manifest["dims"]
    n_repeats = int(manifest.get("n_repeats", 1) or 1)

    runs: dict[tuple[int, str], list[dict]] = {
        (d, arm): collect_runs(out_dir, config_name(d, arm), CUT_LINES.get(d))
        for d in dims for arm in ARMS
    }
    # (mean, ci half-width, n) per (dim, arm, metric) — the one quantity every table
    # below is built from, computed once so the tables cannot disagree.
    stat: dict[tuple[int, str, str], tuple] = {
        (d, arm, m): mean_ci([r[m] for r in runs[(d, arm)]])
        for d in dims for arm in ARMS for m in METRICS + CUT_METRICS
    }

    L: list[str] = []
    A = L.append
    A("# rf_benchmark — tuned vs LLM-chosen hyperparameters on the real-campaign GP landscapes")
    A("")
    A(f"Landscape: `fullgp` — {manifest.get('landscape_note')}.  "
      f"Budget: {manifest.get('budget_points')} measured points/run "
      f"({manifest.get('max_lines')} optimizer lines), "
      f"{n_repeats} repeats per cell.  "
      "Both metrics are **lower is better**, so a negative percent change is the "
      "tuned arm winning.  Intervals are 95% Student-t on the repeats.")
    A("")

    A("## Means by dimension")
    A("")
    A("| dim | arm | runs | dist to needles | dup fraction |")
    A("|---|---|---|---|---|")
    for d in dims:
        for arm in ARMS:
            dm, dh, n = stat[(d, arm, "dist_to_needles")]
            um, uh, _ = stat[(d, arm, "dup_fraction")]
            A(f"| {d}d | `{arm}` | {n}/{n_repeats} | {_fmt_ci(dm, dh)} | "
              f"{_fmt_ci(um, uh)} |")
    A("")

    A("## Percent change, tuned vs llm")
    A("")
    A("| dim | dist to needles | dup fraction |")
    A("|---|---|---|")
    for d in dims:
        cells = [_fmt_pct(pct_change(stat[(d, "tuned", m)][0], stat[(d, "llm", m)][0]))
                 for m in METRICS]
        A(f"| {d}d | " + " | ".join(cells) + " |")
    A("")

    A("## Overall")
    A("")
    # The mean of the three per-dimension means, NOT a pooled average of all 30 runs
    # per arm: the three landscapes are different surfaces whose metrics live on
    # different natural scales, so pooling would silently weight the comparison by
    # whichever dimension happens to produce the largest numbers. Equal weight per
    # dimension is the question actually being asked. The interval is across the
    # three dimension means (n=3), so it describes consistency ACROSS dimensions,
    # which is a different and much wider thing than the per-cell intervals above.
    A("Each dimension weighs equally: the overall number is the mean of the three "
      "per-dimension means, and its interval is the spread across those three "
      "(n=3), not across runs.")
    A("")
    A("| metric | tuned | llm | percent change |")
    A("|---|---|---|---|")
    overall: dict[str, dict[str, float | None]] = {}
    for m in METRICS:
        row = {}
        for arm in ARMS:
            mu, half, _ = mean_ci([stat[(d, arm, m)][0] for d in dims])
            row[arm] = mu
            row[f"{arm}_ci"] = half
        overall[m] = row
        A(f"| {METRIC_LABEL[m]} | {_fmt_ci(row['tuned'], row['tuned_ci'])} | "
          f"{_fmt_ci(row['llm'], row['llm_ci'])} | "
          f"{_fmt_pct(pct_change(row['tuned'], row['llm']))} |")
    A("")

    chart = write_chart(stat, dims, out_dir, METRICS, CHART,
                        "rf_benchmark — real-campaign GP landscapes, ±95% CI over repeats")
    if chart:
        A(f"![Mean by dimension and arm, ±95% CI]({chart})")
        A("")

    cut_dims = [d for d in dims if d in CUT_LINES]
    if cut_dims:
        A("## Early budget — the same comparison, truncated")
        A("")
        A("The tables above spend the whole 125-line budget. This section stops each "
          "dimension early, at "
          + ", ".join(f"**{CUT_LINES[d]} lines** ({CUT_LINES[d] * POINTS_PER_LINE} "
                      f"points) for {d}d" for d in cut_dims)
          + " — the point at which that dimension is still being decided rather than "
            "already level. Nothing was rerun: both metrics are read back out of each "
            "run's `metrics_over_time.csv`, which evaluates them at every line, so "
            "these are the exact values those same runs held at the cut.")
        A("")
        A("`dist to needles` is the identical quantity as above, just earlier. "
          "**`dup fraction` is not:** `metrics_over_time.csv` measures it at one "
          "global radius, while the full-budget column scales the radius per point by "
          "the zoom zone it was sampled in, and the zoom sizes are not persisted, so "
          "the zoom-scaled version cannot be recovered at a cut. The unscaled number "
          "reads systematically higher. Compare `tuned` against `llm` within this "
          "section; do not compare it against the dup fraction above.")
        A("")
        A("| dim | cut | arm | runs | dist to needles | dup fraction (unscaled radius) |")
        A("|---|---|---|---|---|---|")
        for d in cut_dims:
            for arm in ARMS:
                dm, dh_, n = stat[(d, arm, "dist_to_needles_cut")]
                um, uh, _ = stat[(d, arm, "dup_fraction_cut")]
                A(f"| {d}d | {CUT_LINES[d]} lines | `{arm}` | {n}/{n_repeats} | "
                  f"{_fmt_ci(dm, dh_)} | {_fmt_ci(um, uh)} |")
        A("")
        A("| dim | cut | dist to needles | dup fraction (unscaled radius) |")
        A("|---|---|---|---|")
        for d in cut_dims:
            cells = [_fmt_pct(pct_change(stat[(d, "tuned", m)][0],
                                         stat[(d, "llm", m)][0])) for m in CUT_METRICS]
            A(f"| {d}d | {CUT_LINES[d]} lines | " + " | ".join(cells) + " |")
        A("")
        cut_chart = write_chart(
            stat, cut_dims, out_dir, CUT_METRICS, CHART_CUT,
            "rf_benchmark — truncated budget ("
            + ", ".join(f"{d}d: {CUT_LINES[d]}" for d in cut_dims)
            + " lines), ±95% CI over repeats",
            xlabels={d: f"{d}d\n{CUT_LINES[d]} lines" for d in cut_dims})
        if cut_chart:
            A(f"![Mean by dimension and arm at the truncated budget, ±95% CI]"
              f"({cut_chart})")
            A("")

    runs_csv, stats_csv = write_csvs(runs, stat, dims, out_dir, n_repeats)

    path = os.path.join(out_dir, "rf_benchmark_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    n_have = sum(len(v) for v in runs.values())
    print(f"  summary ({n_have}/{len(dims) * len(ARMS) * n_repeats} run(s) present) "
          f"-> {path}")
    print(f"  raw per-repeat values -> {runs_csv}")
    print(f"  cell statistics       -> {stats_csv}")
    return path


def write_csvs(runs, stat, dims, out_dir: str, n_repeats: int) -> tuple[str, str]:
    """The raw per-run ledger and the per-cell statistics.

    Two files because they answer different questions: the ledger is what a later
    significance test or box plot is re-analysed from, and survives a change of mind
    about which statistics matter; the statistics file is the summary's own tables
    in machine-readable form.
    """
    runs_path = os.path.join(out_dir, "rf_benchmark_runs.csv")
    with open(runs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "arm", "repeat", "dist_to_needles", "dup_fraction",
                    "cut_lines", "dist_to_needles_cut", "dup_fraction_cut",
                    "n_needles", "runtime_s", "run_dir"])
        for d in dims:
            for arm in ARMS:
                for r in runs[(d, arm)]:
                    w.writerow([d, arm, r["repeat"], r["dist_to_needles"],
                                r["dup_fraction"], CUT_LINES.get(d, ""),
                                r["dist_to_needles_cut"], r["dup_fraction_cut"],
                                r["n_needles"], r["runtime_s"],
                                os.path.relpath(r["run_dir"], out_dir)])

    stats_path = os.path.join(out_dir, "rf_benchmark_stats.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "dim", "arm", "metric", "n", "mean", "ci95_half",
                    "pct_change_vs_llm"])
        for d in dims:
            for m in METRICS + CUT_METRICS:
                base = stat[(d, "llm", m)][0]
                for arm in ARMS:
                    mu, half, n = stat[(d, arm, m)]
                    w.writerow(["cell", d, arm, m, n, mu, half,
                                pct_change(mu, base) if arm == "tuned" else ""])
        for m in METRICS:
            base_mu, _, _ = mean_ci([stat[(d, "llm", m)][0] for d in dims])
            for arm in ARMS:
                mu, half, n = mean_ci([stat[(d, arm, m)][0] for d in dims])
                w.writerow(["overall", "all", arm, m, n, mu, half,
                            pct_change(mu, base_mu) if arm == "tuned" else ""])
    return runs_path, stats_path


CHART = "rf_benchmark_means.png"
CHART_CUT = "rf_benchmark_means_cut.png"
ARM_COLOR = {"tuned": "#2a78d6", "llm": "#eb6834"}


def write_chart(stat, dims, out_dir: str, metrics=METRICS, fname: str = CHART,
                suptitle: str = "", xlabels: dict | None = None) -> str | None:
    """Grouped bars: both arms, both metrics, one group per dimension.

    Parameterised over the metric set so the full-budget and truncated-budget
    figures are the SAME code — two renders of one chart, which is the only way the
    reader can trust that a difference between them is the budget and not the
    drawing.

    Returns None when matplotlib is unavailable — a missing chart must never cost
    the summary, which is the part that has to be writable anywhere.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [chart] skipped: {exc}")
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.2))
    axes = axes if len(metrics) > 1 else [axes]
    for ax, m in zip(axes, metrics):
        width, top = 0.36, 0.0
        for k, arm in enumerate(ARMS):
            xs = [i + (k - 0.5) * width for i in range(len(dims))]
            mus = [stat[(d, arm, m)][0] or 0.0 for d in dims]
            errs = [stat[(d, arm, m)][1] or 0.0 for d in dims]
            ax.bar(xs, mus, width=width, color=ARM_COLOR[arm], label=arm, zorder=3,
                   yerr=errs, capsize=4,
                   error_kw={"ecolor": "#3b3a37", "elinewidth": 1.3,
                             "capthick": 1.3, "zorder": 4})
            top = max([top] + [a + b for a, b in zip(mus, errs)])
            for x, mu, err in zip(xs, mus, errs):
                ax.text(x, mu + err, f"{mu:.3f}", ha="center", va="bottom",
                        fontsize=8.5, color="#52514e")
        ax.set_title(f"{METRIC_LABEL[m]} (lower is better)", fontsize=11.5, pad=8)
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([(xlabels or {}).get(d, f"{d}d") for d in dims],
                           fontsize=10, color="#52514e")
        ax.tick_params(axis="y", labelsize=9, colors="#52514e", length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, (top or 1.0) * 1.22)
        ax.grid(axis="y", color="#e4e3e0", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c4")
        ax.legend(frameon=False, fontsize=9.5)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    return fname


# ─── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write configs, queue and SLURM script")
    p.add_argument("--out", required=True)
    p.add_argument("--dims", default=",".join(str(d) for d in DIMS))
    p.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    p.add_argument("--budget-points", type=int, default=DEFAULT_BUDGET_POINTS,
                   help="measured points per run; 0 reverts to wall-clock only")
    p.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_H,
                   help="per-run wall-clock SAFETY CAP in hours (default: %(default)s)")
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    p.add_argument("--worker-hours", type=float, default=8)
    p.add_argument("--walltime-margin", type=float, default=0.75,
                   help="hours a worker holds back before claiming one more task")
    p.add_argument("--job-name", default=None)
    p.add_argument("--no-precompute-peaks", dest="precompute_peaks",
                   action="store_false",
                   help="skip building each landscape at plan time; workers then "
                        "race to compute and cache the same reference peaks")
    p.set_defaults(precompute_peaks=True)

    for name, helptext in (("status", "queue progress"),
                           ("reset-stale", "clear claims left by killed workers"),
                           ("summary", "write the summary, CSVs and chart")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "plan":
        plan(args)
    elif args.cmd == "status":
        print_status(os.path.abspath(args.out))
    elif args.cmd == "reset-stale":
        reset_stale(os.path.abspath(args.out))
    elif args.cmd == "summary":
        write_summary(args.out)


if __name__ == "__main__":
    main()
