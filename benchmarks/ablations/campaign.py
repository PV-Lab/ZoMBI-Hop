"""
benchmarks/ablations/campaign.py
================================
Planning and draining an ablation campaign.

A campaign is a full-factorial grid — every arm x every landscape x every repeat —
written to a queue file and drained by one or more workers. Modelled on
``optimize/showdown.py``, and for the same reason: the arms have to be compared on
the *same* landscapes or a difference between them can just be one arm drawing an
easier surface. The differences from showdown are deliberate:

* **The baseline is shared.** It appears in all four ablations, so it is queued once
  and read into all four figures. Running it per-ablation would spend a quarter of
  the campaign re-measuring the same arm.
* **``metrics.json`` is the completion marker**, not a separate ``done/`` directory.
  The artifact that proves a cell finished and the record that it finished are then
  the same file, so they cannot disagree — a half-written cell has neither.
* **Repeat-major task order.** Tasks are ordered ``(repeat, landscape, arm)``, so a
  campaign cut short by wall-time has every arm on every landscape at repeat 1
  rather than all repeats of the first arm and nothing for the last. A partial
  campaign is then still a fair comparison, just a noisier one.
* **It runs locally too.** ``run`` drains the queue in-process on any machine; SLURM
  is one way to get several of those going at once, not a requirement.

Usage
-----
  # Plan a 6d campaign: 4 ablations, 5 shared landscapes, 3 repeats.
  python -m benchmarks.ablations plan --out benchmarks/ablations/runs/first \\
      --dim 6 --n-landscapes 5 --n-repeats 3 --time-limit-min 30

  # Drain it here, or submit the generated array to SLURM.
  python -m benchmarks.ablations run --out benchmarks/ablations/runs/first
  sbatch benchmarks/ablations/runs/first/ablations.sbatch

  # Progress, requeueing after killed workers, and the summary.
  python -m benchmarks.ablations status      --out benchmarks/ablations/runs/first
  python -m benchmarks.ablations reset-stale --out benchmarks/ablations/runs/first
  python -m benchmarks.ablations summarize   --out benchmarks/ablations/runs/first
"""

from __future__ import annotations

import datetime
import json
import os
import time
import traceback
from typing import Any

from ._paths import REPO_ROOT, ensure_paths

ensure_paths()

from .arms import ABLATION_KEYS, ABLATIONS, ARMS, arms_for  # noqa: E402
from .landscapes import parse_landscape_args, resolve_landscape  # noqa: E402
from .runner import (  # noqa: E402
    default_base_hparams,
    is_complete,
    run_ablation_trial,
)

MANIFEST = "manifest.json"
QUEUE = "tasks.tsv"
CLAIMS = "claims"
RUNS = "runs"
LOGS = "logs"


# ─── Layout ──────────────────────────────────────────────────────────────────────

def cell_dir(out_dir: str, arm: str, landscape_index: int, repeat: int) -> str:
    """Where one (arm, landscape, repeat) cell's artifacts live."""
    return os.path.join(out_dir, RUNS, arm, f"ls{int(landscape_index):05d}_r{int(repeat):03d}")


def load_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.isfile(path):
        raise SystemExit(f"no {MANIFEST} in {out_dir} — run `plan` first")
    with open(path) as f:
        return json.load(f)


def read_tasks(out_dir: str) -> list[tuple[str, str, int, int]]:
    """``[(tid, arm, landscape_index, repeat), …]`` from the queue file."""
    path = os.path.join(out_dir, QUEUE)
    if not os.path.isfile(path):
        raise SystemExit(f"no {QUEUE} in {out_dir} — run `plan` first")
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            tid, arm, ls, rep = line.rstrip("\n").split("\t")
            out.append((tid, arm, int(ls), int(rep)))
    return out


def factory_from_manifest(manifest: dict):
    """Rebuild the landscape factory a worker needs from the recorded plan.

    A worker gets only the output directory, so the manifest has to be a complete
    reconstruction recipe — this is the function that proves it is one.
    """
    return resolve_landscape(
        manifest["landscape_ref"],
        dim=int(manifest["dim"]),
        time_limit_hours=manifest["time_limit_hours"],
        extra=manifest.get("landscape_args") or {},
    )


# ─── Plan ────────────────────────────────────────────────────────────────────────

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

# PERSISTENT WORKER POOL — {n_workers} long-lived workers, not {n_tasks} SLURM tasks.
#
# Each worker drains the shared queue in benchmarks/ablations/campaign.py's `run`
# until it is empty or wall-time runs low, so a finished cell hands its GPU to the
# next cell inside the same allocation instead of returning it to the scheduler.
# Claiming is one atomic mkdir per cell (the only primitive that is reliably atomic
# on a shared filesystem), and a cell's metrics.json is its completion marker — so
# re-submitting this script skips finished work. Clear the claims left behind by
# killed workers first, or those cells are never retried:
#
#     python -m benchmarks.ablations reset-stale --out {out_dir}
#     sbatch {out_dir}/ablations.sbatch

cd {repo}

uv run python -m benchmarks.ablations run \\
    --out {out_dir} \\
    --worker ${{SLURM_ARRAY_TASK_ID}} \\
    --n-workers {n_workers} \\
    --worker-hours {worker_hours} \\
    --device cuda < /dev/null
"""


def plan(args) -> str:
    """Write the manifest, the queue and the SLURM array script. Returns *out_dir*."""
    out_dir = os.path.abspath(args.out)
    ablation_keys = [k.strip().upper() for k in args.ablations.split(",") if k.strip()]
    unknown = [k for k in ablation_keys if k not in ABLATIONS]
    if unknown:
        raise SystemExit(f"unknown ablation(s) {unknown}; known: {ABLATION_KEYS}")

    time_limit_hours = float(args.time_limit_min) / 60.0
    landscape_args = parse_landscape_args(args.landscape_arg)
    factory = resolve_landscape(args.landscape, dim=args.dim,
                                time_limit_hours=time_limit_hours,
                                extra=landscape_args)

    n_landscapes = int(args.n_landscapes)
    available = getattr(factory, "n_available", None)
    if available is not None and n_landscapes > available:
        raise SystemExit(
            f"--landscape {args.landscape} offers {available} distinct landscape(s) "
            f"but --n-landscapes is {n_landscapes}. Planning more would run the SAME "
            f"surface several times under different names and make the confidence "
            f"bands look tighter than the evidence supports — use "
            f"--n-landscapes {available} and raise --n-repeats instead.")

    arm_names = arms_for(ablation_keys)
    runner_overrides: dict[str, dict[str, Any]] = {}
    if args.n_restarts is not None:
        runner_overrides["k_restarts"] = {"n_restarts": int(args.n_restarts)}
    if args.activations_per_restart is not None:
        runner_overrides.setdefault("k_restarts", {})["max_activations_per_restart"] = \
            int(args.activations_per_restart)
    if args.no_fill_budget:
        runner_overrides.setdefault("k_restarts", {})["fill_budget"] = False

    base_hparams = default_base_hparams()
    if args.hparams_json:
        with open(args.hparams_json) as f:
            blob = json.load(f)
        # Accept a bare hyperparameter dict or a trial.json-style blob, matching
        # what optimize/evaluate.py and optimize/showdown.py already accept.
        base_hparams = blob.get("hparams", blob)

    for sub in (RUNS, LOGS, CLAIMS):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    n_repeats = max(1, int(args.n_repeats))
    landscape_indices = [int(args.landscape_start) + i for i in range(n_landscapes)]

    # Repeat-major, then landscape, then arm — see the module docstring on why a
    # truncated campaign must be balanced across arms rather than across repeats.
    tasks: list[tuple[str, str, int, int]] = []
    for rep in range(1, n_repeats + 1):
        for ls in landscape_indices:
            for arm in arm_names:
                tasks.append((f"{len(tasks):05d}", arm, ls, rep))

    with open(os.path.join(out_dir, QUEUE), "w") as f:
        for tid, arm, ls, rep in tasks:
            f.write(f"{tid}\t{arm}\t{ls}\t{rep}\n")

    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "ablations": [ABLATIONS[k].to_dict() for k in ablation_keys],
        "arms": {name: ARMS[name].to_dict() for name in arm_names},
        "runner_overrides": runner_overrides,
        "landscape_ref": args.landscape,
        "landscape_args": landscape_args,
        "landscape_spec": factory.spec(),
        "landscape_indices": landscape_indices,
        "dim": int(args.dim),
        "time_limit_min": float(args.time_limit_min),
        "time_limit_hours": time_limit_hours,
        "n_landscapes": n_landscapes,
        "n_repeats": n_repeats,
        "n_tasks": len(tasks),
        "seed_base": int(args.seed_base),
        "base_hparams": base_hparams,
        "hparams_source": args.hparams_json or "src/default_hparams.DEFAULT_HPARAMS",
    }
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    n_workers = max(1, int(args.n_workers))
    worker_hours = float(args.worker_hours)
    sbatch = SBATCH_TEMPLATE.format(
        job_name=args.job_name or f"ablations_{args.dim}d",
        out_dir=out_dir, repo=REPO_ROOT,
        walltime=f"{max(1, int(round(worker_hours)))}:00:00",
        last_worker=n_workers - 1, n_workers=n_workers,
        n_tasks=len(tasks), worker_hours=worker_hours,
    )
    sbatch_path = os.path.join(out_dir, "ablations.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch)
    try:
        os.chmod(sbatch_path, 0o755)
    except OSError:
        pass  # Windows: the mode bit is meaningless and chmod may refuse

    # Per-cell cost is the optimiser budget plus artifact rendering; the CoNet UMAP
    # renders on ensemble landscapes are the slow tail (run_mobo caps each at 0.5 h).
    per_cell_h = time_limit_hours + 0.05
    drain_h = len(tasks) * per_cell_h / n_workers
    print(f"\n  plan -> {out_dir}")
    print(f"    ablations: {', '.join(ablation_keys)}")
    print(f"    arms ({len(arm_names)}): {', '.join(arm_names)}")
    print(f"    {len(arm_names)} arm(s) x {n_landscapes} landscape(s) x "
          f"{n_repeats} repeat(s) = {len(tasks)} cell(s)")
    print(f"    landscape: {args.landscape} {factory.spec()}")
    print(f"    budget: {args.time_limit_min:g} min/cell  ->  ~{drain_h:.1f} h "
          f"across {n_workers} worker(s)")
    print(f"    queue -> {os.path.join(out_dir, QUEUE)}")
    print(f"    drain here:  python -m benchmarks.ablations run --out {out_dir}")
    print(f"    or submit:   sbatch {sbatch_path}")
    if drain_h > worker_hours:
        # Not an error: a worker stops claiming cleanly when a cell no longer fits,
        # and `reset-stale` + re-submit picks the campaign up where it left off.
        print(f"    NOTE: ~{drain_h:.1f} h of work per worker exceeds the "
              f"{worker_hours:g} h wall-time. Raise --n-workers to "
              f"{int(-(-len(tasks) * per_cell_h // worker_hours))}, or plan on "
              "re-submitting (after `reset-stale`) to finish the remainder.")
    return out_dir


# ─── Run (drain the queue) ───────────────────────────────────────────────────────

def _rotated(tasks: list, worker: int, n_workers: int) -> list:
    """Worker *k* starts ``k/n_workers`` of the way down the queue and wraps.

    Read top-down in lockstep, every worker would race for the same first cell and
    lose ``n_workers-1`` claim races before finding work, and the pool would grind
    through one arm before touching the next — so a campaign killed halfway would
    leave the last arms with nothing. Rotating starts each worker on a different
    part of the grid.
    """
    if n_workers <= 1 or not tasks:
        return list(tasks)
    offset = (worker * len(tasks)) // n_workers
    return tasks[offset:] + tasks[:offset]


def run(args) -> None:
    """Claim and execute cells until the queue drains or wall-time runs low."""
    out_dir = os.path.abspath(args.out)
    manifest = load_manifest(out_dir)
    tasks = read_tasks(out_dir)
    factory = factory_from_manifest(manifest)
    base_hparams = manifest["base_hparams"]
    seed_base = int(manifest.get("seed_base", 0))
    overrides = manifest.get("runner_overrides") or {}

    claims_dir = os.path.join(out_dir, CLAIMS)
    os.makedirs(claims_dir, exist_ok=True)

    per_cell_h = float(manifest["time_limit_hours"]) + float(args.cell_margin_hours)
    deadline = (time.time() + float(args.worker_hours) * 3600.0
                if args.worker_hours and args.worker_hours > 0 else None)

    queue = _rotated(tasks, int(args.worker), max(1, int(args.n_workers)))
    n_ran = n_failed = 0
    pass_no = 0

    print(f"  [worker {args.worker}] {len(queue)} cell(s) in view, "
          f"{'no wall-time limit' if deadline is None else f'{args.worker_hours:g} h wall-time'}")

    # Outer rescan: one pass can walk past a cell that was claimed at the time but
    # never finished (a transient failure, or a claim released by reset-stale while
    # this worker was mid-pass). A pass that claims nothing means the queue is
    # genuinely drained, so this terminates — but a worker never exits while
    # unclaimed work is still sitting in the file.
    while True:
        pass_no += 1
        claimed_this_pass = 0
        for tid, arm, ls, rep in queue:
            target = cell_dir(out_dir, arm, ls, rep)
            if is_complete(target):
                continue
            if deadline is not None and time.time() + per_cell_h * 3600.0 > deadline:
                print(f"  [worker {args.worker}] out of wall-time for another cell "
                      f"(ran {n_ran})")
                return
            claim = os.path.join(claims_dir, tid)
            try:
                os.mkdir(claim)   # atomic: exactly one worker wins
            except FileExistsError:
                continue
            except OSError:
                continue
            claimed_this_pass += 1

            if args.dry_run:
                print(f"  [dry-run] would run {tid}: arm={arm} ls={ls} rep={rep} -> {target}")
                os.rmdir(claim)
                continue

            try:
                run_ablation_trial(
                    arm=arm, factory=factory, landscape_index=ls, repeat=rep,
                    trial_dir=target, base_hparams=base_hparams,
                    device=args.device, seed_base=seed_base,
                    runner_kwargs=overrides.get(arm),
                )
                n_ran += 1
            except Exception:
                n_failed += 1
                print(f"  [worker {args.worker}] cell {tid} ({arm}, ls={ls}, rep={rep}) "
                      f"FAILED — claim kept; `reset-stale` requeues it")
                traceback.print_exc()
                log = os.path.join(out_dir, LOGS, f"fail_{tid}_{arm}.log")
                try:
                    with open(log, "a") as f:
                        f.write(f"\n=== {datetime.datetime.now().isoformat()} ===\n")
                        traceback.print_exc(file=f)
                except OSError:
                    pass

        if claimed_this_pass == 0:
            break
        print(f"  [worker {args.worker}] pass {pass_no} claimed {claimed_this_pass} "
              f"cell(s); rescanning")

    print(f"  [worker {args.worker}] done — ran {n_ran} cell(s), {n_failed} failed, "
          f"{pass_no} pass(es)")


# ─── Status / reset ──────────────────────────────────────────────────────────────

def status(args) -> None:
    """Per-arm progress: done, claimed-but-unfinished, pending."""
    out_dir = os.path.abspath(args.out)
    tasks = read_tasks(out_dir)
    claims_dir = os.path.join(out_dir, CLAIMS)

    by_arm: dict[str, list[int]] = {}
    for tid, arm, ls, rep in tasks:
        done = is_complete(cell_dir(out_dir, arm, ls, rep))
        claimed = os.path.isdir(os.path.join(claims_dir, tid))
        row = by_arm.setdefault(arm, [0, 0, 0])
        if done:
            row[0] += 1
        elif claimed:
            row[1] += 1
        else:
            row[2] += 1

    total = [0, 0, 0]
    print(f"  {os.path.basename(out_dir)}: {len(tasks)} cell(s)")
    print(f"    {'arm':<24} {'done':>6} {'running':>8} {'pending':>8}")
    for arm in sorted(by_arm):
        d, c, p = by_arm[arm]
        total = [total[0] + d, total[1] + c, total[2] + p]
        print(f"    {arm:<24} {d:>6} {c:>8} {p:>8}")
    print(f"    {'TOTAL':<24} {total[0]:>6} {total[1]:>8} {total[2]:>8}")


def reset_stale(args) -> None:
    """Drop claims on cells that never produced a ``metrics.json``, so they retry.

    A worker killed by wall-time, a node failure or an OOM leaves its claim behind
    with no result, and nothing else ever clears it — without this the cell is
    invisible to every later submission and the campaign looks drained while
    silently missing runs. Safe only when no workers are live: a claim on a RUNNING
    cell is indistinguishable from an abandoned one.
    """
    out_dir = os.path.abspath(args.out)
    tasks = read_tasks(out_dir)
    claims_dir = os.path.join(out_dir, CLAIMS)
    if not os.path.isdir(claims_dir):
        print(f"  no claims dir in {out_dir} — nothing to reset")
        return
    n_reset = n_done = 0
    for tid, arm, ls, rep in tasks:
        claim = os.path.join(claims_dir, tid)
        if not os.path.isdir(claim):
            continue
        if is_complete(cell_dir(out_dir, arm, ls, rep)):
            n_done += 1
            continue
        try:
            os.rmdir(claim)
            n_reset += 1
        except OSError as exc:
            print(f"  could not release claim {tid}: {exc}")
    print(f"  reset {n_reset} stale claim(s); {n_done} cell(s) already complete")
