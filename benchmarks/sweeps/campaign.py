"""
benchmarks/sweeps/campaign.py
=============================
Planning and draining the landscape sweep.

A campaign is the full-factorial grid ``n_needles x basin_width x dim`` (4 x 4 x 4
= 64 landscape configurations) times ``--n-draws`` independent placements of the
optima, written to a queue and drained by a pool of persistent workers. It is built
on the same primitives as ``benchmarks/ablations`` and ``optimize/showdown.py`` —
one atomic ``mkdir`` per claim, a per-cell artifact that doubles as the completion
marker — with three differences the sweep needs:

* **The grid varies the dimension**, so ``dim`` lives in the queue row rather than
  the manifest, and the landscape factory and the hyperparameters are resolved per
  cell instead of once per campaign.
* **The budget is measured in lines, not wall-clock** (see
  :mod:`benchmarks.sweeps.budget`), so every cell gets the same 3000 experiments
  whatever the dimension does to the cost of an iteration.
* **It heals itself across restarts.** A worker heartbeats its claim; a claim that
  has stopped beating for ``--reclaim-after-min`` is released automatically by the
  next worker that walks past it. That is what lets the generated sbatch resubmit
  itself indefinitely without anyone running ``reset-stale`` between submissions —
  the requirement that a campaign of this length runs unattended.

Draw-major order
----------------
Tasks are ordered ``(draw, dim, n_needles, basin_width)``, so a campaign cut short
has every one of the 64 grid configurations at draw 1 rather than all five draws of
the first few configurations and nothing for the rest. A partial sweep is then
still a complete picture of the grid, just a noisier one.

Usage
-----
  python -m benchmarks.sweeps plan --out benchmarks/sweeps/runs/first --n-draws 5
  sbatch benchmarks/sweeps/runs/first/sweep.sbatch

  python -m benchmarks.sweeps status    --out benchmarks/sweeps/runs/first
  python -m benchmarks.sweeps summarize --out benchmarks/sweeps/runs/first
"""

from __future__ import annotations

import datetime
import itertools
import json
import os
import shutil
import threading
import time
import traceback

from ._paths import REPO_ROOT, ensure_paths

ensure_paths()

from benchmarks.ablations.runner import (  # noqa: E402
    is_complete as _run_complete,
    run_ablation_trial,
)

from . import needles as nd  # noqa: E402
from .budget import BudgetState, line_budget  # noqa: E402
from .hparams import parse_hparam_overrides, resolve_all  # noqa: E402

MANIFEST = "manifest.json"
QUEUE = "tasks.tsv"
CLAIMS = "claims"
RUNS = "runs"
LOGS = "logs"
CELL_FILE = "sweep_cell.json"
HEARTBEAT = "heartbeat"

#: How often a running worker touches its claim's heartbeat file.
HEARTBEAT_EVERY_S = 60.0


# ─── Layout ──────────────────────────────────────────────────────────────────────

def cell_name(dim: int, n_needles: int, basin_width: float) -> str:
    """Directory-safe name for one grid configuration."""
    return f"d{int(dim):02d}_n{int(n_needles):02d}_b{float(basin_width):g}"


def cell_dir(out_dir: str, name: str, draw: int) -> str:
    return os.path.join(out_dir, RUNS, name, f"draw{int(draw):03d}")


def load_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.isfile(path):
        raise SystemExit(f"no {MANIFEST} in {out_dir} — run `plan` first")
    with open(path) as f:
        return json.load(f)


def read_tasks(out_dir: str) -> list[dict]:
    """The queue, one dict per cell."""
    path = os.path.join(out_dir, QUEUE)
    if not os.path.isfile(path):
        raise SystemExit(f"no {QUEUE} in {out_dir} — run `plan` first")
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            tid, name, dim, n, b, draw = line.rstrip("\n").split("\t")
            out.append({"tid": tid, "name": name, "dim": int(dim),
                        "n_needles": int(n), "basin_width": float(b),
                        "draw": int(draw)})
    return out


def is_complete(target: str) -> bool:
    """A cell is done when it has BOTH the run's metrics and this sweep's record.

    ``metrics.json`` is written by the shared runner and proves the optimiser
    finished; :data:`CELL_FILE` is written after it and proves the budget and
    landscape bookkeeping landed too. Requiring both means a cell interrupted
    between the two is re-run rather than counted with half its record.
    """
    return _run_complete(target) and os.path.isfile(os.path.join(target, CELL_FILE))


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
#SBATCH --signal=B:USR1@300

# SELF-RESTARTING PERSISTENT WORKER POOL -- {n_workers} workers, {n_tasks} cells.
#
# Each array element is one long-lived worker that claims cell after cell off
# tasks.tsv until the queue drains, so a finished cell hands its GPU to the next
# one inside the same allocation instead of going back to the scheduler.
#
# The pool RESTARTS ITSELF, which is what lets a multi-day campaign run unattended:
#
#   * The worker stops claiming with less than one cell's budget of wall-time left
#     and exits cleanly; the tail of this script then resubmits THIS array index if
#     the queue still has work. A clean exit with an empty queue submits nothing,
#     so the chain ends on its own when the campaign finishes.
#   * If wall-time arrives anyway (a cell overran its estimate), SLURM sends USR1
#     300 s early and the trap below resubmits and exits before the kill.
#   * Claims are HEARTBEATED. A worker killed outright -- node failure, OOM, a
#     SIGKILL past the grace period -- leaves a claim whose heartbeat stops, and the
#     next worker to walk past it releases it automatically after
#     {reclaim_after_min:g} minutes. Nothing has to be reset by hand between
#     submissions.
#
# Stop the chain with `scancel`. To drain it by hand instead:
#     python -m benchmarks.sweeps run --out {out_dir} --device cuda

cd {repo}

RESUBMITTED=0
resubmit_if_work_remains() {{
    if [ "$RESUBMITTED" -eq 1 ]; then return; fi
    PENDING=$(uv run python -m benchmarks.sweeps status --out {out_dir} --pending-count 2>/dev/null | tail -1)
    case "$PENDING" in
        ''|*[!0-9]*) echo "[$(date)] could not read pending count ('$PENDING'); NOT resubmitting"; return ;;
    esac
    if [ "$PENDING" -gt 0 ]; then
        echo "[$(date)] $PENDING cell(s) still pending; resubmitting worker $SLURM_ARRAY_TASK_ID"
        sbatch --array="$SLURM_ARRAY_TASK_ID" "$0"
        RESUBMITTED=1
    else
        echo "[$(date)] queue drained; chain ends here"
    fi
}}

on_time_limit() {{
    echo "[$(date)] wall-time approaching on worker $SLURM_ARRAY_TASK_ID"
    resubmit_if_work_remains
    exit 0
}}
trap on_time_limit USR1

uv run python -m benchmarks.sweeps run \\
    --out {out_dir} \\
    --worker "$SLURM_ARRAY_TASK_ID" \\
    --n-workers {n_workers} \\
    --worker-hours {worker_hours} \\
    --reclaim-after-min {reclaim_after_min} \\
    --device cuda < /dev/null &
wait $!
rc=$?

# Reached only on a clean exit (the USR1 trap exits before this). A crashed worker
# resubmits too: the cell it died on keeps its claim only until the heartbeat goes
# stale, and one bad cell should not end a campaign of {n_tasks}.
echo "[$(date)] worker $SLURM_ARRAY_TASK_ID exited rc=$rc"
resubmit_if_work_remains
"""


def plan(args) -> str:
    """Write the manifest, the queue and the self-restarting SLURM array script."""
    out_dir = os.path.abspath(args.out)
    dims = [int(v) for v in args.dims.split(",") if v.strip()]
    counts = [int(v) for v in args.n_needles.split(",") if v.strip()]
    widths = [float(v) for v in args.basin_widths.split(",") if v.strip()]
    n_draws = max(1, int(args.n_draws))

    # Resolved first: a missing or unreadable hyperparameter file must stop the plan
    # here, not on a worker three hours in.
    hp_map = resolve_all(dims, parse_hparam_overrides(args.hparams))

    for sub in (RUNS, LOGS, CLAIMS):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # Draw-major: the whole grid at draw 1 before any of draw 2 (see the docstring).
    tasks = []
    for draw in range(1, n_draws + 1):
        for dim, n, b in itertools.product(dims, counts, widths):
            tasks.append({"tid": f"{len(tasks):05d}", "name": cell_name(dim, n, b),
                          "dim": dim, "n_needles": n, "basin_width": b, "draw": draw})
    with open(os.path.join(out_dir, QUEUE), "w") as f:
        for t in tasks:
            f.write(f"{t['tid']}\t{t['name']}\t{t['dim']}\t{t['n_needles']}\t"
                    f"{t['basin_width']:g}\t{t['draw']}\n")

    feasibility = nd.plan_feasibility(dims, counts, widths)
    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "grid": {"dims": dims, "n_needles": counts, "basin_widths": widths},
        "n_draws": n_draws,
        "n_configurations": len(dims) * len(counts) * len(widths),
        "n_tasks": len(tasks),
        "n_lines": int(args.n_lines),
        "points_per_line": int(_points_per_line()),
        "points_budget": int(args.n_lines) * int(_points_per_line()),
        "cell_max_hours": float(args.cell_max_hours),
        "seed_base": int(args.seed_base),
        "hparams": hp_map,
        "landscape": {
            "kind": "needles",
            "description": ("bumps-only Ensemble: n negated-Ackley optima of "
                            "sharpness b on a flat plain, every other feature off"),
            "sigma_x": float(nd.SIGMA_X),
            "sigma_y_at_peak": round(float(nd.sigma_y_at_peak()), 6),
            "plain_y": nd.PLAIN_Y, "peak_y": nd.PEAK_Y,
        },
        "feasibility": feasibility,
        "reclaim_after_min": float(args.reclaim_after_min),
    }
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    n_workers = max(1, int(args.n_workers))
    worker_hours = float(args.worker_hours)
    sbatch = SBATCH_TEMPLATE.format(
        job_name=args.job_name or "zh_sweep",
        out_dir=out_dir, repo=REPO_ROOT,
        walltime=f"{max(1, int(round(args.walltime_hours)))}:00:00",
        last_worker=n_workers - 1, n_workers=n_workers, n_tasks=len(tasks),
        worker_hours=worker_hours,
        reclaim_after_min=float(args.reclaim_after_min),
    )
    sbatch_path = os.path.join(out_dir, "sweep.sbatch")
    # Explicit UTF-8 and an ASCII-only template: `plan` may be run from a Windows
    # checkout, where the default encoding is cp1252 and any non-ASCII in the
    # script would be written as mojibake for bash to choke on.
    # LF line endings explicitly: a CRLF sbatch written from a Windows checkout is
    # rejected by bash on the cluster ("\r: command not found").
    with open(sbatch_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sbatch)
    try:
        os.chmod(sbatch_path, 0o755)
    except OSError:
        pass  # Windows: the mode bit is meaningless and chmod may refuse

    print(f"\n  plan -> {out_dir}")
    print(f"    grid: dims {dims} x needles {counts} x basin widths {widths} "
          f"= {manifest['n_configurations']} configuration(s)")
    print(f"    x {n_draws} draw(s) = {len(tasks)} cell(s)")
    print(f"    budget: {args.n_lines} lines x {_points_per_line()} points "
          f"= {manifest['points_budget']} measured compositions per cell "
          f"(wall-clock ceiling {args.cell_max_hours:g} h)")
    for dim in dims:
        rec = hp_map[dim]
        flag = "  [STAND-IN]" if rec["is_stand_in"] else ""
        print(f"    dim {dim:>2}: {rec['path']}{flag}")
    tight = [r for r in feasibility if not r["feasible"]]
    if tight:
        print(f"    NOTE: {len(tight)} cell(s) sit above the optimistic packing "
              "bound; placement falls back to the input-noise floor and records it:")
        for r in tight:
            print(f"           dim {r['dim']} / n {r['n_needles']} / b "
                  f"{r['basin_width']:g}: wants s>={r['separation_target']:.4f}, "
                  f"~{r['capacity_estimate']:g} fit")
    print(f"    queue -> {os.path.join(out_dir, QUEUE)}")
    print(f"    {n_workers} self-restarting worker(s) @ {args.walltime_hours:g} h")
    print(f"    submit:      sbatch {sbatch_path}")
    print(f"    drain here:  python -m benchmarks.sweeps run --out {out_dir}")
    return out_dir


def _points_per_line() -> int:
    import run_mobo as rm

    return int(rm.NUM_EXPERIMENTS)


# ─── Claims: atomic, heartbeated, self-releasing ─────────────────────────────────

def _claim_path(out_dir: str, tid: str) -> str:
    return os.path.join(out_dir, CLAIMS, tid)


def _claim_age_s(claim: str) -> float:
    """Seconds since this claim last showed a sign of life.

    The heartbeat file if there is one, else the claim directory's own mtime — a
    claim made microseconds ago has not written its first beat yet, and treating
    that as infinitely stale would let a second worker steal a cell that is fine.
    """
    for candidate in (os.path.join(claim, HEARTBEAT), claim):
        try:
            return max(0.0, time.time() - os.path.getmtime(candidate))
        except OSError:
            continue
    return 0.0


def _release_stale(out_dir: str, tasks: list[dict], max_age_s: float) -> int:
    """Release claims that have stopped beating and produced no result.

    This is what makes the campaign survive restarts unattended. It is safe to run
    while other workers are live *because* it keys on the heartbeat: a running
    worker touches its claim once a minute (:data:`HEARTBEAT_EVERY_S`), so a claim
    silent for tens of minutes is not one somebody is working on. Two workers
    racing to release the same claim is harmless — the ``mkdir`` that follows is
    still atomic and exactly one of them ends up owning the cell.
    """
    n = 0
    for t in tasks:
        claim = _claim_path(out_dir, t["tid"])
        if not os.path.isdir(claim):
            continue
        if is_complete(cell_dir(out_dir, t["name"], t["draw"])):
            continue
        if _claim_age_s(claim) < max_age_s:
            continue
        try:
            shutil.rmtree(claim)
            n += 1
        except OSError:
            pass  # another worker got there first
    return n


class _Heartbeat:
    """Touch a claim's heartbeat file on a daemon thread while a cell runs.

    A cell can take an hour inside one blocking call, so the beat cannot be driven
    from the run loop. The thread is a daemon and the flag is checked every second,
    so it never holds up interpreter shutdown or the next cell.
    """

    def __init__(self, claim: str) -> None:
        self._path = os.path.join(claim, HEARTBEAT)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Heartbeat":
        self._beat()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _beat(self) -> None:
        try:
            with open(self._path, "w") as f:
                f.write(f"{time.time():.0f}\n")
        except OSError:
            pass  # a filesystem hiccup must not take the cell down

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_EVERY_S):
            self._beat()


# ─── Run (drain the queue) ───────────────────────────────────────────────────────

def _rotated(tasks: list, worker: int, n_workers: int) -> list:
    """Worker *k* starts ``k/n_workers`` of the way down the queue and wraps.

    Read top-down in lockstep every worker would race for the same first cell and
    lose ``n_workers-1`` claim races before finding work, and the pool would grind
    through one region of the grid at a time. Rotating starts each worker in a
    different part of the grid.
    """
    if n_workers <= 1 or not tasks:
        return list(tasks)
    offset = (worker * len(tasks)) // n_workers
    return tasks[offset:] + tasks[:offset]


def run_one_cell(task: dict, out_dir: str, manifest: dict, target: str,
                 device: str | None, verbose: bool = True) -> dict:
    """Run a single grid cell under the line budget, and write its sweep record.

    Routes through ``benchmarks.ablations.runner.run_ablation_trial`` on the
    unmodified baseline arm, so the cell gets the same artifact set as a MOBO trial
    or an ablation cell (points.csv, needles.csv, metrics_over_time.csv, the plots,
    the CoNet renders, ``metrics.json``) and drops into the existing tooling
    unchanged. This function adds only what the ablations harness has no concept
    of: the line budget wrapped around the call, and the landscape/budget record
    written after it.
    """
    factory = nd.NeedleFactory(
        dim=task["dim"], n_needles=task["n_needles"],
        basin_width=task["basin_width"], seed=int(manifest.get("seed_base", 0)),
        time_limit_hours=float(manifest["cell_max_hours"]),
    )
    hparams = manifest["hparams"][str(task["dim"])]["hparams"]

    # The landscape is built once here so its verified record (exact optima count,
    # measured prominence, achieved separation) can be written even though
    # ``run_ablation_trial`` builds its own copy from the same seed. Both calls go
    # through ``place_optima`` with the same seed, so they are the same landscape;
    # this one exists to be *checked*, and build_landscape raises if the count is
    # ever not exactly n.
    seed = factory.placement_seed(task["draw"])
    built = nd.build_landscape(task["dim"], task["n_needles"],
                               task["basin_width"], seed)

    state = BudgetState(n_lines=int(manifest["n_lines"]),
                        n_init_lines=_n_init_lines(),
                        points_per_line=_points_per_line())
    t0 = time.time()
    with line_budget(state=state):
        metrics = run_ablation_trial(
            arm="zombi_hop", factory=factory,
            landscape_index=task["draw"], repeat=1,
            trial_dir=target, base_hparams=hparams, device=device,
            # Distinct per grid configuration, so two cells at the same draw index
            # do not start from correlated initial designs.
            seed_base=_cell_seed_base(task, int(manifest.get("seed_base", 0))),
            verbose=verbose,
        )

    record = {
        "tid": task["tid"], "cell": task["name"], "draw": task["draw"],
        "dim": task["dim"], "n_needles": task["n_needles"],
        "basin_width": task["basin_width"],
        "hparams_source": manifest["hparams"][str(task["dim"])]["path"],
        "hparams_is_stand_in": manifest["hparams"][str(task["dim"])]["is_stand_in"],
        "landscape": built["record"],
        "budget": state.to_dict(),
        "metrics": metrics,
        "wall_s": round(time.time() - t0, 3),
    }
    with open(os.path.join(target, CELL_FILE), "w") as f:
        json.dump(record, f, indent=2, default=str)
    if verbose:
        b = state.to_dict()
        print(f"  [cell] {task['name']} draw {task['draw']}: "
              f"dist={metrics['dist_to_needles']:.4f} "
              f"needles={metrics['n_needles']}/{task['n_needles']} "
              f"points={metrics['n_points']}/{b['points_budget']} "
              f"budget_hit={b['budget_hit']}", flush=True)
    return record


def _n_init_lines() -> int:
    import run_mobo as rm

    return int(rm.N_INIT_LINES)


def _cell_seed_base(task: dict, base: int) -> int:
    """Seed offset unique to a grid configuration (the draw is mixed in downstream)."""
    h = (int(base) * 1_000_003
         ^ int(task["dim"]) * 2_654_435_761
         ^ int(task["n_needles"]) * 40_503
         ^ int(round(float(task["basin_width"]) * 10)) * 97_499)
    return int(abs(h) % (2 ** 31 - 1))


def run(args) -> None:
    """Claim and execute cells until the queue drains or wall-time runs low."""
    out_dir = os.path.abspath(args.out)
    manifest = load_manifest(out_dir)
    tasks = read_tasks(out_dir)
    os.makedirs(os.path.join(out_dir, CLAIMS), exist_ok=True)

    reclaim_after_s = float(args.reclaim_after_min) * 60.0
    # A cell's worst case is the wall-clock ceiling plus artifact rendering (the
    # CoNet UMAP renders are the slow tail; run_mobo caps each at 0.5 h). A worker
    # stops claiming when less than this is left, so it exits cleanly instead of
    # being killed mid-cell.
    per_cell_h = float(manifest["cell_max_hours"]) + float(args.cell_margin_hours)
    deadline = (time.time() + float(args.worker_hours) * 3600.0
                if args.worker_hours and args.worker_hours > 0 else None)

    queue = _rotated(tasks, int(args.worker), max(1, int(args.n_workers)))
    n_ran = n_failed = pass_no = 0
    print(f"  [worker {args.worker}] {len(queue)} cell(s) in view; "
          + ("no wall-time limit" if deadline is None
             else f"{args.worker_hours:g} h wall-time, stops claiming with "
                  f"{per_cell_h:.2f} h left"))

    while True:
        pass_no += 1
        # Release abandoned claims before every pass, so a worker restarted after a
        # node failure picks up the cells that died with it without anyone running
        # reset-stale first. See _release_stale on why this is safe while live.
        released = _release_stale(out_dir, tasks, reclaim_after_s)
        if released:
            print(f"  [worker {args.worker}] released {released} claim(s) with no "
                  f"heartbeat for {args.reclaim_after_min:g} min")

        claimed_this_pass = 0
        for task in queue:
            target = cell_dir(out_dir, task["name"], task["draw"])
            if is_complete(target):
                continue
            if deadline is not None and time.time() + per_cell_h * 3600.0 > deadline:
                print(f"  [worker {args.worker}] out of wall-time for another cell "
                      f"(ran {n_ran}) — exiting cleanly so the job can resubmit")
                return
            claim = _claim_path(out_dir, task["tid"])
            try:
                os.mkdir(claim)   # atomic: exactly one worker wins
            except OSError:
                continue
            claimed_this_pass += 1

            if args.dry_run:
                print(f"  [dry-run] {task['tid']}: {task['name']} draw {task['draw']}")
                shutil.rmtree(claim, ignore_errors=True)
                continue

            try:
                with _Heartbeat(claim):
                    run_one_cell(task, out_dir, manifest, target, args.device)
                n_ran += 1
            except Exception:
                n_failed += 1
                print(f"  [worker {args.worker}] cell {task['tid']} "
                      f"({task['name']} draw {task['draw']}) FAILED", flush=True)
                traceback.print_exc()
                log = os.path.join(out_dir, LOGS, f"fail_{task['tid']}.log")
                try:
                    with open(log, "a") as f:
                        f.write(f"\n=== {datetime.datetime.now().isoformat()} ===\n")
                        traceback.print_exc(file=f)
                except OSError:
                    pass
                # The claim is left in place but stops beating, so it is retried
                # after reclaim_after_min rather than immediately — a cell that
                # fails deterministically must not spin the pool.

        if claimed_this_pass == 0:
            break
        print(f"  [worker {args.worker}] pass {pass_no} claimed "
              f"{claimed_this_pass} cell(s); rescanning")

    print(f"  [worker {args.worker}] done — ran {n_ran} cell(s), {n_failed} failed, "
          f"{pass_no} pass(es)")


# ─── Status / reset ──────────────────────────────────────────────────────────────

def _counts(out_dir: str, tasks: list[dict], reclaim_after_s: float):
    done = running = stale = pending = 0
    for t in tasks:
        if is_complete(cell_dir(out_dir, t["name"], t["draw"])):
            done += 1
            continue
        claim = _claim_path(out_dir, t["tid"])
        if os.path.isdir(claim):
            if _claim_age_s(claim) >= reclaim_after_s:
                stale += 1
            else:
                running += 1
        else:
            pending += 1
    return done, running, stale, pending


def status(args) -> None:
    """Progress by dimension, plus the count the sbatch chain reads to decide."""
    out_dir = os.path.abspath(args.out)
    manifest = load_manifest(out_dir)
    tasks = read_tasks(out_dir)
    reclaim_after_s = float(manifest.get("reclaim_after_min", 30.0)) * 60.0

    if args.pending_count:
        # Machine-readable, and the last line of stdout: the generated sbatch
        # parses this to decide whether to resubmit itself. A stale claim counts as
        # outstanding work, because it will be released and re-run.
        _, _, stale, pending = _counts(out_dir, tasks, reclaim_after_s)
        print(stale + pending)
        return

    by_dim: dict[int, list[int]] = {}
    for t in tasks:
        row = by_dim.setdefault(t["dim"], [0, 0, 0, 0])
        if is_complete(cell_dir(out_dir, t["name"], t["draw"])):
            row[0] += 1
            continue
        claim = _claim_path(out_dir, t["tid"])
        if os.path.isdir(claim):
            row[2 if _claim_age_s(claim) >= reclaim_after_s else 1] += 1
        else:
            row[3] += 1

    print(f"  {os.path.basename(out_dir)}: {len(tasks)} cell(s)")
    print(f"    {'dim':<8} {'done':>6} {'running':>8} {'stale':>7} {'pending':>8}")
    total = [0, 0, 0, 0]
    for dim in sorted(by_dim):
        r = by_dim[dim]
        total = [a + b for a, b in zip(total, r)]
        print(f"    dim {dim:<4} {r[0]:>6} {r[1]:>8} {r[2]:>7} {r[3]:>8}")
    print(f"    {'TOTAL':<8} {total[0]:>6} {total[1]:>8} {total[2]:>7} {total[3]:>8}")
    if total[2]:
        print(f"    ({total[2]} stale claim(s) will be released automatically by the "
              "next worker)")


def reset_stale(args) -> None:
    """Release stale claims now, rather than waiting for a worker to do it.

    Rarely needed — workers release stale claims at the top of every pass, which is
    the whole point of the heartbeat. Kept for the case where you have shortened
    the campaign's patience and want the queue re-opened immediately, and for
    ``--all``, which releases every unfinished claim regardless of heartbeat and is
    therefore only safe when no workers are live.
    """
    out_dir = os.path.abspath(args.out)
    manifest = load_manifest(out_dir)
    tasks = read_tasks(out_dir)
    max_age = 0.0 if args.all else float(
        args.reclaim_after_min
        if args.reclaim_after_min is not None
        else manifest.get("reclaim_after_min", 30.0)) * 60.0
    if args.all:
        print("  --all: releasing every unfinished claim, heartbeat or not. This is "
              "only safe with no workers running.")
    n = _release_stale(out_dir, tasks, max_age)
    done, running, stale, pending = _counts(
        out_dir, tasks, float(manifest.get("reclaim_after_min", 30.0)) * 60.0)
    print(f"  released {n} claim(s); {done} done, {running} running, "
          f"{stale} stale, {pending} pending")
