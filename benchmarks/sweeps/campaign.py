"""
benchmarks/sweeps/campaign.py
=============================
Planning and draining a method x landscape sweep.

A campaign is ``methods x dims x n_needles x basin_widths x draws``: every method
runs on every landscape, and for a given ``(dim, n, b, draw)`` every method gets the
byte-identical landscape and the same measurement-noise seed, so method-vs-method
differences are paired. Cells go into a queue drained by a pool of persistent,
self-restarting SLURM workers, on the same primitives as ``benchmarks/ablations``
and ``optimize/showdown.py`` — one atomic ``mkdir`` per claim, a per-cell artifact
as the completion marker — plus:

* **Heartbeated claims.** A worker touches its claim once a minute; a claim silent
  for ``--reclaim-after-min`` is released by the next worker that walks past it, so
  the pool survives node failures without anyone running ``reset-stale``.
* **One process per cell.** A worker runs each cell as
  ``python -m benchmarks.sweeps cell ...`` in a child process, under a hard timeout.
  Importing ZoMBI-Hop switches torch's global default device and dtype, HEBO puts a
  vendored directory on ``sys.path``, and a GP that exhausts GPU memory can take the
  process down: none of that may leak into the next method's cell. Process startup
  costs seconds against a cell of minutes to hours.
* **Bounded retries.** A cell whose child fails ``--max-attempts`` times is marked
  FAILED (its claim holds a ``FAILED`` file) and is no longer retried or counted as
  pending, so one deterministic crash cannot keep the pool resubmitting forever.
  ``reset-stale --failed`` re-opens them.
* **A point budget, enforced identically.** Every cell measures exactly
  ``--budget-per-dim`` x dim points (default 100 x dim: 200 at 2d, 900 at 9d), or a
  flat ``--budget`` if one is given, in batches of ``--batch-size`` (1), through the
  shared :class:`benchmarks.methods.Problem`. Every method gets the same budget at a
  given dim. ``--cell-max-hours`` is only a safety ceiling, and a cell stopped by it
  is recorded ``budget_hit: false``.

Queue order is draw-major, then landscape, then method: a campaign cut short has
every configuration at draw 1, and every method on each landscape it reached.
"""

from __future__ import annotations

import datetime
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

from ._paths import REPO_ROOT, ensure_paths

ensure_paths()

from benchmarks.methods import make_extractor, make_method  # noqa: E402
from benchmarks.methods.base import Problem  # noqa: E402
from benchmarks.methods.runner import (  # noqa: E402
    GroundTruth,
    atomic_write_json,
    run_method,
)
from benchmarks.methods.runner import is_complete as _run_complete  # noqa: E402

from . import needles as nd  # noqa: E402
from .configs import (  # noqa: E402
    method_names,
    parse_method_overrides,
    resolve_method_configs,
)
from .hparams import parse_hparam_overrides  # noqa: E402

SCHEMA_VERSION = 2
MANIFEST = "manifest.json"
QUEUE = "tasks.tsv"
CLAIMS = "claims"
RUNS = "runs"
LOGS = "logs"
CELL_FILE = "sweep_cell.json"
HEARTBEAT = "heartbeat"
FAILED = "FAILED"

#: How often a running worker touches its claim's heartbeat file.
HEARTBEAT_EVERY_S = 60.0

DEFAULT_METHODS = ("zombi_hop", "random", "gp_bo", "turbo", "hebo")
#: Measured points per cell = this x dim, unless ``plan --budget`` fixes one number.
DEFAULT_BUDGET_PER_DIM = 100
#: One point per call for every method (POINTWISE.md). Was 24: one LineBO line.
DEFAULT_BATCH = 1


# ─── Layout ──────────────────────────────────────────────────────────────────────

def cell_name(dim: int, n_needles: int, basin_width: float) -> str:
    """Directory-safe name for one landscape configuration."""
    return f"d{int(dim):02d}_n{int(n_needles):02d}_b{float(basin_width):g}"


def cell_dir(out_dir: str, method: str, name: str, draw: int) -> str:
    return os.path.join(out_dir, RUNS, method, name, f"draw{int(draw):03d}")


def task_dir(out_dir: str, task: dict) -> str:
    return cell_dir(out_dir, task["method"], task["name"], task["draw"])


def load_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.isfile(path):
        raise SystemExit(f"no {MANIFEST} in {out_dir} — run `plan` first")
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(
            f"{out_dir} was planned by the pre-2026-09 single-method SIMPLEX sweep "
            f"(schema {manifest.get('schema_version', 1)}); this code runs the "
            f"multi-method CUBE sweep (schema {SCHEMA_VERSION}). Its summary/ is "
            "already on disk; to re-run or re-summarise it, check out commit 285424f.")
    return manifest


def cell_budget(manifest: dict, dim: int) -> int:
    """Measured points a cell at ``dim`` gets. Manifests planned before per-dim
    budgets carry only the flat ``budget``."""
    per_dim = manifest.get("budgets")
    return int(per_dim[str(dim)]) if per_dim else int(manifest["budget"])


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
            tid, method, name, dim, n, b, draw = line.rstrip("\n").split("\t")
            out.append({"tid": tid, "method": method, "name": name, "dim": int(dim),
                        "n_needles": int(n), "basin_width": float(b), "draw": int(draw)})
    return out


def is_complete(target: str) -> bool:
    """Done = the runner's ``metrics.json`` AND this sweep's ``sweep_cell.json``.

    The sweep record is written after the metrics, so a cell interrupted between
    the two is re-run rather than counted with half its record.
    """
    return _run_complete(target) and os.path.isfile(os.path.join(target, CELL_FILE))


# ─── Seeds ───────────────────────────────────────────────────────────────────────

def cell_seed(seed_base: int, task: dict) -> int:
    """Seed for a cell's measurement noise and the method's own RNG.

    Excludes the method on purpose: every method on a landscape draws from the same
    noise stream (common random numbers), so a paired difference between two
    methods is not inflated by one of them drawing kinder noise.
    """
    h = (int(seed_base) * 7_919
         ^ int(task["dim"]) * 104_729
         ^ int(task["n_needles"]) * 1_299_709
         ^ int(round(float(task["basin_width"]) * 10)) * 15_485_863
         ^ int(task["draw"]) * 2_654_435_761)
    return int(abs(h) % (2 ** 31 - 1))


def landscape_seed(seed_base: int, task: dict) -> int:
    return nd.placement_seed(seed_base, task["dim"], task["n_needles"],
                             task["basin_width"], task["draw"])


# ─── Plan ────────────────────────────────────────────────────────────────────────

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={out_dir}/logs/%x_%A_%a.out
#SBATCH --error={out_dir}/logs/%x_%A_%a.err
#SBATCH --time={walltime}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --partition={partition}
{gres_line}#SBATCH --array=0-{last_worker}
#SBATCH --signal=B:USR1@300

# SELF-RESTARTING PERSISTENT WORKER POOL -- {n_workers} workers, {n_tasks} cells
# ({methods}).
#
# Each array element is one long-lived worker that claims cell after cell off
# tasks.tsv until the queue drains; each cell runs in its own child process.
#
# The pool RESTARTS ITSELF:
#   * A worker stops claiming with less than one cell's ceiling of wall-time left
#     and exits cleanly; the tail of this script resubmits THIS array index if the
#     queue still has work, and submits nothing once it does not.
#   * If wall-time arrives anyway, SLURM sends USR1 300 s early and the trap below
#     resubmits and exits before the kill.
#   * Claims are HEARTBEATED; a claim silent for {reclaim_after_min:g} minutes is
#     released by the next worker. A cell that fails {max_attempts} times is marked
#     FAILED and left alone (`status` lists it; `reset-stale --failed` retries).
#
# Stop the chain with `scancel`. To drain it by hand instead:
#     python -m benchmarks.sweeps run --out {out_dir}

cd {repo}
export OMP_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}

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
    --reclaim-after-min {reclaim_after_min}{device_flag} < /dev/null &
wait $!
rc=$?

echo "[$(date)] worker $SLURM_ARRAY_TASK_ID exited rc=$rc"
resubmit_if_work_remains
"""


def _csv(raw: str, cast):
    return [cast(v) for v in raw.split(",") if v.strip()]


def _parse_kv(pairs: list[str] | None) -> dict:
    out = {}
    for raw in pairs or []:
        k, sep, v = raw.partition("=")
        if not sep:
            raise ValueError(f"{raw!r} is not key=value")
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def plan(args) -> str:
    """Write the manifest, the queue and the self-restarting SLURM array script."""
    out_dir = os.path.abspath(args.out)
    if os.path.isfile(os.path.join(out_dir, QUEUE)) and not args.force:
        raise SystemExit(
            f"{out_dir} already holds a planned campaign. Re-planning rewrites the "
            "queue under any claims and finished cells it has; plan into a new "
            "directory, or pass --force if that is really what you want.")
    dims = _csv(args.dims, int)
    counts = _csv(args.n_needles, int)
    widths = _csv(args.basin_widths, float)
    method_refs = method_names(_csv(args.methods, str))   # unknown names fail here
    methods = list(method_refs)
    n_draws = max(1, int(args.n_draws))
    if args.budget is not None:
        budgets = {str(d): int(args.budget) for d in dims}
    else:
        budgets = {str(d): int(args.budget_per_dim) * d for d in dims}

    # Resolved up front: a bad config key, a missing hyperparameter file or an
    # uninstalled HEBO must stop the plan here, not a worker hours in.
    configs = resolve_method_configs(
        list(method_refs.values()), dims,
        zombi_hparam_files=parse_hparam_overrides(args.hparams),
        overrides=parse_method_overrides(args.method_config, args.method_set))
    if "hebo" in methods:
        from benchmarks.methods.hebo_method import _import_hebo

        _import_hebo()
    extractor_kwargs = _parse_kv(args.extractor_arg)
    extractor_cfg = make_extractor(args.extractor, noise_frac=float(args.output_noise_frac),
                                   **extractor_kwargs).config()

    for sub in (RUNS, LOGS, CLAIMS):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    tasks = []
    for draw in range(1, n_draws + 1):
        for dim, n, b in itertools.product(dims, counts, widths):
            for method in methods:
                tasks.append({"tid": f"{len(tasks):06d}", "method": method,
                              "name": cell_name(dim, n, b), "dim": dim,
                              "n_needles": n, "basin_width": b, "draw": draw})
    with open(os.path.join(out_dir, QUEUE), "w") as f:
        for t in tasks:
            f.write(f"{t['tid']}\t{t['method']}\t{t['name']}\t{t['dim']}\t"
                    f"{t['n_needles']}\t{t['basin_width']:g}\t{t['draw']}\n")

    feasibility = nd.plan_feasibility(dims, counts, widths)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "domain": nd.DOMAIN,
        "methods": methods,
        "method_refs": method_refs,
        "grid": {"dims": dims, "n_needles": counts, "basin_widths": widths},
        "n_draws": n_draws,
        "n_configurations": len(dims) * len(counts) * len(widths),
        "n_tasks": len(tasks),
        # Flat budget if --budget was given, else None; the per-dim numbers are in
        # "budgets" either way (read them with cell_budget).
        "budget": None if args.budget is None else int(args.budget),
        "budget_per_dim": None if args.budget is not None else int(args.budget_per_dim),
        "budgets": budgets,
        "batch_size": int(args.batch_size),
        "input_noise": float(args.input_noise),
        "output_noise_frac": float(args.output_noise_frac),
        "extractor": {"name": args.extractor, "kwargs": extractor_kwargs,
                      "resolved": extractor_cfg},
        "trace_every": int(args.trace_every),
        "cell_max_hours": float(args.cell_max_hours),
        "seed_base": int(args.seed_base),
        "method_configs": configs,
        "landscape": {
            "kind": "needles", "domain": nd.DOMAIN,
            "description": ("bumps-only CartesianEnsemble on [0,1]^dim: n negated-"
                            "Ackley optima of sharpness b on a flat plain, every "
                            "other feature off"),
            "sigma_x": float(nd.SIGMA_X),
            "sigma_y_at_peak": round(float(nd.sigma_y_at_peak()), 6),
            "plain_y": nd.PLAIN_Y, "peak_y": nd.PEAK_Y,
        },
        "feasibility": feasibility,
        "reclaim_after_min": float(args.reclaim_after_min),
        "max_attempts": int(args.max_attempts),
    }
    atomic_write_json(os.path.join(out_dir, MANIFEST), manifest)

    n_workers = max(1, int(args.n_workers))
    sbatch = SBATCH_TEMPLATE.format(
        job_name=args.job_name or "zh_bench",
        out_dir=out_dir, repo=REPO_ROOT,
        walltime=f"{max(1, int(round(args.walltime_hours)))}:00:00",
        cpus=int(args.cpus_per_task), mem=args.mem, partition=args.partition,
        gres_line=(f"#SBATCH --gres={args.gres}\n" if args.gres else ""),
        device_flag=(" \\\n    --device cuda" if args.gres else ""),
        last_worker=n_workers - 1, n_workers=n_workers, n_tasks=len(tasks),
        methods=", ".join(methods), worker_hours=float(args.worker_hours),
        reclaim_after_min=float(args.reclaim_after_min),
        max_attempts=int(args.max_attempts),
    )
    sbatch_path = os.path.join(out_dir, "sweep.sbatch")
    # UTF-8, ASCII-only template, LF endings: a plan written from a Windows checkout
    # must still be a script bash on the cluster will run.
    with open(sbatch_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sbatch)
    try:
        os.chmod(sbatch_path, 0o755)
    except OSError:
        pass

    print(f"\n  plan -> {out_dir}")
    print(f"    methods: {', '.join(methods)}")
    print(f"    landscapes: dims {dims} x needles {counts} x basin widths {widths} "
          f"= {manifest['n_configurations']} configuration(s) on the unit cube")
    print(f"    x {n_draws} draw(s) x {len(methods)} method(s) = {len(tasks)} cell(s)")
    rule = (f"{args.budget} at every dim" if args.budget is not None
            else f"{args.budget_per_dim} x dim")
    print(f"    budget: {rule} = " + ", ".join(f"{n} at {d}d" for d, n in budgets.items())
          + f" points per cell, in batches of {args.batch_size} "
          f"(wall-clock ceiling {args.cell_max_hours:g} h)")
    small = [d for d, n in budgets.items() if n <= 48]
    if small:
        print(f"    WARNING: the budget at dim {', '.join(small)} does not exceed the "
              "48-point initial design, so no method gets to choose a point there.")
    print(f"    noise: input {args.input_noise:g}, output {args.output_noise_frac:g} x |y|;"
          f" extractor {args.extractor}")
    for method in methods:
        for dim, rec in configs[method].items():
            if method == "zombi_hop" or dim == str(dims[0]):
                flag = "  [STAND-IN]" if rec["is_stand_in"] else ""
                where = f"dim {dim:>2}" if method == "zombi_hop" else "all dims"
                print(f"    {method:<10} {where}: {rec['source']}{flag}")
    zombi_cfg = next(iter(configs.get("zombi_hop", {}).values()), {}).get("config", {})
    if zombi_cfg.get("sampling") == "point" and int(args.batch_size) != 1:
        print(f"    WARNING: zombi_hop measures 1 point per call but the baselines get "
              f"batches of {args.batch_size}; the comparison is not like-for-like.")
    tight = [r for r in feasibility if not r["feasible"]]
    if tight:
        print(f"    NOTE: {len(tight)} configuration(s) above the packing bound; "
              "placement falls back to the input-noise floor and records it.")
    print(f"    {n_workers} self-restarting worker(s) @ {args.walltime_hours:g} h on "
          f"{args.partition}" + (f" ({args.gres})" if args.gres else ""))
    print(f"    submit:      sbatch {sbatch_path}")
    print(f"    drain here:  python -m benchmarks.sweeps run --out {out_dir}")
    return out_dir


# ─── Claims: atomic, heartbeated, self-releasing, bounded retries ────────────────

def _claim_path(out_dir: str, tid: str) -> str:
    return os.path.join(out_dir, CLAIMS, tid)


def _is_failed(claim: str) -> bool:
    return os.path.isfile(os.path.join(claim, FAILED))


def _claim_age_s(claim: str) -> float:
    """Seconds since the claim last showed a sign of life (heartbeat, else mkdir)."""
    for candidate in (os.path.join(claim, HEARTBEAT), claim):
        try:
            return max(0.0, time.time() - os.path.getmtime(candidate))
        except OSError:
            continue
    return 0.0


def _attempts_log(out_dir: str, tid: str) -> str:
    """Failure count, kept OUTSIDE the claim so releasing a stale claim keeps it."""
    return os.path.join(out_dir, LOGS, f"attempts_{tid}")


def _count_attempts(out_dir: str, tid: str) -> int:
    try:
        with open(_attempts_log(out_dir, tid)) as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def _record_failure(out_dir: str, tid: str, claim: str, max_attempts: int) -> bool:
    """Bump the failure count; mark the claim FAILED at the limit. True if marked."""
    n = _count_attempts(out_dir, tid) + 1
    with open(_attempts_log(out_dir, tid), "w") as f:
        f.write(f"{n}\n")
    if n >= int(max_attempts):
        with open(os.path.join(claim, FAILED), "w") as f:
            f.write(f"{n} failed attempt(s); last {datetime.datetime.now().isoformat()}\n")
        return True
    return False


def _release_stale(out_dir: str, tasks: list[dict], max_age_s: float,
                   include_failed: bool = False) -> int:
    """Release claims that stopped beating and produced no result.

    Safe while other workers are live because it keys on the heartbeat. FAILED
    claims are kept (that is what stops the retries) unless ``include_failed``.
    """
    n = 0
    for t in tasks:
        claim = _claim_path(out_dir, t["tid"])
        if not os.path.isdir(claim) or is_complete(task_dir(out_dir, t)):
            continue
        failed = _is_failed(claim)
        if failed and not include_failed:
            continue
        if not failed and _claim_age_s(claim) < max_age_s:
            continue
        try:
            shutil.rmtree(claim)
            if failed:
                try:
                    os.remove(_attempts_log(out_dir, t["tid"]))
                except OSError:
                    pass
            n += 1
        except OSError:
            pass
    return n


class _Heartbeat:
    """Touch a claim's heartbeat file on a daemon thread while its cell runs."""

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
            pass

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_EVERY_S):
            self._beat()


# ─── One cell (runs in the child process) ────────────────────────────────────────

def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run_one_cell(task: dict, out_dir: str, manifest: dict, target: str,
                 device: str | None = None, verbose: bool = True) -> dict:
    """Build the landscape, run the method on it, write the cell and its record."""
    device = device or _default_device()
    seed_base = int(manifest.get("seed_base", 0))
    built = nd.build_landscape(task["dim"], task["n_needles"], task["basin_width"],
                               landscape_seed(seed_base, task))
    fn = built["fn"]
    truth = GroundTruth(optima=built["centers"],
                        peak_value=float(fn.predict(built["centers"]).max()))
    seed = cell_seed(seed_base, task)
    cfg = manifest["method_configs"][task["method"]][str(task["dim"])]

    problem = Problem(
        fn.predict, task["dim"], budget=cell_budget(manifest, task["dim"]),
        batch_size=int(manifest["batch_size"]),
        input_noise=float(manifest["input_noise"]),
        output_noise_frac=float(manifest["output_noise_frac"]),
        seed=seed, deadline=time.time() + float(manifest["cell_max_hours"]) * 3600.0)
    ref = manifest.get("method_refs", {}).get(task["method"], task["method"])
    method = make_method(ref, cfg["config"], seed=seed, device=device)
    extractor = make_extractor(manifest["extractor"]["name"], device=device, seed=seed,
                               noise_frac=float(manifest["output_noise_frac"]),
                               **manifest["extractor"]["kwargs"])

    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "ensemble_config.json"), "w") as f:
        json.dump(built["config"], f, indent=2)
    t0 = time.time()
    metrics = run_method(method, problem, truth, target, extractor=extractor,
                         trace_every=int(manifest["trace_every"]),
                         extra_record={"task": task, "config_source": cfg["source"]},
                         verbose=verbose)
    record = {
        "tid": task["tid"], "method": task["method"], "cell": task["name"],
        "draw": task["draw"], "dim": task["dim"], "n_needles": task["n_needles"],
        "basin_width": task["basin_width"],
        "config_source": cfg["source"], "config_is_stand_in": cfg["is_stand_in"],
        "landscape": built["record"],
        "metrics": metrics,
        "device": device,
        "wall_s": round(time.time() - t0, 3),
    }
    atomic_write_json(os.path.join(target, CELL_FILE), record)
    return record


def cell_command(args) -> None:
    """``python -m benchmarks.sweeps cell --out DIR --tid T``: one cell, in-process."""
    out_dir = os.path.abspath(args.out)
    manifest = load_manifest(out_dir)
    task = next((t for t in read_tasks(out_dir) if t["tid"] == args.tid), None)
    if task is None:
        raise SystemExit(f"no task {args.tid!r} in {out_dir}/{QUEUE}")
    run_one_cell(task, out_dir, manifest, task_dir(out_dir, task), args.device)


# ─── Run (drain the queue) ───────────────────────────────────────────────────────

def _rotated(tasks: list, worker: int, n_workers: int) -> list:
    """Worker *k* starts ``k/n_workers`` of the way down the queue and wraps, so the
    pool does not race for the same first cell or grind through one region."""
    if n_workers <= 1 or not tasks:
        return list(tasks)
    offset = (worker * len(tasks)) // n_workers
    return tasks[offset:] + tasks[:offset]


def _spawn_cell(out_dir: str, task: dict, device: str | None, timeout_s: float) -> int:
    cmd = [sys.executable, "-m", "benchmarks.sweeps", "cell", "--out", out_dir,
           "--tid", task["tid"]]
    if device:
        cmd += ["--device", device]
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        print(f"  [cell {task['tid']}] killed after {timeout_s / 3600:.2f} h "
              "(hard timeout)", flush=True)
        return -9


def run(args) -> None:
    """Claim and execute cells until the queue drains or wall-time runs low."""
    out_dir = os.path.abspath(args.out)
    manifest = load_manifest(out_dir)
    tasks = read_tasks(out_dir)
    os.makedirs(os.path.join(out_dir, CLAIMS), exist_ok=True)

    reclaim_after_s = float(args.reclaim_after_min) * 60.0
    max_attempts = int(manifest.get("max_attempts", 3))
    # A cell's worst case: the in-cell wall-clock ceiling, then scoring (the
    # extractor's GP fits) and process start-up. The child is killed past this.
    per_cell_h = float(manifest["cell_max_hours"]) + float(args.cell_margin_hours)
    deadline = (time.time() + float(args.worker_hours) * 3600.0
                if args.worker_hours and args.worker_hours > 0 else None)

    queue = _rotated(tasks, int(args.worker), max(1, int(args.n_workers)))
    n_ran = n_failed = pass_no = 0
    print(f"  [worker {args.worker}] {len(queue)} cell(s) in view; "
          + ("no wall-time limit" if deadline is None
             else f"{args.worker_hours:g} h wall-time, stops claiming with "
                  f"{per_cell_h:.2f} h left"), flush=True)

    while True:
        pass_no += 1
        released = _release_stale(out_dir, tasks, reclaim_after_s)
        if released:
            print(f"  [worker {args.worker}] released {released} claim(s) with no "
                  f"heartbeat for {args.reclaim_after_min:g} min", flush=True)

        claimed_this_pass = 0
        for task in queue:
            target = task_dir(out_dir, task)
            if is_complete(target):
                continue
            if deadline is not None and time.time() + per_cell_h * 3600.0 > deadline:
                print(f"  [worker {args.worker}] out of wall-time for another cell "
                      f"(ran {n_ran}) — exiting cleanly so the job can resubmit",
                      flush=True)
                return
            claim = _claim_path(out_dir, task["tid"])
            try:
                os.mkdir(claim)   # atomic: exactly one worker wins
            except OSError:
                continue
            claimed_this_pass += 1

            if args.dry_run:
                print(f"  [dry-run] {task['tid']}: {task['method']} {task['name']} "
                      f"draw {task['draw']}")
                shutil.rmtree(claim, ignore_errors=True)
                continue

            print(f"  [worker {args.worker}] cell {task['tid']}: {task['method']} "
                  f"{task['name']} draw {task['draw']}", flush=True)
            with _Heartbeat(claim):
                rc = _spawn_cell(out_dir, task, args.device, per_cell_h * 3600.0)
            if rc == 0 and is_complete(target):
                n_ran += 1
                continue
            n_failed += 1
            gave_up = _record_failure(out_dir, task["tid"], claim, max_attempts)
            msg = (f"  [worker {args.worker}] cell {task['tid']} ({task['method']} "
                   f"{task['name']} draw {task['draw']}) FAILED rc={rc}"
                   + (f" — {max_attempts} attempts, marked FAILED" if gave_up
                      else " — will be retried once its claim goes stale"))
            print(msg, flush=True)
            try:
                with open(os.path.join(out_dir, LOGS, f"fail_{task['tid']}.log"), "a") as f:
                    f.write(f"=== {datetime.datetime.now().isoformat()} rc={rc} ===\n"
                            f"see {os.path.join(target, 'error.log')} and the worker log\n")
            except OSError:
                traceback.print_exc()

        if claimed_this_pass == 0:
            break
        print(f"  [worker {args.worker}] pass {pass_no} claimed "
              f"{claimed_this_pass} cell(s); rescanning", flush=True)

    print(f"  [worker {args.worker}] done — ran {n_ran} cell(s), {n_failed} failed, "
          f"{pass_no} pass(es)", flush=True)


# ─── Status / reset ──────────────────────────────────────────────────────────────

def _state(out_dir: str, task: dict, reclaim_after_s: float) -> str:
    if is_complete(task_dir(out_dir, task)):
        return "done"
    claim = _claim_path(out_dir, task["tid"])
    if not os.path.isdir(claim):
        return "pending"
    if _is_failed(claim):
        return "failed"
    return "stale" if _claim_age_s(claim) >= reclaim_after_s else "running"


STATES = ("done", "running", "stale", "pending", "failed")


def status(args) -> None:
    """Progress by method and dimension, plus the count the sbatch chain reads."""
    out_dir = os.path.abspath(args.out)
    manifest = load_manifest(out_dir)
    tasks = read_tasks(out_dir)
    reclaim_after_s = float(manifest.get("reclaim_after_min", 30.0)) * 60.0
    states = [(t, _state(out_dir, t, reclaim_after_s)) for t in tasks]

    if args.pending_count:
        # Machine-readable, last line of stdout: the sbatch resubmits while > 0. A
        # stale claim is outstanding work (it will be released); FAILED is not.
        print(sum(1 for _, s in states if s in ("pending", "stale")))
        return

    table: dict[tuple[str, int], dict[str, int]] = {}
    for t, s in states:
        row = table.setdefault((t["method"], t["dim"]), dict.fromkeys(STATES, 0))
        row[s] += 1
    print(f"  {os.path.basename(out_dir)}: {len(tasks)} cell(s)")
    print(f"    {'method':<13} {'dim':>4} " + " ".join(f"{s:>8}" for s in STATES))
    total = dict.fromkeys(STATES, 0)
    for (method, dim) in sorted(table):
        row = table[(method, dim)]
        for s in STATES:
            total[s] += row[s]
        print(f"    {method:<13} {dim:>4} " + " ".join(f"{row[s]:>8}" for s in STATES))
    print(f"    {'TOTAL':<13} {'':>4} " + " ".join(f"{total[s]:>8}" for s in STATES))
    if total["stale"]:
        print(f"    ({total['stale']} stale claim(s) will be released by the next worker)")
    if total["failed"]:
        print(f"    {total['failed']} cell(s) FAILED {manifest.get('max_attempts', 3)} "
              "times and are no longer retried — see logs/fail_*.log and each cell's "
              "error.log; `reset-stale --failed` re-opens them:")
        for t, s in states:
            if s == "failed":
                print(f"      {t['tid']}  {t['method']:<10} {t['name']} draw {t['draw']}")


def reset_stale(args) -> None:
    """Release stale claims now (``--all``: every unfinished claim; ``--failed``:
    also re-open cells marked FAILED). Workers release stale claims on their own."""
    out_dir = os.path.abspath(args.out)
    manifest = load_manifest(out_dir)
    tasks = read_tasks(out_dir)
    max_age = 0.0 if args.all else float(
        args.reclaim_after_min if args.reclaim_after_min is not None
        else manifest.get("reclaim_after_min", 30.0)) * 60.0
    if args.all:
        print("  --all: releasing every unfinished claim, heartbeat or not. Only safe "
              "with no workers running.")
    n = _release_stale(out_dir, tasks, max_age, include_failed=args.failed)
    counts = dict.fromkeys(STATES, 0)
    for t in tasks:
        counts[_state(out_dir, t, float(manifest.get("reclaim_after_min", 30.0)) * 60.0)] += 1
    print(f"  released {n} claim(s); " + ", ".join(f"{counts[s]} {s}" for s in STATES))
