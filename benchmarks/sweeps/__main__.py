"""
benchmarks/sweeps/__main__.py
=============================
The sweep CLI: ``python -m benchmarks.sweeps <command> --out RUNS_DIR``.

    plan         write the manifest, the task queue and a self-restarting sbatch
    run          drain the queue (here, or as one worker of a SLURM pool)
    cell         run ONE cell in this process (what `run` spawns per cell)
    status       progress by method and dimension (``--pending-count`` for the sbatch)
    reset-stale  release claims by hand; ``--failed`` re-opens cells marked FAILED
    summarize    tables and figures comparing the methods
    describe     print the grid, the separations it needs and every method's config
    selftest     check the landscape module's closed-form identities

``plan`` and ``run`` are separate on purpose: a plan is an artifact you can read,
diff and re-submit, and the queue it writes is what makes the campaign resumable
and parallelisable.
"""

from __future__ import annotations

import argparse
import json

from ._paths import ensure_paths

ensure_paths()

from benchmarks.methods import available_methods  # noqa: E402
from benchmarks.methods.base import DEFAULT_OUTPUT_NOISE_FRAC  # noqa: E402

from . import needles as nd  # noqa: E402
from .campaign import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_BUDGET,
    DEFAULT_METHODS,
    cell_command,
    plan,
    reset_stale,
    run,
    status,
)
from .configs import parse_method_overrides, resolve_method_configs  # noqa: E402
from .hparams import HPARAM_MAP, parse_hparam_overrides  # noqa: E402
from .summarize import DEFAULT_CI, DEFAULT_N_BOOT, summarize  # noqa: E402


def _add_out(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, metavar="DIR",
                   help="campaign directory (holds the manifest, queue and runs)")


def _add_grid_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                   help=f"comma-separated method names (registered: "
                        f"{', '.join(available_methods())}) or module:Class refs")
    p.add_argument("--dims", default=",".join(str(v) for v in nd.GRID_DIM),
                   help="cube dimensions to sweep")
    p.add_argument("--n-needles", default=",".join(str(v) for v in nd.GRID_N_NEEDLES),
                   help="true-optima counts to sweep")
    p.add_argument("--basin-widths",
                   default=",".join(f"{v:g}" for v in nd.GRID_BASIN_WIDTH),
                   help="basin sharpness values (Ackley b) to sweep")


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hparams", action="append", metavar="DIM=PATH", default=None,
                   help="override zombi_hop's per-dimension hyperparameter file; "
                        f"repeatable. Defaults: "
                        f"{', '.join(f'{d}={v[0]}' for d, v in HPARAM_MAP.items())}")
    p.add_argument("--method-config", action="append", metavar="NAME=PATH",
                   default=None, help="a JSON object of config keys for one method; "
                                      "repeatable")
    p.add_argument("--method-set", action="append", metavar="NAME.KEY=VALUE",
                   default=None, help="one config key for one method (value parsed "
                                      "as JSON); repeatable, applied after "
                                      "--method-config")


def describe(args) -> None:
    """Print what a campaign would run, without planning or running anything."""
    dims = [int(v) for v in args.dims.split(",") if v.strip()]
    counts = [int(v) for v in args.n_needles.split(",") if v.strip()]
    widths = [float(v) for v in args.basin_widths.split(",") if v.strip()]
    methods = [m for m in args.methods.split(",") if m.strip()]
    rows = nd.plan_feasibility(dims, counts, widths)
    configs = resolve_method_configs(
        methods, dims, zombi_hparam_files=parse_hparam_overrides(args.hparams),
        overrides=parse_method_overrides(args.method_config, args.method_set))
    if args.json:
        print(json.dumps({"grid": {"dims": dims, "n_needles": counts,
                                   "basin_widths": widths},
                          "methods": methods, "cells": rows,
                          "method_configs": configs}, indent=2, default=str))
        return

    print(f"\nGrid: {len(methods)} method(s) x {len(dims)} dim(s) x {len(counts)} "
          f"needle count(s) x {len(widths)} sharpness value(s) = "
          f"{len(methods) * len(rows)} cell(s) per draw, on the unit cube")
    print(f"  methods      {methods}")
    print(f"  needles      {counts}")
    print(f"  sharpness    {widths}")
    print(f"  dimensions   {dims}")
    print(f"\nResolvability (sigma_x = {nd.SIGMA_X}, sigma_y at a peak = "
          f"{nd.sigma_y_at_peak():.4f}):")
    print(f"  {'dim':>4} {'b':>5} {'sep target':>11} {'binds':>7} "
          f"{'basin radius':>13} {'capacity':>12}")
    seen = set()
    for r in rows:
        key = (r["dim"], r["basin_width"])
        if key in seen:
            continue
        seen.add(key)
        binds = "prom" if r["prominence_binds"] else "noise"
        print(f"  {r['dim']:>4} {r['basin_width']:>5g} {r['separation_target']:>11.4f} "
              f"{binds:>7} {r['basin_plain_radius']:>13.4f} "
              f"{r['capacity_estimate']:>12.4g}")
    tight = [r for r in rows if not r["feasible"]]
    print(f"\n  {len(tight)} configuration(s) above the optimistic packing bound.")

    print("\nMethod configurations:")
    for method in configs:
        for dim, rec in configs[method].items():
            if method == "zombi_hop":
                flag = "  [STAND-IN]" if rec["is_stand_in"] else ""
                print(f"  {method:<10} dim {dim:>2}: {rec['source']}{flag}")
            elif dim == str(dims[0]):
                shown = {k: v for k, v in rec["config"].items()}
                print(f"  {method:<10} all dims: {rec['source']}  {shown}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.sweeps", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    # ── plan ──
    p = sub.add_parser("plan", help="write the manifest, queue and sbatch script",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_out(p)
    _add_grid_args(p)
    _add_config_args(p)
    p.add_argument("--n-draws", type=int, default=5,
                   help="independent optima placements per landscape configuration; "
                        "every method runs on each")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="measured points per cell, initial design included")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help="points per batch: ZoMBI-Hop's points per line and the q of "
                        "the batch baselines")
    p.add_argument("--input-noise", type=float, default=0.0,
                   help="sd of Gaussian actuation noise added to every requested "
                        "coordinate (0 = the optimiser gets the point it asked for)")
    p.add_argument("--output-noise-frac", type=float, default=DEFAULT_OUTPUT_NOISE_FRAC,
                   help="multiplicative measurement noise, sd = frac * |y|")
    p.add_argument("--extractor", default="gp_peaks",
                   help="needle extractor for methods that do not declare needles "
                        "(and the controlled *_extracted metrics for all methods)")
    p.add_argument("--extractor-arg", action="append", metavar="KEY=VALUE",
                   default=None, help="extractor setting; repeatable")
    p.add_argument("--trace-every", type=int, default=10,
                   help="run the extractor every this many batches for the "
                        "trajectories (0 = final only). Each run is a GP fit")
    p.add_argument("--cell-max-hours", type=float, default=6.0,
                   help="wall-clock ceiling per cell. NOT the budget — a safety "
                        "valve; a cell stopped by it is flagged budget_hit=false")
    p.add_argument("--seed-base", type=int, default=0,
                   help="offsets every landscape placement and cell seed")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="a cell that fails this many times is marked FAILED and "
                        "no longer retried")
    p.add_argument("--n-workers", type=int, default=5,
                   help="SLURM array elements (jobs the campaign ever has queued)")
    p.add_argument("--walltime-hours", type=float, default=24.0)
    p.add_argument("--worker-hours", type=float, default=23.0,
                   help="when a worker stops claiming; keep below --walltime-hours")
    p.add_argument("--reclaim-after-min", type=float, default=30.0,
                   help="a claim whose heartbeat is silent this long is released")
    p.add_argument("--partition", default="sched_mit_sloan_gpu_r8")
    p.add_argument("--gres", default="gpu:1",
                   help="SLURM --gres; '' for a CPU-only pool (workers then run "
                        "cells on the CPU)")
    p.add_argument("--cpus-per-task", type=int, default=4,
                   help="HEBO and the acquisition optimisers are CPU-bound")
    p.add_argument("--mem", default="64G")
    p.add_argument("--job-name", default=None)
    p.add_argument("--force", action="store_true",
                   help="re-plan into a directory that already has a queue")
    p.set_defaults(func=plan)

    # ── run ──
    p = sub.add_parser("run", help="drain the queue",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_out(p)
    p.add_argument("--worker", type=int, default=0,
                   help="this worker's index; workers start at different points")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--worker-hours", type=float, default=0.0,
                   help="stop claiming with less than one cell's ceiling left; 0 = "
                        "no limit")
    p.add_argument("--cell-margin-hours", type=float, default=0.5,
                   help="added to --cell-max-hours for scoring and start-up; the "
                        "cell's process is killed past the sum")
    p.add_argument("--reclaim-after-min", type=float, default=30.0)
    p.add_argument("--device", choices=("cpu", "cuda"), default=None,
                   help="device for the cells (default: cuda if available)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the cells that would run, claim nothing")
    p.set_defaults(func=run)

    # ── cell ──
    p = sub.add_parser("cell", help="run one cell in-process (spawned by `run`)")
    _add_out(p)
    p.add_argument("--tid", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default=None)
    p.set_defaults(func=cell_command)

    # ── status / reset-stale ──
    p = sub.add_parser("status", help="progress by method and dimension")
    _add_out(p)
    p.add_argument("--pending-count", action="store_true",
                   help="print only the number of outstanding cells (pending plus "
                        "stale; FAILED excluded). The sbatch chain reads this")
    p.set_defaults(func=status)

    p = sub.add_parser("reset-stale", help="release claims by hand")
    _add_out(p)
    p.add_argument("--reclaim-after-min", type=float, default=None)
    p.add_argument("--all", action="store_true",
                   help="release EVERY unfinished claim regardless of heartbeat. "
                        "Only safe with no workers running")
    p.add_argument("--failed", action="store_true",
                   help="also re-open cells marked FAILED (resets their attempts)")
    p.set_defaults(func=reset_stale)

    # ── summarize ──
    p = sub.add_parser("summarize", help="tables and figures comparing the methods",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_out(p)
    p.add_argument("--ci", type=float, default=DEFAULT_CI)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.set_defaults(func=lambda a: summarize(a.out, ci=a.ci, n_boot=a.n_boot,
                                            seed=a.seed))

    # ── describe / selftest ──
    p = sub.add_parser("describe", help="print the grid and every method's config")
    _add_grid_args(p)
    _add_config_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=describe)

    p = sub.add_parser("selftest",
                       help="verify the landscape module's closed-form identities")
    p.set_defaults(func=lambda a: nd.selftest())

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
