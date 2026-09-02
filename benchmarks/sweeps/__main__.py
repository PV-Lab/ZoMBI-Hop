"""
benchmarks/sweeps/__main__.py
=============================
The sweep CLI: ``python -m benchmarks.sweeps <command> --out RUNS_DIR``.

    plan         write the manifest, the task queue and a self-restarting sbatch
    run          drain the queue (here, or as one worker of a SLURM pool)
    status       progress by dimension (``--pending-count`` for the sbatch chain)
    reset-stale  release claims by hand; workers normally do this on their own
    summarize    heatmaps and tables for a finished or partial campaign
    describe     print the grid, the separations it needs and the hyperparameter map
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

from . import needles as nd  # noqa: E402
from .budget import DEFAULT_N_LINES  # noqa: E402
from .campaign import plan, reset_stale, run, status  # noqa: E402
from .hparams import HPARAM_MAP, parse_hparam_overrides, resolve_all  # noqa: E402
from .summarize import DEFAULT_CI, DEFAULT_N_BOOT, summarize  # noqa: E402


def _add_out(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, metavar="DIR",
                   help="campaign directory (holds the manifest, queue and runs)")


def describe(args) -> None:
    """Print what a campaign would run, without planning or running anything."""
    dims = [int(v) for v in args.dims.split(",") if v.strip()]
    counts = [int(v) for v in args.n_needles.split(",") if v.strip()]
    widths = [float(v) for v in args.basin_widths.split(",") if v.strip()]
    rows = nd.plan_feasibility(dims, counts, widths)
    if args.json:
        print(json.dumps({"grid": {"dims": dims, "n_needles": counts,
                                   "basin_widths": widths},
                          "cells": rows,
                          "hparams": {d: {k: v for k, v in rec.items()
                                          if k != "hparams"}
                                      for d, rec in resolve_all(dims).items()}},
                         indent=2))
        return

    print(f"\nGrid: {len(dims)} dim(s) x {len(counts)} needle count(s) x "
          f"{len(widths)} sharpness value(s) = {len(rows)} configuration(s)")
    print(f"  needles      {counts}")
    print(f"  sharpness    {widths}")
    print(f"  dimensions   {dims}")
    print(f"\nResolvability (sigma_x = {nd.SIGMA_X}, sigma_y at a peak = "
          f"{nd.sigma_y_at_peak():.4f}):")
    # "placed at" is the row's SHARED separation (the strictest width's target, see
    # needles.placement_width); "own" is what this width would have asked for alone.
    # They differ only where the prominence rule beats the sigma_x floor, and that
    # gap is exactly the confound the shared placement removes.
    print(f"  {'dim':>4} {'b':>5} {'placed at':>11} {'own':>9} {'binds':>7} "
          f"{'basin radius':>13} {'capacity':>9}")
    seen = set()
    for r in rows:
        key = (r["dim"], r["basin_width"])
        if key in seen:
            continue
        seen.add(key)
        binds = "prom" if r["prominence_binds"] else "noise"
        print(f"  {r['dim']:>4} {r['basin_width']:>5g} {r['separation_target']:>11.4f} "
              f"{r['separation_own_target']:>9.4f} "
              f"{binds:>7} {r['basin_plain_radius']:>13.4f} "
              f"{r['capacity_estimate']:>9g}")
    tight = [r for r in rows if not r["feasible"]]
    print(f"\n  {len(tight)} configuration(s) above the optimistic packing bound"
          + (":" if tight else "."))
    for r in tight:
        print(f"    dim {r['dim']} / n {r['n_needles']} / b {r['basin_width']:g}")

    print("\nHyperparameters:")
    for dim, rec in resolve_all(dims).items():
        flag = "  [STAND-IN]" if rec["is_stand_in"] else ""
        print(f"  dim {dim:>2}: {rec['path']}{flag}")
        print(f"          {rec['provenance']}")


def _add_grid_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dims", default=",".join(str(v) for v in nd.GRID_DIM),
                   help="simplex dimensions to sweep")
    p.add_argument("--n-needles", default=",".join(str(v) for v in nd.GRID_N_NEEDLES),
                   help="true-optima counts to sweep")
    p.add_argument("--basin-widths",
                   default=",".join(f"{v:g}" for v in nd.GRID_BASIN_WIDTH),
                   help="basin sharpness values (Ackley b) to sweep")


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
    p.add_argument("--n-draws", type=int, default=5,
                   help="independent optima placements per grid configuration. One "
                        "draw is a single landscape AND a single draw from a "
                        "stochastic optimiser, so this is what turns a tile of the "
                        "heatmap into a distribution rather than an anecdote")
    p.add_argument("--n-lines", type=int, default=DEFAULT_N_LINES,
                   help="measurement budget per cell, in LineBO lines. 24 points "
                        "per line, and the 2 initial-design lines are charged to it, "
                        "so the default is 3000 measured compositions")
    p.add_argument("--cell-max-hours", type=float, default=6.0,
                   help="wall-clock ceiling per cell. NOT the budget (that is "
                        "--n-lines) — a safety valve so one pathological cell "
                        "cannot hold a worker indefinitely. A cell stopped by this "
                        "is flagged budget_hit=false in the summary")
    p.add_argument("--hparams", action="append", metavar="DIM=PATH", default=None,
                   help="override the per-dimension hyperparameter file; repeatable. "
                        f"Defaults: {', '.join(f'{d}={v[0]}' for d, v in HPARAM_MAP.items())}")
    p.add_argument("--seed-base", type=int, default=0,
                   help="offsets every landscape placement and cell seed")
    p.add_argument("--n-workers", type=int, default=5,
                   help="SLURM array elements. This is how many jobs the campaign "
                        "ever has queued, regardless of how many cells it holds")
    p.add_argument("--walltime-hours", type=float, default=24.0,
                   help="wall-time requested per worker job (partition maximum)")
    p.add_argument("--worker-hours", type=float, default=23.0,
                   help="when a worker stops claiming new cells, so it exits "
                        "cleanly and resubmits instead of being killed mid-cell. "
                        "Keep below --walltime-hours")
    p.add_argument("--reclaim-after-min", type=float, default=30.0,
                   help="a claim whose heartbeat has been silent this long is "
                        "released automatically by the next worker. This is what "
                        "lets the pool restart itself without `reset-stale`")
    p.add_argument("--job-name", default=None)
    p.set_defaults(func=plan)

    # ── run ──
    p = sub.add_parser("run", help="drain the queue",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_out(p)
    p.add_argument("--worker", type=int, default=0,
                   help="this worker's index; workers start at different points in "
                        "the queue so the pool spreads across the grid")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--worker-hours", type=float, default=0.0,
                   help="stop claiming with less than one cell's budget left. "
                        "0 = no limit")
    p.add_argument("--cell-margin-hours", type=float, default=0.25,
                   help="reserve on top of a cell's ceiling for artifact rendering "
                        "(the CoNet UMAP renders are the slow tail)")
    p.add_argument("--reclaim-after-min", type=float, default=30.0,
                   help="release claims whose heartbeat has been silent this long")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None,
                   help="override the device run_mobo picked at import")
    p.add_argument("--dry-run", action="store_true",
                   help="print the cells that would run, claim nothing")
    p.set_defaults(func=run)

    # ── status / reset-stale ──
    p = sub.add_parser("status", help="progress by dimension")
    _add_out(p)
    p.add_argument("--pending-count", action="store_true",
                   help="print only the number of outstanding cells (pending plus "
                        "stale-claimed). The generated sbatch reads this to decide "
                        "whether to resubmit itself")
    p.set_defaults(func=status)

    p = sub.add_parser("reset-stale", help="release claims by hand")
    _add_out(p)
    p.add_argument("--reclaim-after-min", type=float, default=None,
                   help="override the manifest's staleness threshold")
    p.add_argument("--all", action="store_true",
                   help="release EVERY unfinished claim regardless of heartbeat. "
                        "Only safe with no workers running")
    p.set_defaults(func=reset_stale)

    # ── summarize ──
    p = sub.add_parser("summarize", help="heatmaps and tables for the campaign",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_out(p)
    p.add_argument("--ci", type=float, default=DEFAULT_CI,
                   help="confidence level for the intervals")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                   help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.set_defaults(func=lambda a: summarize(a.out, ci=a.ci, n_boot=a.n_boot,
                                            seed=a.seed))

    # ── describe / selftest ──
    p = sub.add_parser("describe", help="print the grid and hyperparameter map")
    _add_grid_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=describe)

    p = sub.add_parser("selftest",
                       help="verify the landscape module's closed-form identities")
    p.set_defaults(func=lambda a: nd.selftest())

    args = ap.parse_args()
    if getattr(args, "hparams", None):
        parse_hparam_overrides(args.hparams)   # fail fast on a malformed override
    args.func(args)


if __name__ == "__main__":
    main()
