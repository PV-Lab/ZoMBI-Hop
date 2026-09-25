"""
benchmarks/ablations/__main__.py
================================
The campaign CLI: ``python -m benchmarks.ablations <command> --out RUNS_DIR``.

    plan         write the manifest, the task queue and a SLURM array script
    run          drain the queue (here, or as one worker of a pool)
    status       per-arm progress
    reset-stale  release claims left by killed workers so those cells retry
    summarize    write the figures and tables for a finished (or partial) campaign
    describe     print the arm/ablation registry without running anything

``plan`` and ``run`` are separate on purpose: a plan is a artifact you can read,
diff and re-submit, and the queue it writes is what makes a campaign resumable and
parallelisable. For a quick local check, ``plan`` then ``run`` back-to-back is fine.
"""

from __future__ import annotations

import argparse
import json

from ._paths import ensure_paths

ensure_paths()

from .arms import ABLATION_KEYS, ABLATIONS, ARMS  # noqa: E402
from .campaign import plan, reset_stale, run, status  # noqa: E402
from .landscapes import BUILTIN_KINDS  # noqa: E402
from .summarize import DEFAULT_CI, DEFAULT_N_BOOT, summarize  # noqa: E402


def _add_out(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, metavar="DIR",
                   help="campaign directory (holds the manifest, queue and runs)")


def describe(args) -> None:
    """Print the registry, so `plan --help` is not the only way to see what exists."""
    if args.json:
        print(json.dumps({"ablations": {k: a.to_dict() for k, a in ABLATIONS.items()},
                          "arms": {k: a.to_dict() for k, a in ARMS.items()}}, indent=2))
        return
    for key, ab in ABLATIONS.items():
        print(f"\n{key} — {ab.title}")
        print(f"  {ab.question}")
        for name in ab.arms:
            arm = ARMS[name]
            tag = " (baseline)" if arm.is_baseline else ""
            print(f"    · {arm.label}{tag}  [{name}]")
            print(f"        {arm.description}")
            if arm.hparam_overrides:
                print(f"        hparam overrides: {arm.hparam_overrides}")
            if arm.patches:
                print(f"        patches: {', '.join(arm.patches)}")
            if arm.runner != "single":
                print(f"        runner: {arm.runner} {arm.runner_kwargs}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.ablations",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    # ── plan ──
    p = sub.add_parser("plan", help="write the manifest, queue and sbatch script",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_out(p)
    p.add_argument("--ablations", default=",".join(ABLATION_KEYS),
                   help=f"comma-separated subset of {ABLATION_KEYS}")
    p.add_argument("--landscape", default="ensemble",
                   help=f"landscape factory: one of {BUILTIN_KINDS}, or "
                        "'module:<path.py or dotted.module>:<factory_attr>'")
    p.add_argument("--landscape-arg", action="append", metavar="K=V", default=None,
                   help="extra keyword for the landscape factory; repeatable. "
                        "Values are parsed as JSON when possible")
    p.add_argument("--dim", type=int, default=6, help="simplex dimension")
    p.add_argument("--time-limit-min", type=float, default=30.0,
                   help="wall-clock budget per cell, in MINUTES. For the k-restarts "
                        "arm this is the budget shared across its restarts, so every "
                        "arm gets the same total")
    p.add_argument("--n-landscapes", type=int, default=5,
                   help="distinct landscapes every arm is run on")
    p.add_argument("--landscape-start", type=int, default=0,
                   help="first landscape index (shift it to extend a campaign with "
                        "fresh landscapes rather than re-running the old ones)")
    p.add_argument("--n-repeats", type=int, default=3,
                   help="repeats per (arm, landscape). One run of a cell is a single "
                        "draw from a stochastic optimiser, so repeats are what turn "
                        "the comparison into a distribution")
    p.add_argument("--seed-base", type=int, default=0,
                   help="offsets every cell seed; arms still share a cell's seed")
    p.add_argument("--hparams-json", default=None, metavar="PATH",
                   help="base hyperparameters (bare dict or trial.json-style blob). "
                        "Default: src/default_hparams.DEFAULT_HPARAMS")
    p.add_argument("--n-restarts", type=int, default=None,
                   help="A1: restarts per cell (default: the arm's own, 4)")
    p.add_argument("--activations-per-restart", type=int, default=None,
                   help="A1: activations one restart may run (default 1 = plain ZoMBI)")
    p.add_argument("--no-fill-budget", action="store_true",
                   help="A1: do NOT launch extra restarts when the planned ones "
                        "finish early. Leaves the arm spending less than the baseline, "
                        "so only use it when that is the comparison you want")
    p.add_argument("--n-workers", type=int, default=1,
                   help="workers the sbatch array will start")
    p.add_argument("--worker-hours", type=float, default=24.0,
                   help="wall-time per worker in the generated sbatch script")
    p.add_argument("--job-name", default=None)
    p.set_defaults(func=plan)

    # ── run ──
    p = sub.add_parser("run", help="drain the queue",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_out(p)
    p.add_argument("--worker", type=int, default=0,
                   help="this worker's index; workers start at different points in "
                        "the queue so the pool spreads across arms")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--worker-hours", type=float, default=0.0,
                   help="stop claiming new cells with less than one cell's budget "
                        "left. 0 = no limit")
    p.add_argument("--cell-margin-hours", type=float, default=0.25,
                   help="reserve on top of a cell's optimiser budget for artifact "
                        "rendering (the CoNet UMAP renders are the slow tail)")
    p.add_argument("--device", choices=("cpu", "cuda"), default=None,
                   help="override the device run_mobo picked at import")
    p.add_argument("--dry-run", action="store_true",
                   help="print the cells that would run, claim nothing")
    p.set_defaults(func=run)

    # ── status / reset-stale ──
    p = sub.add_parser("status", help="per-arm progress")
    _add_out(p)
    p.set_defaults(func=status)

    p = sub.add_parser("reset-stale",
                       help="release claims with no metrics.json so those cells retry")
    _add_out(p)
    p.set_defaults(func=reset_stale)

    # ── summarize ──
    p = sub.add_parser("summarize", help="figures and tables for the campaign",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_out(p)
    p.add_argument("--ci", type=float, default=DEFAULT_CI,
                   help="confidence level for the bands and tables")
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                   help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--ablations", default=None,
                   help="comma-separated subset to summarise (default: whatever the "
                        "manifest planned)")
    p.set_defaults(func=lambda a: summarize(
        a.out, ci=a.ci, n_boot=a.n_boot, seed=a.seed,
        ablation_keys=([k.strip().upper() for k in a.ablations.split(",")]
                       if a.ablations else None)))

    # ── describe ──
    p = sub.add_parser("describe", help="print the arm / ablation registry")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=describe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
