"""
benchmarks/methods/__main__.py
==============================
    python -m benchmarks.methods list
    python -m benchmarks.methods install-hebo
    python -m benchmarks.methods smoke [--methods a,b] [--dim 3] [--budget 96]
    python -m benchmarks.methods calibrate [--extractor gp_peaks] [--out DIR]

``smoke`` and ``calibrate`` fit models: run them on a compute node (see
``benchmarks/scripts/validate_methods.sbatch``), not the login node.

``smoke`` runs each method end to end on a tiny needle landscape and writes real cell directories under ``--out``, so an adapter can be
checked before it is let loose on a sweep. It is a plumbing check, not a benchmark:
the budget is far too small for the scores to mean anything.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

from . import available_methods, get_method, make_extractor, make_method
from .base import Problem
from .runner import GroundTruth, run_method


def _smoke_landscape(dim: int, seed: int):
    """Three well-separated needles on a flat plain in the unit box."""
    from synthetic_data.ensemble import CartesianEnsemble

    rng = np.random.default_rng(seed)
    centers = 0.2 + 0.6 * rng.random((3, dim))
    fn = CartesianEnsemble(
        dim=dim, n_optima=0, pinned_optima=centers, basin_width=6.0,
        n_weak=0, weak_amp=0.0, n_ridges=0, ridge_amp=0.0, noise_amp=0.0,
        aniso_strength=0.0, n_plateaus=0, plateau_amp=0.0, edge_region=None,
        edge_amp=0.0, input_noise=0.0, seed=seed)
    return fn, np.asarray(fn.centers)


def cmd_list(_args) -> None:
    for name in available_methods():
        cls = get_method(name)
        flag = "  [declares needles]" if cls.declares_needles else ""
        print(f"  {name:<10} {cls.description}{flag}")
        for k, v in cls.defaults.items():
            if k in ("hparams", "fixed"):
                v = "(see zombihop.py)" if v is not None else v
            print(f"      {k} = {v!r}")


def cmd_install_hebo(_args) -> None:
    from .hebo_method import install_hebo

    install_hebo()


def cmd_smoke(args) -> None:
    names = [m for m in args.methods.split(",") if m] if args.methods else available_methods()
    out = args.out or tempfile.mkdtemp(prefix="methods_smoke_")
    fn, optima = _smoke_landscape(args.dim, args.seed)
    truth = GroundTruth(optima=optima, peak_value=float(fn.predict(optima).max()))
    failures = []
    for name in names:
        overrides = {}
        if name == "gp_bo":
            overrides = {"num_restarts": 2, "raw_samples": 64}
        if name == "turbo":
            overrides = {"n_candidates": 500}
        method = make_method(name, overrides, seed=args.seed, device=args.device)
        problem = Problem(fn.predict, args.dim, budget=args.budget,
                          batch_size=args.batch_size, seed=args.seed)
        t0 = time.time()
        try:
            m = run_method(method, problem, truth, os.path.join(out, name),
                           extractor=make_extractor(args.extractor, device=args.device,
                                                    noise_frac=problem.output_noise_frac),
                           trace_every=2, verbose=args.verbose)
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"  FAIL  {name:<10} {type(exc).__name__}: {exc}")
            continue
        ok = m["n_points"] == args.budget and m["budget_hit"]
        if not ok:
            failures.append(name)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<10} points={m['n_points']:>4} "
              f"stop={m['stop_reason']:<7} dist={m['dist_to_needles']:.3f} "
              f"needles={m['n_needles']:>2} regret={m['simple_regret']:.3f} "
              f"({time.time() - t0:.1f}s)")
    print(f"\n  cells written to {out}")
    if failures:
        raise SystemExit(f"  smoke FAILED for: {', '.join(failures)}")


def cmd_calibrate(args) -> None:
    from .calibrate import calibrate

    calibrate(args.extractor, seeds=[int(v) for v in args.seeds.split(",") if v],
              device=args.device, out=args.out)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.methods", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="registered methods and their config").set_defaults(
        func=cmd_list)
    sub.add_parser("install-hebo", help="vendor HEBO without touching .venv").set_defaults(
        func=cmd_install_hebo)
    p = sub.add_parser("smoke", help="run each method end to end on a tiny problem",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--methods", default=None, help="comma-separated; default all")
    p.add_argument("--dim", type=int, default=3)
    p.add_argument("--budget", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--extractor", default="gp_peaks")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="default: a fresh temp dir")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("calibrate", help="score an extractor on known sample sets",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--extractor", default="gp_peaks")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_calibrate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
