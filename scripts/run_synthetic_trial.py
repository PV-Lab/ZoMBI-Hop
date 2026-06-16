#!/usr/bin/env python3
"""
Backward-compatible wrapper — use ``optimize/evaluate.py`` directly.

This script forwards to the unified evaluate CLI.  Examples that previously worked
here now map to evaluate.py as follows:

  python scripts/run_synthetic_trial.py \\
      --hparams optimize/runs/mobo_05_06_15_32/trial_112 \\
      --oracle gaussian --time-limit-hours 0.05

  → python optimize/evaluate.py \\
      --hparams optimize/runs/mobo_05_06_15_32/trial_112 \\
      --dataset gaussian --num-runs 1 --time-limit-hours 0.05

  python scripts/run_synthetic_trial.py \\
      --hparams .../trial_112 \\
      --config optimize/mobo_batch_configs/synthetic_3d_gaussian.json

  → python optimize/evaluate.py \\
      --hparams .../trial_112 \\
      --config optimize/mobo_batch_configs/synthetic_3d_gaussian.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_EVAL = os.path.join(_REPO, "optimize", "evaluate.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper — forwards to optimize/evaluate.py.",
    )
    parser.add_argument("--hparams", required=True, metavar="PATH")
    parser.add_argument("--config", metavar="PATH", default=None)
    parser.add_argument("--oracle", default=None)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--layout", default="2", choices=["1", "2", "3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit-hours", type=float, default=None)
    parser.add_argument("--out-dir", metavar="DIR", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-convergence-plot", action="store_true")
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Repeats per trial (default: 1 for backward compat).")
    args, extra = parser.parse_known_args()

    cmd = [sys.executable, _EVAL, "--hparams", args.hparams,
           "--num-runs", str(args.num_runs)]
    if args.config:
        cmd.extend(["--config", args.config])
    elif args.oracle:
        cmd.extend(["--dataset", args.oracle])
    else:
        parser.error("Provide --config or --oracle (or pass --dataset via evaluate.py).")

    cmd.extend(["--dim", str(args.dim), "--layout", args.layout, "--seed", str(args.seed)])
    if args.time_limit_hours is not None:
        cmd.extend(["--time-limit-hours", str(args.time_limit_hours)])
    if args.out_dir:
        cmd.extend(["--out-dir", args.out_dir])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.no_video:
        cmd.append("--no-video")
    if args.no_convergence_plot:
        cmd.append("--no-convergence-plot")
    if args.log_x:
        cmd.append("--log-x")
    if args.log_y:
        cmd.append("--log-y")
    cmd.extend(extra)

    print(f"[run_synthetic_trial] forwarding to: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
