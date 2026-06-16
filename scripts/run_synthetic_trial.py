#!/usr/bin/env python3
"""
Run ZoMBI-Hop once on a 3D synthetic oracle using hyperparameters from a MOBO trial.

Transfers hparams from an existing MOBO trial (e.g. campaign RF tuning) onto a
direct analytic synthetic landscape (gaussian, messy, …) and writes the same
artifact set as ``optimize/run_mobo.py`` trials: CSVs, static plots, per-iteration
ternary frames, timelapse MP4, and a convergence metrics PNG.

Usage
-----
  conda activate zombi-hop-linebo

  # trial_112 hparams on synthetic_3d_gaussian (default batch JSON fields)
  python scripts/run_synthetic_trial.py \\
      --hparams optimize/runs/mobo_05_06_15_32/trial_112 \\
      --config optimize/mobo_batch_configs/synthetic_3d_gaussian.json

  # Quick smoke test (~3 min per trial budget)
  python scripts/run_synthetic_trial.py \\
      --hparams optimize/runs/mobo_05_06_15_32/trial_112 \\
      --oracle gaussian --time-limit-hours 0.05

  # Explicit output directory, CPU only, skip video
  python scripts/run_synthetic_trial.py \\
      --hparams optimize/runs/mobo_05_06_15_32/trial_112/trial.json \\
      --config optimize/mobo_batch_configs/synthetic_3d_messy.json \\
      --out-dir optimize/runs/messy_trial112 \\
      --device cpu --no-video
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys

# Repo root on sys.path (same pattern as optimize/evaluate.py).
_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "optimize"))

import run_mobo as rm
from mobo_landscapes import build_synthetic_landscape, parse_synthetic_batch_fields

try:
    import make_videos as mv
except Exception:
    mv = None


def _load_hparams(path: str) -> dict:
    """Load hparams from a trial_* dir, trial.json, or mobo_progress.json (latest trial)."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        trial_json = os.path.join(path, "trial.json")
        if os.path.isfile(trial_json):
            path = trial_json
        elif os.path.basename(path).startswith("mobo_"):
            prog = os.path.join(path, "mobo_progress.json")
            if not os.path.isfile(prog):
                sys.exit(f"No trial.json in {path} and no mobo_progress.json found.")
            with open(prog) as f:
                trials = json.load(f).get("trials", [])
            if not trials:
                sys.exit(f"No trials in {prog}.")
            hp = trials[-1].get("hparams")
            if not hp:
                sys.exit(f"Latest trial in {prog} has no hparams.")
            path = None  # loaded
        else:
            sys.exit(f"Directory has no trial.json: {path}")

    if path is not None:
        if not os.path.isfile(path):
            sys.exit(f"Hparams path not found: {path}")
        with open(path) as f:
            data = json.load(f)
        hp = data.get("hparams", data)

    missing = [k for k in rm.HPARAM_NAMES if k not in hp]
    if missing:
        sys.exit(f"Hparams missing keys {missing} (expected MOBO trial hyperparameters).")
    return {k: hp[k] for k in rm.HPARAM_NAMES}


def _load_synthetic_config(
    config_path: str | None,
    *,
    oracle: str | None,
    dim: int,
    layout: str,
    seed: int,
    time_limit_hours: float | None,
):
    """Build a synthetic LandscapeSpec from a batch JSON or CLI oracle fields."""
    if config_path:
        cfg_path = os.path.abspath(config_path)
        if not os.path.isfile(cfg_path):
            sys.exit(f"--config not found: {cfg_path}")
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("landscape") != "synthetic":
            sys.exit(f"--config must have landscape=synthetic (got {cfg.get('landscape')!r}).")
        syn = parse_synthetic_batch_fields(cfg)
        tl = time_limit_hours if time_limit_hours is not None else cfg.get("time_limit_hours")
        name = cfg.get("name") or os.path.splitext(os.path.basename(cfg_path))[0]
    else:
        if oracle is None:
            sys.exit("Provide --config or --oracle.")
        syn = {"oracle": oracle, "dim": dim, "layout": layout, "seed": seed}
        tl = time_limit_hours
        name = f"synthetic_{oracle}"

    landscape = build_synthetic_landscape(
        syn["oracle"], syn["dim"], syn["layout"],
        seed=syn["seed"],
        time_limit_hours=tl,
    )
    return landscape, name, syn, tl


def _save_convergence_plot(csv_path: str, out_png: str, *, log_x: bool = False, log_y: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    metrics = [
        ("dist_to_needles", "Distance to True Needles", "steelblue"),
        ("dup_fraction", "Duplicate Sample Fraction", "tomato"),
        ("pct_matched", "Pct Needles Matching True Needle", "seagreen"),
        ("avg_pairwise_dist", "Avg Pairwise Needle Distance", "mediumpurple"),
        ("recent_needle_value", "Most Recent Needle Value", "darkorange"),
    ]
    metrics = [m for m in metrics if m[0] in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Convergence metrics over iterations")

    for ax, (col, label, color) in zip(axes.flat, metrics):
        if col == "recent_needle_value":
            ax.plot(df["iteration"], df[col], color=color, drawstyle="steps-post", marker="o", ms=3)
        else:
            ax.plot(df["iteration"], df[col], color=color)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[len(metrics):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  convergence plot -> {out_png}")


def _default_out_dir(oracle: str, hparams_path: str) -> str:
    base = os.path.basename(os.path.normpath(hparams_path.rstrip("/")))
    if base == "trial.json":
        base = os.path.basename(os.path.dirname(hparams_path))
    stamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    return os.path.join(_REPO, "optimize", "runs", f"synthetic_{oracle}_{base}_{stamp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one ZoMBI trial on a 3D synthetic oracle with MOBO trial hparams.",
    )
    parser.add_argument(
        "--hparams", required=True, metavar="PATH",
        help="trial_* directory, trial.json, or mobo_* run dir (uses latest trial).",
    )
    parser.add_argument(
        "--config", metavar="PATH", default=None,
        help="Synthetic batch JSON (e.g. optimize/mobo_batch_configs/synthetic_3d_gaussian.json).",
    )
    parser.add_argument(
        "--oracle", default=None,
        help="Oracle name if --config omitted (messy, gaussian, ackley, planted_bumps, rastrigin_ilr).",
    )
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--layout", default="2", choices=["1", "2", "3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--time-limit-hours", type=float, default=None,
        help="ZoMBI wall-clock budget per run (overrides batch JSON).",
    )
    parser.add_argument("--out-dir", metavar="DIR", default=None, help="Output directory.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--no-video", action="store_true", help="Skip timelapse MP4 assembly.")
    parser.add_argument(
        "--no-convergence-plot", action="store_true",
        help="Skip saving convergence_metrics.png.",
    )
    parser.add_argument("--log-x", action="store_true", help="Log x-axis on convergence plot.")
    parser.add_argument("--log-y", action="store_true", help="Log y-axis on convergence plot.")
    args = parser.parse_args()

    rm._configure_mpl_backend(headless=True)
    rm._apply_runtime_overrides(device=args.device, time_limit_hours=None)

    hparams = _load_hparams(args.hparams)
    landscape, name, syn, tl = _load_synthetic_config(
        args.config,
        oracle=args.oracle,
        dim=args.dim,
        layout=args.layout,
        seed=args.seed,
        time_limit_hours=args.time_limit_hours,
    )
    if args.time_limit_hours is not None:
        landscape.time_limit_hours = args.time_limit_hours
        tl = args.time_limit_hours

    out_dir = os.path.abspath(args.out_dir or _default_out_dir(syn["oracle"], args.hparams))

    print("=" * 70)
    print(f"ZoMBI synthetic trial  |  {name}")
    print(f"  oracle={syn['oracle']}  dim={syn['dim']}  layout={syn['layout']}  seed={syn['seed']}")
    print(f"  time_limit_hours={tl}  device={rm.DEVICE}")
    print(f"  hparams from: {args.hparams}")
    print(f"  output: {out_dir}")
    print("=" * 70)

    result = rm.run_single_trial(hparams, landscape, out_dir)

    true_optima_json = [
        (o.tolist() if hasattr(o, "tolist") else list(o)) for o in landscape.true_optima
    ]
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({
            "landscape": "synthetic",
            "oracle": syn["oracle"],
            "dim": syn["dim"],
            "layout": syn["layout"],
            "seed": syn["seed"],
            "maximize": True,
            "time_limit_hours": tl,
            "true_optima": true_optima_json,
            "hparams_source": os.path.abspath(args.hparams),
            "hparams": hparams,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)

    with open(os.path.join(out_dir, "trial.json"), "w") as f:
        json.dump({
            "phase": "synthetic_eval",
            "metrics": {
                "dist_to_needles": round(result["dist"], 6),
                "dup_fraction": round(result["dup"], 6),
                "runtime_s": round(result["runtime"], 3),
            },
            "hparams": hparams,
        }, f, indent=2)

    if not args.no_video and mv is not None:
        plots_dir = os.path.join(out_dir, "plots")
        video_path = os.path.join(out_dir, "zombihop_timelapse.mp4")
        if os.path.isdir(plots_dir) and glob.glob(os.path.join(plots_dir, "iter_*.png")):
            print("  building timelapse video …")
            try:
                mv.make_video_from_dir(plots_dir, video_path)
            except Exception as exc:
                print(f"  [warn] video assembly failed: {exc}")
        else:
            print("  [warn] no plots/ frames — skipping video.")
    elif not args.no_video:
        print("  [warn] make_videos unavailable — skipping video.")

    metrics_csv = os.path.join(out_dir, "metrics_over_time.csv")
    if not args.no_convergence_plot and os.path.isfile(metrics_csv):
        _save_convergence_plot(
            metrics_csv,
            os.path.join(out_dir, "convergence_metrics.png"),
            log_x=args.log_x,
            log_y=args.log_y,
        )

    print(f"\nDone. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
