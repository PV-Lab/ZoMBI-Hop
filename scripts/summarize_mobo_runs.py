#!/usr/bin/env python3
"""
Summarize MOBO run timing and trial counts from optimize/runs/.

Usage:
  python scripts/summarize_mobo_runs.py
  python scripts/summarize_mobo_runs.py optimize/runs/mobo_05_06_15_10
  python scripts/summarize_mobo_runs.py --latest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _load_progress(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _run_label(run_dir: str) -> str:
    cfg_path = os.path.join(run_dir, "run_config.json")
    label = os.path.basename(run_dir)
    if not os.path.exists(cfg_path):
        return label
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return label
    landscape = cfg.get("landscape", "rf")
    if landscape == "ackley":
        return f"{label}  (Ackley d={cfg.get('dim', '?')} L{cfg.get('ackley_layout', '?')})"
    return f"{label}  (RF)"


def summarize_run(run_dir: str) -> dict | None:
    prog = os.path.join(run_dir, "mobo_progress.json")
    data = _load_progress(prog)
    if not data or not data.get("trials"):
        return None
    runtimes = [float(t["metrics"]["runtime_s"]) for t in data["trials"]]
    n = len(runtimes)
    cfg = {}
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            pass
    per_trial_h = cfg.get("time_limit_hours")
    max_trials_cfg = None
    batch_cfg = cfg.get("batch_config_path")
    if batch_cfg and os.path.exists(batch_cfg):
        try:
            with open(batch_cfg) as f:
                max_trials_cfg = json.load(f).get("max_trials")
        except Exception:
            pass
    return {
        "run_dir": run_dir,
        "label": _run_label(run_dir),
        "n_trials": n,
        "total_zombi_s": sum(runtimes),
        "mean_trial_s": sum(runtimes) / n,
        "max_trial_s": max(runtimes),
        "min_trial_s": min(runtimes),
        "per_trial_limit_h": per_trial_h,
        "max_activations": cfg.get("max_activations"),
        "config_max_trials": max_trials_cfg,
        "best_dist": data.get("best_dist"),
    }


def _fmt_duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def print_summary(rec: dict) -> None:
    est_full = rec["mean_trial_s"] * (rec["config_max_trials"] or rec["n_trials"])
    print(f"\n{rec['label']}")
    print(f"  path:           {rec['run_dir']}")
    print(f"  trials done:    {rec['n_trials']}")
    print(f"  ZoMBI total:    {_fmt_duration(rec['total_zombi_s'])}  "
          f"(mean {_fmt_duration(rec['mean_trial_s'])}/trial, "
          f"max {_fmt_duration(rec['max_trial_s'])})")
    if rec["per_trial_limit_h"] is not None:
        print(f"  per-trial cap:  {rec['per_trial_limit_h']} h  (time_limit_hours)")
    elif rec["max_activations"] is not None:
        print(f"  per-trial cap:  max_activations={rec['max_activations']}")
    if rec["config_max_trials"]:
        print(f"  config trials:  {rec['config_max_trials']}  "
              f"→ est. ZoMBI {_fmt_duration(est_full)} at current mean/trial")
    if rec.get("best_dist"):
        bd = rec["best_dist"]
        print(f"  best dist:      {bd.get('value')}  (trial {bd.get('trial')})")


def find_runs(runs_dir: str, paths: list[str], latest: bool) -> list[str]:
    if paths:
        out = []
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                out.append(p)
            else:
                print(f"  [skip] not a directory: {p}", file=sys.stderr)
        return out
    cands = sorted(
        glob.glob(os.path.join(runs_dir, "mobo_*")),
        key=lambda d: os.path.getmtime(os.path.join(d, "mobo_progress.json"))
        if os.path.exists(os.path.join(d, "mobo_progress.json"))
        else os.path.getmtime(d),
    )
    if latest and cands:
        return [cands[-1]]
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize MOBO run timing from mobo_progress.json")
    ap.add_argument("runs", nargs="*", help="Run dir(s); default: all under optimize/runs")
    ap.add_argument("--runs-dir", default=None, help="Base runs directory (default: optimize/runs)")
    ap.add_argument("--latest", action="store_true", help="Only summarize the newest run")
    args = ap.parse_args()

    repo = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    runs_dir = os.path.abspath(args.runs_dir or os.path.join(repo, "optimize", "runs"))

    run_dirs = find_runs(runs_dir, args.runs, args.latest)
    if not run_dirs:
        print(f"No runs found under {runs_dir}", file=sys.stderr)
        sys.exit(1)

    n_printed = 0
    for rd in run_dirs:
        rec = summarize_run(rd)
        if rec is None:
            print(f"\n{os.path.basename(rd)}  — no completed trials in mobo_progress.json")
            continue
        print_summary(rec)
        n_printed += 1

    if n_printed == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
