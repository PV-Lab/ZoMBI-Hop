#!/usr/bin/env python3
"""Backfill dimension-aware metrics into existing evaluate run artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import (
    metric_dist_to_needles,
    metric_dup_fraction,
    metric_pct_matched,
)

_NEEDLE_META_COLS = frozenset({
    "needle_idx", "value", "median_value", "activation", "zoom",
    "iteration", "reason", "dist_to_centre",
})

_POINTS_META_COLS = frozenset({
    "sample_idx", "Y", "penalized", "activation", "zoom",
    "iter_num", "n_points", "n_needles", "dist_to_centre",
    "value", "median_value", "iteration", "reason",
})


def _coord_cols(dataset: str | None, dim: int) -> list[str]:
    """Composition column names (matches ``evaluate.coord_cols``)."""
    if dataset == "RF" and dim == 3:
        return ["FA", "MA", "Br"]
    return [f"x{i}" for i in range(1, dim + 1)]


def _pick_coord_cols(df: pd.DataFrame, dataset: str | None, dim: int) -> list[str]:
    expected = _coord_cols(dataset, dim)
    if expected and all(c in df.columns for c in expected):
        return expected
    x_cols = sorted(
        (c for c in df.columns if c.startswith("x") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    if len(x_cols) >= dim:
        return x_cols[:dim]
    if {"FA", "MA", "Br"}.issubset(df.columns) and dim == 3:
        return ["FA", "MA", "Br"]
    return []


def _find_rerun_config(start: Path) -> Path | None:
    for parent in start.parents:
        candidate = parent / "rerun_config.json"
        if candidate.is_file():
            return candidate
    return None


def _load_rerun_config(run_dir: Path) -> dict | None:
    cfg_path = _find_rerun_config(run_dir)
    if cfg_path is None:
        return None
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _load_true_optima(run_dir: Path) -> tuple[list[list[float]], int, str | None]:
    cfg = _load_rerun_config(run_dir)
    if cfg is None:
        return [], 3, None
    true_optima = cfg.get("true_optima") or []
    dim = int(cfg.get("dim") or (len(true_optima[0]) if true_optima else 3))
    return true_optima, dim, cfg.get("dataset")


def _load_needles(run_dir: Path, dim: int, *, dataset: str | None = None) -> np.ndarray:
    path = run_dir / "needles.csv"
    if not path.is_file():
        return np.empty((0, dim))
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return np.empty((0, dim))
    if df.empty:
        return np.empty((0, dim))
    coord_cols = _pick_coord_cols(df, dataset, dim)
    if not coord_cols:
        coord_cols = [c for c in df.columns if c not in _NEEDLE_META_COLS]
    if not coord_cols:
        return np.empty((0, dim))
    return df[coord_cols].to_numpy(dtype=float)


def _load_points(run_dir: Path, dim: int, *, dataset: str | None = None) -> np.ndarray:
    path = run_dir / "points.csv"
    if not path.is_file():
        return np.empty((0, dim))
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return np.empty((0, dim))
    if df.empty:
        return np.empty((0, dim))
    coord_cols = _pick_coord_cols(df, dataset, dim)
    if not coord_cols:
        coord_cols = [
            c for c in df.columns
            if c not in _POINTS_META_COLS and not c.startswith("Unnamed")
        ]
    if len(coord_cols) != dim:
        return np.empty((0, dim))
    return df[coord_cols].to_numpy(dtype=float)


def metrics_for_run_dir(run_dir: Path) -> dict[str, float] | None:
    true_optima, dim, dataset = _load_true_optima(run_dir)
    if not true_optima:
        return None
    optima = [np.asarray(o, dtype=float) for o in true_optima]
    discovered = _load_needles(run_dir, dim, dataset=dataset)
    X_all = _load_points(run_dir, dim, dataset=dataset)
    return {
        "dist_to_needles": round(metric_dist_to_needles(discovered, optima, dim=dim), 6),
        "dup_fraction": round(metric_dup_fraction(X_all, dim=dim), 6),
        "pct_matched": round(metric_pct_matched(discovered, optima, dim=dim), 4),
    }


def backfill_metrics_json(path: Path, *, dry_run: bool = False) -> bool:
    computed = metrics_for_run_dir(path.parent)
    if computed is None:
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changed = False
    for key, val in computed.items():
        if data.get(key) != val:
            data[key] = val
            changed = True
    if changed and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    return changed


def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=0)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "n": int(arr.size),
    }


def backfill_rerun_summary(path: Path, *, dry_run: bool = False) -> bool:
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    changed = False
    root = path.parent
    metric_keys = ("dist_to_needles", "dup_fraction", "pct_matched")
    for trial in summary.get("trials", []):
        trial_num = trial.get("trial")
        if trial_num is None:
            continue
        for run in trial.get("runs", []):
            run_num = run.get("run")
            if run_num is None:
                continue
            run_dir = root / f"trial_{trial_num}" / f"run_{run_num}"
            computed = metrics_for_run_dir(run_dir)
            if computed is None:
                metrics_path = run_dir / "metrics.json"
                if metrics_path.is_file():
                    with open(metrics_path, encoding="utf-8") as f:
                        m = json.load(f)
                    computed = {k: m[k] for k in metric_keys if k in m}
            if not computed:
                continue
            for key in metric_keys:
                if key in computed and run.get(key) != computed[key]:
                    run[key] = computed[key]
                    changed = True

        runs = trial.get("runs", [])
        if runs:
            for key in metric_keys:
                vals = [r[key] for r in runs if key in r]
                if vals:
                    agg = _agg(vals)
                    if trial.get(key) != agg:
                        trial[key] = agg
                        changed = True

    if changed and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    return changed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Root directory to scan (default: optimize/runs)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write.")
    args = parser.parse_args()

    root = Path(args.runs_root)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    n_metrics = 0
    for metrics_path in sorted(root.rglob("metrics.json")):
        if backfill_metrics_json(metrics_path, dry_run=args.dry_run):
            n_metrics += 1
            print(f"  metrics: {metrics_path.relative_to(root)}")

    n_summaries = 0
    for summary_path in sorted(root.rglob("rerun_summary.json")):
        if backfill_rerun_summary(summary_path, dry_run=args.dry_run):
            n_summaries += 1
            print(f"  summary: {summary_path.relative_to(root)}")

    action = "would update" if args.dry_run else "updated"
    print(f"\n{action} {n_metrics} metrics.json and {n_summaries} rerun_summary.json under {root}")


if __name__ == "__main__":
    main()
