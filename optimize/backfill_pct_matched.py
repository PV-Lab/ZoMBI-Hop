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
    metric_avg_pairwise_dist,
    metric_dist_to_needles,
    metric_dup_fraction,
    metric_pct_matched_comp,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_REFERENCE_OPTIMA = (
    _SCRIPT_DIR / "reference_optima" / "mobo_05_06_15_32_campaign1a.json"
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


def load_reference_optima(path: Path) -> tuple[list[list[float]], str]:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    raw = cfg.get("true_optima")
    if not raw:
        raise ValueError(f"{path}: missing 'true_optima'")
    return raw, str(path.resolve())


def patch_rerun_config(path: Path, true_optima: list[list[float]], *, source: str) -> bool:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    changed = False
    if cfg.get("true_optima") != true_optima:
        cfg["true_optima"] = true_optima
        changed = True
    if cfg.get("true_optima_source") != source:
        cfg["true_optima_source"] = source
        changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
    return changed


def find_rerun_configs(
    root: Path,
    *,
    dataset: str | None = None,
) -> list[Path]:
    out: list[Path] = []
    for cfg_path in sorted(root.rglob("rerun_config.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ds = str(cfg.get("dataset") or cfg_path.parent.name)
        if dataset is not None and ds != dataset:
            continue
        out.append(cfg_path)
    return out


def patch_rerun_configs(
    root: Path,
    true_optima: list[list[float]],
    *,
    source: str,
    dataset: str | None = None,
    dry_run: bool = False,
) -> list[Path]:
    patched: list[Path] = []
    for cfg_path in find_rerun_configs(root, dataset=dataset):
        if dry_run:
            patched.append(cfg_path)
            continue
        if patch_rerun_config(cfg_path, true_optima, source=source):
            patched.append(cfg_path)
    return patched


def _find_rerun_config(start: Path) -> Path | None:
    for parent in start.parents:
        candidate = parent / "rerun_config.json"
        if candidate.is_file():
            return candidate
    return None


def _find_mobo_run_config(start: Path) -> Path | None:
    for parent in start.parents:
        if parent.name.startswith("mobo_"):
            candidate = parent / "run_config.json"
            if candidate.is_file():
                return candidate
    return None


def _load_run_config(run_dir: Path) -> tuple[list[list[float]], int, str | None] | None:
    cfg_path = _find_rerun_config(run_dir)
    if cfg_path is not None:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        true_optima = cfg.get("true_optima") or []
        dim = int(cfg.get("dim") or (len(true_optima[0]) if true_optima else 3))
        return true_optima, dim, cfg.get("dataset")

    mobo_path = _find_mobo_run_config(run_dir)
    if mobo_path is not None:
        with open(mobo_path, encoding="utf-8") as f:
            cfg = json.load(f)
        true_optima = cfg.get("true_optima") or []
        dim = len(true_optima[0]) if true_optima else 3
        return true_optima, int(dim), "RF"

    return None


def _load_true_optima(run_dir: Path) -> tuple[list[list[float]], int, str | None]:
    loaded = _load_run_config(run_dir)
    if loaded is None:
        return [], 3, None
    return loaded


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
        "pct_matched_comp": round(metric_pct_matched_comp(discovered, optima, dim=dim), 4),
        "pct_matched": round(metric_pct_matched_comp(discovered, optima, dim=dim), 4),
    }


def backfill_metrics_over_time_csv(path: Path, run_dir: Path) -> bool:
    """Recompute dist/pct (and avg pairwise) from needles.csv using rerun_config optima."""
    if not path.is_file():
        return False
    true_optima, dim, dataset = _load_true_optima(run_dir)
    if not true_optima:
        return False
    optima = [np.asarray(o, dtype=float) for o in true_optima]
    needles_path = run_dir / "needles.csv"
    if not needles_path.is_file():
        return False
    try:
        ndf = pd.read_csv(needles_path)
    except pd.errors.EmptyDataError:
        return False
    if ndf.empty or "iteration" not in ndf.columns:
        return False
    coord_cols = _pick_coord_cols(ndf, dataset, dim)
    if not coord_cols:
        return False

    mot = pd.read_csv(path)
    if mot.empty or "iteration" not in mot.columns:
        return False

    for col in ("pct_matched_comp", "pct_matched"):
        if col not in mot.columns:
            mot[col] = np.nan

    changed = False
    for idx, row in mot.iterrows():
        iter_num = int(row["iteration"])
        sub = ndf[ndf["iteration"] <= iter_num]
        if sub.empty:
            disc = np.empty((0, dim))
        else:
            disc = sub[coord_cols].to_numpy(dtype=float)
        dist = round(metric_dist_to_needles(disc, optima, dim=dim), 6)
        pct_comp = round(metric_pct_matched_comp(disc, optima, dim=dim), 4)
        apd = round(metric_avg_pairwise_dist(disc), 6)
        if row.get("dist_to_needles") != dist:
            mot.at[idx, "dist_to_needles"] = dist
            changed = True
        for col, val in (
            ("pct_matched_comp", pct_comp),
            ("pct_matched", pct_comp),
        ):
            if row.get(col) != val:
                mot.at[idx, col] = val
                changed = True
        if "avg_pairwise_dist" in mot.columns and row.get("avg_pairwise_dist") != apd:
            mot.at[idx, "avg_pairwise_dist"] = apd
            changed = True

    if changed:
        mot.to_csv(path, index=False)
    return changed


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
    metric_keys = ("dist_to_needles", "dup_fraction",
                   "pct_matched_comp", "pct_matched")
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
        default=str(_SCRIPT_DIR / "runs"),
        help="Root directory to scan (default: optimize/runs)",
    )
    parser.add_argument(
        "--true-optima-json",
        default=None,
        metavar="PATH",
        help="Patch rerun_config.json true_optima before recomputing metrics "
             f"(default when --dataset RF: {_DEFAULT_REFERENCE_OPTIMA.name})",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only patch/backfill runs for this evaluate.py dataset name (e.g. RF)",
    )
    parser.add_argument(
        "--skip-patch",
        action="store_true",
        help="Do not patch rerun_config.json; only recompute metrics from existing config",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write.")
    args = parser.parse_args()

    root = Path(args.runs_root)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    ref_path: Path | None = None
    if args.true_optima_json:
        ref_path = Path(args.true_optima_json)
    elif args.dataset == "RF":
        ref_path = _DEFAULT_REFERENCE_OPTIMA

    n_patched = 0
    if ref_path and not args.skip_patch:
        if not ref_path.is_file():
            sys.exit(f"Reference optima not found: {ref_path}")
        true_optima, source = load_reference_optima(ref_path)
        patched = patch_rerun_configs(
            root, true_optima, source=source, dataset=args.dataset, dry_run=args.dry_run,
        )
        n_patched = len(patched)
        label = "would patch" if args.dry_run else "patched"
        for p in patched:
            rel = p.relative_to(root)
            ds = json.loads(p.read_text(encoding="utf-8")).get("dataset", "?")
            print(f"  config ({label}): {rel}  [{ds}, {len(true_optima)} optima]")
        print(f"\n{label} {n_patched} rerun_config.json file(s)")

    n_metrics = 0
    n_mot = 0
    seen_run_dirs: set[Path] = set()

    def _should_backfill(run_dir: Path) -> tuple[list, int, str | None] | None:
        loaded = _load_run_config(run_dir)
        if loaded is None:
            return None
        _, _, ds = loaded
        if args.dataset is not None and str(ds or "") != args.dataset:
            return None
        return loaded

    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        if _should_backfill(run_dir) is None:
            continue
        seen_run_dirs.add(run_dir)
        if backfill_metrics_json(metrics_path, dry_run=args.dry_run):
            n_metrics += 1
            print(f"  metrics: {metrics_path.relative_to(root)}")

    for mot_path in sorted(root.rglob("metrics_over_time.csv")):
        run_dir = mot_path.parent
        if _should_backfill(run_dir) is None:
            continue
        if args.dry_run:
            continue
        if backfill_metrics_over_time_csv(mot_path, run_dir):
            n_mot += 1
            print(f"  metrics_over_time: {mot_path.relative_to(root)}")
        metrics_path = run_dir / "metrics.json"
        if run_dir not in seen_run_dirs and metrics_path.is_file():
            if backfill_metrics_json(metrics_path, dry_run=args.dry_run):
                n_metrics += 1
                print(f"  metrics: {metrics_path.relative_to(root)}")
            seen_run_dirs.add(run_dir)

    n_summaries = 0
    for summary_path in sorted(root.rglob("rerun_summary.json")):
        cfg_path = summary_path.parent / "rerun_config.json"
        if cfg_path.is_file() and args.dataset is not None:
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cfg = {}
            if str(cfg.get("dataset") or summary_path.parent.name) != args.dataset:
                continue
        if backfill_rerun_summary(summary_path, dry_run=args.dry_run):
            n_summaries += 1
            print(f"  summary: {summary_path.relative_to(root)}")

    action = "would update" if args.dry_run else "updated"
    print(
        f"\n{action} {n_patched} config(s), {n_metrics} metrics.json, "
        f"{n_mot} metrics_over_time.csv, and {n_summaries} rerun_summary.json under {root}"
    )


if __name__ == "__main__":
    main()
