#!/usr/bin/env python3
"""
Scan ``optimize/runs`` for ``rerun_*`` directories and aggregate metric summaries.

Filters by trial number and dataset name (prefix match).  Useful for collecting
results spread across many single-dataset evaluate runs, e.g. trial 112 on
ackley3d / ackley4d / ackley10d at different time budgets.

Usage
-----
  # Default: trial 112, ackley* datasets, write to optimize/runs/
  python optimize/collect_rerun_summaries.py

  # Custom trial / datasets / output path
  python optimize/collect_rerun_summaries.py \\
      --trials 112 --datasets ackley3d,ackley4d,ackley10d \\
      --out optimize/runs/ackley_trial112_summary.json

  # Print table only (no files written)
  python optimize/collect_rerun_summaries.py --print-only
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_RUNS = _REPO / "optimize" / "runs"
_ACKLEY_DATASETS = ("ackley3d", "ackley4d", "ackley10d")
_METRIC_KEYS = ("dist_to_needles", "dup_fraction", "pct_matched", "runtime_s")


def _parse_csv_ints(raw: str) -> list[int]:
    out = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not out:
        raise ValueError("no trial numbers parsed")
    return out


def _parse_datasets(raw: str) -> list[str]:
    out = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not out:
        raise ValueError("no dataset names parsed")
    return out


def _dataset_matches(name: str, patterns: list[str]) -> bool:
    return any(name == p or name.startswith(p) for p in patterns)


def find_rerun_dirs(
    runs_root: Path,
    *,
    trials: set[int],
    datasets: list[str],
) -> list[tuple[Path, dict]]:
    """Return (dir, rerun_config) for matching single-dataset reruns."""
    rows: list[tuple[Path, dict]] = []
    if not runs_root.is_dir():
        raise FileNotFoundError(runs_root)

    for cfg_path in sorted(runs_root.rglob("rerun_config.json")):
        root = cfg_path.parent
        summary_path = root / "rerun_summary.json"
        if not summary_path.is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cfg_trials = set(cfg.get("trials") or [])
        if not cfg_trials.issubset(trials) or not cfg_trials:
            continue
        ds = str(cfg.get("dataset") or root.name)
        if not _dataset_matches(ds, datasets):
            continue
        rows.append((root, cfg))
    return rows


def collect_rows(rerun_dirs: list[tuple[Path, dict]], *, runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for root, cfg in rerun_dirs:
        summary = json.loads((root / "rerun_summary.json").read_text(encoding="utf-8"))
        ds = str(cfg.get("dataset") or root.name)
        time_limit = cfg.get("time_limit_min")
        dim = cfg.get("dim")
        n_optima = len(cfg.get("true_optima") or [])
        rel = str(root.relative_to(runs_root))
        for trial_entry in summary.get("trials", []):
            trial = trial_entry["trial"]
            for run in trial_entry.get("runs", []):
                rows.append({
                    "rerun_dir": rel,
                    "dataset": ds,
                    "dim": dim,
                    "time_limit_min": time_limit,
                    "trial": trial,
                    "run": run["run"],
                    "dist_to_needles": run["dist_to_needles"],
                    "dup_fraction": run["dup_fraction"],
                    "pct_matched": run.get("pct_matched"),
                    "runtime_s": run["runtime_s"],
                    "n_true_optima": n_optima,
                    "path": str(root / f"trial_{trial}" / f"run_{run['run']}"),
                })
            agg = trial_entry
            if len(trial_entry.get("runs", [])) > 1:
                rows.append({
                    "rerun_dir": rel,
                    "dataset": ds,
                    "dim": dim,
                    "time_limit_min": time_limit,
                    "trial": trial,
                    "run": "mean",
                    "dist_to_needles": agg["dist_to_needles"]["mean"],
                    "dup_fraction": agg["dup_fraction"]["mean"],
                    "pct_matched": agg.get("pct_matched", {}).get("mean"),
                    "runtime_s": agg["runtime_s"]["mean"],
                    "n_true_optima": n_optima,
                    "path": str(root / f"trial_{trial}"),
                })
    return rows


def coverage_matrix(
    rows: list[dict],
    *,
    datasets: list[str],
    time_limits: list[float],
    dims: list[int] | None = None,
) -> dict:
    """Report which (row_key, time_limit_min) combos have at least one run row."""
    have_ds: set[tuple[str, float]] = set()
    have_dim: set[tuple[int, float]] = set()
    for r in rows:
        if r["run"] == "mean":
            continue
        tl = float(r["time_limit_min"])
        have_ds.add((r["dataset"], tl))
        if r["dim"] is not None:
            have_dim.add((int(r["dim"]), tl))

    if dims is not None:
        matrix = {}
        for d in dims:
            matrix[str(d)] = {tl: (d, tl) in have_dim for tl in time_limits}
        return matrix

    matrix = {}
    for ds in datasets:
        matrix[ds] = {tl: (ds, tl) in have_ds for tl in time_limits}
    return matrix


def print_table(rows: list[dict]) -> None:
    show = [r for r in rows if r["run"] != "mean"] or rows
    header = (
        f"{'dataset':<12} {'dim':>3} {'time_min':>8} {'trial':>5} {'run':>4} "
        f"{'dist':>10} {'dup_frac':>10} {'pct_match':>10} {'runtime_s':>10}  {'rerun_dir'}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(show, key=lambda x: (x["dataset"], x["time_limit_min"], x["rerun_dir"], x["run"])):
        pct = r.get("pct_matched")
        pct_s = f"{pct:.2f}" if pct is not None else "?"
        print(
            f"{r['dataset']:<12} {r['dim']:>3} {r['time_limit_min']:>8g} {r['trial']:>5} {str(r['run']):>4} "
            f"{r['dist_to_needles']:>10.4f} {r['dup_fraction']:>10.4f} "
            f"{pct_s:>10} {r['runtime_s']:>10.1f}  {r['rerun_dir']}"
        )


def print_coverage(matrix: dict, *, row_label: str, row_keys: list[str], time_limits: list[float]) -> None:
    print(f"\nCoverage ({row_label} × time_limit_min):")
    hdr = f"{row_label:<12}" + "".join(f"{tl:>10g}" for tl in time_limits)
    print(hdr)
    print("-" * len(hdr))
    for key in row_keys:
        cells = "".join(
            f"{'✓':>10}" if matrix[key][tl] else f"{'—':>10}"
            for tl in time_limits
        )
        print(f"{key:<12}{cells}")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "rerun_dir", "dataset", "dim", "time_limit_min", "trial", "run",
        "dist_to_needles", "dup_fraction", "pct_matched", "runtime_s",
        "n_true_optima", "path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default=str(_DEFAULT_RUNS),
                        help=f"Root to scan for rerun_* dirs (default: {_DEFAULT_RUNS})")
    parser.add_argument("--trials", default="112", help="Comma-separated trial filter (default: 112)")
    parser.add_argument("--datasets", default=",".join(_ACKLEY_DATASETS),
                        help=f"Dataset filter (default: {','.join(_ACKLEY_DATASETS)})")
    parser.add_argument("--time-limits", default="3,10",
                        help="Time limits for coverage matrix (default: 3,10)")
    parser.add_argument("--dims", default=None,
                        help="Coverage matrix rows by dim (e.g. 3,4,10 for rastrigin_ilr)")
    parser.add_argument("--out", default=None,
                        help="Write JSON summary here (default: <runs-root>/<dataset>_trial112_summary.json)")
    parser.add_argument("--print-only", action="store_true", help="Print table; do not write files")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    trials = set(_parse_csv_ints(args.trials))
    datasets = _parse_datasets(args.datasets)
    time_limits = [float(tok.strip()) for tok in args.time_limits.split(",") if tok.strip()]
    dims = [int(tok.strip()) for tok in args.dims.split(",")] if args.dims else None

    rerun_dirs = find_rerun_dirs(runs_root, trials=trials, datasets=datasets)
    rows = collect_rows(rerun_dirs, runs_root=runs_root)
    matrix = coverage_matrix(rows, datasets=datasets, time_limits=time_limits, dims=dims)
    if dims is not None:
        row_label, row_keys = "dim", [str(d) for d in dims]
    else:
        row_label, row_keys = "dataset", datasets

    print(f"Found {len(rerun_dirs)} rerun dir(s), {len([r for r in rows if r['run'] != 'mean'])} run row(s)")
    print_table(rows)
    print_coverage(matrix, row_label=row_label, row_keys=row_keys, time_limits=time_limits)

    if args.print_only:
        return

    default_stem = datasets[0] if len(datasets) == 1 else "multi"
    out_json = Path(args.out) if args.out else runs_root / f"{default_stem}_trial112_summary.json"
    out_csv = out_json.with_suffix(".csv")
    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "runs_root": str(runs_root),
        "trials": sorted(trials),
        "datasets": datasets,
        "time_limits": time_limits,
        "n_rerun_dirs": len(rerun_dirs),
        "coverage": matrix,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(out_csv, rows)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
