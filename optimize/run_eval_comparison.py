#!/usr/bin/env python3
"""
Run ``evaluate.py`` on RF + synthetic landscapes and aggregate a comparison table.

Default sweep: transfer the same MOBO-tuned hyperparameters (from ``--runs-path``)
onto campaign1a RF, realistic Gaussian (gaussian3d), and Rastrigin-in-ILR, with
10 min / 1 run per (trial × dataset).

Output layout (multi-dataset evaluate run)::

    optimize/runs/compare_DD_MM_HH_MM/
      comparison_config.json
      comparison_summary.json
      comparison_summary.csv
      RF/
        rerun_config.json
        rerun_summary.json
        trial_<n>/run_1/...
      gaussian3d/
        ...
      rastrigin_ilr/
        ...

Usage
-----
  # Default: trial 112, RF + gaussian3d + rastrigin_ilr @ 10 min
  python optimize/run_eval_comparison.py

  # Several MOBO trials (different hparam sets) on the same landscapes
  python optimize/run_eval_comparison.py --trials 57,112,145

  # Dry-run (print the evaluate.py command only)
  python optimize/run_eval_comparison.py --dry-run

  # Custom source MOBO run or time budget
  python optimize/run_eval_comparison.py \\
      --runs-path optimize/runs/mobo_05_06_15_32 \\
      --trials 112 --time-limit-min 10 --num-runs 1
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_EVAL = _REPO / "optimize" / "evaluate.py"
_DEFAULT_RUNS = _REPO / "optimize" / "runs" / "mobo_05_06_15_32"
_DEFAULT_OUT = _REPO / "optimize" / "runs"
_DEFAULT_DATASETS = ("RF", "gaussian3d", "rastrigin_ilr")


def _parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("no trial numbers parsed")
    return out


def _parse_datasets(raw: str) -> list[str]:
    out = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not out:
        raise ValueError("no dataset names parsed")
    return out


def _dataset_dirs(eval_root: Path) -> list[tuple[str, Path]]:
    """Return (dataset_name, dir) for each landscape in an evaluate output tree."""
    eval_root = eval_root.resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(eval_root)

    subdirs = [
        p for p in sorted(eval_root.iterdir())
        if p.is_dir() and (p / "rerun_summary.json").is_file()
    ]
    if subdirs:
        rows: list[tuple[str, Path]] = []
        for sub in subdirs:
            cfg_path = sub / "rerun_config.json"
            name = sub.name
            if cfg_path.is_file():
                with open(cfg_path, encoding="utf-8") as f:
                    name = json.load(f).get("dataset", name)
            rows.append((str(name), sub))
        return rows

    if (eval_root / "rerun_summary.json").is_file():
        name = eval_root.name
        cfg_path = eval_root / "rerun_config.json"
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as f:
                name = json.load(f).get("dataset", name)
        return [(str(name), eval_root)]

    raise FileNotFoundError(
        f"No rerun_summary.json found under {eval_root} "
        f"(expected multi-dataset subdirs or a single-dataset rerun)."
    )


def collect_rows(eval_root: Path) -> list[dict]:
    rows: list[dict] = []
    for dataset, ds_dir in _dataset_dirs(eval_root):
        with open(ds_dir / "rerun_summary.json", encoding="utf-8") as f:
            summary = json.load(f)
        n_optima = None
        cfg_path = ds_dir / "rerun_config.json"
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            n_optima = len(cfg.get("true_optima") or [])
        for trial_entry in summary.get("trials", []):
            trial = trial_entry["trial"]
            for run in trial_entry.get("runs", []):
                rows.append({
                    "dataset": dataset,
                    "trial": trial,
                    "run": run["run"],
                    "dist_to_needles": run["dist_to_needles"],
                    "dup_fraction": run["dup_fraction"],
                    "runtime_s": run["runtime_s"],
                    "n_true_optima": n_optima,
                    "path": str(ds_dir / f"trial_{trial}" / f"run_{run['run']}"),
                })
            runs = trial_entry.get("runs", [])
            if len(runs) > 1:
                rows.append({
                    "dataset": dataset,
                    "trial": trial,
                    "run": "mean",
                    "dist_to_needles": trial_entry["dist_to_needles"]["mean"],
                    "dup_fraction": trial_entry["dup_fraction"]["mean"],
                    "runtime_s": trial_entry["runtime_s"]["mean"],
                    "n_true_optima": n_optima,
                    "path": str(ds_dir / f"trial_{trial}"),
                })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "dataset", "trial", "run",
        "dist_to_needles", "dup_fraction", "runtime_s",
        "n_true_optima", "path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    show_rows = [r for r in rows if r["run"] != "mean"] or rows
    header = f"{'dataset':<16} {'trial':>6} {'run':>4} {'dist':>10} {'dup_frac':>10} {'runtime_s':>10} {'#optima':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(show_rows, key=lambda x: (x["trial"], x["dataset"], str(x["run"]))):
        n_opt = r["n_true_optima"]
        n_opt_s = str(n_opt) if n_opt is not None else "?"
        print(
            f"{r['dataset']:<16} {r['trial']:>6} {str(r['run']):>4} "
            f"{r['dist_to_needles']:>10.4f} {r['dup_fraction']:>10.4f} "
            f"{r['runtime_s']:>10.1f} {n_opt_s:>8}"
        )


def build_eval_command(
    *,
    runs_path: Path,
    trials: list[int],
    datasets: list[str],
    time_limit_min: float,
    num_runs: int,
    out_dir: Path,
    seed: int,
    extra: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(_EVAL),
        "--runs-path", str(runs_path),
        "--trials", ",".join(str(t) for t in trials),
        "--dataset", ",".join(datasets),
        "--num-runs", str(num_runs),
        "--time-limit-min", str(time_limit_min),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ]
    cmd.extend(extra)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ZoMBI-Hop on RF vs gaussian3d vs rastrigin_ilr via evaluate.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs-path",
        default=str(_DEFAULT_RUNS),
        help=f"Source MOBO run for --trials and RF landscape (default: {_DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--trials",
        default="112",
        help="Comma-separated trial numbers to re-evaluate (default: 112)",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(_DEFAULT_DATASETS),
        help=f"Comma-separated evaluate.py dataset names (default: {','.join(_DEFAULT_DATASETS)})",
    )
    parser.add_argument("--time-limit-min", type=float, default=10.0)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for synthetic landscapes (default: 42)")
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Parent directory for comparison output (default: optimize/runs)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Exact output directory (default: auto compare_DD_MM_HH_MM under --out)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional suffix for auto output dir: compare_<name>_DD_MM_HH_MM",
    )
    parser.add_argument(
        "--collect-only",
        metavar="DIR",
        help="Skip evaluate; rebuild comparison_summary from an existing output dir",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print evaluate.py command and exit")
    parser.add_argument(
        "evaluate_extra",
        nargs="*",
        help="Extra flags forwarded to evaluate.py (e.g. --no-video)",
    )
    args = parser.parse_args()

    if args.collect_only:
        eval_root = Path(args.collect_only).resolve()
        rows = collect_rows(eval_root)
        summary_path = eval_root / "comparison_summary.json"
        csv_path = eval_root / "comparison_summary.csv"
        payload = {
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "eval_root": str(eval_root),
            "rows": rows,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        write_csv(csv_path, rows)
        print(f"Wrote {summary_path}")
        print(f"Wrote {csv_path}")
        print_table(rows)
        return

    trials = _parse_csv_ints(args.trials)
    datasets = _parse_datasets(args.datasets)
    runs_path = Path(args.runs_path).resolve()
    if not runs_path.is_dir():
        sys.exit(f"--runs-path not found: {runs_path}")
    if "RF" in datasets and not (runs_path / "run_config.json").is_file():
        sys.exit(f"RF dataset requires run_config.json in {runs_path}")

    if args.out_dir:
        eval_root = Path(args.out_dir).resolve()
        eval_root.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.datetime.now().strftime("%d_%m_%H_%M")
        prefix = "compare"
        if args.name:
            prefix = f"compare_{args.name}"
        eval_root = Path(args.out).resolve() / f"{prefix}_{stamp}"
        eval_root.mkdir(parents=True, exist_ok=True)

    cmd = build_eval_command(
        runs_path=runs_path,
        trials=trials,
        datasets=datasets,
        time_limit_min=args.time_limit_min,
        num_runs=args.num_runs,
        out_dir=eval_root,
        seed=args.seed,
        extra=list(args.evaluate_extra),
    )

    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "runs_path": str(runs_path),
        "trials": trials,
        "datasets": datasets,
        "time_limit_min": args.time_limit_min,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "eval_root": str(eval_root),
    }
    with open(eval_root / "comparison_config.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print("Landscape comparison evaluate")
    print(f"  trials:   {trials}")
    print(f"  datasets: {datasets}")
    print(f"  budget:   {args.num_runs} run(s) × {args.time_limit_min:g} min")
    print(f"  output:   {eval_root}")
    print("=" * 72)
    print("Command:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        print("\n(dry-run — evaluate.py not executed)")
        return

    print()
    result = subprocess.run(cmd, cwd=str(_REPO))
    if result.returncode != 0:
        sys.exit(result.returncode)

    try:
        manifest["landscape_configs"] = {
            name: json.loads((path / "rerun_config.json").read_text(encoding="utf-8")).get("landscape_config")
            for name, path in _dataset_dirs(eval_root)
            if (path / "rerun_config.json").is_file()
        }
        with open(eval_root / "comparison_config.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except FileNotFoundError:
        pass

    rows = collect_rows(eval_root)
    summary_path = eval_root / "comparison_summary.json"
    csv_path = eval_root / "comparison_summary.csv"
    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        **{k: manifest[k] for k in ("runs_path", "trials", "datasets", "time_limit_min", "num_runs", "seed")},
        "eval_root": str(eval_root),
        "rows": rows,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_csv(csv_path, rows)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {csv_path}")
    print_table(rows)
    print(f"\nDone. Artifacts in {eval_root}")


if __name__ == "__main__":
    main()
