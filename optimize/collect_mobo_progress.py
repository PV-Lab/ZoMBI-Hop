#!/usr/bin/env python3
"""
Merge MOBO trial progress from one or more run directories into a single summary.

Each run's ``mobo_progress.json`` records only its own trials, so the union across
runs never double-counts. Trials are tagged with ``source_run`` (and keep their
original ``trial`` number within that run).

Usage
-----
  # All mobo_* runs under optimize/runs/
  python optimize/collect_mobo_progress.py

  # Specific run directories
  python optimize/collect_mobo_progress.py \\
      optimize/runs/mobo_19_06_03_59_53_88071 \\
      optimize/runs/mobo_19_06_11_34_11_10647

  # Parent directory (expands to mobo_* children)
  python optimize/collect_mobo_progress.py optimize/runs

  # Pareto-style filters (same syntax as optimize/pareto.py --only)
  python optimize/collect_mobo_progress.py optimize/runs \\
      --only mobo_19_06_03_59_53_88071,mobo_19_06_11_34_11_10647
  python optimize/collect_mobo_progress.py optimize/runs --only 4d
  python optimize/collect_mobo_progress.py optimize/runs \\
      --only mobo_05_06_15_32/trial_112

  # Write JSON + CSV into an output folder
  python optimize/collect_mobo_progress.py optimize/runs --only 4d \\
      --out optimize/runs/mobo_4d_collected

  # Or a single JSON file path (--csv adds a sibling .csv)
  python optimize/collect_mobo_progress.py optimize/runs --only 4d \\
      --out optimize/runs/mobo_4d_collected.json --csv

  # Print summary table only
  python optimize/collect_mobo_progress.py optimize/runs --print-only
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_RUNS = _REPO / "optimize" / "runs"


def _parse_only(only_str: str) -> tuple[set[str], dict[str, set[int]], set[str]]:
    """Parse ``--only`` (same syntax as optimize/pareto.py)."""
    import re

    run_names: set[str] = set()
    run_trials: dict[str, set[int]] = {}
    run_prefixes: set[str] = set()
    for part in only_str.split(","):
        part = part.strip().replace("\\", "/").rstrip("/")
        if not part:
            continue
        segments = part.split("/")
        mobo_seg = None
        trial_seg = None
        for seg in segments:
            if seg.startswith("mobo_"):
                mobo_seg = seg
            elif re.fullmatch(r"trial_\d+", seg):
                trial_seg = seg
        if mobo_seg is None:
            if re.fullmatch(r"\d+d", part):
                run_prefixes.add(f"mobo_{part}_")
            else:
                print(f"  [--only] skipping unrecognised entry: {part}")
            continue
        if trial_seg is not None:
            num = int(trial_seg.replace("trial_", ""))
            run_trials.setdefault(mobo_seg, set()).add(num)
        else:
            run_names.add(mobo_seg)
    return run_names, run_trials, run_prefixes


def _trial_passes_filter(
    run_name: str,
    trial_num: int | None,
    *,
    only_runs: set[str] | None,
    only_trials: dict[str, set[int]] | None,
    only_prefixes: set[str] | None,
) -> bool:
    has_filter = only_runs or only_trials or only_prefixes
    if not has_filter:
        return True
    matches_prefix = any(run_name.startswith(p) for p in (only_prefixes or set()))
    if run_name in (only_runs or set()) or matches_prefix:
        trial_filter = (only_trials or {}).get(run_name)
        return trial_filter is None or trial_num in trial_filter
    trial_filter = (only_trials or {}).get(run_name)
    return trial_filter is not None and trial_num in trial_filter


def resolve_run_dirs(paths: list[str], runs_root: Path) -> list[Path]:
    """Expand CLI paths to concrete mobo run directories."""
    if paths:
        found: list[Path] = []
        for raw in paths:
            p = Path(raw).expanduser().resolve()
            if (p / "mobo_progress.json").is_file():
                found.append(p)
                continue
            children = sorted(p.glob("mobo_*/mobo_progress.json"))
            if children:
                for prog in children:
                    if not prog.parent.name.startswith("IGNORE_"):
                        found.append(prog.parent)
                continue
            print(f"  [skip] no mobo_progress.json under {p}", file=sys.stderr)
        return sorted({d.resolve() for d in found}, key=lambda d: d.name)

    if not runs_root.is_dir():
        raise FileNotFoundError(runs_root)
    return sorted(
        (
            d for d in runs_root.iterdir()
            if d.is_dir()
            and d.name.startswith("mobo_")
            and not d.name.startswith("IGNORE_")
            and (d / "mobo_progress.json").is_file()
        ),
        key=lambda d: d.name,
    )


def resolve_only_dirs(
    runs_root: Path,
    *,
    only_runs: set[str],
    only_trials: dict[str, set[int]],
    only_prefixes: set[str],
) -> list[Path]:
    """Resolve explicit ``--only`` run names to directories (even without progress JSON)."""
    found: dict[str, Path] = {}
    wanted_names = sorted(only_runs | set(only_trials.keys()))
    for name in wanted_names:
        candidates = [
            runs_root / name,
            _REPO / name,
            Path(name).expanduser(),
        ]
        hit = None
        for candidate in candidates:
            c = candidate.resolve()
            if c.is_dir():
                hit = c
                break
        if hit is None:
            print(f"  [warn] --only run directory not found: {name}", file=sys.stderr)
            continue
        found[hit.name] = hit

    if only_prefixes and runs_root.is_dir():
        for d in runs_root.iterdir():
            if not d.is_dir():
                continue
            if any(d.name.startswith(p) for p in only_prefixes):
                found[d.name] = d.resolve()

    return sorted(found.values(), key=lambda d: d.name)


def _run_label(run_dir: Path) -> str:
    cfg_path = run_dir / "run_config.json"
    label = run_dir.name
    if not cfg_path.is_file():
        return label
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return label
    landscape = cfg.get("landscape", "rf")
    dataset = cfg.get("dataset")
    if dataset and str(dataset).startswith("ackley"):
        return f"{label}  (Ackley d={cfg.get('dim', '?')}, {dataset})"
    if landscape == "ackley" or (landscape == "synthetic" and cfg.get("oracle") == "ackley"):
        return f"{label}  (Ackley d={cfg.get('dim', '?')})"
    if landscape == "synthetic":
        oracle = cfg.get("oracle", "?")
        return f"{label}  (synthetic {oracle} d={cfg.get('dim', '?')})"
    return f"{label}  (RF)"


def collect_trials(
    run_dirs: list[Path],
    *,
    only_runs: set[str] | None = None,
    only_trials: dict[str, set[int]] | None = None,
    only_prefixes: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load trials from mobo_progress.json; return (trial_records, per_run_meta)."""
    trials: list[dict] = []
    runs_meta: list[dict] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        prog_path = run_dir / "mobo_progress.json"
        if not prog_path.is_file():
            print(f"  [skip] {run_name}: no mobo_progress.json", file=sys.stderr)
            continue
        try:
            data = json.loads(prog_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [skip] {run_name}: unreadable ({exc})", file=sys.stderr)
            continue

        raw_n = len(data.get("trials", []))
        n_used = 0
        n_bad_metrics = 0
        for t in data.get("trials", []):
            trial_num = t.get("trial")
            if not _trial_passes_filter(
                run_name, trial_num,
                only_runs=only_runs, only_trials=only_trials, only_prefixes=only_prefixes,
            ):
                continue
            metrics = t.get("metrics", {})
            if "dist_to_needles" not in metrics or "dup_fraction" not in metrics:
                n_bad_metrics += 1
                continue
            rec = {
                "source_run": run_name,
                "trial": trial_num,
                "phase": t.get("phase"),
                "metrics": metrics,
                "hparams": t.get("hparams", {}),
            }
            trials.append(rec)
            n_used += 1

        if n_used:
            print(f"  [collect] {run_name}: {n_used} trial(s)")
            runs_meta.append({
                "run": run_name,
                "label": _run_label(run_dir),
                "path": str(run_dir),
                "n_trials": n_used,
                "best_dist": data.get("best_dist"),
            })
        elif raw_n:
            print(
                f"  [warn] {run_name}: {raw_n} trial(s) in mobo_progress.json but "
                f"none matched the filter"
                + (f" ({n_bad_metrics} missing dist/dup metrics)" if n_bad_metrics else ""),
                file=sys.stderr,
            )
        else:
            print(f"  [warn] {run_name}: mobo_progress.json has no trials yet", file=sys.stderr)

    return trials, runs_meta


def build_summary(
    trials: list[dict],
    runs_meta: list[dict],
    *,
    runs_root: str,
) -> dict:
    dists = [float(t["metrics"]["dist_to_needles"]) for t in trials]
    dups = [float(t["metrics"]["dup_fraction"]) for t in trials]

    time_key = None
    time_vals: list[float] = []
    keys_seen: set[str] = set()
    for t in trials:
        m = t["metrics"]
        for k in ("avg_time_per_iter_s", "runtime_s"):
            if k in m:
                keys_seen.add(k)
                time_key = k
                time_vals.append(float(m[k]))
                break

    averages: dict[str, float] = {
        "dist_to_needles": round(float(sum(dists) / len(dists)), 6),
        "dup_fraction": round(float(sum(dups) / len(dups)), 6),
    }
    if time_vals and time_key:
        averages[time_key] = round(float(sum(time_vals) / len(time_vals)), 4)

    best_i = min(range(len(dists)), key=lambda i: dists[i])
    best_trial = trials[best_i]

    return {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "runs_root": runs_root,
        "n_runs": len(runs_meta),
        "n_trials": len(trials),
        "source_runs": runs_meta,
        "time_metric_keys": sorted(keys_seen),
        "averages": averages,
        "best_dist": {
            "value": round(dists[best_i], 6),
            "trial": best_trial.get("trial"),
            "source_run": best_trial["source_run"],
        },
        "trials": trials,
    }


def print_table(runs_meta: list[dict], summary: dict) -> None:
    print("\n" + "=" * 72)
    print(f"MOBO progress collection  |  {summary['n_trials']} trial(s) from "
          f"{summary['n_runs']} run(s)")
    print("=" * 72)
    hdr = f"{'run':<36} {'trials':>7} {'best_dist':>10}  label"
    print(hdr)
    print("-" * len(hdr))
    for r in runs_meta:
        bd = r.get("best_dist") or {}
        bd_val = bd.get("value")
        bd_s = f"{bd_val:.4f}" if bd_val is not None else "?"
        print(f"{r['run']:<36} {r['n_trials']:>7} {bd_s:>10}  {r['label']}")

    bd = summary["best_dist"]
    print(f"\nGlobal best dist: {bd['value']:.6f}  "
          f"({bd['source_run']} trial {bd['trial']})")
    av = summary["averages"]
    parts = [f"dist={av['dist_to_needles']:.4f}", f"dup={av['dup_fraction']:.4f}"]
    for k in ("avg_time_per_iter_s", "runtime_s"):
        if k in av:
            parts.append(f"{k}={av[k]:.4g}")
    print("Means:           " + "  ".join(parts))
    if len(summary.get("time_metric_keys", [])) > 1:
        print("  [warn] collection mixes avg_time_per_iter_s and runtime_s trials")


def write_csv(path: Path, trials: list[dict]) -> None:
    fields = [
        "source_run", "trial", "phase",
        "dist_to_needles", "dup_fraction", "avg_time_per_iter_s", "runtime_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trials:
            m = t.get("metrics", {})
            writer.writerow({
                "source_run": t["source_run"],
                "trial": t.get("trial"),
                "phase": t.get("phase"),
                "dist_to_needles": m.get("dist_to_needles"),
                "dup_fraction": m.get("dup_fraction"),
                "avg_time_per_iter_s": m.get("avg_time_per_iter_s"),
                "runtime_s": m.get("runtime_s"),
            })


def resolve_output_paths(
    out: str | None,
    paths: list[str],
    runs_root: Path,
    *,
    write_csv: bool,
) -> tuple[Path, Path | None]:
    """Return (json_path, csv_path). csv_path is None when CSV not requested."""
    stem = "mobo_collected"

    if out:
        p = Path(out).expanduser().resolve()
        if p.suffix.lower() == ".json":
            csv_path = p.with_suffix(".csv") if write_csv else None
            p.parent.mkdir(parents=True, exist_ok=True)
            return p, csv_path
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{stem}.json", p / f"{stem}.csv"

    if paths:
        first = Path(paths[0]).expanduser().resolve()
        if first.name.startswith("mobo_") and (first / "mobo_progress.json").is_file():
            out_dir = first.parent / stem
        elif (first / "mobo_progress.json").is_file():
            out_dir = first.parent / stem
        else:
            out_dir = first / stem
    else:
        out_dir = runs_root / stem

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}.json", out_dir / f"{stem}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge MOBO mobo_progress.json trials from selected run directories.")
    parser.add_argument(
        "paths", nargs="*",
        help="Run dir(s) or parent dir(s) containing mobo_* runs "
             "(default: all mobo_* under --runs-dir)",
    )
    parser.add_argument(
        "--runs-dir", default=None,
        help=f"Default parent when no paths given (default: {_DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated run/trial filter (same syntax as optimize/pareto.py)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output folder (writes mobo_collected.json + .csv) or a .json file path",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="When --out is a .json file, also write a sibling .csv "
             "(folder output always includes CSV)",
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="Print summary table only; do not write files",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_dir or _DEFAULT_RUNS).expanduser().resolve()
    only_runs = only_trials = only_prefixes = None
    if args.only:
        only_runs, only_trials, only_prefixes = _parse_only(args.only)

    if only_runs or only_trials or only_prefixes:
        run_dirs = resolve_only_dirs(
            runs_root,
            only_runs=only_runs or set(),
            only_trials=only_trials or {},
            only_prefixes=only_prefixes or set(),
        )
    else:
        run_dirs = resolve_run_dirs(args.paths, runs_root)
        if not run_dirs:
            sys.exit(f"No MOBO runs found (looked under {runs_root} and {args.paths!r}).")

    if not run_dirs:
        sys.exit("No MOBO runs matched the selection.")

    print("=" * 72)
    print(f"Collecting from {len(run_dirs)} run(s)")
    print("=" * 72)
    trials, runs_meta = collect_trials(
        run_dirs,
        only_runs=only_runs,
        only_trials=only_trials,
        only_prefixes=only_prefixes,
    )
    if not trials:
        sys.exit("No usable trials matched the selection.")

    summary = build_summary(trials, runs_meta, runs_root=str(runs_root))
    print_table(runs_meta, summary)

    if args.print_only:
        return

    out_json, out_csv = resolve_output_paths(
        args.out, args.paths, runs_root, write_csv=args.csv,
    )
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")

    if out_csv is not None:
        write_csv(out_csv, trials)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
