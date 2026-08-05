#!/usr/bin/env python3
"""Rebuild MOBO/eval ``convergence.png`` with running-best reset per activation.

Reads each trial's ``points.csv`` (needs ``Y`` + ``activation``) and overwrites
``convergence.png`` beside it.

Usage
-----
  # All trials under optimize/runs (default)
  python optimize/recompile_convergence.py

  # Specific run / trial dirs
  python optimize/recompile_convergence.py optimize/runs/rerun_trial112_campaign1a
  python optimize/recompile_convergence.py optimize/runs/mobo_05_06_15_32/trial_112
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _iter_points_csv(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        root = root.resolve()
        if root.is_file() and root.name == "points.csv":
            found.append(root)
            continue
        if not root.is_dir():
            print(f"[recompile] skip missing path: {root}", file=sys.stderr)
            continue
        found.extend(sorted(root.rglob("points.csv")))
    # de-dupe while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompile convergence.png with per-activation running best.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["optimize/runs"],
        help="Run/trial dirs (or points.csv paths). Default: optimize/runs",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Unused for the envelope (Y is stored higher-is-better); kept for CLI parity.",
    )
    args = parser.parse_args()

    # Late import so --help is fast / avoids torch at parse time when possible.
    from optimize.run_mobo import plot_convergence_from_points_csv

    roots = [(ROOT / p if not Path(p).is_absolute() else Path(p)) for p in args.paths]
    csvs = _iter_points_csv(roots)
    print(f"[recompile] {len(csvs)} points.csv under {', '.join(str(r) for r in roots)}")
    ok = skip = 0
    for i, csv in enumerate(csvs, 1):
        written = plot_convergence_from_points_csv(
            str(csv),
            maximize=not args.minimize,
        )
        if written:
            ok += 1
            if ok <= 5 or ok % 100 == 0 or i == len(csvs):
                print(f"  [{i}/{len(csvs)}] {written}")
        else:
            skip += 1
    print(f"[recompile] done: wrote={ok} skipped={skip}")


if __name__ == "__main__":
    main()
