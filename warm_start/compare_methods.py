"""
warm_start/compare_methods.py
=============================
Aggregate the three warm-start deployment methods (static / dynamic / mixture)
into one head-to-head table.

``deploy_eval_Nd.sbatch`` runs ``optimize/evaluate.py`` once per method, each on the
warm-start GP landscape and on re-randomized ensemble landscapes, writing an
``eval/<method>/<landscape>/rerun_summary.json``. This script reads those summaries
and reports, per (method × landscape), the mean ± std across runs of the metrics
that matter — dist_to_needles (lower = closer to the true peaks), dup_fraction,
runtime — so the methods can be compared on the task they were tuned for and on
generic landscapes at once.

Writes ``comparison.md`` and ``comparison.csv`` into the base dir and prints the
table. Pure stdlib (json) — no torch/pandas — so it is cheap to run after the jobs.

Usage
-----
    python warm_start/compare_methods.py --base optimize/runs/deploy_3d_job123/eval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# baseline = production BEST_HPARAMS.md, no warm-start influence at all (the
# reference row); static/dynamic/mixture are the warm-start methods.
METHODS = ["baseline", "static", "dynamic", "mixture"]
# fullgp = GP fit to the ENTIRE real campaign (the honest ground truth); ensemble =
# generic re-randomized landscapes. (The warm-start GP is only the tuning surrogate,
# never an evaluation target.)
LANDSCAPES = ["fullgp", "ensemble"]
# (summary key, column header, lower_is_better)
METRICS = [
    ("dist_to_needles", "dist", True),
    ("dup_fraction", "dup", True),
    ("runtime_s", "runtime_s", True),
]


def _read_summary(path: Path) -> dict | None:
    """Return the first trial's aggregated metrics from a rerun_summary.json."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    trials = data.get("trials", [])
    return trials[0] if trials else None


def collect(base: Path) -> list[dict]:
    """One row per (method, landscape) that has a summary on disk."""
    rows: list[dict] = []
    for method in METHODS:
        for land in LANDSCAPES:
            summ = _read_summary(base / method / land / "rerun_summary.json")
            if summ is None:
                print(f"  [compare] missing: {method}/{land} (skipped)")
                continue
            row = {"method": method, "landscape": land,
                   "n_runs": summ.get("n_runs", 0)}
            for key, _hdr, _low in METRICS:
                agg = summ.get(key)
                if isinstance(agg, dict):
                    row[f"{key}_mean"] = agg.get("mean")
                    row[f"{key}_std"] = agg.get("std")
            rows.append(row)
    return rows


def _fmt(mean, std) -> str:
    if mean is None:
        return "—"
    return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = ["method", "landscape", "n"] + [hdr for _k, hdr, _l in METRICS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = [r["method"], r["landscape"], str(r["n_runs"])]
        for key, _hdr, _low in METRICS:
            cells.append(_fmt(r.get(f"{key}_mean"), r.get(f"{key}_std")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("_dist_to_needles / dup_fraction / runtime: lower is better._")
    path.write_text("\n".join(lines) + "\n")


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["method", "landscape", "n_runs"]
    for key, _hdr, _low in METRICS:
        cols += [f"{key}_mean", f"{key}_std"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate static/dynamic/mixture eval into one table.")
    parser.add_argument("--base", required=True,
                        help="Dir holding <method>/<landscape>/rerun_summary.json.")
    args = parser.parse_args()
    base = Path(args.base)
    rows = collect(base)
    if not rows:
        raise SystemExit(f"no rerun_summary.json found under {base}")
    md_path = base / "comparison.md"
    csv_path = base / "comparison.csv"
    write_markdown(rows, md_path)
    write_csv(rows, csv_path)
    print("\n" + md_path.read_text())
    print(f"  comparison -> {md_path}")
    print(f"  comparison -> {csv_path}")


if __name__ == "__main__":
    main()
