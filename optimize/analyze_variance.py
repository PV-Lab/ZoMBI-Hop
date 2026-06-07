"""
analyze_variance.py
===================
Rank each configuration ("set") in ``optimize/variance_results.json`` by how
reproducible its three MOBO objectives are.

Every set holds ``reruns + 1`` data points (the ORIGINAL stored point plus each
re-run; 3 points by default).  For each set we compute, across those points, the
variance of every objective and then average the three variances.  Sets are
ranked from MOST reproducible (lowest average variance) to least.

Because the three objectives live on very different scales
(``runtime_s`` ~ 10²,  ``dist_to_needles`` ~ 10⁰,  ``dup_fraction`` ~ 10⁻¹),
the raw average variance is dominated by ``runtime_s``.  We therefore also report
the mean coefficient of variation (std / |mean|, averaged over the three
objectives) as a scale-free companion ranking.

Usage
-----
  conda activate zombi-hop
  python optimize/analyze_variance.py                      # default: rank by mean CV
  python optimize/analyze_variance.py path/to/variance_results.json
  python optimize/analyze_variance.py --sort-by variance   # rank by raw avg variance
  python optimize/analyze_variance.py --rank-by runtime    # rank by one metric's variance
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np

OBJECTIVES = ("dist_to_needles", "dup_fraction", "runtime_s")

# Short --rank-by aliases -> full objective name.
RANK_BY_ALIASES = {"dist": "dist_to_needles", "dup": "dup_fraction", "runtime": "runtime_s"}


def _points(s: dict) -> list[dict]:
    """All data points for a set: the original plus every re-run."""
    return [s["original"]] + list(s.get("reruns", []))


def analyze_set(s: dict) -> dict:
    """Per-objective variance / mean / std + the aggregate ranking metrics."""
    pts = _points(s)
    per_metric = {}
    variances, cvs = [], []
    for obj in OBJECTIVES:
        vals = np.array([float(p[obj]) for p in pts], dtype=float)
        mean = float(vals.mean())
        # population variance (ddof=0): describes the spread of these N points.
        var = float(vals.var(ddof=0))
        std = float(vals.std(ddof=0))
        cv = std / abs(mean) if mean != 0 else float("nan")
        per_metric[obj] = {"mean": mean, "var": var, "std": std, "cv": cv}
        variances.append(var)
        cvs.append(cv)
    return {
        "trial":        s.get("trial"),
        "n_points":     len(pts),
        "per_metric":   per_metric,
        "avg_variance": float(np.mean(variances)),
        "mean_cv":      float(np.nanmean(cvs)),
    }


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_file = os.path.join(script_dir, "variance_results.json")

    parser = argparse.ArgumentParser(
        description="Rank variance_results.json sets by average objective variance.")
    parser.add_argument("results_path", nargs="?", default=default_file,
                        help="Path to variance_results.json (default: optimize/variance_results.json).")
    parser.add_argument("--sort-by", choices=("variance", "cv"), default="cv",
                        help="Rank by mean coefficient of variation (default) or raw average variance.")
    parser.add_argument("--rank-by", choices=tuple(RANK_BY_ALIASES), default=None,
                        help="Rank by a single objective's variance instead of the aggregate "
                             "(overrides --sort-by).")
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        sys.exit(f"No such file: {args.results_path}")
    with open(args.results_path) as f:
        data = json.load(f)

    sets = data.get("sets", [])
    if not sets:
        sys.exit(f"No 'sets' found in {args.results_path}.")

    rows = [analyze_set(s) for s in sets]
    if args.rank_by is not None:
        obj = RANK_BY_ALIASES[args.rank_by]
        rows.sort(key=lambda r: r["per_metric"][obj]["var"])
        sort_label = f"variance of {obj}"
    else:
        key = "avg_variance" if args.sort_by == "variance" else "mean_cv"
        rows.sort(key=lambda r: r[key])
        sort_label = "average variance" if key == "avg_variance" else "mean CV"

    print("=" * 78)
    print(f"Variance ranking  |  {os.path.abspath(args.results_path)}")
    print(f"  {len(rows)} set(s), {rows[0]['n_points']} points each, "
          f"sorted by {sort_label} (lowest = most reproducible)")
    print("=" * 78)

    header = (f"{'rank':>4}  {'trial':>6}  {'avg_var':>12}  {'mean_cv':>8}  "
              f"{'var(dist)':>11}  {'var(dup)':>10}  {'var(runtime)':>13}")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        pm = r["per_metric"]
        print(f"{i:>4}  {r['trial']:>6}  {r['avg_variance']:>12.4f}  "
              f"{r['mean_cv']:>8.3f}  "
              f"{pm['dist_to_needles']['var']:>11.4f}  "
              f"{pm['dup_fraction']['var']:>10.4f}  "
              f"{pm['runtime_s']['var']:>13.2f}")

    best = rows[0]
    print("-" * len(header))
    print(f"Most reproducible: trial {best['trial']}  "
          f"(avg_variance={best['avg_variance']:.4f}, mean_cv={best['mean_cv']:.3f})")


if __name__ == "__main__":
    main()
