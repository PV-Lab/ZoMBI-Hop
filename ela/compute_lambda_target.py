#!/usr/bin/env python3
"""
Compute ELA features from a real campaign DB.

Default: 10-feature Tier-1 target (``--tier1``).
``--full``: Muñoz Table-1 (33) + flacco classical blocks + ZoMBI extras.

Usage
-----
  python ela/compute_lambda_target.py --db data/2nd_real_run.db
  python ela/compute_lambda_target.py --db data/2nd_real_run.db --full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ela.features import FEATURE_NAMES, characterize_campaign_surrogate, save_lambda_target


def _print_group(title: str, feats: dict) -> None:
    print(f"\n{title} ({len(feats)} features)")
    print("-" * (len(title) + 12))
    w = max(len(k) for k in feats) + 2
    for k, v in sorted(feats.items()):
        if isinstance(v, float):
            print(f"  {k:<{w}} {v:.6g}")
        else:
            print(f"  {k:<{w}} {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ELA features from a campaign DB.")
    parser.add_argument("--db", default="2nd_real_run.db", help="SQLite DB path or name under data/")
    parser.add_argument("--objective", default="Objective", help="Objective column")
    parser.add_argument("--minimize", action="store_true", help="Treat objective as minimization")
    parser.add_argument("--seed", type=int, default=42, help="Sobol / ILR sample seed")
    parser.add_argument("--full", action="store_true", help="Compute all Muñoz + flacco ELA groups")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = characterize_campaign_surrogate(
        args.db,
        objective_column=args.objective,
        maximize=not args.minimize,
        sample_seed=args.seed,
        full=args.full,
    )

    db_stem = Path(args.db).stem
    suffix = "_ela_full" if args.full else "_lambda_target"
    out_path = args.out or ROOT / "data" / f"{db_stem}{suffix}.json"
    save_lambda_target(result, out_path)

    print(f"Campaign: {result['db_path']}")
    print(f"  dim={result['dim']}  n_campaign={result['n_campaign']}  "
          f"n_dense={result['n_dense_sample']}  maximize={result['maximize']}")
    print(f"  Y_campaign ∈ [{result['y_campaign_range'][0]:.4f}, {result['y_campaign_range'][1]:.4f}]")
    print(f"  Y_rf_dense ∈ [{result['y_dense_range'][0]:.4f}, {result['y_dense_range'][1]:.4f}]")
    print(f"\nSaved → {out_path}")

    if args.full and "feature_groups" in result:
        print(f"  Total features: {result['n_features_total']}")
        for name, group in result["feature_groups"].items():
            _print_group(name, group)
    else:
        print(f"\nλ_T Tier-1 ({len(FEATURE_NAMES)} features)\n")
        w = max(len(n) for n in FEATURE_NAMES) + 2
        for name in FEATURE_NAMES:
            print(f"  {name:<{w}} {result['features'][name]:.6g}")


if __name__ == "__main__":
    main()
