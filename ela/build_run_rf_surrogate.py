#!/usr/bin/env python3
"""
Build a scaled RF surrogate from a ZoMBI-Hop hardware run (same recipe as 3D).

Pipeline (mirrors ``2nd_real_run.db`` → RF → dense λ_T / ela_full):
  1. Read ``composition_log.jsonl`` measured compositions + y
  2. Write campaign CSV + SQLite ``results`` DB under ``data/``
  3. Train RF (500 trees, seed 42) on row-normalized X
  4. Evaluate on dense Sobol simplex sample → ``y_dense_range`` for GP affine scale
  5. Write ``*_lambda_target.json``, optional ``*_ela_full.json``, RF joblib

Usage
-----
  conda activate zombi-hop-linebo
  python ela/build_run_rf_surrogate.py --run runs/run_9dfe
  python ela/build_run_rf_surrogate.py --run runs/run_9dfe --full
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ela.features import (  # noqa: E402
    DEFAULT_OBJECTIVE,
    characterize_campaign_surrogate,
    save_lambda_target,
)


def _resolve_run_dir(run_arg: str | Path) -> Path:
    p = Path(run_arg)
    if p.is_dir():
        return p.resolve()
    for base in (ROOT / "runs", ROOT / "data"):
        cand = base / p.name
        if cand.is_dir():
            return cand.resolve()
        cand = base / f"run_{p.name}"
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"Run directory not found: {run_arg}")


def load_measured_from_composition_log(
    run_dir: Path,
) -> tuple[list[list[float]], list[float], list[int], dict]:
    """Return (X rows, y, call indices, meta) from composition_log.jsonl."""
    log_path = run_dir / "composition_log.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing {log_path}")

    hw = {}
    hw_path = run_dir / "hw_config.json"
    if hw_path.is_file():
        hw = json.loads(hw_path.read_text())
    cfg = {}
    cfg_path = run_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())

    optimizing_dims = None
    xs: list[list[float]] = []
    ys: list[float] = []
    calls: list[int] = []
    n_mismatch = 0

    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        call = int(rec.get("call", len(calls)))
        if optimizing_dims is None and rec.get("optimizing_dims"):
            optimizing_dims = list(rec["optimizing_dims"])
        for rail in rec.get("rails", []):
            measured = rail.get("measured") or []
            yvals = rail.get("y") or []
            n = min(len(measured), len(yvals))
            if len(measured) != len(yvals):
                n_mismatch += 1
            for i in range(n):
                x = [float(v) for v in measured[i]]
                s = sum(x)
                if s <= 0:
                    continue
                xs.append([v / s for v in x])
                ys.append(float(yvals[i]))
                calls.append(call)

    if not xs:
        raise RuntimeError(f"No measured points in {log_path}")

    dim = len(xs[0])
    if optimizing_dims is None and hw.get("dims"):
        optimizing_dims = [int(d) for d in str(hw["dims"]).split(",")]
    if optimizing_dims is None:
        optimizing_dims = list(range(dim))

    meta = {
        "run_dir": str(run_dir),
        "run_uuid": cfg.get("run_uuid") or run_dir.name.replace("run_", ""),
        "optimizing_dims": optimizing_dims,
        "dim": dim,
        "n_points": len(ys),
        "n_length_mismatches": n_mismatch,
        "source": "composition_log.jsonl measured+y",
        "hw_config": hw,
    }
    return xs, ys, calls, meta


def write_campaign_csv(
    out_csv: Path,
    xs: list[list[float]],
    ys: list[float],
    calls: list[int],
    *,
    comp_cols: list[str],
) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", *comp_cols, DEFAULT_OBJECTIVE])
        for i, (x, y, call) in enumerate(zip(xs, ys, calls)):
            writer.writerow([call, *[f"{v:.12g}" for v in x], f"{y:.12g}"])
    return out_csv


def write_campaign_db(
    out_db: Path,
    xs: list[list[float]],
    ys: list[float],
    calls: list[int],
    *,
    comp_cols: list[str],
) -> Path:
    out_db.parent.mkdir(parents=True, exist_ok=True)
    if out_db.exists():
        out_db.unlink()
    cols_sql = ", ".join(f'"{c}" REAL' for c in comp_cols)
    con = sqlite3.connect(str(out_db))
    try:
        con.execute(
            f'CREATE TABLE results ("Iteration" REAL, {cols_sql}, '
            f'"{DEFAULT_OBJECTIVE}" REAL)'
        )
        placeholders = ", ".join("?" for _ in range(len(comp_cols) + 2))
        rows = [(float(call), *x, float(y)) for x, y, call in zip(xs, ys, calls)]
        con.executemany(f"INSERT INTO results VALUES ({placeholders})", rows)
        con.commit()
    finally:
        con.close()
    return out_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build scaled RF surrogate from a ZoMBI hardware run.",
    )
    parser.add_argument(
        "--run", default="runs/run_9dfe",
        help="Run directory (or name under runs/ or data/)",
    )
    parser.add_argument("--out-stem", default=None,
                        help="Output stem under data/ (default: run dir name)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true",
                        help="Also write Muñoz+flacco ela_full JSON")
    parser.add_argument("--no-rf-joblib", action="store_true",
                        help="Skip saving the trained RF joblib")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    stem = args.out_stem or run_dir.name
    out_dir = ROOT / "data"
    analysis_dir = out_dir / "analysis"
    out_csv = out_dir / f"{stem}.csv"
    out_db = out_dir / f"{stem}.db"
    out_meta = out_dir / f"{stem}_rf_meta.json"
    out_joblib = analysis_dir / f"{stem}_RF_trained.joblib"
    out_lambda = out_dir / f"{stem}_lambda_target.json"
    out_ela = out_dir / f"{stem}_ela_full.json"

    xs, ys, calls, meta = load_measured_from_composition_log(run_dir)
    dim = meta["dim"]
    comp_cols = [f"Comp{i + 1}" for i in range(dim)]

    write_campaign_csv(out_csv, xs, ys, calls, comp_cols=comp_cols)
    write_campaign_db(out_db, xs, ys, calls, comp_cols=comp_cols)
    print(f"Exported {len(ys)} points (d={dim}) → {out_csv.name}, {out_db.name}")
    print(f"  optimizing_dims={meta['optimizing_dims']}")
    print(f"  Y ∈ [{min(ys):.4f}, {max(ys):.4f}]")

    result = characterize_campaign_surrogate(
        out_csv,
        objective_column=DEFAULT_OBJECTIVE,
        comp_cols=comp_cols,
        maximize=True,
        sample_seed=args.seed,
        full=args.full,
        return_model=True,
    )
    rf = result.pop("_rf")
    x_campaign = result.pop("_x_campaign")
    y_campaign = result.pop("_y_campaign")
    x_dense = result.pop("_x_dense")
    y_dense = result.pop("_y_dense")

    result["db_path"] = str(out_db.relative_to(ROOT))
    result["csv_path"] = str(out_csv.relative_to(ROOT))
    result["composition_columns"] = comp_cols
    result["optimizing_dims"] = meta["optimizing_dims"]
    result["run_uuid"] = meta["run_uuid"]
    result["source"] = meta["source"]

    if args.full:
        save_lambda_target(result, out_ela)
        print(f"Saved → {out_ela}")
        # Dedicated Tier-1 target for evolve_context / pilot configs.
        tier1 = characterize_campaign_surrogate(
            out_csv,
            objective_column=DEFAULT_OBJECTIVE,
            comp_cols=comp_cols,
            maximize=True,
            sample_seed=args.seed,
            full=False,
        )
        tier1["db_path"] = str(out_db.relative_to(ROOT))
        tier1["csv_path"] = str(out_csv.relative_to(ROOT))
        tier1["composition_columns"] = comp_cols
        tier1["optimizing_dims"] = meta["optimizing_dims"]
        tier1["run_uuid"] = meta["run_uuid"]
        tier1["source"] = meta["source"]
        save_lambda_target(tier1, out_lambda)
    else:
        save_lambda_target(result, out_lambda)
    print(f"Saved → {out_lambda}")

    meta_out = {
        **meta,
        "csv_path": str(out_csv.relative_to(ROOT)),
        "db_path": str(out_db.relative_to(ROOT)),
        "composition_columns": comp_cols,
        "n_estimators": 500,
        "random_state": 42,
        "y_campaign_range": result["y_campaign_range"],
        "y_dense_range": result["y_dense_range"],
        "n_dense_sample": result["n_dense_sample"],
        "lambda_target": str(out_lambda.relative_to(ROOT)),
        "ela_full": str(out_ela.relative_to(ROOT)) if args.full else None,
        "rf_joblib": None if args.no_rf_joblib else str(out_joblib.relative_to(ROOT)),
    }
    out_meta.write_text(json.dumps(meta_out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved → {out_meta}")

    if not args.no_rf_joblib:
        import joblib
        analysis_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "rf": rf,
                "composition_columns": comp_cols,
                "optimizing_dims": meta["optimizing_dims"],
                "dim": dim,
                "objective_column": DEFAULT_OBJECTIVE,
                "y_campaign_range": result["y_campaign_range"],
                "y_dense_range": result["y_dense_range"],
                "n_campaign": int(x_campaign.shape[0]),
                "n_dense_sample": int(x_dense.shape[0]),
                "sample_seed": args.seed,
                "csv_path": str(out_csv.relative_to(ROOT)),
                "run_uuid": meta["run_uuid"],
            },
            out_joblib,
        )
        print(f"Saved → {out_joblib}")

    print(
        f"\nRF dense scale (y_dense_range): "
        f"[{result['y_dense_range'][0]:.4f}, {result['y_dense_range'][1]:.4f}]"
    )
    print(
        f"Campaign Y: "
        f"[{result['y_campaign_range'][0]:.4f}, {result['y_campaign_range'][1]:.4f}]  "
        f"n={result['n_campaign']}  dense={result['n_dense_sample']}"
    )
    _ = (y_campaign, y_dense)


if __name__ == "__main__":
    main()
