from __future__ import annotations

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import load_yaml, write_csv, write_json, write_yaml
from .runner import run_trial


FINAL_METRICS = [
    "best_y_so_far",
    "dist_to_needles",
    "pct_matched",
    "dup_fraction",
    "runtime_s",
    "num_points",
    "num_lines",
]


def run_suite(config: dict[str, Any], repo_root: Path) -> Path:
    experiment = config["experiment"]
    suite_name = experiment["name"]
    output_root = Path(experiment.get("output_root", "benchmark_runs"))
    aggregate_dir = output_root / suite_name / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(aggregate_dir / "suite_config_resolved.yaml", config)

    run_index: list[dict[str, Any]] = []
    metrics_long: list[dict[str, Any]] = []
    line_metrics_long: list[dict[str, Any]] = []
    final_metrics: list[dict[str, Any]] = []
    started = time.time()

    for optimizer_config in config.get("optimizers", []):
        for seed in experiment.get("seeds", [0]):
            trial_config = _trial_config(config, optimizer_config)
            optimizer_kind = optimizer_config.get("kind", "unknown")
            status = "success"
            error = ""
            run_dir: Path | None = None
            try:
                run_dir = run_trial(trial_config, int(seed), repo_root)
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            summary = _load_summary(run_dir) if run_dir is not None else {}
            optimizer_name = summary.get("optimizer", optimizer_kind)
            row = {
                "optimizer": optimizer_name,
                "optimizer_kind": optimizer_kind,
                "seed": int(seed),
                "run_dir": "" if run_dir is None else str(run_dir),
                "status": status if not summary else summary.get("status", status),
                "error": error if error else summary.get("error", ""),
                "num_lines": summary.get("num_lines", ""),
                "num_points": summary.get("num_points", ""),
                "line_budget_requested": summary.get("line_budget_requested", ""),
                "line_budget_reached": summary.get("line_budget_reached", ""),
            }
            run_index.append(row)
            if run_dir is not None:
                rows = _read_csv(run_dir / "metrics_over_time.csv")
                for metric_row in rows:
                    metrics_long.append(
                        {
                            "optimizer": optimizer_name,
                            "optimizer_kind": optimizer_kind,
                            "seed": int(seed),
                            "run_dir": str(run_dir),
                            **metric_row,
                        }
                    )
                if rows:
                    final_metrics.append(
                        {
                            "optimizer": optimizer_name,
                            "optimizer_kind": optimizer_kind,
                            "seed": int(seed),
                            "run_dir": str(run_dir),
                            **rows[-1],
                        }
                    )
                line_rows = _read_csv(run_dir / "line_metrics.csv")
                for line_row in line_rows:
                    line_metrics_long.append(
                        {
                            "optimizer": optimizer_name,
                            "optimizer_kind": optimizer_kind,
                            "seed": int(seed),
                            "run_dir": str(run_dir),
                            **line_row,
                        }
                    )

    summary_by_optimizer = _summary_by_optimizer(final_metrics)
    write_csv(aggregate_dir / "run_index.csv", run_index, _fieldnames(run_index))
    write_csv(aggregate_dir / "metrics_over_time_long.csv", metrics_long, _fieldnames(metrics_long))
    write_csv(aggregate_dir / "line_metrics_long.csv", line_metrics_long, _fieldnames(line_metrics_long))
    write_csv(aggregate_dir / "final_metrics.csv", final_metrics, _fieldnames(final_metrics))
    write_csv(aggregate_dir / "summary_by_optimizer.csv", summary_by_optimizer, _fieldnames(summary_by_optimizer))
    write_json(
        aggregate_dir / "summary.json",
        {
            "suite": suite_name,
            "status": "success" if all(r["status"] == "success" for r in run_index) else "partial_failure",
            "num_runs": len(run_index),
            "num_success": sum(1 for r in run_index if r["status"] == "success"),
            "runtime_s": time.time() - started,
            "aggregate_dir": str(aggregate_dir),
        },
    )
    return aggregate_dir


def _trial_config(config: dict[str, Any], optimizer_config: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(config)
    out["optimizer"] = deepcopy(optimizer_config)
    out.pop("optimizers", None)
    return out


def _load_summary(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def _summary_by_optimizer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["optimizer"], []).append(row)
    out = []
    for optimizer, group in sorted(groups.items()):
        summary: dict[str, Any] = {"optimizer": optimizer, "n_runs": len(group)}
        for metric in FINAL_METRICS:
            values = [_to_float(row.get(metric)) for row in group]
            values = [v for v in values if not math.isnan(v)]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_std"] = _std(values)
            summary[f"{metric}_median"] = _median(values)
        out.append(summary)
    return out


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _mean(values: list[float]) -> float:
    return math.nan if not values else sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else math.nan
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a ZoMBI-Hop benchmark optimizer suite.")
    parser.add_argument("--config", required=True, help="Path to suite YAML config")
    args = parser.parse_args(argv)
    aggregate_dir = run_suite(load_yaml(args.config), Path.cwd())
    print(aggregate_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
