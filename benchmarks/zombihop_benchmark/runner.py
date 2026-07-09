from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io import git_state, load_yaml, make_run_dir, observation_rows, write_csv, write_json, write_yaml
from .line_mode import LineModeOptimizerWrapper
from .metrics import compute_metrics
from .objectives.base import build_objective
from .optimizers.base import build_optimizer
from .seeding import set_global_seed
from .spaces import validate_simplex
from .types import BatchObservation


POINT_FIELDS = [
    "step",
    "point_index",
    "mode",
    "line_index",
    "line_id",
    "point_index_in_line",
    "line_num_points",
    "line_score",
    "line_score_method",
    "line_length_l2",
    "line_length_ilr",
    "is_initial_point",
    "x_expected",
    "x_actual",
    "y",
    "elapsed_s",
    "optimizer",
    "objective",
    "seed",
    "metadata",
]
METRIC_FIELDS = [
    "step",
    "line_index",
    "point_index_in_line",
    "best_y_so_far",
    "dist_to_needles",
    "pct_matched",
    "dup_fraction",
    "dist_to_needles_ilr",
    "pct_matched_ilr",
    "dup_fraction_ilr",
    "dist_to_needles_comp",
    "pct_matched_comp",
    "dup_fraction_comp",
    "dup_fraction_comp_all_points",
    "dup_fraction_comp_cross_line",
    "runtime_s",
    "num_points",
    "num_lines",
]
LINE_METRIC_FIELDS = [
    "line_index",
    "optimizer",
    "seed",
    "n_points",
    "line_id",
    "line_score",
    "line_score_method",
    "line_best_y",
    "line_mean_y",
    "line_min_y",
    "line_std_y",
    "line_length_l2",
    "line_length_ilr",
    "line_endpoint_min",
    "line_endpoint_min_left",
    "line_endpoint_min_right",
    "line_endpoint_sum_deviation",
    "line_endpoints_finite",
    "line_endpoints_nonnegative",
    "line_endpoints_normalized",
    "line_endpoints_valid_simplex",
    "line_length_l2_audit",
    "line_length_ilr_audit",
    "line_length_ilr_finite",
    "line_length_l2_coordinate_system",
    "line_length_ilr_coordinate_system",
    "line_length_l2_simplex_diameter",
    "line_length_l2_within_simplex_diameter",
    "line_adapter",
    "line_adapter_caveat",
    "runtime_s_line",
    "selected_left",
    "selected_right",
    "n_ranked_candidate_lines",
    "zombihop_internal_linebo",
    "activation",
    "zoom",
    "iteration",
    "global_iteration",
    "candidate_anchor",
    "bounds_lower",
    "bounds_upper",
    "line_budget_reached",
    "runtime_s_cumulative",
]


def run_trial(config: dict[str, Any], seed: int, repo_root: Path) -> Path:
    set_global_seed(seed)
    objective = build_objective(config["objective"])
    optimizer = build_optimizer(config["optimizer"])
    experiment = config["experiment"]
    budget = config.get("budget", {})
    line_cfg = config.get("line_mode", {})
    metrics_cfg = config.get("metrics", {})
    mode = experiment.get("mode", "point")

    points_per_line = int(
        budget.get(
            "points_per_line",
            budget.get("n_line_points", line_cfg.get("points_per_line", 24)),
        )
    )
    n_line_candidates = int(line_cfg.get("n_line_candidates", budget.get("n_line_candidates", 256)))
    n_lines = int(budget.get("n_lines", line_cfg.get("n_lines", budget.get("n_steps", 0))))
    if mode == "line" and optimizer.name != "zombihop" and not optimizer.supports_line:
        optimizer_cfg = config.get("optimizer", {})
        line_score = optimizer_cfg.get("line_score") or line_cfg.get("score")
        if line_score is None:
            line_score = "random" if optimizer.name == "random_simplex" else "mean_acq"
        optimizer = LineModeOptimizerWrapper(
            optimizer,
            points_per_line=points_per_line,
            n_line_candidates=n_line_candidates,
            line_score=line_score,
            include_endpoints=bool(line_cfg.get("include_endpoints", True)),
        )
    elif mode == "line" and hasattr(optimizer, "configure_line_mode"):
        optimizer.configure_line_mode(
            points_per_line=points_per_line,
            n_line_candidates=n_line_candidates,
            line_score=line_cfg.get("score"),
        )

    output_root = experiment.get("output_root", "benchmark_runs")
    run_dir = make_run_dir(output_root, experiment["name"], optimizer.name, objective.info.name, seed)
    write_yaml(run_dir / "config_resolved.yaml", config)
    write_json(run_dir / "git_state.json", git_state(repo_root))
    objective_metadata = _objective_metadata(objective)
    if objective_metadata:
        write_json(run_dir / "objective_metadata.json", objective_metadata)
    objective_needle_rows = _objective_needle_rows(objective)
    if objective_needle_rows:
        write_csv(run_dir / "objective_needles.csv", objective_needle_rows, _fieldnames(objective_needle_rows))
    objective_distribution_rows = _objective_distribution_rows(objective)
    if objective_distribution_rows:
        fields = _fieldnames(objective_distribution_rows)
        write_csv(run_dir / "objective_distribution.csv", objective_distribution_rows, fields)
        n_components = objective_metadata.get("n_components")
        if n_components:
            write_csv(run_dir / f"objective_distribution_{int(n_components)}d.csv", objective_distribution_rows, fields)

    n_init = int(budget.get("n_init", 5))
    X_init = objective.initial_design(n_init, seed)
    init_obs = objective.evaluate_points(X_init, seed=seed)
    validate_simplex(init_obs.X_expected)
    validate_simplex(init_obs.X_actual)

    start = time.time()
    point_rows = observation_rows(
        init_obs,
        0,
        0.0,
        optimizer.name,
        objective.info.name,
        seed,
        mode=mode,
        is_initial_point=True,
        point_offset=0,
    )
    next_point_index = len(init_obs.y)
    all_X = init_obs.X_actual.copy()
    all_y = init_obs.y.copy()
    all_line_group_ids: list[str] = [f"init_{i}" for i in range(len(init_obs.y))]
    metric_rows = [
        _metric_row(
            _compute_metrics_with_config(
                all_X,
                all_y,
                objective.info,
                metrics_cfg,
                0.0,
                0,
                all_line_group_ids,
            ),
            line_index=None,
            point_index_in_line=None,
            num_lines=0,
        )
    ]
    line_rows: list[dict[str, Any]] = []

    optimizer.initialize(init_obs.X_actual, init_obs.y, objective.info, seed)
    status = "success"
    error_message = None
    extra_summary: dict[str, Any] = {}

    try:
        if mode == "point":
            if not optimizer.supports_point:
                raise RuntimeError(f"Optimizer {optimizer.name} does not support point mode")
            n_steps = int(budget.get("n_steps", 20))
            for step in range(1, n_steps + 1):
                X_next = optimizer.suggest(1)
                obs = objective.evaluate_points(X_next, seed=seed + step)
                validate_simplex(obs.X_expected)
                validate_simplex(obs.X_actual)
                optimizer.observe(obs)
                elapsed = time.time() - start
                point_rows.extend(
                    observation_rows(
                        obs,
                        step,
                        elapsed,
                        optimizer.name,
                        objective.info.name,
                        seed,
                        mode=mode,
                        is_initial_point=False,
                        point_offset=next_point_index,
                    )
                )
                next_point_index += len(obs.y)
                all_X = np.vstack([all_X, obs.X_actual])
                all_y = np.concatenate([all_y, obs.y])
                all_line_group_ids.extend(f"point_{step}_{i}" for i in range(len(obs.y)))
                metric_rows.append(
                    _metric_row(
                        _compute_metrics_with_config(
                            all_X,
                            all_y,
                            objective.info,
                            metrics_cfg,
                            elapsed,
                            step,
                            all_line_group_ids,
                        ),
                        line_index=None,
                        point_index_in_line=None,
                        num_lines=0,
                    )
                )
        elif mode == "line" and optimizer.name == "zombihop":
            if n_lines <= 0:
                raise ValueError("ZoMBI-Hop line mode requires budget.n_lines or line_mode.n_lines")
            result = optimizer.run_full_trial(
                objective=objective,
                X_init_actual=init_obs.X_actual,
                X_init_expected=init_obs.X_expected,
                Y_init=init_obs.y,
                run_dir=run_dir,
                seed=seed,
                n_line_budget=n_lines,
                points_per_line=points_per_line,
            )
            elapsed = result["runtime_s"]
            extra_summary = {
                "runtime_s": float(elapsed),
                "line_budget_requested": result.get("line_budget_requested"),
                "line_budget_reached": bool(result.get("line_budget_reached", False)),
                "zombihop_internal_linebo": True,
            }
            line_observations = result.get("line_observations", [])
            for line_record, obs in zip(result.get("line_records", []), line_observations):
                line_index = int(line_record["line_index"])
                validate_simplex(obs.X_expected)
                validate_simplex(obs.X_actual)
                line_metadata = obs.metadata.get("line", {})
                line_elapsed = _to_float_or_default(line_record.get("runtime_s_cumulative"), elapsed)

                point_rows.extend(
                    observation_rows(
                        obs,
                        line_index,
                        line_elapsed,
                        optimizer.name,
                        objective.info.name,
                        seed,
                        line_index=line_index,
                        mode=mode,
                        line_metadata=line_metadata,
                        is_initial_point=False,
                        point_offset=next_point_index,
                    )
                )
                next_point_index += len(obs.y)

                X_before_line = all_X
                y_before_line = all_y
                groups_before_line = list(all_line_group_ids)
                line_group_id = str(line_metadata.get("line_id") or f"zombihop_line_{line_index}")
                for point_index_in_line in range(len(obs.y)):
                    X_prefix = np.vstack([X_before_line, obs.X_actual[: point_index_in_line + 1]])
                    y_prefix = np.concatenate([y_before_line, obs.y[: point_index_in_line + 1]])
                    group_prefix = groups_before_line + [line_group_id] * (point_index_in_line + 1)
                    metric_rows.append(
                        _metric_row(
                            _compute_metrics_with_config(
                                X_prefix,
                                y_prefix,
                                objective.info,
                                metrics_cfg,
                                line_elapsed,
                                line_index,
                                group_prefix,
                            ),
                            line_index=line_index,
                            point_index_in_line=point_index_in_line,
                            num_lines=line_index,
                        )
                    )

                all_X = np.vstack([all_X, obs.X_actual])
                all_y = np.concatenate([all_y, obs.y])
                all_line_group_ids.extend([line_group_id] * len(obs.y))
                line_rows.append(line_record)
            write_json(run_dir / "zombihop_result.json", result)
        elif mode == "line":
            if not optimizer.supports_line or not hasattr(optimizer, "suggest_line"):
                raise RuntimeError(f"Optimizer {optimizer.name} does not support line mode")
            if n_lines <= 0:
                raise ValueError("line mode requires budget.n_lines or line_mode.n_lines to be positive")
            for line_index in range(1, n_lines + 1):
                line_start = time.time()
                line = optimizer.suggest_line()
                obs_raw = objective.evaluate_points(line.points, seed=seed + 10_000 + line_index)
                line_metadata = line.metadata()
                obs = _observation_with_metadata(obs_raw, {"line": line_metadata})
                validate_simplex(obs.X_expected)
                validate_simplex(obs.X_actual)
                # Fairness rule: the base optimizer sees the line only after all points are evaluated.
                optimizer.observe(obs)
                elapsed = time.time() - start
                runtime_s_line = time.time() - line_start

                point_rows.extend(
                    observation_rows(
                        obs,
                        line_index,
                        elapsed,
                        optimizer.name,
                        objective.info.name,
                        seed,
                        line_index=line_index,
                        mode=mode,
                        line_metadata=line_metadata,
                        is_initial_point=False,
                        point_offset=next_point_index,
                    )
                )
                next_point_index += len(obs.y)

                X_before_line = all_X
                y_before_line = all_y
                groups_before_line = list(all_line_group_ids)
                line_group_id = str(line_metadata.get("line_id") or f"line_{line_index}")
                for point_index_in_line in range(len(obs.y)):
                    X_prefix = np.vstack([X_before_line, obs.X_actual[: point_index_in_line + 1]])
                    y_prefix = np.concatenate([y_before_line, obs.y[: point_index_in_line + 1]])
                    group_prefix = groups_before_line + [line_group_id] * (point_index_in_line + 1)
                    metric_rows.append(
                        _metric_row(
                            _compute_metrics_with_config(
                                X_prefix,
                                y_prefix,
                                objective.info,
                                metrics_cfg,
                                elapsed,
                                line_index,
                                group_prefix,
                            ),
                            line_index=line_index,
                            point_index_in_line=point_index_in_line,
                            num_lines=line_index,
                        )
                    )

                all_X = np.vstack([all_X, obs.X_actual])
                all_y = np.concatenate([all_y, obs.y])
                all_line_group_ids.extend([line_group_id] * len(obs.y))
                line_rows.append(
                    {
                        "line_index": line_index,
                        "optimizer": optimizer.name,
                        "seed": seed,
                        "n_points": int(len(obs.y)),
                        "line_id": line_metadata["line_id"],
                        "line_score": line_metadata["line_score"],
                        "line_score_method": line_metadata["line_score_method"],
                        "line_best_y": float(np.max(obs.y)),
                        "line_mean_y": float(np.mean(obs.y)),
                        "line_min_y": float(np.min(obs.y)),
                        "line_std_y": float(np.std(obs.y)),
                        "line_length_l2": line_metadata["line_length_l2"],
                        "line_length_ilr": line_metadata["line_length_ilr"],
                        **_line_metadata_fields(line_metadata),
                        "runtime_s_line": runtime_s_line,
                        "selected_left": _endpoint_json(line_metadata, 0),
                        "selected_right": _endpoint_json(line_metadata, 1),
                        "n_ranked_candidate_lines": line_metadata.get("n_ranked_candidate_lines", ""),
                        "candidate_anchor": _json_or_empty(line_metadata.get("candidate_anchor", "")),
                    }
                )
        else:
            raise RuntimeError(f"Mode {mode!r} is not implemented for optimizer {optimizer.name!r}")
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"[benchmark] {error_message}", file=sys.stderr)
        if optimizer.name != "zombihop":
            raise

    write_csv(run_dir / "points.csv", point_rows, POINT_FIELDS)
    write_csv(run_dir / "metrics_over_time.csv", metric_rows, METRIC_FIELDS)
    if mode == "line":
        write_csv(run_dir / "line_metrics.csv", line_rows, LINE_METRIC_FIELDS)
    line_summary = _line_summary(line_rows)
    summary = {
        "status": status,
        "error": error_message,
        "experiment": experiment["name"],
        "mode": mode,
        "optimizer": optimizer.name,
        "objective": objective.info.name,
        "seed": seed,
        "run_dir": str(run_dir),
        "num_points": int(len(all_y)),
        "num_lines": int(line_summary["num_lines"]),
        "points_per_line": points_per_line if mode == "line" else None,
        "best_y": float(np.max(all_y)) if len(all_y) else None,
        **line_summary,
        **extra_summary,
        **objective_metadata,
        "optimizer_state": optimizer.get_state(),
    }
    write_json(run_dir / "summary.json", summary)
    if status == "failed":
        raise RuntimeError(error_message)
    return run_dir


def _metric_row(
    row: dict[str, Any],
    line_index: int | None,
    point_index_in_line: int | None,
    num_lines: int,
) -> dict[str, Any]:
    out = dict(row)
    out["line_index"] = "" if line_index is None else int(line_index)
    out["point_index_in_line"] = "" if point_index_in_line is None else int(point_index_in_line)
    out["num_lines"] = int(num_lines)
    return out


def _compute_metrics_with_config(
    X: np.ndarray,
    y: np.ndarray,
    objective_info,
    metrics_cfg: dict[str, Any],
    runtime_s: float,
    step: int,
    line_group_ids: list[str] | None = None,
) -> dict[str, Any]:
    return compute_metrics(
        X,
        y,
        objective_info,
        duplicate_radius_ilr=float(metrics_cfg.get("duplicate_radius_ilr", 0.03)),
        match_radius_ilr=metrics_cfg.get("match_radius_ilr", objective_info.match_radius_ilr),
        runtime_s=runtime_s,
        step=step,
        duplicate_radius_comp=float(metrics_cfg.get("duplicate_radius_comp", 0.032)),
        match_radius_comp=metrics_cfg.get("match_radius_comp", objective_info.match_radius_comp),
        line_group_ids=line_group_ids,
    )


def _observation_with_metadata(obs: BatchObservation, extra_metadata: dict[str, Any]) -> BatchObservation:
    metadata = dict(obs.metadata)
    metadata.update(extra_metadata)
    return BatchObservation(
        X_expected=obs.X_expected,
        X_actual=obs.X_actual,
        y=obs.y,
        metadata=metadata,
    )


def _to_float_or_default(value: Any, default: float) -> float:
    try:
        if value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _objective_metadata(objective) -> dict[str, Any]:
    if hasattr(objective, "get_metadata"):
        metadata = objective.get_metadata()
        if isinstance(metadata, dict):
            return metadata
    return {}


def _objective_needle_rows(objective) -> list[dict[str, Any]]:
    if hasattr(objective, "true_needle_rows"):
        rows = objective.true_needle_rows()
        if isinstance(rows, list):
            return rows
    return []


def _objective_distribution_rows(objective) -> list[dict[str, Any]]:
    if hasattr(objective, "objective_distribution_rows"):
        rows = objective.objective_distribution_rows()
        if isinstance(rows, list):
            return rows
    return []


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def _line_metadata_fields(line_metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "line_endpoint_min",
        "line_endpoint_min_left",
        "line_endpoint_min_right",
        "line_endpoint_sum_deviation",
        "line_endpoints_finite",
        "line_endpoints_nonnegative",
        "line_endpoints_normalized",
        "line_endpoints_valid_simplex",
        "line_length_l2_audit",
        "line_length_ilr_audit",
        "line_length_ilr_finite",
        "line_length_l2_coordinate_system",
        "line_length_ilr_coordinate_system",
        "line_length_l2_simplex_diameter",
        "line_length_l2_within_simplex_diameter",
        "line_adapter",
        "line_adapter_caveat",
    ]
    return {key: line_metadata.get(key, "") for key in keys}


def _endpoint_json(line_metadata: dict[str, Any], index: int) -> str:
    endpoints = line_metadata.get("endpoints")
    try:
        return json.dumps(endpoints[index])
    except Exception:
        return ""


def _json_or_empty(value: Any) -> str:
    if value == "":
        return ""
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def _line_summary(line_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not line_rows:
        return {
            "num_lines": 0,
            "line_best_y_mean": None,
            "line_best_y_max": None,
            "line_length_l2_mean": None,
            "line_length_ilr_mean": None,
        }
    line_best = np.asarray([row["line_best_y"] for row in line_rows], dtype=float)
    line_l2 = np.asarray([row["line_length_l2"] for row in line_rows], dtype=float)
    line_ilr = np.asarray([row["line_length_ilr"] for row in line_rows], dtype=float)
    return {
        "num_lines": int(len(line_rows)),
        "line_best_y_mean": float(np.mean(line_best)),
        "line_best_y_max": float(np.max(line_best)),
        "line_length_l2_mean": float(np.mean(line_l2)),
        "line_length_ilr_mean": float(np.mean(line_ilr)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ZoMBI-Hop benchmark smoke trials.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_yaml(config_path)
    repo_root = Path.cwd()
    run_dirs = []
    for seed in config.get("experiment", {}).get("seeds", [0]):
        run_dirs.append(run_trial(config, int(seed), repo_root))
    for run_dir in run_dirs:
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

