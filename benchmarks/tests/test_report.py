import csv
import json
from pathlib import Path

import pandas as pd
import yaml

from benchmarks.zombihop_benchmark.report import (
    build_final_metrics_by_run,
    compute_dimension_rank_delta,
    compute_transfer_rank_delta,
    load_aggregate,
    run_report,
)


def test_report_loader_handles_point_aggregate_without_line_metrics(tmp_path):
    aggregate_dir = _write_aggregate(tmp_path, "synthetic_point", mode="point", family="synthetic")
    suite_config = _write_suite_config(tmp_path, "synthetic_point", mode="point", kind="synthetic_simplex")

    loaded = load_aggregate(
        {
            "label": "synthetic_3d_point",
            "suite_config": str(suite_config),
            "aggregate_dir": str(aggregate_dir),
            "objective_family": "synthetic",
        },
        repo_root=Path.cwd(),
    )

    assert loaded.mode == "point"
    assert loaded.objective_family == "synthetic"
    assert len(loaded.run_index) == 2
    assert loaded.line_metrics.empty


def test_report_loader_handles_line_aggregate_with_line_metrics(tmp_path):
    aggregate_dir = _write_aggregate(tmp_path, "real_line", mode="line", family="real_rf", include_line=True)
    suite_config = _write_suite_config(tmp_path, "real_line", mode="line", kind="real_rf_surrogate")

    loaded = load_aggregate(
        {
            "label": "real_rf_3d_line",
            "suite_config": str(suite_config),
            "aggregate_dir": str(aggregate_dir),
            "objective_family": "real_rf",
        },
        repo_root=Path.cwd(),
    )

    assert loaded.mode == "line"
    assert loaded.objective_kind == "real_rf_surrogate"
    assert len(loaded.line_metrics) == 4


def test_final_metric_aggregation_produces_expected_columns(tmp_path):
    aggregate_dir = _write_aggregate(tmp_path, "synthetic_line", mode="line", family="synthetic", include_line=True)
    suite_config = _write_suite_config(tmp_path, "synthetic_line", mode="line", kind="synthetic_simplex")
    loaded = load_aggregate(
        {
            "label": "synthetic_3d_line",
            "suite_config": str(suite_config),
            "aggregate_dir": str(aggregate_dir),
            "objective_family": "synthetic",
        },
        repo_root=Path.cwd(),
    )

    final_by_run = build_final_metrics_by_run([loaded])

    assert len(final_by_run) == 2
    assert "final_best_y_so_far" in final_by_run.columns
    assert "line_best_y_mean" in final_by_run.columns
    assert final_by_run["line_best_y_max"].notna().all()


def test_transfer_rank_delta_uses_metric_directionality(tmp_path):
    aggregate_dirs = [
        _write_aggregate(tmp_path, "synthetic_point", mode="point", family="synthetic", values=(0.9, 0.8)),
        _write_aggregate(tmp_path, "real_point", mode="point", family="real_rf", values=(0.7, 0.95)),
    ]
    suite_configs = [
        _write_suite_config(tmp_path, "synthetic_point", mode="point", kind="synthetic_simplex"),
        _write_suite_config(tmp_path, "real_point", mode="point", kind="real_rf_surrogate"),
    ]
    loaded = [
        load_aggregate(
            {
                "label": label,
                "suite_config": str(suite_config),
                "aggregate_dir": str(aggregate_dir),
                "objective_family": family,
            },
            repo_root=Path.cwd(),
        )
        for label, suite_config, aggregate_dir, family in [
            ("synthetic_3d_point", suite_configs[0], aggregate_dirs[0], "synthetic"),
            ("real_rf_3d_point", suite_configs[1], aggregate_dirs[1], "real_rf"),
        ]
    ]
    final_by_run = build_final_metrics_by_run(loaded)

    delta = compute_transfer_rank_delta(
        final_by_run,
        {"higher_is_better": ["best_y_so_far"], "lower_is_better": ["dist_to_needles"]},
    )

    best_y = delta[(delta["metric"] == "final_best_y_so_far") & (delta["optimizer"] == "random_simplex")].iloc[0]
    assert best_y["synthetic_rank"] == 1
    assert best_y["real_rank"] == 2
    assert best_y["rank_delta"] == 1


def test_dimension_report_writes_3d_to_4d_delta_tables(tmp_path):
    inputs = []
    for label, n_components, values in [
        ("realistic_ackley_3d_point", 3, (0.9, 0.8)),
        ("realistic_ackley_4d_point", 4, (0.7, 0.95)),
    ]:
        aggregate_dir = _write_aggregate(tmp_path, label, mode="point", family="synthetic", values=values)
        suite_config = _write_suite_config(
            tmp_path,
            label,
            mode="point",
            kind="realistic_ackley_simplex",
            n_components=n_components,
        )
        inputs.append(
            {
                "label": label,
                "objective_family": "synthetic",
                "objective_kind": "realistic_ackley_simplex",
                "objective": f"realistic_ackley_{n_components}d",
                "n_components": n_components,
                "suite_config": str(suite_config),
                "aggregate_dir": str(aggregate_dir),
            }
        )
    loaded = [load_aggregate(input_cfg, repo_root=Path.cwd()) for input_cfg in inputs]
    final_by_run = build_final_metrics_by_run(loaded)

    delta = compute_dimension_rank_delta(final_by_run, {"comparison_axis": "n_components"})
    best_y = delta[(delta["metric"] == "final_best_y_so_far") & (delta["optimizer"] == "random_simplex")].iloc[0]
    assert best_y["base_n_components"] == 3
    assert best_y["target_n_components"] == 4
    assert best_y["base_rank"] == 1
    assert best_y["target_rank"] == 2

    report_dir = run_report(
        {
            "name": "test_dimension_report",
            "output_root": str(tmp_path / "reports"),
            "comparison_axis": "n_components",
            "inputs": inputs,
            "plots": {"include_seed_traces": False, "include_iqr_band": True, "dpi": 80},
        },
        repo_root=Path.cwd(),
    )

    assert (report_dir / "dimension_rank_delta.csv").exists()
    assert (report_dir / "dimension_metric_delta.csv").exists()
    assert "3D-to-4D" in (report_dir / "report.md").read_text(encoding="utf-8")


def test_report_cli_core_writes_markdown_csvs_and_plots(tmp_path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    inputs = []
    for label, mode, family, kind, include_line, values in [
        ("synthetic_3d_point", "point", "synthetic", "synthetic_simplex", False, (0.9, 0.8)),
        ("real_rf_3d_point", "point", "real_rf", "real_rf_surrogate", False, (0.7, 0.95)),
        ("synthetic_3d_line", "line", "synthetic", "synthetic_simplex", True, (0.85, 0.82)),
        ("real_rf_3d_line", "line", "real_rf", "real_rf_surrogate", True, (0.75, 0.93)),
    ]:
        aggregate_dir = _write_aggregate(tmp_path, label, mode=mode, family=family, include_line=include_line, values=values)
        suite_config = _write_suite_config(tmp_path, label, mode=mode, kind=kind)
        inputs.append(
            {
                "label": label,
                "objective_family": family,
                "suite_config": str(suite_config),
                "aggregate_dir": str(aggregate_dir),
            }
        )
    config = {
        "name": "test_report",
        "output_root": str(tmp_path / "reports"),
        "inputs": inputs,
        "plots": {"include_seed_traces": False, "include_iqr_band": True, "dpi": 80},
    }

    report_dir = run_report(config, repo_root=Path.cwd())

    assert (report_dir / "report.md").exists()
    assert (report_dir / "report_manifest.json").exists()
    assert (report_dir / "final_metrics_by_run.csv").exists()
    assert (report_dir / "transfer_rank_delta.csv").exists()
    assert list((report_dir / "plots").glob("*.png"))


def _write_suite_config(tmp_path: Path, name: str, mode: str, kind: str, n_components: int = 3) -> Path:
    path = tmp_path / f"{name}_suite.yaml"
    objective_name = "real_3d_perovskite_rf" if kind == "real_rf_surrogate" else f"realistic_ackley_{n_components}d"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "experiment": {"name": name, "mode": mode},
                "objective": {"kind": kind, "name": objective_name, "n_components": n_components},
            },
            f,
        )
    return path


def _write_aggregate(
    tmp_path: Path,
    name: str,
    mode: str,
    family: str,
    include_line: bool = False,
    values: tuple[float, float] = (0.8, 0.9),
) -> Path:
    aggregate_dir = tmp_path / name / "aggregate"
    aggregate_dir.mkdir(parents=True)
    optimizers = ["random_simplex", "gp_ard_ei"]
    run_rows = []
    metric_rows = []
    final_rows = []
    line_rows = []
    for opt_index, optimizer in enumerate(optimizers):
        value = values[opt_index]
        run_dir = str(tmp_path / name / f"{optimizer}_seed0")
        run_rows.append(
            {
                "optimizer": optimizer,
                "optimizer_kind": optimizer,
                "seed": 0,
                "run_dir": run_dir,
                "status": "success",
                "error": "",
                "num_lines": 2 if mode == "line" else 0,
                "num_points": 6,
            }
        )
        for step in range(3):
            metric_rows.append(
                {
                    "optimizer": optimizer,
                    "optimizer_kind": optimizer,
                    "seed": 0,
                    "run_dir": run_dir,
                    "step": step,
                    "line_index": step if mode == "line" else "",
                    "best_y_so_far": value - 0.1 + step * 0.05,
                    "dist_to_needles": 1.0 - value + 0.05 * (2 - step),
                    "pct_matched": step / 2,
                    "dup_fraction": 0.2 - step * 0.05,
                    "runtime_s": 0.5 + opt_index + step,
                    "num_points": 4 + step,
                    "num_lines": step if mode == "line" else 0,
                }
            )
        final_rows.append(metric_rows[-1].copy())
        final_rows[-1].update(
            {
                "optimizer": optimizer,
                "optimizer_kind": optimizer,
                "seed": 0,
                "run_dir": run_dir,
            }
        )
        if include_line:
            for line_index in [1, 2]:
                line_rows.append(
                    {
                        "optimizer": optimizer,
                        "optimizer_kind": optimizer,
                        "seed": 0,
                        "run_dir": run_dir,
                        "line_index": line_index,
                        "line_best_y": value + 0.01 * line_index,
                        "line_mean_y": value - 0.02,
                        "line_length_l2": 0.4 + 0.1 * opt_index,
                        "line_length_ilr": 0.5 + 0.1 * opt_index,
                        "runtime_s_line": 0.1,
                    }
                )

    _write_csv(aggregate_dir / "run_index.csv", run_rows)
    _write_csv(aggregate_dir / "metrics_over_time_long.csv", metric_rows)
    _write_csv(aggregate_dir / "final_metrics.csv", final_rows)
    if include_line:
        _write_csv(aggregate_dir / "line_metrics_long.csv", line_rows)
    with open(aggregate_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"suite": name, "family": family, "status": "success"}, f)
    return aggregate_dir


def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
