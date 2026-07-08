import csv
import json
from pathlib import Path

import pytest

from benchmarks.zombihop_benchmark.runner import run_trial
from benchmarks.zombihop_benchmark.suite import run_suite


pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")


def _objective_config():
    return {
        "kind": "synthetic_simplex",
        "name": "synthetic_3d_planted",
        "n_components": 3,
        "maximize": True,
        "params": {
            "n_needles": 2,
            "basin_width": 20.0,
            "noise_std": 0.0,
            "seed": 123,
            "match_radius_ilr": 0.25,
        },
    }


def _zombihop_optimizer_config():
    return {
        "kind": "zombihop",
        "params": {
            "device": "cpu",
            "dtype": "float64",
            "max_activations": 5,
            "max_zooms": 1,
            "max_iterations": 5,
            "n_restarts": 3,
            "raw": 32,
            "max_gp_points": 128,
            "acquisition_type": "ucb",
            "ucb_beta": 0.1,
            "num_lines": 4,
            "checkpoint_subdir": "checkpoints",
        },
    }


def test_zombihop_line_smoke_writes_line_metrics_and_exact_budget(tmp_path):
    config = {
        "experiment": {"name": "test_zombihop_line", "mode": "line", "seeds": [0], "output_root": str(tmp_path)},
        "objective": _objective_config(),
        "optimizer": _zombihop_optimizer_config(),
        "budget": {"n_init": 5, "n_lines": 2, "points_per_line": 5},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
    }

    run_dir = run_trial(config, seed=0, repo_root=Path.cwd())

    with open(run_dir / "points.csv", newline="", encoding="utf-8") as f:
        point_rows = list(csv.DictReader(f))
    with open(run_dir / "line_metrics.csv", newline="", encoding="utf-8") as f:
        line_rows = list(csv.DictReader(f))
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(point_rows) == 15
    assert len(line_rows) == 2
    assert summary["optimizer"] == "zombihop"
    assert summary["mode"] == "line"
    assert summary["num_lines"] == 2
    assert summary["line_budget_requested"] == 2
    assert summary["line_budget_reached"] is True
    assert summary["points_per_line"] == 5
    assert line_rows[0]["zombihop_internal_linebo"] == "True"
    assert line_rows[0]["selected_left"]
    assert point_rows[5]["line_index"] == "1"
    assert point_rows[5]["point_index_in_line"] == "0"
    assert point_rows[5]["is_initial_point"] == "False"


def test_zombihop_line_suite_aggregates(tmp_path):
    config = {
        "experiment": {"name": "test_zombihop_line_suite", "mode": "line", "seeds": [0], "output_root": str(tmp_path)},
        "objective": _objective_config(),
        "budget": {"n_init": 5, "n_lines": 1, "points_per_line": 5},
        "line_mode": {"n_line_candidates": 8, "score": "mean_acq", "include_endpoints": True},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
        "optimizers": [
            {"kind": "random_simplex", "line_score": "random", "params": {}},
            _zombihop_optimizer_config(),
        ],
    }

    aggregate_dir = run_suite(config, repo_root=Path.cwd())

    with open(aggregate_dir / "line_metrics_long.csv", newline="", encoding="utf-8") as f:
        line_rows = list(csv.DictReader(f))
    with open(aggregate_dir / "run_index.csv", newline="", encoding="utf-8") as f:
        run_rows = list(csv.DictReader(f))
    assert len(line_rows) == 2
    assert any(row["optimizer"] == "zombihop" for row in line_rows)
    zombi_run = [row for row in run_rows if row["optimizer"] == "zombihop"][0]
    assert zombi_run["line_budget_reached"] == "True"
