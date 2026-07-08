import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.zombihop_benchmark.line_mode import (
    LineModeOptimizerWrapper,
    generate_simplex_line_candidates,
)
from benchmarks.zombihop_benchmark.objectives.synthetic import SyntheticSimplexObjective
from benchmarks.zombihop_benchmark.runner import run_trial
from benchmarks.zombihop_benchmark.spaces import validate_simplex
from benchmarks.zombihop_benchmark.suite import run_suite


def _objective_config():
    return {
        "kind": "synthetic_simplex",
        "name": "synthetic_3d_planted",
        "n_components": 3,
        "maximize": True,
        "params": {"n_needles": 2, "basin_width": 20.0, "noise_std": 0.0, "seed": 123},
    }


def test_line_generator_outputs_valid_simplex_points_and_is_deterministic():
    first = generate_simplex_line_candidates(4, 3, 6, seed=17)
    second = generate_simplex_line_candidates(4, 3, 6, seed=17)

    assert len(first) == 4
    assert first[0].points.shape == (6, 3)
    assert validate_simplex(first[0].points)
    assert first[0].length_l2 >= 0.0
    assert first[0].length_ilr >= 0.0
    np.testing.assert_allclose(first[0].points, second[0].points)
    np.testing.assert_allclose(first[0].endpoints, second[0].endpoints)


def test_line_wrapper_updates_base_optimizer_with_whole_batch():
    class RecordingOptimizer:
        name = "recording"
        supports_point = True
        supports_line = False

        def __init__(self):
            self.batch_lengths = []
            self.score_calls = 0

        def initialize(self, X, y, objective_info, seed):
            self.n_components = objective_info.n_components

        def score_candidates(self, X_candidates):
            self.score_calls += 1
            return np.asarray(X_candidates)[:, 0]

        def observe(self, obs):
            self.batch_lengths.append(len(obs.y))

        def get_state(self):
            return {"batch_lengths": self.batch_lengths}

    objective = SyntheticSimplexObjective(n_components=3, n_needles=2, seed=123)
    init = objective.evaluate_points(objective.initial_design(5, seed=0), seed=0)
    base = RecordingOptimizer()
    wrapper = LineModeOptimizerWrapper(base, points_per_line=4, n_line_candidates=5, line_score="mean_acq")
    wrapper.initialize(init.X_actual, init.y, objective.info, seed=5)

    line = wrapper.suggest_line()
    obs = objective.evaluate_points(line.points, seed=1)
    wrapper.observe(obs)

    assert base.score_calls == 1
    assert base.batch_lengths == [4]


def test_random_line_smoke_run_writes_points_line_metrics_and_summary(tmp_path):
    config = {
        "experiment": {"name": "test_random_line", "mode": "line", "seeds": [0], "output_root": str(tmp_path)},
        "objective": _objective_config(),
        "optimizer": {"kind": "random_simplex", "line_score": "random", "params": {}},
        "budget": {"n_init": 3, "n_lines": 2, "points_per_line": 4},
        "line_mode": {"n_line_candidates": 8, "score": "random", "include_endpoints": True},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
    }

    run_dir = run_trial(config, seed=0, repo_root=Path.cwd())

    with open(run_dir / "points.csv", newline="", encoding="utf-8") as f:
        point_rows = list(csv.DictReader(f))
    with open(run_dir / "line_metrics.csv", newline="", encoding="utf-8") as f:
        line_rows = list(csv.DictReader(f))
    with open(run_dir / "metrics_over_time.csv", newline="", encoding="utf-8") as f:
        metric_rows = list(csv.DictReader(f))
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(point_rows) == 11
    assert len(line_rows) == 2
    assert len(metric_rows) == 9
    assert point_rows[3]["line_index"] == "1"
    assert point_rows[3]["point_index_in_line"] == "0"
    assert point_rows[3]["is_initial_point"] == "False"
    assert summary["mode"] == "line"
    assert summary["num_lines"] == 2
    assert summary["points_per_line"] == 4


def test_gp_line_smoke_run_writes_line_metrics(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    config = {
        "experiment": {"name": "test_gp_line", "mode": "line", "seeds": [0], "output_root": str(tmp_path)},
        "objective": _objective_config(),
        "optimizer": {
            "kind": "gp_ard_ei",
            "line_score": "mean_acq",
            "params": {"candidate_pool_size": 16, "device": "cpu", "dtype": "float64"},
        },
        "budget": {"n_init": 5, "n_lines": 1, "points_per_line": 3},
        "line_mode": {"n_line_candidates": 4, "score": "mean_acq", "include_endpoints": True},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
    }

    run_dir = run_trial(config, seed=0, repo_root=Path.cwd())

    with open(run_dir / "line_metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["line_score_method"] == "mean_acq"


def test_line_suite_aggregation_writes_line_metrics_long(tmp_path):
    config = {
        "experiment": {"name": "test_line_suite", "mode": "line", "seeds": [0, 1], "output_root": str(tmp_path)},
        "objective": _objective_config(),
        "budget": {"n_init": 3, "n_lines": 2, "points_per_line": 4},
        "line_mode": {"n_line_candidates": 8, "score": "mean_acq", "include_endpoints": True},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
        "optimizers": [{"kind": "random_simplex", "line_score": "random", "params": {}}],
    }

    aggregate_dir = run_suite(config, repo_root=Path.cwd())

    assert (aggregate_dir / "line_metrics_long.csv").exists()
    with open(aggregate_dir / "line_metrics_long.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    with open(aggregate_dir / "summary_by_optimizer.csv", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert summary_rows[0]["num_lines_mean"] == "2.0"
