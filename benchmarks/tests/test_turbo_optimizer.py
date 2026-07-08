import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.zombihop_benchmark.objectives.synthetic import SyntheticSimplexObjective
from benchmarks.zombihop_benchmark.optimizers.base import build_optimizer
from benchmarks.zombihop_benchmark.optimizers.turbo_optimizer import (
    TuRBOOptimizer,
    TurboTrustRegionState,
)
from benchmarks.zombihop_benchmark.runner import run_trial
from benchmarks.zombihop_benchmark.spaces import validate_simplex
from benchmarks.zombihop_benchmark.suite import run_suite
from benchmarks.zombihop_benchmark.types import BatchObservation


pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")


def test_turbo_registry_returns_local_adapter():
    optimizer = build_optimizer({"kind": "turbo", "params": {"candidate_pool_size": 16}})

    assert isinstance(optimizer, TuRBOOptimizer)
    assert optimizer.name == "turbo"


def test_turbo_state_expands_shrinks_and_restarts():
    state = TurboTrustRegionState.from_config(
        {
            "initial_length": 0.5,
            "min_length": 0.2,
            "max_length": 1.0,
            "success_tolerance": 2,
            "failure_tolerance": 2,
        }
    )
    state.initialize_best(np.array([1.0]), maximize=True)

    assert state.update(np.array([2.0]), maximize=True) is True
    assert state.length == 0.5
    assert state.update(np.array([3.0]), maximize=True) is True
    assert state.length == 1.0

    state.length = 0.25
    assert state.update(np.array([2.5]), maximize=True) is False
    assert state.update(np.array([2.4]), maximize=True) is False
    assert state.length == 0.5
    assert state.restart_count == 1


def test_turbo_suggests_valid_simplex_and_serializable_state():
    objective = SyntheticSimplexObjective(n_components=3, n_needles=2, seed=123)
    init = objective.evaluate_points(objective.initial_design(5, seed=0), seed=0)
    opt = TuRBOOptimizer(
        acquisition="ucb",
        candidate_pool_size=32,
        min_tr_candidates=8,
        device="cpu",
        dtype="float64",
        trust_region={"initial_length": 0.8, "success_tolerance": 2, "failure_tolerance": 2},
    )
    opt.initialize(init.X_actual, init.y, objective.info, seed=0)

    X_next = opt.suggest(1)
    obs = objective.evaluate_points(X_next, seed=1)
    opt.observe(obs)
    state = opt.get_state()

    assert X_next.shape == (1, 3)
    assert validate_simplex(X_next)
    assert state["n_observations"] == 6
    assert state["turbo_length"] > 0.0
    assert state["last_candidate_pool_size"] == 32
    json.dumps(state)


def test_turbo_scores_candidate_lines_with_metadata(tmp_path):
    config = {
        "experiment": {"name": "test_turbo_line", "mode": "line", "seeds": [0], "output_root": str(tmp_path)},
        "objective": {
            "kind": "synthetic_simplex",
            "name": "synthetic_3d_planted",
            "n_components": 3,
            "maximize": True,
            "params": {"n_needles": 2, "basin_width": 20.0, "noise_std": 0.0, "seed": 123},
        },
        "optimizer": {
            "kind": "turbo",
            "line_score": "mean_acq",
            "params": {
                "acquisition": "ucb",
                "candidate_pool_size": 32,
                "min_tr_candidates": 8,
                "device": "cpu",
                "dtype": "float64",
            },
        },
        "budget": {"n_init": 5, "n_lines": 1, "points_per_line": 3},
        "line_mode": {"n_line_candidates": 4, "score": "mean_acq", "include_endpoints": True},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
    }

    run_dir = run_trial(config, seed=0, repo_root=Path.cwd())

    with open(run_dir / "line_metrics.csv", newline="", encoding="utf-8") as f:
        line_rows = list(csv.DictReader(f))
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(line_rows) == 1
    assert line_rows[0]["line_score_method"] == "mean_acq"
    assert line_rows[0]["line_adapter"] == "turbo_acq_line"
    assert line_rows[0]["line_endpoints_valid_simplex"] == "True"
    assert summary["optimizer_state"]["base_optimizer_state"]["algorithm"] == "benchmark_local_finite_pool_turbo_1"


def test_turbo_observe_accepts_batched_line_update():
    objective = SyntheticSimplexObjective(n_components=3, n_needles=2, seed=123)
    init = objective.evaluate_points(objective.initial_design(5, seed=0), seed=0)
    opt = TuRBOOptimizer(acquisition="ucb", candidate_pool_size=16, min_tr_candidates=4)
    opt.initialize(init.X_actual, init.y, objective.info, seed=0)

    X_batch = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
    obs = BatchObservation(X_expected=X_batch, X_actual=X_batch, y=np.array([1.0, 2.0, 3.0]))
    opt.observe(obs)

    assert opt.get_state()["n_observations"] == 8


def test_turbo_suite_aggregation_writes_outputs(tmp_path):
    config = {
        "experiment": {"name": "test_turbo_suite", "mode": "point", "seeds": [0], "output_root": str(tmp_path)},
        "objective": {
            "kind": "synthetic_simplex",
            "name": "synthetic_3d_planted",
            "n_components": 3,
            "maximize": True,
            "params": {"n_needles": 2, "basin_width": 20.0, "noise_std": 0.0, "seed": 123},
        },
        "budget": {"n_init": 5, "n_steps": 1},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
        "optimizers": [
            {
                "kind": "turbo",
                "params": {
                    "acquisition": "ucb",
                    "candidate_pool_size": 16,
                    "min_tr_candidates": 4,
                    "device": "cpu",
                    "dtype": "float64",
                },
            }
        ],
    }

    aggregate_dir = run_suite(config, repo_root=Path.cwd())

    assert (aggregate_dir / "run_index.csv").exists()
    assert (aggregate_dir / "final_metrics.csv").exists()
    with open(aggregate_dir / "run_index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["optimizer"] == "turbo"
    assert rows[0]["status"] == "success"
