import numpy as np
import pytest

from benchmarks.zombihop_benchmark.objectives.synthetic import SyntheticSimplexObjective
from benchmarks.zombihop_benchmark.optimizers.gp_botorch import GPBoTorchOptimizer
from benchmarks.zombihop_benchmark.runner import run_trial
from benchmarks.zombihop_benchmark.spaces import validate_simplex


pytest.importorskip("torch")
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")


def test_gp_botorch_ei_suggests_valid_simplex():
    objective = SyntheticSimplexObjective(n_components=3, n_needles=2, seed=123)
    init = objective.evaluate_points(objective.initial_design(5, seed=0), seed=0)
    opt = GPBoTorchOptimizer(kind="ei", candidate_pool_size=32, device="cpu", dtype="float64")
    opt.initialize(init.X_actual, init.y, objective.info, seed=0)

    X_next = opt.suggest(1)
    obs = objective.evaluate_points(X_next, seed=1)
    opt.observe(obs)

    assert X_next.shape == (1, 3)
    assert validate_simplex(X_next)
    assert opt.get_state()["n_observations"] == 6


def test_gp_botorch_runner_writes_outputs(tmp_path):
    config = {
        "experiment": {"name": "test_gp", "mode": "point", "seeds": [0], "output_root": str(tmp_path)},
        "objective": {
            "kind": "synthetic_simplex",
            "name": "synthetic_3d_planted",
            "n_components": 3,
            "maximize": True,
            "params": {"n_needles": 2, "basin_width": 20.0, "noise_std": 0.0, "seed": 123},
        },
        "optimizer": {
            "kind": "gp_ard_ucb",
            "params": {"candidate_pool_size": 32, "ucb_beta": 0.2, "device": "cpu", "dtype": "float64"},
        },
        "budget": {"n_init": 5, "n_steps": 1},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
    }

    run_dir = run_trial(config, seed=0, repo_root=tmp_path)

    assert (run_dir / "summary.json").exists()
    assert (run_dir / "points.csv").exists()
    assert (run_dir / "metrics_over_time.csv").exists()
