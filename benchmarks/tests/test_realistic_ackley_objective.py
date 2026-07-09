import numpy as np

from benchmarks.zombihop_benchmark.objectives.base import build_objective
from benchmarks.zombihop_benchmark.objectives.realistic_ackley import RealisticAckleySimplexObjective


def test_realistic_ackley_objective_builds_from_factory():
    objective = build_objective(
        {
            "kind": "realistic_ackley_simplex",
            "name": "realistic_ackley_3d",
            "n_components": 3,
            "maximize": True,
            "params": {
                "n_optima": 20,
                "basin_width": 86.0,
                "noise_freq": 9.0,
                "noise_amp": 400.0,
                "peak_seed": 0,
                "noise_seed": 42,
            },
        }
    )

    assert isinstance(objective, RealisticAckleySimplexObjective)
    assert objective.info.name == "realistic_ackley_3d"
    assert objective.info.true_needles.shape == (20, 3)
    assert objective.info.match_radius_comp == 0.05
    np.testing.assert_allclose(objective.info.true_needles.sum(axis=1), 1.0)


def test_realistic_ackley_evaluates_points_and_lines():
    objective = RealisticAckleySimplexObjective(n_components=3, n_optima=3)
    X = objective.initial_design(4, seed=7)
    obs = objective.evaluate_points(X, seed=7)
    assert obs.X_actual.shape == (4, 3)
    assert obs.y.shape == (4,)
    assert np.isfinite(obs.y).all()

    line_obs = objective.evaluate_line(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), n_points=5, seed=8)
    assert line_obs.X_actual.shape == (5, 3)
    assert line_obs.y.shape == (5,)


def test_realistic_ackley_4d_metadata_points_lines_and_distribution():
    objective = RealisticAckleySimplexObjective(
        name="realistic_ackley_4d",
        n_components=4,
        n_optima=30,
        basin_width=65.0,
        noise_freq=9.0,
        noise_amp=300.0,
    )

    assert objective.info.name == "realistic_ackley_4d"
    assert objective.info.true_needles.shape == (30, 4)
    metadata = objective.get_metadata()
    assert metadata["n_components"] == 4
    assert metadata["n_optima"] == 30
    assert metadata["basin_width"] == 65.0
    assert metadata["noise_amp"] == 300.0
    assert metadata["num_true_needles"] == 30
    assert metadata["synthetic_role"] == "headline_brianna_realistic_4d"
    assert metadata["y_star"] == objective.info.y_star

    X = objective.initial_design(6, seed=11)
    obs = objective.evaluate_points(X, seed=11)
    assert obs.X_actual.shape == (6, 4)
    np.testing.assert_allclose(obs.X_actual.sum(axis=1), 1.0)
    assert np.isfinite(obs.y).all()

    line_obs = objective.evaluate_line(
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        n_points=24,
        seed=12,
    )
    assert line_obs.X_actual.shape == (24, 4)
    np.testing.assert_allclose(line_obs.X_actual.sum(axis=1), 1.0)

    distribution = objective.objective_distribution_rows(n_samples=128, seed=13)[0]
    assert distribution["n_samples"] == 128
    assert distribution["y_min"] <= distribution["y_median"] <= distribution["y_max"]
    assert 0.0 <= distribution["fraction_above_0.9"] <= 1.0
