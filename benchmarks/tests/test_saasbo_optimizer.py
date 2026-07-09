import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.zombihop_benchmark.line_mode import LineModeOptimizerWrapper
from benchmarks.zombihop_benchmark.objectives.synthetic import SyntheticSimplexObjective
from benchmarks.zombihop_benchmark.optimizers.base import build_optimizer
from benchmarks.zombihop_benchmark.optimizers.saasbo_optimizer import (
    SAASBOOptimizer,
    get_saasbo_dependency_status,
    is_saasbo_available,
)
from benchmarks.zombihop_benchmark.spaces import validate_simplex
from benchmarks.zombihop_benchmark.types import BatchObservation


class FakePosterior:
    def __init__(self, mean):
        self.mean = mean


class FakeSAASModel:
    def __init__(self, train_X, train_Y, train_Yvar=None, **kwargs):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.kwargs = kwargs
        self.median_lengthscale = train_X.new_tensor([0.25, 0.75])
        self.fit_kwargs = None

    def posterior(self, X):
        mean = X.sum(dim=-1, keepdim=True)
        return FakePosterior(mean)


def fake_fit_func(model, **kwargs):
    model.fit_kwargs = kwargs


def _fake_optimizer(**kwargs):
    return SAASBOOptimizer(
        acquisition="posterior_mean",
        candidate_pool_size=16,
        ilr_bounds={"lower": [-3.0, -3.0], "upper": [3.0, 3.0]},
        model_cls=FakeSAASModel,
        fit_func=fake_fit_func,
        **kwargs,
    )


def _initialized_fake_optimizer():
    objective = SyntheticSimplexObjective(n_components=3, n_needles=2, seed=123)
    init = objective.evaluate_points(objective.initial_design(5, seed=0), seed=0)
    opt = _fake_optimizer()
    opt.initialize(init.X_actual, init.y, objective.info, seed=0)
    return opt, objective


def test_saasbo_registry_returns_adapter():
    optimizer = build_optimizer(
        {
            "kind": "saasbo",
            "params": {
                "candidate_pool_size": 16,
                "model_cls": FakeSAASModel,
                "fit_func": fake_fit_func,
                "acquisition": "posterior_mean",
            },
        }
    )

    assert isinstance(optimizer, SAASBOOptimizer)
    assert optimizer.name == "saasbo"


def test_saasbo_dependency_status_has_required_fields():
    status = get_saasbo_dependency_status()

    assert "available" in status
    assert status["backend"] == "botorch_saas_fully_bayesian"
    assert "botorch" in status["modules"]
    assert isinstance(is_saasbo_available(), bool)


def test_saasbo_missing_dependency_error_is_clear():
    if is_saasbo_available():
        pytest.skip("real SAASBO fully Bayesian dependencies are installed")
    objective = SyntheticSimplexObjective(n_components=3, n_needles=2, seed=123)
    init = objective.evaluate_points(objective.initial_design(5, seed=0), seed=0)
    opt = SAASBOOptimizer(candidate_pool_size=16)

    with pytest.raises(ImportError, match="fully Bayesian optional dependencies"):
        opt.initialize(init.X_actual, init.y, objective.info, seed=0)


def test_saasbo_fake_backend_suggests_scores_and_observes_valid_simplex():
    opt, objective = _initialized_fake_optimizer()

    X_next = opt.suggest(1)
    scores = opt.score_candidates(np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]]))
    obs = objective.evaluate_points(X_next, seed=1)
    opt.observe(obs)
    state = opt.get_state()

    assert X_next.shape == (1, 3)
    assert validate_simplex(X_next)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert state["n_observations"] == 6
    assert state["fit_calls"] >= 1
    assert state["median_lengthscale_values"] == [0.25, 0.75]
    json.dumps(state)


def test_saasbo_fake_backend_line_wrapper_labels_acquisition_line():
    opt, objective = _initialized_fake_optimizer()
    init = objective.evaluate_points(objective.initial_design(5, seed=1), seed=1)
    wrapper = LineModeOptimizerWrapper(opt, points_per_line=5, n_line_candidates=4, line_score="mean_acq")
    wrapper.initialize(init.X_actual, init.y, objective.info, seed=3)

    line = wrapper.suggest_line()
    metadata = line.metadata()

    assert validate_simplex(line.points)
    assert line.points.shape == (5, 3)
    assert metadata["line_adapter"] == "saasbo_acq_line"
    assert metadata["line_score_method"] == "mean_acq"
    assert metadata["line_endpoints_valid_simplex"] is True


def test_saasbo_line_metrics_present_with_fake_backend(tmp_path):
    opt, objective = _initialized_fake_optimizer()
    wrapper = LineModeOptimizerWrapper(opt, points_per_line=4, n_line_candidates=3, line_score="mean_acq")
    init = objective.evaluate_points(objective.initial_design(5, seed=2), seed=2)
    wrapper.initialize(init.X_actual, init.y, objective.info, seed=2)
    line = wrapper.suggest_line()
    obs_raw = objective.evaluate_points(line.points, seed=3)
    obs = BatchObservation(obs_raw.X_expected, obs_raw.X_actual, obs_raw.y, {"line": line.metadata()})
    wrapper.observe(obs)

    rows = [
        {
            "line_index": 1,
            "optimizer": wrapper.name,
            "seed": 2,
            "line_adapter": line.metadata()["line_adapter"],
            "line_score_method": line.metadata()["line_score_method"],
        }
    ]
    path = Path(tmp_path) / "line_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(path, newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert loaded[0]["optimizer"] == "saasbo"
    assert loaded[0]["line_adapter"] == "saasbo_acq_line"
