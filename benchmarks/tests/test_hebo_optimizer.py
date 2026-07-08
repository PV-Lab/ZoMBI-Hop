import importlib.util

import numpy as np
import pandas as pd
import pytest

from benchmarks.zombihop_benchmark.line_audit import audit_line_endpoints
from benchmarks.zombihop_benchmark.optimizers.base import build_optimizer
from benchmarks.zombihop_benchmark.optimizers.hebo_optimizer import HEBOOptimizer
from benchmarks.zombihop_benchmark.spaces import composition_to_ilr_np, validate_simplex
from benchmarks.zombihop_benchmark.types import BatchObservation, ObjectiveInfo


class FakeDesignSpace:
    def parse(self, specs):
        self.specs = specs
        self.numeric_names = [spec["name"] for spec in specs]
        return self


class FakeHEBO:
    def __init__(self, space, **kwargs):
        self.space = space
        self.kwargs = kwargs
        self.observed = []
        self.suggest_calls = 0

    def suggest(self, n_suggestions=1):
        self.suggest_calls += 1
        rows = []
        for idx in range(n_suggestions):
            rows.append({name: 0.05 * (idx + self.suggest_calls) for name in self.space.numeric_names})
        return pd.DataFrame(rows)

    def observe(self, X, y):
        self.observed.append((X.copy(), np.asarray(y, dtype=float).copy()))


def test_hebo_registry_returns_optional_adapter():
    optimizer = build_optimizer(
        {
            "kind": "hebo",
            "params": {"hebo_cls": FakeHEBO, "design_space_cls": FakeDesignSpace},
        }
    )

    assert isinstance(optimizer, HEBOOptimizer)
    assert optimizer.name == "hebo"


def test_hebo_point_adapter_uses_ilr_and_minimization_sign():
    optimizer = HEBOOptimizer(
        hebo_cls=FakeHEBO,
        design_space_cls=FakeDesignSpace,
        ilr_bounds={"lower": [-2.0, -2.0], "upper": [2.0, 2.0]},
    )
    X_init = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    y_init = np.array([1.5, 2.0])

    optimizer.initialize(
        X_init,
        y_init,
        ObjectiveInfo(name="toy", n_components=3, maximize=True),
        seed=0,
    )
    X_next = optimizer.suggest(2)

    assert validate_simplex(X_next)
    assert optimizer._hebo.observed[0][1].reshape(-1).tolist() == [-1.5, -2.0]

    obs = BatchObservation(X_expected=X_next, X_actual=X_next, y=np.array([3.0, 4.0]))
    optimizer.observe(obs)

    assert optimizer._hebo.observed[-1][1].reshape(-1).tolist() == [-3.0, -4.0]
    assert optimizer.get_state()["n_observations"] == 4


def test_hebo_line_adapter_returns_valid_audited_simplex_line():
    optimizer = HEBOOptimizer(
        hebo_cls=FakeHEBO,
        design_space_cls=FakeDesignSpace,
        ilr_bounds={"lower": [-2.0, -2.0], "upper": [2.0, 2.0]},
        points_per_line=6,
        n_line_candidates=8,
    )
    X_init = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    y_init = np.array([1.5, 2.0])
    optimizer.initialize(X_init, y_init, ObjectiveInfo(name="toy", n_components=3), seed=5)

    line = optimizer.suggest_line()
    metadata = line.metadata()

    assert validate_simplex(line.points)
    assert line.points.shape == (6, 3)
    assert metadata["line_adapter"] == "hebo_anchor_chord"
    assert metadata["line_endpoints_valid_simplex"] is True
    assert metadata["line_length_l2_within_simplex_diameter"] is True
    assert metadata["line_length_l2_coordinate_system"] == "raw_simplex_l2"
    assert metadata["line_length_ilr_coordinate_system"].startswith("helmert_ilr_l2")


def test_line_audit_identifies_near_boundary_ilr_behavior():
    endpoints = np.array([[1.0 - 1e-12, 1e-12, 0.0], [1e-12, 1.0 - 1e-12, 0.0]])
    audit = audit_line_endpoints(endpoints)

    assert audit["line_endpoints_valid_simplex"] is True
    assert audit["line_endpoint_min"] == 0.0
    assert audit["line_length_l2_within_simplex_diameter"] is True
    assert audit["line_length_ilr_finite"] is True
    assert audit["line_length_ilr_audit"] > 10.0


def test_hebo_missing_dependency_has_helpful_error():
    if importlib.util.find_spec("hebo") is not None:
        pytest.skip("real HEBO is installed in this environment")
    optimizer = HEBOOptimizer()
    X_init = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    y_init = np.array([1.5, 2.0])

    with pytest.raises(ImportError, match="optional HEBO package"):
        optimizer.initialize(X_init, y_init, ObjectiveInfo(name="toy", n_components=3), seed=0)
