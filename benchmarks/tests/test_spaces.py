import numpy as np
import pytest

from benchmarks.zombihop_benchmark.spaces import (
    composition_to_ilr_np,
    ilr_to_composition_np,
    ilr_distance,
    project_simplex,
    sample_simplex,
    validate_simplex,
)


def test_sample_simplex_valid_rows():
    X = sample_simplex(10, 3, seed=0)
    assert X.shape == (10, 3)
    assert np.all(X >= 0)
    assert np.allclose(X.sum(axis=1), 1.0)
    assert validate_simplex(X)


def test_project_simplex_valid_rows():
    X = np.array([[0.2, -0.1, 4.0], [2.0, 2.0, 2.0]])
    P = project_simplex(X)
    assert np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_ilr_distance_shapes():
    X = sample_simplex(5, 3, seed=1)
    Y = sample_simplex(7, 3, seed=2)
    assert composition_to_ilr_np(X).shape == (5, 2)
    assert ilr_distance(X, Y).shape == (5, 7)


def test_ilr_round_trip_returns_valid_simplex_rows():
    X = sample_simplex(12, 3, seed=11)
    Z = composition_to_ilr_np(X)
    X_round = ilr_to_composition_np(Z, n_components=3)

    assert validate_simplex(X_round)
    np.testing.assert_allclose(X_round, X, atol=1e-10)


def test_invalid_simplex_raises():
    with pytest.raises(ValueError, match="sum"):
        validate_simplex(np.array([[0.2, 0.2, 0.2]]))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_simplex(np.array([[1.1, -0.1, 0.0]]))

