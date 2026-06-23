"""Smoke tests for dimension-general synthetic oracle expressions."""

from __future__ import annotations

import numpy as np
import pytest

from synthetic_data.oracles import ORACLE_CHOICES, ORACLE_EXPRESSIONS, build_oracle


@pytest.mark.parametrize("oracle", ORACLE_CHOICES)
def test_build_oracle_evaluates_at_centroid(oracle: str) -> None:
    d, layout, seed = 3, "2", 42
    fn, optima, label = build_oracle(oracle, d, layout, seed=seed)
    x = np.ones(d) / d
    y = float(fn.predict(x)) if hasattr(fn, "predict") and oracle == "ackley" else float(fn(x))
    assert np.isfinite(y), f"{oracle}: non-finite at centroid"
    assert label
    assert oracle in ORACLE_EXPRESSIONS
    assert len(optima) >= 1
    for peak in optima:
        assert peak.shape == (d,)
        assert abs(float(peak.sum()) - 1.0) < 1e-8


def test_vertex_max_peaks_at_vertices() -> None:
    fn, optima, _ = build_oracle("vertex_max", 3, "2", seed=0)
    for v in optima:
        assert abs(float(fn(v)) - 1.0) < 1e-12
    centroid = np.ones(3) / 3
    assert float(fn(centroid)) < 1.0


def test_aitchison_differs_from_euclidean_gaussian() -> None:
    _, _, _ = build_oracle("gaussian", 3, "2", seed=0)
    g_fn, g_opt, _ = build_oracle("gaussian", 3, "2", seed=0)
    a_fn, a_opt, _ = build_oracle("aitchison_gaussian", 3, "2", seed=0)
    grid = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [1 / 3, 1 / 3, 1 / 3],
    ])
    g_vals = np.array([float(g_fn(x)) for x in grid])
    a_vals = np.array([float(a_fn(x)) for x in grid])
    assert not np.allclose(g_vals, a_vals)
    assert len(g_opt) == len(a_opt)
