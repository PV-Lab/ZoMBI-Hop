import csv
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.zombihop_benchmark.objectives.real_rf_surrogate import RealRFSurrogateObjective
from benchmarks.zombihop_benchmark.runner import run_trial
from benchmarks.zombihop_benchmark.spaces import ilr_distance, validate_simplex


def _write_fixture_csv(path: Path) -> None:
    rows = [
        (1.0, 0.0, 0.0, 0.10),
        (0.0, 1.0, 0.0, 0.90),
        (0.0, 0.0, 1.0, 0.30),
        (0.5, 0.5, 0.0, 0.70),
        (0.2, 0.6, 0.2, 0.95),
        (0.2, 0.2, 0.6, 0.40),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["A", "B", "C", "Objective"])
        writer.writerows(rows)


def _objective(path: Path, **overrides):
    params = {
        "name": "fixture_rf",
        "n_components": 3,
        "maximize": True,
        "data_path": str(path),
        "component_labels": ["A", "B", "C"],
        "composition_columns": ["A", "B", "C"],
        "target_column": "Objective",
        "feature_transform": "raw_composition",
        "train_surrogate": {"enabled": True, "n_estimators": 20, "random_state": 0, "n_jobs": 1},
        "needle_detection": {
            "method": "grid_local_maxima",
            "grid_resolution": 12,
            "top_k": 3,
            "min_separation_ilr": 0.1,
            "match_radius_ilr": 0.25,
        },
    }
    params.update(overrides)
    return RealRFSurrogateObjective(**params)


def test_real_rf_surrogate_trains_from_csv_and_evaluates_points(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    _write_fixture_csv(csv_path)
    objective = _objective(csv_path)

    obs = objective.evaluate_points(np.array([[0.2, 0.6, 0.2], [2.0, 1.0, 1.0]]), seed=0)

    assert obs.X_actual.shape == (2, 3)
    assert obs.y.shape == (2,)
    assert validate_simplex(obs.X_actual)
    np.testing.assert_allclose(obs.X_actual[1].sum(), 1.0)
    assert obs.metadata["kind"] == "real_rf_surrogate"
    assert objective.get_metadata()["num_true_needles"] <= 3


def test_real_rf_surrogate_evaluate_line_and_needle_separation(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    _write_fixture_csv(csv_path)
    objective = _objective(csv_path)

    obs = objective.evaluate_line(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), n_points=5, seed=0)

    assert obs.X_actual.shape == (5, 3)
    assert obs.metadata["left"] == [1.0, 0.0, 0.0]
    assert obs.metadata["right"] == [0.0, 1.0, 0.0]
    needles = objective.true_needles
    assert needles.shape[0] <= 3
    if needles.shape[0] > 1:
        dists = ilr_distance(needles, needles)
        lower = dists[np.tril_indices_from(dists, k=-1)]
        assert float(np.min(lower)) >= 0.1 - 1e-9


def test_real_rf_surrogate_can_load_fitted_model(tmp_path):
    sklearn = pytest.importorskip("sklearn.ensemble")
    joblib = pytest.importorskip("joblib")
    csv_path = tmp_path / "fixture.csv"
    _write_fixture_csv(csv_path)
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    X = np.column_stack([data["A"], data["B"], data["C"]])
    y = data["Objective"]
    model = sklearn.RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=1).fit(X, y)
    model_path = tmp_path / "fixture.joblib"
    joblib.dump(model, model_path)

    objective = RealRFSurrogateObjective(
        name="fixture_model",
        n_components=3,
        maximize=True,
        model_path=str(model_path),
        model_format="joblib",
        component_labels=["A", "B", "C"],
        feature_transform="raw_composition",
        needle_detection={"grid_resolution": 8, "top_k": 2, "min_separation_ilr": 0.1},
    )

    obs = objective.evaluate_points(np.array([[0.2, 0.6, 0.2]]))
    assert obs.y.shape == (1,)
    assert objective.get_metadata()["source_kind"] == "model"


def test_real_rf_runner_smoke_writes_metadata_and_needles(tmp_path):
    csv_path = tmp_path / "fixture.csv"
    _write_fixture_csv(csv_path)
    config = {
        "experiment": {"name": "test_real_rf", "mode": "point", "seeds": [0], "output_root": str(tmp_path / "runs")},
        "objective": {
            "kind": "real_rf_surrogate",
            "name": "fixture_rf",
            "n_components": 3,
            "maximize": True,
            "params": {
                "data_path": str(csv_path),
                "component_labels": ["A", "B", "C"],
                "composition_columns": ["A", "B", "C"],
                "target_column": "Objective",
                "train_surrogate": {"enabled": True, "n_estimators": 10, "random_state": 0, "n_jobs": 1},
                "needle_detection": {"grid_resolution": 8, "top_k": 2, "min_separation_ilr": 0.1},
            },
        },
        "optimizer": {"kind": "random_simplex", "params": {}},
        "budget": {"n_init": 3, "n_steps": 2},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
    }

    run_dir = run_trial(config, seed=0, repo_root=Path.cwd())

    assert (run_dir / "objective_metadata.json").exists()
    assert (run_dir / "objective_needles.csv").exists()
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["objective_kind"] == "real_rf_surrogate"
    assert summary["num_true_needles"] <= 2
