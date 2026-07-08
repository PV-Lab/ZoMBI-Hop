from pathlib import Path

from benchmarks.zombihop_benchmark.runner import run_trial


def test_runner_random_smoke(tmp_path):
    config = {
        "experiment": {
            "name": "test_smoke_random",
            "mode": "point",
            "seeds": [0],
            "output_root": str(tmp_path),
        },
        "objective": {
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
        },
        "optimizer": {"kind": "random_simplex", "params": {}},
        "budget": {"n_init": 3, "n_steps": 2},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
    }
    run_dir = run_trial(config, seed=0, repo_root=Path.cwd())
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "points.csv").exists()
    assert (run_dir / "metrics_over_time.csv").exists()

