import csv

from benchmarks.zombihop_benchmark.suite import run_suite


def test_suite_runner_writes_aggregate_outputs(tmp_path):
    config = {
        "experiment": {"name": "test_suite", "mode": "point", "seeds": [0, 1], "output_root": str(tmp_path)},
        "objective": {
            "kind": "synthetic_simplex",
            "name": "synthetic_3d_planted",
            "n_components": 3,
            "maximize": True,
            "params": {"n_needles": 2, "basin_width": 20.0, "noise_std": 0.0, "seed": 123},
        },
        "budget": {"n_init": 3, "n_steps": 2},
        "metrics": {"duplicate_radius_ilr": 0.03, "match_radius_ilr": 0.25},
        "optimizers": [{"kind": "random_simplex", "params": {}}],
    }

    aggregate_dir = run_suite(config, repo_root=tmp_path)

    assert (aggregate_dir / "run_index.csv").exists()
    assert (aggregate_dir / "metrics_over_time_long.csv").exists()
    assert (aggregate_dir / "final_metrics.csv").exists()
    assert (aggregate_dir / "summary_by_optimizer.csv").exists()
    with open(aggregate_dir / "run_index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"success"}
