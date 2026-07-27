"""
warm_start/select_best.py
=========================
Collapse a finished MOBO tuning run to the single hyperparameter configuration to
deploy, and write it as an ``evaluate.py --hparams-json`` file.

Each tuning run leaves a 3-objective Pareto front (dist_to_needles, dup_fraction,
avg_time_per_iter_s — all minimised). Deployment needs *one* config, so this picks
the trial **lexicographically by dist_to_needles first, then dup_fraction, then the
time metric** — the "find the needles, break ties by cleanliness then speed" rule.
The lexicographic minimum is always a Pareto-optimal point, so no separate
front-masking step is needed.

Failed trials are excluded exactly as ``optimize/pareto.py`` does: a non-positive
time metric marks a trial that completed zero ZoMBI iterations (its
avg_time_per_iter_s is 0 and its distance is the unmatched-needle penalty), which
would otherwise masquerade as the best point on the minimised objectives.

Reads the run's ``mobo_progress.json`` directly (no torch/botorch import) so it
stays a light pre-deploy helper, like ``best_hparams_seed.py``.

Usage
-----
    python warm_start/select_best.py --run-dir optimize/runs/mobo_warmgp_3d_job123 \
        --out <deploy_dir>/warm_best_3d
    # prints the written JSON path on the last stdout line; pass it to
    # evaluate.py --hparams-json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DIST_KEY = "dist_to_needles"
DUP_KEY = "dup_fraction"
# Current runs record avg_time_per_iter_s; older runs recorded only runtime_s.
TIME_KEYS = ("avg_time_per_iter_s", "runtime_s")


def _time_value(metrics: dict) -> float | None:
    for k in TIME_KEYS:
        if k in metrics:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                return None
    return None


def _valid_trials(progress: dict) -> list[dict]:
    """Trials that carry all three real objectives (failures dropped)."""
    out: list[dict] = []
    for t in progress.get("trials", []):
        m = t.get("metrics", {})
        if DIST_KEY not in m or DUP_KEY not in m:
            continue
        tv = _time_value(m)
        if tv is None or tv <= 0:  # zero-iteration failure sentinel — skip
            continue
        try:
            dist = float(m[DIST_KEY])
            dup = float(m[DUP_KEY])
        except (TypeError, ValueError):
            continue
        out.append({"trial": t.get("trial"), "dist": dist, "dup": dup,
                    "time": tv, "hparams": t.get("hparams", {})})
    return out


def select_best(run_dir: Path) -> dict:
    """Return the lexicographic-(dist, dup, time) best trial record of ``run_dir``."""
    run_dir = Path(run_dir)
    progress_path = run_dir / "mobo_progress.json"
    if not progress_path.exists():
        raise FileNotFoundError(f"no mobo_progress.json in {run_dir}")
    progress = json.loads(progress_path.read_text())
    trials = _valid_trials(progress)
    if not trials:
        raise ValueError(f"{progress_path} has no completed (non-failed) trials")
    best = min(trials, key=lambda r: (r["dist"], r["dup"], r["time"]))
    print(f"  [select] {run_dir.name}: {len(trials)} valid trial(s); "
          f"best = trial {best['trial']} "
          f"(dist={best['dist']:.4f} dup={best['dup']:.4f} time={best['time']:.3f})")
    return best


def write_best(run_dir: Path, out_dir: Path) -> Path:
    """Write ``out_dir/hparams.json`` (a trial.json-style dict) for the best trial."""
    best = select_best(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial": best["trial"],
        "source_run": str(Path(run_dir).resolve()),
        "selection": "lexicographic(dist_to_needles, dup_fraction, time)",
        "metrics": {DIST_KEY: best["dist"], DUP_KEY: best["dup"],
                    "time": best["time"]},
        "hparams": best["hparams"],
    }
    out_path = out_dir / "hparams.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick the deploy config from a finished MOBO tuning run.")
    parser.add_argument("--run-dir", required=True,
                        help="Finished run dir (holds mobo_progress.json).")
    parser.add_argument("--out", required=True,
                        help="Output dir; hparams.json is written inside it.")
    args = parser.parse_args()
    out_path = write_best(Path(args.run_dir), Path(args.out))
    # The sbatch captures stdout for --hparams-json, so print ONLY the path last.
    print(str(out_path.resolve()))


if __name__ == "__main__":
    main()
