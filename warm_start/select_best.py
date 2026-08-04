"""
warm_start/select_best.py
=========================
Collapse a finished MOBO tuning run to the single hyperparameter configuration to
deploy, and write it as an ``evaluate.py --hparams-json`` file.

Each tuning run leaves a 3-objective Pareto front (dist_to_needles, dup_fraction,
avg_time_per_iter_s — all minimised). Deployment needs *one* config, so this picks
the trial with the best **weighted score ``w_dist * dist + w_dup * dup``** (default
0.8 / 0.2) — a deliberate dist/dup trade-off rather than "lowest dist, dup be
damned".

Because dist and dup live on very different scales (dist ranges ~0.2–9, dup is a
fraction in ~0.6–1.0), each metric is first **min-max normalised to [0,1] over the
run's completed trials** before weighting; otherwise a raw sum is ~99 % dist and dup
barely counts. So the weights are shares of the *observed spread* of each metric, and
``w_dup=0.2`` genuinely gives dup a fifth of the say. The time metric is kept only as
a deterministic tie-breaker. Weights are configurable via ``--w-dist`` / ``--w-dup``.

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


def _minmax(vals: list[float]) -> list[float]:
    """Min-max scale ``vals`` to [0,1]; a degenerate (zero-span) axis maps to all 0
    so it contributes nothing to the weighted score rather than dividing by zero."""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return [0.0] * len(vals)
    return [(v - lo) / span for v in vals]


def select_best(run_dirs, *, w_dist: float = 0.8, w_dup: float = 0.2,
                pin_trial: int | None = None) -> dict:
    """Return the chosen trial pooled across ``run_dirs``.

    Default: the best weighted score ``w_dist * dist_norm + w_dup * dup_norm`` where
    dist and dup are each min-max normalised to [0,1] over the pooled completed trials
    (see module docstring for why normalisation is required). Lower is better; the time
    metric breaks ties.

    ``pin_trial``: bypass the score and return that exact trial number instead — used
    when a config is picked by hand off the Pareto plot rather than by the weighting.
    It must be a completed (non-failed) trial in the pool, else a ValueError lists what
    is available. The record still carries the pool-normalised dist/dup and score for
    context.

    Accepts a single run dir or a list of them: the parallel, shared-history tuning
    (``tune_par.sbatch``) splits one budget across several worker dirs, so the deploy
    config is chosen over EVERY worker of the submission. A single dir (the old
    single-job scripts) still works — it is just the one-element case.
    """
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    run_dirs = [Path(d) for d in run_dirs]

    pooled: list[dict] = []
    for run_dir in run_dirs:
        progress_path = run_dir / "mobo_progress.json"
        if not progress_path.exists():
            raise FileNotFoundError(f"no mobo_progress.json in {run_dir}")
        trials = _valid_trials(json.loads(progress_path.read_text()))
        for t in trials:
            t["source_run"] = str(run_dir.resolve())
        print(f"  [select] {run_dir.name}: {len(trials)} valid trial(s)")
        pooled.extend(trials)
    if not pooled:
        raise ValueError(f"no completed (non-failed) trials across {len(run_dirs)} "
                         f"run dir(s)")

    dist_n = _minmax([t["dist"] for t in pooled])
    dup_n = _minmax([t["dup"] for t in pooled])
    for t, dn, un in zip(pooled, dist_n, dup_n):
        t["dist_norm"], t["dup_norm"] = dn, un
        t["score"] = w_dist * dn + w_dup * un

    if pin_trial is not None:
        matches = [t for t in pooled if t["trial"] == pin_trial]
        if not matches:
            avail = sorted(t["trial"] for t in pooled)
            raise ValueError(f"--trial {pin_trial} is not a completed trial in the "
                             f"pool; available: {avail}")
        best = matches[0]
        how = "manual pin"
    else:
        # Lowest weighted score; break exact ties by raw dist then time (deterministic).
        best = min(pooled, key=lambda r: (r["score"], r["dist"], r["time"]))
        how = f"weighted dist={w_dist} dup={w_dup}"
    print(f"  [select] pooled {len(pooled)} valid trial(s) over {len(run_dirs)} "
          f"dir(s); selection: {how}\n"
          f"  [select] chosen = trial {best['trial']} of {Path(best['source_run']).name} "
          f"(score={best['score']:.4f}  dist={best['dist']:.4f} [n={best['dist_norm']:.2f}] "
          f"dup={best['dup']:.4f} [n={best['dup_norm']:.2f}] time={best['time']:.3f})")
    return best


def write_best(run_dirs, out_dir: Path, *,
               w_dist: float = 0.8, w_dup: float = 0.2,
               pin_trial: int | None = None) -> Path:
    """Write ``out_dir/hparams.json`` (a trial.json-style dict) for the chosen trial."""
    best = select_best(run_dirs, w_dist=w_dist, w_dup=w_dup, pin_trial=pin_trial)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = (f"manual pin trial {pin_trial}" if pin_trial is not None else
                 f"min({w_dist}*dist_norm + {w_dup}*dup_norm), "
                 f"metrics min-max normalised over the trial pool; time as tie-breaker")
    payload = {
        "trial": best["trial"],
        "source_run": best["source_run"],
        "selection": selection,
        "score": best["score"],
        "metrics": {DIST_KEY: best["dist"], DUP_KEY: best["dup"],
                    "time": best["time"],
                    "dist_norm": best["dist_norm"], "dup_norm": best["dup_norm"]},
        "hparams": best["hparams"],
    }
    out_path = out_dir / "hparams.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick the deploy config from finished MOBO tuning run(s).")
    parser.add_argument("--run-dir", required=True, nargs="+", metavar="RUN_DIR",
                        help="Finished run dir(s) holding mobo_progress.json. Pass "
                             "several (or a shell glob) to pool parallel workers.")
    parser.add_argument("--out", required=True,
                        help="Output dir; hparams.json is written inside it.")
    parser.add_argument("--w-dist", type=float, default=0.8,
                        help="Weight on normalised dist_to_needles (default 0.8).")
    parser.add_argument("--w-dup", type=float, default=0.2,
                        help="Weight on normalised dup_fraction (default 0.2).")
    parser.add_argument("--trial", type=int, default=None,
                        help="Pin this exact trial number (a hand-picked config off "
                             "the Pareto plot) instead of the weighted score.")
    args = parser.parse_args()
    out_path = write_best(args.run_dir, Path(args.out),
                          w_dist=args.w_dist, w_dup=args.w_dup,
                          pin_trial=args.trial)
    # The sbatch captures stdout for --hparams-json, so print ONLY the path last.
    print(str(out_path.resolve()))


if __name__ == "__main__":
    main()
