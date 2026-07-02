"""
llm/evaluate_llm_sweep.py
=========================
Sweep the LLM hyperparameter-tuning experiment across injection points.

Runs evaluate_llm.run_evaluation once per injection iteration, starting at
``INJECTION_INTERVAL`` and incrementing by ``INJECTION_INTERVAL`` each time
(so, with an interval of 5: iterations 5, 10, 15, 20, ...). All runs of a sweep
land under one timestamped directory, and a ``sweep_summary.csv`` /
``sweep_summary.json`` tabulates the LLM's decision, latency, and the
baseline-vs-LLM differences at every injection point.

The model and prompt are configured in llm/llm_config.py (shared by every run).

Usage:
  conda activate zombi-hop
  python llm/evaluate_llm_sweep.py
"""

from __future__ import annotations

# ─── HARDCODED CONFIG ──────────────────────────────────────────────────────────
INJECTION_INTERVAL: int = 5       # step between successive injection iterations
MAX_INJECTION_ITER: int = 40      # last injection iteration to try (db has iters 0..40)
START_ITER: int | None = None     # first injection iter (default: INJECTION_INTERVAL)

# Repeats per injection point (variance reduction). The LLM is still called once
# per point; only the ZoMBI-Hop continuation is repeated.
N_LLM_REPEATS: int = 5            # RF continuations with the LLM's hyperparameters
# Baseline = the real campaign2 run (sample #1) + (N_BASELINE_REPEATS - 1) RF
# continuations with the ORIGINAL (trial_112) hyperparameters. Set to 5 → 1 real
# + 4 RF reruns, all pooled for mean/variance.
N_BASELINE_REPEATS: int = 5
# ───────────────────────────────────────────────────────────────────────────────

import csv
import datetime
import json
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import evaluate_llm as E  # noqa: E402


def _ms(stats: dict, key: str, field: str):
    """stats[key][field] or None."""
    if not stats:
        return None
    return (stats.get(key) or {}).get(field)


def _flatten(comparison: dict) -> dict:
    """One flat row per injection point for the summary CSV (mean ± std columns)."""
    m = comparison.get("mapping", {})
    llm = comparison.get("llm_decision", {})
    bstats = comparison.get("baseline_stats") or {}
    lstats = comparison.get("llm_stats")
    diff = comparison.get("difference_llm_minus_baseline_mean") or {}
    row = {
        "injection_iter": comparison.get("injection_iter"),
        "snapshot": m.get("snapshot_name"),
        "n_points_at_injection": m.get("n_points_at_injection"),
        "budget_iterations": m.get("budget_iterations"),
        "n_baseline_repeats": comparison.get("n_baseline_repeats"),
        "n_llm_repeats": comparison.get("n_llm_repeats"),
        "model": llm.get("model"),
        "effort": llm.get("effort"),
        "latency_s": llm.get("latency_s"),
        "changed": llm.get("changed_hyperparameters"),
        "changes": json.dumps(llm.get("validated_changes", {})),
        "reasoning": llm.get("reasoning"),
        "baseline_best_mean": _ms(bstats, "best_objective", "mean"),
        "baseline_best_std": _ms(bstats, "best_objective", "std"),
        "baseline_needles_mean": _ms(bstats, "n_needles", "mean"),
        "baseline_dup_mean": _ms(bstats, "dup_fraction", "mean"),
        "llm_best_mean": _ms(lstats, "best_objective", "mean"),
        "llm_best_std": _ms(lstats, "best_objective", "std"),
        "llm_needles_mean": _ms(lstats, "n_needles", "mean"),
        "llm_dup_mean": _ms(lstats, "dup_fraction", "mean"),
        "diff_best": diff.get("best_objective"),
        "diff_needles": diff.get("n_needles"),
        "diff_dup": diff.get("dup_fraction"),
        "out_dir": comparison.get("out_dir"),
    }
    return row


def main() -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = E.RESULTS_ROOT / f"sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    start = START_ITER if START_ITER is not None else INJECTION_INTERVAL
    iters = list(range(start, MAX_INJECTION_ITER + 1, INJECTION_INTERVAL))
    print(f"Sweep over injection iterations: {iters}")
    print(f"Sweep dir: {sweep_dir}")

    rows = []
    for it in iters:
        try:
            comparison = E.run_evaluation(
                it, out_root=sweep_dir / f"inj_{it:03d}",
                n_llm_repeats=N_LLM_REPEATS, n_baseline_repeats=N_BASELINE_REPEATS)
            rows.append(_flatten(comparison))
        except Exception as e:
            print(f"  [sweep] injection {it} FAILED: {e}")
            traceback.print_exc()
            rows.append({"injection_iter": it, "changes": f"ERROR: {e}"})
        # Incremental write so a crash mid-sweep still leaves a usable summary.
        _write_summary(sweep_dir, rows)

    print(f"\nSweep complete. Summary → {sweep_dir / 'sweep_summary.csv'}")


def _write_summary(sweep_dir: Path, rows: list) -> None:
    if not rows:
        return
    fields = ["injection_iter", "snapshot", "n_points_at_injection", "budget_iterations",
              "n_baseline_repeats", "n_llm_repeats",
              "model", "effort", "latency_s", "changed", "changes", "reasoning",
              "baseline_best_mean", "baseline_best_std", "baseline_needles_mean",
              "baseline_dup_mean",
              "llm_best_mean", "llm_best_std", "llm_needles_mean", "llm_dup_mean",
              "diff_best", "diff_needles", "diff_dup", "out_dir"]
    with open(sweep_dir / "sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
