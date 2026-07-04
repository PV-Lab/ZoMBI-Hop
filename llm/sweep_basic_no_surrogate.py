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

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import evaluate_llm as E  # noqa: E402

# Two-sided significance level for the diff_best hypothesis test.
SIG_ALPHA: float = 0.05


def _ms(stats: dict, key: str, field: str):
    """stats[key][field] or None."""
    if not stats:
        return None
    return (stats.get(key) or {}).get(field)


# ─── statistical significance of diff_best ──────────────────────────────────────

def welch_significance(baseline_vals, llm_vals, alpha: float = SIG_ALPHA) -> dict:
    """Welch's two-sample t-test on the per-repeat best-objective values.

    Tests H0: mean(LLM) == mean(baseline) with unequal variances (Welch), which
    is appropriate because the baseline and LLM repeats have different spreads
    and small, possibly-unequal n. Returns the two-sided p-value, a boolean
    ``significant`` flag (p < alpha ⇒ diff_best is distinguishable from 0 at the
    (1-alpha) confidence level), and the (1-alpha) confidence interval for the
    mean difference (LLM − baseline). All None when the test is undefined
    (fewer than 2 finite values in either group)."""
    null = {"p_value": None, "significant": None, "t_stat": None, "df": None,
            "ci95_low": None, "ci95_high": None}

    def _clean(v):
        arr = np.array([x for x in (v or []) if x is not None], dtype=float)
        return arr[np.isfinite(arr)]

    a = _clean(baseline_vals)   # baseline
    b = _clean(llm_vals)        # LLM
    if a.size < 2 or b.size < 2:
        return null

    diff = float(b.mean() - a.mean())          # LLM − baseline (matches diff_best sign)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = float(np.sqrt(vb / b.size + va / a.size))

    try:
        from scipy import stats
    except Exception:
        # No scipy: fall back to a normal-approx CI and z-test decision.
        if se == 0.0:
            return {**null, "significant": bool(diff != 0.0), "t_stat": None,
                    "ci95_low": diff, "ci95_high": diff}
        z = diff / se
        p = float(2.0 * (1.0 - _norm_cdf(abs(z))))
        half = 1.959963984540054 * se  # z_{0.975}
        return {"p_value": p, "significant": bool(p < alpha), "t_stat": float(z),
                "df": None, "ci95_low": diff - half, "ci95_high": diff + half}

    if se == 0.0:
        # Both groups have zero variance: difference is deterministic.
        return {"p_value": (0.0 if diff != 0.0 else 1.0),
                "significant": bool(diff != 0.0), "t_stat": None, "df": None,
                "ci95_low": diff, "ci95_high": diff}

    res = stats.ttest_ind(b, a, equal_var=False)   # LLM vs baseline
    df = (vb / b.size + va / a.size) ** 2 / (
        (vb / b.size) ** 2 / (b.size - 1) + (va / a.size) ** 2 / (a.size - 1))
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return {"p_value": float(res.pvalue), "significant": bool(res.pvalue < alpha),
            "t_stat": float(res.statistic), "df": float(df),
            "ci95_low": diff - tcrit * se, "ci95_high": diff + tcrit * se}


def _norm_cdf(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ─── best declared-needle value ─────────────────────────────────────────────────

def _needle_max_from_csv(path: Path):
    """Max of the ``value`` column of a needles.csv, or nan if empty/missing."""
    if not path.exists():
        return float("nan")
    best = float("nan")
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    v = float(r["value"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(v) and (not np.isfinite(best) or v > best):
                    best = v
    except Exception:
        return float("nan")
    return best


_REAL_DB_BEST_NEEDLE = None  # cached: best needle value of the real campaign2 run


def _real_db_best_needle():
    """Best declared-needle value of the real campaign2 trajectory (final snapshot).

    This is the needle counterpart of the real-run ``best_objective`` that seeds
    baseline sample #1, so it is pooled with the RF-repeat needle values. Same for
    every injection point, so compute it once and cache."""
    global _REAL_DB_BEST_NEEDLE
    if _REAL_DB_BEST_NEEDLE is None:
        try:
            needles = E.needles_at_snapshot(E._default_final_snapshot())
            vals = [n.get("value", float("nan")) for n in needles]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            _REAL_DB_BEST_NEEDLE = float(max(vals)) if vals else float("nan")
        except Exception as e:
            print(f"  [best_needle] could not read real-run needles: {e}")
            _REAL_DB_BEST_NEEDLE = float("nan")
    return _REAL_DB_BEST_NEEDLE


def _nanmean(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def best_needle_stats(comparison: dict) -> dict:
    """(baseline_mean, llm_mean, diff) for the best declared-needle value.

    Prefers the ``best_needle`` metric captured upstream (new runs). Falls back to
    reconstructing it from the per-repeat needles.csv artifacts on disk (so an
    older sweep can be retrofitted), pooling the real-run needle with the RF
    baseline repeats to mirror how baseline ``best_objective`` is pooled."""
    bstats = comparison.get("baseline_stats") or {}
    lstats = comparison.get("llm_stats")
    if "best_needle" in bstats:
        bm = _ms(bstats, "best_needle", "mean")
        lm = _ms(lstats, "best_needle", "mean") if lstats else None
        dm = (comparison.get("difference_llm_minus_baseline_mean") or {}).get("best_needle")
        return {"baseline_best_needle": bm, "llm_best_needle": lm, "diff_best_needle": dm}

    out_dir = comparison.get("out_dir")
    if not out_dir:
        return {"baseline_best_needle": None, "llm_best_needle": None, "diff_best_needle": None}
    out_dir = Path(out_dir)

    base_vals = [_real_db_best_needle()]
    base_vals += [_needle_max_from_csv(d / "needles.csv")
                  for d in sorted((out_dir / "baseline_rf").glob("rep*"))]
    llm_vals = [_needle_max_from_csv(d / "needles.csv")
                for d in sorted((out_dir / "continuation").glob("rep*"))]

    bm = _nanmean(base_vals)
    lm = _nanmean(llm_vals) if llm_vals else None
    dm = (lm - bm) if (bm is not None and lm is not None) else None
    return {"baseline_best_needle": bm, "llm_best_needle": lm, "diff_best_needle": dm}


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

    # Best declared-needle value (baseline vs LLM) and its change.
    row.update(best_needle_stats(comparison))

    # p-value for H0: mean(LLM best) == mean(baseline best) (Welch t-test).
    # Small p ⇒ diff_best is unlikely to be noise; also log the 95% CI of the diff.
    sig = welch_significance(_ms(bstats, "best_objective", "values"),
                             _ms(lstats, "best_objective", "values") if lstats else None)
    row["diff_best_p_value"] = sig["p_value"]
    row["diff_best_ci95_low"] = sig["ci95_low"]
    row["diff_best_ci95_high"] = sig["ci95_high"]
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
              "baseline_dup_mean", "baseline_best_needle",
              "llm_best_mean", "llm_best_std", "llm_needles_mean", "llm_dup_mean",
              "llm_best_needle",
              "diff_best", "diff_needles", "diff_dup", "diff_best_needle",
              "diff_best_p_value", "diff_best_ci95_low", "diff_best_ci95_high", "out_dir"]
    with open(sweep_dir / "sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2))


def regenerate_summary(sweep_dir: Path) -> None:
    """Rebuild sweep_summary.{json,csv} for an existing sweep from its per-point
    ``inj_XXX/baseline_vs_llm.json`` artifacts, adding the significance and
    best-declared-needle columns without re-running any ZoMBI-Hop continuations."""
    sweep_dir = Path(sweep_dir)
    inj_dirs = sorted(d for d in sweep_dir.glob("inj_*") if d.is_dir())
    if not inj_dirs:
        raise SystemExit(f"No inj_* directories under {sweep_dir}")
    print(f"Regenerating summary for {len(inj_dirs)} injection points in {sweep_dir}")

    rows = []
    for d in inj_dirs:
        cmp_path = d / "baseline_vs_llm.json"
        if not cmp_path.exists():
            print(f"  [skip] {d.name}: no baseline_vs_llm.json")
            continue
        comparison = json.loads(cmp_path.read_text())
        comparison.setdefault("out_dir", str(d))  # needed for on-disk needle lookup
        rows.append(_flatten(comparison))
        p = rows[-1].get("diff_best_p_value")
        print(f"  {d.name}: diff_best={rows[-1].get('diff_best')}, "
              f"p_value={p}, diff_best_needle={rows[-1].get('diff_best_needle')}")

    rows.sort(key=lambda r: (r.get("injection_iter") is None, r.get("injection_iter")))
    _write_summary(sweep_dir, rows)
    print(f"\nWrote {sweep_dir / 'sweep_summary.json'} and .csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] in ("--regenerate", "-r"):
        if len(args) < 2:
            raise SystemExit("usage: evaluate_llm_sweep.py --regenerate <sweep_dir>")
        regenerate_summary(Path(args[1]))
    else:
        main()
