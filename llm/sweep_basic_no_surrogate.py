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

import evaluate_llm as E  # noqa: E402  (also sets sys.path for optimize/ + repo root)
import run_mobo as R  # noqa: E402  (NUM_EXPERIMENTS: droplets per BO iteration)

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

    # Overlaid running-best convergence: baseline vs LLM per injection iteration.
    print("\n[plot] convergence comparison …")
    plot_convergence_comparison(sweep_dir)

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


# ─── convergence comparison plot ─────────────────────────────────────────────────
#
# Unlike the cadence sweeps (sweep_basic_surrogate / sweep_volume_control), this
# experiment sweeps the *injection iteration* rather than an injection cadence, and
# every run shares the SAME real-campaign prefix up to its injection point (the
# prefix is deterministic replayed data, not a fresh run). So the analogous plot is:
#
#   * a single BASELINE line — the real campaign trajectory itself (the true
#     "no-LLM, original-hyperparameter" run), on which every LLM line coincides up
#     to its injection iteration. It is one deterministic run, so it carries no CI
#     band.
#   * one LLM line per swept injection iteration (5, 10, … 40), each the mean±95% CI
#     over its RF continuation repeats, drawn ONLY from its injection iteration
#     onward — so it branches off the baseline exactly at injection and never before.
#
# The x-axis is the ZoMBI-Hop iteration. The real prefix records a variable number
# of real droplets per iteration while the RF continuation adds a fixed
# ``NUM_EXPERIMENTS`` per iteration, so cumulative droplets are NOT comparable across
# runs — the iteration axis is. We recover it from each injection point's
# (iteration → cumulative-points) anchor (mapping.n_points_at_injection).

def _running_best_from_points(path: Path):
    """Running-best (cumulative max) Objective from a rep's points.csv ``Y`` column.

    points.csv holds the FULL trajectory from point 0 (the real-run prefix + the RF
    continuation)."""
    if not path.exists():
        return None
    ys = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    ys.append(float(r["Y"]))
                except (KeyError, TypeError, ValueError):
                    continue
    except Exception:
        return None
    if not ys:
        return None
    return np.maximum.accumulate(np.asarray(ys, float))


def _ci95_halfwidth(std, n):
    """Half-width of the 95% CI for the mean (Student-t for small n; 1.96 fallback)."""
    if n < 2:
        return np.zeros_like(std)
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        t = 1.959963984540054
    return t * std / np.sqrt(n)


def _n_bo_iterations(rep_dir: Path):
    """BO iterations a rep ran, from its metrics_over_time.csv (one row per
    iteration). For a continuation this is the post-injection iteration count
    (== budget_iterations). None if unavailable."""
    mot = rep_dir / "metrics_over_time.csv"
    if not mot.exists():
        return None
    try:
        with open(mot, newline="", encoding="utf-8") as f:
            return max(sum(1 for _ in csv.reader(f)) - 1, 0)
    except Exception:
        return None


def _continuation_iteration_curve(rep_dir: Path, n_inj: int, batch: int):
    """Running-best per iteration for one RF-continuation rep, indexed so element 0
    is the injection iteration (running-best at the injection point) and element j is
    after j post-injection iterations. The continuation adds a fixed ``batch``
    droplets per iteration, so iteration j ends at droplet ``n_inj + j*batch``."""
    rb = _running_best_from_points(rep_dir / "points.csv")
    if rb is None:
        return None
    L = rb.size
    n_inj = min(n_inj, L)
    n_cont = _n_bo_iterations(rep_dir)
    if not n_cont or n_cont <= 0:
        n_cont = max((L - n_inj) // batch, 0)
    idx = np.clip(n_inj + np.arange(n_cont + 1) * batch - 1, 0, L - 1)
    return rb[idx]


def _mean_ci_band(curves):
    """(mean, halfwidth, n) over a list of monotone per-iteration curves. Shorter
    curves (early-converged reps) are forward-filled with their final value to the
    common max length, so a converged run adds a flat tail rather than dropping out."""
    curves = [c for c in curves if c is not None and len(c)]
    if not curves:
        return None
    L = max(len(c) for c in curves)
    M = np.vstack([np.concatenate([c, np.full(L - len(c), c[-1])]) for c in curves])
    mean = M.mean(axis=0)
    std = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(L)
    return mean, _ci95_halfwidth(std, M.shape[0]), M.shape[0]


def plot_convergence_comparison(sweep_dir: Path, out_png: Path | None = None) -> Path | None:
    """Overlay running-best-Objective convergence on a ZoMBI-Hop-iteration axis: the
    deterministic real-campaign baseline plus one mean±95% CI LLM line per swept
    injection iteration, each branching off the baseline exactly at its injection."""
    import matplotlib.pyplot as plt  # Agg backend already selected via evaluate_llm

    sweep_dir = Path(sweep_dir)
    if out_png is None:
        out_png = sweep_dir / "convergence_comparison.png"
    inj_dirs = sorted(d for d in sweep_dir.glob("inj_*") if d.is_dir())
    if not inj_dirs:
        print(f"  [plot] no inj_* directories under {sweep_dir}; skipped")
        return None
    batch = int(R.NUM_EXPERIMENTS)

    # (injection_iter, n_points_at_injection) anchors of the shared real prefix.
    anchors = []  # (iter, cum_points)
    inj_info = []  # (dir, iter, n_inj)
    for d in inj_dirs:
        cmp_path = d / "baseline_vs_llm.json"
        if not cmp_path.exists():
            continue
        m = json.loads(cmp_path.read_text()).get("mapping", {})
        it = m.get("injection_iter")
        n_inj = m.get("n_points_at_injection")
        if it is None or n_inj is None:
            continue
        anchors.append((int(it), int(n_inj)))
        inj_info.append((d, int(it), int(n_inj)))
    if not anchors:
        print(f"  [plot] no injection anchors under {sweep_dir}; skipped")
        return None
    anchors = sorted(set(anchors))
    anchor_iters = np.array([0] + [a[0] for a in anchors], float)
    anchor_pts = np.array([0.0] + [a[1] for a in anchors], float)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # ── Baseline: the real campaign run itself (deterministic; no CI) ──────────────
    # Its running-best is the shared prefix embedded in every run; read it from the
    # longest available prefix (the largest-iteration injection point).
    d_max, it_max, n_max = max(inj_info, key=lambda t: t[2])
    rb_real = None
    for sub in ("baseline_rf", "continuation"):
        for r in sorted((d_max / sub).glob("rep*")):
            rb_real = _running_best_from_points(r / "points.csv")
            if rb_real is not None:
                break
        if rb_real is not None:
            break
    if rb_real is not None:
        base_iters = np.arange(0, it_max + 1)
        # iteration → cumulative droplet (piecewise-linear over the real anchors).
        base_pts = np.interp(base_iters, anchor_iters, anchor_pts)
        base_idx = np.clip(base_pts.astype(int) - 1, 0, rb_real.size - 1)
        ax.plot(base_iters, rb_real[base_idx], color="#111111", lw=2.4, zorder=6,
                label="baseline: real campaign (original HPs)")

    # ── One LLM line per swept injection iteration (mean ± 95% CI) ─────────────────
    cmap = plt.cm.viridis
    all_iters = [it for _, it, _ in inj_info]
    lo, hi = min(all_iters), max(all_iters)
    plotted = 0
    for d, it, n_inj in sorted(inj_info, key=lambda t: t[1]):
        curves = [_continuation_iteration_curve(r, n_inj, batch)
                  for r in sorted((d / "continuation").glob("rep*"))]
        band = _mean_ci_band(curves)
        if band is None or band[0].size < 2:
            continue  # nothing to draw past injection (e.g. budget 0)
        mean, half, n = band
        x = it + np.arange(mean.size)  # iterations from injection onward
        color = cmap((it - lo) / (hi - lo)) if hi > lo else cmap(0.5)
        ax.plot(x, mean, color=color, lw=1.8, zorder=5,
                label=f"LLM inject @ iter {it} (n={n})")
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.14,
                        linewidth=0, zorder=3)
        plotted += 1

    if plotted == 0 and rb_real is None:
        plt.close(fig)
        print(f"  [plot] no running-best curves under {sweep_dir}; skipped")
        return None

    ax.set_xlabel("ZoMBI-Hop iteration")
    ax.set_ylabel("running-best Objective")
    ax.set_title("Convergence: baseline vs LLM by injection iteration\n"
                 "(LLM lines: mean ± 95% CI over RF repeats)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", ncol=1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")
    return out_png


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
            raise SystemExit("usage: sweep_basic_no_surrogate.py --regenerate <sweep_dir>")
        regenerate_summary(Path(args[1]))
    elif args and args[0] in ("--plot", "-p"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_no_surrogate.py --plot <sweep_dir>")
        plot_convergence_comparison(Path(args[1]))
    else:
        main()
