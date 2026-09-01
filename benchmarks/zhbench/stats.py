"""Paired statistics over seeds, so the tables say what is resolved and what is not.

``RESULTS.md`` reported means. Means alone cannot tell "ZoMBI-Hop is best" from
"ZoMBI-Hop's mean is highest and the gap is a third of one optimum", and on these
landscapes that distinction decides what may be claimed. Recall is quantised at
``1 / n_true`` -- 0.0714 on real3d, 0.0370 on real4d, 0.0147 on real6d -- so a
difference of 0.02 is not a small effect, it is *no* effect plus rounding.

**Everything here is paired by seed, and that pairing is legitimate**: a seed fixes
the initial design, and the initial 48 samples are byte-identical across all six
methods at a given seed (``test_protocol`` asserts it on the artifacts). So seed
``s`` is the same landscape draw and the same starting information for every
method, and the per-seed difference removes that shared variance. Unpaired tests on
n=10 would throw that away.

Two tests are reported for every comparison because they fail in different ways:

* **paired t** uses the magnitudes, and is the more powerful of the two when the
  per-seed differences are roughly symmetric. It is also the one that a single
  outlying seed can carry.
* **sign test** (exact binomial on wins vs losses, ties dropped) uses only the
  direction. It is nearly assumption-free and is the honest test when the metric is
  coarsely quantised, which this one is. Ties are *reported*, never counted as
  evidence either way.

A paired bootstrap CI on the mean difference is included as the effect-size
statement; it is the number to quote when a p-value would over-claim.

No reruns: this reads ``aggregate.csv`` and ``curves.json`` from a finished suite.

    python -m benchmarks.zhbench.stats benchmarks/runs/s1_real_20260824_221242
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np

# Matched-|S| recall is read off each cell's precision-recall curve at a single
# k. It is NOT `posthoc_peak_ratio_at_n_declared`, which metrics.py only writes
# for methods that declare their own optima (the ZoMBI-Hop arms) and leaves NaN
# for every baseline.
_MATCHED_METRIC = "pr_curve_peak_ratio"

_ORDER = ["random", "gp_qucb", "gp_qlogei", "gp_ts", "zombihop", "zombihop_nc5",
          "zombihop_mz0",
          "hebo", "turbo", "rf_bo", "saasbo", "robot"]

_BOOTSTRAP_N = 10000
_BOOTSTRAP_SEED = 0


# --------------------------------------------------------------------------- io

def load(suite_dir: str) -> tuple[list[dict], dict]:
    """Rows of aggregate.csv (numeric where possible) and the curves blob."""
    with open(os.path.join(suite_dir, "aggregate.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("objective", "optimizer", "error", "declared_source"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = float("nan")
    cpath = os.path.join(suite_dir, "curves.json")
    if not os.path.exists(cpath):
        # report.py degrades silently when this file is missing; here it is fatal,
        # because every matched-|S| number in this module comes out of it.
        raise FileNotFoundError(
            f"{cpath} not found. Matched-|S| statistics are computed from the "
            f"per-seed PR curves, which live only in curves.json. Point this at "
            f"the suite run directory under benchmarks/runs/, not at "
            f"benchmarks/results/s1_real (which ships without curves.json)."
        )
    with open(cpath, encoding="utf-8") as fh:
        curves = json.load(fh)
    return rows, curves


def objectives(rows) -> list[str]:
    seen = []
    for r in rows:
        if r["objective"] not in seen:
            seen.append(r["objective"])
    return seen


def optimizers(rows) -> list[str]:
    present = {r["optimizer"] for r in rows}
    return ([o for o in _ORDER if o in present] + sorted(present - set(_ORDER)))


# ------------------------------------------------------------------ extraction

def matched_k(rows, objective: str, reference: str = "zombihop") -> int:
    """The |S| at which every method is compared on `objective`.

    The reference method declares what it declares; everyone else is cut to the
    same size. Rounding the reference's mean declaration count is what produced
    the published tables (real3d 6.30 -> 6, real4d/real6d 15.10 -> 15).
    """
    v = [r["n_declared"] for r in rows
         if r["objective"] == objective and r["optimizer"] == reference
         and np.isfinite(r["n_declared"])]
    if not v:
        raise ValueError(f"no {reference} rows for {objective}")
    return int(round(float(np.mean(v))))


def per_seed_matched(curves, objective: str, optimizer: str, k: int) -> dict[int, float]:
    """seed -> recall at |S| = k, from that cell's own PR curve."""
    out: dict[int, float] = {}
    for cell, c in curves.items():
        if c.get("objective") != objective or c.get("optimizer") != optimizer:
            continue
        ks, pr = c.get("pr_curve_k"), c.get(_MATCHED_METRIC)
        if not ks or not pr:
            continue
        idx = [i for i, kk in enumerate(ks) if kk == k]
        if not idx:
            continue
        out[_seed_of(cell)] = float(pr[idx[0]])
    return out


def per_seed_column(rows, objective: str, optimizer: str, key: str) -> dict[int, float]:
    """seed -> value of an aggregate.csv column."""
    return {int(r["seed"]): float(r[key]) for r in rows
            if r["objective"] == objective and r["optimizer"] == optimizer
            and np.isfinite(r.get(key, float("nan")))}


def _seed_of(cell: str) -> int:
    # cells are named "<objective_spec>__<optimizer>__s<seed>"
    return int(cell.rsplit("__s", 1)[1])


# ------------------------------------------------- matched-|S| distance curves

_MATCHED_CURVES = "matched_curves.json"


def matched_curves(suite_dir: str, reference: str = "zombihop",
                   *, rebuild: bool = False, ks: dict | None = None) -> dict:
    """``dist_to_needles`` at MATCHED |S|, per cell per checkpoint.

    ``curves.json``'s ``by_n`` blocks cannot answer this. There, a baseline's set is
    the post-hoc extraction at ``k = n_true`` while ZoMBI-Hop's is its own ~6
    declarations, and ``metric_dist_to_needles`` caps every unmatched true optimum
    at 0.5 -- so a method that declares fewer sets is charged 0.5 for each optimum
    it never claimed. Plotting those side by side reproduces exactly the unfairness
    that ``fig3`` exists to remove, and it looks like a large ZoMBI-Hop deficit.

    So this recomputes the distance with the SAME post-hoc extractor applied to
    every method's own samples at the same ``|S| = k``, which is the construction
    behind the published "post-hoc recall @|S|" column -- the distance analogue of
    the headline rather than a second, contradictory story.

    Reads only ``points.csv`` and ``true_optima.csv`` from each cell directory, so
    no optimizer is re-run. Cached to ``matched_curves.json``.
    """
    # `ks` bypasses the cache in both directions: it is a DIFFERENT quantity from
    # the one cached here. A combined bundle takes its ZoMBI-Hop rows from one run
    # and its baselines from another, so the matched |S| must be computed on the
    # combined table and pushed down -- deriving it per source directory would score
    # each half at its own |S| and silently make the two halves incomparable, which
    # is the exact failure this whole metric exists to prevent.
    cache = os.path.join(suite_dir, _MATCHED_CURVES)
    if ks is None and os.path.exists(cache) and not rebuild:
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)

    from .metrics import merge_true_optima, posthoc_solution_set, solution_set_scores

    rows, _ = load(suite_dir)
    ks_was_derived = ks is None
    if ks is None:
        ks = {obj: matched_k(rows, obj, reference) for obj in objectives(rows)}
    eval_at = sorted({int(c.split("@")[1]) for c in rows[0] if "@" in c})

    out: dict[str, dict] = {}
    for r in rows:
        cell = _cell_dir(suite_dir, r)
        if cell is None:
            continue
        pts = os.path.join(cell, "points.csv")
        tru = os.path.join(cell, "true_optima.csv")
        if not (os.path.exists(pts) and os.path.exists(tru)):
            continue
        P = np.genfromtxt(pts, delimiter=",", names=True)
        T_raw = np.genfromtxt(tru, delimiter=",", names=True)
        # Score on the ACTUAL compositions, which is what was really deposited and
        # what every other metric in the suite uses.
        xcols = [c for c in P.dtype.names if c.startswith("x_act_")]
        X = np.column_stack([P[c] for c in xcols])
        y = np.asarray(P["y_observed"], dtype=float)
        tcols = [c for c in T_raw.dtype.names if c.startswith("x_")]
        T0 = np.column_stack([np.atleast_1d(T_raw[c]) for c in tcols])
        tv0 = np.atleast_1d(T_raw["value"]).astype(float)
        T, tv = merge_true_optima(T0, tv0)
        k = ks[r["objective"]]
        r_match = float(r.get("match_radius") or 0.05)

        by_n = {}
        for n in eval_at:
            if n > r["n_samples"]:
                continue
            S = posthoc_solution_set(X[:n], y[:n], k=k, min_sep=2.0 * r_match)
            sc = solution_set_scores(S, T, tv, r=r_match)
            by_n[str(n)] = {"dist_to_needles": sc["dist_to_needles"],
                            "peak_ratio": sc["peak_ratio"],
                            "n_declared": sc["n_declared"]}
        out[os.path.basename(cell)] = {
            "objective": r["objective"], "optimizer": r["optimizer"],
            "seed": int(r["seed"]), "matched_k": k, "by_n": by_n,
        }

    if ks_was_derived:
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
    return out


def _cell_dir(suite_dir: str, row: dict) -> str | None:
    """Locate a row's run directory.

    Cell directory names encode the objective SPEC (``real_gp_dim3``) while
    aggregate.csv carries the display name (``real3d``), and the two are not
    derivable from one another in general -- an ensemble spec collapses several
    distinct landscapes onto one dim. Each cell's own ``metrics.json`` records the
    display name, so match on that.
    """
    suffix = f"__{row['optimizer']}__s{int(row['seed'])}"
    for name in sorted(os.listdir(suite_dir)):
        if not name.endswith(suffix):
            continue
        p = os.path.join(suite_dir, name)
        mpath = os.path.join(p, "metrics.json")
        if not (os.path.isdir(p) and os.path.exists(mpath)):
            continue
        with open(mpath, encoding="utf-8") as fh:
            if json.load(fh).get("objective") == row["objective"]:
                return p
    return None


# ------------------------------------------------------------------ statistics

def paired_compare(a: dict[int, float], b: dict[int, float],
                   *, bootstrap: int = _BOOTSTRAP_N) -> dict:
    """Compare `a` against `b` on the seeds they share. Positive favours `a`.

    Returns wins/ties/losses, the paired t test, an exact sign test, and a paired
    bootstrap CI on the mean difference. `ties` are differences that are exactly
    zero -- common here, because recall can only move in steps of 1/n_true.
    """
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    out = {
        "n": n,
        "mean_a": float(np.mean([a[s] for s in seeds])) if seeds else float("nan"),
        "mean_b": float(np.mean([b[s] for s in seeds])) if seeds else float("nan"),
        "mean_diff": float(np.mean(d)) if n else float("nan"),
        "sd_diff": float(np.std(d, ddof=1)) if n > 1 else float("nan"),
        "wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
        "t": float("nan"), "p_t": float("nan"),
        "p_sign": float("nan"),
        "ci_lo": float("nan"), "ci_hi": float("nan"),
    }
    if n < 2 or not np.any(d != 0):
        # All-tie comparisons are real results (nc5 vs random on real4d is exactly
        # 0.0000), but no test has anything to work with. Leave the p-values NaN
        # rather than emitting a misleading 1.0.
        return out

    from scipy import stats as _st
    t, p = _st.ttest_rel(
        np.array([a[s] for s in seeds], dtype=float),
        np.array([b[s] for s in seeds], dtype=float),
    )
    out["t"], out["p_t"] = float(t), float(p)

    eff = out["wins"] + out["losses"]          # ties carry no directional evidence
    if eff:
        out["p_sign"] = float(_st.binomtest(out["wins"], eff, 0.5).pvalue)

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    boot = np.mean(rng.choice(d, size=(bootstrap, n), replace=True), axis=1)
    out["ci_lo"], out["ci_hi"] = (float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5)))
    return out


def resolved(cmp: dict, alpha: float = 0.05) -> bool:
    """A comparison counts as resolved only if BOTH tests agree at `alpha`.

    Deliberately conservative. The point of this module is to stop a mean
    ordering from being reported as a finding, so a claim that only one of the
    two tests supports is exactly the case to leave open.
    """
    return (np.isfinite(cmp["p_t"]) and np.isfinite(cmp["p_sign"])
            and cmp["p_t"] < alpha and cmp["p_sign"] < alpha)


# ---------------------------------------------------------------------- tables

def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def mean_std_table(rows, objective: str, keys: list[str], opts: list[str]) -> list[str]:
    head = "| method | " + " | ".join(keys) + " |"
    sep = "|---" * (len(keys) + 1) + "|"
    out = [head, sep]
    for o in opts:
        cells = []
        for k in keys:
            v = [r[k] for r in rows if r["objective"] == objective
                 and r["optimizer"] == o and np.isfinite(r.get(k, float("nan")))]
            cells.append(f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
                         if len(v) > 1 else ("--" if not v else f"{v[0]:.3f}"))
        out.append(f"| `{o}` | " + " | ".join(cells) + " |")
    return out


def comparison_table(per_seed: dict[str, dict[int, float]], reference: str,
                     opts: list[str]) -> list[str]:
    out = ["| vs | mean diff | 95% CI | W/T/L | paired t | p(t) | p(sign) | resolved |",
           "|---|---|---|---|---|---|---|---|"]
    for o in opts:
        if o == reference or o not in per_seed:
            continue
        c = paired_compare(per_seed[reference], per_seed[o])
        out.append(
            f"| `{o}` | {c['mean_diff']:+.4f} | "
            f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | "
            f"{c['wins']}/{c['ties']}/{c['losses']} | "
            f"{c['t']:+.2f} | {_fmt_p(c['p_t'])} | {_fmt_p(c['p_sign'])} | "
            f"{'**yes**' if resolved(c) else 'no'} |"
        )
    return out


def build(suite_dir: str, reference: str = "zombihop") -> str:
    rows, curves = load(suite_dir)
    objs, opts = objectives(rows), optimizers(rows)
    md: list[str] = [
        f"# Paired statistics -- `{os.path.basename(suite_dir)}`",
        "",
        "Every comparison is paired by seed (seeds fix the initial design, which is "
        "byte-identical across methods), positive favours the reference. `W/T/L` "
        "counts seeds; ties are exact zeros and are dropped from the sign test. "
        "A comparison is marked **resolved** only when the paired t and the exact "
        "sign test *both* clear p<0.05 -- deliberately conservative, because the "
        "purpose here is to stop a mean ordering being reported as a finding.",
        "",
        f"Bootstrap: {_BOOTSTRAP_N} paired resamples, seed {_BOOTSTRAP_SEED}.",
        "",
    ]

    for obj in objs:
        k = matched_k(rows, obj, reference)
        n_true = next((r["n_true_optima"] for r in rows if r["objective"] == obj), float("nan"))
        md += [f"## {obj}", "",
               f"`n_true = {n_true:.0f}`, quantum = 1/n_true = {1.0 / n_true:.4f}, "
               f"matched |S| = {k} (mean declarations by `{reference}`).", ""]

        per_seed = {o: per_seed_matched(curves, obj, o, k) for o in opts}
        per_seed = {o: v for o, v in per_seed.items() if v}
        md += [f"### Matched-|S| recall at |S|={k}, vs `{reference}`", ""]
        md += comparison_table(per_seed, reference, opts) + [""]

        md += ["### Means ± std over seeds", ""]
        md += mean_std_table(
            rows, obj,
            ["peak_ratio", "precision", "reached_ratio_final", "input_cost", "wall_s"],
            opts) + [""]

        # nc2 vs nc5 is a hyperparameter question, not a method question, so it
        # gets its own paired line on the metric it was chosen by.
        if "zombihop" in per_seed and "zombihop_nc5" in opts:
            a = per_seed_column(rows, obj, "zombihop", "peak_ratio")
            b = per_seed_column(rows, obj, "zombihop_nc5", "peak_ratio")
            if a and b:
                c = paired_compare(a, b)
                md += [
                    "### `n_consecutive_converged` 2 vs 5 (declared recall)", "",
                    f"mean diff {c['mean_diff']:+.4f} "
                    f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}], "
                    f"W/T/L {c['wins']}/{c['ties']}/{c['losses']}, "
                    f"t={c['t']:+.2f} p={_fmt_p(c['p_t'])}, "
                    f"sign p={_fmt_p(c['p_sign'])} -- "
                    f"{'**resolved**' if resolved(c) else 'not resolved'}.", "",
                ]

        # Aleks's prediction: standard BO recovers fewer optima than uniform random.
        # Reference is `random` here, and a POSITIVE diff means random wins.
        gp = [o for o in opts if o.startswith("gp_")]
        if "random" in per_seed and gp:
            md += ["### Is standard BO worse than uniform random here?", "",
                   "Positive favours `random` (Aleks's prediction).", ""]
            md += comparison_table(
                {o: per_seed[o] for o in ["random"] + gp if o in per_seed},
                "random", gp) + [""]

    path = os.path.join(suite_dir, "STATS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suite_dir")
    ap.add_argument("--reference", default="zombihop",
                    help="method every other is compared against (default: zombihop)")
    a = ap.parse_args(argv)
    print(build(a.suite_dir, a.reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
