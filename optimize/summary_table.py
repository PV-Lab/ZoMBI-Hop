"""
optimize/summary_table.py
=========================
Markdown summary of a showdown, with rows grouped **by landscape**.

``pareto.write_top_trials_summary`` and ``collect_rerun_summaries`` both group by
trial: all of one configuration's runs, then all of the next. That is the right
shape when each configuration was scored on its own landscapes, because the rows
within a group are comparable to each other and nothing else.

A showdown inverts that. Every configuration sees the SAME landscapes, so the
comparison that matters is *across* configurations on one landscape — which is
only readable when those rows are adjacent. Hence: the first N rows are every
configuration on landscape 1, the next N on landscape 2, and so on. Reading down
a landscape block answers "which configuration handled this landscape best"; the
old layout scattered that across the table.

Written by ``optimize/showdown.py`` (directly, or via ``--summary-only``).
"""

from __future__ import annotations

import os
import sys
import csv
import json
import glob
import datetime
from collections import Counter


# Per-run plots shown as columns (file name -> column heading).
SUMMARY_PLOTS: list[tuple[str, str]] = [
    ("convergence.png",    "convergence"),
    ("needle_values.png",  "needle values"),
    ("conet.png",          "conet"),
    ("conet_uniform.png",  "conet uniform"),
]

# Lines measured before the optimizer loop starts (``run_mobo.N_INIT_LINES``).
# Only used by the legacy points.csv fallback below.
N_INIT_LINES = 2


def _read_csv_column(path: str, column: str) -> list[str]:
    """All values of one column of a CSV, or [] if the file/column is missing."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames is None or column not in r.fieldnames:
                return []
            return [row[column] for row in r]
    except Exception:
        return []


def lines_per_activation(run_dir: str | None) -> list[int]:
    """Lines measured in each activation, in order.

    One "line" is one objective call — the unit the convergence plot's orange
    running-best envelope restarts on, since that envelope resets at every
    activation boundary. So this counts how long each of those sawtooth segments
    ran for.

    Preferred source is ``metrics_over_time.csv``'s ``activation`` column (one row
    per line, exact). Runs written before that column existed fall back to
    ``points.csv``: every line contributes the same fixed number of points, so
    dividing each activation's point count by that constant recovers its line
    count.
    """
    if not run_dir:
        return []
    acts = _read_csv_column(os.path.join(run_dir, "metrics_over_time.csv"), "activation")
    if acts and any(a not in ("", "nan") for a in acts):
        counts: Counter = Counter()
        order: list[str] = []
        for a in acts:
            if a in ("", "nan"):
                continue
            if a not in counts:
                order.append(a)
            counts[a] += 1
        return [counts[a] for a in order]

    # Fallback for legacy runs.
    n_lines = len(_read_csv_column(os.path.join(run_dir, "metrics_over_time.csv"),
                                   "iteration"))
    pt_acts = _read_csv_column(os.path.join(run_dir, "points.csv"), "activation")
    if not n_lines or not pt_acts:
        return []
    per_line = len(pt_acts) / float(n_lines + N_INIT_LINES)
    if per_line <= 0:
        return []
    counts = Counter()
    order = []
    for a in pt_acts:
        if a not in counts:
            order.append(a)
        counts[a] += 1
    return [int(round(counts[a] / per_line)) for a in order]


def mean_full_activations(counts: list[int]) -> float | None:
    """Mean of ``counts``, ignoring the final (time-truncated) activation.

    A run stops on its wall-clock budget, so its last activation is cut off
    mid-flight; averaging it in would understate how long an activation actually
    runs. It is only included when it is the run's *only* activation.
    """
    if not counts:
        return None
    full = counts[:-1] if len(counts) > 1 else counts
    return sum(full) / float(len(full))


def avg_lines_per_activation(run_dir: str | None) -> float | None:
    """``mean_full_activations`` of one run directory's activations."""
    return mean_full_activations(lines_per_activation(run_dir))


def ensure_needle_values_plot(run_dir: str | None) -> None:
    """Generate ``needle_values.png`` for a run that predates it (best-effort).

    The plot is derived entirely from the run's CSVs plus its
    ``ensemble_config.json``, so finished runs can be backfilled without re-running
    the optimizer.
    """
    if not run_dir or os.path.isfile(os.path.join(run_dir, "needle_values.png")):
        return
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import plot_metrics
        plot_metrics.plot_needle_values(run_dir)
    except Exception as exc:
        print(f"  needle_values plot failed for {run_dir}: {exc}")


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ─── dist_to_needles, recomputed with an OPTIMAL assignment ──────────────────────
#
# ``eval_metrics.metric_dist_to_needles`` matches greedily: it walks true_optima in
# list order and lets each take its nearest unclaimed needle. That is order-dependent
# and can cost more than the best pairing — an optimum early in the list can take the
# one needle a later optimum needed, sending the later one to the full penalty even
# though a perfectly good alternative existed. The intended quantity is the MINIMUM
# total matching cost, which ``scipy.optimize.linear_sum_assignment`` computes exactly
# (Hungarian, O(n^3) — trivial at the ~100 needles per run seen here).
#
# The cutoff is also lowered, from ``eval_metrics.UNMATCHED_PENALTY`` (10.0) to a value
# on the same scale as the distances it is averaged with. Measured matched-needle
# distances in these 6d runs are 0.05–0.55 (median 0.16–0.37) and composition L2 on the
# simplex cannot exceed its diameter, ~1.414 — so a 10.0 penalty is 20–200x anything it
# is added to. The consequence is not subtle: on the 8-peak landscape the localization
# term contributes 0.6% of the score, i.e. the metric is ~99% a comparison of how many
# needles were declared against how many exist, and the distances it is named for are
# noise on top. That is also why ``pct_matched`` reads 0.0 everywhere.
#
# TRADEOFF, recorded because it is the reason the old value was chosen: the cutoff is
# the whole anti-spam deterrent. Scoring a real run against the same run plus 200
# needles jittered into the densest optima cluster gives a spam/honest ratio of 51.8x
# at cutoff 10.0 but only 2.6x at 0.5. Lowering it therefore buys localization signal
# with deterrence. The deterrence is recoverable without the cutoff — the cardinality
# error (max-min)/max separates honest (0.000) from that spam (0.879) on its own and
# needs no constant — but it has to be tracked as its own number, which this module
# does not yet do.
#
# The true optima are rebuilt from each run's ``ensemble_config.json`` (deterministic,
# CPU-only), so finished runs are re-scored without re-running the optimizer. That
# needs numpy/scipy plus evaluate.py, which this module otherwise does NOT depend on
# — it is imported by the SLURM epilogue path where the scientific stack may not be
# loaded. Hence the lazy imports and the fall back to the stored metrics.json value:
# a summary must still be writable where the recomputation cannot run.

# Composition-L2 radius past which a needle earns no credit, and the cost charged for
# every unmatched member of the larger set.
#
# As of 2026-08-11 this is ``eval_metrics.UNMATCHED_PENALTY``: that module now does the
# optimal assignment at this cutoff itself, so runs scored after that date already
# carry this definition in their ``metrics.json`` and the recomputation below merely
# reproduces it. It is kept because it is the only way to put OLDER runs — scored
# greedily at 10.0 — on the same axis as new ones, which is exactly what a showdown
# spanning the change needs.
#
# Duplicated as a literal rather than imported from eval_metrics ON PURPOSE: that
# import pulls in numpy/torch/scipy, and this module is imported by the SLURM epilogue
# path where the scientific stack may not be loaded (importing torch there would cost
# seconds, or fail outright). It must be kept equal to eval_metrics.UNMATCHED_PENALTY
# — ``recomputed_dist`` asserts that whenever the stack is importable, so a drift shows
# up as a loud warning on the first run scored rather than as two summaries that
# quietly disagree.
DEFAULT_DIST_CUTOFF = 0.5

# Keyed by the ensemble config's canonical JSON: identical config <=> identical
# landscape, so every repeat of every config on one landscape shares one rebuild.
_OPTIMA_CACHE: dict[str, object] = {}


def _true_optima_for(run_dir: str):
    """True optima of a run's landscape, rebuilt from its ``ensemble_config.json``.

    Returns an ``(n_optima, d)`` array, or None when the landscape cannot be rebuilt
    (no config file, or the scientific stack is unavailable).
    """
    cfg = _read_json(os.path.join(run_dir, "ensemble_config.json"))
    if not cfg:
        return None
    key = json.dumps(cfg, sort_keys=True)
    if key in _OPTIMA_CACHE:
        return _OPTIMA_CACHE[key]
    try:
        import numpy as np
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import evaluate
        ds = evaluate.build_ensemble_ds(cfg, "ensemble")
        T = np.array([np.asarray(o, dtype=float).ravel() for o in ds["true_optima"]])
    except Exception as exc:
        print(f"  [dist] could not rebuild landscape for {run_dir}: {exc}")
        T = None
    _OPTIMA_CACHE[key] = T
    return T


def _needles_of(run_dir: str):
    """Declared needle positions from ``needles.csv`` as an ``(n, d)`` array.

    The dimension is inferred from however many ``x<i>`` columns the file has, so this
    does not need the manifest and works for any run dimension. An empty needles.csv
    (a run that declared nothing) correctly yields an ``(0, d)`` array rather than None
    — that is a real measurement, not a missing one.
    """
    path = os.path.join(run_dir, "needles.csv")
    if not os.path.isfile(path):
        return None
    try:
        import numpy as np
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames:
                return None
            cols = [c for c in r.fieldnames
                    if len(c) > 1 and c[0] == "x" and c[1:].isdigit()]
            if not cols:
                return None
            cols.sort(key=lambda c: int(c[1:]))
            rows = [[float(row[c]) for c in cols] for row in r]
        return np.asarray(rows, dtype=float).reshape(len(rows), len(cols))
    except Exception:
        return None


_CUTOFF_CHECKED = False


def _check_cutoff_matches_eval_metrics() -> None:
    """Warn once if ``DEFAULT_DIST_CUTOFF`` has drifted from ``eval_metrics``.

    The two have to agree or new runs (scored by eval_metrics at write time) and old
    ones (re-scored here) land on different axes, which is precisely the comparison
    this recomputation exists to make possible. Checked lazily and once: by the time
    it runs the scientific stack is already imported, so it costs nothing, and it
    stays silent on the epilogue path where eval_metrics cannot be imported at all.
    """
    global _CUTOFF_CHECKED
    if _CUTOFF_CHECKED:
        return
    _CUTOFF_CHECKED = True
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import eval_metrics
    except Exception:
        return
    if abs(float(eval_metrics.UNMATCHED_PENALTY) - DEFAULT_DIST_CUTOFF) > 1e-12:
        print(f"  ⚠ [dist] DEFAULT_DIST_CUTOFF={DEFAULT_DIST_CUTOFF} but "
              f"eval_metrics.UNMATCHED_PENALTY={eval_metrics.UNMATCHED_PENALTY}; "
              f"re-scored runs will not be comparable to newly-written metrics.json")


def recomputed_dist(run_dir: str, cutoff: float = DEFAULT_DIST_CUTOFF):
    """``dist_to_needles`` for one run under the optimal assignment, or None.

    Same definition as ``eval_metrics.metric_dist_to_needles`` — mean over
    ``max(n_discovered, n_true)`` of the matched distances plus ``cutoff`` for every
    unmatched member of the larger set — with the greedy pairing replaced by the
    minimum-cost one. None means the recomputation could not run, and the caller
    should keep the value stored in metrics.json.
    """
    T = _true_optima_for(run_dir)
    if T is None or len(T) == 0:
        return None
    D = _needles_of(run_dir)
    if D is None:
        return None
    _check_cutoff_matches_eval_metrics()
    if len(D) == 0:
        return float(cutoff)
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        C = np.minimum(np.linalg.norm(D[:, None, :] - T[None, :, :], axis=2), cutoff)
        rows, cols = linear_sum_assignment(C)
        n = max(len(D), len(T))
        return float((C[rows, cols].sum() + cutoff * (n - len(rows))) / n)
    except Exception as exc:
        print(f"  [dist] recompute failed for {run_dir}: {exc}")
        return None


def _md_img(src: str | None, alt: str, width: int = 460) -> str:
    """Markdown-table cell for a plot: a thumbnail linking to the full image.

    Table cells cannot hold block elements, so this uses an inline ``<img>`` (which
    every Markdown renderer passes through) rather than ``![]()`` sizing tricks.
    """
    if not src:
        return "—"
    return f'<a href="{src}"><img src="{src}" alt="{alt}" width="{width}"></a>'


def _fmt(v, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _mean(vals) -> float | None:
    """Mean of the non-missing values, or None if there are none.

    Cells whose run has not finished contribute nothing rather than a zero, so an
    average over a partially-complete block still describes the runs that exist.
    """
    xs = []
    for v in vals:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x:  # drop NaN
            xs.append(x)
    return sum(xs) / len(xs) if xs else None


# Metrics carried through the repeat statistics, in CSV column order.
STAT_METRICS: list[str] = ["dist_to_needles", "dup_fraction",
                          "lines_per_activation", "activations", "runtime_s"]


def _percentile(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile of a sorted-able list (q in [0, 100]).

    Hand-rolled rather than ``numpy.percentile`` only so this module keeps its
    stdlib-only import set — it is imported by the SLURM epilogue path where the
    scientific stack may not be loaded.
    """
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def describe(xs: list[float]) -> dict:
    """mean / median / variance / p5 / p95 / min / max / n of one metric's repeats.

    Variance is the SAMPLE variance (ddof=1): these repeats are a sample of the
    optimizer's run-to-run behaviour, not the whole population, and with n=10 the
    difference from the population form is not negligible. It is undefined for a
    single repeat, which is reported as an empty cell rather than 0 — a lone run
    carries no information about spread, and 0 would claim it carries perfect
    information.
    """
    vals = [x for x in xs if x is not None and x == x]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "variance": None,
                "p5": None, "p95": None, "min": None, "max": None}
    mean = sum(vals) / n
    var = (sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "median": _percentile(vals, 50),
        "variance": var,
        "p5": _percentile(vals, 5),
        "p95": _percentile(vals, 95),
        "min": min(vals),
        "max": max(vals),
    }


def _run_dir_under(base: str) -> str | None:
    """The single ``trial_<n>/run_<k>`` artifact dir under one evaluate.py output dir.

    ``evaluate.py`` nests its output as ``<out-dir>/trial_<n>/run_<k>``; a
    ``--hparams-json`` invocation lands on ``trial_0/run_1``, but the trial number
    is not guaranteed, so this globs rather than hard-coding it.
    """
    hits = sorted(glob.glob(os.path.join(base, "trial_*", "run_*")))
    for h in hits:
        if os.path.isfile(os.path.join(h, "metrics.json")):
            return h
    return hits[0] if hits else None


def find_run_dirs(showdown_dir: str, config: str,
                  landscape: int) -> list[tuple[int, str]]:
    """Every repeat of one (config, landscape) cell, as ``(repeat, run_dir)`` in order.

    Repeats live in sibling directories suffixed ``__r<N>``. A showdown planned before
    repeats existed (or with ``--n-repeats 1``) has no suffix at all, so that layout is
    accepted too and reported as the single repeat 1 — which keeps every previously-run
    showdown summarisable by this script.
    """
    runs = os.path.join(showdown_dir, "runs")
    out: list[tuple[int, str]] = []
    for base in sorted(glob.glob(os.path.join(runs, f"{config}__ls{landscape}__r*"))):
        try:
            rep = int(base.rsplit("__r", 1)[1])
        except (IndexError, ValueError):
            continue
        rd = _run_dir_under(base)
        if rd:
            out.append((rep, rd))
    if out:
        return sorted(out)
    legacy = _run_dir_under(os.path.join(runs, f"{config}__ls{landscape}"))
    return [(1, legacy)] if legacy else []


def write_repeat_csvs(reps: dict[tuple[str, int], list[dict]],
                      configs: list[dict], landscapes: list[int],
                      out_dir: str) -> tuple[str, str]:
    """Write the raw per-repeat CSV and the aggregated statistics CSV.

    Two files rather than one because they answer different questions and have
    different shapes:

    ``showdown_runs.csv`` is the raw ledger — one row per completed run, every value
    exactly as measured. It is the file to re-analyse from (a significance test, a
    box plot), and the only one that survives a change of mind about which statistics
    matter.

    ``showdown_stats.csv`` is long-format (one row per group × metric) rather than
    one column per statistic per metric, so it stays readable as metrics are added
    and filters cleanly with ``df[df.metric == 'dist_to_needles']``. It aggregates at
    three levels: each cell, each config across landscapes, and each landscape across
    configs — matching the three tables in the Markdown summary.
    """
    runs_path = os.path.join(out_dir, "showdown_runs.csv")
    with open(runs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "landscape", "repeat"] + STAT_METRICS + ["run_dir"])
        for c in configs:
            for ls in landscapes:
                for r in reps.get((c["name"], ls), []):
                    w.writerow([c["name"], ls, r["repeat"]]
                               + [r[m] for m in STAT_METRICS]
                               + [os.path.relpath(r["run_dir"], out_dir)])

    # (level, group label, config, landscape, the rows it pools)
    groups: list[tuple[str, str, str, str, list[dict]]] = []
    for c in configs:
        for ls in landscapes:
            groups.append(("cell", f"{c['name']}__ls{ls}", c["name"], str(ls),
                           reps.get((c["name"], ls), [])))
    for c in configs:
        pooled = [r for ls in landscapes for r in reps.get((c["name"], ls), [])]
        groups.append(("config", c["name"], c["name"], "all", pooled))
    for ls in landscapes:
        pooled = [r for c in configs for r in reps.get((c["name"], ls), [])]
        groups.append(("landscape", str(ls), "all", str(ls), pooled))

    stats_path = os.path.join(out_dir, "showdown_stats.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "group", "config", "landscape", "metric",
                    "n", "mean", "median", "variance", "p5", "p95", "min", "max"])
        for level, label, cname, lname, rows in groups:
            for m in STAT_METRICS:
                d = describe([r[m] for r in rows])
                w.writerow([level, label, cname, lname, m, d["n"],
                            d["mean"], d["median"], d["variance"],
                            d["p5"], d["p95"], d["min"], d["max"]])
    return runs_path, stats_path


# ─── the by-configuration bar chart ─────────────────────────────────────────────

CONFIG_BAR_PLOT = "config_means.png"

# Panels of that chart: (metric key, panel title, bar colour). Both metrics are
# "lower is better", which the caption states rather than the axes.
BAR_METRICS: list[tuple[str, str, str]] = [
    ("dist_to_needles", "dist to needles", "#2a78d6"),
    ("dup_fraction",    "dup fraction",    "#eb6834"),
]

# Half-width multiplier for a 95% interval under the normal approximation. The
# within-landscape form pools ~n_repeats x n_landscapes runs, so the t correction is
# under 3% and not worth carrying a t table for.
Z95 = 1.96


def _mean_and_error(cells: list[dict], within: bool) -> tuple[float | None, float | None]:
    """Mean over one configuration's cells, and the half-width of its 95% CI.

    ``cells`` is one ``describe`` dict per landscape. The mean is the mean of the
    cell means — exactly the number the "Averages by configuration" table carries,
    so the bars and the table cannot drift apart.

    There are two defensible error bars on that mean, and the choice matters because
    the landscapes are a FIXED set that every configuration saw:

    ``within=True`` uses only the repeat-to-repeat variance inside each landscape,
    ``sqrt(sum(var_i / n_i)) / k``. This treats the landscapes as constants, which is
    what the paired design makes them: landscape-to-landscape variation is common to
    every configuration and cancels in the comparison, so folding it in would widen
    every bar by the same large amount and bury the differences the showdown exists
    to measure.

    ``within=False`` is the fallback for when some cell has a single repeat and no
    variance to speak of: the spread of the cell means themselves, ``stdev / sqrt(k)``.
    That does carry landscape variation and is correspondingly wider.

    Either error is None when it cannot be computed (one cell, or none).
    """
    used = [d for d in cells if d["n"] and d["mean"] is not None]
    if not used:
        return None, None
    k = len(used)
    mean = sum(d["mean"] for d in used) / k
    if within:
        if any(d["variance"] is None for d in used):
            return mean, None
        se = (sum(d["variance"] / d["n"] for d in used) ** 0.5) / k
    else:
        if k < 2:
            return mean, None
        var = sum((d["mean"] - mean) ** 2 for d in used) / (k - 1)
        se = (var / k) ** 0.5
    return mean, Z95 * se


def write_config_bar_chart(reps: dict[tuple[str, int], list[dict]],
                           configs: list[dict], landscapes: list[int],
                           out_dir: str) -> tuple[str, bool] | None:
    """Bar chart of mean ``dist_to_needles`` / ``dup_fraction`` per configuration.

    The by-configuration table is the headline comparison, and with repeats its
    numbers are means of a distribution — five means in a row give no sense of
    whether the gaps between them survive run-to-run noise. This draws the same five
    means with a 95% CI on each, which answers that at a glance.

    Returns ``(file name relative to out_dir, used_within_landscape_error)``, or None
    when matplotlib is unavailable — this module is imported by the SLURM epilogue
    path where the scientific stack may not be loaded, and a missing chart must not
    cost the summary.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [chart] config bar chart skipped: {exc}")
        return None

    names = [c["name"] for c in configs]
    stats = {(n, m): [describe([r[m] for r in reps.get((n, ls), [])])
                      for ls in landscapes]
             for n in names for m, _, _ in BAR_METRICS}
    # A cell that has not finished (n == 0) is simply absent from the average; a cell
    # with exactly one repeat has no within-landscape variance, and that is what forces
    # the whole chart onto the wider across-landscape error (one error definition per
    # chart — two would make the bars incomparable).
    within = all(d["variance"] is not None
                 for cs in stats.values() for d in cs if d["n"])

    fig, axes = plt.subplots(1, len(BAR_METRICS),
                             figsize=(5.4 * len(BAR_METRICS), 4.2))
    for ax, (metric, title, colour) in zip(axes, BAR_METRICS):
        pairs = [_mean_and_error(stats[(n, metric)], within) for n in names]
        idx = [i for i, (mu, _) in enumerate(pairs) if mu is not None]
        mus = [pairs[i][0] for i in idx]
        errs = [pairs[i][1] or 0.0 for i in idx]

        ax.bar(idx, mus, width=0.62, color=colour, zorder=3, yerr=errs, capsize=4,
               error_kw={"ecolor": "#3b3a37", "elinewidth": 1.4, "capthick": 1.4,
                         "zorder": 4})
        top = max([m + e for m, e in zip(mus, errs)], default=1.0) or 1.0
        for i, mu, err in zip(idx, mus, errs):
            ax.text(i, mu + err + top * 0.03, f"{mu:.3f}", ha="center", va="bottom",
                    fontsize=9, color="#52514e")

        ax.set_title(title, fontsize=11.5, color="#0b0b0b", pad=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9.5, color="#52514e")
        ax.tick_params(axis="y", labelsize=9, colors="#52514e", length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, top * 1.22)
        ax.grid(axis="y", color="#e4e3e0", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c4")

    scope = ("repeat-to-repeat noise, landscapes held fixed" if within
             else "spread across landscape means")
    fig.suptitle(f"Mean by configuration, ±95% CI ({scope}) — lower is better",
                 fontsize=12, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, CONFIG_BAR_PLOT)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return CONFIG_BAR_PLOT, within


def write_landscape_summary(showdown_dir: str, out_path: str | None = None, *,
                            dist_cutoff: float = DEFAULT_DIST_CUTOFF,
                            legacy_dist: bool = False) -> str:
    """Write the landscape-grouped Markdown table for a finished showdown.

    Image paths are written relative to the Markdown file so it renders in place.
    Cells for runs that have not finished (or failed) show as ``—`` rather than
    being dropped, so a partially-complete campaign is still readable and the gaps
    are visible.

    ``dist_to_needles`` is recomputed per run with the optimal assignment and a cutoff
    on the scale of the distances (see ``recomputed_dist``) rather than read from
    metrics.json, so the summary and its CSVs disagree with metrics.json by design —
    the note under the results table records which definition and cutoff produced the
    numbers. ``dist_cutoff=10`` reproduces the stored metric's weighting and
    ``legacy_dist=True`` the stored values verbatim.
    """
    showdown_dir = os.path.abspath(showdown_dir)
    manifest = _read_json(os.path.join(showdown_dir, "showdown_manifest.json"))
    if not manifest:
        raise SystemExit(f"no showdown_manifest.json in {showdown_dir}")

    configs = manifest["configs"]
    landscapes = manifest["landscapes"]
    n_repeats = int(manifest.get("n_repeats", 1) or 1)
    out_path = out_path or os.path.join(showdown_dir, "showdown_summary.md")
    out_dir = os.path.dirname(out_path)

    L: list[str] = []
    A = L.append
    A(f"# Showdown — {manifest['dim']}D ensemble, "
      f"{len(configs)} configs × {len(landscapes)} landscapes")
    A("")
    A(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')}. "
      f"Planned {manifest['generated']}.")
    A("")
    A(f"Every configuration was run on the **same** {len(landscapes)} ensemble "
      f"landscapes (Sobol indices `{landscapes}`, seed `{manifest['landscape_seed']}`) "
      f"at a {manifest['time_limit_hours']} h per-run budget, so differences between "
      "rows within a landscape block are attributable to the hyperparameters rather "
      "than to landscape luck.")
    A("")
    A("**Rows are grouped by landscape**, not by configuration: read down a block to "
      "compare all configurations on one landscape.")
    A("")

    A("## Configurations")
    A("")
    A("| config | selected for | rank | source run | trial | source dist | source dup |")
    A("|---|---|---|---|---|---|---|")
    for c in configs:
        m = c.get("source_metrics") or {}
        A(f"| `{c['name']}` | {c.get('selected_for') or '—'} | "
          f"{c.get('rank') if c.get('rank') is not None else '—'} | "
          f"`{c.get('source_run') or '—'}` | {c.get('trial') if c.get('trial') is not None else '—'} | "
          f"{_fmt(m.get('dist_to_needles'))} | {_fmt(m.get('dup_fraction'))} |")
    A("")
    A("`selected for` is the objective the configuration topped on the source Pareto "
      "front; `source dist` / `source dup` are its metrics THERE (each on its own "
      "landscapes), which is exactly what this showdown re-measures on common ground.")
    A("")

    A("## Results")
    A("")
    header = (["landscape", "config", "n", "dist to needles", "dup fraction",
               "lines/activation", "activations", "runtime (s)"]
              + [h for _, h in SUMMARY_PLOTS])
    A("| " + " | ".join(header) + " |")
    A("|" + "|".join(["---"] * len(header)) + "|")

    # Every repeat of every cell, keyed by (config, landscape). Collected while writing
    # the results table so the aggregate tables, the winners table and the CSVs below
    # can reuse it instead of re-reading the CSVs (the legacy fallback walks every
    # point, so a second pass over a 10-repeat campaign is not cheap).
    reps: dict[tuple[str, int], list[dict]] = {}
    n_rows = n_missing = n_dist_fallback = 0
    for ls in landscapes:
        for i, c in enumerate(configs):
            found = find_run_dirs(showdown_dir, c["name"], ls)
            rows: list[dict] = []
            for rep, run_dir in found:
                met = _read_json(os.path.join(run_dir, "metrics.json"))
                if not met:
                    continue
                # Only repeat 1 is rendered, so it is the only one worth backfilling.
                if rep == 1:
                    ensure_needle_values_plot(run_dir)
                act_lines = lines_per_activation(run_dir)
                dist = met.get("dist_to_needles")
                if not legacy_dist:
                    fixed = recomputed_dist(run_dir, dist_cutoff)
                    if fixed is None:
                        n_dist_fallback += 1
                    else:
                        dist = fixed
                rows.append({
                    "repeat": rep,
                    "run_dir": run_dir,
                    "dist_to_needles": dist,
                    "dup_fraction": met.get("dup_fraction"),
                    "lines_per_activation": mean_full_activations(act_lines),
                    "activations": len(act_lines) or None,
                    "runtime_s": met.get("runtime_s"),
                })
            reps[(c["name"], ls)] = rows
            if len(rows) < n_repeats:
                n_missing += n_repeats - len(rows)

            # The Markdown carries the MEAN over repeats; the spread lives in the CSVs,
            # because five statistics per cell in a table nobody can read is not a
            # summary. `n` makes an under-filled cell visible at a glance.
            st = {m: describe([r[m] for r in rows]) for m in STAT_METRICS}
            plot_dir = next((rd for rp, rd in found if rp == 1),
                            found[0][1] if found else None)
            # Name the landscape once per block; repeating it on every row is what
            # makes the grouping hard to see.
            cells = [f"**{ls}**" if i == 0 else "",
                     f"`{c['name']}`",
                     f"{len(rows)}/{n_repeats}",
                     _fmt(st["dist_to_needles"]["mean"]),
                     _fmt(st["dup_fraction"]["mean"]),
                     _fmt(st["lines_per_activation"]["mean"], 1),
                     _fmt(st["activations"]["mean"], 1),
                     _fmt(st["runtime_s"]["mean"], 1)]
            for fname, head in SUMMARY_PLOTS:
                path = os.path.join(plot_dir, fname) if plot_dir else None
                rel = (os.path.relpath(path, out_dir)
                       if path and os.path.isfile(path) else None)
                cells.append(_md_img(rel, f"{c['name']} ls{ls} {head}"))
            A("| " + " | ".join(cells) + " |")
            n_rows += 1

    # Mean-per-cell view, which is what the aggregate/winner tables consume: they
    # compare cells, and a cell is now a distribution summarised by its mean.
    cell: dict[tuple[str, int], dict] = {
        k: {
            "dist": describe([r["dist_to_needles"] for r in v])["mean"],
            "dup": describe([r["dup_fraction"] for r in v])["mean"],
            "lines": describe([r["lines_per_activation"] for r in v])["mean"],
            "acts": describe([r["activations"] for r in v])["mean"],
            "runtime": describe([r["runtime_s"] for r in v])["mean"],
        }
        for k, v in reps.items()
    }

    A("")
    A("Every number in the table above is a MEAN over that cell's repeats; `n` is how "
      "many completed. Median, variance, p5, p95 and the individual per-repeat values "
      "are in `showdown_stats.csv` and `showdown_runs.csv` beside this file — the "
      "spread is the point of repeating, and it does not fit in a table cell.")
    A("")
    A("`lines/activation` is the mean number of measured lines (objective calls) per "
      "activation — the length of one orange running-best segment in the convergence "
      "plot, which restarts at every activation. The final activation is excluded "
      "because the wall-clock budget cuts it off mid-flight. `activations` is how many "
      "the run got through in total.")
    A("")
    A("`needle values` plots each needle at the iteration it was declared on, against "
      "the landscape's true best (green) and the best objective value the run actually "
      "observed (blue). Observed Y can sit above true best: each measurement carries "
      "multiplicative output noise.")
    A("")
    if legacy_dist:
        A("`dist to needles` is the value stored in each run's `metrics.json`, matched "
          "GREEDILY (`--legacy-dist`).")
    else:
        A(f"`dist to needles` is **recomputed here**, not read from `metrics.json`. "
          f"Needle positions come from each run's `needles.csv` and the true optima "
          f"are rebuilt from its `ensemble_config.json`; the two sets are then paired "
          f"by MINIMUM-COST assignment (`scipy.optimize.linear_sum_assignment`) with "
          f"each matched distance capped at `{dist_cutoff:g}`, which is also the "
          f"penalty charged for every unmatched needle or optimum.")
        A("")
        A(f"Since 2026-08-11 that is exactly what `eval_metrics.metric_dist_to_needles` "
          f"does at write time, so for runs scored after that date this column simply "
          f"REPRODUCES `metrics.json` and the recomputation is a no-op. Its purpose is "
          f"runs scored BEFORE it: those were matched greedily — walking the optima in "
          f"list order and letting each take its nearest unclaimed needle, which is "
          f"order-dependent and can only over-state the distance — at an unmatched "
          f"penalty of 10.0. Recomputing puts both eras on one axis, which a showdown "
          f"spanning the change needs.")
        A("")
        A(f"The penalty moved from 10.0 to {dist_cutoff:g} because of scale: measured "
          f"matched distances are 0.05–0.55 and composition L2 cannot exceed ~1.414, "
          f"so 10.0 was 20–200x anything it was averaged with and left the score ~99% "
          f"a needle-COUNT comparison with the distances as noise on top. The cost is "
          f"deterrence — a run padded with 200 needles scattered through the densest "
          f"optima cluster scores 51.8x an honest run at 10.0 but only 2.6x at 0.5. "
          f"Over-declaration is still penalised, via the `(n_declared - n_true)` term, "
          f"just no longer to the exclusion of everything else. Use `--dist-cutoff 10` "
          f"for the old weighting, or `--legacy-dist` for the stored values verbatim.")
    A("")
    if n_dist_fallback:
        A(f"⚠ {n_dist_fallback} run(s) could not be re-scored (no `needles.csv` / "
          "`ensemble_config.json`, or the landscape could not be rebuilt); those cells "
          "keep their stored greedy value.")
        A("")
    if n_missing:
        A(f"⚠ {n_missing} of {n_rows * n_repeats} runs are missing `metrics.json` — "
          "they have not finished or they failed. Their cells are kept (see the `n` "
          "column) so the gaps stay visible; every statistic covers only the repeats "
          "that exist.")
        A("")

    def agg_table(title: str, label: str, groups: list[tuple[str, list[dict]]],
                  note: str) -> None:
        """One aggregate table: a mean of each metric over a group of cells."""
        A(f"## {title}")
        A("")
        A(f"| {label} | runs | dist to needles | dup fraction | lines/activation | "
          "activations | runtime (s) |")
        A("|---|---|---|---|---|---|---|")
        for name, cs in groups:
            n = sum(1 for c in cs if c["dist"] is not None)
            A(f"| {name} | {n}/{len(cs)} | "
              f"{_fmt(_mean(c['dist'] for c in cs))} | "
              f"{_fmt(_mean(c['dup'] for c in cs))} | "
              f"{_fmt(_mean(c['lines'] for c in cs), 1)} | "
              f"{_fmt(_mean(c['acts'] for c in cs), 1)} | "
              f"{_fmt(_mean(c['runtime'] for c in cs), 1)} |")
        A("")
        A(note)
        A("")

    agg_table("Averages by configuration", "config",
              [(f"`{c['name']}`", [cell[(c["name"], ls)] for ls in landscapes])
               for c in configs],
              "Each row averages one configuration over all "
              f"{len(landscapes)} landscapes — the headline comparison, since every "
              "configuration saw the same set. `runs` is how many of those cells "
              "have finished; the averages cover only those.")

    chart = write_config_bar_chart(reps, configs, landscapes, out_dir)
    if chart:
        chart_path, within = chart
        A(f"![Mean dist to needles and dup fraction by configuration]({chart_path})")
        A("")
        A("Bars are the two headline columns of the table above — the same means, so "
          "they cannot drift apart from it.")
        A("")
        if within:
            A(f"Error bars are the 95% CI of each mean, built from the repeat-to-repeat "
              f"variance WITHIN each landscape (`1.96 · sqrt(Σ var/n) / {len(landscapes)}`). "
              f"Landscape-to-landscape variation is deliberately excluded: every "
              f"configuration saw the same {len(landscapes)} landscapes, so that "
              f"variation is common to all five bars and cancels in the comparison. "
              f"Including it would widen every bar by the same large amount and hide "
              f"the differences this showdown exists to measure. Two bars whose "
              f"intervals do not overlap differ by more than run-to-run noise; two "
              f"that overlap heavily are not separated by this campaign.")
        else:
            A(f"At least one cell has a single repeat and therefore no within-landscape "
              f"variance, so the error bars fall back to the 95% CI across the "
              f"{len(landscapes)} landscape means (`1.96 · stdev / sqrt(k)`). That "
              f"interval includes landscape-to-landscape variation, which is common to "
              f"every configuration and so is wider than the comparison between "
              f"configurations warrants — read it as a bound, not as the resolution of "
              f"the comparison.")
        A("")

    agg_table("Averages by landscape", "landscape",
              [(f"**{ls}**", [cell[(c["name"], ls)] for c in configs])
               for ls in landscapes],
              "Each row averages all configurations on one landscape, which measures "
              "the landscape rather than the hyperparameters: a high mean distance "
              "means every configuration struggled there.")

    # The spread section. Imported here rather than at module scope because it needs
    # matplotlib, and this module is imported by the SLURM epilogue path where the
    # scientific stack may not be loaded; it returns [] if it cannot draw anything,
    # so a missing section never costs the summary.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import variance_plots
        L.extend(variance_plots.variance_section(
            reps, [c["name"] for c in configs], landscapes, out_dir))
    except Exception as exc:
        print(f"  [chart] variance section skipped: {exc}")

    A("## Per-landscape winners")
    A("")
    A("| landscape | best dist to needles | best dup fraction |")
    A("|---|---|---|")
    for ls in landscapes:
        rows = [(c["name"], float(d["dist"]),
                 float(d["dup"]) if d["dup"] is not None else float("nan"))
                for c in configs
                for d in [cell[(c["name"], ls)]]
                if d["dist"] is not None]
        if not rows:
            A(f"| {ls} | — | — |")
            continue
        bd = min(rows, key=lambda r: r[1])
        bu = min(rows, key=lambda r: r[2])
        A(f"| {ls} | `{bd[0]}` ({bd[1]:.4f}) | `{bu[0]}` ({bu[2]:.4f}) |")
    A("")

    runs_csv, stats_csv = write_repeat_csvs(reps, configs, landscapes, out_dir)

    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  summary ({n_rows} cells, {n_missing} run(s) missing) -> {out_path}")
    print(f"  raw per-repeat values -> {runs_csv}")
    print(f"  repeat statistics     -> {stats_csv}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("showdown_dir", help="a showdown output directory")
    ap.add_argument("--out", default=None, help="output Markdown path")
    ap.add_argument("--dist-cutoff", type=float, default=DEFAULT_DIST_CUTOFF,
                    help="unmatched-needle penalty used when recomputing "
                         "dist_to_needles, and the cap on any single matched pair. "
                         "The default is on the scale of the distances themselves; "
                         "pass 10 for eval_metrics.UNMATCHED_PENALTY, which weights "
                         "needle count ~99%% of the score but is a far stronger "
                         "deterrent against spurious needles (default: %(default)s)")
    ap.add_argument("--legacy-dist", action="store_true",
                    help="report the greedy dist_to_needles stored in metrics.json "
                         "instead of recomputing it with an optimal assignment")
    a = ap.parse_args()
    write_landscape_summary(a.showdown_dir, a.out,
                            dist_cutoff=a.dist_cutoff, legacy_dist=a.legacy_dist)
