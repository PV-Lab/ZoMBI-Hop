"""
benchmarks/ablations/summarize.py
=================================
The campaign-level artifacts: for each ablation, ``dist_to_needles`` and
``dup_fraction`` over time with confidence bands, plus the tables behind them.

What gets written, per ablation
-------------------------------
``dist_to_needles_over_time.png`` / ``dup_fraction_over_time.png``
    Mean across every (landscape, repeat) cell of an arm, with a bootstrap
    confidence band. One line per arm, one metric per figure — never two y-scales
    on one plot.
``curves_<metric>.csv``
    The plotted numbers: per arm and iteration, mean / CI / median / p25 / p75 and
    ``n_active`` (how many cells were still running at that iteration).
``summary.csv``
    Per-arm end-of-run metrics: mean and CI for every objective, plus needle and
    sample counts.
``paired.csv`` / ``summary.md``
    The paired comparison. Cells are matched on ``(landscape, repeat)``, which
    share a seed and therefore an initial design (see runner.py), so the
    baseline-vs-variant difference is measured within a cell rather than across the
    landscape draw. That is where nearly all of the statistical power in a campaign
    this size comes from.

Two decisions worth knowing about
---------------------------------
**Runs that end early are held, not dropped.** Cells are wall-clock budgeted, so
they end at different iteration counts. Dropping a finished cell from the average
would make the tail of the curve a different population from the head — arms that
finish early would silently leave the comparison, and the curve would bend for a
reason that has nothing to do with the optimiser. Instead a finished cell's last
value is carried forward (it really did end there, with that score), and the
iteration where the FIRST cell ended is marked on the figure so the held region is
visible rather than implied. ``n_active`` in the CSV is the exact count.

**The band is a bootstrap CI of the MEAN, not a spread.** ``dist_to_needles`` is
bounded and skewed, so a symmetric mean ± s.d. band would run outside the metric's
own range. The percentile bootstrap over cells makes no distributional assumption
and cannot do that. Median/p25/p75 are in the CSV for anyone who wants the spread
instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ._paths import ensure_paths

ensure_paths()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .arms import ABLATIONS, ARMS  # noqa: E402
from .campaign import RUNS, load_manifest  # noqa: E402
from .runner import load_cell_metrics  # noqa: E402

SUMMARY_DIR = "summary"

# Validated categorical slots 1-3 (blue / orange / aqua): all-pairs CVD ΔE 9.2,
# normal-vision ΔE 24.0 on a light surface. Capped at three on purpose — a fourth
# slot puts yellow next to orange and fails the normal-vision floor. An ablation
# with more than three arms should be split into figures, not given a new hue.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
MAX_SERIES = len(SERIES_COLORS)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

METRICS = (
    ("dist_to_needles", "Distance to true optima", "lower is better"),
    ("dup_fraction", "Duplicate sample fraction", "lower is better"),
)

DEFAULT_CI = 0.95
DEFAULT_N_BOOT = 2000


# ─── Collection ──────────────────────────────────────────────────────────────────

@dataclass
class Cell:
    arm: str
    landscape_index: int
    repeat: int
    path: str
    metrics: dict


def collect_cells(out_dir: str) -> list[Cell]:
    """Every finished cell under ``<out_dir>/runs``.

    Driven by what is on disk rather than by the queue, so a partially drained
    campaign summarises cleanly and a cell copied in from elsewhere is picked up.
    """
    root = os.path.join(out_dir, RUNS)
    cells: list[Cell] = []
    if not os.path.isdir(root):
        return cells
    for arm in sorted(os.listdir(root)):
        arm_dir = os.path.join(root, arm)
        if not os.path.isdir(arm_dir):
            continue
        for name in sorted(os.listdir(arm_dir)):
            cell_path = os.path.join(arm_dir, name)
            m = load_cell_metrics(cell_path)
            if m is None:
                continue
            cells.append(Cell(arm=m.get("arm", arm),
                              landscape_index=int(m["landscape_index"]),
                              repeat=int(m["repeat"]),
                              path=cell_path, metrics=m))
    return cells


def cells_frame(cells: list[Cell]) -> pd.DataFrame:
    """One row per finished cell — the flat table every statistic is computed from."""
    rows = []
    for c in cells:
        row = {"arm": c.arm, "landscape_index": c.landscape_index,
               "repeat": c.repeat, "path": c.path}
        row.update({k: v for k, v in c.metrics.items()
                    if k not in ("arm", "landscape_index", "repeat")})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["arm", "landscape_index", "repeat"]).reset_index(drop=True)
    return df


def load_curve(cell_path: str, column: str) -> np.ndarray | None:
    """One cell's per-iteration series for *column*, or None if unreadable."""
    path = os.path.join(cell_path, "metrics_over_time.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if column not in df.columns or df.empty:
        return None
    if "iteration" in df.columns:
        df = df.sort_values("iteration")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    # A leading NaN has no earlier value to inherit; back-fill it from the first
    # real reading so the curve starts on the grid rather than at index k>0.
    if np.isnan(values).all():
        return None
    values = pd.Series(values).ffill().bfill().to_numpy(dtype=float)
    return values


# ─── Alignment & statistics ──────────────────────────────────────────────────────

def align_curves(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged per-cell curves onto one iteration grid, holding the last value.

    Returns ``(matrix (n_cells, T), n_active (T,))`` where ``T`` is the longest
    cell's length and ``n_active[t]`` counts cells that were genuinely still
    running at iteration ``t+1``. See the module docstring for why holding beats
    dropping.
    """
    if not curves:
        return np.empty((0, 0)), np.empty(0, dtype=int)
    T = max(len(c) for c in curves)
    matrix = np.empty((len(curves), T), dtype=float)
    n_active = np.zeros(T, dtype=int)
    for i, c in enumerate(curves):
        n = len(c)
        matrix[i, :n] = c
        if n < T:
            matrix[i, n:] = c[-1]
        n_active[:n] += 1
    return matrix, n_active


def bootstrap_band(matrix: np.ndarray, *, ci: float = DEFAULT_CI,
                   n_boot: int = DEFAULT_N_BOOT,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile-bootstrap CI of the column mean, resampling CELLS.

    Resampling whole cells (rows), not individual iterations, is what makes the
    band a statement about "another draw of landscapes and repeats" rather than
    about noise within one run — the former is the uncertainty a reader cares
    about when deciding whether an arm is really better.

    With one or two cells the bootstrap has nothing to resample; the band collapses
    to the mean and the figure says so in its subtitle rather than drawing a
    misleadingly narrow ribbon.
    """
    n, T = matrix.shape
    mean = matrix.mean(axis=0) if n else np.empty(0)
    if n < 3:
        return mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = matrix[idx].mean(axis=1)          # (n_boot, T)
    alpha = (1.0 - ci) / 2.0
    lo = np.quantile(boot_means, alpha, axis=0)
    hi = np.quantile(boot_means, 1.0 - alpha, axis=0)
    return mean, lo, hi


def bootstrap_scalar(values: np.ndarray, *, ci: float = DEFAULT_CI,
                     n_boot: int = DEFAULT_N_BOOT,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap CI of a 1-D sample."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size < 3:
        m = float(v.mean())
        return m, m, m
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(v.mean()), float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def sign_flip_p(deltas: np.ndarray, *, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p-value for "the paired mean difference is zero".

    The exact test for a paired design under the only assumption it needs — that
    under the null, the sign of each pair's difference is arbitrary. No normality,
    no equal variances, and it is honest at the small pair counts these campaigns
    produce, where a t-test is not.
    """
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    # +1 in both terms: the observed arrangement is itself one of the permutations,
    # which keeps the p-value from ever being exactly 0.
    return float((np.sum(null >= observed - 1e-15) + 1) / (n_perm + 1))


# ─── Figures ─────────────────────────────────────────────────────────────────────

def _style_axes(ax, *, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)
    # Iterations are whole measured lines; matplotlib's default locator happily
    # labels 1.5 and 2.5, which name nothing that exists.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)


def plot_metric_over_time(
    path: str,
    series: list[dict],
    *,
    title: str,
    subtitle: str,
    ylabel: str,
) -> None:
    """One metric, one figure, one line + confidence band per arm.

    *series* entries carry ``label``, ``mean``, ``lo``, ``hi``, ``n_active``,
    ``n_cells``. Identity is never colour alone: every arm gets a legend entry AND
    a direct label at the end of its line, with the label text in ink and a dot in
    the series colour carrying the identity.
    """
    fig = Figure(figsize=(9.0, 5.2), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    _style_axes(ax, xlabel="LineBO iteration (measured lines)", ylabel=ylabel)

    if not series:
        ax.text(0.5, 0.5, "no finished cells", ha="center", va="center",
                color=INK_MUTED, transform=ax.transAxes)
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
        fig.clear()
        return

    def _first_end(s) -> int:
        """Iteration at which this arm's earliest-finishing cell stopped."""
        below = np.nonzero(s["n_active"] < s["n_cells"])[0]
        return int(below[0]) if below.size else len(s["n_active"])

    first_end = min(_first_end(s) for s in series)
    x_max = max(len(s["mean"]) for s in series)

    for i, s in enumerate(series):
        color = SERIES_COLORS[i % MAX_SERIES]
        x = np.arange(1, len(s["mean"]) + 1)
        ax.fill_between(x, s["lo"], s["hi"], color=color, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, s["mean"], color=color, lw=2.0, zorder=3,
                label=f"{s['label']}  (n={s['n_cells']})")

    # End-of-line dots. The labels themselves are placed after the layout is final,
    # because separating them correctly needs to know how tall the axes ended up.
    ends = sorted(((float(s["mean"][-1]), i, s) for i, s in enumerate(series)),
                  key=lambda t: t[0])
    for _, i, s in ends:
        ax.plot([len(s["mean"])], [s["mean"][-1]], marker="o", ms=6,
                color=SERIES_COLORS[i % MAX_SERIES], zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)

    if 0 < first_end < x_max:
        ax.axvline(first_end, color=INK_MUTED, ls=":", lw=1.0, zorder=1)
        ax.annotate("first cell ends —\nvalues held beyond here",
                    xy=(first_end, ax.get_ylim()[1]), xytext=(4, -6),
                    textcoords="offset points", color=INK_MUTED, fontsize=8,
                    va="top", ha="left", zorder=5)

    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=18)
    ax.annotate(subtitle, xy=(0, 1), xytext=(0, 6), xycoords="axes fraction",
                textcoords="offset points", color=INK_SECONDARY, fontsize=9,
                ha="left", va="bottom")
    leg = ax.legend(loc="best", fontsize=9, frameon=False)
    for text in leg.get_texts():
        text.set_color(INK_PRIMARY)
    # Room on the right for the direct labels — sized off the longest one, since a
    # fixed fraction of the x-range crowds the frame as soon as an arm name is long.
    longest = max(len(f"{s['label']}: 0.000") for s in series)
    ax.set_xlim(left=1, right=x_max * (1.0 + min(0.9, 0.022 * longest)))
    fig.tight_layout()

    # Direct labels last, once the axes box is final: text in ink, with the coloured
    # dot beside it carrying identity. Arms that converge to the same value (a
    # perfectly ordinary outcome for an ablation that changes nothing) would print
    # on top of each other, so labels are pushed apart in POINT space — the units
    # `offset points` is actually in. Doing this in data units, or before
    # tight_layout, gets the separation wrong by whatever the axes height turns out
    # to be.
    MIN_LABEL_GAP_PT = 13.0
    px_per_pt = fig.dpi / 72.0
    placed_pt = -np.inf
    for yv, i, s in ends:                      # ascending, so pushes go upward
        xv = len(s["mean"])
        y_pt = ax.transData.transform((xv, yv))[1] / px_per_pt
        target_pt = max(y_pt, placed_pt + MIN_LABEL_GAP_PT)
        placed_pt = target_pt
        ax.annotate(f"{s['label']}: {yv:.3f}", xy=(xv, yv),
                    xytext=(9, target_pt - y_pt), textcoords="offset points",
                    color=INK_PRIMARY, fontsize=9, va="center", zorder=5,
                    annotation_clip=False)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    fig.clear()


# ─── Per-ablation summary ────────────────────────────────────────────────────────

def _curve_table(arm: str, matrix: np.ndarray, n_active: np.ndarray,
                 mean: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "arm": arm,
        "iteration": np.arange(1, len(mean) + 1),
        "n_cells": matrix.shape[0],
        "n_active": n_active,
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "median": np.median(matrix, axis=0),
        "p25": np.percentile(matrix, 25, axis=0),
        "p75": np.percentile(matrix, 75, axis=0),
    })


def _paired_table(df: pd.DataFrame, baseline: str, variant: str,
                  metric: str) -> pd.DataFrame:
    """Baseline and variant side by side for every cell they both completed."""
    b = df[df["arm"] == baseline][["landscape_index", "repeat", metric]]
    v = df[df["arm"] == variant][["landscape_index", "repeat", metric]]
    merged = b.merge(v, on=["landscape_index", "repeat"],
                     suffixes=("_baseline", "_variant"))
    if merged.empty:
        return merged
    merged["delta"] = merged[f"{metric}_variant"] - merged[f"{metric}_baseline"]
    merged.insert(0, "metric", metric)
    return merged


def write_ablation_summary(
    out_dir: str,
    key: str,
    df: pd.DataFrame,
    cells: list[Cell],
    *,
    ci: float = DEFAULT_CI,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict:
    """Write one ablation's figures and tables. Returns a headline dict for the index."""
    ablation = ABLATIONS[key]
    dest = os.path.join(out_dir, SUMMARY_DIR, key)
    os.makedirs(dest, exist_ok=True)

    arms = [a for a in ablation.arms if (df["arm"] == a).any()]
    missing = [a for a in ablation.arms if a not in arms]
    if len(arms) > MAX_SERIES:
        raise ValueError(
            f"ablation {key} has {len(arms)} arms but the validated palette carries "
            f"{MAX_SERIES}; split it into separate figures rather than adding a hue")

    by_arm_cells = {a: [c for c in cells if c.arm == a] for a in arms}

    # ── curves ──
    headline: dict = {"key": key, "title": ablation.title,
                      "question": ablation.question, "missing_arms": missing,
                      "n_cells": {a: len(by_arm_cells[a]) for a in arms},
                      "metrics": {}}

    for metric, metric_label, direction in METRICS:
        series = []
        tables = []
        for a in arms:
            curves = [c for c in (load_curve(cell.path, metric)
                                  for cell in by_arm_cells[a]) if c is not None]
            if not curves:
                continue
            matrix, n_active = align_curves(curves)
            mean, lo, hi = bootstrap_band(matrix, ci=ci, n_boot=n_boot, seed=seed)
            series.append({"label": ARMS[a].label, "arm": a, "mean": mean,
                           "lo": lo, "hi": hi, "n_active": n_active,
                           "n_cells": matrix.shape[0]})
            tables.append(_curve_table(a, matrix, n_active, mean, lo, hi))

        if tables:
            pd.concat(tables, ignore_index=True).to_csv(
                os.path.join(dest, f"curves_{metric}.csv"), index=False)

        n_min = min((s["n_cells"] for s in series), default=0)
        band = (f"{int(ci * 100)}% bootstrap CI of the mean"
                if n_min >= 3 else "too few cells for a CI band — mean only")
        plot_metric_over_time(
            os.path.join(dest, f"{metric}_over_time.png"),
            series,
            title=f"{key} · {ablation.title}",
            subtitle=f"{metric_label} ({direction}) · {band}",
            ylabel=metric_label,
        )

        # ── paired comparison against the baseline ──
        baseline = ablation.arms[0]
        for variant in ablation.arms[1:]:
            if variant not in arms or baseline not in arms:
                continue
            paired = _paired_table(df, baseline, variant, metric)
            if paired.empty:
                continue
            deltas = paired["delta"].to_numpy(dtype=float)
            mean_d, lo_d, hi_d = bootstrap_scalar(deltas, ci=ci, n_boot=n_boot, seed=seed)
            headline["metrics"].setdefault(metric, []).append({
                "variant": variant,
                "n_pairs": int(len(paired)),
                "baseline_mean": float(paired[f"{metric}_baseline"].mean()),
                "variant_mean": float(paired[f"{metric}_variant"].mean()),
                "mean_delta": mean_d, "ci_lo": lo_d, "ci_hi": hi_d,
                "p_value": sign_flip_p(deltas, seed=seed),
                "variant_better_frac": float((deltas < 0).mean()),
            })
            paired.to_csv(os.path.join(dest, f"paired_{metric}_{variant}.csv"),
                          index=False)

    # ── per-arm end-of-run table ──
    scalar_cols = [c for c in ("dist_to_needles", "dup_fraction", "n_needles",
                               "n_iters", "n_points", "avg_time_per_iter_s",
                               "runtime_s", "n_restarts_actual")
                   if c in df.columns]
    rows = []
    for a in arms:
        sub = df[df["arm"] == a]
        row = {"arm": a, "label": ARMS[a].label, "n_cells": int(len(sub))}
        for col in scalar_cols:
            values = sub[col].to_numpy(dtype=float)
            m, lo, hi = bootstrap_scalar(values, ci=ci, n_boot=n_boot, seed=seed)
            row[f"{col}_mean"] = m
            row[f"{col}_ci_lo"] = lo
            row[f"{col}_ci_hi"] = hi
            # A column only some arms report (n_restarts_actual) is all-NaN for the
            # rest, and nanmedian warns on that instead of just returning NaN.
            row[f"{col}_median"] = (float(np.nanmedian(values))
                                    if values.size and not np.isnan(values).all()
                                    else float("nan"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(dest, "summary.csv"), index=False)

    _write_ablation_markdown(dest, key, ablation, headline, rows, ci)
    return headline


def _fmt(v: float, digits: int = 4) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _write_ablation_markdown(dest: str, key: str, ablation, headline: dict,
                             rows: list[dict], ci: float) -> None:
    lines = [f"# {key} — {ablation.title}", "", f"**{ablation.question}**", ""]
    if headline["missing_arms"]:
        lines += [f"> No finished cells for: {', '.join(headline['missing_arms'])}. "
                  "The figures below show only the arms that ran.", ""]

    lines += ["## Arms", ""]
    for a in ablation.arms:
        if a in ARMS:
            lines += [f"- **{ARMS[a].label}** (`{a}`) — {ARMS[a].description}"]
    lines += ["", "## End-of-run metrics", "",
              f"Mean over cells, with a {int(ci * 100)}% bootstrap CI.", "",
              "| arm | n | dist_to_needles | dup_fraction | needles | iterations | samples |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        def cell(col, digits=4):
            if f"{col}_mean" not in r:
                return "—"
            return (f"{_fmt(r[f'{col}_mean'], digits)} "
                    f"[{_fmt(r[f'{col}_ci_lo'], digits)}, {_fmt(r[f'{col}_ci_hi'], digits)}]")
        lines.append(
            f"| {r['label']} | {r['n_cells']} | {cell('dist_to_needles')} | "
            f"{cell('dup_fraction')} | {cell('n_needles', 1)} | "
            f"{cell('n_iters', 1)} | {cell('n_points', 0)} |")

    lines += ["", "## Paired comparison vs the baseline", "",
              "Cells are matched on (landscape, repeat); a matched pair shares its "
              "seed, so both arms started from the same initial design. "
              "`delta = variant − baseline`, and both metrics are minimised, so a "
              "**negative delta favours the variant**. The p-value is a two-sided "
              "sign-flip permutation test over the pairs.", "",
              "| metric | variant | pairs | baseline | variant | delta "
              f"[{int(ci * 100)}% CI] | p | variant better |",
              "|---|---|---|---|---|---|---|---|"]
    for metric, entries in headline["metrics"].items():
        for e in entries:
            lines.append(
                f"| {metric} | {ARMS[e['variant']].label} | {e['n_pairs']} | "
                f"{_fmt(e['baseline_mean'])} | {_fmt(e['variant_mean'])} | "
                f"{_fmt(e['mean_delta'])} [{_fmt(e['ci_lo'])}, {_fmt(e['ci_hi'])}] | "
                f"{_fmt(e['p_value'], 3)} | {e['variant_better_frac'] * 100:.0f}% |")

    lines += ["", "## Figures", "",
              "![dist_to_needles](dist_to_needles_over_time.png)", "",
              "![dup_fraction](dup_fraction_over_time.png)", ""]
    with open(os.path.join(dest, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─── Campaign index ──────────────────────────────────────────────────────────────

def _write_index(out_dir: str, manifest: dict, df: pd.DataFrame,
                 headlines: list[dict], ci: float) -> None:
    dest = os.path.join(out_dir, SUMMARY_DIR)
    lines = [f"# Ablation campaign — {os.path.basename(os.path.abspath(out_dir))}", "",
             f"- Landscape: `{manifest.get('landscape_ref')}` "
             f"{manifest.get('landscape_spec')}",
             f"- Dimension: {manifest.get('dim')}",
             f"- Budget: {manifest.get('time_limit_min')} min per cell",
             f"- Grid: {manifest.get('n_landscapes')} landscape(s) x "
             f"{manifest.get('n_repeats')} repeat(s) x {len(manifest.get('arms', {}))} arm(s)",
             f"- Finished cells: {len(df)} / {manifest.get('n_tasks')}",
             f"- Hyperparameters: `{manifest.get('hparams_source')}`", ""]

    lines += ["## Headline", "",
              "Both metrics are minimised; `delta = variant − baseline`, so a "
              "negative delta favours the variant.", "",
              "| ablation | question | metric | variant | delta "
              f"[{int(ci * 100)}% CI] | p |", "|---|---|---|---|---|---|"]
    for h in headlines:
        for metric, entries in h["metrics"].items():
            for e in entries:
                lines.append(
                    f"| [{h['key']}]({h['key']}/summary.md) | {h['question']} | "
                    f"{metric} | {ARMS[e['variant']].label} | "
                    f"{_fmt(e['mean_delta'])} [{_fmt(e['ci_lo'])}, {_fmt(e['ci_hi'])}] | "
                    f"{_fmt(e['p_value'], 3)} |")

    lines += ["", "## Per ablation", ""]
    for h in headlines:
        counts = ", ".join(f"{ARMS[a].label}: {n}" for a, n in h["n_cells"].items())
        lines += [f"### [{h['key']} — {h['title']}]({h['key']}/summary.md)", "",
                  f"{h['question']}", "", f"Cells: {counts}", "",
                  f"![dist]({h['key']}/dist_to_needles_over_time.png)", "",
                  f"![dup]({h['key']}/dup_fraction_over_time.png)", ""]

    with open(os.path.join(dest, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def summarize(out_dir: str, *, ci: float = DEFAULT_CI, n_boot: int = DEFAULT_N_BOOT,
              seed: int = 0, ablation_keys: list[str] | None = None) -> None:
    """Write every summary artifact for a campaign directory."""
    out_dir = os.path.abspath(out_dir)
    manifest = load_manifest(out_dir)
    keys = ablation_keys or [a["key"] for a in manifest.get("ablations", [])]

    cells = collect_cells(out_dir)
    df = cells_frame(cells)
    dest = os.path.join(out_dir, SUMMARY_DIR)
    os.makedirs(dest, exist_ok=True)
    df.to_csv(os.path.join(dest, "cells.csv"), index=False)

    if df.empty:
        print(f"  no finished cells under {os.path.join(out_dir, RUNS)} — "
              "nothing to summarise yet")
        return

    headlines = []
    for key in keys:
        if key not in ABLATIONS:
            print(f"  skipping unknown ablation {key!r}")
            continue
        arms_present = [a for a in ABLATIONS[key].arms if (df["arm"] == a).any()]
        if not arms_present:
            print(f"  {key}: no finished cells for any of its arms — skipped")
            continue
        headlines.append(write_ablation_summary(
            out_dir, key, df, cells, ci=ci, n_boot=n_boot, seed=seed))
        print(f"  {key}: wrote {os.path.join(dest, key)}")

    _write_index(out_dir, manifest, df, headlines, ci)
    with open(os.path.join(dest, "headlines.json"), "w") as f:
        json.dump(headlines, f, indent=2, default=float)
    print(f"  index -> {os.path.join(dest, 'index.md')}")
