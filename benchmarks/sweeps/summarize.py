"""
benchmarks/sweeps/summarize.py
==============================
Turn a drained (or partly drained) sweep into tables and figures.

The question the sweep asks is "which landscapes does ZoMBI-Hop hold up on", so the
output is organised as **one heatmap per (dimension, metric)** over the
``n_needles x basin_width`` plane, with the cell value averaged across draws. That
is the shape the grid actually has: dimension is not commensurable with the other
two axes (it changes the hyperparameters as well as the surface, see
:mod:`benchmarks.sweeps.hparams`), so it gets its own panel rather than being
averaged over.

Metrics
-------
The deliverable is the *set* of optima (METHODS §1), so the headline is how much of
that set was recovered and how much of what was reported is real. Those are two
numbers, not one, and they move in opposite directions:

    recall            fraction of the landscape's TRUE optima with a declared
                      needle inside ``MATCH_RADIUS``. Denominator is n.
    precision         fraction of the DECLARED needles that sit inside
                      ``MATCH_RADIUS`` of some true optimum. Denominator is the
                      declared count — this is ``metric_pct_matched_comp / 100``,
                      which earlier versions of this module reported under the name
                      ``recall``. It never had n as its denominator.
    dist_to_needles   the MOBO objective: mean distance from each true optimum to
                      the nearest declared needle, with a penalty for unmatched
                      ones. Lower is better. Sensitive to *how badly* a miss missed,
                      which neither recall nor precision is.
    n_needles         how many needles were declared, matched or not. Read next to
                      recall it separates "found few" from "declared few".
    median_nn_spacing median nearest-neighbour distance between the samples a cell
                      took, in composition L2. HIGHER is better. It replaces
                      ``dup_fraction``, which saturates at 0.999 on every cell of
                      this campaign and so separates nothing; see
                      ``_median_nn_spacing``. dup_fraction is still written to
                      ``cells.csv``.

Trajectories
------------
The heatmaps score each cell by its FINAL state, which cannot distinguish a cell
that converged early and sat still from one that was still improving when the
budget ran out. ``dist_to_needles`` is therefore also plotted against measured
lines, read from each cell's ``metrics_over_time.csv`` — once per swept axis with
the other two marginalised out, once faceted on dim x n, and once as all 320
individual trajectories with nothing averaged away.

Bootstrap confidence intervals resample *draws*, so a band answers "what if we drew
another set of optima placements", which is the question a reader has. With five
draws per cell the interval is wide by construction — it is there to stop a
one-draw fluke being read as a trend, not to make a significance claim.

Held, not dropped
-----------------
Cells that failed or have not run yet are left out of the mean and counted in
``n_draws`` in the CSV, so a heatmap tile computed from two draws is visibly
different from one computed from five. Nothing is imputed.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ._paths import ensure_paths

ensure_paths()

from .campaign import CELL_FILE, cell_dir, load_manifest, read_tasks  # noqa: E402

DEFAULT_CI = 0.95
DEFAULT_N_BOOT = 2000

#: Metric key -> (column label, is-lower-better).
METRICS: dict[str, tuple[str, bool]] = {
    "recall": ("recall (true optima found / n)", False),
    "precision": ("precision (declared needles that are real)", False),
    "dist_to_needles": ("dist_to_needles", True),
    "n_needles": ("needles declared", False),
    "median_nn_spacing": ("median NN spacing (x10^-3)", False),
}

#: Display scale applied at PLOT time only. Every stored value — cells.csv,
#: grid.csv — stays in the metric's own units; this exists so the heatmap
#: annotations are not four leading zeros. Matches
#: ``optimize/summary_table.NN_SPACING_SCALE``.
METRIC_SCALE: dict[str, float] = {"median_nn_spacing": 1e3}


# ─── Collection ──────────────────────────────────────────────────────────────────

def _match_stats(trial_dir: str, n_true: int) -> tuple[float | None, float | None]:
    """``(recall, precision)`` of one cell's declared needles, or ``(None, None)``.

    Both are computed at the same ``eval_metrics.match_radius_comp`` (0.05 in
    composition L2, dimension-independent) so they are two readings of one pairing,
    but they divide by opposite things and answer opposite questions:

        recall     |{true optima with SOME needle within r}| / n_true
                   "how much of the landscape did it find". Falls when the
                   optimiser declares too few needles.
        precision  |{needles within r of SOME optimum}| / n_declared
                   "how much of what it declared is real". Falls when it declares
                   spurious needles.

    Neither alone is a verdict — an optimiser that declares one perfect needle on a
    50-optimum landscape scores precision 1.0 and recall 0.02 — which is exactly
    why both are carried.

    HISTORY, because it changes how earlier figures read: this function used to
    return ``metric_pct_matched_comp / 100`` under the name ``recall``. That
    function divides by ``len(discovered)``, so it was *precision* all along, and
    the docstring's claim that "the placement guarantees the denominator is exactly
    n" was never true of the code. The value is still reported, under its right
    name; ``recall`` is now computed here rather than borrowed.
    """
    import pandas as pd
    from eval_metrics import match_radius_comp

    needles_csv = os.path.join(trial_dir, "needles.csv")
    ens_cfg = os.path.join(trial_dir, "ensemble_config.json")
    if not (os.path.isfile(needles_csv) and os.path.isfile(ens_cfg)):
        return (None, None)
    with open(ens_cfg) as f:
        cfg = json.load(f)
    true_optima = [np.asarray(c, dtype=float) for c in cfg.get("pinned_optima", [])]
    if len(true_optima) != n_true:
        return (None, None)
    dim = len(true_optima[0])
    df = pd.read_csv(needles_csv)
    # 3d runs write the composition columns by name (FA/MA/Br); every other
    # dimension writes x0..x{d-1}.
    coord_cols = [c for c in df.columns if len(c) > 1 and c[0] == "x" and c[1:].isdigit()] \
        or [c for c in ("FA", "MA", "Br") if c in df.columns]
    coord_cols.sort(key=lambda c: int(c[1:]) if c[1:].isdigit() else 0)
    disc = (df[coord_cols].to_numpy(dtype=float) if len(df) and coord_cols
            else np.empty((0, dim)))
    T = np.asarray(true_optima, dtype=float)
    if len(disc) == 0:
        # Declared nothing: found none of the optima, and has no declarations for a
        # precision to be about. 0.0 rather than None — this is a measurement.
        return (0.0, 0.0)
    r = float(match_radius_comp(dim))
    C = np.linalg.norm(disc[:, None, :] - T[None, :, :], axis=2)
    recall = float((C.min(axis=0) <= r).mean())      # over true optima
    precision = float((C.min(axis=1) <= r).mean())   # over declared needles
    return (recall, precision)


def _median_nn_spacing(trial_dir: str) -> float | None:
    """Median nearest-neighbour distance between a cell's samples, composition L2.

    REPLACES ``dup_fraction`` as the sweep's read on whether a landscape drives the
    optimiser into re-measuring one spot. Dup fraction counts samples whose nearest
    neighbour falls inside ``NOISE_LEVEL/2``, which on this campaign saturates: every
    cell lands at 0.999x, because LineBO measures 24 points along a line and
    consecutive points on one line are always closer than that radius. A metric that
    reads 0.999 on all 320 cells separates nothing.

    Spacing has no radius, no noise level and no zoom scaling in it, so it measures
    the same thing without a threshold to saturate against. Same definition as
    ``optimize/summary_table.median_nn_spacing``, so a sweep cell and a showdown run
    are on one axis. **Higher is better** — wide spacing means the run spread out.

    Compare only between runs of similar length: spacing scales as ``N^(-1/d)``. Every
    cell here spent exactly 3000 points, so within this campaign that is moot.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    path = os.path.join(trial_dir, "points.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError):
        return None
    cols = [c for c in df.columns if len(c) > 1 and c[0] == "x" and c[1:].isdigit()] \
        or [c for c in ("FA", "MA", "Br") if c in df.columns]
    if not cols or len(df) < 2:
        return None
    cols.sort(key=lambda c: int(c[1:]) if c[1:].isdigit() else 0)
    X = df[cols].to_numpy(dtype=float)
    nn, _ = cKDTree(X).query(X, k=2)   # k=2: self (0) + nearest other
    return float(np.median(nn[:, 1]))


def collect(out_dir: str) -> list[dict]:
    """One row per finished cell, with the metrics the summary plots."""
    rows = []
    for task in read_tasks(out_dir):
        target = cell_dir(out_dir, task["name"], task["draw"])
        path = os.path.join(target, CELL_FILE)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        m = rec.get("metrics", {})
        land = rec.get("landscape", {})
        bud = rec.get("budget", {})
        recall, precision = _match_stats(target, rec["n_needles"])
        rows.append({
            "cell": rec["cell"], "draw": rec["draw"], "dim": rec["dim"],
            "n_needles": rec["n_needles"], "basin_width": rec["basin_width"],
            "recall": recall,
            "precision": precision,
            "median_nn_spacing": _median_nn_spacing(target),
            "dist_to_needles": m.get("dist_to_needles"),
            "n_needles_found": m.get("n_needles"),
            "dup_fraction": m.get("dup_fraction"),
            "n_points": m.get("n_points"),
            "n_iters": m.get("n_iters"),
            "budget_hit": bud.get("budget_hit"),
            "points_budget": bud.get("points_budget"),
            "separation_achieved": land.get("separation_achieved"),
            "separation_target": land.get("separation_target"),
            "prominence_target_met": land.get("prominence_target_met"),
            "n_prominence_resolved": land.get("n_prominence_resolved"),
            "basin_plain_radius": round(
                float(np.sqrt(rec["dim"]) * np.log(2.0) / rec["basin_width"]), 6),
            "hparams_source": rec.get("hparams_source"),
            "runtime_s": m.get("runtime_s"),
            "wall_s": rec.get("wall_s"),
        })
    # ``n_needles`` names the landscape's TRUE count on the row; the metric column
    # is the declared count, so the two are never confused in the CSV.
    for r in rows:
        r["n_needles_true"] = r.pop("n_needles")
        r["n_needles"] = r.pop("n_needles_found")
    return rows


# ─── Statistics ──────────────────────────────────────────────────────────────────

def _boot_ci(values: np.ndarray, ci: float, n_boot: int, rng) -> tuple[float, float]:
    """Percentile bootstrap CI of the MEAN, resampling draws.

    A symmetric mean +/- sd band would run outside the range of a bounded metric
    like ``recall``; resampling the draws keeps the interval inside whatever range
    the metric actually has.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    means = values[idx].mean(axis=1)
    lo = (1.0 - ci) / 2.0
    return (float(np.quantile(means, lo)), float(np.quantile(means, 1.0 - lo)))


def aggregate(rows: list[dict], *, ci: float = DEFAULT_CI,
              n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> list[dict]:
    """Collapse draws into one record per grid configuration, per metric."""
    rng = np.random.default_rng(seed)
    by_cell: dict[tuple, list[dict]] = {}
    for r in rows:
        by_cell.setdefault((r["dim"], r["n_needles_true"], r["basin_width"]), []).append(r)

    out = []
    for (dim, n, b), group in sorted(by_cell.items()):
        rec = {"dim": dim, "n_needles_true": n, "basin_width": b,
               "n_draws": len(group),
               "budget_hit_frac": round(float(np.mean(
                   [bool(g["budget_hit"]) for g in group])), 4)}
        for key in METRICS:
            vals = np.asarray([g[key] for g in group if g.get(key) is not None],
                              dtype=float)
            if not len(vals):
                rec[f"{key}_mean"] = rec[f"{key}_lo"] = rec[f"{key}_hi"] = None
                rec[f"{key}_n"] = 0
                continue
            lo, hi = _boot_ci(vals, ci, n_boot, rng)
            rec[f"{key}_mean"] = round(float(vals.mean()), 6)
            rec[f"{key}_median"] = round(float(np.median(vals)), 6)
            rec[f"{key}_lo"] = None if np.isnan(lo) else round(lo, 6)
            rec[f"{key}_hi"] = None if np.isnan(hi) else round(hi, 6)
            rec[f"{key}_n"] = int(len(vals))
        out.append(rec)
    return out


# ─── Figures ─────────────────────────────────────────────────────────────────────

def _heatmap_grid(agg: list[dict], metric: str, manifest: dict,
                  path: str) -> None:
    """One panel per dimension: ``n_needles`` (rows) x ``basin_width`` (columns).

    A shared colour scale across the panels is the point — the comparison a reader
    makes is "does this get worse as the dimension rises", and per-panel scales
    would flatten exactly that. Tiles with no data are left blank rather than drawn
    at zero, which would read as a *result*.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = sorted({r["dim"] for r in agg})
    counts = sorted({r["n_needles_true"] for r in agg})
    widths = sorted({r["basin_width"] for r in agg})
    if not (dims and counts and widths):
        return
    label, lower_better = METRICS[metric]
    scale = METRIC_SCALE.get(metric, 1.0)

    grids = {}
    for dim in dims:
        M = np.full((len(counts), len(widths)), np.nan)
        for r in agg:
            if r["dim"] != dim or r.get(f"{metric}_mean") is None:
                continue
            M[counts.index(r["n_needles_true"]), widths.index(r["basin_width"])] = \
                r[f"{metric}_mean"] * scale
        grids[dim] = M

    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()]) \
        if any(np.isfinite(g).any() for g in grids.values()) else np.array([0.0, 1.0])
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    # Viridis reads low-to-high; reverse it for a minimised metric so "good" is
    # bright in every panel of every figure.
    cmap = plt.get_cmap("viridis_r" if lower_better else "viridis").copy()
    cmap.set_bad("#e8e8e8")

    fig, axes = plt.subplots(1, len(dims), figsize=(3.4 * len(dims) + 1.6, 3.9),
                             squeeze=False)
    im = None
    for ax, dim in zip(axes[0], dims):
        im = ax.imshow(np.ma.masked_invalid(grids[dim]), cmap=cmap, vmin=vmin,
                       vmax=vmax, origin="lower", aspect="auto")
        ax.set_xticks(range(len(widths)), [f"{w:g}" for w in widths])
        ax.set_yticks(range(len(counts)), [str(c) for c in counts])
        ax.set_xlabel("basin sharpness $b$")
        src = manifest.get("hparams", {}).get(str(dim), {})
        tag = " *" if src.get("is_stand_in") else ""
        ax.set_title(f"dim {dim}{tag}", fontsize=11)
        for i in range(len(counts)):
            for j in range(len(widths)):
                v = grids[dim][i, j]
                if np.isfinite(v):
                    # Text contrast against the tile, not against the figure.
                    rel = (v - vmin) / (vmax - vmin or 1.0)
                    if lower_better:
                        rel = 1.0 - rel
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if rel < 0.55 else "black")
    axes[0][0].set_ylabel("number of needles $n$")
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02, label=label)
    arrow = "lower is better" if lower_better else "higher is better"
    n_draws = manifest.get("n_draws", "?")
    fig.suptitle(f"{label} — mean of {n_draws} draw(s), {arrow}"
                 + ("   (* hyperparameters are a stand-in for this dim)"
                    if any(manifest.get("hparams", {}).get(str(d), {}).get("is_stand_in")
                           for d in dims) else ""),
                 fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _recall_vs_axis(agg: list[dict], manifest: dict, path: str) -> None:
    """Recall AND precision against each swept axis, marginalised over the other two.

    The heatmaps show the interaction; this shows the main effects, which is what a
    one-line answer to "what is ZoMBI-Hop robust to" needs. Error bars are the
    bootstrap CI of the mean over every cell in that slice.

    The two series are drawn together because they are the pair that has to be read
    together: on the ``n`` axis they cross, and a figure showing only one of them
    invites exactly the wrong conclusion — precision rises with n purely because the
    declared count stays flat while the number of things there are to hit goes up.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_spec = [("n_needles_true", "number of needles $n$"),
                 ("basin_width", "basin sharpness $b$"),
                 ("dim", "dimension $d$")]
    # Both series on one axis on purpose: they are the same kind of quantity (a
    # fraction in [0, 1] at one match radius), so they share a scale honestly, and
    # the whole point is that they move in OPPOSITE directions on the n axis —
    # which is invisible in two separate figures.
    series = [("recall", "recall (of true optima)", "#2b6cb0", "o", "-"),
              ("precision", "precision (of declarations)", "#d97706", "s", "--")]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), squeeze=False)
    for ax, (key, label) in zip(axs[0], axes_spec):
        levels = sorted({r[key] for r in agg})
        for metric, mlabel, colour, marker, ls in series:
            xs, ys, los, his = [], [], [], []
            for lv in levels:
                vals = [r[f"{metric}_mean"] for r in agg
                        if r[key] == lv and r.get(f"{metric}_mean") is not None]
                if not vals:
                    continue
                v = np.asarray(vals, dtype=float)
                lo, hi = _boot_ci(v, DEFAULT_CI, DEFAULT_N_BOOT,
                                  np.random.default_rng(0))
                xs.append(lv)
                ys.append(v.mean())
                los.append(v.mean() - (lo if np.isfinite(lo) else v.mean()))
                his.append((hi if np.isfinite(hi) else v.mean()) - v.mean())
            if not xs:
                continue
            ax.errorbar(range(len(xs)), ys, yerr=[los, his], marker=marker,
                        capsize=3, color=colour, ls=ls, lw=1.8, ms=5,
                        label=mlabel)
            ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_xlabel(label)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axs[0][0].set_ylabel("fraction")
    handles, labels = axs[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
                   fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Recall and precision by axis, marginalised over the other two "
                 f"({manifest.get('n_draws', '?')} draw(s) per configuration)",
                 fontsize=11, y=1.07)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Trajectories ────────────────────────────────────────────────────────────────
#
# The heatmaps score a cell by its FINAL dist_to_needles, which cannot tell a cell
# that converged at line 20 and sat still apart from one still improving when the
# budget ran out. Every cell writes ``metrics_over_time.csv`` — one row per measured
# line, carrying the metrics as they stood after that line — so the trajectory is
# already on disk and only needs to be read and pooled.
#
# All three figures share one x-axis, MEASURED COMPOSITIONS (line index x
# ``points_per_line``), because that is the quantity the budget is denominated in and
# the quantity an experimentalist pays for. It is not wall-clock: a dim-10 line costs
# more seconds than a dim-3 line, and plotting against seconds would show the model
# fit getting slower rather than the search doing better or worse.

#: Levels of every swept axis are ORDERED (3<4<6<10, 2<10<30<50, 2.2<6<10<15), so
#: the colour job is sequential — one hue ramp, light to dark — not categorical.
#: Viridis is the same family the heatmaps use and is perceptually uniform and
#: CVD-safe, so a reader carries one colour intuition across the whole summary.
TRAJ_CMAP = "viridis"

#: Trajectories are truncated to the shortest curve in a group before averaging.
#: The alternative — forward-filling short curves to the longest — invents a flat
#: tail that reads as "converged and held" when it actually means "this cell stopped
#: here", which is the one confusion these figures exist to remove.
def _level_colors(n: int):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(TRAJ_CMAP)
    # Stop short of 0.95: the very top of viridis is a pale yellow that disappears
    # against white at 1.5px line width.
    return [cmap(v) for v in np.linspace(0.08, 0.88, max(n, 1))]


def collect_curves(out_dir: str, manifest: dict) -> list[dict]:
    """One record per finished cell carrying its ``dist_to_needles`` trajectory.

    Kept out of :func:`collect` and out of ``cells.csv`` on purpose: a cell's row is
    one line of a table, but its trajectory is ~123 numbers, and inlining those would
    make the CSV unreadable for the sake of a figure. The two are joined on
    ``(dim, n_needles_true, basin_width, draw)``.

    Cells whose ``metrics_over_time.csv`` is missing or unreadable are skipped rather
    than zero-filled — same rule as the heatmaps.
    """
    import pandas as pd

    per_line = float(manifest.get("points_per_line", 24) or 24)
    curves = []
    for task in read_tasks(out_dir):
        target = cell_dir(out_dir, task["name"], task["draw"])
        rec_path = os.path.join(target, CELL_FILE)
        mot = os.path.join(target, "metrics_over_time.csv")
        if not (os.path.isfile(rec_path) and os.path.isfile(mot)):
            continue
        try:
            with open(rec_path) as f:
                rec = json.load(f)
            df = pd.read_csv(mot)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if "dist_to_needles" not in df.columns or not len(df):
            continue
        y = df["dist_to_needles"].to_numpy(dtype=float)
        x = (df["iteration"].to_numpy(dtype=float) if "iteration" in df.columns
             else np.arange(1, len(y) + 1, dtype=float)) * per_line
        keep = np.isfinite(y)
        if keep.sum() < 2:
            continue
        curves.append({"cell": rec["cell"], "draw": rec["draw"], "dim": rec["dim"],
                       "n_needles_true": rec["n_needles"],
                       "basin_width": rec["basin_width"],
                       "x": x[keep], "y": y[keep]})
    return curves


def _pool(group: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(x, mean, lo, hi)`` over a group of trajectories, truncated to the shortest.

    ``lo``/``hi`` are the interquartile range, not a bootstrap CI: these bands overlay
    four at a time and a percentile of the raw curves is the honest thing to stack —
    it says "half the cells in this slice lived in here", which does not need a
    resampling assumption to read.
    """
    k = min(len(g["y"]) for g in group)
    Y = np.vstack([g["y"][:k] for g in group])
    x = group[0]["x"][:k]
    return x, Y.mean(axis=0), np.quantile(Y, 0.25, axis=0), np.quantile(Y, 0.75, axis=0)


def _traj_by_axis(curves: list[dict], manifest: dict, path: str) -> None:
    """dist_to_needles against budget spent, once per swept axis.

    Three panels, one per axis; within a panel one curve per level of that axis with
    the other two axes marginalised out. This is the main-effects view — the answer to
    "does raising d/n/b change the SHAPE of the search, or only where it ends up".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [("dim", "dimension $d$"),
             ("n_needles_true", "needles $n$"),
             ("basin_width", "sharpness $b$")]
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.8), sharey=True, squeeze=False)
    for ax, (key, label) in zip(axs[0], specs):
        levels = sorted({c[key] for c in curves})
        for colour, lv in zip(_level_colors(len(levels)), levels):
            group = [c for c in curves if c[key] == lv]
            if not group:
                continue
            x, mu, lo, hi = _pool(group)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.14, linewidth=0)
            ax.plot(x, mu, color=colour, lw=2.0, label=f"{lv:g}")
        ax.set_xlabel("measured compositions")
        ax.grid(alpha=0.22, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # A legend on every panel, because each panel's levels mean something
        # different — one shared legend could only label one of the three axes.
        ax.legend(title=label, fontsize=8, title_fontsize=8.5, frameon=False,
                  loc="lower right", ncol=2)
    axs[0][0].set_ylabel("dist_to_needles  (lower is better)")
    fig.suptitle("dist_to_needles over the budget, by axis — mean over every cell in "
                 "the slice, band is the IQR", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _traj_grid(curves: list[dict], manifest: dict, path: str) -> None:
    """A facet per (dimension, needle count), with one curve per basin sharpness.

    The by-axis figure marginalises, which hides interactions; this shows all three
    axes at once. Rows are dimension and columns are needle count because those are
    the two that carry the effect, leaving sharpness — the weakest axis — as the
    within-panel comparison where four overlaid curves stay legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = sorted({c["dim"] for c in curves})
    counts = sorted({c["n_needles_true"] for c in curves})
    widths = sorted({c["basin_width"] for c in curves})
    if not (dims and counts and widths):
        return
    colours = _level_colors(len(widths))

    fig, axs = plt.subplots(len(dims), len(counts), figsize=(3.0 * len(counts),
                                                             2.5 * len(dims)),
                            sharex=True, sharey=True, squeeze=False)
    for i, dim in enumerate(dims):
        for j, n in enumerate(counts):
            ax = axs[i][j]
            for colour, b in zip(colours, widths):
                group = [c for c in curves if c["dim"] == dim
                         and c["n_needles_true"] == n and c["basin_width"] == b]
                if not group:
                    continue
                x, mu, _, _ = _pool(group)
                ax.plot(x, mu, color=colour, lw=1.6, label=f"{b:g}")
            ax.grid(alpha=0.2, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(f"$n$ = {n}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"dim {dim}", fontsize=10)
            if i == len(dims) - 1:
                ax.set_xlabel("measured compositions", fontsize=9)
    # One legend for the whole figure: sharpness means the same thing in all 16
    # panels, so repeating it 16 times would only cost space.
    handles, labels = axs[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="sharpness $b$", frameon=False,
                   loc="upper center", ncol=len(widths), fontsize=9,
                   title_fontsize=9.5, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("dist_to_needles over the budget — mean of "
                 f"{manifest.get('n_draws', '?')} draw(s) per curve, "
                 "shared axes (lower is better)", fontsize=11, y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _traj_all(curves: list[dict], manifest: dict, path: str) -> None:
    """Every individual trajectory, nothing averaged, coloured by dimension.

    Averages can only show where the middle of a slice went; this shows the spread
    the averages are made of, which is where a bimodal slice or a single divergent
    cell would live. Thin and translucent so density reads as darkness, with the
    per-dimension means drawn over the top so the figure is still navigable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dims = sorted({c["dim"] for c in curves})
    colours = dict(zip(dims, _level_colors(len(dims))))

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for c in curves:
        ax.plot(c["x"], c["y"], color=colours[c["dim"]], lw=0.5, alpha=0.16,
                zorder=1, solid_capstyle="round")
    for dim in dims:
        group = [c for c in curves if c["dim"] == dim]
        x, mu, _, _ = _pool(group)
        # A 2px white underlay keeps each mean readable where it crosses the others.
        ax.plot(x, mu, color="white", lw=4.0, zorder=2, solid_capstyle="round")
        ax.plot(x, mu, color=colours[dim], lw=2.4, zorder=3, solid_capstyle="round",
                label=f"dim {dim}  (n={len(group)})")
    ax.set_xlabel("measured compositions")
    ax.set_ylabel("dist_to_needles  (lower is better)")
    ax.grid(alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(title="mean per dimension", frameon=False, fontsize=9,
              title_fontsize=9.5, loc="lower right")
    ax.set_title(f"All {len(curves)} cell trajectories overlaid — one faint line per "
                 "cell, bold line the per-dimension mean", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Entry point ─────────────────────────────────────────────────────────────────

def summarize(out_dir: str, *, ci: float = DEFAULT_CI, n_boot: int = DEFAULT_N_BOOT,
              seed: int = 0) -> None:
    """Write ``summary/`` for a finished or partial campaign."""
    import pandas as pd

    out_dir = os.path.abspath(out_dir)
    manifest = load_manifest(out_dir)
    sdir = os.path.join(out_dir, "summary")
    os.makedirs(sdir, exist_ok=True)

    rows = collect(out_dir)
    if not rows:
        print(f"  no finished cells in {out_dir} yet — nothing to summarise")
        return
    agg = aggregate(rows, ci=ci, n_boot=n_boot, seed=seed)

    pd.DataFrame(rows).to_csv(os.path.join(sdir, "cells.csv"), index=False)
    pd.DataFrame(agg).to_csv(os.path.join(sdir, "grid.csv"), index=False)

    for metric in METRICS:
        _heatmap_grid(agg, metric, manifest,
                      os.path.join(sdir, f"{metric}_heatmap.png"))
    _recall_vs_axis(agg, manifest, os.path.join(sdir, "recall_by_axis.png"))

    curves = collect_curves(out_dir, manifest)
    if curves:
        _traj_by_axis(curves, manifest,
                      os.path.join(sdir, "dist_over_time_by_axis.png"))
        _traj_grid(curves, manifest,
                   os.path.join(sdir, "dist_over_time_grid.png"))
        _traj_all(curves, manifest,
                  os.path.join(sdir, "dist_over_time_all.png"))

    n_expected = manifest["n_tasks"]
    budget_short = [r for r in rows if r.get("budget_hit") is False]
    lines = [
        "# Landscape sweep — how robust is ZoMBI-Hop to the surface?",
        "",
        f"{len(rows)} of {n_expected} cell(s) finished "
        f"({manifest['n_configurations']} configurations x "
        f"{manifest['n_draws']} draw(s)).",
        "",
        f"Budget: **{manifest['n_lines']} lines / {manifest['points_budget']} "
        f"measured compositions** per cell, identical at every dimension.",
        "",
        "Landscape: bumps-only `Ensemble` — exactly *n* negated-Ackley optima of "
        "sharpness *b* on a flat plain at 0.75, every optimum peaking at 1.0, no "
        "roughness / ridges / plateaus / distractors / edge bias. Optima are placed "
        "at least `max(sigma_x, s_prom(b, d))` apart so all *n* are resolvable in "
        "both senses METHODS section 1 requires; see "
        "`benchmarks/sweeps/needles.py`.",
        "",
        "## Hyperparameters",
        "",
        "| dim | config | provenance |",
        "|---|---|---|",
    ]
    for dim in sorted(int(d) for d in manifest["hparams"]):
        h = manifest["hparams"][str(dim)]
        star = " **(stand-in)**" if h["is_stand_in"] else ""
        lines.append(f"| {dim} | `{h['path']}`{star} | {h['provenance']} |")
    lines += [
        "",
        "A stand-in means no configuration tuned at that dimension exists in the "
        "repo, so a neighbour's is used; dim-to-dim differences there include the "
        "hyperparameters, not only the landscape.",
        "",
        "## Metrics",
        "",
        "| metric | denominator | direction |",
        "|---|---|---|",
        "| `recall` | the landscape's *n* true optima | higher |",
        "| `precision` | the needles the run declared | higher |",
        "| `dist_to_needles` | — (mean distance + unmatched penalty) | lower |",
        "| `n_needles` | — (count declared) | higher is not better |",
        "| `median_nn_spacing` | — (composition L2, shown x10^-3) | higher |",
        "",
        "`recall` and `precision` share one match radius "
        "(`eval_metrics.MATCH_RADIUS` = 0.05, composition L2, dimension-independent) "
        "and differ only in what they divide by, so they must be read as a pair: an "
        "optimiser that declares four needles on a 50-optimum landscape and gets all "
        "four right scores precision 1.00 and recall 0.08.",
        "",
        "**`precision` is the quantity earlier versions of this summary labelled "
        "`recall`.** It came from `eval_metrics.metric_pct_matched_comp`, which "
        "divides by `len(discovered)`; the denominator was never *n*, so every "
        "figure written before this change reads as precision regardless of its "
        "axis label. `recall` is now computed here rather than borrowed.",
        "",
        "`median_nn_spacing` replaces `dup_fraction`, which saturates at 0.999 on "
        "every cell of this campaign — LineBO measures 24 points along one line, and "
        "consecutive points on a line are always closer than the `NOISE_LEVEL/2` "
        "radius dup_fraction counts against, so it separates nothing here. Spacing "
        "has no radius to saturate against and is the same definition "
        "`optimize/summary_table.py` uses, so a sweep cell and a showdown run sit on "
        "one axis. `dup_fraction` is still written to `cells.csv`.",
        "",
        "## Figures",
        "",
        "![recall](recall_heatmap.png)",
        "![precision](precision_heatmap.png)",
        "![recall and precision by axis](recall_by_axis.png)",
        "![dist_to_needles](dist_to_needles_heatmap.png)",
        "![needles declared](n_needles_heatmap.png)",
        "![median NN spacing](median_nn_spacing_heatmap.png)",
        "",
        "### Trajectories",
        "",
        "Every heatmap above scores a cell by its FINAL state, which cannot separate "
        "a cell that converged early and held from one still improving when the "
        "budget ran out. These read `dist_to_needles` off each cell's "
        "`metrics_over_time.csv` and plot it against measured compositions — the "
        "unit the budget is denominated in, not wall-clock, so a dim-10 line and a "
        "dim-3 line cost the same on the x-axis.",
        "",
        "![dist over time by axis](dist_over_time_by_axis.png)",
        "![dist over time faceted](dist_over_time_grid.png)",
        "![all trajectories](dist_over_time_all.png)",
        "",
        "The first marginalises each axis over the other two (main effects, IQR "
        "band); the second facets dim x n with sharpness overlaid inside each panel, "
        "so interactions stay visible; the third overlays every individual cell with "
        "nothing averaged away, which is where a bimodal slice or a lone divergent "
        "cell would show up.",
        "",
        "`cells.csv` has every finished cell; `grid.csv` has the per-configuration "
        f"means with {int(ci * 100)}% bootstrap intervals over draws.",
        "",
    ]
    if budget_short:
        lines += [
            "## Cells that did not spend their budget",
            "",
            f"{len(budget_short)} cell(s) stopped before the line budget — the "
            "wall-clock ceiling or the optimiser's own termination got there first. "
            "Their metrics are not comparable to a full-budget cell on equal terms.",
            "",
            "| cell | draw | points | budget |",
            "|---|---|---|---|",
        ] + [f"| {r['cell']} | {r['draw']} | {r['n_points']} | "
             f"{r['points_budget']} |" for r in budget_short[:25]] + [""]

    with open(os.path.join(sdir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  summary -> {sdir}  ({len(rows)} cell(s), "
          f"{len(agg)} configuration(s))")
