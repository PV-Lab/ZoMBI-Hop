"""
benchmarks/sweeps/summarize.py
==============================
Turn a drained (or partly drained) method x landscape sweep into tables and figures.

Running it
----------
Not a script on its own (relative imports): run it through the sweep CLI from the
repo root, pointing ``--out`` at the campaign directory (the one holding
``manifest.json`` and ``tasks.tsv``)::

    cd /path/to/ZoMBI-Hop
    uv run python -m benchmarks.sweeps summarize \
        --out benchmarks/sweeps/runs/full_20260926

Options: ``--ci`` (default 0.95), ``--n-boot`` (bootstrap resamples, default 2000),
``--seed`` (bootstrap seed, default 0). Output goes to ``<out>/summary/`` and is
overwritten on every run, so it is safe to re-run while the sweep is still draining.
From Python: ``summarize("benchmarks/sweeps/runs/full_20260926")``.

The question is "which method gets near every needle, and on which landscapes", so
the summary is organised around the METHOD:

    summary/
    ├── index.md                     headline table, paired comparisons, every figure
    ├── cells.csv                    one row per finished cell
    ├── grid.csv                     per (method, dim, n, b): means + bootstrap CIs
    ├── methods.csv                  per (method, dim) and overall: means + CIs
    ├── paired.csv                   each method vs the reference, paired by landscape
    ├── method_by_dim.png            the headline: each metric vs dim, one line per method
    ├── <metric>_heatmap.png         rows = method, columns = dim, tile = n x b
    ├── greedy_over_time.png         greedy_dist vs measured points, panel per dim
    ├── greedy_over_time_all.png     every cell's trajectory, panel per method
    ├── greedy_over_time_by_axis.png row per method, column per swept axis (d, n, b)
    ├── greedy_over_time_grid_b<b>.png  one per sharpness b: row per dim, column
    │                                per n, one line per method
    ├── greedy_over_time_grid_b<b>_trial1.png  the same, draw 1 only (no averaging)
    ├── needles_found_grid_b<b>.png  the same layout for the number of optima with a
    │                                sample within the found radius (see Metrics)
    ├── needles_found_grid_b<b>_trial1.png  the same, draw 1 only
    ├── best_f_over_time_grid.png    the same layout for the running best noiseless f
    └── sampling_<cell>.png          2-D landscapes only: the landscape (draw 1), one
                                     panel per method with its samples on it; under
                                     it running best f and greedy_dist for two single
                                     draws and the mean over draws (bootstrap CI)

Metrics
-------
    greedy_dist          headline. For each true optimum the distance to the nearest
                         point the method MEASURED, averaged over the optima
                         (``eval_metrics.metric_greedy_dist``). Lower is better; no
                         penalty, no cap. Scores samples, not declared needles, so
                         every method is scored the same way and no extractor enters.
                         Recomputed here from each cell's ``points.csv`` and
                         ``ensemble_config.json`` (``pinned_optima``), at every
                         measured point, so it needs nothing the runner did not save.
    needles_found        true optima with a sample within the FOUND radius
                         ``found_radius(b, d)`` of the cell's landscape kind
                         (``needles.py`` / ``varied_height.py``) — the distance at which the
                         objective is one output-noise sd below the peak, so inside
                         it a measurement is indistinguishable from the optimum.
                         Set by the landscape (b, d) alone; at the default grid it
                         runs from 0.009 (2-D, b = 15) to 0.13 (9-D, b = 2.2). A
                         count, unlike greedy_dist's mean, separates "localised k
                         needles" from "equally far from all of them".
    frac_optima_visited  true optima with a sample within the match radius: the
                         same per-optimum minima, thresholded instead of averaged
                         (from ``metrics.json``). Higher is better.

``dist_to_needles`` (declared / extracted needles) is still written to every cell's
``metrics.json`` by the runner but is deliberately not summarised: it scores ZoMBI-Hop
on its declarations and everyone else through an extractor, and its unmatched
penalty dominates on low-``n`` landscapes.

Paired comparisons use the landscape as the unit: for every ``(dim, n, b, draw)`` on
which both a method and the reference finished, the difference in the metric, and
whether the method won. Every method saw the identical landscape and noise stream
there, so the pairing removes the landscape-to-landscape variance that dominates an
unpaired mean. Bootstrap intervals resample those landscapes.

Nothing is imputed: an unfinished or failed cell is simply absent, and ``n`` columns
say how many cells each number rests on.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ._paths import ensure_paths

ensure_paths()

from .campaign import (CELL_FILE, cell_budget, load_manifest, read_tasks,  # noqa: E402
                       task_dir)
from .needles import fn_from_config, landscape_module  # noqa: E402

DEFAULT_CI = 0.95
DEFAULT_N_BOOT = 2000
REFERENCE_METHOD = "zombi_hop"

#: Metric key -> (label, lower-is-better).
METRICS: dict[str, tuple[str, bool]] = {
    "greedy_dist": ("greedy_dist (optima → nearest sample)", True),
    "frac_optima_visited": ("fraction of optima visited", False),
}
#: The metric the trajectory figures trace.
CURVE_METRIC = "greedy_dist"
#: Extra columns carried into cells.csv (from ``metrics.json``) but not plotted.
EXTRA = ("best_f", "median_nn_spacing", "n_points", "budget_hit", "stop_reason",
         "runtime_s")

#: Categorical slots 1-8 of the dataviz reference palette (light surface), in
#: fixed order. A method keeps its slot by its position in the manifest's method
#: list, so its colour is the same in every figure of a campaign.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#9b59d0", "#7a6f5a")
#: The reference palette's blue ramp, light -> dark, for heatmaps.
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
              "#0d366b")
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


# ─── Collection ──────────────────────────────────────────────────────────────────

def greedy_curve(target: str, dim: int, basin_width: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """``(greedy_dist, running best f, needles found, found radius)`` after every
    measured point of one cell, or None if unreadable.

    ``D`` holds each optimum's distance to each sample; ``np.minimum.accumulate``
    turns it into "closest sample among the first k", and the mean over optima is
    the metric at k. Its last entry equals ``eval_metrics.metric_greedy_dist`` on
    the whole sample set. The running best is the cumulative max of the noiseless
    ``f`` — what the noisy ``y`` hid from the method, so it scores where the method
    sampled, not what it believed. Needles found counts the optima whose closest
    sample so far is within the found radius of the cell's landscape kind (read
    from ``ensemble_config.json``; a config without ``kind`` is a needles one)."""
    import pandas as pd

    try:
        with open(os.path.join(target, "ensemble_config.json")) as f:
            cfg = json.load(f)
        optima = np.asarray(cfg.get("pinned_optima") or [], dtype=float)
        df = pd.read_csv(os.path.join(target, "points.csv"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    cols = [f"x{i}" for i in range(int(dim))]
    if (not len(optima) or not len(df) or not set(cols) <= set(df.columns)
            or "f" not in df.columns):
        return None
    X = df[cols].to_numpy(float)
    D = np.minimum.accumulate(
        np.linalg.norm(X[:, None, :] - optima[None, :, :], axis=2), axis=0)  # (n_samples, k)
    radius = landscape_module(cfg.get("kind")).found_radius(basin_width, dim)
    return (D.mean(axis=1), np.maximum.accumulate(df["f"].to_numpy(float)),
            (D <= radius).sum(axis=1).astype(float), float(radius))


def collect(out_dir: str, manifest: dict) -> tuple[list[dict], list[dict]]:
    """One row per finished cell, and each cell's ``greedy_dist`` trajectory.

    ``f`` in a curve is the x axis as a fraction of the cell's budget, for figures
    that pool dimensions (budgets differ by dim)."""
    rows, curves = [], []
    for task in read_tasks(out_dir):
        target = task_dir(out_dir, task)
        path = os.path.join(target, CELL_FILE)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        got = greedy_curve(target, rec["dim"], rec["basin_width"])
        if got is None:
            continue
        g, best, found, radius = got
        m, land = rec.get("metrics", {}), rec.get("landscape", {})
        row = {"method": rec["method"], "cell": rec["cell"], "draw": rec["draw"],
               "dim": rec["dim"], "n_needles_true": rec["n_needles"],
               "basin_width": rec["basin_width"],
               "greedy_dist": round(float(g[-1]), 6),
               "needles_found": int(found[-1]),
               "found_radius": round(radius, 6)}
        for key in (*METRICS, *EXTRA):
            if key != "greedy_dist":
                row[key] = m.get(key)
        x = np.arange(1, len(g) + 1, dtype=float)
        curves.append({"method": rec["method"], "dim": rec["dim"],
                       "n_needles_true": rec["n_needles"],
                       "basin_width": rec["basin_width"], "draw": rec["draw"],
                       "x": x, "f": x / cell_budget(manifest, rec["dim"]), "y": g,
                       "best": best, "found": found, "found_radius": radius})
        row.update({
            "separation_achieved": land.get("separation_achieved"),
            "prominence_target_met": land.get("prominence_target_met"),
            "n_prominence_resolved": land.get("n_prominence_resolved"),
            "basin_plain_radius": land.get("basin_plain_radius"),
            "config_source": rec.get("config_source"),
        })
        rows.append(row)
    return rows, curves


# ─── Statistics ──────────────────────────────────────────────────────────────────

def _boot_ci(values: np.ndarray, ci: float, n_boot: int, rng) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    means = values[idx].mean(axis=1)
    lo = (1.0 - ci) / 2.0
    return (float(np.quantile(means, lo)), float(np.quantile(means, 1.0 - lo)))


def _summ(vals: list, ci: float, n_boot: int, rng, prefix: str) -> dict:
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {f"{prefix}_mean": None, f"{prefix}_lo": None, f"{prefix}_hi": None,
                f"{prefix}_n": 0}
    lo, hi = _boot_ci(v, ci, n_boot, rng)
    return {f"{prefix}_mean": round(float(v.mean()), 6),
            f"{prefix}_lo": None if np.isnan(lo) else round(lo, 6),
            f"{prefix}_hi": None if np.isnan(hi) else round(hi, 6),
            f"{prefix}_n": int(len(v))}


def aggregate(rows: list[dict], keys: tuple[str, ...], *, ci: float, n_boot: int,
              seed: int) -> list[dict]:
    """Collapse rows sharing ``keys`` into means + CIs for every metric."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for gk in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        group = groups[gk]
        rec = dict(zip(keys, gk))
        rec["n_cells"] = len(group)
        rec["budget_hit_frac"] = round(float(np.mean(
            [bool(g.get("budget_hit")) for g in group])), 4)
        for m in METRICS:
            rec.update(_summ([g.get(m) for g in group], ci, n_boot, rng, m))
        out.append(rec)
    return out


def paired(rows: list[dict], reference: str, *, ci: float, n_boot: int,
           seed: int) -> list[dict]:
    """Each method minus the reference, paired on (dim, n, b, draw)."""
    rng = np.random.default_rng(seed + 1)
    key = ("dim", "n_needles_true", "basin_width", "draw")
    ref = {tuple(r[k] for k in key): r for r in rows if r["method"] == reference}
    out = []
    for method in sorted({r["method"] for r in rows} - {reference}):
        mine = [r for r in rows if r["method"] == method]
        for dim in [None] + sorted({r["dim"] for r in mine}):
            rec = {"method": method, "reference": reference,
                   "dim": "all" if dim is None else dim}
            for metric, (_, lower_better) in METRICS.items():
                diffs, wins = [], []
                for r in mine:
                    if dim is not None and r["dim"] != dim:
                        continue
                    other = ref.get(tuple(r[k] for k in key))
                    if other is None or r.get(metric) is None or other.get(metric) is None:
                        continue
                    d = float(r[metric]) - float(other[metric])
                    diffs.append(d)
                    wins.append((d < 0) if lower_better else (d > 0))
                rec.update(_summ(diffs, ci, n_boot, rng, f"{metric}_diff"))
                rec[f"{metric}_win_rate"] = (round(float(np.mean(wins)), 4)
                                             if wins else None)
            out.append(rec)
    return out


# ─── Figures ─────────────────────────────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "text.color": INK, "axes.titlecolor": INK,
        "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _method_colors(methods: list[str]) -> dict[str, str]:
    return {m: CATEGORICAL[i % len(CATEGORICAL)] for i, m in enumerate(methods)}


def _style(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _method_by_dim(agg_md: list[dict], methods: list[str], path: str, ci: float) -> None:
    """One panel per metric: mean vs dimension, one line per method, CI bars."""
    plt = _plt()
    colors = _method_colors(methods)
    dims = sorted({r["dim"] for r in agg_md})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.6 * len(METRICS) + 0.6, 3.6),
                             squeeze=False)
    for ax, (metric, (label, lower_better)) in zip(axes[0], METRICS.items()):
        for j, method in enumerate(methods):
            pts = [r for r in agg_md if r["method"] == method
                   and r.get(f"{metric}_mean") is not None]
            if not pts:
                continue
            pts.sort(key=lambda r: r["dim"])
            # Small horizontal dodge so overlapping CI bars stay readable.
            x = np.array([dims.index(r["dim"]) for r in pts], float) \
                + (j - (len(methods) - 1) / 2) * 0.06
            y = np.array([r[f"{metric}_mean"] for r in pts])
            lo = np.array([r[f"{metric}_lo"] if r[f"{metric}_lo"] is not None else m
                           for r, m in zip(pts, y)])
            hi = np.array([r[f"{metric}_hi"] if r[f"{metric}_hi"] is not None else m
                           for r, m in zip(pts, y)])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=colors[method], lw=2,
                        marker="o", ms=6, mec="white", mew=1.5, capsize=0,
                        elinewidth=1.2, label=method)
        ax.set_xticks(range(len(dims)), [str(d) for d in dims])
        ax.set_xlabel("dimension")
        ax.set_title(f"{label}\n({'lower' if lower_better else 'higher'} is better)",
                     fontsize=9.5)
        _style(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Mean over landscapes, {int(ci * 100)}% bootstrap CI", y=1.10,
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heatmaps(agg: list[dict], metric: str, methods: list[str], path: str) -> None:
    """Rows = method, columns = dim; each tile an n x b grid; one shared scale."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    label, lower_better = METRICS[metric]
    dims = sorted({r["dim"] for r in agg})
    counts = sorted({r["n_needles_true"] for r in agg})
    widths = sorted({r["basin_width"] for r in agg})
    methods = [m for m in methods if any(r["method"] == m for r in agg)]
    if not (dims and counts and widths and methods):
        return
    grids = {}
    for m in methods:
        for d in dims:
            M = np.full((len(counts), len(widths)), np.nan)
            for r in agg:
                if r["method"] == m and r["dim"] == d and r.get(f"{metric}_mean") is not None:
                    M[counts.index(r["n_needles_true"]),
                      widths.index(r["basin_width"])] = r[f"{metric}_mean"]
            grids[(m, d)] = M
    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    if not len(finite):
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    cmap.set_bad("#f1f0ea")

    fig, axes = plt.subplots(len(methods), len(dims),
                             figsize=(2.6 * len(dims) + 1.4, 2.3 * len(methods) + 0.8),
                             squeeze=False)
    im = None
    for i, m in enumerate(methods):
        for j, d in enumerate(dims):
            ax = axes[i][j]
            G = grids[(m, d)]
            im = ax.imshow(np.ma.masked_invalid(G), cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower", aspect="auto")
            ax.set_xticks(range(len(widths)), [f"{w:g}" for w in widths], fontsize=8)
            ax.set_yticks(range(len(counts)), [str(c) for c in counts], fontsize=8)
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            for a in range(len(counts)):
                for b in range(len(widths)):
                    v = G[a, b]
                    if np.isfinite(v):
                        dark = (v - vmin) / (vmax - vmin) > 0.55
                        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if dark else INK)
            if i == 0:
                ax.set_title(f"dim {d}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{m}\nneedles n", fontsize=9)
            if i == len(methods) - 1:
                ax.set_xlabel("sharpness b", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label=f"{label} ({'lower' if lower_better else 'higher'} is better)")
    fig.suptitle(f"{label} — mean over draws per landscape configuration", fontsize=10.5)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _stack(group: list[dict], xkey: str = "x"):
    """``(xs, Y)``: a group of curves on a common x grid (one row per curve), or None.

    The common grid is the x values every curve reaches, so a short curve is never
    extended with an invented flat tail. Step interpolation: a metric holds its last
    measured value between points."""
    x_max = min(c[xkey][-1] for c in group)
    xs = np.unique(np.concatenate([c[xkey] for c in group]))
    xs = xs[xs <= x_max]
    if len(xs) < 2:
        return None
    Y = np.vstack([c["y"][np.clip(np.searchsorted(c[xkey], xs, side="right") - 1,
                                  0, len(c["y"]) - 1)] for c in group])
    return xs, Y


def _pool(group: list[dict], xkey: str = "x"):
    """``(xs, mean, q25, q75)`` over a group of curves, or None (see ``_stack``)."""
    stacked = _stack(group, xkey)
    if stacked is None:
        return None
    xs, Y = stacked
    return xs, Y.mean(axis=0), np.quantile(Y, 0.25, axis=0), np.quantile(Y, 0.75, axis=0)


def _pool_ci(group: list[dict], ci: float, n_boot: int, rng):
    """``(xs, mean, lo, hi)``: the mean curve with a pointwise percentile bootstrap CI
    of the mean (curves resampled whole), or None. With one curve the band is the
    curve itself."""
    stacked = _stack(group)
    if stacked is None:
        return None
    xs, Y = stacked
    mu = Y.mean(axis=0)
    if len(Y) < 2:
        return xs, mu, mu, mu
    means = Y[rng.integers(0, len(Y), size=(int(n_boot), len(Y)))].mean(axis=1)
    lo = (1.0 - ci) / 2.0
    return xs, mu, np.quantile(means, lo, axis=0), np.quantile(means, 1.0 - lo, axis=0)


def _level_colors(n: int) -> list[str]:
    """``n`` ordered levels (dim, n, b) -> steps of the sequential ramp, light to dark.
    The two palest steps are skipped: a 2px line in them vanishes on white."""
    ramp = SEQUENTIAL[2:]
    return [ramp[i] for i in np.linspace(0, len(ramp) - 1, max(n, 1)).round().astype(int)]


def _curves_by_dim(curves: list[dict], methods: list[str], ylabel: str,
                   path: str, title: str) -> None:
    """Panel per dim; per method the mean curve (on a common x grid) + IQR band."""
    plt = _plt()
    colors = _method_colors(methods)
    dims = sorted({c["dim"] for c in curves})
    if not dims:
        return
    fig, axes = plt.subplots(1, len(dims), figsize=(3.4 * len(dims) + 0.6, 3.4),
                             sharey=True, squeeze=False)
    for ax, d in zip(axes[0], dims):
        for m in methods:
            group = [c for c in curves if c["dim"] == d and c["method"] == m]
            pooled = _pool(group) if group else None
            if pooled is None:
                continue
            xs, mu, lo, hi = pooled
            ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.14, lw=0)
            ax.plot(xs, mu, color=colors[m], lw=2, label=f"{m} ({len(group)})")
        ax.set_title(f"dim {d}", fontsize=10)
        ax.set_xlabel("measured points")
        _style(ax)
        ax.legend(fontsize=7.5, frameon=False, title="method (cells)",
                  title_fontsize=7.5)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title + " — mean, band is the IQR across cells", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _traj_all(curves: list[dict], methods: list[str], path: str) -> None:
    """Panel per method: every cell's trajectory, faint and coloured by dim, with the
    per-dim mean drawn bold over the top. Shows the spread the means are made of —
    where a bimodal slice or a single divergent cell would live."""
    plt = _plt()
    methods = [m for m in methods if any(c["method"] == m for c in curves)]
    dims = sorted({c["dim"] for c in curves})
    if not (methods and dims):
        return
    colours = dict(zip(dims, _level_colors(len(dims))))
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods) + 0.4, 3.8),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        mine = [c for c in curves if c["method"] == m]
        for c in mine:
            ax.plot(c["x"], c["y"], color=colours[c["dim"]], lw=0.6, alpha=0.25, zorder=1)
        for d in dims:
            group = [c for c in mine if c["dim"] == d]
            pooled = _pool(group) if group else None
            if pooled is None:
                continue
            xs, mu, _, _ = pooled
            # A white underlay keeps each mean readable where it crosses the others.
            ax.plot(xs, mu, color="white", lw=4.0, zorder=2)
            ax.plot(xs, mu, color=colours[d], lw=2.0, zorder=3,
                    label=f"dim {d} ({len(group)})")
        ax.set_title(f"{m}  ({len(mine)} cells)", fontsize=10)
        ax.set_xlabel("measured points")
        _style(ax)
    axes[0][0].set_ylabel(f"{CURVE_METRIC} (lower is better)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, [lb.split(" (")[0] for lb in labels], loc="upper center",
               ncol=len(dims), frameon=False, bbox_to_anchor=(0.5, 1.05),
               title="bold = mean per dimension")
    fig.suptitle(f"Every cell's {CURVE_METRIC} trajectory — one faint line per cell",
                 y=1.15, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _traj_by_axis(curves: list[dict], methods: list[str], path: str) -> None:
    """Row per method, column per swept axis: one curve per level of that axis with
    the other two pooled. The main-effects view — does raising d, n or b change the
    shape of the search, or only where it ends up. Budgets differ by dim, so x is the
    fraction of the cell's budget spent."""
    plt = _plt()
    methods = [m for m in methods if any(c["method"] == m for c in curves)]
    if not methods:
        return
    specs = [("dim", "dimension d"), ("n_needles_true", "needles n"),
             ("basin_width", "sharpness b")]
    fig, axes = plt.subplots(len(methods), 3, figsize=(12.0, 2.7 * len(methods) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(methods):
        mine = [c for c in curves if c["method"] == m]
        for j, (key, label) in enumerate(specs):
            ax = axes[i][j]
            levels = sorted({c[key] for c in curves})
            for colour, lv in zip(_level_colors(len(levels)), levels):
                group = [c for c in mine if c[key] == lv]
                pooled = _pool(group, "f") if group else None
                if pooled is None:
                    continue
                xs, mu, lo, hi = pooled
                ax.fill_between(xs, lo, hi, color=colour, alpha=0.12, lw=0)
                ax.plot(xs, mu, color=colour, lw=2, label=f"{lv:g}")
            _style(ax)
            if i == 0:
                # One legend per column: each column's levels mean something different.
                ax.legend(title=label, fontsize=8, title_fontsize=8.5, frameon=False,
                          ncol=len(levels), loc="lower center",
                          bbox_to_anchor=(0.5, 1.02))
            if j == 0:
                ax.set_ylabel(f"{m}\n{CURVE_METRIC}", fontsize=9)
            if i == len(methods) - 1:
                ax.set_xlabel("fraction of budget spent")
    fig.suptitle(f"{CURVE_METRIC} over the budget, by axis — mean over every cell in "
                 "the slice, band is the IQR (lower is better)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _traj_grid(curves: list[dict], methods: list[str], manifest: dict,
               path: str, *, metric: str = CURVE_METRIC,
               better: str = "lower", width: float | None = None,
               draw: int | None = None, ci: float | None = None,
               n_boot: int = DEFAULT_N_BOOT, rng=None, count: bool = False) -> None:
    """Row per dim, column per needle count, one line per method (mean over
    draws unless ``draw`` picks one, and over sharpness unless ``width`` picks
    one). The by-axis view marginalises, which hides interactions; this keeps
    dim x n and compares methods inside each panel.

    With ``ci`` set, a mean line gets its pointwise bootstrap CI as a band (curves
    resampled whole, see ``_pool_ci``); a single draw has no band. ``count`` is for
    a needles-found curve: each column's y axis runs 0..n, and each row is labelled
    with its found radius (one per dim at a fixed ``width``)."""
    plt = _plt()
    if width is not None:
        curves = [c for c in curves if np.isclose(float(c["basin_width"]), width)]
    if draw is not None:
        curves = [c for c in curves if int(c["draw"]) == draw]
    banded = ci is not None and draw is None
    if banded and rng is None:
        rng = np.random.default_rng(0)
    colors = _method_colors(methods)
    dims = sorted({c["dim"] for c in curves})
    counts = sorted({c["n_needles_true"] for c in curves})
    if not (dims and counts):
        return
    fig, axes = plt.subplots(len(dims), len(counts),
                             figsize=(3.0 * len(counts) + 0.6, 2.5 * len(dims) + 0.8),
                             sharex="row", sharey="col" if count else True,
                             squeeze=False)
    for i, d in enumerate(dims):
        for j, n in enumerate(counts):
            ax = axes[i][j]
            for m in methods:
                group = [c for c in curves if c["method"] == m and c["dim"] == d
                         and c["n_needles_true"] == n]
                if not group:
                    continue
                if banded:
                    pooled = _pool_ci(group, ci, n_boot, rng)
                    if pooled is None:
                        continue
                    xs, mu, lo, hi = pooled
                    ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.16, lw=0)
                else:
                    pooled = _pool(group)
                    if pooled is None:
                        continue
                    xs, mu, _, _ = pooled
                ax.plot(xs, mu, color=colors[m], lw=2, label=m)
            _style(ax)
            if count:
                ax.set_ylim(-0.03 * n, 1.05 * n)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if j == 0:
                label = f"dim {d}"
                radii = {c["found_radius"] for c in curves
                         if c["dim"] == d and "found_radius" in c}
                if count and len(radii) == 1:
                    label += f"\nr = {radii.pop():.3g}"
                ax.set_ylabel(label, fontsize=10)
            if i == len(dims) - 1:
                ax.set_xlabel("measured points", fontsize=9)
    handles, labels = [], []
    for row in axes:
        for ax in row:
            for h, lb in zip(*ax.get_legend_handles_labels()):
                if lb not in labels:
                    handles.append(h)
                    labels.append(lb)
    order = [labels.index(m) for m in methods if m in labels]
    fig.legend([handles[k] for k in order], [labels[k] for k in order],
               loc="upper center", ncol=len(order), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    over = "" if width is None else f"sharpness b = {width:g}: "
    if draw is None:
        over += ("mean over " + ("sharpness and " if width is None else "")
                 + f"{manifest.get('n_draws', '?')} draw(s) per line")
        if banded:
            over += f", band = {int(round(ci * 100))}% bootstrap CI of the mean"
    else:
        over += f"single draw (draw {draw})" \
            + (", mean over sharpness" if width is None else "")
    fig.suptitle(f"{metric} over the budget — {over} ({better} is better)",
                 fontsize=10.5, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#: Sampling maps are drawn only for landscapes a plane can show exactly.
SAMPLING_DIM = 2
SAMPLING_GRID_N = 201
#: Height of the trajectory block under each sampling map, and how many single
#: draws it shows before the mean-over-draws column.
SAMPLING_TRAJ_H = 6.0
SAMPLING_SINGLE_DRAWS = 2
#: The one draw (1-based) the single-trial greedy grids show instead of the mean.
SINGLE_TRIAL = 1


def _sampling_trajectories(sub, curves: list[dict], methods: list[str], *, ci: float,
                           n_boot: int, rng) -> None:
    """The block under a sampling map: row 1 running best f, row 2 greedy_dist;
    columns are single draws (no averaging) then the mean over every finished draw
    with a bootstrap CI. One line per method.

    The single draws are the lowest-numbered ones every plotted method finished, so
    each panel compares methods on one identical landscape (draw 1 whenever it is
    complete, i.e. the landscape drawn above). Different draws are different
    landscapes — same (d, n, b), new needle positions."""
    plt = _plt()
    colors = _method_colors(methods)
    shown = [m for m in methods if any(c["method"] == m for c in curves)]
    by_draw: dict[int, dict[str, dict]] = {}
    for c in curves:
        by_draw.setdefault(int(c["draw"]), {})[c["method"]] = c
    complete = [d for d in sorted(by_draw) if set(shown) <= set(by_draw[d])]
    singles = complete[:SAMPLING_SINGLE_DRAWS]
    n_cols = SAMPLING_SINGLE_DRAWS + 1
    axes = sub.subplots(2, n_cols, sharex=True, squeeze=False)
    rows = [("best", "running best f (noiseless)", "higher"),
            ("y", CURVE_METRIC, "lower")]
    for i, (key, label, better) in enumerate(rows):
        for j in range(n_cols):
            ax = axes[i][j]
            _style(ax)
            if j < SAMPLING_SINGLE_DRAWS:
                if j >= len(singles):
                    ax.text(0.5, 0.5, "no further draw finished\nby every method",
                            ha="center", va="center", color=INK_MUTED,
                            transform=ax.transAxes, fontsize=9)
                    title = f"single draw #{j + 1}"
                else:
                    d = singles[j]
                    for m in shown:
                        c = by_draw[d][m]
                        ax.plot(c["x"], c[key], color=colors[m], lw=1.8, label=m)
                    title = f"draw {d} only"
            else:
                for m in shown:
                    group = [{**c, "y": c[key]} for c in curves if c["method"] == m]
                    pooled = _pool_ci(group, ci, n_boot, rng) if group else None
                    if pooled is None:
                        continue
                    xs, mu, lo, hi = pooled
                    ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.16, lw=0)
                    ax.plot(xs, mu, color=colors[m], lw=1.8, label=m)
                n_by = sorted({sum(c["method"] == m for c in curves) for m in shown})
                n_txt = (f"{n_by[0]}" if len(n_by) == 1
                         else f"{n_by[0]}–{n_by[-1]}")
                title = f"mean over {n_txt} draws, {int(ci * 100)}% bootstrap CI"
            if i == 0:
                ax.set_title(title, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{label}\n({better} is better)", fontsize=9)
            if i == 1:
                ax.set_xlabel("measured points")
    handles = [plt.Line2D([], [], color=colors[m], lw=2) for m in shown]
    sub.legend(handles, shown, loc="outside upper center", ncol=len(shown),
               frameon=False)


def _sampling_maps(out_dir: str, methods: list[str], sdir: str, curves: list[dict],
                   *, ci: float, n_boot: int, seed: int) -> list[str]:
    """Where each method sampled, one figure per 2-D landscape configuration.

    Draw 1 only (any draw would do; every method on a draw sees the same landscape).
    Each panel is the true noiseless landscape with one method's samples on it. A
    sample is filled with the colour the landscape has AT that sample (its noiseless
    ``f`` on the same scale), so it reads as a see-through ring on the map: a ring
    darker than its surroundings sits on a peak. Under the maps, the running best f
    and greedy_dist trajectories on this configuration (``_sampling_trajectories``).
    Returns the files written."""
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap, Normalize


    plt = _plt()
    rng = np.random.default_rng(seed + 2)
    tasks = [t for t in read_tasks(out_dir)
             if int(t["dim"]) == SAMPLING_DIM and int(t["draw"]) == 1]
    written = []
    for name in sorted({t["name"] for t in tasks}):
        cells = {}
        for t in tasks:
            target = task_dir(out_dir, t)
            if (t["name"] == name and os.path.isfile(os.path.join(target, CELL_FILE))
                    and os.path.isfile(os.path.join(target, "points.csv"))):
                cells[t["method"]] = target
        shown = [m for m in methods if m in cells]
        if not shown:
            continue
        with open(os.path.join(cells[shown[0]], "ensemble_config.json")) as f:
            fn = fn_from_config(json.load(f))
        g = np.linspace(0.0, 1.0, SAMPLING_GRID_N)
        gx, gy = np.meshgrid(g, g)
        Z = np.asarray(fn.predict(np.column_stack([gx.ravel(), gy.ravel()])),
                       float).reshape(gx.shape)
        pts = {m: pd.read_csv(os.path.join(cells[m], "points.csv")) for m in shown}
        f_all = np.concatenate([p["f"].to_numpy(float) for p in pts.values()])
        norm = Normalize(vmin=min(Z.min(), f_all.min()), vmax=max(Z.max(), f_all.max()))
        cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)

        t0 = next(t for t in tasks if t["name"] == name)
        fig = plt.figure(figsize=(3.3 * len(shown) + 0.9, 3.6 + SAMPLING_TRAJ_H),
                         layout="constrained")
        top, bottom = fig.subfigures(2, 1, height_ratios=[3.6, SAMPLING_TRAJ_H],
                                     hspace=0.04)
        axes = top.subplots(1, len(shown), squeeze=False)
        im = None
        for ax, m in zip(axes[0], shown):
            im = ax.pcolormesh(g, g, Z, cmap=cmap, norm=norm, shading="nearest",
                               rasterized=True)
            p = pts[m]
            ax.scatter(p["x0"], p["x1"], c=cmap(norm(p["f"].to_numpy(float))), s=16,
                       edgecolors=INK, linewidths=0.5, zorder=3)
            ax.set_title(f"{m}  ({len(p)} points)", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel("x0")
            ax.set_yticks([0, 0.5, 1])
            ax.set_xticks([0, 0.5, 1])
        axes[0][0].set_ylabel("x1")
        top.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                     label="objective (noiseless)")
        top.suptitle(f"Where each method sampled — dim {SAMPLING_DIM}, "
                     f"n = {t0['n_needles']}, b = {float(t0['basin_width']):g}, draw 1. "
                     "Each point is filled with the landscape's colour at that point.",
                     fontsize=10.5)
        _sampling_trajectories(
            bottom, [c for c in curves if c["dim"] == SAMPLING_DIM
                     and c["n_needles_true"] == int(t0["n_needles"])
                     and np.isclose(float(c["basin_width"]), float(t0["basin_width"]))],
            methods, ci=ci, n_boot=n_boot, rng=rng)
        path = os.path.join(sdir, f"sampling_{name}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(os.path.basename(path))
    return written


# ─── Entry point ─────────────────────────────────────────────────────────────────

def _fmt(rec: dict, key: str, digits: int = 3) -> str:
    m = rec.get(f"{key}_mean")
    if m is None:
        return "—"
    lo, hi = rec.get(f"{key}_lo"), rec.get(f"{key}_hi")
    band = f" [{lo:.{digits}f}, {hi:.{digits}f}]" if lo is not None else ""
    return f"{m:.{digits}f}{band}"


def summarize(out_dir: str, *, ci: float = DEFAULT_CI, n_boot: int = DEFAULT_N_BOOT,
              seed: int = 0) -> None:
    """Write ``summary/`` for a finished or partial campaign."""
    import pandas as pd

    out_dir = os.path.abspath(out_dir)
    manifest = load_manifest(out_dir)
    sdir = os.path.join(out_dir, "summary")
    os.makedirs(sdir, exist_ok=True)

    rows, curves = collect(out_dir, manifest)
    if not rows:
        print(f"  no finished cells in {out_dir} yet — nothing to summarise")
        return
    methods = [m for m in manifest["methods"] if any(r["method"] == m for r in rows)]
    kw = dict(ci=ci, n_boot=n_boot, seed=seed)
    agg = aggregate(rows, ("method", "dim", "n_needles_true", "basin_width"), **kw)
    agg_md = aggregate(rows, ("method", "dim"), **kw)
    agg_m = aggregate(rows, ("method",), **kw)
    ref = REFERENCE_METHOD if REFERENCE_METHOD in methods else methods[0]
    pair = paired(rows, ref, **kw)

    pd.DataFrame(rows).to_csv(os.path.join(sdir, "cells.csv"), index=False)
    pd.DataFrame(agg).to_csv(os.path.join(sdir, "grid.csv"), index=False)
    pd.DataFrame([{**r, "dim": "all"} for r in agg_m] + agg_md).to_csv(
        os.path.join(sdir, "methods.csv"), index=False)
    pd.DataFrame(pair).to_csv(os.path.join(sdir, "paired.csv"), index=False)

    _method_by_dim(agg_md, methods, os.path.join(sdir, "method_by_dim.png"), ci)
    for metric in METRICS:
        _heatmaps(agg, metric, methods, os.path.join(sdir, f"{metric}_heatmap.png"))
    _curves_by_dim(curves, methods, f"{CURVE_METRIC} (lower is better)",
                   os.path.join(sdir, "greedy_over_time.png"),
                   f"{CURVE_METRIC} over the budget")
    _traj_all(curves, methods, os.path.join(sdir, "greedy_over_time_all.png"))
    _traj_by_axis(curves, methods, os.path.join(sdir, "greedy_over_time_by_axis.png"))
    widths = sorted({float(c["basin_width"]) for c in curves})
    band = dict(ci=ci, n_boot=n_boot, rng=np.random.default_rng(seed + 3))
    found_curves = [{**c, "y": c["found"]} for c in curves]
    found_kw = dict(metric="needles found", better="higher", count=True)
    for w in widths:
        _traj_grid(curves, methods, manifest,
                   os.path.join(sdir, f"greedy_over_time_grid_b{w:g}.png"), width=w,
                   **band)
        _traj_grid(curves, methods, manifest,
                   os.path.join(sdir, f"greedy_over_time_grid_b{w:g}_trial{SINGLE_TRIAL}.png"),
                   width=w, draw=SINGLE_TRIAL)
        _traj_grid(found_curves, methods, manifest,
                   os.path.join(sdir, f"needles_found_grid_b{w:g}.png"), width=w,
                   **found_kw, **band)
        _traj_grid(found_curves, methods, manifest,
                   os.path.join(sdir, f"needles_found_grid_b{w:g}_trial{SINGLE_TRIAL}.png"),
                   width=w, draw=SINGLE_TRIAL, **found_kw)
    _traj_grid([{**c, "y": c["best"]} for c in curves], methods, manifest,
               os.path.join(sdir, "best_f_over_time_grid.png"),
               metric="running best f (noiseless)", better="higher", **band)
    sampling = _sampling_maps(out_dir, methods, sdir, curves, **kw)

    n_expected = manifest["n_tasks"]
    dims_all = sorted({int(d) for d in manifest["grid"]["dims"]})
    budget_text = (", ".join(f"{cell_budget(manifest, d)} at {d}d" for d in dims_all)
                   if manifest.get("budgets") else str(manifest["budget"]))
    short = [r for r in rows if r.get("budget_hit") is False]
    lines = [
        "# Method sweep — ZoMBI-Hop vs. black-box baselines on needle landscapes",
        "",
        f"{len(rows)} of {n_expected} cell(s) finished — {len(methods)} method(s) x "
        f"{manifest['n_configurations']} landscape configuration(s) x "
        f"{manifest['n_draws']} draw(s).",
        "",
        f"Budget: **{budget_text} measured points per cell** in batches of "
        f"{manifest['batch_size']}, identical for every method at a given dimension. "
        f"Noise: "
        f"input {manifest['input_noise']:g}, output {manifest['output_noise_frac']:g} x |y|. "
        + ("Landscape: bumps-only `CartesianEnsemble` on the unit cube — *n* "
           "negated-Ackley needles of sharpness *b* peaking at 1.0 on a plain at 0.75 "
           "(`benchmarks/sweeps/needles.py`). "
           if manifest.get("landscape", {}).get("kind", "needles") == "needles" else
           "Landscape: **varied height** — *n* negated-Ackley needles of sharpness *b* "
           "on the unit cube, heights drawn U(0.5, 1) with the tallest set to 1.0, on "
           "a flat plain at 0 (`benchmarks/sweeps/varied_height.py`). The output noise "
           "is multiplicative, so the plain reads exactly 0 and the noise at a peak is "
           "4.5% of its height. ")
        + "Every method saw the identical landscape and noise stream for a given cell.",
        "",
        "Every method is scored on its **samples**, the same way: `greedy_dist` is, "
        "for each true optimum, the distance to the nearest point the method measured, "
        "averaged over the optima (`eval_metrics.metric_greedy_dist`). No declared "
        "needles, no extractor, no unmatched penalty; two optima may share a nearest "
        "sample. Computed from each cell's `points.csv` at every measured point, so "
        "its curves only ever fall.",
        "",
        "## Headline (all landscapes)",
        "",
        "| method | cells | greedy_dist | optima visited |",
        "|---|---|---|---|",
    ]
    for r in sorted(agg_m, key=lambda r: methods.index(r["method"])):
        lines.append(f"| {r['method']} | {r['n_cells']} | {_fmt(r, 'greedy_dist')} | "
                     f"{_fmt(r, 'frac_optima_visited')} |")
    lines += ["", f"Means with {int(ci * 100)}% bootstrap intervals over cells. Lower "
              "is better except *optima visited*.", "",
              f"## Paired against `{ref}` (all landscapes)", "",
              "Difference = method − reference on the same landscape (negative is "
              "better for the distances). Win rate = share of landscapes "
              "where the method did strictly better.", "",
              "| method | pairs | Δ greedy_dist | win rate | Δ optima visited "
              "| win rate |",
              "|---|---|---|---|---|---|"]
    for r in pair:
        if r["dim"] != "all":
            continue
        lines.append(
            f"| {r['method']} | {r['greedy_dist_diff_n']} | "
            f"{_fmt(r, 'greedy_dist_diff')} | {r['greedy_dist_win_rate']} | "
            f"{_fmt(r, 'frac_optima_visited_diff')} | "
            f"{r['frac_optima_visited_win_rate']} |")
    lines += ["", "Per-dimension pairs are in `paired.csv`.", "",
              "## Configurations", "", "| method | dim | source |", "|---|---|---|"]
    for method in methods:
        for dim, rec in manifest["method_configs"][method].items():
            star = " **(stand-in)**" if rec["is_stand_in"] else ""
            lines.append(f"| {method} | {dim} | {rec['source']}{star} |")
    lines += [
        "",
        "ZoMBI-Hop's hyperparameters were tuned by MOBO on the *simplex*; the baselines "
        "run their published defaults. Neither was tuned on these landscapes.",
        "",
        "## Figures",
        "",
        "![method by dim](method_by_dim.png)",
        "",
        *[f"![{m}]({m}_heatmap.png)" for m in METRICS],
        "",
        "![greedy over time](greedy_over_time.png)",
        "![greedy over time by axis](greedy_over_time_by_axis.png)",
        *[f"![greedy over time faceted, b = {w:g}](greedy_over_time_grid_b{w:g}.png)"
          for w in widths],
        *[f"![greedy over time faceted, b = {w:g}, draw {SINGLE_TRIAL} only]"
          f"(greedy_over_time_grid_b{w:g}_trial{SINGLE_TRIAL}.png)" for w in widths],
        *[f"![needles found, b = {w:g}](needles_found_grid_b{w:g}.png)" for w in widths],
        *[f"![needles found, b = {w:g}, draw {SINGLE_TRIAL} only]"
          f"(needles_found_grid_b{w:g}_trial{SINGLE_TRIAL}.png)" for w in widths],
        "![running best faceted](best_f_over_time_grid.png)",
        "![all trajectories](greedy_over_time_all.png)",
        "",
        "The by-axis figure pools dimensions with different budgets, so its x axis is "
        "the fraction of the cell's budget spent; the others are in measured points.",
        "",
        "## Where each method sampled",
        "",
        *([f"Draw 1 of every {SAMPLING_DIM}-D landscape: the true landscape, one panel "
           "per method, each sample filled with the landscape's colour at that point. "
           "Under it, running best f (top) and greedy_dist (bottom) on that "
           "configuration: two single draws (the lowest-numbered draws every method "
           "finished, one line per method, no averaging) and the mean over every "
           f"finished draw with a {int(ci * 100)}% bootstrap CI.",
           "", *[f"![{f}]({f})" for f in sampling], ""] if sampling else
          [f"Drawn only for {SAMPLING_DIM}-D landscapes; this campaign has none.", ""]),
    ]
    if short:
        lines += ["## Cells that did not spend their budget", "",
                  f"{len(short)} cell(s) stopped early (wall-clock ceiling or the "
                  "method returning on its own); they are not comparable on equal terms.",
                  "", "| method | cell | draw | points | stop |", "|---|---|---|---|---|"]
        lines += [f"| {r['method']} | {r['cell']} | {r['draw']} | {r['n_points']} | "
                  f"{r['stop_reason']} |" for r in short[:40]]
        lines.append("")
    with open(os.path.join(sdir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  summary -> {sdir}  ({len(rows)} cell(s), {len(methods)} method(s))")
