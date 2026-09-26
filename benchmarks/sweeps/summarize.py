"""
benchmarks/sweeps/summarize.py
==============================
Turn a drained (or partly drained) method x landscape sweep into tables and figures.

The question is "which method recovers the set of needles best, and on which
landscapes", so the summary is organised around the METHOD:

    summary/
    ├── index.md                   headline table, paired comparisons, every figure
    ├── cells.csv                  one row per finished cell
    ├── grid.csv                   per (method, dim, n, b): means + bootstrap CIs
    ├── methods.csv                per (method, dim) and overall: means + CIs
    ├── paired.csv                 each method vs the reference, paired by landscape
    ├── method_by_dim.png          the headline: each metric vs dim, one line per method
    ├── <metric>_heatmap.png       rows = method, columns = dim, tile = n x b
    ├── dist_over_time.png         dist_to_needles vs measured points, panel per dim
    └── regret_over_time.png       simple regret vs measured points, panel per dim

Metrics (all from each cell's ``metrics.json``; see ``benchmarks/methods/runner.py``)
------------------------------------------------------------------------------------
    dist_to_needles            headline. Each method's own needles (ZoMBI-Hop's
                               declarations; the extractor's for the rest). Lower
                               is better, range [0, 0.5].
    dist_to_needles_extracted  the SAME extractor on every method's samples — the
                               comparison in which methods differ only in where they
                               sampled. Lower is better.
    frac_optima_visited        true optima with a sample within the match radius:
                               did the method ever measure there. Higher is better.
    simple_regret              1 - best noiseless value sampled: what single-optimum
                               BO optimises for. Lower is better.

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

from .campaign import CELL_FILE, load_manifest, read_tasks, task_dir  # noqa: E402

DEFAULT_CI = 0.95
DEFAULT_N_BOOT = 2000
REFERENCE_METHOD = "zombi_hop"

#: Metric key -> (label, lower-is-better).
METRICS: dict[str, tuple[str, bool]] = {
    "dist_to_needles": ("dist_to_needles (own needles)", True),
    "dist_to_needles_extracted": ("dist_to_needles (common extractor)", True),
    "frac_optima_visited": ("fraction of optima visited", False),
    "simple_regret": ("simple regret", True),
}
#: Extra columns carried into cells.csv but not plotted.
EXTRA = ("n_needles", "n_needles_extracted", "frac_optima_found",
         "frac_optima_found_extracted", "needle_precision", "needle_precision_extracted",
         "best_f", "median_nn_spacing", "n_points", "budget_hit", "stop_reason",
         "runtime_s", "scoring_s", "needles_source")

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

def collect(out_dir: str) -> list[dict]:
    """One row per finished cell."""
    rows = []
    for task in read_tasks(out_dir):
        path = os.path.join(task_dir(out_dir, task), CELL_FILE)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        m, land = rec.get("metrics", {}), rec.get("landscape", {})
        row = {"method": rec["method"], "cell": rec["cell"], "draw": rec["draw"],
               "dim": rec["dim"], "n_needles_true": rec["n_needles"],
               "basin_width": rec["basin_width"]}
        for key in (*METRICS, *EXTRA):
            row[key] = m.get(key)
        row.update({
            "separation_achieved": land.get("separation_achieved"),
            "prominence_target_met": land.get("prominence_target_met"),
            "n_prominence_resolved": land.get("n_prominence_resolved"),
            "basin_plain_radius": land.get("basin_plain_radius"),
            "config_source": rec.get("config_source"),
        })
        rows.append(row)
    return rows


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


def collect_curves(out_dir: str, column: str) -> list[dict]:
    """``column`` vs measured points for every finished cell (NaN rows dropped)."""
    import pandas as pd

    curves = []
    for task in read_tasks(out_dir):
        target = task_dir(out_dir, task)
        mot = os.path.join(target, "metrics_over_time.csv")
        if not (os.path.isfile(os.path.join(target, CELL_FILE)) and os.path.isfile(mot)):
            continue
        try:
            df = pd.read_csv(mot)
        except (OSError, ValueError):
            continue
        if column not in df.columns:
            continue
        df = df[["n_points", column]].dropna()
        if len(df) < 2:
            continue
        curves.append({"method": task["method"], "dim": task["dim"],
                       "x": df["n_points"].to_numpy(float),
                       "y": df[column].to_numpy(float)})
    return curves


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
            if not group:
                continue
            # Common grid = the points every curve in the group reaches, so a short
            # curve is never extended with an invented flat tail.
            x_max = min(c["x"][-1] for c in group)
            xs = np.unique(np.concatenate([c["x"] for c in group]))
            xs = xs[xs <= x_max]
            if len(xs) < 2:
                continue
            # Step interpolation: a metric holds its last measured value.
            Y = np.vstack([c["y"][np.clip(np.searchsorted(c["x"], xs, side="right") - 1,
                                          0, len(c["y"]) - 1)] for c in group])
            ax.fill_between(xs, np.quantile(Y, 0.25, axis=0), np.quantile(Y, 0.75, axis=0),
                            color=colors[m], alpha=0.14, lw=0)
            ax.plot(xs, Y.mean(axis=0), color=colors[m], lw=2, label=f"{m} ({len(group)})")
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

    rows = collect(out_dir)
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
    _curves_by_dim(collect_curves(out_dir, "dist_to_needles"), methods,
                   "dist_to_needles (lower is better)",
                   os.path.join(sdir, "dist_over_time.png"),
                   "dist_to_needles over the budget")
    _curves_by_dim(collect_curves(out_dir, "simple_regret"), methods,
                   "simple regret (lower is better)",
                   os.path.join(sdir, "regret_over_time.png"),
                   "Simple regret over the budget")

    n_expected = manifest["n_tasks"]
    short = [r for r in rows if r.get("budget_hit") is False]
    lines = [
        "# Method sweep — ZoMBI-Hop vs. black-box baselines on needle landscapes",
        "",
        f"{len(rows)} of {n_expected} cell(s) finished — {len(methods)} method(s) x "
        f"{manifest['n_configurations']} landscape configuration(s) x "
        f"{manifest['n_draws']} draw(s).",
        "",
        f"Budget: **{manifest['budget']} measured points per cell** in batches of "
        f"{manifest['batch_size']}, identical for every method and dimension. Noise: "
        f"input {manifest['input_noise']:g}, output {manifest['output_noise_frac']:g} x |y|. "
        "Landscape: bumps-only `CartesianEnsemble` on the unit cube — *n* negated-Ackley "
        "needles of sharpness *b* peaking at 1.0 on a plain at 0.75 "
        "(`benchmarks/sweeps/needles.py`). Every method saw the identical landscape "
        "and noise stream for a given cell.",
        "",
        f"Methods that do not declare needles are scored on the needles the "
        f"`{manifest['extractor']['name']}` extractor finds in their samples; "
        "`dist_to_needles_extracted` applies that same extractor to every method "
        "(`benchmarks/methods/extract.py`).",
        "",
        "## Headline (all landscapes)",
        "",
        "| method | cells | dist_to_needles | dist (common extractor) | optima visited "
        "| simple regret |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(agg_m, key=lambda r: methods.index(r["method"])):
        lines.append(f"| {r['method']} | {r['n_cells']} | {_fmt(r, 'dist_to_needles')} | "
                     f"{_fmt(r, 'dist_to_needles_extracted')} | "
                     f"{_fmt(r, 'frac_optima_visited')} | {_fmt(r, 'simple_regret')} |")
    lines += ["", f"Means with {int(ci * 100)}% bootstrap intervals over cells. Lower "
              "is better except *optima visited*.", "",
              f"## Paired against `{ref}` (all landscapes)", "",
              "Difference = method − reference on the same landscape (negative is "
              "better for the distances and regret). Win rate = share of landscapes "
              "where the method did strictly better.", "",
              "| method | pairs | Δ dist_to_needles | win rate | Δ dist (common extractor) "
              "| win rate | Δ simple regret | win rate |",
              "|---|---|---|---|---|---|---|---|"]
    for r in pair:
        if r["dim"] != "all":
            continue
        lines.append(
            f"| {r['method']} | {r['dist_to_needles_diff_n']} | "
            f"{_fmt(r, 'dist_to_needles_diff')} | {r['dist_to_needles_win_rate']} | "
            f"{_fmt(r, 'dist_to_needles_extracted_diff')} | "
            f"{r['dist_to_needles_extracted_win_rate']} | "
            f"{_fmt(r, 'simple_regret_diff')} | {r['simple_regret_win_rate']} |")
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
        "![dist over time](dist_over_time.png)",
        "![regret over time](regret_over_time.png)",
        "",
        f"The over-time curves for extractor-scored methods have a point every "
        f"{manifest['trace_every']} batches (each is a GP fit); ZoMBI-Hop's own "
        "needles are traced every batch.",
        "",
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
