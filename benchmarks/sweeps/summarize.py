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
``recall`` is the headline. The deliverable is the *set* of optima (METHODS §1), so
the first thing to know about a landscape is what fraction of its needles were
found at all:

    recall            fraction of the landscape's true optima with a declared
                      needle inside ``MATCH_RADIUS`` — ``metric_pct_matched_comp``
                      over 100. Directly comparable across cells because the
                      placement guarantees the denominator is exactly n.
    dist_to_needles   the MOBO objective: mean distance from each true optimum to
                      the nearest declared needle, with a penalty for unmatched
                      ones. Lower is better. Sensitive to *how badly* a miss missed,
                      which recall is not.
    n_needles         how many needles were declared, matched or not. Read next to
                      recall it separates "found few" from "declared few".
    dup_fraction      fraction of samples landing within half the input noise of an
                      earlier one. Lower is better; it is the sweep's read on
                      whether a landscape drives the optimiser into re-measuring.

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
    "recall": ("recall (matched / true optima)", False),
    "dist_to_needles": ("dist_to_needles", True),
    "n_needles": ("needles declared", False),
    "dup_fraction": ("dup_fraction", True),
}


# ─── Collection ──────────────────────────────────────────────────────────────────

def _recall(trial_dir: str, n_true: int) -> float | None:
    """Fraction of the landscape's true optima that got a needle within MATCH_RADIUS.

    Recomputed from ``needles.csv`` and the cell's own optima rather than read off
    ``metrics.json``, which does not carry a matched-fraction. Uses the shared
    ``eval_metrics`` implementation so "matched" means here exactly what it means in
    ``pareto.py`` and the MOBO objectives.
    """
    import pandas as pd
    from eval_metrics import metric_pct_matched_comp

    needles_csv = os.path.join(trial_dir, "needles.csv")
    ens_cfg = os.path.join(trial_dir, "ensemble_config.json")
    if not (os.path.isfile(needles_csv) and os.path.isfile(ens_cfg)):
        return None
    with open(ens_cfg) as f:
        cfg = json.load(f)
    true_optima = [np.asarray(c, dtype=float) for c in cfg.get("pinned_optima", [])]
    if len(true_optima) != n_true:
        return None
    df = pd.read_csv(needles_csv)
    coord_cols = [c for c in df.columns if c.startswith("x")] or \
                 [c for c in ("FA", "MA", "Br") if c in df.columns]
    disc = (df[coord_cols].to_numpy(dtype=float) if len(df) and coord_cols
            else np.empty((0, len(true_optima[0]))))
    return float(metric_pct_matched_comp(disc, true_optima,
                                         dim=len(true_optima[0]))) / 100.0


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
        rows.append({
            "cell": rec["cell"], "draw": rec["draw"], "dim": rec["dim"],
            "n_needles": rec["n_needles"], "basin_width": rec["basin_width"],
            "recall": _recall(target, rec["n_needles"]),
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

    grids = {}
    for dim in dims:
        M = np.full((len(counts), len(widths)), np.nan)
        for r in agg:
            if r["dim"] != dim or r.get(f"{metric}_mean") is None:
                continue
            M[counts.index(r["n_needles_true"]), widths.index(r["basin_width"])] = \
                r[f"{metric}_mean"]
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
    """Recall against each swept axis, marginalised over the other two.

    The heatmaps show the interaction; this shows the main effects, which is what a
    one-line answer to "what is ZoMBI-Hop robust to" needs. Error bars are the
    bootstrap CI of the mean over every cell in that slice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_spec = [("n_needles_true", "number of needles $n$"),
                 ("basin_width", "basin sharpness $b$"),
                 ("dim", "dimension $d$")]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), squeeze=False)
    for ax, (key, label) in zip(axs[0], axes_spec):
        levels = sorted({r[key] for r in agg})
        xs, ys, los, his = [], [], [], []
        for lv in levels:
            vals = [r["recall_mean"] for r in agg
                    if r[key] == lv and r.get("recall_mean") is not None]
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
        ax.errorbar(range(len(xs)), ys, yerr=[los, his], marker="o", capsize=3,
                    color="#2b6cb0")
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
        ax.set_xlabel(label)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, axis="y")
    axs[0][0].set_ylabel("recall")
    fig.suptitle("Recall by axis, marginalised over the other two "
                 f"({manifest.get('n_draws', '?')} draw(s) per configuration)",
                 fontsize=11)
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
        "## Figures",
        "",
        "![recall](recall_heatmap.png)",
        "![recall by axis](recall_by_axis.png)",
        "![dist_to_needles](dist_to_needles_heatmap.png)",
        "![needles declared](n_needles_heatmap.png)",
        "![dup_fraction](dup_fraction_heatmap.png)",
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
