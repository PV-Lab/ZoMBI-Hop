"""Figures and tables for a finished suite.

Five figures, no more. Each one answers a question somebody actually asked:

1. ``reached_ratio(t)``      -- are we finding optima faster than random? (Aleks)
2. ``peak_ratio`` / ``precision`` / ``dist_to_needles`` bars at the end of budget
3. ``peak_ratio`` vs ``|S|`` -- the matched-declaration comparison, so ZoMBI-Hop's
   7 needles are not scored against a baseline's 24 guesses
4. ``needles_declared(t)``   -- the structural ceiling on ZoMBI-Hop's needle count
5. ``peak_ratio`` vs ``n_optima`` -- the needle-count hypothesis (s2 only)

Everything is drawn from ``aggregate.csv`` + ``curves.json``, so a report can be
regenerated without re-running anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

_ORDER = ["random", "gp_qucb", "gp_qlogei", "gp_ts", "zombihop", "zombihop_nc5",
          "hebo", "turbo", "rf_bo", "saasbo", "robot"]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load(suite_dir: str) -> tuple[list[dict], dict]:
    with open(os.path.join(suite_dir, "aggregate.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("objective", "optimizer", "error", "declared_source"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = np.nan
    curves = {}
    cpath = os.path.join(suite_dir, "curves.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as fh:
            curves = json.load(fh)
    return rows, curves


def _opts(rows) -> list[str]:
    present = {r["optimizer"] for r in rows}
    return ([o for o in _ORDER if o in present]
            + sorted(present - set(_ORDER)))


def _colors(opts):
    plt = _mpl()
    cmap = plt.get_cmap("tab10")
    return {o: ("0.45" if o == "random" else cmap(i % 10))
            for i, o in enumerate(opts)}


def _mean_std(rows, opt, key):
    v = [r[key] for r in rows if r["optimizer"] == opt
         and isinstance(r.get(key), float) and np.isfinite(r[key])]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def fig_reached(rows, curves, suite_dir, objectives) -> str:
    plt = _mpl()
    opts = _opts(rows)
    col = _colors(opts)
    fig, axes = plt.subplots(1, len(objectives), figsize=(5.2 * len(objectives), 4.2),
                             squeeze=False)
    for ax, obj in zip(axes[0], objectives):
        for o in opts:
            cs = [c for c in curves.values()
                  if c.get("objective") == obj and c.get("optimizer") == o
                  and c.get("reached_curve_ratio")]
            if not cs:
                continue
            t = np.asarray(cs[0]["reached_curve_t"], dtype=float)
            Y = np.vstack([np.asarray(c["reached_curve_ratio"], dtype=float)
                           for c in cs if len(c["reached_curve_ratio"]) == t.size])
            m, s = Y.mean(0), Y.std(0)
            ax.plot(t, m, label=f"{o} (n={Y.shape[0]})", color=col[o],
                    lw=2.2 if o.startswith("zombihop") else 1.6,
                    ls="--" if o == "random" else "-")
            ax.fill_between(t, m - s, m + s, color=col[o], alpha=0.13, lw=0)
        ax.set_title(obj, fontsize=10)
        ax.set_xlabel("samples")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("fraction of true optima reached")
    axes[0][-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Sample efficiency: optima reached (near AND high) vs budget",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(suite_dir, "fig1_reached_ratio.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_endpoint_bars(rows, suite_dir, objectives) -> str:
    plt = _mpl()
    opts = _opts(rows)
    col = _colors(opts)
    keys = [("peak_ratio", "peak ratio (recall)"), ("precision", "precision"),
            ("dist_to_needles", "dist to needles (lower better)")]
    fig, axes = plt.subplots(len(objectives), 3,
                             figsize=(13, 3.1 * len(objectives)), squeeze=False)
    for i, obj in enumerate(objectives):
        sub = [r for r in rows if r["objective"] == obj]
        for j, (key, label) in enumerate(keys):
            ax = axes[i][j]
            m = [_mean_std(sub, o, key) for o in opts]
            ax.bar(range(len(opts)), [x[0] for x in m],
                   yerr=[x[1] for x in m], capsize=3,
                   color=[col[o] for o in opts])
            ax.set_xticks(range(len(opts)))
            ax.set_xticklabels(opts, rotation=40, ha="right", fontsize=7)
            ax.grid(alpha=0.25, axis="y")
            if i == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(obj, fontsize=9)
    fig.suptitle("End of budget, mean +/- std over seeds", fontsize=11)
    fig.tight_layout()
    path = os.path.join(suite_dir, "fig2_endpoint.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_matched_declarations(rows, curves, suite_dir, objectives) -> str:
    """peak_ratio against the number of optima a method declares.

    The figure that keeps the headline honest. ZoMBI-Hop volunteers only the
    needles it is confident in; a post-hoc set can declare as many as it likes.
    Scoring both at |S| = n_true silently hands the baselines more guesses, so the
    fair reading is where the curves sit at the SAME |S| -- marked for ZoMBI-Hop.
    """
    plt = _mpl()
    opts = _opts(rows)
    col = _colors(opts)
    fig, axes = plt.subplots(1, len(objectives), figsize=(5.2 * len(objectives), 4.2),
                             squeeze=False)
    for ax, obj in zip(axes[0], objectives):
        for o in opts:
            cs = [c for c in curves.values()
                  if c.get("objective") == obj and c.get("optimizer") == o
                  and c.get("pr_curve_k")]
            if not cs:
                continue
            n = min(len(c["pr_curve_k"]) for c in cs)
            k = np.asarray(cs[0]["pr_curve_k"][:n], dtype=float)
            Y = np.vstack([np.asarray(c["pr_curve_peak_ratio"][:n], dtype=float)
                           for c in cs])
            ax.plot(k, Y.mean(0), color=col[o], label=o,
                    lw=2.2 if o.startswith("zombihop") else 1.6,
                    ls="--" if o == "random" else "-")
        for o in [x for x in opts if x.startswith("zombihop")]:
            nd, _, cnt = _mean_std([r for r in rows if r["objective"] == obj],
                                   o, "n_declared")
            pr, _, _ = _mean_std([r for r in rows if r["objective"] == obj],
                                 o, "peak_ratio")
            if cnt and np.isfinite(nd):
                ax.scatter([nd], [pr], marker="*", s=170, zorder=5,
                           color=col[o], edgecolor="k", linewidth=0.6)
        ax.set_title(obj, fontsize=10)
        ax.set_xlabel("|S| = number of optima declared")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("peak ratio (recall)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Matched declarations: recall vs how many optima were declared "
                 "(star = ZoMBI-Hop's own count)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(suite_dir, "fig3_matched_declarations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_needles(rows, curves, suite_dir, objectives) -> str | None:
    plt = _mpl()
    zh = [o for o in _opts(rows) if o.startswith("zombihop")]
    if not zh:
        return None
    col = _colors(_opts(rows))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    any_line = False
    for obj in objectives:
        for o in zh:
            cs = [c for c in curves.values()
                  if c.get("objective") == obj and c.get("optimizer") == o
                  and c.get("needles_curve_n")]
            if not cs:
                continue
            n = min(len(c["needles_curve_t"]) for c in cs)
            t = np.asarray(cs[0]["needles_curve_t"][:n], dtype=float)
            Y = np.vstack([np.asarray(c["needles_curve_n"][:n], dtype=float) for c in cs])
            ax.plot(t, Y.mean(0), label=f"{obj} / {o}", color=col[o],
                    ls="-" if o == "zombihop" else ":")
            any_line = True
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel("samples")
    ax.set_ylabel("needles declared")
    ax.set_title("ZoMBI-Hop's declaration budget\n"
                 "(an activation needs 4-6 lines before it may declare)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(suite_dir, "fig4_needles_declared.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_needle_count(rows, suite_dir) -> str | None:
    """peak_ratio vs n_optima -- the needle-count hypothesis. s2 only."""
    plt = _mpl()
    # x-axis is the ACHIEVED number of distinct optima, not the requested one. The
    # 2-simplex cannot hold 80 optima at 2r separation: asking for 80 at d=3 yields
    # 33 after merging (40 -> 26, 20 -> 18). Plotting the request would compress
    # three different landscapes onto one x value and imply a plateau that is really
    # a packing limit.
    byn = defaultdict(list)
    for r in rows:
        n = r.get("n_true_optima")
        if not isinstance(n, float) or not np.isfinite(n):
            return None
        byn[(r["optimizer"], int(n))].append(r.get("peak_ratio", np.nan))
    opts = _opts(rows)
    col = _colors(opts)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for o in opts:
        ns = sorted({n for (oo, n) in byn if oo == o})
        if not ns:
            continue
        m = [np.nanmean(byn[(o, n)]) for n in ns]
        s = [np.nanstd(byn[(o, n)]) for n in ns]
        ax.errorbar(ns, m, yerr=s, marker="o", capsize=3, color=col[o], label=o,
                    lw=2.2 if o.startswith("zombihop") else 1.6,
                    ls="--" if o == "random" else "-")
    ax.set_xlabel("number of distinct true optima (achieved, after merging at 2r)")
    ax.set_ylabel("peak ratio (recall)")
    ax.set_title("Does performance degrade as needles multiply?", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(suite_dir, "fig5_needle_count.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def budget_table(rows, objectives, opts) -> list[str]:
    """peak_ratio at each budget checkpoint -- where small-N columns matter."""
    ns = sorted({int(k.split("@")[1]) for r in rows for k in r
                 if k.startswith("peak_ratio@")})
    if not ns:
        return []
    out = ["## peak_ratio vs budget", ""]
    for obj in objectives:
        sub = [r for r in rows if r["objective"] == obj]
        out += [f"### {obj}", "",
                "| optimizer | " + " | ".join(f"N={n}" for n in ns) + " |",
                "|" + "---|" * (len(ns) + 1)]
        for o in opts:
            cells = []
            for n in ns:
                m, s, c = _mean_std(sub, o, f"peak_ratio@{n}")
                cells.append(f"{m:.3f} ± {s:.3f}" if c else "-")
            out.append(f"| {o} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def build(suite_dir: str) -> str:
    rows, curves = _load(suite_dir)
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        raise SystemExit(f"no successful runs in {suite_dir}")
    objectives = sorted({r["objective"] for r in rows})
    opts = _opts(rows)

    figs = [fig_reached(rows, curves, suite_dir, objectives),
            fig_endpoint_bars(rows, suite_dir, objectives),
            fig_matched_declarations(rows, curves, suite_dir, objectives),
            fig_needles(rows, curves, suite_dir, objectives),
            fig_needle_count(rows, suite_dir)]
    figs = [f for f in figs if f]

    md = ["# Report", "",
          f"Suite: `{os.path.basename(suite_dir)}`  ",
          f"{len(rows)} successful runs, {len(objectives)} objectives, "
          f"{len(opts)} optimizers.", "",
          "Figures:", ""]
    md += [f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in figs]
    md += ["", "Read `peak_ratio` next to `fig3`: a method that declares fewer "
           "optima is capped on recall for a structural reason, so the comparison "
           "that means something is at matched |S|.", ""]
    md += budget_table(rows, objectives, opts)

    path = os.path.join(suite_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("\n".join(f"wrote {f}" for f in figs + [path]))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build figures for a finished suite")
    ap.add_argument("suite_dir")
    args = ap.parse_args(argv)
    build(args.suite_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
