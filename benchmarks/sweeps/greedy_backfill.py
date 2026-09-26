"""
benchmarks/sweeps/greedy_backfill.py
====================================
Back-calculate ``greedy_dist`` on a sweep that finished before the metric existed,
and draw it the way that sweep's own summary draws ``dist_to_needles``.

    python -m benchmarks.sweeps.greedy_backfill --out benchmarks/sweeps/runs/first

``greedy_dist`` (``eval_metrics.metric_greedy_dist``) is the mean over true optima of
the distance to the nearest point the run MEASURED. That makes it recoverable after
the fact from artifacts every cell already wrote — ``points.csv`` (every sample) and
``ensemble_config.json`` (``pinned_optima``, the landscape's true optima) — with no
model to refit and no run to repeat. Nothing else about the metric depends on the
optimiser's internal state, which is exactly why it can be added retroactively where
``dist_to_needles`` could not: that one scores the needle set as it stood at each
line, and old cells stored only its *value*, not the set.

Which runs this is for
----------------------
The pre-2026-09 single-method **simplex** sweep (schema 1), whose cells live at
``runs/<cell>/draw<NNN>/`` and whose ``summary/`` was written by the version of
``summarize.py`` at commit 285424f. Current multi-method **cube** campaigns
(``runs/<method>/<cell>/draw<NNN>/``) need nothing from this module: their runner
writes ``greedy_dist`` into ``metrics.json`` and ``metrics_over_time.csv`` as the
cell runs, and ``python -m benchmarks.sweeps summarize`` plots it per method. This
script refuses that layout rather than drawing a second, differently-made version of
a figure the sweep already has.

What it writes — and only writes
--------------------------------
Every output is new. No existing plot, CSV or line of ``index.md`` is replaced::

    <cell>/draw<NNN>/greedy_over_time.csv   n_points, greedy_dist (per line measured)
    summary/greedy_cells.csv                one row per cell: final greedy_dist
    summary/greedy_grid.csv                 per (dim, n, b): mean + bootstrap CI
    summary/greedy_dist_heatmap.png         panel per dim, n x b tile  <- the headline
    summary/greedy_over_time_by_axis.png    main effects, one panel per swept axis
    summary/greedy_over_time_grid.png       faceted dim x n, sharpness inside a panel
    summary/greedy_over_time_all.png        every cell trajectory, nothing averaged
    summary/index.md                        ONE appended section (re-runs replace it)

The three trajectory figures and the heatmap deliberately mirror
``dist_to_needles``'s — same panel layout, same viridis_r ramp, same x-axis (measured
compositions), same IQR bands, same truncate-to-the-shortest-curve rule — so the two
metrics can be read side by side without correcting for the drawing.

Trajectory resolution
---------------------
``dist_to_needles`` was traced once per measured line, so this is too: the running
distance is sampled at every ``points_per_line``-th sample (24 by default, read from
the manifest) plus the final sample. Because the minima are running, the curve is
non-increasing and sampling it cannot hide a rise — there are none to hide.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from ._paths import ensure_paths

ensure_paths()

from eval_metrics import metric_greedy_dist  # noqa: E402

CELL_FILE = "sweep_cell.json"
CURVE_FILE = "greedy_over_time.csv"
#: Marks the section this script owns in ``index.md``; a re-run replaces exactly the
#: text between it and the end of file, and touches nothing above it.
INDEX_MARKER = "<!-- greedy_dist: appended by benchmarks/sweeps/greedy_backfill.py -->"

METRIC_LABEL = "greedy dist"
DEFAULT_CI = 0.95
DEFAULT_N_BOOT = 2000
TRAJ_CMAP = "viridis"


# ─── Reading a legacy cell ───────────────────────────────────────────────────────

def _coord_columns(columns) -> list[str]:
    """The sample-coordinate columns of a ``points.csv``, in axis order.

    Two conventions are in the runs: ``x0..x{d-1}`` (every dimension of the current
    runner, and dims > 3 of the legacy sweep) and the named simplex components
    ``FA, MA, Br`` (legacy dim-3 cells, which logged the composition by name).
    """
    xs = [c for c in columns if len(c) > 1 and c[0] == "x" and c[1:].isdigit()]
    if xs:
        return sorted(xs, key=lambda c: int(c[1:]))
    named = [c for c in ("FA", "MA", "Br") if c in columns]
    return named


def load_cell(cell_dir: str) -> dict | None:
    """``{record, X, optima}`` for one finished cell, or None if it is unusable.

    Unusable means a missing completion marker, landscape or sample file — the same
    rule the summaries use. Nothing is imputed and nothing is guessed.
    """
    import pandas as pd

    rec_path = os.path.join(cell_dir, CELL_FILE)
    cfg_path = os.path.join(cell_dir, "ensemble_config.json")
    pts_path = os.path.join(cell_dir, "points.csv")
    if not all(os.path.isfile(p) for p in (rec_path, cfg_path, pts_path)):
        return None
    try:
        with open(rec_path) as f:
            rec = json.load(f)
        with open(cfg_path) as f:
            cfg = json.load(f)
        df = pd.read_csv(pts_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    optima = np.asarray(cfg.get("pinned_optima") or [], dtype=float)
    cols = _coord_columns(df.columns)
    if not len(optima) or not cols or not len(df):
        return None
    X = df[cols].to_numpy(dtype=float)
    if X.shape[1] != optima.shape[1]:
        raise SystemExit(
            f"{cell_dir}: {X.shape[1]} sample coordinate(s) {cols} but the landscape's "
            f"optima are {optima.shape[1]}-dimensional — the wrong columns were read")
    return {"record": rec, "X": X, "optima": optima}


def greedy_curve(X: np.ndarray, optima: np.ndarray, stride: int) -> tuple[np.ndarray,
                                                                         np.ndarray]:
    """``(n_points, greedy_dist)`` after every ``stride`` samples, plus the last one.

    The whole trajectory comes out of one running minimum: ``D`` holds each optimum's
    distance to each sample, ``np.minimum.accumulate`` turns that into "closest
    sample among the first k", and the mean over optima is the metric at k. One pass,
    and the final value is identical to ``metric_greedy_dist`` on the whole set
    (asserted by ``--check``).
    """
    D = np.linalg.norm(X[:, None, :] - optima[None, :, :], axis=2)   # (n_samples, k)
    running = np.minimum.accumulate(D, axis=0).mean(axis=1)
    n = len(X)
    idx = np.unique(np.concatenate([
        np.arange(max(1, int(stride)) - 1, n, max(1, int(stride))), [n - 1]]))
    return (idx + 1).astype(float), running[idx]


# ─── Collection ──────────────────────────────────────────────────────────────────

def _is_method_major(out_dir: str) -> bool:
    """True for the current layout, ``runs/<method>/<cell>/draw<NNN>``."""
    return bool(glob.glob(os.path.join(out_dir, "runs", "*", "*", "draw*", CELL_FILE)))


def cell_dirs(out_dir: str) -> list[str]:
    """Every legacy cell directory, ``runs/<cell>/draw<NNN>`` (sorted, bounded glob)."""
    return sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(out_dir, "runs", "*", "draw*", CELL_FILE)))


def backfill(out_dir: str, *, stride: int, check: bool, verbose: bool = True) -> list[dict]:
    """Compute and write every cell's curve; return one row per cell.

    A row carries the grid coordinates, the final ``greedy_dist``, and — for the
    figures — the trajectory itself, which stays out of ``greedy_cells.csv`` for the
    same reason the old summary kept ``dist_to_needles``'s out: a cell is one line of
    a table, its trajectory is ~125 numbers.
    """
    import pandas as pd

    rows = []
    for cdir in cell_dirs(out_dir):
        cell = load_cell(cdir)
        if cell is None:
            if verbose:
                print(f"  [skip] {os.path.relpath(cdir, out_dir)} — incomplete")
            continue
        rec, X, optima = cell["record"], cell["X"], cell["optima"]
        x, y = greedy_curve(X, optima, stride)
        final = float(y[-1])
        if check:
            direct = metric_greedy_dist(X, list(optima))
            if not np.isclose(final, direct, atol=1e-9):
                raise SystemExit(f"{cdir}: running curve ends at {final} but "
                                 f"metric_greedy_dist says {direct}")
        pd.DataFrame({"n_points": x.astype(int), "greedy_dist": np.round(y, 8)}).to_csv(
            os.path.join(cdir, CURVE_FILE), index=False)
        m = rec.get("metrics", {})
        rows.append({
            "cell": rec["cell"], "draw": rec["draw"], "dim": rec["dim"],
            "n_needles_true": rec["n_needles"], "basin_width": rec["basin_width"],
            "greedy_dist": round(final, 6),
            # Carried so a reader can see the two metrics on one row: greedy_dist
            # scores where the run MEASURED, dist_to_needles what it DECLARED.
            "dist_to_needles": m.get("dist_to_needles"),
            "n_needles_declared": m.get("n_needles"),
            "n_samples": int(len(X)),
            "n_optima": int(len(optima)),
            "budget_hit": rec.get("budget", {}).get("budget_hit"),
            "x": x, "y": y,
        })
        if verbose:
            print(f"  [cell] {rec['cell']} draw{rec['draw']:03d}  greedy_dist="
                  f"{final:.4f}  (dist_to_needles {m.get('dist_to_needles')})")
    return rows


def _boot_ci(values: np.ndarray, ci: float, n_boot: int, rng) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling draws (as the old summary does)."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    means = values[idx].mean(axis=1)
    lo = (1.0 - ci) / 2.0
    return (float(np.quantile(means, lo)), float(np.quantile(means, 1.0 - lo)))


def aggregate(rows: list[dict], *, ci: float, n_boot: int, seed: int) -> list[dict]:
    """Collapse draws into one record per ``(dim, n, b)`` configuration."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["dim"], r["n_needles_true"], r["basin_width"]), []).append(r)
    out = []
    for (dim, n, b), group in sorted(groups.items()):
        vals = np.asarray([g["greedy_dist"] for g in group], dtype=float)
        lo, hi = _boot_ci(vals, ci, n_boot, rng)
        out.append({
            "dim": dim, "n_needles_true": n, "basin_width": b, "n_draws": len(group),
            "greedy_dist_mean": round(float(vals.mean()), 6),
            "greedy_dist_median": round(float(np.median(vals)), 6),
            "greedy_dist_lo": None if np.isnan(lo) else round(lo, 6),
            "greedy_dist_hi": None if np.isnan(hi) else round(hi, 6),
            "greedy_dist_n": int(len(vals)),
        })
    return out


# ─── Figures ─────────────────────────────────────────────────────────────────────
#
# Drawn to match the dist_to_needles figures of the same summary directory, down to
# the reversed viridis ramp and the "measured compositions" x-axis, because the whole
# point of back-calculating is to read the two metrics against each other.

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _level_colors(n: int):
    plt = _plt()
    cmap = plt.get_cmap(TRAJ_CMAP)
    # Stop short of the pale top of viridis, which vanishes against white at 1.5px.
    return [cmap(v) for v in np.linspace(0.08, 0.88, max(n, 1))]


def _heatmap(agg: list[dict], manifest: dict, path: str) -> None:
    """One panel per dimension: needle count (rows) x sharpness (columns), shared scale."""
    plt = _plt()
    dims = sorted({r["dim"] for r in agg})
    counts = sorted({r["n_needles_true"] for r in agg})
    widths = sorted({r["basin_width"] for r in agg})
    if not (dims and counts and widths):
        return
    grids = {}
    for dim in dims:
        M = np.full((len(counts), len(widths)), np.nan)
        for r in agg:
            if r["dim"] == dim and r["greedy_dist_mean"] is not None:
                M[counts.index(r["n_needles_true"]),
                  widths.index(r["basin_width"])] = r["greedy_dist_mean"]
        grids[dim] = M
    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    if not len(finite):
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    cmap = plt.get_cmap("viridis_r").copy()   # reversed: bright = good, as elsewhere
    cmap.set_bad("#e8e8e8")

    fig, axes = plt.subplots(1, len(dims), figsize=(3.4 * len(dims) + 1.6, 3.9),
                             squeeze=False)
    im = None
    for ax, dim in zip(axes[0], dims):
        im = ax.imshow(np.ma.masked_invalid(grids[dim]), cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", aspect="auto")
        ax.set_xticks(range(len(widths)), [f"{w:g}" for w in widths])
        ax.set_yticks(range(len(counts)), [str(c) for c in counts])
        ax.set_xlabel("basin sharpness $b$")
        src = manifest.get("hparams", {}).get(str(dim), {})
        ax.set_title(f"dim {dim}{' *' if src.get('is_stand_in') else ''}", fontsize=11)
        for i in range(len(counts)):
            for j in range(len(widths)):
                v = grids[dim][i, j]
                if np.isfinite(v):
                    rel = 1.0 - (v - vmin) / (vmax - vmin or 1.0)
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                            color="white" if rel < 0.55 else "black")
    axes[0][0].set_ylabel("number of needles $n$")
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02,
                 label=f"{METRIC_LABEL} (composition L2)")
    stand_in = any(manifest.get("hparams", {}).get(str(d), {}).get("is_stand_in")
                   for d in dims)
    fig.suptitle(f"{METRIC_LABEL} — mean of {manifest.get('n_draws', '?')} draw(s), "
                 "lower is better"
                 + ("   (* hyperparameters are a stand-in for this dim)" if stand_in
                    else ""), fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pool(group: list[dict]):
    """``(x, mean, q25, q75)`` over a group of trajectories, truncated to the shortest.

    Truncating rather than forward-filling: an invented flat tail reads as "converged
    and held" when it means "this cell stopped here".
    """
    k = min(len(g["y"]) for g in group)
    Y = np.vstack([g["y"][:k] for g in group])
    return (group[0]["x"][:k], Y.mean(axis=0),
            np.quantile(Y, 0.25, axis=0), np.quantile(Y, 0.75, axis=0))


def _traj_by_axis(rows: list[dict], path: str) -> None:
    """One panel per swept axis, the other two marginalised out — the main effects."""
    plt = _plt()
    specs = [("dim", "dimension $d$"), ("n_needles_true", "needles $n$"),
             ("basin_width", "sharpness $b$")]
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.8), sharey=True, squeeze=False)
    for ax, (key, label) in zip(axs[0], specs):
        levels = sorted({r[key] for r in rows})
        for colour, lv in zip(_level_colors(len(levels)), levels):
            group = [r for r in rows if r[key] == lv]
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
        ax.legend(title=label, fontsize=8, title_fontsize=8.5, frameon=False,
                  loc="upper right", ncol=2)
    axs[0][0].set_ylabel(f"{METRIC_LABEL}  (lower is better)")
    fig.suptitle(f"{METRIC_LABEL} over the budget, by axis — mean over every cell in "
                 "the slice, band is the IQR", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _traj_grid(rows: list[dict], manifest: dict, path: str) -> None:
    """A facet per (dim, needle count) with one curve per sharpness — interactions kept."""
    plt = _plt()
    dims = sorted({r["dim"] for r in rows})
    counts = sorted({r["n_needles_true"] for r in rows})
    widths = sorted({r["basin_width"] for r in rows})
    if not (dims and counts and widths):
        return
    colours = _level_colors(len(widths))
    fig, axs = plt.subplots(len(dims), len(counts),
                            figsize=(3.0 * len(counts), 2.5 * len(dims)),
                            sharex=True, sharey=True, squeeze=False)
    for i, dim in enumerate(dims):
        for j, n in enumerate(counts):
            ax = axs[i][j]
            for colour, b in zip(colours, widths):
                group = [r for r in rows if r["dim"] == dim
                         and r["n_needles_true"] == n and r["basin_width"] == b]
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
    handles, labels = axs[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="sharpness $b$", frameon=False,
                   loc="upper center", ncol=len(widths), fontsize=9, title_fontsize=9.5,
                   bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"{METRIC_LABEL} over the budget — mean of "
                 f"{manifest.get('n_draws', '?')} draw(s) per curve, shared axes "
                 "(lower is better)", fontsize=11, y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _traj_all(rows: list[dict], path: str) -> None:
    """Every trajectory, nothing averaged, coloured by dimension with means over the top."""
    plt = _plt()
    dims = sorted({r["dim"] for r in rows})
    colours = dict(zip(dims, _level_colors(len(dims))))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for r in rows:
        ax.plot(r["x"], r["y"], color=colours[r["dim"]], lw=0.5, alpha=0.16, zorder=1,
                solid_capstyle="round")
    for dim in dims:
        group = [r for r in rows if r["dim"] == dim]
        x, mu, _, _ = _pool(group)
        ax.plot(x, mu, color="white", lw=4.0, zorder=2, solid_capstyle="round")
        ax.plot(x, mu, color=colours[dim], lw=2.4, zorder=3, solid_capstyle="round",
                label=f"dim {dim}  (n={len(group)})")
    ax.set_xlabel("measured compositions")
    ax.set_ylabel(f"{METRIC_LABEL}  (lower is better)")
    ax.grid(alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(title="mean per dimension", frameon=False, fontsize=9, title_fontsize=9.5,
              loc="upper right")
    ax.set_title(f"All {len(rows)} cell trajectories overlaid — one faint line per cell, "
                 "bold line the per-dimension mean", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── index.md ────────────────────────────────────────────────────────────────────

def _index_section(rows: list[dict], agg: list[dict], ci: float, stride: int) -> str:
    """The section appended to ``index.md`` — describes the metric and its figures."""
    dims = sorted({r["dim"] for r in rows})
    by_dim = {d: np.mean([r["greedy_dist"] for r in rows if r["dim"] == d]) for d in dims}
    lines = [
        INDEX_MARKER,
        "",
        "## Greedy dist (back-calculated)",
        "",
        f"`greedy_dist` for all {len(rows)} cell(s), computed after the fact from each "
        "cell's `points.csv` and `ensemble_config.json` "
        "(`python -m benchmarks.sweeps.greedy_backfill`). **Nothing above this line "
        "changed** — this is an addition to the summary, not a re-scoring of it.",
        "",
        "For each true optimum, the distance to the nearest point the run *measured*; "
        "averaged over the optima. Lower is better, and 0 would mean every optimum was "
        "sampled exactly.",
        "",
        "| | `dist_to_needles` | `greedy_dist` |",
        "|---|---|---|",
        "| scores | the needles the run **declared** | the samples the run **took** |",
        "| pairing | one-to-one (optimal assignment) | greedy — two optima may share "
        "a sample |",
        "| missing term | `UNMATCHED_PENALTY` = 0.5 per unmatched needle or optimum | "
        "none: with ≥1 sample every optimum has a nearest one |",
        "| over a budget | can rise (a badly placed needle costs) | can only fall "
        "(samples accumulate) |",
        "",
        "So the two answer different questions. `greedy_dist` asks whether the search "
        "ever *went* to each optimum; `dist_to_needles` asks whether it *reported* "
        "them. A cell can be good at the first and bad at the second (it measured the "
        "needle and declared nothing), and the pair localises which half of the "
        "pipeline a landscape breaks. The greedy pairing is deliberate: a sample is "
        "not a claim, so two optima sharing one nearest measurement double-counts "
        "nothing.",
        "",
        "Mean over all cells, by dimension: "
        + ", ".join(f"dim {d} **{by_dim[d]:.3f}**" for d in dims)
        + f". Per-configuration means with {int(ci * 100)}% bootstrap intervals over "
          "draws are in `greedy_grid.csv`; every cell's final value, next to its "
          "`dist_to_needles`, is in `greedy_cells.csv`.",
        "",
        "![greedy dist](greedy_dist_heatmap.png)",
        "",
        "### Trajectories",
        "",
        f"Sampled every {stride} measured compositions (one measured line, the same "
        "resolution `dist_to_needles` was traced at) from each cell's running minima, "
        "written per cell to `greedy_over_time.csv`. Same panel layout, colour ramp and "
        "x-axis as the `dist_to_needles` trajectories above, so the two can be read "
        "against each other directly.",
        "",
        "![greedy dist over time by axis](greedy_over_time_by_axis.png)",
        "![greedy dist over time faceted](greedy_over_time_grid.png)",
        "![all greedy dist trajectories](greedy_over_time_all.png)",
        "",
        "A curve that flattens means the search stopped reaching ground it had not "
        "already covered — it cannot mean a declaration went wrong, which is the one "
        "reading a flat `dist_to_needles` tail leaves open.",
        "",
    ]
    return "\n".join(lines)


def _append_index(path: str, section: str) -> None:
    """Append the section, replacing only a previous run's copy of it."""
    old = ""
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            old = f.read()
        old = old.split(INDEX_MARKER)[0].rstrip("\n") + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{old}\n{section}")


# ─── Entry point ─────────────────────────────────────────────────────────────────

def run(out_dir: str, *, stride: int | None = None, ci: float = DEFAULT_CI,
        n_boot: int = DEFAULT_N_BOOT, seed: int = 0, check: bool = True,
        verbose: bool = True) -> None:
    import pandas as pd

    out_dir = os.path.abspath(out_dir)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"no manifest.json in {out_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if _is_method_major(out_dir):
        raise SystemExit(
            f"{out_dir} is a multi-method campaign (runs/<method>/<cell>/draw...). Its "
            "cells score greedy_dist as they run, so use `python -m benchmarks.sweeps "
            "summarize --out ...` instead of back-calculating it here.")
    if stride is None:
        stride = int(manifest.get("points_per_line") or 24)

    rows = backfill(out_dir, stride=stride, check=check, verbose=verbose)
    if not rows:
        raise SystemExit(f"no finished cells with samples under {out_dir}/runs")
    agg = aggregate(rows, ci=ci, n_boot=n_boot, seed=seed)

    sdir = os.path.join(out_dir, "summary")
    os.makedirs(sdir, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("x", "y")} for r in rows]) \
        .to_csv(os.path.join(sdir, "greedy_cells.csv"), index=False)
    pd.DataFrame(agg).to_csv(os.path.join(sdir, "greedy_grid.csv"), index=False)

    _heatmap(agg, manifest, os.path.join(sdir, "greedy_dist_heatmap.png"))
    _traj_by_axis(rows, os.path.join(sdir, "greedy_over_time_by_axis.png"))
    _traj_grid(rows, manifest, os.path.join(sdir, "greedy_over_time_grid.png"))
    _traj_all(rows, os.path.join(sdir, "greedy_over_time_all.png"))
    _append_index(os.path.join(sdir, "index.md"),
                  _index_section(rows, agg, ci, stride))
    print(f"  greedy_dist -> {sdir}  ({len(rows)} cell(s), {len(agg)} configuration(s))")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks.sweeps.greedy_backfill",
        description="back-calculate greedy_dist on a legacy sweep and plot it "
                    "alongside its dist_to_needles figures")
    ap.add_argument("--out", required=True, metavar="DIR", help="campaign directory")
    ap.add_argument("--stride", type=int, default=None,
                    help="trajectory resolution in samples (default: the manifest's "
                         "points_per_line)")
    ap.add_argument("--ci", type=float, default=DEFAULT_CI)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true",
                    help="skip verifying each curve's last point against "
                         "eval_metrics.metric_greedy_dist")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(args.out, stride=args.stride, ci=args.ci, n_boot=args.n_boot, seed=args.seed,
        check=not args.no_check, verbose=not args.quiet)


if __name__ == "__main__":
    main()
