"""
pareto.py
=========
Collect every MOBO trial across ``runs/mobo_*/mobo_progress.json``, determine the
Pareto-optimal set of hyperparameter configurations, write it to ``pareto.json``,
and render a Pareto-front figure.

This replaces the old on-the-fly ``pareto`` flag that ``run_mobo.py`` used to
stamp into each ``mobo_progress.json`` (which was computed per-run and therefore
inconsistent). Pareto membership is a global property of all trials, so it is
determined here, after the fact, over the union of every run.

Objectives (all MINIMISED):
    dist_to_needles      – distance from discovered needles to the reference optima
    dup_fraction         – fraction of duplicated samples
    avg_time_per_iter_s  – average wall-clock seconds per ZoMBI iteration (the
                           current run_mobo objective). Older runs minimised total
                           ``runtime_s`` instead; this script reads whichever key
                           each trial recorded and labels the third axis to match
                           (it warns if a single collection mixes the two).

Each run's ``mobo_progress.json`` records only its own trials, so the union over
all runs never double-counts (a resumed run seeds the GP with prior history but
writes only its own new trials).  ``IGNORE_mobo_*`` directories are excluded by
the ``mobo_*`` glob.

Clicking a Pareto star opens that trial's landscape view (3D per-iteration frame,
or the 4D interactive rotatable point_cloud.html). Higher-dimensional trials have
no such landscape, so a click there is a no-op — the cross-subplot hover
highlighting still works for every trial regardless of dimension.

Usage
-----
  conda activate zombi-hop
  python optimize/pareto.py                 # crawl optimize/runs, write there
  python optimize/pareto.py <runs_dir>      # crawl a specific runs directory
  python optimize/pareto.py <run_dir>       # a single run dir: pools its config-
                                            #   matching siblings (shared history)
  python optimize/pareto.py <run_dir> --no-shared-history  # that one run only
  python optimize/pareto.py --out <dir>     # write pareto.json / .png elsewhere
  python optimize/pareto.py --no-interactive # save static PNG instead of live window
  python optimize/pareto.py --with-old       # include mobo_old_jackson (excluded by default)
  python optimize/pareto.py --show-numberline # show hyperparameter number-line figure
  python optimize/pareto.py --only mobo_00_01,mobo_00_02          # only these runs
  python optimize/pareto.py --only mobo_00_01/trial_3,mobo_00_02  # specific trials
  python optimize/pareto.py --only 4d        # all runs with _4d_ (mobo_4d_*, mobo_ensemble_4d_*, ...)
  python optimize/pareto.py --only 10d       # all runs with _10d_ (mobo_10d_*, mobo_ensemble_10d_*, ...)
  python optimize/pareto.py --only 4d,ensemble   # only the mobo_ensemble_4d_* runs (variant AND)
  python optimize/pareto.py --only 4d,-ensemble  # only the plain mobo_4d_* runs (exclude ensemble)
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import datetime

import subprocess
import platform

import numpy as np

import matplotlib
# Backend is set later: "Agg" for static PNG, system default for interactive.
import matplotlib.pyplot as plt

# The first two objectives are fixed; the third is a time metric whose key varies
# by run age: current run_mobo writes ``avg_time_per_iter_s``, older runs wrote
# total ``runtime_s``. ``_time_metric`` reads whichever a trial has (preferring the
# current one) and the third plot axis is labelled to match the data collected.
DIST_KEY  = "dist_to_needles"
DUP_KEY   = "dup_fraction"
TIME_KEYS = ("avg_time_per_iter_s", "runtime_s")

HPARAM_SPACE: dict[str, tuple] = {
    "nat_grad_step":               (0.001,  0.5,   "log"),
    "nat_grad_max_steps":          (10,     200,   "int"),
    "n_restarts":                  (20,     300,   "int"),
    "raw":                         (1,      300,   "int"),
    "ucb_beta":                    (0.05,   3.0,   "linear"),
    "max_zooms":                   (2,      10,    "int"),
    "max_iterations":              (2,      10,    "int"),
    "top_m_points":                (2,      8,     "int"),
    "n_consecutive_converged":     (1,      5,     "int"),
    "convergence_pi_threshold":    (1e-4,   0.05,  "log"),
    "input_noise_threshold_mult":  (0.5,    6.0,   "linear"),
    "output_noise_threshold_mult": (0.1,    2.0,   "linear"),
    "max_penalty_radius":          (0.2,    5.0,   "linear"),
    "needle_shrink_factor":        (0.55,   0.99,  "linear"),
    "needle_stop_noise_multiplier":(1.0,    8.0,   "linear"),
    "paring_spatial_halfnoise":    (0.1,    2.0,   "linear"),
    "paring_y_noise_multiplier":   (0.1,    5.0,   "linear"),
}
HPARAM_NAMES = list(HPARAM_SPACE.keys())


# ─── Run-config signatures (shared history) ──────────────────────────────────────
# Mirror of run_mobo.py's --share-history matching: a single run dir's trials are
# only comparable to a sibling's when the objective they were scored against is the
# same, which is pinned down by these run_config.json fields.

def _run_signature(cfg: dict) -> dict:
    """Config fields that must match for another run's stored trials to be pooled.

    These pin down the *objective* a trial was scored against — the same
    hyperparameters yield comparable (dist, dup, avg_time_per_iter) only when the
    dataset, dimension, per-trial time budget, search direction, and (for the
    ensemble objective) the landscape difficulty/averaging all agree. Fields absent
    in older configs come back as ``None`` and simply have to match ``None`` on both
    sides. Kept in sync with ``run_mobo._run_signature``.
    """
    return {
        "dataset": cfg.get("dataset") or cfg.get("oracle") or cfg.get("landscape"),
        "dim": int(cfg["dim"]) if cfg.get("dim") is not None else None,
        "time_limit_hours": cfg.get("time_limit_hours"),
        "maximize": bool(cfg.get("maximize", False)),
        "ensemble_optima_margin": cfg.get("ensemble_optima_margin"),
        "ensemble_repeats": cfg.get("ensemble_repeats"),
    }


def _signatures_match(a: dict, b: dict) -> bool:
    """Equality over signature fields, with float tolerance for the numeric ones."""
    def eq(x, y) -> bool:
        if isinstance(x, bool) or isinstance(y, bool):
            return x == y
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return abs(float(x) - float(y)) < 1e-9
        return x == y
    return all(eq(v, b.get(k)) for k, v in a.items())


def _load_run_signature(run_dir: str) -> dict | None:
    """Read ``run_dir/run_config.json`` and return its signature, or None if absent."""
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            return _run_signature(json.load(f))
    except Exception as exc:
        print(f"  [shared-history] {run_dir}: run_config.json unreadable ({exc}).")
        return None


# ─── Collection ────────────────────────────────────────────────────────────────

def _time_metric(m: dict) -> tuple[float, str] | None:
    """Return (value, key) for a trial's time objective, or None if it has none.

    Prefers the current ``avg_time_per_iter_s``; falls back to legacy ``runtime_s``.
    """
    for k in TIME_KEYS:
        if k in m:
            try:
                return float(m[k]), k
            except (TypeError, ValueError):
                return None
    return None


def _parse_only(
    only_str: str,
) -> tuple[set[str], dict[str, set[int]], set[str], set[str], set[str]]:
    """Parse ``--only`` into (run_names, {run_name: {trials}}, dim_tokens,
    require_subs, exclude_subs).

    Entries like ``mobo_00_01`` add the full run. Entries like
    ``mobo_00_01/trial_3`` add only that trial from that run. Full paths
    (e.g. ``optimize/runs/mobo_00_01/trial_3``) and backslashes are handled.

    A bare dimension token such as ``4d`` or ``10d`` is a shorthand that selects
    every run directory whose name contains that dimension as an
    underscore-delimited segment — e.g. ``10d`` matches both ``mobo_10d_*`` and
    ``mobo_ensemble_10d_*`` (matched as the substring ``_10d_`` in
    ``collect_trials``). Several dim tokens union (``4d,10d`` -> both dims).

    Any other bare word is a substring *variant* filter that further narrows the
    shorthand selection (logical AND with the dim tokens): ``ensemble`` keeps only
    runs whose name contains ``ensemble``; a leading ``-``/``!`` excludes instead
    (``-ensemble`` drops the ensemble runs). So ``4d,ensemble`` selects only
    ``mobo_ensemble_4d_*`` while ``4d`` alone still catches the plain runs too.
    """
    import re
    run_names: set[str] = set()
    run_trials: dict[str, set[int]] = {}
    dim_tokens: set[str] = set()
    require_subs: set[str] = set()
    exclude_subs: set[str] = set()
    for part in only_str.split(","):
        part = part.strip().replace("\\", "/").rstrip("/")
        if not part:
            continue
        segments = part.split("/")
        mobo_seg = None
        trial_seg = None
        for seg in segments:
            if seg.startswith("mobo_"):
                mobo_seg = seg
            elif re.fullmatch(r"trial_\d+", seg):
                trial_seg = seg
        if mobo_seg is None:
            # Shorthand: a bare dimension token (e.g. "4d", "10d") expands to a
            # substring match over every mobo_*_<dim>_* run directory, so it
            # catches plain (mobo_10d_*) and variant (mobo_ensemble_10d_*) runs.
            if re.fullmatch(r"\d+d", part):
                dim_tokens.add(f"_{part}_")
            elif part[0] in "-!^" and len(part) > 1:
                # Variant exclusion, e.g. "-ensemble" drops ensemble runs.
                exclude_subs.add(part[1:])
            else:
                # Variant requirement, e.g. "ensemble" keeps only ensemble runs.
                require_subs.add(part)
            continue
        if trial_seg is not None:
            num = int(trial_seg.replace("trial_", ""))
            run_trials.setdefault(mobo_seg, set()).add(num)
        else:
            run_names.add(mobo_seg)
    return run_names, run_trials, dim_tokens, require_subs, exclude_subs


def collect_trials(
    runs_dir: str,
    *,
    exclude_old: bool = False,
    only_runs: set[str] | None = None,
    only_trials: dict[str, set[int]] | None = None,
    only_prefixes: set[str] | None = None,
    require_subs: set[str] | None = None,
    exclude_subs: set[str] | None = None,
    only_signature: dict | None = None,
) -> list[dict]:
    """Crawl ``runs_dir/mobo_*/mobo_progress.json`` → list of trial records.

    Each record: {source_run, trial, metrics{...}, time_key, time_value,
    hparams{...}}. Trials missing dist_to_needles, dup_fraction, or any time
    objective (avg_time_per_iter_s / runtime_s) are skipped.

    *only_runs*: if set, include only these run directories (all trials).
    *only_trials*: if set, maps run names to specific trial numbers to include.
    *only_prefixes*: dim-token substrings (e.g. ``_4d_`` from ``--only 4d``);
    a run matches if its name contains *any* of them.
    *require_subs* / *exclude_subs*: variant substrings that further narrow the
    shorthand selection — a run must contain *every* required substring and
    *none* of the excluded ones (e.g. ``ensemble`` / ``-ensemble``).
    """
    has_filter = (only_runs or only_trials or only_prefixes
                  or require_subs or exclude_subs)
    records: list[dict] = []
    # Accept either a runs *parent* directory (containing mobo_*/mobo_progress.json)
    # or a single run directory (containing mobo_progress.json directly).
    progress_paths = sorted(glob.glob(os.path.join(runs_dir, "mobo_*", "mobo_progress.json")))
    if not progress_paths and os.path.isfile(os.path.join(runs_dir, "mobo_progress.json")):
        progress_paths = [os.path.join(runs_dir, "mobo_progress.json")]
    for path in progress_paths:
        run_name = os.path.basename(os.path.dirname(path))
        if has_filter:
            # Shorthand selection: dim tokens union, then variant subs AND-narrow.
            shorthand_active = bool(only_prefixes or require_subs or exclude_subs)
            dim_ok = (not only_prefixes
                      or any(tok in run_name for tok in only_prefixes))
            require_ok = all(sub in run_name for sub in (require_subs or set()))
            exclude_ok = not any(sub in run_name for sub in (exclude_subs or set()))
            matches_shorthand = (shorthand_active
                                 and dim_ok and require_ok and exclude_ok)
            if (run_name not in (only_runs or set())
                    and run_name not in (only_trials or {})
                    and not matches_shorthand):
                continue
        if exclude_old and run_name == "mobo_old_jackson":
            continue
        if only_signature is not None:
            sig = _load_run_signature(os.path.dirname(path))
            if sig is None or not _signatures_match(only_signature, sig):
                continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  [collect] {run_name}: unreadable ({exc}); skipping.")
            continue
        trial_filter = (only_trials or {}).get(run_name)
        used = 0
        skipped_failed = 0
        for t in data.get("trials", []):
            if trial_filter is not None and t.get("trial") not in trial_filter:
                continue
            m = t.get("metrics", {})
            if DIST_KEY not in m or DUP_KEY not in m:
                continue
            tm = _time_metric(m)
            if tm is None:
                continue
            time_value, time_key = tm
            # A non-positive time metric marks a failed trial: it completed zero
            # ZoMBI iterations (avg_time_per_iter_s = runtime/0 -> 0.0) and so
            # carries only failure sentinels (the unmatched-needle penalty
            # distance, time=0). Those aren't real measurements; on the minimised
            # objectives they masquerade as Pareto-optimal, so exclude them.
            if time_value <= 0:
                skipped_failed += 1
                continue
            try:
                metrics = {DIST_KEY: float(m[DIST_KEY]), DUP_KEY: float(m[DUP_KEY]),
                           time_key: time_value}
            except (TypeError, ValueError):
                continue
            records.append({
                "source_run": run_name,
                "trial":      t.get("trial"),
                "metrics":    metrics,
                "time_key":   time_key,
                "time_value": time_value,
                "hparams":    t.get("hparams", {}),
            })
            used += 1
        if used or skipped_failed:
            note = (f"  ({skipped_failed} failed trial(s) skipped)"
                    if skipped_failed else "")
            print(f"  [collect] {run_name}: {used} trial(s){note}")
    return records


# ─── Pareto front (minimisation on all objectives) ─────────────────────────────

def pareto_mask_min(M: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of ``M`` (all columns minimised).

    Row j dominates row i iff ``M[j] <= M[i]`` elementwise and ``M[j] < M[i]`` in
    at least one objective. A row kept iff no other row dominates it (so points
    with identical objective vectors are all kept).
    """
    n = len(M)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(M <= M[i], axis=1) & np.any(M < M[i], axis=1)
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


# ─── Visualisation ─────────────────────────────────────────────────────────────

def _obj_pairs(obj_labels: list[str]) -> list[tuple[int, int, str, str]]:
    """The three pairwise (x_idx, y_idx, x_label, y_label) panels."""
    return [
        (0, 2, obj_labels[0], obj_labels[2]),
        (0, 1, obj_labels[0], obj_labels[1]),
        (1, 2, obj_labels[1], obj_labels[2]),
    ]


def plot_pareto(M: np.ndarray, mask: np.ndarray, obj_labels: list[str],
                out_path: str) -> None:
    """Pairwise objective scatter; Pareto-optimal points starred (static PNG)."""
    matplotlib.use("Agg")
    plt.switch_backend("Agg")
    pairs = _obj_pairs(obj_labels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"MOBO Pareto front across all runs  "
                 f"(★ = Pareto-optimal, {int(mask.sum())}/{len(mask)})", fontsize=12)
    for ax, (ix, iy, xl, yl) in zip(axes, pairs):
        ax.scatter(M[~mask, ix], M[~mask, iy], c="steelblue", alpha=0.6,
                   edgecolors="k", linewidths=0.3, label="dominated")
        ax.scatter(M[mask, ix], M[mask, iy], marker="*", s=220, c="gold",
                   zorder=5, edgecolors="k", linewidths=0.5, label="Pareto")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto plot -> {out_path}")


def _has_display() -> bool:
    """True if an interactive GUI window can plausibly be shown.

    On Linux a window needs an X11/Wayland display, so a missing ``DISPLAY`` /
    ``WAYLAND_DISPLAY`` (e.g. an SSH session or batch node) means headless. macOS
    and Windows always have a native display server.
    """
    if platform.system() in ("Darwin", "Windows"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_file(path: str) -> None:
    """Open a file with the OS default viewer."""
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _final_plot_for_trial(runs_dir: str, source_run: str, trial: int) -> str | None:
    """Return a trial's landscape view to open on click, or None if it has none.

    3D trials render per-iteration frames (``plots/iter_*.png`` — open the last);
    4D trials render an interactive (rotatable) ``point_cloud.html`` (falling back
    to a legacy static ``coverage.png`` for older runs). Higher-dimensional trials
    have no landscape view, so this returns None and clicking is a silent no-op.

    Ensemble trials are nested one level deeper: instead of writing landscape
    plots directly under ``trial_N``, they hold one sub-run per ensemble repeat
    (``trial_N/run_1`` ... ``trial_N/run_5``), each with its own ``plots``. For
    those we open the first sub-run's final plot.
    """
    trial_dir = os.path.join(runs_dir, source_run, f"trial_{trial}")

    def _landscape_in(d: str) -> str | None:
        pngs = sorted(glob.glob(os.path.join(d, "plots", "iter_*.png")))
        if pngs:
            return pngs[-1]
        for name in ("point_cloud.html", "coverage.png"):
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                return candidate
        return None

    found = _landscape_in(trial_dir)
    if found:
        return found
    # Ensemble trial: fall back to the first sub-run (trial_N/run_1, run_2, ...).
    run_dirs = sorted(glob.glob(os.path.join(trial_dir, "run_*")))
    for run_dir in run_dirs:
        found = _landscape_in(run_dir)
        if found:
            return found
    return None


def _hparam_normalised(value: float, name: str) -> float:
    """Map a raw hyperparameter value to [0, 1] within its HPARAM_SPACE bounds."""
    lo, hi, tfm = HPARAM_SPACE[name]
    if tfm == "log":
        import math
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        return (value - lo) / (hi - lo)


def plot_pareto_interactive(
    M: np.ndarray,
    mask: np.ndarray,
    records: list[dict],
    runs_dir: str,
    obj_labels: list[str],
    *,
    show_numberline: bool = False,
) -> None:
    """Interactive Pareto plot: hover highlights across all subplots, click opens trial image.

    Hovering and clicking work on *every* trial — Pareto stars and dominated
    points alike. The hovered point is ringed in red across all three subplots,
    its metrics shown in the tooltip, and a click opens its landscape view (if
    any). The hyperparameter number-line (``--show-numberline``) only lists
    Pareto points, so it highlights when a Pareto point is hovered and clears
    when a dominated one is.
    """
    pairs = _obj_pairs(obj_labels)
    pareto_idx = np.where(mask)[0]
    pareto_M = M[pareto_idx]
    n_pareto = len(pareto_idx)

    # --- Build hparam matrix for Pareto points (normalised to [0,1]) ---
    if show_numberline:
        available_hparams = [
            name for name in HPARAM_NAMES
            if all(name in records[i]["hparams"] for i in pareto_idx)
        ]
        hp_norm = np.full((n_pareto, len(available_hparams)), np.nan)
        for j, name in enumerate(available_hparams):
            for k, ri in enumerate(pareto_idx):
                hp_norm[k, j] = _hparam_normalised(
                    float(records[ri]["hparams"][name]), name,
                )

    # --- Figure 1: Pareto scatter ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    fig.suptitle(
        f"MOBO Pareto front across all runs  "
        f"(★ = Pareto-optimal, {n_pareto}/{len(mask)})  —  hover/click any point",
        fontsize=12,
    )

    for ax, (ix, iy, xl, yl) in zip(axes, pairs):
        ax.scatter(
            M[~mask, ix], M[~mask, iy],
            c="steelblue", alpha=0.6, edgecolors="k", linewidths=0.3, label="dominated",
        )
        ax.scatter(
            pareto_M[:, ix], pareto_M[:, iy],
            marker="*", s=220, c="gold", zorder=5,
            edgecolors="k", linewidths=0.5, label="Pareto",
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)

    # A red ring marks the hovered point, whether it is a gold Pareto star or a
    # steelblue dominated dot — an open circle reads clearly over either marker.
    highlight_artists = []
    for ax, (ix, iy, _, _) in zip(axes, pairs):
        hl = ax.scatter(
            [], [], marker="o", s=320, facecolors="none",
            edgecolors="red", linewidths=2.0, zorder=10,
        )
        highlight_artists.append(hl)

    tooltip = fig.text(0.5, 0.01, "", ha="center", fontsize=9, color="gray")

    # --- Figure 2: Hyperparameter number lines (optional) ---
    fig_hp = None
    if show_numberline:
        n_hp = len(available_hparams)
        fig_hp, ax_hp = plt.subplots(figsize=(10, max(4, n_hp * 0.45 + 1.5)))
        fig_hp.suptitle("Pareto-optimal hyperparameters  —  hover a star on the other figure", fontsize=11)

        for j in range(n_hp):
            ax_hp.axhline(j, color="lightgray", linewidth=1.0, zorder=0)
            lo, hi, tfm = HPARAM_SPACE[available_hparams[j]]
            ax_hp.text(-0.02, j, f"{lo}", ha="right", va="center", fontsize=7, color="gray",
                       transform=ax_hp.get_yaxis_transform())
            ax_hp.text(1.02, j, f"{hi}", ha="left", va="center", fontsize=7, color="gray",
                       transform=ax_hp.get_yaxis_transform())

        hp_dots = []
        for j in range(n_hp):
            dots = ax_hp.scatter(
                hp_norm[:, j], np.full(n_pareto, j),
                c="gold", edgecolors="k", linewidths=0.3, s=50, alpha=0.5, zorder=2,
            )
            hp_dots.append(dots)

        hp_highlight = []
        for j in range(n_hp):
            hl = ax_hp.scatter([], [], c="red", edgecolors="k", linewidths=0.8,
                               s=120, zorder=5, marker="D")
            hp_highlight.append(hl)

        hp_val_labels = []
        for j in range(n_hp):
            lbl = ax_hp.text(0, j, "", fontsize=7, color="red", fontweight="bold",
                             ha="center", va="bottom", zorder=6)
            hp_val_labels.append(lbl)

        ax_hp.set_xlim(-0.05, 1.05)
        ax_hp.set_ylim(-0.8, n_hp - 0.2)
        ax_hp.set_yticks(range(n_hp))
        ax_hp.set_yticklabels(available_hparams, fontsize=8)
        ax_hp.set_xlabel("normalised value (0 = lower bound, 1 = upper bound)", fontsize=9)
        ax_hp.invert_yaxis()
        fig_hp.tight_layout()

    # --- Shared interaction state ---
    active_idx = [None]

    def _nearest_point(event) -> int | None:
        """Global record index of the point nearest the cursor, or None.

        Searches *all* trials (Pareto and dominated) so both are interactive.
        Returns an index into ``records`` / ``M``.
        """
        if event.inaxes is None:
            return None
        ax = event.inaxes
        try:
            panel = list(axes).index(ax)
        except ValueError:
            return None
        ix, iy = pairs[panel][0], pairs[panel][1]
        dx = M[:, ix] - event.xdata
        dy = M[:, iy] - event.ydata
        sx = ax.get_xlim()
        sy = ax.get_ylim()
        x_range = sx[1] - sx[0]
        y_range = sy[1] - sy[0]
        if x_range == 0 or y_range == 0:
            return None
        dist = np.sqrt((dx / x_range) ** 2 + (dy / y_range) ** 2)
        best = int(np.argmin(dist))
        if dist[best] < 0.05:
            return best
        return None

    def _update_hp_highlight(idx: int | None) -> None:
        """Highlight the hovered point on the Pareto-only number-line.

        *idx* is a global record index. Dominated points are absent from the
        number-line, so they clear it; Pareto points map to their row in
        ``pareto_idx`` / ``hp_norm``.
        """
        if fig_hp is None:
            return
        pareto_pos = None
        if idx is not None:
            where = np.where(pareto_idx == idx)[0]
            if len(where):
                pareto_pos = int(where[0])
        if pareto_pos is None:
            for hl in hp_highlight:
                hl.set_offsets(np.empty((0, 2)))
            for lbl in hp_val_labels:
                lbl.set_text("")
        else:
            rec = records[idx]
            for j, name in enumerate(available_hparams):
                val_norm = hp_norm[pareto_pos, j]
                hp_highlight[j].set_offsets([[val_norm, j]])
                raw_val = rec["hparams"].get(name)
                if raw_val is not None:
                    txt = f"{raw_val:.4g}" if isinstance(raw_val, float) else str(raw_val)
                    hp_val_labels[j].set_position((val_norm, j))
                    hp_val_labels[j].set_text(txt)
                else:
                    hp_val_labels[j].set_text("")
        fig_hp.canvas.draw_idle()

    def _on_motion(event):
        idx = _nearest_point(event)
        if idx == active_idx[0]:
            return
        active_idx[0] = idx
        if idx is None:
            for hl in highlight_artists:
                hl.set_offsets(np.empty((0, 2)))
            tooltip.set_text("")
        else:
            for hl, (ix, iy, _, _) in zip(highlight_artists, pairs):
                hl.set_offsets([[M[idx, ix], M[idx, iy]]])
            rec = records[idx]
            m = rec["metrics"]
            kind = "Pareto" if mask[idx] else "dominated"
            tooltip.set_text(
                f"{rec['source_run']} trial {rec['trial']} ({kind})  |  "
                f"dist={m[DIST_KEY]:.4f}  dup={m[DUP_KEY]:.4f}  "
                f"{rec['time_key']}={rec['time_value']:.4g}"
            )
        fig.canvas.draw_idle()
        _update_hp_highlight(idx)

    def _on_click(event):
        idx = _nearest_point(event)
        if idx is None:
            return
        rec = records[idx]
        img = _final_plot_for_trial(runs_dir, rec["source_run"], rec["trial"])
        # Higher-dimensional trials have no landscape image: silently do nothing
        # (no error, no popup) — hover highlighting still works for them.
        if not img:
            return
        try:
            print(f"  Opening: {img}")
            _open_file(img)
        except Exception as exc:
            print(f"  Could not open {img}: {exc}")

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    print("  Interactive Pareto plot open. Hover any point (Pareto or dominated) to "
          "highlight, click to open its trial image.")
    if show_numberline:
        print("  Hyperparameter figure shows values for the hovered Pareto point.")
    plt.show()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the global Pareto-optimal MOBO trials and write pareto.json.")
    parser.add_argument("runs_dir", nargs="?", default=None,
                        help="Directory containing runs/mobo_* (default: optimize/runs).")
    parser.add_argument("--out", default=None,
                        help="Output directory for pareto.json / pareto_front.png "
                             "(default: the runs directory).")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Save a static PNG instead of opening the interactive window.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of runs or specific trials to include "
                             "(e.g. mobo_00_01,mobo_00_02/trial_3). A bare dimension "
                             "token like 4d or 10d includes every run whose name contains "
                             "that dimension, e.g. both mobo_10d_* and mobo_ensemble_10d_*. "
                             "Add a variant word to narrow it: '4d,ensemble' keeps only "
                             "mobo_ensemble_4d_*, '4d,-ensemble' keeps only plain mobo_4d_*.")
    parser.add_argument("--with-old", action="store_true",
                        help="Include trials from mobo_old_jackson (excluded by default).")
    parser.add_argument("--show-numberline", action="store_true",
                        help="Show hyperparameter number-line figure in interactive mode.")
    parser.add_argument("--no-shared-history", action="store_true",
                        help="When a single run dir is given, do NOT pool sibling runs "
                             "with a matching run_config; use only that one run's trials.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.abspath(args.runs_dir) if args.runs_dir else os.path.join(script_dir, "runs")
    out_dir = os.path.abspath(args.out) if args.out else runs_dir

    # If runs_dir is itself a single run directory (holds mobo_progress.json
    # directly), trial dirs live in its parent under the run's own name, so the
    # click-to-open-image lookup must resolve relative to that parent.
    single_run = (
        not glob.glob(os.path.join(runs_dir, "mobo_*", "mobo_progress.json"))
        and os.path.isfile(os.path.join(runs_dir, "mobo_progress.json"))
    )
    plot_runs_dir = os.path.dirname(runs_dir) if single_run else runs_dir

    # Shared-history pooling: when a single run dir is given, default to pooling its
    # config-matching siblings (same dataset/dim/time-budget/direction/ensemble
    # settings), mirroring run_mobo's --share-history. This crawls the parent runs
    # dir, filtered to the target run's signature. --no-shared-history opts out.
    only_signature = None
    if single_run and not args.no_shared_history:
        only_signature = _load_run_signature(runs_dir)
        if only_signature is None:
            print(f"  [shared-history] {os.path.basename(runs_dir)} has no run_config.json; "
                  "using this run's trials only.")
        else:
            print(f"  [shared-history] pooling sibling runs matching "
                  f"{os.path.basename(runs_dir)}'s config: {only_signature}")
            runs_dir = plot_runs_dir  # crawl the parent, filtered by signature below

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"MOBO Pareto collection  |  runs: {runs_dir}")
    print("=" * 70)

    only_runs, only_trials, only_prefixes, require_subs, exclude_subs = (
        _parse_only(args.only) if args.only else (None, None, None, None, None))
    records = collect_trials(runs_dir, exclude_old=not args.with_old,
                             only_runs=only_runs or None,
                             only_trials=only_trials or None,
                             only_prefixes=only_prefixes or None,
                             require_subs=require_subs or None,
                             exclude_subs=exclude_subs or None,
                             only_signature=only_signature)
    if not records:
        sys.exit(f"No usable trials found under {runs_dir}/mobo_*/mobo_progress.json.")

    # Third objective label tracks the time key the collected trials recorded.
    time_keys_used = {r["time_key"] for r in records}
    if time_keys_used == {"runtime_s"}:
        time_label = "runtime_s"
    elif time_keys_used == {"avg_time_per_iter_s"}:
        time_label = "avg_time_per_iter_s"
    else:
        time_label = "avg_time_per_iter_s | runtime_s (MIXED)"
        print("  [warn] collection mixes avg_time_per_iter_s and runtime_s trials; "
              "the third objective combines per-iteration and total-runtime seconds "
              "(different units). Filter with --only to compare like with like.")
    obj_labels = [DIST_KEY, DUP_KEY, time_label]

    M = np.array([[r["metrics"][DIST_KEY], r["metrics"][DUP_KEY], r["time_value"]]
                  for r in records], dtype=float)
    mask = pareto_mask_min(M)
    n_total, n_pareto = len(records), int(mask.sum())
    print(f"\n  {n_total} trial(s) total -> {n_pareto} Pareto-optimal.")

    # Pareto records, best dist_to_needles first.
    pareto = [records[i] for i in np.where(mask)[0]]
    pareto.sort(key=lambda r: r["metrics"][DIST_KEY])

    out = {
        "generated":      datetime.datetime.now().isoformat(timespec="seconds"),
        "runs_dir":       runs_dir,
        "objectives":     {lbl: "minimize" for lbl in obj_labels},
        "n_trials_total": n_total,
        "n_pareto":       n_pareto,
        "pareto":         pareto,
    }
    json_path = os.path.join(out_dir, "pareto.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  pareto.json -> {json_path}")

    if args.no_interactive:
        plot_pareto(M, mask, obj_labels, os.path.join(out_dir, "pareto_front.png"))
    elif not _has_display():
        # Headless system (e.g. SSH / batch node): an interactive window can't be
        # shown, so save a static PNG into optimize/ instead of failing.
        png_path = os.path.join(script_dir, "pareto_front.png")
        print("  No display detected (headless); saving static PNG instead of "
              "opening an interactive window.")
        plot_pareto(M, mask, obj_labels, png_path)
    else:
        plot_pareto_interactive(M, mask, records, plot_runs_dir, obj_labels,
                                show_numberline=args.show_numberline)

    print("\n  Pareto-optimal configurations (best dist first):")
    for r in pareto:
        m = r["metrics"]
        print(f"    {r['source_run']} trial {r['trial']}:  "
              f"dist={m[DIST_KEY]:.4f}  dup={m[DUP_KEY]:.4f}  "
              f"{r['time_key']}={r['time_value']:.4g}")


if __name__ == "__main__":
    main()
