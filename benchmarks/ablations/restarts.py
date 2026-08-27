"""
benchmarks/ablations/restarts.py
================================
A1's arm: **k independent ZoMBI restarts**, and the artifact merge that makes them
comparable to one ZoMBI-Hop run.

What "independent" has to mean
------------------------------
Splitting the budget is the easy half. The hard half is that a restart must inherit
*nothing*: not the measured points, not the GP posterior, not the needle penalties,
not the trust region. That rules out running one ``ZoMBIHop`` with its history
cleared between activations — the object's whole design is to carry that state
forward. So each restart is a separate ``run_mobo.run_single_trial`` call with its
own optimiser, its own initial lines and its own empty ``DataHandler``, exactly as
if it were a fresh job. The only thing shared is the landscape.

Each restart is also capped at ``max_activations_per_restart`` activations (default
1). That is what makes it plain ZoMBI rather than a short ZoMBI-Hop: one activation
zooms in, converges, declares one needle, and stops. k of them yield up to k needles
with no memory between them, which is precisely the strategy hopping claims to beat.

Why the artifacts get merged rather than reported per restart
------------------------------------------------------------
``dist_to_needles`` scores a *set* of discovered optima against the landscape's true
optima, and ``dup_fraction`` scores a *pooled* sample cloud. Reporting five restarts
separately would answer a different question ("how good is one short ZoMBI run?")
than the one A1 asks ("does the strategy of restarting match the strategy of
hopping?"). So the restarts are pooled into one trial directory carrying the same
artifact set every other arm writes, and every metric is recomputed over the pool:

* ``points.csv``  — all restarts concatenated in time order, ``sample_idx``
  re-indexed, ``activation`` offset so activations stay globally unique, plus a
  ``restart`` column.
* ``needles.csv`` — likewise, ``needle_idx`` re-indexed, ``iteration``/``activation``
  offset.
* ``metrics_over_time.csv`` — **recomputed**, not concatenated. At merged iteration
  *t* the discovered set is (all needles from finished restarts) ∪ (this restart's
  needles so far), which is the only reading under which the curve is comparable to
  a hopping run's.
* the plots, redrawn from the merged data.

Budget fairness
---------------
Two things have to be right or A1 measures a handicap rather than a strategy.

``fill_budget=True`` (the default) keeps launching restarts after the planned k while
budget remains, because a restart that converges in a third of its slice would
otherwise leave the arm spending far less than the baseline. The actual count lands in
``restarts.json``.

And the budget is counted in **optimiser time**, not wall-clock. Every restart renders
a full artifact set when it finishes, which on a short cell can cost as much as the
search itself; charging that to the budget would starve the later restarts, while the
baseline — which renders once, after its whole run — would pay nothing.
``ZoMBIHop.run``'s own ``time_limit_hours`` covers only the search, so summing the
per-restart ``runtime`` puts both arms on the same clock.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from dataclasses import replace
from typing import Any, Iterator

from ._paths import ensure_paths

ensure_paths()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import run_mobo as rm  # noqa: E402
from eval_metrics import (  # noqa: E402
    metric_dist_to_needles,
    metric_dup_fraction,
)
from mobo_landscapes import LandscapeSpec, composition_column_names  # noqa: E402

from .arms import capture_convergence


# Stop launching fill-in restarts once less than this fraction of a normal slice is
# left: a stub restart that cannot finish its initial design contributes points
# without ever declaring a needle, which drags dup_fraction without a chance at dist.
MIN_FILL_SLICE_FRACTION = 0.35
# Hard ceiling on fill-in restarts, as a multiple of the planned k. Guards against a
# pathological landscape where every restart dies in seconds and the loop spins.
MAX_RESTART_MULTIPLIER = 4


# ─── Activation cap ──────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _cap_activations(max_activations: int) -> Iterator[None]:
    """Cap ``ZoMBIHop.run``'s activation budget without touching its time limit.

    ``run_single_trial`` hard-codes ``max_activations=inf`` whenever the landscape
    carries a wall-clock budget, which is right for every other arm and wrong for
    this one: a restart that keeps hopping after its first needle is not an
    independent ZoMBI run. Clamping inside ``run`` gets both — the slice's wall-clock
    still bounds the restart, and the activation count still makes it plain ZoMBI.
    """
    from src.core.zombihop import ZoMBIHop

    original = ZoMBIHop.run

    def capped_run(self, *args, **kwargs):
        # max_activations is run()'s first parameter, so it can arrive either way.
        # Re-passing it in the same position it came in keeps the remaining
        # positional arguments (time_limit_hours, pause_event, …) bound correctly.
        if args:
            requested, rest = args[0], args[1:]
            return original(self, min(float(requested), float(max_activations)),
                            *rest, **kwargs)
        requested = kwargs.pop("max_activations", 5)
        kwargs["max_activations"] = min(float(requested), float(max_activations))
        return original(self, **kwargs)

    ZoMBIHop.run = capped_run
    try:
        yield
    finally:
        ZoMBIHop.run = original


# ─── Small helpers ───────────────────────────────────────────────────────────────

def _read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    try:
        # An empty-but-valid frame is meaningful (a restart that declared no
        # needles), so it is returned as-is; only an unreadable file is None.
        return pd.read_csv(path)
    except Exception:
        return None


def _resolve_truth(landscape: LandscapeSpec,
                   ensemble_config: dict | None) -> tuple[Any, list[np.ndarray]]:
    """The objective and true optima this trial is actually scored against.

    ``run_single_trial`` does this reseeding internally and then discards it; the
    merge needs the same pair to recompute pooled metrics, so it is redone here from
    the same inputs. Both calls are deterministic in ``ensemble_config``, so they
    cannot disagree.
    """
    if ensemble_config is not None:
        fn, true_optima, _ = rm.reseed_ensemble(landscape, ensemble_config)
        return fn, true_optima
    return landscape.fn_callable, [np.asarray(o, dtype=float)
                                   for o in landscape.true_optima]


class _MergedDataHandler:
    """The three arrays ``run_mobo.plot_convergence`` reads, and nothing else.

    The real ``DataHandler`` for each restart is released the moment its
    ``run_single_trial`` returns — deliberately, since holding k of them is what used
    to OOM-kill these jobs. ``arms.capture_convergence`` taps the arrays on the way
    past instead, and this shim presents the pooled versions back to the same
    plotting function, so the merged convergence plot is drawn by exactly the code
    that draws every other arm's.
    """

    def __init__(self, Y_all: np.ndarray, penalty_mask: np.ndarray | None,
                 needle_indices: np.ndarray | None):
        self.Y_all = torch.as_tensor(np.asarray(Y_all, dtype=float)).reshape(-1, 1)
        self._mask = (torch.as_tensor(np.asarray(penalty_mask, dtype=bool))
                      if penalty_mask is not None else None)
        self.needle_indices = (torch.as_tensor(np.asarray(needle_indices, dtype=np.int64))
                               if needle_indices is not None and len(needle_indices)
                               else None)

    def get_penalty_mask(self):
        return self._mask


def _plot_dist_from_centre_from_csv(path: str, points: pd.DataFrame,
                                    needles: pd.DataFrame | None, dim: int) -> None:
    """The ``dist_from_centre.png`` of the pooled restarts.

    A CSV-driven twin of ``run_mobo.plot_dist_from_centre``, which needs a live
    ``DataHandler``. Same axes, same encodings (sample index as colour, needles as
    crimson stars) so the merged figure sits beside a single-run one without a
    reader having to translate.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    cols = composition_column_names(dim)
    fig = Figure(figsize=(7, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    if points is None or points.empty or not all(c in points.columns for c in cols):
        ax.text(0.5, 0.5, "no data", ha="center")
        fig.savefig(path, dpi=120)
        fig.clear()
        return

    X = points[cols].to_numpy(dtype=float)
    Y = points["Y"].to_numpy(dtype=float)
    centroid = np.full(X.shape[1], 1.0 / X.shape[1])
    dists = np.linalg.norm(X - centroid, axis=1)
    sc = ax.scatter(dists, Y, c=np.arange(len(Y)), cmap="viridis", s=14, alpha=0.7,
                    zorder=3, label="samples")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("sample index", fontsize=8)

    if needles is not None and not needles.empty and all(c in needles.columns for c in cols):
        nx = needles[cols].to_numpy(dtype=float)
        nv = needles["value"].to_numpy(dtype=float)
        nd = np.linalg.norm(nx - centroid, axis=1)
        ax.scatter(nd, nv, marker="*", s=220, color="crimson", edgecolors="darkred",
                   lw=0.8, zorder=5, label="needle")
    ax.set_xlabel("‖x − centroid‖₂")
    ax.set_ylabel("Objective Y")
    ax.set_title("Distance from simplex centre  (pooled restarts)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.clear()


# ─── The merge ───────────────────────────────────────────────────────────────────

def merge_restart_artifacts(
    trial_dir: str,
    restart_dirs: list[str],
    results: list[dict],
    *,
    dim: int,
    true_optima: list[np.ndarray],
    fn_callable: Any,
    convergence_records: list[dict | None],
) -> dict:
    """Pool k restart directories into one trial directory's artifact set.

    Returns the pooled ``{"dist", "dup", "n_points", "n_iters"}``. Everything else
    the caller needs (runtime, per-restart detail) it already has.
    """
    comp_cols = composition_column_names(dim)

    points_frames: list[pd.DataFrame] = []
    needle_frames: list[pd.DataFrame] = []
    merged_payloads: list[dict] = []
    conv_Y: list[np.ndarray] = []
    conv_mask: list[np.ndarray] = []
    conv_needle_idx: list[np.ndarray] = []

    point_offset = 0        # points contributed by finished restarts
    iter_offset = 0         # LineBO lines contributed by finished restarts
    activation_offset = 0   # activations contributed by finished restarts
    prior_needle_xy: list[np.ndarray] = []   # final needle coords of finished restarts
    prior_needle_vals: list[np.ndarray] = []

    for j, (rdir, res) in enumerate(zip(restart_dirs, results)):
        pts = _read_csv(os.path.join(rdir, "points.csv"))
        nds = _read_csv(os.path.join(rdir, "needles.csv"))
        payloads = res.get("payloads") or []
        n_pts_here = 0 if pts is None else len(pts)

        # ── points ──
        if pts is not None and len(pts):
            block = pts.copy()
            block.insert(0, "restart", j)
            if "activation" in block.columns:
                block["activation"] = block["activation"].fillna(0).astype(int) + activation_offset
            points_frames.append(block)

        # ── needles ──
        if nds is not None and len(nds):
            nblock = nds.copy()
            nblock.insert(0, "restart", j)
            if "activation" in nblock.columns:
                nblock["activation"] = (nblock["activation"].fillna(0).astype(int)
                                        + activation_offset)
            if "iteration" in nblock.columns:
                nblock["iteration"] = (nblock["iteration"].fillna(0).astype(int)
                                       + iter_offset)
            needle_frames.append(nblock)

        # ── metrics-over-time payloads ──
        # The discovered set at merged iteration t is every needle from a FINISHED
        # restart plus this restart's needles as of t. Concatenating the restarts'
        # own metrics_over_time.csv files instead would report each restart's dist
        # against its own handful of needles and never show the pooled curve, which
        # is the only one comparable to a hopping run.
        for p in payloads:
            q = {
                "iter_num": int(p.get("iter_num", 0)) + iter_offset,
                "n_points_before": int(p.get("n_points_before", 0)) + point_offset,
                "activation": (int(p["activation"]) + activation_offset
                               if p.get("activation") is not None else None),
                "line_0": p.get("line_0"),
                "line_1": p.get("line_1"),
            }
            own = p.get("needles")
            stack = list(prior_needle_xy)
            if own is not None and len(own):
                stack.append(np.asarray(own, dtype=float).reshape(-1, dim))
            q["needles"] = np.vstack(stack) if stack else None

            own_v = p.get("needle_vals")
            vstack = list(prior_needle_vals)
            if own_v is not None and len(own_v):
                vstack.append(np.asarray(own_v, dtype=float).ravel())
            q["needle_vals"] = np.concatenate(vstack) if vstack else None
            merged_payloads.append(q)

        # ── convergence arrays ──
        rec = convergence_records[j] if j < len(convergence_records) else None
        if rec is not None and rec.get("Y_all") is not None:
            y = np.asarray(rec["Y_all"], dtype=float).ravel()
            conv_Y.append(y)
            m = rec.get("penalty_mask")
            conv_mask.append(np.asarray(m, dtype=bool).ravel() if m is not None
                             else np.ones(len(y), dtype=bool))
            ni = rec.get("needle_indices")
            if ni is not None and len(ni):
                conv_needle_idx.append(np.asarray(ni, dtype=np.int64).ravel() + point_offset)

        # ── advance the offsets ──
        point_offset += n_pts_here
        iter_offset += len(payloads)
        if pts is not None and len(pts) and "activation" in pts.columns:
            activation_offset += int(pd.to_numeric(pts["activation"],
                                                   errors="coerce").fillna(0).max()) + 1
        else:
            activation_offset += 1
        if nds is not None and len(nds) and all(c in nds.columns for c in comp_cols):
            prior_needle_xy.append(nds[comp_cols].to_numpy(dtype=float))
            prior_needle_vals.append(
                pd.to_numeric(nds["value"], errors="coerce").fillna(0.0).to_numpy(dtype=float))

    # ── write the pooled CSVs ──
    if points_frames:
        merged_points = pd.concat(points_frames, ignore_index=True)
        merged_points["sample_idx"] = np.arange(len(merged_points))
    else:
        merged_points = pd.DataFrame(columns=["restart", "sample_idx", "Y", "penalized",
                                              "activation", "zoom", *comp_cols])
    merged_points.to_csv(os.path.join(trial_dir, "points.csv"), index=False)

    if needle_frames:
        merged_needles = pd.concat(needle_frames, ignore_index=True)
        merged_needles["needle_idx"] = np.arange(len(merged_needles))
    else:
        merged_needles = pd.DataFrame(columns=["restart", "needle_idx", *comp_cols,
                                               "value", "median_value", "activation",
                                               "zoom", "iteration", "reason",
                                               "dist_to_centre"])
    merged_needles.to_csv(os.path.join(trial_dir, "needles.csv"), index=False)

    # ── pooled trial-level metrics ──
    if len(merged_points) and all(c in merged_points.columns for c in comp_cols):
        X_all = merged_points[comp_cols].to_numpy(dtype=float)
    else:
        X_all = np.empty((0, dim))
    if len(merged_needles) and all(c in merged_needles.columns for c in comp_cols):
        discovered = merged_needles[comp_cols].to_numpy(dtype=float)
    else:
        discovered = np.empty((0, dim))

    # zoom_size comes from arms.patch_points_csv_zoom_size; without it the duplicate
    # radius would not be shrunk inside zoom boxes and this arm's dup would be on a
    # different scale from every other arm's.
    if "zoom_size" in merged_points.columns and len(merged_points):
        zoom_sizes = pd.to_numeric(merged_points["zoom_size"],
                                   errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        zoom_sizes = None

    dist = metric_dist_to_needles(discovered, true_optima, dim=dim)
    dup = metric_dup_fraction(X_all, dim=dim, zoom_sizes=zoom_sizes)

    # ── pooled metrics_over_time.csv (+ the plots that read it) ──
    try:
        rm.write_metrics_over_time_csv(
            os.path.join(trial_dir, "metrics_over_time.csv"),
            merged_payloads, X_all, true_optima, dim=dim)
    except Exception as exc:
        print(f"    [restarts] merged metrics_over_time failed: {exc}")

    try:
        _plot_dist_from_centre_from_csv(
            os.path.join(trial_dir, "dist_from_centre.png"),
            merged_points, merged_needles, dim)
    except Exception as exc:
        print(f"    [restarts] merged dist_from_centre failed: {exc}")

    try:
        rm.plot_line_length_hist(os.path.join(trial_dir, "line_length_hist.png"),
                                 merged_payloads)
    except Exception as exc:
        print(f"    [restarts] merged line_length_hist failed: {exc}")

    try:
        if conv_Y:
            shim = _MergedDataHandler(
                np.concatenate(conv_Y),
                np.concatenate(conv_mask) if conv_mask else None,
                np.concatenate(conv_needle_idx) if conv_needle_idx else None,
            )
            acts = (pd.to_numeric(merged_points["activation"], errors="coerce")
                    .fillna(0).to_numpy(dtype=int)
                    if "activation" in merged_points.columns and len(merged_points)
                    else None)
            # plot_convergence resets the running-best envelope at each activation
            # boundary; because activations were offset per restart, each restart's
            # envelope is already disconnected from the next one's — which is exactly
            # the right reading for independent runs.
            rm.plot_convergence(os.path.join(trial_dir, "convergence.png"), shim,
                                True, activations=acts)
    except Exception as exc:
        print(f"    [restarts] merged convergence failed: {exc}")

    try:
        true_best = (max(float(fn_callable(np.asarray(o, dtype=float).ravel()))
                         for o in true_optima) if true_optima else None)
    except Exception:
        true_best = None
    rm._auto_generate_plots(trial_dir, dim, true_best=true_best)

    return {"dist": float(dist), "dup": float(dup),
            "n_points": int(len(merged_points)), "n_iters": int(len(merged_payloads))}


# ─── The arm ─────────────────────────────────────────────────────────────────────

def run_restart_trial(
    hparams: dict,
    landscape: LandscapeSpec,
    trial_dir: str,
    *,
    ensemble_config: dict | None = None,
    n_restarts: int = 4,
    max_activations_per_restart: int = 1,
    fill_budget: bool = True,
    verbose: bool = True,
) -> dict:
    """Run k independent ZoMBI restarts on one landscape and pool their artifacts.

    Signature-compatible with ``run_mobo.run_single_trial`` in everything the
    ablation runner uses, so the two arms are interchangeable at the call site.

    Parameters
    ----------
    n_restarts : planned restarts. The wall-clock budget on ``landscape`` is split
        evenly across them.
    max_activations_per_restart : activations one restart may run (default 1 —
        plain ZoMBI: zoom in, converge, declare one needle, stop).
    fill_budget : keep launching restarts past ``n_restarts`` while wall-clock
        remains, so the arm actually spends the budget the baseline gets. Ignored
        when the landscape has no wall-clock budget.
    """
    os.makedirs(trial_dir, exist_ok=True)
    dim = landscape.dim
    fn_callable, true_optima = _resolve_truth(landscape, ensemble_config)

    if ensemble_config is not None:
        with open(os.path.join(trial_dir, "ensemble_config.json"), "w") as f:
            json.dump(ensemble_config, f, indent=2)

    n_restarts = max(1, int(n_restarts))
    total_h = landscape.time_limit_hours
    if total_h is not None:
        slice_h = float(total_h) / n_restarts
    else:
        # Activation-budgeted landscape: there is no clock to split, so each restart
        # simply runs its activation allowance and fill-in is meaningless.
        slice_h = None
        fill_budget = False

    if verbose:
        budget = f"{slice_h * 60:.1f} min each" if slice_h is not None else "no time limit"
        print(f"    [restarts] {n_restarts} independent ZoMBI restart(s), {budget}, "
              f"<= {max_activations_per_restart} activation(s) each"
              + (" (+fill)" if fill_budget else ""))

    restart_dirs: list[str] = []
    results: list[dict] = []
    conv_records: list[dict | None] = []
    conv_sink: list[dict] = []
    per_restart: list[dict] = []

    max_total = n_restarts * MAX_RESTART_MULTIPLIER
    j = 0
    # The budget is spent in OPTIMISER time, not wall-clock. ``run_single_trial``
    # renders a full artifact set after every restart, and on a short cell that
    # rendering can cost as much as the search — charging it to the budget would
    # leave later restarts with a fraction of a slice while the baseline arm, which
    # renders once after its whole run, pays nothing. ZoMBIHop.run's own
    # time_limit_hours covers only the search, so summing the returned ``runtime``
    # measures the arms on exactly the same clock.
    spent_h = 0.0
    with _cap_activations(max_activations_per_restart), capture_convergence(conv_sink):
        while True:
            remaining_h = (float(total_h) - spent_h) if total_h is not None else None
            if j >= n_restarts:
                if not fill_budget or j >= max_total:
                    break
                if remaining_h is None or remaining_h < slice_h * MIN_FILL_SLICE_FRACTION:
                    break
            this_slice = slice_h
            if remaining_h is not None:
                if remaining_h <= 0:
                    if verbose:
                        print(f"    [restarts] budget spent after {j} restart(s)")
                    break
                this_slice = min(slice_h, remaining_h)

            rdir = os.path.join(trial_dir, f"restart_{j}")
            if os.path.isdir(rdir):
                shutil.rmtree(rdir, ignore_errors=True)
            sub_landscape = replace(
                landscape,
                time_limit_hours=this_slice,
                max_activations=float(max_activations_per_restart),
            )
            if verbose:
                slice_txt = f"{this_slice * 60:.1f} min" if this_slice is not None else "untimed"
                print(f"    [restart {j + 1}] budget {slice_txt}", flush=True)

            n_before = len(conv_sink)
            t0 = time.time()
            res = rm.run_single_trial(hparams, sub_landscape, rdir,
                                      ensemble_config=ensemble_config)
            wall = time.time() - t0
            optimiser_s = float(res.get("runtime", 0.0))
            spent_h += optimiser_s / 3600.0

            restart_dirs.append(rdir)
            results.append(res)
            conv_records.append(conv_sink[n_before] if len(conv_sink) > n_before else None)
            per_restart.append({
                "restart": j,
                "budget_hours": (round(this_slice, 6) if this_slice is not None else None),
                # runtime_s is the optimiser's own clock, matching what a
                # run_single_trial reports; wall_s adds this restart's artifact
                # rendering, which is outside every arm's budget.
                "runtime_s": round(optimiser_s, 3),
                "wall_s": round(wall, 3),
                "n_iters": int(res.get("n_iters", 0)),
                "n_points": int(res.get("n_points", 0)),
                "dist": round(float(res.get("dist", float("nan"))), 6),
                "dup": round(float(res.get("dup", float("nan"))), 6),
            })
            j += 1

    runtime = float(sum(r["runtime_s"] for r in per_restart))
    merged = merge_restart_artifacts(
        trial_dir, restart_dirs, results, dim=dim, true_optima=true_optima,
        fn_callable=fn_callable, convergence_records=conv_records)

    n_iters = merged["n_iters"]
    avg_time_per_iter = runtime / n_iters if n_iters > 0 else 0.0

    # A dim-3 coverage plot reads the landscape truth from the trial dir; the
    # restarts each wrote an identical copy (same landscape), so lift one up.
    truth_src = os.path.join(restart_dirs[0], "coverage_ground_truth.npz") if restart_dirs else None
    if truth_src and os.path.isfile(truth_src):
        shutil.copyfile(truth_src, os.path.join(trial_dir, "coverage_ground_truth.npz"))

    with open(os.path.join(trial_dir, "restarts.json"), "w") as f:
        json.dump({
            "n_restarts_planned": n_restarts,
            "n_restarts_actual": len(per_restart),
            "max_activations_per_restart": max_activations_per_restart,
            "fill_budget": fill_budget,
            "total_budget_hours": total_h,
            "slice_hours": slice_h,
            "runtime_s": round(runtime, 3),
            "wall_s": round(float(sum(r["wall_s"] for r in per_restart)), 3),
            "restarts": per_restart,
        }, f, indent=2)

    if verbose:
        print(f"    [restarts] pooled over {len(per_restart)} restart(s) — "
              f"dist={merged['dist']:.4f}  dup={merged['dup']:.4f}  "
              f"iters={n_iters}  points={merged['n_points']}  "
              f"({runtime:.1f}s)", flush=True)

    return {
        "dist": merged["dist"],
        "dup": merged["dup"],
        "runtime": runtime,
        "avg_time_per_iter": avg_time_per_iter,
        "n_iters": n_iters,
        "n_points": merged["n_points"],
        # Deliberately empty: the payloads of k restarts have already been consumed
        # by the merge, and holding them past it is what makes these jobs run out of
        # host RAM. Nothing downstream of the runner reads them.
        "payloads": [],
        "ackley_seed": None,
        "ensemble_config": ensemble_config,
        "restarts": per_restart,
        "n_restarts_actual": len(per_restart),
    }
