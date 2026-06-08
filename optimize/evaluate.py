"""
optimize/evaluate.py
====================
Re-evaluate selected ZoMBI-Hop hyperparameter configurations several times on a
chosen objective, writing the SAME per-run artifact set as ``run_mobo.py`` (CSVs,
static plots, per-iteration landscape frames / video where the dimension allows).

Where ``run_mobo.py`` *searches* for good hyperparameters on the 3-element
campaign RF surrogate, this script takes hyperparameters that a MOBO run already
found (``--trials`` from a ``--runs-path mobo_*`` folder) and simply RUNS them —
``--num-runs`` times each — so you can quantify run-to-run variance or transfer a
configuration onto a higher-dimensional benchmark (``--dataset``).

Datasets
--------
``--dataset`` selects both the objective and its reference optima.  Pass one
name, or several comma-separated (``--dataset RF,ackley4d``) to evaluate the same
trials on each in turn; with multiple datasets every dataset gets its own
sub-folder (``rerun_*/RF/``, ``rerun_*/ackley4d/``) holding the usual artifacts.

  RF         the 3-simplex Random-Forest surrogate from the source run's
             ``run_config.json`` (search direction, csv_path and reference optima
             are inherited from ``--runs-path`` — fully non-interactive).
  ackley3d   negated analytic Ackley on the 3-simplex   (maximise; peaks known).
  ackley4d   negated analytic Ackley on the 4-simplex   (maximise; peaks known).
  ackley10d  negated analytic Ackley on the 10-simplex  (maximise; peaks known).

The Ackley variant defaults to ``realistic`` (the materials-tuned, multi-peak
mode); override with ``--ackley-variant``.  Hyperparameters are dimension-
independent, so a configuration tuned on the 3-simplex RF transfers unchanged to
the 4-/10-simplex Ackley benchmarks.

Output
------
A single ``rerun_DD_MM_HH_MM/`` folder (military clock, mirroring ``mobo_*``):

  rerun_DD_MM_HH_MM/
    rerun_config.json          static config (dataset, dim, optima, time limit …)
    rerun_summary.json         per-trial mean/std of the three objectives + every
                               individual run's metrics
    trial_<n>/                 one per selected source trial
        hparams.json           the hyperparameters being re-evaluated
        run_<k>/               one per --num-runs repeat
            metrics.json
            points.csv         (sample_idx, <coords>, Y, penalized, activation, zoom)
            needles.csv        (needle_idx, <coords>, value, median_value,
                                activation, zoom, iteration, dist_to_centre)
            metrics_over_time.csv
            dist_from_centre.png
            line_length_hist.png
            hparam_edge_proximity.png
            plots/iter_*.png    (dim == 3 only)  +  zombihop_timelapse.mp4
            point_cloud.html    (dim == 4 only — final-state interactive cloud)
            (dim >= 5: metrics + non-landscape plots only, no landscape view)

``<coords>`` are ``FA/MA/Br`` for the 3-simplex RF (drop-in compatible with the
mobo trial CSVs) and ``x1..xd`` otherwise.

Usage
-----
  conda activate zombi-hop
  python optimize/evaluate.py --runs-path optimize/runs/mobo_05_06_15_32 \
         --trials 12,34,56 --dataset RF --num-runs 3
  python optimize/evaluate.py --runs-path optimize/runs/mobo_05_06_15_32 \
         --trials 12,34 --dataset ackley4d --num-runs 5 --time-limit-min 10
  python optimize/evaluate.py --runs-path optimize/runs/mobo_05_06_15_32 \
         --trials 12 --dataset ackley10d --num-runs 3
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

# ZoMBI's GP code prints unicode glyphs (→, ×) unconditionally; on a non-UTF-8
# console (e.g. Windows cp1252) those prints raise UnicodeEncodeError, which would
# otherwise propagate out of ``optimizer.run`` and abort an otherwise-fine run.
# Make stdout/stderr tolerant so logging can never kill a ZoMBI evaluation.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# Reuse the surrogate, metrics, plotting and ZoMBI trial machinery from run_mobo.
# (Importing it also wires up sys.path to the repo root and the simplex helpers.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_mobo as rm

# Analytic benchmark objectives (negated Ackley on the d-simplex).
from synthetic_data.ackley import Ackley

# Per-run video assembly (best-effort: needs imageio + ffmpeg).
try:
    import make_videos as mv
except Exception:                                   # pragma: no cover - optional dep
    mv = None

DATASET_DIMS = {"RF": 3, "ackley3d": 3, "ackley4d": 4, "ackley10d": 10}


class _TopKReached(Exception):
    """Internal signal: ``--top-k`` needles found, stop this run early."""


# ─── Dataset resolution ─────────────────────────────────────────────────────────

def resolve_dataset(dataset: str, runs_path: str, ackley_variant: str) -> dict:
    """Build the objective + reference optima for ``dataset``.

    Returns a dict with: ``dim``, ``maximize``, ``fn`` (scalar callable
    ``(d,)->float`` for the ZoMBI sim-objective), ``true_optima`` (list of (d,)),
    ``grid_pts`` / ``grid_vals`` (ternary render grid for dim==3 else None),
    ``ackley_fn`` (the ``Ackley`` instance for dim==4 point clouds else None),
    and ``label``.
    """
    if dataset == "RF":
        cfg_path = os.path.join(runs_path, "run_config.json")
        if not os.path.exists(cfg_path):
            sys.exit(f"--dataset RF needs the source run's config, but {cfg_path} "
                     f"does not exist.")
        with open(cfg_path) as f:
            cfg = json.load(f)
        csv_path = cfg["csv_path"]
        if not os.path.exists(csv_path):
            sys.exit(f"Surrogate CSV from run_config.json no longer exists: {csv_path}")
        maximize    = bool(cfg["maximize"])
        true_optima = [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
        print(f"  [dataset] RF surrogate from {csv_path} "
              f"({'maximize' if maximize else 'minimize'}, "
              f"{len(true_optima)} reference optima inherited from {os.path.basename(runs_path)})")
        _, rf_fn, grid_pts, grid_vals = rm.build_rf_and_grid(csv_path)
        return dict(dim=3, maximize=maximize, fn=rf_fn, true_optima=true_optima,
                    grid_pts=grid_pts, grid_vals=grid_vals, ackley_fn=None,
                    label="RF", csv_path=os.path.abspath(csv_path))

    # ── Analytic Ackley benchmarks ──
    dim = DATASET_DIMS[dataset]
    fn  = Ackley(ackley_variant, dim=dim)
    true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
    maximize = True                                  # negated Ackley → peaks are maxima
    print(f"  [dataset] {dataset}: Ackley('{ackley_variant}', dim={dim}) — "
          f"maximize, {len(true_optima)} analytic peak(s)")
    if dim == 3:
        grid_pts  = rm.ternary_grid(rm.TERNARY_GRID_N)
        grid_vals = fn.predict(grid_pts)
    else:
        grid_pts = grid_vals = None
    return dict(dim=dim, maximize=maximize, fn=fn, true_optima=true_optima,
                grid_pts=grid_pts, grid_vals=grid_vals,
                ackley_fn=(fn if dim == 4 else None), label=dataset, csv_path=None)


# ─── Trial-hparam selection from the source mobo run ────────────────────────────

def load_trial_hparams(runs_path: str, trial_nums: list[int]) -> dict[int, dict]:
    """Read the hyperparameters of the requested trials from ``--runs-path``.

    Prefers ``mobo_progress.json`` (single source of truth); falls back to each
    ``trial_<n>/trial.json``.  Aborts if a trial is missing or its hparams don't
    cover the current ``HPARAM_SPACE`` (stale hyperparameter set).
    """
    by_trial: dict[int, dict] = {}
    prog_path = os.path.join(runs_path, "mobo_progress.json")
    if os.path.exists(prog_path):
        try:
            with open(prog_path) as f:
                for t in json.load(f).get("trials", []):
                    by_trial[int(t["trial"])] = t.get("hparams", {})
        except Exception as exc:
            print(f"  [trials] {prog_path} unreadable ({exc}); falling back to trial.json files.")

    out: dict[int, dict] = {}
    for n in trial_nums:
        hp = by_trial.get(n)
        if hp is None:
            tj = os.path.join(runs_path, f"trial_{n}", "trial.json")
            if os.path.exists(tj):
                try:
                    with open(tj) as f:
                        hp = json.load(f).get("hparams")
                except Exception as exc:
                    sys.exit(f"--trials: {tj} unreadable ({exc}).")
        if hp is None:
            sys.exit(f"--trials: trial {n} not found in {runs_path} "
                     f"(checked mobo_progress.json and trial_{n}/trial.json).")
        missing = [k for k in rm.HPARAM_NAMES if k not in hp]
        if missing:
            sys.exit(f"--trials: trial {n} is missing hparams {missing} "
                     f"(stale hyperparameter set?).")
        out[n] = {k: hp[k] for k in rm.HPARAM_NAMES}
        print(f"  [trials] trial {n}: loaded {len(out[n])} hyperparameters")
    return out


# ─── Generalised init data (dimension-aware copy of run_mobo._gen_init_data) ────

def gen_init_data(fn_callable, maximize: bool, dim: int):
    """Generate ``N_INIT_LINES`` random simplex lines on the ``dim``-simplex."""
    x_a_list, x_e_list, y_list = [], [], []
    for _ in range(rm.N_INIT_LINES):
        x0   = torch.full((dim,), 1.0 / dim, device=rm.DEVICE, dtype=rm.DTYPE)
        dir_ = rm.zero_sum_dirs(1, dim, device=rm.DEVICE, dtype=rm.DTYPE).squeeze(0)
        seg  = rm.line_simplex_segment(x0, dir_)
        if seg is None:
            continue
        _, _, x_left, x_right = seg
        t = torch.linspace(0.0, 1.0, rm.NUM_EXPERIMENTS, dtype=torch.float64, device=rm.DEVICE)
        pts_t = (x_left.to(torch.float64).unsqueeze(0)
                 + t.unsqueeze(1) * (x_right - x_left).to(torch.float64).unsqueeze(0))
        z = rm.composition_to_ilr(pts_t)
        z = z + torch.randn_like(z) * rm.NOISE_LEVEL_ILR
        pts_t = rm.ilr_to_composition(z, d=dim)
        pts_np = pts_t.detach().cpu().numpy()
        raw    = np.array([fn_callable(x) for x in pts_np], dtype=float)
        y_t    = torch.tensor(raw if maximize else -raw, dtype=rm.DTYPE, device=rm.DEVICE)
        y_t    = y_t + torch.randn_like(y_t) * rm.NOISE_LEVEL
        pts_out = pts_t.to(dtype=rm.DTYPE, device=rm.DEVICE)
        x_a_list.append(pts_out)
        x_e_list.append(pts_out)
        y_list.append(y_t)
    if not x_a_list:
        raise RuntimeError("Could not generate any initial simplex lines.")
    return (torch.cat(x_a_list, dim=0),
            torch.cat(x_e_list, dim=0),
            torch.cat(y_list, dim=0).reshape(-1, 1))


# ─── Generalised CSV writers (dimension-aware coordinate columns) ───────────────

def coord_cols(dim: int, dataset: str) -> list[str]:
    """Coordinate column names: FA/MA/Br for the 3-simplex RF, else x1..xd."""
    if dataset == "RF" and dim == 3:
        return ["FA", "MA", "Br"]
    return [f"x{i + 1}" for i in range(dim)]


def write_points_csv(path: str, dh, snap_records: list[tuple], cols: list[str]) -> None:
    import pandas as pd
    X = dh.X_all_actual.detach().cpu().numpy()
    Y = dh.Y_all.detach().cpu().numpy().ravel()
    n = X.shape[0]
    mask = dh.get_penalty_mask()                     # True = NOT penalized
    penalized = (~mask.detach().cpu().numpy()) if mask is not None else np.zeros(n, bool)
    act, zm = rm._activation_zoom_per_point(n, snap_records)
    data = {"sample_idx": np.arange(n)}
    for j, c in enumerate(cols):
        data[c] = X[:, j]
    data.update({"Y": Y, "penalized": penalized.astype(int),
                 "activation": act, "zoom": zm})
    pd.DataFrame(data).to_csv(path, index=False)


def write_needles_csv(path: str, dh, cols: list[str], dim: int) -> None:
    import math
    import pandas as pd
    centroid = np.full(dim, 1.0 / dim)
    rows = []
    for i, r in enumerate(dh.get_all_needle_results()):
        pt = r["point"].detach().cpu().numpy().ravel()
        mv_ = r.get("median_value")
        row = {"needle_idx": i}
        for j, c in enumerate(cols):
            row[c] = pt[j]
        row.update({
            "value": r.get("value"),
            "median_value": (None if mv_ is None or (isinstance(mv_, float) and math.isnan(mv_)) else mv_),
            "activation": r.get("activation"),
            "zoom": r.get("zoom"),
            "iteration": r.get("iteration"),
            "dist_to_centre": float(np.linalg.norm(pt - centroid)),
        })
        rows.append(row)
    columns = ["needle_idx"] + cols + ["value", "median_value", "activation",
                                       "zoom", "iteration", "dist_to_centre"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


# ─── 4D landscape: final-state interactive point cloud ──────────────────────────

def write_point_cloud_html(path: str, ackley_fn, dh, last_payload: dict) -> None:
    """Render the final ZoMBI state over the 4-simplex Ackley cloud (one HTML).

    Uses ``synthetic_data/point_cloud_4d``'s overlay API.  Pared points, needle
    markers, and the last LineBO lines are all exact (plain simplex compositions);
    needle penalisation ellipsoids are intentionally omitted because the run's
    tangent basis differs from the Helmert ILR basis the overlay assumes.
    """
    import plotly.graph_objects as go
    import synthetic_data.point_cloud_4d as pc4

    comp = pc4.build_simplex_lattice(pc4.GRID_N)
    obj  = ackley_fn.predict(comp)
    xyz  = pc4.to_3d(comp)
    obj_min, obj_max = float(obj.min()), float(obj.max())

    hover = [f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
             for (a, b, c, d), v in zip(comp, obj)]
    cloud = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name="objective",
        text=hover, hoverinfo="text",
        marker=dict(color=obj, colorscale="Viridis", cmin=obj_min, cmax=obj_max,
                    size=pc4.MARKER_SIZE, opacity=pc4.MARKER_OPACITY,
                    showscale=True, colorbar=dict(title="Objective")),
    )
    peaks_xyz = pc4.to_3d(np.array(ackley_fn.centers))
    peaks_trace = go.Scatter3d(
        x=peaks_xyz[:, 0], y=peaks_xyz[:, 1], z=peaks_xyz[:, 2], mode="markers",
        name="known peak",
        marker=dict(symbol="diamond", color="red", size=6,
                    line=dict(color="white", width=1)),
        hoverinfo="name",
    )
    fig = go.Figure(data=[cloud, pc4.tetra_edges_trace(),
                          pc4.vertex_labels_trace(), peaks_trace])
    fig.update_layout(
        title="ZoMBI-Hop final state on the 4-simplex (negated Ackley) point cloud",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data"),
        legend=dict(x=0.0, y=1.0), width=pc4.FIG_W, height=pc4.FIG_H,
    )

    # ── Overlay data pulled from the final data-handler state ──
    pared_X = pared_Y = recency = None
    if dh.X_pared is not None and dh.X_pared.shape[0] > 0:
        pared_X = dh.X_pared.detach().cpu().numpy()
        pared_Y = dh.Y_pared.detach().cpu().numpy().ravel()
        recency = np.arange(pared_X.shape[0], dtype=float)

    needle_t = dh.get_all_needle_locations()
    needles  = (needle_t.detach().cpu().numpy()
                if needle_t is not None and needle_t.numel() > 0 else None)

    main_line  = (np.array(last_payload["line_0"]) if last_payload
                  and last_payload.get("line_0") is not None else None)
    cache_line = (np.array(last_payload["line_1"]) if last_payload
                  and last_payload.get("line_1") is not None else None)

    pc4.add_simplex_overlays(
        fig, obj_cmin=obj_min, obj_cmax=obj_max,
        pared_points=pared_X, pared_values=pared_Y, recency=recency,
        main_line=main_line, cache_line=cache_line, needles=needles,
    )
    fig.write_html(path, include_plotlyjs="cdn", auto_open=False)


# ─── A single evaluation run (full run_mobo-style artifact set) ─────────────────

def run_single_eval(hparams: dict, ds: dict, dataset: str, out_dir: str,
                    time_limit_min: float, top_k: int | None = None) -> dict:
    """Run one time-limited ZoMBI trial on ``ds['fn']``; write all artifacts.

    Mirrors ``run_mobo.run_single_trial`` but is dimension-general: it writes the
    same CSVs and static plots for every dimension, and dispatches the landscape
    view by dimension (ternary frames + mp4 for 3D, an interactive point cloud
    for 4D, none for ≥5D).  Returns ``{"dist", "dup", "runtime"}``.

    If ``top_k`` is given, the run terminates as soon as ``top_k`` needles have
    been found (in addition to the wall-clock ``time_limit_min`` budget); the
    same artifact set / metrics are written regardless of why the run stopped.
    """
    os.makedirs(out_dir, exist_ok=True)
    dim       = ds["dim"]
    fn        = ds["fn"]
    maximize  = ds["maximize"]
    true_optima = ds["true_optima"]
    cols      = coord_cols(dim, dataset)

    plot_state: dict = {"line_0": None, "line_1": None}
    payloads: list[dict] = []
    snap_records: list[tuple] = []
    call_counter = [0]
    dh_ref = [None]

    sim_obj = rm.make_sim_obj(fn, rm.DEVICE, rm.DTYPE, maximize=maximize)
    inner   = rm.make_linebo_wrapper(sim_obj, dim, rm.NUM_LINES, rm.DEVICE, rm.DTYPE, plot_state)

    def obj_wrapper(x_tell, bounds, acq_fn):
        x_req, x_act, y = inner(x_tell, bounds, acq_fn)
        call_counter[0] += 1
        dh = dh_ref[0]
        xp, yp = dh.X_pared, dh.Y_pared
        if xp is not None and xp.shape[0] > 0:
            pared_X = xp.detach().cpu().numpy()
            pared_Y = yp.detach().cpu().numpy().ravel()
            if not maximize:
                pared_Y = -pared_Y
        else:
            pared_X = pared_Y = None
        needles = dh.needles
        payloads.append(dict(
            iter_num=call_counter[0],
            pared_X=pared_X, pared_Y=pared_Y,
            needles=(needles.detach().cpu().numpy()
                     if needles is not None and needles.shape[0] > 0 else None),
            needle_M_list=[m.detach().cpu().clone() if m is not None else None
                           for m in dh.needle_M_list],
            needle_B=(dh.needle_B.detach().cpu().clone() if dh.needle_B is not None else None),
            bounds=(dh.bounds.detach().cpu().clone() if dh.bounds is not None else None),
            line_0=plot_state.get("line_0"),
            line_1=plot_state.get("line_1"),
            n_points_before=(dh.X_all_actual.shape[0] if dh.X_all_actual is not None else 0),
        ))
        if top_k is not None:
            n_needles = needles.shape[0] if needles is not None else 0
            if n_needles >= top_k:
                raise _TopKReached(n_needles)
        return x_req, x_act, y

    try:
        X_a, X_e, Y = gen_init_data(fn, maximize, dim)
    except RuntimeError as exc:
        print(f"      [run] init failed: {exc}")
        return {"dist": rm.UNMATCHED_PENALTY, "dup": 1.0, "runtime": 0.0}

    optimizer = rm.ZoMBIHop(
        objective=obj_wrapper,
        X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
        **rm.ZOMBI_FIXED, **hparams,
        device=str(rm.DEVICE), dtype=rm.DTYPE,
        run_uuid=None, checkpoint_dir=None,
    )
    dh = optimizer.data_handler
    dh_ref[0] = dh

    orig_snap = dh.take_snapshot
    def snap_wrap(*a, **k):
        orig_snap(*a, **k)
        if dh.X_all_actual is not None:
            snap_records.append((dh.X_all_actual.shape[0],
                                 dh.current_activation, dh.current_zoom))
    dh.take_snapshot = snap_wrap

    t0 = time.time()
    try:
        optimizer.run(max_activations=float("inf"), time_limit_hours=time_limit_min / 60.0)
    except _TopKReached as stop:
        print(f"      [run] top-k reached: {int(stop.args[0])} needle(s) found — stopping early")
    except Exception as exc:
        print(f"      [run] ZoMBI crashed: {exc}")
    runtime = time.time() - t0

    # ── Metrics ──
    needle_t   = dh.get_all_needle_locations()
    discovered = needle_t.detach().cpu().numpy() if needle_t.numel() > 0 else np.empty((0, dim))
    X_all_np   = (dh.X_all_actual.detach().cpu().numpy()
                  if dh.X_all_actual is not None else np.empty((0, dim)))
    dist = rm.metric_dist_to_needles(discovered, true_optima)
    dup  = rm.metric_dup_fraction(X_all_np, rm.NOISE_LEVEL / 2.0)
    print(f"      [run]  iters={call_counter[0]}  dist={dist:.4f}  dup={dup:.4f}"
          f"  t={runtime:.1f}s  needles={len(discovered)}/{len(true_optima)}")

    # ── CSV artifacts ──
    try:
        write_points_csv(os.path.join(out_dir, "points.csv"), dh, snap_records, cols)
        write_needles_csv(os.path.join(out_dir, "needles.csv"), dh, cols, dim)
        rm.write_metrics_over_time_csv(
            os.path.join(out_dir, "metrics_over_time.csv"), payloads, X_all_np, true_optima)
    except Exception as exc:
        print(f"      [run] CSV write failed: {exc}")

    # ── Static (dimension-general) plots ──
    try:
        rm.plot_dist_from_centre(os.path.join(out_dir, "dist_from_centre.png"), dh, maximize)
        rm.plot_line_length_hist(os.path.join(out_dir, "line_length_hist.png"), payloads)
        rm.plot_hparam_edge_proximity(
            os.path.join(out_dir, "hparam_edge_proximity.png"),
            rm.hparams_to_norm(hparams))
    except Exception as exc:
        print(f"      [run] static plot failed: {exc}")

    # ── Landscape view (dimension-dispatched, rendered AFTER timing) ──
    try:
        if dim == 3:
            plots_dir = os.path.join(out_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            print(f"      [run] rendering {len(payloads)} ternary frames …", flush=True)
            for p in payloads:
                rm.render_frame(p, ds["grid_pts"], ds["grid_vals"], true_optima, maximize,
                                os.path.join(plots_dir, f"iter_{p['iter_num'] - 1:04d}.png"))
            if mv is not None:
                try:
                    mv.make_video_from_dir(plots_dir, os.path.join(out_dir, "zombihop_timelapse.mp4"))
                except Exception as exc:
                    print(f"      [run] video assembly failed: {exc}")
        elif dim == 4 and ds["ackley_fn"] is not None:
            last_payload = payloads[-1] if payloads else None
            print("      [run] rendering 4D point-cloud HTML …", flush=True)
            write_point_cloud_html(os.path.join(out_dir, "point_cloud.html"),
                                   ds["ackley_fn"], dh, last_payload)
        # dim >= 5: metrics + non-landscape plots only (no landscape view).
    except Exception as exc:
        print(f"      [run] landscape render failed: {exc}")

    metrics = {"dist_to_needles": round(dist, 6),
               "dup_fraction":    round(dup, 6),
               "runtime_s":       round(runtime, 3)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return {"dist": dist, "dup": dup, "runtime": runtime}


# ─── Summary across runs ────────────────────────────────────────────────────────

def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": round(float(arr.mean()), 6),
            "std":  round(float(arr.std(ddof=0)), 6),
            "min":  round(float(arr.min()), 6),
            "max":  round(float(arr.max()), 6),
            "n":    int(arr.size)}


def write_summary(path: str, per_trial: dict) -> None:
    """Per-trial mean/std of the three objectives plus every individual run."""
    summary = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
               "trials": []}
    for trial_num, runs in per_trial.items():
        entry = {"trial": trial_num, "n_runs": len(runs), "runs": runs}
        if runs:
            for key in ("dist_to_needles", "dup_fraction", "runtime_s"):
                entry[key] = _agg([r[key] for r in runs])
        summary["trials"].append(entry)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary -> {path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def _parse_trials(raw: str) -> list[int]:
    nums = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            nums.append(int(tok))
        except ValueError:
            sys.exit(f"--trials: '{tok}' is not an integer trial number.")
    if not nums:
        sys.exit("--trials: no trial numbers parsed.")
    return nums


def _parse_datasets(raw: str) -> list[str]:
    """Parse the comma-separated ``--dataset`` value, preserving order, dropping
    duplicates, and validating each name against ``DATASET_DIMS``."""
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in DATASET_DIMS:
            sys.exit(f"--dataset: '{tok}' is not a known dataset "
                     f"(choose from {', '.join(sorted(DATASET_DIMS))}).")
        if tok not in out:
            out.append(tok)
    if not out:
        sys.exit("--dataset: no dataset names parsed.")
    return out


def evaluate_dataset(dataset: str, out_dir: str, runs_path: str,
                     hparams_by_trial: dict[int, dict], trial_nums: list[int],
                     args) -> int:
    """Run every selected trial (``--num-runs`` times) on a single ``dataset``,
    writing the full rerun artifact set into ``out_dir``.  Returns the number of
    runs completed (for the closing tally)."""
    ds = resolve_dataset(dataset, runs_path, args.ackley_variant)

    # Static config for this dataset's rerun.
    with open(os.path.join(out_dir, "rerun_config.json"), "w") as f:
        json.dump({
            "generated":       datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset":         dataset,
            "dim":             ds["dim"],
            "maximize":        ds["maximize"],
            "ackley_variant":  (args.ackley_variant if dataset != "RF" else None),
            "csv_path":        ds.get("csv_path"),
            "runs_path":       runs_path,
            "trials":          trial_nums,
            "num_runs":        args.num_runs,
            "time_limit_min":  args.time_limit_min,
            "top_k":           args.top_k,
            "true_optima":     [list(map(float, t.ravel())) for t in ds["true_optima"]],
        }, f, indent=2)

    total = len(trial_nums) * args.num_runs
    done = 0
    per_trial: dict = {}
    for trial_num in trial_nums:
        hparams = hparams_by_trial[trial_num]
        trial_dir = os.path.join(out_dir, f"trial_{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)

        print(f"\n=== [{dataset}] trial {trial_num}  "
              f"({args.num_runs} run(s) @ {args.time_limit_min} min) ===")
        per_trial[trial_num] = []
        for k in range(1, args.num_runs + 1):
            done += 1
            print(f"  [run {k}/{args.num_runs}]  (overall {done}/{total})")
            run_dir = os.path.join(trial_dir, f"run_{k}")
            try:
                res = run_single_eval(hparams, ds, dataset, run_dir,
                                      args.time_limit_min, top_k=args.top_k)
                per_trial[trial_num].append({
                    "run": k,
                    "dist_to_needles": round(res["dist"], 6),
                    "dup_fraction":    round(res["dup"], 6),
                    "runtime_s":       round(res["runtime"], 3),
                })
            except KeyboardInterrupt:
                print("\n[!] Interrupted by user — writing summary so far …")
                write_summary(os.path.join(out_dir, "rerun_summary.json"), per_trial)
                raise
            except Exception as exc:
                print(f"  [run {k}] FAILED: {exc}")
            # Persist the summary after every run so a crash never loses progress.
            write_summary(os.path.join(out_dir, "rerun_summary.json"), per_trial)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate selected ZoMBI-Hop hyperparameter sets on a chosen "
                    "objective, writing run_mobo-style artifacts.")
    parser.add_argument("--runs-path", required=True, metavar="MOBO_DIR",
                        help="Source mobo_* run directory holding the trials to re-run.")
    parser.add_argument("--trials", required=True, metavar="N,N,...",
                        help="Comma-separated source trial numbers (e.g. 12,34,56).")
    parser.add_argument("--dataset", required=True, metavar="DS[,DS...]",
                        help="Objective(s) to evaluate on, comma-separated "
                             "(RF | ackley3d | ackley4d | ackley10d). Multiple "
                             "datasets each get their own sub-folder in the rerun.")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Repeats per selected trial (default: 1).")
    parser.add_argument("--time-limit-min", type=float, default=10.0,
                        help="Wall-clock budget per run, minutes (default: 10).")
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="Stop each run as soon as K needles are found "
                             "(in addition to --time-limit-min; default: no needle cap).")
    parser.add_argument("--ackley-variant", default="realistic",
                        choices=sorted(Ackley.VARIANTS),
                        help="Ackley variant for the ackley* datasets (default: realistic).")
    parser.add_argument("--out", default=None,
                        help="Parent directory for the rerun_* folder "
                             "(default: optimize/runs).")
    args = parser.parse_args()

    runs_path = os.path.abspath(args.runs_path)
    if not os.path.isdir(runs_path):
        sys.exit(f"--runs-path not found: {runs_path}")
    if args.num_runs < 1:
        sys.exit("--num-runs must be >= 1.")
    if args.top_k is not None and args.top_k < 1:
        sys.exit("--top-k must be >= 1.")
    trial_nums = _parse_trials(args.trials)
    datasets = _parse_datasets(args.dataset)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_parent = os.path.abspath(args.out) if args.out else os.path.join(script_dir, "runs")
    os.makedirs(out_parent, exist_ok=True)
    rerun_dir = os.path.join(out_parent, datetime.datetime.now().strftime("rerun_%d_%m_%H_%M"))
    os.makedirs(rerun_dir, exist_ok=True)

    print("=" * 72)
    print(f"ZoMBI-Hop evaluate  |  datasets={datasets}  trials={trial_nums}  "
          f"num_runs={args.num_runs}  limit={args.time_limit_min} min"
          + (f"  top_k={args.top_k}" if args.top_k is not None else ""))
    print(f"source: {runs_path}")
    print(f"output: {rerun_dir}")
    print("=" * 72)

    # Trial hyperparameters are dataset-independent — load them once.
    hparams_by_trial = load_trial_hparams(runs_path, trial_nums)

    # One dataset writes its artifacts directly into rerun_dir (unchanged layout);
    # multiple datasets each get their own sub-folder under it.
    done = 0
    for dataset in datasets:
        out_dir = rerun_dir if len(datasets) == 1 else os.path.join(rerun_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)
        done += evaluate_dataset(dataset, out_dir, runs_path,
                                 hparams_by_trial, trial_nums, args)

    print(f"\nDone. {done} run(s) across {len(trial_nums)} trial(s) "
          f"and {len(datasets)} dataset(s). Results in {rerun_dir}")


if __name__ == "__main__":
    main()
