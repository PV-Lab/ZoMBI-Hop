"""
interface/app.py
================
ZoMBI-Hop GUI: browse run history, visualise convergence and distance-from-
centre, explore sampled points and discovered needles, query the GP posterior,
and launch new experiments.  Dimensionality-agnostic — works for d=3…10+.

Usage
-----
  conda activate zombi-hop
  python interface/app.py [<checkpoint_dir>]
"""
from __future__ import annotations

import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path
from uuid import uuid4
from typing import Optional

import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

import matplotlib
import matplotlib.patches
import matplotlib.lines
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── project root on sys.path ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from botorch.exceptions import InputDataWarning
warnings.filterwarnings("ignore", category=InputDataWarning)

try:
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    _BOTORCH_OK = True
except ImportError:
    _BOTORCH_OK = False

from src import ZoMBIHop, LineBO
from src.core.linebo import line_simplex_segment, zero_sum_dirs
from src.utils.simplex import composition_to_ilr, ilr_to_composition, proj_simplex
from src.utils.datahandler import reconstruct_snapshot_tensors

# ── constants ─────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE           = torch.float64
NOISE_LEVEL     = 0.01
NOISE_LEVEL_ILR = 0.03
NUM_EXPERIMENTS = 24
NUM_LINES       = 10
N_INIT_LINES    = 2
POLL_MS         = 5_000

DEFAULT_CKPT_DIR = str(_HERE.parent / "runs")


class _StopRunRequested(Exception):
    """Raised inside the objective wrapper when the user clicks Stop Run."""
    pass

# ── ternary helpers ───────────────────────────────────────────────────────────

_TERNARY_SQRT3_2 = math.sqrt(3) / 2

# Regular-tetrahedron vertices for the 4-simplex → 3D point-cloud view (one per
# composition component). Mirrors synthetic_data/point_cloud_4d.TETRA_VERTICES so
# the GUI's d=4 cloud matches the offline plotter, without importing plotly.
_TETRA_VERTICES = np.array([
    [1.0,  1.0,  1.0],
    [1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
], dtype=float)
_TETRA_VERTICES = _TETRA_VERTICES - _TETRA_VERTICES.mean(axis=0)


def _ternary_xy(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Convert ternary (a, b, c) → Cartesian (x, y); a+b+c need not equal 1."""
    s = a + b + c
    s = np.where(s > 0, s, 1.0)
    x = b / s + (c / s) * 0.5
    y = (c / s) * _TERNARY_SQRT3_2
    return x, y


def _draw_ternary_frame_ax(ax, la: str = "A", lb: str = "B", lc: str = "C") -> None:
    """Draw equilateral triangle frame with grid lines and corner labels."""
    ax.plot([0, 1, 0.5, 0], [0, 0, _TERNARY_SQRT3_2, 0], "k-", lw=1.2, zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.10, _TERNARY_SQRT3_2 + 0.18)
    ax.text(-0.04, -0.04, la, ha="right", va="top",    fontsize=9)
    ax.text( 1.04, -0.04, lb, ha="left",  va="top",    fontsize=9)
    ax.text( 0.50, _TERNARY_SQRT3_2 + 0.03, lc, ha="center", va="bottom", fontsize=9)
    for t in (0.25, 0.50, 0.75):
        r = 1.0 - t
        for _a, _b, _c, _a2, _b2, _c2 in [
            (t, 0.0, r,   t,   r, 0.0),   # iso-a
            (0.0, t, r,   r,   t, 0.0),   # iso-b
            (0.0, r, t,   r, 0.0,   t),   # iso-c
        ]:
            x1, y1 = _ternary_xy(np.array([_a]),  np.array([_b]),  np.array([_c]))
            x2, y2 = _ternary_xy(np.array([_a2]), np.array([_b2]), np.array([_c2]))
            ax.plot([x1[0], x2[0]], [y1[0], y2[0]], color="#cccccc", lw=0.5, zorder=0)

# Hyperparameter categories: (name, type, default, display_label, tooltip_description)
HPARAM_CATEGORIES: dict[str, list[tuple]] = {
    "Acquisition Opt": [
        ("nat_grad_step",      "float", 0.02, "Nat-grad step",
         "Step size for natural gradient ascent used during acquisition optimization."),
        ("nat_grad_max_steps", "int",   50,   "Nat-grad max steps",
         "Maximum number of natural gradient steps per acquisition optimization call."),
        ("n_restarts",         "int",   30,   "Restarts",
         "Number of random restarts for multi-start acquisition function optimization."),
        ("raw",                "int",   500,  "Raw samples",
         "Number of raw random candidates sampled before gradient-based refinement begins."),
    ],
    "Acq Function": [
        ("acquisition_type", "str",   "ucb", "Acq. type",
         "Acquisition function used to select the next query point.\n"
         "  ucb – Upper Confidence Bound (exploration via β)\n"
         "  ei  – Expected Improvement\n"
         "  pi  – Probability of Improvement"),
        ("ucb_beta",         "float", 0.1,  "UCB β",
         "Exploration–exploitation trade-off for UCB: higher β = more exploration. "
         "Ignored when acquisition_type is 'ei' or 'pi'."),
        ("input_noise_ilr",  "float", 0.03, "Input noise (ILR)",
         "Std dev of isotropic Gaussian noise added to inputs in ILR space at every "
         "evaluation. Controls how spread-out the sampled line is around the suggested point."),
    ],
    "Zoom / Convergence": [
        ("max_zooms",                   "int",   3,    "Max zooms",
         "Maximum number of zoom (trust-region shrink) steps per activation before "
         "the optimizer hops to a new region."),
        ("max_iterations",              "int",   10,   "Max iters / zoom",
         "Maximum BO iterations (objective calls) allowed within each zoom level."),
        ("top_m_points",                "int",   4,    "Top-M points",
         "Number of top-performing data points used to define the new trust region "
         "after zooming in."),
        ("n_consecutive_converged",     "int",   2,    "Consec. converged",
         "Number of consecutive iterations that must meet the convergence criterion "
         "before the optimizer zooms in."),
        ("convergence_pi_threshold",    "float", 0.01, "Conv. PI thresh",
         "Probability of Improvement below this threshold counts as a converged iteration."),
        ("input_noise_threshold_mult",  "float", 2.0,  "Input noise mult",
         "Convergence gate: if the best observed improvement < mult × input_noise, "
         "the iteration is considered converged."),
        ("output_noise_threshold_mult", "float", 2.0,  "Output noise mult",
         "Convergence gate: if the best observed improvement < mult × output_noise std, "
         "the iteration is considered converged."),
        ("max_gp_points",               "int",   3000, "Max GP points",
         "Maximum number of data points passed to the GP. Older points are dropped "
         "when the dataset exceeds this size."),
        ("zoom_jaccard_threshold",      "float", 0.75, "Zoom Jaccard thresh",
         "If a new trust region overlaps a previously visited one by more than this "
         "Jaccard fraction, the activation exits early (avoids re-exploring)."),
        ("jaccard_window",              "int",   3,    "Jaccard window",
         "Number of recent bounds snapshots to compare when checking for "
         "Jaccard-based stagnation."),
        ("jaccard_threshold",           "float", 0.9,  "Jaccard stag. thresh",
         "Jaccard similarity above this threshold between consecutive bounds "
         "snapshots signals stagnation and triggers an early zoom."),
    ],
    "Penalty & Needle": [
        ("max_penalty_radius",           "float", 1.0,  "Max penalty radius",
         "Maximum radius (in ILR space) of the penalization ellipsoid placed around "
         "each discovered needle to prevent re-discovering it."),
        ("needle_shrink_factor",         "float", 0.85, "Needle shrink factor",
         "Multiplicative factor applied to needle ellipsoid radii each time the "
         "optimizer re-enters a needle's penalization region."),
        ("needle_stop_noise_multiplier", "float", 3.0,  "Needle stop mult",
         "Stop shrinking the needle ellipsoid when its radius falls below "
         "mult × input_noise (prevents the region from collapsing to zero)."),
        ("ellipsoid_drop_fraction",      "float", 0.25, "Ellipsoid drop frac",
         "Fraction of the smallest eigenvalue retained when trimming the needle "
         "ellipsoid matrix (regularization to avoid singular matrices)."),
        ("ellipsoid_eigenvalue_floor",   "float", 1e-6, "Eigenvalue floor",
         "Minimum eigenvalue for the needle penalty ellipsoid (numerical stability guard)."),
        ("bounds_shrink_factor",         "float", 0.8,  "Bounds shrink factor",
         "Multiplicative factor by which the trust-region AABB contracts at each zoom step."),
        ("min_axis_noise_mult",          "float", 2.0,  "Min axis noise mult",
         "Minimum trust-region half-width as a multiple of input_noise; prevents "
         "the search region from collapsing below the noise floor."),
    ],
    "Point Paring": [
        ("paring_spatial_halfnoise",  "float", 0.5, "Spatial halfnoise",
         "Half-bandwidth (in units of input_noise_ilr) for spatial clustering when "
         "deduplicating repeated nearby measurements. Points within this radius "
         "are candidates for merging."),
        ("paring_y_noise_multiplier", "float", 1.0, "Y-noise mult",
         "When paring duplicates, observations within mult × output_noise in Y are "
         "averaged together into a single representative point."),
    ],
}

# Flat set of every tunable hyperparameter name the UI knows about.
_KNOWN_HPARAM_NAMES: set[str] = {
    p[0] for params in HPARAM_CATEGORIES.values() for p in params
}


def load_hparams_json(path: str) -> dict:
    """
    Read a hyperparameter JSON file and return a {name: value} dict.

    Accepts either a file shaped like ``optimize/runs/.../trial.json`` (a
    top-level ``"hparams"`` object) or a flat ``{name: value}`` object. Only
    keys recognised by the UI (``_KNOWN_HPARAM_NAMES``) are returned; unknown
    keys (e.g. "metrics", "trial") are ignored.
    """
    with open(path, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Hyperparameter JSON must be an object.")
    hp = data.get("hparams", data)
    if not isinstance(hp, dict):
        raise ValueError("'hparams' must be an object.")
    return {k: v for k, v in hp.items() if k in _KNOWN_HPARAM_NAMES}


# Default hyperparameters used when no JSON file is supplied. Copied verbatim
# from optimize/runs/mobo_05_06_15_32/trial_112 (a strong MOBO result). Keep in
# sync with DEFAULT_HW_HPARAMS in scripts/run_zombi_main.py.
DEFAULT_HPARAMS: dict = {
    "nat_grad_step": 0.00100187,
    "nat_grad_max_steps": 54,
    "n_restarts": 285,
    "raw": 200,
    "ucb_beta": 1.45791911,
    "max_zooms": 3,
    "max_iterations": 8,
    "top_m_points": 8,
    "n_consecutive_converged": 1,
    "convergence_pi_threshold": 0.05,
    "input_noise_threshold_mult": 3.98353261,
    "output_noise_threshold_mult": 0.52251166,
    "max_penalty_radius": 4.56475059,
    "needle_shrink_factor": 0.637987,
    "needle_stop_noise_multiplier": 2.81839283,
    "paring_spatial_halfnoise": 1.21375814,
    "paring_y_noise_multiplier": 4.19507699,
}


# ── synthetic-run helpers ─────────────────────────────────────────────────────
# The objective itself comes from optimize/evaluate.resolve_dataset (RF surrogate
# + analytic Ackleys); these helpers turn a scalar objective into the LineBO/
# ZoMBI machinery and are dimension-general.

def _make_sim_obj(fn_callable, device, dtype, *, maximize: bool):
    def _obj(endpoints: torch.Tensor):
        left  = endpoints[0, 0].to(torch.float64)
        right = endpoints[0, 1].to(torch.float64)
        t     = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS,
                               dtype=torch.float64, device=left.device)
        pts   = left.unsqueeze(0) + t.unsqueeze(1) * (right - left).unsqueeze(0)
        z     = composition_to_ilr(pts)
        z     = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts   = ilr_to_composition(z, d=pts.shape[1])
        raw   = np.array([fn_callable(x) for x in pts.detach().cpu().numpy()], float)
        y     = torch.tensor(raw if maximize else -raw, dtype=dtype, device=device)
        y     = y + torch.randn_like(y) * NOISE_LEVEL
        return pts.to(dtype=dtype, device=device), y
    return _obj


def _make_linebo_wrapper(sim_obj, dim: int, device, dtype, plot_state: dict | None = None):
    linebo = LineBO(sim_obj, dim,
                   num_points_per_line=100, num_lines=NUM_LINES, device=str(device))

    def _wrap(x_tell, bounds, acq_fn):
        xl, xr  = linebo.ranked_line_endpoints(x_tell, bounds, acq_fn)
        if plot_state is not None:
            n_valid = xl.shape[0]
            plot_state["line_0"] = (
                (xl[0].cpu().numpy(), xr[0].cpu().numpy()) if n_valid > 0 else None
            )
            plot_state["line_1"] = (
                (xl[1].cpu().numpy(), xr[1].cpu().numpy()) if n_valid > 1 else None
            )
        x_act, y = sim_obj(torch.stack([xl, xr], dim=1))
        x_act = x_act.to(device=device, dtype=dtype)
        y     = y.to(device=device, dtype=dtype).ravel()
        if x_act.shape[0] > 1:
            xc  = x_act - x_act.mean(dim=0, keepdim=True)
            _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
            d_  = Vt[0]
            p   = xc @ d_
            tv  = torch.linspace(p.min().item(), p.max().item(),
                                 x_act.shape[0], device=device, dtype=dtype)
            xr_ = x_act.mean(0).unsqueeze(0) + tv.unsqueeze(1) * d_.unsqueeze(0)
            xr_ = proj_simplex(xr_)
        else:
            xr_ = x_act.clone()
        return xr_, x_act, y

    return _wrap


def _gen_init_data(fn_callable, d: int, maximize: bool):
    xa, xe, yl = [], [], []
    for _ in range(N_INIT_LINES):
        x0  = torch.full((d,), 1.0 / d, device=DEVICE, dtype=DTYPE)
        dir_ = zero_sum_dirs(1, d, device=DEVICE, dtype=DTYPE).squeeze(0)
        seg  = line_simplex_segment(x0, dir_)
        if seg is None:
            continue
        _, _, xl, xr = seg
        t    = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS, dtype=torch.float64, device=DEVICE)
        pts  = xl.to(torch.float64).unsqueeze(0) + t.unsqueeze(1) * (xr - xl).to(torch.float64).unsqueeze(0)
        z    = composition_to_ilr(pts)
        z    = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts  = ilr_to_composition(z, d=pts.shape[1])
        raw  = np.array([fn_callable(x) for x in pts.detach().cpu().numpy()], float)
        yt   = torch.tensor(raw if maximize else -raw, dtype=DTYPE, device=DEVICE)
        yt   = yt + torch.randn_like(yt) * NOISE_LEVEL
        pts  = pts.to(dtype=DTYPE, device=DEVICE)
        xa.append(pts); xe.append(pts); yl.append(yt)
    if not xa:
        raise RuntimeError("Could not generate any initial simplex lines.")
    return (torch.cat(xa), torch.cat(xe), torch.cat(yl).reshape(-1, 1))


# ── run analytics (mirrors optimize/run_mobo.py metrics + plots) ─────────────

def _metric_dup_fraction(X_all: np.ndarray, threshold: float) -> float:
    n = len(X_all)
    if n <= 1:
        return 0.0
    diff = X_all[:, None, :] - X_all[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    return float((dists < threshold).any(axis=1).mean())


def _metric_avg_pairwise_dist(discovered: np.ndarray) -> float:
    disc = np.asarray(discovered, dtype=float)
    n = len(disc)
    if n < 2:
        return 0.0
    diff = disc[:, None, :] - disc[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    return float(dists[iu].mean())


def _activation_zoom_per_point(n_points: int, snap_records: list[tuple]) -> tuple:
    act = np.zeros(n_points, dtype=int)
    zm = np.zeros(n_points, dtype=int)
    prev = 0
    for (n, a, z) in snap_records:
        n = min(int(n), n_points)
        if n > prev:
            act[prev:n] = a
            zm[prev:n] = z
            prev = n
    if prev < n_points and snap_records:
        act[prev:] = snap_records[-1][1]
        zm[prev:] = snap_records[-1][2]
    return act, zm


def _write_run_analytics(dh, run_dir, payloads, snap_records, maximize, log_fn):
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    run_dir = Path(run_dir) if not isinstance(run_dir, Path) else run_dir
    X_all = dh.X_all_actual.detach().cpu().numpy() if dh.X_all_actual is not None else np.empty((0, 0))
    Y_all = dh.Y_all.detach().cpu().numpy().ravel() if dh.Y_all is not None else np.empty(0)
    d = X_all.shape[1] if X_all.ndim == 2 and X_all.size > 0 else 0

    # ── points.csv ────────────────────────────────────────────────────────
    try:
        n = X_all.shape[0]
        mask = dh.get_penalty_mask()
        penalized = (~mask.detach().cpu().numpy()) if mask is not None else np.zeros(n, bool)
        act, zm = _activation_zoom_per_point(n, snap_records)
        data = {"sample_idx": np.arange(n)}
        for i in range(d):
            data[f"x{i}"] = X_all[:, i]
        data["Y"] = Y_all
        data["penalized"] = penalized.astype(int)
        data["activation"] = act
        data["zoom"] = zm
        pd.DataFrame(data).to_csv(str(run_dir / "points.csv"), index=False)
        log_fn("  Wrote points.csv", tag="info")
    except Exception as exc:
        log_fn(f"  points.csv failed: {exc}", tag="error")

    # ── needles.csv ───────────────────────────────────────────────────────
    try:
        centroid = np.full(d, 1.0 / d) if d > 0 else np.empty(0)
        rows = []
        for i, r in enumerate(dh.get_all_needle_results()):
            pt = r["point"].detach().cpu().numpy().ravel()
            mv = r.get("median_value")
            row = {"needle_idx": i}
            for j in range(d):
                row[f"x{j}"] = pt[j]
            row.update({
                "value": r.get("value"),
                "median_value": (None if mv is None or (isinstance(mv, float) and math.isnan(mv)) else mv),
                "activation": r.get("activation"),
                "zoom": r.get("zoom"),
                "iteration": r.get("iteration"),
                "dist_to_centre": float(np.linalg.norm(pt - centroid)) if d > 0 else 0.0,
            })
            rows.append(row)
        cols = ["needle_idx"] + [f"x{j}" for j in range(d)] + [
            "value", "median_value", "activation", "zoom", "iteration", "dist_to_centre"]
        pd.DataFrame(rows, columns=cols).to_csv(str(run_dir / "needles.csv"), index=False)
        log_fn("  Wrote needles.csv", tag="info")
    except Exception as exc:
        log_fn(f"  needles.csv failed: {exc}", tag="error")

    # ── metrics_over_time.csv ─────────────────────────────────────────────
    try:
        thr = NOISE_LEVEL / 2.0
        met_rows = []
        for p in payloads:
            needles = p.get("needles")
            disc = needles if needles is not None else np.empty((0, d))
            n_before = p.get("n_points_before", len(X_all))
            X_upto = X_all[:n_before] if n_before > 0 else np.empty((0, d))
            nvals = p.get("needle_vals")
            recent = float(nvals[-1]) if nvals is not None and len(nvals) > 0 else np.nan
            met_rows.append({
                "iteration": p["iter_num"],
                "dup_fraction": round(_metric_dup_fraction(X_upto, thr), 6),
                "avg_pairwise_dist": round(_metric_avg_pairwise_dist(disc), 6),
                "recent_needle_value": (round(recent, 6) if not math.isnan(recent) else np.nan),
                "n_needles": len(disc),
                "n_points": n_before,
            })
        pd.DataFrame(met_rows, columns=[
            "iteration", "dup_fraction", "avg_pairwise_dist",
            "recent_needle_value", "n_needles", "n_points",
        ]).to_csv(str(run_dir / "metrics_over_time.csv"), index=False)
        log_fn("  Wrote metrics_over_time.csv", tag="info")
    except Exception as exc:
        log_fn(f"  metrics_over_time.csv failed: {exc}", tag="error")

    # ── dist_from_centre.png ──────────────────────────────────────────────
    try:
        fig = Figure(figsize=(7, 5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        if X_all.shape[0] > 0 and d > 0:
            centroid = np.full(d, 1.0 / d)
            dists = np.linalg.norm(X_all - centroid, axis=1)
            idx = np.arange(len(Y_all))
            sc = ax.scatter(dists, Y_all, c=idx, cmap="viridis", s=14, alpha=0.7, zorder=3, label="samples")
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("sample index", fontsize=8)
            needle_t = dh.get_all_needle_locations()
            nvals_t = dh.get_all_needle_vals()
            if needle_t is not None and needle_t.numel() > 0:
                nd = np.linalg.norm(needle_t.detach().cpu().numpy() - centroid, axis=1)
                nv = nvals_t.detach().cpu().numpy().ravel()
                ax.scatter(nd, nv, marker="*", s=220, color="crimson", edgecolors="darkred",
                           lw=0.8, zorder=5, label="needle")
            ax.set_xlabel("‖x − centroid‖₂")
            ax.set_ylabel("Objective Y" + ("" if maximize else "  (ZoMBI-internal)"))
            ax.set_title("Distance from simplex centre", fontsize=10)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(str(run_dir / "dist_from_centre.png"), dpi=120, bbox_inches="tight")
        log_fn("  Saved dist_from_centre.png", tag="info")
    except Exception as exc:
        log_fn(f"  dist_from_centre.png failed: {exc}", tag="error")

    # ── line_length_hist.png ──────────────────────────────────────────────
    try:
        lengths = []
        for p in payloads:
            line = p.get("line_0")
            if line is not None:
                left, right = np.asarray(line[0]), np.asarray(line[1])
                lengths.append(float(np.linalg.norm(right - left)))
        fig = Figure(figsize=(7, 5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        if lengths:
            ax.hist(lengths, bins=min(40, max(5, len(lengths) // 2)),
                    color="orange", edgecolor="black", alpha=0.8)
            ax.axvline(float(np.mean(lengths)), color="navy", ls="--", lw=1.5,
                       label=f"mean = {np.mean(lengths):.3f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no lines recorded", ha="center")
        ax.set_xlabel("LineBO main-line length  (composition L2)")
        ax.set_ylabel("count")
        ax.set_title("Line length distribution", fontsize=10)
        fig.tight_layout()
        fig.savefig(str(run_dir / "line_length_hist.png"), dpi=120, bbox_inches="tight")
        log_fn("  Saved line_length_hist.png", tag="info")
    except Exception as exc:
        log_fn(f"  line_length_hist.png failed: {exc}", tag="error")

    # ── final summary metrics ─────────────────────────────────────────────
    needle_t = dh.get_all_needle_locations()
    discovered = needle_t.detach().cpu().numpy() if needle_t is not None and needle_t.numel() > 0 else np.empty((0, d))
    dup = _metric_dup_fraction(X_all, NOISE_LEVEL / 2.0)
    avg_pw = _metric_avg_pairwise_dist(discovered)
    nvals_t = dh.get_all_needle_vals()
    recent_nv = float(nvals_t[-1].item()) if nvals_t is not None and nvals_t.numel() > 0 else float('nan')
    log_fn(f"  Final: dup_frac={dup:.4f}  avg_pw_dist={avg_pw:.4f}"
           f"  recent_needle={recent_nv:.4f}  needles={len(discovered)}"
           f"  points={len(X_all)}", tag="info")


# ── data loading ──────────────────────────────────────────────────────────────

class RunData:
    """State loaded from one ZoMBI-Hop snapshot."""

    def __init__(self):
        self.run_id        = ""
        self.run_dir: Path = Path(".")
        self.config: dict  = {}
        self.snapshot_name = ""
        self.snapshots: list[str] = []
        self.X_all: Optional[np.ndarray]         = None  # (n, d)
        self.Y_all: Optional[np.ndarray]         = None  # (n,)
        self.penalty_mask: Optional[np.ndarray]  = None  # (n,) bool
        self.needles: Optional[np.ndarray]       = None  # (k, d)
        self.needle_vals: Optional[np.ndarray]   = None  # (k,)
        self.needle_indices: Optional[np.ndarray]= None  # (k,) int
        self.needles_json: list                  = []
        self.summary: dict                       = {}
        self.d: int                              = 0
        self.log_lines: list[str]                = []
        # Trust region + penalty-ellipsoid params (for the ternary overlays)
        self.zoom_bounds: Optional[np.ndarray]   = None  # (2, d) composition AABB
        self.needle_M_list: list                 = []    # per-needle (d-1,d-1) or None
        self.needle_B: Optional[np.ndarray]      = None  # (d, d-1) or None (ILR mode)

    @property
    def n_points(self):
        return len(self.X_all) if self.X_all is not None else 0

    @property
    def n_needles(self):
        return len(self.needles) if self.needles is not None else 0

    @property
    def centroid_dist(self) -> Optional[np.ndarray]:
        if self.X_all is None or self.d == 0:
            return None
        return np.linalg.norm(self.X_all - np.full(self.d, 1.0 / self.d), axis=1)


def _list_snapshots(run_dir: Path) -> list[str]:
    snap_dir = run_dir / "snapshots"
    return sorted(s.name for s in snap_dir.iterdir() if s.is_dir()) if snap_dir.exists() else []


def load_run(run_dir: Path, snapshot_name: Optional[str] = None) -> RunData:
    rd = RunData()
    rd.run_id  = run_dir.name
    rd.run_dir = run_dir

    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            rd.config = json.loads(cfg.read_text())
        except Exception:
            pass

    rd.snapshots = _list_snapshots(run_dir)
    if not rd.snapshots:
        return rd

    if snapshot_name is None:
        latest = run_dir / "latest.txt"
        snapshot_name = latest.read_text().strip() if latest.exists() else rd.snapshots[-1]
    if snapshot_name not in rd.snapshots:
        snapshot_name = rd.snapshots[-1]
    rd.snapshot_name = snapshot_name

    snap = run_dir / "snapshots" / snapshot_name

    # Load tensors — supports both new delta format and legacy tensors.pt
    try:
        s = reconstruct_snapshot_tensors(run_dir, snapshot_name, device="cpu")

        def _npy(key, ravel=False):
            t = s.get(key)
            if t is None:
                return None
            a = t.float().numpy()
            return a.ravel() if ravel else a

        rd.X_all        = _npy("X_all_actual")
        rd.Y_all        = _npy("Y_all", ravel=True)
        rd.needles      = _npy("needles")
        rd.needle_vals  = _npy("needle_vals", ravel=True)
        pm = s.get("penalty_mask")
        if pm is not None:
            rd.penalty_mask = pm.bool().numpy().ravel()
        ni = s.get("needle_indices")
        if ni is not None:
            rd.needle_indices = ni.long().numpy().ravel()

        # Trust region (current zoom bounds, fall back to full bounds) +
        # per-needle penalty-ellipsoid params, for the ternary overlays.
        bd = s.get("current_zoom_bounds")
        if bd is None:
            bd = s.get("bounds")
        if bd is not None:
            rd.zoom_bounds = bd.float().numpy()
        m_stack = s.get("needle_M_stack")
        has_m   = s.get("needle_has_M")
        if (isinstance(m_stack, torch.Tensor) and isinstance(has_m, torch.Tensor)
                and has_m.numel() > 0):
            rd.needle_M_list = [
                m_stack[i].float().numpy() if bool(has_m[i].item()) else None
                for i in range(has_m.shape[0])
            ]
        nb = s.get("needle_B")
        if isinstance(nb, torch.Tensor):
            rd.needle_B = nb.float().numpy()
    except Exception as e:
        print(f"[warn] tensors load: {e}")

    np_ = snap / "needles.json"
    if np_.exists():
        try:
            rd.needles_json = json.loads(np_.read_text())
        except Exception:
            pass

    sp = snap / "summary.json"
    if sp.exists():
        try:
            rd.summary = json.loads(sp.read_text())
        except Exception:
            pass

    if rd.X_all is not None:
        rd.d = rd.X_all.shape[1]

    log_path = run_dir / "run.log"
    if log_path.exists():
        try:
            rd.log_lines = log_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            pass

    return rd


def scan_runs(base_dir: str) -> list[dict]:
    base = Path(base_dir)
    if not base.exists():
        return []
    runs = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        latest = d / "latest.txt"
        mtime  = latest.stat().st_mtime if latest.exists() else d.stat().st_mtime
        try:
            cfg = json.loads((d / "config.json").read_text())
        except Exception:
            cfg = {}
        runs.append({"run_id": d.name, "run_dir": d, "mtime": mtime, "config": cfg})
    return sorted(runs, key=lambda r: r["mtime"], reverse=True)


# ── embedded matplotlib figure ────────────────────────────────────────────────

class PlotFrame(ttk.Frame):
    def __init__(self, parent, figsize=(7, 4), **kwargs):
        super().__init__(parent, **kwargs)
        self.fig    = Figure(figsize=figsize, dpi=96, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(self.canvas, self)
        tb.update()

    def draw(self):
        self.canvas.draw_idle()


# ── run browser ───────────────────────────────────────────────────────────────

class RunBrowserPanel(ttk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app  = app
        self._runs: list[dict] = []

        ttk.Label(self, text="Checkpoint directory:").pack(anchor="w", padx=4, pady=(4, 0))
        df = ttk.Frame(self)
        df.pack(fill="x", padx=4)
        self._dir_var = tk.StringVar(value=app.ckpt_dir)
        ttk.Entry(df, textvariable=self._dir_var, width=28).pack(side="left", fill="x", expand=True)
        ttk.Button(df, text="…", width=3, command=self._browse).pack(side="left")

        bf = ttk.Frame(self)
        bf.pack(fill="x", padx=4, pady=2)
        _bw = 8
        ttk.Button(bf, text="Refresh",  width=_bw, command=self.refresh).pack(side="left", padx=2)
        ttk.Button(bf, text="Load",     width=_bw, command=self._load_selected).pack(side="left", padx=2)
        ttk.Button(bf, text="New Run",  width=_bw, command=app.open_new_run_dialog).pack(side="left", padx=2)
        ttk.Button(bf, text="Stop Run", width=_bw, command=self._stop_selected,
                   style="Danger.TButton").pack(side="left", padx=2)
        ttk.Button(bf, text="Delete",   width=_bw, command=self._delete_selected,
                   style="Danger.TButton").pack(side="left", padx=2)

        # Active hardware-run UUID indicator
        self._hw_uuid_var = tk.StringVar(value="")
        self._hw_uuid_lbl = ttk.Label(
            self, textvariable=self._hw_uuid_var,
            foreground="#007700", font=("Consolas", 8), wraplength=290)
        self._hw_uuid_lbl.pack(anchor="w", padx=4, pady=(0, 2))

        lf = ttk.Frame(self)
        lf.pack(fill="both", expand=True, padx=4, pady=2)
        sb = ttk.Scrollbar(lf, orient="vertical")
        sb.pack(side="right", fill="y")
        self._lb = tk.Listbox(lf, yscrollcommand=sb.set, selectmode="single",
                               font=("Consolas", 8))
        self._lb.pack(side="left", fill="both", expand=True)
        sb.config(command=self._lb.yview)
        self._lb.bind("<<ListboxSelect>>", self._on_select)
        self._lb.bind("<Double-Button-1>", lambda _e: self._load_selected())
        self.refresh()

    def _browse(self):
        d = filedialog.askdirectory(initialdir=self._dir_var.get())
        if d:
            self._dir_var.set(d)
            self._app.ckpt_dir = d
            self.refresh()

    def refresh(self):
        self._app.ckpt_dir = self._dir_var.get()
        self._runs = scan_runs(self._app.ckpt_dir)
        active_ids = set(self._app._active_runs.keys())
        viewed_id  = self._app._viewed_run_id

        # Remember which index was selected so we can restore it
        prev_sel = self._lb.curselection()
        prev_idx = prev_sel[0] if prev_sel else None

        self._lb.delete(0, "end")
        import datetime
        for i, r in enumerate(self._runs):
            ts  = datetime.datetime.fromtimestamp(r["mtime"]).strftime("%m-%d %H:%M")
            rid = r["run_id"]
            if rid in active_ids:
                label = f"▶ {rid}  [{ts}]  RUNNING"
            else:
                label = f"   {rid}  [{ts}]"
            self._lb.insert("end", label)
            # Colour-code running and viewed entries
            if rid in active_ids:
                self._lb.itemconfig(i, foreground="#007700", selectforeground="#007700")
            if rid == viewed_id:
                self._lb.itemconfig(i, background="#ddeeff", selectbackground="#aaccff")

        # Restore selection if the same index still exists
        if prev_idx is not None and prev_idx < self._lb.size():
            self._lb.selection_set(prev_idx)

    def _on_select(self, _event=None):
        """Single-click: load the run data and show its log without waiting for double-click."""
        sel = self._lb.curselection()
        if sel:
            self._app.load_run(self._runs[sel[0]]["run_dir"])

    def _load_selected(self):
        sel = self._lb.curselection()
        if not sel:
            messagebox.showinfo("Select", "Click a run first.")
            return
        HardwareResumeDialog(self, self._app, self._runs[sel[0]])

    def _delete_selected(self):
        sel = self._lb.curselection()
        if not sel:
            messagebox.showinfo("Select", "Click a run first.")
            return
        run_info = self._runs[sel[0]]
        run_id  = run_info["run_id"]
        run_dir = run_info["run_dir"]

        # Don't delete a run that is currently executing
        if run_id in self._app._active_runs:
            messagebox.showwarning("Active run",
                                   "Cannot delete a run that is currently executing.")
            return

        if not messagebox.askyesno("Delete run",
                                   f"Permanently delete run  {run_id} ?\n\n"
                                   f"{run_dir}\n\nThis cannot be undone."):
            return
        try:
            shutil.rmtree(str(run_dir))
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))
            return

        # If this was the displayed run, clear the UI
        if self._app.current_run and self._app.current_run.run_id == run_id:
            self._app.current_run = None
            self._app._viewed_run_id = None
            self._app._live_log.clear()
            self._app._refresh_pause_btn()
            self._app.set_status("Run deleted.")

        self.refresh()

    def _stop_selected(self):
        sel = self._lb.curselection()
        if not sel:
            messagebox.showinfo("Select", "Click a running run first.")
            return
        run_info = self._runs[sel[0]]
        run_id = run_info["run_id"]
        if run_id not in self._app._active_runs:
            messagebox.showinfo("Not running", f"{run_id} is not currently running.")
            return
        if not messagebox.askyesno("Stop run",
                                    f"Stop the active run  {run_id} ?\n\n"
                                    "The run will be terminated after the current iteration."):
            return
        self._app.stop_run(run_id)

    def set_hw_uuid(self, uuid_or_none: str | None):
        """Update the active-hardware-run UUID indicator in the left panel."""
        if uuid_or_none:
            self._hw_uuid_var.set(f"HW run: {uuid_or_none}")
        else:
            self._hw_uuid_var.set("")


# ── config / summary panel ────────────────────────────────────────────────────

class ConfigPanel(ttk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Run Info", **kwargs)
        self._text = scrolledtext.ScrolledText(
            self, width=32, height=10, font=("Consolas", 8), state="disabled")
        self._text.pack(fill="both", expand=True, padx=2, pady=2)

    def update(self, rd: RunData):
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        if rd.config:
            self._text.insert("end", "── config ──\n")
            for k, v in rd.config.items():
                self._text.insert("end", f"  {k}: {v}\n")
        if rd.summary:
            self._text.insert("end", "\n── snapshot ──\n")
            for k, v in rd.summary.items():
                self._text.insert("end", f"  {k}: {v}\n")
        self._text.insert("end", f"\n  d = {rd.d}\n")
        self._text.insert("end", f"  points = {rd.n_points}\n")
        self._text.insert("end", f"  needles = {rd.n_needles}\n")
        self._text.config(state="disabled")


# ── snapshot slider ───────────────────────────────────────────────────────────

class SnapshotSliderFrame(ttk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app = app
        self._snapshots: list[str] = []
        self._slide_job: Optional[str] = None

        ttk.Label(self, text="Snapshot:").pack(side="left", padx=4)
        self._var = tk.IntVar(value=0)
        self._slider = ttk.Scale(self, from_=0, to=0, orient="horizontal",
                                  variable=self._var, command=self._on_slide)
        self._slider.pack(side="left", fill="x", expand=True, padx=4)
        self._lbl = ttk.Label(self, text="", width=26, font=("Consolas", 8))
        self._lbl.pack(side="left", padx=4)

    def set_snapshots(self, snapshots: list[str], current: str):
        self._snapshots = snapshots
        if not snapshots:
            self._slider.config(to=0)
            self._lbl.config(text="")
            return
        self._slider.config(to=max(len(snapshots) - 1, 0))
        idx = snapshots.index(current) if current in snapshots else len(snapshots) - 1
        self._var.set(idx)
        self._lbl.config(text=current)

    def _on_slide(self, val):
        idx = int(float(val))
        if 0 <= idx < len(self._snapshots):
            self._lbl.config(text=self._snapshots[idx])
        if self._slide_job:
            self.after_cancel(self._slide_job)
        self._slide_job = self.after(350, self._fire_load)

    def _fire_load(self):
        idx = int(self._var.get())
        if 0 <= idx < len(self._snapshots):
            self._app.load_snapshot(self._snapshots[idx])


# ── convergence plot ──────────────────────────────────────────────────────────

class ConvergencePlotFrame(PlotFrame):
    def update(self, rd: RunData):
        self.fig.clf()
        if rd.Y_all is None or len(rd.Y_all) == 0:
            self.draw(); return
        ax  = self.fig.add_subplot(111)
        Y   = rd.Y_all
        idx = np.arange(len(Y))
        pm  = rd.penalty_mask

        if pm is not None and pm.any():
            ax.scatter(idx[~pm], Y[~pm], s=10, alpha=0.35, color="#aaaaaa",
                       label="penalized", zorder=2)
            ax.scatter(idx[pm],  Y[pm],  s=10, alpha=0.65, color="steelblue",
                       label="valid", zorder=3)
            running_best = np.maximum.accumulate(np.where(pm, Y, -np.inf))
        else:
            ax.scatter(idx, Y, s=10, alpha=0.65, color="steelblue", label="obs", zorder=2)
            running_best = np.maximum.accumulate(Y)

        ax.plot(idx, running_best, color="darkorange", lw=1.8,
                label="running best", zorder=4)

        if rd.needle_indices is not None and len(rd.needle_indices) > 0:
            labeled = False
            for ni in rd.needle_indices:
                if 0 <= ni < len(Y):
                    kw = dict(color="crimson", alpha=0.55, lw=0.9, ls="--")
                    if not labeled:
                        kw["label"] = "needle found"
                        labeled = True
                    ax.axvline(float(ni), **kw)

        ax.set_xlabel("Sample index")
        ax.set_ylabel("Objective Y")
        ax.set_title(f"{rd.run_id} — Convergence  "
                     f"(snap: {rd.snapshot_name},  {rd.n_points} pts, {rd.n_needles} needles)",
                     fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        self.draw()


# ── distance-from-centre plot ─────────────────────────────────────────────────

class DistancePlotFrame(PlotFrame):
    def update(self, rd: RunData):
        self.fig.clf()
        if rd.Y_all is None or rd.X_all is None or rd.d == 0:
            self.draw(); return
        ax    = self.fig.add_subplot(111)
        dists = rd.centroid_dist
        Y     = rd.Y_all
        idx   = np.arange(len(Y))

        sc = ax.scatter(dists, Y, c=idx, cmap="viridis", s=14, alpha=0.7,
                        zorder=3, label="samples")
        cb = self.fig.colorbar(sc, ax=ax)
        cb.set_label("sample index", fontsize=8)

        if rd.needles is not None and len(rd.needles) > 0:
            centroid  = np.full(rd.d, 1.0 / rd.d)
            nd = np.linalg.norm(rd.needles - centroid, axis=1)
            nv = rd.needle_vals if rd.needle_vals is not None else np.zeros(len(rd.needles))
            ax.scatter(nd, nv, marker="*", s=220, color="crimson",
                       edgecolors="darkred", lw=0.8, zorder=5, label="needle")

        ax.set_xlabel("‖x − centroid‖₂")
        ax.set_ylabel("Objective Y")
        ax.set_title(f"{rd.run_id} — Distance from simplex centre", fontsize=9)
        ax.legend(fontsize=8)
        self.draw()


# ── points table ──────────────────────────────────────────────────────────────

class PointsTableFrame(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._inner = ttk.Frame(self)
        self._inner.pack(fill="both", expand=True)

    def update(self, rd: RunData):
        for w in self._inner.winfo_children():
            w.destroy()
        if rd.X_all is None:
            ttk.Label(self._inner, text="No data loaded.").pack(pady=20)
            return

        d    = rd.d
        cols = ["#"] + [f"x[{i}]" for i in range(d)] + ["Y", "valid"]
        tree = ttk.Treeview(self._inner, columns=cols, show="headings", height=22)
        for c in cols:
            tree.heading(c, text=c)
            w = 40 if c == "#" else (65 if c.startswith("x[") else 80)
            tree.column(c, width=w, anchor="e", minwidth=w)

        Y  = rd.Y_all
        pm = rd.penalty_mask
        for i, (x, y) in enumerate(zip(rd.X_all, Y)):
            comp = [f"{v:.4f}" for v in x]
            ok   = "✓" if (pm is not None and i < len(pm) and pm[i]) else ""
            tree.insert("", "end", values=[str(i)] + comp + [f"{y:.5f}", ok])

        sb_y = ttk.Scrollbar(self._inner, orient="vertical",   command=tree.yview)
        sb_x = ttk.Scrollbar(self._inner, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x.grid(row=1, column=0, sticky="ew")
        self._inner.rowconfigure(0, weight=1)
        self._inner.columnconfigure(0, weight=1)


# ── needles panel ─────────────────────────────────────────────────────────────

class NeedlesFrame(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        pw = ttk.PanedWindow(self, orient="vertical")
        pw.pack(fill="both", expand=True)

        top = ttk.Frame(pw)
        pw.add(top, weight=3)
        self._tree = ttk.Treeview(top, show="headings", height=12)
        sb = ttk.Scrollbar(top, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        top.rowconfigure(0, weight=1); top.columnconfigure(0, weight=1)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        bot = ttk.LabelFrame(pw, text="Needle detail")
        pw.add(bot, weight=1)
        self._detail = scrolledtext.ScrolledText(
            bot, height=7, font=("Consolas", 9), state="disabled")
        self._detail.pack(fill="both", expand=True, padx=2, pady=2)

        self._data: list = []

    def update(self, rd: RunData):
        self._tree.delete(*self._tree.get_children())
        self._data = rd.needles_json
        if not self._data:
            self._tree["columns"] = ["info"]
            self._tree.heading("info", text="No needles")
            self._tree.column("info", width=300)
            return

        d    = rd.d
        ccols = [f"x[{i}]" for i in range(d)]
        cols  = ["#"] + ccols + ["value", "med_val", "activation", "zoom", "iter"]
        self._tree["columns"] = cols
        for c in cols:
            self._tree.heading(c, text=c)
            w = 35 if c == "#" else (60 if c.startswith("x[") else 75)
            self._tree.column(c, width=w, anchor="e", minwidth=w)

        for i, n in enumerate(self._data):
            pt   = n.get("point", [])
            comp = [f"{v:.4f}" for v in pt]
            val  = f"{n['value']:.5f}"   if n.get("value")        is not None else ""
            mv   = f"{n['median_value']:.5f}" if n.get("median_value") is not None else ""
            self._tree.insert("", "end", iid=str(i),
                              values=[str(i)] + comp + [val, mv,
                                      n.get("activation", ""),
                                      n.get("zoom", ""),
                                      n.get("iteration", "")])

    def _on_select(self, _e):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self._data):
            return
        n = self._data[idx]
        self._detail.config(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("end", json.dumps(n, indent=2))
        self._detail.config(state="disabled")


# ── GP query panel ────────────────────────────────────────────────────────────

class GPQueryFrame(ttk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self._app     = app
        self._gp      = None
        self._gp_lock = threading.Lock()
        self._gp_rid  = ""
        self._entries: list[tk.StringVar] = []
        self._d_built = 0

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="1. Fit GP on the loaded run's points:").pack(anchor="w")
        bf = ttk.Frame(top)
        bf.pack(anchor="w", pady=2)
        self._fit_btn = ttk.Button(bf, text="Fit GP", command=self._fit_bg)
        self._fit_btn.pack(side="left")
        self._fit_lbl = tk.StringVar(value="(not fitted)")
        ttk.Label(bf, textvariable=self._fit_lbl, foreground="gray").pack(side="left", padx=8)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=8, pady=4)

        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=8)
        ttk.Label(mid, text="2. Enter a composition then click Predict:").pack(anchor="w")

        self._ef = ttk.LabelFrame(mid, text="Composition x[ ]")
        self._ef.pack(fill="x", pady=4)

        nb_f = ttk.Frame(mid)
        nb_f.pack(anchor="w", pady=2)
        ttk.Button(nb_f, text="Normalise (÷ sum)",  command=self._normalise).pack(side="left")
        ttk.Button(nb_f, text="Set to centroid",    command=self._set_centroid).pack(side="left", padx=4)

        self._pred_btn = ttk.Button(mid, text="Predict →", command=self._predict)
        self._pred_btn.pack(anchor="w", pady=4)

        self._result_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._result_var,
                  font=("Consolas", 12, "bold"), foreground="navy").pack(anchor="w", padx=8, pady=6)

        self._sum_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._sum_var, foreground="gray").pack(anchor="w", padx=8)

    def _rebuild_entries(self, d: int):
        for w in self._ef.winfo_children():
            w.destroy()
        self._entries = []
        self._d_built = d
        n_cols = 4
        for i in range(d):
            r, c = divmod(i, n_cols)
            ttk.Label(self._ef, text=f"x[{i}]:", width=6).grid(
                row=r, column=2*c, sticky="e", padx=2, pady=1)
            v = tk.StringVar(value=f"{1.0/d:.6f}")
            ttk.Entry(self._ef, textvariable=v, width=10).grid(
                row=r, column=2*c+1, sticky="w", padx=2)
            self._entries.append(v)
        for i in range(n_cols):
            self._ef.columnconfigure(2*i+1, weight=1)

    def _get_comp(self) -> Optional[np.ndarray]:
        try:
            v = np.array([float(e.get()) for e in self._entries])
            return v if np.all(np.isfinite(v)) else None
        except Exception:
            return None

    def _normalise(self):
        v = self._get_comp()
        if v is None:
            return
        s = v.sum()
        if s > 0:
            v /= s
        for var, val in zip(self._entries, v):
            var.set(f"{val:.6f}")
        self._update_sum_label()

    def _set_centroid(self):
        d = self._d_built
        if d > 0:
            for var in self._entries:
                var.set(f"{1.0/d:.6f}")
        self._update_sum_label()

    def _update_sum_label(self):
        v = self._get_comp()
        if v is not None:
            self._sum_var.set(f"sum = {v.sum():.6f}")

    def update_for_run(self, rd: RunData):
        if rd.d != self._d_built:
            self._rebuild_entries(max(rd.d, 1))
        if rd.run_id != self._gp_rid:
            with self._gp_lock:
                self._gp = None
            self._gp_rid = ""
            self._fit_lbl.set("(not fitted — click 'Fit GP')")
            self._result_var.set("")

    def _fit_bg(self):
        rd = self._app.current_run
        if rd is None or rd.X_all is None or rd.Y_all is None:
            messagebox.showwarning("No data", "Load a run first.")
            return
        if not _BOTORCH_OK:
            messagebox.showerror("Missing", "botorch not installed.")
            return
        self._fit_btn.config(state="disabled")
        self._fit_lbl.set("Fitting…")
        _X, _Y, _rid = rd.X_all.copy(), rd.Y_all.copy(), rd.run_id

        def _work():
            try:
                X_ilr = composition_to_ilr(
                    torch.tensor(_X, dtype=torch.float64)
                ).to(dtype=DTYPE, device=DEVICE)
                Y_t = torch.tensor(_Y, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
                gp  = SingleTaskGP(X_ilr, Y_t, outcome_transform=Standardize(m=1))
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                gp.eval()
                with self._gp_lock:
                    self._gp = gp
                self._gp_rid = _rid
                self.after(0, lambda: (
                    self._fit_lbl.set(f"GP fitted on {len(_X)} pts"),
                    self._fit_btn.config(state="normal"),
                ))
            except Exception as exc:
                self.after(0, lambda: (
                    self._fit_lbl.set(f"Fit failed: {exc}"),
                    self._fit_btn.config(state="normal"),
                ))

        threading.Thread(target=_work, daemon=True).start()

    def _predict(self):
        with self._gp_lock:
            gp = self._gp
        if gp is None:
            messagebox.showwarning("Not fitted", "Click 'Fit GP' first.")
            return
        v = self._get_comp()
        if v is None:
            messagebox.showwarning("Input error", "Enter valid numbers.")
            return
        s = v.sum()
        self._sum_var.set(f"sum = {s:.6f}")
        if abs(s - 1.0) > 0.05:
            if messagebox.askyesno("Normalise?", f"Sum = {s:.4f}. Normalise to 1?"):
                v /= s
            else:
                return
        elif abs(s - 1.0) > 1e-6:
            v /= s
        try:
            x_ilr = composition_to_ilr(
                torch.tensor(v, dtype=torch.float64).unsqueeze(0)
            ).to(dtype=DTYPE, device=DEVICE)
            with torch.no_grad():
                post = gp.posterior(x_ilr)
                mean = post.mean.item()
                std  = post.variance.sqrt().item()
            self._result_var.set(f"mean = {mean:.5f}    std = {std:.5f}")
        except Exception as exc:
            self._result_var.set(f"Error: {exc}")


# ── ternary plot tab ─────────────────────────────────────────────────────────


def _sample_bounds_comp(lo, hi, n_sample: int = 4000):
    """
    Uniformly sample compositions inside the trust-region box ``[lo, hi]`` on the
    3-simplex (composition space, NOT ILR). Mirrors ``_draw_bounds_region`` in
    interactive_test_zombi.py so the GUI shows the same dashed-red trust region.
    Returns an ``(N, 3)`` composition array, or None on failure.
    """
    from src.utils.simplex import random_simplex
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    if lo.shape[0] != 3 or hi.shape[0] != 3:
        return None
    try:
        samp = random_simplex(
            n_sample,
            torch.tensor(lo, dtype=torch.float64),
            torch.tensor(hi, dtype=torch.float64),
            device="cpu", torch_dtype=torch.float64,
        )
    except Exception:
        return None
    pts = samp.detach().cpu().numpy()
    return pts if (pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] == 3) else None


def _needle_ellipsoid_comp(needle_comp, M, B, n: int = 200):
    """
    Penalization-ellipsoid boundary as ``(n, 3)`` compositions. Direct port of
    ``_draw_needle_ellipsoid`` in interactive_test_zombi.py:

      tangent-space mode (B given): x = x* + B @ u,  u^T M u = 1
      ILR mode      (B is None):    x = ilr⁻¹(ilr(x*) + u),  u^T M u = 1

    Returns None if ``M`` is None or on any failure.
    """
    if M is None:
        return None
    try:
        needle_comp = np.asarray(needle_comp, dtype=float).ravel()
        d = needle_comp.shape[0]
        M_np = np.asarray(M, dtype=float)
        eigvals, eigvecs = np.linalg.eigh(M_np)
        eigvals = np.maximum(eigvals, 1e-12)
        angles = np.linspace(0, 2 * np.pi, n)
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        u_ell = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ circle.T).T   # (n, d-1)
        if B is not None:
            B_np = np.asarray(B, dtype=float)               # (d, d-1)
            ell = needle_comp.reshape(1, d) + (B_np @ u_ell.T).T
        else:
            needle_ilr = composition_to_ilr(
                torch.tensor(needle_comp, dtype=torch.float64).unsqueeze(0)
            ).squeeze(0).cpu().numpy()                       # (d-1,)
            z_ell = needle_ilr + u_ell                       # (n, d-1)
            ell = ilr_to_composition(
                torch.tensor(z_ell, dtype=torch.float64), d).cpu().numpy()
        ell = np.clip(ell, 0, 1)
        s = ell.sum(axis=1, keepdims=True)
        return ell / np.where(s < 1e-9, 1.0, s)
    except Exception:
        return None


class TernaryPlotFrame(ttk.Frame):
    """
    Ternary-plot tab.

    • d < 3  → placeholder message; no plot.
    • d ≥ 3  → dim selectors always visible; user picks which hardware dim goes
               to which vertex.  Selections are saved per run-id so they persist
               when switching tabs or snapshots.
    • "Update" button commits the current selection and re-renders.

    Dim labels always reflect the *hardware* indices (e.g. dim-0, dim-8, dim-9),
    read from live_plot_state.json → hw_config.json → config.json in that order.
    """

    _VERTEX_LABELS = ["A  (bottom-left)", "B  (bottom-right)", "C  (top)"]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # ── dim-selector row (always packed for d ≥ 3) ───────────────────
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", padx=4, pady=(3, 1))
        ttk.Label(ctrl, text="Vertex dims:").pack(side="left", padx=(4, 6))

        # Use StringVars so the combobox text IS the human-readable label.
        self._col_svars: list[tk.StringVar] = [tk.StringVar() for _ in range(3)]
        self._combos:    list[ttk.Combobox] = []
        for svar, vtx in zip(self._col_svars, self._VERTEX_LABELS):
            ttk.Label(ctrl, text=f"{vtx}: ").pack(side="left")
            cb = ttk.Combobox(ctrl, textvariable=svar, width=8, state="readonly")
            cb.pack(side="left", padx=(0, 8))
            self._combos.append(cb)

        ttk.Button(ctrl, text="Update", command=self._on_update).pack(side="left", padx=(4, 0))
        self._ctrl = ctrl

        # ── matplotlib figure ─────────────────────────────────────────────
        self.fig    = Figure(figsize=(8, 6), dpi=96, tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(self.canvas, self)
        tb.update()

        # ── internal state ────────────────────────────────────────────────
        self._last_rd: Optional[RunData]       = None
        self._live_state: Optional[dict]       = None
        self._live_state_mtime: float          = 0.0
        self._saved_dims: dict[str, list[str]] = {}   # run_id → [labelA, labelB, labelC]
        self._current_labels: list[str]        = []   # labels for the current data set

    # ── public API ────────────────────────────────────────────────────────

    def draw(self):
        self.canvas.draw_idle()

    def try_reload_live_state(self, rd: "RunData") -> bool:
        """Re-read live_plot_state.json if its mtime changed. Returns True if changed."""
        if rd is None:
            return False
        p = rd.run_dir / "live_plot_state.json"
        if not p.exists():
            if self._live_state is not None:
                self._live_state       = None
                self._live_state_mtime = 0.0
                return True
            return False
        try:
            mt = p.stat().st_mtime
            if mt == self._live_state_mtime:
                return False
            self._live_state       = json.loads(p.read_text())
            self._live_state_mtime = mt
            return True
        except Exception:
            return False

    def update(self, rd: "RunData"):
        """Called by the app whenever a run or snapshot changes."""
        self._last_rd = rd
        self.try_reload_live_state(rd)
        X, n_cols, labels = self._data_info()
        self._current_labels = labels
        self._sync_combos(labels, n_cols, rd)
        self._replot(X, n_cols, labels)

    # ── dim-selector helpers ──────────────────────────────────────────────

    def _data_info(self):
        """
        Return (X, n_cols, labels).

        X      – ndarray (n, n_cols), may be empty
        n_cols – int
        labels – list[str] of length n_cols using actual hardware dim names
                 (e.g. ["dim-0","dim-8","dim-9"] not ["d0","d1","d2"])
        """
        # Priority 1: live_plot_state.json
        if self._live_state is not None:
            raw      = self._live_state.get("x_actual", [])
            X        = np.array(raw, dtype=float) if raw else np.empty((0, 0))
            n        = X.shape[1] if X.ndim == 2 and X.size > 0 else 0
            opt_dims = self._live_state.get("optimizing_dims", list(range(n)))
            labels   = [f"dim-{d}" for d in opt_dims]
            return X, n, labels

        rd = self._last_rd
        if rd is None or rd.X_all is None or rd.d < 1:
            return np.empty((0, 0)), 0, []

        # Priority 2: read actual hardware dim indices from config files
        opt_dims = self._read_hw_dims(rd)
        if opt_dims is not None and len(opt_dims) == rd.d:
            labels = [f"dim-{d}" for d in opt_dims]
        else:
            labels = [f"dim-{i}" for i in range(rd.d)]

        return rd.X_all, rd.d, labels

    def _read_hw_dims(self, rd: "RunData") -> Optional[list]:
        """Try to read hardware optimizing dims from config files in the run dir."""
        for fname in ("hw_config.json", "config.json"):
            p = rd.run_dir / fname
            if not p.exists():
                continue
            try:
                data     = json.loads(p.read_text())
                dims_raw = data.get("dims", "")
                if not dims_raw:
                    continue
                parsed = [int(x.strip()) for x in str(dims_raw).split(",")
                          if x.strip().lstrip("-").isdigit()]
                if parsed:
                    return parsed
            except Exception:
                pass
        return None

    def _sync_combos(self, labels: list[str], n_cols: int, rd: "RunData"):
        """Rebuild combobox choices and restore any saved selection for this run."""
        for cb in self._combos:
            cb["values"] = labels if n_cols >= 3 else []

        if n_cols < 3:
            return

        # Restore saved selection for this run if available, else use first 3 labels
        run_id = rd.run_id if rd else ""
        if run_id and run_id in self._saved_dims:
            saved = self._saved_dims[run_id]
            for svar, saved_lbl in zip(self._col_svars, saved):
                svar.set(saved_lbl if saved_lbl in labels else "")
        else:
            for i, svar in enumerate(self._col_svars):
                current = svar.get()
                if current not in labels:
                    svar.set(labels[min(i, len(labels) - 1)])

    def _on_update(self):
        """Update button: save selection then replot."""
        rd = self._last_rd
        if rd and rd.run_id:
            self._saved_dims[rd.run_id] = [v.get() for v in self._col_svars]
        X, n_cols, labels = self._data_info()
        self._current_labels = labels
        self._replot(X, n_cols, labels)

    def _resolve_cols(self, labels: list[str]):
        """
        Map the 3 StringVar selections back to column indices.
        Returns (da, db, dc) or None if any selection is invalid/duplicate.
        """
        cols = []
        for svar in self._col_svars:
            lbl = svar.get()
            if lbl not in labels:
                return None
            cols.append(labels.index(lbl))
        da, db, dc = cols
        if len({da, db, dc}) < 3:
            return None
        return da, db, dc

    # ── rendering ─────────────────────────────────────────────────────────

    def _replot_tetra(self, X: np.ndarray, labels: list[str]):
        """
        d == 4 view: project the 4-simplex onto a regular tetrahedron and show a
        3D point cloud of the *sampled* points only (coloured by objective),
        plus discovered needles and the latest LineBO lines.

        This simulates a real run, so the ground-truth objective lattice/peaks
        are intentionally NOT drawn — only what the optimizer has actually
        visited appears.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

        self.fig.clf()
        ax = self.fig.add_subplot(111, projection="3d")
        rd   = self._last_rd
        live = self._live_state
        V    = _TETRA_VERTICES
        vlabels = (labels[:4] if len(labels) >= 4
                   else [f"x{i + 1}" for i in range(4)])

        # Tetrahedron wireframe + corner labels for orientation.
        for i in range(4):
            for j in range(i + 1, 4):
                ax.plot([V[i, 0], V[j, 0]], [V[i, 1], V[j, 1]], [V[i, 2], V[j, 2]],
                        color="0.6", lw=1.0, zorder=1)
        lp = V * 1.12
        for k in range(4):
            ax.text(lp[k, 0], lp[k, 1], lp[k, 2], vlabels[k],
                    fontsize=9, fontweight="bold", ha="center", va="center")

        legend_handles: list = []

        # ── sampled points (no ground-truth background cloud) ─────────────
        if X.shape[0] > 0 and X.shape[1] >= 4:
            P = X[:, :4] @ V
            Y = None
            if live is not None:
                yraw = live.get("y_values", [])
                if len(yraw) == P.shape[0]:
                    Y = np.array(yraw, dtype=float)
            elif rd is not None and rd.Y_all is not None and len(rd.Y_all) == P.shape[0]:
                Y = rd.Y_all
            if Y is not None:
                sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=Y, cmap="viridis",
                                s=18, depthshade=True, zorder=3)
                cb = self.fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
                cb.set_label("Objective Y", fontsize=8)
            else:
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=18,
                           color="steelblue", zorder=3)
            legend_handles.append(matplotlib.lines.Line2D(
                [], [], linestyle="none", marker="o", markersize=6,
                color="steelblue", label=f"Samples ({P.shape[0]})"))

        # ── LineBO lines (live runs only; snapshots don't carry them) ─────
        def _draw_lines(key, style, color, label):
            drawn = False
            for ep in (live.get(key, []) if live else [])[:2]:
                try:
                    L = np.asarray(ep[0], float); R = np.asarray(ep[1], float)
                    if L.shape[0] >= 4 and R.shape[0] >= 4:
                        seg = np.vstack([L[:4], R[:4]]) @ V
                        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], style,
                                color=color, lw=2.0, alpha=0.85, zorder=4)
                        drawn = True
                except Exception:
                    pass
            if drawn:
                legend_handles.append(matplotlib.lines.Line2D(
                    [], [], linestyle=style, color=color, lw=2.0, label=label))

        _draw_lines("prior_line_endpoints", "--", "#7799ee", "Prior LineBO lines")
        _draw_lines("line_endpoints",       "-",  "#0044dd", "Current LineBO lines")

        # ── needles ───────────────────────────────────────────────────────
        needles_data: list = live.get("needles", []) if live else []
        if not needles_data and rd is not None and rd.needles is not None \
                and rd.needles.shape[0] > 0:
            needles_data = [{"point": npt.tolist()} for npt in rd.needles]
        nd_pts = []
        for nd in needles_data:
            try:
                pt = np.asarray(nd["point"], float)
                if pt.shape[0] >= 4:
                    nd_pts.append(pt[:4])
            except Exception:
                pass
        if nd_pts:
            NP = np.asarray(nd_pts) @ V
            ax.scatter(NP[:, 0], NP[:, 1], NP[:, 2], marker="X", s=160,
                       color="#ff3300", edgecolors="darkred", lw=0.8, zorder=6)
            legend_handles.append(matplotlib.lines.Line2D(
                [], [], linestyle="none", marker="X", markersize=10,
                color="#ff3300", markeredgecolor="darkred",
                label=f"Needles ({len(nd_pts)})"))

        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.set_axis_off()
        run_lbl = rd.run_id if rd else ""
        n_pts   = X.shape[0] if X.ndim == 2 else 0
        src_tag = (" [live]" if live else
                   (f" [snap: {rd.snapshot_name}]" if rd and rd.snapshot_name else ""))
        ax.set_title(f"{run_lbl}  —  4-simplex point cloud  ·  {n_pts} pts{src_tag}",
                     fontsize=9)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left",
                      fontsize=7, framealpha=0.88)
        self.draw()

    def _replot(self, X: np.ndarray, n_cols: int, labels: list[str]):
        # d == 4 → tetrahedron point-cloud view instead of a 3-of-d ternary.
        if n_cols == 4:
            self._replot_tetra(X, labels)
            return

        self.fig.clf()
        ax = self.fig.add_subplot(111)
        rd   = self._last_rd
        live = self._live_state

        # ── placeholder for d < 3 ─────────────────────────────────────────
        if n_cols < 3:
            if rd is None or rd.d == 0:
                msg = "No run loaded."
            elif rd.d < 3:
                msg = f"Ternary requires d ≥ 3.\nThis run has d = {rd.d}."
            else:
                msg = "Waiting for first objective call …"
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray",
                    multialignment="center")
            ax.axis("off")
            self.draw()
            return

        # ── resolve column assignments ────────────────────────────────────
        resolved = self._resolve_cols(labels)
        if resolved is None:
            ax.text(0.5, 0.5, "Select 3 distinct dimensions\nthen click Update",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray", multialignment="center")
            ax.axis("off")
            self.draw()
            return

        da, db, dc = resolved
        la, lb, lc = labels[da], labels[db], labels[dc]
        max_col    = max(da, db, dc)
        _draw_ternary_frame_ax(ax, la, lb, lc)
        legend_handles: list = []

        # Project a composition array (N, 3) to ternary xy using the current
        # vertex-dim selection, so overlays line up with the scatter points.
        def _proj(comp: np.ndarray):
            x, y = _ternary_xy(comp[:, da], comp[:, db], comp[:, dc])
            return np.column_stack([x, y])

        # ── trust / zoom region: sample composition bounds → convex hull ──
        # Matches _draw_bounds_region in interactive_test_zombi.py. Source the
        # bounds from the snapshot (works for synthetic runs) or live state.
        lo = hi = None
        if live is not None and live.get("zoom_bounds_lo") is not None \
                and live.get("zoom_bounds_hi") is not None:
            lo, hi = live["zoom_bounds_lo"], live["zoom_bounds_hi"]
        elif rd is not None and rd.zoom_bounds is not None \
                and rd.zoom_bounds.shape == (2, 3):
            lo, hi = rd.zoom_bounds[0], rd.zoom_bounds[1]
        if lo is not None and n_cols == 3:
            comp = _sample_bounds_comp(lo, hi)
            if comp is not None:
                bxy = _proj(comp)
                try:
                    from scipy.spatial import ConvexHull
                    verts = bxy[ConvexHull(bxy).vertices]
                    ax.add_patch(matplotlib.patches.Polygon(
                        verts, closed=True, facecolor="#ff000010",
                        edgecolor="red", lw=2.0, linestyle="--",
                        zorder=2))
                    legend_handles.append(matplotlib.patches.Patch(
                        facecolor="#ff000018", edgecolor="red",
                        linestyle="--", label="Zoom bounds"))
                except Exception:
                    pass

        # ── penalty / needle penalization ellipsoids ─────────────────────
        # Matches _draw_needle_ellipsoid in interactive_test_zombi.py. Prefer the
        # snapshot's true ellipsoid (needle M + B); fall back to live isotropic
        # spheres (radius_ilr) when no snapshot ellipsoid is available.
        penalty_drawn = False
        if n_cols == 3 and rd is not None and rd.needles is not None \
                and rd.needles.shape[0] > 0 and rd.needle_M_list:
            for i, ncomp in enumerate(rd.needles):
                M = rd.needle_M_list[i] if i < len(rd.needle_M_list) else None
                ell = _needle_ellipsoid_comp(ncomp, M, rd.needle_B)
                if ell is None:
                    continue
                ax.add_patch(matplotlib.patches.Polygon(
                    _proj(ell), closed=True, facecolor="#80008022",
                    edgecolor="purple", lw=0.9, zorder=2))
                penalty_drawn = True
        if not penalty_drawn and live is not None and n_cols == 3:
            for pr in live.get("penalty_regions", []):
                try:
                    cpt = np.asarray(pr["center"], float)
                    rad = float(pr["radius_ilr"])
                    if cpt.shape[0] != 3 or rad <= 0:
                        continue
                    # isotropic ILR sphere of radius rad ⇒ M = I / rad²
                    ell = _needle_ellipsoid_comp(cpt, np.eye(2) / (rad ** 2), None)
                    if ell is None:
                        continue
                    ax.add_patch(matplotlib.patches.Polygon(
                        _proj(ell), closed=True, facecolor="#80008022",
                        edgecolor="purple", lw=0.9, zorder=2))
                    penalty_drawn = True
                except Exception:
                    pass
        if penalty_drawn:
            legend_handles.append(matplotlib.patches.Patch(
                facecolor="#80008033", edgecolor="purple",
                label="Penalty region"))

        # ── scatter points ────────────────────────────────────────────────
        if X.shape[0] > 0 and X.shape[1] > max_col:
            A, B, C = X[:, da], X[:, db], X[:, dc]
            xp, yp  = _ternary_xy(A, B, C)

            Y = None
            if live is not None:
                yraw = live.get("y_values", [])
                if len(yraw) == len(xp):
                    Y = np.array(yraw, dtype=float)
            elif rd is not None and rd.Y_all is not None and len(rd.Y_all) == len(xp):
                Y = rd.Y_all

            if Y is not None:
                sc = ax.scatter(xp, yp, c=Y, cmap="viridis", s=22, alpha=0.80, zorder=3)
                cb_bar = self.fig.colorbar(sc, ax=ax, shrink=0.60, pad=0.02)
                cb_bar.set_label("Objective Y", fontsize=8)
            else:
                ax.scatter(xp, yp, s=22, alpha=0.80, color="steelblue", zorder=3)
            legend_handles.append(matplotlib.lines.Line2D(
                [], [], linestyle="none", marker="o", markersize=6,
                color="steelblue", label=f"Samples ({len(xp)})"))

        # ── prior LineBO lines (dashed, faded) ────────────────────────────
        prior_eps    = live.get("prior_line_endpoints", []) if live else []
        prior_colors = ["#7799ee", "#77cc77"]
        prior_drawn  = False
        for i, ep in enumerate(prior_eps[:2]):
            try:
                L = np.asarray(ep[0], float); R = np.asarray(ep[1], float)
                if L.shape[0] > max_col and R.shape[0] > max_col:
                    xl, yl = _ternary_xy(L[[da]], L[[db]], L[[dc]])
                    xr, yr = _ternary_xy(R[[da]], R[[db]], R[[dc]])
                    ax.plot([xl[0], xr[0]], [yl[0], yr[0]],
                            "--", color=prior_colors[i % 2], lw=1.6, alpha=0.70, zorder=4)
                    if not prior_drawn:
                        legend_handles.append(matplotlib.lines.Line2D(
                            [], [], linestyle="--", color=prior_colors[0],
                            lw=1.6, label="Prior LineBO lines"))
                        prior_drawn = True
            except Exception:
                pass

        # ── current LineBO lines (solid, prominent) ───────────────────────
        curr_eps    = live.get("line_endpoints", []) if live else []
        curr_colors = ["#0044dd", "#007700"]
        curr_drawn  = False
        for i, ep in enumerate(curr_eps[:2]):
            try:
                L = np.asarray(ep[0], float); R = np.asarray(ep[1], float)
                if L.shape[0] > max_col and R.shape[0] > max_col:
                    xl, yl = _ternary_xy(L[[da]], L[[db]], L[[dc]])
                    xr, yr = _ternary_xy(R[[da]], R[[db]], R[[dc]])
                    ax.plot([xl[0], xr[0]], [yl[0], yr[0]],
                            "-", color=curr_colors[i % 2], lw=2.0, alpha=0.88, zorder=5)
                    if not curr_drawn:
                        legend_handles.append(matplotlib.lines.Line2D(
                            [], [], linestyle="-", color=curr_colors[0],
                            lw=2.0, label="Current LineBO lines"))
                        curr_drawn = True
            except Exception:
                pass

        # ── needles ───────────────────────────────────────────────────────
        needles_data: list = live.get("needles", []) if live else []
        if not needles_data and rd is not None and rd.needles is not None and rd.needles.shape[0] > 0:
            nv = rd.needle_vals
            for i, npt in enumerate(rd.needles):
                val = float(nv[i]) if nv is not None and i < len(nv) else 0.0
                needles_data.append({"point": npt.tolist(), "y": val})

        needle_drawn = False
        for nd in needles_data:
            try:
                pt = np.asarray(nd["point"], float)
                if pt.shape[0] > max_col:
                    xn, yn = _ternary_xy(pt[[da]], pt[[db]], pt[[dc]])
                    ax.scatter(xn, yn, marker="*", s=260, color="#ff4400",
                               edgecolors="#880000", lw=0.8, zorder=7)
                    if not needle_drawn:
                        legend_handles.append(matplotlib.lines.Line2D(
                            [], [], linestyle="none", marker="*", markersize=12,
                            color="#ff4400", markeredgecolor="#880000",
                            label="Needle (optimum)"))
                        needle_drawn = True
            except Exception:
                pass

        # ── legend ────────────────────────────────────────────────────────
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right",
                      fontsize=7, framealpha=0.88, ncol=1)

        # ── title ─────────────────────────────────────────────────────────
        run_lbl = rd.run_id if rd else ""
        n_pts   = X.shape[0] if X.ndim == 2 else 0
        src_tag = (" [live]" if live else
                   (f" [snap: {rd.snapshot_name}]" if rd and rd.snapshot_name else ""))
        ax.set_title(
            f"{run_lbl}  —  Ternary ({la}, {lb}, {lc})  ·  {n_pts} pts{src_tag}",
            fontsize=9)
        self.draw()


# ── manual composition control ───────────────────────────────────────────────

# Try to detect whether pyserial is available so we can give a clear error early
try:
    import serial as _pyserial  # noqa: F401
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False


class ManualControlFrame(ttk.Frame):
    """
    Tab for direct hardware composition control — bypasses ZoMBI-Hop.

    Workflow:
      1. Set the COM port and baud rate, then click "Connect" to start serial IO.
      2. Fill in the 10-component Inlet (start) and Outlet (end) compositions.
      3. Click "▶ Update" to write those compositions to compositions.db.
         The composition_sender in the serial process immediately picks it up
         and begins sending the new line to the hardware continuously.
      4. The "Currently Sending" log shows what is active and serial messages.

    "▶ Update" can also be clicked *before* connecting — it will write to the DB
    so compositions are ready when the serial process starts.
    """

    N_DIMS = 10

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._proc: Optional[subprocess.Popen] = None
        self._poll_job: Optional[str] = None
        self._start_vars: list[tk.StringVar] = []
        self._end_vars:   list[tk.StringVar] = []
        self._start_sum_var = tk.StringVar(value="")
        self._end_sum_var   = tk.StringVar(value="")
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Scrollable canvas wrapper so long content doesn't get clipped
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vsb    = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        fid   = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize_window(e, c=canvas, f=fid):
            c.itemconfig(f, width=e.width)
        canvas.bind("<Configure>", _resize_window)

        def _update_scroll(e, c=canvas):
            c.configure(scrollregion=c.bbox("all"))
        inner.bind("<Configure>", _update_scroll)

        self._build_connection(inner)
        self._build_comp_section(inner, "Inlet  (Start) Composition", is_start=True)
        self._build_comp_section(inner, "Outlet (End) Composition",   is_start=False)
        self._build_controls(inner)
        self._build_display(inner)

    def _build_connection(self, parent):
        fr = ttk.LabelFrame(parent, text="Serial Connection")
        fr.pack(fill="x", padx=6, pady=(6, 3))

        r1 = ttk.Frame(fr); r1.pack(fill="x", padx=4, pady=2)
        ttk.Label(r1, text="COM port:").pack(side="left")
        self._com_var = tk.StringVar(value="COM5")
        ttk.Entry(r1, textvariable=self._com_var, width=9).pack(side="left", padx=(4, 18))
        ttk.Label(r1, text="Baud:").pack(side="left")
        self._baud_var = tk.IntVar(value=9600)
        ttk.Entry(r1, textvariable=self._baud_var, width=8).pack(side="left", padx=4)

        r2 = ttk.Frame(fr); r2.pack(fill="x", padx=4, pady=2)
        ttk.Label(r2, text="Compositions DB:").pack(side="left")
        _proj = Path(__file__).resolve().parent.parent
        self._compdb_var = tk.StringVar(value=str(_proj / "sql" / "compositions.db"))
        ttk.Entry(r2, textvariable=self._compdb_var, width=40).pack(side="left", padx=4)
        ttk.Button(r2, text="…", width=3, command=self._browse_db).pack(side="left")

        r3 = ttk.Frame(fr); r3.pack(fill="x", padx=4, pady=(4, 6))
        self._conn_btn = ttk.Button(r3, text="▶ Connect",
                                    command=self._connect,
                                    state="normal" if _SERIAL_OK else "disabled")
        self._conn_btn.pack(side="left", padx=4)
        self._disc_btn = ttk.Button(r3, text="■ Disconnect",
                                    command=self._disconnect, state="disabled")
        self._disc_btn.pack(side="left", padx=4)
        self._conn_sv = tk.StringVar(value="○  Disconnected")
        self._conn_lbl = ttk.Label(r3, textvariable=self._conn_sv,
                                   foreground="#880000", font=("Consolas", 9))
        self._conn_lbl.pack(side="left", padx=8)
        if not _SERIAL_OK:
            ttk.Label(r3, text="(pyserial not installed)",
                      foreground="gray").pack(side="left")

    def _build_comp_section(self, parent, title: str, is_start: bool):
        fr = ttk.LabelFrame(parent, text=title)
        fr.pack(fill="x", padx=6, pady=3)

        entries: list[tk.StringVar] = []
        n_cols = 5
        for i in range(self.N_DIMS):
            row, col = divmod(i, n_cols)
            ttk.Label(fr, text=f"d{i}:", width=4, anchor="e").grid(
                row=row, column=2*col,   sticky="e", padx=(6, 2), pady=3)
            sv = tk.StringVar(value=f"{1.0/self.N_DIMS:.6f}")
            ttk.Entry(fr, textvariable=sv, width=10).grid(
                row=row, column=2*col+1, sticky="w", padx=(0, 6), pady=3)
            entries.append(sv)
            sv.trace_add("write", lambda *_, s=is_start: self._update_sum(s))

        n_rows = math.ceil(self.N_DIMS / n_cols)
        sf = ttk.Frame(fr)
        sf.grid(row=n_rows, column=0, columnspan=2*n_cols, sticky="w", padx=6, pady=(0, 4))
        sv_sum = tk.StringVar(value="")
        ttk.Label(sf, textvariable=sv_sum, font=("Consolas", 8),
                  width=22, anchor="w").pack(side="left")
        ttk.Button(sf, text="Normalize",
                   command=lambda s=is_start: self._normalize(s)).pack(side="left", padx=8)
        ttk.Button(sf, text="Set uniform",
                   command=lambda s=is_start: self._set_uniform(s)).pack(side="left")

        if is_start:
            self._start_vars   = entries
            self._start_sum_var = sv_sum
        else:
            self._end_vars     = entries
            self._end_sum_var  = sv_sum

        # compute initial sum label
        self._update_sum(is_start)

    def _build_controls(self, parent):
        fr = ttk.Frame(parent)
        fr.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(fr, text="N samples along line:").pack(side="left")
        self._n_var = tk.IntVar(value=24)
        ttk.Entry(fr, textvariable=self._n_var, width=6).pack(side="left", padx=4)
        ttk.Button(fr, text="▶ Update",
                   command=self._do_update,
                   style="Accent.TButton" if "Accent.TButton" in ttk.Style().theme_names()
                   else "TButton").pack(side="left", padx=20)
        ttk.Label(fr,
                  text="← writes to compositions.db; hardware starts sending immediately",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side="left")

    def _build_display(self, parent):
        fr = ttk.LabelFrame(parent, text="Currently Sending  (serial log)")
        fr.pack(fill="both", expand=True, padx=6, pady=(2, 6))
        self._disp = scrolledtext.ScrolledText(
            fr, height=10, font=("Consolas", 8), state="disabled", wrap="none")
        self._disp.pack(fill="both", expand=True, padx=2, pady=2)
        self._disp.tag_configure("header",   foreground="#004488", font=("Consolas", 8, "bold"))
        self._disp.tag_configure("serial",   foreground="#555555")
        self._disp.tag_configure("error",    foreground="red")
        self._disp.tag_configure("pending",  foreground="gray")
        self._append_display("(not sent yet — fill in compositions and click ▶ Update)\n",
                             tag="pending")

    # ── connection management ──────────────────────────────────────────────

    def _connect(self):
        if self._proc is not None and self._proc.poll() is None:
            messagebox.showinfo("Already connected", "Disconnect first.")
            return
        if not _SERIAL_OK:
            messagebox.showerror("Missing dependency",
                                 "pyserial is not installed.\n"
                                 "Run:  pip install pyserial")
            return

        com    = self._com_var.get().strip()
        baud   = self._baud_var.get()
        comp_db = self._compdb_var.get().strip()

        script  = str(Path(__file__).resolve().parent.parent / "scripts" / "serial_only.py")
        proj    = str(Path(__file__).resolve().parent.parent)

        cmd = [sys.executable, script,
               "--com",  com, "--baud", str(baud), "--comp-db", comp_db]

        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding="utf-8", errors="replace", bufsize=1, cwd=proj, env=env)
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        self._set_conn_state(connected=True)
        self._append_display(
            f"{'─'*60}\n"
            f"[{self._ts()}] Connected: {com} @ {baud} baud\n",
            tag="header")

        def _pump():
            for line in self._proc.stdout:
                s = line.rstrip()
                if s:
                    tag = "error" if "error" in s.lower() else "serial"
                    self.after(0, lambda msg=s, t=tag: self._append_display(msg + "\n", t))
            self.after(0, self._on_proc_exit)

        threading.Thread(target=_pump, daemon=True).start()
        self._schedule_proc_poll()

    def _disconnect(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._on_proc_exit()

    def _schedule_proc_poll(self):
        self._poll_job = self.after(2000, self._check_proc)

    def _check_proc(self):
        if self._proc is not None and self._proc.poll() is not None:
            self._on_proc_exit()
        else:
            self._schedule_proc_poll()

    def _on_proc_exit(self):
        if self._poll_job:
            self.after_cancel(self._poll_job)
            self._poll_job = None
        self._proc = None
        self._set_conn_state(connected=False)
        self._append_display(
            f"[{self._ts()}] Serial connection closed.\n", tag="header")

    def _set_conn_state(self, connected: bool):
        if connected:
            self._conn_sv.set("●  Connected")
            self._conn_lbl.config(foreground="#007700")
            self._conn_btn.config(state="disabled")
            self._disc_btn.config(state="normal")
        else:
            self._conn_sv.set("○  Disconnected")
            self._conn_lbl.config(foreground="#880000")
            self._conn_btn.config(state="normal" if _SERIAL_OK else "disabled")
            self._disc_btn.config(state="disabled")

    # ── composition helpers ────────────────────────────────────────────────

    def _update_sum(self, is_start: bool):
        try:
            vars_   = self._start_vars if is_start else self._end_vars
            sum_var = self._start_sum_var if is_start else self._end_sum_var
            if not vars_:
                return
            total = sum(float(v.get()) for v in vars_)
            ok    = abs(total - 1.0) < 1e-4
            sum_var.set(f"Sum: {total:.6f}  {'✓' if ok else '⚠ ≠ 1.0'}")
        except (ValueError, tk.TclError, AttributeError):
            pass

    def _normalize(self, is_start: bool):
        vars_ = self._start_vars if is_start else self._end_vars
        try:
            vals = np.array([float(v.get()) for v in vars_])
        except ValueError:
            messagebox.showwarning("Invalid input", "Cannot parse all values as numbers.")
            return
        s = vals.sum()
        if s > 0:
            vals /= s
        for var, val in zip(vars_, vals):
            var.set(f"{val:.6f}")
        self._update_sum(is_start)

    def _set_uniform(self, is_start: bool):
        vars_ = self._start_vars if is_start else self._end_vars
        u = 1.0 / self.N_DIMS
        for v in vars_:
            v.set(f"{u:.6f}")
        self._update_sum(is_start)

    def _get_comp(self, is_start: bool) -> Optional[np.ndarray]:
        vars_ = self._start_vars if is_start else self._end_vars
        try:
            return np.array([float(v.get()) for v in vars_])
        except (ValueError, tk.TclError):
            return None

    # ── update (write to DB) ──────────────────────────────────────────────

    def _do_update(self):
        start = self._get_comp(is_start=True)
        end   = self._get_comp(is_start=False)

        if start is None or end is None:
            messagebox.showerror("Invalid input",
                                 "All 10 values in both rows must be valid numbers.")
            return

        s_sum, e_sum = start.sum(), end.sum()
        needs_norm = abs(s_sum - 1.0) > 0.01 or abs(e_sum - 1.0) > 0.01
        if needs_norm:
            if not messagebox.askyesno(
                    "Sum ≠ 1",
                    f"Inlet sum = {s_sum:.6f}\nOutlet sum = {e_sum:.6f}\n\n"
                    f"Normalize both to 1 before sending?"):
                return
            if s_sum > 0: start = start / s_sum
            if e_sum > 0: end   = end   / e_sum
        else:
            if s_sum > 0: start = start / s_sum
            if e_sum > 0: end   = end   / e_sum

        try:
            n = max(2, int(self._n_var.get()))
        except (ValueError, tk.TclError):
            n = 24

        # Build interpolated line; re-normalize each row to sum to 1
        array = np.linspace(start, end, n)
        row_sums = array.sum(axis=1, keepdims=True)
        array /= np.where(row_sums == 0, 1.0, row_sums)

        comp_db = self._compdb_var.get().strip()
        os.makedirs(Path(comp_db).parent, exist_ok=True)

        try:
            from scripts.communication import write_compositions
            write_compositions(
                start=start, end=end, array=array,
                timestamp=time.time(),
                start_cache=start.copy(), end_cache=end.copy(),
                array_cache=array.copy(),
                db_path=comp_db,
            )
        except Exception as exc:
            messagebox.showerror("Write failed",
                                 f"Could not write to:\n{comp_db}\n\n{exc}")
            return

        # ── update the display ─────────────────────────────────────────────
        def _fmt(arr: np.ndarray) -> str:
            return "[" + ",  ".join(f"{v:.5f}" for v in arr) + "]"

        header = (
            f"{'═'*62}\n"
            f"  Updated: {self._ts()}\n"
            f"  N samples: {n}   DB: {comp_db}\n"
            f"{'─'*62}\n"
            f"  Inlet  (start):\n    {_fmt(start)}\n"
            f"  Outlet (end):\n    {_fmt(end)}\n"
            f"{'─'*62}\n"
            f"  Array (first 3 / last row):\n"
        )
        rows_shown = [array[0], array[n//2]] + ([array[-1]] if n > 2 else [])
        row_strs   = "".join(
            f"    row {i:>3}: {_fmt(array[i])}\n"
            for i in ([0, n//2, n-1] if n > 2 else [0]))
        header += row_strs + f"{'═'*62}\n"

        self._append_display(header, tag="header")

        # Refresh sum labels with normalised values
        for var, val in zip(self._start_vars, start):
            var.set(f"{val:.6f}")
        for var, val in zip(self._end_vars, end):
            var.set(f"{val:.6f}")

    # ── display helpers ────────────────────────────────────────────────────

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_display(self, text: str, tag: str = "serial"):
        try:
            self._disp.config(state="normal")
            self._disp.insert("end", text, tag)
            self._disp.see("end")
            self._disp.config(state="disabled")
        except tk.TclError:
            pass

    def _browse_db(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("SQLite DB", "*.db"), ("All", "*.*")],
            initialdir=str(Path(self._compdb_var.get()).parent),
            title="Select / create compositions.db")
        if p:
            self._compdb_var.set(p)


# ── tooltip helper ───────────────────────────────────────────────────────────

class ToolTip:
    """Lightweight hover tooltip for any Tkinter widget."""

    def __init__(self, widget: tk.Widget, text: str, delay: int = 450):
        self._widget   = widget
        self._text     = text
        self._delay    = delay
        self._tip_win: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>",       self._schedule, add="+")
        widget.bind("<Leave>",       self._cancel,   add="+")
        widget.bind("<ButtonPress>", self._cancel,   add="+")

    def _schedule(self, _e=None):
        self._cancel()
        self._after_id = self._widget.after(self._delay, self._show)

    def _cancel(self, _e=None):
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip_win:
            return
        x = self._widget.winfo_rootx() + self._widget.winfo_width() + 6
        y = self._widget.winfo_rooty()
        self._tip_win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self._text, justify="left",
            background="#fffbe6", relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9), wraplength=340, padx=6, pady=4,
        ).pack()

    def _hide(self):
        if self._tip_win:
            self._tip_win.destroy()
            self._tip_win = None


# ── new-run dialog ────────────────────────────────────────────────────────────

class _LogStream:
    """Thread-safe stdout redirect: forwards each line to a callback."""

    def __init__(self, callback, real_stdout=None):
        self._cb   = callback
        self._real = real_stdout
        self._buf  = ""

    def write(self, text: str):
        if self._real:
            self._real.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.rstrip()
            if stripped:
                self._cb(stripped)

    def flush(self):
        if self._real:
            self._real.flush()


# ── colour palette for log tags ──────────────────────────────────────────────
_LOG_TAGS = {
    "needle":  {"foreground": "crimson",   "font": ("Consolas", 8, "bold")},
    "fail":    {"foreground": "#c07000",   "font": ("Consolas", 8)},
    "status":  {"foreground": "navy",      "font": ("Consolas", 8, "bold")},
    "gp":      {"foreground": "#207060",   "font": ("Consolas", 8)},
    "info":    {"foreground": "#555555",   "font": ("Consolas", 8)},
    "error":   {"foreground": "red",       "font": ("Consolas", 8, "bold")},
    "done":    {"foreground": "green",     "font": ("Consolas", 8, "bold")},
    "default": {"foreground": "#333333",   "font": ("Consolas", 8)},
}


def _log_tag_for(line: str) -> str:
    low = line.lower()
    if "error" in low or "traceback" in low:
        return "error"
    if "done" in low or "finished" in low:
        return "done"
    if "needle" in low or "declare" in low:
        return "needle"
    if "fail" in low or "retry" in low or "shrink" in low:
        return "fail"
    if "[zombi]" in low or "activation" in low or "zoom" in low:
        return "status"
    if "[gp" in low or "mll" in low or "fitting" in low:
        return "gp"
    if "init" in low or "run_uuid" in low or "starting" in low:
        return "info"
    return "default"


# Static status regex patterns
_RE_ACTIVATION      = re.compile(r"activation\s*[:#=]\s*(\d+)", re.IGNORECASE)
_RE_ACTIVATION_SLASH= re.compile(r"activation\s+(\d+)/\d+", re.IGNORECASE)
_RE_AZI_BRACKET     = re.compile(r"\[A(\d+)/Z(\d+)/I(\d+)\]", re.IGNORECASE)
_RE_ZOOM            = re.compile(r"zoom\s*[:#=]\s*(\d+)", re.IGNORECASE)
_RE_ZOOM_DASH       = re.compile(r"---\s*zoom\s+(\d+)/\d+", re.IGNORECASE)
_RE_ITER            = re.compile(r"iter(?:ation)?\s*[:#=]\s*(\d+)", re.IGNORECASE)
_RE_ITER_DOT        = re.compile(r"·\s*iter\s+(\d+)/\d+", re.IGNORECASE)
_RE_BOUNDS          = re.compile(r"bounds[^[]*(\[.+)", re.IGNORECASE)
_RE_CANDIDATE       = re.compile(r"candidate.*?:\s*(\[[\d.,\s\-]+\])", re.IGNORECASE)
_RE_LINE0_LEFT      = re.compile(r"line_0\s+left\s*:\s*(\[[\d.,\s\-]+\])", re.IGNORECASE)
_RE_LINE0_RIGHT     = re.compile(r"line_0\s+right\s*:\s*(\[[\d.,\s\-]+\])", re.IGNORECASE)
_RE_FAILURE         = re.compile(r"\[failure\]|activation failed|no valid candidate", re.IGNORECASE)


def _parse_status_from_line(line: str, state: dict):
    """Update mutable state dict from a log line (activation, zoom, iter, bounds, etc.)."""
    # [A1/Z2/I3] bracket format from _log_status
    m = _RE_AZI_BRACKET.search(line)
    if m:
        state["act"] = m.group(1)
        state["zoom"] = m.group(2)
        state["iter"] = m.group(3)
        return  # fully parsed this line

    m = _RE_ACTIVATION_SLASH.search(line) or _RE_ACTIVATION.search(line)
    if m:
        state["act"] = m.group(1)
    m = _RE_ZOOM_DASH.search(line) or _RE_ZOOM.search(line)
    if m:
        state["zoom"] = m.group(1)
    m = _RE_ITER_DOT.search(line) or _RE_ITER.search(line)
    if m:
        state["iter"] = m.group(1)
    m = _RE_BOUNDS.search(line)
    if m:
        state["bounds"] = m.group(1)[:80]
    m = _RE_CANDIDATE.search(line)
    if m:
        state["candidate"] = m.group(1)[:60]
    m = _RE_LINE0_LEFT.search(line)
    if m:
        state["line0_left"] = m.group(1)[:60]
    m = _RE_LINE0_RIGHT.search(line)
    if m:
        state["line0_right"] = m.group(1)[:60]
    if _RE_FAILURE.search(line):
        state["status_tag"] = "FAIL"
    elif "needle" in line.lower() and "declare" in line.lower():
        state["status_tag"] = "NEEDLE"
    elif "done" in line.lower() or "finished" in line.lower():
        state["status_tag"] = "DONE"
    else:
        state["status_tag"] = ""


def _fmt_status(state: dict) -> str:
    bounds_str    = f"\n  bnds:{state.get('bounds','')}" if state.get("bounds") else ""
    cand_str      = f"\n  cand:{state.get('candidate','')}" if state.get("candidate") else ""
    l0l           = state.get("line0_left", "")
    l0r           = state.get("line0_right", "")
    lines_str     = f"\n  L0: {l0l[:40]} → {l0r[:40]}" if l0l or l0r else ""
    tag           = state.get("status_tag", "")
    tag_str       = f"  [{tag}]" if tag else ""
    return (
        f"A:{state.get('act','—')}  Z:{state.get('zoom','—')}  I:{state.get('iter','—')}"
        f"{bounds_str}{tag_str}{cand_str}{lines_str}"
    )


class LiveLogPanel(ttk.LabelFrame):
    """Bottom-right panel: static status line + scrollable colour log."""

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("text", "Live Run Log")
        super().__init__(parent, **kwargs)
        self._state = {
            "act": "—", "zoom": "—", "iter": "—", "bounds": "",
            "candidate": "", "line0_left": "", "line0_right": "",
            "status_tag": "",
        }
        # Static status line (single non-wrapping label)
        self._status_var = tk.StringVar(value="A:—  Z:—  I:—")
        ttk.Label(
            self, textvariable=self._status_var,
            font=("Consolas", 8, "bold"), foreground="navy",
            relief="sunken", anchor="w",
        ).pack(fill="x", padx=2, pady=(2, 0))

        # Scrollable log
        self._log = scrolledtext.ScrolledText(
            self, height=9, font=("Consolas", 8), state="disabled", wrap="none"
        )
        self._log.pack(fill="both", expand=True, padx=2, pady=(2, 4))
        for tag, cfg in _LOG_TAGS.items():
            self._log.tag_configure(tag, **cfg)

    def log(self, msg: str, tag: Optional[str] = None):
        if tag is None:
            tag = _log_tag_for(msg)
        _parse_status_from_line(msg, self._state)

        def _do():
            try:
                if not self.winfo_exists():
                    return
                self._status_var.set(_fmt_status(self._state))
                self._log.config(state="normal")
                self._log.insert("end", msg + "\n", tag)
                self._log.see("end")
                self._log.config(state="disabled")
            except tk.TclError:
                pass

        self.after(0, _do)

    def clear(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")
        self._state.update({
            "act": "—", "zoom": "—", "iter": "—", "bounds": "",
            "candidate": "", "line0_left": "", "line0_right": "",
            "status_tag": "",
        })
        self._status_var.set("A:—  Z:—  I:—")


class NewRunDialog(tk.Toplevel):
    def __init__(self, parent, app: "ZoMBIApp"):
        super().__init__(parent)
        self.title("New ZoMBI-Hop Run")
        self.resizable(True, True)
        self.geometry("640x520")
        self._app    = app
        self._thread: Optional[threading.Thread] = None
        self._hw_proc: Optional[subprocess.Popen] = None
        self._build()
        self.grab_set()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        # ── synthetic tab ──────────────────────────────────────────────────
        tf = ttk.Frame(nb)
        nb.add(tf, text="Synthetic")

        row = 0
        ttk.Label(tf, text="Dataset:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=8, pady=4)
        # Datasets mirror optimize/evaluate.py (RF surrogate + analytic Ackleys).
        self._ds_var = tk.StringVar(value="ackley3d")
        ttk.OptionMenu(tf, self._ds_var, "ackley3d",
                       "RF", "ackley3d", "ackley4d", "ackley10d",
                       command=lambda _v: self._sync_ds_fields()).grid(
            row=row, column=1, sticky="w", padx=4)
        ttk.Label(tf, text="(dimension is set by the dataset)",
                  foreground="gray").grid(row=row, column=2, sticky="w")

        row += 1
        ttk.Label(tf, text="Ackley variant:").grid(
            row=row, column=0, sticky="w", padx=8)
        from synthetic_data.ackley import Ackley as _Ackley
        self._variant_var = tk.StringVar(value="realistic")
        self._variant_menu = ttk.OptionMenu(
            tf, self._variant_var, "realistic", *sorted(_Ackley.VARIANTS))
        self._variant_menu.grid(row=row, column=1, sticky="w", padx=4)

        # RF source (mobo_* dir) is hardcoded to the default surrogate run; the
        # max number of activations is hardcoded to infinite (run until stopped).
        self._rf_src_default = str(_HERE.parent / "optimize" / "runs" / "mobo_05_06_15_32")

        row += 1
        ttk.Label(tf, text="Output directory:").grid(
            row=row, column=0, sticky="w", padx=8)
        self._outdir_var = tk.StringVar(value=self._app.ckpt_dir)
        odf = ttk.Frame(tf)
        odf.grid(row=row, column=1, columnspan=2, sticky="w", pady=4)
        ttk.Entry(odf, textvariable=self._outdir_var, width=26).pack(side="left")
        ttk.Button(odf, text="…", width=3, command=self._browse_out).pack(side="left")

        row += 1
        ttk.Label(tf, text="Hyperparameters JSON:").grid(
            row=row, column=0, sticky="w", padx=8)
        self._syn_hparams_var = tk.StringVar(value="")
        shp = ttk.Frame(tf)
        shp.grid(row=row, column=1, columnspan=2, sticky="w", pady=2)
        ttk.Entry(shp, textvariable=self._syn_hparams_var, width=26).pack(side="left")
        ttk.Button(shp, text="…", width=3,
                   command=lambda: self._browse_hparams_into(self._syn_hparams_var)).pack(side="left")

        row += 1
        ttk.Label(tf, text="(trial.json-style file. If blank, arbitrary defaults "
                           "are used instead — from mobo trial_112.)",
                  foreground="gray").grid(row=row, column=0, columnspan=3,
                                          sticky="w", padx=8)

        # Enable/disable the variant / RF-source rows to match the dataset.
        self._sync_ds_fields()

        # ── hardware tab ───────────────────────────────────────────────────
        hw = ttk.Frame(nb)
        nb.add(hw, text="Hardware")

        hrow = 0
        ttk.Label(hw, text="Resume UUID (blank = new run):").grid(
            row=hrow, column=0, sticky="w", padx=8, pady=4)
        self._uuid_var = tk.StringVar(value="")
        ttk.Entry(hw, textvariable=self._uuid_var, width=38).grid(
            row=hrow, column=1, sticky="w", padx=4)

        hrow += 1
        ttk.Label(hw, text="Optimizing dims (comma-sep):").grid(
            row=hrow, column=0, sticky="w", padx=8, pady=4)
        self._hw_dims_var = tk.StringVar(value="0,8,9")
        ttk.Entry(hw, textvariable=self._hw_dims_var, width=20).grid(
            row=hrow, column=1, sticky="w", padx=4)

        hrow += 1
        ttk.Label(hw, text="Script path:").grid(
            row=hrow, column=0, sticky="w", padx=8)
        _proj_root = Path(__file__).resolve().parent.parent
        self._hw_script_var = tk.StringVar(
            value=str(_proj_root / "scripts" / "main.py"))
        sf = ttk.Frame(hw)
        sf.grid(row=hrow, column=1, sticky="w", pady=2)
        ttk.Entry(sf, textvariable=self._hw_script_var, width=34).pack(side="left")
        ttk.Button(sf, text="…", width=3, command=self._browse_hw_script).pack(side="left")

        hrow += 1
        ttk.Label(hw, text="Python executable:").grid(
            row=hrow, column=0, sticky="w", padx=8)
        self._hw_python_var = tk.StringVar(value=sys.executable)
        ttk.Entry(hw, textvariable=self._hw_python_var, width=38).grid(
            row=hrow, column=1, sticky="w", padx=4)

        hrow += 1
        ttk.Label(hw, text="Hyperparameters JSON:").grid(
            row=hrow, column=0, sticky="w", padx=8)
        self._hw_hparams_var = tk.StringVar(value="")
        hpf = ttk.Frame(hw)
        hpf.grid(row=hrow, column=1, sticky="w", pady=2)
        ttk.Entry(hpf, textvariable=self._hw_hparams_var, width=30).pack(side="left")
        ttk.Button(hpf, text="…", width=3,
                   command=lambda: self._browse_hparams_into(self._hw_hparams_var)).pack(side="left")

        hrow += 1
        ttk.Label(hw, text="(trial.json-style file. If blank, arbitrary defaults "
                           "are used instead — from mobo trial_112.)",
                  foreground="gray").grid(
            row=hrow, column=0, columnspan=2, sticky="w", padx=8)

        hrow += 1
        hw_btn_f = ttk.Frame(hw)
        hw_btn_f.grid(row=hrow, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        ttk.Button(
            hw_btn_f, text="▶ Launch Hardware Run", command=self._start_hardware
        ).pack(side="left", padx=4)

        hrow += 1
        ttk.Label(hw, text="(Requires COM5 connected and databases initialised.)",
                  foreground="gray").grid(
            row=hrow, column=0, columnspan=2, sticky="w", padx=8)

        # ── bottom: start / close ──────────────────────────────────────────
        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=6, pady=8)
        ttk.Button(bot, text="▶ Start (Synthetic)", command=self._start).pack(side="left", padx=4)
        ttk.Button(bot, text="Close", command=self.destroy).pack(side="left")

    def _browse_out(self):
        d = filedialog.askdirectory(initialdir=self._outdir_var.get())
        if d:
            self._outdir_var.set(d)

    def _sync_ds_fields(self):
        """Enable only the inputs relevant to the chosen dataset."""
        ds = self._ds_var.get()
        is_rf = (ds == "RF")
        # Ackley variant applies to the analytic datasets only.
        self._variant_menu.config(state="disabled" if is_rf else "normal")

    def _browse_hw_script(self):
        p = filedialog.askopenfilename(
            initialdir=str(Path(self._hw_script_var.get()).parent),
            filetypes=[("Python", "*.py"), ("All", "*")])
        if p:
            self._hw_script_var.set(p)

    def _browse_hparams_into(self, var: tk.StringVar):
        """Pick a trial.json-style hyperparameter file into the given path var."""
        p = filedialog.askopenfilename(
            title="Select hyperparameter JSON",
            filetypes=[("JSON", "*.json"), ("All", "*")])
        if not p:
            return
        # Validate early so the user gets immediate feedback on a bad file.
        try:
            hp = load_hparams_json(p)
        except Exception as exc:
            messagebox.showerror("Load failed",
                                 f"Could not read hyperparameters:\n{exc}")
            return
        if not hp:
            messagebox.showwarning(
                "No hyperparameters",
                "No recognised hyperparameter keys found in that file.")
            return
        var.set(p)

    def _log_msg(self, msg: str, tag: Optional[str] = None):
        if tag is None:
            tag = _log_tag_for(msg)
        self._app.log_to_main(msg, tag)

    def _start(self):
        try:
            # Hyperparameters: defaults (trial_112) unless a JSON file is given.
            hp = dict(DEFAULT_HPARAMS)
            syn_path = self._syn_hparams_var.get().strip()
            if syn_path:
                loaded = load_hparams_json(syn_path)
                if not loaded:
                    raise ValueError(f"No recognised hyperparameters in {syn_path}.")
                hp.update(loaded)
            max_act  = float("inf")   # hardcoded: run until stopped
            outdir   = self._outdir_var.get()
            dataset  = self._ds_var.get()
            variant  = self._variant_var.get()
            rf_src   = self._rf_src_default   # hardcoded default surrogate run
        except Exception as exc:
            messagebox.showerror("Config error", str(exc))
            return

        app = self._app
        app.log_to_main(f"Starting synthetic: dataset={dataset}"
                        + (f" variant={variant}" if dataset != "RF" else "")
                        + f", max_act={max_act}", tag="info")
        app.log_to_main(f"  outdir: {outdir}", tag="info")

        # Pre-generate the UUID and register the run NOW so it appears in the
        # explorer the moment Start is clicked — before any data is collected.
        new_uuid = str(uuid4())[:4]
        run_dir  = Path(outdir) / f"run_{new_uuid}"
        run_id   = run_dir.name
        pause_event = threading.Event()
        pause_event.set()  # start in running state
        stop_event = threading.Event()
        app._register_active_run(pause_event, run_id, stop_event=stop_event)
        app.track_new_run(run_dir, hw=False)  # refreshes explorer → shows RUNNING
        self.destroy()

        run_id_ref: list[Optional[str]] = [run_id]

        def _log(msg: str, tag: Optional[str] = None):
            """Log helper that routes to the correct per-run buffer."""
            app.log_to_main(msg, tag=tag, run_id=run_id_ref[0])

        def _run():
            old_stdout = sys.stdout
            sys.stdout = _LogStream(_log, real_stdout=old_stdout)
            try:
                # Resolve the objective from optimize/evaluate.py so the GUI shares
                # the exact RF surrogate + analytic Ackley datasets the benchmark
                # harness uses. Imported lazily (pulls run_mobo) only on Start.
                try:
                    import optimize.evaluate as _ev
                except Exception as exc:
                    _log(f"ERROR: could not import optimize/evaluate.py: {exc}", tag="error")
                    return
                # resolve_dataset may sys.exit() on bad RF config — catch that too.
                try:
                    ds = _ev.resolve_dataset(dataset, rf_src, variant)
                except SystemExit as exc:
                    _log(f"ERROR resolving dataset {dataset!r}: {exc}", tag="error")
                    return
                fn_obj   = ds["fn"]
                maximize = ds["maximize"]
                d        = ds["dim"]
                _log(f"  dataset={dataset} d={d} maximize={maximize}", tag="info")

                X_a, X_e, Y_i = _gen_init_data(fn_obj, d, maximize)
                _log(f"  init: {X_a.shape[0]} pts", tag="info")
                sim_obj  = _make_sim_obj(fn_obj, DEVICE, DTYPE, maximize=maximize)
                plot_state: dict = {"line_0": None, "line_1": None}
                base_obj = _make_linebo_wrapper(sim_obj, d, DEVICE, DTYPE,
                                                plot_state=plot_state)

                analytics_payloads: list[dict] = []
                analytics_counter = [0]
                analytics_dh_ref = [None]

                def _analytics_obj(x_tell, bounds, acq_fn):
                    if stop_event.is_set():
                        raise _StopRunRequested()
                    x_req, x_act, y = base_obj(x_tell, bounds, acq_fn)
                    analytics_counter[0] += 1
                    _dh = analytics_dh_ref[0]
                    if _dh is not None:
                        _needles = _dh.needles
                        analytics_payloads.append({
                            "iter_num": analytics_counter[0],
                            "needles": (_needles.detach().cpu().numpy()
                                        if _needles is not None and _needles.shape[0] > 0
                                        else None),
                            "needle_vals": (_dh.needle_vals.detach().cpu().numpy().ravel()
                                           if _dh.needle_vals is not None
                                           and _dh.needle_vals.shape[0] > 0 else None),
                            "n_points_before": (_dh.X_all_actual.shape[0]
                                                if _dh.X_all_actual is not None else 0),
                            "line_0": plot_state.get("line_0"),
                            "line_1": plot_state.get("line_1"),
                        })
                    return x_req, x_act, y

                zombi = ZoMBIHop(
                    objective=_analytics_obj,
                    X_init_actual=X_a,
                    X_init_expected=X_e,
                    Y_init=Y_i,
                    verbose=True,
                    device=str(DEVICE),
                    dtype=DTYPE,
                    run_uuid=new_uuid,
                    resume=False,
                    checkpoint_dir=outdir,
                    num_iterations_saved=50,
                    **hp,
                )
                actual_run_dir = zombi.data_handler.run_dir
                analytics_dh_ref[0] = zombi.data_handler
                _log(f"  run_uuid: {zombi.data_handler.run_uuid}", tag="info")

                snap_records: list[tuple] = []
                _orig_snap = zombi.data_handler.take_snapshot
                _snap_dh = zombi.data_handler
                def _snap_wrap(*a, **k):
                    _orig_snap(*a, **k)
                    if _snap_dh.X_all_actual is not None:
                        snap_records.append((_snap_dh.X_all_actual.shape[0],
                                             _snap_dh.current_activation,
                                             _snap_dh.current_zoom))
                zombi.data_handler.take_snapshot = _snap_wrap

                try:
                    zombi.run(max_activations=max_act, time_limit_hours=None,
                              pause_event=pause_event)
                except _StopRunRequested:
                    _log("Run stopped by user.", tag="done")

                _log(f"Done — {actual_run_dir}", tag="done")
                if actual_run_dir is not None:
                    try:
                        _write_run_analytics(
                            zombi.data_handler, actual_run_dir,
                            analytics_payloads, snap_records, maximize, _log)
                    except Exception as exc:
                        _log(f"Analytics write failed: {exc}", tag="error")
                    app.after(0, lambda rd=actual_run_dir: app.load_run(rd))
            except Exception as exc:
                _log(f"ERROR: {exc}", tag="error")
                _log(traceback.format_exc(), tag="error")
            finally:
                sys.stdout = old_stdout
                rid = run_id_ref[0]
                if rid:
                    app.after(0, lambda r=rid: app._unregister_active_run(r))

        threading.Thread(target=_run, daemon=True).start()

    def _start_hardware(self):
        self._app.launch_hardware_process(
            uuid         = self._uuid_var.get().strip() or None,
            dims_raw     = self._hw_dims_var.get().strip(),
            script       = self._hw_script_var.get(),
            python       = self._hw_python_var.get(),
            ckpt_dir     = self._app.ckpt_dir,
            hparams_path = self._hw_hparams_var.get().strip() or None,
        )
        self.destroy()


# ── hardware-resume dialog ────────────────────────────────────────────────────

class HardwareResumeDialog(tk.Toplevel):
    """Dialog for loading / resuming a hardware run from the run browser."""

    def __init__(self, parent, app: "ZoMBIApp", run_info: dict):
        super().__init__(parent)
        self.title("Load / Resume Hardware Run")
        self.resizable(False, False)
        self.geometry("480x300")
        self._app      = app
        self._run_info = run_info
        self._build()
        self.grab_set()

    def _build(self):
        run_id  = self._run_info["run_id"]
        run_dir = Path(self._run_info["run_dir"])

        # Load saved hw dims if available
        saved_dims = "0,8,9"
        hw_cfg = run_dir / "hw_config.json"
        if hw_cfg.exists():
            try:
                saved_dims = json.loads(hw_cfg.read_text()).get("dims", saved_dims)
            except Exception:
                pass
        else:
            # Fall back to config.json
            cfg = run_dir / "config.json"
            if cfg.exists():
                try:
                    saved_dims = json.loads(cfg.read_text()).get("dims", saved_dims)
                except Exception:
                    pass

        _proj_root = Path(__file__).resolve().parent.parent

        pad = {"padx": 8, "pady": 4}
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)

        row = 0
        ttk.Label(f, text="UUID (run ID):").grid(row=row, column=0, sticky="w", **pad)
        ttk.Label(f, text=run_id, foreground="#007700",
                  font=("Consolas", 9, "bold")).grid(row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(f, text="Optimizing dims:").grid(row=row, column=0, sticky="w", **pad)
        self._dims_var = tk.StringVar(value=saved_dims)
        ttk.Entry(f, textvariable=self._dims_var, width=24).grid(
            row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(f, text="Script path:").grid(row=row, column=0, sticky="w", **pad)
        self._script_var = tk.StringVar(
            value=str(_proj_root / "scripts" / "main.py"))
        sf = ttk.Frame(f)
        sf.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Entry(sf, textvariable=self._script_var, width=30).pack(side="left")
        ttk.Button(sf, text="…", width=3, command=self._browse_script).pack(side="left")

        row += 1
        ttk.Label(f, text="Python executable:").grid(row=row, column=0, sticky="w", **pad)
        self._python_var = tk.StringVar(value=sys.executable)
        ttk.Entry(f, textvariable=self._python_var, width=34).grid(
            row=row, column=1, sticky="w", **pad)

        row += 1
        ttk.Label(f, text="Hyperparameters JSON (optional):").grid(
            row=row, column=0, sticky="w", **pad)
        self._hparams_var = tk.StringVar(value="")
        hpf = ttk.Frame(f)
        hpf.grid(row=row, column=1, sticky="w", pady=4)
        ttk.Entry(hpf, textvariable=self._hparams_var, width=26).pack(side="left")
        ttk.Button(hpf, text="…", width=3, command=self._browse_hparams).pack(side="left")

        row += 1
        ttk.Separator(f, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8)

        row += 1
        bf = ttk.Frame(f)
        bf.grid(row=row, column=0, columnspan=2)
        ttk.Button(bf, text="View Only",
                   command=self._view_only).pack(side="left", padx=6)
        ttk.Button(bf, text="Resume Hardware Run",
                   command=self._resume).pack(side="left", padx=6)
        ttk.Button(bf, text="Cancel",
                   command=self.destroy).pack(side="left", padx=6)

    def _browse_script(self):
        p = filedialog.askopenfilename(
            initialdir=str(Path(self._script_var.get()).parent),
            filetypes=[("Python", "*.py"), ("All", "*")])
        if p:
            self._script_var.set(p)

    def _browse_hparams(self):
        p = filedialog.askopenfilename(
            title="Select hyperparameter JSON",
            filetypes=[("JSON", "*.json"), ("All", "*")])
        if p:
            self._hparams_var.set(p)

    def _view_only(self):
        self.destroy()
        self._app.load_run(self._run_info["run_dir"])

    def _resume(self):
        self._app.launch_hardware_process(
            uuid         = self._run_info["run_id"].removeprefix("run_"),
            dims_raw     = self._dims_var.get().strip(),
            script       = self._script_var.get(),
            python       = self._python_var.get(),
            ckpt_dir     = self._app.ckpt_dir,
            hparams_path = self._hparams_var.get().strip() or None,
        )
        self.destroy()


# ── main application window ───────────────────────────────────────────────────

class ZoMBIApp(tk.Tk):
    def __init__(self, ckpt_dir: str = DEFAULT_CKPT_DIR):
        super().__init__()
        self.title("ZoMBI-Hop Interface")
        self.geometry("1500x900")
        self.ckpt_dir = ckpt_dir
        self.current_run: Optional[RunData] = None
        self._poll_job: Optional[str] = None
        # Multi-run tracking:
        #   _active_runs: run_id (dir name, e.g. "run_b470") →
        #       {"pause_event": Event, "log_buffer": [(msg, tag), ...]}
        self._active_runs: dict[str, dict] = {}
        # Which run's log is currently shown in the log panel
        self._viewed_run_id: Optional[str] = None
        self._build_ui()
        self._start_poll()

    def _build_ui(self):
        # ── top toolbar ───────────────────────────────────────────────────
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=4, pady=(4, 0))
        self._pause_btn = ttk.Button(toolbar, text="⏸ Pause",
                                     command=self._toggle_pause, state="disabled")
        self._pause_btn.pack(side="left", padx=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=4)

        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=4, pady=(4, 0))

        # ── left sidebar ──────────────────────────────────────────────────
        left = ttk.Frame(pw)
        pw.add(left, weight=0)
        left.config(width=310)
        left.pack_propagate(False)

        self.run_browser = RunBrowserPanel(left, self)
        self.run_browser.pack(fill="both", expand=True)

        self._cfg_panel = ConfigPanel(left)
        self._cfg_panel.pack(fill="x", pady=4)

        # ── right area ────────────────────────────────────────────────────
        right = ttk.Frame(pw)
        pw.add(right, weight=1)

        right_pw = ttk.PanedWindow(right, orient="vertical")
        right_pw.pack(fill="both", expand=True)

        right_top = ttk.Frame(right_pw)
        right_pw.add(right_top, weight=3)

        self._snap_slider = SnapshotSliderFrame(right_top, self)
        self._snap_slider.pack(fill="x", padx=4, pady=2)

        self._nb = ttk.Notebook(right_top)
        self._nb.pack(fill="both", expand=True, padx=4, pady=2)

        self._conv  = ConvergencePlotFrame(self._nb, figsize=(9, 4))
        self._dist  = DistancePlotFrame   (self._nb, figsize=(9, 4))
        self._pts   = PointsTableFrame    (self._nb)
        self._neds  = NeedlesFrame        (self._nb)
        self._gp    = GPQueryFrame        (self._nb, self)
        self._tern  = TernaryPlotFrame    (self._nb)

        self._nb.add(self._conv,  text="Convergence")
        self._nb.add(self._dist,  text="Distance")
        self._nb.add(self._pts,   text="Points")
        self._nb.add(self._neds,  text="Needles")
        self._nb.add(self._gp,    text="GP Query")
        self._nb.add(self._tern,  text="Ternary")
        self._manual = ManualControlFrame(self._nb)
        self._nb.add(self._manual, text="Manual Ctrl")

        self._live_log = LiveLogPanel(right_pw, text="Live Run Log")
        right_pw.add(self._live_log, weight=1)

        # ── status bar ────────────────────────────────────────────────────
        self._status = tk.StringVar(value="Ready — load a run or start a new one.")
        ttk.Label(self, textvariable=self._status, relief="sunken",
                  anchor="w", font=("Consolas", 8)).pack(
            fill="x", side="bottom", padx=4, pady=2)

    def set_status(self, msg: str):
        self._status.set(msg)

    # ── per-run log routing ───────────────────────────────────────────────────

    def log_to_main(self, msg: str, tag: Optional[str] = None,
                    run_id: Optional[str] = None):
        """
        Route a log line to the live log panel.

        run_id: the directory-name key for this run (e.g. "run_b470").
        Lines are always stored in the run's log_buffer.
        They are displayed in the panel only when that run is currently viewed.
        run_id=None (legacy / hardware) always displays immediately.
        """
        if run_id and run_id in self._active_runs:
            self._active_runs[run_id]["log_buffer"].append((msg, tag))
        if run_id is None or run_id == self._viewed_run_id:
            self._live_log.log(msg, tag)

    # ── pause / resume (acts on the currently-viewed run) ────────────────────

    def _toggle_pause(self):
        info = self._active_runs.get(self._viewed_run_id) if self._viewed_run_id else None
        if info is None:
            return
        ev = info["pause_event"]
        if ev.is_set():
            ev.clear()
            self._pause_btn.config(text="▶ Resume")
            self.set_status(f"{self._viewed_run_id} PAUSED — click Resume to continue.")
        else:
            ev.set()
            self._pause_btn.config(text="⏸ Pause")
            self.set_status(f"{self._viewed_run_id} resumed.")

    def _refresh_pause_btn(self):
        """Update the Pause/Resume button label to reflect the viewed run's state."""
        info = self._active_runs.get(self._viewed_run_id) if self._viewed_run_id else None
        if info is None:
            self._pause_btn.config(state="disabled", text="⏸ Pause")
        else:
            ev = info["pause_event"]
            label = "▶ Resume" if not ev.is_set() else "⏸ Pause"
            self._pause_btn.config(state="normal", text=label)

    def _register_active_run(self, pause_event: threading.Event, run_id: str,
                             stop_event: threading.Event | None = None,
                             proc: subprocess.Popen | None = None):
        """Called (on the main thread) when a new run thread starts."""
        self._active_runs[run_id] = {
            "pause_event": pause_event,
            "stop_event": stop_event,
            "proc": proc,
            "log_buffer": [],
        }
        self.run_browser.refresh()
        if self._viewed_run_id == run_id:
            self._refresh_pause_btn()

    def _unregister_active_run(self, run_id: str):
        """Called (on the main thread) when a run thread finishes."""
        self._active_runs.pop(run_id, None)
        self.run_browser.refresh()
        if self._viewed_run_id == run_id:
            self._refresh_pause_btn()

    def stop_run(self, run_id: str):
        """Stop a running run (synthetic or hardware)."""
        info = self._active_runs.get(run_id)
        if info is None:
            return
        # Hardware run: terminate the subprocess
        proc = info.get("proc")
        if proc is not None and proc.poll() is None:
            self.log_to_main(f"Terminating hardware process for {run_id} …",
                             tag="info", run_id=run_id)
            proc.terminate()
            return
        # Synthetic run: signal the stop event
        stop_ev = info.get("stop_event")
        if stop_ev is not None:
            stop_ev.set()
            # Unblock if paused so the thread can see the stop signal
            pause_ev = info.get("pause_event")
            if pause_ev is not None and not pause_ev.is_set():
                pause_ev.set()
            self.log_to_main(f"Stop requested for {run_id} — will stop after current iteration.",
                             tag="info", run_id=run_id)
            self.set_status(f"{run_id} stopping …")

    # ── run loading ───────────────────────────────────────────────────────────

    def load_run(self, run_dir):
        rd = load_run(Path(run_dir))
        self.current_run = rd
        prev_viewed = self._viewed_run_id
        self._viewed_run_id = rd.run_id
        self._refresh_all(rd)
        self._refresh_pause_btn()

        # Rebuild the log panel for this run
        self._live_log.clear()
        if rd.run_id in self._active_runs:
            # Live run — replay the in-memory log buffer accumulated so far
            for msg, tag in self._active_runs[rd.run_id]["log_buffer"]:
                self._live_log.log(msg, tag)
        else:
            # Completed / external run — load from the persisted log file
            for line in rd.log_lines:
                text = line.split("] ", 1)[-1] if "] " in line else line
                self._live_log.log(line, tag=_log_tag_for(text))

        self.set_status(
            f"Loaded: {rd.run_id}  snap={rd.snapshot_name}  "
            f"({rd.n_points} pts, {rd.n_needles} needles)")

    def launch_hardware_process(
        self,
        uuid:         str | None = None,
        dims_raw:     str        = "",
        script:       str | None = None,
        python:       str | None = None,
        ckpt_dir:     str | None = None,
        hparams_path: str | None = None,
    ):
        """Spawn scripts/main.py as a subprocess and stream its output to the log panel.

        uuid         – resume UUID (None = new run)
        dims_raw     – comma-separated dim indices, e.g. "0,8,9"
        script       – path to main.py
        python       – python executable
        ckpt_dir     – checkpoint directory (defaults to self.ckpt_dir)
        hparams_path – optional path to a trial.json-style hyperparameter file
        """
        if script   is None: script   = str(Path(__file__).resolve().parent.parent / "scripts" / "main.py")
        if python   is None: python   = sys.executable
        if ckpt_dir is None: ckpt_dir = self.ckpt_dir

        proj_root = str(Path(script).resolve().parent.parent)

        # For a new run, pre-generate the UUID so the run dir can be created and
        # shown in the explorer the moment the run is launched (the subprocess
        # adopts it via --run-uuid). Resume runs already know their UUID.
        new_uuid = None if uuid else str(uuid4())[:4]

        cmd = [python, script] + ([uuid] if uuid else [])
        if dims_raw:
            cmd += ["--dims", dims_raw]
        cmd += ["--checkpoint-dir", ckpt_dir]
        if hparams_path:
            cmd += ["--hparams", hparams_path]
        if new_uuid:
            cmd += ["--run-uuid", new_uuid]

        self.log_to_main(f"Launching: {' '.join(cmd)}", tag="info")

        # Register the run_dir immediately so it appears in the explorer the
        # moment the run is launched — for both resume (known UUID) and new runs
        # (pre-generated UUID above).
        known_uuid = uuid or new_uuid
        hw_run_id = f"run_{known_uuid}" if known_uuid else None
        if known_uuid:
            run_dir = Path(ckpt_dir) / hw_run_id
            self.after(0, lambda rd=run_dir, d=dims_raw: self.track_new_run(rd, hw_dims=d))

        _RE_UUID = re.compile(r"Starting new trial with UUID:\s*(\S+)", re.IGNORECASE)
        app = self
        _dims = dims_raw
        proc_ref: list[subprocess.Popen | None] = [None]

        def _pump():
            try:
                import os as _os
                _env = {**_os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    cwd=proj_root,
                    env=_env,
                )
                proc_ref[0] = proc
                if hw_run_id:
                    app.after(0, lambda: app._register_active_run(
                        threading.Event(), hw_run_id, proc=proc))
                for line in proc.stdout:
                    stripped = line.rstrip()
                    if not stripped:
                        continue
                    rid = hw_run_id
                    app.log_to_main(stripped, run_id=rid)
                    if not uuid:          # only parse UUID for new runs
                        m = _RE_UUID.search(stripped)
                        if m:
                            rd = Path(ckpt_dir) / f"run_{m.group(1)}"
                            app.after(0, lambda r=rd, d=_dims: app.track_new_run(r, hw_dims=d))
                proc.wait()
                rc = proc.returncode
                app.log_to_main(
                    f"Hardware process exited (rc={rc})",
                    tag="done" if rc == 0 else "error",
                    run_id=hw_run_id)
                app.after(0, app.run_browser.refresh)
                app.after(0, lambda: app.run_browser.set_hw_uuid(None))
                if hw_run_id:
                    app.after(0, lambda r=hw_run_id: app._unregister_active_run(r))
            except Exception as exc:
                app.log_to_main(f"ERROR launching hardware: {exc}", tag="error")
                app.after(0, lambda: app.run_browser.set_hw_uuid(None))
                if hw_run_id:
                    app.after(0, lambda r=hw_run_id: app._unregister_active_run(r))

        threading.Thread(target=_pump, daemon=True).start()

    def track_new_run(self, run_dir, hw_dims: str = "", hw: bool = True):
        """Register a just-started run so the poll loop watches it from the first snapshot.

        hw : True for hardware runs (writes hw_config.json + shows the green HW-run
             indicator). False for in-process synthetic runs.
        """
        run_dir_path = Path(run_dir)
        # Ensure the directory and a stub config.json exist so scan_runs() lists
        # the run immediately, before ZoMBI writes its own config.
        run_dir_path.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir_path / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(json.dumps(
                {"status": "running", "source": "hardware" if hw else "synthetic"}))
        if hw:
            # Write hw_config.json with dims so Resume dialog can pre-fill them.
            hw_cfg = run_dir_path / "hw_config.json"
            hw_cfg.write_text(json.dumps({"dims": hw_dims, "source": "hardware"}))

        try:
            rd = load_run(run_dir_path)
        except Exception:
            rd = RunData()
            rd.run_dir = run_dir_path
            rd.run_id  = run_dir_path.name
        self.current_run = rd
        self._viewed_run_id = rd.run_id
        self._refresh_all(rd)
        self.run_browser.refresh()
        if hw:
            self.run_browser.set_hw_uuid(rd.run_id)
        self.set_status(f"Running: {rd.run_id} — waiting for first snapshot …")

    def load_snapshot(self, snapshot_name: str):
        if self.current_run is None:
            return
        rd = load_run(self.current_run.run_dir, snapshot_name)
        self.current_run = rd
        self._refresh_plots(rd)
        self.set_status(
            f"{rd.run_id}  snap={snapshot_name}  "
            f"({rd.n_points} pts, {rd.n_needles} needles)")

    def _refresh_all(self, rd: RunData):
        self._cfg_panel.update(rd)
        self._snap_slider.set_snapshots(rd.snapshots, rd.snapshot_name)
        self._refresh_plots(rd)

    def _refresh_plots(self, rd: RunData):
        self._conv.update(rd)
        self._dist.update(rd)
        self._pts.update(rd)
        self._neds.update(rd)
        self._gp.update_for_run(rd)
        self._tern.update(rd)

    def open_new_run_dialog(self):
        NewRunDialog(self, self)

    def _start_poll(self):
        self._poll_job = self.after(POLL_MS, self._poll)

    def _poll(self):
        rd = self.current_run
        if rd is not None:
            try:
                latest_path = rd.run_dir / "latest.txt"
                if latest_path.exists():
                    latest = latest_path.read_text().strip()
                    if latest != rd.snapshot_name:
                        rd_new = load_run(rd.run_dir)
                        self.current_run = rd_new
                        self._refresh_all(rd_new)
                        self.set_status(
                            f"Auto-updated: {rd_new.run_id}  "
                            f"({rd_new.n_points} pts, {rd_new.n_needles} needles)")
            except Exception:
                pass

            # Independent: check for live_plot_state.json changes (hardware runs)
            try:
                if self._tern.try_reload_live_state(rd):
                    X, n_cols, labels = self._tern._data_info()
                    self._tern._current_labels = labels
                    self._tern._sync_combos(labels, n_cols, rd)
                    self._tern._replot(X, n_cols, labels)
            except Exception:
                pass

        self._start_poll()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="ZoMBI-Hop Interface")
    p.add_argument("ckpt_dir", nargs="?", default=DEFAULT_CKPT_DIR,
                   help="Checkpoint directory to browse")
    args = p.parse_args()
    ZoMBIApp(ckpt_dir=args.ckpt_dir).mainloop()


if __name__ == "__main__":
    main()
