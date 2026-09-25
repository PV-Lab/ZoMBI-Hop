"""
Database-driven ZoMBI-Hop runner (LineBO + serial/DB handshake).

This module contains the DB-backed objective + runner that used to live in
`zombihop_linebo_v2.py`.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src import ZoMBIHop, LineBO
from src.core.linebo import (
    batch_line_bounds_segments,
    batch_line_simplex_segments,
    zero_sum_dirs,
)

from scripts import communication

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# # Default Configuration
# DEFAULT_DIMENSIONS = 10
# DEFAULT_NUM_MINIMA = 3
# DEFAULT_TIME_LIMIT_HOURS = 24.0
NUM_EXPERIMENTS = 24
NUM_INIT_DATA = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ARCHERFISH MODULES TO PRINT AND OPTIMIZE FROM
OPTIMIZING_DIMS = [0, 8, 9]

# LineBO + ZoMBI-Hop maximize internal Y. True ⇒ treat hardware/DB y as a cost
# to minimize (we negate for the GP; plots/logs show measured y).
MINIMIZE_OBJECTIVE = False


# ── composition logging ─────────────────────────────────────────────────────────
# Records, per objective call, the SENT (requested) and REAL (measured)
# compositions for BOTH hardware rails (main + cache), so expected-vs-actual can
# be reconstructed exactly later (see visualization/discrepancy.py). The optimizer
# itself only keeps the main rail's measurements; this log is the only recoverable
# record of the requested lines and of the cache rail. Written as newline-delimited
# JSON to <run_dir>/composition_log.jsonl, one record per objective call.
#
# Resume-safe: opened in append mode, with the call counter continued past records
# from before the resume. Records produced before the run directory is known (the
# initial seed-line sampling of a brand-new run) are buffered and flushed once
# set_composition_log_dir() is called.
COMPOSITION_LOG_NAME = "composition_log.jsonl"
_COMP_LOG: Dict[str, Any] = {"path": None, "buffer": [], "call": 0}


def set_composition_log_dir(run_dir: "Path | None") -> None:
    """Point the composition log at a run directory and flush buffered records.

    Safe to call more than once. On resume (file already present) the call counter
    continues from the number of records already logged so indices stay unique.
    """
    if run_dir is None:
        return
    path = Path(run_dir) / COMPOSITION_LOG_NAME
    _COMP_LOG["path"] = path
    if _COMP_LOG["call"] == 0 and path.exists():
        try:
            with open(path, "r") as fh:
                _COMP_LOG["call"] = sum(1 for _ in fh)
        except Exception:
            pass
    if _COMP_LOG["buffer"]:
        buffered, _COMP_LOG["buffer"] = _COMP_LOG["buffer"], []
        for rec in buffered:
            _write_composition_record(rec)


def _write_composition_record(record: Dict[str, Any]) -> None:
    path = _COMP_LOG["path"]
    if path is None:
        _COMP_LOG["buffer"].append(record)
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"[comp-log] WARNING: could not write composition record: {exc}")


def log_compositions(
    sent_main: np.ndarray,
    sent_cache: np.ndarray,
    endpoints_log_ref: "Dict[str, Any] | None",
    measured: np.ndarray,
    y_measured: np.ndarray,
    valid_indices: np.ndarray,
    num_experiments: int,
) -> None:
    """Append one record describing both rails of a single objective call.

    All compositions are in the apparatus/optimizer space (``OPTIMIZING_DIMS``).
    ``measured`` / ``y_measured`` are only the rows the apparatus returned (NaN
    rows already dropped); ``valid_indices`` maps each returned row back to its
    position in the concatenated send order ``[main(0..N-1), cache(N..2N-1)]`` so
    the two rails are separated exactly even when some points are missing.
    """
    try:
        sent = {0: np.asarray(sent_main, float), 1: np.asarray(sent_cache, float)}
        rails: Dict[int, Dict[str, list]] = {
            0: {"sent": [], "measured": [], "y": []},
            1: {"sent": [], "measured": [], "y": []},
        }
        meas = np.atleast_2d(np.asarray(measured, float))
        yv = np.asarray(y_measured, float).ravel()
        for j, idx in enumerate(np.asarray(valid_indices).ravel().astype(int)):
            rail = 0 if idx < num_experiments else 1
            within = int(idx) - (0 if rail == 0 else num_experiments)
            src = sent[rail]
            if 0 <= within < len(src):
                rails[rail]["sent"].append(src[within].tolist())
                rails[rail]["measured"].append(meas[j].tolist() if j < len(meas) else [])
                rails[rail]["y"].append(float(yv[j]) if j < len(yv) else float("nan"))

        ep = endpoints_log_ref or {}

        def _ep(key: str):
            v = ep.get(key)
            return np.asarray(v, float).tolist() if v is not None else None

        record = {
            "ts": time.time(),
            "call": _COMP_LOG["call"],
            "optimizing_dims": list(OPTIMIZING_DIMS),
            "num_experiments": int(num_experiments),
            "rails": [
                {
                    "name": "main",
                    "sent_endpoints": [_ep("line_0_left"), _ep("line_0_right")],
                    "sent": rails[0]["sent"],
                    "measured": rails[0]["measured"],
                    "y": rails[0]["y"],
                },
                {
                    "name": "cache",
                    "sent_endpoints": [_ep("line_1_left"), _ep("line_1_right")],
                    "sent": rails[1]["sent"],
                    "measured": rails[1]["measured"],
                    "y": rails[1]["y"],
                },
            ],
        }
        _COMP_LOG["call"] += 1
        _write_composition_record(record)
    except Exception as exc:
        print(f"[comp-log] WARNING: failed to build composition record: {exc}")

# Built-in (arbitrary) ZoMBI-Hop hyperparameters used when no --hparams JSON file
# is supplied. Single source of truth lives in src/default_hparams.py — edit
# there, not here. The hardware runner folds in the measured physical input
# noise (not a tuned hyperparameter, so kept out of DEFAULT_HPARAMS).
from src.default_hparams import DEFAULT_HPARAMS, DEFAULT_INPUT_NOISE
DEFAULT_HW_HPARAMS: Dict[str, Any] = {**DEFAULT_HPARAMS, "input_noise": DEFAULT_INPUT_NOISE}

# ZoMBIHop tunable kwargs that may be set from a hyperparameter JSON file.
_VALID_HPARAM_KEYS = {
    "max_zooms", "max_iterations", "top_m_points", "n_restarts", "raw",
    "input_noise_threshold_mult",
    "output_noise_threshold_mult", "n_consecutive_converged", "max_gp_points",
    "repulsion_lambda", "acquisition_type", "ucb_beta", "nat_grad_step",
    "nat_grad_max_steps", "ellipsoid_drop_fraction", "ellipsoid_eigenvalue_floor",
    "max_penalty_radius", "paring_spatial_halfnoise", "paring_y_noise_multiplier",
    "input_noise", "needle_shrink_factor", "needle_stop_noise_multiplier",
    "zoom_jaccard_threshold", "bounds_shrink_factor", "min_axis_noise_mult",
    "jaccard_window", "jaccard_threshold",
}


def _load_hparams_file(path: str) -> Dict[str, Any]:
    """
    Load hyperparameters from a trial.json-style file.

    Accepts a top-level ``"hparams"`` object (as in optimize/runs/.../trial.json)
    or a flat ``{name: value}`` object. Returns only keys ZoMBIHop accepts;
    unknown keys (metrics, trial, phase, …) are ignored. Returns {} on failure.
    """
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[ZoMBI] WARNING: could not read hyperparameters from {path!r}: {exc}")
        return {}
    if not isinstance(data, dict):
        print(f"[ZoMBI] WARNING: {path!r} is not a JSON object; ignoring.")
        return {}
    hp = data.get("hparams", data)
    if not isinstance(hp, dict):
        print(f"[ZoMBI] WARNING: 'hparams' in {path!r} is not an object; ignoring.")
        return {}
    return {k: v for k, v in hp.items() if k in _VALID_HPARAM_KEYS}


def y_measured_to_optimizer(y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Map apparatus / DB y to the value the GP + acquisition maximize."""
    if not MINIMIZE_OBJECTIVE:
        return y
    if isinstance(y, torch.Tensor):
        return -y
    return -np.asarray(y, dtype=np.float64)


def y_optimizer_to_measured(y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Inverse: internal (max) y → physical measured y for plots and logs."""
    if not MINIMIZE_OBJECTIVE:
        return y
    if isinstance(y, torch.Tensor):
        return -y
    return -np.asarray(y, dtype=np.float64)

_SQRT3_2 = np.sqrt(3) / 2


def _composition_to_ternary_xy(comp: np.ndarray) -> np.ndarray:
    """
    Map 3-component compositions (rows sum ~1) to Cartesian coordinates in the
    standard equilateral ternary: corner 0 at origin, corner 1 at (1,0), corner 2 at (0.5, √3/2).
    """
    p = np.asarray(comp, dtype=float)
    if p.size == 0:
        return np.zeros((0, 2))
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    p = p / s
    if p.shape[1] != 3:
        raise ValueError(f"ternary expects 3 columns, got shape {p.shape}")
    x = p[:, 1] + 0.5 * p[:, 2]
    y = _SQRT3_2 * p[:, 2]
    return np.column_stack([x, y])


def _draw_ternary_frame(ax, corner_labels: Tuple[str, str, str]) -> None:
    """Draw triangle outline and corner labels."""
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, _SQRT3_2 + 0.12)
    ax.axis("off")
    ax.text(-0.02, -0.03, corner_labels[0], ha="right", va="top", fontsize=9)
    ax.text(1.02, -0.03, corner_labels[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + 0.02, corner_labels[2], ha="center", va="bottom", fontsize=9)


# --- Live plotting and iteration logging ---
def _accumulate_plot_data(
    all_sample_num: List[float],
    all_y: List[float],
    all_x_actual: List[np.ndarray],
    new_x_actual: np.ndarray,
    new_y: np.ndarray,
) -> None:
    """Append new (x, y) batch to the running plot lists."""
    new_x = np.atleast_2d(new_x_actual)
    new_y_flat = np.atleast_1d(new_y).ravel()
    n_new = len(new_y_flat)
    for i in range(n_new):
        all_sample_num.append(len(all_sample_num) + 1)
        all_y.append(float(new_y_flat[i]))
        all_x_actual.append(new_x[i] if new_x.shape[0] > i else new_x[0])


def _endpoints_to_list(ref: Dict[str, Any]) -> List:
    """Extract up to 2 line-endpoint pairs from a log-ref dict."""
    out = []
    for prefix in ("line_0", "line_1"):
        lft = ref.get(f"{prefix}_left")
        rgt = ref.get(f"{prefix}_right")
        if lft is not None and rgt is not None:
            try:
                out.append([lft.tolist(), rgt.tolist()])
            except Exception:
                pass
    return out


def _write_live_plot_state(
    run_dir: Path,
    optimizing_dims: List[int],
    all_x_actual: List[np.ndarray],
    all_y: List[float],
    endpoints_log_ref: Dict[str, Any],
    needle_plot_points: List[Dict[str, float]],
    zoom_bounds: "torch.Tensor | None" = None,
    prior_endpoints_ref: Dict[str, Any] | None = None,
    optimizer_ref: List | None = None,
) -> None:
    """Write current optimisation state to <run_dir>/live_plot_state.json for the GUI ternary tab."""
    if run_dir is None or not all_x_actual:
        return
    try:
        state: Dict[str, Any] = {
            "optimizing_dims": list(optimizing_dims),
            "x_actual":  [x.tolist() for x in all_x_actual],
            "y_values":  [float(y) for y in all_y],
            "line_endpoints":       _endpoints_to_list(endpoints_log_ref) if endpoints_log_ref else [],
            "prior_line_endpoints": _endpoints_to_list(prior_endpoints_ref) if prior_endpoints_ref else [],
            "needles": [],
            "penalty_regions": [],   # [{center: [...], radius_ilr: float}]
            "penalty_mask": None,    # per-point bool aligned with x_actual (True = active)
            "zoom_bounds_lo": None,
            "zoom_bounds_hi": None,
        }
        for n in needle_plot_points:
            try:
                # Prefer the needle's true composition recorded at declaration.
                # Fall back to the sample-index lookup only for legacy records
                # that predate the "point" field.  Note: sample_idx indexes the
                # DataHandler arrays (which include the initial data) while
                # all_x_actual does not, so the lookup is offset and only kept
                # for backward compatibility.
                pt = n.get("point")
                if pt is None:
                    si = int(n["sample_idx"]) - 1
                    if 0 <= si < len(all_x_actual):
                        pt = all_x_actual[si].tolist()
                if pt is not None:
                    state["needles"].append({
                        "point": list(pt),
                        "y": float(n["y"]),
                    })
            except Exception:
                pass
        if zoom_bounds is not None:
            try:
                state["zoom_bounds_lo"] = zoom_bounds[0].cpu().numpy().tolist()
                state["zoom_bounds_hi"] = zoom_bounds[1].cpu().numpy().tolist()
            except Exception:
                pass
        # Penalty regions + per-point penalty mask from the optimizer's handler
        if optimizer_ref is not None and optimizer_ref[0] is not None:
            try:
                dh = optimizer_ref[0].data_handler
                n_comp = dh.needles.cpu().numpy()       # (k, d)
                n_rads = dh.needle_penalty_radii.cpu().numpy().ravel()  # (k,)
                for i in range(len(n_rads)):
                    state["penalty_regions"].append({
                        "center": n_comp[i].tolist(),
                        "radius_ilr": float(n_rads[i]),
                    })
            except Exception:
                pass
            # Per-point pruned/pared flag, for the ternary's white/black outlines.
            # get_penalty_mask() aligns with dh.X_all_actual (which *includes* the
            # initial data); all_x_actual does not, so take the tail that lines up.
            try:
                dh = optimizer_ref[0].data_handler
                full_mask = dh.get_penalty_mask()
                if full_mask is not None:
                    fm = full_mask.detach().cpu().numpy().ravel().astype(bool)
                    n_live = len(all_x_actual)
                    if len(fm) >= n_live:
                        state["penalty_mask"] = fm[len(fm) - n_live:].tolist()
            except Exception:
                pass
        (run_dir / "live_plot_state.json").write_text(json.dumps(state))
    except Exception:
        pass


def setup_live_plots() -> Tuple[Dict[str, Any], List[float], List[float], List[np.ndarray], Dict[str, Any]]:
    """Return (fig_ref, all_sample_num, all_y, all_x_actual, last_call_ref). No window yet; plots are created on first update and then closed/recreated each time."""
    if not _HAS_MPL:
        return {}, [], [], [], {}
    fig_ref: Dict[str, Any] = {}  # will hold 'fig' so we can close it before recreating
    return fig_ref, [], [], [], {}


def update_live_plots(
    fig_ref: Dict[str, Any],
    all_sample_num: List[float],
    all_y: List[float],
    all_x_actual: List[np.ndarray],
    new_x_actual: np.ndarray,
    new_y: np.ndarray,
    needle_plot_points: List[Dict[str, float]] | None = None,
    suggested_lines: List[Tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Append new points, close previous plot window, create a new figure with full state, show it (non-blocking). Plots are not kept active."""
    if not _HAS_MPL:
        return
    _accumulate_plot_data(all_sample_num, all_y, all_x_actual, new_x_actual, new_y)
    if not all_x_actual:
        return
    # Close previous figure so we don't keep active windows
    prev_fig = fig_ref.get("fig")
    if prev_fig is not None:
        try:
            plt.close(prev_fig)
        except Exception:
            pass
        fig_ref["fig"] = None
    X = np.array(all_x_actual)
    center = np.mean(X, axis=0)
    # center = np.ones((len(OPTIMIZING_DIMS),)) * 1.0 / len(OPTIMIZING_DIMS)
    distances = np.linalg.norm(X - center, axis=1)
    n_pts = len(all_y)
    y_arr = np.asarray(all_y, dtype=float)
    y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
    if y_max <= y_min:
        y_max = y_min + 1e-9

    d_comp = X.shape[1]
    use_ternary = d_comp == 3
    if use_ternary:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax3 = None

    _mode = "minimize" if MINIMIZE_OBJECTIVE else "maximize"
    fig.suptitle(f"ZoMBI-Hop live ({_mode} measured y)")
    ax1.set_xlabel("Sample number")
    ax1.set_ylabel("Measured objective")
    ax1.set_title("Measured objective vs sample number")
    ax1.plot(all_sample_num, all_y, "b.-", markersize=4, alpha=0.7, zorder=1)
    sc1 = ax1.scatter(all_sample_num, all_y, c=y_arr, cmap="viridis", s=25, ec="k", lw=0.25, zorder=2, vmin=y_min, vmax=y_max)
    fig.colorbar(sc1, ax=ax1, label="Measured objective", fraction=0.046, pad=0.02)

    ax2.set_xlabel("Distance from center of mass")
    ax2.set_ylabel("Measured objective")
    ax2.set_title("Measured objective vs distance from center")
    iteration_order = np.arange(n_pts)
    scatter_ax2 = ax2.scatter(
        distances,
        all_y,
        c=y_arr,
        ec="k",
        lw=0.3,
        cmap="viridis",
        s=22,
        alpha=1,
        vmin=y_min,
        vmax=y_max,
    )
    fig.colorbar(scatter_ax2, ax=ax2, label="Measured objective", fraction=0.046, pad=0.02)

    if needle_plot_points:
        for n in needle_plot_points:
            # Needle y from ZoMBI is in optimizer (max) space; plot measured y
            ny = float(y_optimizer_to_measured(n["y"]))
            ax1.scatter(n["sample_idx"], ny, marker="*", s=200, c="gold", zorder=5, edgecolors="darkgoldenrod")
            ax2.scatter(n["distance"], ny, marker="*", s=200, c="gold", zorder=5, edgecolors="darkgoldenrod")

    if use_ternary and ax3 is not None:
        labels = tuple(f"dim {i}" for i in OPTIMIZING_DIMS)
        _draw_ternary_frame(ax3, labels)
        ax3.set_title("Composition (ternary) — points colored by measured objective")

        Xn = normalize_last_axis(X)
        xy = _composition_to_ternary_xy(Xn)
        sc3 = ax3.scatter(
            xy[:, 0], xy[:, 1], c=y_arr, cmap="viridis", s=28, ec="k", lw=0.3, zorder=3, vmin=y_min, vmax=y_max
        )
        fig.colorbar(sc3, ax=ax3, label="Measured objective", fraction=0.046, pad=0.02)

        if suggested_lines:
            line_styles = [("-", "C0", 2.0), ("--", "C1", 2.0)]
            for i, seg in enumerate(suggested_lines[:2]):
                if seg is None or len(seg) != 2:
                    continue
                a, b = np.asarray(seg[0], dtype=float).ravel(), np.asarray(seg[1], dtype=float).ravel()
                if a.size != 3 or b.size != 3:
                    continue
                la = normalize_last_axis(a.reshape(1, 3))
                lb = normalize_last_axis(b.reshape(1, 3))
                lxy = _composition_to_ternary_xy(np.vstack([la, lb]))
                sty, col, lw = line_styles[i % len(line_styles)]
                ax3.plot(lxy[:, 0], lxy[:, 1], linestyle=sty, color=col, lw=lw, alpha=0.85, zorder=4, label=f"LineBO line {i}")

        if needle_plot_points:
            for n in needle_plot_points:
                si = int(n["sample_idx"]) - 1
                if 0 <= si < len(all_x_actual):
                    pn = normalize_last_axis(np.asarray(all_x_actual[si], dtype=float).reshape(1, 3))
                    nxy = _composition_to_ternary_xy(pn)
                    ax3.scatter(
                        nxy[0, 0], nxy[0, 1], marker="*", s=280, c="gold", zorder=6, edgecolors="darkgoldenrod", linewidths=1.0
                    )

        h, lab = ax3.get_legend_handles_labels()
        if h:
            ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig_ref["fig"] = fig
    plt.show(block=False)
    plt.pause(0.001)  # allow GUI to update


def log_iteration(
    candidate: torch.Tensor,
    endpoints_top2: Dict[str, Any],
    x_expected: torch.Tensor,
    x_actual: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """Log candidate, best two endpoints from LineBO, and resultant expected, actual, y (measured) to terminal."""
    print("\n" + "=" * 60)
    print("[ITERATION LOG]")
    print("  candidate (x_tell):", candidate.cpu().numpy().tolist())
    if endpoints_top2:
        print("  best two endpoints (LineBO):")
        print("    line_0 left :", endpoints_top2.get("line_0_left", np.array([])).tolist())
        print("    line_0 right:", endpoints_top2.get("line_0_right", np.array([])).tolist())
        print("    line_1 left :", endpoints_top2.get("line_1_left", np.array([])).tolist())
        print("    line_1 right:", endpoints_top2.get("line_1_right", np.array([])).tolist())
    print("  expected / sent (LineBO x_requested — the points ZoMBI-Hop wanted):")
    x_exp = x_expected.cpu().numpy()
    for i in range(len(x_exp)):
        print("   ", x_exp[i].tolist())
    print(f"    ({len(x_exp)} expected points)")
    print("  actual / received (x_actual measured by hardware):")
    x_act = x_actual.cpu().numpy()
    for i in range(len(x_act)):
        print("   ", x_act[i].tolist())
    print(f"    ({len(x_act)} received points)")
    y_flat = y.cpu().numpy().ravel()
    _tag = "measured (minimize)" if MINIMIZE_OBJECTIVE else "measured (maximize)"
    y_print = y.reshape(-1)
    print(
        f"  y ({_tag}):",
        y_print.tolist() if len(y_print) <= 12 else y_print[:6].tolist() + ["..."] + y_print[-6:].tolist(),
    )
    print("=" * 60 + "\n")


def normalize_last_axis(arr: np.ndarray) -> np.ndarray:
    """Normalize array along last axis to sum to 1, handling edge cases."""
    a = np.asarray(arr, dtype=float)
    sums = a.sum(axis=-1, keepdims=True)
    sums = np.where(sums == 0, 1.0, sums)
    result = a / sums
    mask = ~np.isfinite(result).all(axis=-1, keepdims=True)
    if np.any(mask):
        d = a.shape[-1]
        result = np.where(mask, 1.0 / d, result)
    return result


def get_y_measurements(
    x,
    db: str = "./sql/objective.db",
    verbose: bool = False,
    ready_for_objectives: bool = False,
    return_indices: bool = False,
):
    """
    Read objective values (and compositions → x_meas) from the objective DB.

    Handshake (when ready_for_objectives=True):
    - Receiver sets handshake.new_objective_available = 1 when it writes a new objective row.
    - We wait until flag == 1, then read the objective table, then set flag = 0 (consumed).
    - This avoids reading stale data. On resume, reset_objective() clears the table and
      sets flag = 0 so the first read waits for fresh data from the apparatus.

    When ``return_indices`` is True the return is ``(y, x_meas, valid_indices)``
    where ``valid_indices`` maps each returned (non-NaN) row back to its row in the
    input ``x`` send order, so callers can attribute rows to the correct rail even
    when some points were dropped.
    """
    import os as _os
    import sqlite3
    import time as _time

    consecutive_errors = 0
    max_consecutive_errors = 10

    if ready_for_objectives:
        while True:
            try:
                conn = sqlite3.connect(db, timeout=10.0)
                cur = conn.cursor()
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS handshake (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    new_objective_available INTEGER DEFAULT 0
                )"""
                )
                cur.execute("INSERT OR IGNORE INTO handshake (id, new_objective_available) VALUES (1, 0)")
                cur.execute("SELECT new_objective_available FROM handshake WHERE id = 1")
                flag = cur.fetchone()
                conn.close()
                if flag and flag[0] == 1:
                    break
                _time.sleep(1)
            except Exception:
                _time.sleep(1)
                continue

    while True:
        try:
            if not _os.path.exists(db):
                _time.sleep(1)
                continue
            from scripts.communication import _objective_db_lock, _objective_writing

            if _objective_writing:
                _time.sleep(0.1)
                continue
            with _objective_db_lock:
                conn = sqlite3.connect(db, timeout=30.0)
                cur = conn.cursor()
                cur.execute("SELECT * FROM objective")
                all_rows = cur.fetchall()
                if not all_rows:
                    conn.close()
                    _time.sleep(1)
                    continue
                if len(all_rows) == 1 and len(all_rows[0]) > 1:
                    flat = list(all_rows[0])
                elif len(all_rows) > 1 and len(all_rows[0]) == 1:
                    flat = [r[0] for r in all_rows]
                else:
                    conn.close()
                    _time.sleep(1)
                    continue
                y_all = np.array(flat, dtype=float)
                valid_mask = ~np.isnan(y_all)
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) == 0:
                    conn.close()
                    _time.sleep(1)
                    continue
                y = y_all[valid_indices].reshape(-1)
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='compositions'",
                )
                if cur.fetchone():
                    cur.execute("SELECT * FROM compositions")
                    comp_rows = cur.fetchall()
                    X_meas_full = np.array(comp_rows, dtype=float) if comp_rows else None
                else:
                    X_meas_full = None
                if X_meas_full is not None and X_meas_full.shape[0] >= len(flat):
                    x_meas = X_meas_full[valid_indices][:, OPTIMIZING_DIMS]
                elif X_meas_full is not None and X_meas_full.shape[0] > 0:
                    x_meas = np.zeros((len(valid_indices), len(OPTIMIZING_DIMS)), dtype=float)
                    n_to_copy = min(X_meas_full.shape[0], len(valid_indices))
                    x_meas[:n_to_copy] = X_meas_full[:n_to_copy][:, OPTIMIZING_DIMS]
                else:
                    x_meas = np.zeros((len(valid_indices), len(OPTIMIZING_DIMS)), dtype=float)
                if verbose:
                    print(f"[get_y_measurements] ✅ NEW DATA RECEIVED: {len(y)} objective values")
                conn.close()
                break
        except Exception:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                _time.sleep(5.0)
                consecutive_errors = 0
            else:
                _time.sleep(1)
            continue

    if ready_for_objectives:
        try:
            conn = sqlite3.connect(db, timeout=10.0)
            cur = conn.cursor()
            cur.execute("UPDATE handshake SET new_objective_available = 0 WHERE id = 1")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[get_y_measurements] Error clearing handshake flag: {e}")
    if return_indices:
        return y, x_meas, valid_indices
    return y, x_meas


def _pad_to_10d(arr):
    arr = np.atleast_2d(arr)
    out = np.zeros((arr.shape[0], 10), dtype=arr.dtype)
    out[:, OPTIMIZING_DIMS] = arr
    return out


def expected_from_actual(x_actual: torch.Tensor) -> torch.Tensor:
    """
    Compute expected (requested) points from actual points the same way LineBO does:
    points evenly spaced along the first principal direction of x_actual.
    """
    if x_actual.shape[0] > 1:
        x_centered = x_actual - x_actual.mean(dim=0, keepdim=True)
        U, S, V = torch.linalg.svd(x_centered, full_matrices=False)
        direction = V[0]  # (d,) first right singular vector
        projections = torch.matmul(x_centered, direction.unsqueeze(1)).squeeze(1)
        t_vals = torch.linspace(
            projections.min().item(),
            projections.max().item(),
            x_actual.shape[0],
            device=x_actual.device,
            dtype=x_actual.dtype,
        )
        x_requested = x_actual.mean(dim=0, keepdim=True) + t_vals.unsqueeze(1) * direction.unsqueeze(0)
    else:
        x_requested = x_actual.clone()
    return x_requested


def objective(
    endpoints: torch.Tensor,
    ready_for_objectives: bool = True,
    endpoints_log_ref: Dict[str, Any] | None = None,
    num_experiments: int = NUM_EXPERIMENTS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single objective: accepts (n, 2, d) torch tensor with n >= 2.
    Sends first two lines endpoints[0] and endpoints[1] (each 2,d = left, right) to communication,
    waits on response, returns (x_actual, y) for the first line as tensors.
    Logs the endpoints passed in to endpoints_log_ref when provided.
    """
    print(
        "[objective] called with params:\n"
        f"  endpoints.shape={getattr(endpoints, 'shape', None)}, dtype={getattr(endpoints, 'dtype', None)}, device={getattr(endpoints, 'device', None)}\n"
        f"  ready_for_objectives={ready_for_objectives}\n"
        f"  endpoints_log_ref keys={list(endpoints_log_ref.keys()) if endpoints_log_ref is not None else None}\n"
        f"  num_experiments={num_experiments}"
    )
    assert endpoints.dim() == 3 and endpoints.shape[0] >= 2 and endpoints.shape[1] == 2
    device = endpoints.device
    dtype = endpoints.dtype
    line0 = endpoints[0]  # (2, d) left, right
    line1 = endpoints[1]  # (2, d) left, right
    line_0_left = line0[0].cpu().numpy()
    line_0_right = line0[1].cpu().numpy()
    line_1_left = line1[0].cpu().numpy()
    line_1_right = line1[1].cpu().numpy()

    if endpoints_log_ref is not None:
        endpoints_log_ref["line_0_left"] = line_0_left
        endpoints_log_ref["line_0_right"] = line_0_right
        endpoints_log_ref["line_1_left"] = line_1_left
        endpoints_log_ref["line_1_right"] = line_1_right

    x_main = np.array([line_0_left + t * (line_0_right - line_0_left) for t in np.linspace(0, 1, num_experiments)])
    x_cache = np.array([line_1_left + t * (line_1_right - line_1_left) for t in np.linspace(0, 1, num_experiments)])
    left_norm = _pad_to_10d(normalize_last_axis(np.round(line_0_left, 3)))[0]
    right_norm = _pad_to_10d(normalize_last_axis(np.round(line_0_right, 3)))[0]
    x_main_norm = _pad_to_10d(normalize_last_axis(np.round(x_main, 3)))
    cache_left_norm = _pad_to_10d(normalize_last_axis(np.round(line_1_left, 3)))[0]
    cache_right_norm = _pad_to_10d(normalize_last_axis(np.round(line_1_right, 3)))[0]
    x_cache_norm = _pad_to_10d(normalize_last_axis(np.round(x_cache, 3)))

    communication.write_compositions(
        start=left_norm,
        end=right_norm,
        array=x_main_norm,
        start_cache=cache_left_norm,
        end_cache=cache_right_norm,
        array_cache=x_cache_norm,
        timestamp=time.time(),
    )
    # Clear before reading so we only accept data sent in response to this request.
    # Always reset (not just when ready_for_objectives) so stale rows from a previous
    # call never bleed into the next one.
    communication.reset_objective()
    y_all, x_meas_all, valid_indices = get_y_measurements(
        np.vstack([x_main, x_cache]), verbose=True,
        ready_for_objectives=ready_for_objectives, return_indices=True,
    )
    # Clear after reading so objective_receiver sees obj_empty=True for the next call.
    communication.reset_objective()
    # Record the sent vs. measured compositions for BOTH rails before the optimizer
    # discards the cache rail — the only recoverable record of the requested lines.
    log_compositions(
        sent_main=x_main, sent_cache=x_cache, endpoints_log_ref=endpoints_log_ref,
        measured=x_meas_all, y_measured=y_all, valid_indices=valid_indices,
        num_experiments=num_experiments,
    )
    x_meas_main = x_meas_all[:num_experiments].astype(np.float64)
    y_main = np.asarray(y_all[:num_experiments]).ravel().astype(np.float64)
    y_for_gp = y_measured_to_optimizer(y_main)
    return (
        torch.tensor(x_meas_main, device=device, dtype=dtype),
        torch.tensor(y_for_gp, device=device, dtype=dtype),
    )


def linebo_sampler_wrapper(
    dimensions: int,
    num_lines: int = 10,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
    resume_plot_data: Tuple[List[float], List[float], List[np.ndarray]] | None = None,
    needle_plot_points: List[Dict[str, float]] | None = None,
    run_dir_ref: List | None = None,
    optimizer_ref: List | None = None,
    show_live_plots: bool = True,
):
    """
    Wrapper for LineBO.sampler: calls linebo.sampler, then logs expected, actual, y
    and makes them available to all logging (live plots + log_iteration).

    Parameters
    ----------
    run_dir_ref : list[Path|None]
        Single-element mutable list.  Set [0] to the run directory after the
        ZoMBIHop instance is created so the wrapper can write live_plot_state.json.
    optimizer_ref : list[ZoMBIHop|None]
        Same pattern — used to read current zoom bounds for the ternary tab.
    show_live_plots : bool
        When False (hardware mode), suppresses the matplotlib popup window but
        still accumulates data for live_plot_state.json.
    """
    if needle_plot_points is None:
        needle_plot_points = []
    endpoints_log_ref: Dict[str, Any] = {}
    prior_endpoints_ref: Dict[str, Any] = {}   # previous iteration's endpoints
    linebo = LineBO(
        lambda ep: objective(ep, ready_for_objectives=True, endpoints_log_ref=endpoints_log_ref),
        dimensions,
        num_points_per_line=100,
        num_lines=num_lines,
        device=str(device),
    )
    fig_ref: Dict[str, Any] = {}
    all_sample_num: List[float] = []
    all_y: List[float] = []
    all_x_actual: List[np.ndarray] = []

    if resume_plot_data is not None:
        sample_nums, y_vals, x_actuals = resume_plot_data
        all_sample_num.extend(sample_nums)
        for v in y_vals:
            all_y.append(float(y_optimizer_to_measured(v)))
        all_x_actual.extend(x_actuals)
        if show_live_plots and _HAS_MPL and all_y:
            update_live_plots(
                fig_ref, all_sample_num, all_y, all_x_actual,
                np.zeros((0, dimensions)), np.array([]),
                needle_plot_points=needle_plot_points,
                suggested_lines=None,
            )

    def _flush_state() -> None:
        """Write current accumulated state to live_plot_state.json immediately."""
        if run_dir_ref is None or run_dir_ref[0] is None:
            return
        zoom_bounds = None
        if optimizer_ref is not None and optimizer_ref[0] is not None:
            try:
                zoom_bounds = optimizer_ref[0].data_handler.bounds
            except Exception:
                pass
        _write_live_plot_state(
            run_dir_ref[0], OPTIMIZING_DIMS, all_x_actual, all_y,
            endpoints_log_ref, needle_plot_points, zoom_bounds,
            prior_endpoints_ref=prior_endpoints_ref,
            optimizer_ref=optimizer_ref,
        )

    def wrapper(
        x_tell: torch.Tensor,
        bounds: torch.Tensor | None = None,
        acquisition_function=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Save the endpoints from the previous iteration before this call overwrites them
        prior_endpoints_ref.clear()
        prior_endpoints_ref.update(endpoints_log_ref)
        x_requested, x_actual, y = linebo.sampler(x_tell, bounds, acquisition_function)
        y_flat = y.reshape(-1)
        x_act_np = x_actual.cpu().numpy()
        y_meas = y_optimizer_to_measured(y_flat)
        y_plot_np = y_meas.detach().cpu().numpy()
        suggested: List[Tuple[np.ndarray, np.ndarray]] | None = None
        if endpoints_log_ref:
            try:
                suggested = [
                    (endpoints_log_ref["line_0_left"], endpoints_log_ref["line_0_right"]),
                    (endpoints_log_ref["line_1_left"], endpoints_log_ref["line_1_right"]),
                ]
            except KeyError:
                suggested = None
        if show_live_plots and _HAS_MPL:
            update_live_plots(
                fig_ref, all_sample_num, all_y, all_x_actual, x_act_np, y_plot_np,
                needle_plot_points=needle_plot_points,
                suggested_lines=suggested,
            )
        else:
            _accumulate_plot_data(all_sample_num, all_y, all_x_actual, x_act_np, y_plot_np)
        _flush_state()
        log_iteration(x_tell, endpoints_log_ref, x_requested, x_actual, y_meas)
        return x_requested, x_actual, y_flat

    return wrapper, _flush_state


def initial_lines_on_boundary(
    num_lines: int,
    bounds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    max_retries: int = 10,
) -> np.ndarray:
    """
    Generate num_lines with endpoints on the search-box boundary.

    Sample num_lines interior points inside the box-constrained simplex and
    num_lines random zero-sum directions; for each point + direction, extend to
    the boundary of ``[bounds[0], bounds[1]]`` (linebo.batch_line_bounds_segments)
    to get (x_left, x_right). Clipping to the box — not to the bare simplex face —
    matters whenever a dim is capped below 1 (e.g. [0, 0.3]): extending to the
    simplex face would seed the run with points outside the requested box.
    Returns ordered_endpoints of shape (num_lines, 2, d) where [i, 0] is left
    and [i, 1] is right endpoint on the boundary.
    """
    d = bounds.shape[1]
    low, high = bounds[0], bounds[1]
    # Interior points on simplex (sum=1, within the per-dim box)
    points = ZoMBIHop.random_simplex(num_lines, low, high, device=device)
    bounds = bounds.to(device=points.device, dtype=points.dtype)
    endpoints_list = []
    for i in range(num_lines):
        x0 = points[i]
        for _ in range(max_retries):
            direction = zero_sum_dirs(1, d, device=device, dtype=dtype).squeeze(0)
            x_left, x_right, _t_min, _t_max, mask = batch_line_bounds_segments(
                x0, direction.unsqueeze(0).to(x0.dtype), bounds
            )
            if bool(mask.any()):
                endpoints_list.append([x_left[0].cpu().numpy(),
                                       x_right[0].cpu().numpy()])
                break
        else:
            # Fallback: chord between two independently sampled in-box points, so a
            # degenerate direction never seeds a line outside the search box.
            pair = ZoMBIHop.random_simplex(2, low, high, device=device)
            endpoints_list.append([pair[0].cpu().numpy(), pair[1].cpu().numpy()])
    return np.array(endpoints_list)


def _latest_snapshot_dir(run_dir: Path) -> Path | None:
    """Return the latest snapshot directory, trying new format then old format."""
    latest_txt = run_dir / "latest.txt"
    if latest_txt.exists():
        name = latest_txt.read_text().strip()
        candidate = run_dir / "snapshots" / name
        if candidate.exists():
            return candidate
    # Old format fallback
    current_state_file = run_dir / "current_state.txt"
    if current_state_file.exists():
        label = current_state_file.read_text().strip()
        candidate = run_dir / "states" / label
        if candidate.exists():
            return candidate
    return None


def _load_bounds_from_run(run_dir: Path, device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Load bounds tensor from a saved run directory (for resuming). Handles both snapshot formats."""
    snap = _latest_snapshot_dir(run_dir)
    if snap is None:
        raise FileNotFoundError(f"No valid checkpoint found in {run_dir}; cannot load bounds for resume.")
    tensors_path = snap / "tensors.pt"
    if not tensors_path.exists():
        raise FileNotFoundError(f"No tensors.pt in {snap}.")
    tensors = torch.load(tensors_path, map_location=device, weights_only=False)
    if "bounds" not in tensors:
        raise KeyError(f"tensors.pt in {snap} has no 'bounds' key.")
    return tensors["bounds"].to(device=device, dtype=dtype)


def _load_plot_data_from_run(run_dir: Path) -> Tuple[List[float], List[float], List[np.ndarray]] | None:
    """Load all evaluated (x_actual, y) pairs from a resumed run for live plots.
    Handles both new snapshot format (tensors.pt) and old format (all_points.csv)."""
    snap = _latest_snapshot_dir(run_dir)
    if snap is None:
        return None

    # ── New format: reconstruct from tensors.pt ───────────────────────────────
    tensors_path = snap / "tensors.pt"
    if tensors_path.exists():
        try:
            tensors = torch.load(tensors_path, map_location="cpu", weights_only=False)
            X_all = tensors.get("X_all_actual")
            Y_all = tensors.get("Y_all")
            if X_all is not None and Y_all is not None and X_all.shape[0] > 0:
                X_np = X_all.numpy()
                Y_np = Y_all.numpy().ravel()
                all_sample_num = list(range(1, len(Y_np) + 1))
                all_y = [float(v) for v in Y_np]
                all_x_actual = [X_np[i] for i in range(len(Y_np))]
                return (all_sample_num, all_y, all_x_actual)
        except Exception:
            pass

    # ── Old format: all_points.csv ────────────────────────────────────────────
    csv_path = snap / "all_points.csv"
    if not csv_path.exists():
        return None
    all_sample_num = []
    all_y = []
    all_x_actual = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            x_cols = sorted(
                [c for c in reader.fieldnames if c.startswith("x_actual_") and c[len("x_actual_"):].isdigit()],
                key=lambda c: int(c.split("_")[-1]),
            )
            for row in reader:
                try:
                    y_val = float(row["y_value"])
                    x_vals = [float(row[c]) for c in x_cols]
                except (KeyError, ValueError):
                    continue
                all_sample_num.append(len(all_sample_num) + 1)
                all_y.append(y_val)
                all_x_actual.append(np.array(x_vals))
    except Exception:
        return None
    return (all_sample_num, all_y, all_x_actual) if all_y else None


def _load_needles_for_plot(
    run_dir: Path,
    all_x_actual: List[np.ndarray],
) -> List[Dict[str, float]]:
    """Load needle positions for live-plot stars. Handles both new (needles.json) and old (needles_results.json) formats."""
    out: List[Dict[str, float]] = []
    if not all_x_actual:
        return out
    snap = _latest_snapshot_dir(run_dir)
    if snap is None:
        return out

    # New format uses needles.json; old format used needles_results.json
    for fname in ("needles.json", "needles_results.json"):
        needles_path = snap / fname
        if needles_path.exists():
            try:
                with open(needles_path) as f:
                    needles_data = json.load(f)
                break
            except Exception:
                continue
    else:
        return out

    X = np.array(all_x_actual)
    center = np.mean(X, axis=0)
    for rec in needles_data:
        try:
            pt = np.array(rec["point"], dtype=float)
            y_val = float(rec["value"])
        except (KeyError, TypeError, ValueError):
            continue
        # Closest point index in all_x_actual (1-based sample number); used only
        # for the matplotlib sample-number axis.  The live_plot_state "point" is
        # taken from the needle's true composition below.
        dists = np.linalg.norm(X - pt, axis=1)
        idx = int(np.argmin(dists))
        sample_idx = idx + 1
        distance = float(np.linalg.norm(pt - center))
        out.append({"sample_idx": sample_idx, "point": pt.ravel().tolist(),
                    "y": y_val, "distance": distance})
    return out


def run_zombi_main(resume_uuid: str | None = None, optimizing_dims: list | None = None,
                   checkpoint_dir: str | None = None, hparams_path: str | None = None,
                   new_run_uuid: str | None = None,
                   bounds_lo: list | None = None, bounds_hi: list | None = None):
    """Run DB-driven ZoMBI-Hop loop (new or resume).

    hparams_path : optional path to a trial.json-style JSON file whose 'hparams'
        override DEFAULT_HW_HPARAMS for this run.
    new_run_uuid : optional caller-provided UUID for a *new* run. Lets the GUI
        pre-create and display the run directory the moment the run is launched,
        before any data has been collected. Ignored when resuming.
    bounds_lo / bounds_hi : optional per-dim lower/upper search-box bounds aligned
        to OPTIMIZING_DIMS (default 0 / 1 for every dim). A tightened box — e.g. a
        dim capped at 0.3 — constrains sampling, zoom-resets and space-filling to
        that box. On resume, if omitted, they are restored from the run's
        hw_config.json so the box survives across resumes.
    """
    global OPTIMIZING_DIMS
    if optimizing_dims is not None:
        OPTIMIZING_DIMS = list(optimizing_dims)

    # Merge built-in defaults with any user-supplied hyperparameter file.
    hw_hparams: Dict[str, Any] = dict(DEFAULT_HW_HPARAMS)
    if hparams_path:
        overrides = _load_hparams_file(hparams_path)
        if overrides:
            hw_hparams.update(overrides)
            print(f"[ZoMBI] Applied {len(overrides)} hyperparameter override(s) "
                  f"from {hparams_path}: {overrides}")
        else:
            print(f"[ZoMBI] No usable hyperparameters found in {hparams_path}; "
                  f"using built-in defaults.")

    # Clear objective DB and handshake so the first read waits for fresh data (new run or resume).
    communication.reset_objective()

    dimensions = len(OPTIMIZING_DIMS)
    device = torch.device(DEVICE)
    dtype = torch.float64

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
    else:
        ckpt_path = Path("actual_runs") / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Per-dim search box (2, dimensions). Defaults to the full [0,1] simplex, but a
    # caller (GUI) may pass a tightened box; on resume, restore it from hw_config.json
    # so the box persists. Bounds are aligned to OPTIMIZING_DIMS order.
    if (bounds_lo is None or bounds_hi is None) and resume_uuid is not None:
        try:
            _hw = ckpt_path / f"run_{resume_uuid}" / "hw_config.json"
            if _hw.exists():
                _cfg = json.loads(_hw.read_text())
                if bounds_lo is None and _cfg.get("bounds_lo"):
                    bounds_lo = [float(x) for x in str(_cfg["bounds_lo"]).split(",")]
                if bounds_hi is None and _cfg.get("bounds_hi"):
                    bounds_hi = [float(x) for x in str(_cfg["bounds_hi"]).split(",")]
        except Exception as _e:
            print(f"[ZoMBI] Could not restore bounds from hw_config.json: {_e}")
    bounds = torch.zeros((2, dimensions), device=device, dtype=dtype)
    bounds[0] = torch.tensor(bounds_lo, device=device, dtype=dtype) if bounds_lo else 0.0
    bounds[1] = torch.tensor(bounds_hi, device=device, dtype=dtype) if bounds_hi else 1.0
    if bounds_lo or bounds_hi:
        print(f"[ZoMBI] Search box: lo={bounds[0].tolist()} hi={bounds[1].tolist()}")

    resume_plot_data: Tuple[List[float], List[float], List[np.ndarray]] | None = None
    needle_plot_points: List[Dict[str, float]] = []
    if resume_uuid is not None:
        run_dir = ckpt_path / f"run_{resume_uuid}"
        if run_dir.exists():
            resume_plot_data = _load_plot_data_from_run(run_dir)
            if resume_plot_data is not None:
                print(f"[Resume] Loaded {len(resume_plot_data[1])} points into live plot.")
                resume_needles = _load_needles_for_plot(run_dir, resume_plot_data[2])
                needle_plot_points.extend(resume_needles)
                if resume_needles:
                    print(f"[Resume] Loaded {len(resume_needles)} needle(s) for plot stars.")

    run_dir_ref: List = [None]
    optimizer_ref: List = [None]

    objective_wrapper, _flush_initial_state = linebo_sampler_wrapper(
        dimensions=dimensions,
        num_lines=10,
        device=device,
        dtype=dtype,
        resume_plot_data=resume_plot_data,
        needle_plot_points=needle_plot_points,
        run_dir_ref=run_dir_ref,
        optimizer_ref=optimizer_ref,
        show_live_plots=False,  # hardware mode: use GUI ternary tab, no popup
    )

    if resume_uuid is None:
        print("=" * 80)
        print("STARTING NEW ZOMBIHOP TRIAL (DATABASE-DRIVEN)")
        print("=" * 80)
        print(f"Dimensions: {dimensions} (from OPTIMIZING_DIMS: {OPTIMIZING_DIMS})")
        print(f"Device: {device}")
        if MINIMIZE_OBJECTIVE:
            print("Objective: minimize measured y (LineBO+ZoMBI maximize -y internally).")
        else:
            print("Objective: maximize measured y.")
        print("=" * 80 + "\n")

        print("Generating initial data via database...")
        n_init_data = NUM_INIT_DATA
        ordered_endpoints = initial_lines_on_boundary(
            2 * n_init_data, bounds, device, dtype=torch.float64
        )
        n_total = len(ordered_endpoints)
        x_actual_list: List[torch.Tensor] = []
        x_expected_list: List[torch.Tensor] = []
        y_list: List[torch.Tensor] = []
        for i in range(n_init_data):
            idx0 = 2 * i
            idx1 = 2 * i + 1
            line0 = ordered_endpoints[idx0]  # (2, d) left, right
            line1 = ordered_endpoints[idx1]
            # Ensure main and cache are different (avoid same values in cache and real)
            if np.allclose(line0, line1, rtol=1e-6, atol=1e-8):
                idx1 = (2 * i + 2) % n_total
                if idx1 == idx0:
                    idx1 = (idx0 + 1) % n_total
                line1 = ordered_endpoints[idx1]
            ep = torch.tensor(
                np.stack([line0, line1], axis=0),
                device=device,
                dtype=torch.float64,
            )
            x_act, y_act = objective(ep, ready_for_objectives=True)
            x_exp = expected_from_actual(x_act)
            x_actual_list.append(x_act)
            x_expected_list.append(x_exp)
            y_list.append(y_act)
        X_init_actual = torch.cat(x_actual_list, dim=0)
        X_init_expected = torch.cat(x_expected_list, dim=0)
        Y_init = torch.cat(y_list, dim=0).reshape(-1, 1)

        # ZoMBI hyperparameters: DEFAULT_HW_HPARAMS, optionally overridden by --hparams.
        new_hparams = dict(hw_hparams)
        new_hparams.setdefault("top_m_points", max(dimensions + 1, 4))
        optimizer = ZoMBIHop(
            objective=objective_wrapper,
            X_init_actual=X_init_actual,
            X_init_expected=X_init_expected,
            Y_init=Y_init,
            device=str(device),
            dtype=dtype,
            bounds=bounds,
            run_uuid=new_run_uuid,
            resume=False,
            checkpoint_dir=str(ckpt_path),
            num_iterations_saved=50,
            verbose=True,
            needle_plot_points_ref=needle_plot_points,
            **new_hparams,
        )

        run_dir_ref[0] = ckpt_path / f"run_{optimizer.run_uuid}"
        optimizer_ref[0] = optimizer
        set_composition_log_dir(run_dir_ref[0])  # flush any buffered seed-line records
        print(f"✅ Starting new trial with UUID: {optimizer.run_uuid}")
    else:
        run_dir = ckpt_path / f"run_{resume_uuid}"
        if not run_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found for UUID {resume_uuid} (expected {run_dir})")

        # For resume: pass dummy init tensors — they are ignored when run_uuid is set.
        _dummy = torch.zeros(0, dimensions, device=device, dtype=dtype)
        optimizer = ZoMBIHop(
            objective=objective_wrapper,
            X_init_actual=_dummy,
            X_init_expected=_dummy,
            Y_init=torch.zeros(0, 1, device=device, dtype=dtype),
            device=str(device),
            dtype=dtype,
            bounds=bounds,
            run_uuid=resume_uuid,
            checkpoint_dir=str(ckpt_path),
            num_iterations_saved=50,
            verbose=True,
            needle_plot_points_ref=needle_plot_points,
            **hw_hparams,
        )
        run_dir_ref[0] = run_dir
        optimizer_ref[0] = optimizer
        set_composition_log_dir(run_dir_ref[0])  # append to existing log across resume
        _flush_initial_state()  # write historical data to GUI immediately
        print(
            f"✅ Resumed from activation={optimizer.current_activation}, "
            f"zoom={optimizer.current_zoom}, iteration={optimizer.current_iteration}\n"
        )

    # Live-mirror discovered needles into sql/needles.db (optimizing dims → full
    # 10-col composition, sorted best-first). Rewritten on every needle add; the
    # immediate sync here reflects any needles already loaded on resume.
    try:
        optimizer.data_handler._enable_needles_db(OPTIMIZING_DIMS)
    except Exception as e:
        print(f"[ZoMBI] needles.db live-sync setup failed: {e}")

    print("=" * 80)
    print("STARTING OPTIMIZATION")
    if MINIMIZE_OBJECTIVE:
        print("Mode: minimization of measured objective (set MINIMIZE_OBJECTIVE = False to maximize).")
    print("=" * 80 + "\n")

    # never_terminate: the campaign must not stop on its own. Any internal stop
    # pathway (over-penalisation, activation failure, noise-floor exhaustion) is
    # converted into "shrink all penalty volumes by 70%, reset bounds, continue".
    # Only the user Stop button ends the run.
    optimizer.run(max_activations=float("inf"), time_limit_hours=None,
                  never_terminate=True)


if __name__ == "__main__":
    import sys
    resume_uuid = None
    if len(sys.argv) >= 2:
        a1 = sys.argv[1].strip().lower()
        if a1 in ("-h", "--help", "help"):
            print("Usage: python -m scripts.run_zombi_main [UUID]")
            print("  UUID     Resume this run (e.g. 6877). Omit for a new run.")
            sys.exit(0)
        resume_uuid = sys.argv[1]
    run_zombi_main(resume_uuid=resume_uuid)
