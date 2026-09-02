"""
interactive_test_zombi.py
=========================
Interactive ZoMBI-Hop simulator.

By default the objective is a Random-Forest surrogate built from
``campaign1a.csv``.  Passing ``--ackley {centroid|edge|vertex|multimodal}``
swaps the RF for one of the analytic negated-Ackley test functions defined in
``synthetic_data/ackley.py`` (the same objectives used by
``scripts/run_zombi_test.py``), which is handy for visualising a ZoMBI-Hop run
against a landscape with known optima.

Workflow
--------
1. Build the objective:
     * default — load ``data/campaign1a.csv`` and train a 500-tree
       Random-Forest on (FAPbI3, MAPbI3, MAPbBr3) → Objective; or
     * ``--ackley VARIANT`` — use the analytic Ackley surrogate (no CSV / RF
       training). Its peaks are maxima, so the run is forced to **maximize**
       and the analytic optima are used as the reference extrema (no clicking).
2. Choose **minimize** or **maximize** at the prompt (ZoMBI always runs as a
   maximiser internally; minimizing uses negated observations).  Skipped under
   ``--ackley`` (always maximizes the Ackley peaks).
3. Open an interactive ternary plot.  Left-click near a local minimum or
   maximum (per your choice); L-BFGS-B refinement (gradient-based on the
   surrogate) snaps to a nearby extremum.  Press Enter / Q when done selecting.
   (Skipped under ``--background``, which has no window to click on, and under
   ``--ackley``, whose reference optima are known analytically.)
4. Run ZoMBI-Hop (with LineBO), exactly as in ``scripts/run_zombi_main.py``,
   but evaluating the surrogate (RF or Ackley) instead of a physical instrument.
   Input noise (ILR std=NOISE_LEVEL_ILR) and multiplicative output noise
   (OUTPUT_NOISE_FRAC × |y|) are added at every sample, matched to data/2nd_real_run.db.
5. After every objective call, save a two-panel ternary figure to
   ``interactive_testing/plots/`` and display it (non-blocking; PNGs are still
   saved but not displayed under ``--background``):
     Left  – reference surrogate landscape + blue ★ for confirmed extrema (min or max).
     Right – ZoMBI-Hop exploration:
               • all sampled points (older = more transparent),
               • red ★ + faded-purple ellipse per discovered needle,
               • grey ✕ + faded-grey region per capped-activation exclusion zone,
               • dashed-red boundary for the current zoom box (grey dotted when the
                 search is back on the full box) — exact box ∩ simplex polygon,
               • dashed-blue circle of radius ``needle_move_tol`` around the
                 incumbent (the best unpenalized point in the active box, whose
                 value is the local median of its replicates): the incumbent
                 staying inside it between lines is half the needle test,
               • orange solid line for LineBO's main suggested line,
               • dotted cornflower-blue line for LineBO's cache line,
               • thin dim-grey lines for every candidate line sampled this step
                 (only with ``--show-sampling``),
               • blue ★ for confirmed reference extrema.

Usage
-----
  cd <repo_root>
  python interactive_testing/interactive_test_zombi.py

Flags
-----
  --ackley {centroid,edge,vertex,multimodal}
      Use an analytic negated-Ackley objective (from ``synthetic_data/ackley.py``)
      instead of the campaign1a RF surrogate. The chosen variant's peak(s) are
      the maxima, so the run is forced to maximize and the analytic optima are
      drawn as reference extrema (the interactive picker / CSV load are skipped).
        python interactive_testing/interactive_test_zombi.py --ackley centroid

  --ensemble
      Use the layered ``synthetic_data/ensemble.py`` objective on the 3-simplex
      (true optima + weak distractors + ridges + roughness + plateaus + edge bias)
      instead of the RF surrogate. A NEW RANDOM landscape is drawn every run. Like
      ``--ackley`` the optima are known analytically, so the run maximizes and
      neither the min/max prompt nor the click-picker appears. The landscape that
      was drawn is written to ``interactive_testing/plots/ensemble_config.json``,
      so a run worth repeating can be rebuilt with ``Ensemble(**config)``.
        python interactive_testing/interactive_test_zombi.py --ensemble

  --hparams PATH
      Load ZoMBI hyperparameters from a previous hparam-opt run and use them to
      override the built-in defaults. PATH may be a ``trial_*`` directory (or a
      ``trial.json`` file) to use that specific trial's hyperparameters, or a
      ``mobo_*`` directory (or ``mobo_progress.json`` file) to use the latest
      trial's hyperparameters.
        python interactive_testing/interactive_test_zombi.py --hparams path/to/mobo_00_00_00_00/trial_42

  --show-sampling
      Overlay a thin, semi-transparent line for every candidate line the
      acquisition function was integrated over at each step.
        python interactive_testing/interactive_test_zombi.py --show-sampling

  --background
      Run headless on the Agg backend: no plot window ever pops up or steals
      focus, and per-iteration PNGs are still saved to ``interactive_testing/
      plots/``. Skips the interactive extrema picker (step 3).
        python interactive_testing/interactive_test_zombi.py --background
"""

from __future__ import annotations

import os
import gc
import time
import sys
import json
import glob
import argparse
import queue
import threading
import warnings

# Ensure project root is on sys.path so ``src`` is importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# GP is trained in ILR space (ℝ^{d-1}) — BoTorch's unit-cube check is a false positive.
from botorch.exceptions import InputDataWarning
warnings.filterwarnings("ignore", category=InputDataWarning)

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize as sp_minimize
from scipy.spatial import ConvexHull

import matplotlib

# --background runs fully headless on the non-interactive Agg backend, so no plot
# window ever pops up or steals focus. Per-iteration PNGs are still written to
# interactive_testing/plots/. Detected from argv here because the backend must be
# chosen before pyplot is imported (argparse runs much later, in __main__).
_BACKGROUND = "--background" in sys.argv
matplotlib.use("Agg" if _BACKGROUND else "TkAgg")  # must be called before pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

from src import ZoMBIHop, LineBO
from src.core.linebo import line_simplex_segment, zero_sum_dirs
from src.utils.simplex import Ellipsoid, composition_to_ilr, ilr_to_composition, proj_simplex
from synthetic_data.ackley import Ackley
from synthetic_data.ensemble import Ensemble, random_ensemble_config
from optimize.composition_prediction import physics_simulate_line

# ── Configuration ─────────────────────────────────────────────────────────────
COMPOSITION_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
OBJECTIVE_COL = "Objective"
CORNER_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")

RF_N_ESTIMATORS = 500
OUTPUT_NOISE_FRAC = 0.045  # output noise as a fraction of the true y (measured ≈ within 4.5%)
NOISE_LEVEL_ILR = 0.384     # input noise std in ILR space (≈ 0.128 ambient per-component × 3; see default_hparams.DEFAULT_INPUT_NOISE)
NUM_EXPERIMENTS = 24     # points sampled per suggested line (mirrors run_zombi_main.py)
NUM_LINES = 10           # LineBO candidate lines per iteration
TERNARY_GRID_N = 120     # ternary grid resolution for reference heatmap
N_INIT_LINES = 2         # random lines to build the initial GP dataset
ENSEMBLE_OPTIMA_MARGIN = 0.2  # --ensemble: gap between the true optima and the capped background

SAVE_PLOTS = True        # save per-iteration PNG to interactive_testing/plots/

# Guard: this script must be run directly, NOT via `conda run`.
# `conda run` intercepts Tk event-loop signals and kills the window.
# Correct usage:
#   conda activate zombi-hop
#   python interactive_testing/interactive_test_zombi.py
import sys as _sys
if not _sys.stdin.isatty():
    print(
        "\n  WARNING: stdin is not a TTY — this script may be running under\n"
        "  `conda run`, which intercepts Tk signals and kills the window.\n\n"
        "  Run the script like this instead:\n"
        "    conda activate zombi-hop\n"
        "    python interactive_testing/interactive_test_zombi.py\n",
        flush=True,
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# ── Boundary penalty ─────────────────────────────────────────────────────────
# Always active. Strength = repulsion_lambda (auto-scaled to acquisition magnitude);
# ε = input_noise_ilr (existing ZoMBI param, default 0.03).
# No additional hyperparameters required.
# ─────────────────────────────────────────────────────────────────────────────

# ZoMBI hyperparameters (mirror run_zombi_main.py)
#
# This script targets the basin flood-fill loop: an activation is a flat sequence
# of measured lines, zoom depth is decided by the basin's volume ratio rather than
# by a zoom/iteration budget, and a needle is declared when the incumbent stops
# moving.  The old budget knobs (max_zooms / max_iterations / top_m_points /
# n_consecutive_converged / output_noise_threshold_mult / needle_shrink_factor /
# needle_stop_noise_multiplier) are RETIRED — ZoMBIHop still accepts them so old
# configs construct, but nothing reads them, so they are not set here.
ZOMBI_PARAMS: dict = dict(
    n_restarts=100,
    raw=1323,
    input_noise_threshold_mult=3.0,
    max_gp_points=3000,
    acquisition_type="ucb",
    ucb_beta=0.0,
    nat_grad_step=0.024757059158665974,
    nat_grad_max_steps=67,
    max_penalty_radius=1.0,
    # Point paring
    paring_spatial_halfnoise=0.5,
    paring_y_noise_multiplier=1.0,
    input_noise_ilr=NOISE_LEVEL_ILR,
    # Basin flood fill → zoom
    flood_k=6,
    flood_ci_z=2.0,
    zoom_volume_fraction=0.5,
    max_lines_per_activation=30,
    # Needle stability (the incumbent must stay within needle_move_tol of where it
    # was on the previous line — this is the circle drawn on the exploration panel)
    needle_move_tol=0.10,
    needle_ci_tol=0.15,
    verbose=True,
)

_SQRT3_2 = np.sqrt(3) / 2

# ── IMPORTANT: run this script directly, NOT via `conda run` ─────────────────
# `conda run` intercepts signals from the Tk event loop and will kill the
# window immediately.  Activate the environment first, then call Python:
#
#   conda activate zombi-hop
#   python interactive_testing/interactive_test_zombi.py

# ── Ternary utilities ─────────────────────────────────────────────────────────

def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """
    (N, 3) simplex compositions → (N, 2) Cartesian ternary coordinates.

    Corner mapping:
        comp[:, 0]  (FAPbI3)  →  origin (0, 0)
        comp[:, 1]  (MAPbI3)  →  (1, 0)
        comp[:, 2]  (MAPbBr3) →  (0.5, √3/2)
    """
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def xy_to_comp(x: float, y: float) -> np.ndarray:
    """Ternary Cartesian → (3,) composition (may contain small negatives near edges)."""
    c2 = y / _SQRT3_2
    c1 = x - 0.5 * c2
    c0 = 1.0 - c1 - c2
    return np.array([c0, c1, c2])


def draw_ternary_frame(ax, pad: float = 0.04) -> None:
    """Draw triangle outline, equal aspect, and corner labels."""
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, CORNER_LABELS[0], ha="right", va="top", fontsize=9)
    ax.text(1 + pad, -pad, CORNER_LABELS[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + pad, CORNER_LABELS[2], ha="center", va="bottom", fontsize=9)


def ternary_grid(n: int = 120) -> np.ndarray:
    """Return (N, 3) uniform grid on the probability simplex."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load composition inputs and Objective output from the CSV."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=COMPOSITION_COLS + [OBJECTIVE_COL])
    X = df[COMPOSITION_COLS].values.astype(float)
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)   # normalise rows to sum 1
    y = df[OBJECTIVE_COL].values.astype(float)
    return X, y


def train_rf(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    return rf


# ── Step 2: interactive extrema selection ────────────────────────────────────


def prompt_minimize_or_maximize() -> bool:
    """
    Ask whether to maximize or minimize the RF objective.

    ZoMBIHop always maximizes observations internally; for minimization the
    script feeds negated RF values (same pattern as before).

    Returns
    -------
    True  → maximize RF (gradient refinement ascends RF; no negation for ZoMBI).
    False → minimize RF (negate for ZoMBI; refinement minimizes RF).
    """
    while True:
        raw = input(
            "\nOptimize RF surrogate: type 'max' to maximize or 'min' to minimize\n"
            "(default: min). Refinement uses L-BFGS-B (SciPy finite-diff gradients).\n"
            "> "
        ).strip().lower()
        if raw in ("", "min", "m", "minimize"):
            return False
        if raw in ("max", "x", "maximize"):
            return True
        print("  Please enter 'min' or 'max'.")


def _log_params_to_simplex(log_x: np.ndarray) -> np.ndarray:
    """Unconstrained log-parameters → normalized composition (positive, sums to 1)."""
    log_x = np.asarray(log_x, dtype=float)
    z = log_x - np.max(log_x)
    x = np.exp(z)
    s = x.sum()
    return x / (s if s > 0 else 1.0)


def _refine_extremum(
    rf: RandomForestRegressor,
    x0: np.ndarray,
    *,
    maximize: bool,
    max_l1_displacement: float = 0.10,
) -> tuple[np.ndarray, float]:
    """
    Locally refine a clicked simplex point toward a nearby RF minimum or maximum.

    Uses L-BFGS-B on unconstrained log-coordinates (softmax map to the simplex).
    SciPy supplies gradients via finite differences (ascent on RF when
    ``maximize=True`` is implemented by minimizing ``-RF``).  If the optimum
    moves farther than ``max_l1_displacement`` in L1 composition distance from
    the click, the raw clicked point is returned.

    Note: piecewise-constant RFs have noisy finite-diff gradients; results are
    still useful as a local polish near the click.
    """
    x0 = np.clip(x0, 1e-12, None)
    x0 = x0 / x0.sum()
    log_x0 = np.log(np.maximum(x0, 1e-300))

    def objective(log_x: np.ndarray) -> float:
        x = _log_params_to_simplex(log_x)
        val = float(rf.predict(x.reshape(1, -1))[0])
        return -val if maximize else val

    res = sp_minimize(
        objective,
        log_x0,
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-9},
    )
    x_opt = _log_params_to_simplex(res.x)

    if np.abs(x_opt - x0).sum() > max_l1_displacement:
        return x0, float(rf.predict(x0.reshape(1, -1))[0])

    return x_opt, float(rf.predict(x_opt.reshape(1, -1))[0])


class ExtremaPicker:
    """
    Interactive ternary heatmap for clicking reference minima or maxima.

    Left-click near an extremum; L-BFGS-B refinement snaps to a nearby RF
    optimum (min or max per ``maximize``).  Press Enter / Q / Escape to proceed.
    """

    def __init__(
        self,
        rf: RandomForestRegressor,
        grid_pts: np.ndarray,
        grid_vals: np.ndarray,
        *,
        maximize: bool,
    ) -> None:
        self.rf = rf
        self.grid_pts = grid_pts
        self.grid_vals = grid_vals
        self.maximize = maximize
        self.extrema: list[tuple[np.ndarray, float]] = []
        self._fig = None
        self._ax = None

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax or event.button != 1:
            return
        comp = xy_to_comp(event.xdata, event.ydata)
        if np.any(comp < -0.05):
            return
        comp = np.clip(comp, 0, None)
        comp = comp / comp.sum()
        x_ref, y_ref = _refine_extremum(self.rf, comp, maximize=self.maximize)
        tag = "maximum" if self.maximize else "minimum"
        print(f"  → refined {tag}: {np.round(x_ref, 4)},  y = {y_ref:.5f}")
        self.extrema.append((x_ref, y_ref))
        xy = comp_to_xy(x_ref.reshape(1, 3))
        self._ax.scatter(
            xy[0, 0], xy[0, 1],
            marker="*", s=340, c="blue", zorder=12,
            edgecolors="navy", linewidths=1.3,
        )
        self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key in ("enter", "q", "escape"):
            self._done = True

    def run(self) -> list[tuple[np.ndarray, float]]:
        self._done = False
        fig, ax = plt.subplots(figsize=(7.5, 6.8))
        self._fig, self._ax = fig, ax
        draw_ternary_frame(ax)
        goal = "maxima" if self.maximize else "minima"
        goal_sg = "maximum" if self.maximize else "minimum"
        ax.set_title(
            f"RF landscape  —  click near a local {goal_sg}, then Enter / Q",
            fontsize=10,
        )
        gxy = comp_to_xy(self.grid_pts)
        sc = ax.scatter(
            gxy[:, 0], gxy[:, 1],
            c=self.grid_vals, cmap="viridis",
            s=8, alpha=0.80, zorder=2, rasterized=True,
        )
        cbar_lbl = (
            "RF Objective (maximize — higher better)"
            if self.maximize
            else "RF Objective (minimize — lower better)"
        )
        fig.colorbar(sc, ax=ax, label=cbar_lbl, fraction=0.046, pad=0.04)
        ax.text(
            0.5, -0.07,
            f"Click to mark reference {goal}.  Enter / Q to finish.",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic",
        )
        cid1 = fig.canvas.mpl_connect("button_press_event", self._on_click)
        cid2 = fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.tight_layout()
        fig.canvas.draw()
        plt.show(block=False)

        while not self._done:
            try:
                fig.canvas.flush_events()
            except Exception:
                break
            time.sleep(0.05)

        fig.canvas.mpl_disconnect(cid1)
        fig.canvas.mpl_disconnect(cid2)
        plt.close(fig)
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        time.sleep(0.1)
        return self.extrema


# ── RF objective simulation ───────────────────────────────────────────────────

def make_sim_objective(
    rf: RandomForestRegressor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    maximize: bool,
):
    """
    Return ``sim_objective(endpoints) → (x_actual, y)`` where:
      * ``endpoints`` is a ``(k, 2, d)`` tensor of ranked line endpoints,
      * the first line (index 0) is sampled at NUM_EXPERIMENTS evenly-spaced points,
      * Logistic-normal input noise (std=NOISE_LEVEL_ILR in ILR space) preserves the
        compositional geometry; multiplicative output noise (std=OUTPUT_NOISE_FRAC × |y|)
        is added to outputs,
      * returns ``(x_actual: (N, d), y: (N,))`` tensors on ``device``.

    ZoMBIHop maximizes ``y``. When ``maximize`` is False, RF outputs are negated so
    maximizing ``y`` corresponds to minimizing the RF surrogate.
    """

    def sim_objective(endpoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = endpoints[0, 0].to(dtype=torch.float64)
        right = endpoints[0, 1].to(dtype=torch.float64)
        # Physics-based "actual" compositions: push the requested start→end line
        # through the deterministic hardware print model (ramp lag/overshoot +
        # junction-volume diffusion mixing) instead of adding random ILR-space
        # input noise.
        pts_t = physics_simulate_line(left, right, num_points=NUM_EXPERIMENTS,
                                      device=left.device, dtype=torch.float64)  # (N, 3)
        pts_np = pts_t.detach().cpu().numpy()                  # numpy only at sklearn boundary
        raw = torch.tensor(rf.predict(pts_np).ravel(), dtype=dtype, device=device)
        y = raw if maximize else -raw
        y = y + torch.randn_like(y) * (OUTPUT_NOISE_FRAC * y.abs())
        return pts_t.to(dtype=dtype, device=device), y

    return sim_objective


# ── LineBO wrapper ────────────────────────────────────────────────────────────

def make_linebo_wrapper(
    sim_obj,
    dim: int,
    num_lines: int,
    device: torch.device,
    dtype: torch.dtype,
    plot_state: dict,
):
    """
    Build a ZoMBIHop-compatible objective wrapper::

        wrapper(x_tell, bounds, acq_fn) → (x_requested, x_actual, y)

    Internally uses ``LineBO.ranked_line_endpoints`` (not ``LineBO.sampler``)
    so we can capture the top-2 ranked lines before evaluation and store them
    in ``plot_state`` for the per-iteration plots.
    """
    linebo = LineBO(
        sim_obj,
        dim,
        num_points_per_line=100,
        num_lines=num_lines,
        device=str(device),
    )

    def wrapper(x_tell, bounds: torch.Tensor, acquisition_function):
        print(f"  [LineBO] ranking {num_lines} candidate lines …", flush=True)
        x_left_ranked, x_right_ranked = linebo.ranked_line_endpoints(
            x_tell, bounds, acquisition_function
        )
        n_valid = x_left_ranked.shape[0]
        print(f"  [LineBO] {n_valid} valid lines ranked. evaluating RF on best line …", flush=True)

        # ── capture top-2 lines for plotting ──────────────────────────────────
        plot_state["line_0"] = (
            (x_left_ranked[0].cpu().numpy(), x_right_ranked[0].cpu().numpy())
            if n_valid > 0 else None
        )
        plot_state["line_1"] = (
            (x_left_ranked[1].cpu().numpy(), x_right_ranked[1].cpu().numpy())
            if n_valid > 1 else None
        )

        # ── capture every candidate line the acquisition was integrated over ──
        # (all ranked lines for this step, used by the --show-sampling overlay).
        xl_np = x_left_ranked.cpu().numpy()
        xr_np = x_right_ranked.cpu().numpy()
        plot_state["sampling_lines"] = [
            (xl_np[i], xr_np[i]) for i in range(n_valid)
        ]

        # ── call simulated objective ───────────────────────────────────────────
        endpoints_ranked = torch.stack([x_left_ranked, x_right_ranked], dim=1)
        x_actual, y = sim_obj(endpoints_ranked)
        x_actual = x_actual.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype).ravel()
        print(f"  [LineBO] done — {x_actual.shape[0]} pts, y=[{y.min():.4f}, {y.max():.4f}]", flush=True)

        # ── compute x_requested (principal direction of x_actual) ─────────────
        # The PCA-reconstructed line lives in ℝ^d and can have negative components
        # or not sum to 1 — project onto the simplex before returning.
        if x_actual.shape[0] > 1:
            xc = x_actual - x_actual.mean(dim=0, keepdim=True)
            _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
            direction = Vt[0]
            projs = xc @ direction
            t_vals = torch.linspace(
                projs.min().item(), projs.max().item(),
                x_actual.shape[0], device=device, dtype=dtype,
            )
            x_requested = (
                x_actual.mean(dim=0).unsqueeze(0)
                + t_vals.unsqueeze(1) * direction.unsqueeze(0)
            )
            x_requested = proj_simplex(x_requested)   # guarantee simplex membership
        else:
            x_requested = x_actual.clone()

        return x_requested, x_actual, y

    return wrapper


def _gp_landscape_vals(gp_handler, grid_pts, maximize: bool):
    """GP posterior mean over the ternary grid (display orientation), or None.

    Captures what ZoMBI-Hop's GP currently believes the objective landscape
    looks like, for use as the exploration-panel background.  Un-negated for
    minimize runs to match the ``pared_Y`` display convention.
    """
    if (gp_handler is None or getattr(gp_handler, "gp", None) is None
            or grid_pts is None):
        return None
    try:
        gt = torch.as_tensor(grid_pts, dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            mean, _ = gp_handler.predict(gt)
        m = mean.detach().cpu().numpy().ravel()
        return m if maximize else -m
    except Exception:
        return None


def make_plotting_wrapper(
    inner_wrapper,
    dh_ref: list,
    plot_state: dict,
    grid_pts: np.ndarray,
    grid_vals: np.ndarray,
    true_minima: list,
    save_dir: str | None,
    plot_queue: "queue.Queue",
    *,
    maximize: bool,
    show_sampling: bool = False,
    gp_ref: list | None = None,
    opt_ref: list | None = None,
    ref_name: str = "RF",
):
    """
    Wrap ``inner_wrapper`` so that after every objective call a snapshot of
    plot data is pushed to ``plot_queue``.

    Uses the DataHandler's *pared* dataset (``X_pared`` / ``Y_pared``) for
    the exploration panel so duplicate / noisy points are collapsed and the
    display matches what the GP actually trains on.

    The main thread drains ``plot_queue`` and calls ``_plot_iteration``.
    All Tkinter / matplotlib work therefore stays on the main thread,
    eliminating the "main thread is not in main loop" RuntimeErrors that
    occur when figures are garbage-collected from a worker thread.
    """

    def wrapped(x_tell, bounds: torch.Tensor, acq_fn):
        x_requested, x_actual, y = inner_wrapper(x_tell, bounds, acq_fn)

        plot_state["iter"] = plot_state.get("iter", 0) + 1
        print(f"  [plot] queuing iteration-{plot_state['iter']} ternary …", flush=True)

        # Harvest DataHandler state and snapshot tensors for thread safety.
        dh = dh_ref[0]
        needles       = getattr(dh, "needles", None)
        needle_M_list = getattr(dh, "needle_M_list", None)
        needle_B      = getattr(dh, "needle_B", None)
        # The ACTIVE search region of the flood-fill loop. ``bounds`` and
        # ``current_zoom_bounds`` are kept in sync by every zoom-in / zoom-out /
        # needle-declaration path in ZoMBIHop, so either reads the live box;
        # prefer current_zoom_bounds, which is also what checkpoints store.
        curr_bounds   = getattr(dh, "current_zoom_bounds", None)
        if curr_bounds is None:
            curr_bounds = getattr(dh, "bounds", None)

        # Zoom state: depth, and whether the active box is still the full one.
        opt = opt_ref[0] if opt_ref else None
        full_bounds = getattr(opt, "full_bounds", None)
        is_full_box = bool(
            curr_bounds is not None and full_bounds is not None
            and torch.allclose(curr_bounds, full_bounds)
        )
        zoom_depth = getattr(dh, "current_zoom", None)

        # Incumbent + needle-stability tolerance: the best unpenalized point in
        # the active box (its Y is the local median of the replicates around it),
        # and the radius it must stay inside for a needle to be declared.
        best_X = None
        move_tol = float(getattr(opt, "needle_move_tol", 0.10)) if opt else 0.10
        try:
            if curr_bounds is not None and hasattr(dh, "get_best_in_bounds"):
                bx, _, _ = dh.get_best_in_bounds(curr_bounds)
            else:
                bx, _, _ = dh.get_best_unpenalized()
            if bx is not None:
                best_X = bx.detach().cpu().numpy().ravel()
        except Exception:
            best_X = None

        # Capped-activation exclusion zones (regions later activations are repelled
        # from without a needle ever being declared there).
        excl = getattr(dh, "exclusions", None)
        if excl is not None and excl.numel() > 0:
            excl_np = excl.detach().cpu().numpy()
            excl_radii = getattr(dh, "exclusion_radii", None)
            excl_radii_np = (excl_radii.detach().cpu().numpy().ravel()
                             if excl_radii is not None and excl_radii.numel() else
                             np.zeros(len(excl_np)))
            excl_M = list(getattr(dh, "exclusion_M_list", []) or [])
        else:
            excl_np = None
            excl_radii_np = None
            excl_M = []

        # Pared dataset — what the GP sees (noise-deduplicated).
        xp = getattr(dh, "X_pared", None)
        yp = getattr(dh, "Y_pared", None)
        if xp is not None and xp.shape[0] > 0:
            pared_X = xp.detach().cpu().numpy()
            pared_Y = yp.detach().cpu().numpy().ravel()
            if not maximize:
                pared_Y = -pared_Y   # un-negate so display shows true RF values
        else:
            pared_X = None
            pared_Y = None

        def _clone(t):
            return t.clone() if isinstance(t, torch.Tensor) else t

        payload = dict(
            grid_pts=grid_pts,
            grid_vals=grid_vals,
            true_minima=list(true_minima),
            maximize=maximize,
            pared_X=pared_X,
            pared_Y=pared_Y,
            needles=_clone(needles) if (needles is not None and isinstance(needles, torch.Tensor) and needles.numel() > 0) else None,
            needle_M_list=[_clone(m) for m in (needle_M_list or [])],
            needle_B=_clone(needle_B),
            trust_ellipsoid=_clone(curr_bounds),
            bounds_is_full=is_full_box,
            zoom_depth=zoom_depth,
            best_X=best_X,
            move_tol=move_tol,
            exclusions=excl_np,
            exclusion_radii=excl_radii_np,
            exclusion_M_list=[_clone(m) for m in excl_M],
            line_0=plot_state.get("line_0"),
            line_1=plot_state.get("line_1"),
            sampling_lines=plot_state.get("sampling_lines") if show_sampling else None,
            iteration_num=plot_state["iter"],
            ref_name=ref_name,
            save_dir=save_dir,
            gp_grid_vals=_gp_landscape_vals(
                gp_ref[0] if gp_ref else None, grid_pts, maximize),
        )
        plot_queue.put(payload)
        return x_requested, x_actual, y

    return wrapped


# ── Plotting helpers ──────────────────────────────────────────────────────────

# comp_to_xy restricted to the zero-sum plane is a similarity: an orthonormal
# tangent basis maps to orthogonal ternary vectors of length 1/√2 (check with
# u = (1,-1,0)/√2 → (-1/√2, 0) and v = (1,1,-2)/√6 → (0, -1/√2)).  So a ball of
# radius r in composition L2 is exactly a CIRCLE of radius r/√2 on the ternary —
# no sampling or projection needed to draw one.
_COMP_TO_XY_SCALE = 1.0 / np.sqrt(2.0)


def clip_simplex_box(bounds: torch.Tensor) -> np.ndarray | None:
    """Exact polygon of ``{x ∈ simplex : lo ≤ x ≤ hi}`` in composition coords.

    The current zoom region is an axis-aligned box (``_basin_box``), and its
    intersection with the 2-simplex is a convex polygon with up to 9 vertices — a
    corner-clipped triangle, not a rectangle.  Sutherland–Hodgman clips the
    triangle against the 2·d half-planes ``x_i ≥ lo_i`` and ``x_i ≤ hi_i``;
    every vertex stays on the simplex plane, so the clip is planar and exact.

    Returns ``(k, 3)`` polygon vertices in order, or None if the box misses the
    simplex entirely (or ``bounds`` is not a 3-simplex box).
    """
    b = bounds.detach().cpu().numpy().astype(float)
    if b.shape != (2, 3):
        return None
    lo, hi = b[0], b[1]
    poly = [np.eye(3)[i] for i in range(3)]

    def _clip(poly_in: list, axis: int, bound: float, keep_upper: bool) -> list:
        """Keep the side of ``x[axis] == bound`` requested; interpolate crossings."""
        if not poly_in:
            return []
        # signed distance > 0 == inside
        sd = [(p[axis] - bound) if keep_upper else (bound - p[axis]) for p in poly_in]
        out = []
        n = len(poly_in)
        for i in range(n):
            p, s = poly_in[i], sd[i]
            q, t = poly_in[(i + 1) % n], sd[(i + 1) % n]
            if s >= 0:
                out.append(p)
            if (s > 0) != (t > 0) and abs(s - t) > 1e-15:
                out.append(p + (s / (s - t)) * (q - p))
        return out

    for axis in range(3):
        poly = _clip(poly, axis, lo[axis], keep_upper=True)
        poly = _clip(poly, axis, hi[axis], keep_upper=False)
        if not poly:
            return None
    # Vertices sitting exactly on a clip plane are emitted twice (once as "inside",
    # once as the crossing); drop the consecutive duplicates.
    out = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - out[-1]) > 1e-12:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= 1e-12:
        out.pop()
    return np.array(out, dtype=float)


def _draw_bounds_region(ax, bounds, *, is_full: bool = False,
                        n_sample: int = 5000):
    """Draw the active search region (the current zoom box) on the ternary.

    The basin flood-fill loop keeps the active region as an axis-aligned box
    ``(2, d)`` — ``DataHandler.bounds`` / ``current_zoom_bounds`` — which is drawn
    exactly via :func:`clip_simplex_box`.  ``is_full`` marks the un-zoomed state
    (the box equals the run's full search bounds): it is drawn grey and dotted so
    "zoomed into a basin" is visually distinct from "searching the whole simplex".
    The legacy ``Ellipsoid`` branch is kept only for old checkpoints.
    """
    colour = "grey" if is_full else "red"
    style = ":" if is_full else "--"
    label = "Search box (full)" if is_full else "Zoom box"

    if isinstance(bounds, torch.Tensor) and bounds.shape[0] == 2:
        poly = clip_simplex_box(bounds)
        if poly is None or poly.shape[0] < 3:
            return
        verts = comp_to_xy(poly)
        verts_c = np.vstack([verts, verts[0]])
        if not is_full:
            ax.fill(verts[:, 0], verts[:, 1], color=colour, alpha=0.06, zorder=4)
        (h,) = ax.plot(
            verts_c[:, 0], verts_c[:, 1],
            style, color=colour, lw=2.0, alpha=0.75, zorder=5, label=label,
        )
        return h

    if not isinstance(bounds, Ellipsoid):
        return
    # Fallback for old Ellipsoid checkpoints (pre-AABB bounds).
    try:
        from src.utils.simplex import sample_ellipsoid as _se
        samp = _se(n_sample, bounds, scale=1.0)
    except Exception:
        return
    pts = samp.detach().cpu().numpy()
    if pts.shape[0] < 3:
        return
    xy = comp_to_xy(pts)
    try:
        hull = ConvexHull(xy)
        verts = xy[hull.vertices]
        verts_c = np.vstack([verts, verts[0]])
        ax.fill(verts[:, 0], verts[:, 1], color="red", alpha=0.06, zorder=4)
        ax.plot(
            verts_c[:, 0], verts_c[:, 1],
            "--", color="red", lw=2.0, alpha=0.75, zorder=5, label="Trust bounds",
        )
    except Exception:
        pass


def _draw_move_tol_ball(ax, center: np.ndarray, radius: float):
    """Draw the needle-stability tolerance around the current incumbent.

    ``center`` is the best unpenalized point in the active box — the point a
    needle would be declared at, and whose value is the MEDIAN of the replicate
    measurements around it (``DataHandler._relabel_pared_with_medians`` /
    ``_median_ci_halfwidth`` both work on that local median).  ``radius`` is
    ``needle_move_tol`` (default 0.10, i.e. ten composition points in L2): if the
    incumbent's next position stays inside this circle — and its median CI has
    stopped sharpening — the needle is declared here and the search zooms out.

    Composition-L2 balls are exact circles on the ternary (see
    ``_COMP_TO_XY_SCALE``), so this is drawn analytically rather than sampled.
    """
    c_xy = comp_to_xy(np.asarray(center, dtype=float).reshape(1, 3))[0]
    r_xy = float(radius) * _COMP_TO_XY_SCALE
    ang = np.linspace(0, 2 * np.pi, 181)
    ring = np.column_stack([c_xy[0] + r_xy * np.cos(ang),
                            c_xy[1] + r_xy * np.sin(ang)])
    ax.fill(ring[:, 0], ring[:, 1], color="deepskyblue", alpha=0.10, zorder=5)
    (h,) = ax.plot(
        ring[:, 0], ring[:, 1], "--", color="deepskyblue", lw=1.8, alpha=0.95,
        zorder=8, label=f"needle_move_tol ({radius:.2f}) around incumbent",
    )
    ax.scatter(
        c_xy[0], c_xy[1], marker="P", s=130, c="deepskyblue",
        zorder=10, edgecolors="black", linewidths=0.9,
    )
    return h


def _draw_exclusion_zone(ax, center: np.ndarray, M, B, radius: float) -> None:
    """Draw a capped-activation exclusion zone (grey ✕ + faded grey region).

    An activation that burns ``max_lines_per_activation`` without declaring a
    needle records the region it was stuck in as an exclusion zone
    (``_penalize_capped_zone``), which repels later activations exactly like a
    needle's penalty does.  Drawn so points being avoided have a visible reason.
    """
    xy = comp_to_xy(np.asarray(center, dtype=float).reshape(1, 3))
    ax.scatter(
        xy[0, 0], xy[0, 1], marker="X", s=170, c="dimgray",
        zorder=9, edgecolors="black", linewidths=0.9,
    )
    try:
        if M is not None:
            for ring, penalty in _needle_penalty_bands(np.asarray(center, float), M, B):
                ax.add_patch(MplPolygon(
                    comp_to_xy(ring), closed=True,
                    facecolor="dimgray", edgecolor="none",
                    alpha=0.45 * penalty, linewidth=0, zorder=6,
                ))
        elif radius and radius > 0:
            # Sphere fallback: exact circle on the ternary (see _COMP_TO_XY_SCALE).
            r_xy = float(radius) * _COMP_TO_XY_SCALE
            ang = np.linspace(0, 2 * np.pi, 181)
            ax.fill(xy[0, 0] + r_xy * np.cos(ang), xy[0, 1] + r_xy * np.sin(ang),
                    color="dimgray", alpha=0.15, zorder=6)
    except Exception as exc:
        print(f"  [exclusion warn] {exc}")


def _needle_penalty_bands(
    needle_x: np.ndarray,
    M: torch.Tensor,
    B: torch.Tensor | None,
    *,
    n_bands: int = 18,
    n_ang: int = 160,
) -> list[tuple[np.ndarray, float]]:
    """Non-overlapping annular rings coloured by the smooth repulsion penalty.

    The acquisition repulsion is ``violation**2 = clamp(1 - quad, 0)**2`` with
    ``quad = delta_z^T M delta_z``: strongest (1.0) at the needle centre, fading
    smoothly to 0 at the boundary (``quad = 1``).  Each ring between contours
    ``quad = s_out**2`` and ``quad = s_in**2`` carries penalty ``(1 - s_mid**2)**2``
    at its mid-radius.  Because the rings don't overlap, filling each with
    ``alpha = base * penalty`` yields a rendered opacity *exactly proportional* to
    the true penalty at every radius.

    Tangent-space mode (B is not None):  boundary {x* + B @ u : u^T M u = 1}
    ILR mode (B is None):                boundary {ilr⁻¹(ilr(x*) + u) : u^T M u = 1}
    """
    d = needle_x.shape[0]
    M_np = M.cpu().numpy()
    eigvals, eigvecs = np.linalg.eigh(M_np)
    eigvals = np.maximum(eigvals, 1e-12)
    angles = np.linspace(0, 2 * np.pi, n_ang)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    # Unit ellipse boundary: u = V diag(1/√λ) [cos,sin]^T  → u^T M u = 1
    u_unit = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ circle.T).T   # (n_ang, d-1)
    needle_x = np.asarray(needle_x, dtype=float).ravel()
    needle_ilr = (composition_to_ilr(
        torch.tensor(needle_x, dtype=torch.float64).unsqueeze(0)).squeeze(0).cpu().numpy()
        if B is None else None)

    def _contour(s: float) -> np.ndarray:
        u_ell = s * u_unit
        if B is not None:
            ell = needle_x.reshape(1, d) + (B.cpu().numpy() @ u_ell.T).T
        else:
            ell = ilr_to_composition(
                torch.tensor(needle_ilr + u_ell, dtype=torch.float64), d).cpu().numpy()
        ell = np.clip(ell, 0, 1)
        ssum = ell.sum(axis=1, keepdims=True)
        return ell / np.where(ssum < 1e-9, 1.0, ssum)

    scales = np.linspace(1.0, 0.0, n_bands + 1)         # edge (1) → centre (0)
    contours = [_contour(s) for s in scales]
    bands: list[tuple[np.ndarray, float]] = []
    for k in range(n_bands):
        s_mid = 0.5 * (scales[k] + scales[k + 1])
        penalty = float((1.0 - s_mid ** 2) ** 2)
        if scales[k + 1] <= 1e-9:
            ring = contours[k]                          # innermost: filled centre disk
        else:
            ring = np.vstack([contours[k], contours[k + 1][::-1]])  # annulus
        bands.append((ring, penalty))
    return bands


def _draw_needle_ellipsoid(
    ax,
    needle_x: np.ndarray,
    M: torch.Tensor | None,
    B: torch.Tensor | None,
) -> None:
    """
    Plot a red star for the needle and, if ellipsoid parameters are available, a
    red penalty gradient over the penalization ellipsoid: opaque red at the needle
    centre (where the repulsion is strongest) fading to fully transparent at the
    boundary (where the penalty vanishes).  Opacity is proportional to the penalty.
    """
    xy = comp_to_xy(needle_x.reshape(1, 3))
    ax.scatter(
        xy[0, 0], xy[0, 1],
        marker="*", s=280, c="red",
        zorder=9, edgecolors="darkred", linewidths=1.0,
    )
    if M is None:
        return
    try:
        for ring, penalty in _needle_penalty_bands(needle_x, M, B):
            ax.add_patch(MplPolygon(
                comp_to_xy(ring), closed=True,
                facecolor="red", edgecolor="none",
                alpha=0.5 * penalty, linewidth=0, zorder=6,
            ))
    except Exception as exc:
        print(f"  [ellipse warn] {exc}")


def _plot_iteration(
    grid_pts: np.ndarray,
    grid_vals: np.ndarray,
    true_minima: list,
    maximize: bool,
    pared_X: np.ndarray | None,
    pared_Y: np.ndarray | None,
    needles: torch.Tensor | None,
    needle_M_list: list | None,
    needle_B: torch.Tensor | None,
    trust_ellipsoid: Ellipsoid | None,
    line_0: tuple | None,
    line_1: tuple | None,
    iteration_num: int,
    save_dir: str | None = None,
    sampling_lines: list | None = None,
    gp_grid_vals: np.ndarray | None = None,
    bounds_is_full: bool = False,
    zoom_depth: int | None = None,
    best_X: np.ndarray | None = None,
    move_tol: float = 0.10,
    exclusions: np.ndarray | None = None,
    exclusion_radii: np.ndarray | None = None,
    exclusion_M_list: list | None = None,
    ref_name: str = "RF",
) -> plt.Figure:
    """
    Generate the two-panel ternary figure for one iteration and optionally save it.

    Parameters
    ----------
    grid_pts, grid_vals : reference heatmap data.
    true_minima         : list of (composition, value) from the interactive picker.
    maximize            : if True, legend labels reference maxima and colour scale
                          semantics match maximization.
    pared_X, pared_Y    : noise-deduplicated dataset the GP trains on (newest last).
    needles             : (k, d) tensor or None.
    needle_M_list       : per-needle ellipsoid M matrices (list of (2,2) tensors or None).
    needle_B            : shared tangent basis (3, 2) tensor or None.
    trust_ellipsoid     : active search region — a ``(2, d)`` box in the current
                          (flood-fill) loop, or a legacy ``Ellipsoid``; None to skip.
    bounds_is_full      : True when that box is still the full search box (drawn
                          grey/dotted rather than red/dashed).
    zoom_depth          : current zoom depth, for the summary box.
    best_X, move_tol    : incumbent (best unpenalized point in the active box, whose
                          value is the local median of its replicates) and the
                          ``needle_move_tol`` radius it must stay inside for a needle
                          to be declared.  None skips the circle.
    exclusions, exclusion_radii, exclusion_M_list :
                          capped-activation exclusion zones (grey ✕ + faded region).
    line_0, line_1      : each is (left_np, right_np) or None.
    iteration_num       : counter shown in the title and filename.
    save_dir            : directory to write the PNG, or None.
    sampling_lines      : list of (left_np, right_np) for every candidate line the
                          acquisition was integrated over this step, or None to skip
                          the thin sampling overlay (``--show-sampling``).
    """
    fig, (ax_ref, ax_exp) = plt.subplots(1, 2, figsize=(16, 6.8))
    fig.suptitle(
        f"ZoMBI-Hop Interactive Test  —  iteration {iteration_num}",
        fontsize=13,
    )

    # ── Left panel: RF reference ───────────────────────────────────────────────
    draw_ternary_frame(ax_ref)
    ax_ref.set_title(f"Reference: {ref_name} landscape", fontsize=11)
    gxy = comp_to_xy(grid_pts)
    sc_ref = ax_ref.scatter(
        gxy[:, 0], gxy[:, 1],
        c=grid_vals, cmap="viridis",
        s=6, alpha=0.72, zorder=2, rasterized=True,
    )
    fig.colorbar(sc_ref, ax=ax_ref, label=f"{ref_name} objective",
                 fraction=0.046, pad=0.04)
    ref_lbl = "True maxima" if maximize else "True minima"
    if true_minima:
        mc = np.array([m[0] for m in true_minima])
        mxy = comp_to_xy(mc)
        ax_ref.scatter(
            mxy[:, 0], mxy[:, 1],
            marker="*", s=360, c="blue",
            zorder=11, edgecolors="navy", linewidths=1.3, label=ref_lbl,
        )
        ax_ref.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ── Right panel: ZoMBI-Hop exploration ────────────────────────────────────
    draw_ternary_frame(ax_exp)
    ax_exp.set_title("ZoMBI-Hop exploration", fontsize=11)
    legend_handles = []

    # Background = the GP's current belief about the objective landscape (its
    # posterior mean over the ternary grid).
    if gp_grid_vals is not None and len(gp_grid_vals) == len(grid_pts):
        sc_gp = ax_exp.scatter(
            gxy[:, 0], gxy[:, 1],
            c=gp_grid_vals, cmap="viridis",
            s=6, alpha=0.72, zorder=1, rasterized=True,
        )
        fig.colorbar(sc_gp, ax=ax_exp, label="GP posterior mean",
                     fraction=0.046, pad=0.04)

    # Pared (noise-deduplicated) dataset — matches what the GP trains on.
    # Older points are more transparent; colour encodes true RF objective value.
    if pared_X is not None and len(pared_X) > 0:
        n      = len(pared_Y)
        alphas = np.linspace(0.15, 0.92, n) if n > 1 else np.array([0.92])
        xy_pts = comp_to_xy(pared_X)
        y_lo, y_hi = pared_Y.min(), pared_Y.max()
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-9
        # Draw oldest first so newest points sit on top
        for i in range(n):
            ax_exp.scatter(
                xy_pts[i, 0], xy_pts[i, 1],
                c=[[pared_Y[i]]], cmap="viridis",
                vmin=y_lo, vmax=y_hi,
                s=22, alpha=float(alphas[i]), zorder=3,
                edgecolors="black", linewidths=0.9,
            )
        ax_exp.set_title(
            f"ZoMBI-Hop exploration  ({n} pared pts)", fontsize=11
        )

    # Active search region: the current zoom box (red dashed) or the full search
    # box (grey dotted) — see _draw_bounds_region.
    if trust_ellipsoid is not None:
        h_box = _draw_bounds_region(ax_exp, trust_ellipsoid, is_full=bounds_is_full)
        if h_box is not None:
            legend_handles.append(h_box)

    # Active needles: red ★ + purple ellipse
    if needles is not None and needles.shape[0] > 0:
        nx_np = needles.cpu().numpy()
        M_list = needle_M_list or [None] * len(nx_np)
        for i, nx in enumerate(nx_np):
            Mi = M_list[i] if i < len(M_list) else None
            _draw_needle_ellipsoid(ax_exp, nx, Mi, needle_B)

    # Capped-activation exclusion zones: grey ✕ + faded grey penalty region.
    if exclusions is not None and len(exclusions) > 0:
        eM = exclusion_M_list or [None] * len(exclusions)
        eR = (exclusion_radii if exclusion_radii is not None
              else np.zeros(len(exclusions)))
        for i, ex in enumerate(exclusions):
            _draw_exclusion_zone(
                ax_exp, ex,
                eM[i] if i < len(eM) else None,
                needle_B,
                float(eR[i]) if i < len(eR) else 0.0,
            )
        legend_handles.append(Line2D(
            [], [], marker="X", color="none", markerfacecolor="dimgray",
            markeredgecolor="black", markersize=9,
            label=f"Exclusion zone ({len(exclusions)})",
        ))

    # Needle-stability tolerance around the incumbent (the median-valued best
    # point): stay inside this circle between lines and a needle is declared.
    if best_X is not None:
        legend_handles.append(_draw_move_tol_ball(ax_exp, best_X, move_tol))

    # Sampling overlay: every candidate line the acquisition was integrated over.
    # Thin and semi-transparent so the chosen main/cache lines remain readable.
    if sampling_lines:
        for k, sl in enumerate(sampling_lines):
            sxy = comp_to_xy(np.array(sl))  # (2, 2)
            ax_exp.plot(
                sxy[:, 0], sxy[:, 1],
                "-", color="dimgray", lw=0.8, alpha=0.35,
                zorder=6,
                label="Sampled lines" if k == 0 else None,
            )
        # Register one legend handle for the whole sampling set.
        h_samp = Line2D(
            [], [], color="dimgray", lw=0.8, alpha=0.6,
            label=f"Sampled lines ({len(sampling_lines)})",
        )
        legend_handles.append(h_samp)

    # LineBO suggested lines
    if line_0 is not None:
        ll = comp_to_xy(np.array(line_0))  # (2, 2)
        (h0,) = ax_exp.plot(
            ll[:, 0], ll[:, 1],
            "-", color="orange", lw=2.5, alpha=0.90,
            zorder=7, label="LineBO (main)",
        )
        legend_handles.append(h0)
    if line_1 is not None:
        ll = comp_to_xy(np.array(line_1))
        (h1,) = ax_exp.plot(
            ll[:, 0], ll[:, 1],
            ":", color="cornflowerblue", lw=2.2, alpha=0.85,
            zorder=7, label="LineBO (cache)",
        )
        legend_handles.append(h1)

    # Reference extrema on the exploration panel too (blue stars)
    if true_minima:
        mc = np.array([m[0] for m in true_minima])
        mxy = comp_to_xy(mc)
        h_min = ax_exp.scatter(
            mxy[:, 0], mxy[:, 1],
            marker="*", s=360, c="blue",
            zorder=11, edgecolors="navy", linewidths=1.3,
            label=ref_lbl,
        )
        legend_handles.append(h_min)

    if legend_handles:
        ax_exp.legend(
            handles=legend_handles, loc="upper right",
            fontsize=8, framealpha=0.9,
        )

    # ── Top-left summary box ───────────────────────────────────────────────────
    summary_lines = []
    if trust_ellipsoid is not None:
        if isinstance(trust_ellipsoid, torch.Tensor) and trust_ellipsoid.shape[0] == 2:
            lo_cpu = trust_ellipsoid[0].detach().cpu().numpy()
            hi_cpu = trust_ellipsoid[1].detach().cpu().numpy()
            depth_str = "full box" if bounds_is_full else (
                f"zoom {zoom_depth}" if zoom_depth is not None else "zoomed")
            summary_lines.append(f"Box ({depth_str})")
            summary_lines.append("  lo: [" + " ".join(f"{v:.4f}" for v in lo_cpu) + "]")
            summary_lines.append("  hi: [" + " ".join(f"{v:.4f}" for v in hi_cpu) + "]")
            summary_lines.append("  w : [" + " ".join(f"{v:.4f}" for v in (hi_cpu - lo_cpu)) + "]")
        elif isinstance(trust_ellipsoid, Ellipsoid):
            c_cpu = trust_ellipsoid.c.detach().cpu().numpy()
            wM, _ = torch.linalg.eigh(trust_ellipsoid.M)
            wcpu = np.sort(wM.detach().cpu().numpy())
            summary_lines.append("Trust c: [" + " ".join(f"{v:.4f}" for v in c_cpu) + "]")
            summary_lines.append("Trust M eig: [" + " ".join(f"{v:.4f}" for v in wcpu) + "]")
    if best_X is not None:
        summary_lines.append(
            "Incumbent: [" + " ".join(f"{v:.4f}" for v in best_X) + "]"
            f"  ±{move_tol:.2f}")
    if line_0 is not None:
        a, b = np.round(line_0[0], 3), np.round(line_0[1], 3)
        summary_lines.append(f"Line0: {list(a)} → {list(b)}")
    if line_1 is not None:
        a, b = np.round(line_1[0], 3), np.round(line_1[1], 3)
        summary_lines.append(f"Line1: {list(a)} → {list(b)}")
    if summary_lines:
        ax_exp.text(
            0.01, 0.99, "\n".join(summary_lines),
            transform=ax_exp.transAxes,
            va="top", ha="left",
            fontsize=6.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.75),
            zorder=20,
        )

    fig.tight_layout()

    if SAVE_PLOTS and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, f"iter_{iteration_num:04d}.png")
        tmp_path = final_path + ".tmp"
        fig.savefig(tmp_path, dpi=120, bbox_inches="tight", format="png")
        os.replace(tmp_path, final_path)

    # Headless (Agg) mode: the PNG is already written above; skip the GUI
    # show/draw/flush so no window pops up or steals focus.
    if not matplotlib.get_backend().lower().startswith("agg"):
        plt.show(block=False)
        fig.canvas.draw()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
    return fig


# ── Initial data generation ───────────────────────────────────────────────────

def generate_init_data(
    rf: RandomForestRegressor,
    n_lines: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    maximize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample ``n_lines`` random simplex lines, evaluate the RF on each, and
    return tensors suitable as ``X_init_actual / X_init_expected / Y_init``
    for ZoMBIHop.

    ``Y_init`` matches ZoMBI's convention: raw RF (+ noise) when ``maximize``,
    negated RF (+ noise) when minimizing.
    """
    x_actual_list, x_exp_list, y_list = [], [], []
    for line_idx in range(n_lines):
        print(f"      init line {line_idx + 1}/{n_lines} …", flush=True)
        x0 = torch.full((3,), 1.0 / 3, device=device, dtype=dtype)
        direction = zero_sum_dirs(1, 3, device=device, dtype=dtype).squeeze(0)
        seg = line_simplex_segment(x0, direction)
        if seg is None:
            print(f"      init line {line_idx + 1}: no valid segment, skipping.")
            continue
        _t_min, _t_max, x_left, x_right = seg
        t = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS, dtype=torch.float64, device=device)
        # Clean straight segment = requested line; physics-simulated print = actual
        # (replaces the random ILR-space input noise).
        pts_clean = (
            x_left.to(torch.float64).unsqueeze(0)
            + t.unsqueeze(1) * (x_right - x_left).to(torch.float64).unsqueeze(0)
        )
        pts_t = physics_simulate_line(x_left, x_right, num_points=NUM_EXPERIMENTS,
                                      device=device, dtype=torch.float64)
        pts_np = pts_t.detach().cpu().numpy()
        print(f"      evaluating RF on {len(pts_np)} points …", flush=True)
        raw = torch.tensor(rf.predict(pts_np).ravel(), dtype=dtype, device=device)
        y_vals = raw + torch.randn_like(raw) * (OUTPUT_NOISE_FRAC * raw.abs())
        print(f"      done. y range: [{y_vals.min():.4f}, {y_vals.max():.4f}]", flush=True)
        y_zombi = y_vals if maximize else -y_vals
        pts_out = pts_t.to(dtype=dtype, device=device)
        pts_clean = pts_clean.to(dtype=dtype, device=device)
        x_actual_list.append(pts_out)
        x_exp_list.append(pts_clean)
        y_list.append(y_zombi)
    if not x_actual_list:
        raise RuntimeError("Could not generate any initial data lines.")
    return (
        torch.cat(x_actual_list, dim=0),
        torch.cat(x_exp_list, dim=0),
        torch.cat(y_list, dim=0).reshape(-1, 1),
    )


# ── Hyperparameter loading ────────────────────────────────────────────────────

def load_hparams(path: str) -> dict:
    """
    Load a set of ZoMBI hyperparameters from a previous hparam-opt run.

    ``path`` may be any of:
      * a ``trial_*`` directory containing a ``trial.json`` (or a ``trial.json``
        file directly) — that specific trial's hyperparameters are used; or
      * a ``mobo_*`` run directory containing ``mobo_progress.json`` (or a
        ``mobo_progress.json`` file directly) — the latest trial's
        hyperparameters are used (the trial with the highest ``trial`` index).

    Returns the ``hparams`` dict for the selected trial.
    """
    # ── Single-trial source: a trial_* directory or a trial.json file ─────────
    if os.path.isdir(path):
        trial_path = os.path.join(path, "trial.json")
    else:
        trial_path = path
    if os.path.basename(trial_path) == "trial.json" and os.path.exists(trial_path):
        with open(trial_path, "r") as f:
            trial = json.load(f)
        hparams = trial.get("hparams")
        if not hparams:
            raise ValueError(f"Trial file {trial_path} has no hparams.")
        print(f"    Loaded hyperparameters from {trial_path}")
        print(f"    Using trial {trial.get('trial')} (phase={trial.get('phase')}):")
        for k, v in hparams.items():
            print(f"      {k} = {v}")
        return hparams

    # ── Whole-run source: a mobo_* directory or mobo_progress.json file ───────
    if os.path.isdir(path):
        progress_path = os.path.join(path, "mobo_progress.json")
    else:
        progress_path = path
    if not os.path.exists(progress_path):
        # Tolerate being handed a parent dir: search for a mobo_progress.json.
        matches = glob.glob(os.path.join(path, "**", "mobo_progress.json"), recursive=True)
        if not matches:
            raise FileNotFoundError(
                f"Could not find a trial.json or mobo_progress.json at or under: {path}"
            )
        progress_path = matches[0]

    with open(progress_path, "r") as f:
        progress = json.load(f)

    trials = progress.get("trials", [])
    if not trials:
        raise ValueError(f"No trials found in {progress_path}")

    latest = max(trials, key=lambda t: t.get("trial", -1))
    hparams = latest.get("hparams")
    if not hparams:
        raise ValueError(
            f"Trial {latest.get('trial')} in {progress_path} has no hparams."
        )

    print(f"    Loaded hyperparameters from {progress_path}")
    print(f"    Using trial {latest.get('trial')} (phase={latest.get('phase')}, "
          f"pareto={latest.get('pareto')}):")
    for k, v in hparams.items():
        print(f"      {k} = {v}")
    return hparams


# ── Main ──────────────────────────────────────────────────────────────────────

def build_ensemble() -> tuple[Ensemble, dict]:
    """Draw a fresh random layered ``synthetic_data.ensemble`` objective.

    Returns ``(fn, config)``.  Every run gets a different landscape — random
    feature counts / widths / amplitudes / on-off toggles / optima layout — the
    same "Randomize" draw ``synthetic_data/plot_ensemble.py`` and
    ``optimize/evaluate.py --dataset ensemble`` use, from a seed taken from the OS
    entropy pool rather than a fixed sweep position.  The kwargs are returned (and
    written next to the plots by the caller) so a run that turns up something
    interesting can still be reconstructed exactly with ``Ensemble(**config)``.

    The dimension is pinned to 3: everything downstream here is ternary (the grid,
    both panels, the LineBO wrapper).  Use ``optimize/evaluate.py --dataset
    ensemble --dim D`` for higher-dimensional ensembles.
    """
    rng = np.random.default_rng()
    seed = int(rng.integers(0, 2 ** 31 - 1))
    index = int(rng.integers(0, 1_000_000))
    cfg = random_ensemble_config(3, index, seed=seed,
                                 optima_margin=ENSEMBLE_OPTIMA_MARGIN)
    return Ensemble(**cfg), cfg


def main(
    hparams_path: str | None = None,
    show_sampling: bool = False,
    background: bool = False,
    ackley: str | None = None,
    ensemble: bool = False,
) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "campaign1a.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(script_dir, "data", "campaign1a.csv")
    save_dir = os.path.join(script_dir, "plots")
    ckpt_dir = os.path.join(script_dir, "checkpoints")

    if ackley:
        surrogate_name = f"Ackley ({ackley})"
    elif ensemble:
        surrogate_name = "Ensemble (random landscape)"
    else:
        surrogate_name = "RF Surrogate on campaign1a.csv"
    print("=" * 70)
    print(f"ZoMBI-Hop Interactive Test — {surrogate_name}")
    print(f"Device : {DEVICE}")
    print(f"Noise  : input ILR std={NOISE_LEVEL_ILR}  |  output frac={OUTPUT_NOISE_FRAC} (× |y|)")
    print("=" * 70)

    # ── Step 1: build the objective surrogate ─────────────────────────────────
    # ``rf`` is any object exposing a scikit-learn-style ``predict((N, d)) → (N,)``
    # method: either a trained RandomForestRegressor (default) or an analytic
    # ``Ackley`` instance (``--ackley``). Everything downstream is agnostic to which.
    if ackley:
        print(f"\n[1] Using analytic Ackley objective ('{ackley}') — no CSV / RF training.")
        rf = Ackley(ackley)
        grid_pts = ternary_grid(TERNARY_GRID_N)
        grid_vals = rf.predict(grid_pts)
        # Ackley peaks are maxima; ZoMBI maximizes them directly.
        maximize = True
        print("    Mode forced: MAXIMIZE (Ackley peaks are maxima).")
    elif ensemble:
        print("\n[1] Drawing a random layered Ensemble objective — no CSV / RF training.")
        rf, ens_cfg = build_ensemble()
        print(f"    {rf!r}")
        # Record the exact landscape next to the plots (same file evaluate.py /
        # run_mobo.py write) so this run's random surface can be rebuilt later with
        # ``Ensemble(**json.load(open(path)))``.
        os.makedirs(save_dir, exist_ok=True)
        cfg_out = os.path.join(save_dir, "ensemble_config.json")
        with open(cfg_out, "w") as f:
            json.dump(ens_cfg, f, indent=2)
        print(f"    Landscape config written to {cfg_out}")
        grid_pts = ternary_grid(TERNARY_GRID_N)
        grid_vals = rf.predict(grid_pts)
        # Ensemble optima are maxima (its output is mapped into [0.5, 1]).
        maximize = True
        print("    Mode forced: MAXIMIZE (Ensemble optima are maxima).")
    else:
        print("\n[1] Loading data and training RF …")
        X_data, y_data = load_data(csv_path)
        print(f"    {X_data.shape[0]} samples loaded.")
        rf = train_rf(X_data, y_data)
        print(f"    Train R² = {rf.score(X_data, y_data):.4f}")

        print("    Building ternary evaluation grid …")
        grid_pts = ternary_grid(TERNARY_GRID_N)
        grid_vals = rf.predict(grid_pts)

        maximize = prompt_minimize_or_maximize()
        mode_str = "MAXIMIZE (RF ascent; ZoMBI maximizes RF directly)" if maximize else (
            "MINIMIZE (RF negated for ZoMBI; GP still uses natural-gradient ascent on acquisition)"
        )
        print(f"\n    Mode selected: {mode_str}")

    # ── Step 2: reference extrema ─────────────────────────────────────────────
    # The picker needs a GUI to click on, so it is skipped under --background
    # (Agg has no window). Reference extrema are display-only overlays, so an
    # empty list just omits the blue ★ markers.
    # Under --ackley the optima are known analytically, so the picker is skipped
    # and the analytic maxima are used directly.
    goal_pl = "maxima" if maximize else "minima"
    if ackley or ensemble:
        # Both analytic objectives expose ``known_maxima`` (composition, value), so
        # the reference optima are known exactly — no min/max prompt, no clicking.
        true_minima = rf.known_maxima
        flag = "--ackley" if ackley else "--ensemble"
        if not background:
            plt.ion()   # keep Tk root alive for the live per-iteration figures
        print(f"\n[2] {flag}: using {len(true_minima)} analytic reference {goal_pl}:")
        for i, (c, v) in enumerate(true_minima):
            print(f"      #{i + 1}  comp={np.round(c, 4)}  y={v:.5f}")
    elif background:
        print("\n[2] --background: skipping interactive extrema picker (headless).")
        true_minima: list = []
    else:
        print(f"\n[2] Opening interactive ternary — click near reference {goal_pl}, then Enter / Q.")
        plt.ion()   # interactive mode: keeps Tk root alive across multiple figures
        picker = ExtremaPicker(rf, grid_pts, grid_vals, maximize=maximize)
        true_minima = picker.run()
        print(f"    {len(true_minima)} reference {goal_pl} confirmed:")
        for i, (c, v) in enumerate(true_minima):
            print(f"      #{i + 1}  comp={np.round(c, 4)}  y={v:.5f}")

    # ── Step 3: ZoMBI-Hop setup ───────────────────────────────────────────────
    print("\n[3] Initialising ZoMBI-Hop …")
    dim = 3
    zparams = dict(ZOMBI_PARAMS)
    if hparams_path is not None:
        zparams.update(load_hparams(hparams_path))

    plot_state: dict = {"line_0": None, "line_1": None, "fig": None, "iter": 0}
    dh_ref: list = [None]   # filled with optimizer.data_handler after construction
    gp_ref: list = [None]   # filled with optimizer.gp_handler after construction
    opt_ref: list = [None]  # filled with the optimizer itself (zoom state, tolerances)
    plot_queue: queue.Queue = queue.Queue()

    ref_name = ("Ackley" if ackley else "Ensemble" if ensemble else "RF")
    print("    Building sim objective …")
    sim_obj = make_sim_objective(rf, DEVICE, DTYPE, maximize=maximize)
    print("    Building LineBO wrapper …")
    inner_wrap = make_linebo_wrapper(
        sim_obj, dim, NUM_LINES, DEVICE, DTYPE, plot_state,
    )
    print("    Building plotting wrapper …")
    full_wrap = make_plotting_wrapper(
        inner_wrap, dh_ref, plot_state,
        grid_pts, grid_vals, true_minima,
        save_dir=save_dir if SAVE_PLOTS else None,
        plot_queue=plot_queue,
        maximize=maximize,
        show_sampling=show_sampling,
        gp_ref=gp_ref,
        opt_ref=opt_ref,
        ref_name=ref_name,
    )

    print(f"    Generating initial data ({N_INIT_LINES} lines × {NUM_EXPERIMENTS} pts) …")
    X_init_a, X_init_e, Y_init = generate_init_data(
        rf, N_INIT_LINES, DEVICE, DTYPE, maximize=maximize,
    )
    print(f"    {X_init_a.shape[0]} initial points generated.")
    y_rng_lbl = "ZoMBI-internal Y" if maximize else "ZoMBI-internal Y (negated RF)"
    print(f"    Y_init range ({y_rng_lbl}): [{Y_init.min().item():.4f}, {Y_init.max().item():.4f}]")

    print("    Constructing ZoMBIHop optimizer (fits initial GP) …")
    optimizer = ZoMBIHop(
        objective=full_wrap,
        X_init_actual=X_init_a,
        X_init_expected=X_init_e,
        Y_init=Y_init,
        **zparams,
        device=str(DEVICE),
        dtype=DTYPE,
        run_uuid=None,
        checkpoint_dir=ckpt_dir,
        num_iterations_saved=50,
    )
    dh_ref[0] = optimizer.data_handler  # connect plotting wrapper to live DataHandler
    gp_ref[0] = optimizer.gp_handler    # GP belief landscape for the panel background
    opt_ref[0] = optimizer             # full_bounds / needle_move_tol for the overlays
    print("    ZoMBIHop ready.")

    # ── Step 4: run ZoMBI in background; main thread owns all Tk/matplotlib ───
    # Tkinter is not thread-safe: figure creation, display, and destruction must
    # all happen on the main thread.  ZoMBI pushes plot payloads onto plot_queue;
    # the main-thread loop below drains the queue and renders each figure.
    print("\n[4] Running ZoMBI-Hop …  (Ctrl+C to stop early)")
    print("=" * 70)

    zombi_exc: list = []

    def _run_zombi():
        try:
            optimizer.run(max_activations=float("inf"), time_limit_hours=None)
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            zombi_exc.append(exc)
            import traceback
            traceback.print_exc()

    zombi_thread = threading.Thread(target=_run_zombi, daemon=True, name="zombi-worker")
    zombi_thread.start()

    # Matplotlib TkAgg figures live in reference cycles (figure↔axes↔canvas), so
    # refcounting alone never frees their child Tk PhotoImage/Variable objects —
    # they wait for the cyclic garbage collector. That collector runs on whatever
    # thread crosses the allocation threshold, which is the allocation-heavy
    # zombi-worker thread; finalizing Tk objects off the main thread raises
    # "main thread is not in main loop". Disable automatic GC and drive cyclic
    # collection explicitly from the main-thread loop below so every Tk finalizer
    # runs on the main thread. (Harmless in Agg/headless mode: no Tk objects.)
    _tk_windowed = not matplotlib.get_backend().lower().startswith("agg")
    if _tk_windowed:
        gc.disable()

    current_fig: list = [None]

    def _drain_queue():
        """Render all pending plot payloads on the main thread."""
        while True:
            try:
                payload = plot_queue.get_nowait()
            except queue.Empty:
                break
            # Close the previous figure before opening the next one.
            if current_fig[0] is not None:
                try:
                    plt.close(current_fig[0])
                except Exception:
                    pass
                current_fig[0] = None
            fig = _plot_iteration(**payload)
            current_fig[0] = fig
            plot_state["fig"] = fig
            print(f"  [plot] iteration-{payload['iteration_num']} figure ready.", flush=True)

    try:
        while zombi_thread.is_alive():
            _drain_queue()
            # Run cyclic GC here (main thread) so any Tk finalizers for closed
            # figures execute on-thread rather than on the worker thread.
            if _tk_windowed:
                gc.collect()
            # plt.pause pumps the Tk event loop AND yields the CPU briefly.
            try:
                plt.pause(0.05)
            except Exception:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")

    # Drain any plots that arrived after the thread finished.
    _drain_queue()
    if _tk_windowed:
        gc.collect()
        gc.enable()

    if zombi_exc:
        print(f"\n[!] ZoMBI-Hop raised an exception: {zombi_exc[0]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[5] Optimisation finished.")
    dh = optimizer.data_handler
    all_locs = dh.get_all_needle_locations()
    all_vals = dh.get_all_needle_vals()
    all_results = dh.get_all_needle_results()
    if all_locs.shape[0] > 0:
        print(f"  Discovered needles ({all_locs.shape[0]} total):")
        for i, (nx, nv) in enumerate(
            zip(all_locs.cpu().numpy(), all_vals.cpu().numpy().ravel())
        ):
            y_rf = float(nv) if maximize else -float(nv)
            print(f"    #{i + 1}  {np.round(nx, 4)}  y (RF) = {y_rf:.5f}")
    else:
        print("  No needles found yet.")

    if true_minima:
        ref_word = "maxima" if maximize else "minima"
        print(f"  Reference {ref_word} (interactive selection):")
        for i, (c, v) in enumerate(true_minima):
            print(f"    #{i + 1}  {np.round(c, 4)}  y = {v:.5f}")

    print(f"\n  Plots saved to: {save_dir}")
    # Keep the last iteration figure open using flush_events + sleep so we
    # don't start a nested Tk mainloop (which conda run intercepts).
    # In --background there is no window to keep open, so just exit.
    last_fig = plot_state.get("fig")
    if last_fig is not None and not background:
        print("  Close the final plot window (or press Ctrl+C) to exit.")
        try:
            while plt.fignum_exists(last_fig.number):
                try:
                    last_fig.canvas.flush_events()
                except Exception:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive ZoMBI-Hop simulator on an RF or Ackley surrogate."
    )
    parser.add_argument(
        "--ackley",
        choices=Ackley.VARIANTS,
        default=None,
        metavar="{centroid,edge,vertex,multimodal}",
        help="Use an analytic negated-Ackley objective from synthetic_data/ackley.py "
             "instead of the campaign1a RF surrogate. The variant's peak(s) are the "
             "maxima, so the run is forced to maximize, the analytic optima are used "
             "as reference extrema, and the interactive picker / CSV load are skipped.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use the layered synthetic_data/ensemble.py objective (3-simplex) "
             "instead of the campaign1a RF surrogate. A NEW RANDOM landscape is "
             "drawn every run. Like --ackley the optima are known analytically, so "
             "the run is forced to maximize and the min/max prompt and the "
             "interactive picker are both skipped. The landscape that was drawn is "
             "written to interactive_testing/plots/ensemble_config.json.",
    )
    parser.add_argument(
        "--hparams",
        metavar="PATH",
        default=None,
        help="Path to a trial_* directory (or a trial.json file) — that "
             "specific trial's hyperparameters are used. A mobo_* run directory "
             "(or mobo_progress.json file) is also accepted, in which case the "
             "latest trial's hyperparameters are used. Either way they override "
             "the built-in ZoMBI defaults.",
    )
    parser.add_argument(
        "--show-sampling",
        action="store_true",
        help="Overlay a very thin, semi-transparent line for every candidate "
             "line the acquisition function was integrated over at each step.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run headless (Agg backend): no plot window pops up or steals focus "
             "during the run, and per-iteration PNGs are still saved to "
             "interactive_testing/plots/. Skips the interactive extrema picker.",
    )
    args = parser.parse_args()
    use_ensemble = bool(args.ensemble)
    if args.ackley and use_ensemble:
        parser.error("--ackley and --ensemble select different objectives; pick one.")
    main(
        hparams_path=args.hparams,
        show_sampling=args.show_sampling,
        background=args.background,
        ackley=args.ackley,
        ensemble=use_ensemble,
    )
