"""
interactive_test_zombi.py
=========================
Interactive ZoMBI-Hop simulator.

At startup you choose the search space (or pass ``--domain``):

  * **simplex**   — 3-component compositions (sum to 1), drawn as ternary plots.
  * **cartesian** — the unit square [0, 1]^2 (ZoMBI-Hop's ``BoxDomain``), drawn
    as ordinary square height maps.

By default the objective is a synthetic ``Ensemble`` landscape from
``synthetic_data/ensemble.py`` (``Ensemble(dim=3)`` on the simplex,
``CartesianEnsemble(dim=2)`` on the square), drawn from the same Sobol' config
sweep the benchmarks use (``--seed`` / ``--index``) with **anisotropy forced
off** (``aniso_strength = 0``).  Two simplex-only alternatives remain:
``--rf`` trains a Random-Forest surrogate on ``campaign1a.csv``, and
``--ackley {centroid|edge|vertex|multimodal}`` uses one of the analytic
negated-Ackley test functions in ``synthetic_data/ackley.py``.

Workflow
--------
0. Choose **simplex** or **cartesian** at the prompt (skipped with ``--domain``).
1. Build the objective:
     * default — the Ensemble landscape for the chosen domain (anisotropy off).
       Its true optima are maxima, so the run is forced to **maximize** and the
       known optima are used as the reference extrema (no clicking);
     * ``--rf`` (simplex only) — load ``data/campaign1a.csv`` and train a
       500-tree Random-Forest on (FAPbI3, MAPbI3, MAPbBr3) → Objective; or
     * ``--ackley VARIANT`` (simplex only) — the analytic Ackley surrogate,
       likewise forced to maximize with analytic reference optima.
2. ``--rf`` only: choose **minimize** or **maximize** at the prompt (ZoMBI
   always runs as a maximiser internally; minimizing uses negated observations).
3. ``--rf`` only: open an interactive ternary plot.  Left-click near a local
   minimum or maximum (per your choice); L-BFGS-B refinement (gradient-based on
   the surrogate) snaps to a nearby extremum.  Press Enter / Q when done
   selecting.  (Skipped under ``--background``, which has no window to click on.)
4. Run ZoMBI-Hop (with LineBO), exactly as in ``scripts/run_zombi_main.py``,
   but evaluating the surrogate instead of a physical instrument.  On the
   simplex, the requested line is pushed through the physics print model; on the
   square it is sampled as-is.  Multiplicative output noise
   (OUTPUT_NOISE_FRAC × |y|) is added at every sample, matched to data/2nd_real_run.db.
   With ``--pointwise`` each call instead measures the single proposed point
   (no LineBO, no print model), as ``benchmarks/sweeps`` does.
5. After every objective call, save a two-panel figure (ternary or square, per
   the domain) to ``interactive_testing/plots/`` and display it (non-blocking;
   PNGs are still saved but not displayed under ``--background``):
     Left  – reference surrogate landscape + blue ★ for confirmed extrema (min or max).
     Right – ZoMBI-Hop exploration:
               • all sampled points (older = more transparent),
               • red ★ + faded-purple ellipse per discovered needle,
               • grey ✕ + grey circle per "old" (demoted) needle,
               • dashed-red boundary for current search bounds,
               • orange solid line for LineBO's main suggested line,
               • dotted cornflower-blue line for LineBO's cache line,
               • thin dim-grey lines for every candidate line sampled this step
                 (only with ``--show-sampling``),
               • orange ✕ for the point measured this step (``--pointwise``),
               • blue ★ for confirmed reference extrema.

Usage
-----
  cd <repo_root>
  python interactive_testing/interactive_test_zombi.py

Flags
-----
  --domain {simplex,cartesian}
      Search space; skips the startup prompt. Defaults to simplex when stdin is
      not a terminal.
        python interactive_testing/interactive_test_zombi.py --domain cartesian

  --seed N / --index N
      Pick the Ensemble landscape: the ``index``-th config of the Sobol' sweep
      scrambled by ``seed`` (``random_ensemble_config``). Both default to 0.
      Anisotropy is always switched off regardless of what the config draws.

  --rf
      (simplex only) Use the campaign1a Random-Forest surrogate instead of the
      Ensemble landscape, with the min/max prompt and interactive picker.

  --ackley {centroid,edge,vertex,multimodal}
      (simplex only) Use an analytic negated-Ackley objective (from
      ``synthetic_data/ackley.py``) instead of the Ensemble landscape. The
      chosen variant's peak(s) are the maxima, so the run is forced to maximize
      and the analytic optima are drawn as reference extrema.
        python interactive_testing/interactive_test_zombi.py --ackley centroid

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

  --pointwise
      Point-wise sampling, the ``benchmarks/sweeps`` regime (see
      ``benchmarks/sweeps/POINTWISE.md``): every objective call measures ONE
      point, the candidate ZoMBI-Hop proposes (projected into the domain; on the
      simplex the print model is skipped). The initial design is 48 scrambled-
      Sobol' points instead of 2 random lines, and the call-counted
      hyperparameters (``max_iterations``, ``min_iters_per_zoom``,
      ``max_lines_per_activation``) are multiplied by 24 so each zoom and
      activation keeps its point budget. ``--show-sampling`` is ignored.
        python interactive_testing/interactive_test_zombi.py --domain cartesian --pointwise

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
from src.core.linebo import batch_line_bounds_segments, line_simplex_segment, zero_sum_dirs
from src.utils.domain import BoxDomain, SimplexDomain
from src.utils.simplex import Ellipsoid, composition_to_ilr, ilr_to_composition
from synthetic_data.ackley import Ackley
from synthetic_data.ensemble import Ensemble, random_ensemble_config
from optimize.composition_prediction import physics_simulate_line
from benchmarks.methods.base import sobol_design
from benchmarks.methods.zombihop import CALL_COUNTED, _zombihop_defaults

# ── Configuration ─────────────────────────────────────────────────────────────
COMPOSITION_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
OBJECTIVE_COL = "Objective"
CORNER_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")
AXIS_LABELS = ("x₁", "x₂")   # cartesian (unit-square) axes

# Search spaces offered at startup: dimensionality and the ensemble.py domain name.
DOMAIN_DIMS = {"simplex": 3, "cartesian": 2}
ENSEMBLE_DOMAINS = {"simplex": "simplex", "cartesian": "cube"}

RF_N_ESTIMATORS = 500
OUTPUT_NOISE_FRAC = 0.045  # output noise as a fraction of the true y (measured ≈ within 4.5%)
NOISE_LEVEL_ILR = 0.384     # input noise std in ILR space (≈ 0.128 ambient per-component × 3; see default_hparams.DEFAULT_INPUT_NOISE)
NUM_EXPERIMENTS = 24     # points sampled per suggested line (mirrors run_zombi_main.py)
NUM_LINES = 10           # LineBO candidate lines per iteration
TERNARY_GRID_N = 120     # ternary grid resolution for reference heatmap
N_INIT_LINES = 2         # random lines to build the initial GP dataset
# --pointwise (benchmarks/sweeps' regime, see benchmarks/sweeps/POINTWISE.md):
N_INIT_POINTS = N_INIT_LINES * NUM_EXPERIMENTS   # Sobol' points in the initial design (48)
LINE_EQUIVALENT = NUM_EXPERIMENTS                # CALL_COUNTED hparams are scaled by this

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
ZOMBI_PARAMS: dict = dict(
    max_zooms=6,
    max_iterations=6,
    top_m_points=4,
    n_restarts=100,
    raw=1323,
    input_noise_threshold_mult=3.0,
    output_noise_threshold_mult=0.5,
    n_consecutive_converged=5,
    max_gp_points=3000,
    acquisition_type="ucb",
    ucb_beta=0.6677950695897094,
    nat_grad_step=0.024757059158665974,
    nat_grad_max_steps=67,
    max_penalty_radius=1.0,
    # Point paring
    paring_spatial_halfnoise=0.5,
    paring_y_noise_multiplier=1.0,
    input_noise_ilr=NOISE_LEVEL_ILR,
    # Needle-ellipsoid failure handling
    needle_shrink_factor=0.85,
    needle_stop_noise_multiplier=3.0,
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


# ── Domain-aware plot geometry ───────────────────────────────────────────────
# ``domain`` is "simplex" (ternary drawing) or "cartesian" (unit square, where a
# point's plot coordinates are just its two components).

def to_xy(pts: np.ndarray, domain: str) -> np.ndarray:
    """(N, d) domain points → (N, 2) plot coordinates."""
    if domain == "simplex":
        return comp_to_xy(pts)
    p = np.asarray(pts, dtype=float)
    return p.reshape(1, -1) if p.ndim == 1 else p


def draw_frame(ax, domain: str, pad: float = 0.04) -> None:
    """Triangle outline (simplex) or unit-square outline + axes (cartesian)."""
    if domain == "simplex":
        draw_ternary_frame(ax, pad)
        return
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(AXIS_LABELS[0])
    ax.set_ylabel(AXIS_LABELS[1])


def domain_grid(domain: str, n: int = TERNARY_GRID_N) -> np.ndarray:
    """Uniform evaluation grid over the domain: ternary (simplex) or n×n (square)."""
    if domain == "simplex":
        return ternary_grid(n)
    g = np.linspace(0.0, 1.0, n + 1)
    gx, gy = np.meshgrid(g, g)
    return np.column_stack([gx.ravel(), gy.ravel()])


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


def prompt_domain() -> str:
    """Ask whether to search the 3-component simplex or the 2-D unit square."""
    if not sys.stdin.isatty():
        print("\n  stdin is not a terminal and --domain was not given — using 'simplex'.")
        return "simplex"
    while True:
        raw = input(
            "\nSearch space: type 'simplex' (3-component compositions, ternary plots)\n"
            "or 'cartesian' (unit square [0,1]², square plots)  (default: simplex).\n"
            "> "
        ).strip().lower()
        if raw in ("", "s", "simplex"):
            return "simplex"
        if raw in ("c", "cart", "cartesian", "box", "cube", "square"):
            return "cartesian"
        print("  Please enter 'simplex' or 'cartesian'.")


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
    domain: str = "simplex",
):
    """
    Return ``sim_objective(endpoints) → (x_actual, y)`` where:
      * ``endpoints`` is a ``(k, 2, d)`` tensor of ranked line endpoints,
      * the first line (index 0) is sampled at NUM_EXPERIMENTS points — through
        the physics print model on the simplex, evenly spaced on the square,
      * multiplicative output noise (std=OUTPUT_NOISE_FRAC × |y|) is added to outputs,
      * returns ``(x_actual: (N, d), y: (N,))`` tensors on ``device``.

    ZoMBIHop maximizes ``y``. When ``maximize`` is False, RF outputs are negated so
    maximizing ``y`` corresponds to minimizing the RF surrogate.
    """

    def sim_objective(endpoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = endpoints[0, 0].to(dtype=torch.float64)
        right = endpoints[0, 1].to(dtype=torch.float64)
        if domain == "simplex":
            # Physics-based "actual" compositions: push the requested start→end line
            # through the deterministic hardware print model (ramp lag/overshoot +
            # junction-volume diffusion mixing) instead of adding random ILR-space
            # input noise.
            pts_t = physics_simulate_line(left, right, num_points=NUM_EXPERIMENTS,
                                          device=left.device, dtype=torch.float64)  # (N, 3)
        else:
            # The print model is composition-specific; on the square the requested
            # line is measured as-is.
            t = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS, device=left.device, dtype=torch.float64)
            pts_t = left.unsqueeze(0) + t.unsqueeze(1) * (right - left).unsqueeze(0)
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
    zdomain=None,
):
    """
    Build a ZoMBIHop-compatible objective wrapper::

        wrapper(x_tell, bounds, acq_fn) → (x_requested, x_actual, y)

    Internally uses ``LineBO.ranked_line_endpoints`` (not ``LineBO.sampler``)
    so we can capture the top-2 ranked lines before evaluation and store them
    in ``plot_state`` for the per-iteration plots.  ``zdomain`` is the ZoMBI-Hop
    ``Domain`` (simplex or box) the lines are drawn in.
    """
    zdomain = zdomain if zdomain is not None else SimplexDomain()
    linebo = LineBO(
        sim_obj,
        dim,
        num_points_per_line=100,
        num_lines=num_lines,
        device=str(device),
        domain=zdomain,
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
        # The PCA-reconstructed line lives in ℝ^d and can leave the domain
        # (e.g. negative components / not summing to 1) — project back first.
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
            x_requested = zdomain.project(x_requested)   # guarantee domain membership
        else:
            x_requested = x_actual.clone()

        return x_requested, x_actual, y

    return wrapper


def make_point_wrapper(
    rf,
    device: torch.device,
    dtype: torch.dtype,
    plot_state: dict,
    *,
    maximize: bool,
    zdomain,
):
    """
    Point-mode counterpart of :func:`make_linebo_wrapper` (``--pointwise``)::

        wrapper(x_tell, bounds, acq_fn) → (x_requested, x_actual, y)

    Measures exactly the candidate ZoMBI-Hop proposed, projected into the domain,
    as ``benchmarks/methods/zombihop.py`` does with ``sampling="point"``. LineBO is
    not used. The physics print model is a model of a printed *line*, so on the
    simplex the point is measured as requested (``x_actual == x_requested``).
    Output noise is the same multiplicative OUTPUT_NOISE_FRAC × |y|.
    """

    def wrapper(x_tell, bounds: torch.Tensor, acquisition_function):
        x = zdomain.project(x_tell.detach().to(device=device, dtype=dtype).reshape(1, -1))
        plot_state["point"] = x[0].cpu().numpy()
        raw = torch.tensor(rf.predict(x.cpu().numpy()).ravel(), dtype=dtype, device=device)
        y = raw if maximize else -raw
        y = y + torch.randn_like(y) * (OUTPUT_NOISE_FRAC * y.abs())
        print(f"  [point] measured {np.round(plot_state['point'], 4)}  y={y.item():.4f}", flush=True)
        return x, x.clone(), y

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
    domain: str = "simplex",
    landscape_name: str = "RF",
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
        curr_bounds   = getattr(dh, "bounds", None)

        # Pared dataset — what the GP sees (noise-deduplicated).
        xp = getattr(dh, "X_pared", None)
        yp = getattr(dh, "Y_pared", None)
        if xp is not None and xp.shape[0] > 0:
            pared_X = xp.detach().cpu().numpy()
            pared_Y = yp.detach().cpu().numpy().ravel()
            if not maximize:
                pared_Y = -pared_Y   # un-negate so display shows true objective values
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
            line_0=plot_state.get("line_0"),
            line_1=plot_state.get("line_1"),
            point=plot_state.get("point"),
            sampling_lines=plot_state.get("sampling_lines") if show_sampling else None,
            iteration_num=plot_state["iter"],
            save_dir=save_dir,
            gp_grid_vals=_gp_landscape_vals(
                gp_ref[0] if gp_ref else None, grid_pts, maximize),
            domain=domain,
            landscape_name=landscape_name,
        )
        plot_queue.put(payload)
        return x_requested, x_actual, y

    return wrapped


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _draw_bounds_region(ax, bounds, n_sample: int = 5000, domain: str = "simplex") -> None:
    """
    Draw the trust-region (tensor bounds or Ellipsoid) as a dashed-red region:
    the convex hull of simplex ∩ box on the ternary, or the box itself on the square.
    """
    if domain == "cartesian":
        if not (isinstance(bounds, torch.Tensor) and bounds.shape[0] == 2):
            return
        lo = bounds[0].detach().cpu().numpy()
        hi = bounds[1].detach().cpu().numpy()
        xs = [lo[0], hi[0], hi[0], lo[0], lo[0]]
        ys = [lo[1], lo[1], hi[1], hi[1], lo[1]]
        ax.fill(xs, ys, color="red", alpha=0.06, zorder=4)
        ax.plot(xs, ys, "--", color="red", lw=2.0, alpha=0.75, zorder=5, label="Trust bounds")
        return
    from src.utils.simplex import random_simplex
    try:
        if isinstance(bounds, torch.Tensor) and bounds.shape[0] == 2:
            lo = bounds[0]
            hi = bounds[1]
            samp = random_simplex(n_sample, lo, hi, device=str(lo.device), torch_dtype=lo.dtype)
        elif isinstance(bounds, Ellipsoid):
            # Fallback for old Ellipsoid format (should not occur after refactor)
            from src.utils.simplex import sample_ellipsoid as _se
            samp = _se(n_sample, bounds, scale=1.0)
        else:
            return
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


def _needle_penalty_bands(
    needle_x: np.ndarray,
    M: torch.Tensor,
    B: torch.Tensor | None,
    *,
    n_bands: int = 18,
    n_ang: int = 160,
    domain: str = "simplex",
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
    On the square (``domain="cartesian"``) the box's basis is the identity, and
    contours are clipped to [0, 1]^2 rather than renormalised onto the simplex.
    """
    d = needle_x.shape[0]
    if domain == "cartesian" and B is None:
        B = torch.eye(d, dtype=torch.float64)
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
        if domain == "cartesian":
            return ell
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
    domain: str = "simplex",
) -> None:
    """
    Plot a red star for the needle and, if ellipsoid parameters are available, a
    red penalty gradient over the penalization ellipsoid: opaque red at the needle
    centre (where the repulsion is strongest) fading to fully transparent at the
    boundary (where the penalty vanishes).  Opacity is proportional to the penalty.
    """
    xy = to_xy(needle_x.reshape(1, -1), domain)
    ax.scatter(
        xy[0, 0], xy[0, 1],
        marker="*", s=280, c="red",
        zorder=9, edgecolors="darkred", linewidths=1.0,
    )
    if M is None:
        return
    try:
        for ring, penalty in _needle_penalty_bands(needle_x, M, B, domain=domain):
            ax.add_patch(MplPolygon(
                to_xy(ring, domain), closed=True,
                facecolor="red", edgecolor="none",
                alpha=0.5 * penalty, linewidth=0, zorder=6,
            ))
    except Exception as exc:
        print(f"  [ellipse warn] {exc}")


def _draw_landscape(ax, grid_pts: np.ndarray, vals: np.ndarray, domain: str, zorder: int):
    """Heat-map a grid of values: point cloud on the ternary, raster on the square."""
    if domain == "cartesian":
        n = int(round(np.sqrt(len(grid_pts))))
        g = grid_pts[:n, 0]   # domain_grid is row-major in x, so the first row is the x axis
        return ax.pcolormesh(
            g, g, np.asarray(vals).reshape(n, n),
            cmap="viridis", shading="nearest", alpha=0.85, zorder=zorder, rasterized=True,
        )
    gxy = comp_to_xy(grid_pts)
    return ax.scatter(
        gxy[:, 0], gxy[:, 1],
        c=vals, cmap="viridis",
        s=6, alpha=0.72, zorder=zorder, rasterized=True,
    )


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
    domain: str = "simplex",
    landscape_name: str = "RF",
    point: np.ndarray | None = None,
) -> plt.Figure:
    """
    Generate the two-panel figure for one iteration and optionally save it —
    ternary panels on the simplex, unit-square panels when ``domain="cartesian"``.

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
    trust_ellipsoid     : Trust-region ``Ellipsoid``, or None.
    line_0, line_1      : each is (left_np, right_np) or None.
    iteration_num       : counter shown in the title and filename.
    save_dir            : directory to write the PNG, or None.
    sampling_lines      : list of (left_np, right_np) for every candidate line the
                          acquisition was integrated over this step, or None to skip
                          the thin sampling overlay (``--show-sampling``).
    point               : (d,) point measured this step (``--pointwise``), or None.
    """
    fig, (ax_ref, ax_exp) = plt.subplots(1, 2, figsize=(16, 6.8))
    fig.suptitle(
        f"ZoMBI-Hop Interactive Test  —  iteration {iteration_num}",
        fontsize=13,
    )

    # ── Left panel: surrogate reference ────────────────────────────────────────
    draw_frame(ax_ref, domain)
    ax_ref.set_title(f"Reference: {landscape_name} landscape", fontsize=11)
    sc_ref = _draw_landscape(ax_ref, grid_pts, grid_vals, domain, zorder=2)
    fig.colorbar(sc_ref, ax=ax_ref, label=f"{landscape_name} Objective", fraction=0.046, pad=0.04)
    ref_lbl = "True maxima" if maximize else "True minima"
    if true_minima:
        mc = np.array([m[0] for m in true_minima])
        mxy = to_xy(mc, domain)
        ax_ref.scatter(
            mxy[:, 0], mxy[:, 1],
            marker="*", s=360, c="blue",
            zorder=11, edgecolors="navy", linewidths=1.3, label=ref_lbl,
        )
        ax_ref.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ── Right panel: ZoMBI-Hop exploration ────────────────────────────────────
    draw_frame(ax_exp, domain)
    ax_exp.set_title("ZoMBI-Hop exploration", fontsize=11)
    legend_handles = []

    # Background = the GP's current belief about the objective landscape (its
    # posterior mean over the evaluation grid).
    if gp_grid_vals is not None and len(gp_grid_vals) == len(grid_pts):
        sc_gp = _draw_landscape(ax_exp, grid_pts, gp_grid_vals, domain, zorder=1)
        fig.colorbar(sc_gp, ax=ax_exp, label="GP posterior mean",
                     fraction=0.046, pad=0.04)

    # Pared (noise-deduplicated) dataset — matches what the GP trains on.
    # Older points are more transparent; colour encodes true objective value.
    if pared_X is not None and len(pared_X) > 0:
        n      = len(pared_Y)
        alphas = np.linspace(0.15, 0.92, n) if n > 1 else np.array([0.92])
        xy_pts = to_xy(pared_X, domain)
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

    # Current trust ellipsoid (dashed red polygon)
    if trust_ellipsoid is not None:
        _draw_bounds_region(ax_exp, trust_ellipsoid, domain=domain)

    # Active needles: red ★ + purple ellipse
    if needles is not None and needles.shape[0] > 0:
        nx_np = needles.cpu().numpy()
        M_list = needle_M_list or [None] * len(nx_np)
        for i, nx in enumerate(nx_np):
            Mi = M_list[i] if i < len(M_list) else None
            _draw_needle_ellipsoid(ax_exp, nx, Mi, needle_B, domain=domain)

    # Sampling overlay: every candidate line the acquisition was integrated over.
    # Thin and semi-transparent so the chosen main/cache lines remain readable.
    if sampling_lines:
        for k, sl in enumerate(sampling_lines):
            sxy = to_xy(np.array(sl), domain)  # (2, 2)
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
        ll = to_xy(np.array(line_0), domain)  # (2, 2)
        (h0,) = ax_exp.plot(
            ll[:, 0], ll[:, 1],
            "-", color="orange", lw=2.5, alpha=0.90,
            zorder=7, label="LineBO (main)",
        )
        legend_handles.append(h0)
    if line_1 is not None:
        ll = to_xy(np.array(line_1), domain)
        (h1,) = ax_exp.plot(
            ll[:, 0], ll[:, 1],
            ":", color="cornflowerblue", lw=2.2, alpha=0.85,
            zorder=7, label="LineBO (cache)",
        )
        legend_handles.append(h1)

    # Point measured this step (--pointwise)
    if point is not None:
        pxy = to_xy(np.asarray(point).reshape(1, -1), domain)
        h_pt = ax_exp.scatter(
            pxy[:, 0], pxy[:, 1],
            marker="X", s=160, c="orange",
            zorder=10, edgecolors="black", linewidths=1.0,
            label="Measured point",
        )
        legend_handles.append(h_pt)

    # Reference extrema on the exploration panel too (blue stars)
    if true_minima:
        mc = np.array([m[0] for m in true_minima])
        mxy = to_xy(mc, domain)
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
            summary_lines.append("Trust lo: [" + " ".join(f"{v:.4f}" for v in lo_cpu) + "]")
            summary_lines.append("Trust hi: [" + " ".join(f"{v:.4f}" for v in hi_cpu) + "]")
        elif isinstance(trust_ellipsoid, Ellipsoid):
            c_cpu = trust_ellipsoid.c.detach().cpu().numpy()
            wM, _ = torch.linalg.eigh(trust_ellipsoid.M)
            wcpu = np.sort(wM.detach().cpu().numpy())
            summary_lines.append("Trust c: [" + " ".join(f"{v:.4f}" for v in c_cpu) + "]")
            summary_lines.append("Trust M eig: [" + " ".join(f"{v:.4f}" for v in wcpu) + "]")
    def _fmt(v) -> str:
        return "[" + " ".join(f"{x:.3f}" for x in np.asarray(v).ravel()) + "]"

    if line_0 is not None:
        summary_lines.append(f"Line0: {_fmt(line_0[0])} → {_fmt(line_0[1])}")
    if line_1 is not None:
        summary_lines.append(f"Line1: {_fmt(line_1[0])} → {_fmt(line_1[1])}")
    if point is not None:
        summary_lines.append(f"Point: {_fmt(point)}")
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
    domain: str = "simplex",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample ``n_lines`` random lines through the domain's centre (simplex
    centroid, or the middle of the unit square), evaluate the surrogate on each,
    and return tensors suitable as ``X_init_actual / X_init_expected / Y_init``
    for ZoMBIHop.

    ``Y_init`` matches ZoMBI's convention: raw objective (+ noise) when
    ``maximize``, negated objective (+ noise) when minimizing.
    """
    dim = DOMAIN_DIMS[domain]
    x_actual_list, x_exp_list, y_list = [], [], []
    for line_idx in range(n_lines):
        print(f"      init line {line_idx + 1}/{n_lines} …", flush=True)
        if domain == "simplex":
            x0 = torch.full((dim,), 1.0 / dim, device=device, dtype=dtype)
            direction = zero_sum_dirs(1, dim, device=device, dtype=dtype).squeeze(0)
            seg = line_simplex_segment(x0, direction)
        else:
            x0 = torch.full((dim,), 0.5, device=device, dtype=dtype)
            direction = BoxDomain().directions(1, dim, device=str(device), dtype=dtype)
            unit = BoxDomain().default_bounds(dim, device, dtype)
            xl, xr, t_lo, t_hi, mask = batch_line_bounds_segments(x0, direction, unit)
            seg = (t_lo[0], t_hi[0], xl[0], xr[0]) if bool(mask.any()) else None
        if seg is None:
            print(f"      init line {line_idx + 1}: no valid segment, skipping.")
            continue
        _t_min, _t_max, x_left, x_right = seg
        t = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS, dtype=torch.float64, device=device)
        # Clean straight segment = requested line; on the simplex the actual
        # compositions are the physics-simulated print (replaces the random
        # ILR-space input noise), on the square the line is measured as-is.
        pts_clean = (
            x_left.to(torch.float64).unsqueeze(0)
            + t.unsqueeze(1) * (x_right - x_left).to(torch.float64).unsqueeze(0)
        )
        if domain == "simplex":
            pts_t = physics_simulate_line(x_left, x_right, num_points=NUM_EXPERIMENTS,
                                          device=device, dtype=torch.float64)
        else:
            pts_t = pts_clean.clone()
        pts_np = pts_t.detach().cpu().numpy()
        print(f"      evaluating surrogate on {len(pts_np)} points …", flush=True)
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


def generate_init_points(
    rf,
    n_points: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    maximize: bool,
    domain: str = "simplex",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Point-mode initial design (``--pointwise``): ``n_points`` scrambled-Sobol'
    points, as ``benchmarks/methods`` gives every method. On the square they are
    Sobol' points in [0, 1]^2; on the simplex a (d-1)-dim Sobol' design is mapped
    uniformly onto the simplex by the sorted-spacings construction. Measured as
    requested, so ``X_init_actual == X_init_expected``.
    """
    dim = DOMAIN_DIMS[domain]
    if domain == "simplex":
        u = np.sort(sobol_design(n_points, dim - 1, seed), axis=1)
        pts = np.diff(np.hstack([np.zeros((n_points, 1)), u, np.ones((n_points, 1))]), axis=1)
    else:
        pts = sobol_design(n_points, dim, seed)
    raw = torch.tensor(rf.predict(pts).ravel(), dtype=dtype, device=device)
    y = raw + torch.randn_like(raw) * (OUTPUT_NOISE_FRAC * raw.abs())
    y_zombi = y if maximize else -y
    X = torch.as_tensor(pts, dtype=dtype, device=device)
    return X, X.clone(), y_zombi.reshape(-1, 1)


def scale_call_counted(zparams: dict, k: int) -> dict:
    """Point-mode hparams: the keys in ``CALL_COUNTED`` (budgets counted in objective
    calls, tuned when a call was a ``k``-point line) are multiplied by ``k``, so the
    per-zoom and per-activation *point* budgets stay as tuned — the same rule as
    ``ZoMBIHopMethod.resolved_hparams`` (see benchmarks/sweeps/POINTWISE.md). Keys
    absent from ``zparams`` are scaled from the ``ZoMBIHop`` constructor defaults."""
    out = dict(zparams)
    missing = [key for key in CALL_COUNTED if key not in out]
    base = {**(_zombihop_defaults(missing) if missing else {}),
            **{key: out[key] for key in CALL_COUNTED if key in out}}
    for key in CALL_COUNTED:
        out[key] = int(base[key]) * k
    return out


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

def main(
    hparams_path: str | None = None,
    show_sampling: bool = False,
    background: bool = False,
    ackley: str | None = None,
    use_rf: bool = False,
    domain: str | None = None,
    seed: int = 0,
    index: int = 0,
    pointwise: bool = False,
) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "campaign1a.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(script_dir, "data", "campaign1a.csv")
    save_dir = os.path.join(script_dir, "plots")
    ckpt_dir = os.path.join(script_dir, "checkpoints")

    # ── Step 0: search space ──────────────────────────────────────────────────
    if domain is None:
        domain = prompt_domain()
    if domain != "simplex" and (ackley or use_rf):
        raise SystemExit(
            "--rf and --ackley are simplex-only objectives; use --domain simplex, "
            "or drop them to run the cartesian Ensemble landscape.")
    dim = DOMAIN_DIMS[domain]
    zdomain = SimplexDomain() if domain == "simplex" else BoxDomain()

    if ackley:
        surrogate_name, landscape_name = f"Ackley ({ackley})", "Ackley"
    elif use_rf:
        surrogate_name, landscape_name = "RF Surrogate on campaign1a.csv", "RF"
    else:
        surrogate_name = f"Ensemble (seed={seed}, index={index}, anisotropy off)"
        landscape_name = "Ensemble"
    print("=" * 70)
    print(f"ZoMBI-Hop Interactive Test — {surrogate_name}")
    print(f"Domain : {domain}  (d={dim}, {zdomain!r})")
    print(f"Device : {DEVICE}")
    print(f"Noise  : input ILR std={NOISE_LEVEL_ILR}  |  output frac={OUTPUT_NOISE_FRAC} (× |y|)")
    print(f"Sampling: {'point (one point per call)' if pointwise else f'line (LineBO, {NUM_EXPERIMENTS} pts per call)'}")
    print("=" * 70)
    if pointwise and show_sampling:
        print("  --show-sampling draws LineBO candidate lines; ignored with --pointwise.")
        show_sampling = False

    # ── Step 1: build the objective surrogate ─────────────────────────────────
    # ``rf`` is any object exposing a scikit-learn-style ``predict((N, d)) → (N,)``
    # method: an ``Ensemble`` landscape (default), a trained RandomForestRegressor
    # (``--rf``) or an analytic ``Ackley`` instance (``--ackley``). Everything
    # downstream is agnostic to which.
    grid_pts = domain_grid(domain)
    if ackley:
        print(f"\n[1] Using analytic Ackley objective ('{ackley}') — no CSV / RF training.")
        rf = Ackley(ackley)
        grid_vals = rf.predict(grid_pts)
        # Ackley peaks are maxima; ZoMBI maximizes them directly.
        maximize = True
        print("    Mode forced: MAXIMIZE (Ackley peaks are maxima).")
    elif not use_rf:
        cfg = random_ensemble_config(dim, index, seed=seed, domain=ENSEMBLE_DOMAINS[domain])
        cfg["aniso_strength"] = 0.0   # anisotropy always off in this harness
        rf = Ensemble(**cfg)
        assert np.allclose(rf.axis_scale, 1.0), "anisotropy should be off"
        print(f"\n[1] Using {rf!r}")
        print("    Anisotropy: OFF (aniso_strength = 0, isotropic distance metric).")
        grid_vals = rf.predict(grid_pts)
        # The ensemble's true optima are its global maxima.
        maximize = True
        print("    Mode forced: MAXIMIZE (Ensemble true optima are maxima).")
    else:
        print("\n[1] Loading data and training RF …")
        X_data, y_data = load_data(csv_path)
        print(f"    {X_data.shape[0]} samples loaded.")
        rf = train_rf(X_data, y_data)
        print(f"    Train R² = {rf.score(X_data, y_data):.4f}")

        print("    Evaluating RF on the ternary grid …")
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
    # For Ackley / Ensemble the optima are known, so the picker is skipped and
    # the known maxima are used directly.
    goal_pl = "maxima" if maximize else "minima"
    if not use_rf:
        true_minima = rf.known_maxima
        if not background:
            plt.ion()   # keep Tk root alive for the live per-iteration figures
        print(f"\n[2] Using {len(true_minima)} known reference {goal_pl}:")
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
    zparams = dict(ZOMBI_PARAMS)
    if hparams_path is not None:
        zparams.update(load_hparams(hparams_path))
    if pointwise:
        zparams = scale_call_counted(zparams, LINE_EQUIVALENT)
        print(f"    Point mode: call-counted hparams × {LINE_EQUIVALENT}: "
              + ", ".join(f"{k}={zparams[k]}" for k in CALL_COUNTED))

    plot_state: dict = {"line_0": None, "line_1": None, "point": None, "fig": None, "iter": 0}
    dh_ref: list = [None]   # filled with optimizer.data_handler after construction
    gp_ref: list = [None]   # filled with optimizer.gp_handler after construction
    plot_queue: queue.Queue = queue.Queue()

    if pointwise:
        print("    Building point wrapper …")
        inner_wrap = make_point_wrapper(
            rf, DEVICE, DTYPE, plot_state, maximize=maximize, zdomain=zdomain,
        )
    else:
        print("    Building sim objective …")
        sim_obj = make_sim_objective(rf, DEVICE, DTYPE, maximize=maximize, domain=domain)
        print("    Building LineBO wrapper …")
        inner_wrap = make_linebo_wrapper(
            sim_obj, dim, NUM_LINES, DEVICE, DTYPE, plot_state, zdomain=zdomain,
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
        domain=domain,
        landscape_name=landscape_name,
    )

    if pointwise:
        print(f"    Generating initial data ({N_INIT_POINTS} Sobol' points) …")
        X_init_a, X_init_e, Y_init = generate_init_points(
            rf, N_INIT_POINTS, DEVICE, DTYPE, maximize=maximize, domain=domain, seed=seed,
        )
    else:
        print(f"    Generating initial data ({N_INIT_LINES} lines × {NUM_EXPERIMENTS} pts) …")
        X_init_a, X_init_e, Y_init = generate_init_data(
            rf, N_INIT_LINES, DEVICE, DTYPE, maximize=maximize, domain=domain,
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
        domain=zdomain,
    )
    dh_ref[0] = optimizer.data_handler  # connect plotting wrapper to live DataHandler
    gp_ref[0] = optimizer.gp_handler    # GP belief landscape for the panel background
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
            y_obj = float(nv) if maximize else -float(nv)
            print(f"    #{i + 1}  {np.round(nx, 4)}  y ({landscape_name}) = {y_obj:.5f}")
    else:
        print("  No needles found yet.")

    if true_minima:
        ref_word = "maxima" if maximize else "minima"
        ref_src = "interactive selection" if use_rf else "known optima"
        print(f"  Reference {ref_word} ({ref_src}):")
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
        description="Interactive ZoMBI-Hop simulator on an Ensemble, RF or Ackley "
                    "surrogate, in simplex or cartesian space."
    )
    parser.add_argument(
        "--domain",
        choices=tuple(DOMAIN_DIMS),
        default=None,
        help="Search space: 'simplex' (3-component compositions, ternary plots) or "
             "'cartesian' (unit square, square plots). Prompted for if omitted.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Ensemble landscape: Sobol' scramble / placement seed (default 0).",
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Ensemble landscape: index into the Sobol' config sweep (default 0).",
    )
    objective = parser.add_mutually_exclusive_group()
    objective.add_argument(
        "--rf",
        action="store_true",
        help="(simplex only) Use the campaign1a Random-Forest surrogate instead of "
             "the Ensemble landscape, with the min/max prompt and interactive picker.",
    )
    objective.add_argument(
        "--ackley",
        choices=Ackley.VARIANTS,
        default=None,
        metavar="{centroid,edge,vertex,multimodal}",
        help="(simplex only) Use an analytic negated-Ackley objective from "
             "synthetic_data/ackley.py instead of the Ensemble landscape. The "
             "variant's peak(s) are the maxima, so the run is forced to maximize "
             "and the analytic optima are used as reference extrema.",
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
    parser.add_argument(
        "--pointwise",
        action="store_true",
        help="Measure ONE point per objective call (the candidate ZoMBI-Hop "
             f"proposes) instead of a LineBO line, with a {N_INIT_POINTS}-point "
             "Sobol' initial design and the call-counted hyperparameters scaled "
             f"x{LINE_EQUIVALENT} — the benchmarks/sweeps regime "
             "(see benchmarks/sweeps/POINTWISE.md).",
    )
    args = parser.parse_args()
    main(
        hparams_path=args.hparams,
        show_sampling=args.show_sampling,
        background=args.background,
        ackley=args.ackley,
        use_rf=args.rf,
        domain=args.domain,
        seed=args.seed,
        index=args.index,
        pointwise=args.pointwise,
    )
