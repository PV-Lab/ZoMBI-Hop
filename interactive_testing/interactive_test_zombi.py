"""
interactive_test_zombi.py
=========================
Interactive ZoMBI-Hop simulator using a Random-Forest surrogate built from
campaign1a.csv.

Workflow
--------
1. Load ``data/campaign1a.csv``, train a 500-tree Random-Forest on
   (FAPbI3, MAPbI3, MAPbBr3) → Objective.
2. Choose **minimize** or **maximize** at the prompt (ZoMBI always runs as a
   maximiser internally; minimizing RF uses negated observations).
3. Open an interactive ternary plot.  Left-click near a local minimum or
   maximum (per your choice); L-BFGS-B refinement (gradient-based on the RF)
   snaps to a nearby extremum.  Press Enter / Q when done selecting.
4. Run ZoMBI-Hop (with LineBO), exactly as in ``scripts/run_zombi_main.py``,
   but evaluating the RF surrogate instead of a physical instrument.
   0.01 Gaussian noise is added to both inputs and outputs at every sample.
5. After every objective call, save a two-panel ternary figure to
   ``interactive_testing/plots/`` and display it (non-blocking):
     Left  – reference RF landscape + blue ★ for confirmed extrema (min or max).
     Right – ZoMBI-Hop exploration:
               • all sampled points (older = more transparent),
               • red ★ + faded-purple ellipse per discovered needle,
               • grey ✕ + grey circle per "old" (demoted) needle,
               • dashed-red boundary for current search bounds,
               • orange solid line for LineBO's main suggested line,
               • dotted cornflower-blue line for LineBO's cache line,
               • blue ★ for confirmed reference extrema.

Usage
-----
  cd <repo_root>
  conda run -n zombi-hop python interactive_testing/interactive_test_zombi.py
"""

from __future__ import annotations

import os
import time
import sys
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

matplotlib.use("TkAgg")  # must be called before pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from src import ZoMBIHop, LineBO
from src.core.linebo import line_simplex_segment, zero_sum_dirs
from src.utils.simplex import Ellipsoid, composition_to_ilr, ilr_to_composition, proj_simplex

# ── Configuration ─────────────────────────────────────────────────────────────
COMPOSITION_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
OBJECTIVE_COL = "Objective"
CORNER_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")

RF_N_ESTIMATORS = 500
NOISE_LEVEL = 0.01       # output noise std (added to y at every evaluation)
NOISE_LEVEL_ILR = 0.03  # input noise std in ILR space (≈ NOISE_LEVEL=0.01 in ambient at centroid, d=3)
NUM_EXPERIMENTS = 24     # points sampled per suggested line (mirrors run_zombi_main.py)
NUM_LINES = 10           # LineBO candidate lines per iteration
TERNARY_GRID_N = 120     # ternary grid resolution for reference heatmap
N_INIT_LINES = 2         # random lines to build the initial GP dataset

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
    convergence_pi_threshold=0.001,
    input_noise_threshold_mult=2.0,
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
        compositional geometry; Gaussian noise of std=NOISE_LEVEL is added to outputs,
      * returns ``(x_actual: (N, d), y: (N,))`` tensors on ``device``.

    ZoMBIHop maximizes ``y``. When ``maximize`` is False, RF outputs are negated so
    maximizing ``y`` corresponds to minimizing the RF surrogate.
    """

    def sim_objective(endpoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = endpoints[0, 0].to(dtype=torch.float64)
        right = endpoints[0, 1].to(dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS, dtype=torch.float64, device=left.device)
        pts_t = left.unsqueeze(0) + t.unsqueeze(1) * (right - left).unsqueeze(0)  # (N, 3)
        # Logistic-normal input noise: perturb in ILR space, invert back to simplex.
        # Respects the Aitchison geometry; no clip/renorm needed.
        z = composition_to_ilr(pts_t)
        z = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts_t = ilr_to_composition(z, d=pts_t.shape[1])       # (N, 3) on simplex
        pts_np = pts_t.detach().cpu().numpy()                  # numpy only at sklearn boundary
        raw = torch.tensor(rf.predict(pts_np).ravel(), dtype=dtype, device=device)
        y = raw if maximize else -raw
        y = y + torch.randn_like(y) * NOISE_LEVEL
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
            line_0=plot_state.get("line_0"),
            line_1=plot_state.get("line_1"),
            iteration_num=plot_state["iter"],
            save_dir=save_dir,
        )
        plot_queue.put(payload)
        return x_requested, x_actual, y

    return wrapped


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _draw_bounds_region(ax, bounds, n_sample: int = 5000) -> None:
    """
    Draw the trust-region (tensor bounds or Ellipsoid) as a dashed-red convex hull on the ternary.
    """
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


def _draw_needle_ellipsoid(
    ax,
    needle_x: np.ndarray,
    M: torch.Tensor | None,
    B: torch.Tensor | None,
) -> None:
    """
    Plot a red star for the needle and, if ellipsoid parameters are available,
    a faded purple polygon tracing the penalization ellipsoid in ternary space.

    Tangent-space mode (B is not None):
        Boundary: {x* + B @ u : u^T M u = 1}
    ILR mode (B is None):
        Boundary: {ilr_to_composition(ilr(x*) + u) : u^T M u = 1}
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
        d = needle_x.shape[0]
        M_np = M.cpu().numpy()
        eigvals, eigvecs = np.linalg.eigh(M_np)
        eigvals = np.maximum(eigvals, 1e-12)
        angles = np.linspace(0, 2 * np.pi, 200)
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        # Ellipse boundary: u = V diag(1/√λ) [cos,sin]^T  → u^T M u = 1
        u_ell = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ circle.T).T   # (200, d-1)

        if B is not None:
            # Legacy tangent-space mode: x = x* + B @ u, project back to simplex
            B_np = B.cpu().numpy()   # (d, d-1)
            delta_x = (B_np @ u_ell.T).T   # (200, d)
            ell_pts = needle_x.reshape(1, d) + delta_x
        else:
            # ILR mode: boundary in ILR space → map back to composition
            needle_t = torch.tensor(needle_x, dtype=torch.float64)
            needle_ilr = composition_to_ilr(needle_t.unsqueeze(0)).squeeze(0).cpu().numpy()  # (d-1,)
            z_ell = needle_ilr + u_ell   # (200, d-1)
            z_t = torch.tensor(z_ell, dtype=torch.float64)
            ell_pts = ilr_to_composition(z_t, d).cpu().numpy()   # (200, d)

        ell_pts = np.clip(ell_pts, 0, 1)
        s = ell_pts.sum(axis=1, keepdims=True)
        ell_pts = ell_pts / np.where(s < 1e-9, 1.0, s)
        ell_xy = comp_to_xy(ell_pts)
        poly = MplPolygon(
            ell_xy, closed=True,
            alpha=0.15, facecolor="purple", edgecolor="purple",
            linewidth=0.9, zorder=6,
        )
        ax.add_patch(poly)
        ax.plot(ell_xy[:, 0], ell_xy[:, 1], c="purple", lw=0.7, alpha=0.45, zorder=6)
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
    trust_ellipsoid     : Trust-region ``Ellipsoid``, or None.
    line_0, line_1      : each is (left_np, right_np) or None.
    iteration_num       : counter shown in the title and filename.
    save_dir            : directory to write the PNG, or None.
    """
    fig, (ax_ref, ax_exp) = plt.subplots(1, 2, figsize=(16, 6.8))
    fig.suptitle(
        f"ZoMBI-Hop Interactive Test  —  iteration {iteration_num}",
        fontsize=13,
    )

    # ── Left panel: RF reference ───────────────────────────────────────────────
    draw_ternary_frame(ax_ref)
    ax_ref.set_title("Reference: RF landscape", fontsize=11)
    gxy = comp_to_xy(grid_pts)
    sc_ref = ax_ref.scatter(
        gxy[:, 0], gxy[:, 1],
        c=grid_vals, cmap="viridis",
        s=6, alpha=0.72, zorder=2, rasterized=True,
    )
    fig.colorbar(sc_ref, ax=ax_ref, label="RF Objective", fraction=0.046, pad=0.04)
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
                edgecolors="gray", linewidths=0.2,
            )
        ax_exp.set_title(
            f"ZoMBI-Hop exploration  ({n} pared pts)", fontsize=11
        )

    # Current trust ellipsoid (dashed red polygon)
    if trust_ellipsoid is not None:
        _draw_bounds_region(ax_exp, trust_ellipsoid)

    # Active needles: red ★ + purple ellipse
    if needles is not None and needles.shape[0] > 0:
        nx_np = needles.cpu().numpy()
        M_list = needle_M_list or [None] * len(nx_np)
        for i, nx in enumerate(nx_np):
            Mi = M_list[i] if i < len(M_list) else None
            _draw_needle_ellipsoid(ax_exp, nx, Mi, needle_B)

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
            summary_lines.append("Trust lo: [" + " ".join(f"{v:.4f}" for v in lo_cpu) + "]")
            summary_lines.append("Trust hi: [" + " ".join(f"{v:.4f}" for v in hi_cpu) + "]")
        elif isinstance(trust_ellipsoid, Ellipsoid):
            c_cpu = trust_ellipsoid.c.detach().cpu().numpy()
            wM, _ = torch.linalg.eigh(trust_ellipsoid.M)
            wcpu = np.sort(wM.detach().cpu().numpy())
            summary_lines.append("Trust c: [" + " ".join(f"{v:.4f}" for v in c_cpu) + "]")
            summary_lines.append("Trust M eig: [" + " ".join(f"{v:.4f}" for v in wcpu) + "]")
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
        pts_t = (
            x_left.to(torch.float64).unsqueeze(0)
            + t.unsqueeze(1) * (x_right - x_left).to(torch.float64).unsqueeze(0)
        )
        z = composition_to_ilr(pts_t)
        z = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts_t = ilr_to_composition(z, d=pts_t.shape[1])
        pts_np = pts_t.detach().cpu().numpy()
        print(f"      evaluating RF on {len(pts_np)} points …", flush=True)
        raw = torch.tensor(rf.predict(pts_np).ravel(), dtype=dtype, device=device)
        y_vals = raw + torch.randn_like(raw) * NOISE_LEVEL
        print(f"      done. y range: [{y_vals.min():.4f}, {y_vals.max():.4f}]", flush=True)
        y_zombi = y_vals if maximize else -y_vals
        pts_out = pts_t.to(dtype=dtype, device=device)
        x_actual_list.append(pts_out)
        x_exp_list.append(pts_out)
        y_list.append(y_zombi)
    if not x_actual_list:
        raise RuntimeError("Could not generate any initial data lines.")
    return (
        torch.cat(x_actual_list, dim=0),
        torch.cat(x_exp_list, dim=0),
        torch.cat(y_list, dim=0).reshape(-1, 1),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "campaign1a.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(script_dir, "data", "campaign1a.csv")
    save_dir = os.path.join(script_dir, "plots")
    ckpt_dir = os.path.join(script_dir, "checkpoints")

    print("=" * 70)
    print("ZoMBI-Hop Interactive Test — RF Surrogate on campaign1a.csv")
    print(f"Device : {DEVICE}")
    print(f"Noise  : {NOISE_LEVEL}  (applied to both inputs and outputs)")
    print("=" * 70)

    # ── Step 1: data + RF ─────────────────────────────────────────────────────
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

    # ── Step 2: interactive reference extrema ─────────────────────────────────
    goal_pl = "maxima" if maximize else "minima"
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

    plot_state: dict = {"line_0": None, "line_1": None, "fig": None, "iter": 0}
    dh_ref: list = [None]   # filled with optimizer.data_handler after construction
    plot_queue: queue.Queue = queue.Queue()

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
            # plt.pause pumps the Tk event loop AND yields the CPU briefly.
            try:
                plt.pause(0.05)
            except Exception:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")

    # Drain any plots that arrived after the thread finished.
    _drain_queue()

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
    last_fig = plot_state.get("fig")
    if last_fig is not None:
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
    main()
