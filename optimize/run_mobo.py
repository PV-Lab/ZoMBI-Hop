"""
optimize/run_mobo.py
====================
Multi-objective Bayesian optimisation (MOBO) of ZoMBI-Hop hyperparameters.

Three objectives (all minimised):
  1. dist_to_needles    – mean greedy distance from discovered needles to the
                          nearest true optima (no-repeat matching; unmatched
                          true optima incur a penalty of 1.0)
  2. dup_fraction       – fraction of sampled points whose nearest neighbour
                          in input space is within noise/2
  3. runtime            – total wall-clock seconds for the ZoMBI run

Test functions (choose at startup):
  A. RF surrogate on campaign1a.csv  – interactive extrema picker with
                                       L-BFGS-B gradient refinement
  B. Gaussian mixture on simplex     – 3 known peaks near each corner
  C. Ackley in ILR space             – global minimum at centroid [1/3,1/3,1/3]

MOBO engine: qLogNEHVI (BoTorch, maximises negated objectives).
max_activations for each trial = n_true_needles × 2.

Usage
-----
  conda activate zombi-hop
  python optimize/run_mobo.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import tempfile
import warnings

import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize as sp_minimize

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# BoTorch MOBO imports
from botorch.exceptions import InputDataWarning
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement as qLogNEHVI
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples

warnings.filterwarnings("ignore", category=InputDataWarning)

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src import ZoMBIHop, LineBO
from src.core.linebo import line_simplex_segment, zero_sum_dirs
from src.utils.simplex import composition_to_ilr, ilr_to_composition, proj_simplex

# ─── Global config ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64

NOISE_LEVEL     = 0.01
NOISE_LEVEL_ILR = 0.03
NUM_EXPERIMENTS = 24
NUM_LINES       = 10
N_INIT_LINES    = 2

# MOBO settings
N_INIT_TRIALS    = 8
N_MOBO_TRIALS    = 40
N_MOBO_RESTARTS  = 10
N_MOBO_SAMPLES   = 512

# ZoMBI-Hop fixed params (not optimised — infrastructure / noise model constants)
ZOMBI_FIXED = dict(
    max_gp_points=3000,
    acquisition_type="ucb",
    input_noise_ilr=NOISE_LEVEL_ILR,
    verbose=False,
)

RF_N_ESTIMATORS = 500
TERNARY_GRID_N  = 80
_SQRT3_2        = math.sqrt(3) / 2
CORNER_LABELS   = ("FAPbI3", "MAPbI3", "MAPbBr3")

# ─── Hyperparameter search space ──────────────────────────────────────────────
# Each entry: (lo, hi, transform) — transform ∈ {"log", "linear", "int"}
# Normalised to [0, 1] for MOBO; unnormalised when calling ZoMBI.

HPARAM_SPACE: dict[str, tuple] = {
    # Acquisition optimisation
    "nat_grad_step":               (0.001,  0.5,   "log"),
    "nat_grad_max_steps":          (10,     200,   "int"),
    "n_restarts":                  (20,     300,   "int"),
    "raw":                         (200,    2000,  "int"),
    # Acquisition function
    "ucb_beta":                    (0.05,   3.0,   "linear"),
    # Zoom / convergence
    "max_zooms":                   (2,      10,    "int"),
    "max_iterations":              (2,      10,    "int"),
    "top_m_points":                (2,      8,     "int"),
    "n_consecutive_converged":     (2,      10,    "int"),
    "convergence_pi_threshold":    (1e-4,   0.05,  "log"),
    "input_noise_threshold_mult":  (0.5,    6.0,   "linear"),
    "output_noise_threshold_mult": (0.1,    2.0,   "linear"),
    # Penalisation & needle
    "max_penalty_radius":          (0.2,    3.0,   "linear"),
    "needle_shrink_factor":        (0.55,   0.99,  "linear"),
    "needle_stop_noise_multiplier":(1.0,    8.0,   "linear"),
    # Point paring (deduplication)
    "paring_spatial_halfnoise":    (0.1,    2.0,   "linear"),
    "paring_y_noise_multiplier":   (0.1,    3.0,   "linear"),
}
HPARAM_NAMES = list(HPARAM_SPACE.keys())
N_HPARAMS    = len(HPARAM_NAMES)


def norm_to_hparams(x_norm: torch.Tensor) -> dict:
    params = {}
    for i, name in enumerate(HPARAM_NAMES):
        lo, hi, tfm = HPARAM_SPACE[name]
        v = float(x_norm[i].clamp(0.0, 1.0).item())
        if tfm == "log":
            params[name] = math.exp(math.log(lo) + v * (math.log(hi) - math.log(lo)))
        elif tfm == "int":
            params[name] = int(round(lo + v * (hi - lo)))
        else:
            params[name] = lo + v * (hi - lo)
    return params


# ─── Standard simplex objective functions ─────────────────────────────────────

class GaussianMixture:
    """
    Sum of Gaussians centred at known simplex compositions; maximise.
    Known true optima = peak centres.
    """
    maximize = True

    def __init__(self, peaks: list[list | np.ndarray], sigma: float = 0.12):
        self.peaks = [np.asarray(p, dtype=float) for p in peaks]
        self.peaks = [p / p.sum() for p in self.peaks]
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> float:
        return float(sum(
            np.exp(-np.sum((x - p) ** 2) / (2 * self.sigma ** 2))
            for p in self.peaks
        ))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return list(self.peaks)


class AckleyILR:
    """
    Ackley in ILR space; global minimum at centroid [1/3, 1/3, 1/3].
    Minimise.
    """
    maximize = False
    CENTROID = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

    def __call__(self, x: np.ndarray) -> float:
        z = composition_to_ilr(
            torch.tensor(x, dtype=torch.float64).unsqueeze(0)
        ).squeeze(0).cpu().numpy()
        a, b, c = 20.0, 0.2, 2 * math.pi
        t1 = -a * math.exp(-b * math.sqrt((z ** 2).mean()))
        t2 = -math.exp(float(np.cos(c * z).mean()))
        return float(t1 + t2 + a + math.e)

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [self.CENTROID.copy()]


# ─── Ternary helpers (RF interactive picker) ──────────────────────────────────

def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def xy_to_comp(x: float, y: float) -> np.ndarray:
    c2 = y / _SQRT3_2
    c1 = x - 0.5 * c2
    return np.array([1.0 - c1 - c2, c1, c2])


def draw_ternary_frame(ax, pad: float = 0.04) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, CORNER_LABELS[0], ha="right", va="top",    fontsize=9)
    ax.text(1+pad, -pad, CORNER_LABELS[1], ha="left",  va="top",    fontsize=9)
    ax.text(0.5, _SQRT3_2+pad, CORNER_LABELS[2], ha="center", va="bottom", fontsize=9)


def ternary_grid(n: int = 80) -> np.ndarray:
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


def _log_softmax(log_x: np.ndarray) -> np.ndarray:
    z = log_x - np.max(log_x)
    x = np.exp(z)
    s = x.sum()
    return x / (s if s > 0 else 1.0)


def _refine_extremum(
    rf: RandomForestRegressor,
    x0: np.ndarray,
    *,
    maximize: bool,
    max_l1: float = 0.10,
) -> tuple[np.ndarray, float]:
    """L-BFGS-B refinement toward a nearby RF extremum in log-softmax coords."""
    x0 = np.clip(x0, 1e-12, None)
    x0 = x0 / x0.sum()
    log_x0 = np.log(np.maximum(x0, 1e-300))

    def obj(log_x: np.ndarray) -> float:
        val = float(rf.predict(_log_softmax(log_x).reshape(1, -1))[0])
        return -val if maximize else val

    res = sp_minimize(obj, log_x0, method="L-BFGS-B",
                      options={"maxiter": 400, "ftol": 1e-9})
    x_opt = _log_softmax(res.x)
    if np.abs(x_opt - x0).sum() > max_l1:
        return x0, float(rf.predict(x0.reshape(1, -1))[0])
    return x_opt, float(rf.predict(x_opt.reshape(1, -1))[0])


class ExtremaPicker:
    """Interactive ternary RF landscape; left-click → L-BFGS-B refinement."""

    def __init__(self, rf, grid_pts, grid_vals, *, maximize: bool):
        self.rf, self.grid_pts, self.grid_vals = rf, grid_pts, grid_vals
        self.maximize = maximize
        self.extrema: list[tuple[np.ndarray, float]] = []
        self._fig = self._ax = None
        self._done = False

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
        print(f"  → {tag}: {np.round(x_ref, 4)}  y={y_ref:.5f}")
        self.extrema.append((x_ref, y_ref))
        xy = comp_to_xy(x_ref.reshape(1, 3))
        self._ax.scatter(xy[0, 0], xy[0, 1], marker="*", s=340, c="blue",
                         zorder=12, edgecolors="navy", linewidths=1.3)
        self._fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key in ("enter", "q", "escape"):
            self._done = True

    def run(self) -> list[tuple[np.ndarray, float]]:
        self._done = False
        fig, ax = plt.subplots(figsize=(7, 6.5))
        self._fig, self._ax = fig, ax
        draw_ternary_frame(ax)
        goal = "maximum" if self.maximize else "minimum"
        ax.set_title(f"RF landscape — click near {goal}, then Enter / Q", fontsize=10)
        gxy = comp_to_xy(self.grid_pts)
        sc = ax.scatter(gxy[:, 0], gxy[:, 1], c=self.grid_vals,
                        cmap="viridis", s=8, alpha=0.80, zorder=2, rasterized=True)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.text(0.5, -0.07, "Click extrema.  Enter / Q when done.",
                transform=ax.transAxes, ha="center", fontsize=9, style="italic")
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
        return self.extrema


# ─── Metrics ──────────────────────────────────────────────────────────────────

UNMATCHED_PENALTY = 10.0   # added to total distance for each unmatched true optimum


def metric_dist_to_needles(
    discovered: np.ndarray,
    true_optima: list[np.ndarray],
) -> float:
    """
    Greedy no-repeat matching: for each true optimum find the closest
    unmatched discovered needle.  Unmatched true optima add UNMATCHED_PENALTY.
    Returns mean distance.
    """
    if not true_optima:
        return 0.0
    if len(discovered) == 0:
        return UNMATCHED_PENALTY
    used: set[int] = set()
    total = 0.0
    for t in true_optima:
        best_d, best_j = float("inf"), -1
        for j, d in enumerate(discovered):
            if j in used:
                continue
            dist = float(np.linalg.norm(np.asarray(d) - np.asarray(t)))
            if dist < best_d:
                best_d, best_j = dist, j
        if best_j >= 0:
            used.add(best_j)
            total += best_d
        else:
            total += UNMATCHED_PENALTY
    return total / len(true_optima)


def metric_dup_fraction(X_all: np.ndarray, threshold: float) -> float:
    """Fraction of points that have at least one neighbour within threshold."""
    n = len(X_all)
    if n <= 1:
        return 0.0
    diff  = X_all[:, None, :] - X_all[None, :, :]     # (n, n, d)
    dists = np.sqrt((diff ** 2).sum(axis=-1))           # (n, n)
    np.fill_diagonal(dists, np.inf)
    return float((dists < threshold).any(axis=1).mean())


# ─── ZoMBI sim-objective factory ──────────────────────────────────────────────

def make_sim_obj(fn_callable, device, dtype, *, maximize: bool):
    """Wrap any f: (d,) np.ndarray → float as a ZoMBI sim_objective."""

    def sim_objective(endpoints: torch.Tensor):
        left  = endpoints[0, 0].to(torch.float64)
        right = endpoints[0, 1].to(torch.float64)
        t     = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS,
                               dtype=torch.float64, device=left.device)
        pts_t = left.unsqueeze(0) + t.unsqueeze(1) * (right - left).unsqueeze(0)
        z     = composition_to_ilr(pts_t)
        z     = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts_t = ilr_to_composition(z, d=pts_t.shape[1])
        pts_np = pts_t.detach().cpu().numpy()
        raw    = np.array([fn_callable(x) for x in pts_np], dtype=float)
        y      = torch.tensor(raw if maximize else -raw, dtype=dtype, device=device)
        y      = y + torch.randn_like(y) * NOISE_LEVEL
        return pts_t.to(dtype=dtype, device=device), y

    return sim_objective


def make_linebo_wrapper(sim_obj, dim: int, num_lines: int, device, dtype):
    """LineBO wrapper identical to interactive testing (with simplex projection)."""
    linebo = LineBO(sim_obj, dim,
                   num_points_per_line=100, num_lines=num_lines, device=str(device))

    def wrapper(x_tell, bounds, acq_fn):
        x_left_r, x_right_r = linebo.ranked_line_endpoints(x_tell, bounds, acq_fn)
        endpoints = torch.stack([x_left_r, x_right_r], dim=1)
        x_actual, y = sim_obj(endpoints)
        x_actual = x_actual.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype).ravel()
        if x_actual.shape[0] > 1:
            xc = x_actual - x_actual.mean(dim=0, keepdim=True)
            _, _, Vt = torch.linalg.svd(xc, full_matrices=False)
            direction = Vt[0]
            projs = xc @ direction
            t_vals = torch.linspace(projs.min().item(), projs.max().item(),
                                    x_actual.shape[0], device=device, dtype=dtype)
            x_requested = (x_actual.mean(dim=0).unsqueeze(0)
                           + t_vals.unsqueeze(1) * direction.unsqueeze(0))
            x_requested = proj_simplex(x_requested)
        else:
            x_requested = x_actual.clone()
        return x_requested, x_actual, y

    return wrapper


# ─── Trial evaluation ─────────────────────────────────────────────────────────

def _gen_init_data(fn_callable, maximize: bool):
    """Generate N_INIT_LINES random simplex lines; return (X_a, X_e, Y, all_X)."""
    x_a_list, x_e_list, y_list, all_X = [], [], [], []
    for _ in range(N_INIT_LINES):
        x0  = torch.full((3,), 1.0 / 3, device=DEVICE, dtype=DTYPE)
        dir_ = zero_sum_dirs(1, 3, device=DEVICE, dtype=DTYPE).squeeze(0)
        seg  = line_simplex_segment(x0, dir_)
        if seg is None:
            continue
        _, _, x_left, x_right = seg
        t     = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS, dtype=torch.float64, device=DEVICE)
        pts_t = (x_left.to(torch.float64).unsqueeze(0)
                 + t.unsqueeze(1) * (x_right - x_left).to(torch.float64).unsqueeze(0))
        z     = composition_to_ilr(pts_t)
        z     = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts_t = ilr_to_composition(z, d=pts_t.shape[1])
        pts_np = pts_t.detach().cpu().numpy()
        raw    = np.array([fn_callable(x) for x in pts_np], dtype=float)
        y_t    = torch.tensor(raw if maximize else -raw, dtype=DTYPE, device=DEVICE)
        y_t    = y_t + torch.randn_like(y_t) * NOISE_LEVEL
        pts_out = pts_t.to(dtype=DTYPE, device=DEVICE)
        x_a_list.append(pts_out)
        x_e_list.append(pts_out)
        y_list.append(y_t)
        all_X.extend(pts_np)
    if not x_a_list:
        raise RuntimeError("Could not generate any initial simplex lines.")
    return (
        torch.cat(x_a_list, dim=0),
        torch.cat(x_e_list, dim=0),
        torch.cat(y_list, dim=0).reshape(-1, 1),
        np.array(all_X),
    )


def _run_single_zombi(
    hparams: dict,
    fn_callable,
    true_optima: list[np.ndarray],
    n_activations: int,
    maximize: bool,
    label: str,
) -> tuple[float, float, float]:
    """Run one ZoMBI trial on a single test function. Returns (dist, dup, runtime)."""
    import shutil
    try:
        X_init_a, X_init_e, Y_init, init_X_np = _gen_init_data(fn_callable, maximize)
    except RuntimeError as exc:
        print(f"    [{label}] init failed: {exc}")
        return UNMATCHED_PENALTY, 1.0, 600.0

    all_X_run: list[np.ndarray] = list(init_X_np)
    sim_obj  = make_sim_obj(fn_callable, DEVICE, DTYPE, maximize=maximize)
    base_obj = make_linebo_wrapper(sim_obj, 3, NUM_LINES, DEVICE, DTYPE)

    def objective_with_accum(x_tell, bounds, acq_fn):
        x_req, x_act, y = base_obj(x_tell, bounds, acq_fn)
        all_X_run.extend(x_act.detach().cpu().numpy())
        return x_req, x_act, y

    ckpt_dir = tempfile.mkdtemp(prefix="zombi_mobo_")
    t0 = time.time()
    try:
        optimizer = ZoMBIHop(
            objective=objective_with_accum,
            X_init_actual=X_init_a,
            X_init_expected=X_init_e,
            Y_init=Y_init,
            **ZOMBI_FIXED,
            **hparams,
            device=str(DEVICE),
            dtype=DTYPE,
            run_uuid=None,
            checkpoint_dir=ckpt_dir,
            num_iterations_saved=5,
        )
        optimizer.run(max_activations=n_activations, time_limit_hours=None)
        runtime = time.time() - t0
    except Exception as exc:
        print(f"    [{label}] ZoMBI crashed: {exc}")
        return UNMATCHED_PENALTY, 1.0, time.time() - t0
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    dh        = optimizer.data_handler
    needle_t  = dh.get_all_needle_locations()
    discovered = needle_t.detach().cpu().numpy() if needle_t.numel() > 0 else np.empty((0, 3))
    X_sampled  = np.array(all_X_run) if all_X_run else np.empty((0, 3))

    d   = metric_dist_to_needles(discovered, true_optima)
    dup = metric_dup_fraction(X_sampled, NOISE_LEVEL / 2.0)
    print(f"    [{label}]  dist={d:.4f}  dup={dup:.4f}  t={runtime:.1f}s"
          f"  needles={len(discovered)}/{len(true_optima)}")
    return d, dup, runtime


# TestFn: (fn_callable, true_optima, n_activations, maximize, name)
TestFn = tuple


def evaluate_trial(
    x_norm: torch.Tensor,
    test_fns: list[TestFn],
    trial_idx: int,
) -> tuple[float, float, float]:
    """
    Run ZoMBI on every test function with the hyperparameters in x_norm.
    Returns (mean_dist, mean_dup, total_runtime) averaged / summed across functions.
    """
    hparams = norm_to_hparams(x_norm)
    hp_str  = "  ".join(
        f"{k}={round(v,4) if isinstance(v,float) else v}"
        for k, v in hparams.items()
    )
    print(f"\n  [trial {trial_idx:>3}]  {hp_str}")

    dists, dups, runtimes = [], [], []
    for fn, optima, n_act, maximize, name in test_fns:
        d, dup, t = _run_single_zombi(hparams, fn, optima, n_act, maximize, name)
        dists.append(d); dups.append(dup); runtimes.append(t)

    mean_dist = float(np.mean(dists))
    mean_dup  = float(np.mean(dups))
    total_t   = float(sum(runtimes))
    print(f"  [trial {trial_idx:>3}]  MEAN dist={mean_dist:.4f}  dup={mean_dup:.4f}"
          f"  total_t={total_t:.1f}s  ({len(test_fns)} functions)")
    return mean_dist, mean_dup, total_t


# ─── Running summary ──────────────────────────────────────────────────────────

def _save_running_summary(
    X_obs: list[torch.Tensor],
    Y_obs: list[torch.Tensor],
    save_path: str,
) -> None:
    """Write a JSON summary of all trials completed so far to save_path."""
    n = len(Y_obs)
    if n == 0:
        return
    Y_stk       = torch.stack(Y_obs)
    pareto_mask = is_non_dominated(Y_stk)

    metrics_all = [
        {
            "dist_to_needles": round(-Y_obs[i][0].item(), 6),
            "dup_fraction":    round(-Y_obs[i][1].item(), 6),
            "runtime_s":       round(-Y_obs[i][2].item(), 3),
        }
        for i in range(n)
    ]

    trials = [
        {
            "trial":   i + 1,
            "phase":   "sobol" if i < N_INIT_TRIALS else "mobo",
            "pareto":  bool(pareto_mask[i].item()),
            "metrics": metrics_all[i],
            "hparams": {
                k: (round(v, 8) if isinstance(v, float) else v)
                for k, v in norm_to_hparams(X_obs[i]).items()
            },
        }
        for i in range(n)
    ]

    dists    = [m["dist_to_needles"] for m in metrics_all]
    dups     = [m["dup_fraction"]    for m in metrics_all]
    runtimes = [m["runtime_s"]       for m in metrics_all]

    summary = {
        "n_trials":  n,
        "n_pareto":  int(pareto_mask.sum().item()),
        "averages": {
            "dist_to_needles": round(float(np.mean(dists)),    6),
            "dup_fraction":    round(float(np.mean(dups)),     6),
            "runtime_s":       round(float(np.mean(runtimes)), 3),
        },
        "best_dist": {
            "value":   round(min(dists), 6),
            "trial":   int(np.argmin(dists)) + 1,
        },
        "trials": trials,
    }

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [summary] {save_path}  ({n} trials, {summary['n_pareto']} Pareto)", flush=True)


# ─── MOBO loop ────────────────────────────────────────────────────────────────

def run_mobo(
    test_fns: list[TestFn],
    save_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    bounds = torch.zeros(2, N_HPARAMS, dtype=DTYPE, device=DEVICE)
    bounds[1] = 1.0

    X_obs: list[torch.Tensor] = []
    Y_obs: list[torch.Tensor] = []   # maximised: Y = (-dist, -dup, -runtime)

    fn_names = [tf[4] for tf in test_fns]
    print(f"\n{'='*70}")
    print(f"MOBO  |  {N_INIT_TRIALS} Sobol init + {N_MOBO_TRIALS} BO trials")
    print(f"Hyperparameters ({N_HPARAMS}): {HPARAM_NAMES}")
    print(f"Test functions ({len(test_fns)}): {fn_names}")
    print(f"{'='*70}")

    json_path = os.path.join(save_dir, "mobo_progress.json")

    # ── Sobol initial trials ──────────────────────────────────────────────────
    X_sobol = draw_sobol_samples(bounds=bounds, n=N_INIT_TRIALS, q=1).squeeze(1)
    for i, x in enumerate(X_sobol):
        d, dup, t = evaluate_trial(x, test_fns, trial_idx=i)
        X_obs.append(x.cpu())
        Y_obs.append(torch.tensor([-d, -dup, -t], dtype=DTYPE))
        _save_running_summary(X_obs, Y_obs, json_path)

    # ── MOBO iterations ───────────────────────────────────────────────────────
    for trial in range(N_MOBO_TRIALS):
        X_t = torch.stack(X_obs).to(DEVICE)
        Y_t = torch.stack(Y_obs).to(DEVICE)

        span      = (Y_t.max(dim=0).values - Y_t.min(dim=0).values).clamp(min=1e-6)
        ref_point = (Y_t.min(dim=0).values - 0.1 * span).tolist()

        model = SingleTaskGP(
            X_t, Y_t,
            input_transform=Normalize(d=N_HPARAMS),
            outcome_transform=Standardize(m=3),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq = qLogNEHVI(model=model, ref_point=ref_point, X_baseline=X_t)
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds.to(DEVICE),
            q=1,
            num_restarts=N_MOBO_RESTARTS,
            raw_samples=N_MOBO_SAMPLES,
        )
        x_new = candidate.squeeze(0).detach()

        d, dup, t = evaluate_trial(x_new, test_fns, trial_idx=N_INIT_TRIALS + trial)
        X_obs.append(x_new.cpu())
        Y_obs.append(torch.tensor([-d, -dup, -t], dtype=DTYPE))
        _save_running_summary(X_obs, Y_obs, json_path)

    return torch.stack(X_obs), torch.stack(Y_obs)


# ─── Results display ──────────────────────────────────────────────────────────

def show_pareto(
    X_obs: torch.Tensor,
    Y_obs: torch.Tensor,
    save_dir: str,
) -> torch.Tensor:
    pareto_mask = is_non_dominated(Y_obs)
    n_par = pareto_mask.sum().item()
    print(f"\n{'='*70}")
    print(f"Pareto front: {n_par} / {len(Y_obs)} trials")
    print(f"{'rank':>4}  {'dist':>8}  {'dup%':>8}  {'time(s)':>9}  hparams")
    print("-" * 70)
    pareto_idx = torch.where(pareto_mask)[0]
    # Sort by dist (best first)
    order = pareto_idx[Y_obs[pareto_idx, 0].argsort(descending=True)]
    for rank, idx in enumerate(order):
        y  = Y_obs[idx]
        hp = norm_to_hparams(X_obs[idx])
        hp_str = "  ".join(
            f"{k}={round(v, 4) if isinstance(v, float) else v}"
            for k, v in hp.items()
        )
        print(f"{rank+1:>4}  {-y[0].item():>8.4f}  {-y[1].item():>8.4f}"
              f"  {-y[2].item():>9.1f}  {hp_str}")

    # ── Scatter plots (3 pairwise objective planes) ───────────────────────────
    Y_np   = (-Y_obs).cpu().numpy()   # (n, 3): dist, dup, runtime
    pm_np  = pareto_mask.numpy()
    pairs  = [
        (0, 2, "dist_to_needles", "runtime (s)"),
        (0, 1, "dist_to_needles", "dup_fraction"),
        (1, 2, "dup_fraction",    "runtime (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MOBO Pareto front  (★ = Pareto-optimal)", fontsize=12)
    for ax, (ix, iy, xl, yl) in zip(axes, pairs):
        ax.scatter(Y_np[~pm_np, ix], Y_np[~pm_np, iy],
                   c="steelblue", alpha=0.6, edgecolors="k", linewidths=0.3, label="dominated")
        ax.scatter(Y_np[pm_np, ix], Y_np[pm_np, iy],
                   marker="*", s=220, c="gold", zorder=5,
                   edgecolors="k", linewidths=0.5, label="Pareto")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "pareto_front.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\nPareto plot saved to {fig_path}")
    plt.show(block=True)
    return order


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("ZoMBI-Hop Hyperparameter Optimisation (MOBO)")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    print("\nSelect test functions (any combination, e.g. '123', '2 3', Enter = all):")
    print("  1. RF surrogate on campaign1a.csv  (interactive extrema selection)")
    print("  2. Gaussian mixture on simplex     (3 known peaks near each corner)")
    print("  3. Ackley in ILR space             (minimum at centroid)")
    raw_choice = input("> ").strip()
    selected = set(raw_choice.replace(" ", "").replace(",", "")) or {"1", "2", "3"}
    if not selected.issubset({"1", "2", "3"}):
        sys.exit(f"Invalid selection '{raw_choice}' — use digits 1, 2, 3.")
    print(f"  Using functions: {sorted(selected)}")

    plt.ion()
    test_fns: list[TestFn] = []

    # ── Function 1: RF surrogate ──────────────────────────────────────────────
    if "1" in selected:
        csv_candidates = [
            os.path.join(script_dir, "..", "interactive_testing", "data", "campaign1a.csv"),
            os.path.join(script_dir, "..", "..", "interactive_testing", "data", "campaign1a.csv"),
        ]
        csv_path = next(
            (os.path.normpath(p) for p in csv_candidates if os.path.exists(p)), None
        )
        if csv_path is None:
            sys.exit(f"CSV not found. Tried:\n" + "\n".join(f"  {os.path.normpath(p)}" for p in csv_candidates))
        print(f"\n[RF] Loading {csv_path} …")
        df = pd.read_csv(csv_path).dropna(
            subset=["FAPbI3", "MAPbI3", "MAPbBr3", "Objective"]
        )
        X_data = df[["FAPbI3", "MAPbI3", "MAPbBr3"]].values.astype(float)
        X_data /= X_data.sum(axis=1, keepdims=True)
        y_data = df["Objective"].values.astype(float)
        print(f"  {X_data.shape[0]} samples.  Training RF ({RF_N_ESTIMATORS} trees) …")
        rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
        rf.fit(X_data, y_data)
        print(f"  Train R² = {rf.score(X_data, y_data):.4f}")
        grid_pts  = ternary_grid(TERNARY_GRID_N)
        grid_vals = rf.predict(grid_pts)
        raw_mm = input("  Maximize or minimize RF?  [max/min, default min]: ").strip().lower()
        rf_maximize = raw_mm in ("max", "x", "maximize")
        goal = "maxima" if rf_maximize else "minima"
        print(f"  Click near reference {goal}, then Enter / Q.")
        picker  = ExtremaPicker(rf, grid_pts, grid_vals, maximize=rf_maximize)
        extrema = picker.run()
        if not extrema:
            print("  No extrema selected — using centroid as fallback.")
            extrema = [(np.array([1/3, 1/3, 1/3]), 0.0)]
        rf_optima = [x for x, _ in extrema]
        rf_fn     = lambda x, _rf=rf: float(_rf.predict(x.reshape(1, -1))[0])
        test_fns.append((rf_fn, rf_optima, max(2, len(rf_optima) * 2), rf_maximize, "RF"))
        print(f"  RF ready: {len(rf_optima)} reference {goal}")

    # ── Function 2: Gaussian mixture ─────────────────────────────────────────
    if "2" in selected:
        peaks  = [[0.70, 0.15, 0.15], [0.15, 0.70, 0.15], [0.15, 0.15, 0.70]]
        gm     = GaussianMixture(peaks, sigma=0.12)
        test_fns.append((gm, gm.true_optima, max(2, len(gm.true_optima) * 2),
                         gm.maximize, "GaussianMixture"))
        print(f"\n[GaussianMixture] 3 peaks at {[list(np.round(p,2)) for p in gm.peaks]}")

    # ── Function 3: Ackley ILR ────────────────────────────────────────────────
    if "3" in selected:
        ackley = AckleyILR()
        test_fns.append((ackley, ackley.true_optima, 2, ackley.maximize, "AckleyILR"))
        print(f"\n[AckleyILR] minimum at centroid {AckleyILR.CENTROID}")

    # ── MOBO ──────────────────────────────────────────────────────────────────
    X_obs, Y_obs = run_mobo(test_fns, script_dir)

    # ── Results ───────────────────────────────────────────────────────────────
    pareto_order = show_pareto(X_obs, Y_obs, save_dir=script_dir)

    results_path = os.path.join(script_dir, "mobo_results.pt")
    torch.save(
        {"X_obs": X_obs, "Y_obs": Y_obs, "hparam_names": HPARAM_NAMES},
        results_path,
    )
    print(f"\nAll results saved to {results_path}")

    if len(pareto_order) > 0:
        best_idx = pareto_order[0]   # top Pareto by dist
        best_hp  = norm_to_hparams(X_obs[best_idx])
        print("\nBest config (min dist_to_needles on Pareto front):")
        for k, v in best_hp.items():
            print(f"  {k}: {round(v, 6) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()
