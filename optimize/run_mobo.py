"""
optimize/run_mobo.py
====================
Multi-objective Bayesian optimisation (MOBO) of ZoMBI-Hop hyperparameters,
evaluated on the Random-Forest surrogate built from campaign1a.csv.

Three objectives (all minimised):
  1. dist_to_needles    – mean greedy distance from discovered needles to the
                          nearest true optimum (no-repeat matching; unmatched
                          true optima incur UNMATCHED_PENALTY)
  2. dup_fraction       – fraction of sampled points whose nearest neighbour
                          in input space is within noise/2
  3. runtime            – wall-clock seconds for the (timed) ZoMBI run only
                          (per-iteration plotting + video are rendered AFTER
                          the timed region so they don't pollute this metric)

MOBO engine: qLogNEHVI (BoTorch, maximises negated objectives).

Run layout
----------
Each run creates a folder ``runs/mobo_DD_MM_HH_MM/`` (start date/time, military
clock) containing:
  • mobo_progress.json / mobo_results.json / mobo_results.pt  – running summary
  • pareto_front.png                                          – on exit
  • trial_<n>/                                                – one per trial
        ├─ trial.json                 (phase / pareto / metrics / hparams)
        ├─ points.csv                 (sample_idx, FA, MA, Br, Y, penalized,
        │                              activation, zoom)
        ├─ needles.csv                (needle_idx, FA, MA, Br, value,
        │                              median_value, activation, zoom,
        │                              iteration, dist_to_centre)
        ├─ metrics_over_time.csv      (iteration, dist_to_needles, dup_fraction,
        │                              pct_matched, avg_pairwise_dist)
        ├─ dist_from_centre.png
        ├─ line_length_hist.png
        ├─ hparam_edge_proximity.png
        ├─ plots/iter_0000.png …      (one frame per iteration)
        └─ zombihop_timelapse.mp4

Each trial runs ZoMBI-Hop until its wall-clock budget (TIME_LIMIT_HOURS) expires.
The number of trials is unbounded — the MOBO loop runs Sobol init then BO
indefinitely until you press Ctrl+C.

Usage
-----
  conda activate zombi-hop
  python optimize/run_mobo.py            # TIME_LIMIT_HOURS per trial
  python optimize/run_mobo.py --resume   # crawl every runs/mobo_*/mobo_progress.json,
                                         #   collect all (X,Y) pairs, and seed a NEW
                                         #   runs/mobo_* run from the full landscape
  python optimize/run_mobo.py --make-videos            # newest run
  python optimize/run_mobo.py --make-videos <run_dir>  # specific run
  python optimize/run_mobo.py --make-videos <run_dir> --force-videos   # rebuild all
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import shutil
import sys
import time
import traceback
import warnings

import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize as sp_minimize
from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use("TkAgg")              # interactive backend for the extrema picker
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon as MplPolygon

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
from src.utils.simplex import (
    Ellipsoid,
    random_simplex,
    composition_to_ilr,
    ilr_to_composition,
    proj_simplex,
)

# ─── Global config ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64

NOISE_LEVEL     = 0.01
NOISE_LEVEL_ILR = 0.03
NUM_EXPERIMENTS = 24
NUM_LINES       = 10
N_INIT_LINES    = 2

# MOBO settings
N_INIT_TRIALS    = 8       # Sobol initial designs before BO begins
N_MOBO_RESTARTS  = 10
N_MOBO_SAMPLES   = 512

# Per-trial wall-clock budget (hours) passed to ZoMBIHop.run(time_limit_hours=…).
TIME_LIMIT_HOURS = 0.4

# pct_matched: a true optimum counts as "found" if a needle is within this
# Euclidean (composition L2) radius of it.
MATCH_RADIUS = 0.05

# Resumability: abort the overnight run only after this many trials fail back-to-back
# (guards against a runaway loop on a systemic failure; transient failures just retry).
MAX_CONSEC_FAIL = 5

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

# Video timelapse target
VIDEO_TARGET_DURATION_S = 30.0
VIDEO_MIN_FPS           = 1.0
VIDEO_MAX_FPS           = 60.0

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


def hparams_to_norm(hparams: dict) -> torch.Tensor:
    """Inverse of norm_to_hparams: map a stored hyperparameter dict → [0,1] vector.

    Used when crawling past runs' mobo_progress.json (which stores unnormalised
    hparams) to reconstruct the normalised MOBO design points for resume.
    """
    v = torch.zeros(N_HPARAMS, dtype=DTYPE)
    for i, name in enumerate(HPARAM_NAMES):
        lo, hi, tfm = HPARAM_SPACE[name]
        val = float(hparams[name])
        if tfm == "log":
            u = (math.log(val) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:  # "int" or "linear"
            u = (val - lo) / (hi - lo)
        v[i] = min(1.0, max(0.0, u))
    return v


# ─── Crash-safe persistence helpers ─────────────────────────────────────────────

def _atomic_write_text(path: str, text: str) -> None:
    """Write text to a temp file then os.replace — never leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _atomic_torch_save(obj, path: str) -> None:
    """torch.save via temp file + os.replace, keeping the previous file as `.bak`."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    if os.path.exists(path):
        try:
            shutil.copy2(path, path + ".bak")
        except Exception:
            pass
    os.replace(tmp, path)


def _log_error(run_dir: str, trial_num: int, exc: Exception) -> None:
    """Append a failing trial's traceback to errors.log (best-effort)."""
    try:
        with open(os.path.join(run_dir, "errors.log"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] "
                    f"trial {trial_num}: {exc}\n")
            f.write(traceback.format_exc() + "\n")
    except Exception:
        pass


# ─── Deterministic RF surrogate (rebuilt identically on resume) ─────────────────

def build_rf_and_grid(csv_path: str):
    """Train the RF surrogate from campaign1a.csv and build the ternary grid.

    Deterministic (fixed random_state / tree count), so a resumed run rebuilds an
    identical surrogate without re-prompting the interactive extrema picker.
    Returns (rf, rf_fn, grid_pts, grid_vals).
    """
    df = pd.read_csv(csv_path).dropna(subset=["FAPbI3", "MAPbI3", "MAPbBr3", "Objective"])
    X_data = df[["FAPbI3", "MAPbI3", "MAPbBr3"]].values.astype(float)
    X_data /= X_data.sum(axis=1, keepdims=True)
    y_data = df["Objective"].values.astype(float)
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X_data, y_data)
    grid_pts  = ternary_grid(TERNARY_GRID_N)
    grid_vals = rf.predict(grid_pts)
    rf_fn = lambda x, _rf=rf: float(_rf.predict(x.reshape(1, -1))[0])
    return rf, rf_fn, grid_pts, grid_vals


# ─── Run-config persistence + resume ────────────────────────────────────────────

def write_run_config(run_dir, maximize, csv_path, true_optima) -> None:
    """Persist the static run state needed for a fully non-interactive resume."""
    cfg = {
        "maximize":      bool(maximize),
        "csv_path":      os.path.abspath(csv_path),
        "true_optima":   [list(map(float, np.asarray(t).ravel())) for t in true_optima],
        "n_init_trials": N_INIT_TRIALS,
        "hparam_names":  HPARAM_NAMES,
        "created":       datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write_text(os.path.join(run_dir, "run_config.json"), json.dumps(cfg, indent=2))


def resolve_run_dir(arg: str, runs_dir: str) -> str:
    """Resolve a --make-videos argument to a run directory under runs/.

    arg == "__latest__"  → newest mobo_* folder under runs_dir.
    otherwise            → the given path (absolute, cwd-relative, or runs-relative).
    """
    if arg == "__latest__":
        cands = [c for c in glob.glob(os.path.join(runs_dir, "mobo_*")) if os.path.isdir(c)]
        if not cands:
            sys.exit(f"No mobo_* run found under {runs_dir}.")
        return max(cands, key=os.path.getmtime)
    for cand in (arg, os.path.join(runs_dir, arg), os.path.join(os.path.dirname(runs_dir), arg)):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    sys.exit(f"Run directory not found: {arg}")


def load_latest_run_config(runs_dir: str) -> dict:
    """Return the run_config.json from the most recently created runs/mobo_* run.

    Used on resume to reuse the static config (min/max, csv_path,
    reference optima) without re-prompting interactively.
    """
    cands = glob.glob(os.path.join(runs_dir, "mobo_*", "run_config.json"))
    if not cands:
        sys.exit(f"No runs/mobo_*/run_config.json found under {runs_dir} — nothing to resume.")
    latest = max(cands, key=os.path.getmtime)
    with open(latest) as f:
        cfg = json.load(f)
    print(f"  [resume] reusing config from {os.path.basename(os.path.dirname(latest))}")
    return cfg


def collect_all_observations(runs_dir: str):
    """Crawl every runs/mobo_*/mobo_progress.json and collect all (X_norm, Y) pairs.

    X = normalised hyperparameter vector (inverted from the stored hparams),
    Y = (-dist_to_needles, -dup_fraction, -runtime_s)  [the maximised objectives].

    Each run's progress.json records only its own trials, so the union across all
    runs has no double-counting.  Trials whose hparam keys don't cover the current
    HPARAM_SPACE are skipped (stale hyperparameter set).
    Returns (X_obs, Y_obs, n_runs).
    """
    X_obs, Y_obs = [], []
    n_runs = 0
    for path in sorted(glob.glob(os.path.join(runs_dir, "mobo_*", "mobo_progress.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  [collect] {path} unreadable ({exc}); skipping.")
            continue
        used = 0
        for t in data.get("trials", []):
            hp, m = t.get("hparams", {}), t.get("metrics", {})
            if not all(name in hp for name in HPARAM_NAMES):
                continue  # stale / different hyperparameter set
            try:
                x = hparams_to_norm(hp)
                y = torch.tensor([-float(m["dist_to_needles"]),
                                  -float(m["dup_fraction"]),
                                  -float(m["runtime_s"])], dtype=DTYPE)
            except (KeyError, ValueError, TypeError):
                continue
            X_obs.append(x)
            Y_obs.append(y)
            used += 1
        if used:
            n_runs += 1
            print(f"  [collect] {os.path.basename(os.path.dirname(path))}: {used} trial(s)")
    return X_obs, Y_obs, n_runs


def load_or_make_sobol(run_dir: str, bounds: torch.Tensor) -> torch.Tensor:
    """Load the persisted Sobol init design, or draw + persist it on first use.

    Persisting keeps the 8 init trials identical across restarts.
    """
    path = os.path.join(run_dir, "sobol_design.pt")
    if os.path.exists(path):
        try:
            return torch.load(path, map_location="cpu").to(device=DEVICE, dtype=DTYPE)
        except Exception as exc:
            print(f"  [resume] sobol_design.pt unreadable ({exc}); redrawing.")
    X_sobol = draw_sobol_samples(bounds=bounds, n=N_INIT_TRIALS, q=1).squeeze(1)
    _atomic_torch_save(X_sobol.cpu(), path)
    return X_sobol


# ─── Ternary helpers (RF interactive picker + plotting) ────────────────────────

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
    """Greedy no-repeat matching; unmatched true optima add UNMATCHED_PENALTY."""
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


def metric_pct_matched(
    discovered: np.ndarray,
    true_optima: list[np.ndarray],
    radius: float = MATCH_RADIUS,
) -> float:
    """Percentage of TRUE optima that have at least one needle within `radius`."""
    if not true_optima:
        return 100.0
    if len(discovered) == 0:
        return 0.0
    disc = np.asarray(discovered, dtype=float)
    matched = 0
    for t in true_optima:
        dmin = float(np.linalg.norm(disc - np.asarray(t), axis=1).min())
        if dmin <= radius:
            matched += 1
    return 100.0 * matched / len(true_optima)


def metric_avg_pairwise_dist(discovered: np.ndarray) -> float:
    """Average pairwise Euclidean distance between discovered needles."""
    disc = np.asarray(discovered, dtype=float)
    n = len(disc)
    if n < 2:
        return 0.0
    diff  = disc[:, None, :] - disc[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    return float(dists[iu].mean())


# ─── ZoMBI sim-objective + LineBO wrapper ──────────────────────────────────────

def make_sim_obj(fn_callable, device, dtype, *, maximize: bool):
    """Wrap f: (d,) np.ndarray → float as a ZoMBI sim_objective."""

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


def make_linebo_wrapper(sim_obj, dim, num_lines, device, dtype, plot_state: dict):
    """LineBO wrapper that also stashes the top-2 ranked lines into plot_state."""
    linebo = LineBO(sim_obj, dim,
                   num_points_per_line=100, num_lines=num_lines, device=str(device))

    def wrapper(x_tell, bounds, acq_fn):
        x_left_r, x_right_r = linebo.ranked_line_endpoints(x_tell, bounds, acq_fn)
        n_valid = x_left_r.shape[0]
        plot_state["line_0"] = (
            (x_left_r[0].cpu().numpy(), x_right_r[0].cpu().numpy()) if n_valid > 0 else None
        )
        plot_state["line_1"] = (
            (x_left_r[1].cpu().numpy(), x_right_r[1].cpu().numpy()) if n_valid > 1 else None
        )
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


def _gen_init_data(fn_callable, maximize: bool):
    """Generate N_INIT_LINES random simplex lines; return (X_a, X_e, Y)."""
    x_a_list, x_e_list, y_list = [], [], []
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
    if not x_a_list:
        raise RuntimeError("Could not generate any initial simplex lines.")
    return (
        torch.cat(x_a_list, dim=0),
        torch.cat(x_e_list, dim=0),
        torch.cat(y_list, dim=0).reshape(-1, 1),
    )


# ─── Per-iteration plotting (rendered AFTER the timed run) ──────────────────────

def _draw_bounds_region(ax, bounds, n_sample: int = 5000) -> None:
    """Draw the trust-region (tensor bounds) as a dashed-red convex hull."""
    try:
        if isinstance(bounds, torch.Tensor) and bounds.shape[0] == 2:
            lo, hi = bounds[0], bounds[1]
            samp = random_simplex(n_sample, lo, hi, device=str(lo.device), torch_dtype=lo.dtype)
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
        ax.plot(verts_c[:, 0], verts_c[:, 1], "--", color="red", lw=2.0,
                alpha=0.75, zorder=5, label="Trust bounds")
    except Exception:
        pass


def _draw_needle_ellipsoid(ax, needle_x, M, B) -> None:
    """Red star for the needle + faded purple penalisation ellipsoid."""
    xy = comp_to_xy(needle_x.reshape(1, 3))
    ax.scatter(xy[0, 0], xy[0, 1], marker="*", s=280, c="red",
               zorder=9, edgecolors="darkred", linewidths=1.0)
    if M is None:
        return
    try:
        d = needle_x.shape[0]
        M_np = M.cpu().numpy()
        eigvals, eigvecs = np.linalg.eigh(M_np)
        eigvals = np.maximum(eigvals, 1e-12)
        angles = np.linspace(0, 2 * np.pi, 200)
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        u_ell = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ circle.T).T
        if B is not None:
            B_np = B.cpu().numpy()
            ell_pts = needle_x.reshape(1, d) + (B_np @ u_ell.T).T
        else:
            needle_t = torch.tensor(needle_x, dtype=torch.float64)
            needle_ilr = composition_to_ilr(needle_t.unsqueeze(0)).squeeze(0).cpu().numpy()
            z_ell = needle_ilr + u_ell
            ell_pts = ilr_to_composition(torch.tensor(z_ell, dtype=torch.float64), d).cpu().numpy()
        ell_pts = np.clip(ell_pts, 0, 1)
        s = ell_pts.sum(axis=1, keepdims=True)
        ell_pts = ell_pts / np.where(s < 1e-9, 1.0, s)
        ell_xy = comp_to_xy(ell_pts)
        poly = MplPolygon(ell_xy, closed=True, alpha=0.15, facecolor="purple",
                          edgecolor="purple", linewidth=0.9, zorder=6)
        ax.add_patch(poly)
        ax.plot(ell_xy[:, 0], ell_xy[:, 1], c="purple", lw=0.7, alpha=0.45, zorder=6)
    except Exception:
        pass


def render_frame(payload: dict, grid_pts, grid_vals, true_optima, maximize: bool,
                 out_path: str) -> None:
    """Render one two-panel ternary iteration figure to out_path (no GUI)."""
    fig = Figure(figsize=(16, 6.8))
    FigureCanvasAgg(fig)
    ax_ref, ax_exp = fig.subplots(1, 2)
    fig.suptitle(f"ZoMBI-Hop MOBO trial — iteration {payload['iter_num']}", fontsize=13)

    # ── Left: RF reference ──
    draw_ternary_frame(ax_ref)
    ax_ref.set_title("Reference: RF landscape", fontsize=11)
    gxy = comp_to_xy(grid_pts)
    sc_ref = ax_ref.scatter(gxy[:, 0], gxy[:, 1], c=grid_vals, cmap="viridis",
                            s=6, alpha=0.72, zorder=2, rasterized=True)
    fig.colorbar(sc_ref, ax=ax_ref, label="RF Objective", fraction=0.046, pad=0.04)
    ref_lbl = "True maxima" if maximize else "True minima"
    if true_optima:
        mxy = comp_to_xy(np.array(true_optima))
        ax_ref.scatter(mxy[:, 0], mxy[:, 1], marker="*", s=360, c="blue",
                       zorder=11, edgecolors="navy", linewidths=1.3, label=ref_lbl)
        ax_ref.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ── Right: ZoMBI exploration ──
    draw_ternary_frame(ax_exp)
    legend_handles = []
    pared_X, pared_Y = payload.get("pared_X"), payload.get("pared_Y")
    if pared_X is not None and len(pared_X) > 0:
        n = len(pared_Y)
        alphas = np.linspace(0.15, 0.92, n) if n > 1 else np.array([0.92])
        xy_pts = comp_to_xy(pared_X)
        y_lo, y_hi = float(pared_Y.min()), float(pared_Y.max())
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-9
        for i in range(n):
            ax_exp.scatter(xy_pts[i, 0], xy_pts[i, 1], c=[[pared_Y[i]]], cmap="viridis",
                           vmin=y_lo, vmax=y_hi, s=22, alpha=float(alphas[i]),
                           zorder=3, edgecolors="gray", linewidths=0.2)
        ax_exp.set_title(f"ZoMBI-Hop exploration  ({n} pared pts)", fontsize=11)
    else:
        ax_exp.set_title("ZoMBI-Hop exploration", fontsize=11)

    if payload.get("bounds") is not None:
        _draw_bounds_region(ax_exp, payload["bounds"])

    needles = payload.get("needles")
    if needles is not None and len(needles) > 0:
        M_list = payload.get("needle_M_list") or [None] * len(needles)
        B = payload.get("needle_B")
        for i, nx in enumerate(needles):
            Mi = M_list[i] if i < len(M_list) else None
            _draw_needle_ellipsoid(ax_exp, nx, Mi, B)

    if payload.get("line_0") is not None:
        ll = comp_to_xy(np.array(payload["line_0"]))
        (h0,) = ax_exp.plot(ll[:, 0], ll[:, 1], "-", color="orange", lw=2.5,
                            alpha=0.90, zorder=7, label="LineBO (main)")
        legend_handles.append(h0)
    if payload.get("line_1") is not None:
        ll = comp_to_xy(np.array(payload["line_1"]))
        (h1,) = ax_exp.plot(ll[:, 0], ll[:, 1], ":", color="cornflowerblue", lw=2.2,
                            alpha=0.85, zorder=7, label="LineBO (cache)")
        legend_handles.append(h1)

    if true_optima:
        mxy = comp_to_xy(np.array(true_optima))
        h_min = ax_exp.scatter(mxy[:, 0], mxy[:, 1], marker="*", s=360, c="blue",
                               zorder=11, edgecolors="navy", linewidths=1.3, label=ref_lbl)
        legend_handles.append(h_min)

    if legend_handles:
        ax_exp.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)

    # NB: no bbox_inches="tight" — that yields variable / odd pixel dimensions
    # across frames, which libx264 rejects (frames must share an even-sized
    # canvas).  A fixed figsize × dpi gives a constant even-dimension canvas.
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    fig.clear()


def make_video_from_dir(plots_dir: str, out_path: str) -> bool:
    """Compile iter_*.png frames in plots_dir into a ~30s MP4 at out_path.

    Returns True on success. Tries imageio+ffmpeg (h264) then OpenCV; both paths
    verify the output is non-empty, since ffmpeg/cv2 can fail without raising.
    """
    frames = sorted(glob.glob(os.path.join(plots_dir, "iter_*.png")))
    if not frames:
        print(f"    [video] no frames in {plots_dir} — skipping.")
        return False
    fps = max(VIDEO_MIN_FPS, min(VIDEO_MAX_FPS, len(frames) / VIDEO_TARGET_DURATION_S))

    def _even(img):
        """Crop to even height/width (libx264 requirement)."""
        h, w = img.shape[:2]
        return img[: h - (h % 2), : w - (w % 2)]

    def _ok() -> bool:
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0

    # Prefer imageio + imageio-ffmpeg (h264); fall back to OpenCV.
    try:
        import imageio.v2 as iio
        from PIL import Image as PILImage
        imgs = [iio.imread(f)[:, :, :3] for f in frames]
        h, w = imgs[0].shape[:2]
        fixed = []
        for img in imgs:
            if img.shape[:2] != (h, w):
                img = np.array(PILImage.fromarray(img).resize((w, h), PILImage.LANCZOS))
            fixed.append(_even(img))
        iio.mimwrite(out_path, fixed, fps=fps, codec="libx264", macro_block_size=None)
        if not _ok():
            raise RuntimeError("imageio/ffmpeg produced an empty file")
        print(f"    [video] {out_path}  ({len(frames)} frames @ {fps:.2f} fps)")
        return True
    except Exception as exc:
        print(f"    [video] imageio failed ({exc}); trying OpenCV …")

    try:
        import cv2
        first = _even(cv2.imread(frames[0]))
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open (codec unavailable?)")
        for path in frames:
            img = cv2.imread(path)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
        writer.release()
        if not _ok():
            raise RuntimeError("OpenCV produced an empty file")
        print(f"    [video] {out_path}  ({len(frames)} frames @ {fps:.2f} fps, OpenCV)")
        return True
    except Exception as exc:
        print(f"    [video] OpenCV failed too ({exc}) — no video written.")
        return False


# ─── Per-trial artifact writers ────────────────────────────────────────────────

def _activation_zoom_per_point(n_points: int, snap_records: list[tuple]) -> tuple:
    """Map each stored point index → (activation, zoom) from snapshot records.

    snap_records is the chronological list of (cumulative_n_points, activation,
    zoom) captured at every take_snapshot.  Points in [prev_n, n) were added by
    that snapshot's objective call; init points fall into the first record.
    """
    act = np.zeros(n_points, dtype=int)
    zm  = np.zeros(n_points, dtype=int)
    prev = 0
    for (n, a, z) in snap_records:
        n = min(int(n), n_points)
        if n > prev:
            act[prev:n] = a
            zm[prev:n] = z
            prev = n
    if prev < n_points and snap_records:   # tail safety
        act[prev:] = snap_records[-1][1]
        zm[prev:]  = snap_records[-1][2]
    return act, zm


def write_points_csv(path: str, dh, snap_records: list[tuple]) -> None:
    X = dh.X_all_actual.detach().cpu().numpy()
    Y = dh.Y_all.detach().cpu().numpy().ravel()
    n = X.shape[0]
    mask = dh.get_penalty_mask()                 # True = NOT penalized
    penalized = (~mask.detach().cpu().numpy()) if mask is not None else np.zeros(n, bool)
    act, zm = _activation_zoom_per_point(n, snap_records)
    df = pd.DataFrame({
        "sample_idx": np.arange(n),
        "FA": X[:, 0], "MA": X[:, 1], "Br": X[:, 2],
        "Y": Y,
        "penalized": penalized.astype(int),
        "activation": act,
        "zoom": zm,
    })
    df.to_csv(path, index=False)


def write_needles_csv(path: str, dh) -> None:
    centroid = np.full(3, 1.0 / 3)
    rows = []
    for i, r in enumerate(dh.get_all_needle_results()):
        pt = r["point"].detach().cpu().numpy().ravel()
        mv = r.get("median_value")
        rows.append({
            "needle_idx": i,
            "FA": pt[0], "MA": pt[1], "Br": pt[2],
            "value": r.get("value"),
            "median_value": (None if mv is None or (isinstance(mv, float) and math.isnan(mv)) else mv),
            "activation": r.get("activation"),
            "zoom": r.get("zoom"),
            "iteration": r.get("iteration"),
            "dist_to_centre": float(np.linalg.norm(pt - centroid)),
        })
    cols = ["needle_idx", "FA", "MA", "Br", "value", "median_value",
            "activation", "zoom", "iteration", "dist_to_centre"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def write_metrics_over_time_csv(path: str, payloads: list[dict], X_all: np.ndarray,
                                true_optima: list[np.ndarray]) -> None:
    thr = NOISE_LEVEL / 2.0
    rows = []
    for p in payloads:
        needles = p.get("needles")
        disc = needles if needles is not None else np.empty((0, 3))
        n_before = p.get("n_points_before", len(X_all))
        X_upto = X_all[:n_before] if n_before > 0 else np.empty((0, 3))
        rows.append({
            "iteration": p["iter_num"],
            "dist_to_needles":  round(metric_dist_to_needles(disc, true_optima), 6),
            "dup_fraction":     round(metric_dup_fraction(X_upto, thr), 6),
            "pct_matched":      round(metric_pct_matched(disc, true_optima), 4),
            "avg_pairwise_dist":round(metric_avg_pairwise_dist(disc), 6),
        })
    cols = ["iteration", "dist_to_needles", "dup_fraction", "pct_matched", "avg_pairwise_dist"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def plot_dist_from_centre(path: str, dh, maximize: bool) -> None:
    """Distance-from-simplex-centre scatter (mirrors interface/app.py)."""
    fig = Figure(figsize=(7, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    X = dh.X_all_actual.detach().cpu().numpy()
    Y = dh.Y_all.detach().cpu().numpy().ravel()
    if X.shape[0] == 0:
        ax.text(0.5, 0.5, "no data", ha="center"); fig.savefig(path); fig.clear(); return
    centroid = np.full(X.shape[1], 1.0 / X.shape[1])
    dists = np.linalg.norm(X - centroid, axis=1)
    idx = np.arange(len(Y))
    sc = ax.scatter(dists, Y, c=idx, cmap="viridis", s=14, alpha=0.7, zorder=3, label="samples")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("sample index", fontsize=8)
    needles = dh.get_all_needle_locations()
    nvals = dh.get_all_needle_vals()
    if needles is not None and needles.shape[0] > 0:
        nd = np.linalg.norm(needles.detach().cpu().numpy() - centroid, axis=1)
        nv = nvals.detach().cpu().numpy().ravel()
        ax.scatter(nd, nv, marker="*", s=220, color="crimson", edgecolors="darkred",
                   lw=0.8, zorder=5, label="needle")
    ax.set_xlabel("‖x − centroid‖₂")
    ax.set_ylabel("Objective Y" + ("" if maximize else "  (ZoMBI-internal, = −RF)"))
    ax.set_title("Distance from simplex centre", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.clear()


def plot_line_length_hist(path: str, payloads: list[dict]) -> None:
    """Histogram of LineBO main-line lengths (composition L2)."""
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
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.clear()


def plot_hparam_edge_proximity(path: str, x_norm: torch.Tensor) -> None:
    """How close each chosen hyperparameter is to its nearest search-space edge.

    Values are normalised to [0, 1] internally, so proximity = min(v, 1−v):
    0 = sitting on lo or hi, 0.5 = dead centre of the range.
    """
    v = x_norm.clamp(0.0, 1.0).cpu().numpy()
    prox = np.minimum(v, 1.0 - v)
    order = np.argsort(prox)
    names = [HPARAM_NAMES[i] for i in order]
    vals = prox[order]
    colors = ["crimson" if p < 0.05 else "steelblue" for p in vals]
    fig = Figure(figsize=(8, 7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.barh(range(len(names)), vals, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 0.5)
    ax.axvline(0.05, color="crimson", ls=":", lw=1.0, alpha=0.6, label="edge zone (<0.05)")
    ax.set_xlabel("distance to nearest search-space edge  (normalised; 0 = at edge, 0.5 = centre)")
    ax.set_title("Hyperparameter edge proximity", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    fig.clear()


def write_trial_json(path: str, trial_num: int, phase: str, is_pareto: bool,
                     metrics: dict, hparams: dict) -> None:
    obj = {
        "trial": trial_num,
        "phase": phase,
        "pareto": bool(is_pareto),
        "metrics": {
            "dist_to_needles": round(metrics["dist"], 6),
            "dup_fraction":    round(metrics["dup"], 6),
            "runtime_s":       round(metrics["runtime"], 3),
        },
        "hparams": {
            k: (round(v, 8) if isinstance(v, float) else v) for k, v in hparams.items()
        },
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ─── Single trial: run ZoMBI on RF + write all artifacts ───────────────────────

def run_single_trial(
    hparams: dict,
    rf_fn,
    true_optima: list[np.ndarray],
    grid_pts: np.ndarray,
    grid_vals: np.ndarray,
    maximize: bool,
    trial_dir: str,
) -> dict:
    """Run one time-limited ZoMBI trial on the RF surrogate, then write all
    per-trial artifacts.  Returns {"dist", "dup", "runtime", "payloads", ...}."""
    # Wipe any partial folder left by a crashed/interrupted attempt at this trial
    # number so resumed runs never mix stale frames with fresh ones.
    if os.path.isdir(trial_dir):
        shutil.rmtree(trial_dir, ignore_errors=True)
    os.makedirs(trial_dir, exist_ok=True)

    plot_state: dict = {"line_0": None, "line_1": None}
    payloads: list[dict] = []
    snap_records: list[tuple] = []
    call_counter = [0]
    dh_ref = [None]

    sim_obj = make_sim_obj(rf_fn, DEVICE, DTYPE, maximize=maximize)
    inner   = make_linebo_wrapper(sim_obj, 3, NUM_LINES, DEVICE, DTYPE, plot_state)

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
        return x_req, x_act, y

    try:
        X_a, X_e, Y = _gen_init_data(rf_fn, maximize)
    except RuntimeError as exc:
        print(f"    [trial] init failed: {exc}")
        return {"dist": UNMATCHED_PENALTY, "dup": 1.0, "runtime": 0.0, "payloads": []}

    # checkpoint_dir=None → no disk snapshots (keeps runtime_s clean); we still
    # capture activation/zoom in-memory because take_snapshot updates the
    # current_* counters before the save_enabled early-return.
    optimizer = ZoMBIHop(
        objective=obj_wrapper,
        X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
        **ZOMBI_FIXED, **hparams,
        device=str(DEVICE), dtype=DTYPE,
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
        optimizer.run(max_activations=float("inf"), time_limit_hours=TIME_LIMIT_HOURS)
    except Exception as exc:
        print(f"    [trial] ZoMBI crashed: {exc}")
    runtime = time.time() - t0

    # ── Trial-level metrics ──
    needle_t   = dh.get_all_needle_locations()
    discovered = needle_t.detach().cpu().numpy() if needle_t.numel() > 0 else np.empty((0, 3))
    X_all_np   = dh.X_all_actual.detach().cpu().numpy() if dh.X_all_actual is not None else np.empty((0, 3))
    dist = metric_dist_to_needles(discovered, true_optima)
    dup  = metric_dup_fraction(X_all_np, NOISE_LEVEL / 2.0)
    print(f"    [trial]  iters={call_counter[0]}  dist={dist:.4f}  dup={dup:.4f}"
          f"  t={runtime:.1f}s  needles={len(discovered)}/{len(true_optima)}")

    # ── CSV / table artifacts ──
    try:
        write_points_csv(os.path.join(trial_dir, "points.csv"), dh, snap_records)
        write_needles_csv(os.path.join(trial_dir, "needles.csv"), dh)
        write_metrics_over_time_csv(
            os.path.join(trial_dir, "metrics_over_time.csv"), payloads, X_all_np, true_optima)
    except Exception as exc:
        print(f"    [trial] CSV write failed: {exc}")

    # ── Static plots ──
    try:
        plot_dist_from_centre(os.path.join(trial_dir, "dist_from_centre.png"), dh, maximize)
        plot_line_length_hist(os.path.join(trial_dir, "line_length_hist.png"), payloads)
    except Exception as exc:
        print(f"    [trial] static plot failed: {exc}")

    # ── Per-iteration frames + video (rendered AFTER timing) ──
    plots_dir = os.path.join(trial_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"    [trial] rendering {len(payloads)} frames …", flush=True)
    for p in payloads:
        try:
            render_frame(p, grid_pts, grid_vals, true_optima, maximize,
                         os.path.join(plots_dir, f"iter_{p['iter_num'] - 1:04d}.png"))
        except Exception as exc:
            print(f"    [trial] frame {p['iter_num']} failed: {exc}")

    return {"dist": dist, "dup": dup, "runtime": runtime, "payloads": payloads}


def regenerate_videos(run_dir: str, force: bool = False) -> None:
    """Rebuild zombihop_timelapse.mp4 for every trial_* folder from its frames.

    Skips trials that already have a non-empty video unless force=True.
    """
    trial_dirs = sorted(
        glob.glob(os.path.join(run_dir, "trial_*")),
        key=lambda p: int(p.split("_")[-1]) if p.split("_")[-1].isdigit() else 0,
    )
    if not trial_dirs:
        print(f"No trial_* folders found in {run_dir}")
        return
    n_ok = n_skip = n_fail = 0
    for tdir in trial_dirs:
        plots_dir = os.path.join(tdir, "plots")
        out_path  = os.path.join(tdir, "zombihop_timelapse.mp4")
        name = os.path.basename(tdir)
        if not os.path.isdir(plots_dir) or not glob.glob(os.path.join(plots_dir, "iter_*.png")):
            print(f"  {name}: no frames — skipping."); continue
        if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"  {name}: video already present — skipping (use --force-videos to rebuild).")
            n_skip += 1; continue
        print(f"  {name}: building video …")
        if make_video_from_dir(plots_dir, out_path):
            n_ok += 1
        else:
            n_fail += 1
    print(f"\nVideos: {n_ok} written, {n_skip} skipped, {n_fail} failed.  ({run_dir})")


# ─── Running summary (mobo_progress.json / mobo_results.json) ───────────────────

def _build_summary(X_obs: list[torch.Tensor], Y_obs: list[torch.Tensor],
                   prior_count: int = 0) -> dict:
    n = len(Y_obs)
    Y_stk = torch.stack(Y_obs)
    pareto_mask = is_non_dominated(Y_stk)
    metrics_all = [
        {
            "dist_to_needles": round(-Y_obs[i][0].item(), 6),
            "dup_fraction":    round(-Y_obs[i][1].item(), 6),
            "runtime_s":       round(-Y_obs[i][2].item(), 3),
        }
        for i in range(n)
    ]
    # A resumed run is seeded with prior history, so it never runs Sobol init.
    trials = [
        {
            "trial":   i + 1,
            "phase":   "sobol" if (prior_count == 0 and i < N_INIT_TRIALS) else "mobo",
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
    return {
        "n_trials": n,
        "n_pareto": int(pareto_mask.sum().item()),
        "averages": {
            "dist_to_needles": round(float(np.mean(dists)),    6),
            "dup_fraction":    round(float(np.mean(dups)),     6),
            "runtime_s":       round(float(np.mean(runtimes)), 3),
        },
        "best_dist": {"value": round(min(dists), 6), "trial": int(np.argmin(dists)) + 1},
        "trials": trials,
    }, pareto_mask


def save_running_summary(X_obs, Y_obs, run_dir: str, prior_count: int = 0) -> torch.Tensor:
    """Write mobo_progress.json + mobo_results.json + mobo_results.pt. Returns pareto mask."""
    if not Y_obs:
        return torch.zeros(0, dtype=torch.bool)
    summary, pareto_mask = _build_summary(X_obs, Y_obs, prior_count=prior_count)
    summary_txt = json.dumps(summary, indent=2)
    _atomic_write_text(os.path.join(run_dir, "mobo_progress.json"), summary_txt)
    _atomic_write_text(os.path.join(run_dir, "mobo_results.json"), summary_txt)
    # mobo_results.pt is the resume source-of-truth → atomic + .bak.
    _atomic_torch_save(
        {"X_obs": torch.stack(X_obs).cpu(), "Y_obs": torch.stack(Y_obs).cpu(),
         "hparam_names": HPARAM_NAMES},
        os.path.join(run_dir, "mobo_results.pt"),
    )
    print(f"  [summary] {len(Y_obs)} trials, {int(pareto_mask.sum().item())} Pareto", flush=True)
    return pareto_mask


def save_pareto_plot(X_obs, Y_obs, run_dir: str) -> None:
    if not Y_obs:
        return
    Y = torch.stack(Y_obs)
    pareto_mask = is_non_dominated(Y).cpu().numpy()
    Y_np = (-Y).cpu().numpy()   # dist, dup, runtime
    pairs = [
        (0, 2, "dist_to_needles", "runtime (s)"),
        (0, 1, "dist_to_needles", "dup_fraction"),
        (1, 2, "dup_fraction",    "runtime (s)"),
    ]
    fig = Figure(figsize=(15, 5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    fig.suptitle("MOBO Pareto front  (★ = Pareto-optimal)", fontsize=12)
    for ax, (ix, iy, xl, yl) in zip(axes, pairs):
        ax.scatter(Y_np[~pareto_mask, ix], Y_np[~pareto_mask, iy], c="steelblue",
                   alpha=0.6, edgecolors="k", linewidths=0.3, label="dominated")
        ax.scatter(Y_np[pareto_mask, ix], Y_np[pareto_mask, iy], marker="*", s=220,
                   c="gold", zorder=5, edgecolors="k", linewidths=0.5, label="Pareto")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "pareto_front.png"), dpi=120, bbox_inches="tight")
    fig.clear()
    print(f"  Pareto plot saved to {os.path.join(run_dir, 'pareto_front.png')}")


# ─── MOBO loop (unbounded, resumable) ───────────────────────────────────────────

def run_mobo(rf_fn, true_optima, grid_pts, grid_vals, maximize, run_dir,
             max_trials=None, X_prior=None, Y_prior=None) -> None:
    """Unbounded MOBO loop, writing trials into a fresh ``run_dir``.

    ``X_prior``/``Y_prior`` seed the GP with (X, Y) pairs harvested from past runs
    (see ``collect_all_observations``) so resume continues from the full landscape.
    Prior data only feeds GP fitting + the Pareto plot; the run's own
    progress.json / results record only this run's trials, so re-crawling later
    never double-counts.  When prior history is present, Sobol init is skipped.
    """
    bounds = torch.zeros(2, N_HPARAMS, dtype=DTYPE, device=DEVICE)
    bounds[1] = 1.0

    # maximised objectives Y = (-dist, -dup, -runtime)
    X_prior = [x.detach().cpu() for x in X_prior] if X_prior else []
    Y_prior = [y.detach().cpu() for y in Y_prior] if Y_prior else []
    n_prior = len(Y_prior)
    X_obs: list[torch.Tensor] = []   # this run's own trials only (written to disk)
    Y_obs: list[torch.Tensor] = []

    X_sobol = load_or_make_sobol(run_dir, bounds)

    print(f"\n{'='*70}")
    print(f"MOBO  |  {N_INIT_TRIALS} Sobol init, then BO until Ctrl+C")
    print(f"Time limit / trial: {TIME_LIMIT_HOURS} h    Run dir: {run_dir}")
    if n_prior:
        print(f"PRIOR HISTORY — seeding GP with {n_prior} (X,Y) pair(s) from past runs; "
              f"skipping Sobol init")
    print(f"Hyperparameters ({N_HPARAMS}): {HPARAM_NAMES}")
    print(f"{'='*70}")

    consec_fail = 0
    try:
        while max_trials is None or len(Y_obs) < max_trials:
            n_done    = len(Y_obs)
            use_sobol = (n_prior == 0 and n_done < N_INIT_TRIALS)
            phase     = "sobol" if use_sobol else "mobo"
            trial_num = n_done + 1
            trial_dir = os.path.join(run_dir, f"trial_{trial_num}")

            try:
                # ── Pick the next hyperparameter vector ──
                if use_sobol:
                    x_new = X_sobol[n_done].clone()
                else:
                    X_t = torch.stack(X_prior + X_obs).to(DEVICE)
                    Y_t = torch.stack(Y_prior + Y_obs).to(DEVICE)
                    span = (Y_t.max(dim=0).values - Y_t.min(dim=0).values).clamp(min=1e-6)
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
                        acq_function=acq, bounds=bounds.to(DEVICE), q=1,
                        num_restarts=N_MOBO_RESTARTS, raw_samples=N_MOBO_SAMPLES,
                    )
                    x_new = candidate.squeeze(0).detach().cpu()

                hparams = norm_to_hparams(x_new)
                hp_str = "  ".join(f"{k}={round(v,4) if isinstance(v,float) else v}"
                                   for k, v in hparams.items())
                print(f"\n[trial {trial_num} | {phase}]  {hp_str}")

                # ── Run the trial + write its artifacts ──
                res = run_single_trial(hparams, rf_fn, true_optima, grid_pts, grid_vals,
                                       maximize, trial_dir)
                try:
                    plot_hparam_edge_proximity(
                        os.path.join(trial_dir, "hparam_edge_proximity.png"), x_new)
                except Exception as exc:
                    print(f"    [trial] hparam_edge_proximity failed: {exc}")

                # ── Record (this is the point the trial becomes "done") ──
                # Keep observations on CPU so prior + new entries share a device
                # (zombihop sets the global default device to CUDA on import).
                X_obs.append(x_new.detach().cpu())
                Y_obs.append(torch.tensor([-res["dist"], -res["dup"], -res["runtime"]],
                                          dtype=DTYPE, device="cpu"))
                save_running_summary(X_obs, Y_obs, run_dir, prior_count=n_prior)
                # Pareto status reported against the FULL landscape (prior + this run).
                global_mask = is_non_dominated(torch.stack(Y_prior + Y_obs))
                is_pareto = bool(global_mask[-1].item())
                write_trial_json(
                    os.path.join(trial_dir, "trial.json"),
                    trial_num, phase, is_pareto,
                    {"dist": res["dist"], "dup": res["dup"], "runtime": res["runtime"]},
                    hparams,
                )
                consec_fail = 0

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # Log & continue: a single failed trial must not end the night.
                consec_fail += 1
                _log_error(run_dir, trial_num, exc)
                print(f"    [trial {trial_num}] FAILED: {exc} — logged to errors.log "
                      f"({consec_fail}/{MAX_CONSEC_FAIL} consecutive); retrying.")
                if consec_fail >= MAX_CONSEC_FAIL:
                    print(f"\n[!] {MAX_CONSEC_FAIL} consecutive failures — aborting "
                          f"(see errors.log). Fix the issue and rerun with --resume.")
                    break
                continue
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user — finalising results …")

    if Y_obs:
        save_running_summary(X_obs, Y_obs, run_dir, prior_count=n_prior)
        # Pareto plot spans the full landscape (prior history + this run's trials).
        save_pareto_plot(X_prior + X_obs, Y_prior + Y_obs, run_dir)
    print(f"\nDone. {len(Y_obs)} trials completed this run "
          f"({n_prior} prior + {len(Y_obs)} new = {n_prior + len(Y_obs)} total). Results in {run_dir}")
    print(f"Resume (crawls all runs) with:  python optimize/run_mobo.py --resume")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ZoMBI-Hop MOBO hyperparameter optimisation (RF surrogate).")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Optional cap on total number of trials (default: unbounded, Ctrl+C to stop).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume optimisation using ALL prior data: crawl every "
                             "runs/mobo_*/mobo_progress.json, collect all (X,Y) pairs, and "
                             "seed a NEW runs/mobo_* run with them (reusing the latest run's "
                             "saved RF settings + picked optima). Non-interactive.")
    parser.add_argument("--make-videos", nargs="?", const="__latest__", default=None,
                        metavar="RUN_DIR",
                        help="Regenerate zombihop_timelapse.mp4 for every trial in a run from "
                             "its saved frames, then exit. Give a runs/mobo_* folder, or pass "
                             "with no value for the newest run. No optimisation is run.")
    parser.add_argument("--force-videos", action="store_true",
                        help="With --make-videos, rebuild videos even if one already exists.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir   = os.path.join(script_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # ── Video regeneration only (no optimisation, no GUI) ──
    if args.make_videos is not None:
        run_dir = resolve_run_dir(args.make_videos, runs_dir)
        print(f"Regenerating videos for {run_dir}")
        regenerate_videos(run_dir, force=args.force_videos)
        return

    # ── Resume path: crawl all prior runs, seed a new run, rebuild RF, no GUI ──
    if args.resume:
        cfg         = load_latest_run_config(runs_dir)
        rf_maximize = cfg["maximize"]
        csv_path    = cfg["csv_path"]
        true_optima = [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
        if cfg.get("hparam_names") != HPARAM_NAMES:
            print("  [resume] WARNING: latest run's hparam_names differ from the current "
                  "HPARAM_SPACE; only matching trials are collected.")

        print("=" * 70)
        print("ZoMBI-Hop MOBO — RESUMING (crawling all prior runs)")
        print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h   |   "
              f"{'maximize' if rf_maximize else 'minimize'}")
        print("=" * 70)

        print("\n[collect] Crawling runs/mobo_*/mobo_progress.json for all (X,Y) pairs …")
        X_prior, Y_prior, n_runs = collect_all_observations(runs_dir)
        print(f"  [collect] {len(Y_prior)} trial(s) from {n_runs} run(s) -> prior history.")

        if not os.path.exists(csv_path):
            sys.exit(f"Saved CSV path no longer exists: {csv_path}")
        print(f"\n[RF] Rebuilding surrogate from {csv_path} …")
        _, rf_fn, grid_pts, grid_vals = build_rf_and_grid(csv_path)
        print(f"  RF ready: reusing {len(true_optima)} saved reference optima")

        run_dir = os.path.join(runs_dir, datetime.datetime.now().strftime("mobo_%d_%m_%H_%M"))
        os.makedirs(run_dir, exist_ok=True)
        write_run_config(run_dir, rf_maximize, csv_path, true_optima)
        print(f"[run] New output folder: {run_dir}")

        run_mobo(rf_fn, true_optima, grid_pts, grid_vals, rf_maximize, run_dir,
                 max_trials=args.max_trials, X_prior=X_prior, Y_prior=Y_prior)
        return

    # ── Fresh run ──
    print("=" * 70)
    print("ZoMBI-Hop Hyperparameter Optimisation (MOBO)  —  RF surrogate")
    print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h")
    print("=" * 70)

    # RF surrogate on campaign1a.csv
    csv_candidates = [
        os.path.join(script_dir, "..", "interactive_testing", "data", "campaign1a.csv"),
        os.path.join(script_dir, "..", "interactive_testing", "campaign1a.csv"),
    ]
    csv_path = next((os.path.normpath(p) for p in csv_candidates if os.path.exists(p)), None)
    if csv_path is None:
        sys.exit("campaign1a.csv not found. Tried:\n" +
                 "\n".join(f"  {os.path.normpath(p)}" for p in csv_candidates))
    print(f"\n[RF] Loading {csv_path} …  Training RF ({RF_N_ESTIMATORS} trees) …")
    rf, rf_fn, grid_pts, grid_vals = build_rf_and_grid(csv_path)
    print("  RF trained.")

    raw_mm = input("  Maximize or minimize RF?  [max/min, default min]: ").strip().lower()
    rf_maximize = raw_mm in ("max", "x", "maximize")
    goal = "maxima" if rf_maximize else "minima"

    plt.ion()
    print(f"  Click near reference {goal}, then Enter / Q.")
    picker  = ExtremaPicker(rf, grid_pts, grid_vals, maximize=rf_maximize)
    extrema = picker.run()
    if not extrema:
        print("  No extrema selected — using centroid as fallback.")
        extrema = [(np.array([1/3, 1/3, 1/3]), 0.0)]
    true_optima = [x for x, _ in extrema]
    print(f"  RF ready: {len(true_optima)} reference {goal}")

    # Run directory: runs/mobo_DD_MM_HH_MM (military time)
    run_dir = os.path.join(runs_dir, datetime.datetime.now().strftime("mobo_%d_%m_%H_%M"))
    os.makedirs(run_dir, exist_ok=True)
    write_run_config(run_dir, rf_maximize, csv_path, true_optima)
    print(f"\n[run] Output folder: {run_dir}")

    run_mobo(rf_fn, true_optima, grid_pts, grid_vals, rf_maximize, run_dir,
             max_trials=args.max_trials)


if __name__ == "__main__":
    main()
