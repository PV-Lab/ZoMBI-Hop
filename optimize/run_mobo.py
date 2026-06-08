"""
optimize/run_mobo.py
====================
Multi-objective Bayesian optimisation (MOBO) of ZoMBI-Hop hyperparameters,
evaluated on the Random-Forest surrogate built from campaign1a.csv.

Three objectives (all minimised):
  1. dist_to_needles    – symmetric greedy matching distance between needles and
                          true optima (no-repeat matching; both unmatched true
                          optima AND unmatched/spurious needles incur
                          UNMATCHED_PENALTY, mean over max(#needles, #optima))
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

Run modes
---------
A run is configured by (a) the RF direction (max/min), (b) the campaign CSV, and
(c) the reference optima. Each mode sources those three differently and may or
may not seed the new run with data harvested from past runs:

  fresh (default)   interactive picker for config; no prior data.
  --resume          reuse the LATEST run's saved config; seed the GP with ALL
                    (X,Y) pairs crawled from every runs/mobo_*/mobo_progress.json.
  --resume-scratch  re-prompt for config (picker); seed with ALL prior (X,Y),
                    re-deriving dist_to_needles against the freshly-picked optima.
  --copy-config P   reuse a SPECIFIC run's saved config (P = run dir or
                    run_config.json), but start with NO prior data — a normal
                    Sobol-init + BO run under someone else's config.

Modifiers (combinable with any mode above):
  --start-from-best DIR [DIR ...]  copy the (hyperparameters, metrics) of the
                    given trial_* dir(s) (or trial.json files) straight into the
                    GP prior history; never re-evaluated, and skips Sobol init.
  --max-trials N    cap total trials (default: unbounded, Ctrl+C to stop).

Usage
-----
  conda activate zombi-hop
  python optimize/run_mobo.py                                   # fresh, interactive
  python optimize/run_mobo.py --resume                          # seed from all past runs
  python optimize/run_mobo.py --resume-scratch                  # re-pick config + seed
  python optimize/run_mobo.py --copy-config runs/mobo_04_06_11_47   # reuse config, no data
  python optimize/run_mobo.py --start-from-best runs/mobo_04_06_11_47/trial_1 [trial_dir ...]
  python optimize/make_videos.py                       # newest run
  python optimize/make_videos.py <run_dir>             # specific run
  python optimize/make_videos.py <run_dir> --force     # rebuild all
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

# pct_matched: a needle counts as "valid" if it is within this Euclidean
# (composition L2) radius of some true optimum.
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
TERNARY_GRID_N  = 80     # render/metric grid (kept coarse: drawn every trial frame)
PICKER_GRID_N   = 120    # interactive extrema picker only (matches interactive_test_zombi.py)
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
    "raw":                         (1,    300,  "int"),
    # Acquisition function
    "ucb_beta":                    (0.05,   3.0,   "linear"),
    # Zoom / convergence
    "max_zooms":                   (2,      10,    "int"),
    "max_iterations":              (2,      10,    "int"),
    "top_m_points":                (2,      8,     "int"),
    "n_consecutive_converged":     (1,      5,    "int"),
    "convergence_pi_threshold":    (1e-4,   0.05,  "log"),
    "input_noise_threshold_mult":  (0.5,    6.0,   "linear"),
    "output_noise_threshold_mult": (0.1,    2.0,   "linear"),
    # Penalisation & needle
    "max_penalty_radius":          (0.2,    5.0,   "linear"),
    "needle_shrink_factor":        (0.55,   0.99,  "linear"),
    "needle_stop_noise_multiplier":(1.0,    8.0,   "linear"),
    # Point paring (deduplication)
    "paring_spatial_halfnoise":    (0.1,    2.0,   "linear"),
    "paring_y_noise_multiplier":   (0.1,    5.0,   "linear"),
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


def load_latest_run_config(runs_dir: str) -> dict:
    """Return the run_config.json from the most recently created runs/mobo_* run.

    Used on resume to reuse the static config (min/max, csv_path, reference
    optima) without re-prompting interactively.

    Recency is decided by the ``created`` timestamp that ``write_run_config``
    stamps in at run start — *before* any trial runs — so the newest run's config
    is chosen regardless of whether that run completed (or even started) any
    trials. This is also more robust than file mtime, which Dropbox sync can
    rewrite; mtime is only a fallback when ``created`` is missing/unparseable.
    """
    cands = glob.glob(os.path.join(runs_dir, "mobo_*", "run_config.json"))
    if not cands:
        sys.exit(f"No runs/mobo_*/run_config.json found under {runs_dir} — nothing to resume.")

    def created_key(cfg_path: str):
        # (1, created_ts) sorts above (0, mtime) so a parseable timestamp always
        # wins over a fallback; within each tier, larger = more recent.
        try:
            with open(cfg_path) as f:
                created = json.load(f).get("created")
            if created:
                return (1, datetime.datetime.fromisoformat(created).timestamp())
        except Exception:
            pass
        return (0, os.path.getmtime(cfg_path))

    latest = max(cands, key=created_key)
    with open(latest) as f:
        cfg = json.load(f)
    print(f"  [resume] reusing config from {os.path.basename(os.path.dirname(latest))} "
          f"(created {cfg.get('created', '?')})")
    return cfg


def load_run_config_from_path(path: str) -> dict:
    """Load one specific run's run_config.json, given its run dir or the json file.

    Used by ``--copy-config``: reuse another run's static config (min/max,
    csv_path, reference optima) for a NEW run without re-prompting the picker and
    without inheriting that run's data points.
    """
    cfg_path = path if path.lower().endswith(".json") else os.path.join(path, "run_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"--copy-config: no run_config.json found at {cfg_path}")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as exc:
        sys.exit(f"--copy-config: {cfg_path} unreadable ({exc}).")
    print(f"  [copy-config] reusing config from {cfg_path} "
          f"(created {cfg.get('created', '?')})")
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


def _read_needles_csv(path: str) -> np.ndarray:
    """Read a trial's needles.csv → (N, 3) array of FA/MA/Br needle coords.

    A trial that discovered no needles still writes a header-only file, so an
    empty result is meaningful (0 needles → max distance penalty on rescore).
    """
    import csv as _csv
    pts = []
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            pts.append([float(row["FA"]), float(row["MA"]), float(row["Br"])])
    return np.array(pts, dtype=float) if pts else np.empty((0, 3))


def collect_rederived_observations(runs_dir: str, new_maximize: bool,
                                   new_optima: list[np.ndarray]):
    """Crawl prior runs and rebuild (X, Y) under a freshly-picked run config.

    Used by ``--resume-scratch``. ``dist_to_needles`` is the only objective that
    depends on the picked optima, and it is a post-hoc geometric score over the
    needles a trial discovered — those are persisted per trial in needles.csv, so
    it can be recomputed against ``new_optima`` with no ZoMBI re-runs. ``dup`` and
    ``runtime`` don't depend on the config and are reused as stored.

    Trials whose needles.csv is missing (older runs that never wrote per-trial
    dirs) cannot be re-scored. Rather than drop them, the stored
    ``dist_to_needles`` is reused as-is with a per-run warning — an approximation
    that is reasonable only when the new optima are close to the run's original
    ones.

    This is only valid when the search direction is unchanged: flipping max/min
    changes which points are "good", so the stored needles would be invalid. If
    any prior run's direction differs from ``new_maximize`` (case 2), this aborts
    via ``sys.exit`` and nothing is run.

    Returns (X_obs, Y_obs, n_runs).
    """
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "mobo_*")))

    # ── Pass 1: verify every prior run's direction matches the new selection ──
    mismatched, no_config = [], []
    for rd in run_dirs:
        if not os.path.exists(os.path.join(rd, "mobo_progress.json")):
            continue
        cfg_path = os.path.join(rd, "run_config.json")
        if not os.path.exists(cfg_path):
            no_config.append(os.path.basename(rd))
            continue
        try:
            with open(cfg_path) as f:
                prior_max = bool(json.load(f).get("maximize"))
        except Exception:
            no_config.append(os.path.basename(rd))
            continue
        if prior_max != new_maximize:
            mismatched.append((os.path.basename(rd), prior_max))

    if mismatched:
        new_dir = "maximize" if new_maximize else "minimize"
        print("\n" + "=" * 70)
        print("  [resume-scratch] ABORT — direction flip detected (case 2).")
        print(f"  You selected '{new_dir}', but these prior runs used the opposite "
              f"direction:")
        for name, pm in mismatched:
            print(f"    - {name}: {'maximize' if pm else 'minimize'}")
        print("  Flipping max/min changes the entire ZoMBI search, so the stored "
              "needles are\n  invalid and dist_to_needles cannot be re-derived "
              "without re-running every\n  trial. Nothing has been run. Re-pick the "
              "matching direction, or start a\n  fresh run instead.")
        print("=" * 70)
        sys.exit(1)
    if no_config:
        print(f"  [resume-scratch] WARNING: skipping run(s) without a readable "
              f"run_config.json (direction unverifiable): {', '.join(no_config)}")

    # ── Pass 2: re-derive dist from saved needles; reuse stored dist if absent ──
    X_obs, Y_obs = [], []
    n_runs = 0
    for rd in run_dirs:
        run_name = os.path.basename(rd)
        prog_path = os.path.join(rd, "mobo_progress.json")
        if not os.path.exists(prog_path) or not os.path.exists(os.path.join(rd, "run_config.json")):
            continue
        try:
            with open(prog_path) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  [resume-scratch] {prog_path} unreadable ({exc}); skipping.")
            continue
        n_rederived = n_reused = 0
        for t in data.get("trials", []):
            hp, m = t.get("hparams", {}), t.get("metrics", {})
            if not all(name in hp for name in HPARAM_NAMES):
                continue  # stale / different hyperparameter set
            needles_path = os.path.join(rd, f"trial_{t.get('trial')}", "needles.csv")
            try:
                if os.path.exists(needles_path):
                    needles = _read_needles_csv(needles_path)
                    dist = metric_dist_to_needles(needles, new_optima)
                    rederived = True
                else:
                    # No saved needles → cannot re-score; reuse stored dist as-is.
                    dist = float(m["dist_to_needles"])
                    rederived = False
                x = hparams_to_norm(hp)
                y = torch.tensor([-float(dist),
                                  -float(m["dup_fraction"]),
                                  -float(m["runtime_s"])], dtype=DTYPE)
            except (KeyError, ValueError, TypeError) as exc:
                print(f"  [resume-scratch] {run_name} trial "
                      f"{t.get('trial')}: could not build observation ({exc}); skipping.")
                continue
            X_obs.append(x)
            Y_obs.append(y)
            if rederived:
                n_rederived += 1
            else:
                n_reused += 1
        used = n_rederived + n_reused
        if used:
            n_runs += 1
            parts = []
            if n_rederived:
                parts.append(f"{n_rederived} re-derived")
            if n_reused:
                parts.append(f"{n_reused} reused as-is")
            print(f"  [resume-scratch] {run_name}: {used} trial(s) ({', '.join(parts)})")
            if n_reused:
                print(f"  [resume-scratch]   WARNING: {run_name} has no saved needles — "
                      f"its dist_to_needles was NOT re-scored to the new optima; reused as "
                      f"stored (assumed close enough).")
    return X_obs, Y_obs, n_runs


def load_or_make_sobol(run_dir: str, bounds: torch.Tensor, n: int) -> torch.Tensor:
    """Load the persisted Sobol init design, or draw + persist it on first use.

    Persisting keeps the init trials identical across restarts. ``n`` is the
    number of Sobol draws (normally ``N_INIT_TRIALS``, but reduced by one per
    "start from best" seed); a persisted design of a different size is redrawn.
    """
    path = os.path.join(run_dir, "sobol_design.pt")
    if os.path.exists(path):
        try:
            X = torch.load(path, map_location="cpu").to(device=DEVICE, dtype=DTYPE)
            if X.shape[0] == n:
                return X
            print(f"  [resume] sobol_design.pt has {X.shape[0]} rows, expected {n}; redrawing.")
        except Exception as exc:
            print(f"  [resume] sobol_design.pt unreadable ({exc}); redrawing.")
    if n <= 0:
        X_sobol = torch.empty(0, bounds.shape[1], dtype=DTYPE, device=DEVICE)
    else:
        X_sobol = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(1)
    _atomic_torch_save(X_sobol.cpu(), path)
    return X_sobol


def load_seed_observations(trial_paths: list[str]):
    """Load (X_norm, Y) pairs from one or more trial_* dirs → prior history.

    Each path is a trial directory (or its trial.json directly) produced by a
    prior run. These seed the GP for "start from best": the stored (hyperparameters,
    metrics) are COPIED straight into the prior history (exactly like
    ``collect_all_observations``) and never re-evaluated, so the GP starts already
    knowing these known-good points. Trials whose hparams/metrics don't cover the
    current HPARAM_SPACE abort the run (stale hyperparameter set).
    Returns (X_obs, Y_obs).
    """
    X_obs, Y_obs = [], []
    for p in trial_paths:
        json_path = p if p.lower().endswith(".json") else os.path.join(p, "trial.json")
        if not os.path.exists(json_path):
            sys.exit(f"--start-from-best: no trial.json found at {json_path}")
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as exc:
            sys.exit(f"--start-from-best: {json_path} unreadable ({exc}).")
        hp = data.get("hparams", {})
        missing = [name for name in HPARAM_NAMES if name not in hp]
        if missing:
            sys.exit(f"--start-from-best: {json_path} is missing hparams {missing} "
                     f"(stale hyperparameter set?).")
        m = data.get("metrics", {})
        try:
            y = torch.tensor([-float(m["dist_to_needles"]),
                              -float(m["dup_fraction"]),
                              -float(m["runtime_s"])], dtype=DTYPE)
        except (KeyError, ValueError, TypeError):
            sys.exit(f"--start-from-best: {json_path} is missing metrics "
                     f"(dist_to_needles/dup_fraction/runtime_s).")
        X_obs.append(hparams_to_norm(hp))
        Y_obs.append(y)
        print(f"  [seed] {p}  (trial {data.get('trial', '?')}, "
              f"dist_to_needles={m.get('dist_to_needles', '?')})")
    return X_obs, Y_obs


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
        # High-resolution landscape for picking — evaluated on a denser grid than
        # the coarse render/metric grid so the displayed surface matches the
        # interactive_test_zombi.py picker.
        pick_pts  = ternary_grid(PICKER_GRID_N)
        pick_vals = self.rf.predict(pick_pts)
        gxy = comp_to_xy(pick_pts)
        sc = ax.scatter(gxy[:, 0], gxy[:, 1], c=pick_vals,
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

UNMATCHED_PENALTY = 10.0   # per-term penalty for an unmatched optimum OR needle


def metric_dist_to_needles(
    discovered: np.ndarray,
    true_optima: list[np.ndarray],
) -> float:
    """Symmetric greedy matching distance between needles and true optima.

    Each true optimum is greedily matched (no repeats) to its nearest discovered
    needle and contributes that Euclidean (composition L2) distance.  Two kinds of
    leftovers each contribute UNMATCHED_PENALTY:

      • a true optimum with no needle left to match  (missed optimum — recall),
      • a needle matched to no optimum               (spurious/duplicate — precision).

    The score is the mean over ``max(#needles, #optima)`` terms.  Including a
    per-needle penalty is what makes the metric symmetric: spawning extra needles
    near already-matched optima now *raises* the score instead of lowering it, so
    "needle spam" is penalised rather than rewarded.  When the needle and optimum
    counts are equal and all match, this reduces to the old mean-distance score.
    """
    n_opt = len(true_optima)
    if n_opt == 0:
        return 0.0
    n_disc = len(discovered)
    if n_disc == 0:
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
            total += UNMATCHED_PENALTY          # fewer needles than optima (missed)
    # Every needle not claimed by an optimum is a false positive (precision term).
    total += (n_disc - len(used)) * UNMATCHED_PENALTY
    return total / max(n_disc, n_opt)


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
    """Percentage of DISCOVERED needles that lie within `radius` of a true optimum.

    A precision-style score (true positives / all needles): spurious or spammed
    needles that sit far from every true optimum drag it down, so — unlike a
    coverage/recall score over the fixed set of true optima — it is NOT monotonic
    in needle count.  Returns 0.0 when there are no needles (no valid needles yet)
    or no reference optima (nothing to validate against).
    """
    if len(discovered) == 0 or not true_optima:
        return 0.0
    disc = np.asarray(discovered, dtype=float)
    opt  = np.asarray(true_optima, dtype=float)
    valid = 0
    for d in disc:
        if float(np.linalg.norm(opt - d, axis=1).min()) <= radius:
            valid += 1
    return 100.0 * valid / len(disc)


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


def write_trial_json(path: str, trial_num: int, phase: str,
                     metrics: dict, hparams: dict) -> None:
    obj = {
        "trial": trial_num,
        "phase": phase,
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


# ─── Running summary (mobo_progress.json / mobo_results.json) ───────────────────

def _build_summary(X_obs: list[torch.Tensor], Y_obs: list[torch.Tensor],
                   prior_count: int = 0) -> dict:
    n = len(Y_obs)
    metrics_all = [
        {
            "dist_to_needles": round(-Y_obs[i][0].item(), 6),
            "dup_fraction":    round(-Y_obs[i][1].item(), 6),
            "runtime_s":       round(-Y_obs[i][2].item(), 3),
        }
        for i in range(n)
    ]
    # Init phase = Sobol, which only runs on a fresh run; a run seeded with prior
    # history (resume or --start-from-best) skips Sobol entirely.
    # Pareto membership is intentionally NOT recorded here — it is determined
    # across all runs after the fact by optimize/pareto.py.
    def _phase(i: int) -> str:
        if prior_count == 0 and i < N_INIT_TRIALS:
            return "sobol"
        return "mobo"
    trials = [
        {
            "trial":   i + 1,
            "phase":   _phase(i),
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
        "averages": {
            "dist_to_needles": round(float(np.mean(dists)),    6),
            "dup_fraction":    round(float(np.mean(dups)),     6),
            "runtime_s":       round(float(np.mean(runtimes)), 3),
        },
        "best_dist": {"value": round(min(dists), 6), "trial": int(np.argmin(dists)) + 1},
        "trials": trials,
    }


def save_running_summary(X_obs, Y_obs, run_dir: str, prior_count: int = 0) -> None:
    """Write mobo_progress.json + mobo_results.json + mobo_results.pt."""
    if not Y_obs:
        return
    summary = _build_summary(X_obs, Y_obs, prior_count=prior_count)
    summary_txt = json.dumps(summary, indent=2)
    _atomic_write_text(os.path.join(run_dir, "mobo_progress.json"), summary_txt)
    _atomic_write_text(os.path.join(run_dir, "mobo_results.json"), summary_txt)
    # mobo_results.pt is the resume source-of-truth → atomic + .bak.
    _atomic_torch_save(
        {"X_obs": torch.stack(X_obs).cpu(), "Y_obs": torch.stack(Y_obs).cpu(),
         "hparam_names": HPARAM_NAMES},
        os.path.join(run_dir, "mobo_results.pt"),
    )
    print(f"  [summary] {len(Y_obs)} trials recorded", flush=True)


# ─── MOBO loop (unbounded, resumable) ───────────────────────────────────────────

def run_mobo(rf_fn, true_optima, grid_pts, grid_vals, maximize, run_dir,
             max_trials=None, X_prior=None, Y_prior=None) -> None:
    """Unbounded MOBO loop, writing trials into a fresh ``run_dir``.

    ``X_prior``/``Y_prior`` seed the GP with (X, Y) pairs harvested from past runs
    (see ``collect_all_observations``) or copied from chosen trials (``--start-from-best``,
    see ``load_seed_observations``) so the run continues from the full landscape.
    Prior data only feeds GP fitting; the run's own progress.json / results
    record only this run's trials, so re-crawling later never double-counts.
    When prior history is present, Sobol init is skipped.
    """
    bounds = torch.zeros(2, N_HPARAMS, dtype=DTYPE, device=DEVICE)
    bounds[1] = 1.0

    # maximised objectives Y = (-dist, -dup, -runtime)
    X_prior = [x.detach().cpu() for x in X_prior] if X_prior else []
    Y_prior = [y.detach().cpu() for y in Y_prior] if Y_prior else []
    n_prior = len(Y_prior)
    X_obs: list[torch.Tensor] = []   # this run's own trials only (written to disk)
    Y_obs: list[torch.Tensor] = []

    # ── Build the initial design: Sobol only, and only on a fresh run. A run
    # seeded with prior history (resume or --start-from-best) skips Sobol. ──
    n_sobol = N_INIT_TRIALS if n_prior == 0 else 0
    X_sobol = load_or_make_sobol(run_dir, bounds, n_sobol)
    init_design = [(X_sobol[i], "sobol") for i in range(X_sobol.shape[0])]
    n_init = len(init_design)

    print(f"\n{'='*70}")
    print(f"MOBO  |  {X_sobol.shape[0]} Sobol init, then BO until Ctrl+C")
    print(f"Time limit / trial: {TIME_LIMIT_HOURS} h    Run dir: {run_dir}")
    if n_prior:
        print(f"PRIOR HISTORY — seeding GP with {n_prior} (X,Y) pair(s) "
              f"(prior runs and/or --start-from-best); skipping Sobol init")
    print(f"Hyperparameters ({N_HPARAMS}): {HPARAM_NAMES}")
    print(f"{'='*70}")

    consec_fail = 0
    try:
        while max_trials is None or len(Y_obs) < max_trials:
            n_done    = len(Y_obs)
            use_init  = n_done < n_init
            phase     = init_design[n_done][1] if use_init else "mobo"
            trial_num = n_done + 1
            trial_dir = os.path.join(run_dir, f"trial_{trial_num}")

            try:
                # ── Pick the next hyperparameter vector ──
                if use_init:
                    x_new = init_design[n_done][0].detach().cpu().clone()
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
                write_trial_json(
                    os.path.join(trial_dir, "trial.json"),
                    trial_num, phase,
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
    print(f"\nDone. {len(Y_obs)} trials completed this run "
          f"({n_prior} prior + {len(Y_obs)} new = {n_prior + len(Y_obs)} total). Results in {run_dir}")
    print(f"Resume (crawls all runs) with:  python optimize/run_mobo.py --resume")
    print(f"Pareto front across all runs:   python optimize/pareto.py")


# ─── Main ───────────────────────────────────────────────────────────────────────

def _interactive_run_config(script_dir: str):
    """Interactive run-config creation: locate campaign1a.csv, train the RF,
    prompt max/min, and interactively pick the reference optima.

    Shared by the fresh run and ``--resume-scratch``. Returns
    ``(rf_fn, grid_pts, grid_vals, csv_path, rf_maximize, true_optima)``.
    """
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
    return rf_fn, grid_pts, grid_vals, csv_path, rf_maximize, true_optima


def _launch_run(runs_dir, rf_fn, true_optima, grid_pts, grid_vals, rf_maximize,
                csv_path, max_trials, X_prior=None, Y_prior=None) -> None:
    """Create a fresh runs/mobo_* folder, persist its config, and run MOBO."""
    run_dir = os.path.join(runs_dir, datetime.datetime.now().strftime("mobo_%d_%m_%H_%M"))
    os.makedirs(run_dir, exist_ok=True)
    write_run_config(run_dir, rf_maximize, csv_path, true_optima)
    print(f"\n[run] Output folder: {run_dir}")
    run_mobo(rf_fn, true_optima, grid_pts, grid_vals, rf_maximize, run_dir,
             max_trials=max_trials, X_prior=X_prior, Y_prior=Y_prior)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZoMBI-Hop MOBO hyperparameter optimisation (RF surrogate).")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Optional cap on total number of trials (default: unbounded, Ctrl+C to stop).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume optimisation using ALL prior data: crawl every "
                             "runs/mobo_*/mobo_progress.json, collect all (X,Y) pairs, and "
                             "seed a NEW runs/mobo_* run with them (reusing the latest run's "
                             "saved RF settings + picked optima). Non-interactive.")
    parser.add_argument("--resume-scratch", action="store_true",
                        help="Like --resume (seed a NEW run with ALL prior (X,Y) pairs), but "
                             "RE-PROMPT for run config (max/min + interactive optima picking) "
                             "instead of reusing the latest saved config. Note: prior Y values "
                             "were computed under the previous run's optima/direction, so a "
                             "different selection makes the seeded history inconsistent.")
    parser.add_argument("--start-from-best", nargs="+", metavar="TRIAL_DIR", default=None,
                        help="One or more trial_* directories (or trial.json files) whose "
                             "(hyperparameters, metrics) are COPIED straight into the GP "
                             "prior history (never re-evaluated), so the run starts already "
                             "knowing these points and skips Sobol init. Combinable with any "
                             "run mode.")
    parser.add_argument("--copy-config", metavar="PATH", default=None,
                        help="Reuse another run's run_config.json (max/min, csv_path, picked "
                             "optima) for a NEW run, WITHOUT inheriting its data points. PATH "
                             "is a run dir or a run_config.json file. Non-interactive; runs a "
                             "normal Sobol-init + BO run. Cannot combine with "
                             "--resume / --resume-scratch.")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.resume, args.resume_scratch, args.copy_config)) > 1:
        sys.exit("Use only one of --resume / --resume-scratch / --copy-config.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir   = os.path.join(script_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # "Start from best" seeds: (X, Y) pairs copied straight into the GP prior
    # history (never re-evaluated), shared by all run modes.
    X_seed, Y_seed = [], []
    if args.start_from_best:
        print("\n[seed] Loading 'start from best' (X,Y) pairs into prior history …")
        X_seed, Y_seed = load_seed_observations(args.start_from_best)

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
        X_prior += X_seed; Y_prior += Y_seed

        if not os.path.exists(csv_path):
            sys.exit(f"Saved CSV path no longer exists: {csv_path}")
        print(f"\n[RF] Rebuilding surrogate from {csv_path} …")
        _, rf_fn, grid_pts, grid_vals = build_rf_and_grid(csv_path)
        print(f"  RF ready: reusing {len(true_optima)} saved reference optima")

        _launch_run(runs_dir, rf_fn, true_optima, grid_pts, grid_vals, rf_maximize,
                    csv_path, args.max_trials, X_prior=X_prior, Y_prior=Y_prior)
        return

    # ── Resume-from-scratch: seed with all prior (X,Y), but re-pick config ──
    if args.resume_scratch:
        print("=" * 70)
        print("ZoMBI-Hop MOBO — RESUMING FROM SCRATCH (prior data + fresh config)")
        print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h")
        print("=" * 70)

        # Re-pick the run config first; the new optima/direction determine how the
        # prior (X,Y) pairs are re-derived (see collect_rederived_observations).
        rf_fn, grid_pts, grid_vals, csv_path, rf_maximize, true_optima = \
            _interactive_run_config(script_dir)

        print("\n[collect] Re-deriving prior trials against the freshly-picked optima …")
        X_prior, Y_prior, n_runs = collect_rederived_observations(
            runs_dir, rf_maximize, true_optima)
        print(f"  [collect] {len(Y_prior)} trial(s) from {n_runs} run(s) -> prior history "
              f"(dist re-scored where needles saved, else reused; dup/runtime reused).")
        X_prior += X_seed; Y_prior += Y_seed

        _launch_run(runs_dir, rf_fn, true_optima, grid_pts, grid_vals, rf_maximize,
                    csv_path, args.max_trials, X_prior=X_prior, Y_prior=Y_prior)
        return

    # ── Copy-config: reuse another run's config, but start with NO prior data ──
    if args.copy_config:
        cfg         = load_run_config_from_path(args.copy_config)
        rf_maximize = cfg["maximize"]
        csv_path    = cfg["csv_path"]
        true_optima = [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
        if cfg.get("hparam_names") and cfg["hparam_names"] != HPARAM_NAMES:
            print("  [copy-config] WARNING: copied run's hparam_names differ from the current "
                  "HPARAM_SPACE; only the static config (min/max, csv, optima) is reused.")

        print("=" * 70)
        print("ZoMBI-Hop MOBO — COPY-CONFIG (reused config, no inherited data)")
        print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h   |   "
              f"{'maximize' if rf_maximize else 'minimize'}")
        print("=" * 70)

        if not os.path.exists(csv_path):
            sys.exit(f"Copied CSV path no longer exists: {csv_path}")
        print(f"\n[RF] Rebuilding surrogate from {csv_path} …")
        _, rf_fn, grid_pts, grid_vals = build_rf_and_grid(csv_path)
        print(f"  RF ready: reusing {len(true_optima)} saved reference optima")

        _launch_run(runs_dir, rf_fn, true_optima, grid_pts, grid_vals, rf_maximize,
                    csv_path, args.max_trials, X_prior=X_seed, Y_prior=Y_seed)
        return

    # ── Fresh run ──
    print("=" * 70)
    print("ZoMBI-Hop Hyperparameter Optimisation (MOBO)  —  RF surrogate")
    print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h")
    print("=" * 70)

    rf_fn, grid_pts, grid_vals, csv_path, rf_maximize, true_optima = \
        _interactive_run_config(script_dir)

    _launch_run(runs_dir, rf_fn, true_optima, grid_pts, grid_vals, rf_maximize,
                csv_path, args.max_trials, X_prior=X_seed, Y_prior=Y_seed)


if __name__ == "__main__":
    main()
