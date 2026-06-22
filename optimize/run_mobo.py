"""
optimize/run_mobo.py
====================
Multi-objective Bayesian optimisation (MOBO) of ZoMBI-Hop hyperparameters.

Landscapes (``--landscape`` or batch JSON ``"landscape"`` field):
  • ``rf`` (default) — Random-Forest surrogate on a composition CSV (campaign1a)
  • ``synthetic``    — Direct analytic oracle (messy, gaussian, …) — MOBO default
                       for synthetic benchmarks; RF is comparison-only (see
                       synthetic_data/compare_campaign_datasets.py)
  • ``ackley``       — Multi-Ackley sum on the d-dimensional probability simplex

Objectives are selectable via ``--dataset``: RF (default, 3-simplex campaign1a
surrogate), analytic negated-Ackley benchmarks on the 3-/4-/10-simplex
(``ackley3d`` / ``ackley4d`` / ``ackley10d``), or ``ensemble`` (the layered
``synthetic_data.ensemble`` objective, dimension from ``--dim``). The
hyperparameters are dimension-independent. ``--landscape ackley`` is an alternate
Multi-Ackley benchmark (layout/b configurable); ``--dataset ackley*`` uses the
``synthetic_data.ackley.Ackley`` oracle (``--ackley-variant``).

``--dataset ensemble`` RE-RANDOMIZES the landscape for every evaluation (random
feature counts/widths/amplitudes + a fresh master seed; the number of true optima
is drawn from a dimension-specific range — 5–30 at dim 3, 20–50 at dim 4, 50–150
at dim 10). To keep the MOBO model from chasing that landscape noise, each
hyperparameter set is evaluated on ``ENSEMBLE_N_REPEATS`` (5) independently
randomized landscapes per trial and the three metrics are AVERAGED before the next
MOBO iteration. Every landscape is saved for reproducibility: each repeat writes
``trial_<n>/run_<k>/ensemble_config.json`` and the full ``ensemble_configs`` list
(plus per-repeat metrics) is recorded in ``trial_<n>/trial.json``. ``--ensemble-seed``
makes the per-trial landscape sequence reproducible; ``--ensemble-margin`` sets the
optima/background gap.

Three objectives (all minimised):
  1. dist_to_needles    – symmetric greedy matching distance between needles and
                          true optima (no-repeat matching; both unmatched true
                          optima AND unmatched/spurious needles incur
                          UNMATCHED_PENALTY, mean over max(#needles, #optima))
  2. dup_fraction       – fraction of sampled points whose nearest neighbour
                          in input space is within noise/2
  3. avg_time_per_iter_s  – average wall-clock seconds per ZoMBI iteration, where
                          an iteration is one LineBO main-line pick (== one
                          obj_wrapper call / one would-be plot frame): total
                          (timed) runtime / number of iterations. Plotting is
                          rendered AFTER the timed region so it never pollutes
                          this metric. (trial.json also records runtime_s and
                          n_iters for context.)

MOBO engine: qLogNEHVI (BoTorch, maximises negated objectives).

Landscape plotting scales with dimension: 3D renders per-iteration ternary frames
plus an end-of-trial static coverage plot; 4D renders a single end-of-trial
interactive (rotatable) point-cloud HTML (point_cloud.html; no per-iteration
frames); ≥5D renders no landscape view. The metrics time-series plot
(plot_metrics) is generated for every dimension, and the 3D coverage plot
(coverage_plot) at the end of each trial — no videos.

Run layout
----------
Each run creates a folder ``runs/mobo_DD_MM_HH_MM_SS_PID/`` (local or HPC;
date/time + process id for uniqueness) containing:
  • mobo_progress.json / mobo_results.json / mobo_results.pt  – running summary
  • pareto_front.png                                          – on exit
  • trial_<n>/                                                – one per trial
        ├─ trial.json                 (phase / pareto / metrics / hparams)
        ├─ points.csv                 (sample_idx, FA, MA, Br, Y, penalized,
        │                              activation, zoom)
        ├─ needles.csv                (needle_idx, FA, MA, Br, value,
        │                              median_value, zoom, iteration, reason,
        │                              dist_to_centre)
        ├─ metrics_over_time.csv      (iteration, dist_to_needles, dup_fraction,
        │                              pct_matched, avg_pairwise_dist,
        │                              recent_needle_value)
        ├─ convergence.png
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
  --dataset DS      objective: RF (default, interactive picker) or ackley3d /
                    ackley4d / ackley10d (analytic, non-interactive). Inherited
                    from saved config by --resume / --copy-config / --resume-from.
  --ackley-variant V  Ackley variant for ackley* datasets (default: realistic).
  --time-limit H    per-trial wall-clock budget in hours (alias: --time-limit-hours).
  --resume-from DIR resume from one specific run (trust its metrics, reuse config).
  --resume-dim      resume on the --dataset objective, seeding the new run with ALL
                    prior runs/mobo_* of the SAME dimensionality (crawl every run
                    whose run_config dim matches the --dataset dim, trust their
                    (X,Y) pairs, no re-evaluation). Non-interactive (ackley only).
  --start-from-best DIR [DIR ...]  RE-EVALUATE hparams from trial dir(s) as
                    initial trials (metrics ignored; Sobol init is NOT skipped).
  --max-trials N    cap total trials (default: unbounded, Ctrl+C to stop).

Usage
-----
  conda activate zombi-hop
  python optimize/run_mobo.py                                   # fresh RF, interactive
  python optimize/run_mobo.py --dataset ackley4d                # fresh 4D Ackley
  python optimize/run_mobo.py --dataset ackley10d --time-limit 0.5
  python optimize/run_mobo.py --dataset ensemble --dim 4        # re-randomized ensemble,
                                                                #   5 landscapes averaged/trial
  python optimize/run_mobo.py --dataset ensemble --dim 3 --ensemble-seed 7 --ensemble-margin 0.15
  python optimize/run_mobo.py --resume                          # seed from all past runs
  python optimize/run_mobo.py --resume-scratch                  # re-pick config + seed
  python optimize/run_mobo.py --resume-dim --dataset ackley4d   # seed from all past 4D runs
  python optimize/run_mobo.py --copy-config optimize/runs/mobo_05_06_15_32   # reuse config, no data
  python optimize/run_mobo.py --start-from-best optimize/runs/mobo_05_06_15_32/trial_112 [trial_dir ...]
  python optimize/make_videos.py                       # newest run
  python optimize/make_videos.py <run_dir>             # specific run
  python optimize/make_videos.py <run_dir> --force     # rebuild all

Non-interactive / MIT ORCD HPC
------------------------------
  Edit ``slurm/run_mobo_manual.sbatch`` (MOBO_CMD + #SBATCH headers), then:

    cd ~/ZoMBI-Hop && sbatch slurm/run_mobo_manual.sbatch

  Same flags as local CLI; always include ``--no-show``. Examples:

    # Fresh RF from batch JSON (--batch required with --config):
    python optimize/run_mobo.py --no-show --batch --config optimize/mobo_batch_configs/....json

    # Resume after failure:
    python optimize/run_mobo.py --no-show --resume --device cuda

    # Ackley 4D benchmark:
    python optimize/run_mobo.py --no-show --dataset ackley4d --device cuda

  ``scripts/submit_mobo.sh`` is optional; it only wraps ``--batch --config`` jobs.

Each trial appends to ``trials_log.csv`` (hyperparameters + metrics) and
``all_samples.csv`` (every ZoMBI sample point with trial/phase context).
``mobo_progress.json`` is still rewritten atomically after every trial.

``run_config.json`` also records ``device``, ``invocation`` (full argv, resolved
effective settings, and notes on which source won when CLI/batch/saved-config
values conflict), plus ``slurm`` metadata on HPC.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import random
import shlex
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


def _configure_mpl_backend(*, headless: bool) -> None:
    """Pick a matplotlib backend: respect MPLBACKEND, else Agg when headless."""
    if os.environ.get("MPLBACKEND"):
        matplotlib.use(os.environ["MPLBACKEND"])
    elif headless:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")


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

from optimize.mobo_landscapes import (
    LandscapeSpec,
    build_ackley_landscape,
    build_rf_landscape,
    build_synthetic_landscape,
    composition_column_names,
    infer_composition_columns,
    interactive_ackley_startup,
    landscape_from_run_config,
    parse_ackley_batch_fields,
    parse_synthetic_batch_fields,
)
from synthetic_data.ackley import Ackley
from synthetic_data.campaign_datasets import load_metadata, resolve_metadata_path
from synthetic_data.ensemble import (
    Ensemble,
    optima_count_range,
    random_ensemble_config,
)

DATASET_DIMS = {"RF": 3, "ackley3d": 3, "ackley4d": 4, "ackley10d": 10}
# Layered Ensemble objective: re-randomized per trial (dim from --dim), so it is
# not a fixed-dim entry in DATASET_DIMS.
ENSEMBLE_DATASET = "ensemble"
ENSEMBLE_DEFAULT_DIM = 3
DATASET_CHOICES = sorted([*DATASET_DIMS, ENSEMBLE_DATASET])

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

# For the re-randomized 'ensemble' dataset only: evaluate each hyperparameter set
# on this many independently randomized landscapes per MOBO trial and average the
# three metrics, to reduce the run-to-run landscape noise the MOBO model sees.
ENSEMBLE_N_REPEATS = 5

# Per-trial wall-clock budget (hours) passed to ZoMBIHop.run(time_limit_hours=…).
TIME_LIMIT_HOURS = 0.4

# Needle-match / duplicate thresholds: see optimize/eval_metrics.py (ILR, dim-scaled).

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


def unique_run_dir(parent: str, prefix: str) -> str:
    """Create a unique timestamped directory under *parent*, appending ``_2``,
    ``_3``, … if the base name already exists (prevents collisions between
    concurrent runs).

    The base name carries second resolution plus the PID so two runs launched in
    the same minute (e.g. from a sweep script) get distinct names without relying
    on the suffix loop. Creation is atomic — ``os.makedirs(exist_ok=False)`` fails
    if another process won the race — so two processes can never share a run directory.
    """
    base = datetime.datetime.now().strftime(f"{prefix}_%d_%m_%H_%M_%S") + f"_{os.getpid()}"
    n = 1
    while True:
        name = base if n == 1 else f"{base}_{n}"
        candidate = os.path.join(parent, name)
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            n += 1


# ─── Hyperparameter search space ──────────────────────────────────────────────
# Each entry: (lo, hi, transform) — transform ∈ {"log", "linear", "int"}
# Normalised to [0, 1] for MOBO; unnormalised when calling ZoMBI.

HPARAM_SPACE: dict[str, tuple] = {
    # Acquisition optimisation
    "nat_grad_step":               (0.001,  0.5,   "log"),
    "nat_grad_max_steps":          (10,     400,   "int"),
    "n_restarts":                  (20,     300,   "int"),
    "raw":                         (1,    300,  "int"),
    # Acquisition function
    "ucb_beta":                    (0.001,   3.0,   "linear"),
    # Zoom / convergence
    "max_zooms":                   (2,      10,    "int"),
    "max_iterations":              (2,      30,    "int"),
    "top_m_points":                (2,      8,     "int"),
    "n_consecutive_converged":     (1,      5,    "int"),
    "convergence_pi_threshold":    (1e-4,   0.05,  "log"),
    "input_noise_threshold_mult":  (0.5,    6.0,   "linear"),
    "output_noise_threshold_mult": (0.01,    2.0,   "linear"),
    # Penalisation & needle
    "max_penalty_radius":          (0.01,    5.0,   "linear"),
    "needle_shrink_factor":        (0.1,   0.99,  "linear"),
    "needle_stop_noise_multiplier":(1.0,    8.0,   "linear"),
    # Point paring (deduplication)
    "paring_spatial_halfnoise":    (0.1,    5.0,   "linear"),
    "paring_y_noise_multiplier":   (0.1,    10.0,   "linear"),
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

def build_rf_and_grid(
    csv_path: str,
    objective_column: str = "Objective",
    composition_columns: list[str] | None = None,
):
    """Train the RF surrogate from a campaign or synthetic CSV.

    Deterministic (fixed random_state / tree count), so a resumed run rebuilds an
    identical surrogate without re-prompting the interactive extrema picker.
    Returns (rf, rf_fn, grid_pts, grid_vals, composition_columns, dim).
    """
    df = pd.read_csv(csv_path)
    comp_cols = infer_composition_columns(df, explicit=composition_columns)
    dim = len(comp_cols)
    df = df.dropna(subset=comp_cols + [objective_column])
    X_data = df[comp_cols].values.astype(float)
    X_data /= X_data.sum(axis=1, keepdims=True)
    y_data = df[objective_column].values.astype(float)
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X_data, y_data)
    if dim == 3:
        grid_pts = ternary_grid(TERNARY_GRID_N)
        grid_vals = rf.predict(grid_pts)
    else:
        grid_pts = grid_vals = None
    rf_fn = lambda x, _rf=rf: float(_rf.predict(x.reshape(1, -1))[0])
    return rf, rf_fn, grid_pts, grid_vals, comp_cols, dim


def _load_rf_batch_extras(cfg: dict, csv_path: str, repo_root: str) -> dict:
    """Merge optional synthetic metadata sidecar into a batch RF config."""
    meta_path = cfg.get("metadata_path")
    if meta_path and not os.path.isabs(meta_path):
        meta_path = os.path.normpath(os.path.join(repo_root, meta_path))
    resolved = resolve_metadata_path(csv_path, meta_path)
    extras = {
        "composition_columns": cfg.get("composition_columns"),
        "oracle": cfg.get("oracle"),
        "metadata_path": resolved,
        "maximize": bool(cfg.get("maximize", False)),
        "true_optima": [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
        if cfg.get("true_optima") else [],
    }
    if not resolved:
        return extras
    try:
        meta = load_metadata(resolved)
    except Exception as exc:
        print(f"  [batch] WARNING: metadata unreadable ({resolved}): {exc}")
        return extras
    if not extras["composition_columns"] and meta.get("composition_columns"):
        extras["composition_columns"] = list(meta["composition_columns"])
    if not extras["oracle"] and meta.get("oracle"):
        extras["oracle"] = str(meta["oracle"])
    if not extras["true_optima"] and meta.get("true_optima"):
        extras["true_optima"] = [np.asarray(t, dtype=float) for t in meta["true_optima"]]
    if "maximize" not in cfg and "maximize" in meta:
        extras["maximize"] = bool(meta["maximize"])
    return extras


# ─── Run-config persistence + resume ────────────────────────────────────────────

def _slurm_metadata() -> dict:
    """Capture Slurm / host context for reproducibility (best-effort)."""
    keys = (
        "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID",
        "SLURM_JOB_NAME", "SLURM_CLUSTER_NAME", "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE", "SLURM_JOB_PARTITION", "HOSTNAME",
    )
    meta = {k.lower(): os.environ[k] for k in keys if os.environ.get(k)}
    meta["submitted_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return meta


def _cli_snapshot(args: argparse.Namespace) -> dict:
    """Non-default CLI flags actually passed on the command line."""
    snap: dict = {}
    for key, val in vars(args).items():
        if val is None or val == []:
            continue
        if val is False and key not in ("no_show", "batch", "resume", "resume_scratch"):
            continue
        if key == "dataset" and val == "RF":
            continue
        if key == "landscape" and val == "rf":
            continue
        if key == "ackley_variant" and val == "realistic":
            continue
        snap[key] = val
    return snap


def _effective_run_settings(
    landscape: LandscapeSpec,
    *,
    max_trials: int | None,
    n_init_trials: int,
    runs_dir: str,
    run_dir: str | None = None,
) -> dict:
    return {
        "device": str(DEVICE),
        "time_limit_hours": landscape.time_limit_hours,
        "max_activations": landscape.max_activations,
        "max_trials": max_trials,
        "n_init_trials": n_init_trials,
        "landscape": landscape.landscape,
        "landscape_label": landscape.label,
        "dim": landscape.dim,
        "runs_dir": runs_dir,
        "run_dir": run_dir,
    }


def build_invocation_log(
    *,
    argv: list[str],
    run_mode: str,
    cli: dict,
    effective: dict,
    resolutions: list[str] | None = None,
) -> dict:
    """Full snapshot of how this run was launched and what values were actually used."""
    device_requested = cli.get("device")
    return {
        "argv": list(argv),
        "command": shlex.join(argv),
        "run_mode": run_mode,
        "cli": cli,
        "effective": effective,
        "device_requested": device_requested,
        "device_effective": effective.get("device", str(DEVICE)),
        "cuda_available": torch.cuda.is_available(),
        "resolutions": resolutions or [],
        "hparam_space": {k: list(v) for k, v in HPARAM_SPACE.items()},
    }


def _log_invocation(invocation: dict) -> None:
    eff = invocation["effective"]
    print(f"  [invocation] mode={invocation['run_mode']}  "
          f"device={eff.get('device')}  "
          f"time_limit_hours={eff.get('time_limit_hours')}  "
          f"max_trials={eff.get('max_trials')}", flush=True)
    for note in invocation.get("resolutions", []):
        print(f"  [invocation] {note}", flush=True)


def write_run_config(run_dir, landscape: LandscapeSpec, *,
                     batch_name: str | None = None,
                     batch_config_path: str | None = None,
                     n_init_trials: int = N_INIT_TRIALS,
                     dataset: str | None = None,
                     ackley_variant: str | None = None,
                     ensemble_spec: dict | None = None,
                     invocation: dict | None = None) -> None:
    """Persist the static run state needed for a fully non-interactive resume."""
    cfg = {
        "landscape":       landscape.landscape,
        "dim":             landscape.dim,
        "maximize":        bool(landscape.maximize),
        "true_optima":     [list(map(float, np.asarray(t).ravel())) for t in landscape.true_optima],
        "n_init_trials":   n_init_trials,
        "hparam_names":    HPARAM_NAMES,
        "time_limit_hours": landscape.time_limit_hours,
        "max_activations": landscape.max_activations,
        "created":         datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if dataset is not None:
        cfg["dataset"] = dataset
    if ackley_variant is not None:
        cfg["ackley_variant"] = ackley_variant
    if landscape.landscape == "rf":
        cfg["csv_path"] = landscape.csv_path
        cfg["objective_column"] = landscape.objective_column
        if landscape.composition_columns:
            cfg["composition_columns"] = landscape.composition_columns
        if landscape.oracle:
            cfg["oracle"] = landscape.oracle
        if landscape.metadata_path:
            cfg["metadata_path"] = landscape.metadata_path
    if landscape.landscape == "ackley":
        cfg["ackley_layout"] = landscape.ackley_layout
        cfg["ackley_b"] = landscape.ackley_b
    if landscape.landscape == "synthetic":
        cfg["oracle"] = landscape.oracle
        cfg["ackley_layout"] = landscape.ackley_layout
        if landscape.synthetic_seed is not None:
            cfg["seed"] = landscape.synthetic_seed
    if ensemble_spec is not None:
        cfg["ensemble_random_per_run"] = True
        cfg["ensemble_seed"] = ensemble_spec["seed"]
        cfg["ensemble_optima_margin"] = ensemble_spec["optima_margin"]
    if batch_name:
        cfg["batch_name"] = batch_name
    if batch_config_path:
        cfg["batch_config_path"] = os.path.abspath(batch_config_path)
    cfg["device"] = str(DEVICE)
    if invocation is not None:
        cfg["invocation"] = invocation
    from synthetic_data.landscape_config_log import build_landscape_config_log, dataset_label_for_landscape
    cfg["landscape_config"] = build_landscape_config_log(
        dataset=dataset_label_for_landscape(
            landscape.landscape, dim=landscape.dim, oracle=landscape.oracle,
        ),
        ds={
            "dim": landscape.dim,
            "oracle": landscape.oracle,
            "landscape": landscape.landscape,
            "fn": landscape.fn_callable,
            "csv_path": landscape.csv_path,
            "metadata_path": landscape.metadata_path,
            "seed": landscape.synthetic_seed,
            "layout": landscape.ackley_layout,
        },
    )
    cfg["slurm"] = _slurm_metadata()
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


def auto_detect_rf_optima(
    rf: RandomForestRegressor,
    grid_pts: np.ndarray,
    grid_vals: np.ndarray,
    *,
    maximize: bool,
    n_peaks: int = 3,
    min_sep: float = 0.08,
) -> list[np.ndarray]:
    """Greedy top-grid peak picking + L-BFGS-B refinement (headless picker)."""
    order = np.argsort(grid_vals)
    if not maximize:
        order = order[::-1]
    chosen: list[np.ndarray] = []
    for idx in order:
        if len(chosen) >= n_peaks:
            break
        pt = grid_pts[idx]
        if any(float(np.linalg.norm(pt - c)) < min_sep for c in chosen):
            continue
        x_ref, _ = _refine_extremum(rf, pt, maximize=maximize)
        if any(float(np.linalg.norm(x_ref - c)) < min_sep for c in chosen):
            continue
        chosen.append(x_ref)
    if not chosen:
        chosen = [np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)]
    return chosen


def load_batch_config(path: str, script_dir: str) -> dict:
    """Load a headless run config JSON (paths resolved relative to repo root)."""
    cfg_path = os.path.abspath(path)
    if not os.path.exists(cfg_path):
        sys.exit(f"--config: file not found: {cfg_path}")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as exc:
        sys.exit(f"--config: {cfg_path} unreadable ({exc}).")

    repo_root = os.path.normpath(os.path.join(script_dir, ".."))
    landscape_type = cfg.get("landscape", "rf")
    maximize = bool(cfg.get("maximize", False))
    time_limit = cfg.get("time_limit_hours")

    if landscape_type == "ackley":
        try:
            ack = parse_ackley_batch_fields(cfg)
        except ValueError as exc:
            sys.exit(f"--config: {exc}")
        landscape = build_ackley_landscape(
            ack["dim"], ack["layout"], b=ack["ackley_b"],
            time_limit_hours=time_limit,
            max_activations=ack["max_activations"],
        )
        print(f"  [batch] Multi-Ackley d={ack['dim']} layout={ack['layout']} b={ack['ackley_b']}")
        print(f"  [batch] {len(landscape.true_optima)} planted peaks  |  "
              f"max_activations={landscape.max_activations}")
        return {
            "name":              cfg.get("name") or os.path.splitext(os.path.basename(cfg_path))[0],
            "landscape":         landscape,
            "maximize":          True,
            "true_optima":       landscape.true_optima,
            "max_trials":        cfg.get("max_trials"),
            "time_limit_hours":  time_limit,
            "n_init_trials":     cfg.get("n_init_trials", N_INIT_TRIALS),
            "auto_optima":       None,
            "config_path":       cfg_path,
            "csv_path":          None,
            "objective_column":  None,
        }

    if landscape_type == "synthetic":
        try:
            syn = parse_synthetic_batch_fields(cfg)
        except ValueError as exc:
            sys.exit(f"--config: {exc}")
        landscape = build_synthetic_landscape(
            syn["oracle"], syn["dim"], syn["layout"],
            seed=syn["seed"],
            time_limit_hours=time_limit,
        )
        maximize = bool(cfg.get("maximize", True))
        print(f"  [batch] Synthetic oracle={syn['oracle']} d={syn['dim']} "
              f"layout={syn['layout']} seed={syn['seed']}")
        print(f"  [batch] {len(landscape.true_optima)} planted peaks (direct oracle, no RF)")
        return {
            "name":              cfg.get("name") or os.path.splitext(os.path.basename(cfg_path))[0],
            "landscape":         landscape,
            "maximize":          maximize,
            "true_optima":       landscape.true_optima,
            "max_trials":        cfg.get("max_trials"),
            "time_limit_hours":  time_limit,
            "n_init_trials":     cfg.get("n_init_trials", N_INIT_TRIALS),
            "auto_optima":       None,
            "config_path":       cfg_path,
            "csv_path":          None,
            "objective_column":  None,
        }

    csv_path = cfg.get("csv_path")
    if not csv_path:
        sys.exit(f"--config: {cfg_path} must include 'csv_path' for RF landscape.")
    if not os.path.isabs(csv_path):
        csv_path = os.path.normpath(os.path.join(repo_root, csv_path))
    if not os.path.exists(csv_path):
        sys.exit(f"--config: csv_path does not exist: {csv_path}")

    rf_extras = _load_rf_batch_extras(cfg, csv_path, repo_root)
    composition_columns = rf_extras.get("composition_columns")
    oracle = rf_extras.get("oracle")
    metadata_path = rf_extras.get("metadata_path")
    maximize = rf_extras["maximize"]

    true_optima: list[np.ndarray] = list(rf_extras.get("true_optima") or [])
    if cfg.get("true_optima"):
        true_optima = [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
    elif not true_optima and cfg.get("regions_objective"):
        regions_path = cfg.get("regions_path", "scripts/max_min_regions.json")
        if not os.path.isabs(regions_path):
            regions_path = os.path.normpath(os.path.join(repo_root, regions_path))
        if not os.path.exists(regions_path):
            sys.exit(f"--config: regions_path not found: {regions_path}")
        with open(regions_path) as f:
            regions = json.load(f)
        obj_key = cfg["regions_objective"]
        obj = regions.get("objectives", {}).get(obj_key)
        if obj is None:
            sys.exit(f"--config: regions_objective '{obj_key}' not in {regions_path}")
        seed_key = "max_seeds" if maximize else "min_seeds"
        seeds = obj.get(seed_key) or []
        if not seeds:
            sys.exit(f"--config: no {seed_key} for '{obj_key}' in {regions_path}")
        true_optima = [np.asarray(s, dtype=float) for s in seeds]
        print(f"  [batch] loaded {len(true_optima)} reference optima from "
              f"{obj_key}/{seed_key}")

    return {
        "name":              cfg.get("name") or os.path.splitext(os.path.basename(cfg_path))[0],
        "landscape":         None,
        "csv_path":          csv_path,
        "objective_column":  cfg.get("objective_column", "Objective"),
        "composition_columns": composition_columns,
        "oracle":            oracle,
        "metadata_path":     metadata_path,
        "maximize":          maximize,
        "true_optima":       true_optima,
        "max_trials":        cfg.get("max_trials"),
        "time_limit_hours":  time_limit,
        "n_init_trials":     cfg.get("n_init_trials", N_INIT_TRIALS),
        "auto_optima":       cfg.get("auto_optima"),
        "config_path":       cfg_path,
    }


def _append_csv_rows(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Append rows to a CSV, writing the header when the file is new."""
    if not rows:
        return
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def append_trial_logs(
    run_dir: str,
    trial_num: int,
    phase: str,
    hparams: dict,
    metrics: dict,
    trial_dir: str,
    *,
    dim: int = 3,
) -> None:
    """Append one trial row + all sample points for live analysis on HPC."""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    trial_row = {
        "timestamp": ts,
        "trial": trial_num,
        "phase": phase,
        "dist_to_needles": round(metrics["dist"], 6),
        "dup_fraction": round(metrics["dup"], 6),
        "runtime_s": round(metrics["runtime"], 3),
    }
    for k, v in hparams.items():
        trial_row[f"hp_{k}"] = v
    hp_cols = [f"hp_{k}" for k in HPARAM_NAMES]
    _append_csv_rows(
        os.path.join(run_dir, "trials_log.csv"),
        ["timestamp", "trial", "phase", "dist_to_needles", "dup_fraction", "runtime_s"]
        + hp_cols,
        [trial_row],
    )

    points_path = os.path.join(trial_dir, "points.csv")
    if not os.path.exists(points_path):
        return
    comp_cols = composition_column_names(dim)
    sample_rows: list[dict] = []
    with open(points_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample_row = {
                "timestamp": ts,
                "trial": trial_num,
                "phase": phase,
                "sample_idx": row.get("sample_idx"),
                "Y": row.get("Y"),
                "penalized": row.get("penalized"),
                "activation": row.get("activation"),
                "zoom": row.get("zoom"),
            }
            for col in comp_cols:
                sample_row[col] = row.get(col)
            sample_rows.append(sample_row)
    _append_csv_rows(
        os.path.join(run_dir, "all_samples.csv"),
        ["timestamp", "trial", "phase", "sample_idx"] + comp_cols
        + ["Y", "penalized", "activation", "zoom"],
        sample_rows,
    )


def _time_objective(metrics: dict) -> float:
    """Third MOBO objective: average wall-clock seconds per ZoMBI iteration.

    Prefers the current ``avg_time_per_iter_s`` key. Older runs stored total
    ``runtime_s`` instead (the objective used to be whole-trial wall clock), so it
    is used as a fallback to keep prior data seedable on resume — note such values
    are totals, not per-iteration, so a resume mixing old and new trials is only
    approximate on this objective.
    """
    if "avg_time_per_iter_s" in metrics:
        return float(metrics["avg_time_per_iter_s"])
    return float(metrics["runtime_s"])


def _collect_from_progress(path: str, X_obs: list, Y_obs: list) -> int:
    """Load (X_norm, Y) pairs from one mobo_progress.json; append to X_obs/Y_obs.

    Returns the number of usable trials found.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  [collect] {path} unreadable ({exc}); skipping.")
        return 0
    used = 0
    for t in data.get("trials", []):
        hp, m = t.get("hparams", {}), t.get("metrics", {})
        if not all(name in hp for name in HPARAM_NAMES):
            continue  # stale / different hyperparameter set
        try:
            x = hparams_to_norm(hp)
            y = torch.tensor([-float(m["dist_to_needles"]),
                              -float(m["dup_fraction"]),
                              -_time_objective(m)], dtype=DTYPE)
        except (KeyError, ValueError, TypeError):
            continue
        X_obs.append(x)
        Y_obs.append(y)
        used += 1
    return used


def collect_all_observations(runs_dir: str):
    """Crawl every runs/mobo_*/mobo_progress.json and collect all (X_norm, Y) pairs.

    X = normalised hyperparameter vector (inverted from the stored hparams),
    Y = (-dist_to_needles, -dup_fraction, -avg_time_per_iter_s)  [maximised objectives].

    Each run's progress.json records only its own trials, so the union across all
    runs has no double-counting.  Trials whose hparam keys don't cover the current
    HPARAM_SPACE are skipped (stale hyperparameter set).
    Returns (X_obs, Y_obs, n_runs).
    """
    X_obs, Y_obs = [], []
    n_runs = 0
    for path in sorted(glob.glob(os.path.join(runs_dir, "mobo_*", "mobo_progress.json"))):
        used = _collect_from_progress(path, X_obs, Y_obs)
        if used:
            n_runs += 1
            print(f"  [collect] {os.path.basename(os.path.dirname(path))}: {used} trial(s)")
    return X_obs, Y_obs, n_runs


def _read_needles_csv(path: str, dim: int = 3) -> np.ndarray:
    """Read a trial's needles.csv → (N, dim) composition array."""
    import csv as _csv
    cols = composition_column_names(dim)
    pts = []
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            pts.append([float(row[c]) for c in cols])
    return np.array(pts, dtype=float) if pts else np.empty((0, dim))


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
        try:
            with open(os.path.join(rd, "run_config.json")) as f:
                run_dim = int(json.load(f).get("dim", 3))
        except Exception:
            run_dim = 3
        for t in data.get("trials", []):
            hp, m = t.get("hparams", {}), t.get("metrics", {})
            if not all(name in hp for name in HPARAM_NAMES):
                continue  # stale / different hyperparameter set
            needles_path = os.path.join(rd, f"trial_{t.get('trial')}", "needles.csv")
            try:
                if os.path.exists(needles_path):
                    needles = _read_needles_csv(needles_path, dim=run_dim)
                    dist = metric_dist_to_needles(needles, new_optima, dim=run_dim)
                    rederived = True
                else:
                    # No saved needles → cannot re-score; reuse stored dist as-is.
                    dist = float(m["dist_to_needles"])
                    rederived = False
                x = hparams_to_norm(hp)
                y = torch.tensor([-float(dist),
                                  -float(m["dup_fraction"]),
                                  -_time_objective(m)], dtype=DTYPE)
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


def collect_observations_from_run(run_dir: str):
    """Load (X_norm, Y) pairs from a single run's mobo_progress.json.

    Like ``collect_all_observations`` but scoped to one run directory. Used by
    ``--resume-from`` to trust only a specific run's stored metrics without
    re-evaluating anything.
    Returns (X_obs, Y_obs).
    """
    path = os.path.join(run_dir, "mobo_progress.json")
    if not os.path.exists(path):
        sys.exit(f"--resume-from: no mobo_progress.json found in {run_dir}")
    X_obs, Y_obs = [], []
    used = _collect_from_progress(path, X_obs, Y_obs)
    print(f"  [collect] {os.path.basename(run_dir)}: {used} trial(s)")
    return X_obs, Y_obs


def collect_observations_for_dim(runs_dir: str, dim: int):
    """Crawl every runs/mobo_* run whose saved config dimension matches ``dim`` and
    collect all (X_norm, Y) pairs from their mobo_progress.json.

    Used by ``--resume-dim`` to seed a new run with the union of all prior runs of
    the SAME dimensionality (e.g. every past 4-simplex run), trusting their stored
    metrics without re-evaluation. Runs of a different dimension — or whose config
    is missing/unreadable — are skipped, since their objective is incomparable.
    Returns (X_obs, Y_obs, n_runs).
    """
    X_obs, Y_obs = [], []
    n_runs = 0
    for cfg_path in sorted(glob.glob(os.path.join(runs_dir, "mobo_*", "run_config.json"))):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception as exc:
            print(f"  [collect] {cfg_path} unreadable ({exc}); skipping.")
            continue
        if int(cfg.get("dim", -1)) != int(dim):
            continue
        prog = os.path.join(os.path.dirname(cfg_path), "mobo_progress.json")
        if not os.path.exists(prog):
            continue
        used = _collect_from_progress(prog, X_obs, Y_obs)
        if used:
            n_runs += 1
            print(f"  [collect] {os.path.basename(os.path.dirname(cfg_path))}: {used} trial(s)")
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


def load_seed_hparams(trial_paths: list[str]) -> list[torch.Tensor]:
    """Load hyperparameter vectors from one or more trial_* dirs → init design.

    Each path is a trial directory (or its trial.json directly) produced by a
    prior run. For "start from best", only the HYPERPARAMETERS are read; their
    stored metrics are ignored. Each becomes a normalised design point that the
    run RE-EVALUATES as a real initial trial (alongside — not instead of — the
    full Sobol init), so the GP learns these known-good configs under the current
    objective/dataset rather than trusting a copied score. Trials whose hparams
    don't cover the current HPARAM_SPACE abort the run (stale hyperparameter set).
    Returns a list of normalised [0,1] hyperparameter vectors.
    """
    X_seed: list[torch.Tensor] = []
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
        X_seed.append(hparams_to_norm(hp))
        print(f"  [seed] {p}  (trial {data.get('trial', '?')}) — will be re-evaluated")
    return X_seed


# ─── Ternary helpers (RF interactive picker + plotting) ────────────────────────

def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    p = as_numpy(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def _line_endpoints_array(line) -> np.ndarray | None:
    """Stack a LineBO (left, right) pair as (2, d) host array."""
    if line is None:
        return None
    left, right = line
    return np.stack([as_numpy(left).ravel(), as_numpy(right).ravel()], axis=0)


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


# ─── Metrics (dimension-aware; see optimize/eval_metrics.py) ───────────────────

from eval_metrics import (  # noqa: E402  — after sys.path setup in callers
    MATCH_RADIUS,
    UNMATCHED_PENALTY,
    as_numpy,
    dup_threshold_ilr,
    match_radius_ilr,
    metric_avg_pairwise_dist,
    metric_dist_to_needles,
    metric_dup_fraction,
    metric_pct_matched,
    metric_pct_matched_comp,
    metric_pct_matched_ilr,
)
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
            (as_numpy(x_left_r[0]), as_numpy(x_right_r[0])) if n_valid > 0 else None
        )
        plot_state["line_1"] = (
            (as_numpy(x_left_r[1]), as_numpy(x_right_r[1])) if n_valid > 1 else None
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


def _gen_init_data(fn_callable, maximize: bool, dim: int = 3):
    """Generate ``N_INIT_LINES`` random simplex lines on the ``dim``-simplex;
    return (X_a, X_e, Y)."""
    x_a_list, x_e_list, y_list = [], [], []
    for _ in range(N_INIT_LINES):
        x0 = torch.full((dim,), 1.0 / dim, device=DEVICE, dtype=DTYPE)
        dir_ = zero_sum_dirs(1, dim, device=DEVICE, dtype=DTYPE).squeeze(0)
        seg  = line_simplex_segment(x0, dir_)
        if seg is None:
            continue
        _, _, x_left, x_right = seg
        t     = torch.linspace(0.0, 1.0, NUM_EXPERIMENTS, dtype=torch.float64, device=DEVICE)
        pts_t = (x_left.to(torch.float64).unsqueeze(0)
                 + t.unsqueeze(1) * (x_right - x_left).to(torch.float64).unsqueeze(0))
        z     = composition_to_ilr(pts_t)
        z     = z + torch.randn_like(z) * NOISE_LEVEL_ILR
        pts_t = ilr_to_composition(z, d=dim)
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
        elif isinstance(bounds, np.ndarray) and bounds.shape[0] == 2:
            lo = torch.as_tensor(bounds[0], dtype=DTYPE, device=DEVICE)
            hi = torch.as_tensor(bounds[1], dtype=DTYPE, device=DEVICE)
        else:
            return
        samp = random_simplex(n_sample, lo, hi, device=str(lo.device), torch_dtype=lo.dtype)
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
        M_np = as_numpy(M, dtype=float)
        eigvals, eigvecs = np.linalg.eigh(M_np)
        eigvals = np.maximum(eigvals, 1e-12)
        angles = np.linspace(0, 2 * np.pi, 200)
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        u_ell = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ circle.T).T
        if B is not None:
            B_np = as_numpy(B, dtype=float)
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
                 out_path: str, *, ref_title: str = "Reference: RF landscape") -> None:
    """Render one two-panel ternary iteration figure to out_path (no GUI)."""
    fig = Figure(figsize=(16, 6.8))
    FigureCanvasAgg(fig)
    ax_ref, ax_exp = fig.subplots(1, 2)
    fig.suptitle(f"ZoMBI-Hop MOBO trial — iteration {payload['iter_num']}", fontsize=13)

    # ── Left: RF reference ──
    draw_ternary_frame(ax_ref)
    ax_ref.set_title(ref_title, fontsize=11)
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
        ll = comp_to_xy(_line_endpoints_array(payload["line_0"]))
        (h0,) = ax_exp.plot(ll[:, 0], ll[:, 1], "-", color="orange", lw=2.5,
                            alpha=0.90, zorder=7, label="LineBO (main)")
        legend_handles.append(h0)
    if payload.get("line_1") is not None:
        ll = comp_to_xy(_line_endpoints_array(payload["line_1"]))
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


def plot_convergence(path: str, dh, maximize: bool) -> None:
    """Save a convergence plot: all Y values, running best, needle vlines."""
    Y_all = dh.Y_all.detach().cpu().numpy().ravel()
    if Y_all.size == 0:
        return
    mask = dh.get_penalty_mask()
    pm = mask.detach().cpu().numpy().ravel() if mask is not None else None
    needle_indices = (dh.needle_indices.detach().cpu().numpy().ravel()
                      if dh.needle_indices is not None and dh.needle_indices.numel() > 0
                      else None)

    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(Y_all))

    if pm is not None and pm.any():
        ax.scatter(idx[~pm], Y_all[~pm], s=10, alpha=0.35, color="#aaaaaa",
                   label="penalized", zorder=2)
        ax.scatter(idx[pm], Y_all[pm], s=10, alpha=0.65, color="steelblue",
                   label="valid", zorder=3)
        running_best = np.maximum.accumulate(np.where(pm, Y_all, -np.inf))
    else:
        ax.scatter(idx, Y_all, s=10, alpha=0.65, color="steelblue",
                   label="obs", zorder=2)
        running_best = np.maximum.accumulate(Y_all)

    ax.plot(idx, running_best, color="darkorange", lw=1.8,
            label="running best", zorder=4)

    if needle_indices is not None:
        labeled = False
        for ni in needle_indices:
            if 0 <= ni < len(Y_all):
                kw = dict(color="crimson", alpha=0.55, lw=0.9, ls="--")
                if not labeled:
                    kw["label"] = "needle found"
                    labeled = True
                ax.axvline(float(ni), **kw)

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Objective Y")
    ax.set_title(f"Convergence  ({len(Y_all)} pts, "
                 f"{len(needle_indices) if needle_indices is not None else 0} needles)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def write_points_csv(path: str, dh, snap_records: list[tuple], *, dim: int = 3) -> None:
    X = dh.X_all_actual.detach().cpu().numpy()
    Y = dh.Y_all.detach().cpu().numpy().ravel()
    n = X.shape[0]
    mask = dh.get_penalty_mask()
    penalized = (~mask.detach().cpu().numpy()) if mask is not None else np.zeros(n, bool)
    act, zm = _activation_zoom_per_point(n, snap_records)
    comp_cols = composition_column_names(dim)
    data = {
        "sample_idx": np.arange(n),
        "Y": Y,
        "penalized": penalized.astype(int),
        "activation": act,
        "zoom": zm,
    }
    for i, col in enumerate(comp_cols):
        data[col] = X[:, i]
    pd.DataFrame(data).to_csv(path, index=False)


def write_needles_csv(path: str, dh, *, dim: int = 3) -> None:
    centroid = np.full(dim, 1.0 / dim)
    comp_cols = composition_column_names(dim)
    rows = []
    for i, r in enumerate(dh.get_all_needle_results()):
        pt = r["point"].detach().cpu().numpy().ravel()
        mv = r.get("median_value")
        row = {
            "needle_idx": i,
            "value": r.get("value"),
            "median_value": (None if mv is None or (isinstance(mv, float) and math.isnan(mv)) else mv),
            "activation": r.get("activation"),
            "zoom": r.get("zoom"),
            "iteration": r.get("iteration"),
            "reason": r.get("reason"),
            "dist_to_centre": float(np.linalg.norm(pt - centroid)),
        }
        for j, col in enumerate(comp_cols):
            row[col] = pt[j]
        rows.append(row)
    cols = ["needle_idx"] + comp_cols + [
        "value", "median_value", "activation", "zoom", "iteration", "reason", "dist_to_centre",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def write_metrics_over_time_csv(path: str, payloads: list[dict], X_all: np.ndarray,
                                true_optima: list[np.ndarray], *, dim: int = 3) -> None:
    rows = []
    for p in payloads:
        needles = p.get("needles")
        if needles is not None:
            disc = as_numpy(needles, dtype=float)
        else:
            disc = np.empty((0, X_all.shape[1] if X_all.ndim == 2 and X_all.shape[0] else (true_optima[0].shape[0] if true_optima else dim)))
        n_before = p.get("n_points_before", len(X_all))
        X_upto = as_numpy(X_all[:n_before], dtype=float) if n_before > 0 else np.empty_like(disc)
        # Value of the most recently discovered needle as of this iteration
        # (needles accumulate in chronological order, so the last one is newest).
        nvals = p.get("needle_vals")
        recent = float(nvals[-1]) if nvals is not None and len(nvals) > 0 else np.nan
        rows.append({
            "iteration": p["iter_num"],
            "dist_to_needles":  round(metric_dist_to_needles(disc, true_optima, dim=dim), 6),
            "dup_fraction":     round(metric_dup_fraction(X_upto, dim=dim), 6),
            "pct_matched_comp": round(metric_pct_matched_comp(disc, true_optima, dim=dim), 4),
            "pct_matched_ilr":  round(metric_pct_matched_ilr(disc, true_optima, dim=dim), 4),
            "pct_matched":      round(metric_pct_matched_comp(disc, true_optima, dim=dim), 4),
            "avg_pairwise_dist":round(metric_avg_pairwise_dist(disc), 6),
            "recent_needle_value": (round(recent, 6) if not math.isnan(recent) else np.nan),
        })
    cols = ["iteration", "dist_to_needles", "dup_fraction",
            "pct_matched_comp", "pct_matched_ilr", "pct_matched",
            "avg_pairwise_dist", "recent_needle_value"]
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
            left, right = as_numpy(line[0]).ravel(), as_numpy(line[1]).ravel()
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
                     metrics: dict, hparams: dict,
                     ackley_seed: int | None = None,
                     ensemble_config: dict | None = None,
                     ensemble_configs: list[dict] | None = None,
                     repeats: list[dict] | None = None) -> None:
    obj = {
        "trial": trial_num,
        "phase": phase,
        "metrics": {
            "dist_to_needles":     round(metrics["dist"], 6),
            "dup_fraction":        round(metrics["dup"], 6),
            "avg_time_per_iter_s": round(metrics["avg_time_per_iter"], 4),
            "runtime_s":           round(metrics["runtime"], 3),
            "n_iters":             int(metrics["n_iters"]),
        },
        "hparams": {
            k: (round(v, 8) if isinstance(v, float) else v) for k, v in hparams.items()
        },
    }
    if ackley_seed is not None:
        obj["ackley_seed"] = int(ackley_seed)
    if ensemble_config is not None:
        obj["ensemble_config"] = ensemble_config
    if ensemble_configs is not None:
        # Averaged ensemble trial: metrics above are the mean over these landscapes.
        obj["ensemble_n_repeats"] = len(ensemble_configs)
        obj["ensemble_configs"] = ensemble_configs
    if repeats is not None:
        obj["repeats"] = repeats
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)



# ─── End-of-trial auto plots (plot_metrics + coverage_plot; no videos) ─────────

def _auto_generate_plots(trial_dir: str, dim: int) -> None:
    """Generate the metrics time-series plot (all dims) and, for 3D, the static
    coverage plot — automatically, at the end of a trial, with no videos and no
    interactive windows. (4D's landscape is the interactive point cloud written
    separately by ``_render_4d_point_cloud``; ≥5D has no landscape.) Best-effort:
    a plotting failure never fails the trial."""
    # Metrics time series (dimension-independent).
    try:
        import plot_metrics
        csv_path = os.path.join(trial_dir, "metrics_over_time.csv")
        if os.path.isfile(csv_path):
            plot_metrics.plot_metrics(
                csv_path, save_path=os.path.join(trial_dir, "metrics_over_time.png"))
    except Exception as exc:
        print(f"    [trial] plot_metrics failed: {exc}")
    # Static coverage plot: ternary (3D) only. 4D uses the interactive point cloud.
    if dim == 3:
        try:
            import coverage_plot
            coverage_plot.save_coverage_image(trial_dir)
        except Exception as exc:
            print(f"    [trial] coverage_plot failed: {exc}")


def _render_4d_point_cloud(out_path: str, ackley_fn, dh, last_payload: dict | None) -> None:
    """Write one interactive (rotatable) 4-simplex point-cloud HTML for a 4D trial.

    Mirrors ``evaluate.write_point_cloud_html``: the final ZoMBI state (pared
    points, discovered needles, last LineBO lines) drawn over the negated-Ackley
    objective cloud on the tetrahedron, using ``synthetic_data/plot``'s overlay
    API. The only interactivity is 3D rotation (Plotly Scatter3d). Needle
    penalisation ellipsoids are omitted (the run's tangent basis differs from the
    Helmert ILR basis the overlay assumes). Best-effort: needs plotly.
    """
    import plotly.graph_objects as go
    import synthetic_data.plot_ackley as pc4

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

    # Overlay the final data-handler state.
    pared_X = pared_Y = recency = None
    if dh.X_pared is not None and dh.X_pared.shape[0] > 0:
        pared_X = dh.X_pared.detach().cpu().numpy()
        pared_Y = dh.Y_pared.detach().cpu().numpy().ravel()
        recency = np.arange(pared_X.shape[0], dtype=float)

    needle_t = dh.get_all_needle_locations()
    needles  = (needle_t.detach().cpu().numpy()
                if needle_t is not None and needle_t.numel() > 0 else None)

    main_line  = (_line_endpoints_array(last_payload["line_0"]) if last_payload
                  and last_payload.get("line_0") is not None else None)
    cache_line = (_line_endpoints_array(last_payload["line_1"]) if last_payload
                  and last_payload.get("line_1") is not None else None)

    pc4.add_simplex_overlays(
        fig, obj_cmin=obj_min, obj_cmax=obj_max,
        pared_points=pared_X, pared_values=pared_Y, recency=recency,
        main_line=main_line, cache_line=cache_line, needles=needles,
    )
    fig.write_html(out_path, include_plotlyjs="cdn", auto_open=False)


# ─── Per-trial realistic-Ackley reseeding ──────────────────────────────────────

def draw_ackley_seed() -> int:
    """Fresh entropy-seeded value in [0, 2**31) for one trial's realistic Ackley."""
    return int(np.random.default_rng().integers(0, 2**31 - 1))


def reseed_ackley_dataset(ds: dict, seed: int) -> dict:
    """Return a shallow copy of an Ackley dataset bundle rebuilt for ``seed``.

    The seed drives BOTH the optima placement (peak_seed) and the background noise
    (noise_seed), so each trial faces a fresh random realisation of the
    realistic-Ackley landscape. The function, its known optima, and — for 3D — the
    render-grid values are recomputed to match (4D renders from ``fn`` directly;
    ≥5D has no landscape view, so ``grid_vals`` stays ``None``).
    """
    dim = ds["dim"]
    fn = Ackley(ds["ackley_variant"], dim=dim, peak_seed=seed, noise_seed=seed)
    new_ds = dict(ds)
    new_ds["fn"] = fn
    new_ds["true_optima"] = [np.asarray(c, dtype=float) for c in fn.centers]
    if dim == 3 and ds.get("grid_pts") is not None:
        new_ds["grid_vals"] = fn.predict(ds["grid_pts"])
    return new_ds

def reseed_ackley_dataset(ds: dict, seed: int) -> dict:
    """Return a shallow copy of an Ackley dataset bundle rebuilt for ``seed``.

    The seed drives BOTH the optima placement (peak_seed) and the background noise
    (noise_seed), so each trial faces a fresh random realisation of the
    realistic-Ackley landscape. The function, its known optima, and — for 3D — the
    render-grid values are recomputed to match (4D renders from ``fn`` directly;
    ≥5D has no landscape view, so ``grid_vals`` stays ``None``).
    """
    dim = ds["dim"]
    fn = Ackley(ds["ackley_variant"], dim=dim, peak_seed=seed, noise_seed=seed)
    new_ds = dict(ds)
    new_ds["fn"] = fn
    new_ds["true_optima"] = [np.asarray(c, dtype=float) for c in fn.centers]
    if dim == 3 and ds.get("grid_pts") is not None:
        new_ds["grid_vals"] = fn.predict(ds["grid_pts"])
    return new_ds


# ─── Per-trial Ensemble re-randomization ───────────────────────────────────────

def build_ensemble_landscape(
    dim: int, *, optima_margin: float, seed: int, time_limit_hours: float | None,
) -> LandscapeSpec:
    """Base ``LandscapeSpec`` for the layered Ensemble objective.

    The objective is re-randomized for every trial (see :func:`reseed_ensemble`),
    so the function/optima carried here are just an initial draw that gives the
    spec a valid ``true_optima`` and (for dim 3) a render grid. ``oracle`` is
    tagged ``"ensemble"`` so resume can recognise and rebuild it.
    """
    rng = random.Random(f"{seed}-base")
    cfg = random_ensemble_config(dim, rng, optima_margin=optima_margin)
    fn = Ensemble(**cfg)
    true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
    grid_pts = grid_vals = None
    if dim == 3:
        grid_pts = ternary_grid(TERNARY_GRID_N)
        grid_vals = fn.predict(grid_pts)
    lo, hi = optima_count_range(dim)
    print(f"  [dataset] ensemble: Ensemble(dim={dim}) re-randomized per trial "
          f"(n_optima in [{lo}, {hi}], ensemble_seed={seed}, margin={optima_margin})")
    return LandscapeSpec(
        landscape="synthetic", dim=dim, maximize=True, true_optima=true_optima,
        fn_callable=fn, grid_pts=grid_pts, grid_vals=grid_vals,
        time_limit_hours=time_limit_hours, max_activations=float("inf"),
        oracle="ensemble", synthetic_seed=seed,
    )


def reseed_ensemble(landscape: LandscapeSpec, config: dict):
    """Rebuild the Ensemble objective for one trial from a saved ``config``.

    Returns ``(fn_callable, true_optima, grid_vals)`` — ``grid_vals`` is ``None``
    unless the landscape renders a 3-simplex ternary grid.
    """
    fn = Ensemble(**config)
    true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
    grid_vals = None
    if landscape.dim == 3 and landscape.grid_pts is not None:
        grid_vals = fn.predict(landscape.grid_pts)
    return fn, true_optima, grid_vals


# ─── Single trial: run ZoMBI on the objective + write all artifacts ────────────

def run_single_trial(
    hparams: dict,
    landscape: LandscapeSpec,
    trial_dir: str,
    *,
    ackley_variant: str | None = None,
    ensemble_config: dict | None = None,
) -> dict:
    """Run one ZoMBI trial on the configured landscape, then write per-trial artifacts.

    Returns {"dist", "dup", "runtime", "avg_time_per_iter", "n_iters", "payloads",
    "ackley_seed"}.
    """
    ackley_seed = None
    fn_callable = landscape.fn_callable
    true_optima = list(landscape.true_optima)
    grid_pts = landscape.grid_pts
    grid_vals = landscape.grid_vals
    dim = landscape.dim
    maximize = landscape.maximize

    if ackley_variant == "realistic":
        ackley_seed = draw_ackley_seed()
        reseeded = reseed_ackley_dataset(
            {"dim": dim, "ackley_variant": ackley_variant, "grid_pts": grid_pts},
            ackley_seed,
        )
        fn_callable = reseeded["fn"]
        true_optima = reseeded["true_optima"]
        if reseeded.get("grid_vals") is not None:
            grid_vals = reseeded["grid_vals"]
        print(f"    [trial] realistic Ackley seed = {ackley_seed}")

    if ensemble_config is not None:
        fn_callable, true_optima, ens_grid_vals = reseed_ensemble(landscape, ensemble_config)
        if ens_grid_vals is not None:
            grid_vals = ens_grid_vals
        print(f"    [trial] ensemble seed={ensemble_config.get('seed')}  "
              f"n_optima={ensemble_config.get('n_optima')}  "
              f"({len(true_optima)} true optima)")

    if os.path.isdir(trial_dir):
        shutil.rmtree(trial_dir, ignore_errors=True)
    os.makedirs(trial_dir, exist_ok=True)
    if ensemble_config is not None:
        # Persist the exact landscape so this trial is recreatable.
        with open(os.path.join(trial_dir, "ensemble_config.json"), "w") as f:
            json.dump(ensemble_config, f, indent=2)

    plot_state: dict = {"line_0": None, "line_1": None}
    payloads: list[dict] = []
    snap_records: list[tuple] = []
    call_counter = [0]
    dh_ref = [None]

    sim_obj = make_sim_obj(fn_callable, DEVICE, DTYPE, maximize=maximize)
    inner   = make_linebo_wrapper(sim_obj, dim, NUM_LINES, DEVICE, DTYPE, plot_state)

    # The per-iteration "heavy" payload fields (pared point cloud, per-needle
    # penalisation matrices, bounds) are consumed ONLY by render_frame, i.e. the
    # 3D per-iteration ternary frames. At dim ≥ 4 no frames are drawn (the 4D
    # point cloud reads final state straight from the data handler, not payloads),
    # so storing them every iteration just grows host RAM without bound — on a
    # long high-dim trial that exhausts the cgroup limit and triggers an OOM kill.
    # Keep them only when ternary frames will actually be rendered.
    keep_heavy = landscape.render_ternary

    def obj_wrapper(x_tell, bounds, acq_fn):
        x_req, x_act, y = inner(x_tell, bounds, acq_fn)
        call_counter[0] += 1
        dh = dh_ref[0]
        needles = dh.needles
        payload = dict(
            iter_num=call_counter[0],
            needles=(as_numpy(needles)
                     if needles is not None and needles.shape[0] > 0 else None),
            needle_vals=(as_numpy(dh.needle_vals).ravel()
                         if dh.needle_vals is not None and dh.needle_vals.shape[0] > 0 else None),
            line_0=plot_state.get("line_0"),
            line_1=plot_state.get("line_1"),
            n_points_before=(dh.X_all_actual.shape[0] if dh.X_all_actual is not None else 0),
        )
        if keep_heavy:
            xp, yp = dh.X_pared, dh.Y_pared
            if xp is not None and xp.shape[0] > 0:
                pared_X = xp.detach().cpu().numpy()
                pared_Y = yp.detach().cpu().numpy().ravel()
                if not maximize:
                    pared_Y = -pared_Y
            else:
                pared_X = pared_Y = None
            payload.update(
                pared_X=pared_X, pared_Y=pared_Y,
                needle_M_list=[as_numpy(m) if m is not None else None
                               for m in dh.needle_M_list],
                needle_B=(as_numpy(dh.needle_B) if dh.needle_B is not None else None),
                bounds=(as_numpy(dh.bounds) if dh.bounds is not None else None),
            )
        payloads.append(payload)
        return x_req, x_act, y

    try:
        X_a, X_e, Y = _gen_init_data(fn_callable, maximize, dim=dim)
    except Exception as exc:
        print(f"    [trial] init failed: {exc}")
        return {"dist": UNMATCHED_PENALTY, "dup": 1.0, "runtime": 0.0,
                "avg_time_per_iter": 0.0, "n_iters": 0, "payloads": [],
                "ackley_seed": ackley_seed}

    hp = dict(hparams)
    if dim > 3 and (hp.get("top_m_points") is None or hp.get("top_m_points", 0) < dim + 1):
        hp["top_m_points"] = max(dim + 1, 4)

    try:
        optimizer = ZoMBIHop(
            objective=obj_wrapper,
            X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
            **ZOMBI_FIXED, **hp,
            device=str(DEVICE), dtype=DTYPE,
            run_uuid=None, checkpoint_dir=None,
        )
    except Exception as exc:
        print(f"    [trial] ZoMBI init failed: {exc}")
        return {"dist": UNMATCHED_PENALTY, "dup": 1.0, "runtime": 0.0,
                "avg_time_per_iter": 0.0, "n_iters": 0, "payloads": [],
                "ackley_seed": ackley_seed}
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
        if landscape.time_limit_hours is not None:
            optimizer.run(
                max_activations=float("inf"),
                time_limit_hours=landscape.time_limit_hours,
            )
        else:
            optimizer.run(
                max_activations=landscape.max_activations or float("inf"),
                time_limit_hours=None,
            )
    except Exception as exc:
        print(f"    [trial] ZoMBI crashed: {exc}")
    runtime = time.time() - t0

    n_iters = call_counter[0]
    avg_time_per_iter = runtime / n_iters if n_iters > 0 else 0.0

    needle_t   = dh.get_all_needle_locations()
    discovered = (
        as_numpy(needle_t)
        if needle_t.numel() > 0 else np.empty((0, dim))
    )
    X_all_np   = (
        as_numpy(dh.X_all_actual)
        if dh.X_all_actual is not None else np.empty((0, dim))
    )
    dist = metric_dist_to_needles(discovered, true_optima, dim=dim)
    dup  = metric_dup_fraction(X_all_np, dim=dim)
    print(f"    [trial]  iters={n_iters}  dist={dist:.4f}  dup={dup:.4f}"
          f"  t/iter={avg_time_per_iter:.3f}s  (total {runtime:.1f}s)"
          f"  needles={len(discovered)}/{len(true_optima)}", flush=True)

    try:
        write_points_csv(os.path.join(trial_dir, "points.csv"), dh, snap_records, dim=dim)
        write_needles_csv(os.path.join(trial_dir, "needles.csv"), dh, dim=dim)
        write_metrics_over_time_csv(
            os.path.join(trial_dir, "metrics_over_time.csv"), payloads, X_all_np, true_optima,
            dim=dim,
        )
    except Exception as exc:
        print(f"    [trial] CSV write failed: {exc}")

    try:
        plot_dist_from_centre(os.path.join(trial_dir, "dist_from_centre.png"), dh, maximize)
        plot_line_length_hist(os.path.join(trial_dir, "line_length_hist.png"), payloads)
        plot_convergence(os.path.join(trial_dir, "convergence.png"), dh, maximize)
    except Exception as exc:
        print(f"    [trial] static plot failed: {exc}")

    if landscape.render_ternary and grid_pts is not None and grid_vals is not None:
        plots_dir = os.path.join(trial_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        print(f"    [trial] rendering {len(payloads)} ternary frames …", flush=True)
        for p in payloads:
            try:
                ref_title = (
                    "Reference: oracle landscape"
                    if landscape.landscape == "synthetic"
                    else "Reference: RF landscape"
                )
                render_frame(
                    p, grid_pts, grid_vals, true_optima, maximize,
                    os.path.join(plots_dir, f"iter_{p['iter_num'] - 1:04d}.png"),
                    ref_title=ref_title,
                )
            except Exception as exc:
                print(f"    [trial] frame {p['iter_num']} failed: {exc}")

    if dim == 4 and hasattr(fn_callable, "predict"):
        try:
            print("    [trial] rendering 4D interactive point cloud …", flush=True)
            _render_4d_point_cloud(
                os.path.join(trial_dir, "point_cloud.html"),
                fn_callable, dh, payloads[-1] if payloads else None)
        except Exception as exc:
            print(f"    [trial] 4D point cloud failed: {exc}")

    _auto_generate_plots(trial_dir, dim)

    return {"dist": dist, "dup": dup, "runtime": runtime,
            "avg_time_per_iter": avg_time_per_iter, "n_iters": n_iters,
            "payloads": payloads, "ackley_seed": ackley_seed,
            "ensemble_config": ensemble_config}


# ─── One MOBO trial: single eval, or averaged ensemble repeats ──────────────────

def evaluate_hparams(
    hparams: dict,
    landscape: LandscapeSpec,
    trial_dir: str,
    trial_num: int,
    *,
    ackley_variant: str | None = None,
    ensemble_spec: dict | None = None,
) -> dict:
    """Evaluate one hyperparameter set and return the metrics MOBO scores against.

    For every dataset except the re-randomized ``ensemble``, this is a single
    ``run_single_trial`` writing its artifacts straight into ``trial_dir``.

    For ``ensemble`` it instead evaluates the same hyperparameters on
    ``ENSEMBLE_N_REPEATS`` independently randomized landscapes — each in its own
    ``trial_<n>/run_<k>/`` subfolder with the full artifact set — and **averages
    the three MOBO metrics** (``dist_to_needles``, ``dup_fraction``,
    ``avg_time_per_iter_s``) across the repeats before the next MOBO iteration.
    The repeats are reproducible per ``(ensemble_seed, trial, k)``; every landscape
    config is saved (per-run ``ensemble_config.json`` + the ``ensemble_configs``
    list in ``trial.json``). The returned ``runtime``/``n_iters`` are summed over
    the repeats (whole-trial cost), and ``repeats`` holds each repeat's metrics.
    """
    if ensemble_spec is None:
        return run_single_trial(
            hparams, landscape, trial_dir, ackley_variant=ackley_variant)

    if os.path.isdir(trial_dir):
        shutil.rmtree(trial_dir, ignore_errors=True)
    os.makedirs(trial_dir, exist_ok=True)

    configs: list[dict] = []
    repeats: list[dict] = []
    for k in range(1, ENSEMBLE_N_REPEATS + 1):
        rng = random.Random(f"{ensemble_spec['seed']}-{trial_num}-{k}")
        cfg = random_ensemble_config(
            ensemble_spec["dim"], rng, optima_margin=ensemble_spec["optima_margin"])
        configs.append(cfg)
        run_dir = os.path.join(trial_dir, f"run_{k}")
        print(f"    [ensemble repeat {k}/{ENSEMBLE_N_REPEATS}]", flush=True)
        r = run_single_trial(hparams, landscape, run_dir, ensemble_config=cfg)
        repeats.append({
            "run": k,
            "dist": round(r["dist"], 6),
            "dup": round(r["dup"], 6),
            "avg_time_per_iter": round(r["avg_time_per_iter"], 4),
            "runtime": round(r["runtime"], 3),
            "n_iters": int(r["n_iters"]),
        })

    dist = float(np.mean([r["dist"] for r in repeats]))
    dup = float(np.mean([r["dup"] for r in repeats]))
    avg_t = float(np.mean([r["avg_time_per_iter"] for r in repeats]))
    runtime = float(np.sum([r["runtime"] for r in repeats]))
    n_iters = int(np.sum([r["n_iters"] for r in repeats]))
    print(f"    [trial] ensemble avg over {ENSEMBLE_N_REPEATS} landscapes — "
          f"dist={dist:.4f}  dup={dup:.4f}  t/iter={avg_t:.3f}s")
    return {
        "dist": dist, "dup": dup, "runtime": runtime,
        "avg_time_per_iter": avg_t, "n_iters": n_iters,
        "payloads": [], "ackley_seed": None,
        "ensemble_config": None, "ensemble_configs": configs, "repeats": repeats,
    }


# ─── Running summary (mobo_progress.json / mobo_results.json) ───────────────────

def _build_summary(X_obs: list[torch.Tensor], Y_obs: list[torch.Tensor],
                   n_seed: int = 0, n_sobol: int = 0) -> dict:
    n = len(Y_obs)
    metrics_all = [
        {
            "dist_to_needles":     round(-Y_obs[i][0].item(), 6),
            "dup_fraction":        round(-Y_obs[i][1].item(), 6),
            "avg_time_per_iter_s": round(-Y_obs[i][2].item(), 4),
        }
        for i in range(n)
    ]

    def _phase(i: int) -> str:
        if i < n_seed:
            return "seed"
        if i < n_seed + n_sobol:
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
    dists = [m["dist_to_needles"]     for m in metrics_all]
    dups  = [m["dup_fraction"]        for m in metrics_all]
    times = [m["avg_time_per_iter_s"] for m in metrics_all]
    return {
        "n_trials": n,
        "averages": {
            "dist_to_needles":     round(float(np.mean(dists)), 6),
            "dup_fraction":        round(float(np.mean(dups)),  6),
            "avg_time_per_iter_s": round(float(np.mean(times)), 4),
        },
        "best_dist": {"value": round(min(dists), 6), "trial": int(np.argmin(dists)) + 1},
        "trials": trials,
    }


def save_running_summary(X_obs, Y_obs, run_dir: str, n_seed: int = 0,
                         n_sobol: int = 0) -> None:
    """Write mobo_progress.json + mobo_results.json + mobo_results.pt."""
    if not Y_obs:
        return
    summary = _build_summary(X_obs, Y_obs, n_seed=n_seed, n_sobol=n_sobol)
    summary_txt = json.dumps(summary, indent=2)
    _atomic_write_text(os.path.join(run_dir, "mobo_progress.json"), summary_txt)
    _atomic_write_text(os.path.join(run_dir, "mobo_results.json"), summary_txt)
    _atomic_torch_save(
        {"X_obs": torch.stack(X_obs).cpu(), "Y_obs": torch.stack(Y_obs).cpu(),
         "hparam_names": HPARAM_NAMES},
        os.path.join(run_dir, "mobo_results.pt"),
    )
    print(f"  [summary] {len(Y_obs)} trials recorded", flush=True)



# ─── MOBO loop (unbounded, resumable) ───────────────────────────────────────────

def run_mobo(landscape: LandscapeSpec, run_dir,
             max_trials=None, seed_X=None, X_prior=None, Y_prior=None, *,
             n_init_trials: int = N_INIT_TRIALS,
             ackley_variant: str | None = None,
             ensemble_spec: dict | None = None) -> None:
    """Unbounded MOBO loop, writing trials into a fresh ``run_dir``."""
    bounds = torch.zeros(2, N_HPARAMS, dtype=DTYPE, device=DEVICE)
    bounds[1] = 1.0

    X_prior = [x.detach().cpu() for x in X_prior] if X_prior else []
    Y_prior = [y.detach().cpu() for y in Y_prior] if Y_prior else []
    n_prior = len(Y_prior)
    seed_X  = [x.detach().cpu() for x in seed_X] if seed_X else []
    X_obs: list[torch.Tensor] = []
    Y_obs: list[torch.Tensor] = []

    n_seed  = len(seed_X)
    n_sobol = n_init_trials if n_prior == 0 else 0
    X_sobol = load_or_make_sobol(run_dir, bounds, n_sobol)
    init_design = ([(x, "seed") for x in seed_X]
                   + [(X_sobol[i], "sobol") for i in range(X_sobol.shape[0])])
    n_init = len(init_design)

    stop_desc = (
        f"time limit / trial: {landscape.time_limit_hours} h"
        if landscape.time_limit_hours is not None
        else f"max_activations / trial: {landscape.max_activations}"
    )
    print(f"\n{'='*70}")
    print(f"MOBO  |  {landscape.label}  |  {n_seed} re-evaluated seed(s) + "
          f"{X_sobol.shape[0]} Sobol init, then BO until Ctrl+C")
    print(f"{stop_desc}    Run dir: {run_dir}")
    if n_prior:
        print(f"PRIOR HISTORY — seeding GP with {n_prior} (X,Y) pair(s) "
              f"from prior runs; skipping Sobol init")
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
                print(f"\n[trial {trial_num} | {phase}]  {hp_str}", flush=True)

                res = evaluate_hparams(
                    hparams, landscape, trial_dir, trial_num,
                    ackley_variant=ackley_variant, ensemble_spec=ensemble_spec)
                try:
                    plot_hparam_edge_proximity(
                        os.path.join(trial_dir, "hparam_edge_proximity.png"), x_new)
                except Exception as exc:
                    print(f"    [trial] hparam_edge_proximity failed: {exc}")

                X_obs.append(x_new.detach().cpu())
                Y_obs.append(torch.tensor(
                    [-res["dist"], -res["dup"], -res["avg_time_per_iter"]],
                    dtype=DTYPE, device="cpu"))
                save_running_summary(X_obs, Y_obs, run_dir, n_seed=n_seed, n_sobol=n_sobol)
                write_trial_json(
                    os.path.join(trial_dir, "trial.json"),
                    trial_num, phase,
                    {"dist": res["dist"], "dup": res["dup"],
                     "avg_time_per_iter": res["avg_time_per_iter"],
                     "runtime": res["runtime"], "n_iters": res["n_iters"]},
                    hparams,
                    ackley_seed=res.get("ackley_seed"),
                    ensemble_config=res.get("ensemble_config"),
                    ensemble_configs=res.get("ensemble_configs"),
                    repeats=res.get("repeats"),
                )
                append_trial_logs(
                    run_dir, trial_num, phase, hparams,
                    {"dist": res["dist"], "dup": res["dup"], "runtime": res["runtime"]},
                    trial_dir,
                    dim=landscape.dim,
                )
                consec_fail = 0

            except KeyboardInterrupt:
                raise
            except Exception as exc:
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

    if max_trials is not None and len(Y_obs) >= max_trials:
        print(f"\n[stop] max_trials={max_trials} reached ({len(Y_obs)} completed).")

    if Y_obs:
        save_running_summary(X_obs, Y_obs, run_dir, n_seed=n_seed, n_sobol=n_sobol)
    print(f"\nDone. {len(Y_obs)} trials completed this run "
          f"({n_prior} prior + {len(Y_obs)} new = {n_prior + len(Y_obs)} total). Results in {run_dir}")
    print(f"Resume (crawls all runs) with:  python optimize/run_mobo.py --resume")
    print(f"Pareto front across all runs:   python optimize/pareto.py")


# ─── Main ───────────────────────────────────────────────────────────────────────



def _landscape_from_legacy_ackley(ds: dict) -> LandscapeSpec:
    """Wrap a legacy ``--dataset ackley*`` bundle as a ``LandscapeSpec``."""
    return LandscapeSpec(
        landscape="ackley",
        dim=ds["dim"],
        maximize=ds["maximize"],
        true_optima=ds["true_optima"],
        fn_callable=ds["fn"],
        grid_pts=ds.get("grid_pts"),
        grid_vals=ds.get("grid_vals"),
        time_limit_hours=TIME_LIMIT_HOURS,
    )


def _ds_ackley(dataset: str, ackley_variant: str) -> dict:
    """Build an analytic negated-Ackley objective bundle on the d-simplex."""
    dim = DATASET_DIMS[dataset]
    fn  = Ackley(ackley_variant, dim=dim)
    true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
    if dim == 3:
        grid_pts  = ternary_grid(TERNARY_GRID_N)
        grid_vals = fn.predict(grid_pts)
    else:
        grid_pts = grid_vals = None
    print(f"  [dataset] {dataset}: Ackley('{ackley_variant}', dim={dim}) — "
          f"maximize, {len(true_optima)} analytic peak(s)")
    return dict(dim=dim, label=dataset, fn=fn, maximize=True, true_optima=true_optima,
                grid_pts=grid_pts, grid_vals=grid_vals, ackley_variant=ackley_variant,
                csv_path=None)


def _legacy_ackley_dataset_key(cfg: dict) -> str | None:
    """Return ackley3d/ackley4d/ackley10d when config is a legacy Ackley benchmark."""
    dataset = cfg.get("dataset")
    if dataset in DATASET_DIMS and dataset != "RF":
        return dataset
    # Runs saved before ``dataset`` was persisted: infer from variant + dim.
    if (cfg.get("landscape") == "ackley"
            and cfg.get("ackley_variant")
            and not cfg.get("ackley_layout")):
        dim = int(cfg.get("dim", 3))
        key = f"ackley{dim}d"
        if key in DATASET_DIMS:
            return key
    return None


def _persist_dataset_fields(cfg: dict) -> tuple[str | None, str | None]:
    """Extract dataset / ackley_variant to re-save in a resumed run's config."""
    legacy = _legacy_ackley_dataset_key(cfg)
    if legacy:
        return legacy, cfg.get("ackley_variant")
    ds = cfg.get("dataset")
    if ds and ds != "RF":
        return ds, cfg.get("ackley_variant")
    return None, cfg.get("ackley_variant")


def _landscape_from_legacy_ackley_cfg(cfg: dict) -> tuple[LandscapeSpec, str]:
    """Rebuild a ``--dataset ackley*`` landscape from run_config.json."""
    dataset = _legacy_ackley_dataset_key(cfg)
    if dataset is None:
        raise ValueError("not a legacy ackley dataset config")
    ackley_variant = cfg.get("ackley_variant") or "realistic"
    ds = _ds_ackley(dataset, ackley_variant)
    if "maximize" in cfg:
        ds["maximize"] = bool(cfg["maximize"])
    if cfg.get("true_optima"):
        ds["true_optima"] = [np.asarray(t, dtype=float) for t in cfg["true_optima"]]
    return _landscape_from_legacy_ackley(ds), ds["ackley_variant"]


def _ensemble_spec_from_cfg(cfg: dict) -> dict | None:
    """Reconstruct the per-trial ensemble randomization spec from a run_config."""
    if cfg.get("dataset") == ENSEMBLE_DATASET or cfg.get("oracle") == "ensemble":
        return {
            "dim": int(cfg.get("dim", ENSEMBLE_DEFAULT_DIM)),
            "optima_margin": float(cfg.get("ensemble_optima_margin", 0.2)),
            "seed": int(cfg.get("ensemble_seed", cfg.get("seed", 0))),
        }
    return None


def _landscape_from_run_config_legacy(cfg: dict) -> tuple[LandscapeSpec, str | None]:
    """Rebuild landscape from run_config.json (RF, legacy ackley*, Multi-Ackley,
    or the re-randomized Ensemble objective)."""
    ens = _ensemble_spec_from_cfg(cfg)
    if ens is not None:
        landscape = build_ensemble_landscape(
            ens["dim"], optima_margin=ens["optima_margin"], seed=ens["seed"],
            time_limit_hours=cfg.get("time_limit_hours"),
        )
        return landscape, None
    ackley_variant = cfg.get("ackley_variant")
    legacy_key = _legacy_ackley_dataset_key(cfg)
    if legacy_key is not None:
        return _landscape_from_legacy_ackley_cfg(cfg)
    if cfg.get("landscape"):
        return landscape_from_run_config(cfg, build_rf_and_grid=build_rf_and_grid), ackley_variant
    return landscape_from_run_config(
        {"landscape": "rf", **cfg}, build_rf_and_grid=build_rf_and_grid,
    ), None


def _interactive_run_config(script_dir: str):
    """Interactive run-config creation: locate campaign1a.csv, train the RF,
    prompt max/min, and interactively pick the reference optima.

    Shared by the fresh run and ``--resume-scratch``. Returns a ``LandscapeSpec``.
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
    rf, rf_fn, grid_pts, grid_vals, _, _ = build_rf_and_grid(csv_path)
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
    return build_rf_landscape(
        rf_fn, true_optima, grid_pts, grid_vals,
        maximize=rf_maximize, csv_path=csv_path,
        time_limit_hours=TIME_LIMIT_HOURS,
    )


def _launch_run(runs_dir, landscape: LandscapeSpec, max_trials,
                seed_X=None, X_prior=None, Y_prior=None, *,
                batch_name: str | None = None,
                batch_config_path: str | None = None,
                run_dir: str | None = None,
                n_init_trials: int = N_INIT_TRIALS,
                dataset: str | None = None,
                ackley_variant: str | None = None,
                ensemble_spec: dict | None = None,
                invocation: dict | None = None) -> None:
    """Create a fresh runs/mobo_* folder, persist its config, and run MOBO."""
    if run_dir is None:
        run_dir = unique_run_dir(runs_dir, "mobo")
    else:
        os.makedirs(run_dir, exist_ok=True)
    if invocation is not None:
        _log_invocation(invocation)
    write_run_config(
        run_dir, landscape,
        batch_name=batch_name, batch_config_path=batch_config_path,
        n_init_trials=n_init_trials,
        dataset=dataset, ackley_variant=ackley_variant,
        ensemble_spec=ensemble_spec,
        invocation=invocation,
    )
    print(f"\n[run] Output folder: {run_dir}")
    run_mobo(landscape, run_dir, max_trials=max_trials,
             seed_X=seed_X, X_prior=X_prior, Y_prior=Y_prior,
             n_init_trials=n_init_trials, ackley_variant=ackley_variant,
             ensemble_spec=ensemble_spec)



def _apply_runtime_overrides(*, device: str | None = None,
                             time_limit_hours: float | None = None) -> None:
    """Apply CLI overrides to module-level DEVICE and TIME_LIMIT_HOURS."""
    global DEVICE, TIME_LIMIT_HOURS
    if device is not None:
        DEVICE = torch.device(device)
    if time_limit_hours is not None:
        TIME_LIMIT_HOURS = float(time_limit_hours)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZoMBI-Hop MOBO hyperparameter optimisation (RF or Multi-Ackley).",
    )
    parser.add_argument("--landscape", choices=("rf", "ackley"), default="rf",
                        help="Objective landscape (default: rf). Ackley = synthetic Δ^d benchmark.")
    parser.add_argument("--dim", type=int, default=None,
                        help="Simplex dimension for --landscape ackley (default 10) or "
                             "--dataset ensemble (default 3).")
    parser.add_argument("--layout", type=str, default=None, choices=["1", "2", "3"],
                        help="Multi-Ackley peak layout (1/2/3).")
    parser.add_argument("--ackley-b", type=float, default=None,
                        help="Ackley peak width b (default 1.2 skinny).")
    parser.add_argument("--n-init-trials", type=int, default=None,
                        help=f"Sobol init trials before BO (default: {N_INIT_TRIALS}).")
    parser.add_argument("--batch", action="store_true",
                        help="Headless mode: read --config JSON, skip interactive picker "
                             "(for Slurm / ORCD HPC).")
    parser.add_argument("--config", metavar="PATH", default=None,
                        help="Batch run config JSON (csv_path, maximize, true_optima or "
                             "regions_objective, max_trials, time_limit_hours).")
    parser.add_argument("--no-show", action="store_true",
                        help="Use Agg matplotlib backend (no GUI windows). Implied by --batch.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None,
                        help="Torch device (default: cuda if available else cpu).")
    parser.add_argument("--time-limit-hours", type=float, default=None,
                        help=f"Per-trial ZoMBI wall-clock budget in hours (default: {TIME_LIMIT_HOURS}).")
    parser.add_argument("--time-limit", type=float, default=None, metavar="HOURS",
                        help="Alias for --time-limit-hours (parametric sweep compatibility).")
    parser.add_argument("--dataset", default="RF", choices=DATASET_CHOICES,
                        help="Objective to search on (default: RF). RF uses the 3-simplex "
                             "Random-Forest surrogate (interactive extrema picker); "
                             "ackley3d/ackley4d/ackley10d are analytic negated-Ackley "
                             "benchmarks (known optima, non-interactive); 'ensemble' is the "
                             "layered synthetic objective, RE-RANDOMIZED PER TRIAL (dim from "
                             "--dim). Ignored by --resume / --copy-config / --resume-from "
                             "(inherit saved config).")
    parser.add_argument("--ackley-variant", default="realistic",
                        choices=sorted(Ackley.VARIANTS),
                        help="Ackley variant for --dataset ackley* (default: realistic).")
    parser.add_argument("--ensemble-seed", type=int, default=0,
                        help="Master seed for per-trial 'ensemble' randomization (same value "
                             "reproduces the per-trial landscape sequence; default: 0).")
    parser.add_argument("--ensemble-margin", type=float, default=0.2,
                        help="optima_margin for --dataset ensemble — normalized gap the "
                             "background stays below the optima (default: 0.2).")
    parser.add_argument("--resume-from", metavar="RUN_DIR", default=None,
                        help="Resume from a SPECIFIC run: trust its stored metrics "
                             "as prior history (no re-evaluation), reuse its config.")
    parser.add_argument("--runs-dir", metavar="DIR", default=None,
                        help="Override output runs directory (default: optimize/runs).")
    parser.add_argument("--run-dir", metavar="DIR", default=None,
                        help="Explicit run output folder (array jobs: one dir per task).")
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
    parser.add_argument("--resume-dim", action="store_true",
                        help="Resume on the dataset given by --dataset, seeding the new run "
                             "with ALL prior runs/mobo_* runs of the SAME dimensionality: "
                             "crawl every run whose run_config dim matches the --dataset dim, "
                             "collect their (X,Y) pairs, and trust them as prior history (no "
                             "re-evaluation). Non-interactive (ackley datasets only).")
    parser.add_argument("--start-from-best", nargs="+", metavar="TRIAL_DIR", default=None,
                        help="One or more trial_* directories (or trial.json files) whose "
                             "HYPERPARAMETERS are RE-EVALUATED as initial trials on the "
                             "current objective (stored metrics ignored). Runs IN ADDITION "
                             "to the full Sobol init. Combinable with any run mode.")
    parser.add_argument("--copy-config", metavar="PATH", default=None,
                        help="Reuse another run's run_config.json (max/min, csv_path, picked "
                             "optima) for a NEW run, WITHOUT inheriting its data points. PATH "
                             "is a run dir or a run_config.json file. Non-interactive; runs a "
                             "normal Sobol-init + BO run. Cannot combine with "
                             "--resume / --resume-scratch / --resume-from.")
    args = parser.parse_args()

    if args.batch and not args.config:
        sys.exit("--batch requires --config PATH.")
    n_exclusive = sum([
        args.resume, args.resume_scratch, args.resume_dim,
        args.copy_config is not None, args.resume_from is not None,
    ])
    if args.config and not args.batch and not n_exclusive:
        sys.exit(
            "--config requires --batch for a headless run from JSON.\n"
            "  Example: python optimize/run_mobo.py --no-show --batch "
            "--config optimize/mobo_batch_configs/foo.json\n"
            "  Do not pass --dataset RF with --config — that skips the batch "
            "path and opens the interactive max/min picker (will hang on HPC)."
        )
    if args.config and n_exclusive:
        print("  [warning] --config is ignored when using --resume / "
              "--resume-from / --copy-config (config comes from the saved run).")
    if args.config and not args.batch and not args.no_show:
        print("  [hint] --config is usually paired with --batch on HPC.")
    if n_exclusive > 1:
        sys.exit("Use only one of --resume / --resume-scratch / --copy-config / --resume-from.")
    if args.batch and n_exclusive:
        sys.exit("--batch cannot combine with --resume / --resume-scratch / --copy-config / --resume-from.")

    headless = bool(args.batch or args.no_show or args.config
                    or args.resume or args.resume_scratch or args.copy_config
                    or args.resume_from
                    or (args.dataset != "RF"))
    _configure_mpl_backend(headless=headless)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir   = os.path.abspath(args.runs_dir or os.path.join(script_dir, "runs"))
    os.makedirs(runs_dir, exist_ok=True)

    if args.time_limit is not None and args.time_limit_hours is not None:
        sys.exit("Use only one of --time-limit / --time-limit-hours.")
    time_limit_override = args.time_limit_hours if args.time_limit_hours is not None else args.time_limit

    max_trials = args.max_trials
    batch_name = None
    batch_config_path = None
    run_dir_override = os.path.abspath(args.run_dir) if args.run_dir else None

    if args.batch:
        batch = load_batch_config(args.config, script_dir)
        batch_name = batch["name"]
        batch_config_path = batch["config_path"]
        n_init = batch.get("n_init_trials") or args.n_init_trials or N_INIT_TRIALS
        _apply_runtime_overrides(
            device=args.device,
            time_limit_hours=batch.get("time_limit_hours") or time_limit_override,
        )
        max_trials = (
            args.max_trials
            if args.max_trials is not None
            else batch.get("max_trials")
        )

        print("=" * 70)
        print(f"ZoMBI-Hop MOBO — BATCH  |  {batch_name}")
        print(f"Device: {DEVICE}   |   Sobol init: {n_init} trials   |   "
              f"max_trials: {max_trials if max_trials is not None else 'unbounded (Ctrl+C)'}")
        print("=" * 70)

        optima_note = ""
        if batch.get("landscape") is not None:
            landscape = batch["landscape"]
            if landscape.time_limit_hours is None and TIME_LIMIT_HOURS is not None:
                pass  # Ackley uses max_activations
            elif batch.get("time_limit_hours") is not None:
                landscape.time_limit_hours = batch["time_limit_hours"]
            optima_note = "true_optima: planted by landscape builder (batch JSON)"
        else:
            csv_path = batch["csv_path"]
            obj_col = batch["objective_column"]
            comp_cols = batch.get("composition_columns")
            rf_maximize = batch["maximize"]
            oracle = batch.get("oracle")
            label_bits = [obj_col]
            if oracle:
                label_bits.append(f"oracle={oracle}")
            if comp_cols:
                label_bits.append(f"cols={','.join(comp_cols)}")
            print(f"\n[RF] Loading {csv_path} ({', '.join(label_bits)}) …  "
                  f"Training RF ({RF_N_ESTIMATORS} trees) …")
            rf, rf_fn, grid_pts, grid_vals, comp_cols, dim = build_rf_and_grid(
                csv_path, objective_column=obj_col, composition_columns=comp_cols,
            )
            print(f"  RF trained ({dim}D, {len(comp_cols)} composition columns).")

            true_optima = batch["true_optima"]
            if not true_optima:
                auto = batch.get("auto_optima") or {}
                n_peaks = int(auto.get("n_peaks", 3))
                min_sep = float(auto.get("min_sep", 0.08))
                print(f"  [batch] auto-detecting {n_peaks} reference "
                      f"{'maxima' if rf_maximize else 'minima'} …")
                true_optima = auto_detect_rf_optima(
                    rf, grid_pts, grid_vals,
                    maximize=rf_maximize, n_peaks=n_peaks, min_sep=min_sep,
                )
                optima_note = (
                    f"true_optima: auto_detect_rf_optima(n_peaks={n_peaks}, "
                    f"min_sep={min_sep}) — NOT from batch JSON"
                )
            else:
                optima_note = "true_optima: batch JSON or metadata sidecar"
            goal = "maxima" if rf_maximize else "minima"
            print(f"  RF ready: {len(true_optima)} reference {goal}")
            for i, t in enumerate(true_optima):
                print(f"    #{i + 1}  {np.round(t, 4).tolist()}")

            landscape = build_rf_landscape(
                rf_fn, true_optima, grid_pts, grid_vals,
                maximize=rf_maximize, csv_path=csv_path,
                objective_column=obj_col,
                composition_columns=comp_cols,
                dim=dim,
                oracle=oracle,
                metadata_path=batch.get("metadata_path"),
                time_limit_hours=TIME_LIMIT_HOURS,
            )

        if landscape.landscape in ("ackley", "synthetic"):
            for i, p in enumerate(landscape.true_optima):
                print(f"  peak {i + 1}: {np.round(p, 4).tolist()}")
        stop = (
            f"time limit/trial: {landscape.time_limit_hours} h"
            if landscape.time_limit_hours is not None
            else f"max_activations/trial: {landscape.max_activations}"
        )
        print(f"  {landscape.label}  |  {stop}")

        resolutions: list[str] = []
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )
        if batch.get("time_limit_hours") is not None:
            resolutions.append(
                f"time_limit_hours: batch JSON ({batch['time_limit_hours']}) "
                f"(CLI --time-limit-hours ignored)"
                if time_limit_override is not None else
                f"time_limit_hours: batch JSON ({batch['time_limit_hours']})"
            )
        elif time_limit_override is not None:
            resolutions.append(f"time_limit_hours: CLI ({time_limit_override})")
        else:
            resolutions.append(f"time_limit_hours: default ({TIME_LIMIT_HOURS})")
        if args.max_trials is not None:
            resolutions.append(f"max_trials: CLI ({args.max_trials})")
        elif batch.get("max_trials") is not None:
            resolutions.append(f"max_trials: batch JSON ({batch['max_trials']})")
        else:
            resolutions.append("max_trials: unbounded")
        if batch.get("n_init_trials") is not None:
            resolutions.append(f"n_init_trials: batch JSON ({n_init})")
        elif args.n_init_trials is not None:
            resolutions.append(f"n_init_trials: CLI ({n_init})")
        else:
            resolutions.append(f"n_init_trials: default ({n_init})")
        if optima_note:
            resolutions.append(optima_note)

        invocation = build_invocation_log(
            argv=sys.argv,
            run_mode="batch",
            cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(
            runs_dir, landscape, max_trials,
            batch_name=batch_name, batch_config_path=batch_config_path,
            run_dir=run_dir_override, n_init_trials=n_init,
            invocation=invocation,
        )
        return

    _apply_runtime_overrides(device=args.device, time_limit_hours=time_limit_override)

    seed_X: list[torch.Tensor] = []
    if args.start_from_best:
        print("\n[seed] Loading 'start from best' hyperparameters to re-evaluate …")
        seed_X = load_seed_hparams(args.start_from_best)

    n_init = args.n_init_trials or N_INIT_TRIALS
    ackley_variant: str | None = (
        args.ackley_variant if args.dataset.startswith("ackley") else None
    )

    if args.resume:
        cfg = load_latest_run_config(runs_dir)
        if cfg.get("hparam_names") != HPARAM_NAMES:
            print("  [resume] WARNING: latest run's hparam_names differ from the current "
                  "HPARAM_SPACE; only matching trials are collected.")
        if cfg.get("time_limit_hours") is not None:
            _apply_runtime_overrides(time_limit_hours=cfg["time_limit_hours"])
        landscape, av = _landscape_from_run_config_legacy(cfg)
        if av:
            ackley_variant = av

        print("=" * 70)
        print(f"ZoMBI-Hop MOBO — RESUMING (crawling all prior runs)  |  {landscape.label}")
        print(f"Device: {DEVICE}")
        print("=" * 70)

        print("\n[collect] Crawling runs/mobo_*/mobo_progress.json for all (X,Y) pairs …")
        X_prior, Y_prior, n_runs = collect_all_observations(runs_dir)
        print(f"  [collect] {len(Y_prior)} trial(s) from {n_runs} run(s) -> prior history.")

        resolutions = [
            "landscape + true_optima: saved run_config from latest mobo_* run",
            f"time_limit_hours: saved run_config ({cfg.get('time_limit_hours')})"
            if cfg.get("time_limit_hours") is not None else
            f"time_limit_hours: default ({TIME_LIMIT_HOURS})",
        ]
        if time_limit_override is not None and cfg.get("time_limit_hours") is not None:
            resolutions.append(
                "CLI --time-limit-hours ignored on --resume when saved config has time_limit_hours"
            )
        if args.config:
            resolutions.append("--config ignored on --resume (using saved run_config)")
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )

        invocation = build_invocation_log(
            argv=sys.argv, run_mode="resume", cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                    X_prior=X_prior, Y_prior=Y_prior,
                    run_dir=run_dir_override, n_init_trials=n_init,
                    dataset=_persist_dataset_fields(cfg)[0],
                    ackley_variant=ackley_variant,
                    ensemble_spec=_ensemble_spec_from_cfg(cfg),
                    invocation=invocation)
        return

    # ── Resume-dim: seed from ALL prior runs of the same dimensionality ──
    if args.resume_dim:
        if args.dataset not in DATASET_DIMS or args.dataset == "RF":
            sys.exit("--resume-dim requires a fixed-dim analytic --dataset (e.g. ackley4d); "
                     "RF is interactive and 3-simplex only, and 'ensemble' is re-randomized "
                     "per trial (no shared landscape to pool across runs).")
        dim = DATASET_DIMS[args.dataset]
        print("=" * 70)
        print(f"ZoMBI-Hop MOBO — RESUME-DIM (all {dim}-simplex runs) — dataset: {args.dataset}")
        print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h")
        print("=" * 70)

        print(f"\n[collect] Crawling runs/mobo_*/ for dim-{dim} runs' (X,Y) pairs …")
        X_prior, Y_prior, n_runs = collect_observations_for_dim(runs_dir, dim)
        print(f"  [collect] {len(Y_prior)} trial(s) from {n_runs} run(s) -> prior history.")

        ds = _ds_ackley(args.dataset, args.ackley_variant)
        landscape = _landscape_from_legacy_ackley(ds)
        resolutions = [
            f"dataset: CLI --dataset {args.dataset} (resume-dim, all dim-{dim} runs)",
            f"ackley_variant: {args.ackley_variant}",
            f"time_limit_hours: {'CLI' if time_limit_override is not None else 'default'} "
            f"({TIME_LIMIT_HOURS})",
        ]
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )
        invocation = build_invocation_log(
            argv=sys.argv, run_mode="resume_dim", cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                    X_prior=X_prior, Y_prior=Y_prior,
                    run_dir=run_dir_override, n_init_trials=n_init,
                    dataset=args.dataset, ackley_variant=ds["ackley_variant"],
                    invocation=invocation)
        return

    if args.resume_scratch:
        print("=" * 70)
        print("ZoMBI-Hop MOBO — RESUMING FROM SCRATCH (prior data + fresh config)")
        print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h")
        print("=" * 70)

        landscape = _interactive_run_config(script_dir)

        print("\n[collect] Re-deriving prior trials against the freshly-picked optima …")
        X_prior, Y_prior, n_runs = collect_rederived_observations(
            runs_dir, landscape.maximize, landscape.true_optima)
        print(f"  [collect] {len(Y_prior)} trial(s) from {n_runs} run(s) -> prior history "
              f"(dist re-scored where needles saved, else reused; dup/runtime reused).")

        resolutions = [
            "landscape + true_optima: interactive picker (fresh config)",
            f"time_limit_hours: {'CLI' if time_limit_override is not None else 'default'} "
            f"({TIME_LIMIT_HOURS})",
        ]
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )

        invocation = build_invocation_log(
            argv=sys.argv, run_mode="resume_scratch", cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                    X_prior=X_prior, Y_prior=Y_prior,
                    run_dir=run_dir_override, n_init_trials=n_init,
                    invocation=invocation)
        return

    if args.resume_from:
        run_path = os.path.normpath(args.resume_from)
        if not os.path.isdir(run_path):
            sys.exit(f"--resume-from: directory not found: {run_path}")
        cfg = load_run_config_from_path(run_path)
        if cfg.get("hparam_names") and cfg["hparam_names"] != HPARAM_NAMES:
            print("  [resume-from] WARNING: source run's hparam_names differ from the "
                  "current HPARAM_SPACE; only matching trials are collected.")
        if cfg.get("time_limit_hours") is not None:
            _apply_runtime_overrides(time_limit_hours=cfg["time_limit_hours"])
        landscape, av = _landscape_from_run_config_legacy(cfg)
        if av:
            ackley_variant = av

        print("=" * 70)
        print(f"ZoMBI-Hop MOBO — RESUME-FROM {os.path.basename(run_path)}  |  {landscape.label}")
        print(f"Device: {DEVICE}")
        print("=" * 70)

        print(f"\n[collect] Loading prior data from {run_path} …")
        X_prior, Y_prior = collect_observations_from_run(run_path)
        print(f"  [collect] {len(Y_prior)} trial(s) -> prior history.")

        resolutions = [
            f"landscape + true_optima: saved run_config from {run_path}",
            f"time_limit_hours: saved run_config ({cfg.get('time_limit_hours')})"
            if cfg.get("time_limit_hours") is not None else
            f"time_limit_hours: default ({TIME_LIMIT_HOURS})",
        ]
        if time_limit_override is not None and cfg.get("time_limit_hours") is not None:
            resolutions.append(
                "CLI --time-limit-hours ignored on --resume-from when saved config "
                "has time_limit_hours"
            )
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )

        invocation = build_invocation_log(
            argv=sys.argv, run_mode="resume_from", cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                    X_prior=X_prior, Y_prior=Y_prior,
                    run_dir=run_dir_override, n_init_trials=n_init,
                    dataset=_persist_dataset_fields(cfg)[0],
                    ackley_variant=ackley_variant,
                    ensemble_spec=_ensemble_spec_from_cfg(cfg),
                    invocation=invocation)
        return

    if args.copy_config:
        cfg = load_run_config_from_path(args.copy_config)
        if cfg.get("hparam_names") and cfg["hparam_names"] != HPARAM_NAMES:
            print("  [copy-config] WARNING: copied run's hparam_names differ from the current "
                  "HPARAM_SPACE; only the static config is reused.")
        tl = time_limit_override if time_limit_override is not None else cfg.get("time_limit_hours")
        if tl is None and cfg.get("landscape", "rf") == "rf":
            tl = TIME_LIMIT_HOURS
            print(f"  [copy-config] no time_limit_hours in saved config — using default {tl} h")
        if tl is not None:
            _apply_runtime_overrides(time_limit_hours=tl)
            cfg = {**cfg, "time_limit_hours": tl}
        landscape, av = _landscape_from_run_config_legacy(cfg)
        if av:
            ackley_variant = av

        print("=" * 70)
        print(f"ZoMBI-Hop MOBO — COPY-CONFIG  |  {landscape.label}")
        print(f"Device: {DEVICE}   |   time limit/trial: {landscape.time_limit_hours} h")
        print("=" * 70)

        resolutions = [
            f"landscape + true_optima: copied from {args.copy_config}",
        ]
        if time_limit_override is not None:
            resolutions.append(f"time_limit_hours: CLI ({tl})")
        elif cfg.get("time_limit_hours") is not None:
            resolutions.append(f"time_limit_hours: saved run_config ({tl})")
        else:
            resolutions.append(f"time_limit_hours: default ({tl})")
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )

        invocation = build_invocation_log(
            argv=sys.argv, run_mode="copy_config", cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                    run_dir=run_dir_override, n_init_trials=n_init,
                    dataset=_persist_dataset_fields(cfg)[0],
                    ackley_variant=ackley_variant,
                    ensemble_spec=_ensemble_spec_from_cfg(cfg),
                    invocation=invocation)
        return

    if args.dataset == ENSEMBLE_DATASET:
        dim = args.dim if args.dim is not None else ENSEMBLE_DEFAULT_DIM
        if dim < 2:
            sys.exit("--dataset ensemble requires --dim >= 2.")
        print("=" * 70)
        print(f"ZoMBI-Hop MOBO — dataset ensemble (re-randomized per trial)  dim={dim}")
        print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h")
        print("=" * 70)
        landscape = build_ensemble_landscape(
            dim, optima_margin=args.ensemble_margin, seed=args.ensemble_seed,
            time_limit_hours=TIME_LIMIT_HOURS)
        ensemble_spec = {
            "dim": dim, "optima_margin": args.ensemble_margin, "seed": args.ensemble_seed,
        }
        lo, hi = optima_count_range(dim)
        resolutions = [
            f"dataset: CLI --dataset ensemble (re-randomized per trial, n_optima in [{lo}, {hi}])",
            f"ensemble_seed: {args.ensemble_seed}   ensemble_margin: {args.ensemble_margin}",
            f"time_limit_hours: {'CLI' if time_limit_override is not None else 'default'} "
            f"({TIME_LIMIT_HOURS})",
        ]
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )
        invocation = build_invocation_log(
            argv=sys.argv, run_mode="dataset_ensemble", cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                    run_dir=run_dir_override, n_init_trials=n_init,
                    dataset=ENSEMBLE_DATASET, ensemble_spec=ensemble_spec,
                    invocation=invocation)
        return

    if args.dataset != "RF":
        print("=" * 70)
        print(f"ZoMBI-Hop MOBO — dataset {args.dataset}  (variant={args.ackley_variant})")
        print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h")
        print("=" * 70)
        ds = _ds_ackley(args.dataset, args.ackley_variant)
        landscape = _landscape_from_legacy_ackley(ds)
        resolutions = [
            f"dataset: CLI --dataset {args.dataset}",
            f"ackley_variant: {args.ackley_variant}",
            f"time_limit_hours: {'CLI' if time_limit_override is not None else 'default'} "
            f"({TIME_LIMIT_HOURS})",
        ]
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )
        invocation = build_invocation_log(
            argv=sys.argv, run_mode="dataset", cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                    run_dir=run_dir_override, n_init_trials=n_init,
                    dataset=args.dataset, ackley_variant=ds["ackley_variant"],
                    invocation=invocation)
        return

    if args.landscape == "ackley":
        print("=" * 70)
        print("ZoMBI-Hop MOBO — Multi-Ackley synthetic benchmark")
        print(f"Device: {DEVICE}")
        print("=" * 70)
        if args.dim is not None or args.layout is not None:
            dim = args.dim if args.dim is not None else 10
            layout = args.layout if args.layout is not None else "1"
            b = args.ackley_b if args.ackley_b is not None else 1.2
            landscape = build_ackley_landscape(dim, layout, b=b, time_limit_hours=None)
        else:
            landscape = interactive_ackley_startup()
        resolutions = [
            "landscape: Multi-Ackley (--landscape ackley)",
            "time_limit_hours: None (max_activations budget)",
        ]
        if args.device:
            resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
        else:
            resolutions.append(
                f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
            )
        invocation = build_invocation_log(
            argv=sys.argv, run_mode="landscape_ackley", cli=_cli_snapshot(args),
            effective=_effective_run_settings(
                landscape, max_trials=max_trials, n_init_trials=n_init,
                runs_dir=runs_dir, run_dir=run_dir_override,
            ),
            resolutions=resolutions,
        )
        _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                    run_dir=run_dir_override, n_init_trials=n_init,
                    invocation=invocation)
        return

    print("=" * 70)
    label = "RF surrogate"
    print(f"ZoMBI-Hop Hyperparameter Optimisation (MOBO)  —  {label}")
    print(f"Device: {DEVICE}   |   time limit/trial: {TIME_LIMIT_HOURS} h")
    print("=" * 70)

    landscape = _interactive_run_config(script_dir)

    resolutions = [
        "landscape: interactive RF picker",
        f"time_limit_hours: {'CLI' if time_limit_override is not None else 'default'} "
        f"({TIME_LIMIT_HOURS})",
    ]
    if args.device:
        resolutions.append(f"device: CLI --device {args.device} -> {DEVICE}")
    else:
        resolutions.append(
            f"device: auto (cuda_available={torch.cuda.is_available()}) -> {DEVICE}"
        )
    invocation = build_invocation_log(
        argv=sys.argv, run_mode="fresh_rf", cli=_cli_snapshot(args),
        effective=_effective_run_settings(
            landscape, max_trials=max_trials, n_init_trials=n_init,
            runs_dir=runs_dir, run_dir=run_dir_override,
        ),
        resolutions=resolutions,
    )
    _launch_run(runs_dir, landscape, max_trials, seed_X=seed_X,
                run_dir=run_dir_override, n_init_trials=n_init,
                invocation=invocation)


if __name__ == "__main__":
    main()
