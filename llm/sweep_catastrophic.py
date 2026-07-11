"""
llm/sweep_catastrophic.py
=========================
"Catastrophic-perturbation" sweep on a *single fixed* synthetic Ensemble
landscape (``synthetic_data/ensemble.py``) — the layered oracle that
``optimize/run_mobo.py`` optimizes, NOT the RF/generative surrogate used by
``sweep_basic_RF.py`` / ``sweep_basic_surrogate.py``.

The experiment compares three groups, all on the SAME frozen landscape and with
common random numbers (repeat ``r`` of every group shares a seed, so the Sobol
init + output-noise stream are identical and trajectories diverge only through
the hyperparameters / the LLM's edits):

    1. baseline   — the hyperparameters read from ``TRIAL_JSON`` (5 repeats).
    2. perturbed  — the same hyperparameters but with ``PERTURB`` applied, i.e.
                    one or more knobs pushed to a (deliberately catastrophic)
                    value (5 repeats).
    3. llm        — START from the perturbed hyperparameters, then let the LLM
                    inject a hyperparameter change every ``INJECT_INTERVAL``
                    ZoMBI-Hop iterations, resuming the exact on-disk state each
                    time (5 repeats).  Tests whether the LLM can recover the run
                    from the bad starting point.

The very first artifact printed is a plot of the fixed Ensemble landscape
(``ensemble_landscape.png``) so you can eyeball whether the hardcoded
``ENSEMBLE_SEED`` pulled a good landscape; the sweep then proceeds automatically
(bump ``ENSEMBLE_SEED`` and re-run if you don't like the draw).

Dimensionality and the baseline hyperparameters are inferred from ``TRIAL_JSON``.
The runner is dimension-general; the coverage plot and the needles-on-landscape
plot are only produced at ``dim == 3`` (ternary), matching run_mobo.

Per trial we write:
    * ``convergence_segments.png`` — a convergence plot whose running-best line
      RESTARTS at every ZoMBI-Hop activation (zoom-out / hop to a new region),
      so one figure carries several discrete running-best segments (unlike
      run_mobo's single monotone envelope).
    * ``coverage.png``            — 3D only (coverage_plot.save_coverage_image).
    * ``needles_on_ensemble.png`` — 3D only: final declared needles overlaid on
      the Ensemble Objective background.
    * the usual points.csv / needles.csv / metrics_over_time.csv / convergence.png.

Usage:
  # repo-root uv venv (see MEMORY.md), NOT `conda activate zombi-hop`
  python llm/sweep_catastrophic.py            # full 3-group sweep
  python llm/sweep_catastrophic.py --no-llm   # baseline + perturbed only (no LLM)
  python llm/sweep_catastrophic.py --smoke-llm # one real LLM rep, shake-out
  python llm/sweep_catastrophic.py --resume   # continue newest sweep dir,
                                              # skipping reps that already have
                                              # metrics.json (else fresh sweep)

Plain-group reuse (automatic, no flag): the ``baseline`` and ``perturbed`` groups
are blind to the LLM, so a fresh sweep first copies their reps from the newest
earlier sweep whose stored signature for that group (landscape + effective
hyperparameters + budget) matches — no recompute. ``baseline`` matches on the base
hyperparameters; ``perturbed`` matches only when ``PERTURB`` (hence the perturbed
hyperparameters) is identical too. Reps are matched by seed and only missing ones
are filled, so this composes with ``--resume`` and with priors of a different
N_REPEATS.
"""

from __future__ import annotations

# ═══ HARDCODED CONFIG ════════════════════════════════════════════════════════
# Path to the MOBO trial.json whose "hparams" block seeds the baseline (and whose
# dimensionality drives the whole sweep). Relative paths resolve against repo root.
TRIAL_JSON: str = "optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112/trial.json"

# The catastrophic perturbation: {hparam_name: value, ...}. These override the
# baseline hyperparameters for BOTH the "perturbed" group and the LLM group's
# starting point. Every name must be a valid ZoMBI-Hop hyperparameter.

# extra exploitative
PERTURB: dict = {
    "ucb_beta": 0.01,
    "max_zooms": 10,
    "n_consecutive_converged": 5,
    "max_penalty_radius": 0.1
}

# extra explorative
# PERTURB: dict = {
#     "ucb_beta": 2.986,
#     "max_zooms": 2,
#     "n_consecutive_converged": 1,
#     "max_penalty_radius": 4.556
# }

# Full list of hyperparameters:
# HPARAM_SPACE: dict[str, tuple] = {
#     # Acquisition optimisation
#     "nat_grad_step":               (0.001,  0.5,   "log"),
#     "nat_grad_max_steps":          (10,     400,   "int"),
#     "n_restarts":                  (20,     300,   "int"),
#     "raw":                         (1,    300,  "int"),
#     # Acquisition function
#     "ucb_beta":                    (0.001,   3.0,   "linear"),
#     # Zoom / convergence
#     "max_zooms":                   (2,      10,    "int"),
#     "max_iterations":              (2,      30,    "int"),
#     "top_m_points":                (2,      8,     "int"),
#     "n_consecutive_converged":     (1,      5,    "int"),
#     "input_noise_threshold_mult":  (0.5,    6.0,   "linear"),
#     "output_noise_threshold_mult": (0.01,    2.0,   "linear"),
#     # Penalisation & needle
#     "max_penalty_radius":          (0.01,    5.0,   "linear"),
#     "needle_shrink_factor":        (0.1,   0.99,  "linear"),
#     "needle_stop_noise_multiplier":(1.0,    8.0,   "linear"),
#     # Point paring (deduplication)
#     "paring_spatial_halfnoise":    (0.1,    5.0,   "linear"),
#     "paring_y_noise_multiplier":   (0.1,    10.0,   "linear"),
# }

# "optimize/runs/mobo_ensemble_3d_job17402634/trial_3/trial.json"
# "hparams": {
#     "nat_grad_step": 0.24614024,
#     "nat_grad_max_steps": 10,
#     "n_restarts": 300,
#     "raw": 1,
#     "ucb_beta": 2.73573432,
#     "max_zooms": 10,
#     "max_iterations": 30,
#     "top_m_points": 8,
#     "n_consecutive_converged": 1,
#     "input_noise_threshold_mult": 2.27923891,
#     "output_noise_threshold_mult": 1.08670437,
#     "max_penalty_radius": 0.50107598,
#     "needle_shrink_factor": 0.99,
#     "needle_stop_noise_multiplier": 1.0,
#     "paring_spatial_halfnoise": 0.1,
#     "paring_y_noise_multiplier": 10.0
#   },

# Which landscape the sweep runs on:
#   "ensemble" — the layered synthetic Ensemble oracle (dimension-general), drawn
#                fresh from ENSEMBLE_SEED below. The original behaviour.
#   "rf"       — the noise-free RF conditional-mean reconstruction of the Objective
#                from llm/surrogate.py (exactly the landscape sweep_basic_RF.py
#                measures). 3D only; ENSEMBLE_SEED / ENSEMBLE_OPTIMA_MARGIN are
#                ignored, and there are NO ground-truth optima (the needle-distance
#                metric is NaN and no reference markers are drawn). dim + baseline
#                hyperparameters are still read from TRIAL_JSON.
LANDSCAPE: str = "rf"   # "ensemble" | "rf"

# Fixed Ensemble landscape draw. random_ensemble_config(dim, seed=ENSEMBLE_SEED)
# generates a FRESH landscape (the trial.json's own ensemble_configs are ignored).
# Bump this until ensemble_landscape.png looks like a landscape you want.
# (LANDSCAPE == "ensemble" only.)
ENSEMBLE_SEED: int = 1
ENSEMBLE_OPTIMA_MARGIN: float = 0.2   # optima/background gap (run_mobo default)

MAX_ITERS: int = 40          # total ZoMBI-Hop objective iterations per trial
N_REPEATS: int = 5           # repeats per group (variance)
INJECT_INTERVAL: int = 10     # LLM injects every k iterations (group 3 only)

# Multiplicative output/objective noise seen by the algorithm at EVERY sampled
# point: each measured y is perturbed by N(0, (OUTPUT_NOISE_FRAC·|y|)^2), applied
# in run_mobo.make_sim_obj + _gen_init_data. run_mobo's real-hardware default is
# 0.045 (~4.5%); crank this up to make the landscape dramatically noisier. Set to
# None to leave run_mobo.OUTPUT_NOISE_FRAC untouched.
OUTPUT_NOISE_FRAC: Optional[float] = 0.2

RESULTS_ROOT: str = "llm/results"   # sweep dir created under here (repo-root relative)
# ═════════════════════════════════════════════════════════════════════════════

import datetime
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# evaluate_llm configures sys.path for optimize/ + repo root and does the heavy
# botorch / run_mobo import; reuse its ZoMBI + LLM plumbing.
import evaluate_llm as E  # noqa: E402
import torch  # noqa: E402
import run_mobo as R  # noqa: E402
import pareto as P  # noqa: E402  (canonical MOBO-history collection; optimize/ on path)
import llm_config  # noqa: E402
from eval_metrics import (  # noqa: E402
    as_numpy,
    metric_avg_pairwise_dist,
    metric_dist_to_needles,
    metric_dup_fraction,
)
from mobo_landscapes import composition_column_names  # noqa: E402
from src.core.zombihop import ZoMBIHop  # noqa: E402

from synthetic_data.ensemble import Ensemble, random_ensemble_config  # noqa: E402

_REPO_ROOT = Path(R.__file__).resolve().parent.parent
MAXIMIZE = True
TOP_K_DROPLETS = 12
MOBO_HISTORY_TOP_K = 16


# ─── repo-root path resolution ───────────────────────────────────────────────
def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (_REPO_ROOT / q)


# ═══════════════════════════════════════════════════════════════════════════
# Trial.json → dimensionality + baseline hyperparameters
# ═══════════════════════════════════════════════════════════════════════════

def load_trial(trial_json: Path) -> Tuple[int, Dict[str, Any]]:
    """(dim, baseline_hparams) from a MOBO trial.json.

    dim is read from the first entry of ``ensemble_configs`` (the landscapes the
    trial was scored on); the baseline hyperparameters are the trial's ``hparams``
    block, filtered to the current ZoMBI-Hop hyperparameter names.
    """
    obj = json.loads(trial_json.read_text())
    hp_raw = obj.get("hparams") or {}
    hp = {k: hp_raw[k] for k in R.HPARAM_NAMES if k in hp_raw}
    missing = [k for k in R.HPARAM_NAMES if k not in hp]
    if missing:
        raise SystemExit(f"trial.json {trial_json} missing hparams: {missing}")
    cfgs = obj.get("ensemble_configs") or []
    if cfgs and "dim" in cfgs[0]:
        dim = int(cfgs[0]["dim"])
    else:
        ec = obj.get("ensemble_config") or {}
        dim = int(ec.get("dim", 3))
    return dim, hp


# ═══════════════════════════════════════════════════════════════════════════
# Fixed landscape (Ensemble oracle or noise-free RF mean) + objective
# ═══════════════════════════════════════════════════════════════════════════

# A "landscape" here is anything with a ``.predict(X)`` returning the noise-free
# Objective at composition rows ``X`` (m×dim). ``Ensemble`` satisfies this directly;
# for the RF option we use sweep_basic_surrogate._ObjMeanPredictor, which wraps the
# generative surrogate's deterministic RF conditional mean with the same interface.

# Human-readable landscape name for plot titles / colorbar labels.
LANDSCAPE_LABEL: str = "RF" if LANDSCAPE == "rf" else "Ensemble"


def _build_rf_landscape(dim: int) -> Tuple[Any, List[np.ndarray]]:
    """The noise-free RF conditional-mean landscape (sweep_basic_RF's landscape):
    load/fit the generative surrogate and wrap its mean-Objective in a predictor.
    3D only.

    Unlike the Ensemble oracle, the RF reconstruction has NO ground-truth optima, so
    ``true_optima`` is empty; the needle-distance metric is skipped (NaN) and the
    plots draw no reference-optima markers for RF runs."""
    if dim != 3:
        raise SystemExit(f"LANDSCAPE='rf' is 3D only, but TRIAL_JSON has dim={dim}")
    # Lazy import: only pull in the surrogate stack (sklearn/pickle) for RF runs.
    import sweep_basic_surrogate as SBS  # noqa: E402
    if SBS.SURROGATE_PICKLE and Path(SBS.SURROGATE_PICKLE).exists():
        print(f"  [landscape] loading surrogate ← {SBS.SURROGATE_PICKLE}")
        surr = SBS.Surrogate.load(SBS.SURROGATE_PICKLE)
    else:
        print("  [landscape] fitting generative surrogate …")
        surr = SBS.Surrogate.fit(verbose=False)
    predictor = SBS._ObjMeanPredictor(surr)
    print("  [landscape] noise-free RF mean (llm/surrogate.py); no ground-truth optima")
    return predictor, []


def _build_ensemble_landscape(dim: int) -> Tuple[Ensemble, List[np.ndarray]]:
    """The single frozen Ensemble landscape used by every trial in the sweep."""
    cfg = random_ensemble_config(dim, index=0, total=1, seed=ENSEMBLE_SEED,
                                 optima_margin=ENSEMBLE_OPTIMA_MARGIN)
    ens = Ensemble(**cfg)
    true_optima = [np.asarray(c, float) for c in ens.centers]
    print(f"  [landscape] {ens!r}\n"
          f"  [landscape] {len(true_optima)} true optima "
          f"(from {len(ens.peak_centers)} placed basins), seed={ENSEMBLE_SEED}")
    return ens, true_optima


def build_landscape(dim: int) -> Tuple[Any, List[np.ndarray]]:
    """The single frozen landscape used by every trial, dispatched on ``LANDSCAPE``.

    Returns ``(landscape, true_optima)`` where ``landscape`` exposes ``.predict(X)``
    (Objective at composition rows) — an ``Ensemble`` for ``LANDSCAPE == 'ensemble'``
    or an RF mean predictor for ``LANDSCAPE == 'rf'``."""
    if LANDSCAPE == "rf":
        return _build_rf_landscape(dim)
    if LANDSCAPE == "ensemble":
        return _build_ensemble_landscape(dim)
    raise SystemExit(f"LANDSCAPE must be 'ensemble' or 'rf', got {LANDSCAPE!r}")


def make_objective(landscape):
    """(fn_callable, feature_log): ``fn(x)`` evaluates the noise-free landscape
    Objective at composition ``x`` and logs (composition, Objective) per droplet.

    Works on any landscape exposing ``.predict`` (the Ensemble oracle or the RF
    conditional mean). The physics print-model + output noise are applied downstream
    by ``run_mobo.make_sim_obj`` exactly as in a real trial; ``feature_log`` records
    the clean landscape value at the measured (physics) composition — the quantity
    the LLM prompt and the needle plots report."""
    feature_log: List[Dict[str, Any]] = []

    def fn(x) -> float:
        comp = np.asarray(x, float)
        s = comp.sum()
        comp = comp / (s if s != 0 else 1.0)
        val = float(landscape.predict(comp.reshape(1, -1))[0])
        feature_log.append({"comp": comp.tolist(), "Objective": val})
        return val

    return fn, feature_log


# ═══════════════════════════════════════════════════════════════════════════
# Landscape preview plot (first artifact)
# ═══════════════════════════════════════════════════════════════════════════

_TERNARY_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
_TERNARY_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")


def _bary_to_cart(pts) -> np.ndarray:
    return np.atleast_2d(np.asarray(pts, float)) @ _TERNARY_VERTS


def plot_ensemble_landscape(out_png: Path, ens: Ensemble, true_optima, dim: int) -> Path:
    """First artifact: draw the fixed Ensemble landscape so the seed can be judged.

    dim == 3 → filled ternary contour of the Objective with true optima marked.
    Other dims → distribution of Objective over Dirichlet samples (no ternary)."""
    import matplotlib.pyplot as plt  # Agg already selected via evaluate_llm

    if dim == 3:
        grid_pts = R.ternary_grid(R.TERNARY_GRID_N)
        grid_vals = ens.predict(grid_pts)
        xy = _bary_to_cart(grid_pts)
        fig, ax = plt.subplots(figsize=(6.6, 5.8))
        tcf = ax.tricontourf(xy[:, 0], xy[:, 1], np.asarray(grid_vals, float),
                             levels=24, cmap="viridis")
        fig.colorbar(tcf, ax=ax, shrink=0.82, label=f"{LANDSCAPE_LABEL} Objective")
        tri = np.vstack([_TERNARY_VERTS, _TERNARY_VERTS[:1]])
        ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1.4, zorder=3)
        offs = [(-0.06, -0.05), (1.06, -0.05), (0.5, np.sqrt(3.0) / 2.0 + 0.05)]
        for lab, (ox, oy) in zip(_TERNARY_LABELS, offs):
            ax.text(ox, oy, lab, ha="center", va="center", fontsize=10, fontweight="bold")
        if len(true_optima):
            tc = _bary_to_cart(np.asarray(true_optima, float))
            ax.scatter(tc[:, 0], tc[:, 1], marker="*", s=240, c="red",
                       edgecolors="white", linewidths=1.0, zorder=5,
                       label=f"true optima (n={len(true_optima)})")
            ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Fixed {LANDSCAPE_LABEL} landscape (dim=3" + (f", seed={ENSEMBLE_SEED}" if LANDSCAPE == "ensemble" else "") + ")", fontsize=11)
    else:
        rng = np.random.default_rng(12345)
        samples = rng.dirichlet(np.ones(dim), size=40000)
        vals = ens.predict(samples)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.hist(vals, bins=60, color="steelblue", alpha=0.85)
        for c in true_optima:
            ax.axvline(float(ens.predict(np.asarray(c).reshape(1, -1))[0]),
                       color="red", lw=0.8, alpha=0.6)
        ax.set_xlabel(f"{LANDSCAPE_LABEL} Objective")
        ax.set_ylabel("count (Dirichlet samples)")
        ax.set_title(f"Fixed {LANDSCAPE_LABEL} landscape (dim={dim}, seed={ENSEMBLE_SEED}); "
                     f"{len(true_optima)} true optima (red)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")
    return out_png


def plot_needles_on_ensemble(out_png: Path, needles: np.ndarray, ens: Ensemble,
                             true_optima, title: Optional[str] = None) -> Optional[Path]:
    """3D only: Ensemble Objective as a filled ternary contour with the run's final
    declared needles overlaid as red stars (true optima as hollow white rings)."""
    import matplotlib.pyplot as plt

    grid_pts = R.ternary_grid(R.TERNARY_GRID_N)
    grid_vals = ens.predict(grid_pts)
    xy = _bary_to_cart(grid_pts)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    tcf = ax.tricontourf(xy[:, 0], xy[:, 1], np.asarray(grid_vals, float),
                         levels=24, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.82, label=f"{LANDSCAPE_LABEL} Objective")
    tri = np.vstack([_TERNARY_VERTS, _TERNARY_VERTS[:1]])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1.4, zorder=3)
    offs = [(-0.06, -0.05), (1.06, -0.05), (0.5, np.sqrt(3.0) / 2.0 + 0.05)]
    for lab, (ox, oy) in zip(_TERNARY_LABELS, offs):
        ax.text(ox, oy, lab, ha="center", va="center", fontsize=10, fontweight="bold")
    if true_optima is not None and len(true_optima):
        tc = _bary_to_cart(np.asarray(true_optima, float))
        ax.scatter(tc[:, 0], tc[:, 1], marker="o", s=90, facecolors="none",
                   edgecolors="white", linewidths=1.4, zorder=4, label="true optima")
    needles = np.atleast_2d(np.asarray(needles, float)) if needles is not None else np.empty((0, 3))
    if needles.size and needles.shape[0]:
        nc = _bary_to_cart(needles)
        ax.scatter(nc[:, 0], nc[:, 1], marker="*", s=260, c="red",
                   edgecolors="white", linewidths=1.0, zorder=5,
                   label=f"final needles (n={needles.shape[0]})")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"Final needles on {LANDSCAPE_LABEL} Objective landscape", fontsize=11)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ═══════════════════════════════════════════════════════════════════════════
# Segmented convergence plot (running-best restarts at each activation)
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence_segments(out_png: Path, dh, snap_records: List[tuple],
                              maximize: bool, title: Optional[str] = None) -> Optional[Path]:
    """Convergence plot whose running-best line RESTARTS at every ZoMBI-Hop
    activation (zoom-out / hop to a new region), so the figure shows several
    discrete running-best segments rather than one monotone envelope.

    Each measured droplet is bucketed to its activation via the same snapshot
    records used elsewhere (see ``run_mobo._activation_zoom_per_point``); within
    each activation the running max is accumulated from that activation's first
    point."""
    import matplotlib.pyplot as plt

    if dh.Y_all is None or dh.Y_all.numel() == 0:
        return None
    Y = as_numpy(dh.Y_all).ravel().astype(float)
    if not maximize:
        Y = -Y
    n = Y.size
    act, _zm = R._activation_zoom_per_point(n, snap_records)
    idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.scatter(idx, Y, s=10, alpha=0.30, color="#999999", zorder=2, label="measured")

    # Contiguous runs of equal activation → one restarting running-best segment.
    boundaries = np.flatnonzero(np.diff(act)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n]])
    labeled = False
    for s, e in zip(starts, ends):
        seg_best = np.maximum.accumulate(Y[s:e])
        kw = dict(color="darkorange", lw=1.9, zorder=4)
        if not labeled:
            kw["label"] = "running best (resets per activation)"
            labeled = True
        ax.plot(idx[s:e], seg_best, **kw)
    # Activation boundaries as faint vertical markers.
    vlabeled = False
    for b in boundaries:
        kw = dict(color="crimson", alpha=0.45, lw=0.9, ls="--")
        if not vlabeled:
            kw["label"] = "activation / zoom-out"
            vlabeled = True
        ax.axvline(float(b), **kw)

    ax.set_xlabel("Sample index (droplet)")
    ax.set_ylabel("Objective Y")
    ax.set_title(title or f"Convergence — running best resets per activation "
                          f"({int(act.max()) + 1 if n else 0} activations)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ═══════════════════════════════════════════════════════════════════════════
# ZoMBI-Hop segment runner (dimension-general; checkpoint/resume for injection)
# ═══════════════════════════════════════════════════════════════════════════

_CONFIG_READ = E._CONFIG_READ_HPARAMS
_CONSTRUCTOR = E._CONSTRUCTOR_HPARAMS


def _patch_config(run_dir: Path, hp: Dict[str, Any]) -> None:
    """Write config-read hyperparameters into a checkpoint's config.json so a
    resume applies them (load_state overwrites constructor values from this file)."""
    cfg_path = run_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    for k, v in hp.items():
        if k in _CONFIG_READ:
            cfg[k] = v
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def run_segment(ckpt_dir: Path, run_uuid: str, fresh: bool, hp: Dict[str, Any],
                fn_callable, dim: int, stop_at: int, call_counter: List[int],
                payloads: List[dict], snap_records: List[tuple]):
    """Run ZoMBI-Hop until the global objective-call counter reaches ``stop_at``.

    ``fresh`` cold-starts a run (initial design + config.json from ``hp``); else it
    resumes the exact on-disk state with config-read hyperparameters patched and
    constructor ones passed as kwargs. Returns the live DataHandler."""
    run_dir = ckpt_dir / f"run_{run_uuid}"
    constructor_hp = {k: v for k, v in hp.items() if k in _CONSTRUCTOR}

    plot_state: Dict[str, Any] = {"line_0": None, "line_1": None}
    sim_obj = R.make_sim_obj(fn_callable, R.DEVICE, R.DTYPE, maximize=MAXIMIZE)
    inner = R.make_linebo_wrapper(sim_obj, dim, R.NUM_LINES, R.DEVICE, R.DTYPE, plot_state)
    dh_ref: List[Any] = [None]

    def obj_wrapper(x_tell, bounds, acq_fn):
        if call_counter[0] >= stop_at:
            raise E.BudgetExhausted()
        x_req, x_act, y = inner(x_tell, bounds, acq_fn)
        call_counter[0] += 1
        dh = dh_ref[0]
        needles = dh.needles
        payloads.append(dict(
            iter_num=call_counter[0],
            needles=(as_numpy(needles) if needles is not None and needles.shape[0] > 0 else None),
            needle_vals=(as_numpy(dh.needle_vals).ravel()
                         if dh.needle_vals is not None and dh.needle_vals.shape[0] > 0 else None),
            line_0=plot_state.get("line_0"), line_1=plot_state.get("line_1"),
            n_points_before=(dh.X_all_actual.shape[0] if dh.X_all_actual is not None else 0),
        ))
        return x_req, x_act, y

    hp_run = dict(hp)
    if dim > 3 and (hp_run.get("top_m_points") is None or hp_run.get("top_m_points", 0) < dim + 1):
        hp_run["top_m_points"] = max(dim + 1, 4)

    if fresh:
        X_a, X_e, Y = R._gen_init_data(fn_callable, MAXIMIZE, dim=dim)
        optimizer = ZoMBIHop(
            objective=obj_wrapper, X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
            **R.ZOMBI_FIXED, **hp_run, device=str(R.DEVICE), dtype=R.DTYPE,
            run_uuid=run_uuid, checkpoint_dir=str(ckpt_dir), resume=False,
        )
    else:
        _patch_config(run_dir, hp)
        dummy_Xa = torch.full((1, dim), 1.0 / dim, device=R.DEVICE, dtype=R.DTYPE)
        dummy_Y = torch.zeros(1, 1, device=R.DEVICE, dtype=R.DTYPE)
        optimizer = ZoMBIHop(
            objective=obj_wrapper, X_init_actual=dummy_Xa,
            X_init_expected=dummy_Xa.clone(), Y_init=dummy_Y,
            input_noise=R.NOISE_LEVEL, acquisition_type="ucb", max_gp_points=3000,
            device=str(R.DEVICE), dtype=R.DTYPE, verbose=False,
            run_uuid=run_uuid, resume=True, checkpoint_dir=str(ckpt_dir),
            **{k: v for k, v in constructor_hp.items()},
        )
    dh = optimizer.data_handler
    dh_ref[0] = dh

    orig_snap = dh.take_snapshot

    def snap_wrap(*a, **k):
        orig_snap(*a, **k)
        if dh.X_all_actual is not None:
            czb = dh.current_zoom_bounds if dh.current_zoom_bounds is not None else dh.bounds
            zoom_size = R.zoom_size_fraction(czb) if czb is not None else 1.0
            snap_records.append((dh.X_all_actual.shape[0], dh.current_activation,
                                 dh.current_zoom, zoom_size))
    dh.take_snapshot = snap_wrap

    try:
        # never_terminate: the optimiser must not stop on its own (over-penalisation,
        # activation failure, noise-floor exhaustion); it only ends when obj_wrapper
        # raises BudgetExhausted at stop_at, so every group spends the full budget.
        optimizer.run(max_activations=float("inf"), time_limit_hours=None,
                      never_terminate=True)
    except E.BudgetExhausted:
        pass
    except Exception as e:
        print(f"      [segment] ZoMBI-Hop stopped early: {e}")
    return dh


# ═══════════════════════════════════════════════════════════════════════════
# Finalize a trial: metrics + artifacts
# ═══════════════════════════════════════════════════════════════════════════

def finalize_trial(dh, true_optima, payloads, snap_records, trial_dir: Path,
                   ens: Ensemble, dim: int) -> Dict[str, Any]:
    needle_t = dh.get_all_needle_locations()
    discovered = as_numpy(needle_t) if needle_t.numel() > 0 else np.empty((0, dim))
    needle_vals_t = dh.get_all_needle_vals()
    best_needle = (float(as_numpy(needle_vals_t).ravel().max())
                   if needle_vals_t.numel() > 0 else float("nan"))
    X_all = as_numpy(dh.X_all_actual) if dh.X_all_actual is not None else np.empty((0, dim))
    Y_all = as_numpy(dh.Y_all).ravel() if dh.Y_all is not None else np.empty((0,))
    best_obj = float(Y_all.max()) if Y_all.size else float("nan")
    dist = metric_dist_to_needles(discovered, true_optima, dim=dim) if len(true_optima) else float("nan")
    zoom_sizes = R._zoom_size_per_point(X_all.shape[0], snap_records) if X_all.shape[0] else None
    dup = (metric_dup_fraction(X_all, dim=dim, zoom_sizes=zoom_sizes)
           if X_all.shape[0] else float("nan"))
    needle_spread = (metric_avg_pairwise_dist(discovered)
                     if discovered.shape[0] >= 2 else float("nan"))

    trial_dir.mkdir(parents=True, exist_ok=True)
    try:
        R.write_points_csv(str(trial_dir / "points.csv"), dh, snap_records, dim=dim)
        R.write_needles_csv(str(trial_dir / "needles.csv"), dh, dim=dim)
        R.write_metrics_over_time_csv(str(trial_dir / "metrics_over_time.csv"),
                                      payloads, X_all, true_optima, dim=dim)
    except Exception as e:
        print(f"      [finalize] CSV write failed: {e}")

    # Segmented convergence plot (the headline artifact of this sweep).
    try:
        plot_convergence_segments(trial_dir / "convergence_segments.png", dh,
                                  snap_records, MAXIMIZE)
    except Exception as e:
        print(f"      [finalize] segmented convergence plot failed: {e}")
    # run_mobo's standard single-envelope convergence for reference.
    try:
        R.plot_convergence(str(trial_dir / "convergence.png"), dh, MAXIMIZE)
    except Exception as e:
        print(f"      [finalize] convergence plot failed: {e}")

    if dim == 3:
        # Per-trial ground truth so coverage_plot draws THIS landscape.
        try:
            grid_pts = R.ternary_grid(R.TERNARY_GRID_N)
            np.savez(trial_dir / "coverage_ground_truth.npz",
                     grid_pts=np.asarray(grid_pts, float),
                     grid_vals=np.asarray(ens.predict(grid_pts), float),
                     true_optima=np.asarray(true_optima, float))
        except Exception as e:
            print(f"      [finalize] ground-truth write failed: {e}")
        try:
            import coverage_plot
            coverage_plot.save_coverage_image(str(trial_dir))
        except Exception as e:
            print(f"      [finalize] coverage plot failed: {e}")
        try:
            plot_needles_on_ensemble(trial_dir / "needles_on_ensemble.png",
                                     discovered, ens, true_optima)
        except Exception as e:
            print(f"      [finalize] needles-on-ensemble plot failed: {e}")

    return {
        "n_iters": len(payloads),
        "n_points_total": int(X_all.shape[0]),
        "n_needles": int(discovered.shape[0]),
        "best_objective": best_obj,
        "best_needle": best_needle,
        "dist_to_true_optima": float(dist),
        "dup_fraction": float(dup),
        "mean_pairwise_needle_dist": float(needle_spread),
        "Y_all_running_best": np.maximum.accumulate(Y_all).tolist() if Y_all.size else [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Offline MOBO hyperparameter-search history (collected like optimize/pareto.py)
# ═══════════════════════════════════════════════════════════════════════════

def _mobo_run_dir() -> Path:
    """The MOBO run directory ``TRIAL_JSON`` was drawn from — its grandparent, i.e.
    the single run dir you would hand to ``optimize/pareto.py``. For
    ``optimize/runs/mobo_ensemble_3d_job.../trial_3/trial.json`` this is
    ``optimize/runs/mobo_ensemble_3d_job...``."""
    return _resolve(TRIAL_JSON).parent.parent


def _fmt_hist_num(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:.5g}"
    return str(v)


def _same_hparams(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True iff two hparam dicts agree on every ZoMBI-Hop hyperparameter (float
    tolerance for numerics). Used to spot the baseline config in the pooled history."""
    for name in R.HPARAM_NAMES:
        va, vb = a.get(name), b.get(name)
        if va is None or vb is None:
            return False
        try:
            if abs(float(va) - float(vb)) > 1e-9:
                return False
        except (TypeError, ValueError):
            if va != vb:
                return False
    return True


def mobo_history_block(run_dir: Path, top_k: int, base_hp: Dict[str, Any]) -> str:
    """Offline hyperparameter-search history for the injection prompt.

    Collects EXACTLY what ``optimize/pareto.py`` would for ``run_dir``: mirroring its
    single-run path, it pools the run's config-signature-matching sibling runs
    (shared history) and collaborators' runs directories, reads each trial's three
    MINIMIZED objectives (``dist_to_needles``, ``dup_fraction``, and a time metric),
    keeps only the Pareto-optimal trials, then shows the best ``top_k`` of those by
    ``dist_to_needles`` (best needle-location first).

    Any pooled trial whose hyperparameters match ``base_hp`` (the config TRIAL_JSON
    was drawn from, which the perturbed/LLM runs start from) is dropped from the pool
    BEFORE the front is taken, so the LLM can't recognise the current setting as that
    baseline row with one knob moved and reverse-engineer the perturbation."""
    run_dir = Path(run_dir)
    # Mirror pareto.main's single-run path: a signature pools signature-matching
    # siblings under the parent runs dir + collaborators; no signature (no
    # run_config.json) falls back to this one run's own trials.
    only_signature = P._load_run_signature(str(run_dir))
    if only_signature is not None:
        crawl_dirs = P._dedup_realpath([str(run_dir.parent)] + P._collaborator_runs_dirs())
    else:
        crawl_dirs = [str(run_dir)]

    records: List[dict] = []
    for d in crawl_dirs:
        records += P.collect_trials(d, exclude_old=True, only_signature=only_signature)
    # Hide the baseline config: drop every trial whose hparams equal base_hp so the
    # front is computed over — and shows — only the OTHER trials.
    n_pooled = len(records)
    records = [r for r in records if not _same_hparams(r.get("hparams", {}), base_hp)]
    n_base_dropped = n_pooled - len(records)
    if not records:
        return f"(no usable MOBO trials found for {run_dir.name})"

    # Pareto front over the three MINIMIZED objectives (dist, dup, time), verbatim
    # pareto.py column order.
    M = np.array([[r["metrics"][P.DIST_KEY], r["metrics"][P.DUP_KEY], r["time_value"]]
                  for r in records], dtype=float)
    pareto = [records[i] for i in np.where(P.pareto_mask_min(M))[0]]
    n_pareto = len(pareto)

    def _dist(r) -> float:
        try:
            d = float(r["metrics"].get(P.DIST_KEY))
        except (TypeError, ValueError):
            return float("inf")
        return d if np.isfinite(d) else float("inf")
    pareto.sort(key=_dist)
    shown = pareto[:top_k]

    # Time-column label follows the collected trials' key(s) (pareto.py convention:
    # current runs write avg_time_per_iter_s, older ones runtime_s).
    time_keys = {r["time_key"] for r in shown}
    if time_keys == {"runtime_s"}:
        time_label = "runtime_s"
    elif time_keys == {"avg_time_per_iter_s"}:
        time_label = "avg_time_per_iter_s"
    else:
        time_label = "avg_time_per_iter_s|runtime_s"

    header = ["trial"] + list(R.HPARAM_NAMES) + [P.DIST_KEY, P.DUP_KEY, time_label]
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    for r in shown:
        hp = r.get("hparams", {})
        row = [str(r.get("trial", ""))]
        row += [_fmt_hist_num(hp.get(name)) for name in R.HPARAM_NAMES]
        row += [_fmt_hist_num(r["metrics"].get(P.DIST_KEY)),
                _fmt_hist_num(r["metrics"].get(P.DUP_KEY)),
                _fmt_hist_num(r["time_value"])]
        lines.append(" | ".join(row))

    note = (f"{len(shown)} of {n_pareto} Pareto-optimal trials shown "
            f"(top {top_k} by lowest dist_to_needles, best-first) "
            f"out of {len(records)} pooled MOBO trials.")
    if n_base_dropped:
        print(f"      [mobo-history] dropped {n_base_dropped} baseline-matching "
              f"trial(s) from the pooled history")
    return f"{note}\n\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# LLM injection prompt (Objective-only, dimension-general)
# ═══════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = """\
You are helping tune a Bayesian-optimization algorithm, ZoMBI-Hop, on the fly. It
is optimizing a synthetic materials-discovery objective over the {dim}-simplex
(compositions summing to 1); the `Objective` is MAXIMIZED. ZoMBI-Hop is a zooming
multi-basin optimizer that hunts MULTIPLE optima ("needles"): it fits a GP, uses
an acquisition function each iteration to pick a LineBO line of measured points,
declares a needle when expected improvement hits the output-noise floor, penalizes
that region, and moves on.

## The ZoMBI-Hop hyperparameters you may tune

(ranges are the allowed bounds; stay inside them)
{hparam_descriptions}

## Current ZoMBI-Hop hyperparameters (in effect right now)

{current_hparams}

## Offline hyperparameter-optimization history

Each row is one offline trial: its hyperparameter values followed by three
MINIMIZED objectives — `dist_to_needles`, `dup_fraction`, `runtime_s`. Use it as a
strong prior on which hyperparameter regions are good.

{hparam_search_history}

## This run so far

This is injection #{injection_idx}. ZoMBI-Hop has completed {iters_done} of
{budget} iterations. Progress:

{progress_summary}

## Change since the last injection

{trend_summary}

Recent measured points (composition + `Objective`), most recent last:

{history_table}

Best point so far:

{best_point_summary}

## Top {top_k} points by Objective

{top_k_summary}

## Declared needles (local optima)

{needle_summary}

## Your task

Decide whether to change any ZoMBI-Hop hyperparameters for the NEXT {interval}
iteration(s). ZoMBI-Hop continues from its exact current internal state (same data,
needles, zoom bounds) with whatever you specify; anything you don't specify keeps
its current value. Prioritize climbing the global optimum AND discovering distinct
secondary high-value basins, rather than re-sampling one basin. Then answer through
the schema:

- `reasoning`: a concise, quantitative justification (2-6 sentences) grounded in
  the numbers above.
- `hyperparameter_changes`: a list of {{"name", "value"}} entries — ONLY the
  hyperparameters you want to change, each within its allowed range. Leave the list
  EMPTY if the current settings are already appropriate.
"""


def _comp_str(comp, comp_cols) -> str:
    return ", ".join(f"{c}={v:.3f}" for c, v in zip(comp_cols, comp))


def _progress_summary(feature_log, iters_done, budget) -> str:
    n = len(feature_log)
    if n == 0:
        return "No measured points yet."
    obj = np.array([r["Objective"] for r in feature_log], float)
    return (f"- Measured points so far: {n} ({iters_done}/{budget} iterations).\n"
            f"- Objective range: min {obj.min():.4f}, max {obj.max():.4f}, "
            f"mean {obj.mean():.4f}.")


def _trend_summary(feature_log, prev_best) -> str:
    if not feature_log:
        return "(no points yet)"
    cur = max(r["Objective"] for r in feature_log)
    if prev_best is None:
        return f"First injection; best Objective so far is {cur:.4f}."
    d = cur - prev_best
    verb = "improved" if d > 1e-9 else ("unchanged" if abs(d) <= 1e-9 else "regressed")
    return f"Best Objective {verb}: {prev_best:.4f} → {cur:.4f} (Δ {d:+.4f})."


def _history_table(feature_log, comp_cols, max_rows=48) -> str:
    if not feature_log:
        return "(no measured points yet)"
    rows = feature_log[-max_rows:]
    omitted = len(feature_log) - len(rows)
    header = " | ".join(comp_cols) + " | Objective"
    lines = [header]
    if omitted > 0:
        lines.append(f"... ({omitted} earlier points omitted)")
    for r in rows:
        comp = " | ".join(f"{v:.3f}" for v in r["comp"])
        lines.append(f"{comp} | {r['Objective']:.4f}")
    return "\n".join(lines)


def _best_point_summary(feature_log, comp_cols) -> str:
    if not feature_log:
        return "(none yet)"
    b = max(feature_log, key=lambda r: r["Objective"])
    return f"- Objective {b['Objective']:.4f} at {_comp_str(b['comp'], comp_cols)}"


def _top_k_summary(feature_log, comp_cols, k) -> str:
    if not feature_log:
        return "(none yet)"
    ranked = sorted(feature_log, key=lambda r: r["Objective"], reverse=True)[:k]
    header = "rank | " + " | ".join(comp_cols) + " | Objective"
    lines = [header]
    for i, r in enumerate(ranked, 1):
        comp = " | ".join(f"{v:.3f}" for v in r["comp"])
        lines.append(f"{i} | {comp} | {r['Objective']:.4f}")
    return "\n".join(lines)


def _needle_summary(dh, comp_cols) -> str:
    loc = dh.get_all_needle_locations()
    if loc is None or loc.numel() == 0:
        return "(no needles declared yet)"
    locs = as_numpy(loc, dtype=float)
    vals_t = dh.get_all_needle_vals()
    vals = (as_numpy(vals_t, dtype=float).ravel() if vals_t.numel() > 0
            else np.full(len(locs), np.nan))
    header = "needle | " + " | ".join(comp_cols) + " | declared Objective"
    lines = [header]
    for i in range(len(locs)):
        comp = " | ".join(f"{v:.3f}" for v in locs[i])
        v = vals[i] if i < len(vals) else float("nan")
        lines.append(f"{i + 1} | {comp} | {v:.4f}")
    return "\n".join(lines)


def build_injection_prompt(feature_log, dh, hp, injection_idx, iters_done, budget,
                           interval, dim, comp_cols, prev_best=None) -> str:
    return PROMPT_TEMPLATE.format(
        dim=dim,
        hparam_descriptions=E.hparam_descriptions_block(),
        current_hparams=E.format_current_hparams(hp),
        hparam_search_history=mobo_history_block(
            _mobo_run_dir(), MOBO_HISTORY_TOP_K, load_trial(_resolve(TRIAL_JSON))[1]),
        injection_idx=injection_idx,
        iters_done=iters_done,
        budget=budget,
        progress_summary=_progress_summary(feature_log, iters_done, budget),
        trend_summary=_trend_summary(feature_log, prev_best),
        history_table=_history_table(feature_log, comp_cols),
        best_point_summary=_best_point_summary(feature_log, comp_cols),
        top_k=TOP_K_DROPLETS,
        top_k_summary=_top_k_summary(feature_log, comp_cols, TOP_K_DROPLETS),
        needle_summary=_needle_summary(dh, comp_cols),
        interval=interval,
    )


# ═══════════════════════════════════════════════════════════════════════════
# One trial: baseline / perturbed (no LLM), or LLM-injection
# ═══════════════════════════════════════════════════════════════════════════

def run_plain_trial(ens, true_optima, hp, dim, seed, trial_dir: Path,
                    source: str) -> Dict[str, Any]:
    """Whole budget in one cold-started run, no LLM (baseline or perturbed group)."""
    E._seed_everything(seed)
    fn_callable, feature_log = make_objective(ens)
    tmp = Path(tempfile.mkdtemp(prefix="zombi_cat_"))
    payloads: List[dict] = []
    snap_records: List[tuple] = []
    call_counter = [0]
    try:
        dh = run_segment(tmp, "plain", True, hp, fn_callable, dim, MAX_ITERS,
                         call_counter, payloads, snap_records)
        metrics = finalize_trial(dh, true_optima, payloads, snap_records, trial_dir,
                                 ens, dim)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    metrics["source"] = source
    metrics["n_injections"] = 0
    metrics["n_changes"] = 0
    return metrics


def run_llm_trial(ens, true_optima, start_hp, dim, seed, trial_dir: Path,
                  comp_cols) -> Dict[str, Any]:
    """Cold-start from ``start_hp`` (the perturbed setting), then inject the LLM
    every ``INJECT_INTERVAL`` iterations, resuming ZoMBI-Hop's exact state."""
    E._seed_everything(seed)
    fn_callable, feature_log = make_objective(ens)
    tmp = Path(tempfile.mkdtemp(prefix="zombi_cat_"))
    inj_dir = trial_dir / "injections"
    payloads: List[dict] = []
    snap_records: List[tuple] = []
    call_counter = [0]
    hp = dict(start_hp)
    injections: List[Dict[str, Any]] = []
    n_changes = 0
    prev_best: Optional[float] = None

    try:
        fresh = True
        injection_idx = 0
        dh = None
        while call_counter[0] < MAX_ITERS:
            stop_at = min(call_counter[0] + INJECT_INTERVAL, MAX_ITERS)
            before = call_counter[0]
            dh = run_segment(tmp, "llm", fresh, hp, fn_callable, dim, stop_at,
                             call_counter, payloads, snap_records)
            fresh = False
            if call_counter[0] >= MAX_ITERS:
                break
            if call_counter[0] == before:
                print(f"      [trial] no progress at iter {before}; stopping trial")
                break

            prompt = build_injection_prompt(feature_log, dh, hp, injection_idx,
                                            call_counter[0], MAX_ITERS, INJECT_INTERVAL,
                                            dim, comp_cols, prev_best)
            prev_best = max((r["Objective"] for r in feature_log), default=None)
            this_dir = inj_dir / f"inj_{injection_idx:02d}"
            this_dir.mkdir(parents=True, exist_ok=True)
            (this_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            llm_out = llm_config.call_llm(prompt, R.HPARAM_NAMES)
            decision = llm_out.get("decision") or {}
            raw_changes = decision.get("hyperparameter_changes", []) if isinstance(decision, dict) else []
            changes, warns = E.validate_changes(raw_changes)
            if changes:
                hp.update(changes)
                n_changes += 1
            rec = {
                "injection_idx": injection_idx,
                "iters_done": call_counter[0],
                "latency_s": llm_out["latency_s"],
                "error": llm_out["error"],
                "reasoning": decision.get("reasoning") if isinstance(decision, dict) else None,
                "validated_changes": changes,
                "validation_warnings": warns,
                "hparams_after": dict(hp),
            }
            injections.append(rec)
            (this_dir / "decision.json").write_text(json.dumps(rec, indent=2))
            print(f"      inj {injection_idx} @ iter {call_counter[0]}: "
                  f"{('CHANGE ' + str(changes)) if changes else 'KEEP'} "
                  f"({llm_out['latency_s']:.1f}s)"
                  + (f"  [ERROR: {llm_out['error']}]" if llm_out["error"] else ""))
            if llm_out["error"]:
                raise llm_config.LLMCallError(
                    f"LLM call failed at injection {injection_idx} "
                    f"(iter {call_counter[0]}): {llm_out['error']}")
            injection_idx += 1

        metrics = finalize_trial(dh, true_optima, payloads, snap_records, trial_dir,
                                 ens, dim)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    (trial_dir / "injections.json").write_text(json.dumps(injections, indent=2))
    metrics["source"] = f"llm_every_{INJECT_INTERVAL}"
    metrics["n_injections"] = len(injections)
    metrics["n_changes"] = n_changes
    total_latency = float(sum((inj.get("latency_s") or 0.0) for inj in injections))
    n_iters = metrics.get("n_iters") or 0
    metrics["llm_total_latency_s"] = total_latency
    metrics["llm_latency_per_iter_s"] = total_latency / n_iters if n_iters else 0.0
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def _group_signature(dim: int, hp: Dict[str, Any]) -> Dict[str, Any]:
    """The config fields that FULLY determine a plain (no-LLM) group's output.

    Such a group runs a fixed hyperparameter set ``hp`` on the frozen landscape with
    the fixed common-random-number seeds, so two sweeps that agree on all of these
    produce byte-identical trajectories — meaning one can copy the other's reps
    instead of recomputing them. ``INJECT_INTERVAL``, ``N_REPEATS`` and every LLM
    knob are intentionally EXCLUDED (a plain group is blind to them; reps are matched
    individually by seed). For ``baseline`` ``hp`` is ``base_hp``; for ``perturbed``
    it is ``base_hp`` with ``PERTURB`` applied — so a perturbed group only matches
    when PERTURB (and the base it modifies) is identical."""
    return {
        "landscape": LANDSCAPE,
        "ensemble_seed": ENSEMBLE_SEED,
        "ensemble_optima_margin": ENSEMBLE_OPTIMA_MARGIN,
        "dim": dim,
        "max_iters": MAX_ITERS,
        "output_noise_frac": OUTPUT_NOISE_FRAC,
        "trial_json": str(_resolve(TRIAL_JSON)),
        "hp": {k: hp.get(k) for k in R.HPARAM_NAMES},
    }


def _group_sig_match(a: Optional[Dict[str, Any]], b: Dict[str, Any]) -> bool:
    """True iff a stored signature ``a`` matches the current signature ``b`` on every
    non-hparam field (exact) and every ZoMBI-Hop hyperparameter (float-tolerant)."""
    if not isinstance(a, dict):
        return False
    for k in b:
        if k == "hp":
            continue
        if a.get(k) != b.get(k):
            return False
    return _same_hparams(a.get("hp", {}) or {}, b.get("hp", {}))


def _write_run_config(sweep_dir: Path, dim: int, base_hp: Dict[str, Any],
                      perturbed_hp: Dict[str, Any]) -> None:
    """run_config.json at the sweep root so coverage_plot._find_config (which walks
    up to the grandparent of a rep dir) can classify the landscape.

    Also records a per-group signature (``group_signatures``) so a later fresh sweep
    can copy this sweep's plain (no-LLM) groups verbatim when their settings match
    (see ``_reuse_group``)."""
    (sweep_dir / "run_config.json").write_text(json.dumps({
        "dataset": LANDSCAPE, "dim": dim, "maximize": True,
        "landscape": "synthetic", "hparam_names": R.HPARAM_NAMES,
        "ensemble_seed": ENSEMBLE_SEED, "trial_json": str(_resolve(TRIAL_JSON)),
        "group_signatures": {
            "baseline": _group_signature(dim, base_hp),
            "perturbed": _group_signature(dim, perturbed_hp),
        },
    }, indent=2))


def _reuse_group(sweep_dir: Path, dim: int, hp: Dict[str, Any], prefix: str,
                 group: str) -> int:
    """Copy ``group``'s reps from the newest prior sweep(s) whose stored signature
    for that group matches ``hp``, so a fresh (non-``--resume``) sweep doesn't
    recompute a plain group it has already produced. Only ``baseline`` and
    ``perturbed`` are reusable (the LLM group is stochastic and never reused); a
    ``perturbed`` group only matches when PERTURB is identical.

    Reps are matched individually by index (== seed) and only missing ones are
    filled, so this composes with ``--resume`` and with priors that had a different
    ``N_REPEATS``. Copies the whole rep dir (metrics.json + all artifacts). Returns
    the number of reps copied."""
    root = _resolve(RESULTS_ROOT)
    if not root.is_dir():
        return 0
    sig = _group_signature(dim, hp)
    dest_gdir = sweep_dir / group
    copied = 0
    priors = sorted((d for d in root.glob(f"{prefix}_*") if d.is_dir()), reverse=True)
    for pdir in priors:
        if pdir.resolve() == sweep_dir.resolve():
            continue
        cfg_path = pdir / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            pcfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        stored = (pcfg.get("group_signatures") or {}).get(group)
        if not _group_sig_match(stored, sig):
            continue
        src_gdir = pdir / group
        if not src_gdir.is_dir():
            continue
        for r in range(N_REPEATS):
            dest_rep = dest_gdir / f"rep{r}"
            if (dest_rep / "metrics.json").exists():
                continue  # already present (resume, or an earlier prior)
            src_rep = src_gdir / f"rep{r}"
            if not (src_rep / "metrics.json").exists():
                continue
            dest_gdir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_rep, dest_rep, dirs_exist_ok=True)
            copied += 1
            print(f"      [reuse] {group}/rep{r} ← {pdir.name} (no recompute)")
        if all((dest_gdir / f"rep{r}" / "metrics.json").exists()
               for r in range(N_REPEATS)):
            break  # group fully populated; no need to scan older sweeps
    if copied:
        print(f"[catastrophic] reused {copied} {group} rep(s) from prior sweep(s) "
              f"with matching settings")
    return copied


def _agg(samples: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    vals = np.array([float(s.get(key, np.nan)) for s in samples], float)
    fin = vals[np.isfinite(vals)]
    return {"mean": float(fin.mean()) if fin.size else float("nan"),
            "std": float(fin.std(ddof=1)) if fin.size > 1 else 0.0,
            "n": int(fin.size)}


def _find_latest_sweep_dir(prefix: str) -> Optional[Path]:
    """Newest existing ``<prefix>_<ts>`` dir under RESULTS_ROOT (timestamps sort
    lexically), or None if there is none to resume."""
    root = _resolve(RESULTS_ROOT)
    if not root.is_dir():
        return None
    cands = sorted(d for d in root.glob(f"{prefix}_*") if d.is_dir())
    return cands[-1] if cands else None


def _setup_sweep(prefix: str, resume: bool = False) -> dict:
    """Shared setup for main() and the smoke run: validate config, create the
    sweep dir, write run_config.json, build the fixed landscape, and draw the
    landscape preview. Returns everything the trial loop needs.

    ``resume=True`` reuses the newest existing ``<prefix>_<ts>`` dir instead of
    minting a fresh one (falling back to a new dir if none exists), so the rep
    loop can skip already-completed reps."""
    trial_json = _resolve(TRIAL_JSON)
    dim, base_hp = load_trial(trial_json)
    bad_keys = [k for k in PERTURB if k not in R.HPARAM_NAMES]
    if bad_keys:
        raise SystemExit(f"PERTURB has invalid hyperparameter names: {bad_keys}")
    perturbed_hp = dict(base_hp)
    perturbed_hp.update(PERTURB)
    comp_cols = composition_column_names(dim)

    print(f"[catastrophic] trial_json={trial_json}")
    print(f"[catastrophic] landscape={LANDSCAPE}")
    print(f"[catastrophic] dim={dim}  budget={MAX_ITERS}  repeats={N_REPEATS}  "
          f"inject_every={INJECT_INTERVAL}")
    print(f"[catastrophic] PERTURB={PERTURB}")

    # Override run_mobo's output-noise fraction for the whole sweep. make_sim_obj +
    # _gen_init_data read this module global at call time, so setting it here makes
    # every group (perturbed/baseline/llm) see the amplified objective noise.
    if OUTPUT_NOISE_FRAC is not None:
        print(f"[catastrophic] output noise: run_mobo.OUTPUT_NOISE_FRAC "
              f"{R.OUTPUT_NOISE_FRAC} → {OUTPUT_NOISE_FRAC}")
        R.OUTPUT_NOISE_FRAC = OUTPUT_NOISE_FRAC

    sweep_dir = _find_latest_sweep_dir(prefix) if resume else None
    if sweep_dir is not None:
        print(f"[catastrophic] --resume: continuing {sweep_dir} "
              f"(reps with metrics.json are skipped)")
    else:
        if resume:
            print(f"[catastrophic] --resume: no existing {prefix}_* dir found; "
                  f"starting a fresh sweep")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = _resolve(RESULTS_ROOT) / f"{prefix}_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(sweep_dir, dim, base_hp, perturbed_hp)

    ens, true_optima = build_landscape(dim)
    # First artifact: the fixed landscape, for eyeballing the seed.
    plot_ensemble_landscape(sweep_dir / "ensemble_landscape.png", ens, true_optima, dim)
    return dict(sweep_dir=sweep_dir, dim=dim, base_hp=base_hp,
                perturbed_hp=perturbed_hp, comp_cols=comp_cols, ens=ens,
                true_optima=true_optima)


def smoke_llm() -> None:
    """Shake-out run: ONE real ``llm_every_{INJECT_INTERVAL}`` rep (real
    hyperparameters, real LLM calls) on the fixed landscape, so the
    checkpoint/resume + injection path is exercised end-to-end before committing
    to the full 3-group sweep. Writes to its own ``smoke_llm_<ts>`` results dir."""
    s = _setup_sweep("smoke_llm")
    group = f"llm_every_{INJECT_INTERVAL}"
    rep_dir = s["sweep_dir"] / group / "rep0"
    seed = 1000
    print(f"\n=== SMOKE {group} / rep0 (seed={seed}) ===")
    m = run_llm_trial(s["ens"], s["true_optima"], s["perturbed_hp"], s["dim"],
                      seed, rep_dir, s["comp_cols"])
    (rep_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"\n[smoke] done → {rep_dir}")
    print(f"[smoke] injections={m['n_injections']} changes={m['n_changes']} "
          f"best_obj={m['best_objective']:.4f} needles={m['n_needles']} "
          f"dist={m['dist_to_true_optima']:.4f}")


def main(no_llm: bool = False, resume: bool = False) -> None:
    s = _setup_sweep("sweep_catastrophic", resume=resume)
    sweep_dir, dim = s["sweep_dir"], s["dim"]
    base_hp, perturbed_hp = s["base_hp"], s["perturbed_hp"]
    comp_cols, ens, true_optima = s["comp_cols"], s["ens"], s["true_optima"]

    # Copy matching plain groups from an earlier sweep (no --resume needed): a plain
    # group is blind to the LLM, so identical landscape + hyperparameters + budget ⇒
    # identical trajectories, safe to reuse verbatim. baseline reuses on base_hp;
    # perturbed reuses only when PERTURB (hence perturbed_hp) is identical too.
    _reuse_group(sweep_dir, dim, base_hp, "sweep_catastrophic", "baseline")
    _reuse_group(sweep_dir, dim, perturbed_hp, "sweep_catastrophic", "perturbed")

    groups: List[Tuple[str, str]] = [
        ("perturbed", "plain"),
        ("baseline", "plain"),
        (f"llm_every_{INJECT_INTERVAL}", "llm"),
    ]
    if no_llm:
        # --no-llm: only the baseline and perturbed-baseline groups (no LLM calls).
        groups = [g for g in groups if g[1] != "llm"]
        print("[catastrophic] --no-llm: running baseline + perturbed only "
              "(LLM group skipped)")
    summary: Dict[str, Any] = {}
    for group, kind in groups:
        gdir = sweep_dir / group
        gdir.mkdir(parents=True, exist_ok=True)
        samples: List[Dict[str, Any]] = []
        for r in range(N_REPEATS):
            seed = 1000 + r  # common random numbers across groups
            rep_dir = gdir / f"rep{r}"
            metrics_path = rep_dir / "metrics.json"
            # Skip any rep that already has metrics.json — from --resume OR from a
            # baseline copied out of a prior matching sweep by _reuse_baseline.
            if metrics_path.exists():
                try:
                    m = json.loads(metrics_path.read_text())
                    samples.append(m)
                    print(f"\n=== {group} / rep{r} (seed={seed}) — SKIP (already "
                          f"done: best_obj={m.get('best_objective', float('nan')):.4f}) ===")
                    continue
                except Exception as e:
                    print(f"\n=== {group} / rep{r}: existing metrics.json unreadable "
                          f"({e}); re-running ===")
            print(f"\n=== {group} / rep{r} (seed={seed}) ===")
            try:
                if kind == "plain":
                    hp = base_hp if group == "baseline" else perturbed_hp
                    m = run_plain_trial(ens, true_optima, hp, dim, seed, rep_dir, group)
                else:
                    m = run_llm_trial(ens, true_optima, perturbed_hp, dim, seed,
                                      rep_dir, comp_cols)
                (rep_dir / "metrics.json").write_text(json.dumps(m, indent=2))
                samples.append(m)
                print(f"    best_obj={m['best_objective']:.4f}  "
                      f"needles={m['n_needles']}  dist={m['dist_to_true_optima']:.4f}")
            except Exception as e:
                import traceback
                print(f"    [rep] FAILED: {e}")
                traceback.print_exc()
        summary[group] = {
            "n_repeats": len(samples),
            "best_objective": _agg(samples, "best_objective"),
            "n_needles": _agg(samples, "n_needles"),
            "dist_to_true_optima": _agg(samples, "dist_to_true_optima"),
            "dup_fraction": _agg(samples, "dup_fraction"),
            "mean_pairwise_needle_dist": _agg(samples, "mean_pairwise_needle_dist"),
        }

    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[catastrophic] done → {sweep_dir}")
    for group, _ in groups:
        s = summary.get(group, {})
        bo = s.get("best_objective", {})
        print(f"  {group:16s}  best_obj={bo.get('mean', float('nan')):.4f} "
              f"± {bo.get('std', 0.0):.4f}  "
              f"needles={s.get('n_needles', {}).get('mean', float('nan')):.1f}  "
              f"dist={s.get('dist_to_true_optima', {}).get('mean', float('nan')):.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smoke-llm" in args:
        smoke_llm()
    else:
        main(no_llm="--no-llm" in args, resume="--resume" in args)
