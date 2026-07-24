#!/usr/bin/env python3
"""
optimize/compare_arb_tuned.py
=============================
Head-to-head comparison of an ARBITRARY hyperparameter set against a MOBO-TUNED
one, using a PAIRED-SEED design.

Landscape (``--dataset``, default ``newRF``):
  • ``newRF`` — a FIXED Random-Forest surrogate trained on live measurements from a
    results DB/CSV (same source as optimize/evaluate.py's newRF and
    visualization/plot_run.py). Built ONCE and shared by every seed, so each seed
    differs only in its random initial design. This script never needs newRF's
    interactively-picked reference optima (it scores the raw objective + a
    grid-based random baseline, not needle-match metrics), so it builds the
    surrogate headlessly — no GUI picker, safe on HPC. Always 3-simplex (3D).
  • ``<run dir>`` (e.g. ``runs/run_9dfe``) — a FIXED Random-Forest surrogate trained
    on the measured (composition, objective) pairs logged in that run's
    ``composition_log.jsonl``. The simplex dimension is read from the run's
    ``config.json`` (``d``), so this is how 4-simplex (4D) campaigns are compared.
    Like ``newRF`` it is built ONCE and shared by every seed.
  • ``ensemble`` — the re-randomized ``ensemble`` objective, a DIFFERENT landscape
    per seed (drawn from ``--ensemble-seed`` + the seed index).

At dim 3 the per-run landscape/needle artifacts are static ternary (3-simplex)
PNGs; at dim 4 they become a single interactive, rotatable 3D 4-simplex
(tetrahedron) point-cloud HTML — the objective cloud with the run's pared points
and discovered needles overlaid — exactly like ``run_mobo.py``'s 4D view.

For each of ``--n-seeds`` seeds we run ZoMBI-Hop twice on that seed's landscape —
once with the arbitrary hparams, once with the tuned hparams — from the SAME
random-number seed and the SAME initial design, so the only thing that differs
within a seed pair is the hyperparameters. Each run is capped at ``--iterations``
ZoMBI iterations (one iteration == one LineBO main-line pick == one
objective-wrapper call, matching run_mobo's iteration count).

Defaults
--------
  arbitrary : optimize/hparams/3d_llm_chosen.json
  tuned     : optimize/runs/mobo_ensemble_4d_job17147232/trial_10/trial.json
ZoMBI-Hop hyperparameters are dimension-independent, so the same arbitrary/tuned
sets apply on a 3-simplex or 4-simplex landscape. Both are overridable with
``--arb-hparams`` / ``--tuned-hparams`` (any file
``evaluate.load_hparams_from_json`` accepts: a flat hparam dict or a trial.json
with an ``"hparams"`` key).

Outputs (under ``--out-dir``, default optimize/runs/compare_arb_tuned_<stamp>/)
------------------------------------------------------------------------------
  per_seed/seed_<i>_convergence.png   one plot PER seed overlaying the arbitrary
                                      and tuned running-best curves (running best
                                      only — individual samples are NOT drawn),
                                      plus a uniform-random baseline (mean + 5–95%
                                      band) on that seed's landscape, à la
                                      llm/plot_vs_random_baseline.py.
  summary_convergence.png             mean ± 95% CI running-best across seeds for
                                      each hparam set, plus the random baseline —
                                      normalised to each landscape's optimum so
                                      the seeds are commensurable.
  slopegraph_final_best.png           slopegraph: one line per seed connecting its
                                      final best (arbitrary → tuned), exploiting
                                      the paired-seed structure.
  per_seed/seed_<i>_<arb|tuned>/      per-run artifacts for EVERY run (both hparam
                                      sets on every seed), à la run_mobo's per-trial
                                      folder and llm/sweep_catastrophic.py:
                                        needles.csv            declared needles (run_mobo schema)
                                        coverage.png           ground-truth landscape + all
                                                               sampled points + found needles
                                        needles_on_landscape.png  final needles as red stars on
                                                               the ternary Objective background
                                        points.csv             every sampled point (coverage input)
                                      At dim 3 these are static ternary PNGs; at
                                      dim 4 they collapse to one interactive
                                      landscape_point_cloud.html (rotatable 4-simplex).
                                      Coverage / landscape artifacts are 3D/4D-only
                                      (skipped at dim > 4); needles.csv is always written.
  compare_summary.json                every number behind the plots.

Convergence x-axis is the number of objective evaluations (sample index), the
standard best-so-far-vs-budget axis and the one llm/plot_vs_random_baseline.py
uses; the random baseline draws the same number of uniform samples. The summary
and slopegraph normalise the objective per seed to ``(y - grid_min) /
(grid_max - grid_min)`` (1.0 == the landscape's global optimum) because each seed
is a different landscape with a different objective scale; the per-seed plots
show the raw objective.

Usage
-----
  conda activate zombi-hop            # (or `uv run`)
  python optimize/compare_arb_tuned.py                          # newRF, default DB
  python optimize/compare_arb_tuned.py --db 2nd_real_run.db --iterations 40 --n-seeds 5
  python optimize/compare_arb_tuned.py \
      --arb-hparams optimize/hparams/3d_llm_chosen.json \
      --tuned-hparams optimize/runs/mobo_ensemble_3d_job17560178/trial_13/trial.json \
      --db data/campaign1a.csv --out-dir optimize/runs/my_compare
  python optimize/compare_arb_tuned.py --dataset ensemble --dim 3 --ensemble-seed 0
  # 4-simplex RF surrogate built from a 4D campaign run:
  python optimize/compare_arb_tuned.py --dataset runs/run_9dfe --n-seeds 5 --iterations 135

On HPC use optimize/scripts/compare_arb_tuned.sbatch (allocates 6h + a GPU).
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# optimize/ on path so `run_mobo` / `evaluate` import; run_mobo puts the repo root
# on sys.path (for synthetic_data.*) as an import side effect.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_mobo as rm  # noqa: E402
import evaluate as ev  # noqa: E402
from synthetic_data.ensemble import random_ensemble_config  # noqa: E402

# Repo-relative defaults (resolved against the repo root, i.e. optimize/..).
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_ARB_HPARAMS = "optimize/hparams/3d_llm_chosen.json"
DEFAULT_TUNED_HPARAMS = "optimize/runs/mobo_ensemble_4d_job17147232/trial_10/trial.json"

# Independent uniform-random searches averaged for a baseline curve (as in
# llm/plot_vs_random_baseline.py).
N_RANDOM_SEARCHES = 500

ARB_COLOR = "darkorange"
TUNED_COLOR = "steelblue"
RANDOM_COLOR = "slategray"


class _MaxItersReached(Exception):
    """Internal signal: the iteration cap was hit — stop this ZoMBI run."""


# ─── Hyperparameter loading ─────────────────────────────────────────────────────

def _resolve(path: str) -> str:
    """Resolve a possibly repo-relative path to an absolute one."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(_REPO_ROOT, path))


def load_one_hparam_set(path: str, label: str) -> dict:
    """Load a single hparam dict (reusing evaluate's validation)."""
    abspath = _resolve(path)
    hp = ev.load_hparams_from_json(abspath)[0]  # {0: hp} -> hp; exits on bad/missing keys
    print(f"  [{label}] {len(hp)} hyperparameters from {abspath}")
    return hp


# ─── Landscapes ─────────────────────────────────────────────────────────────────

def build_newrf_ds(*, db: str, db_value: str, db_minimize: bool,
                   time_limit_min: float, optima_json: str | None,
                   runs_path: str | None) -> dict:
    """Build the FIXED newRF landscape headlessly (no interactive optima picker).

    Mirrors the RF-surrogate half of ``evaluate.resolve_dataset('newRF', …)`` — a
    RandomForestRegressor trained on the DB/CSV compositions vs objective, with a
    dense ternary grid for the random baseline — but skips the ExtremaPicker: this
    script scores the raw objective and a grid-based random baseline, so it never
    needs ``true_optima``. Any cached/provided optima are loaded only to honour a
    saved ``maximize`` flag; the surrogate itself is identical across every seed.
    """
    from sklearn.ensemble import RandomForestRegressor

    db_resolved, X, Y, value_col = ev._newrf_load_db(db or ev.DEFAULT_DB, db_value)
    rf = RandomForestRegressor(n_estimators=rm.RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X, Y)
    rf_fn = lambda x, _rf=rf: float(_rf.predict(x.reshape(1, -1))[0])
    grid_pts = rm.ternary_grid(rm.TERNARY_GRID_N)
    grid_vals = rf.predict(grid_pts)

    maximize = not db_minimize
    true_optima: list = []
    saved = ev._newrf_optima_candidate(optima_json, runs_path)
    if saved is not None:
        true_optima, saved_max = ev._load_newrf_optima(saved)
        if saved_max is not None:
            maximize = saved_max
        print(f"  [newRF] loaded {len(true_optima)} reference optima from {saved} "
              f"({'maximize' if maximize else 'minimize'})")
    print(f"  [newRF] RF surrogate from {db_resolved} "
          f"({'maximize' if maximize else 'minimize'}, value='{value_col}', "
          f"{X.shape[0]} measured points) — fixed across all seeds")

    spec = rm.LandscapeSpec(
        landscape="rf", dim=3, maximize=maximize, true_optima=true_optima,
        fn_callable=rf_fn, grid_pts=grid_pts, grid_vals=grid_vals,
        csv_path=db_resolved, objective_column=value_col,
        composition_columns=list(ev.DB_COMP_COLS),
        time_limit_hours=time_limit_min / 60.0,
    )
    return ev._landscape_to_ds(spec, "newRF")


def _load_run_measurements(run_dir: str) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Read every measured ``(composition, objective)`` pair from a run directory.

    A ZoMBI-Hop run logs each LineBO batch to ``composition_log.jsonl``: one JSON
    object per call, each holding a list of ``rails`` (``main`` / ``cache``) with
    parallel ``measured`` (``d``-vector compositions) and ``y`` (objective) lists.
    We pool them across every call and rail into ``(X (N,d), Y (N,))`` — the
    training set for the fixed RF surrogate. The simplex dimension is read from
    the run's ``config.json`` (``d``). Returns ``(X, Y, dim, config)``.
    """
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        sys.exit(f"--dataset {run_dir}: no config.json (not a ZoMBI-Hop run dir).")
    with open(cfg_path) as f:
        cfg = json.load(f)
    dim = int(cfg.get("d") or cfg.get("dim") or 0)
    log_path = os.path.join(run_dir, "composition_log.jsonl")
    if not os.path.isfile(log_path):
        sys.exit(f"--dataset {run_dir}: no composition_log.jsonl to train a surrogate on.")

    X: list = []
    Y: list = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rail in rec.get("rails", []):
                measured = rail.get("measured") or []
                ys = rail.get("y") or []
                for pt, yv in zip(measured, ys):
                    if pt is None or yv is None:
                        continue
                    X.append([float(c) for c in pt])
                    Y.append(float(yv))
    if not X:
        sys.exit(f"--dataset {run_dir}: composition_log.jsonl held no measured points.")
    Xarr = np.asarray(X, dtype=float)
    Yarr = np.asarray(Y, dtype=float)
    if dim <= 0:
        dim = Xarr.shape[1]
    if Xarr.shape[1] != dim:
        sys.exit(f"--dataset {run_dir}: config d={dim} but measured points are "
                 f"{Xarr.shape[1]}-dimensional.")
    return Xarr, Yarr, dim, cfg


def _fit_gp_surrogate(X: np.ndarray, Y: np.ndarray, length_scale: float):
    """Fit a fixed-length-scale GP interpolant on ``(X, Y)`` over the simplex.

    Mirrors ``visualization/plot_run.py:fit_gp_background``: a Matern(nu=2.5)
    kernel with a FIXED ``length_scale`` (so it directly controls smoothness —
    smaller = more local/wiggly) times a ConstantKernel, plus a WhiteKernel noise
    term. ``Y`` is standardised for numerical stability; the returned predictors
    map back to the original objective scale. Works in any simplex dimension.

    Returns ``(predict_batch, predict_one)`` where ``predict_batch(Xg (M,d)) ->
    (M,)`` and ``predict_one(x (d,)) -> float``.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    y_mean = float(Y.mean())
    y_std = float(Y.std()) or 1.0
    y = (Y - y_mean) / y_std

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=False, n_restarts_optimizer=2, random_state=42
    )
    gp.fit(X, y)

    def predict_batch(Xg, _gp=gp, _m=y_mean, _s=y_std):
        return _gp.predict(np.asarray(Xg, dtype=float)) * _s + _m

    def predict_one(x, _gp=gp, _m=y_mean, _s=y_std):
        return float(_gp.predict(np.asarray(x, dtype=float).reshape(1, -1))[0] * _s + _m)

    return predict_batch, predict_one


def build_run_ds(*, run_dir: str, db_minimize: bool, time_limit_min: float,
                 model: str = "rf", gp_length_scale: float = 0.3) -> dict:
    """Build a FIXED surrogate landscape from a ZoMBI-Hop run's measured data.

    The ``d``-simplex analogue of ``build_newrf_ds``: instead of a results DB it
    trains the surrogate on the ``(composition, objective)`` pairs logged in
    ``run_dir/composition_log.jsonl`` (dimension read from ``config.json``).
    ``model`` selects the surrogate: ``"rf"`` (a RandomForest, the default) or
    ``"gp"`` (a fixed-length-scale Matern GP interpolant, à la
    ``visualization/plot_run.py`` — ``gp_length_scale`` sets its smoothness). The
    dense random-baseline grid is a 3-simplex ternary lattice at dim 3 and a
    4-simplex lattice at dim 4 (``synthetic_data.plot_ackley.build_simplex_lattice``);
    at dim > 4 there is no grid, so the random baseline falls back to each curve's
    own span. No reference optima are picked (this script never needs them).
    """
    run_abspath = os.path.abspath(run_dir)
    X, Y, dim, _cfg = _load_run_measurements(run_abspath)
    # Simplex-normalise each composition (measured rows may drift off the simplex).
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)

    if model == "gp":
        predict_batch, fn = _fit_gp_surrogate(X, Y, gp_length_scale)
        landscape_label = "gp"
        model_desc = f"GP interpolant (Matern nu=2.5, length_scale={gp_length_scale})"
    else:
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=rm.RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
        rf.fit(X, Y)
        fn = lambda x, _rf=rf: float(_rf.predict(x.reshape(1, -1))[0])
        predict_batch = lambda Xg, _rf=rf: _rf.predict(np.asarray(Xg, dtype=float))
        landscape_label = "rf"
        model_desc = "RF surrogate"

    grid_pts = grid_vals = None
    if dim == 3:
        grid_pts = rm.ternary_grid(rm.TERNARY_GRID_N)
        grid_vals = predict_batch(grid_pts)
    elif dim == 4:
        import synthetic_data.plot_ackley as pc4
        grid_pts = pc4.build_simplex_lattice(pc4.GRID_N)
        grid_vals = predict_batch(grid_pts)

    maximize = not db_minimize
    print(f"  [run] {model_desc} from {run_abspath} "
          f"({'maximize' if maximize else 'minimize'}, {X.shape[0]} measured points, "
          f"dim={dim}) — fixed across all seeds")

    spec = rm.LandscapeSpec(
        landscape=landscape_label, dim=dim, maximize=maximize, true_optima=[],
        fn_callable=fn, grid_pts=grid_pts, grid_vals=grid_vals,
        csv_path=run_abspath, objective_column="y",
        composition_columns=[f"x{i + 1}" for i in range(dim)],
        time_limit_hours=time_limit_min / 60.0,
    )
    return ev._landscape_to_ds(spec, "run")


# ─── One capped ZoMBI run ───────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_capped(hparams: dict, ds: dict, *, rng_seed: int, max_iters: int,
               time_limit_min: float, artifact_dir: str | None = None,
               label: str = "", seed_label: str = "") -> dict:
    """Run ZoMBI-Hop on ``ds`` with ``hparams``, stopping after ``max_iters``
    iterations (or the objective's own convergence / the safety time limit,
    whichever comes first).

    ``rng_seed`` seeds Python/NumPy/Torch so the initial design (and the noise
    stream up to the first divergence) is identical to any other run given the
    same seed — the paired-seed guarantee. Returns the full objective trajectory
    (in maximise orientation, higher == better) and its running best.

    When ``artifact_dir`` is given, the per-run artifacts (needles.csv, coverage
    plot, needles-on-landscape plot, points.csv) are written there — the same set
    run_mobo / sweep_catastrophic emit per trial.
    """
    _seed_everything(rng_seed)
    dim = ds["dim"]
    fn = ds["fn"]
    maximize = ds["maximize"]

    plot_state: dict = {"line_0": None, "line_1": None}
    call_counter = [0]

    sim_obj = rm.make_sim_obj(fn, rm.DEVICE, rm.DTYPE, maximize=maximize)
    inner = rm.make_linebo_wrapper(sim_obj, dim, rm.NUM_LINES, rm.DEVICE, rm.DTYPE, plot_state)

    def obj_wrapper(x_tell, bounds, acq_fn):
        # Stop BEFORE evaluating the (max_iters+1)-th line, so exactly max_iters
        # iterations' worth of samples land in the data handler.
        if call_counter[0] >= max_iters:
            raise _MaxItersReached()
        x_req, x_act, y = inner(x_tell, bounds, acq_fn)
        call_counter[0] += 1
        return x_req, x_act, y

    try:
        X_a, X_e, Y = ev.gen_init_data(fn, maximize, dim)
    except RuntimeError as exc:
        print(f"      [run] init failed: {exc}")
        return {"Y": np.empty(0), "running_best": np.empty(0), "n_iters": 0, "n_samples": 0}

    hp = dict(hparams)
    if dim > 3 and (hp.get("top_m_points") is None or hp.get("top_m_points", 0) < dim + 1):
        hp["top_m_points"] = max(dim + 1, 4)
    # Peel any ZOMBI_FIXED overrides out of the hparam dict (none for the default
    # sets, but keep evaluate.py's contract so custom JSONs behave identically).
    zombi_fixed = dict(rm.ZOMBI_FIXED)
    for k in ev._ZOMBI_OVERRIDE_KEYS:
        if k in hp:
            zombi_fixed[k] = hp.pop(k)

    optimizer = rm.ZoMBIHop(
        objective=obj_wrapper,
        X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
        **zombi_fixed, **hp,
        device=str(rm.DEVICE), dtype=rm.DTYPE,
        run_uuid=None, checkpoint_dir=None,
    )
    dh = optimizer.data_handler

    # Track per-point (activation, zoom, zoom-size) snapshots exactly as run_mobo
    # does, so write_points_csv can tag each sampled point (coverage plot input).
    snap_records: list = []
    orig_snap = dh.take_snapshot

    def snap_wrap(*a, **k):
        orig_snap(*a, **k)
        if dh.X_all_actual is not None:
            czb = dh.current_zoom_bounds if dh.current_zoom_bounds is not None else dh.bounds
            zoom_size = rm.zoom_size_fraction(czb) if czb is not None else 1.0
            snap_records.append((dh.X_all_actual.shape[0], dh.current_activation,
                                 dh.current_zoom, zoom_size))

    dh.take_snapshot = snap_wrap

    try:
        optimizer.run(max_activations=float("inf"), time_limit_hours=time_limit_min / 60.0)
    except _MaxItersReached:
        pass
    except Exception as exc:  # keep going: partial trajectory is still usable
        print(f"      [run] ZoMBI crashed after {call_counter[0]} iters: {exc}")

    # dh.Y_all is in maximise orientation already (sim_obj negates only when
    # minimising; the ensemble objective maximises), so running best is a cummax.
    Y_all = dh.Y_all.detach().cpu().numpy().ravel() if dh.Y_all is not None else np.empty(0)
    accum = np.maximum.accumulate if maximize else np.minimum.accumulate
    running_best = accum(Y_all) if Y_all.size else Y_all
    print(f"      [run]  iters={call_counter[0]}  samples={Y_all.size}  "
          f"final_best={running_best[-1]:.4f}" if Y_all.size else
          f"      [run]  iters={call_counter[0]}  samples=0")

    if artifact_dir is not None:
        write_run_artifacts(artifact_dir, dh=dh, ds=ds, snap_records=snap_records,
                            label=label, seed_label=seed_label)

    del optimizer, dh, sim_obj, inner
    rm._reclaim_memory()
    return {"Y": Y_all, "running_best": running_best,
            "n_iters": call_counter[0], "n_samples": int(Y_all.size)}


# ─── Random baseline ────────────────────────────────────────────────────────────

def random_running_best(grid_vals: np.ndarray, n: int, *, maximize: bool,
                        rng: np.random.Generator, n_searches: int) -> np.ndarray:
    """(n_searches, n) running-best curves from uniform draws over ``grid_vals``.

    Mirrors llm/plot_vs_random_baseline.py: each search draws ``n`` points
    uniformly at random from the dense ground-truth grid and tracks its running
    best — the honest "what would random search with the same budget get?" line.
    """
    draws = grid_vals[rng.integers(0, grid_vals.size, size=(n_searches, n))]
    accum = np.maximum.accumulate if maximize else np.minimum.accumulate
    return accum(draws, axis=1)


# ─── Per-run artifacts (needles.csv, coverage plot, needles-on-landscape) ────────

_SQRT3_2 = math.sqrt(3) / 2
_TERNARY_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")


def _comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """Barycentric compositions → ternary cartesian (matches coverage_plot)."""
    p = np.atleast_2d(np.asarray(comp, dtype=float))
    s = p.sum(axis=1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def plot_needles_on_landscape(out_png: str, *, needles: np.ndarray,
                              grid_pts: np.ndarray, grid_vals: np.ndarray,
                              true_optima, label: str, seed_label: str) -> None:
    """3D only: the landscape Objective as a filled ternary contour with the run's
    final declared needles overlaid as red stars (true optima as hollow white
    rings). Mirrors llm/sweep_catastrophic.plot_needles_on_ensemble, but reads the
    background straight off the dataset's ground-truth grid."""
    xy = _comp_to_xy(grid_pts)
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3_2]])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    tcf = ax.tricontourf(xy[:, 0], xy[:, 1], np.asarray(grid_vals, float),
                         levels=24, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.82, label="Objective")
    tri = np.vstack([verts, verts[:1]])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1.4, zorder=3)
    offs = [(-0.06, -0.05), (1.06, -0.05), (0.5, _SQRT3_2 + 0.05)]
    for lab, (ox, oy) in zip(_TERNARY_LABELS, offs):
        ax.text(ox, oy, lab, ha="center", va="center", fontsize=10, fontweight="bold")
    if true_optima is not None and len(true_optima):
        tc = _comp_to_xy(np.asarray(true_optima, dtype=float))
        ax.scatter(tc[:, 0], tc[:, 1], marker="o", s=90, facecolors="none",
                   edgecolors="white", linewidths=1.4, zorder=4, label="true optima")
    needles = (np.atleast_2d(np.asarray(needles, dtype=float))
               if needles is not None and len(needles) else np.empty((0, 3)))
    if needles.shape[0]:
        nc = _comp_to_xy(needles)
        ax.scatter(nc[:, 0], nc[:, 1], marker="*", s=260, c="red",
                   edgecolors="white", linewidths=1.0, zorder=5,
                   label=f"final needles (n={needles.shape[0]})")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Final needles — {seed_label} ({label})", fontsize=11)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_needles_on_landscape_4d(out_html: str, *, needles: np.ndarray,
                                 grid_pts: np.ndarray, grid_vals: np.ndarray,
                                 dh, label: str, seed_label: str) -> None:
    """4D only: an interactive, rotatable 3D 4-simplex (tetrahedron) point cloud of
    the RF-surrogate Objective, with this run's pared points and final declared
    needles overlaid. Mirrors ``run_mobo._render_4d_point_cloud`` but reads the
    background straight off the dataset's ground-truth grid (grid_pts/grid_vals)
    instead of an Ackley callable. Needs plotly (imported lazily)."""
    import plotly.graph_objects as go
    import synthetic_data.plot_ackley as pc4

    comp = np.asarray(grid_pts, dtype=float)
    obj = np.asarray(grid_vals, dtype=float).ravel()
    xyz = pc4.to_3d(comp)
    obj_min, obj_max = float(obj.min()), float(obj.max())

    hover = [f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
             for (a, b, c, d), v in zip(comp, obj)]
    cloud = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        name="objective (surrogate)", text=hover, hoverinfo="text",
        marker=dict(color=obj, colorscale="Viridis", cmin=obj_min, cmax=obj_max,
                    size=pc4.MARKER_SIZE, opacity=pc4.MARKER_OPACITY,
                    showscale=True, colorbar=dict(title="Objective")),
    )
    fig = go.Figure(data=[cloud, pc4.tetra_edges_trace(), pc4.vertex_labels_trace()])
    fig.update_layout(
        title=f"Final state — {seed_label} ({label}) on the 4-simplex surrogate",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data"),
        legend=dict(x=0.0, y=1.0), width=pc4.FIG_W, height=pc4.FIG_H,
    )

    # Overlay the final data-handler state (pared points coloured by objective,
    # sized by recency) plus the discovered needles as red markers.
    pared_X = pared_Y = recency = None
    if dh.X_pared is not None and dh.X_pared.shape[0] > 0:
        pared_X = dh.X_pared.detach().cpu().numpy()
        pared_Y = dh.Y_pared.detach().cpu().numpy().ravel()
        recency = np.arange(pared_X.shape[0], dtype=float)

    needles = (np.atleast_2d(np.asarray(needles, dtype=float))
               if needles is not None and len(needles) else None)
    pc4.add_simplex_overlays(
        fig, obj_cmin=obj_min, obj_cmax=obj_max,
        pared_points=pared_X, pared_values=pared_Y, recency=recency,
        needles=needles,
    )
    fig.write_html(out_html, include_plotlyjs="cdn", auto_open=False)


def write_run_artifacts(artifact_dir: str, *, dh, ds: dict,
                        snap_records: list, label: str, seed_label: str) -> None:
    """Write the per-run artifacts run_mobo / sweep_catastrophic produce per trial:
    needles.csv (always); at dim 3 a static ternary coverage plot + needles-on-
    landscape PNG + the points.csv/ground-truth grid ``coverage_plot`` consumes;
    at dim 4 a single interactive 4-simplex point-cloud HTML (objective cloud +
    pared points + needles). Best-effort: any failure prints and is swallowed so
    it never aborts the comparison."""
    os.makedirs(artifact_dir, exist_ok=True)
    dim = ds["dim"]

    # needles.csv — identical schema to run_mobo's per-trial needles.csv.
    try:
        rm.write_needles_csv(os.path.join(artifact_dir, "needles.csv"), dh, dim=dim)
    except Exception as exc:
        print(f"      [artifacts] needles.csv failed: {exc}")

    if dim not in (3, 4):
        return  # coverage / landscape views are 3D (ternary) or 4D (point cloud) only
    grid_pts = ds.get("grid_pts")
    grid_vals = ds.get("grid_vals")
    true_optima = ds.get("true_optima") or []
    if grid_pts is None or grid_vals is None:
        return

    needle_t = dh.get_all_needle_locations()
    needles = (needle_t.detach().cpu().numpy()
               if needle_t is not None and needle_t.numel() > 0 else np.empty((0, dim)))

    if dim == 4:
        # Interactive 4-simplex point cloud (the 3D-ternary analogue for 4D).
        try:
            rm.write_points_csv(os.path.join(artifact_dir, "points.csv"), dh, snap_records, dim=dim)
        except Exception as exc:
            print(f"      [artifacts] points.csv failed: {exc}")
        try:
            plot_needles_on_landscape_4d(
                os.path.join(artifact_dir, "landscape_point_cloud.html"),
                needles=needles, grid_pts=grid_pts, grid_vals=grid_vals,
                dh=dh, label=label, seed_label=seed_label)
        except Exception as exc:
            print(f"      [artifacts] 4D point cloud failed: {exc}")
        return

    # Needles as red stars on the ternary Objective background.
    try:
        plot_needles_on_landscape(
            os.path.join(artifact_dir, "needles_on_landscape.png"),
            needles=needles, grid_pts=grid_pts, grid_vals=grid_vals,
            true_optima=true_optima, label=label, seed_label=seed_label)
    except Exception as exc:
        print(f"      [artifacts] needles-on-landscape plot failed: {exc}")

    # Coverage plot (run_mobo style): points.csv + a minimal config + this run's
    # ground-truth grid so coverage_plot draws THIS landscape's background/optima.
    try:
        rm.write_points_csv(os.path.join(artifact_dir, "points.csv"), dh, snap_records, dim=dim)
        np.savez(
            os.path.join(artifact_dir, "coverage_ground_truth.npz"),
            grid_pts=np.asarray(grid_pts, dtype=float),
            grid_vals=np.asarray(grid_vals, dtype=float),
            true_optima=(np.asarray(true_optima, dtype=float)
                         if len(true_optima) else np.empty((0, dim))),
        )
        with open(os.path.join(artifact_dir, "run_config.json"), "w") as f:
            json.dump({"dim": dim, "maximize": bool(ds["maximize"]), "dataset": "newRF"}, f)
        import coverage_plot
        coverage_plot.save_coverage_image(artifact_dir)
    except Exception as exc:
        print(f"      [artifacts] coverage plot failed: {exc}")


# ─── Plots ──────────────────────────────────────────────────────────────────────

def plot_per_seed(out_png: str, *, seed_label: str, arb_rb: np.ndarray,
                  tuned_rb: np.ndarray, grid_vals: np.ndarray | None,
                  maximize: bool, rng: np.random.Generator, n_random: int,
                  n_optima: int | None) -> None:
    """Overlay the arbitrary and tuned running-best curves for ONE seed (running
    best only, no scatter) plus a uniform-random baseline on this landscape."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(np.arange(arb_rb.size), arb_rb, color=ARB_COLOR, lw=1.9,
            label="arbitrary (running best)", zorder=4)
    ax.plot(np.arange(tuned_rb.size), tuned_rb, color=TUNED_COLOR, lw=1.9,
            label="tuned (running best)", zorder=4)

    if grid_vals is not None and grid_vals.size:
        n = int(max(arb_rb.size, tuned_rb.size))
        curves = random_running_best(grid_vals, n, maximize=maximize, rng=rng,
                                     n_searches=n_random)
        idx = np.arange(n)
        lo = np.percentile(curves, 5, axis=0)
        hi = np.percentile(curves, 95, axis=0)
        ax.fill_between(idx, lo, hi, color=RANDOM_COLOR, alpha=0.18, lw=0,
                        label="uniform random (5–95%)", zorder=1)
        ax.plot(idx, curves.mean(axis=0), color=RANDOM_COLOR, lw=1.5, ls="--",
                label=f"uniform random (mean of {n_random})", zorder=2)

    ttl = f"Convergence — {seed_label}"
    if n_optima is not None:
        ttl += f"  ({n_optima} true optima)"
    ax.set_xlabel("Objective evaluations (sample index)")
    ax.set_ylabel("Best objective Y found")
    ax.set_title(ttl, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ci95(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% CI (mean ± 1.96·SEM) down axis 0."""
    mean = stack.mean(axis=0)
    n = stack.shape[0]
    sem = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    half = 1.96 * sem
    return mean, mean - half, mean + half


def plot_summary(out_png: str, *, arb_norm: np.ndarray, tuned_norm: np.ndarray,
                 rand_norm: np.ndarray | None, n_seeds: int) -> None:
    """Mean ± 95% CI running-best across seeds for each hparam set (normalised to
    each landscape's optimum), plus the random baseline (5–95% spread)."""
    L = arb_norm.shape[1]
    idx = np.arange(L)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    for stack, color, label in ((arb_norm, ARB_COLOR, "arbitrary"),
                                (tuned_norm, TUNED_COLOR, "tuned")):
        mean, lo, hi = _ci95(stack)
        ax.fill_between(idx, lo, hi, color=color, alpha=0.2, lw=0, zorder=2)
        ax.plot(idx, mean, color=color, lw=2.0,
                label=f"{label} (mean ± 95% CI, n={n_seeds})", zorder=4)

    if rand_norm is not None and rand_norm.size:
        lo = np.percentile(rand_norm, 5, axis=0)
        hi = np.percentile(rand_norm, 95, axis=0)
        ax.fill_between(idx, lo, hi, color=RANDOM_COLOR, alpha=0.18, lw=0,
                        label="uniform random (5–95%)", zorder=1)
        ax.plot(idx, rand_norm.mean(axis=0), color=RANDOM_COLOR, lw=1.6, ls="--",
                label="uniform random (mean)", zorder=3)

    ax.set_xlabel("Objective evaluations (sample index)")
    ax.set_ylabel("Best found (fraction of landscape optimum)")
    ax.set_title(f"Convergence across {n_seeds} paired seeds  "
                 f"(normalised per landscape)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slopegraph(out_png: str, *, arb_finals: list[float], tuned_finals: list[float],
                    n_seeds: int) -> None:
    """Slopegraph: one line per seed connecting its final best (arbitrary→tuned).

    Uses the paired-seed structure directly — each seed is a matched pair, so the
    slope shows whether tuning helped or hurt on THAT landscape. Values are
    normalised to each landscape's optimum (1.0 == global optimum)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    x0, x1 = 0.0, 1.0
    cmap = plt.get_cmap("tab10")
    for i, (a, t) in enumerate(zip(arb_finals, tuned_finals)):
        c = cmap(i % 10)
        ax.plot([x0, x1], [a, t], color=c, lw=1.6, marker="o", ms=6,
                label=f"seed {i}", zorder=3)

    # Group means as a heavy black reference line.
    ma, mt = float(np.mean(arb_finals)), float(np.mean(tuned_finals))
    ax.plot([x0, x1], [ma, mt], color="black", lw=2.6, marker="s", ms=8,
            label="mean", zorder=4)
    ax.annotate(f"{ma:.3f}", (x0, ma), textcoords="offset points", xytext=(-8, 0),
                ha="right", va="center", fontsize=9, fontweight="bold")
    ax.annotate(f"{mt:.3f}", (x1, mt), textcoords="offset points", xytext=(8, 0),
                ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Arbitrary", "Tuned"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Final best (fraction of landscape optimum)")
    ax.set_title(f"Final best per seed  (n={n_seeds}, paired)", fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Driver ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare arbitrary vs tuned ZoMBI-Hop hyperparameters (paired seeds).")
    parser.add_argument("--arb-hparams", default=DEFAULT_ARB_HPARAMS,
                        help=f"arbitrary hparam JSON (default: {DEFAULT_ARB_HPARAMS})")
    parser.add_argument("--tuned-hparams", default=DEFAULT_TUNED_HPARAMS,
                        help=f"tuned hparam JSON / trial.json (default: {DEFAULT_TUNED_HPARAMS})")
    parser.add_argument("--dataset", default="newRF",
                        help="landscape: 'newRF' (fixed RF surrogate from a DB, default), "
                             "'ensemble' (re-randomized ensemble objective), or a path to "
                             "a ZoMBI-Hop run dir (e.g. runs/run_9dfe) to build a fixed RF "
                             "surrogate from its logged measurements — the dim (3 or 4) is "
                             "read from the run's config.json")
    parser.add_argument("--dim", type=int, default=3,
                        help="simplex dimension for --dataset ensemble (newRF is always 3D; "
                             "a run-dir dataset reads its own dim from config.json)")
    parser.add_argument("--landscape-model", choices=("rf", "gp"), default="rf",
                        help="surrogate used for a run-dir --dataset landscape: 'rf' "
                             "(RandomForest, default) or 'gp' (fixed-length-scale Matern "
                             "GP interpolant, à la visualization/plot_run.py)")
    parser.add_argument("--gp-length-scale", type=float, default=0.3,
                        help="fixed Matern length scale for --landscape-model gp "
                             "(smaller = more local/wiggly; default: 0.3)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="number of paired seeds (default: 5)")
    parser.add_argument("--iterations", type=int, default=40,
                        help="ZoMBI iterations per run (default: 40)")
    # newRF (fixed RF surrogate) source
    parser.add_argument("--db", default=ev.DEFAULT_DB, metavar="DB",
                        help=f"newRF results DB or campaign CSV (default: {ev.DEFAULT_DB}; "
                             f"resolved against data/ if not a full path)")
    parser.add_argument("--db-value", default=ev.DEFAULT_DB_VALUE, metavar="COL",
                        help=f"newRF objective column (default: {ev.DEFAULT_DB_VALUE})")
    parser.add_argument("--db-minimize", action="store_true",
                        help="newRF: minimise the objective (default: maximise)")
    parser.add_argument("--optima-json", default=None,
                        help="newRF: optional saved-optima JSON (only used for its "
                             "maximize flag; optima themselves are unused here)")
    parser.add_argument("--runs-path", default=None,
                        help="newRF: dir holding a cached newRF_optima.json (see --optima-json)")
    # ensemble source
    parser.add_argument("--ensemble-seed", type=int, default=0,
                        help="base seed for the ensemble landscape sequence (default: 0)")
    parser.add_argument("--ensemble-margin", type=float, default=0.2,
                        help="ensemble optima/background gap (default: 0.2)")
    parser.add_argument("--seed-base", type=int, default=12345,
                        help="base RNG seed; seed i uses seed-base+i for BOTH hparam "
                             "sets (paired) (default: 12345)")
    parser.add_argument("--time-limit-min", type=float, default=30.0,
                        help="per-run safety wall-clock cap in minutes; the iteration "
                             "cap is the primary stop (default: 30)")
    parser.add_argument("--n-random-searches", type=int, default=N_RANDOM_SEARCHES,
                        help=f"uniform searches averaged per baseline (default: {N_RANDOM_SEARCHES})")
    parser.add_argument("--out-dir", default=None,
                        help="output dir (default: optimize/runs/compare_arb_tuned_<stamp>)")
    parser.add_argument("--device", default=None, help="override torch device (cpu/cuda)")
    args = parser.parse_args()

    if args.device:
        rm.DEVICE = torch.device(args.device)
        ev.rm.DEVICE = rm.DEVICE

    out_dir = args.out_dir or os.path.join(
        _REPO_ROOT, "optimize", "runs",
        f"compare_arb_tuned_{datetime.datetime.now():%d_%m_%H_%M_%S}_{os.getpid()}")
    per_seed_dir = os.path.join(out_dir, "per_seed")
    os.makedirs(per_seed_dir, exist_ok=True)
    print(f"[compare] output -> {out_dir}  device={rm.DEVICE}")

    arb_hp = load_one_hparam_set(args.arb_hparams, "arbitrary")
    tuned_hp = load_one_hparam_set(args.tuned_hparams, "tuned")

    # newRF / a run-dir surrogate are a single fixed landscape shared by every seed;
    # ensemble re-randomizes the landscape per seed. ``landscape_for_seed`` hides
    # that difference. A run-dir dataset is any --dataset that is not newRF/ensemble.
    is_run_dataset = args.dataset not in ("newRF", "ensemble")
    newrf_db_resolved = None
    dataset_dim = 3 if args.dataset == "newRF" else args.dim
    if args.dataset == "newRF":
        newrf_ds = build_newrf_ds(
            db=args.db, db_value=args.db_value, db_minimize=args.db_minimize,
            time_limit_min=args.time_limit_min, optima_json=args.optima_json,
            runs_path=args.runs_path)
        newrf_db_resolved = newrf_ds.get("csv_path")

        def landscape_for_seed(i: int) -> dict:
            return newrf_ds
    elif is_run_dataset:
        run_dir = _resolve(args.dataset)
        if not os.path.isdir(run_dir):
            sys.exit(f"--dataset {args.dataset!r} is neither 'newRF'/'ensemble' nor a "
                     f"run directory (looked for {run_dir}).")
        run_ds = build_run_ds(run_dir=run_dir, db_minimize=args.db_minimize,
                              time_limit_min=args.time_limit_min,
                              model=args.landscape_model,
                              gp_length_scale=args.gp_length_scale)
        newrf_db_resolved = run_ds.get("csv_path")
        dataset_dim = run_ds["dim"]

        def landscape_for_seed(i: int) -> dict:
            return run_ds
    else:
        def landscape_for_seed(i: int) -> dict:
            cfg = random_ensemble_config(args.dim, i, seed=int(args.ensemble_seed),
                                         optima_margin=args.ensemble_margin)
            return ev.build_ensemble_ds(cfg, "ensemble",
                                        time_limit_hours=args.time_limit_min / 60.0)

    seeds_data: list[dict] = []  # per-seed collected trajectories + landscape scale
    for i in range(args.n_seeds):
        rng_seed = args.seed_base + i
        ds = landscape_for_seed(i)
        grid_vals = ds.get("grid_vals")
        gv = np.asarray(grid_vals).ravel() if grid_vals is not None else None
        n_optima = len(ds["true_optima"]) or None
        if args.dataset == "newRF":
            print(f"\n[compare] seed {i}: newRF fixed landscape, rng_seed={rng_seed}")
        elif is_run_dataset:
            print(f"\n[compare] seed {i}: run-dir fixed landscape "
                  f"(dim={dataset_dim}), rng_seed={rng_seed}")
        else:
            print(f"\n[compare] seed {i}: ensemble index={i} "
                  f"ensemble_seed={args.ensemble_seed} rng_seed={rng_seed}  "
                  f"({n_optima} true optima)")

        print(f"  [seed {i}] arbitrary hparams …")
        arb = run_capped(arb_hp, ds, rng_seed=rng_seed, max_iters=args.iterations,
                         time_limit_min=args.time_limit_min,
                         artifact_dir=os.path.join(per_seed_dir, f"seed_{i}_arb"),
                         label="arbitrary", seed_label=f"seed {i}")
        print(f"  [seed {i}] tuned hparams …")
        tuned = run_capped(tuned_hp, ds, rng_seed=rng_seed, max_iters=args.iterations,
                           time_limit_min=args.time_limit_min,
                           artifact_dir=os.path.join(per_seed_dir, f"seed_{i}_tuned"),
                           label="tuned", seed_label=f"seed {i}")

        # Per-seed overlay (raw objective) with the landscape's random baseline.
        plot_per_seed(
            os.path.join(per_seed_dir, f"seed_{i}_convergence.png"),
            seed_label=f"seed {i}", arb_rb=arb["running_best"],
            tuned_rb=tuned["running_best"], grid_vals=gv, maximize=ds["maximize"],
            rng=np.random.default_rng(rng_seed), n_random=args.n_random_searches,
            n_optima=n_optima,
        )
        seeds_data.append({
            "seed_index": i, "rng_seed": rng_seed,
            "n_optima": n_optima,
            "grid_min": float(gv.min()) if gv is not None else None,
            "grid_max": float(gv.max()) if gv is not None else None,
            "grid_vals": gv,
            "maximize": ds["maximize"],
            "arb": arb, "tuned": tuned,
        })

    # ── Cross-seed summary (normalised, truncated to the shortest run) ───────────
    lengths = [d[h]["running_best"].size for d in seeds_data for h in ("arb", "tuned")]
    lengths = [n for n in lengths if n > 0]
    if not lengths:
        print("[compare] no usable runs — aborting summary.")
        return
    L = int(min(lengths))
    print(f"\n[compare] summary aligned to the shortest run: L={L} evaluations "
          f"(run lengths {sorted(set(lengths))})")

    def _norm(rb: np.ndarray, d: dict) -> np.ndarray:
        """Normalise a running-best curve to [0,1] by its landscape's grid range;
        fall back to the curve's own span when no ground-truth grid (dim > 3)."""
        if d["grid_min"] is not None and d["grid_max"] > d["grid_min"]:
            lo, hi = d["grid_min"], d["grid_max"]
        else:
            lo, hi = float(rb.min()), float(rb.max())
        span = (hi - lo) or 1.0
        return (rb[:L] - lo) / span

    arb_norm = np.vstack([_norm(d["arb"]["running_best"], d) for d in seeds_data])
    tuned_norm = np.vstack([_norm(d["tuned"]["running_best"], d) for d in seeds_data])

    rand_norm = None
    have_grids = all(d["grid_vals"] is not None for d in seeds_data)
    if have_grids:
        rand_rows = []
        for d in seeds_data:
            rng = np.random.default_rng(d["rng_seed"])
            curves = random_running_best(d["grid_vals"], L, maximize=d["maximize"],
                                         rng=rng, n_searches=args.n_random_searches)
            lo, hi = d["grid_min"], d["grid_max"]
            span = (hi - lo) or 1.0
            rand_rows.append((curves - lo) / span)  # (n_searches, L), normalised
        rand_norm = np.vstack(rand_rows)  # (n_seeds*n_searches, L)

    plot_summary(os.path.join(out_dir, "summary_convergence.png"),
                 arb_norm=arb_norm, tuned_norm=tuned_norm, rand_norm=rand_norm,
                 n_seeds=args.n_seeds)

    # ── Slopegraph of final best (normalised, paired) ───────────────────────────
    arb_finals = [float(_norm(d["arb"]["running_best"], d)[-1]) for d in seeds_data]
    tuned_finals = [float(_norm(d["tuned"]["running_best"], d)[-1]) for d in seeds_data]
    plot_slopegraph(os.path.join(out_dir, "slopegraph_final_best.png"),
                    arb_finals=arb_finals, tuned_finals=tuned_finals,
                    n_seeds=args.n_seeds)

    # ── Machine-readable summary ────────────────────────────────────────────────
    summary = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "arb_hparams_path": _resolve(args.arb_hparams),
        "tuned_hparams_path": _resolve(args.tuned_hparams),
        "arb_hparams": arb_hp,
        "tuned_hparams": tuned_hp,
        "dataset": args.dataset,
        "dim": dataset_dim,
        "n_seeds": args.n_seeds,
        "iterations": args.iterations,
        "newrf_db": newrf_db_resolved,
        "newrf_db_value": (args.db_value if args.dataset == "newRF" else None),
        "ensemble_seed": args.ensemble_seed,
        "ensemble_margin": args.ensemble_margin,
        "seed_base": args.seed_base,
        "aligned_length": L,
        "device": str(rm.DEVICE),
        "seeds": [
            {
                "seed_index": d["seed_index"], "rng_seed": d["rng_seed"],
                "n_optima": d["n_optima"],
                "grid_min": d["grid_min"], "grid_max": d["grid_max"],
                "arb_n_iters": d["arb"]["n_iters"], "arb_n_samples": d["arb"]["n_samples"],
                "tuned_n_iters": d["tuned"]["n_iters"], "tuned_n_samples": d["tuned"]["n_samples"],
                "arb_final_best_raw": float(d["arb"]["running_best"][-1]) if d["arb"]["n_samples"] else None,
                "tuned_final_best_raw": float(d["tuned"]["running_best"][-1]) if d["tuned"]["n_samples"] else None,
                "arb_final_best_norm": af, "tuned_final_best_norm": tf,
            }
            for d, af, tf in zip(seeds_data, arb_finals, tuned_finals)
        ],
        "arb_final_best_norm_mean": float(np.mean(arb_finals)),
        "tuned_final_best_norm_mean": float(np.mean(tuned_finals)),
    }
    with open(os.path.join(out_dir, "compare_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[compare] done.")
    print(f"  arbitrary final best (norm, mean over seeds): {np.mean(arb_finals):.4f}")
    print(f"  tuned     final best (norm, mean over seeds): {np.mean(tuned_finals):.4f}")
    print(f"  per-seed plots : {per_seed_dir}")
    print(f"  per-run artifacts (needles.csv / coverage.png / "
          f"needles_on_landscape.png): {per_seed_dir}/seed_<i>_<arb|tuned>/")
    print(f"  summary plot   : {os.path.join(out_dir, 'summary_convergence.png')}")
    print(f"  slopegraph     : {os.path.join(out_dir, 'slopegraph_final_best.png')}")
    print(f"  summary json   : {os.path.join(out_dir, 'compare_summary.json')}")


if __name__ == "__main__":
    main()
