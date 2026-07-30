"""
optimize/evaluate.py
====================
Re-evaluate ZoMBI-Hop hyperparameter configurations on one or more objectives,
writing the same per-run artifact set as ``run_mobo.py`` (CSVs, static plots,
per-iteration landscape frames / video where the dimension allows).

Where ``run_mobo.py`` *searches* for good hyperparameters, this script takes
configurations a MOBO run already found and RUNS them ``--num-runs`` times each
(default 1) so you can quantify run-to-run variance or transfer a configuration
onto a different benchmark.

Datasets
--------
``--dataset`` selects the objective.  Pass one name or several comma-separated
(``--dataset RF,gaussian,ackley4d``); with multiple datasets each gets its own
sub-folder under the output directory.

Built-in names:

  RF           3-simplex Random-Forest surrogate from the source run's
               ``run_config.json`` (requires ``--runs-path``).
  newRF        3-simplex Random-Forest surrogate trained on live measurements
               pulled straight from a ``results`` SQLite DB (same source as
               ``visualization/plot_run.py``; ``--db``/``--db-value``).  The first
               time, reference optima are picked interactively (click them on the
               RF map, like ``interactive_test_zombi.py``) and cached to
               ``newRF_optima.json`` in the output dir; later runs can pass
               ``--runs-path THAT_DIR`` (or ``--optima-json``) to skip the picker.
  ackley3d     negated analytic Ackley on the 3-simplex (``--ackley-variant``).
  ackley4d     negated analytic Ackley on the 4-simplex.
  ackley10d    negated analytic Ackley on the 10-simplex.
  gaussian3d   realistic Gaussian mixture on the 3-simplex (random peaks + noise).
  gaussian4d   realistic Gaussian mixture on the 4-simplex.
  gaussian10d  realistic Gaussian mixture on the 10-simplex.

Synthetic oracle names (direct analytic landscapes from ``synthetic_data/oracles.py``):

  messy, gaussian, ackley, planted_bumps, rastrigin_ilr

  ensemble     layered ``synthetic_data/ensemble.py`` objective, RE-RANDOMIZED PER
               RUN like the "Randomize" button in ``plot_ensemble.py`` (random
               feature counts/widths/amplitudes + on/off toggles + a fresh seed).
               Each run writes ``ensemble_config.json`` (full Ensemble kwargs) so
               the exact landscape can be recreated.  ``--ensemble-seed`` makes the
               per-run sequence reproducible; ``--dim`` sets the simplex dimension;
               ``--ensemble-margin`` sets the optima/background gap.

  For ``gaussian``, use ``"variant": "realistic"`` in a batch JSON (or ``--oracle-variant
  realistic``) for random Dirichlet peaks like ``gaussian3d``; default ``layout`` uses
  fixed symmetric peak geometry (3/5/7 peaks via ``--layout``).

For synthetic oracles, dimension / layout / seed come from ``--dim``, ``--layout``,
``--seed``, or a batch JSON via ``--config`` (e.g.
``optimize/mobo_batch_configs/synthetic_3d_gaussian.json``).

Hyperparameter sources (pick one)
---------------------------------
  --trials 12,34       look up trials in ``--runs-path mobo_*``
  --hparams PATH       trial_* dir, trial.json, or mobo_* dir (latest trial)
  --hparams-json PATH  flat hparam dict or trial.json-style file (trial 0)

Output
------
Default parent: ``optimize/runs/rerun_DD_MM_HH_MM/`` (override with ``--out`` or
``--out-dir`` for a fixed directory).

  rerun_DD_MM_HH_MM/
    rerun_config.json          static config (dataset, dim, optima, time limit …)
    rerun_summary.json         per-trial mean/std of the three objectives + every
                               individual run's metrics
    trial_<n>/                 one per selected source trial
        hparams.json           the hyperparameters being re-evaluated
        run_<k>/               one per --num-runs repeat
            metrics.json
            points.csv         (sample_idx, <coords>, Y, penalized, activation, zoom)
            needles.csv        (needle_idx, <coords>, value, median_value,
                                zoom, iteration, reason, dist_to_centre)
            metrics_over_time.csv
            convergence.png
            dist_from_centre.png
            line_length_hist.png
            hparam_edge_proximity.png
            plots/iter_*.png    (dim == 3 only)  +  zombihop_timelapse.mp4
            point_cloud.html    (dim == 4 only — final-state interactive cloud)
            (dim >= 5: metrics + non-landscape plots only, no landscape view)

``<coords>`` are ``FA/MA/Br`` for the 3-simplex RF (drop-in compatible with the
mobo trial CSVs) and ``x1..xd`` otherwise.

Usage
-----

    COMPOSITIONAL TRIAL 112 HPARAMS
    python optimize/evaluate.py \
        --hparams-json optimize/hparams/trial_112_composition.json \
        --dataset RF \
        --num-runs 1 \
        --time-limit-min 3 \
        --device cpu \
        --no-video \
        --out-dir optimize/runs/rerun_trial112_comp \
        --runs-path optimize/runs/mobo_05_06_15_32


  python optimize/evaluate.py --runs-path optimize/runs/mobo_05_06_15_32 \
      --trials 112 --dataset gaussian --num-runs 3 --time-limit-min 3

  python optimize/evaluate.py \
      --hparams optimize/runs/mobo_05_06_15_32/trial_112 \
      --config optimize/mobo_batch_configs/synthetic_3d_gaussian.json

  python optimize/evaluate.py --runs-path optimize/runs/mobo_05_06_15_32 \
      --trials 112 --dataset ackley4d --num-runs 5 --time-limit-min 10

  python optimize/evaluate.py --runs-path optimize/runs/mobo_05_06_15_32 \
      --trials 112 --dataset RF,gaussian --num-runs 1

  python optimize/evaluate.py --runs-path optimize/runs/mobo_05_06_15_32 \
      --trials 12 --dataset ackley10d --num-runs 3

  # newRF: pull data from the DB, pick optima interactively (cached for reuse):
  python optimize/evaluate.py \
      --hparams-json optimize/runs/mobo_.../trial_1/trial.json \
      --dataset newRF --num-runs 3 --time-limit-min 5
  # later, reuse the cached optima (no clicking) by pointing --runs-path at the
  # previous output dir (which holds newRF_optima.json):
  python optimize/evaluate.py \
      --hparams-json optimize/runs/mobo_.../trial_1/trial.json \
      --dataset newRF --num-runs 3 --time-limit-min 5 \
      --runs-path optimize/runs/rerun_DD_MM_HH_MM

Instead of ``--trials``, you can pass ``--hparams`` (trial dir / trial.json /
mobo_* dir) or ``--hparams-json`` with a path to a JSON file containing the
hyperparameters directly (either a trial.json-style dict with an ``"hparams"``
key, or a flat dict of hyperparameter values).  ``--hparams-json`` runs are
labelled ``trial_0``.  ``--runs-path`` is still required for the RF dataset
config::

  python optimize/evaluate.py --runs-path optimize/runs/mobo_05_06_15_32 \
      --hparams-json optimize/hparams.json --dataset RF --num-runs 3

New flags for ensemble data: 
    --ensemble-seed (default 0; makes the per-run landscape sequence reproducible)
    --ensemble-margin (optima/background gap, default 0.2)
    --dim sets the simplex dimension

    python optimize/evaluate.py \
        --hparams optimize/runs/mobo_05_06_15_32/trial_1/trial.json \
        --dataset ensemble --dim 3 --num-runs 5 --ensemble-seed 0
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
import traceback

import numpy as np
import torch

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_mobo as rm
from mobo_landscapes import build_synthetic_landscape, parse_synthetic_batch_fields, resolve_surrogate_csv_path

# Analytic benchmark objectives (negated Ackley on the d-simplex).
from synthetic_data.ackley import Ackley
from synthetic_data.ensemble import Ensemble, random_ensemble_config
from synthetic_data.gaussian_landscape import RealisticGaussian
from synthetic_data.landscape_config_log import (
    build_landscape_config_log,
    format_landscape_config_summary,
)
from synthetic_data.oracles import ORACLE_CHOICES

try:
    import make_videos as mv
except Exception:  # pragma: no cover
    mv = None

ACKLEY_BENCHMARKS = {"ackley3d": 3, "ackley4d": 4, "ackley10d": 10}
GAUSSIAN_BENCHMARKS = {"gaussian3d": 3, "gaussian4d": 4, "gaussian10d": 10}
# Layered ``Ensemble`` objective: re-randomized per run (see random_ensemble_config).
ENSEMBLE_DATASETS = {"ensemble"}
# Fixed warm-start GP landscape (warm_start/warm_gp_landscape.py via
# rm.build_warmgp_landscape): the same surface every run, seeded by --warmgp-seed.
WARMGP_DATASETS = {"warmgp"}
# Full-run GP landscape (rm.build_fullgp_landscape): the same fixed GP fit to the
# ENTIRE real campaign (every scored point), not just the warm-start lines. This is
# the honest evaluation ground truth — the tuner sees the warm-start GP, deployed
# hparams are scored here. Deterministic (no seed dependence).
FULLGP_DATASETS = {"fullgp"}
# ``newRF``: RF surrogate trained on live measurements pulled straight from a
# ``results`` SQLite DB (same source as visualization/plot_run.py).  Its reference
# optima are picked interactively (interactive_test_zombi.ExtremaPicker) the first
# time and cached to ``newRF_optima.json`` so later runs can supply ``--runs-path``.
NEWRF_DATASETS = {"newRF"}
BUILTIN_DATASETS = {"RF", *ACKLEY_BENCHMARKS.keys(), *GAUSSIAN_BENCHMARKS.keys(),
                    *ORACLE_CHOICES, *ENSEMBLE_DATASETS, *WARMGP_DATASETS,
                    *FULLGP_DATASETS,
                    *NEWRF_DATASETS}

# ─── newRF database source (mirrors visualization/plot_run.py) ──────────────────
DEFAULT_DB = "2nd_real_run.db"
DB_COMP_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
DEFAULT_DB_VALUE = "Objective"
# Cached picker selections, written into the rerun output dir; ``--runs-path`` can
# point back at that dir (or any dir / JSON holding a ``true_optima`` list) to skip
# the interactive picker on later runs.
NEWRF_OPTIMA_FILENAME = "newRF_optima.json"

# Canonical campaign1a RF reference optima from mobo_05_06_15_32 (interactive picker).
TRIAL112_REFERENCE_OPTIMA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "reference_optima",
    "mobo_05_06_15_32_campaign1a.json",
)


class _TopKReached(Exception):
    """Internal signal: ``--top-k`` needles found, stop this run early."""


class _LineBudgetReached(Exception):
    """Internal signal: ``--max-lines`` measured lines reached, stop this run.

    Each ZoMBI iteration = one ``obj_wrapper`` call = one measured LineBO line, so
    capping the objective-call count fixes the number of lines every run samples
    (init lines are a fixed deterministic preamble on top). The time limit still
    applies as a secondary safety cap.
    """


# ─── Reference optima ───────────────────────────────────────────────────────────

def load_true_optima_json(path: str) -> list[np.ndarray]:
    """Load ``true_optima`` from a run_config-style JSON sidecar."""
    cfg_path = os.path.abspath(path)
    if not os.path.isfile(cfg_path):
        sys.exit(f"reference optima file not found: {cfg_path}")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as exc:
        sys.exit(f"reference optima file {cfg_path} unreadable ({exc}).")
    raw = cfg.get("true_optima")
    if not raw:
        sys.exit(f"reference optima file {cfg_path} has no 'true_optima' list.")
    return [np.asarray(t, dtype=float) for t in raw]


# ─── newRF: database source + interactive / cached optima ───────────────────────

def _newrf_load_db(db_arg: str, value_col: str):
    """Read ``(X (N,3), Y (N,))`` from a results DB, reusing plot_run's loader.

    Returns ``(db_abspath, X, Y, value_col)``.
    """
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from visualization.plot_run import _resolve_db_path, load_db_dataset
    except Exception as exc:  # pragma: no cover
        sys.exit(f"--dataset newRF: could not import visualization.plot_run ({exc}).")
    try:
        db_path = _resolve_db_path(db_arg)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    X, Y, _labels, title = load_db_dataset(db_path, value_col)
    print(f"  [newRF] {title}")
    return str(db_path), X, Y, value_col


def _newrf_optima_candidate(optima_json: str | None, runs_path: str | None) -> str | None:
    """Resolve a saved-optima file from ``--optima-json`` / ``--runs-path``, or None.

    ``--optima-json`` wins; otherwise ``--runs-path`` is searched for (in order) a
    cached ``newRF_optima.json``, a prior ``rerun_config.json``, or a source
    ``run_config.json`` — any JSON exposing a ``true_optima`` list works.
    """
    if optima_json:
        p = os.path.abspath(optima_json)
        if not os.path.isfile(p):
            sys.exit(f"--optima-json not found: {p}")
        return p
    if runs_path:
        for name in (NEWRF_OPTIMA_FILENAME, "rerun_config.json", "run_config.json"):
            cand = os.path.join(runs_path, name)
            if os.path.isfile(cand):
                return cand
    return None


def _load_newrf_optima(path: str) -> tuple[list[np.ndarray], bool | None]:
    """Load ``true_optima`` (and an optional ``maximize`` flag) from ``path``."""
    optima = load_true_optima_json(path)
    maximize = None
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data.get("maximize"), bool):
            maximize = data["maximize"]
    except Exception:
        pass
    return optima, maximize


def _save_newrf_optima(path: str, optima: list[np.ndarray], *, maximize: bool,
                       db_path: str, value_col: str) -> None:
    payload = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": "newRF",
        "db": db_path,
        "value_col": value_col,
        "maximize": maximize,
        "true_optima": [list(map(float, np.asarray(o, dtype=float).ravel())) for o in optima],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _newrf_pick_optima(rf, grid_pts: np.ndarray, grid_vals: np.ndarray,
                       *, maximize: bool) -> list[np.ndarray]:
    """Open the ExtremaPicker so the user can click reference optima on the RF map.

    Switches matplotlib to an interactive backend for the click session, then
    restores the headless backend so per-iteration landscape frames still render.
    Returns the picked simplex compositions (possibly empty if nothing was clicked).
    """
    import matplotlib
    it_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interactive_testing")
    if it_dir not in sys.path:
        sys.path.insert(0, it_dir)
    try:
        matplotlib.use("TkAgg", force=True)
        import interactive_test_zombi as it  # module sets TkAgg + imports the picker
    except Exception as exc:
        sys.exit(
            f"--dataset newRF needs a GUI (TkAgg) backend to pick optima but it "
            f"failed to start ({exc}). Re-run with --optima-json PATH or "
            f"--runs-path DIR pointing at a saved newRF_optima.json instead."
        )
    import matplotlib.pyplot as plt
    plt.ion()
    try:
        picker = it.ExtremaPicker(rf, grid_pts, grid_vals, maximize=maximize)
        extrema = picker.run()
    finally:
        # Restore the headless backend for the subsequent landscape rendering.
        try:
            rm._configure_mpl_backend(headless=True)
        except Exception:
            matplotlib.use("Agg", force=True)
    return [np.asarray(c, dtype=float) for c, _v in extrema]


# ─── Dataset resolution ─────────────────────────────────────────────────────────

def _load_synthetic_batch_config(config_path: str) -> dict:
    cfg_path = os.path.abspath(config_path)
    if not os.path.isfile(cfg_path):
        sys.exit(f"--config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("landscape") != "synthetic":
        sys.exit(f"--config must have landscape=synthetic (got {cfg.get('landscape')!r}).")
    syn = parse_synthetic_batch_fields(cfg)
    out = {
        "name": cfg.get("name") or os.path.splitext(os.path.basename(cfg_path))[0],
        "oracle": syn["oracle"],
        "dim": syn["dim"],
        "layout": syn["layout"],
        "seed": syn["seed"],
        "variant": syn.get("variant", "layout"),
        "time_limit_hours": cfg.get("time_limit_hours"),
    }
    for key in ("n_peaks", "sigma", "sigma_var", "noise_freq", "noise_amp"):
        if key in syn:
            out[key] = syn[key]
    return out


def _landscape_to_ds(spec, dataset: str, *, ackley_fn=None, variant: str | None = None) -> dict:
    ds = {
        "dim": spec.dim,
        "maximize": spec.maximize,
        "fn": spec.fn_callable,
        "true_optima": spec.true_optima,
        "grid_pts": spec.grid_pts,
        "grid_vals": spec.grid_vals,
        "ackley_fn": ackley_fn,
        "label": dataset,
        "csv_path": spec.csv_path,
        "landscape": spec.landscape,
        "oracle": spec.oracle,
        "layout": spec.ackley_layout,
        "seed": spec.synthetic_seed,
        "variant": variant,
        "time_limit_hours": spec.time_limit_hours,
        "metadata_path": spec.metadata_path,
    }
    log = build_landscape_config_log(dataset=dataset, ds=ds)
    ds["landscape_config"] = log
    print(format_landscape_config_summary(log))
    return ds


def _gaussian_kwargs(config: dict | None) -> dict:
    cfg = config or {}
    out = {}
    for key in ("n_peaks", "sigma", "sigma_var", "noise_freq", "noise_amp"):
        if key in cfg and cfg[key] is not None:
            out[key] = cfg[key]
    return out


def resolve_dataset(
    dataset: str,
    runs_path: str | None,
    ackley_variant: str,
    *,
    dim: int = 3,
    layout: str = "2",
    seed: int = 42,
    oracle_variant: str = "layout",
    config: dict | None = None,
    time_limit_hours: float | None = None,
    db_path: str | None = None,
    db_value: str = DEFAULT_DB_VALUE,
    db_minimize: bool = False,
    optima_json: str | None = None,
    out_dir: str | None = None,
    warmgp_seed: int = 0,
) -> dict:
    """Build the objective + reference optima for ``dataset``."""
    if dataset == "warmgp":
        # Fixed warm-start GP landscape; reference optima are its auto-detected
        # GP peaks (rm.build_warmgp_landscape / warm_start.warm_gp_landscape).
        spec = rm.build_warmgp_landscape(
            dim, seed=warmgp_seed, time_limit_hours=time_limit_hours)
        print(f"  [dataset] warmgp: warm-start GP landscape dim={dim} "
              f"seed={warmgp_seed} — maximize, {len(spec.true_optima)} GP peak(s)")
        return _landscape_to_ds(spec, "warmgp")

    if dataset == "fullgp":
        # Full-run GP landscape (GP fit to the ENTIRE real campaign); reference
        # optima are its auto-detected peaks (rm.build_fullgp_landscape). This is
        # the evaluation ground truth — deterministic, so warmgp_seed is unused.
        spec = rm.build_fullgp_landscape(
            dim, seed=warmgp_seed, time_limit_hours=time_limit_hours)
        print(f"  [dataset] fullgp: full-run GP landscape dim={dim} "
              f"— maximize, {len(spec.true_optima)} GP peak(s)")
        return _landscape_to_ds(spec, "fullgp")

    if dataset == "newRF":
        from sklearn.ensemble import RandomForestRegressor

        db_resolved, X, Y, value_col = _newrf_load_db(db_path or DEFAULT_DB, db_value)
        rf = RandomForestRegressor(n_estimators=rm.RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
        rf.fit(X, Y)
        rf_fn = lambda x, _rf=rf: float(_rf.predict(x.reshape(1, -1))[0])
        grid_pts = rm.ternary_grid(rm.TERNARY_GRID_N)
        grid_vals = rf.predict(grid_pts)

        # Reference optima: load a cached/provided set, else pick interactively and
        # cache the result so a later run can just pass --runs-path.
        maximize = not db_minimize
        saved = _newrf_optima_candidate(optima_json, runs_path)
        if saved is not None:
            true_optima, saved_max = _load_newrf_optima(saved)
            if saved_max is not None:
                maximize = saved_max
            print(f"  [newRF] loaded {len(true_optima)} reference optima from {saved} "
                  f"({'maximize' if maximize else 'minimize'})")
        else:
            print(f"  [newRF] no saved optima — opening interactive picker "
                  f"({'maximize' if maximize else 'minimize'}). Click near each optimum, "
                  f"then press Enter / Q.")
            true_optima = _newrf_pick_optima(rf, grid_pts, grid_vals, maximize=maximize)
            if true_optima:
                save_dir = out_dir or os.getcwd()
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, NEWRF_OPTIMA_FILENAME)
                _save_newrf_optima(save_path, true_optima, maximize=maximize,
                                   db_path=db_resolved, value_col=value_col)
                print(f"  [newRF] cached {len(true_optima)} optima -> {save_path}\n"
                      f"          reuse next time with: --runs-path {save_dir}")
            else:
                print("  [newRF] WARNING: no optima selected — metrics will use an "
                      "empty reference set (distance/pct will be penalized).")

        print(f"  [newRF] RF surrogate from {db_resolved} "
              f"({'maximize' if maximize else 'minimize'}, "
              f"{len(true_optima)} reference optima, value='{value_col}')")
        spec = rm.LandscapeSpec(
            landscape="rf", dim=3, maximize=maximize, true_optima=true_optima,
            fn_callable=rf_fn, grid_pts=grid_pts, grid_vals=grid_vals,
            csv_path=db_resolved, objective_column=value_col,
            composition_columns=list(DB_COMP_COLS),
            time_limit_hours=time_limit_hours,
        )
        return _landscape_to_ds(spec, "newRF")

    if dataset == "RF":
        if not runs_path:
            sys.exit("--dataset RF requires --runs-path (source mobo run).")
        cfg_path = os.path.join(runs_path, "run_config.json")
        if not os.path.exists(cfg_path):
            sys.exit(f"--dataset RF needs the source run's config, but {cfg_path} "
                     f"does not exist.")
        with open(cfg_path) as f:
            cfg = json.load(f)
        try:
            csv_path = resolve_surrogate_csv_path(cfg.get("csv_path"))
        except FileNotFoundError as exc:
            sys.exit(str(exc))
        maximize = bool(cfg["maximize"])
        # Always use the canonical mobo_05_06_15_32 campaign1a optima as the
        # reference set, regardless of which source run supplies the CSV/surrogate.
        true_optima = load_true_optima_json(TRIAL112_REFERENCE_OPTIMA)
        obj_col = cfg.get("objective_column", "Objective")
        comp_cols = cfg.get("composition_columns")
        print(f"  [dataset] RF surrogate from {csv_path} "
              f"({'maximize' if maximize else 'minimize'}, "
              f"{len(true_optima)} reference optima from "
              f"{os.path.basename(TRIAL112_REFERENCE_OPTIMA)})")
        _, rf_fn, grid_pts, grid_vals, resolved_cols, dim = rm.build_rf_and_grid(
            csv_path, objective_column=obj_col, composition_columns=comp_cols,
        )
        spec = rm.LandscapeSpec(
            landscape="rf", dim=dim, maximize=maximize, true_optima=true_optima,
            fn_callable=rf_fn, grid_pts=grid_pts, grid_vals=grid_vals,
            csv_path=os.path.abspath(csv_path),
            objective_column=obj_col,
            composition_columns=resolved_cols,
        )
        return _landscape_to_ds(spec, "RF")

    if dataset in ACKLEY_BENCHMARKS:
        d = ACKLEY_BENCHMARKS[dataset]
        fn = Ackley(ackley_variant, dim=d)
        true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
        print(f"  [dataset] {dataset}: Ackley('{ackley_variant}', dim={d}) — "
              f"maximize, {len(true_optima)} analytic peak(s)")
        grid_pts = grid_vals = None
        if d == 3:
            grid_pts = rm.ternary_grid(rm.TERNARY_GRID_N)
            grid_vals = fn.predict(grid_pts)
        spec = rm.LandscapeSpec(
            landscape="ackley", dim=d, maximize=True, true_optima=true_optima,
            fn_callable=fn, grid_pts=grid_pts, grid_vals=grid_vals,
            time_limit_hours=time_limit_hours,
        )
        return _landscape_to_ds(spec, dataset, ackley_fn=(fn if d == 4 else None))

    if dataset in GAUSSIAN_BENCHMARKS:
        d = GAUSSIAN_BENCHMARKS[dataset]
        gkw = _gaussian_kwargs(config)
        fn = RealisticGaussian(d, peak_seed=seed, **gkw)
        true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
        print(f"  [dataset] {dataset}: RealisticGaussian(dim={d}, "
              f"{len(true_optima)} random peak(s), seed={seed})")
        grid_pts = grid_vals = None
        if d == 3:
            grid_pts = rm.ternary_grid(rm.TERNARY_GRID_N)
            grid_vals = fn.predict(grid_pts)
        spec = rm.LandscapeSpec(
            landscape="synthetic", dim=d, maximize=True, true_optima=true_optima,
            fn_callable=fn, grid_pts=grid_pts, grid_vals=grid_vals,
            time_limit_hours=time_limit_hours, oracle="gaussian",
            ackley_layout=layout, synthetic_seed=seed,
        )
        return _landscape_to_ds(spec, dataset, variant="realistic")

    if dataset in ORACLE_CHOICES:
        syn_cfg = config or {}
        oracle = syn_cfg.get("oracle", dataset)
        if oracle != dataset and dataset in ORACLE_CHOICES:
            oracle = dataset
        d = int(syn_cfg.get("dim", dim))
        ly = str(syn_cfg.get("layout", layout))
        sd = int(syn_cfg.get("seed", seed))
        var = str(syn_cfg.get("variant", oracle_variant))
        tl = time_limit_hours
        if tl is None and syn_cfg.get("time_limit_hours") is not None:
            tl = float(syn_cfg["time_limit_hours"])
        gkw = _gaussian_kwargs(syn_cfg)
        spec = build_synthetic_landscape(
            oracle, d, ly, seed=sd, time_limit_hours=tl, variant=var, **gkw,
        )
        print(f"  [dataset] synthetic '{oracle}' variant={var} dim={d} layout={ly} seed={sd} — "
              f"maximize, {len(spec.true_optima)} reference optima")
        return _landscape_to_ds(spec, dataset, variant=var)

    sys.exit(f"--dataset: '{dataset}' is not known "
             f"(choose from {', '.join(sorted(BUILTIN_DATASETS))}).")


# ─── Ensemble (re-randomized per run) ────────────────────────────────────────────

def build_ensemble_ds(config: dict, dataset: str, *, time_limit_hours: float | None = None) -> dict:
    """Instantiate an ``Ensemble`` from ``config`` and wrap it as a dataset dict.

    The true optima are the strong-basin ``centers`` (the guaranteed global
    maxima).  dim == 3 gets a ternary grid for landscape frames; dim == 4 passes
    the ``Ensemble`` itself as ``ackley_fn`` so ``run_single_eval`` renders the
    final-state point cloud over the objective.
    """
    fn = Ensemble(**config)
    dim = int(config["dim"])
    true_optima = [np.asarray(c, dtype=float) for c in fn.centers]
    grid_pts = grid_vals = None
    if dim == 3:
        grid_pts = rm.ternary_grid(rm.TERNARY_GRID_N)
        grid_vals = fn.predict(grid_pts)
    spec = rm.LandscapeSpec(
        landscape="synthetic", dim=dim, maximize=True, true_optima=true_optima,
        fn_callable=fn, grid_pts=grid_pts, grid_vals=grid_vals,
        time_limit_hours=time_limit_hours, oracle="ensemble",
        synthetic_seed=int(config.get("seed", 0)),
    )
    return _landscape_to_ds(spec, dataset, ackley_fn=(fn if dim == 4 else None),
                            variant="ensemble")


# ─── Hyperparameter loading ─────────────────────────────────────────────────────

# Keys allowed in hparams JSON beyond the MOBO search space (override ZOMBI_FIXED).
_ZOMBI_OVERRIDE_KEYS = frozenset({"input_noise", "max_gp_points", "acquisition_type", "verbose"})


def _split_hparams(hp: dict) -> tuple[dict, dict]:
    """Return (mobo_hparams, zombi_fixed_overrides)."""
    mobo = {k: hp[k] for k in rm.HPARAM_NAMES if k in hp}
    overrides = {k: hp[k] for k in _ZOMBI_OVERRIDE_KEYS if k in hp}
    return mobo, overrides


def _normalize_hparams(hp: dict, *, source: str) -> dict:
    missing = [k for k in rm.HPARAM_NAMES if k not in hp]
    if missing:
        sys.exit(f"{source}: missing hparams {missing} (stale hyperparameter set?).")
    mobo, overrides = _split_hparams(hp)
    return {**mobo, **overrides}


def _trial_num_from_path(path: str) -> int | None:
    base = os.path.basename(os.path.normpath(path.rstrip("/\\")))
    if base == "trial.json":
        base = os.path.basename(os.path.dirname(path))
    m = re.fullmatch(r"trial_(\d+)", base)
    return int(m.group(1)) if m else None


def load_hparams_path(path: str) -> tuple[dict[int, dict], str | None]:
    """Load hparams from trial_* dir, trial.json, or mobo_* dir.

    Returns ``({trial_num: hparams}, inferred_runs_path_or_None)``.
    """
    path = os.path.abspath(path)
    inferred_runs: str | None = None

    if os.path.isdir(path):
        if os.path.basename(path).startswith("mobo_"):
            inferred_runs = path
            prog = os.path.join(path, "mobo_progress.json")
            if not os.path.isfile(prog):
                sys.exit(f"No mobo_progress.json in {path}.")
            with open(prog) as f:
                trials = json.load(f).get("trials", [])
            if not trials:
                sys.exit(f"No trials in {prog}.")
            hp = trials[-1].get("hparams")
            if not hp:
                sys.exit(f"Latest trial in {prog} has no hparams.")
            n = int(trials[-1]["trial"])
            print(f"  [hparams] trial {n}: loaded from latest entry in mobo_progress.json")
            return {n: _normalize_hparams(hp, source=f"trial {n}")}, inferred_runs

        trial_json = os.path.join(path, "trial.json")
        if os.path.isfile(trial_json):
            path = trial_json
        else:
            sys.exit(f"Directory has no trial.json: {path}")

    if not os.path.isfile(path):
        sys.exit(f"--hparams path not found: {path}")

    with open(path) as f:
        data = json.load(f)
    hp = data.get("hparams", data)
    n = _trial_num_from_path(path) or 0
    parent = os.path.dirname(path)
    if os.path.basename(parent).startswith("mobo_"):
        inferred_runs = parent
    print(f"  [hparams] trial {n}: loaded from {path}")
    return {n: _normalize_hparams(hp, source=f"trial {n}")}, inferred_runs


def load_hparams_from_json(json_path: str) -> dict[int, dict]:
    """Load hyperparameters from a standalone JSON file.

    Accepts two formats: (1) a trial.json-style dict with an ``"hparams"`` key,
    or (2) a flat dict whose keys are all hyperparameter names.  Returns a
    single-entry ``{0: hparams}`` dict (trial number 0) so it slots into the
    same ``hparams_by_trial`` flow as ``load_trial_hparams``.
    """
    if not os.path.exists(json_path):
        sys.exit(f"--hparams-json: file not found: {json_path}")
    try:
        with open(json_path) as f:
            raw = json.load(f)
    except Exception as exc:
        sys.exit(f"--hparams-json: {json_path} is not valid JSON ({exc}).")
    hp = raw.get("hparams", raw) if isinstance(raw, dict) else None
    if not isinstance(hp, dict):
        sys.exit(f"--hparams-json: expected a dict with hparam keys (or an "
                 f"'hparams' key containing them).")
    missing = [k for k in rm.HPARAM_NAMES if k not in hp]
    if missing:
        sys.exit(f"--hparams-json: missing hparams {missing} "
                 f"(stale hyperparameter set?).")
    mobo, overrides = _split_hparams(hp)
    hp = {**mobo, **overrides}
    print(f"  [hparams-json] loaded {len(hp)} hyperparameters from {json_path}")
    return {0: hp}


def load_trial_hparams(runs_path: str, trial_nums: list[int]) -> dict[int, dict]:
    by_trial: dict[int, dict] = {}
    prog_path = os.path.join(runs_path, "mobo_progress.json")
    if os.path.exists(prog_path):
        try:
            with open(prog_path) as f:
                for t in json.load(f).get("trials", []):
                    by_trial[int(t["trial"])] = t.get("hparams", {})
        except Exception as exc:
            print(f"  [trials] {prog_path} unreadable ({exc}); falling back to trial.json.")

    out: dict[int, dict] = {}
    for n in trial_nums:
        hp = by_trial.get(n)
        if hp is None:
            tj = os.path.join(runs_path, f"trial_{n}", "trial.json")
            if os.path.exists(tj):
                try:
                    with open(tj) as f:
                        hp = json.load(f).get("hparams")
                except Exception as exc:
                    sys.exit(f"--trials: {tj} unreadable ({exc}).")
        if hp is None:
            sys.exit(f"--trials: trial {n} not found in {runs_path}.")
        out[n] = _normalize_hparams(hp, source=f"trial {n}")
        print(f"  [trials] trial {n}: loaded {len(out[n])} hyperparameters")
    return out


# ─── Init data + CSV writers ────────────────────────────────────────────────────

def gen_init_data(fn_callable, maximize: bool, dim: int):
    """Generate ``run_mobo.N_INIT_LINES`` random simplex lines on the ``dim``-simplex."""
    x_a_list, x_e_list, y_list = [], [], []
    for _ in range(rm.N_INIT_LINES):
        x0 = torch.full((dim,), 1.0 / dim, device=rm.DEVICE, dtype=rm.DTYPE)
        dir_ = rm.zero_sum_dirs(1, dim, device=rm.DEVICE, dtype=rm.DTYPE).squeeze(0)
        seg = rm.line_simplex_segment(x0, dir_)
        if seg is None:
            continue
        _, _, x_left, x_right = seg
        t = torch.linspace(0.0, 1.0, rm.NUM_EXPERIMENTS, dtype=torch.float64, device=rm.DEVICE)
        # Clean straight segment = what we "request"; physics-simulated print =
        # what we "measure" (replaces the random add_composition_noise perturbation).
        pts_clean = (x_left.to(torch.float64).unsqueeze(0)
                     + t.unsqueeze(1) * (x_right - x_left).to(torch.float64).unsqueeze(0))
        pts_t = rm.physics_simulate_line(x_left, x_right, num_points=rm.NUM_EXPERIMENTS,
                                         device=rm.DEVICE, dtype=torch.float64)
        pts_np = pts_t.detach().cpu().numpy()
        raw = np.array([fn_callable(x) for x in pts_np], dtype=float)
        y_t = torch.tensor(raw if maximize else -raw, dtype=rm.DTYPE, device=rm.DEVICE)
        y_t = y_t + torch.randn_like(y_t) * (rm.OUTPUT_NOISE_FRAC * y_t.abs())
        pts_out = pts_t.to(dtype=rm.DTYPE, device=rm.DEVICE)
        pts_clean = pts_clean.to(dtype=rm.DTYPE, device=rm.DEVICE)
        x_a_list.append(pts_out)
        x_e_list.append(pts_clean)
        y_list.append(y_t)
    if not x_a_list:
        raise RuntimeError("Could not generate any initial simplex lines.")
    return (torch.cat(x_a_list, dim=0),
            torch.cat(x_e_list, dim=0),
            torch.cat(y_list, dim=0).reshape(-1, 1))


def coord_cols(dim: int, dataset: str) -> list[str]:
    if dataset == "RF" and dim == 3:
        return ["FA", "MA", "Br"]
    if dataset == "newRF" and dim == 3:
        return list(DB_COMP_COLS)
    return [f"x{i + 1}" for i in range(dim)]


def write_points_csv(path: str, dh, snap_records: list[tuple], cols: list[str]) -> None:
    import pandas as pd
    X = dh.X_all_actual.detach().cpu().numpy()
    Y = dh.Y_all.detach().cpu().numpy().ravel()
    n = X.shape[0]
    mask = dh.get_penalty_mask()
    penalized = (~mask.detach().cpu().numpy()) if mask is not None else np.zeros(n, bool)
    act, zm = rm._activation_zoom_per_point(n, snap_records)
    data = {"sample_idx": np.arange(n)}
    for j, c in enumerate(cols):
        data[c] = X[:, j]
    data.update({"Y": Y, "penalized": penalized.astype(int), "activation": act, "zoom": zm})
    pd.DataFrame(data).to_csv(path, index=False)


def write_needles_csv(path: str, dh, cols: list[str], dim: int) -> None:
    import math
    import pandas as pd
    centroid = np.full(dim, 1.0 / dim)
    rows = []
    for i, r in enumerate(dh.get_all_needle_results()):
        pt = r["point"].detach().cpu().numpy().ravel()
        mv_ = r.get("median_value")
        row = {"needle_idx": i}
        for j, c in enumerate(cols):
            row[c] = pt[j]
        row.update({
            "value": r.get("value"),
            "median_value": (None if mv_ is None or (isinstance(mv_, float) and math.isnan(mv_)) else mv_),
            "zoom": r.get("zoom"),
            "iteration": r.get("iteration"),
            "reason": r.get("reason"),
            "dist_to_centre": float(np.linalg.norm(pt - centroid)),
        })
        rows.append(row)
    columns = ["needle_idx"] + cols + ["value", "median_value",
                                       "zoom", "iteration", "reason",
                                       "dist_to_centre"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def write_point_cloud_html(path: str, predict_fn, peaks, dh, last_payload: dict) -> None:
    """Render the final ZoMBI state over the 4-simplex objective cloud (one HTML).

    Uses ``synthetic_data/plot``'s overlay API.  Pared points, needle markers,
    and the last LineBO lines are all exact (plain simplex compositions);
    needle penalisation ellipsoids are intentionally omitted because the run's
    tangent basis differs from the Helmert ILR basis the overlay assumes.

    ``predict_fn`` maps an (N, 4) composition array to (N,) objective values (an
    Ackley/Ensemble ``.predict`` or a GP-landscape batch predictor). ``peaks`` is an
    (M, 4) array of reference optima to mark. Generalised beyond Ackley so the
    fullgp / ensemble evaluation landscapes all render a 4D cloud.
    """
    import plotly.graph_objects as go
    import synthetic_data.plot_ackley as pc4

    comp = pc4.build_simplex_lattice(pc4.GRID_N)
    obj = np.asarray(predict_fn(comp), dtype=float).ravel()
    xyz = pc4.to_3d(comp)
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
    peaks_arr = np.asarray(peaks, dtype=float) if peaks is not None else np.empty((0, 4))
    fig_data = [cloud, pc4.tetra_edges_trace(), pc4.vertex_labels_trace()]
    if peaks_arr.size:
        peaks_xyz = pc4.to_3d(peaks_arr)
        fig_data.append(go.Scatter3d(
            x=peaks_xyz[:, 0], y=peaks_xyz[:, 1], z=peaks_xyz[:, 2], mode="markers",
            name="known peak",
            marker=dict(symbol="diamond", color="red", size=6, line=dict(color="white", width=1)),
            hoverinfo="name",
        ))
    fig = go.Figure(data=fig_data)
    fig.update_layout(
        title="ZoMBI-Hop final state on the 4-simplex objective point cloud",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data"),
        legend=dict(x=0.0, y=1.0), width=pc4.FIG_W, height=pc4.FIG_H,
    )

    pared_X = pared_Y = recency = None
    if dh.X_pared is not None and dh.X_pared.shape[0] > 0:
        pared_X = dh.X_pared.detach().cpu().numpy()
        pared_Y = dh.Y_pared.detach().cpu().numpy().ravel()
        recency = np.arange(pared_X.shape[0], dtype=float)

    needle_t = dh.get_all_needle_locations()
    needles = (needle_t.detach().cpu().numpy()
               if needle_t is not None and needle_t.numel() > 0 else None)
    main_line = (np.array(last_payload["line_0"]) if last_payload
                 and last_payload.get("line_0") is not None else None)
    cache_line = (np.array(last_payload["line_1"]) if last_payload
                  and last_payload.get("line_1") is not None else None)

    pc4.add_simplex_overlays(
        fig, obj_cmin=obj_min, obj_cmax=obj_max,
        pared_points=pared_X, pared_values=pared_Y, recency=recency,
        main_line=main_line, cache_line=cache_line, needles=needles,
    )
    fig.write_html(path, include_plotlyjs="cdn", auto_open=False)


def save_convergence_metrics_plot(
    csv_path: str, out_png: str, *, log_x: bool = False, log_y: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    metrics = [
        ("dist_to_needles", "Distance to True Needles", "steelblue", False),
        ("dup_fraction", "Duplicate Sample Fraction", "tomato", False),
        ("avg_pairwise_dist", "Avg Pairwise Needle Distance", "mediumpurple", False),
        ("recent_needle_value", "Most Recent Needle Value", "darkorange", True),
    ]
    metrics = [m for m in metrics if m[0] in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Convergence metrics over iterations")
    used = 0
    for ax, (col, label, color, steps) in zip(axes.flat, metrics):
        used += 1
        if col == "recent_needle_value" or steps:
            ax.plot(df["iteration"], df[col], color=color, drawstyle="steps-post", marker="o", ms=3)
        else:
            ax.plot(df["iteration"], df[col], color=color)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    has_comp = "pct_matched_comp" in df.columns or "pct_matched" in df.columns
    if has_comp and used < len(axes.flat):
        pct_axes = axes.flat[used]
        if "pct_matched_comp" in df.columns:
            pct_axes.plot(df["iteration"], df["pct_matched_comp"],
                          color="seagreen", label=f"comp ≤ {rm.MATCH_RADIUS}")
        elif "pct_matched" in df.columns:
            pct_axes.plot(df["iteration"], df["pct_matched"],
                          color="seagreen", label=f"comp ≤ {rm.MATCH_RADIUS} (legacy)")
        pct_axes.set_xlabel("Iteration")
        pct_axes.set_ylabel("Pct matched")
        pct_axes.set_title("Pct Needles Matching True Optimum")
        pct_axes.legend(fontsize=8)
        pct_axes.grid(True, alpha=0.3)
        if log_x:
            pct_axes.set_xscale("log")
        if log_y:
            pct_axes.set_yscale("log")
        used += 1
    for ax in axes.flat[used:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      [run] convergence metrics -> {out_png}")


# ─── Single evaluation run ──────────────────────────────────────────────────────

def _force_zoom_floors() -> tuple[int, int]:
    """Minimum ``(max_zooms, max_iterations)`` that let a run declare a needle.

    Mirrors the HPARAM_SPACE lower bounds run_mobo enforces during hparam search,
    but derived from ZoMBIHop's actual search-discipline defaults so the two can't
    drift: a needle can only be declared once the search has zoomed to level
    ``min_zoom_for_needle + 1`` (so ``max_zooms`` must be able to reach it) and only
    after at least ``min_iters_per_zoom`` lines have been sampled at that zoom.
    """
    import inspect
    params = inspect.signature(rm.ZoMBIHop.__init__).parameters
    min_zoom = int(params["min_zoom_for_needle"].default)
    min_iters = int(params["min_iters_per_zoom"].default)
    return min_zoom + 1, min_iters


def run_single_eval(
    hparams: dict,
    ds: dict,
    dataset: str,
    out_dir: str,
    time_limit_min: float,
    *,
    top_k: int | None = None,
    max_lines: int | None = None,
    final_frame_only: bool = False,
    coverage: bool = False,
    no_video: bool = False,
    no_convergence_plot: bool = False,
    log_x: bool = False,
    log_y: bool = False,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    dim = ds["dim"]
    fn = ds["fn"]
    maximize = ds["maximize"]
    true_optima = ds["true_optima"]
    cols = coord_cols(dim, dataset)

    plot_state: dict = {"line_0": None, "line_1": None}
    payloads: list[dict] = []
    snap_records: list[tuple] = []
    call_counter = [0]
    dh_ref = [None]
    gp_ref = [None]
    grid_pts = ds.get("grid_pts")

    sim_obj = rm.make_sim_obj(fn, rm.DEVICE, rm.DTYPE, maximize=maximize)
    inner = rm.make_linebo_wrapper(sim_obj, dim, rm.NUM_LINES, rm.DEVICE, rm.DTYPE, plot_state)

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
            needle_vals=(dh.needle_vals.detach().cpu().numpy().ravel()
                         if dh.needle_vals is not None and dh.needle_vals.shape[0] > 0 else None),
            needle_M_list=[m.detach().cpu().clone() if m is not None else None
                           for m in dh.needle_M_list],
            needle_B=(dh.needle_B.detach().cpu().clone() if dh.needle_B is not None else None),
            bounds=(dh.bounds.detach().cpu().clone() if dh.bounds is not None else None),
            line_0=plot_state.get("line_0"),
            line_1=plot_state.get("line_1"),
            n_points_before=(dh.X_all_actual.shape[0] if dh.X_all_actual is not None else 0),
            gp_grid_vals=(rm.gp_landscape_vals(gp_ref[0], grid_pts, maximize)
                          if dim == 3 else None),
        ))
        if top_k is not None:
            n_needles = needles.shape[0] if needles is not None else 0
            if n_needles >= top_k:
                raise _TopKReached(n_needles)
        # Hard line budget: each obj_wrapper call is one measured line, so stopping
        # here after `max_lines` calls fixes the number of lines every run samples.
        if max_lines is not None and call_counter[0] >= max_lines:
            raise _LineBudgetReached(call_counter[0])
        return x_req, x_act, y

    try:
        X_a, X_e, Y = gen_init_data(fn, maximize, dim)
    except RuntimeError as exc:
        print(f"      [run] init failed: {exc}")
        return {"dist": rm.UNMATCHED_PENALTY, "dup": 1.0, "runtime": 0.0}

    hp = dict(hparams)
    if dim > 3 and (hp.get("top_m_points") is None or hp.get("top_m_points", 0) < dim + 1):
        hp["top_m_points"] = max(dim + 1, 4)

    # Force-zooming discipline: MOBO clamps these via HPARAM_SPACE, but a fixed
    # re-evaluation reads hparams straight from JSON, so enforce the same floors
    # here or a low-max_zooms config could never zoom deep enough to declare a needle.
    _zoom_floor, _iter_floor = _force_zoom_floors()
    for _key, _floor in (("max_zooms", _zoom_floor), ("max_iterations", _iter_floor)):
        if hp.get(_key) is not None and hp[_key] < _floor:
            print(f"      [run] raising {_key} {hp[_key]} -> {_floor} to satisfy "
                  f"force-zooming constraints.")
            hp[_key] = _floor

    zombi_fixed = dict(rm.ZOMBI_FIXED)
    for k in _ZOMBI_OVERRIDE_KEYS:
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
    dh_ref[0] = dh
    gp_ref[0] = optimizer.gp_handler

    orig_snap = dh.take_snapshot
    def snap_wrap(*a, **k):
        orig_snap(*a, **k)
        if dh.X_all_actual is not None:
            # Capture the current zoom zone's linear size (relative to the full
            # [0,1]^d domain) as the 4th element, exactly as run_mobo does, so the
            # duplicate metric can shrink its radius for points sampled inside a zoom
            # — i.e. so eval scores dup on the SAME definition the tuner optimised.
            czb = dh.current_zoom_bounds if dh.current_zoom_bounds is not None else dh.bounds
            zoom_size = rm.zoom_size_fraction(czb) if czb is not None else 1.0
            snap_records.append((dh.X_all_actual.shape[0], dh.current_activation,
                                 dh.current_zoom, zoom_size))
    dh.take_snapshot = snap_wrap

    interrupted = False
    t0 = time.time()
    try:
        optimizer.run(max_activations=float("inf"), time_limit_hours=time_limit_min / 60.0,
                      never_terminate=True)
    except _TopKReached as stop:
        print(f"      [run] top-k reached: {int(stop.args[0])} needle(s) found — stopping early")
    except _LineBudgetReached as stop:
        print(f"      [run] line budget reached: {int(stop.args[0])} measured line(s) — stopping.")
    except KeyboardInterrupt:
        interrupted = True
        print("\n      [run] interrupted — writing artifacts before exit …")
    except Exception as exc:
        print(f"      [run] ZoMBI crashed: {exc}")
        traceback.print_exc()
    runtime = time.time() - t0

    needle_t = dh.get_all_needle_locations()
    discovered = needle_t.detach().cpu().numpy() if needle_t.numel() > 0 else np.empty((0, dim))
    X_all_np = (dh.X_all_actual.detach().cpu().numpy()
                if dh.X_all_actual is not None else np.empty((0, dim)))
    dist = rm.metric_dist_to_needles(discovered, true_optima, dim=dim)
    # Zoom-scaled duplicate radius (matches run_mobo's trial scoring): points sampled
    # inside a small zoom zone must be closer to count as duplicates, so zooming isn't
    # penalised. Without this eval over-reports dup vs. the tuned/selected objective.
    zoom_sizes = rm._zoom_size_per_point(X_all_np.shape[0], snap_records)
    dup = rm.metric_dup_fraction(X_all_np, dim=dim, zoom_sizes=zoom_sizes)
    pct_comp = rm.metric_pct_matched_comp(discovered, true_optima, dim=dim)
    print(f"      [run]  iters={call_counter[0]}  dist={dist:.4f}  dup={dup:.4f}"
          f"  pct_comp={pct_comp:.2f}  t={runtime:.1f}s"
          f"  needles={len(discovered)}/{len(true_optima)}")

    try:
        write_points_csv(os.path.join(out_dir, "points.csv"), dh, snap_records, cols)
        write_needles_csv(os.path.join(out_dir, "needles.csv"), dh, cols, dim)
        rm.write_metrics_over_time_csv(
            os.path.join(out_dir, "metrics_over_time.csv"), payloads, X_all_np, true_optima,
            dim=dim,
        )
    except Exception as exc:
        print(f"      [run] CSV write failed: {exc}")

    try:
        rm.plot_dist_from_centre(os.path.join(out_dir, "dist_from_centre.png"), dh, maximize)
        if dim == 3 and ds.get("grid_pts") is not None:
            rm.plot_line_length_hist(os.path.join(out_dir, "line_length_hist.png"), payloads)
        rm.plot_hparam_edge_proximity(
            os.path.join(out_dir, "hparam_edge_proximity.png"),
            rm.hparams_to_norm(hparams))
        # Per-sample activation ids (from the snapshot records) so the convergence
        # plot resets its running-best envelope at each activation boundary.
        n_y = int(dh.Y_all.shape[0]) if dh.Y_all is not None else 0
        conv_acts, _ = rm._activation_zoom_per_point(n_y, snap_records)
        rm.plot_convergence(os.path.join(out_dir, "convergence.png"), dh, maximize,
                            activations=conv_acts)
    except Exception as exc:
        print(f"      [run] static plot failed: {exc}")

    metrics_csv = os.path.join(out_dir, "metrics_over_time.csv")
    if not no_convergence_plot and os.path.isfile(metrics_csv):
        try:
            save_convergence_metrics_plot(
                metrics_csv, os.path.join(out_dir, "convergence_metrics.png"),
                log_x=log_x, log_y=log_y,
            )
        except Exception as exc:
            print(f"      [run] convergence metrics plot failed: {exc}")

    try:
        if dim == 3 and ds.get("grid_pts") is not None and ds.get("grid_vals") is not None:
            ref_title = (
                "Reference: oracle landscape"
                if ds.get("landscape") == "synthetic" else
                "Reference: RF landscape" if dataset == "RF" else
                f"Reference: {dataset} landscape"
            )
            if final_frame_only:
                # Only the FINAL ternary frame, written straight into out_dir (no
                # plots/ subdir, no timelapse video).
                if payloads:
                    p = payloads[-1]
                    print("      [run] rendering final ternary frame …", flush=True)
                    rm.render_frame(p, ds["grid_pts"], ds["grid_vals"], true_optima, maximize,
                                    os.path.join(out_dir, f"iter_{p['iter_num'] - 1:04d}.png"),
                                    ref_title=ref_title)
            else:
                plots_dir = os.path.join(out_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                print(f"      [run] rendering {len(payloads)} ternary frames …", flush=True)
                for p in payloads:
                    rm.render_frame(p, ds["grid_pts"], ds["grid_vals"], true_optima, maximize,
                                    os.path.join(plots_dir, f"iter_{p['iter_num'] - 1:04d}.png"),
                                    ref_title=ref_title)
                if not no_video and mv is not None:
                    try:
                        mv.make_video_from_dir(plots_dir, os.path.join(out_dir, "zombihop_timelapse.mp4"))
                    except Exception as exc:
                        print(f"      [run] video assembly failed: {exc}")
        elif dim == 4:
            # Final-state interactive 4-simplex cloud. Works for the Ackley/Ensemble
            # oracles (batch .predict + .centers) and for the GP landscapes (fullgp),
            # whose fn is single-point — wrap it into a batch predictor and mark the
            # true optima as peaks.
            last_payload = payloads[-1] if payloads else None
            predict_fn = peaks = None
            if ds.get("ackley_fn") is not None:
                predict_fn = ds["ackley_fn"].predict
                peaks = np.asarray(ds["ackley_fn"].centers, dtype=float)
            elif ds.get("fn") is not None:
                _f = ds["fn"]
                predict_fn = lambda comp: np.array([float(_f(np.asarray(x, dtype=float)))
                                                    for x in np.asarray(comp, dtype=float)])
                peaks = (np.asarray(true_optima, dtype=float) if true_optima is not None
                         and len(true_optima) else np.empty((0, dim)))
            if predict_fn is not None:
                print("      [run] rendering 4D point-cloud HTML …", flush=True)
                write_point_cloud_html(os.path.join(out_dir, "point_cloud.html"),
                                       predict_fn, peaks, dh, last_payload)
    except Exception as exc:
        print(f"      [run] landscape render failed: {exc}")

    # Coverage plot (3D ternary / 4D tetrahedron): background ground-truth grid + all
    # sampled points, exactly like optimize/coverage_plot.py. Self-contained — write
    # the per-trial ground truth (npz) and a minimal config into out_dir, then reuse
    # save_coverage_image. For 3D we reuse the dataset's prebuilt grid; for 4D (where
    # fullgp/ensemble carry no grid) we build a simplex grid and evaluate the objective
    # on it (batch predictor if available, else the single-point fn — ~sub-second).
    if coverage and dim in (3, 4):
        try:
            import coverage_plot
            cov_grid_pts = ds.get("grid_pts")
            cov_grid_vals = ds.get("grid_vals")
            if cov_grid_pts is None or cov_grid_vals is None:
                cov_grid_pts = (coverage_plot.tetra_grid(coverage_plot.GRID_N_4D)
                                if dim == 4 else coverage_plot.ternary_grid())
                if ds.get("ackley_fn") is not None:
                    cov_grid_vals = np.asarray(ds["ackley_fn"].predict(cov_grid_pts),
                                               dtype=float)
                else:
                    _cf = ds["fn"]
                    cov_grid_vals = np.array([float(_cf(np.asarray(x, dtype=float)))
                                              for x in cov_grid_pts])
            np.savez(os.path.join(out_dir, "coverage_ground_truth.npz"),
                     grid_pts=np.asarray(cov_grid_pts, dtype=float),
                     grid_vals=np.asarray(cov_grid_vals, dtype=float),
                     true_optima=(np.asarray(true_optima, dtype=float)
                                  if true_optima is not None and len(true_optima)
                                  else np.empty((0, dim))))
            with open(os.path.join(out_dir, "run_config.json"), "w") as f:
                json.dump({"dataset": dataset, "dim": dim, "maximize": maximize}, f)
            coverage_plot.save_coverage_image(out_dir)
            print(f"      [run] coverage plot -> {os.path.join(out_dir, 'coverage.png')}")
        except Exception as exc:
            print(f"      [run] coverage plot failed: {exc}")

    metrics = {
        "dist_to_needles": round(dist, 6),
        "dup_fraction": round(dup, 6),
        "pct_matched_comp": round(pct_comp, 4),
        "pct_matched": round(pct_comp, 4),
        "runtime_s": round(runtime, 3),
        "landscape_config": ds.get("landscape_config"),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if interrupted:
        raise KeyboardInterrupt
    return {
        "dist": dist, "dup": dup,
        "pct_comp": pct_comp, "pct": pct_comp,
        "runtime": runtime,
    }


# ─── Summary ────────────────────────────────────────────────────────────────────

def _agg(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=0)), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "n": int(arr.size)}


def write_summary(path: str, per_trial: dict) -> None:
    summary = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "trials": []}
    for trial_num, runs in per_trial.items():
        entry = {"trial": trial_num, "n_runs": len(runs), "runs": runs}
        if runs:
            for key in ("dist_to_needles", "dup_fraction",
                        "pct_matched_comp", "pct_matched", "runtime_s"):
                entry[key] = _agg([r[key] for r in runs])
        summary["trials"].append(entry)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary -> {path}")


def _build_rerun_config(
    dataset: str, ds: dict, *, runs_path: str | None, trial_nums: list[int],
    args, time_limit_min: float, config_meta: dict | None,
) -> dict:
    cfg = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "dim": ds["dim"],
        "maximize": ds["maximize"],
        "landscape": ds.get("landscape"),
        "oracle": ds.get("oracle"),
        "layout": ds.get("layout"),
        "seed": ds.get("seed"),
        "ackley_variant": (args.ackley_variant if dataset in ACKLEY_BENCHMARKS else None),
        "oracle_variant": ds.get("variant"),
        "gaussian_n_peaks": (len(ds["true_optima"]) if dataset in GAUSSIAN_BENCHMARKS
                             or (dataset == "gaussian" and ds.get("variant") == "realistic")
                             else None),
        "csv_path": ds.get("csv_path"),
        "runs_path": runs_path,
        "trials": trial_nums,
        "num_runs": args.num_runs,
        "time_limit_min": time_limit_min,
        "top_k": args.top_k,
        "true_optima": [list(map(float, t.ravel())) for t in ds["true_optima"]],
        "landscape_config": ds.get("landscape_config"),
    }
    if config_meta:
        cfg["batch_config"] = config_meta.get("name")
        cfg["batch_config_path"] = config_meta.get("path")
    if args.hparams:
        cfg["hparams_source"] = os.path.abspath(args.hparams)
    elif args.hparams_json:
        cfg["hparams_source"] = os.path.abspath(args.hparams_json)
    return cfg


# ─── Main ───────────────────────────────────────────────────────────────────────

def _parse_trials(raw: str) -> list[int]:
    nums = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            nums.append(int(tok))
        except ValueError:
            sys.exit(f"--trials: '{tok}' is not an integer trial number.")
    if not nums:
        sys.exit("--trials: no trial numbers parsed.")
    return nums


def _parse_datasets(raw: str) -> list[str]:
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in BUILTIN_DATASETS:
            sys.exit(f"--dataset: '{tok}' is not a known dataset "
                     f"(choose from {', '.join(sorted(BUILTIN_DATASETS))}).")
        if tok not in out:
            out.append(tok)
    if not out:
        sys.exit("--dataset: no dataset names parsed.")
    return out


def evaluate_dataset(
    dataset: str,
    out_dir: str,
    runs_path: str | None,
    hparams_by_trial: dict[int, dict],
    trial_nums: list[int],
    args,
    *,
    config_meta: dict | None = None,
    synthetic_defaults: dict | None = None,
) -> int:
    time_limit_min = args.time_limit_min
    syn_defaults = synthetic_defaults or {}
    is_ensemble = dataset in ENSEMBLE_DATASETS
    ens_dim = int(syn_defaults.get("dim", args.dim))

    if is_ensemble:
        # No single shared landscape: each run gets a fresh randomized Ensemble
        # (built in the run loop below), so true_optima vary per run.
        ds = None
        rerun_cfg = {
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset,
            "dim": ens_dim,
            "maximize": True,
            "landscape": "synthetic",
            "oracle": "ensemble",
            "ensemble_random_per_run": True,
            "ensemble_seed": args.ensemble_seed,
            "ensemble_optima_margin": args.ensemble_margin,
            "runs_path": runs_path,
            "trials": trial_nums,
            "num_runs": args.num_runs,
            "time_limit_min": time_limit_min,
            "top_k": args.top_k,
        }
        if config_meta:
            rerun_cfg["batch_config"] = config_meta.get("name")
            rerun_cfg["batch_config_path"] = config_meta.get("path")
        if args.hparams:
            rerun_cfg["hparams_source"] = os.path.abspath(args.hparams)
        elif args.hparams_json:
            rerun_cfg["hparams_source"] = os.path.abspath(args.hparams_json)
    else:
        ds = resolve_dataset(
            dataset, runs_path, args.ackley_variant,
            dim=ens_dim,
            layout=syn_defaults.get("layout", args.layout),
            seed=syn_defaults.get("seed", args.seed),
            oracle_variant=syn_defaults.get("variant", args.oracle_variant),
            config=synthetic_defaults,
            time_limit_hours=time_limit_min / 60.0,
            db_path=args.db,
            db_value=args.db_value,
            db_minimize=args.db_minimize,
            optima_json=args.optima_json,
            out_dir=out_dir,
            warmgp_seed=args.warmgp_seed,
        )
        rerun_cfg = _build_rerun_config(
            dataset, ds, runs_path=runs_path, trial_nums=trial_nums,
            args=args, time_limit_min=time_limit_min, config_meta=config_meta,
        )

    with open(os.path.join(out_dir, "rerun_config.json"), "w") as f:
        json.dump(rerun_cfg, f, indent=2)

    # DYNAMIC method: --dynamic-alt-hparams-json arms an every-`period`-iteration
    # alternation between the trial's hparams (the WARM-tuned set) and this
    # alternate set (BEST_HPARAMS). Loaded once; armed around each run below.
    dynamic_alt = None
    if getattr(args, "dynamic_alt_hparams_json", None):
        _alt = json.loads(open(args.dynamic_alt_hparams_json).read())
        dynamic_alt = _alt.get("hparams", _alt) if isinstance(_alt, dict) else None
        if not dynamic_alt:
            sys.exit(f"--dynamic-alt-hparams-json: no hparams in "
                     f"{args.dynamic_alt_hparams_json}")
        print(f"  [dynamic] alternating every {args.dynamic_period} iters with "
              f"{len(dynamic_alt)} alternate hparam(s) from "
              f"{args.dynamic_alt_hparams_json}")

    total = len(trial_nums) * args.num_runs
    done = 0
    per_trial: dict = {}
    for trial_num in trial_nums:
        hparams = hparams_by_trial[trial_num]
        trial_dir = os.path.join(out_dir, f"trial_{trial_num}")
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)

        print(f"\n=== [{dataset}] trial {trial_num}  "
              f"({args.num_runs} run(s) @ {time_limit_min:.2g} min) ===")
        per_trial[trial_num] = []
        for k in range(1, args.num_runs + 1):
            done += 1
            print(f"  [run {k}/{args.num_runs}]  (overall {done}/{total})")
            run_dir = os.path.join(trial_dir, f"run_{k}")
            run_ds = ds
            if is_ensemble:
                # Reproducible per (trial, run): same --ensemble-seed regenerates
                # the identical landscape sequence; ensemble_config.json records it.
                idx = int(trial_num) * 1024 + int(k)
                ens_cfg = random_ensemble_config(
                    ens_dim, idx, seed=int(args.ensemble_seed),
                    optima_margin=args.ensemble_margin)
                run_ds = build_ensemble_ds(
                    ens_cfg, dataset, time_limit_hours=time_limit_min / 60.0)
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "ensemble_config.json"), "w") as f:
                    json.dump(ens_cfg, f, indent=2)
            try:
                eval_kwargs = dict(
                    top_k=args.top_k,
                    max_lines=args.max_lines,
                    final_frame_only=args.final_frame_only,
                    coverage=args.coverage,
                    no_video=args.no_video,
                    no_convergence_plot=args.no_convergence_plot,
                    log_x=args.log_x,
                    log_y=args.log_y,
                )
                if dynamic_alt is not None:
                    from warm_start import dynamic_hparams as dyn
                    dyn.arm(hparams, dynamic_alt, period=args.dynamic_period)
                try:
                    res = run_single_eval(
                        hparams, run_ds, dataset, run_dir, time_limit_min, **eval_kwargs,
                    )
                finally:
                    if dynamic_alt is not None:
                        from warm_start import dynamic_hparams as dyn
                        dyn.disarm()
                per_trial[trial_num].append({
                    "run": k,
                    "dist_to_needles": round(res["dist"], 6),
                    "dup_fraction": round(res["dup"], 6),
                    "pct_matched_comp": round(res["pct_comp"], 4),
                    "pct_matched": round(res["pct_comp"], 4),
                    "runtime_s": round(res["runtime"], 3),
                })
            except KeyboardInterrupt:
                print("\n[!] Interrupted by user — writing summary so far …")
                write_summary(os.path.join(out_dir, "rerun_summary.json"), per_trial)
                raise
            except Exception as exc:
                print(f"  [run {k}] FAILED: {exc}")
            write_summary(os.path.join(out_dir, "rerun_summary.json"), per_trial)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate ZoMBI-Hop hyperparameter sets on chosen objective(s).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--trials", default=None, metavar="N,N,...",
                     help="Comma-separated source trial numbers from --runs-path.")
    src.add_argument("--hparams", default=None, metavar="PATH",
                     help="trial_* dir, trial.json, or mobo_* dir (latest trial).")
    src.add_argument("--hparams-json", default=None, metavar="PATH",
                     help="Flat hparam dict or trial.json-style file (labelled trial_0).")

    ds_grp = parser.add_mutually_exclusive_group(required=False)
    ds_grp.add_argument("--dataset", default=None, metavar="DS[,DS...]",
                        help="Objective(s): RF, newRF, ackley3d/4d/10d, or synthetic oracle names.")
    ds_grp.add_argument("--config", default=None, metavar="PATH",
                        help="Synthetic batch JSON (implies --dataset from oracle field).")

    parser.add_argument("--runs-path", default=None, metavar="MOBO_DIR",
                        help="Source mobo_* dir (required for --trials and --dataset RF).")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Repeats per selected trial (default: 1).")
    parser.add_argument("--time-limit-min", type=float, default=10.0,
                        help="Wall-clock budget per run, minutes (default: 10).")
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="Stop each run once K needles are found.")
    parser.add_argument("--ackley-variant", default="realistic",
                        choices=sorted(Ackley.VARIANTS),
                        help="Ackley variant for ackley3d/4d/10d (default: realistic).")
    parser.add_argument("--oracle-variant", default="layout",
                        choices=["layout", "realistic"],
                        help="Synthetic oracle variant; realistic = random peaks for gaussian.")
    parser.add_argument("--db", default=DEFAULT_DB, metavar="DB",
                        help=f"newRF results DB (path or bare name under data/; "
                             f"default: {DEFAULT_DB}).")
    parser.add_argument("--db-value", default=DEFAULT_DB_VALUE, metavar="COL",
                        help=f"newRF DB value column for the objective "
                             f"(default: {DEFAULT_DB_VALUE}).")
    parser.add_argument("--db-minimize", action="store_true",
                        help="newRF: minimize the DB objective instead of maximizing.")
    parser.add_argument("--optima-json", default=None, metavar="PATH",
                        help="newRF: JSON with a 'true_optima' list to use as the "
                             "reference set (skips the interactive picker).")
    parser.add_argument("--warmgp-seed", type=int, default=0,
                        help="Seed for the warmgp dataset's warm-start GP landscape "
                             "(default: 0; matches the tuning jobs).")
    parser.add_argument("--dynamic-alt-hparams-json", default=None, metavar="PATH",
                        help="DYNAMIC method: alternate the trial's (WARM-tuned) "
                             "hparams with the hparams in this JSON every "
                             "--dynamic-period iterations (trial.json-style or flat).")
    parser.add_argument("--dynamic-period", type=int, default=10,
                        help="Iterations per block for --dynamic-alt-hparams-json "
                             "(default: 10).")
    parser.add_argument("--dim", type=int, default=3,
                        help="Simplex dimension for synthetic oracles (default: 3).")
    parser.add_argument("--layout", default="2", choices=["1", "2", "3"],
                        help="Peak layout for synthetic oracles (default: 2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for synthetic oracles (default: 42).")
    parser.add_argument("--ensemble-seed", type=int, default=0,
                        help="Master seed for per-run 'ensemble' randomization "
                             "(same value reproduces the landscape sequence; default: 0).")
    parser.add_argument("--ensemble-margin", type=float, default=0.2,
                        help="optima_margin for the 'ensemble' dataset — normalized gap "
                             "the background stays below the optima (default: 0.2).")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None,
                        help="Override compute device.")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Hard-cap each run at this many measured lines (one per "
                             "ZoMBI iteration / obj call). Fixes the #lines every run "
                             "samples; the --time-limit-min cap still applies.")
    parser.add_argument("--final-frame-only", action="store_true",
                        help="3D: write only the FINAL ternary frame (iter_*.png) into "
                             "the run dir; skip the plots/ subdir and the timelapse MP4.")
    parser.add_argument("--coverage", action="store_true",
                        help="Also write a coverage.png (ground-truth grid + all sampled "
                             "points), like optimize/coverage_plot.py. 3D ternary / 4D "
                             "tetrahedron.")
    parser.add_argument("--no-video", action="store_true", help="Skip timelapse MP4 assembly.")
    parser.add_argument("--no-convergence-plot", action="store_true",
                        help="Skip convergence_metrics.png panel plot.")
    parser.add_argument("--log-x", action="store_true", help="Log x-axis on convergence_metrics.png.")
    parser.add_argument("--log-y", action="store_true", help="Log y-axis on convergence_metrics.png.")
    parser.add_argument("--out", default=None,
                        help="Parent directory for rerun_* output (default: optimize/runs).")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="Exact output directory (skip auto rerun_* naming).")
    args = parser.parse_args()

    if args.num_runs < 1:
        sys.exit("--num-runs must be >= 1.")
    if args.top_k is not None and args.top_k < 1:
        sys.exit("--top-k must be >= 1.")
    config_meta: dict | None = None
    synthetic_defaults: dict | None = None
    if args.config:
        batch = _load_synthetic_batch_config(args.config)
        config_meta = {"name": batch["name"], "path": os.path.abspath(args.config)}
        synthetic_defaults = batch
        if args.dataset is None:
            args.dataset = batch["oracle"]
        if batch.get("time_limit_hours") is not None:
            args.time_limit_min = float(batch["time_limit_hours"]) * 60.0

    if not args.dataset:
        sys.exit("Provide --dataset or --config.")

    datasets = _parse_datasets(args.dataset)
    if args.config and len(datasets) > 1:
        sys.exit("--config can only be used with a single synthetic dataset.")

    runs_path: str | None = os.path.abspath(args.runs_path) if args.runs_path else None

    if args.trials:
        if not runs_path:
            sys.exit("--trials requires --runs-path.")
        if not os.path.isdir(runs_path):
            sys.exit(f"--runs-path not found: {runs_path}")
        trial_nums = _parse_trials(args.trials)
        hparams_by_trial = load_trial_hparams(runs_path, trial_nums)
    elif args.hparams:
        hparams_by_trial, inferred = load_hparams_path(args.hparams)
        trial_nums = list(hparams_by_trial.keys())
        if runs_path is None:
            runs_path = inferred
    else:
        assert args.hparams_json
        hparams_by_trial = load_hparams_from_json(args.hparams_json)
        trial_nums = list(hparams_by_trial.keys())

    if "RF" in datasets and not runs_path:
        sys.exit("--dataset RF requires --runs-path.")
    if runs_path and not os.path.isdir(runs_path):
        sys.exit(f"--runs-path not found: {runs_path}")

    rm._configure_mpl_backend(headless=True)
    rm._apply_runtime_overrides(device=args.device, time_limit_hours=None)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.out_dir:
        eval_dir = os.path.abspath(args.out_dir)
        os.makedirs(eval_dir, exist_ok=True)
    else:
        out_parent = os.path.abspath(args.out) if args.out else os.path.join(script_dir, "runs")
        os.makedirs(out_parent, exist_ok=True)
        eval_dir = rm.unique_run_dir(out_parent, "rerun")


    time_limit_min = args.time_limit_min
    print("=" * 72)
    print(f"ZoMBI-Hop evaluate  |  datasets={datasets}  trials={trial_nums}  "
          f"num_runs={args.num_runs}  limit={time_limit_min:.2g} min"
          + (f"  top_k={args.top_k}" if args.top_k is not None else ""))
    if args.hparams:
        print(f"hparams: {os.path.abspath(args.hparams)}")
    elif args.hparams_json:
        print(f"hparams: {os.path.abspath(args.hparams_json)}")
    if runs_path:
        print(f"source: {runs_path}")
    if args.config:
        print(f"config: {os.path.abspath(args.config)}")
    print(f"output: {eval_dir}")
    print("=" * 72)

    # One dataset writes its artifacts directly into eval_dir (unchanged layout);
    # multiple datasets each get their own sub-folder under it.
    done = 0
    for dataset in datasets:
        out_dir = eval_dir if len(datasets) == 1 else os.path.join(eval_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)
        done += evaluate_dataset(
            dataset, out_dir, runs_path, hparams_by_trial, trial_nums, args,
            config_meta=config_meta, synthetic_defaults=synthetic_defaults,
        )

    print(f"\nDone. {done} run(s) across {len(trial_nums)} trial(s) "
          f"and {len(datasets)} dataset(s). Results in {eval_dir}")


if __name__ == "__main__":
    main()
