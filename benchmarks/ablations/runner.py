"""
benchmarks/ablations/runner.py
==============================
One ablation trial = one (arm, landscape, repeat) cell.

Every arm writes the **same artifact set a ``run_mobo`` trial writes** — that is the
point of routing through ``run_mobo.run_single_trial`` rather than reimplementing a
runner here. A cell directory holds::

    points.csv                 every sample: composition, Y, penalised, activation, zoom,
                               plus the zoom_size column this harness adds
    needles.csv                every declared needle, with the activation/iteration it
                               was declared on
    metrics_over_time.csv      dist_to_needles / dup_fraction / n_needles / … per LineBO line
    metrics_over_time.png      the plot of the above
    needle_values.png          needle value vs iteration against the landscape's ceiling
    convergence.png            all Y, running best (reset per activation), needle vlines
    dist_from_centre.png       Y vs distance from the simplex centroid
    line_length_hist.png       LineBO main-line length distribution
    coverage.png               ternary coverage (dim 3 only)
    point_cloud.html           interactive simplex cloud (dim 4 only)
    conet.png / conet_*.png    co-occurrence network renders (ensemble landscapes only)
    ensemble_config.json       the exact landscape, when it is a reseeded ensemble
    coverage_ground_truth.npz  that landscape's optima + render grid (dim 3)

plus two files this harness adds:

    metrics.json               the cell's scalar results. Written LAST and only on
                               success, so its presence is the completion marker the
                               queue resumes from — a cell killed mid-run leaves none
                               and is re-run rather than silently counted.
    arm.json                   the arm definition, the exact hyperparameters used, the
                               landscape spec and the seed. A cell is reproducible from
                               this file alone.

Common random numbers
---------------------
The RNG is seeded from ``(landscape_index, repeat)`` and deliberately **not** from the
arm. Every arm therefore starts from the identical initial design on a given cell, so
the paired baseline-vs-variant difference the summary reports is not inflated by two
arms having drawn different starting lines. The arms diverge immediately afterwards —
they consume the stream at different rates — which is exactly the intent: shared start,
independent thereafter.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from ._paths import ensure_paths

ensure_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_mobo as rm  # noqa: E402
from eval_metrics import metric_n_points_penalty  # noqa: E402

from .arms import ARMS, Arm, arm_context, patch_points_csv_zoom_size  # noqa: E402
from .landscapes import LandscapeFactory  # noqa: E402
from .restarts import run_restart_trial  # noqa: E402

METRICS_FILENAME = "metrics.json"
ARM_FILENAME = "arm.json"

# Keys metrics.json must carry to count as a finished cell. A file missing any of
# them (truncated by a mid-write kill) is treated as an unfinished cell and re-run.
_REQUIRED_METRIC_KEYS = ("arm", "landscape_index", "repeat", "dist_to_needles",
                         "dup_fraction", "n_iters", "n_points")


# ─── Hyperparameters ─────────────────────────────────────────────────────────────

def default_base_hparams() -> dict:
    """The repo's canonical ZoMBI-Hop hyperparameters.

    ``src/default_hparams.DEFAULT_HPARAMS`` is the 6D MOBO ensemble winner and what
    the GUI and the hardware runner both use, so an ablation run against it is an
    ablation of the optimiser as actually deployed — not of some configuration that
    exists only in this harness.
    """
    from src.default_hparams import DEFAULT_HPARAMS

    return dict(DEFAULT_HPARAMS)


def resolve_hparams(base: dict, arm: Arm) -> dict:
    """``base`` with the arm's overrides applied, minus anything ZOMBI_FIXED owns.

    ``run_single_trial`` builds the optimiser as ``ZoMBIHop(**ZOMBI_FIXED, **hp)``,
    so a key present in both raises ``TypeError: got multiple values``. Silently
    dropping the duplicate is right here: ``ZOMBI_FIXED`` holds infrastructure
    constants (device verbosity, the measured input noise) that an ablation has no
    business varying, and letting a stray key in a hyperparameter JSON abort the
    campaign hours in would be worse than ignoring it.
    """
    hp = dict(base)
    hp.update(arm.hparam_overrides)
    clashes = sorted(set(hp) & set(rm.ZOMBI_FIXED))
    for key in clashes:
        hp.pop(key)
    if clashes:
        print(f"    [ablation] ignoring hyperparameter(s) fixed by run_mobo: {clashes}")
    return hp


# ─── Seeding ─────────────────────────────────────────────────────────────────────

def cell_seed(landscape_index: int, repeat: int, base: int = 0) -> int:
    """A stable seed for a cell, identical across arms (see "Common random numbers")."""
    # Mixed rather than concatenated so adjacent cells do not get adjacent seeds,
    # which correlates the initial designs of neighbouring landscapes.
    h = (int(base) * 1_000_003) ^ (int(landscape_index) * 2_654_435_761) \
        ^ (int(repeat) * 40_503)
    return int(abs(h) % (2 ** 31 - 1))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) for one cell."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Completion marker ───────────────────────────────────────────────────────────

def load_cell_metrics(trial_dir: str) -> dict | None:
    """A finished cell's metrics, or None if it is absent or incomplete."""
    path = os.path.join(trial_dir, METRICS_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception:
        return None
    if not all(k in m for k in _REQUIRED_METRIC_KEYS):
        return None
    return m


def is_complete(trial_dir: str) -> bool:
    return load_cell_metrics(trial_dir) is not None


# ─── The trial ───────────────────────────────────────────────────────────────────

def run_ablation_trial(
    *,
    arm: Arm | str,
    factory: LandscapeFactory,
    landscape_index: int,
    repeat: int,
    trial_dir: str,
    base_hparams: dict | None = None,
    device: str | None = None,
    seed_base: int = 0,
    runner_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict:
    """Run one cell and write its artifacts. Returns the metrics dict.

    Raises whatever the optimiser raises — a failed cell must NOT leave a
    ``metrics.json`` behind, or the queue would count it as done.

    ``runner_kwargs`` overrides the arm's own (the campaign passes ``--n-restarts``
    through here), so the manifest and not this file is the record of what ran.
    """
    if isinstance(arm, str):
        arm = ARMS[arm]
    base_hparams = base_hparams if base_hparams is not None else default_base_hparams()
    os.makedirs(trial_dir, exist_ok=True)

    if device is not None:
        rm._apply_runtime_overrides(device=device)

    landscape, ensemble_config = factory.build(landscape_index)
    hparams = resolve_hparams(base_hparams, arm)
    seed = cell_seed(landscape_index, repeat, seed_base)

    kwargs = dict(arm.runner_kwargs)
    kwargs.update(runner_kwargs or {})

    arm_record = {
        "arm": arm.to_dict(),
        "hparams": hparams,
        "runner_kwargs": kwargs,
        "landscape": factory.spec(),
        "landscape_index": int(landscape_index),
        "repeat": int(repeat),
        "seed": seed,
        "dim": int(landscape.dim),
        "time_limit_hours": landscape.time_limit_hours,
        "device": str(rm.DEVICE),
        "n_true_optima": (len(landscape.true_optima) if ensemble_config is None
                          else None),
    }
    _ensure_run_config(trial_dir, landscape)

    if verbose:
        print(f"  [cell] arm={arm.name}  landscape={landscape_index}  repeat={repeat}  "
              f"dim={landscape.dim}  seed={seed}  device={rm.DEVICE}", flush=True)

    seed_everything(seed)
    t0 = time.time()
    try:
        # patch_points_csv_zoom_size is applied to EVERY arm, baseline included, so
        # the extra column means the same thing in every cell (see its docstring).
        with patch_points_csv_zoom_size(), arm_context(arm):
            if arm.runner == "restarts":
                result = run_restart_trial(hparams, landscape, trial_dir,
                                           ensemble_config=ensemble_config,
                                           verbose=verbose, **kwargs)
            elif arm.runner == "single":
                if kwargs:
                    raise TypeError(
                        f"arm {arm.name!r} uses the 'single' runner but was given "
                        f"runner_kwargs {sorted(kwargs)}")
                result = rm.run_single_trial(hparams, landscape, trial_dir,
                                             ensemble_config=ensemble_config)
            else:
                raise ValueError(f"arm {arm.name!r} has unknown runner {arm.runner!r}")
    finally:
        # AFTER the run, not before: run_single_trial rmtree's its trial directory on
        # entry, so anything written first is gone. In `finally` so a crashed cell
        # still says what it was trying to do — the queue can reconstruct that from
        # the manifest, but only if you know which cell to look up.
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, ARM_FILENAME), "w") as f:
            json.dump(arm_record, f, indent=2, default=str)
    wall = time.time() - t0

    metrics = {
        "arm": arm.name,
        "landscape_index": int(landscape_index),
        "repeat": int(repeat),
        "seed": seed,
        "dim": int(landscape.dim),
        # The three MOBO objectives, under the names pareto.py / summary_table.py use,
        # so an ablation cell drops into the existing tooling unchanged.
        "dist_to_needles": round(float(result["dist"]), 6),
        "dup_fraction": round(float(result["dup"]), 6),
        "avg_time_per_iter_s": round(float(result["avg_time_per_iter"]), 4),
        "n_points_penalty": round(metric_n_points_penalty(int(result["n_points"])), 4),
        "runtime_s": round(float(result["runtime"]), 3),
        "wall_s": round(wall, 3),
        "n_iters": int(result["n_iters"]),
        "n_points": int(result["n_points"]),
        "n_needles": _count_needles(trial_dir),
    }
    if "n_restarts_actual" in result:
        metrics["n_restarts_actual"] = int(result["n_restarts_actual"])

    # LAST, and atomically: this file is the completion marker.
    rm._atomic_write_text(os.path.join(trial_dir, METRICS_FILENAME),
                          json.dumps(metrics, indent=2))
    if verbose:
        print(f"  [cell] done — dist={metrics['dist_to_needles']:.4f}  "
              f"dup={metrics['dup_fraction']:.4f}  iters={metrics['n_iters']}  "
              f"needles={metrics['n_needles']}  ({wall:.1f}s)", flush=True)
    return metrics


def _ensure_run_config(trial_dir: str, landscape) -> None:
    """Put a ``run_config.json`` beside the cell so the dim-3 coverage plot works.

    ``coverage_plot._find_config`` searches a trial directory, its parent and its
    grandparent for ``run_config.json``, and calls ``sys.exit()`` when it finds
    none. ``sys.exit`` raises ``SystemExit``, which derives from ``BaseException``
    and therefore slips straight through ``run_mobo._auto_generate_plots``'s
    ``except Exception`` guard — so a dim-3 cell would be killed at its very last
    artifact with every other file already written, and the queue would see a
    failure with no traceback. A ``run_mobo`` trial never hits this because it sits
    inside a run directory that has the file; an ablation cell's equivalent is its
    parent (``runs/<arm>/``), which also puts it in reach of the restart
    sub-directories one level deeper.

    Only ``dim``, ``maximize`` and ``dataset`` are read from it — the landscape's
    true optima and render grid come from the per-cell ``coverage_ground_truth.npz``,
    which is the only correct source for a reseeded ensemble anyway. Written once
    per arm and skipped if present, so concurrent workers do not fight over it.
    """
    parent = os.path.dirname(os.path.normpath(trial_dir))
    if not parent or parent == os.path.normpath(trial_dir):
        return
    path = os.path.join(parent, "run_config.json")
    if os.path.isfile(path):
        return
    os.makedirs(parent, exist_ok=True)
    try:
        rm.write_run_config(parent, landscape,
                            dataset=("ensemble" if landscape.oracle == "ensemble"
                                     else landscape.landscape))
    except Exception:
        # write_run_config pulls in landscape-config logging and SLURM metadata; if
        # any of that trips on an unfamiliar landscape, the three fields the coverage
        # plot actually reads are still worth writing.
        rm._atomic_write_text(path, json.dumps({
            "landscape": landscape.landscape,
            "dataset": landscape.oracle or landscape.landscape,
            "dim": int(landscape.dim),
            "maximize": bool(landscape.maximize),
            "true_optima": [],
        }, indent=2))


def _count_needles(trial_dir: str) -> int:
    """Declared needles, from the artifact rather than the in-memory run.

    Reading it back also checks the artifact actually landed — a cell whose
    needles.csv failed to write should not report a needle count from a live object
    that no longer matches what is on disk.
    """
    path = os.path.join(trial_dir, "needles.csv")
    if not os.path.isfile(path):
        return 0
    try:
        import pandas as pd

        return int(len(pd.read_csv(path)))
    except Exception:
        return 0
