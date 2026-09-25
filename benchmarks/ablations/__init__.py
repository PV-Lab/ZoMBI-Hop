"""
benchmarks/ablations
====================
The four ZoMBI-Hop component ablations, as a reusable API.

    A1  k independent ZoMBI restarts   vs  ZoMBI-Hop's hopping
    A2  no zooming                     vs  the contracting trust region
    A3  random chords                  vs  acquisition-ranked LineBO line selection
    A4  isotropic basins               vs  anisotropic Hessian ellipsoids

Nothing here is tied to a dataset or a dimension. A campaign names a landscape
factory (``--landscape ensemble``, or ``module:my_file.py:build`` for your own) and
a per-cell wall-clock budget; the arms, the artifacts and the statistics are the
same whatever it points at.

Every trial writes the full ``optimize/run_mobo.py`` artifact set — the arms route
through ``run_mobo.run_single_trial`` rather than reimplementing a runner, so a cell
directory is interchangeable with a MOBO trial directory and the existing tooling
(``plot_metrics``, ``coverage_plot``, ``pareto``) reads it unchanged. On top of that
each campaign gets summary artifacts: per ablation, ``dist_to_needles`` and
``dup_fraction`` over time with bootstrap confidence bands, plus paired
baseline-vs-variant tables.

Command line
------------
    python -m benchmarks.ablations plan       --out RUNS_DIR [options]
    python -m benchmarks.ablations run        --out RUNS_DIR [--worker K --n-workers N]
    python -m benchmarks.ablations status     --out RUNS_DIR
    python -m benchmarks.ablations reset-stale --out RUNS_DIR
    python -m benchmarks.ablations summarize  --out RUNS_DIR

Library
-------
    from benchmarks.ablations import ARMS, run_ablation_trial, resolve_landscape

    factory = resolve_landscape("ensemble", dim=6, time_limit_hours=0.5)
    run_ablation_trial(arm="no_zoom", factory=factory, landscape_index=0,
                       repeat=1, trial_dir="…/cell")

See ``README.md`` in this directory for the full walkthrough.
"""

from __future__ import annotations

from .arms import (
    ABLATION_KEYS,
    ABLATIONS,
    ARMS,
    BASELINE_ARM,
    Ablation,
    Arm,
    arm_context,
    arms_for,
)
from .landscapes import (
    BUILTIN_KINDS,
    EnsembleFactory,
    GPSurfaceFactory,
    LandscapeFactory,
    SyntheticFactory,
    resolve_landscape,
)
from .restarts import run_restart_trial
from .runner import (
    cell_seed,
    default_base_hparams,
    is_complete,
    load_cell_metrics,
    resolve_hparams,
    run_ablation_trial,
)
from .summarize import collect_cells, summarize

__all__ = [
    "ABLATIONS",
    "ABLATION_KEYS",
    "ARMS",
    "BASELINE_ARM",
    "BUILTIN_KINDS",
    "Ablation",
    "Arm",
    "EnsembleFactory",
    "GPSurfaceFactory",
    "LandscapeFactory",
    "SyntheticFactory",
    "arm_context",
    "arms_for",
    "cell_seed",
    "collect_cells",
    "default_base_hparams",
    "is_complete",
    "load_cell_metrics",
    "resolve_hparams",
    "resolve_landscape",
    "run_ablation_trial",
    "run_restart_trial",
    "summarize",
]
