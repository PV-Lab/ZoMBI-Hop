"""
benchmarks/sweeps — ZoMBI-Hop against the standard black-box optimisers, across a
full-factorial grid of needle landscapes on the unit cube.

    methods             zombi_hop, random, gp_bo, turbo, hebo  (any registered in
                        ``benchmarks.methods``, or ``module:Class``)
    number of needles   2, 10, 30, 50
    basin sharpness b   2.2, 6, 10, 15
    dimension           3, 4, 6, 10

on a bumps-only ``CartesianEnsemble`` — negated-Ackley optima on a flat plain in
``[0, 1]^dim``, every other ``Ensemble`` feature off — so a difference between two
cells is attributable to the swept quantities. Every method gets the identical
landscape and noise stream for a given cell, and the same **measurement budget**
(3000 points in batches of 24), enforced by one shared ``Problem``.

See ``benchmarks/sweeps/README.md`` for the workflow, ``benchmarks/methods`` for
the optimiser interface, and ``needles.py`` for what "a resolvable needle" means.
"""

from __future__ import annotations

from ._paths import ensure_paths

ensure_paths()

from .campaign import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_BUDGET,
    DEFAULT_METHODS,
    cell_dir,
    load_manifest,
    read_tasks,
    run_one_cell,
)
from .configs import resolve_method_configs  # noqa: E402
from .hparams import HPARAM_MAP, hparams_for_dim  # noqa: E402
from .needles import (  # noqa: E402
    GRID_BASIN_WIDTH,
    GRID_DIM,
    GRID_N_NEEDLES,
    build_landscape,
    place_optima,
    prominence_separation,
    target_separation,
)

__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_BUDGET",
    "DEFAULT_METHODS",
    "GRID_BASIN_WIDTH",
    "GRID_DIM",
    "GRID_N_NEEDLES",
    "HPARAM_MAP",
    "build_landscape",
    "cell_dir",
    "ensure_paths",
    "hparams_for_dim",
    "load_manifest",
    "place_optima",
    "prominence_separation",
    "read_tasks",
    "resolve_method_configs",
    "run_one_cell",
    "target_separation",
]
