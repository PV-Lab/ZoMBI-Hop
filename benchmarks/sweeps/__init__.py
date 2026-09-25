"""
benchmarks/sweeps — how robust is ZoMBI-Hop to the shape of the landscape?

Where ``benchmarks/ablations`` varies the OPTIMISER on one landscape family, this
package holds the optimiser fixed and varies the LANDSCAPE across a full-factorial
grid:

    number of needles   2, 10, 30, 50
    basin sharpness b   2.2, 6, 10, 15
    dimension           3, 4, 6, 10

on a bumps-only objective — negated-Ackley optima on a flat plain, with every other
``Ensemble`` feature (roughness, ridges, plateaus, distractors, anisotropy, edge
bias) switched off, so a difference between two cells is attributable to the three
swept quantities and nothing else.

Every cell gets the same **measurement budget**: 125 LineBO lines = 3000 measured
compositions, rather than the same wall-clock, so the dimension axis does not
silently become a plot of the GP's cost curve.

See ``benchmarks/sweeps/README.md`` for the campaign workflow, and
``benchmarks/sweeps/needles.py`` for what "a resolvable needle" means here and why
the optima are placed rather than drawn.
"""

from __future__ import annotations

from ._paths import ensure_paths

ensure_paths()

from .budget import (  # noqa: E402
    DEFAULT_N_LINES,
    BudgetExhausted,
    BudgetState,
    line_budget,
)
from .hparams import HPARAM_MAP, hparams_for_dim  # noqa: E402
from .needles import (  # noqa: E402
    GRID_BASIN_WIDTH,
    GRID_DIM,
    GRID_N_NEEDLES,
    NeedleFactory,
    build_landscape,
    place_optima,
    placement_width,
    prominence_separation,
    target_separation,
)

__all__ = [
    "BudgetExhausted",
    "BudgetState",
    "DEFAULT_N_LINES",
    "GRID_BASIN_WIDTH",
    "GRID_DIM",
    "GRID_N_NEEDLES",
    "HPARAM_MAP",
    "NeedleFactory",
    "build_landscape",
    "ensure_paths",
    "hparams_for_dim",
    "line_budget",
    "place_optima",
    "placement_width",
    "prominence_separation",
    "target_separation",
]
