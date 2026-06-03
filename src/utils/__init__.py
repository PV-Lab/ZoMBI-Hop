"""Utility functions for ZoMBI-Hop."""

from .simplex import (
    is_on_simplex,
    simplex_distance,
    proj_simplex,
    random_simplex,
    random_simplex_direction,
    random_zero_sum_directions,
    subset_sums_and_signs,
    polytope_volume,
    composition_to_ilr,
    ilr_to_composition,
)
from .visualization import (
    plot_optimization_progress,
    plot_simplex_2d,
    plot_simplex_3d,
    plot_needles_summary,
)
from .datahandler import DataHandler
from .gp_simplex import (
    GPSimplex,
    RepulsiveAcquisition,
)
from .dataclasses import ZoMBIHopConfig

__all__ = [
    # Simplex utilities
    "is_on_simplex",
    "simplex_distance",
    "proj_simplex",
    "random_simplex",
    "random_simplex_direction",
    "random_zero_sum_directions",
    "subset_sums_and_signs",
    "polytope_volume",
    "composition_to_ilr",
    "ilr_to_composition",
    # Visualization
    "plot_optimization_progress",
    "plot_simplex_2d",
    "plot_simplex_3d",
    "plot_needles_summary",
    # Data handling
    "DataHandler",
    # GP handling
    "GPSimplex",
    "RepulsiveAcquisition",
    # Configuration
    "ZoMBIHopConfig",
]
