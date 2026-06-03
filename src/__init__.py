"""
ZoMBI-Hop: Zooming Multi-Basin Identification with Hopping
==========================================================

A Bayesian optimization framework for discovering multiple optima
in materials research applications, constrained to the simplex space.

Core Components:
    - ZoMBIHop: Main optimization algorithm
    - LineBO: Line-based Bayesian optimization for simplex constraints
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core.zombihop import ZoMBIHop
from .core.linebo import LineBO, line_simplex_segment, batch_line_simplex_segments, batch_line_bounds_segments, batch_line_ellipsoid_segments, zero_sum_dirs

__all__ = [
    "ZoMBIHop",
    "LineBO",
    "line_simplex_segment",
    "batch_line_simplex_segments",
    "batch_line_bounds_segments",
    "batch_line_ellipsoid_segments",
    "zero_sum_dirs",
]