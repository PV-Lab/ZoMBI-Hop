"""
Data classes for ZoMBI-Hop state.

Hyperparameters are declared in src/hparams.py, not here.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class Checkpoint:
    """
    Checkpoint metadata (legacy - kept for compatibility).
    """

    run_uuid: str
    d: int
    max_zooms: int
    max_iterations: int
    top_m_points: Optional[int]
    n_restarts: int
    raw: int
    convergence_pi_threshold: float
    input_noise_threshold_mult: float
    output_noise_threshold_mult: float
    n_consecutive_converged: int = 2
    max_gp_points: int = 3000
    max_penalty_radius: float = 1.0
    bounds_opt_steps: int = 200
    bounds_opt_lr: float = 0.02
    bounds_opt_rho: float = 30.0
    ellipsoid_init_radius: float = 0.5
    ellipsoid_sample_scale: float = 1.0
    barrier_weight_init: float = 0.1
    barrier_weight_final: float = 10.0
    ellipsoid_vol_reg_weight: float = 0.05
    ellipsoid_containment_margin: float = 1.05
    zoom_ellipsoid_eps: float = 1e-4
    paring_spatial_halfnoise: float = 0.5
    paring_y_noise_multiplier: float = 1.0
    input_noise: float = 0.128
    input_noise_ilr: Optional[float] = None
    needle_shrink_factor: float = 0.85
    needle_stop_noise_multiplier: float = 3.0
    device: str = 'cuda'
    dtype: str = 'float64'
    timestamp: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
