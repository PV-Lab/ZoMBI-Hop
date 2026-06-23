"""
Data classes for ZoMBI-Hop configuration and state.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import torch


@dataclass
class ZoMBIHopConfig:
    """
    Configuration for ZoMBI-Hop optimization.

    Contains all hyperparameters and settings for the optimization algorithm.
    Parameters set to None will be auto-computed during optimization.
    """

    # Core optimization parameters
    max_zooms: int = 3
    max_iterations: int = 10
    top_m_points: Optional[int] = None  # Auto-computed as max(d + 1, 4) if None
    n_restarts: int = 30
    raw: int = 500

    # Penalization parameters
    max_penalty_radius: float = 1.0  # Max ellipsoid semi-axis in tangent space

    # Acquisition-guided trust-ellipsoid optimisation (after each needle is found)
    bounds_opt_steps: int = 200
    bounds_opt_lr: float = 0.02
    bounds_opt_rho: float = 30.0

    # Ellipsoid trust-region (replaces axis-aligned bounds)
    ellipsoid_init_radius: float = 0.5
    ellipsoid_sample_scale: float = 1.0
    barrier_weight_init: float = 0.1
    barrier_weight_final: float = 10.0
    ellipsoid_vol_reg_weight: float = 0.05
    ellipsoid_containment_margin: float = 1.05
    zoom_ellipsoid_eps: float = 1e-4

    # Convergence parameters (PI + noise-based Y/X thresholds)
    convergence_pi_threshold: float = 0.01
    input_noise_threshold_mult: float = 2.0
    output_noise_threshold_mult: float = 2.0
    n_consecutive_converged: int = 2

    # GP parameters
    max_gp_points: int = 3000

    # Acquisition parameters
    repulsion_lambda: Optional[float] = None
    acquisition_type: str = "ucb"
    ucb_beta: float = 0.1
    nat_grad_step: float = 0.02
    nat_grad_max_steps: int = 50

    # Point paring (deduplicate near-identical observations before GP fitting)
    paring_spatial_halfnoise: float = 0.5   # spatial threshold = factor × input_noise
    paring_y_noise_multiplier: float = 1.0  # Y threshold = factor × GP output noise
    input_noise: float = 0.01               # known input noise std per composition component (before renorm)
    input_noise_ilr: Optional[float] = None  # deprecated; converted to input_noise when loading old configs

    # Needle-ellipsoid shrink on repeated failure
    needle_shrink_factor: float = 0.85          # per-step exclusion-radius fraction
    needle_stop_noise_multiplier: float = 3.0   # stop when all axes < factor × input_noise

    # Jaccard sliding-window bounds
    jaccard_window: int = 3          # number of recent bounds to compare against
    jaccard_threshold: float = 0.9   # max allowed Jaccard before trying next window

    # Failure-retry params
    bounds_shrink_factor: float = 0.8    # radius shrink factor on no-new-data failure
    min_axis_noise_mult: float = 2.0     # stop shrinking when max axis < mult × input_noise

    # Device and dtype
    device: str = 'cuda'
    dtype: str = 'float64'

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ZoMBIHopConfig':
        """Create config from dictionary."""
        known_fields = {f.name for f in ZoMBIHopConfig.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        if 'input_noise' not in filtered_data and data.get('input_noise_ilr') is not None:
            filtered_data['input_noise'] = float(data['input_noise_ilr']) / 3.0
        return cls(**filtered_data)

    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            'float32': torch.float32,
            'float64': torch.float64,
            'float16': torch.float16,
        }
        return dtype_map.get(self.dtype, torch.float64)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.max_zooms > 0
        assert self.max_iterations > 0
        assert self.top_m_points is None or self.top_m_points > 0
        assert self.n_restarts > 0
        assert self.raw > 0
        assert 0 <= self.convergence_pi_threshold <= 1
        assert self.input_noise_threshold_mult > 0
        assert self.output_noise_threshold_mult > 0
        assert self.n_consecutive_converged >= 1
        assert self.max_gp_points > 0
        assert self.repulsion_lambda is None or self.repulsion_lambda > 0
        assert self.nat_grad_step > 0
        assert self.nat_grad_max_steps >= 1
        assert self.ellipsoid_init_radius > 0
        assert self.ellipsoid_sample_scale > 0
        assert self.barrier_weight_init >= 0
        assert self.barrier_weight_final >= 0
        assert self.ellipsoid_vol_reg_weight >= 0
        assert self.ellipsoid_containment_margin > 0
        assert self.zoom_ellipsoid_eps > 0


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
    input_noise: float = 0.01
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
