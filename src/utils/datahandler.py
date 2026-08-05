"""
Data Handler for ZoMBI-Hop
==========================

Stores ALL data, control variables, and iteration state for ZoMBI-Hop.
All hyperparameters live here as plain attributes and are accessed directly.

Call take_snapshot(label) at any time to save a full checkpoint to disk.
Call load_state() to resume from the latest snapshot.

Backward-compat note: push_checkpoint() is kept as an alias for take_snapshot()
so that existing code (zombihop_exp.py) continues to work unchanged.
"""

import re
import numpy as np
import torch
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union, Callable

from .dataclasses import ZoMBIHopConfig
from .simplex import (
    Ellipsoid,
    proj_simplex,
    get_tangent_basis,
)

# Floor for the zoom-scaled point-paring radius (composition L2). The paring
# radius is ``paring_spatial_halfnoise * input_noise`` at the full domain but
# shrinks with the zoom-zone linear size (see DataHandler._zoom_size_factor), so
# tightly-packed points inside a zoom aren't deduplicated away. The floor stops it
# collapsing to ~0 at deep zooms. Kept below the paring base range so full-domain
# behaviour is unchanged; mirrors eval_metrics.DUP_DIST_FLOOR. Tunable.
PARE_DIST_FLOOR = 0.003


class DataHandler:
    """
    Stores all data, state, and control variables for ZoMBI-Hop.

    All hyperparameters are plain attributes (e.g. self.max_zooms).
    Call take_snapshot(label) to save everything to disk at any point.

    Parameters
    ----------
    max_zooms, max_iterations, top_m_points, n_restarts, raw,
    convergence_pi_threshold, input_noise_threshold_mult,
    output_noise_threshold_mult, n_consecutive_converged, max_gp_points,
    repulsion_lambda, acquisition_type, ucb_beta, nat_grad_step,
    nat_grad_max_steps : control variables
    directory : str, optional
        Base directory for snapshots. If None, no saving occurs.
    run_uuid : str, optional
        UUID to resume an existing run.
    max_saved_recent_checkpoints : int, optional
        Kept for backward compatibility. Use max_snapshots instead.
    max_snapshots : int, optional
        Max snapshots to keep (None = keep all).
    device, dtype : compute settings
    config : ZoMBIHopConfig or dict, optional
        If provided, values override the individual keyword arguments.
        Kept for backward compatibility.
    d : int, optional
        Dimensionality (auto-set in save_init).
    """

    def __init__(
        self,
        # --- Control variables ---
        max_zooms: int = 3,
        max_iterations: int = 10,
        top_m_points: Optional[int] = None,
        n_restarts: int = 30,
        raw: int = 500,
        convergence_pi_threshold: float = 0.01,
        input_noise_threshold_mult: float = 2.0,
        output_noise_threshold_mult: float = 2.0,
        n_consecutive_converged: int = 2,
        max_gp_points: int = 3000,
        repulsion_lambda: Optional[float] = None,
        acquisition_type: str = "ucb",
        ucb_beta: float = 0.1,
        nat_grad_step: float = 0.02,
        nat_grad_max_steps: int = 50,
        # --- Storage settings ---
        directory: Optional[str] = None,
        run_uuid: Optional[str] = None,
        is_resume: bool = True,
        max_saved_recent_checkpoints: Optional[int] = 50,  # backward compat
        max_snapshots: Optional[int] = None,
        # --- Compute settings ---
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
        # --- Backward compat ---
        config: Optional[Union[ZoMBIHopConfig, Dict[str, Any]]] = None,
        d: Optional[int] = None,
        # --- Point paring ---
        paring_spatial_halfnoise: float = 0.5,
        paring_y_noise_multiplier: float = 1.0,
        input_noise: Optional[float] = None,
        input_noise_ilr: Optional[float] = None,
        # --- Jaccard sliding-window bounds ---
        jaccard_window: int = 3,
        jaccard_threshold: float = 0.9,
        # --- Logging ---
        verbose: bool = False,
    ):
        # If a config object/dict is provided, extract values from it
        if config is not None:
            cfg = config if isinstance(config, ZoMBIHopConfig) else ZoMBIHopConfig.from_dict(config)
            max_zooms = cfg.max_zooms
            max_iterations = cfg.max_iterations
            if cfg.top_m_points is not None:
                top_m_points = cfg.top_m_points
            n_restarts = cfg.n_restarts
            raw = cfg.raw
            convergence_pi_threshold = cfg.convergence_pi_threshold
            input_noise_threshold_mult = cfg.input_noise_threshold_mult
            output_noise_threshold_mult = cfg.output_noise_threshold_mult
            n_consecutive_converged = cfg.n_consecutive_converged
            max_gp_points = cfg.max_gp_points
            repulsion_lambda = cfg.repulsion_lambda
            acquisition_type = getattr(cfg, 'acquisition_type', acquisition_type)
            ucb_beta = getattr(cfg, 'ucb_beta', ucb_beta)
            nat_grad_step = getattr(cfg, 'nat_grad_step', nat_grad_step)
            nat_grad_max_steps = getattr(cfg, 'nat_grad_max_steps', nat_grad_max_steps)
            paring_spatial_halfnoise = getattr(cfg, 'paring_spatial_halfnoise', paring_spatial_halfnoise)
            paring_y_noise_multiplier = getattr(cfg, 'paring_y_noise_multiplier', paring_y_noise_multiplier)
            if getattr(cfg, 'input_noise', None) is not None:
                input_noise = cfg.input_noise
            elif getattr(cfg, 'input_noise_ilr', None) is not None:
                input_noise = cfg.input_noise_ilr / 3.0

        # --- Store all control variables as plain attributes ---
        self.max_zooms = max_zooms
        self.max_iterations = max_iterations
        self.top_m_points = top_m_points
        self.n_restarts = n_restarts
        self.raw = raw
        self.convergence_pi_threshold = convergence_pi_threshold
        self.input_noise_threshold_mult = input_noise_threshold_mult
        self.output_noise_threshold_mult = output_noise_threshold_mult
        self.n_consecutive_converged = n_consecutive_converged
        self.max_gp_points = max_gp_points
        self.repulsion_lambda = repulsion_lambda
        self.acquisition_type = acquisition_type
        self.ucb_beta = ucb_beta
        self.nat_grad_step = nat_grad_step
        self.nat_grad_max_steps = nat_grad_max_steps

        # Point paring
        self.paring_spatial_halfnoise = float(paring_spatial_halfnoise)
        self.paring_y_noise_multiplier = float(paring_y_noise_multiplier)
        if input_noise is not None:
            self.input_noise = float(input_noise)
        elif input_noise_ilr is not None:
            self.input_noise = float(input_noise_ilr) / 3.0
        else:
            self.input_noise = 0.064

        # Jaccard sliding-window
        self.jaccard_window = int(jaccard_window)
        self.jaccard_threshold = float(jaccard_threshold)

        self.verbose = verbose

        # Compute settings
        self.device = torch.device(device)
        self.dtype = dtype
        self.d = d

        # Storage settings
        self.max_snapshots = max_snapshots
        self.save_enabled = directory is not None
        self._snapshot_count = 0

        # Backward compat: max_saved_recent_checkpoints drives cleanup if max_snapshots not set
        if self.max_snapshots is None and max_saved_recent_checkpoints:
            self.max_snapshots = max_saved_recent_checkpoints
        self.max_saved_recent_checkpoints = max_saved_recent_checkpoints or 0

        # DP delta snapshot tracking
        self._n_init_points: int = 0
        self._log_file: Optional[Path] = None

        # Initialize in-memory state
        self._init_storage()

        # Checkpoint tracking (kept for backward compat with push_checkpoint)
        self.checkpoint_history: List[Tuple[str, bool]] = []

        # Set up run directory and UUID
        if self.save_enabled:
            base_dir = Path(directory)
            base_dir.mkdir(exist_ok=True)
            if run_uuid is not None:
                self.run_uuid = run_uuid
                self.run_dir = base_dir / f"run_{run_uuid}"
                if is_resume:
                    if not self.run_dir.exists():
                        raise ValueError(f"Run directory {self.run_dir} does not exist!")
                else:
                    # Fresh run with a caller-provided UUID. The directory may
                    # already exist (e.g. the GUI pre-creates a stub so the run
                    # appears in the explorer the moment it is launched).
                    self.run_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.run_uuid = str(uuid.uuid4())[:4]
                self.run_dir = base_dir / f"run_{self.run_uuid}"
                self.run_dir.mkdir(exist_ok=True)
        else:
            self.run_uuid = run_uuid or str(uuid.uuid4())[:4]
            self.checkpoint_base_dir = None
            self.run_dir = None

    def _init_storage(self):
        """Initialize all in-memory tensors and state to defaults."""
        self.X_all_actual: Optional[torch.Tensor] = None
        self.X_all_expected: Optional[torch.Tensor] = None
        self.Y_all: Optional[torch.Tensor] = None
        self.X_init_actual: Optional[torch.Tensor] = None
        self.X_init_expected: Optional[torch.Tensor] = None
        self.Y_init: Optional[torch.Tensor] = None
        self.bounds: Optional[torch.Tensor] = None          # (2, d) tensor
        # Tracks zoom-level trust region at the current zoom (may match self.bounds early on).
        self.current_zoom_bounds: Optional[torch.Tensor] = None  # (2, d) tensor
        # Per-axis extent of the full (global) domain, captured at initialize(); the
        # reference for the zoom-size factor that scales the point-paring radius.
        self._full_extent: Optional[torch.Tensor] = None    # (d,) tensor

        self.needles: Optional[torch.Tensor] = None
        self.needle_vals: Optional[torch.Tensor] = None
        self.needle_indices: Optional[torch.Tensor] = None
        self.needle_penalty_radii: Optional[torch.Tensor] = None
        self.needles_results: List[Dict[str, Any]] = []

        # Optional live mirror of the needle list into sql/needles.db (hardware
        # runs only). Set by the hardware run driver via _enable_needles_db();
        # left None for synthetic/GUI/test runs, where sync is a no-op.
        self._needles_db_path: Optional[str] = None
        self._needles_db_dims: Optional[List[int]] = None

        # Per-needle ellipsoid parameters (None entry = fall back to sphere radius).
        # needle_B is shared across all needles (same simplex tangent space).
        self.needle_M_list: List[Optional[torch.Tensor]] = []  # each (d-1, d-1)
        self.needle_B: Optional[torch.Tensor] = None           # (d, d-1)

        self._penalty_mask: Optional[torch.Tensor] = None

        # Pared dataset (noise-deduplicated view of X_all_actual / Y_all for GP)
        self.X_pared: Optional[torch.Tensor] = None
        self.Y_pared: Optional[torch.Tensor] = None
        self._pared_version: int = 0     # increments whenever X_pared/Y_pared changes
        self._gp_output_noise: float = 1e-4  # updated by GPSimplex after each fit

        # Iteration state
        self.current_activation: int = 0
        self.current_zoom: int = 0
        self.current_iteration: int = 0
        self.no_improvements: int = 0  # kept for backward compat

        # Logging
        self.log_ei_history: List[float] = []

        # Jaccard sliding-window bounds history
        self.bounds_history: List[torch.Tensor] = []

        # DP delta tracking (reset when load_state is called)
        self._last_snap_n_points: int = 0
        self._last_snap_n_bounds_history: int = 0

    def save_init(
        self,
        X_init_actual: torch.Tensor,
        X_init_expected: torch.Tensor,
        Y_init: torch.Tensor,
        bounds: torch.Tensor,
    ):
        """
        Set up data storage with initial observations.
        Call once before optimization starts (not needed when resuming).
        Auto-computes top_m_points if not already set.
        """
        self.d = X_init_actual.shape[1]

        self.bounds = bounds.clone().to(device=self.device, dtype=self.dtype)
        self.current_zoom_bounds = self.bounds.clone()
        # Reference extent for the zoom-size factor: the global domain at run start
        # (ZoMBI passes the full [0,1]^d box here). Clamped away from zero so a
        # degenerate axis can't blow up the ratio.
        self._full_extent = (self.bounds[1] - self.bounds[0]).clamp(min=1e-12)

        if self.top_m_points is None:
            self.top_m_points = max(self.d + 1, 4)

        self.X_init_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype)
        self.X_init_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype)
        self.Y_init = Y_init.clone().to(device=self.device, dtype=self.dtype)

        self.X_all_actual = self.X_init_actual.clone()
        self.X_all_expected = self.X_init_expected.clone()
        self.Y_all = self.Y_init.clone()

        # Build pared dataset from init data
        self.X_pared = None
        self.Y_pared = None
        self._pared_version = 0
        self._update_pared(self.X_all_actual, self.Y_all)

        self.needles = torch.empty((0, self.d), device=self.device, dtype=self.dtype)
        self.needle_vals = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.needle_indices = torch.empty((0, 1), device=self.device, dtype=torch.int64)
        self.needle_penalty_radii = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.needle_M_list = []
        self.needle_B = None

        self._update_penalty_mask()

        if self.save_enabled:
            self._n_init_points = self.X_init_actual.shape[0]
            self._log_file = self.run_dir / 'run.log'
            self._save_config()
            self.take_snapshot("init", permanent=True)

    def _save_config(self):
        """Write config.json once at init (human-readable, for inspection)."""
        config = {
            'run_uuid': self.run_uuid,
            'd': self.d,
            'max_zooms': self.max_zooms,
            'max_iterations': self.max_iterations,
            'top_m_points': self.top_m_points,
            'n_restarts': self.n_restarts,
            'raw': self.raw,
            'convergence_pi_threshold': self.convergence_pi_threshold,
            'input_noise_threshold_mult': self.input_noise_threshold_mult,
            'output_noise_threshold_mult': self.output_noise_threshold_mult,
            'n_consecutive_converged': self.n_consecutive_converged,
            'max_gp_points': self.max_gp_points,
            'repulsion_lambda': self.repulsion_lambda,
            'acquisition_type': self.acquisition_type,
            'ucb_beta': self.ucb_beta,
            'nat_grad_step': self.nat_grad_step,
            'nat_grad_max_steps': self.nat_grad_max_steps,
            'input_noise': self.input_noise,
            'paring_spatial_halfnoise': self.paring_spatial_halfnoise,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # =========================================================================
    # Snapshotting (new, simple system)
    # =========================================================================

    def take_snapshot(
        self,
        label: str = "",
        permanent: bool = False,
        activation: Optional[int] = None,
        zoom: Optional[int] = None,
        iteration: Optional[int] = None,
    ):
        """
        Save complete state to disk.

        Optionally updates current_activation/zoom/iteration before saving so
        a single call replaces the old update_iteration_state + take_snapshot pair.

        Saves all tensors, needle results, iteration state, and a summary.
        Snapshots are numbered sequentially under run_dir/snapshots/.
        Removes oldest non-permanent snapshots if max_snapshots is set.
        Permanent snapshots are never cleaned up.
        """
        if activation is not None:
            self.current_activation = activation
        if zoom is not None:
            self.current_zoom = zoom
        if iteration is not None:
            self.current_iteration = iteration

        if not self.save_enabled:
            return

        self._snapshot_count += 1
        name = f"{self._snapshot_count:04d}_{label}" if label else f"{self._snapshot_count:04d}"
        snapshot_dir = self.run_dir / "snapshots" / name
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        if permanent:
            (snapshot_dir / 'permanent').touch()

        # Serialise ellipsoid M matrices (list of Optional tensors) as a stacked
        # tensor + boolean has_M flag so None entries survive serialisation.
        if self.needle_M_list:
            # Infer (d-1) from first non-None entry or from needle_B
            dm1 = None
            for m in self.needle_M_list:
                if m is not None:
                    dm1 = m.shape[0]; break
            if dm1 is None and self.needle_B is not None:
                dm1 = self.needle_B.shape[1]
            if dm1 is None:
                dm1 = max(self.d - 1, 1)
            needle_has_M = torch.tensor(
                [m is not None for m in self.needle_M_list], dtype=torch.bool, device=self.device
            )
            needle_M_stack = torch.stack([
                m.to(device=self.device, dtype=self.dtype) if m is not None
                else torch.zeros(dm1, dm1, device=self.device, dtype=self.dtype)
                for m in self.needle_M_list
            ], dim=0)
        else:
            needle_has_M = torch.zeros(0, dtype=torch.bool, device=self.device)
            needle_M_stack = torch.zeros(0, 1, 1, device=self.device, dtype=self.dtype)

        # --- DP delta: only save rows added since the previous snapshot ---
        _d = self.d or 1
        n_curr = self.X_all_actual.shape[0] if self.X_all_actual is not None else 0
        n_prev = self._last_snap_n_points

        _empty_x = torch.zeros(0, _d, device=self.device, dtype=self.dtype)
        _empty_y = torch.zeros(0, 1, device=self.device, dtype=self.dtype)

        X_new = (self.X_all_actual[n_prev:].to(device=self.device, dtype=self.dtype)
                 if n_curr > n_prev else _empty_x)
        X_exp_new = (self.X_all_expected[n_prev:].to(device=self.device, dtype=self.dtype)
                     if n_curr > n_prev else _empty_x)
        Y_new = (self.Y_all[n_prev:].to(device=self.device, dtype=self.dtype)
                 if n_curr > n_prev else _empty_y)

        # Bounds-history delta
        n_bh_curr = len(self.bounds_history)
        n_bh_prev = self._last_snap_n_bounds_history
        bh_new = self.bounds_history[n_bh_prev:]
        bounds_history_new = (
            torch.stack([b.to(self.device, self.dtype) for b in bh_new])
            if bh_new
            else torch.zeros(0, 2, _d, device=self.device, dtype=self.dtype)
        )

        _t0 = time.time()
        torch.save({
            'delta_version': 1,
            'n_cumulative_points': n_curr,
            'n_prev_points': n_prev,
            'n_init_points': self._n_init_points,
            # Growing arrays — delta only
            'X_new': X_new,
            'X_exp_new': X_exp_new,
            'Y_new': Y_new,
            # Small tensors — full copy each snapshot
            'bounds': self.bounds,
            'current_zoom_bounds': (
                self.current_zoom_bounds if self.current_zoom_bounds is not None else self.bounds
            ),
            'needles': self.needles,
            'needle_vals': self.needle_vals,
            'needle_indices': self.needle_indices,
            'needle_penalty_radii': self.needle_penalty_radii,
            'needle_M_stack': needle_M_stack,
            'needle_has_M': needle_has_M,
            'needle_B': self.needle_B,
            'penalty_mask': self._penalty_mask,
            # Bounds-history delta
            'bounds_history_new': bounds_history_new,
            'n_prev_bounds_history': n_bh_prev,
        }, snapshot_dir / 'delta.pt')

        if self.verbose:
            print(f"  [DH.snap] delta torch.save: {time.time()-_t0:.2f}s  (+{n_curr - n_prev} pts)")

        # Advance delta tracking
        self._last_snap_n_points = n_curr
        self._last_snap_n_bounds_history = n_bh_curr

        # Needle results as JSON
        import math as _math
        needles_json = [
            {
                'point': r['point'].cpu().tolist(),
                'value': r['value'],
                'median_value': (
                    None if r.get('median_value') is None or _math.isnan(r['median_value'])
                    else r['median_value']
                ),
                'zoom': r['zoom'],
                'iteration': r['iteration'],
                'n_points': r.get('n_points'),
            }
            for r in self.needles_results
        ]
        with open(snapshot_dir / 'needles.json', 'w') as f:
            json.dump(needles_json, f, indent=2)

        # Human-readable summary
        summary = {
            'label': label,
            'timestamp': time.time(),
            'activation': self.current_activation,
            'zoom': self.current_zoom,
            'iteration': self.current_iteration,
            'n_points': self.X_all_actual.shape[0] if self.X_all_actual is not None else 0,
            'n_needles': self.needles.shape[0] if self.needles is not None else 0,
            'best_y': self.Y_all.max().item() if self.Y_all is not None and self.Y_all.numel() > 0 else None,
            'best_y_unpenalized': (
                self.Y_all[self._penalty_mask].max().item()
                if self._penalty_mask is not None and self._penalty_mask.any() else None
            ),
        }
        with open(snapshot_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Track latest
        with open(self.run_dir / 'latest.txt', 'w') as f:
            f.write(name)

        # Cleanup only applies to legacy full-copy snapshots. Delta format stores
        # incremental diffs, so every snapshot is needed for reconstruction.
        # Skip cleanup when using delta format (delta.pt present).

    def _cleanup_old_snapshots(self):
        """Remove oldest non-permanent snapshots, keeping the last max_snapshots."""
        snapshots_dir = self.run_dir / "snapshots"
        if not snapshots_dir.exists():
            return
        all_snapshots = sorted(snapshots_dir.iterdir())
        non_permanent = [s for s in all_snapshots if not (s / 'permanent').exists()]
        for old in non_permanent[:-self.max_snapshots]:
            shutil.rmtree(old, ignore_errors=True)

    # =========================================================================
    # Checkpointing (kept for backward compatibility with zombihop_exp.py)
    # =========================================================================

    def push_checkpoint(self, label: str, is_permanent: bool = False):
        """Backward-compat wrapper: calls take_snapshot(label, permanent=is_permanent)."""
        self.take_snapshot(label, permanent=is_permanent)

    def write_log(self, message: str):
        """Append a timestamped message to the per-run log file."""
        if self._log_file is None:
            return
        try:
            ts = time.strftime("%H:%M:%S")
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] {message}\n")
        except Exception:
            pass

    # =========================================================================
    # State loading
    # =========================================================================

    def load_state(self) -> Tuple[int, int, int, int]:
        """
        Load state from the latest snapshot on disk.
        Returns (current_activation, current_zoom, current_iteration, no_improvements).
        """
        if not self.save_enabled or not self.run_dir.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        # Load config
        config_path = self.run_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            self.d = cfg.get('d', self.d)
            self.max_zooms = cfg.get('max_zooms', self.max_zooms)
            self.max_iterations = cfg.get('max_iterations', self.max_iterations)
            self.top_m_points = cfg.get('top_m_points', self.top_m_points)
            self.n_restarts = cfg.get('n_restarts', self.n_restarts)
            self.raw = cfg.get('raw', self.raw)
            self.convergence_pi_threshold = cfg.get('convergence_pi_threshold', self.convergence_pi_threshold)
            self.input_noise_threshold_mult = cfg.get('input_noise_threshold_mult', self.input_noise_threshold_mult)
            self.output_noise_threshold_mult = cfg.get('output_noise_threshold_mult', self.output_noise_threshold_mult)
            self.n_consecutive_converged = cfg.get('n_consecutive_converged', self.n_consecutive_converged)
            self.max_gp_points = cfg.get('max_gp_points', self.max_gp_points)
            self.repulsion_lambda = cfg.get('repulsion_lambda', self.repulsion_lambda)
            self.acquisition_type = cfg.get('acquisition_type', self.acquisition_type)
            self.ucb_beta = cfg.get('ucb_beta', self.ucb_beta)
            self.nat_grad_step = cfg.get('nat_grad_step', self.nat_grad_step)
            self.nat_grad_max_steps = cfg.get('nat_grad_max_steps', self.nat_grad_max_steps)

        # Try new snapshot format first (latest.txt -> snapshots/)
        latest_path = self.run_dir / 'latest.txt'
        if latest_path.exists():
            return self._load_from_snapshot(latest_path.read_text().strip())

        # Fall back to old checkpoint format (current_state.txt -> states/)
        current_state_file = self.run_dir / 'current_state.txt'
        if current_state_file.exists():
            return self._load_from_old_checkpoint(current_state_file.read_text().strip())

        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    def _load_from_snapshot(self, snapshot_name: str) -> Tuple[int, int, int, int]:
        """Load state from a snapshot directory. Supports both delta and legacy formats."""
        snapshot_dir = self.run_dir / 'snapshots' / snapshot_name
        if not snapshot_dir.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        # DP delta format (new)
        if (snapshot_dir / 'delta.pt').exists():
            return self._load_from_delta_format(snapshot_name)

        # Legacy tensors.pt format
        tensors = torch.load(snapshot_dir / 'tensors.pt', map_location=self.device, weights_only=False)
        self._load_tensors(tensors)

        summary_path = snapshot_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            self.current_activation = summary.get('activation', 0)
            self.current_zoom = summary.get('zoom', 0)
            self.current_iteration = summary.get('iteration', 0)

        self._load_needles_json(snapshot_dir / 'needles.json')

        # Set snapshot count from existing snapshots
        snapshots_dir = self.run_dir / 'snapshots'
        if snapshots_dir.exists():
            existing = sorted(snapshots_dir.iterdir())
            if existing:
                try:
                    self._snapshot_count = int(existing[-1].name.split('_')[0])
                except ValueError:
                    self._snapshot_count = len(existing)

        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    def _load_from_delta_format(self, snapshot_name: str) -> Tuple[int, int, int, int]:
        """Reconstruct full state by replaying all delta snapshots up to snapshot_name."""
        snap_dir = self.run_dir / 'snapshots'
        all_snap_names = sorted(s.name for s in snap_dir.iterdir() if s.is_dir())

        X_parts: List[torch.Tensor] = []
        X_exp_parts: List[torch.Tensor] = []
        Y_parts: List[torch.Tensor] = []
        bh_parts: List[torch.Tensor] = []
        last_delta: Optional[dict] = None

        for name in all_snap_names:
            delta_path = snap_dir / name / 'delta.pt'
            if not delta_path.exists():
                continue  # skip legacy-format snapshots in mixed runs

            d = torch.load(str(delta_path), map_location=self.device, weights_only=False)

            x_new = d.get('X_new')
            if isinstance(x_new, torch.Tensor) and x_new.shape[0] > 0:
                X_parts.append(x_new.to(self.device, self.dtype))
                xe = d.get('X_exp_new', x_new)
                X_exp_parts.append(xe.to(self.device, self.dtype))
                y_new = d.get('Y_new')
                if isinstance(y_new, torch.Tensor):
                    Y_parts.append(y_new.to(self.device, self.dtype))

            bh_new = d.get('bounds_history_new')
            if isinstance(bh_new, torch.Tensor) and bh_new.shape[0] > 0:
                for i in range(bh_new.shape[0]):
                    bh_parts.append(bh_new[i].to(self.device, self.dtype))

            last_delta = d
            if name == snapshot_name:
                break

        if last_delta is None:
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        # Reconstruct accumulated tensors
        _d = self.d or 1
        if X_parts:
            self.X_all_actual = torch.cat(X_parts, dim=0)
            self.X_all_expected = (torch.cat(X_exp_parts, dim=0) if X_exp_parts
                                   else self.X_all_actual.clone())
            self.Y_all = (torch.cat(Y_parts, dim=0) if Y_parts
                          else torch.zeros(self.X_all_actual.shape[0], 1,
                                           device=self.device, dtype=self.dtype))
        else:
            self.X_all_actual = torch.zeros(0, _d, device=self.device, dtype=self.dtype)
            self.X_all_expected = torch.zeros(0, _d, device=self.device, dtype=self.dtype)
            self.Y_all = torch.zeros(0, 1, device=self.device, dtype=self.dtype)

        self.bounds_history = bh_parts

        # Restore X_init from first delta's n_init_points marker
        n_init = last_delta.get('n_init_points', 0)
        if n_init > 0 and self.X_all_actual.shape[0] >= n_init:
            self.X_init_actual = self.X_all_actual[:n_init].clone()
            self.X_init_expected = self.X_all_expected[:n_init].clone()
            self.Y_init = self.Y_all[:n_init].clone()

        if self.d is None and self.X_all_actual.shape[0] > 0:
            self.d = self.X_all_actual.shape[1]

        # Load non-accumulated (small, overwritten-each-snapshot) tensors
        self._load_tensors_from_delta(last_delta)

        # Update delta tracking for any future snapshots
        self._last_snap_n_points = self.X_all_actual.shape[0] if self.X_all_actual is not None else 0
        self._last_snap_n_bounds_history = len(self.bounds_history)

        # Restore snapshot counter
        existing = sorted(snap_dir.iterdir())
        if existing:
            try:
                self._snapshot_count = int(existing[-1].name.split('_')[0])
            except ValueError:
                self._snapshot_count = len(existing)

        # Restore log file handle
        if self.save_enabled and self.run_dir is not None:
            self._log_file = self.run_dir / 'run.log'

        # Load iteration state from summary
        summary_path = self.run_dir / 'snapshots' / snapshot_name / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            self.current_activation = summary.get('activation', 0)
            self.current_zoom = summary.get('zoom', 0)
            self.current_iteration = summary.get('iteration', 0)

        self._load_needles_json(self.run_dir / 'snapshots' / snapshot_name / 'needles.json')
        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    def _load_tensors_from_delta(self, d: dict):
        """Restore non-accumulated tensors (bounds, needles, penalty mask) from a delta dict."""
        def _t(key: str) -> Optional[torch.Tensor]:
            v = d.get(key)
            return v.to(device=self.device, dtype=self.dtype) if isinstance(v, torch.Tensor) else None

        _d = self.d or 1

        raw_bounds = _t('bounds')
        if raw_bounds is not None:
            self.bounds = raw_bounds
        else:
            self.bounds = torch.zeros(2, _d, device=self.device, dtype=self.dtype)
            self.bounds[1] = 1.0

        czb = _t('current_zoom_bounds')
        self.current_zoom_bounds = czb if czb is not None else self.bounds.clone()

        # Use explicit None-checks instead of `tensor or fallback` — the latter calls
        # bool(tensor) for short-circuit evaluation, which raises RuntimeError when the
        # tensor has 0 elements (e.g. no needles yet on first resume).
        _needles = _t('needles')
        self.needles = _needles if _needles is not None else torch.empty(0, _d, device=self.device, dtype=self.dtype)
        _needle_vals = _t('needle_vals')
        self.needle_vals = _needle_vals if _needle_vals is not None else torch.empty(0, 1, device=self.device, dtype=self.dtype)

        ni = d.get('needle_indices')
        self.needle_indices = (ni.to(device=self.device, dtype=torch.int64)
                               if isinstance(ni, torch.Tensor)
                               else torch.empty(0, 1, device=self.device, dtype=torch.int64))

        _npr = _t('needle_penalty_radii')
        self.needle_penalty_radii = _npr if _npr is not None else torch.empty(0, 1, device=self.device, dtype=self.dtype)

        needle_M_stack = d.get('needle_M_stack')
        needle_has_M = d.get('needle_has_M')
        if (isinstance(needle_M_stack, torch.Tensor) and isinstance(needle_has_M, torch.Tensor)
                and needle_has_M.shape[0] > 0):
            self.needle_M_list = [
                needle_M_stack[i].to(self.device, self.dtype) if needle_has_M[i].item() else None
                for i in range(len(needle_has_M))
            ]
        else:
            self.needle_M_list = []

        nb = d.get('needle_B')
        self.needle_B = nb.to(self.device, self.dtype) if isinstance(nb, torch.Tensor) else None

        pm = d.get('penalty_mask')
        self._penalty_mask = pm.to(self.device) if isinstance(pm, torch.Tensor) else None

        if self.top_m_points is None and self.d is not None:
            self.top_m_points = max(self.d + 1, 4)

    def _load_from_old_checkpoint(self, iteration_label: str) -> Tuple[int, int, int, int]:
        """Load state from old checkpoint format (states/ directory)."""
        state_dir = self.run_dir / 'states' / iteration_label
        if not state_dir.exists():
            return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

        tensors = torch.load(state_dir / 'tensors.pt', map_location=self.device)
        self._load_tensors(tensors)

        tracking_path = state_dir / 'tracking.json'
        if tracking_path.exists():
            with open(tracking_path) as f:
                tracking = json.load(f)
            self.current_activation = tracking['current_activation']
            self.current_zoom = tracking['current_zoom']
            self.current_iteration = tracking['current_iteration']
            self.no_improvements = tracking.get('no_improvements', 0)

        # State label is authoritative for position
        match = re.match(r'act(\d+)_zoom(\d+)_iter(\d+)', iteration_label)
        if match:
            self.current_activation = int(match.group(1))
            self.current_zoom = int(match.group(2))
            self.current_iteration = int(match.group(3))

        self._load_needles_json(state_dir / 'needles_results.json')

        # Rebuild checkpoint history
        states_dir = self.run_dir / 'states'
        if states_dir.exists():
            for state_subdir in sorted(states_dir.iterdir()):
                if state_subdir.is_dir():
                    label = state_subdir.name
                    is_permanent = any(keyword in label for keyword in
                                       ['init', 'needle', 'complete', 'finished', 'timeout', 'final'])
                    self.checkpoint_history.append((label, is_permanent))

        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    def _load_tensors(self, tensors: dict):
        """Load all tensors from a checkpoint dict."""
        raw_bounds = tensors.get('bounds')
        if raw_bounds is not None and isinstance(raw_bounds, torch.Tensor):
            self.bounds = raw_bounds.to(device=self.device, dtype=self.dtype)
        else:
            # Fallback: build full-simplex tensor bounds if old format or missing
            d_here = self.d or 3
            self.bounds = torch.zeros(2, d_here, device=self.device, dtype=self.dtype)
            self.bounds[1] = 1.0

        raw_czb = tensors.get('current_zoom_bounds')
        if raw_czb is not None and isinstance(raw_czb, torch.Tensor):
            self.current_zoom_bounds = raw_czb.to(device=self.device, dtype=self.dtype)
        else:
            self.current_zoom_bounds = self.bounds.clone()

        self.X_init_actual = tensors['X_init_actual'].to(device=self.device, dtype=self.dtype)
        self.X_init_expected = tensors['X_init_expected'].to(device=self.device, dtype=self.dtype)
        self.Y_init = tensors['Y_init'].to(device=self.device, dtype=self.dtype)
        self.X_all_actual = tensors['X_all_actual'].to(device=self.device, dtype=self.dtype)
        self.X_all_expected = tensors['X_all_expected'].to(device=self.device, dtype=self.dtype)
        self.Y_all = tensors['Y_all'].to(device=self.device, dtype=self.dtype)
        self.needles = tensors['needles'].to(device=self.device, dtype=self.dtype)
        self.needle_vals = tensors['needle_vals'].to(device=self.device, dtype=self.dtype)
        self.needle_indices = tensors['needle_indices'].to(device=self.device, dtype=torch.int64)
        self.needle_penalty_radii = tensors['needle_penalty_radii'].to(device=self.device, dtype=self.dtype)
        self._penalty_mask = tensors['penalty_mask'].to(device=self.device)

        # Restore ellipsoid data (absent in old checkpoints → safe defaults)
        needle_M_stack = tensors.get('needle_M_stack', None)
        needle_has_M = tensors.get('needle_has_M', None)
        if needle_M_stack is not None and needle_has_M is not None and needle_has_M.numel() > 0:
            needle_M_stack = needle_M_stack.to(device=self.device, dtype=self.dtype)
            needle_has_M = needle_has_M.to(device=self.device)
            self.needle_M_list = [
                needle_M_stack[i].clone() if needle_has_M[i].item() else None
                for i in range(needle_has_M.shape[0])
            ]
        else:
            self.needle_M_list = [None] * self.needles.shape[0]

        nb = tensors.get('needle_B', None)
        self.needle_B = nb.to(device=self.device, dtype=self.dtype) if nb is not None else None

        # Restore bounds_history (absent in old checkpoints → empty)
        bhs = tensors.get('bounds_history_stack', None)
        if bhs is not None and bhs.shape[0] > 0:
            self.bounds_history = [bhs[i].to(device=self.device, dtype=self.dtype) for i in range(bhs.shape[0])]
        else:
            self.bounds_history = []

    def _load_needles_json(self, path: Path):
        """Load needle results from a JSON file (handles both old and new key names)."""
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        self.needles_results = [
            {
                'point': torch.tensor(r['point'], device=self.device, dtype=self.dtype),
                'value': r['value'],
                'median_value': r.get('median_value', None),
                'zoom': r['zoom'],
                'iteration': r['iteration'],
                'n_points': r.get('n_points'),
            }
            for r in data
        ]

    # =========================================================================
    # Iteration state helpers (kept for backward compat with zombihop_exp.py)
    # =========================================================================

    def update_iteration_state(self, activation: int, zoom: int, iteration: int, no_improvements: int):
        """Update iteration tracking state."""
        self.current_activation = activation
        self.current_zoom = zoom
        self.current_iteration = iteration
        self.no_improvements = no_improvements

    def get_iteration_state(self) -> Tuple[int, int, int, int]:
        """Get current iteration state."""
        return self.current_activation, self.current_zoom, self.current_iteration, self.no_improvements

    # =========================================================================
    # Data management
    # =========================================================================

    def add_all_points(
        self,
        new_X_actual: torch.Tensor,
        new_X_expected: torch.Tensor,
        new_Y: torch.Tensor,
    ) -> torch.Tensor:
        """Add new observations. Returns penalty mask for the new points (True = not penalized)."""
        new_X_actual = new_X_actual.to(device=self.device, dtype=self.dtype)
        new_X_expected = new_X_expected.to(device=self.device, dtype=self.dtype)
        new_Y = new_Y.to(device=self.device, dtype=self.dtype)

        if new_Y.ndim == 1:
            new_Y = new_Y.unsqueeze(1)

        # Drop points that are near-identical to already-stored measurements.
        # This prevents device re-transmissions on resume from duplicating checkpoint data.
        # novel_mask tracks which of the original inputs are genuinely new.
        n_new = new_X_actual.shape[0]
        novel_mask = torch.ones(n_new, dtype=torch.bool, device=self.device)
        if self.X_all_actual.shape[0] > 0 and n_new > 0:
            dup_tol = 1e-5
            dists = torch.cdist(
                new_X_actual.to(dtype=torch.float64),
                self.X_all_actual.to(dtype=torch.float64),
            )  # (n_new, n_existing)
            y_diffs = torch.abs(
                new_Y.reshape(-1, 1).to(dtype=torch.float64)
                - self.Y_all.reshape(1, -1).to(dtype=torch.float64)
            )  # (n_new, n_existing)
            is_dup = (dists < dup_tol) & (y_diffs < dup_tol)
            novel_mask = ~is_dup.any(dim=1)  # (n_new,) True = not a duplicate
            if not novel_mask.any():
                if self.verbose:
                    print(f"  [DH.add] all {n_new} point(s) are duplicates of checkpoint data — skipping")
                return self._compute_penalty_mask(new_X_actual)
            if not novel_mask.all():
                n_dup = int((~novel_mask).sum().item())
                if self.verbose:
                    print(f"  [DH.add] skipping {n_dup} duplicate point(s) already in checkpoint")
                new_X_actual = new_X_actual[novel_mask]
                new_X_expected = new_X_expected[novel_mask]
                new_Y = new_Y[novel_mask]

        new_penalty_mask_novel = self._compute_penalty_mask(new_X_actual)
        # Build full-length mask: novel+unpenalized → True, duplicates → False
        if not novel_mask.all():
            full_penalty_mask = torch.zeros(n_new, dtype=torch.bool, device=self.device)
            full_penalty_mask[novel_mask] = new_penalty_mask_novel
            new_penalty_mask = full_penalty_mask
        else:
            new_penalty_mask = new_penalty_mask_novel

        # Append only the novel (non-duplicate) points to the stored data.
        self.X_all_actual = torch.cat([self.X_all_actual, new_X_actual], dim=0)
        self.X_all_expected = torch.cat([self.X_all_expected, new_X_expected], dim=0)
        self.Y_all = torch.cat([self.Y_all, new_Y], dim=0)
        self._penalty_mask = torch.cat([self._penalty_mask, new_penalty_mask_novel], dim=0)

        _t0 = time.time()
        self._update_pared(new_X_actual, new_Y)
        if self.verbose:
            n_pared = self.X_pared.shape[0] if self.X_pared is not None else 0
            print(f"  [DH.add] paring: {time.time()-_t0:.2f}s  n_pared={n_pared}  n_total={self.X_all_actual.shape[0]}")

        # Return full-length mask (size of original input): novel+unpenalized → True, duplicates → False
        return new_penalty_mask

    def get_all_points(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (X_all_actual, X_all_expected, Y_all)."""
        return self.X_all_actual, self.X_all_expected, self.Y_all

    def add_needle(
        self,
        needle: torch.Tensor,
        needle_value: float,
        needle_penalty_radius: float,
        zoom: int,
        iteration: int,
        M: Optional[torch.Tensor] = None,
        B: Optional[torch.Tensor] = None,
        needle_median_value: Optional[float] = None,
        reason: Optional[str] = None,
        activation: Optional[int] = None,
    ):
        """Record a discovered needle (local optimum) and update penalty mask.

        If M and B are provided (the tangent-space Hessian ellipsoid), they are
        stored and used for the ellipsoid penalty mask instead of the sphere.
        ``needle_median_value`` is the median Y of all raw observations within the
        paring spatial distance of the needle; stored alongside the peak value.
        """
        needle = needle.to(device=self.device, dtype=self.dtype)

        distances = torch.norm(self.X_all_actual - needle.unsqueeze(0), dim=1)
        global_idx = distances.argmin()

        self.needles = torch.cat([self.needles, needle.unsqueeze(0)], dim=0)
        self.needle_vals = torch.cat([
            self.needle_vals,
            torch.tensor([[needle_value]], device=self.device, dtype=self.dtype),
        ], dim=0)
        self.needle_indices = torch.cat([self.needle_indices, global_idx.reshape(1, 1)], dim=0)
        self.needle_penalty_radii = torch.cat([
            self.needle_penalty_radii,
            torch.tensor([[needle_penalty_radius]], device=self.device, dtype=self.dtype),
        ], dim=0)

        # Store ellipsoid (or None for sphere fallback)
        self.needle_M_list.append(M.to(device=self.device, dtype=self.dtype) if M is not None else None)
        if M is not None:
            if B is not None:
                self.needle_B = B.to(device=self.device, dtype=self.dtype)
            elif self.needle_B is None:
                self.needle_B = get_tangent_basis(self.d, self.device, self.dtype)

        if activation is None:
            activation = int(getattr(self, "current_activation", 0) or 0)
        self.needles_results.append({
            'point': needle.clone(),
            'value': needle_value,
            'median_value': needle_median_value,
            'activation': int(activation),
            'zoom': zoom,
            'iteration': iteration,
            'reason': reason,
        })

        self._update_penalty_mask()
        self._sync_needles_db()

    def _enable_needles_db(self, optimizing_dims: List[int],
                           path: Optional[str] = None) -> None:
        """Enable live mirroring of the needle list into sql/needles.db.

        ``optimizing_dims`` are the hardware column indices this run optimises
        (e.g. [0, 8, 9]); needle points are scattered into those columns of the
        full 10-wide composition row. Called once by the hardware run driver; a
        no-op mirror otherwise. Immediately syncs so any resume-loaded needles
        are reflected.
        """
        from .needles_db import DEFAULT_NEEDLES_DB_PATH
        self._needles_db_dims = [int(d) for d in optimizing_dims]
        self._needles_db_path = str(path) if path is not None else str(DEFAULT_NEEDLES_DB_PATH)
        # On resume, load_state() restores the needle tensors but not
        # needles_results (its JSON is legacy/never written), which drops each
        # needle's radial median. Backfill it from the checkpoint's needles.json
        # so radial values survive and future live adds stay index-aligned.
        self._backfill_needles_results_from_checkpoint()
        self._sync_needles_db()

    def _backfill_needles_results_from_checkpoint(self) -> None:
        """Rebuild needles_results from the run's needles.json when it is missing
        entries relative to the needle tensors (i.e. after a resume). No-op if
        already complete or if the checkpoint cannot be aligned."""
        n = self.needles.shape[0] if self.needles is not None else 0
        if n == 0 or len(self.needles_results) >= n:
            return
        try:
            from .needles_db import read_run_dir_needles
            recs = read_run_dir_needles(getattr(self, "run_dir", None))
        except Exception:
            recs = []
        if len(recs) < n:
            return  # can't safely align tensors ↔ records; leave as-is
        rebuilt: List[Dict[str, Any]] = []
        for i in range(n):
            r = recs[i] if isinstance(recs[i], dict) else {}
            rebuilt.append({
                'point': self.needles[i].clone(),
                'value': (float(self.needle_vals[i].item())
                          if self.needle_vals is not None and i < self.needle_vals.shape[0]
                          else r.get('value')),
                'median_value': r.get('median_value'),
                'zoom': r.get('zoom'),
                'iteration': r.get('iteration'),
                'reason': r.get('reason'),
            })
        self.needles_results = rebuilt

    def _needle_records_for_db(self) -> List[Dict[str, Any]]:
        """Build DB records from the needle tensors (authoritative for count,
        point and peak value), attaching the radial median from needles_results
        by index when available. Robust even if needles_results is incomplete."""
        n = self.needles.shape[0] if self.needles is not None else 0
        recs: List[Dict[str, Any]] = []
        for i in range(n):
            res = self.needles_results[i] if i < len(self.needles_results) else {}
            recs.append({
                'point': self.needles[i],
                'value': (float(self.needle_vals[i].item())
                          if self.needle_vals is not None and i < self.needle_vals.shape[0]
                          else res.get('value')),
                'median_value': res.get('median_value'),
            })
        return recs

    def _sync_needles_db(self) -> None:
        """Rewrite sql/needles.db from the current needle set (sorted best-first).

        No-op unless _enable_needles_db() has configured a path + dims. Never
        raises — a DB hiccup must not interrupt the optimizer.
        """
        if self._needles_db_path is None or self._needles_db_dims is None:
            return
        try:
            from .needles_db import write_needles
            write_needles(self._needle_records_for_db(), self._needles_db_dims,
                          path=self._needles_db_path)
        except Exception as e:      # noqa: BLE001 — mirroring must not be fatal
            try:
                print(f"[needles.db] sync failed: {e}")
            except Exception:
                pass

    def get_needle_locations(self) -> torch.Tensor:
        """Return (num_needles, d) tensor of needle locations."""
        return self.needles

    def get_needle_results(self) -> List[Dict[str, Any]]:
        """Return list of needle result dicts."""
        return self.needles_results

    def get_needles_and_penalty_radii(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (needles, penalty_radii) tensors."""
        return self.needles, self.needle_penalty_radii

    def get_needle_ellipsoids(self) -> Tuple[List[Optional[torch.Tensor]], Optional[torch.Tensor]]:
        """Return (needle_M_list, needle_B) for use in RepulsiveAcquisition."""
        return self.needle_M_list, self.needle_B

    def get_all_needle_results(self) -> List[Dict[str, Any]]:
        return self.needles_results

    def get_all_needle_locations(self) -> torch.Tensor:
        """Return (n_total, d) tensor of all needle locations."""
        if self.needles is not None and self.needles.shape[0] > 0:
            return self.needles
        return torch.empty((0, self.d), device=self.device, dtype=self.dtype)

    def get_all_needle_vals(self) -> torch.Tensor:
        """Return (n_total, 1) tensor of all needle values."""
        if self.needle_vals is not None and self.needle_vals.shape[0] > 0:
            return self.needle_vals
        return torch.empty((0, 1), device=self.device, dtype=self.dtype)

    # =========================================================================
    # Penalty mask
    # =========================================================================

    def get_penalty_mask(self, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return penalty mask (True = not penalized). If X is None, returns mask for all stored points."""
        if X is None:
            return self._penalty_mask
        return self._compute_penalty_mask(X)

    def _compute_penalty_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Compute penalty mask for given points. True = not inside any penalty region.

        For needles with an ellipsoid (M, B): uses u^T M u <= 1 membership test.
        For needles with only a radius: uses Euclidean sphere test.
        """
        is_2d = X.ndim == 2
        if is_2d:
            n = X.shape[0]
            X_flat = X  # (n, d)
        elif X.ndim == 3:
            n, l, d_x = X.shape
            X_flat = X.reshape(-1, d_x)
        else:
            raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

        num_pts = X_flat.shape[0]
        penalized = torch.zeros(num_pts, dtype=torch.bool, device=X.device)

        # --- Regular needles ---
        if self.needles is not None and self.needles.shape[0] > 0:
            for idx in range(self.needles.shape[0]):
                needle = self.needles[idx]  # (d,)
                diff = X_flat - needle.unsqueeze(0)  # (num_pts, d)

                M = self.needle_M_list[idx] if idx < len(self.needle_M_list) else None
                if M is not None and self.needle_B is not None:
                    u = diff @ self.needle_B
                    quad = (u @ M * u).sum(dim=-1)
                    inside = quad <= 1.0
                else:
                    r = self.needle_penalty_radii[idx].squeeze()
                    inside = torch.norm(diff, dim=-1) <= r
                penalized = penalized | inside

        if not is_2d:
            penalized = penalized.reshape(n, l)

        return ~penalized

    def _update_penalty_mask(self):
        """Recompute cached penalty mask for all stored points and purge pared set.

        Called after every needle addition, ellipsoid refit, or shrink so that:
          • _penalty_mask is always current for X_all_actual, and
          • X_pared never contains penalized points (which would corrupt GP
            training and the deduplication logic in _update_pared).
        """
        if self.X_all_actual is not None and self.X_all_actual.shape[0] > 0:
            self._penalty_mask = self._compute_penalty_mask(self.X_all_actual)
        else:
            self._penalty_mask = torch.ones(0, dtype=torch.bool, device=self.device)

        # Purge any penalized points from the pared set. These arise when a
        # needle ellipsoid is added or refitted after the pared points were
        # recorded (they were unpenalized at the time but are covered now).
        if self.X_pared is not None and self.X_pared.shape[0] > 0:
            pared_keep = self._compute_penalty_mask(self.X_pared)
            if not pared_keep.all():
                if pared_keep.any():
                    self.X_pared = self.X_pared[pared_keep]
                    self.Y_pared = self.Y_pared[pared_keep]
                else:
                    self.X_pared = torch.empty((0, self.d), device=self.device, dtype=self.dtype)
                    self.Y_pared = torch.empty((0, 1), device=self.device, dtype=self.dtype)
                self._pared_version += 1

    # =========================================================================
    # GP helpers
    # =========================================================================

    def get_gp_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (X, Y) for GP fitting using the pared (noise-deduplicated) dataset.

        Applies the penalty mask to the pared set.  Falls back to the full raw
        dataset when the pared set is empty or every pared point is penalized.
        """
        use_pared = self.X_pared is not None and self.X_pared.shape[0] > 0
        X_src = self.X_pared if use_pared else self.X_all_actual
        Y_src = self.Y_pared if use_pared else self.Y_all

        mask = self.get_penalty_mask(X_src)
        if mask.any():
            X = X_src[mask]
            Y = Y_src[mask]
        else:
            # Pared set is empty or fully penalized — fall back to raw unpenalized data.
            raw_mask = self._penalty_mask
            if raw_mask.any():
                X = self.X_all_actual[raw_mask]
                Y = self.Y_all[raw_mask]
            else:
                # Absolute last resort: no unpenalized data at all.
                # Use init data (predates all needles) to keep the GP alive.
                X = self.X_init_actual if self.X_init_actual is not None else self.X_all_actual
                Y = self.Y_init if self.Y_init is not None else self.Y_all

        sorted_idx = torch.argsort(Y.reshape(-1), descending=True)
        n = min(self.max_gp_points, len(sorted_idx))
        top_idx = sorted_idx[:n]
        return X[top_idx], Y[top_idx]

    def get_zoom_gp_data(self, bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Local variant of get_gp_data() restricted to points within bounds.

        Budget (max_gp_points) filled in two passes:

        1. Unpenalized pared points within bounds (deduplicated, sorted by Y desc).
           These give the GP a clean, noise-deduped picture of the local landscape.
        2. Remaining budget: unpenalized raw points within bounds (sorted by Y desc).
           These add higher-density local signal the pared pass may have thinned out.

        Falls back to get_gp_data() when fewer than 2 local points are found.
        """
        lo, hi = bounds[0], bounds[1]

        def _in_bounds(X: torch.Tensor) -> torch.Tensor:
            return ((X >= lo.unsqueeze(0)) & (X <= hi.unsqueeze(0))).all(dim=1)

        X_parts: List[torch.Tensor] = []
        Y_parts: List[torch.Tensor] = []
        n_used = 0

        # Pass 1: pared points within bounds (unpenalized, sorted by Y desc)
        if self.X_pared is not None and self.X_pared.shape[0] > 0:
            pared_keep = self.get_penalty_mask(self.X_pared) & _in_bounds(self.X_pared)
            if pared_keep.any():
                Xp = self.X_pared[pared_keep]
                Yp = self.Y_pared[pared_keep]
                order = Yp.reshape(-1).argsort(descending=True)
                n = min(self.max_gp_points, order.shape[0])
                X_parts.append(Xp[order[:n]])
                Y_parts.append(Yp[order[:n]])
                n_used = n

        # Pass 2: fill remaining budget with raw unpenalized within bounds
        remaining = self.max_gp_points - n_used
        if remaining > 0:
            raw_keep = self._penalty_mask & _in_bounds(self.X_all_actual)
            if raw_keep.any():
                Xr = self.X_all_actual[raw_keep]
                Yr = self.Y_all[raw_keep]
                order = Yr.reshape(-1).argsort(descending=True)
                n = min(remaining, order.shape[0])
                X_parts.append(Xr[order[:n]])
                Y_parts.append(Yr[order[:n]])

        if not X_parts:
            return self.get_gp_data()

        X_out = torch.cat(X_parts, dim=0)
        Y_out = torch.cat(Y_parts, dim=0)

        if X_out.shape[0] < 2:
            return self.get_gp_data()

        return X_out, Y_out

    def get_best_in_bounds(
        self, bounds: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
        """Return (X_best, Y_best, global_index) of best unpenalized point within bounds.

        Falls back to get_best_unpenalized() if no unpenalized points lie within bounds.
        """
        if self._penalty_mask is None or not self._penalty_mask.any():
            return None, None, None

        lo, hi = bounds[0], bounds[1]
        unpen_idx = torch.where(self._penalty_mask)[0]
        X_unpen = self.X_all_actual[unpen_idx]
        Y_unpen = self.Y_all[unpen_idx]

        in_b = ((X_unpen >= lo.unsqueeze(0)) & (X_unpen <= hi.unsqueeze(0))).all(dim=1)
        if not in_b.any():
            return self.get_best_unpenalized()

        local_best = Y_unpen[in_b].reshape(-1).argmax()
        global_idx = unpen_idx[in_b][local_best]
        return self.X_all_actual[global_idx], self.Y_all[global_idx], global_idx.item()

    def _relabel_pared_with_medians(self) -> None:
        """Replace each pared Y with the median of all Y_all values whose X_all
        lies within paring_spatial_halfnoise * input_noise in composition L2.

        Smooths noise spikes so the GP always trains on a clean, representative
        signal.  Called once after each activation completes.
        """
        if self.X_pared is None or self.X_pared.shape[0] == 0:
            return
        if self.X_all_actual is None or self.X_all_actual.shape[0] == 0:
            return

        # Zoom-scaled paring radius (floored), matching _update_pared so relabelling
        # smooths over the same neighbourhood the dedup used. See _zoom_size_factor.
        thresh = max(
            self.paring_spatial_halfnoise * self.input_noise * self._zoom_size_factor(),
            PARE_DIST_FLOOR,
        )
        Y_new = self.Y_pared.clone()
        changed = False
        _t0 = time.time()
        for i in range(self.X_pared.shape[0]):
            dists = torch.norm(self.X_all_actual - self.X_pared[i].unsqueeze(0), dim=1)
            nearby = dists <= thresh
            if nearby.any():
                median_y = self.Y_all[nearby].reshape(-1).median()
                if not torch.isclose(median_y, Y_new[i, 0]):
                    Y_new[i, 0] = median_y
                    changed = True

        if changed:
            self.Y_pared = Y_new
            self._pared_version += 1

        if self.verbose:
            n_pared = self.X_pared.shape[0]
            n_all = self.X_all_actual.shape[0]
            print(f"  [DH.relabel] {time.time()-_t0:.2f}s  n_pared={n_pared}  n_all={n_all}  changed={changed}")

    def update_gp_noise(self, sigma_y: float) -> None:
        """Store the GP's fitted output noise estimate for use in point paring."""
        self._gp_output_noise = float(sigma_y)

    def get_pared_hash(self) -> int:
        """Integer version counter that increments whenever the pared dataset changes."""
        return self._pared_version

    def _zoom_size_factor(self) -> float:
        """Linear size of the current zoom zone relative to the full domain, in (0,1].

        Geometric mean of the per-axis extent ratios of ``current_zoom_bounds`` vs
        the global domain captured at ``initialize()`` (1.0 at the full domain). The
        point-paring radius is multiplied by this so points densely packed inside a
        zoom zone aren't deduplicated away — matching the zoom-scaled dup metric.
        Falls back to 1.0 (no scaling) before any bounds are set, and assumes the
        unit box if no full-domain reference was captured (e.g. a resumed run).
        Mirrors ``eval_metrics.zoom_size_fraction``.
        """
        czb = self.current_zoom_bounds if self.current_zoom_bounds is not None else self.bounds
        if czb is None:
            return 1.0
        ext = (czb[1] - czb[0]).clamp(min=0.0)
        full = self._full_extent if self._full_extent is not None else torch.ones_like(ext)
        ratio = (ext / full).clamp(min=1e-12, max=1.0)
        return float(torch.exp(torch.log(ratio).mean()).item())

    def _update_pared(self, X_new: torch.Tensor, Y_new: torch.Tensor) -> None:
        """Incrementally add new UNPENALIZED points to the pared (noise-deduplicated) dataset.

        For each new point, if a near-duplicate already exists in the pared set
        (within half the composition noise distance AND within one output-noise unit),
        we flip a fair coin: keep the old point or replace it with the new one.
        Equal probability avoids biasing toward noise spikes.
        If no near-duplicate exists, the point is added unconditionally.

        Penalized points are never added — they must not influence GP training
        or corrupt deduplication for unpenalized points near ellipsoid boundaries.

        Distances are computed in composition L2 to match the GP's ambient metric.
        """
        if X_new.shape[0] == 0:
            return

        X_new = X_new.to(device=self.device, dtype=self.dtype)
        Y_new = Y_new.to(device=self.device, dtype=self.dtype)
        if Y_new.ndim == 1:
            Y_new = Y_new.unsqueeze(1)

        # Drop penalized points before they can pollute the pared set.
        unpen = self._compute_penalty_mask(X_new)
        if not unpen.any():
            return
        if not unpen.all():
            X_new = X_new[unpen]
            Y_new = Y_new[unpen]

        # Paring thresholds (both in their natural units after normalization). The
        # spatial radius is scaled down by the current zoom-zone linear size (floored
        # at PARE_DIST_FLOOR) so points packed inside a zoom aren't deduplicated
        # away; at the full domain the factor is 1.0 (unchanged). See
        # _zoom_size_factor.
        spatial_thresh = max(
            self.paring_spatial_halfnoise * self.input_noise * self._zoom_size_factor(),
            PARE_DIST_FLOOR,
        )
        y_thresh = self.paring_y_noise_multiplier * max(self._gp_output_noise, 1e-6)

        d = X_new.shape[1]
        min_pts = max(2 * (d - 1), 5)

        for i in range(X_new.shape[0]):
            x_i = X_new[i]
            y_i = Y_new[i, 0].item()

            if self.X_pared is None or self.X_pared.shape[0] == 0:
                self.X_pared = X_new[i:i+1].clone()
                self.Y_pared = Y_new[i:i+1].clone()
                self._pared_version += 1
                continue

            dists = torch.norm(self.X_pared - x_i.unsqueeze(0), dim=1)
            y_diffs = torch.abs(self.Y_pared.reshape(-1) - y_i)
            is_dup = (dists < spatial_thresh) & (y_diffs < y_thresh)

            if is_dup.any() and self.X_pared.shape[0] >= min_pts:
                # Fair coin: discard new (heads) or replace oldest duplicate (tails)
                if torch.rand(1).item() >= 0.5:
                    dup_idx = torch.where(is_dup)[0][0]
                    self.X_pared[dup_idx] = X_new[i]
                    self.Y_pared[dup_idx] = Y_new[i]
                    self._pared_version += 1
                # else: discard new — pared set unchanged
            else:
                self.X_pared = torch.cat([self.X_pared, X_new[i:i+1]], dim=0)
                self.Y_pared = torch.cat([self.Y_pared, Y_new[i:i+1]], dim=0)
                self._pared_version += 1

    def get_best_unpenalized(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
        """Return (X_best, Y_best, global_index) for the best unpenalized point."""
        if not self._penalty_mask.any():
            return None, None, None
        Y_masked = self.Y_all[self._penalty_mask]
        max_idx = Y_masked.argmax()
        unpenalized_indices = torch.where(self._penalty_mask)[0]
        global_idx = unpenalized_indices[max_idx]
        return self.X_all_actual[global_idx], self.Y_all[global_idx], global_idx.item()

    @staticmethod
    def _jaccard_box(a: torch.Tensor, b: torch.Tensor) -> float:
        """Volume-Jaccard of two (2, d) axis-aligned boxes.

        Dimensions where *both* boxes are zero-width are projected out (factor 1).
        """
        a_np = a.cpu().float().numpy()
        b_np = b.cpu().float().numpy()
        w_a = a_np[1] - a_np[0]
        w_b = b_np[1] - b_np[0]
        both_zero = (w_a < 1e-12) & (w_b < 1e-12)
        active = ~both_zero
        if not active.any():
            return 1.0
        lo_i = np.maximum(a_np[0, active], b_np[0, active])
        hi_i = np.minimum(a_np[1, active], b_np[1, active])
        inter_w = np.maximum(0.0, hi_i - lo_i)
        vol_inter = float(np.prod(inter_w))
        vol_a = float(np.prod(np.maximum(w_a[active], 1e-30)))
        vol_b = float(np.prod(np.maximum(w_b[active], 1e-30)))
        vol_union = vol_a + vol_b - vol_inter
        return 0.0 if vol_union < 1e-30 else vol_inter / vol_union

    def determine_new_bounds(self, add_to_history: bool = True) -> torch.Tensor:
        """Compute new bounds via a Jaccard-aware sliding window over unpenalized points.

        Slides a window of top_m_points along the Y-sorted unpenalized set.  For
        each candidate AABB, computes the max Jaccard overlap against the last
        jaccard_window entries in bounds_history.  Returns the first window whose
        max Jaccard ≤ jaccard_threshold; if none qualifies, returns the least-
        similar window found.  Appends to bounds_history only when
        add_to_history=True.
        """
        if not self._penalty_mask.any():
            return self.bounds.clone()

        Y_masked = self.Y_all[self._penalty_mask].reshape(-1)
        X_masked = self.X_all_actual[self._penalty_mask]
        n = Y_masked.numel()
        k = min(self.top_m_points, n)
        if k == 0:
            return self.bounds.clone()

        order = torch.argsort(Y_masked, descending=True)
        X_sorted = X_masked[order]

        recent = self.bounds_history[-self.jaccard_window:]

        best_bounds: Optional[torch.Tensor] = None
        best_max_jac = float('inf')

        max_start = max(n - k, 0)
        for start in range(max_start + 1):
            window = X_sorted[start:start + k]
            cand = torch.stack([window.min(dim=0).values, window.max(dim=0).values], dim=0)
            max_jac = 0.0
            for prev in recent:
                jac = self._jaccard_box(cand, prev)
                if jac > max_jac:
                    max_jac = jac
            if max_jac <= self.jaccard_threshold:
                if add_to_history:
                    self.bounds_history.append(cand.clone())
                    if len(self.bounds_history) > self.jaccard_window:
                        self.bounds_history.pop(0)
                return cand
            if max_jac < best_max_jac:
                best_max_jac = max_jac
                best_bounds = cand.clone()

        result = best_bounds if best_bounds is not None else self.bounds.clone()
        if add_to_history:
            self.bounds_history.append(result.clone())
            if len(self.bounds_history) > self.jaccard_window:
                self.bounds_history.pop(0)
        return result

    # =========================================================================
    # Needle radius helpers
    # =========================================================================

    def update_all_needle_radii(self, new_M_list: list) -> None:
        """Replace needle_M_list with new_M_list and recompute the penalty mask."""
        assert len(new_M_list) == len(self.needle_M_list), (
            f"Length mismatch: {len(new_M_list)} vs {len(self.needle_M_list)}"
        )
        for i, M in enumerate(new_M_list):
            self.needle_M_list[i] = M
        self._update_penalty_mask()

    def shrink_all_needle_radii(self, factor: float) -> None:
        """Shrink every needle's exclusion semi-axes by *factor* (0 < factor ≤ 1).

        Semi-axis_new = factor × semi-axis_old  ⟺  M_new = M_old / factor².
        """
        f2 = factor ** 2
        for i, M in enumerate(self.needle_M_list):
            if M is not None:
                self.needle_M_list[i] = M / f2
        self._update_penalty_mask()

    def max_needle_radius(self) -> float:
        """Return the largest semi-axis (in tangent/composition units) across all active needles."""
        max_r = 0.0
        for M in self.needle_M_list:
            if M is None:
                continue
            min_eig = torch.linalg.eigvalsh(M).clamp(min=1e-30).min()
            r = (1.0 / min_eig.sqrt()).item()
            if r > max_r:
                max_r = r
        return max_r

    # =========================================================================
    # Input noise helpers
    # =========================================================================

    def _paired_X_actual_expected_2d(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(n, d)`` copies suitable for batch norms (handles legacy 1D rows)."""
        xa = self.X_all_actual
        xe = self.X_all_expected
        if xa is None or xe is None:
            raise RuntimeError("X_all_actual / X_all_expected is not set.")
        if xa.ndim == 2 and xe.ndim == 2:
            return xa, xe
        assert self.d is not None and self.d > 0, "cannot reshape X tensors without known d."
        rem_a, rem_e = xa.numel() % self.d, xe.numel() % self.d
        if rem_a != 0 or rem_e != 0:
            raise ValueError(
                "X_actual / X_expected numel incompatible with dimension d: "
                f"xa {tuple(xa.shape)}, xe {tuple(xe.shape)}, d={self.d}"
            )
        return xa.reshape(-1, self.d), xe.reshape(-1, self.d)

    def get_normalized_input_noise(self) -> float:
        """Median normalized distance between expected and actual points."""
        if (
            self.X_all_expected is None
            or self.X_all_actual is None
            or self.X_all_expected.numel() == 0
        ):
            return 0.0
        xa, xe = self._paired_X_actual_expected_2d()
        if xa.shape[0] == 0:
            return 0.0
        distances = torch.norm(xe - xa, dim=1)
        normalized = distances / torch.sqrt(torch.tensor(self.d, dtype=self.dtype, device=self.device))
        return torch.median(normalized).item()

    def get_input_noise(self) -> float:
        """Median distance between expected and actual points."""
        if (
            self.X_all_expected is None
            or self.X_all_actual is None
            or self.X_all_expected.numel() == 0
        ):
            return 0.0
        xa, xe = self._paired_X_actual_expected_2d()
        if xa.shape[0] == 0:
            return 0.0
        distances = torch.norm(xe - xa, dim=1)
        return torch.median(distances).item()


# =============================================================================
# Module-level helpers
# =============================================================================

def reconstruct_snapshot_tensors(run_dir: Path, snapshot_name: str, device: str = "cpu") -> dict:
    """
    Reconstruct a full tensor dict from delta snapshots (or legacy tensors.pt).

    Returns a dict with the same keys as the old ``tensors.pt`` format so that
    display code (app.py) works without changes.  Falls back transparently for
    runs that were saved with the legacy full-copy format.
    """
    snap_dir = run_dir / "snapshots"
    target_delta = snap_dir / snapshot_name / "delta.pt"

    if not target_delta.exists():
        # Legacy format — load tensors.pt directly
        old_path = snap_dir / snapshot_name / "tensors.pt"
        if old_path.exists():
            return torch.load(str(old_path), map_location=device, weights_only=False)
        return {}

    # Delta format: replay all deltas from the beginning up to snapshot_name
    all_snap_dirs = sorted(s for s in snap_dir.iterdir() if s.is_dir())

    X_parts: list = []
    X_exp_parts: list = []
    Y_parts: list = []
    bh_parts: list = []
    last_d: Optional[dict] = None

    for sdir in all_snap_dirs:
        delta_path = sdir / "delta.pt"
        if not delta_path.exists():
            continue

        d = torch.load(str(delta_path), map_location=device, weights_only=False)

        x_new = d.get("X_new")
        if isinstance(x_new, torch.Tensor) and x_new.shape[0] > 0:
            X_parts.append(x_new)
            xe = d.get("X_exp_new", x_new)
            X_exp_parts.append(xe if isinstance(xe, torch.Tensor) else x_new)
            y_new = d.get("Y_new")
            if isinstance(y_new, torch.Tensor):
                Y_parts.append(y_new)

        bh_new = d.get("bounds_history_new")
        if isinstance(bh_new, torch.Tensor) and bh_new.shape[0] > 0:
            bh_parts.append(bh_new)

        last_d = d
        if sdir.name == snapshot_name:
            break

    if last_d is None:
        return {}

    X_all = torch.cat(X_parts, dim=0) if X_parts else torch.zeros(0, 1)
    X_exp_all = torch.cat(X_exp_parts, dim=0) if X_exp_parts else X_all.clone()
    Y_all = (torch.cat(Y_parts, dim=0) if Y_parts
             else torch.zeros(X_all.shape[0], 1))
    bh_stack = (torch.cat(bh_parts, dim=0) if bh_parts
                else torch.zeros(0, 2, max(X_all.shape[-1], 1)))

    n_init = last_d.get("n_init_points", 0)

    return {
        # Reconstructed growing arrays
        "X_all_actual": X_all,
        "X_all_expected": X_exp_all,
        "Y_all": Y_all,
        "X_init_actual": X_all[:n_init] if n_init > 0 else X_all[:0],
        "X_init_expected": X_exp_all[:n_init] if n_init > 0 else X_exp_all[:0],
        "Y_init": Y_all[:n_init] if n_init > 0 else Y_all[:0],
        # Small tensors — taken from last delta
        "bounds": last_d.get("bounds"),
        "current_zoom_bounds": last_d.get("current_zoom_bounds"),
        "needles": last_d.get("needles"),
        "needle_vals": last_d.get("needle_vals"),
        "needle_indices": last_d.get("needle_indices"),
        "needle_penalty_radii": last_d.get("needle_penalty_radii"),
        "needle_M_stack": last_d.get("needle_M_stack"),
        "needle_has_M": last_d.get("needle_has_M"),
        "needle_B": last_d.get("needle_B"),
        "penalty_mask": last_d.get("penalty_mask"),
        "bounds_history_stack": bh_stack,
    }
