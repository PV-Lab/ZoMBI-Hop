"""
ZoMBI-Hop: Zooming Multi-Basin Identification with Hopping
==========================================================

A novel Bayesian optimization algorithm for discovering multiple optima
in simplex-constrained spaces, designed for materials research applications.
"""

import threading
import torch
import time
from typing import Callable, Tuple, Optional, List, Any

from ..utils.simplex import (
    proj_simplex,
    random_simplex,
    random_zero_sum_directions,
    Ellipsoid,
)
from ..utils.datahandler import DataHandler
from ..utils.gp_simplex import GPSimplex
from .hparam_live import (
    apply_pending as apply_pending_hparams,
    write_effective as write_effective_hparams,
)
from . import retro


# --- CUDA optimization settings (when CUDA is available) ---
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)


def _is_global_bounds(bounds: torch.Tensor, eps: float = 0.01) -> bool:
    """True when bounds essentially span the full [0,1]^d simplex.

    Module-level convenience for the default full simplex. Inside ``ZoMBIHop`` use
    ``self._is_global_bounds`` instead — it compares against the run's configured
    search box (``self.full_bounds``), which may be tighter than [0,1]^d."""
    return bounds[0].max().item() < eps and bounds[1].min().item() > 1.0 - eps


class ZoMBIHop:
    """
    Zooming Multi-Basin Identification with Hopping.

    Trust regions are ellipsoids on the simplex in tangent space (see ``Ellipsoid``).

    Search loop (basin flood-fill formulation)
    ------------------------------------------
    An activation is a flat sequence of measured LineBO lines — there is no longer a
    zoom loop nested inside an iteration loop, because zoom depth is no longer a
    budget to spend but a consequence of what the data says. After every line:

    1. **Sample a line.** Unchanged: the acquisition proposes a point, LineBO turns
       it into a line, the objective measures ``NUM_EXPERIMENTS`` points along it.
    2. **Flood-fill the current basin** (:meth:`_flood_fill_basin`). Nodes are the
       unpenalised measured points inside the active box; edges are a symmetric
       k-nearest-neighbour graph (``flood_k``), which keeps the basin connected
       across LineBO's line-clustered geometry where a fixed radius would not.
       Breadth-first from the best measured point, a neighbour joins the basin when
       its best case clears the incumbent's worst case —
       ``UCB(p) >= LCB(best)`` at ``flood_ci_z`` posterior standard deviations. The
       inequality is what confines the fill to ONE basin: crossing a valley means
       crossing points whose optimistic value cannot reach the incumbent's
       pessimistic one, and the fill stops there.
    3. **Zoom in when the basin is small.** The basin's convex hull is taken as its
       axis-aligned bounding box (every search box downstream is an AABB), floored
       at the input-noise resolution. When that box's volume is
       ``zoom_volume_fraction`` (0.5) or less of the active box's, the search zooms
       into it. Once a basin fills its own box the ratio approaches 1 and zooming
       stops on its own, so nothing caps the depth.
    4. **Declare a needle when the optimum stops moving.** Two stability tests, both
       measured against the previous line: the best point has moved no more than
       ``needle_move_tol`` in composition L2 (0.10 = ten composition points), and
       the bootstrap CI half-width of the MEDIAN of the replicate measurements
       around it (:meth:`_median_ci_halfwidth`) has changed by no more than
       ``needle_ci_tol`` relatively. Both together mean the optimum's location and
       the confidence in its value have both settled: the needle is declared there,
       the region is penalised, and the search zooms back out to the full box for
       the next activation.

    ``max_lines_per_activation`` is the only budget an activation has. On reaching it
    without a needle the region is recorded as an exclusion zone
    (:meth:`_penalize_capped_zone`) so the next activation is repelled from it.

    Retired hyperparameters
    -----------------------
    The following constructor arguments are ACCEPTED AND IGNORED. They belonged to
    the EI-convergence / Jaccard-window formulation this loop replaces, and are kept
    only so existing callers and stored configs still construct: ``max_zooms``,
    ``max_iterations``, ``top_m_points``, ``convergence_pi_threshold``,
    ``output_noise_threshold_mult``, ``n_consecutive_converged``,
    ``needle_shrink_factor``, ``needle_stop_noise_multiplier``,
    ``zoom_jaccard_threshold``, ``jaccard_window``, ``jaccard_threshold``,
    ``min_zoom_for_needle``, ``min_iters_per_zoom``. They are gone from
    ``optimize/hparam_space.HPARAM_SPACE``, so nothing tunes them any more.
    """

    def __init__(self,
                 objective,
                 X_init_actual: torch.Tensor,
                 X_init_expected: torch.Tensor,
                 Y_init: torch.Tensor,
                 proj_fn: Optional[Callable] = None,
                 random_sampler: Optional[Callable] = None,
                 random_direction_sampler: Optional[Callable] = None,
                 max_zooms: int = 3,             # RETIRED — see class docstring
                 max_iterations: int = 10,       # RETIRED — see class docstring
                 top_m_points: Optional[int] = None,  # RETIRED
                 n_restarts: int = 30,
                 raw: int = 500,
                 convergence_pi_threshold: float = 0.01,  # RETIRED
                 input_noise_threshold_mult: float = 3.0,
                 output_noise_threshold_mult: float = 2.0,  # RETIRED
                 n_consecutive_converged: int = 2,          # RETIRED
                 max_gp_points: int = 3000,
                 repulsion_lambda: Optional[float] = None,
                 acquisition_type: str = "ucb",
                 ucb_beta: float = 0.1,
                 nat_grad_step: float = 0.02,
                 nat_grad_max_steps: int = 50,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float64,
                 bounds: Optional[torch.Tensor] = None,
                 run_uuid: Optional[str] = None,
                 resume: Optional[bool] = None,
                 checkpoint_dir: Optional[str] = 'zombihop_checkpoints',
                 num_iterations_saved: int = 50,
                 max_snapshots: Optional[int] = None,
                 verbose: bool = True,
                 needle_plot_points_ref: Optional[List[Any]] = None,
                 ellipsoid_drop_fraction: float = 0.25,
                 ellipsoid_eigenvalue_floor: float = 1e-6,
                 max_penalty_radius: float = 1.0,
                 paring_spatial_halfnoise: float = 0.5,
                 paring_y_noise_multiplier: float = 1.0,
                 input_noise: float = 0.128,
                 input_noise_ilr: Optional[float] = None,
                 needle_shrink_factor: float = 0.85,          # RETIRED
                 needle_stop_noise_multiplier: float = 3.0,   # RETIRED
                 zoom_jaccard_threshold: float = 0.75,        # RETIRED
                 bounds_shrink_factor: float = 0.8,
                 min_axis_noise_mult: float = 2.0,
                 jaccard_window: int = 3,                     # RETIRED
                 jaccard_threshold: float = 0.9,              # RETIRED
                 min_zoom_for_needle: int = 1,                # RETIRED
                 min_iters_per_zoom: int = 3,                 # RETIRED
                 max_lines_per_activation: int = 30,
                 flood_ci_z: float = 2.0,
                 flood_k: int = 6,
                 zoom_volume_fraction: float = 0.5,
                 needle_move_tol: float = 0.10,
                 needle_ci_tol: float = 0.15,
                 needle_ci_bootstrap: int = 200):
        """Initialize ZoMBIHop optimizer.

        Basin parameters (see the class docstring for how the loop uses them):
          * ``flood_ci_z`` — posterior standard deviations used for the flood
            fill's UCB/LCB. A point joins the current basin when
            ``mean(p) + z·sd(p) >= mean(best) - z·sd(best)``; larger z is a more
            permissive (larger) basin.
          * ``flood_k`` — degree of the symmetric kNN graph the fill spreads over.
            Must exceed the number of lines the basin should be able to bridge;
            too small disconnects one LineBO line from the next, too large lets
            the fill jump a valley in one hop.
          * ``zoom_volume_fraction`` — the basin box must be this fraction of the
            active box's VOLUME or less before the search zooms into it.
          * ``needle_move_tol`` — how far, in composition L2 (absolute, NOT
            relative to the box), the best point may move between consecutive
            lines and still count as stationary. 0.10 = ten composition points.
          * ``needle_ci_tol`` — the allowed RELATIVE change, between consecutive
            lines, in the bootstrap CI half-width of the median at the best point.
          * ``needle_ci_bootstrap`` — resamples behind that CI.

        Search-discipline constraint (hard, not tuned):
          * ``max_lines_per_activation`` — hard ceiling on objective lines
            measured in one activation (default 30), and now the ONLY budget an
            activation has: zoom depth is decided by the volume test rather than
            by ``max_zooms × max_iterations``, so nothing else bounds how long an
            activation can grind on one region. On reaching the cap the activation
            ends and the region it was stuck in is penalised (see
            ``_penalize_capped_zone``) so the next activation is repelled from it
            rather than converging straight back in.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose

        d = X_init_actual.shape[1]

        # Per-dimension search box (2, d): row 0 = lower, row 1 = upper. Defaults to
        # the full [0,1]^d simplex, but callers may pass a tighter box (e.g. a dim
        # constrained to [0, 0.3]). Every "reset to the full simplex" throughout the
        # optimiser resets to THIS box, and _is_global_bounds tests against it — so a
        # tightened box is the true global region, not [0,1]^d.
        if bounds is None:
            full_bounds = torch.zeros(2, d, device=self.device, dtype=self.dtype)
            full_bounds[1] = 1.0
        else:
            full_bounds = torch.as_tensor(bounds, device=self.device, dtype=self.dtype).clone()
            assert full_bounds.shape == (2, d), \
                f"bounds must be (2, {d}); got {tuple(full_bounds.shape)}"
            assert (full_bounds[1] >= full_bounds[0]).all(), \
                "bounds upper row must be >= lower row for every dimension"
        self.full_bounds = full_bounds

        self._needle_plot_points_ref = needle_plot_points_ref
        self.ellipsoid_drop_fraction = ellipsoid_drop_fraction
        self.ellipsoid_eigenvalue_floor = ellipsoid_eigenvalue_floor
        self.max_penalty_radius = max_penalty_radius
        self.needle_shrink_factor = float(needle_shrink_factor)
        self.needle_stop_noise_multiplier = float(needle_stop_noise_multiplier)
        self.zoom_jaccard_threshold = float(zoom_jaccard_threshold)
        self.bounds_shrink_factor = float(bounds_shrink_factor)
        self.min_axis_noise_mult = float(min_axis_noise_mult)
        self.min_zoom_for_needle = int(min_zoom_for_needle)      # retired; unread
        self.min_iters_per_zoom = int(min_iters_per_zoom)        # retired; unread
        self.max_lines_per_activation = int(max_lines_per_activation)

        # Basin flood fill (step 2) and the two decisions it feeds (steps 3 & 4).
        self.flood_ci_z = float(flood_ci_z)
        self.flood_k = max(1, int(flood_k))
        self.zoom_volume_fraction = float(zoom_volume_fraction)
        self.needle_move_tol = float(needle_move_tol)
        self.needle_ci_tol = float(needle_ci_tol)
        self.needle_ci_bootstrap = max(1, int(needle_ci_bootstrap))

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if self.verbose:
                print(f"Initialized ZoMBIHop on CUDA device: {torch.cuda.get_device_name()}")
                print(f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # Simplex utilities (functions; not checkpointable)
        self.proj_fn = proj_fn if proj_fn is not None else proj_simplex
        self.random_sampler = random_sampler if random_sampler is not None else random_simplex
        self.random_direction_sampler = (random_direction_sampler if random_direction_sampler is not None
                                         else random_zero_sum_directions)
        self.objective = objective

        if top_m_points is None:
            top_m_points = max(d + 1, 4)
            if self.verbose:
                print(f"Auto-computed top_m_points = {top_m_points} (based on d={d})")

        effective_max_snapshots = max_snapshots if max_snapshots is not None else num_iterations_saved

        # A UUID is normally taken to mean "resume that run". Callers can pass
        # resume=False to start a *fresh* run under a caller-provided UUID (e.g.
        # the GUI pre-creates the run dir so it appears immediately on launch).
        do_resume = resume if resume is not None else (run_uuid is not None)

        # --- DataHandler owns ALL control variables ---
        self.data_handler = DataHandler(
            max_zooms=max_zooms,
            max_iterations=max_iterations,
            top_m_points=top_m_points,
            n_restarts=n_restarts,
            raw=raw,
            convergence_pi_threshold=convergence_pi_threshold,
            input_noise_threshold_mult=input_noise_threshold_mult,
            output_noise_threshold_mult=output_noise_threshold_mult,
            n_consecutive_converged=n_consecutive_converged,
            max_gp_points=max_gp_points,
            repulsion_lambda=repulsion_lambda,
            acquisition_type=acquisition_type,
            ucb_beta=ucb_beta,
            nat_grad_step=nat_grad_step,
            nat_grad_max_steps=nat_grad_max_steps,
            directory=checkpoint_dir,
            run_uuid=run_uuid,
            is_resume=do_resume,
            max_snapshots=effective_max_snapshots,
            device=str(self.device),
            dtype=self.dtype,
            d=d,
            paring_spatial_halfnoise=paring_spatial_halfnoise,
            paring_y_noise_multiplier=paring_y_noise_multiplier,
            input_noise=input_noise,
            input_noise_ilr=input_noise_ilr,
            jaccard_window=jaccard_window,
            jaccard_threshold=jaccard_threshold,
            verbose=verbose,
        )

        # Resume from snapshot or start fresh
        if do_resume:
            if self.verbose:
                print(f"Resuming from saved run: {run_uuid}")
            activation, zoom, iteration, _ = self.data_handler.load_state()
            if self.verbose:
                print(f"Loaded state: activation={activation}, zoom={zoom}, iteration={iteration}")
        else:
            if self.verbose:
                print(f"Starting new run with UUID: {self.data_handler.run_uuid}")
                if checkpoint_dir:
                    print(f"Snapshot directory: {self.data_handler.run_dir}")

            X_init_actual = X_init_actual.clone().to(device=self.device, dtype=self.dtype)
            X_init_expected = X_init_expected.clone().to(device=self.device, dtype=self.dtype)
            Y_init = Y_init.clone().to(device=self.device, dtype=self.dtype)

            assert X_init_actual.shape[1] == d, "X_init_actual must be (n, d)"
            assert X_init_expected.shape[1] == d, "X_init_expected must be (n, d)"
            assert Y_init.shape[1] == 1, "Y_init must be (n, 1)"
            assert X_init_actual.shape[0] == X_init_expected.shape[0] == Y_init.shape[0]

            bounds0 = self.full_bounds.clone()
            self.data_handler.save_init(X_init_actual, X_init_expected, Y_init, bounds0)

        # self.bounds is a convenience alias; always kept in sync with data_handler.bounds
        self.bounds = self.data_handler.bounds

        # Resuming a checkpoint that has no saved bounds (e.g. the run crashed before
        # writing its first snapshot, so load_state restored only the iteration
        # counter) leaves data_handler.bounds / current_zoom_bounds — and this alias —
        # as None. run() then dereferences None at the activation-entry
        # `bounds = self.bounds.clone()` (and the never_terminate bounds clones).
        # A snapshotless resume has no zoom region to preserve, so fall back to the
        # full [0,1]^d simplex — identical to the bounds0 a fresh run builds in
        # save_init above.
        if self.bounds is None:
            self.bounds = self.full_bounds.clone()
            self.data_handler.bounds = self.full_bounds.clone()
            if self.verbose:
                print("[ZoMBIHop] Resumed checkpoint had no saved bounds; "
                      "initialized to the full search box.")
        if self.data_handler.current_zoom_bounds is None:
            self.data_handler.current_zoom_bounds = self.bounds.clone()

        # The minimum-width floor on zoom boxes (DataHandler._apply_min_box_width)
        # widens narrow axes and translates the result back inside the *global* box.
        # save_init records that box, but a resumed handler restores only `bounds`
        # (the current zoom region), and ZoMBI holds the authoritative copy in either
        # case — including a caller-tightened box such as [0, 0.3] on one axis — so
        # set it unconditionally rather than relying on the checkpoint.
        self.data_handler._full_bounds_ref = self.full_bounds.clone()

        # GP handler
        self.gp_handler = GPSimplex(
            data_handler=self.data_handler,
            proj_fn=self.proj_fn,
            random_sampler=self.random_sampler,
            num_restarts=self.data_handler.n_restarts,
            raw_samples=self.data_handler.raw,
            repulsion_lambda=self.data_handler.repulsion_lambda,
            acquisition_type=self.data_handler.acquisition_type,
            ucb_beta=self.data_handler.ucb_beta,
            nat_grad_step=self.data_handler.nat_grad_step,
            nat_grad_max_steps=self.data_handler.nat_grad_max_steps,
            device=str(self.device),
            dtype=self.dtype,
            verbose=self.verbose,
        )

    # --- Properties (expose data handler state) ---

    @property
    def run_uuid(self) -> str:
        return self.data_handler.run_uuid

    @property
    def current_activation(self) -> int:
        return self.data_handler.current_activation

    @property
    def current_zoom(self) -> int:
        return self.data_handler.current_zoom

    @property
    def current_iteration(self) -> int:
        return self.data_handler.current_iteration

    # Convenience shorthands so internal code stays readable
    @property
    def d(self) -> int:
        return self.data_handler.d

    def _is_global_bounds(self, bounds: torch.Tensor, eps: float = 0.01) -> bool:
        """True when ``bounds`` essentially equals the configured search box.

        Generalises the module-level ``_is_global_bounds`` (which assumes the box
        is [0,1]^d) to the run's actual ``self.full_bounds``, so a tightened box —
        e.g. a dimension capped at 0.3 — is correctly recognised as the global
        region rather than looking permanently "zoomed in"."""
        fb = self.full_bounds
        return (torch.abs(bounds[0] - fb[0]).max().item() < eps and
                torch.abs(bounds[1] - fb[1]).max().item() < eps)

    def _all_needle_axes_below_min(self, dh) -> bool:
        """True when every semi-axis is below min_axis_noise_mult × input_noise."""
        threshold = self.min_axis_noise_mult * dh.input_noise
        if threshold <= 0:
            return False
        for M in dh.needle_M_list:
            if M is None:
                continue
            min_eig = torch.linalg.eigvalsh(M).clamp(min=1e-30).min()
            if (1.0 / min_eig.sqrt()).item() > threshold:
                return False
        return True

    def _handle_failure_retry(
        self,
        dh,
        first_failure_handled: bool,
        data_added_since_last_failure: bool,
    ) -> Tuple[bool, bool]:
        """Three-way failure dispatch.

        Returns (should_stop, first_failure_handled_updated).

        Case 1 — first failure: recompute all ellipsoids using clean local GP
          (gp_handler.recompute_all_ellipsoids).
        Case 2 — subsequent failure after objective was called: recompute the
          search box from the basin flood fill (``_flood_fill_basin``), falling
          back to the full search box when no basin can be filled.
        Case 3 — subsequent failure with no new data: shrink radii; terminate
          when all axes fall below min_axis_noise_mult × input_noise.
        """
        n_needles = dh.needles.shape[0] if dh.needles is not None else 0

        if not first_failure_handled:
            self._log("  [failure] first failure — recomputing ellipsoids with clean local GP ...")
            if n_needles > 0 and dh.X_all_actual is not None:
                try:
                    new_M_list = self.gp_handler.recompute_all_ellipsoids(
                        dh.needles,
                        dh.X_all_actual,
                        dh.Y_all.reshape(-1),
                        max_radius=self.max_penalty_radius * 2.0,
                        max_gp_points=dh.max_gp_points,
                        drop_fraction=self.ellipsoid_drop_fraction,
                        eigenvalue_floor=self.ellipsoid_eigenvalue_floor,
                        max_penalty_radius=self.max_penalty_radius,
                    )
                    dh.update_all_needle_radii(new_M_list)
                    self._log(f"  [failure] ellipsoids recomputed for {n_needles} needle(s).")
                except Exception as e:
                    self._log(f"  [failure] recompute_all_ellipsoids failed: {e} — falling back to pared refit.")
                    X_fb, Y_fb = dh.get_gp_data()
                    if X_fb.shape[0] >= 2:
                        self.gp_handler.fit(X_fb, Y_fb)
                        self.gp_handler.create_acquisition(best_f=Y_fb.max().item())
                        for i in range(dh.needles.shape[0]):
                            try:
                                M_new, _ = self.gp_handler.determine_penalty_ellipsoid(
                                    needle=dh.needles[i],
                                    drop_fraction=self.ellipsoid_drop_fraction,
                                    eigenvalue_floor=self.ellipsoid_eigenvalue_floor,
                                    max_radius=self.max_penalty_radius,
                                )
                                dh.needle_M_list[i] = M_new
                            except Exception as e2:
                                self._log(f"  [failure] pared refit needle {i} failed: {e2}")
                        dh._update_penalty_mask()
            return False, True

        elif data_added_since_last_failure:
            self._log("  [failure] subsequent failure with new data — recomputing "
                      "bounds from the basin flood fill ...")
            basin = self._flood_fill_basin(dh, self.bounds)
            if basin is None:
                # Nothing left to fill inside the active box (it is penalised out, or
                # holds too few points) — hand the activation the full search box
                # back rather than re-proposing the region that just failed.
                new_bounds = self.full_bounds.clone()
                self._log("  [failure] no basin available — resetting to the full box.")
            else:
                new_bounds = self._basin_box(dh, basin[0])
            self._log(f"  [failure] new bounds: [{new_bounds[0].cpu().numpy()}] – [{new_bounds[1].cpu().numpy()}]")
            dh.current_zoom_bounds = new_bounds.clone()
            dh.bounds = new_bounds.clone()
            self.bounds = new_bounds.clone()
            return False, True

        else:
            if n_needles == 0 or self._all_needle_axes_below_min(dh):
                self._log("  [failure] all needle axes below noise floor — stopping.")
                return True, True
            self._log(
                f"  [failure] no new data — shrinking radii by factor {self.bounds_shrink_factor:.2f} ..."
            )
            dh.shrink_all_needle_radii(self.bounds_shrink_factor)
            max_r = dh.max_needle_radius()
            self._log(f"  [failure] max needle radius after shrink: {max_r:.4f}")
            if self._all_needle_axes_below_min(dh):
                self._log("  [failure] axes now below noise floor — stopping.")
                return True, True
            return False, True

    def _declare_needle_at_best(
        self,
        dh,
        zoom: int,
        iteration: int,
        reason: str = "converged",
        bounds: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Declare a needle at the best unpenalized point of the active region.

        ``bounds``: the search box the convergence decision was made in. When it
        is a zoom box, the needle is placed at the best unpenalized point *inside
        that box* (``get_best_in_bounds``) — convergence is judged on local EI and
        local improvement, so the declared optimum must come from the same region.
        Placing it at the global best would drop the needle somewhere the current
        zoom never examined. Falls back to ``get_best_unpenalized`` on the full box
        (or when no bounds are given).

        Delegates to ``_declare_needle_from_point``.  Returns the needle tensor,
        or None when no unpenalized points exist.
        """
        if bounds is not None and not self._is_global_bounds(bounds):
            needle_X, needle_Y, global_idx = dh.get_best_in_bounds(bounds)
        else:
            needle_X, needle_Y, global_idx = dh.get_best_unpenalized()
        if needle_X is None:
            self._log("  [declare_needle] no unpenalized points — cannot declare needle.")
            return None
        return self._declare_needle_from_point(
            dh, needle_X, needle_Y, global_idx, zoom, iteration, reason
        )

    def _declare_needle_from_point(
        self,
        dh,
        needle_X: torch.Tensor,
        needle_Y: torch.Tensor,
        global_idx: int,
        zoom: int,
        iteration: int,
        reason: str = "converged",
    ) -> torch.Tensor:
        """
        Declare a needle at ``needle_X`` — the declaration core shared by the
        live convergence path and ``retro_declare_needles``.

        Fits the GP, computes the Hessian ellipsoid, records the needle, and
        resets bounds to the full simplex.  Returns the needle tensor.
        ``reason`` is used only for logging.
        """
        # Median Y of all raw observations within the paring spatial distance
        thresh = dh.paring_spatial_halfnoise * dh.input_noise
        nearby = torch.norm(dh.X_all_actual - needle_X.unsqueeze(0), dim=1) <= thresh
        needle_median = dh.Y_all[nearby].reshape(-1).median().item() if nearby.any() else float('nan')

        self._log(
            f"\n*** Found needle ({reason}) at {needle_X.cpu().numpy()} "
            f"with value {needle_Y.item():.4f}  (local median={needle_median:.4f}) ***"
        )

        # Refit on global data for best Hessian estimate at the needle location
        X, Y = dh.get_gp_data()
        _t0 = time.time()
        self.gp_handler.fit(X, Y)
        self._log(f"  [declare] GP refit: {time.time()-_t0:.2f}s  ({X.shape[0]} pts)")
        self.gp_handler.create_acquisition(best_f=Y.max().item(), penalty_value=-1e6)

        _t0 = time.time()
        M_ellipsoid, B_ellipsoid = self.gp_handler.determine_penalty_ellipsoid(
            needle=needle_X,
            drop_fraction=self.ellipsoid_drop_fraction,
            eigenvalue_floor=self.ellipsoid_eigenvalue_floor,
            max_radius=self.max_penalty_radius,
        )
        self._log(f"  [declare] ellipsoid: {time.time()-_t0:.2f}s")
        self._log("Penalizing with Hessian ellipsoid.")

        dh.add_needle(
            needle=needle_X,
            needle_value=needle_Y.item(),
            needle_penalty_radius=0.0,
            zoom=zoom,
            iteration=iteration,
            M=M_ellipsoid,
            B=B_ellipsoid,
            needle_median_value=needle_median,
            reason=reason,
        )

        if self._needle_plot_points_ref is not None:
            center = dh.X_all_actual.mean(0)
            distance = torch.norm(needle_X - center).item()
            self._needle_plot_points_ref.append({
                "sample_idx": global_idx + 1,
                "point": needle_X.detach().cpu().numpy().ravel().tolist(),
                "y": needle_Y.item(),
                "distance": distance,
            })

        # Reset to the full search box for the next activation
        full_bounds = self.full_bounds.clone()
        self.bounds = full_bounds
        dh.bounds = full_bounds.clone()
        self._log(f"  → bounds reset to full search box for next activation")

        return needle_X

    def retro_declare_needles(self, dry_run: bool = False) -> dict:
        """
        Retroactively declare needles the CURRENT convergence criteria would
        have produced on this run's already-measured history.

        Intended for a resume after the operator loosened the criteria (e.g.
        n_consecutive_converged 5→2 in config.json): replays the convergence
        record stream (see src/core/retro.py for sources and the evidence
        standard), finds at most one trigger per past activation that has not
        already declared a needle, and declares each trigger's needle at that
        activation's best measured point — skipping it when that point is
        already inside a penalty region ("covered"), which also makes repeated
        calls idempotent. After ≥1 declaration the resume position advances to
        a fresh activation on the full search box and a permanent
        "retro_needles" snapshot persists everything.

        ``dry_run=True`` mutates nothing (no GP fit, no snapshot, no state
        change, no file writes — including run.log) and reports what an apply
        would attempt. Never raises: any internal failure returns
        ``{applied: False, error: ...}`` so a hardware resume cannot be
        poisoned.
        """
        try:
            return self._retro_declare_needles(dry_run=dry_run)
        except Exception as e:
            msg = f"  [retro] retro_declare_needles failed: {e!r} — continuing without retro needles."
            try:
                if dry_run:
                    if self.verbose:
                        print(msg)
                else:
                    self._log(msg)
            except Exception:
                pass
            return {"applied": False, "error": repr(e), "triggers": [], "candidates": []}

    def _retro_declare_needles(self, dry_run: bool) -> dict:
        dh = self.data_handler

        def _rlog(message: str):
            # dry_run must leave the run dir byte-identical — bypass run.log.
            if dry_run:
                if self.verbose:
                    print(message)
            else:
                self._log(message)

        result: dict = {"applied": False, "triggers": [], "candidates": []}

        n_consec = int(dh.n_consecutive_converged)
        if n_consec < 1:
            result["error"] = f"invalid n_consecutive_converged={n_consec}"
            _rlog(f"  [retro] {result['error']} — aborting.")
            return result
        if not dh.save_enabled or dh.run_dir is None:
            result["error"] = "run has no directory (saving disabled) — nothing to replay"
            _rlog(f"  [retro] {result['error']}")
            return result

        records, source = retro.load_convergence_history(dh.run_dir)
        if not records:
            result["error"] = f"no convergence history ({source})"
            _rlog(f"  [retro] {result['error']}")
            return result

        skip_activations = set(retro.needle_discovery_activations(dh.run_dir))
        _rlog(f"  [retro] replaying {len(records)} record(s) from {source}  "
              f"(criteria: n_consecutive={n_consec}, "
              f"min_zoom={self.min_zoom_for_needle}, "
              f"min_iters={self.min_iters_per_zoom}; "
              f"skipping needle activations {sorted(skip_activations)})")

        triggers = retro.find_retro_triggers(
            records, n_consec, self.min_zoom_for_needle, self.min_iters_per_zoom,
            skip_activations,
        )
        result["triggers"] = triggers
        if not triggers:
            _rlog("  [retro] no past activation satisfies the needle criteria.")
            return result

        ranges = retro.activation_point_ranges(dh.run_dir)
        n_rows = dh.X_all_actual.shape[0] if dh.X_all_actual is not None else 0
        declared = 0
        candidates_out: List[dict] = []
        earlier_points: List[torch.Tensor] = []
        for trig in triggers:
            act = trig["activation"]
            entry = dict(trig)
            rows = [i for (a, b) in ranges.get(act, [])
                    for i in range(a, min(b, n_rows))]
            if not rows:
                entry["skipped_reason"] = "empty"
                _rlog(f"  [retro] activation {act}: trigger at z{trig['zoom']}/"
                      f"i{trig['iteration']} but no attributable points — skipping.")
                candidates_out.append(entry)
                continue

            # One needle per activation, ever. A needle sits at a measured
            # point, so an activation whose row range already contains one has
            # been accounted for — by the original run or by an earlier retro
            # pass. Without this a resume would keep mining the same activation
            # for its next-best leftover point every time the penalty
            # ellipsoids are too small to cover it, stacking ever-worse needles
            # on each resume.
            if dh.needle_indices is not None and dh.needle_indices.numel():
                existing = {int(i) for i in dh.needle_indices.reshape(-1).tolist()}
                if existing & set(rows):
                    entry["skipped_reason"] = "already declared"
                    _rlog(f"  [retro] activation {act}: already has a needle "
                          f"— skipping.")
                    candidates_out.append(entry)
                    continue

            # The activation's best UNPENALIZED measured point. The live path
            # (``_declare_needle_at_best``) restricts the same rule to the zoom
            # box it converged in; replay has no stored bounds per trigger, so
            # the activation's own rows stand in for that locality.
            # Points already inside a penalty region are skipped rather than
            # re-declared, so an activation that converged onto ground a
            # previous needle already owns contributes its best *uncovered*
            # optimum instead of nothing. Each declaration below updates the
            # mask, so later triggers falling inside a new ellipsoid drop out
            # naturally rather than stacking duplicates.
            idx_t = torch.tensor(rows, device=dh.X_all_actual.device, dtype=torch.long)
            mask_rows = dh.get_penalty_mask()[idx_t]
            if not bool(mask_rows.any().item()):
                entry["skipped_reason"] = "covered"
                _rlog(f"  [retro] activation {act}: every measured point lies "
                      f"inside an existing penalty region — skipping.")
                candidates_out.append(entry)
                continue
            open_idx = idx_t[mask_rows]
            best_local = int(torch.argmax(dh.Y_all[open_idx].reshape(-1)).item())
            row = int(open_idx[best_local].item())
            needle_X = dh.X_all_actual[row]
            needle_Y = dh.Y_all[row]
            entry["x"] = needle_X.detach().cpu().numpy().ravel().tolist()
            entry["y"] = float(needle_Y.item())
            entry["dist_to_earlier_candidates"] = [
                float(torch.norm(needle_X - p).item()) for p in earlier_points
            ]
            earlier_points.append(needle_X.detach().clone())

            if dry_run:
                _rlog(f"  [retro] activation {act}: would declare needle at "
                      f"{needle_X.cpu().numpy()} (Y={entry['y']:.4f}; trigger "
                      f"z{trig['zoom']}/i{trig['iteration']}, "
                      f"counter {trig['counter']}/{n_consec}).")
            else:
                _rlog(f"  [retro] activation {act}: declaring needle "
                      f"(trigger z{trig['zoom']}/i{trig['iteration']}, "
                      f"counter {trig['counter']}/{n_consec}) ...")
                # Same reason string a live convergence needle carries: a
                # retroactive needle IS a convergence needle, just recognised
                # late, and no display or export path may tell them apart.
                # Provenance stays in the run log's [retro] lines.
                needle = self._declare_needle_from_point(
                    dh, needle_X, needle_Y, row,
                    zoom=int(trig["zoom"]), iteration=int(trig["iteration"]),
                    reason="EI convergence",
                )
                entry["declared"] = needle is not None
                if needle is not None:
                    declared += 1
            candidates_out.append(entry)

        result["candidates"] = candidates_out
        if dry_run:
            n_would = len([c for c in candidates_out if "skipped_reason" not in c])
            _rlog(f"  [retro] dry run: {n_would} of {len(triggers)} trigger(s) would "
                  f"declare a needle; nothing was changed. Note: at apply time each "
                  f"new ellipsoid can cover later candidates, collapsing nearby "
                  f"candidates into fewer needles.")
            return result

        result["n_declared"] = declared
        if declared > 0:
            # Fresh activation for the resumed search. The declaration core
            # already reset self.bounds/dh.bounds to the full box, but run()
            # re-enters the resumed activation through dh.current_zoom_bounds —
            # reset it too, BEFORE the snapshot so delta.pt persists it.
            acts = [int(r["activation"]) for r in records
                    if isinstance(r.get("activation"), int)]
            new_act = max([int(dh.current_activation)] + acts) + 1
            dh.current_zoom_bounds = self.full_bounds.clone()
            dh.take_snapshot("retro_needles", permanent=True,
                             activation=new_act, zoom=0, iteration=0)
            result["applied"] = True
            result["new_activation"] = new_act
            self._log(f"  [retro] declared {declared} retroactive needle(s); "
                      f"resuming at fresh activation {new_act} (zoom 0, iter 0) "
                      f"on the full search box.")
        else:
            self._log("  [retro] no retroactive needles declared "
                      "(all candidates covered or empty).")
        return result

    def _penalize_capped_zone(self, dh, bounds: torch.Tensor) -> bool:
        """Penalise the region an activation burned its whole line budget in.

        Counterpart to ``_declare_needle_at_best``: same ellipsoid machinery, but
        recorded as an *exclusion zone* rather than a needle. Hitting the cap means
        the search kept re-converging somewhere without ever satisfying the needle
        criteria — that is a statement about where NOT to look again, not a
        discovered optimum, so it must not enter the needle record (``needles.csv``
        and every metric derived from it, e.g. dist_to_needles).

        The zone is centred on the best unpenalised point inside the active search
        bounds — the attractor the activation kept falling into. Returns True if a
        zone was recorded. Bounds are reset to the full search box either way, so
        the next activation restarts globally instead of inside the dead region.
        """
        center, center_Y, _ = (dh.get_best_unpenalized() if self._is_global_bounds(bounds)
                               else dh.get_best_in_bounds(bounds))
        if center is None:
            center, center_Y, _ = dh.get_best_unpenalized()

        if center is None:
            self._log("  [cap] no unpenalised point to centre an exclusion zone on — "
                      "resetting bounds only.")
        else:
            M_excl = None
            B_excl = None
            try:
                X, Y = dh.get_gp_data()
                if X.shape[0] >= 2:
                    self.gp_handler.fit(X, Y)
                    self.gp_handler.create_acquisition(best_f=Y.max().item(), penalty_value=-1e6)
                    M_excl, B_excl = self.gp_handler.determine_penalty_ellipsoid(
                        needle=center,
                        drop_fraction=self.ellipsoid_drop_fraction,
                        eigenvalue_floor=self.ellipsoid_eigenvalue_floor,
                        max_radius=self.max_penalty_radius,
                    )
            except Exception as e:
                self._log(f"  [cap] ellipsoid fit failed ({e}) — falling back to a sphere.")
                M_excl = None

            if M_excl is not None:
                dh.add_exclusion(center=center, M=M_excl, B=B_excl)
            else:
                # Sphere fallback: half the diagonal of the bounds the activation
                # was working in — literally "the zone it was stuck in". NOT
                # max_penalty_radius, which caps an ellipsoid SEMI-AXIS; read as a
                # sphere radius its default of 1.0 exceeds the simplex diameter
                # (√2) and would penalise the entire search space in one go.
                r = 0.5 * torch.norm(bounds[1] - bounds[0]).item()
                r = min(r, float(self.max_penalty_radius))
                dh.add_exclusion(center=center, M=None, radius=r)
            _y = center_Y.item() if torch.is_tensor(center_Y) else center_Y
            self._log(f"  [cap] exclusion zone added at {center.cpu().numpy()} "
                      f"(best local Y={_y:.4f}); this region is now repelled.")

        full_bounds = self.full_bounds.clone()
        self.bounds = full_bounds
        dh.bounds = full_bounds.clone()
        dh.current_zoom_bounds = full_bounds.clone()
        return center is not None

    def _log(self, message: str):
        if self.verbose:
            print(message)
        try:
            self.data_handler.write_log(message)
        except Exception:
            pass

    def _log_status(self, activation: int, zoom: int, iteration: int,
                    candidate: Optional[torch.Tensor], ei: Optional[float] = None):
        if self.verbose:
            candidate_str = f"{candidate.cpu().numpy()}" if candidate is not None else "None"
            extra = f" | EI={ei:.2e}" if ei is not None else ""
            print(f"[A{activation+1}/Z{zoom+1}/I{iteration+1}] Candidate: {candidate_str}{extra}")

    def _record_convergence(self, **record):
        """Sidecar write for retroactive replay — must never disturb the loop."""
        try:
            self.data_handler.append_convergence_record(**record)
        except Exception:
            pass

    def _record_ei(self, candidate: torch.Tensor, best_f: Optional[float]) -> float:
        """Expected Improvement at *candidate*, logged and recorded — diagnostic only.

        The needle criteria no longer read EI (they are the basin-stability tests in
        :meth:`_needle_stability`), but ``log_ei_history`` is what the convergence
        plots and ``optimize/visualization`` read back, and an EI trace is still the
        cheapest one-number read on whether a region is exhausted. Never raises.
        """
        ei, log_ei = 0.0, float('-inf')
        if best_f is None:
            _, Y_gp = self.data_handler.get_gp_data()
            best_f = Y_gp.max().item() if Y_gp.numel() > 0 else 0.0
        try:
            ei = self.gp_handler.expected_improvement(candidate, best_f)
            log_ei = self.gp_handler.compute_log_ei_at_point(candidate, best_f)
        except Exception:
            pass
        self.data_handler.log_ei_history.append(log_ei)
        return ei

    # ── Step 2: basin flood fill ─────────────────────────────────────────────────

    def _basin_nodes(self, dh, bounds: torch.Tensor,
                     max_nodes: int = 2000) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Nodes the flood fill spreads over: unpenalised measured points in *bounds*.

        The PARED set is preferred over the raw one. Paring already deduplicates
        points that sit within measurement resolution of each other, so it gives
        roughly one node per resolvable cell — which is what a kNN graph wants. The
        raw set would instead put ~24 near-coincident nodes on every LineBO line, so
        each point's k nearest neighbours would all be its own line-mates and the
        fill could never leave the line it started on.

        Falls back to raw points in bounds, then to raw points anywhere (an activation
        that has just reset to a fresh box may have nothing inside it yet). Subsamples
        to *max_nodes* — uniformly, NOT by Y, because a top-Y truncation would delete
        exactly the basin walls whose exclusion defines where the fill stops.
        """
        lo, hi = bounds[0], bounds[1]

        def _in_bounds(X: torch.Tensor) -> torch.Tensor:
            return ((X >= lo.unsqueeze(0)) & (X <= hi.unsqueeze(0))).all(dim=1)

        candidates: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]] = [
            (dh.X_pared, dh.Y_pared, True),
            (dh.X_all_actual, dh.Y_all, True),
            (dh.X_all_actual, dh.Y_all, False),
        ]
        for X_src, Y_src, restrict in candidates:
            if X_src is None or X_src.shape[0] == 0:
                continue
            keep = dh.get_penalty_mask(X_src)
            if restrict:
                keep = keep & _in_bounds(X_src)
            if int(keep.sum()) < 3:
                continue
            X, Y = X_src[keep], Y_src[keep].reshape(-1)
            if X.shape[0] > max_nodes:
                sel = torch.randperm(X.shape[0], device=X.device)[:max_nodes]
                # Keep the incumbent: it is the fill's seed, so losing it to the
                # subsample would restart the basin from an arbitrary point.
                best = int(Y.argmax())
                if not bool((sel == best).any()):
                    sel[0] = best
                X, Y = X[sel], Y[sel]
            return X, Y
        return None

    def _flood_fill_basin(
        self, dh, bounds: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        """The single basin containing the best measured point of *bounds*.

        Breadth-first over a symmetric ``flood_k``-nearest-neighbour graph on the
        nodes from :meth:`_basin_nodes`, admitting a point when its optimistic value
        clears the incumbent's pessimistic one::

            mean(p) + z·sd(p)  >=  mean(best) - z·sd(best)

        with ``z = flood_ci_z`` under the current GP posterior. That single
        inequality is what keeps the fill inside ONE basin: leaving the basin means
        crossing a valley, and a valley point's best case cannot reach the
        incumbent's worst case, so the frontier dies there rather than tunnelling
        into the next peak. The seed is always admitted, so the basin is never empty.

        Returns ``(X_basin, Y_basin, n_nodes)``, or None when there is too little
        data to fill (fewer than 3 nodes) or the posterior cannot be evaluated.
        """
        nodes = self._basin_nodes(dh, bounds)
        if nodes is None:
            return None
        Xn, Yn = nodes
        n = Xn.shape[0]
        if n < 3:
            return None

        try:
            mean, var = self.gp_handler.predict(Xn)
        except Exception as e:
            self._log(f"  [basin] posterior evaluation failed: {e} — skipping flood fill.")
            return None
        mu = mean.reshape(-1)
        sd = var.reshape(-1).clamp(min=0.0).sqrt()
        z = self.flood_ci_z
        ucb = mu + z * sd

        seed = int(Yn.argmax())
        lcb_best = (mu[seed] - z * sd[seed]).item()
        admissible = ucb >= lcb_best
        admissible[seed] = True

        # Symmetric kNN adjacency. topk includes each node itself (distance 0), so
        # ask for k+1 and let the diagonal be cleared afterwards.
        k = min(self.flood_k + 1, n)
        knn = torch.cdist(Xn, Xn).topk(k, largest=False).indices
        A = torch.zeros(n, n, dtype=torch.bool, device=Xn.device)
        rows = torch.arange(n, device=Xn.device).unsqueeze(1).expand_as(knn)
        A[rows.reshape(-1), knn.reshape(-1)] = True
        A = A | A.t()
        A.fill_diagonal_(False)

        visited = torch.zeros(n, dtype=torch.bool, device=Xn.device)
        visited[seed] = True
        frontier = visited.clone()
        while bool(frontier.any()):
            reached = A[frontier].any(dim=0) & admissible & ~visited
            if not bool(reached.any()):
                break
            visited |= reached
            frontier = reached
        return Xn[visited], Yn[visited], n

    def _basin_box(self, dh, X_basin: torch.Tensor) -> torch.Tensor:
        """The basin's convex hull, taken as its axis-aligned bounding box.

        Every search region in this optimiser — ``self.bounds``, ``dh.bounds``, the
        acquisition's optimisation domain, ``get_zoom_gp_data`` — is a per-axis
        ``(2, d)`` box, so the hull is consumed through its AABB. Floored at the
        input-noise resolution (``DataHandler._apply_min_box_width``) so the search
        never zooms below what the printer can actually place.
        """
        box = torch.stack([X_basin.min(dim=0).values, X_basin.max(dim=0).values], dim=0)
        return dh._apply_min_box_width(box)

    @staticmethod
    def _volume_ratio(cand: torch.Tensor, ref: torch.Tensor, floor: float = 1e-9) -> float:
        """vol(*cand*) / vol(*ref*) for two axis-aligned boxes.

        Computed in log space: a d-dimensional volume is a product of d widths, which
        underflows float64 for a tight box in high d, and only the RATIO is ever
        wanted. Widths are floored so a degenerate axis contributes the same factor to
        both boxes and cancels instead of forcing the ratio to 0 or infinity.
        """
        wc = (cand[1] - cand[0]).clamp(min=floor)
        wr = (ref[1] - ref[0]).clamp(min=floor)
        return float(torch.exp((wc.log() - wr.log()).sum()))

    # ── Step 4: needle stability ─────────────────────────────────────────────────

    def _median_ci_halfwidth(self, dh, center: torch.Tensor,
                             max_samples: int = 2000) -> Optional[float]:
        """Bootstrap CI half-width of the MEDIAN of the replicates around *center*.

        The neighbourhood is the same one ``_declare_needle_from_point`` reports a
        needle's median over — every raw observation within
        ``paring_spatial_halfnoise × input_noise`` — so the number the needle records
        and the number the needle criterion watches are the same statistic.

        The median, not the mean: these are repeat measurements of one composition
        under multiplicative output noise, and a single bad print should not move the
        estimate. The median has no closed-form CI worth trusting at the handful of
        replicates available right after a zoom, hence the bootstrap. Returns None
        when fewer than 3 replicates are in range, which the caller treats as "not
        yet stable" rather than as stability.
        """
        if dh.X_all_actual is None or dh.X_all_actual.shape[0] == 0:
            return None
        thresh = dh.paring_spatial_halfnoise * dh.input_noise
        dist = torch.norm(dh.X_all_actual - center.unsqueeze(0), dim=1)
        near = dh.Y_all.reshape(-1)[dist <= thresh]
        n = int(near.numel())
        if n < 3:
            return None
        if n > max_samples:
            near = near[torch.randperm(n, device=near.device)[:max_samples]]
            n = max_samples
        idx = torch.randint(0, n, (self.needle_ci_bootstrap, n), device=near.device)
        meds = near[idx].median(dim=1).values.to(torch.float64)
        q = torch.tensor([0.025, 0.975], device=meds.device, dtype=meds.dtype)
        lo, hi = torch.quantile(meds, q)
        return 0.5 * float(hi - lo)

    def _needle_stability(
        self,
        best_X: torch.Tensor,
        prev_best_X: Optional[torch.Tensor],
        ci_hw: Optional[float],
        prev_ci_hw: Optional[float],
    ) -> Tuple[bool, float, float]:
        """Has the optimum settled? Returns (stable, moved, ci_rel_change).

        Both gates are measured against the PREVIOUS measured line:

        * the best point has moved no more than ``needle_move_tol`` in composition
          L2 — an absolute tolerance in composition units (0.10 = ten composition
          points), deliberately not rescaled by the zoom box, so "the optimum stopped
          moving" means the same thing at every zoom depth;
        * the bootstrap CI half-width of the median at the best point has changed
          relatively by no more than ``needle_ci_tol`` — the value estimate has
          stopped sharpening, so further lines here buy nothing.

        Location alone is not enough (the best point can sit still while its value is
        still being pinned down) and CI alone is not either (a CI can plateau while
        the optimum walks). Missing history — the first line of an activation, or too
        few replicates for a CI — reports not-stable.
        """
        if prev_best_X is None or ci_hw is None or prev_ci_hw is None:
            return False, float('nan'), float('nan')
        moved = torch.norm(best_X - prev_best_X).item()
        denom = max(abs(prev_ci_hw), abs(ci_hw), 1e-12)
        ci_rel = abs(ci_hw - prev_ci_hw) / denom
        stable = (moved <= self.needle_move_tol) and (ci_rel <= self.needle_ci_tol)
        return stable, moved, ci_rel

    def _objective_wrapper(
        self, X: torch.Tensor, bounds: torch.Tensor, acquisition_function
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Call objective and update data handler.
        Returns (unpenalized_X, unpenalized_Y) — points not in any penalty region.
        """
        assert X.shape == (self.d,)
        X_expected, X_actual, Y = self.objective(X, bounds, acquisition_function)

        X_expected = X_expected.to(device=self.device, dtype=self.dtype)
        X_actual = X_actual.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)

        # Project actual measurements onto the simplex so off-simplex apparatus
        # noise doesn't corrupt stored data, distance calculations, or needle positions.
        X_actual = self.proj_fn(X_actual)

        assert X_expected.shape[1] == self.d
        assert X_actual.shape[1] == self.d
        assert Y.ndim == 1
        assert X_expected.shape[0] == X_actual.shape[0] == Y.shape[0]

        penalty_mask = self.data_handler.add_all_points(X_actual, X_expected, Y.unsqueeze(1))
        return X_actual[penalty_mask], Y[penalty_mask]

    def _space_filling_measurement(self, dh, n_pool: int = 2048,
                                   max_ref: int = 3000) -> int:
        """never_terminate fallback for a saturated search: measure one LineBO line
        through the point of the FULL simplex that is farthest from everything
        measured so far (greedy maximin / best-candidate space-filling).

        When the normal search can no longer produce a candidate (the whole simplex
        is penalized and acquisition ascent keeps landing on already-declared
        needles), this spends the remaining budget on the most under-explored region
        instead of idling — repeated calls lay down a low-discrepancy, roughly
        uniform sequence of exploratory lines. Returns the number of new points
        actually measured (0 only in the degenerate case where the objective
        returned nothing, which the caller treats as a hard stop).
        """
        full_lo = self.full_bounds[0].clone()
        full_hi = self.full_bounds[1].clone()

        # Uniform candidate pool over the full search box.
        pool = self.random_sampler(
            n_pool, full_lo, full_hi,
            device=str(self.device), torch_dtype=self.dtype,
        ).to(device=self.device, dtype=self.dtype)

        # Greedy maximin: pick the pool point whose nearest already-measured point
        # is farthest away — i.e. the emptiest spot on the map.
        if dh.X_all_actual is not None and dh.X_all_actual.shape[0] > 0:
            X_meas = dh.X_all_actual.to(device=self.device, dtype=self.dtype)
            if X_meas.shape[0] > max_ref:  # bound the distance-matrix footprint
                sel = torch.randperm(X_meas.shape[0], device=X_meas.device)[:max_ref]
                X_meas = X_meas[sel]
            nn_dist = torch.cdist(pool, X_meas).min(dim=1).values  # (n_pool,)
            center = pool[int(torch.argmax(nn_dist))]
        else:
            center = pool[0]

        # LineBO ranks orientations with the acquisition; make sure one exists.
        if self.gp_handler.acq_fn is None:
            X, Y = dh.get_gp_data()
            if X.shape[0] >= 2:
                self.gp_handler.fit(X, Y)
                self.gp_handler.create_acquisition(best_f=Y.max().item())

        full_bounds = torch.stack([full_lo, full_hi])
        before = dh.X_all_actual.shape[0] if dh.X_all_actual is not None else 0
        self._objective_wrapper(center, full_bounds, self.gp_handler.acq_fn)
        after = dh.X_all_actual.shape[0] if dh.X_all_actual is not None else before
        return after - before

    def run(self, max_activations: int = 5, time_limit_hours: float = None,
            pause_event: Optional[threading.Event] = None,
            never_terminate: bool = False):
        """
        Run ZoMBI-Hop optimization.

        When ``never_terminate`` is True the optimiser is prevented from stopping
        on its own through *any* internal pathway (over-penalisation, activation
        failure with no needles, all-axes-below-noise-floor, or exhausting
        ``max_activations``). For the *heuristic* stops (over-penalisation guess,
        transient activation failure with no needles) it logs a ``[no-stop]`` note,
        shrinks every needle penalty volume to 30% of its size (i.e. by 70%), resets
        the search bounds to the full simplex, and keeps sampling. For the *terminal*
        stop (recovery exhausted — penalty axes already below the noise floor and no
        candidate producible) it instead falls back to **space-filling exploration**:
        it measures a LineBO line through the point of the full simplex farthest from
        everything measured so far (greedy maximin), so leftover budget/precursor is
        spent on the emptiest region rather than idling, and repeated calls tile the
        space roughly uniformly. The only ways to end such a run are the user Stop
        button (``_StopRunRequested``) or the time limit, if one is set. (A bounded
        stall counter is retained purely as a safety net for a genuinely un-measurable
        state — e.g. a persistently singular GP with zero needles — so the process
        can never hard-spin.)

        Returns
        -------
        tuple
            (needles_results, needles, needle_vals, X_all_actual, Y_all)
        """
        dh = self.data_handler  # shorthand

        def _keep_searching(reason: str) -> None:
            """Anti-termination action: log, shrink all penalty volumes by 70%,
            and reset bounds to the full simplex so sampling can continue."""
            self._log(f"  [no-stop] {reason} — shrinking all penalty volumes by "
                      f"70% and resetting to full bounds; continuing.")
            try:
                dh.shrink_all_needle_radii(0.30)
            except Exception as _e:
                self._log(f"  [no-stop] shrink_all_needle_radii failed: {_e}")
            full_bounds = self.full_bounds.clone()
            dh.bounds = full_bounds.clone()
            dh.current_zoom_bounds = full_bounds.clone()
            self.bounds = full_bounds.clone()

        def _spacefill_fallback(context: str) -> bool:
            """never_terminate terminal-stop fallback: measure one space-filling line
            (maximin, in the emptiest part of the full simplex) so the run keeps
            spending budget instead of stopping. Updates loop state; returns True if a
            measurement was made (caller should retry), False if nothing could be
            measured (caller should stop, which only happens on a degenerate simplex
            that can produce no point at all)."""
            nonlocal global_iteration, stalled_retries, bounds, best_f_local
            nonlocal data_added_since_last_failure, lines_this_activation
            full_bounds = self.full_bounds.clone()
            dh.bounds = full_bounds.clone()
            dh.current_zoom_bounds = full_bounds.clone()
            self.bounds = full_bounds.clone()
            n_new = self._space_filling_measurement(dh)
            if n_new <= 0:
                self._log(f"  [no-stop] {context}: space-filling fallback produced no "
                          f"measurement — stopping to avoid a spin loop.")
                return False
            global_iteration += 1
            lines_this_activation += 1
            stalled_retries = 0
            data_added_since_last_failure = True
            dh.take_snapshot(f"act{activation}_z{zoom}_i{iteration}_spacefill",
                             activation=activation, zoom=zoom, iteration=iteration)
            self._log(f"  [no-stop] {context} — space-filling fallback measured "
                      f"{n_new} point(s) in the most under-explored region; continuing.")
            bounds = full_bounds.clone()
            X, Y = dh.get_gp_data()
            best_f_local = Y.max().item() if Y.numel() > 0 else best_f_local
            if X.shape[0] >= 2:
                self.gp_handler.fit(X, Y)
            return True

        def _dispatch_failure(context: str) -> str:
            """A line that produced no usable measurement. Returns 'retry' or 'stop'.

            Same three-way recovery as before (see ``_handle_failure_retry``) — the
            change is only that there is no zoom loop left to restart, so recovery
            hands the caller updated ``bounds`` and the caller simply takes another
            line. Case 2 now recomputes those bounds from the basin flood fill rather
            than from the retired Jaccard sliding window.
            """
            nonlocal first_failure_handled, data_added_since_last_failure
            nonlocal stalled_retries, bounds, best_f_local

            self._log(f"  [failure] {context}")
            # Backstop: reaching here means this line gathered no new measurement. A
            # real objective call resets the counter, so this only fires on a genuine
            # no-progress spin (e.g. a persistently singular GP with zero needles).
            stalled_retries += 1
            if stalled_retries > MAX_STALLED_RETRIES:
                if never_terminate and _spacefill_fallback(
                        f"{stalled_retries} stalled retries with no measurement"):
                    return "retry"
                self._log(
                    f"  [terminate] {stalled_retries} consecutive failure retries with "
                    f"no new objective measurement — the optimiser cannot produce a "
                    f"candidate; stopping."
                )
                return "stop"

            n_needles = dh.needles.shape[0] if dh.needles is not None else 0
            if n_needles == 0:
                if never_terminate:
                    _keep_searching("Activation failed and no needles")
                    data_added_since_last_failure = False
                    first_failure_handled = False
                    bounds = dh.bounds.clone()
                    self.bounds = bounds.clone()
                    X, Y = dh.get_gp_data()
                    best_f_local = Y.max().item() if Y.numel() > 0 else best_f_local
                    if X.shape[0] >= 2:
                        self.gp_handler.fit(X, Y)
                    return "retry"
                self._log("Activation failed and no needles — stopping.")
                return "stop"

            should_stop, first_failure_handled = self._handle_failure_retry(
                dh, first_failure_handled, data_added_since_last_failure
            )
            data_added_since_last_failure = False
            if should_stop:
                # Recovery is exhausted: penalty axes are already below the noise
                # floor and the normal search can no longer produce a candidate.
                # Under never_terminate, spend leftover budget on space-filling
                # exploration instead of stopping.
                if never_terminate and _spacefill_fallback(
                        "simplex saturated (recovery exhausted)"):
                    return "retry"
                return "stop"
            bounds = dh.bounds.clone()
            self.bounds = bounds.clone()
            return "retry"

        if never_terminate:
            # Never exhaust activations on our own — the run only ends via the
            # user Stop button or an explicit time limit.
            max_activations = float("inf")
            self._log("  [no-stop] never_terminate=True — max_activations set to ∞.")

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._log(f"Starting optimization. CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        start_time = time.time() if time_limit_hours is not None else None

        # Publish the in-force hyperparameters (and clear any stale override left
        # by a previous run of this UUID) so a live retune starts from truth.
        write_effective_hparams(self)
        for _chg in apply_pending_hparams(self, log=self._log):
            self._log(f"  [hparams] {_chg}")

        finished = False
        activation, zoom, iteration, _ = dh.get_iteration_state()
        start_activation = activation
        global_iteration = 0

        # Backstop counter for no-measurement loops; see _dispatch_failure.
        MAX_STALLED_RETRIES = 40
        stalled_retries = 0

        # Objective lines measured in the current activation — the ONLY budget an
        # activation has now that zoom depth is data-driven rather than a
        # max_zooms x max_iterations product. Counts every measured line including
        # space-filling fallbacks, and is deliberately NOT reset by the failure-retry
        # path: re-entering the same region is precisely the route by which an
        # activation used to run away to 130+ lines.
        lines_this_activation = 0
        bounds = self.bounds.clone()
        best_f_local: float = float('-inf')
        first_failure_handled = False
        data_added_since_last_failure = False

        while activation < max_activations and not finished:
            self._log(f"\n{'='*50}")
            self._log(f"ACTIVATION {activation+1}/{max_activations}")
            self._log(f"{'='*50}")

            if time_limit_hours is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= time_limit_hours:
                    self._log(f"Time limit of {time_limit_hours} hours reached. Stopping.")
                    finished = True
                    dh.take_snapshot(f"act{activation}_timeout", permanent=True)
                    break
                self._log(f"Elapsed time: {elapsed_hours:.2f} / {time_limit_hours:.2f} hours")

            if self.device.type == 'cuda' and activation > 0:
                torch.cuda.empty_cache()

            first_failure_handled = False
            data_added_since_last_failure = False
            needle = None

            # First attempt: resume from the checkpoint's box; fresh activations
            # start on whatever box the previous activation reset to (the full one).
            if activation == start_activation and dh.current_zoom_bounds is not None:
                bounds = dh.current_zoom_bounds.clone()
            else:
                bounds = self.bounds.clone()
                zoom = 0
                iteration = 0
            dh.current_zoom_bounds = bounds.clone()
            dh.bounds = bounds.clone()
            self.bounds = bounds.clone()

            activation_capped = False
            best_f_local = float('-inf')
            lines_this_activation = 0

            # Stability history for the needle criteria (step 4). Reset per
            # activation: a needle must be justified by THIS activation's evidence,
            # and the previous one ended somewhere else entirely.
            prev_best_X: Optional[torch.Tensor] = None
            prev_ci_hw: Optional[float] = None

            # ── Line loop. One pass = one measured LineBO line, then the four steps
            # of the basin formulation on the data it produced. ──────────────────
            while not finished:
                if lines_this_activation >= self.max_lines_per_activation:
                    activation_capped = True
                    break

                iteration = lines_this_activation
                self._log(f"\n  · line {lines_this_activation+1}/"
                          f"{self.max_lines_per_activation}  (zoom depth {zoom})")
                self._log(f"  Search bounds: [{bounds[0].cpu().numpy()}] – "
                          f"[{bounds[1].cpu().numpy()}]")
                self._record_convergence(activation=activation, zoom=zoom,
                                         iteration=iteration, event="line_entry")

                # Operator hyperparameter changes, applied between measured lines.
                for _chg in apply_pending_hparams(zombi=self, log=self._log):
                    self._log(f"  [hparams] {_chg}")

                if time_limit_hours is not None:
                    elapsed_hours = (time.time() - start_time) / 3600.0
                    if elapsed_hours >= time_limit_hours:
                        self._log("Time limit reached during iteration.")
                        finished = True
                        dh.take_snapshot(f"act{activation}_z{zoom}_i{iteration}_timeout",
                                         activation=activation, zoom=zoom,
                                         iteration=iteration, permanent=True)
                        break

                # Pause check — blocks here until the UI resumes the run.
                if pause_event is not None and not pause_event.is_set():
                    self._log("  [paused — waiting for resume]")
                    pause_event.wait()
                    self._log("  [resumed]")

                # --- Step 1a: fit the GP on the active region ---
                _t0 = time.time()
                if self._is_global_bounds(bounds):
                    X, Y = dh.get_gp_data()
                else:
                    X, Y = dh.get_zoom_gp_data(bounds)
                best_f_local = Y.max().item() if Y.numel() > 0 else float('-inf')
                self.gp_handler.fit(X, Y)
                self._log(f"  GP: {X.shape[0]} pts, best_f_local={best_f_local:.4f}  "
                          f"{time.time()-_t0:.2f}s")

                # --- Step 1b: propose a point and measure a line through it ---
                _t0 = time.time()
                candidate = self.gp_handler.get_candidate(
                    bounds, best_f=(Y.max().item() if Y.numel() > 0 else None))
                self._log(f"  [ZoMBIHop] candidate search done.  {time.time()-_t0:.2f}s")
                if candidate is None:
                    self._log_status(activation, zoom, iteration, None)
                    self._record_convergence(activation=activation, zoom=zoom,
                                             iteration=iteration, measured=False,
                                             event="candidate_none")
                    if _dispatch_failure("no valid candidate (all in penalized "
                                         "regions)") == "stop":
                        finished = True
                        break
                    continue
                self._log(f"  [ZoMBIHop] GP suggested candidate: {candidate.cpu().numpy()}")

                _t0 = time.time()
                unpenalized_X, unpenalized_Y = self._objective_wrapper(
                    candidate, bounds, self.gp_handler.acq_fn
                )
                global_iteration += 1
                lines_this_activation += 1
                data_added_since_last_failure = True
                stalled_retries = 0  # real measurement made → not stalled
                if unpenalized_Y.numel() > 0:
                    y_rng = (f"Y in [{unpenalized_Y.min().item():.4f}, "
                             f"{unpenalized_Y.max().item():.4f}]")
                else:
                    y_rng = "Y=[] (empty)"
                self._log(f"  [ZoMBIHop] Objective returned {unpenalized_X.shape[0]} "
                          f"points, {y_rng}  {time.time()-_t0:.2f}s")

                dh.take_snapshot(
                    f"act{activation}_z{zoom}_i{iteration}",
                    activation=activation, zoom=zoom, iteration=iteration,
                )

                # Refit on the data the line just added — every step below reads this
                # posterior, so it must include the line being judged.
                if self._is_global_bounds(bounds):
                    X, Y = dh.get_gp_data()
                else:
                    X, Y = dh.get_zoom_gp_data(bounds)
                best_f_local = Y.max().item() if Y.numel() > 0 else best_f_local
                self.gp_handler.fit(X, Y)

                if unpenalized_Y.shape[0] == 0:
                    self._record_convergence(activation=activation, zoom=zoom,
                                             iteration=iteration, measured=True,
                                             event="all_penalized")
                    if _dispatch_failure("every point of this line lies inside a "
                                         "penalty region") == "stop":
                        finished = True
                        break
                    continue

                ei = self._record_ei(candidate, best_f_local)
                self._log_status(activation, zoom, iteration, candidate, ei=ei)

                # The incumbent of the active region — the point the basin is filled
                # from and the point a needle would be declared at.
                if self._is_global_bounds(bounds):
                    best_X, best_Y, _ = dh.get_best_unpenalized()
                else:
                    best_X, best_Y, _ = dh.get_best_in_bounds(bounds)
                if best_X is None:
                    self._record_convergence(activation=activation, zoom=zoom,
                                             iteration=iteration, measured=True,
                                             event="no_incumbent")
                    if _dispatch_failure("no unpenalized incumbent to reason "
                                         "about") == "stop":
                        finished = True
                        break
                    continue

                _overall_masked = dh.Y_all[dh.get_penalty_mask()]
                _overall_max_str = (f"{_overall_masked.max().item():.4f}"
                                    if _overall_masked.numel() > 0 else "N/A")
                self._log(f"  Current max Y: {best_Y.item():.4f} | "
                          f"Overall max: {_overall_max_str}")

                # --- Step 2: flood-fill the basin around the incumbent ---
                _t0 = time.time()
                basin = self._flood_fill_basin(dh, bounds)
                if basin is None:
                    self._log("  [basin] too few unpenalized points to flood fill — "
                              "sampling another line.")
                    self._record_convergence(activation=activation, zoom=zoom,
                                             iteration=iteration, measured=True,
                                             converged=False, counter=0, ei=float(ei),
                                             event="basin_unavailable")
                    continue
                X_basin, _Y_basin, n_nodes = basin
                cand_box = self._basin_box(dh, X_basin)
                vol_ratio = self._volume_ratio(cand_box, bounds)
                self._log(f"  [basin] {X_basin.shape[0]}/{n_nodes} node(s) in basin; "
                          f"box volume = {vol_ratio:.3g} x active box  "
                          f"({time.time()-_t0:.2f}s)")

                # --- Step 4: has the optimum settled? ---
                ci_hw = self._median_ci_halfwidth(dh, best_X)
                stable, moved, ci_rel = self._needle_stability(
                    best_X, prev_best_X, ci_hw, prev_ci_hw)
                if prev_best_X is not None:
                    self._log(f"  [stability] best moved {moved:.4f} "
                              f"(tol {self.needle_move_tol:.4f}); median CI half-width "
                              f"{ci_hw if ci_hw is not None else float('nan'):.4g}, "
                              f"rel change {ci_rel:.3f} (tol {self.needle_ci_tol:.3f})"
                              f" -> {'STABLE' if stable else 'moving'}")
                prev_best_X = best_X.clone()
                prev_ci_hw = ci_hw

                self._record_convergence(
                    activation=activation, zoom=zoom, iteration=iteration,
                    measured=True, converged=bool(stable),
                    counter=int(bool(stable)), ei=float(ei),
                    basin_nodes=int(X_basin.shape[0]),
                    basin_volume_ratio=float(vol_ratio),
                    best_moved=float(moved), ci_rel_change=float(ci_rel),
                )

                if stable:
                    # Declare the needle at the incumbent and zoom out: the
                    # declaration itself penalises the region and resets bounds to
                    # the full search box for the next activation.
                    needle = self._declare_needle_at_best(
                        dh, zoom, global_iteration,
                        reason="basin stability", bounds=bounds,
                    )
                    if needle is not None:
                        dh.take_snapshot(
                            f"act{activation}_z{zoom}_i{iteration}_needle",
                            permanent=True)
                        bounds = self.bounds.clone()
                        break
                    # No unpenalized point to place it on — treat as a failed line.
                    if _dispatch_failure("stability reached but no unpenalized point "
                                         "to declare a needle at") == "stop":
                        finished = True
                        break
                    continue

                # --- Step 3: zoom in when the basin is at most half the volume ---
                if vol_ratio <= self.zoom_volume_fraction:
                    zoom += 1
                    bounds = cand_box.clone()
                    dh.current_zoom_bounds = bounds.clone()
                    dh.bounds = bounds.clone()
                    self.bounds = bounds.clone()
                    self._log(f"  → zooming in to depth {zoom}: "
                              f"[{bounds[0].cpu().numpy()}] – [{bounds[1].cpu().numpy()}]")
                    # The incumbent is re-read inside the new box next line; comparing
                    # a global best against an in-box best would register as movement
                    # the optimiser did not actually make.
                    prev_best_X = None
                    prev_ci_hw = None

            # ── After the line loop ──────────────────────────────────────────────
            if finished:
                break

            # Cap reached without a needle: this activation spent its whole line
            # budget circling a region it never localised. Penalise that region so
            # the next activation is repelled from it instead of converging straight
            # back in, and hand it the full simplex.
            if activation_capped and needle is None:
                self._log(
                    f"\n  [cap] activation {activation+1} hit the "
                    f"{self.max_lines_per_activation}-line budget "
                    f"({lines_this_activation} measured) without declaring a "
                    f"needle — ending it and penalising the region."
                )
                self._penalize_capped_zone(dh, bounds)
                dh.take_snapshot(
                    f"act{activation}_z{zoom}_capped",
                    activation=activation, zoom=zoom, iteration=iteration,
                    permanent=True,
                )
                # Exclusion zones accumulate one per capped activation, so they need
                # the same over-penalisation release valve the needle path has —
                # otherwise a run of capped activations can wall off the simplex with
                # nothing to reopen it. Unlike the needle path this never ends the
                # run: a zone records where the search gave up, not something found,
                # so saturating on them is a reason to hand territory back, not stop.
                test_samples = self.random_sampler(
                    dh.raw, self.bounds[0], self.bounds[1],
                    device=str(self.device), torch_dtype=self.dtype,
                )
                penalized_pct = (1 - dh.get_penalty_mask(test_samples).float().mean().item()) * 100
                if penalized_pct > 90:
                    _keep_searching(f"Too much area penalized after cap: {penalized_pct:.2f}%")

            elif needle is not None:
                # Needle found — check over-penalization before the next activation.
                test_samples = self.random_sampler(
                    dh.raw, self.bounds[0], self.bounds[1],
                    device=str(self.device), torch_dtype=self.dtype,
                )
                penalized_pct = (1 - dh.get_penalty_mask(test_samples).float().mean().item()) * 100
                if penalized_pct > 90:
                    if never_terminate:
                        _keep_searching(f"Too much area penalized: {penalized_pct:.2f}%")
                    elif max_activations == float("inf"):
                        full_bounds = self.full_bounds.clone()
                        dh.bounds = full_bounds
                        self.bounds = full_bounds.clone()
                        self._log(f"Too much area penalized: {penalized_pct:.2f}%. Zooming out.")
                    else:
                        self._log(f"Too much area penalized: {penalized_pct:.2f}%. Ending.")
                        finished = True


            # Re-label every pared point with the median of its local neighbourhood
            # so the GP trains on a noise-smoothed signal in subsequent activations.
            _t0 = time.time()
            dh._relabel_pared_with_medians()
            self._log(f"  [time] relabel_pared: {time.time()-_t0:.2f}s")

            activation += 1
            zoom = 0
            iteration = 0

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._log(f"Optimization complete. Final CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        dh.take_snapshot("final", permanent=True)
        self._log(f"\nOptimization complete. Run UUID: {dh.run_uuid}")
        n_current = len(dh.needles_results)
        self._log(f"Found {n_current} needle(s)")

        X_all_actual, _, Y_all = dh.get_all_points()
        return (
            dh.get_all_needle_results(),
            dh.get_all_needle_locations(),
            dh.get_all_needle_vals(),
            X_all_actual,
            Y_all,
        )

    # --- Static methods: backward compatibility ---

    @staticmethod
    def proj_simplex(X):
        """Project points onto the simplex. X: (n, d) -> (n, d)."""
        return proj_simplex(X)

    @staticmethod
    def random_simplex(
        num_samples: int,
        a: torch.Tensor,
        b: torch.Tensor,
        S: float = 1.0,
        max_batch: int = None,
        debug: bool = False,
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.float64,
        **ignored,
    ) -> torch.Tensor:
        """Generate CFS samples from bounded simplex. Returns (num_samples, d)."""
        return random_simplex(num_samples, a, b, S, max_batch, debug, device, torch_dtype, **ignored)

    @staticmethod
    def random_zero_sum_directions(n: int, d: int, device='cuda') -> torch.Tensor:
        """Sample n zero-sum unit vectors of dimension d. Returns (n, d)."""
        return random_zero_sum_directions(n, d, device=device)
