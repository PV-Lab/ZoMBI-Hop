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


# --- CUDA optimization settings (when CUDA is available) ---
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)


def _is_global_bounds(bounds: torch.Tensor, eps: float = 0.01) -> bool:
    """True when bounds essentially span the full simplex (all-zeros lower, all-ones upper)."""
    return bounds[0].max().item() < eps and bounds[1].min().item() > 1.0 - eps


def _bounds_jaccard_simplex(
    bounds_a: torch.Tensor,
    bounds_b: torch.Tensor,
    n_samples: int = 500,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> float:
    """
    Jaccard overlap of two AABB boxes restricted to the simplex, estimated via Monte Carlo.

    Samples uniformly from the simplex; counts what fraction fall inside each box;
    returns  |A ∩ B| / |A ∪ B|  (both sets restricted to the simplex).
    """
    d = bounds_a.shape[1]
    kw: dict = {}
    if device is not None:
        kw["device"] = device
    if dtype is not None:
        kw["dtype"] = dtype
    # Uniform Dirichlet samples
    u = torch.rand(n_samples, d, **kw).clamp(min=1e-9)
    u = u / u.sum(dim=1, keepdim=True)

    def _in_box(pts, lo, hi):
        return ((pts >= lo.unsqueeze(0)) & (pts <= hi.unsqueeze(0))).all(dim=1)

    in_a = _in_box(u, bounds_a[0], bounds_a[1])
    in_b = _in_box(u, bounds_b[0], bounds_b[1])
    n_a = in_a.sum().item()
    n_b = in_b.sum().item()
    n_ab = (in_a & in_b).sum().item()
    denom = n_a + n_b - n_ab
    return 0.0 if denom == 0 else n_ab / denom


class ZoMBIHop:
    """
    Zooming Multi-Basin Identification with Hopping.

    Trust regions are ellipsoids on the simplex in tangent space (see ``Ellipsoid``).
    """

    def __init__(self,
                 objective,
                 X_init_actual: torch.Tensor,
                 X_init_expected: torch.Tensor,
                 Y_init: torch.Tensor,
                 proj_fn: Optional[Callable] = None,
                 random_sampler: Optional[Callable] = None,
                 random_direction_sampler: Optional[Callable] = None,
                 max_zooms: int = 3,
                 max_iterations: int = 10,
                 top_m_points: Optional[int] = None,
                 n_restarts: int = 30,
                 raw: int = 500,
                 convergence_pi_threshold: float = 0.01,  # deprecated/ignored: convergence now uses Expected Improvement vs the output-noise floor
                 input_noise_threshold_mult: float = 2.0,
                 output_noise_threshold_mult: float = 2.0,
                 n_consecutive_converged: int = 2,
                 max_gp_points: int = 3000,
                 repulsion_lambda: Optional[float] = None,
                 acquisition_type: str = "ucb",
                 ucb_beta: float = 0.1,
                 nat_grad_step: float = 0.02,
                 nat_grad_max_steps: int = 50,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float64,
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
                 input_noise: float = 0.064,
                 input_noise_ilr: Optional[float] = None,
                 needle_shrink_factor: float = 0.85,
                 needle_stop_noise_multiplier: float = 3.0,
                 zoom_jaccard_threshold: float = 0.75,
                 bounds_shrink_factor: float = 0.8,
                 min_axis_noise_mult: float = 2.0,
                 jaccard_window: int = 3,
                 jaccard_threshold: float = 0.9,
                 min_zoom_for_needle: int = 2,
                 min_iters_per_zoom: int = 2):
        """Initialize ZoMBIHop optimizer.

        Search-discipline constraints (hard, not tuned):
          * ``min_zoom_for_needle`` — a needle may only be declared once the
            search has zoomed to this 0-indexed level or deeper (default 2 ⇒
            zoom level 3+). This also forces the optimiser to zoom in at least
            ``min_zoom_for_needle + 1`` times before it can localise an optimum.
          * ``min_iters_per_zoom`` — at least this many objective lines must be
            sampled at the current zoom level before the optimiser may declare a
            needle or zoom in/out from it (default 2).
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose

        d = X_init_actual.shape[1]

        self._needle_plot_points_ref = needle_plot_points_ref
        self.ellipsoid_drop_fraction = ellipsoid_drop_fraction
        self.ellipsoid_eigenvalue_floor = ellipsoid_eigenvalue_floor
        self.max_penalty_radius = max_penalty_radius
        self.needle_shrink_factor = float(needle_shrink_factor)
        self.needle_stop_noise_multiplier = float(needle_stop_noise_multiplier)
        self.zoom_jaccard_threshold = float(zoom_jaccard_threshold)
        self.bounds_shrink_factor = float(bounds_shrink_factor)
        self.min_axis_noise_mult = float(min_axis_noise_mult)
        self.min_zoom_for_needle = int(min_zoom_for_needle)
        self.min_iters_per_zoom = int(min_iters_per_zoom)

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

            bounds0 = torch.zeros(2, d, device=self.device, dtype=self.dtype)
            bounds0[1] = 1.0
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
            full_bounds = torch.zeros(2, d, device=self.device, dtype=self.dtype)
            full_bounds[1] = 1.0
            self.bounds = full_bounds
            self.data_handler.bounds = full_bounds.clone()
            if self.verbose:
                print("[ZoMBIHop] Resumed checkpoint had no saved bounds; "
                      "initialized to the full [0,1]^d simplex.")
        if self.data_handler.current_zoom_bounds is None:
            self.data_handler.current_zoom_bounds = self.bounds.clone()

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
        Case 2 — subsequent failure after objective was called: recompute zoom
          bounds with Jaccard-aware determine_new_bounds(add_to_history=False).
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
            self._log("  [failure] subsequent failure with new data — recomputing zoom bounds (Jaccard-aware) ...")
            new_bounds = dh.determine_new_bounds(add_to_history=False)
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
    ) -> Optional[torch.Tensor]:
        """
        Declare a needle at the current best unpenalized point.

        Fits the GP, computes the Hessian ellipsoid, records the needle, and
        resets bounds to the full simplex.  Returns the needle tensor, or None
        when no unpenalized points exist.  ``reason`` is used only for logging.
        """
        needle_X, needle_Y, global_idx = dh.get_best_unpenalized()
        if needle_X is None:
            self._log("  [declare_needle] no unpenalized points — cannot declare needle.")
            return None

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
            activation=int(getattr(dh, "current_activation", 0) or 0),
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

        # Reset to full simplex for the next activation
        d = self.d
        full_bounds = torch.zeros(2, d, device=self.device, dtype=self.dtype)
        full_bounds[1] = 1.0
        self.bounds = full_bounds
        dh.bounds = full_bounds.clone()
        self._log(f"  → bounds reset to full simplex for next activation")

        return needle_X

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

    def _check_convergence_to_needle(
        self,
        candidate: torch.Tensor,
        unpenalized_X: torch.Tensor,
        unpenalized_Y: torch.Tensor,
        prev_best_X: Optional[torch.Tensor],
        prev_best_Y: Optional[torch.Tensor],
        best_f_ref: Optional[float] = None,
    ) -> Tuple[bool, float, float]:
        """
        Check convergence to a local optimum. Returns (converged, ei, log_ei).

        Converge when:
        1. Expected Improvement at candidate < output_noise * output_noise_threshold_mult
           (the GP expects no more than noise-level improvement)
        2. Latest best Y improves by less than output_noise * output_noise_threshold_mult

        ``best_f_ref``: if provided, used as the reference for EI / log-EI instead
        of the global max from get_gp_data().  Pass the max of the local GP training
        data so that EI is computed relative to the best value seen *within the current
        zoom bounds*, avoiding spurious convergence when an unrelated high-value point
        exists outside the active zoom region.
        """
        if unpenalized_X.shape[0] == 0:
            return False, 0.0, float('-inf')

        idx = unpenalized_Y.argmax().item()
        latest_best_X = unpenalized_X[idx : idx + 1].squeeze(0)
        latest_best_Y = unpenalized_Y[idx].item()

        if best_f_ref is not None:
            best_f = best_f_ref
        else:
            _, Y_gp = self.data_handler.get_gp_data()
            best_f = Y_gp.max().item()

        ei = 0.0
        log_ei = float('-inf')
        try:
            ei = self.gp_handler.expected_improvement(candidate, best_f)
            log_ei = self.gp_handler.compute_log_ei_at_point(candidate, best_f)
        except Exception:
            pass
        self.data_handler.log_ei_history.append(log_ei)

        if prev_best_X is None or prev_best_Y is None:
            return False, ei, log_ei

        # Noise floor shared by both gates: the candidate's expected improvement and
        # the realized best-Y improvement must each sit below output-noise scale.
        output_noise = self.gp_handler.get_output_noise()
        noise_floor = output_noise * self.data_handler.output_noise_threshold_mult
        ei_low = ei < noise_floor
        prev_y = prev_best_Y.item() if torch.is_tensor(prev_best_Y) else prev_best_Y
        improvement = latest_best_Y - prev_y
        input_distance = torch.norm(latest_best_X - prev_best_X).item()
        output_within_noise = improvement < noise_floor
        converged = ei_low and output_within_noise

        if converged and self.verbose:
            self._log(f"Converged: EI={ei:.2e}, improvement={improvement:.2e}, "
                      f"input_dist={input_distance:.2e}, logEI={log_ei:.2f}")
        return converged, ei, log_ei

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
        full_lo = torch.zeros(self.d, device=self.device, dtype=self.dtype)
        full_hi = torch.ones(self.d, device=self.device, dtype=self.dtype)

        # Uniform candidate pool over the full simplex.
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
            full_bounds = torch.zeros(2, self.d, device=self.device, dtype=self.dtype)
            full_bounds[1] = 1.0
            dh.bounds = full_bounds.clone()
            dh.current_zoom_bounds = full_bounds.clone()
            self.bounds = full_bounds.clone()

        def _spacefill_fallback(context: str) -> bool:
            """never_terminate terminal-stop fallback: measure one space-filling line
            (maximin, in the emptiest part of the full simplex) so the run keeps
            spending budget instead of stopping. Updates loop state; returns True if a
            measurement was made (caller should ``continue``), False if nothing could
            be measured (caller should stop, which only happens on a degenerate
            simplex that can produce no point at all)."""
            nonlocal global_iteration, stalled_retries, bounds, best_f_local
            nonlocal start_iteration, data_added_since_last_failure
            full_bounds = torch.zeros(2, self.d, device=self.device, dtype=self.dtype)
            full_bounds[1] = 1.0
            dh.bounds = full_bounds.clone()
            dh.current_zoom_bounds = full_bounds.clone()
            self.bounds = full_bounds.clone()
            n_new = self._space_filling_measurement(dh)
            if n_new <= 0:
                self._log(f"  [no-stop] {context}: space-filling fallback produced no "
                          f"measurement — stopping to avoid a spin loop.")
                return False
            global_iteration += 1
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
            start_iteration = 0
            return True

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

        # Backstop counter for no-measurement loops. The main saturation case is
        # handled directly below (should_stop → space-filling fallback), but
        # never_terminate has a second no-measurement path: an activation failure with
        # zero needles (e.g. a persistently singular GP makes get_candidate return
        # None) loops through _keep_searching with nothing to shrink. A run only makes
        # real progress when the objective is measured, so we count consecutive failure
        # dispatches that gather no new measurement and reset to 0 on any objective
        # call. If we rack up this many in a row, under never_terminate we force a
        # space-filling measurement (keeping the "never stop" promise); otherwise we
        # stop rather than spin.
        MAX_STALLED_RETRIES = 40
        stalled_retries = 0

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

            # Failure-handling loop: on activation failure, first refit needle ellipsoids
            # with the updated pared GP; if no new data exists, shrink all needle ellipsoids;
            # stop when all axes fall below the noise floor.
            first_failure_handled: bool = False
            data_added_since_last_failure: bool = False

            needle = None
            # First attempt: resume from checkpoint zoom; fresh activations start at zoom 0.
            if activation == start_activation and dh.current_zoom_bounds is not None:
                bounds = dh.current_zoom_bounds.clone()
                dh.bounds = bounds.clone()
            else:
                bounds = self.bounds.clone()
                dh.current_zoom_bounds = bounds.clone()
                dh.bounds = bounds.clone()
            activation_failed = False
            consecutive_converged = 0
            best_f_local: float = float('-inf')
            zoom_bounds_history: List[torch.Tensor] = []

            start_zoom = zoom if activation == start_activation else 0
            current_zoom = start_zoom

            while current_zoom < dh.max_zooms and not finished:
                zoom = current_zoom
                self._log(f"\n{'─'*50}")
                self._log(f"--- Zoom {zoom+1}/{dh.max_zooms} ---")
                self._log(f"Search bounds: [{bounds[0].cpu().numpy()}] – [{bounds[1].cpu().numpy()}]")

                # Use local GP data when zoomed in so GP posterior is tight
                # within the active region and EI drops to signal convergence.
                _t0 = time.time()
                if _is_global_bounds(bounds):
                    X, Y = dh.get_gp_data()
                else:
                    X, Y = dh.get_zoom_gp_data(bounds)
                best_f_local = Y.max().item() if Y.numel() > 0 else float('-inf')
                self._log(f"GP data points: {X.shape[0]} (best_f_local={best_f_local:.4f})  data_fetch={time.time()-_t0:.2f}s")
                self._log("  fitting GP …")
                _t0 = time.time()
                self.gp_handler.fit(X, Y)
                self._log(f"  GP fitted.  {time.time()-_t0:.2f}s")

                start_iteration = iteration if (activation == start_activation and zoom == start_zoom) else 0

                # Lines sampled at THIS zoom level; gates the min_iters_per_zoom
                # constraint (needle declaration / zoom-in require ≥ this many).
                iters_this_zoom = 0

                # Mirrors `for iteration in range(start_iteration, dh.max_iterations)`
                # exactly (`iteration` holds the last-executed index on exit), but
                # re-reads dh.max_iterations every pass so a manual change to it
                # takes effect within this zoom rather than at the next one.
                _next_iteration = start_iteration
                while _next_iteration < dh.max_iterations:
                    iteration = _next_iteration
                    _next_iteration += 1
                    self._log(f"\n  · iter {iteration+1}/{dh.max_iterations}")

                    # Operator hyperparameter changes, applied between measured lines.
                    for _chg in apply_pending_hparams(zombi=self, log=self._log):
                        self._log(f"  [hparams] {_chg}")

                    # Time limit check
                    if time_limit_hours is not None:
                        elapsed_hours = (time.time() - start_time) / 3600.0
                        if elapsed_hours >= time_limit_hours:
                            self._log(f"Time limit reached during iteration.")
                            finished = True
                            dh.take_snapshot(f"act{activation}_z{zoom}_i{iteration}_timeout",
                                             activation=activation, zoom=zoom, iteration=iteration,
                                             permanent=True)
                            break

                    # Pause check — blocks here until UI resumes the run
                    if pause_event is not None and not pause_event.is_set():
                        self._log("  [paused — waiting for resume]")
                        pause_event.wait()
                        self._log("  [resumed]")

                    _t0 = time.time()
                    candidate = self.gp_handler.get_candidate(bounds, best_f=Y.max().item() if Y.numel() > 0 else None)
                    if self.verbose:
                        self._log(f"  [ZoMBIHop] candidate search done.  {time.time()-_t0:.2f}s")
                    if self.verbose and candidate is not None:
                        self._log(f"  [ZoMBIHop] GP suggested candidate: {candidate.cpu().numpy()}")

                    if candidate is None:
                        self._log("No valid candidate found (all in penalized regions)")
                        activation_failed = True
                        self._log_status(activation, zoom, iteration, None)
                        break

                    # Local reference point: best unpenalized within the active
                    # bounds so convergence is measured against local progress,
                    # not a global high from a different region.
                    if _is_global_bounds(bounds):
                        prev_best_X, prev_best_Y, _ = dh.get_best_unpenalized()
                    else:
                        prev_best_X, prev_best_Y, _ = dh.get_best_in_bounds(bounds)

                    if self.verbose:
                        self._log(f"  [ZoMBIHop] Calling objective (LineBO samples lines through this candidate)...")

                    _t0 = time.time()
                    unpenalized_X, unpenalized_Y = self._objective_wrapper(
                        candidate, bounds, self.gp_handler.acq_fn
                    )
                    global_iteration += 1
                    iters_this_zoom += 1
                    data_added_since_last_failure = True
                    stalled_retries = 0  # real measurement made → not stalled
                    if self.verbose:
                        if unpenalized_Y.numel() > 0:
                            y_rng = f"Y in [{unpenalized_Y.min().item():.4f}, {unpenalized_Y.max().item():.4f}]"
                        else:
                            y_rng = "Y=[] (empty)"
                        self._log(f"  [ZoMBIHop] Objective returned {unpenalized_X.shape[0]} points, "
                                  f"{y_rng}  {time.time()-_t0:.2f}s")

                    # --- One snapshot per received objective ---
                    _t0 = time.time()
                    dh.take_snapshot(
                        f"act{activation}_z{zoom}_i{iteration}",
                        activation=activation, zoom=zoom, iteration=iteration,
                    )
                    self._log(f"  [time] snapshot: {time.time()-_t0:.2f}s")

                    _t0 = time.time()
                    if _is_global_bounds(bounds):
                        X, Y = dh.get_gp_data()
                    else:
                        X, Y = dh.get_zoom_gp_data(bounds)
                    best_f_local = Y.max().item() if Y.numel() > 0 else best_f_local
                    self.gp_handler.fit(X, Y)
                    self._log(f"  [time] post-obj GP refit: {time.time()-_t0:.2f}s  ({X.shape[0]} pts)")

                    if unpenalized_Y.shape[0] == 0:
                        self._log("No unpenalized Y values, breaking — every point in this batch "
                                  "lies inside at least one needle penalty ball.")
                        activation_failed = True
                        break

                    curr_best_X, curr_best_Y, _ = dh.get_best_unpenalized()

                    converged, ei, log_ei = self._check_convergence_to_needle(
                        candidate, unpenalized_X, unpenalized_Y, prev_best_X, prev_best_Y,
                        best_f_ref=best_f_local,
                    )
                    if converged:
                        consecutive_converged += 1
                    else:
                        consecutive_converged = 0

                    self._log_status(activation, zoom, iteration, candidate, ei=ei)
                    if consecutive_converged > 0:
                        self._log(f"Convergence count: {consecutive_converged}/{dh.n_consecutive_converged}")

                    _overall_masked = dh.Y_all[dh.get_penalty_mask()]
                    _overall_max_str = f"{_overall_masked.max().item():.4f}" if _overall_masked.numel() > 0 else "N/A"
                    _curr_y_str = f"{curr_best_Y.item():.4f}" if curr_best_Y is not None and torch.is_tensor(curr_best_Y) else str(curr_best_Y)
                    self._log(f"Current max Y: {_curr_y_str} | Overall max: {_overall_max_str}")

                    # --- Declare needle after N consecutive converged iterations ---
                    if consecutive_converged >= dh.n_consecutive_converged:
                        deep_enough = zoom >= self.min_zoom_for_needle
                        sampled_enough = iters_this_zoom >= self.min_iters_per_zoom
                        if deep_enough and sampled_enough:
                            needle = self._declare_needle_at_best(
                                dh, zoom, global_iteration, reason="EI convergence"
                            )
                            if needle is not None:
                                dh.take_snapshot(
                                    f"act{activation}_z{zoom}_i{iteration}_needle", permanent=True
                                )
                            break
                        elif sampled_enough and not deep_enough:
                            # Converged, but the search is too shallow to declare a
                            # needle (min_zoom_for_needle). Zoom in further instead.
                            self._log(
                                f"  Converged at zoom {zoom+1} < minimum zoom "
                                f"{self.min_zoom_for_needle+1} for needle — zooming in."
                            )
                            consecutive_converged = 0
                            break
                        # else: converged but fewer than min_iters_per_zoom lines
                        # sampled at this zoom — keep sampling until the minimum is met.

                    if finished:
                        break

                # --- After inner iteration loop ---
                if finished:
                    break

                if needle is not None:
                    # Needle found — check over-penalization, then advance activation.
                    test_samples = self.random_sampler(
                        dh.raw, self.bounds[0], self.bounds[1],
                        device=str(self.device), torch_dtype=self.dtype,
                    )
                    unpenalized_mask = dh.get_penalty_mask(test_samples)
                    penalized_pct = (1 - unpenalized_mask.float().mean().item()) * 100
                    if penalized_pct > 90:
                        if never_terminate:
                            _keep_searching(f"Too much area penalized: {penalized_pct:.2f}%")
                        elif max_activations == float("inf"):
                            full_bounds = torch.zeros(2, self.d, device=self.device, dtype=self.dtype)
                            full_bounds[1] = 1.0
                            dh.bounds = full_bounds
                            self.bounds = full_bounds.clone()
                            self._log(f"Too much area penalized: {penalized_pct:.2f}%. Zooming out.")
                        else:
                            self._log(f"Too much area penalized: {penalized_pct:.2f}%. Ending.")
                            finished = True
                    break  # advance to next activation

                if activation_failed:
                    # Backstop (see MAX_STALLED_RETRIES above): reaching the failure
                    # dispatch means this zoom iteration produced no new measurement
                    # (get_candidate returned None). Count it; a real objective call
                    # resets the counter. This catches no-measurement paths that don't
                    # hit the should_stop fallback below — e.g. an activation failure
                    # with zero needles (singular GP) that keeps retrying. Under
                    # never_terminate we honour "never stop" by forcing a space-filling
                    # measurement (progress) instead of terminating; only a genuinely
                    # un-measurable simplex (fallback returns 0) stops the run.
                    stalled_retries += 1
                    if stalled_retries > MAX_STALLED_RETRIES:
                        if never_terminate and _spacefill_fallback(
                                f"{stalled_retries} stalled retries with no measurement"):
                            continue
                        self._log(
                            f"  [terminate] {stalled_retries} consecutive failure "
                            f"retries with no new objective measurement — the "
                            f"optimiser cannot produce a candidate; stopping."
                        )
                        finished = True
                        break
                    # Three-way failure dispatch.
                    n_needles = dh.needles.shape[0] if dh.needles is not None else 0
                    if n_needles == 0:
                        if never_terminate:
                            _keep_searching("Activation failed and no needles")
                            activation_failed = False
                            consecutive_converged = 0
                            data_added_since_last_failure = False
                            first_failure_handled = False
                            bounds = dh.bounds.clone()
                            self.bounds = bounds.clone()
                            X, Y = dh.get_gp_data()
                            best_f_local = Y.max().item() if Y.numel() > 0 else best_f_local
                            if X.shape[0] >= 2:
                                self.gp_handler.fit(X, Y)
                            start_iteration = 0
                            continue
                        self._log("Activation failed and no needles — stopping.")
                        finished = True
                        break
                    should_stop, first_failure_handled = self._handle_failure_retry(
                        dh, first_failure_handled, data_added_since_last_failure
                    )
                    data_added_since_last_failure = False
                    activation_failed = False
                    consecutive_converged = 0
                    if should_stop:
                        # Recovery is exhausted: penalty axes are already below the
                        # noise floor and the normal search can no longer produce a
                        # candidate (the simplex is saturated and acquisition ascent
                        # keeps landing on already-declared needles). Under
                        # never_terminate, spend leftover budget on space-filling
                        # exploration instead of stopping; keeping
                        # first_failure_handled True means the next dispatch is the
                        # cheap Case-2/3 path, not the expensive ellipsoid recompute.
                        if never_terminate and _spacefill_fallback(
                                "simplex saturated (recovery exhausted)"):
                            continue
                        finished = True
                        break
                    # Reload local GP data for the (possibly updated) bounds
                    bounds = dh.bounds.clone()
                    self.bounds = bounds.clone()
                    if _is_global_bounds(bounds):
                        X, Y = dh.get_gp_data()
                    else:
                        X, Y = dh.get_zoom_gp_data(bounds)
                    best_f_local = Y.max().item()
                    self.gp_handler.fit(X, Y)
                    start_iteration = 0
                    # Retry same zoom level — don't advance current_zoom
                    continue

                # No needle, no failure: advance to next zoom level.
                if zoom < dh.max_zooms - 1:
                    _t0 = time.time()
                    new_bounds = dh.determine_new_bounds()  # Jaccard-aware sliding window
                    self._log(f"  [time] determine_new_bounds: {time.time()-_t0:.2f}s")

                    # MC Jaccard guard: force-declare if the new bounds still heavily
                    # overlap the previous zoom (determine_new_bounds may have exhausted
                    # all windows; this catches the residual repeated-region case).
                    if not _is_global_bounds(new_bounds):
                        repeated_jac = 0.0
                        for prev_bounds in zoom_bounds_history:
                            jac = _bounds_jaccard_simplex(
                                new_bounds, prev_bounds, device=self.device, dtype=self.dtype
                            )
                            if jac > self.zoom_jaccard_threshold:
                                repeated_jac = jac
                                break
                        if repeated_jac > self.zoom_jaccard_threshold and zoom >= self.min_zoom_for_needle:
                            self._log(
                                f"  → repeated zoom (Jaccard={repeated_jac:.3f}) — forcing needle."
                            )
                            needle = self._declare_needle_at_best(
                                dh, zoom, global_iteration, reason="Jaccard convergence"
                            )
                            if needle is None:
                                activation_failed = True
                                consecutive_converged = 0
                                data_added_since_last_failure = False
                                continue  # re-enter failure path
                            dh.take_snapshot(
                                f"act{activation}_z{zoom}_jaccard_needle", permanent=True
                            )
                            break  # needle found
                        elif repeated_jac > self.zoom_jaccard_threshold:
                            # Repeated region but too shallow to declare a needle
                            # (min_zoom_for_needle) — advance the zoom anyway.
                            self._log(
                                f"  → repeated zoom (Jaccard={repeated_jac:.3f}) but zoom "
                                f"{zoom+1} < minimum zoom {self.min_zoom_for_needle+1} "
                                f"for needle — advancing zoom."
                            )
                        zoom_bounds_history.append(new_bounds.clone())
                        if len(zoom_bounds_history) > dh.max_zooms:
                            zoom_bounds_history.pop(0)

                    dh.current_zoom_bounds = new_bounds.clone()
                    dh.bounds = new_bounds.clone()
                    bounds = new_bounds.clone()
                    self.bounds = new_bounds.clone()
                    self._log(f"\n  Zoom {zoom+1} complete — best: "
                              f"{dh.Y_all[dh.get_penalty_mask()].max().item():.4f}")
                    self._log(f"  → new bounds: [{new_bounds[0].cpu().numpy()}] – [{new_bounds[1].cpu().numpy()}]")
                    current_zoom += 1
                    start_iteration = 0
                else:
                    break  # exhausted all zoom levels without needle

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
