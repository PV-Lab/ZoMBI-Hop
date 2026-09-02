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
                 input_noise_threshold_mult: float = 3.0,
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
                 needle_shrink_factor: float = 0.85,
                 needle_stop_noise_multiplier: float = 3.0,
                 zoom_jaccard_threshold: float = 0.75,
                 bounds_shrink_factor: float = 0.8,
                 min_axis_noise_mult: float = 2.0,
                 jaccard_window: int = 3,
                 jaccard_threshold: float = 0.9,
                 min_zoom_for_needle: int = 2,
                 min_iters_per_zoom: int = 3,
                 needle_min_repeats: int = 5,
                 needle_repeat_radius_frac: float = 0.10,
                 max_lines_per_activation: int = 30):
        """Initialize ZoMBIHop optimizer.

        Search-discipline constraints (hard, not tuned):
          * ``needle_min_repeats`` / ``needle_repeat_radius_frac`` — the
            REPEATABILITY GATE. A needle may only be declared at a point that has
            at least ``needle_min_repeats`` OTHER unpenalised measurements within
            ``needle_repeat_radius_frac`` of the composition range in L2 (default
            5 within an absolute 0.10; see ``_repeat_radius``). A single high reading is a noise spike until
            the neighbourhood repeats it, so an ungated declaration plants needles
            on outliers. When the gate is not met the optimiser keeps sampling and
            re-tests on the next line. When it IS met, the declared point is not
            the argmax of the cluster but the repeat whose Y is CLOSEST TO THE
            CLUSTER MEDIAN — the median is the noise-free estimate of the local
            value, so declaring the argmax would systematically bias every needle
            value upward by one noise excursion. The gate sits alongside the
            ``min_zoom_for_needle`` depth requirement below rather than replacing
            it: depth buys a tighter region to declare in, repeatability buys
            confidence that the point in it is real.
          * ``min_zoom_for_needle`` — a needle may only be declared once the
            search has zoomed to this 0-indexed level or deeper (default 2 ⇒
            the log's "Zoom 3" and beyond). Zoom index 0 is the full search box,
            so the level index equals the number of zoom-ins performed: the
            default forces the optimiser to zoom in at least TWICE before it can
            localise an optimum, and ``max_zooms`` must be > this to be able to
            reach it at all.
          * ``min_iters_per_zoom`` — at least this many objective lines must be
            sampled at the current zoom level before the optimiser may declare a
            needle or zoom in/out from it (default 3).
          * ``max_lines_per_activation`` — hard ceiling on objective lines
            measured in one activation (default 30). Without it an activation's
            cost is bounded only by ``max_zooms × max_iterations`` multiplied by
            however many times the failure-retry path re-enters the same zoom,
            which in practice produced single activations of 130+ lines (~3200
            measurements) grinding on one unfruitful region. On reaching the cap
            the activation ends and the region it was stuck in is penalised (see
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
        self.min_zoom_for_needle = int(min_zoom_for_needle)
        self.min_iters_per_zoom = int(min_iters_per_zoom)
        self.needle_min_repeats = int(needle_min_repeats)
        # Why the last declaration attempt did not produce a needle; see
        # _select_needle_point. Read by the main loop to choose between
        # "keep sampling" (gate not met) and "leave this zoom" (nothing here).
        self.last_needle_status = "ok"
        self.needle_repeat_radius_frac = float(needle_repeat_radius_frac)
        self.max_lines_per_activation = int(max_lines_per_activation)

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

    def _repeat_radius(self, bounds: Optional[torch.Tensor]) -> float:
        """Radius, in composition L2, inside which measurements count as REPEATS.

        ``needle_repeat_radius_frac`` of the COMPOSITION RANGE — an absolute 0.10
        by default, independent of zoom level and of dimension.

        Scaling it with the active box instead was measured to be much worse: 10%
        of the full 6-d simplex diagonal is 0.245, which on a 60-line smoke run
        swept 76 of 1464 points into the "repeat" cluster. The gate then passes on
        every converged zoom and the median-closest pick lands on a middling point
        somewhere near the peak rather than on a repeat OF the peak — and since
        zoom forcing was removed, most declarations happen at zoom 0 where the box
        IS the full simplex. At a fixed 0.10 the same neighbourhood held 8 points,
        so the gate binds without being unsatisfiable.
        """
        return float(self.needle_repeat_radius_frac)

    def _select_needle_point(
        self,
        dh,
        bounds: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], str]:
        """Pick the point a declaration would land on, or say why it cannot.

        Two rules, in order:

        1. **In-bounds argmax.** The candidate is the best unpenalised point
           *inside* ``bounds``, never the global argmax. Convergence is judged on
           local EI and local improvement, so declaring on a point the active
           region never examined attributes another region's optimum to this one.
           Unlike ``DataHandler.get_best_in_bounds`` this does NOT fall back to the
           global best when the box holds nothing — an empty box means there is
           nothing here to declare.

        2. **Repeatability gate.** That candidate must have at least
           ``needle_min_repeats`` OTHER unpenalised measurements within
           ``_repeat_radius``; otherwise it is an unconfirmed spike and the answer
           is "keep sampling". When the gate passes, the declared point is the
           member of the cluster whose Y is closest to the CLUSTER MEDIAN, not the
           cluster argmax: the median is the noise-free estimate of the local
           value, and the argmax is by construction the largest upward noise
           excursion in the neighbourhood.

        Returns ``(X, Y, global_index, status)`` with status one of ``"ok"``,
        ``"no_points"`` (nothing unpenalised in the box) or ``"not_repeatable"``
        (gate not met yet).
        """
        mask = dh.get_penalty_mask()
        if mask is None or not mask.any():
            return None, None, None, "no_points"
        idx = torch.where(mask)[0]
        X_unpen = dh.X_all_actual[idx]
        Y_unpen = dh.Y_all[idx].reshape(-1)

        if bounds is not None:
            in_b = ((X_unpen >= bounds[0].unsqueeze(0))
                    & (X_unpen <= bounds[1].unsqueeze(0))).all(dim=1)
            if not in_b.any():
                return None, None, None, "no_points"
            idx, X_unpen, Y_unpen = idx[in_b], X_unpen[in_b], Y_unpen[in_b]

        best_X = X_unpen[int(Y_unpen.argmax().item())]

        # Repeats are searched over ALL unpenalised points, not only the in-bounds
        # ones: a measurement sitting just outside the box edge is still a repeat of
        # the same composition. Penalised points are excluded — they belong to an
        # already-declared needle, and confirming a new needle with them would
        # re-declare that one.
        all_unpen = torch.where(mask)[0]
        dist = torch.norm(dh.X_all_actual[all_unpen] - best_X.unsqueeze(0), dim=1)
        radius = self._repeat_radius(bounds)
        near = all_unpen[dist <= radius]
        n_others = int(near.numel()) - 1  # the candidate itself is in `near`
        if n_others < self.needle_min_repeats:
            self._log(
                f"  [repeat-gate] best in-bounds point has {n_others} repeat(s) within "
                f"{radius:.4f} (need {self.needle_min_repeats}) — keep sampling."
            )
            return None, None, None, "not_repeatable"

        Y_near = dh.Y_all[near].reshape(-1)
        median = Y_near.median()
        pick = near[int((Y_near - median).abs().argmin().item())]
        self._log(
            f"  [repeat-gate] {n_others} repeat(s) within {radius:.4f}; "
            f"cluster median Y={median.item():.4f}, best Y={Y_near.max().item():.4f} — "
            f"declaring the median-closest repeat (Y={dh.Y_all[pick].reshape(-1)[0].item():.4f})."
        )
        return dh.X_all_actual[pick], dh.Y_all[pick], int(pick.item()), "ok"

    def _declare_needle_at_best(
        self,
        dh,
        zoom: int,
        iteration: int,
        reason: str = "converged",
        bounds: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Declare a needle in the active region, subject to the repeatability gate.

        The point comes from ``_select_needle_point`` (in-bounds argmax, then the
        median-closest repeat of its cluster); ``self.last_needle_status`` records
        why a declaration was refused so callers can tell "nothing here" from
        "not confirmed yet" — the first ends the zoom, the second keeps sampling.

        Delegates to ``_declare_needle_from_point``.  Returns the needle tensor,
        or None when no needle could be declared.
        """
        needle_X, needle_Y, global_idx, status = self._select_needle_point(dh, bounds)
        self.last_needle_status = status
        if needle_X is None:
            if status == "no_points":
                self._log("  [declare_needle] no unpenalized points in the active "
                          "region — cannot declare needle.")
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
            measurement was made (caller should ``continue``), False if nothing could
            be measured (caller should stop, which only happens on a degenerate
            simplex that can produce no point at all)."""
            nonlocal global_iteration, stalled_retries, bounds, best_f_local
            nonlocal start_iteration, data_added_since_last_failure
            nonlocal lines_this_activation
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

        # Objective lines measured in the current activation; reset per activation
        # and checked against self.max_lines_per_activation. Counts every measured
        # line including space-filling fallbacks, and is deliberately NOT reset by
        # the failure-retry path — re-entering the same zoom is precisely the route
        # by which an activation used to run away to 130+ lines.
        lines_this_activation = 0

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
            activation_capped = False
            # Set when the zoom loop runs out of unsearched territory (see the MC
            # Jaccard guard). Ends the activation on the same terms as the line cap.
            no_novel_window = False
            consecutive_converged = 0
            best_f_local: float = float('-inf')
            zoom_bounds_history: List[torch.Tensor] = []
            lines_this_activation = 0

            start_zoom = zoom if activation == start_activation else 0
            current_zoom = start_zoom

            while current_zoom < dh.max_zooms and not finished:
                # Line-budget guard at the top of the zoom loop. Placed here rather
                # than only in the iteration loop because the failure-retry path
                # `continue`s straight back to this point, and determine_new_bounds
                # advances the zoom from here too — every route that would buy this
                # activation another max_iterations lines passes through here.
                if lines_this_activation >= self.max_lines_per_activation:
                    activation_capped = True
                    break
                zoom = current_zoom
                self._log(f"\n{'─'*50}")
                self._log(f"--- Zoom {zoom+1}/{dh.max_zooms} ---")
                self._record_convergence(activation=activation, zoom=zoom,
                                         event="zoom_entry")
                self._log(f"Search bounds: [{bounds[0].cpu().numpy()}] – [{bounds[1].cpu().numpy()}]")

                # Use local GP data when zoomed in so GP posterior is tight
                # within the active region and EI drops to signal convergence.
                _t0 = time.time()
                if self._is_global_bounds(bounds):
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
                    if lines_this_activation >= self.max_lines_per_activation:
                        activation_capped = True
                        break
                    iteration = _next_iteration
                    _next_iteration += 1
                    self._log(f"\n  · iter {iteration+1}/{dh.max_iterations}  "
                              f"(activation lines {lines_this_activation}/"
                              f"{self.max_lines_per_activation})")

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
                        self._record_convergence(activation=activation, zoom=zoom,
                                                 iteration=iteration, measured=False,
                                                 event="candidate_none")
                        break

                    # Local reference point: best unpenalized within the active
                    # bounds so convergence is measured against local progress,
                    # not a global high from a different region.
                    if self._is_global_bounds(bounds):
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
                    lines_this_activation += 1
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
                    if self._is_global_bounds(bounds):
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
                        self._record_convergence(activation=activation, zoom=zoom,
                                                 iteration=iteration, measured=True,
                                                 event="all_penalized")
                        break

                    # Same locality as prev_best_* above, so the logged "current
                    # max" tracks the region convergence is actually judged in.
                    if self._is_global_bounds(bounds):
                        curr_best_X, curr_best_Y, _ = dh.get_best_unpenalized()
                    else:
                        curr_best_X, curr_best_Y, _ = dh.get_best_in_bounds(bounds)

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
                    self._record_convergence(activation=activation, zoom=zoom,
                                             iteration=iteration, measured=True,
                                             converged=bool(converged),
                                             counter=int(consecutive_converged),
                                             ei=float(ei))

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
                                dh, zoom, global_iteration, reason="EI convergence",
                                bounds=bounds,
                            )
                            if needle is not None:
                                dh.take_snapshot(
                                    f"act{activation}_z{zoom}_i{iteration}_needle", permanent=True
                                )
                                break
                            if self.last_needle_status == "not_repeatable":
                                # Converged on a point the neighbourhood has not
                                # confirmed. Stay in this zoom and keep measuring:
                                # the counter is left standing, so the gate is
                                # re-tested on every subsequent line and declares
                                # as soon as the repeats arrive.
                                self._record_convergence(
                                    activation=activation, zoom=zoom,
                                    iteration=iteration, event="repeat_gate_hold")
                            else:
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

                # Cap reached without a needle — leave the zoom loop; the region is
                # penalised once, after it (the guard at the top of this loop exits
                # the same way, so handling it here as well would double-penalise).
                if activation_capped and needle is None:
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
                            full_bounds = self.full_bounds.clone()
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
                    if self._is_global_bounds(bounds):
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

                    # MC Jaccard guard: the next zoom box still heavily overlaps one
                    # this activation already searched (determine_new_bounds may have
                    # exhausted all its windows), so there is NO NOVEL WINDOW left.
                    # Zooming into it again just re-measures the same region, and it
                    # was previously a licence to FORCE-declare a needle there — which
                    # manufactured a needle out of "I have nowhere else to go", the
                    # weakest possible evidence. Now the activation simply ends and
                    # the region is penalised, exactly as a line-capped activation is,
                    # so the next activation is repelled from it instead of grinding
                    # back in. A genuine optimum here still gets declared: it goes
                    # through EI convergence and the repeatability gate like any other.
                    #
                    # The one exception is a zoom too shallow to declare in at all
                    # (``min_zoom_for_needle``): ending there would abandon the region
                    # before the forced zoom-in that is the only way it could ever
                    # produce a needle, so the search advances the zoom instead.
                    if not self._is_global_bounds(new_bounds):
                        repeated_jac = 0.0
                        for prev_bounds in zoom_bounds_history:
                            jac = _bounds_jaccard_simplex(
                                new_bounds, prev_bounds, device=self.device, dtype=self.dtype
                            )
                            if jac > self.zoom_jaccard_threshold:
                                repeated_jac = jac
                                break
                        if (repeated_jac > self.zoom_jaccard_threshold
                                and zoom >= self.min_zoom_for_needle):
                            self._log(
                                f"  → no novel zoom window (Jaccard={repeated_jac:.3f} vs "
                                f"an already-searched box) — ending the activation."
                            )
                            self._record_convergence(activation=activation, zoom=zoom,
                                                     event="no_novel_window")
                            no_novel_window = True
                            break
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

            # Activation ended without a needle — either it spent its whole line
            # budget circling a region it never localised, or it ran out of novel
            # zoom windows there. Both mean the same thing: this region absorbed
            # measurements and produced nothing. Penalise it so the next activation
            # is repelled from it instead of converging straight back in, and hand
            # that activation the full simplex.
            if (activation_capped or no_novel_window) and needle is None and not finished:
                if activation_capped:
                    self._log(
                        f"\n  [cap] activation {activation+1} hit the "
                        f"{self.max_lines_per_activation}-line budget "
                        f"({lines_this_activation} measured) without declaring a "
                        f"needle — ending it and penalising the region."
                    )
                else:
                    self._log(
                        f"\n  [cap] activation {activation+1} ran out of novel zoom "
                        f"windows after {lines_this_activation} line(s) without "
                        f"declaring a needle — ending it and penalising the region."
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
