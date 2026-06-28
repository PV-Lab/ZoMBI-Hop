"""
GP Simplex Handler
==================

Handles Gaussian Process fitting and candidate selection for simplex-constrained
optimization. Uses repulsive acquisition plus natural-gradient (Fisher-Rao) ascent
on the simplex for candidate optimization.
"""

import time
import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.errors import NotPSDError
from typing import Literal, Optional, Tuple, Callable, List

from .simplex import proj_simplex, random_simplex, Ellipsoid
from .datahandler import DataHandler


class CompWrappedAcquisition(nn.Module):
    """
    Wraps a BoTorch acquisition trained on normalized composition coordinates.

    Accepts simplex inputs (..., d), divides by per-dimension training std before
    calling the underlying BoTorch acquisition.
    """

    def __init__(self, base: nn.Module, comp_std: Optional[torch.Tensor] = None):
        super().__init__()
        self.base = base
        self.comp_std = comp_std  # (d,) or None

    def forward(self, Xq: torch.Tensor) -> torch.Tensor:
        shape = Xq.shape
        d = shape[-1]
        x_flat = Xq.reshape(-1, d)
        if self.comp_std is not None:
            x_flat = x_flat / self.comp_std
        return self.base(x_flat.reshape(*shape))


class RepulsiveAcquisition(nn.Module):
    """
    Acquisition function with smooth repulsion away from discovered needles.

    For needles that have an associated ellipsoid (M, B), the repulsion uses a
    Mahalanobis-distance violation: violation = max(0, 1 - u^T M u) where
    u = B^T (x - needle).  For needles with only a radius (no ellipsoid), the
    original Euclidean violation max(0, r - ||x - needle||) is used as a fallback.

    Parameters
    ----------
    base : nn.Module
        Base acquisition function (e.g., LogExpectedImprovement).
    proj_fn : Callable
        Projection function to simplex.
    needles : torch.Tensor
        Center points of penalty regions (num_needles, d).
    penalty_radii : torch.Tensor
        Radius for each needle (num_needles, 1) or (num_needles,). Used for
        needles without an ellipsoid.
    repulsion_lambda : float
        Strength of repulsion penalty. Default: 1000.0.
    needle_M_list : list of (torch.Tensor or None), optional
        Per-needle (d-1, d-1) Mahalanobis matrix in tangent space. None entries
        fall back to Euclidean sphere repulsion.
    needle_B : torch.Tensor or None, optional
        Shared (d, d-1) tangent basis for all ellipsoid needles.
    boundary_epsilon : float
        Decay length (in composition units) for the boundary penalty.
        penalty_i = exp(−x_i / boundary_epsilon), summed over components and
        scaled by repulsion_lambda.  0.0 disables boundary penalty.
    """

    def __init__(
        self,
        base: nn.Module,
        proj_fn: Callable,
        needles: torch.Tensor,
        penalty_radii: torch.Tensor,
        repulsion_lambda: float = 1000.0,
        needle_M_list: Optional[List] = None,
        needle_B: Optional[torch.Tensor] = None,
        boundary_epsilon: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.proj_fn = proj_fn
        self.needles = needles  # (M, d)
        self.penalty_radii = penalty_radii.view(-1)  # (M,)
        self.repulsion_lambda = repulsion_lambda
        self.needle_M_list = needle_M_list or []
        self.needle_B = needle_B  # (d, d-1) or None
        self.boundary_epsilon = boundary_epsilon

    def forward(self, Xq: torch.Tensor) -> torch.Tensor:
        """
        Evaluate acquisition with smooth repulsion.

        Parameters
        ----------
        Xq : torch.Tensor
            Query points (n, q, d) or (n, d).

        Returns
        -------
        torch.Tensor
            Acquisition values with repulsion penalty applied.
        """
        X_proj = self.proj_fn(Xq)
        base_acq = self.base(X_proj)

        has_needles = self.needles.shape[0] > 0
        has_boundary = self.boundary_epsilon > 0.0
        if not has_needles and not has_boundary:
            return base_acq

        if X_proj.ndim == 3:
            n, q, d = X_proj.shape
            X_flat = X_proj.reshape(-1, d)  # (B, d)
        else:
            X_flat = X_proj.reshape(-1, X_proj.shape[-1])

        total_violation = torch.zeros(X_flat.shape[0], device=X_flat.device, dtype=X_flat.dtype)

        # Needle repulsion
        if has_needles:
            for idx in range(self.needles.shape[0]):
                needle = self.needles[idx]  # (d,)
                diff = X_flat - needle.unsqueeze(0)  # (B, d)
                M = self.needle_M_list[idx] if idx < len(self.needle_M_list) else None

                if M is not None and self.needle_B is not None:
                    u = diff @ self.needle_B
                    quad = (u @ M * u).sum(dim=-1)
                    violation = torch.clamp(1.0 - quad, min=0.0)
                else:
                    r = self.penalty_radii[idx]
                    dist = torch.norm(diff, dim=-1)
                    violation = torch.clamp(r - dist, min=0.0)

                total_violation = total_violation + violation ** 2

        # Boundary repulsion: exp-decay from each simplex face (x_i = 0)
        if has_boundary:
            boundary_terms = torch.exp(-X_flat / self.boundary_epsilon).sum(dim=-1)  # (B,)
            total_violation = total_violation + boundary_terms

        penalty = (-self.repulsion_lambda * total_violation).view(base_acq.shape)
        return base_acq + penalty


class GPSimplex:
    """
    Gaussian Process handler for simplex-constrained optimization.

    Manages GP fitting, acquisition function creation, and candidate selection.

    Parameters
    ----------
    data_handler : DataHandler
        Data handler for accessing points and penalty info.
    proj_fn : Callable, optional
        Projection function to simplex. Default: proj_simplex.
    random_sampler : Callable, optional
        Random sampler for simplex. Default: random_simplex.
    num_restarts : int
        Number of restarts for acquisition optimization.
    raw_samples : int
        Number of raw samples for initial candidates.
    repulsion_lambda : float, optional
        Lambda for repulsive acquisition. If None, auto-computed dynamically
        as 10 * median(|acquisition_values|) when creating acquisition function.
        Default: None (auto).
    acquisition_type : str
        Base acquisition: "ucb" (Upper Confidence Bound) or "ei" (Expected Improvement).
        Both are wrapped with needle repulsion. Default: "ucb".
    ucb_beta : float
        Exploration weight for UCB (mean + beta * std). Only used when acquisition_type=="ucb".
        Default: 0.1.
    nat_grad_step : float
        Step size α for natural-gradient ascent: x ∝ x ⊙ exp(α(g − ḡ_x)).
        Default: 0.02.
    nat_grad_max_steps : int
        Maximum ascent steps per restart. Default: 50.
    device : str
        Torch device.
    dtype : torch.dtype
        Data type.
    """

    def __init__(
        self,
        data_handler: DataHandler,
        proj_fn: Optional[Callable] = None,
        random_sampler: Optional[Callable] = None,
        num_restarts: int = 30,
        raw_samples: int = 500,
        repulsion_lambda: Optional[float] = None,
        acquisition_type: Literal["ucb", "ei"] = "ucb",
        ucb_beta: float = 0.1,
        nat_grad_step: float = 0.02,
        nat_grad_max_steps: int = 50,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float64,
        verbose: bool = True,
    ):
        self.data_handler = data_handler
        self.proj_fn = proj_fn if proj_fn is not None else proj_simplex
        self.random_sampler = random_sampler if random_sampler is not None else random_simplex
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.repulsion_lambda = repulsion_lambda  # None means auto-compute
        self.acquisition_type = acquisition_type.lower()
        if self.acquisition_type not in ("ucb", "ei"):
            raise ValueError(f"acquisition_type must be 'ucb' or 'ei', got {acquisition_type!r}")
        self.ucb_beta = ucb_beta
        self.nat_grad_step = nat_grad_step
        self.nat_grad_max_steps = nat_grad_max_steps
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose

        self.gp: Optional[SingleTaskGP] = None
        self.mll = None
        self.acq_fn = None
        self._last_computed_lambda = None  # Track auto-computed lambda for logging
        self._tangent_basis: Optional[torch.Tensor] = None  # cached (d, d-1)
        self.comp_std: Optional[torch.Tensor] = None  # per-dimension composition std from last fit

    _JITTER_SCHEDULE = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Fit GP to data.

        Parameters
        ----------
        X : torch.Tensor
            Training inputs (n, d) on the simplex.
        Y : torch.Tensor
            Training outputs (n, 1).
        """
        X = X.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)

        comp_std = torch.nan_to_num(X.std(dim=0), nan=1.0).clamp(min=1e-3)
        self.comp_std = comp_std
        X_norm = X / comp_std

        _t0 = time.time()
        last_exc: Exception | None = None
        for jitter in [0.0, *self._JITTER_SCHEDULE]:
            try:
                if jitter > 0.0:
                    noise = torch.randn_like(X_norm) * jitter
                    X_fit = X_norm + noise
                else:
                    X_fit = X_norm
                self.gp = SingleTaskGP(X_fit, Y)
                self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
                fit_gpytorch_mll(self.mll)
                if jitter > 0.0:
                    print(f"  [GP.fit] recovered with jitter={jitter:.0e}")
                break
            except (NotPSDError, RuntimeError) as exc:
                last_exc = exc
                continue
        else:
            raise RuntimeError(
                f"GP fit failed after all jitter levels: {last_exc}"
            ) from last_exc
        if self.verbose:
            print(f"  [GP.fit] MLL: {time.time()-_t0:.2f}s  ({X.shape[0]} pts)")
        self.data_handler.update_gp_noise(self.get_output_noise())

    def fit_from_data_handler(self):
        """Fit GP using data from the data handler."""
        X, Y = self.data_handler.get_gp_data()
        self.fit(X, Y)

    def _gp_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize compositions for GP / BoTorch acquisition evaluation."""
        if self.comp_std is not None:
            return X / self.comp_std
        return X

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the GP.

        Parameters
        ----------
        X : torch.Tensor
            Query points (n, d).

        Returns
        -------
        tuple
            (mean, variance) tensors.
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        X = X.to(device=self.device, dtype=self.dtype)
        x_norm = self._gp_inputs(X)
        try:
            posterior = self.gp.posterior(x_norm)
        except (NotPSDError, RuntimeError):
            mean = torch.zeros(X.shape[0], 1, device=self.device, dtype=self.dtype)
            var = torch.ones(X.shape[0], 1, device=self.device, dtype=self.dtype)
            return mean, var
        return posterior.mean, posterior.variance

    def get_output_noise(self) -> float:
        """Get average output noise from the GP."""
        if self.gp is None:
            return 0.0
        return self.gp.likelihood.noise_covar.noise.mean().item()

    def probability_of_improvement(self, x: torch.Tensor, best_f: float) -> float:
        """
        P(f(x) > best_f) under the GP posterior at x.

        Parameters
        ----------
        x : torch.Tensor
            Query point (d,) or (1, d).
        best_f : float
            Current best observed value.

        Returns
        -------
        float
            Probability of improvement, in [0, 1].
        """
        if self.gp is None:
            return 0.0
        x_2d = x.unsqueeze(0) if x.dim() == 1 else x
        with torch.no_grad():
            x_norm = self._gp_inputs(x_2d)
            try:
                posterior = self.gp.posterior(x_norm)
            except (NotPSDError, RuntimeError):
                return 0.0
            mu = posterior.mean.squeeze().item()
            var = posterior.variance.squeeze().item()
        sigma = max(var ** 0.5, 1e-9)
        z = (mu - best_f) / sigma
        return torch.distributions.Normal(0.0, 1.0).cdf(torch.tensor(z, device=self.device)).item()

    def compute_log_ei_at_point(self, x: torch.Tensor, best_f: float) -> float:
        """
        Log Expected Improvement at x for maximization (best_f = current best).

        Parameters
        ----------
        x : torch.Tensor
            Query point (d,) or (1, d).
        best_f : float
            Current best observed value.

        Returns
        -------
        float
            log(EI(x)).
        """
        if self.gp is None:
            return float('-inf')
        x_1d = x.reshape(-1)
        x_norm = self._gp_inputs(x_1d.unsqueeze(0))  # (1, d)
        x_norm_3d = x_norm.unsqueeze(0)  # (1, 1, d)
        try:
            if self.acquisition_type == "ei":
                base_acq = LogExpectedImprovement(self.gp, best_f=best_f)
                with torch.no_grad():
                    val = base_acq(x_norm_3d).squeeze().item()
                return val
            else:
                base_acq = UpperConfidenceBound(self.gp, beta=self.ucb_beta)
                with torch.no_grad():
                    val = base_acq(x_norm_3d).squeeze().item()
                return val
        except (NotPSDError, RuntimeError):
            return float('-inf')

    def create_acquisition(
        self,
        best_f: Optional[float] = None,
        penalty_value: Optional[float] = None,
    ) -> nn.Module:
        """
        Create acquisition function (base + repulsion).

        Parameters
        ----------
        best_f : float, optional
            Best function value so far. Used only for EI. If None, computed from data.
        penalty_value : float, optional
            Unused, kept for API compatibility.

        Returns
        -------
        nn.Module
            Acquisition function (UCB or EI, wrapped with needle repulsion).
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        if self.acquisition_type == "ucb":
            base_botorch = UpperConfidenceBound(self.gp, beta=self.ucb_beta)
        else:
            if best_f is None:
                _, Y = self.data_handler.get_gp_data()
                best_f = Y.max().item()
            base_botorch = LogExpectedImprovement(self.gp, best_f=best_f)

        base_acq = CompWrappedAcquisition(base_botorch, comp_std=self.comp_std)

        # Auto-compute repulsion_lambda if not provided
        if self.repulsion_lambda is None:
            _t0 = time.time()
            computed_lambda = self._compute_repulsion_lambda(base_acq)
            self._last_computed_lambda = computed_lambda
            if self.verbose:
                print(f"  [GP.acq] repulsion_lambda={computed_lambda:.2f}  ({time.time()-_t0:.2f}s)")
        else:
            computed_lambda = self.repulsion_lambda

        needles, penalty_radii = self.data_handler.get_needles_and_penalty_radii()
        needle_M_list, needle_B = self.data_handler.get_needle_ellipsoids()
        self.acq_fn = RepulsiveAcquisition(
            base=base_acq,
            proj_fn=self.proj_fn,
            needles=needles,
            penalty_radii=penalty_radii,
            repulsion_lambda=computed_lambda,
            needle_M_list=needle_M_list,
            needle_B=needle_B,
            boundary_epsilon=self.data_handler.input_noise,
        )

        return self.acq_fn

    def _compute_repulsion_lambda(self, base_acq: nn.Module, n_samples: int = 100) -> float:
        """
        Auto-compute repulsion_lambda based on acquisition function scale.

        Uses 10 * median(|acquisition_values|) to ensure repulsion is strong
        enough relative to the acquisition function magnitude.

        Parameters
        ----------
        base_acq : nn.Module
            Base acquisition function to evaluate.
        n_samples : int
            Number of samples to estimate scale. Default: 100.

        Returns
        -------
        float
            Computed repulsion lambda value.
        """
        # Sample random points on the simplex within current bounds
        tb = self.data_handler.bounds
        if tb is not None and isinstance(tb, torch.Tensor):
            samples = self.random_sampler(n_samples, tb[0], tb[1], device=str(self.device), torch_dtype=self.dtype)
        else:
            d = self.data_handler.d
            lo = torch.zeros(d, device=self.device, dtype=self.dtype)
            hi = torch.ones(d, device=self.device, dtype=self.dtype)
            samples = self.random_sampler(n_samples, lo, hi, device=str(self.device), torch_dtype=self.dtype)
        samples_3d = samples.unsqueeze(1)  # (n_samples, 1, d)

        # Evaluate base acquisition
        with torch.no_grad():
            acq_values = base_acq(samples_3d).squeeze()

        # Compute lambda as 10 * median(|acq_values|)
        # Use absolute value since LogEI can be negative
        median_abs_acq = torch.median(torch.abs(acq_values)).item()

        # Ensure minimum lambda to avoid numerical issues
        computed_lambda = max(10.0 * median_abs_acq, 100.0)

        return computed_lambda

    def get_last_computed_lambda(self) -> Optional[float]:
        """Get the last auto-computed repulsion_lambda value, if any."""
        return self._last_computed_lambda

    def _sample_random(self, n: int, bounds: torch.Tensor) -> torch.Tensor:
        """Sample random points from bounded simplex."""
        return self.random_sampler(n, bounds[0], bounds[1], device=str(self.device), torch_dtype=self.dtype)

    def get_candidate(
        self,
        bounds: torch.Tensor,
        best_f: Optional[float] = None,
        max_attempts: int = 5,
        exclude_near: Optional[torch.Tensor] = None,
        exclude_near_tol: float = 1e-8,
    ) -> Optional[torch.Tensor]:
        """
        Get next candidate point to evaluate.

        Uses natural-gradient ascent on the simplex to optimize the acquisition.
        If exclude_near is set, the best candidate
        that is not within exclude_near_tol of exclude_near is returned (no two
        same points in a row).

        Parameters
        ----------
        bounds : torch.Tensor
            (2, d) tensor: bounds[0] = lower, bounds[1] = upper.
        best_f : float, optional
            Best function value so far.
        max_attempts : int
            Maximum attempts to find unpenalized candidates.
        exclude_near : torch.Tensor, optional
            Last suggested point(s) to avoid repeating. Shape (d,) or (K, d).
            If not None, skip candidates within exclude_near_tol of *any* of these
            points so we don't bounce between the same few local maxima.
        exclude_near_tol : float
            Minimum distance from each exclude_near point for a candidate to be allowed. Default 1e-8.

        Returns
        -------
        torch.Tensor or None
            Candidate point (d,), or None if no valid candidate found.
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        bounds = bounds.to(device=self.device, dtype=self.dtype)

        _t_total = time.time()

        try:
            return self._get_candidate_inner(bounds, best_f, max_attempts,
                                             exclude_near, exclude_near_tol,
                                             _t_total)
        except (NotPSDError, RuntimeError) as exc:
            if "singular" in str(exc).lower() or "cholesky" in str(exc).lower() or isinstance(exc, NotPSDError):
                print(f"  [GP] get_candidate: numerical failure ({type(exc).__name__}), skipping iteration")
                return None
            raise

    def _get_candidate_inner(self, bounds, best_f, max_attempts,
                             exclude_near, exclude_near_tol, _t_total):
        # Create acquisition function
        acq = self.create_acquisition(best_f=best_f)

        # Sample initial candidates
        _t0 = time.time()
        ic_candidates = self._sample_random(self.raw_samples, bounds)
        ic_candidates_3d = ic_candidates.unsqueeze(1)  # (raw, 1, d)

        # Evaluate acquisition
        with torch.no_grad():
            # reshape(-1), not squeeze(): when raw_samples == 1 the acq output
            # has a single element and .squeeze() would collapse it to a 0-dim
            # scalar, which then fails to torch.cat with additional batches below
            # ("zero-dimensional tensor (at position 0) cannot be concatenated").
            acq_values = acq(ic_candidates_3d).reshape(-1)
        if self.verbose:
            print(f"  [GP.cand] sample+eval: {time.time()-_t0:.2f}s")

        # Find unpenalized candidates
        _t0 = time.time()
        unpenalized_mask = self.data_handler.get_penalty_mask(ic_candidates)
        unpenalized_indices = torch.where(unpenalized_mask.squeeze())[0]

        if self.verbose:
            print(f"  [GP.cand] penalty_mask: {time.time()-_t0:.2f}s  ({unpenalized_mask.sum().item()}/{ic_candidates.shape[0]} unpenalized)")

        # Try to get enough unpenalized candidates
        current_candidates = ic_candidates
        current_candidates_3d = ic_candidates_3d
        current_acq_values = acq_values
        current_unpenalized_indices = unpenalized_indices

        attempt = 0
        while len(current_unpenalized_indices) < self.num_restarts and attempt < max_attempts:
            attempt += 1

            additional_points = self._sample_random(self.raw_samples, bounds)
            additional_points_3d = additional_points.unsqueeze(1)

            with torch.no_grad():
                additional_acq_values = acq(additional_points_3d).reshape(-1)  # see reshape note above

            additional_unpenalized_mask = self.data_handler.get_penalty_mask(additional_points)
            additional_unpenalized_indices = torch.where(additional_unpenalized_mask.squeeze())[0]

            # Offset indices for concatenation
            offset = current_candidates.shape[0]
            additional_unpenalized_indices_offset = additional_unpenalized_indices + offset

            current_candidates = torch.cat([current_candidates, additional_points], dim=0)
            current_candidates_3d = torch.cat([current_candidates_3d, additional_points_3d], dim=0)
            current_acq_values = torch.cat([current_acq_values, additional_acq_values], dim=0)
            current_unpenalized_indices = torch.cat([current_unpenalized_indices, additional_unpenalized_indices_offset], dim=0)

        # Check if we have enough unpenalized candidates
        if len(current_unpenalized_indices) == 0:
            return None  # No valid candidates found

        if len(current_unpenalized_indices) < 0.1 * self.num_restarts:
            return None  # Not enough unpenalized area

        num_restarts_to_use = min(self.num_restarts, len(current_unpenalized_indices))

        if self.verbose:
            print(
                f"  [GP] running nat_grad: {num_restarts_to_use} restarts "
                f"× {self.nat_grad_max_steps} steps …"
            )
        # Select top candidates by acquisition value
        unpenalized_acq_values = current_acq_values[current_unpenalized_indices]
        top_unpenalized_indices = torch.argsort(unpenalized_acq_values, descending=True)[:num_restarts_to_use]
        selected_indices = current_unpenalized_indices[top_unpenalized_indices]
        initial_conditions = current_candidates_3d[selected_indices]  # (num_restarts, 1, d)

        _t_ng = time.time()
        candidates, values = self._optimize_acquisition(
            acq=acq,
            bounds=bounds,
            initial_conditions=initial_conditions,
        )
        if self.verbose:
            print(f"  [GP] nat_grad done ({initial_conditions.shape[0]} restarts × {self.nat_grad_max_steps} steps)  {time.time()-_t_ng:.2f}s")

        if candidates.shape[0] == 0:
            return None

        # Sort by acquisition value descending (best first)
        order = torch.argsort(values, descending=True)
        candidates = candidates[order]
        values = values[order]

        # If exclude_near set, pick first candidate not within tol of *any* excluded point
        if exclude_near is not None:
            exclude_near = exclude_near.to(device=candidates.device, dtype=candidates.dtype)
            if exclude_near.dim() == 1:
                exclude_near = exclude_near.unsqueeze(0)  # (1, d)
            # candidates (R, d), exclude_near (K, d) -> distances (R, K)
            distances = torch.norm(
                candidates.unsqueeze(1) - exclude_near.unsqueeze(0), dim=2
            )
            allowed = (distances >= exclude_near_tol).all(dim=1)  # (R,)
            if allowed.any():
                # Take best allowed candidate
                best_idx = torch.where(allowed)[0][0]
                best_candidate = candidates[best_idx]
                best_value = values[best_idx]
            else:
                # All restarts converged to excluded points; fall back to best so we don't stall
                best_candidate = candidates[0]
                best_value = values[0]
        else:
            best_candidate = candidates[0]
            best_value = values[0]

        if best_candidate is not None and self.verbose:
            print(f"  [GP] get_candidate: best_candidate = {best_candidate.cpu().numpy()}, best_acq_value = {best_value.item():.6f}  total={time.time()-_t_total:.2f}s")

        # Verify the candidate is not penalized
        is_valid = self.data_handler.get_penalty_mask(best_candidate.unsqueeze(0))
        if not is_valid.any():
            return None

        return best_candidate

    def _optimize_acquisition(
        self,
        acq: nn.Module,
        bounds: torch.Tensor,
        initial_conditions: torch.Tensor,
        nat_grad_step: Optional[float] = None,
        nat_grad_max_steps: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Natural-gradient ascent on the simplex, batched across all restarts.

        All R restarts are advanced in a single forward+backward pass per step.
        This reduces the per-step cost from O(R × n²) to O(n²) compared to the
        sequential loop, where n is the GP training-set size.

        Update rule (per restart): g = ∇acq(x), ḡ = Σᵢ xᵢ gᵢ,
        then x ← normalize(x ⊙ exp(α(g − ḡ))), clamped to [lo, hi].
        """
        if "step_size" in kwargs:
            ss = kwargs.pop("step_size")
            if nat_grad_step is None:
                nat_grad_step = ss
        if "max_steps" in kwargs:
            ms = kwargs.pop("max_steps")
            if nat_grad_max_steps is None:
                nat_grad_max_steps = ms
        if kwargs:
            raise TypeError(f"_optimize_acquisition got unexpected keyword arguments: {set(kwargs)}")

        step = self.nat_grad_step if nat_grad_step is None else float(nat_grad_step)
        n_steps = self.nat_grad_max_steps if nat_grad_max_steps is None else int(nat_grad_max_steps)

        lo = bounds[0].to(device=self.device, dtype=self.dtype)
        hi = bounds[1].to(device=self.device, dtype=self.dtype)
        d = initial_conditions.shape[-1]
        R = initial_conditions.shape[0]

        if R == 0:
            return (
                torch.empty(0, d, device=self.device, dtype=self.dtype),
                torch.empty(0, device=self.device, dtype=self.dtype),
            )

        # Initialise all restarts as (R, d) on the simplex.
        x = initial_conditions[:, 0, :].clone().to(device=self.device, dtype=self.dtype)
        x = self.proj_fn(x)  # (R, d) — project onto simplex

        # alive[r] = True while restart r has not degenerated.
        alive = torch.ones(R, dtype=torch.bool, device=self.device)

        for _ in range(n_steps):
            alive_idx = torch.where(alive)[0]  # (A,)
            if alive_idx.numel() == 0:
                break

            # One forward+backward for ALL alive restarts simultaneously.
            x_active = x[alive_idx].detach().requires_grad_(True)  # (A, d)
            try:
                val = acq(x_active.unsqueeze(1))   # (A,)
                val.sum().backward()               # single backward for all A
            except RuntimeError:
                break

            g = x_active.grad                      # (A, d)
            if g is None:
                break

            with torch.no_grad():
                xd = x_active.detach()
                g_bar = (xd * g).sum(dim=1, keepdim=True)         # (A, 1)
                shift = torch.clamp(step * (g - g_bar), -10.0, 10.0)
                x_new = xd * torch.exp(shift)                      # (A, d)

                s = x_new.sum(dim=1, keepdim=True)                 # (A, 1)
                ok = s.squeeze(1) >= 1e-12
                x_new = x_new / s.clamp(min=1e-12)                 # normalize to simplex
                x_new = torch.clamp(x_new, lo, hi)                 # enforce box bounds
                s2 = x_new.sum(dim=1, keepdim=True)
                ok &= s2.squeeze(1) >= 1e-12
                x_new = x_new / s2.clamp(min=1e-12)

                x[alive_idx[ok]] = x_new[ok]
                alive[alive_idx[~ok]] = False

        alive_idx = torch.where(alive)[0]
        if alive_idx.numel() == 0:
            return (
                torch.empty(0, d, device=self.device, dtype=self.dtype),
                torch.empty(0, device=self.device, dtype=self.dtype),
            )

        with torch.no_grad():
            try:
                final_vals = acq(x[alive_idx].unsqueeze(1))  # (A,)
            except RuntimeError:
                return (
                    torch.empty(0, d, device=self.device, dtype=self.dtype),
                    torch.empty(0, device=self.device, dtype=self.dtype),
                )

        return x[alive_idx], final_vals

    def _get_tangent_basis(self, d: int) -> torch.Tensor:
        """
        Return (d, d-1) orthonormal basis for the simplex tangent space
        {v : sum(v) = 0}.  Result is cached and reused while d is unchanged.
        """
        if self._tangent_basis is not None and self._tangent_basis.shape[0] == d:
            return self._tangent_basis
        P = torch.eye(d, device=self.device, dtype=self.dtype) - (1.0 / d)
        Q, _ = torch.linalg.qr(P)
        B = Q[:, :d - 1].contiguous()
        self._tangent_basis = B
        return B

    def determine_penalty_ellipsoid(
        self,
        needle: torch.Tensor,
        drop_fraction: float = 0.25,
        eigenvalue_floor: float = 1e-6,
        max_radius: float = 1.0,
        acq_fn: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the ellipsoidal penalty region via the tangent-space Hessian of the
        base (unpenalized) acquisition function at the needle.

        Returns (M, B) where a point x is inside the basin iff
            u = B^T (x - needle),  u^T M u <= 1.

        M is (d-1, d-1) in tangent space; B is the shared (d, d-1) basis.
        """
        d = needle.shape[0]
        needle = needle.detach().to(device=self.device, dtype=self.dtype)
        B = self._get_tangent_basis(d)

        # Base (unpenalized) acquisition — avoids repulsion contaminating curvature.
        if acq_fn is not None:
            base_acq = acq_fn
        else:
            if self.acq_fn is None:
                self.create_acquisition()
            base_acq = getattr(self.acq_fn, "base", self.acq_fn)

        u0 = torch.zeros(d - 1, device=self.device, dtype=self.dtype)

        def tilde_alpha_u(u: torch.Tensor) -> torch.Tensor:
            x = needle + u @ B.T
            x = self.proj_fn(x.unsqueeze(0)).squeeze(0)
            return base_acq(x.view(1, 1, d)).squeeze()

        _t0 = time.time()
        H = torch.autograd.functional.hessian(tilde_alpha_u, u0)  # (d-1, d-1)
        if self.verbose:
            print(f"  [GP.ellipsoid] Hessian: {time.time()-_t0:.2f}s")
        neg_H = -0.5 * (H + H.T)

        eigvals, eigvecs = torch.linalg.eigh(neg_H)
        eigvals = eigvals.clamp(min=eigenvalue_floor)

        alpha_peak = abs(base_acq(needle.view(1, 1, d)).squeeze().item())
        sigma = self.data_handler.get_input_noise()
        lambda_max = eigvals.max().item()

        Delta_acq = drop_fraction * alpha_peak
        Delta_noise = 0.5 * lambda_max * (3.0 * sigma) ** 2
        Delta = max(Delta_acq, Delta_noise, 1e-12)

        neg_H_clean = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        M = neg_H_clean / (2.0 * Delta)

        M_eigvals, M_eigvecs = torch.linalg.eigh(M)
        M_eigvals_clamped = M_eigvals.clamp(min=1.0 / (max_radius ** 2))
        if (M_eigvals_clamped != M_eigvals).any():
            radii_before = (1.0 / M_eigvals.clamp(min=1e-30).sqrt()).cpu().tolist()
            radii_after  = (1.0 / M_eigvals_clamped.sqrt()).cpu().tolist()
            print(f"  [ellipsoid] radius cap applied — semi-axes before: "
                  f"{[f'{r:.3f}' for r in radii_before]} → "
                  f"after: {[f'{r:.3f}' for r in radii_after]}")
        M = M_eigvecs @ torch.diag(M_eigvals_clamped) @ M_eigvecs.T

        return M, B

    def fit_with_locality(
        self,
        needle: torch.Tensor,
        X_all: torch.Tensor,
        Y_all: torch.Tensor,
        max_radius: float = 2.0,
        max_gp_points: int = 500,
    ):
        """
        Fit GP to a local subset of data centered on needle.

        Ranks all points by composition L2 distance from needle, takes all within
        max_radius (padded to max_gp_points with nearest outside points if the
        inside set is too small).  Operates on the raw X_all/Y_all arrays —
        no penalisation filter — to avoid bias near the needle.
        """
        needle = needle.detach().to(device=self.device, dtype=self.dtype)
        X_all = X_all.to(device=self.device, dtype=self.dtype)
        Y_all = Y_all.to(device=self.device, dtype=self.dtype)

        dists = torch.norm(X_all - needle.unsqueeze(0), dim=1)  # (n,)

        order = torch.argsort(dists)
        n_inside = int((dists <= max_radius).sum().item())
        n_take = max(min(max(n_inside, 1), max_gp_points, len(order)), 2)
        idx = order[:n_take]

        Y_local = Y_all[idx]
        if Y_local.dim() == 1:
            Y_local = Y_local.unsqueeze(1)
        self.fit(X_all[idx], Y_local)

    def create_clean_acquisition(self, best_f: Optional[float] = None) -> nn.Module:
        """
        Return a CompWrappedAcquisition with no RepulsiveAcquisition wrapper.

        Suitable as the acq_fn argument to determine_penalty_ellipsoid so that
        curvature estimates are not contaminated by the repulsion penalty.
        """
        if self.gp is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        if self.acquisition_type == "ucb":
            base_botorch = UpperConfidenceBound(self.gp, beta=self.ucb_beta)
        else:
            if best_f is None:
                _, Y = self.data_handler.get_gp_data()
                best_f = Y.max().item()
            base_botorch = LogExpectedImprovement(self.gp, best_f=best_f)

        return CompWrappedAcquisition(base_botorch, comp_std=self.comp_std)

    def recompute_all_ellipsoids(
        self,
        needles: torch.Tensor,
        X_all: torch.Tensor,
        Y_all: torch.Tensor,
        max_radius: float = 2.0,
        max_gp_points: int = 500,
        drop_fraction: float = 0.25,
        eigenvalue_floor: float = 1e-6,
        max_penalty_radius: float = 1.0,
    ) -> list:
        """
        Refit every needle's ellipsoid using a local, unpenalised GP.

        For each needle: fit_with_locality → create_clean_acquisition →
        determine_penalty_ellipsoid(acq_fn=clean_acq).

        Returns a list of (d-1, d-1) M tensors, one per needle, without
        modifying self.gp or self.acq_fn (both are restored after the loop).
        """
        saved_gp = self.gp
        saved_mll = self.mll
        saved_acq = self.acq_fn
        saved_comp_std = self.comp_std

        M_list = []
        B_shared = None
        try:
            for i in range(needles.shape[0]):
                needle = needles[i]
                self.fit_with_locality(needle, X_all, Y_all, max_radius, max_gp_points)
                clean_acq = self.create_clean_acquisition()
                M, B = self.determine_penalty_ellipsoid(
                    needle,
                    drop_fraction=drop_fraction,
                    eigenvalue_floor=eigenvalue_floor,
                    max_radius=max_penalty_radius,
                    acq_fn=clean_acq,
                )
                M_list.append(M)
                B_shared = B
        finally:
            self.gp = saved_gp
            self.mll = saved_mll
            self.acq_fn = saved_acq
            self.comp_std = saved_comp_std

        if B_shared is not None:
            self.data_handler.needle_B = B_shared

        return M_list
