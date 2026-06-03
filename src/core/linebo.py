"""
LineBO: Line-based Bayesian Optimization for Simplex Constraints
================================================================

LineBO optimizes over the probability simplex by:
1. Sampling lines from the candidate point through random simplex points, extended to the boundary
   (guarantees valid segments), or alternatively zero-sum directions.
2. Finding each line's intersection with the simplex (segment endpoints on the boundary).
3. Integrating an acquisition function along segments and picking the best line.
4. Evaluating the objective along that line and returning requested/actual points and values.

All tensors use float64 for numerical stability. Shapes use k = number of lines, d = dimensions,
n = number of evaluation points.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from ..utils.simplex import (
    Ellipsoid,
    random_simplex,
    random_simplex_direction,
    sample_ellipsoid,
    proj_simplex,
)

# Backward-compatible aliases
random_zero_sum_directions = random_simplex_direction
zero_sum_dirs = random_simplex_direction

SAMPLER_MAX_EXTRA_ATTEMPTS = 50  # max extra direction batches in LineBO.sampler when valid lines < num_lines (default: 50)
MIN_DIRECTION_NORM = 1e-10        # directions with smaller norm (P ≈ x_tell) are dropped when using through-simplex sampling


def directions_through_simplex_points(x0: torch.Tensor, k: int, device=None, dtype=torch.float64,
                                     seed=None) -> torch.Tensor:
    """
    Generate directions from x0 through k random points on the simplex. Each line from x0 through
    a random point is extended to the simplex boundary, guaranteeing a valid segment.

    Parameters
    ----------
    x0 : torch.Tensor
        Candidate point (d,) on the simplex.
    k : int
        Number of directions to generate.
    device : str, optional
        Torch device.
    dtype : torch.dtype
        Data type.
    seed : int, optional
        Random seed (not used; sample_simplex may use global state).

    Returns
    -------
    torch.Tensor
        (k, d) tensor of unit-norm direction vectors. May have fewer than k rows if too many
        random points coincided with x0 (resamples once to try to fill).
    """
    d = x0.shape[0]
    x0 = x0.to(device=device, dtype=dtype)
    # Sample more than k points so we can drop any that equal x0 (or are too close)
    n_sample = max(k * 2, k + 10)
    zeros = torch.zeros(d, device=device, dtype=dtype)
    ones = torch.ones(d, device=device, dtype=dtype)
    P = random_simplex(n_sample, zeros, ones, S=1.0, device=device, torch_dtype=dtype)  # (n_sample, d)
    D = P - x0.unsqueeze(0)  # (n_sample, d)
    norms = D.norm(dim=1, keepdim=True)
    keep = (norms.squeeze(1) > MIN_DIRECTION_NORM)
    D = D[keep] / norms[keep]
    if D.shape[0] >= k:
        return D[:k]
    if D.shape[0] == 0:
        return torch.zeros(0, d, device=device, dtype=dtype)
    # Resample once to try to reach k
    P2 = random_simplex(n_sample, zeros, ones, S=1.0, device=device, torch_dtype=dtype)
    D2 = P2 - x0.unsqueeze(0)
    norms2 = D2.norm(dim=1, keepdim=True)
    keep2 = (norms2.squeeze(1) > MIN_DIRECTION_NORM)
    D2 = D2[keep2] / norms2[keep2]
    D = torch.cat([D, D2], dim=0)
    return D[:k] if D.shape[0] >= k else D



def line_simplex_segment(x0: torch.Tensor, d: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Find the intersection of a line with the simplex.

    Given a starting point x0 on the simplex and a direction d,
    find the segment of the line x0 + t*d that lies within the simplex.

    Parameters
    ----------
    x0 : torch.Tensor
        Starting point (d,) on the simplex.
    d : torch.Tensor
        Direction vector (d,).

    Returns
    -------
    tuple or None
        (t_min, t_max, x_left, x_right) if line intersects simplex,
        None if no intersection.
    """
    # Which components of d are negative / positive (for boundary hit times)
    neg = d < 0   # (d,) bool
    pos = d > 0   # (d,) bool

    inf = torch.tensor(float('inf'), device=d.device, dtype=d.dtype)   # scalar
    ninf = torch.tensor(float('-inf'), device=d.device, dtype=d.dtype)  # scalar

    # For x0 + t*d to stay in simplex [0,1]^d: each coord must stay >= 0.
    # Coord j hits 0 when t = -x0[j]/d[j] (only if d[j]!=0). Take min over neg d, max over pos d.
    t_max = torch.min(torch.where(neg, -x0 / d, inf))   # scalar: largest t keeping all coords >= 0
    t_min = torch.max(torch.where(pos, -x0 / d, ninf))  # scalar: smallest t keeping all coords >= 0

    if t_min > t_max:
        return None  # Line does not intersect simplex

    # Endpoints of the segment on the simplex boundary
    x_left = x0 + t_min * d   # (d,)
    x_right = x0 + t_max * d  # (d,)
    return t_min, t_max, x_left, x_right


def batch_line_simplex_segments(x0: torch.Tensor, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find intersections of multiple lines with the simplex (batched).

    Parameters
    ----------
    x0 : torch.Tensor
        Starting point (d,) on the simplex.
    D : torch.Tensor
        Direction vectors (k, d).

    Returns
    -------
    tuple
        (x_left, x_right, t_min, t_max, mask) where mask indicates valid lines.
    """
    neg = D < 0   # (k, d) bool: which direction components are negative
    pos = D > 0   # (k, d) bool: which direction components are positive

    inf = torch.tensor(float('inf'), device=D.device, dtype=D.dtype)   # scalar
    ninf = torch.tensor(float('-inf'), device=D.device, dtype=D.dtype)  # scalar

    # Per-line: -x0[j]/D[i,j] where D[i,j]!=0; broadcast x0 (d,) to (k, d) via unsqueeze(0)
    t_max = torch.min(torch.where(neg, -x0.unsqueeze(0)/D, inf), dim=1).values   # (k,)
    t_min = torch.max(torch.where(pos, -x0.unsqueeze(0)/D, ninf), dim=1).values  # (k,)

    # Only count segments with positive length (t_min < t_max); reject degenerate/vertex cases
    mask = t_min < t_max   # (k,) bool: valid lines
    t_min = t_min[mask]    # (num_valid,)
    t_max = t_max[mask]    # (num_valid,)
    Dm = D[mask]           # (num_valid, d)

    # Left/right endpoints on simplex boundary: x0 + t*d for each valid line
    x_left = x0.unsqueeze(0) + t_min.unsqueeze(1) * Dm   # (num_valid, d)
    x_right = x0.unsqueeze(0) + t_max.unsqueeze(1) * Dm  # (num_valid, d)

    return x_left, x_right, t_min, t_max, mask


def batch_line_bounds_segments(
    x0: torch.Tensor,
    D: torch.Tensor,
    bounds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find intersections of multiple lines with an axis-aligned zoomed box (batched).

    Clips each line x0 + t*D[i] to the box [bounds[0], bounds[1]] instead of
    the full simplex boundary.  Zero-sum directions preserve the sum=1 constraint,
    so every point on a clipped segment is still on the simplex.

    Parameters
    ----------
    x0 : torch.Tensor
        Starting point (d,) — must lie inside the box.
    D : torch.Tensor
        Direction vectors (k, d).
    bounds : torch.Tensor
        (2, d) tensor: bounds[0] = lower, bounds[1] = upper.

    Returns
    -------
    tuple
        (x_left, x_right, t_min, t_max, mask) same convention as
        batch_line_simplex_segments, where mask marks valid (non-degenerate) lines.
    """
    lo = bounds[0]   # (d,)
    hi = bounds[1]   # (d,)

    pos = D > 0    # (k, d)
    neg = D < 0    # (k, d)

    inf  = torch.tensor(float('inf'),  device=D.device, dtype=D.dtype)
    ninf = torch.tensor(float('-inf'), device=D.device, dtype=D.dtype)

    # Upper bound on t from each component:
    #   d[i] > 0  →  x0[i] + t*d[i] <= hi[i]  →  t <= (hi[i]-x0[i])/d[i]
    #   d[i] < 0  →  x0[i] + t*d[i] >= lo[i]  →  t <= (lo[i]-x0[i])/d[i]
    t_max_hi = torch.where(pos, (hi - x0).unsqueeze(0) / D, inf)   # (k, d)
    t_max_lo = torch.where(neg, (lo - x0).unsqueeze(0) / D, inf)   # (k, d)
    t_max = torch.minimum(
        torch.min(t_max_hi, dim=1).values,
        torch.min(t_max_lo, dim=1).values,
    )  # (k,)

    # Lower bound on t from each component:
    #   d[i] > 0  →  x0[i] + t*d[i] >= lo[i]  →  t >= (lo[i]-x0[i])/d[i]
    #   d[i] < 0  →  x0[i] + t*d[i] <= hi[i]  →  t >= (hi[i]-x0[i])/d[i]
    t_min_lo = torch.where(pos, (lo - x0).unsqueeze(0) / D, ninf)  # (k, d)
    t_min_hi = torch.where(neg, (hi - x0).unsqueeze(0) / D, ninf)  # (k, d)
    t_min = torch.maximum(
        torch.max(t_min_lo, dim=1).values,
        torch.max(t_min_hi, dim=1).values,
    )  # (k,)

    mask = t_min < t_max   # (k,) bool: valid lines
    t_min = t_min[mask]
    t_max = t_max[mask]
    Dm = D[mask]

    x_left  = x0.unsqueeze(0) + t_min.unsqueeze(1) * Dm   # (num_valid, d)
    x_right = x0.unsqueeze(0) + t_max.unsqueeze(1) * Dm   # (num_valid, d)

    return x_left, x_right, t_min, t_max, mask


def batch_line_ellipsoid_segments(
    x0: torch.Tensor,
    D: torch.Tensor,
    ell: Ellipsoid,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Clip each line ``x0 + t D[i]`` to the intersection of the simplex and the ellipsoid
    ``{x : (B^T(x-c))^T M (B^T(x-c)) <= 1}``.
    """
    device, dtype = D.device, D.dtype
    k, d = D.shape
    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    ninf = torch.tensor(float("-inf"), device=device, dtype=dtype)

    neg = D < 0
    pos = D > 0
    t_max_s = torch.min(torch.where(neg, -x0.unsqueeze(0) / D, inf), dim=1).values
    t_min_s = torch.max(torch.where(pos, -x0.unsqueeze(0) / D, ninf), dim=1).values

    B = ell.B
    M = ell.M
    u0 = B.T @ (x0 - ell.c)
    V = D @ B
    VM = V @ M
    a = (VM * V).sum(dim=-1)
    Mu0 = M @ u0
    b = 2.0 * (V @ Mu0)
    cshift = (u0 * Mu0).sum() - 1.0

    eps = 1e-14
    t_lo_e = torch.full((k,), float("-inf"), device=device, dtype=dtype)
    t_hi_e = torch.full((k,), float("inf"), device=device, dtype=dtype)

    quad = a.abs() > eps
    lin = ~quad

    disc_full = b * b - 4.0 * a * cshift
    has_real = quad & (disc_full >= 0)

    if has_real.any():
        sd = torch.sqrt(disc_full[has_real])
        aq = a[has_real]
        bq = b[has_real]
        r1 = (-bq - sd) / (2.0 * aq)
        r2 = (-bq + sd) / (2.0 * aq)
        t_lo_e[has_real] = torch.minimum(r1, r2)
        t_hi_e[has_real] = torch.maximum(r1, r2)

    no_real = quad & (disc_full < 0)
    if no_real.any():
        aan = a[no_real].clamp(min=eps)
        bb = b[no_real]
        vtx = -bb / (2.0 * aan)
        qmin = aan * vtx * vtx + bb * vtx + cshift
        inside_par = qmin <= 0.0
        idx_nr = torch.where(no_real)[0]
        if inside_par.any():
            ii = idx_nr[inside_par]
            t_lo_e[ii] = ninf
            t_hi_e[ii] = inf
        if (~inside_par).any():
            ii = idx_nr[~inside_par]
            t_lo_e[ii] = inf
            t_hi_e[ii] = ninf

    if lin.any():
        b_l = b[lin]
        mask_b = b_l.abs() > eps
        if mask_b.any():
            idx = torch.where(lin)[0][mask_b]
            t_fix = -cshift / b_l[mask_b]
            t_lo_e[idx] = t_fix
            t_hi_e[idx] = t_fix
        idx0 = torch.where(lin)[0][~mask_b]
        if idx0.numel() > 0:
            if cshift <= 0:
                t_lo_e[idx0] = ninf
                t_hi_e[idx0] = inf
            else:
                t_lo_e[idx0] = inf
                t_hi_e[idx0] = ninf

    t_min = torch.maximum(t_min_s, t_lo_e)
    t_max = torch.minimum(t_max_s, t_hi_e)
    mask = t_min < t_max - 1e-10

    t_min = t_min[mask]
    t_max = t_max[mask]
    Dm = D[mask]
    x_left = x0.unsqueeze(0) + t_min.unsqueeze(1) * Dm
    x_right = x0.unsqueeze(0) + t_max.unsqueeze(1) * Dm
    return x_left, x_right, t_min, t_max, mask


def _clip_x_to_ellipsoid(x: torch.Tensor, ell: Ellipsoid) -> torch.Tensor:
    """If x lies outside the ellipsoid, project it radially to the boundary."""
    u = ell.B.T @ (x - ell.c)
    quad = (u * (ell.M @ u)).sum()
    if quad <= 1.0 + 1e-8:
        return x
    # Scale u to the boundary: u_clipped = u / sqrt(quad)  =>  u_clipped^T M u_clipped = 1
    u_clipped = u / quad.sqrt()
    return proj_simplex((ell.c + ell.B @ u_clipped).unsqueeze(0)).squeeze(0)


def random_chords_through_ellipsoid(
    k: int,
    ell: Ellipsoid,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate up to k random chord endpoint pairs within ellipsoid ∩ simplex.

    Each chord passes through a random interior point in a random zero-sum direction.
    Using random interior points (not a fixed anchor) gives uniform coverage of the
    ellipsoid rather than all chords radiating from a single point.
    """
    d = ell.c.shape[0]
    oversample = max(k * 3, k + 20)
    interior_pts = sample_ellipsoid(oversample, ell)  # (oversample, d)
    directions = random_simplex_direction(oversample, d, device=str(device), dtype=dtype)

    lefts: List[torch.Tensor] = []
    rights: List[torch.Tensor] = []

    for i in range(interior_pts.shape[0]):
        xl, xr, _, _, valid = batch_line_ellipsoid_segments(
            interior_pts[i], directions[i:i+1], ell
        )
        if valid.any():
            lefts.append(xl[0])
            rights.append(xr[0])
        if len(lefts) >= k:
            break

    if not lefts:
        return (
            torch.empty(0, d, device=device, dtype=dtype),
            torch.empty(0, d, device=device, dtype=dtype),
        )
    return torch.stack(lefts, dim=0), torch.stack(rights, dim=0)


def random_chords_through_simplex(
    k: int,
    bounds: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate up to k random chord endpoint pairs clipped to the current bounds.
    Interior anchor points are sampled from within the bounds; chords extend to
    the bounds boundary (not the full simplex), keeping evaluations inside the
    current zoom region.
    """
    d = bounds.shape[1]
    lo, hi = bounds[0], bounds[1]
    oversample = max(k * 3, k + 20)
    interior_pts = random_simplex(oversample, lo, hi, device=str(device), torch_dtype=dtype)
    directions = random_simplex_direction(oversample, d, device=str(device), dtype=dtype)

    lefts: List[torch.Tensor] = []
    rights: List[torch.Tensor] = []

    for i in range(interior_pts.shape[0]):
        xl, xr, _, _, valid = batch_line_bounds_segments(
            interior_pts[i], directions[i:i+1], bounds
        )
        if valid.any():
            lefts.append(xl[0])
            rights.append(xr[0])
        if len(lefts) >= k:
            break

    if not lefts:
        return (
            torch.empty(0, d, device=device, dtype=dtype),
            torch.empty(0, d, device=device, dtype=dtype),
        )
    return torch.stack(lefts, dim=0), torch.stack(rights, dim=0)


class LineBO:
    """
    Line-based Bayesian Optimization for simplex-constrained problems.

    Given a current point x_tell on the simplex, samples many zero-sum directions,
    finds the simplex-boundary segment for each, integrates the acquisition along
    each segment, picks the best line, and evaluates the objective on that line.
    Uses float64 throughout for numerical stability.

    Parameters
    ----------
    objective_function : Callable
        Function that accepts a single tensor of shape (num_lines, 2, d): all line
        endpoints ranked by acquisition (highest first). endpoints[i, 0] is left
        and endpoints[i, 1] is right for line i. Returns (x_actual, y) for the best line's evaluation.
    dimensions : int
        Number of dimensions (d).
    num_points_per_line : int
        Number of points to sample along each line for integration. Default: 100.
    num_lines : int or None
        Number of candidate lines to generate and evaluate. Default: None (auto: 10*d).
    device : str
        Device for computations. Default: 'cuda'.
    """

    def __init__(
        self,
        objective_function,
        dimensions: int,
        num_points_per_line: int = 100,
        num_lines: Optional[int] = None,
        device: str = "cuda",
    ):
        """Initialize LineBO sampler: store objective, dimensions, line-discretization and device."""
        self.objective_function = objective_function
        self.d = dimensions
        self.num_points_per_line = num_points_per_line
        self.num_lines = num_lines if num_lines is not None else 10 * dimensions
        self.device = torch.device(device)
        self.dtype = torch.float64

    def _integrate_acquisition_along_lines(self, x_left: torch.Tensor, x_right: torch.Tensor,
                                         acquisition_function: nn.Module) -> torch.Tensor:
        """
        Integrate acquisition function along multiple line segments (average over t in [0,1]).

        Parameters
        ----------
        x_left : torch.Tensor
            (k, d) left endpoints of line segments.
        x_right : torch.Tensor
            (k, d) right endpoints of line segments.
        acquisition_function : nn.Module
            Acquisition function to evaluate.

        Returns
        -------
        torch.Tensor
            (k,) integrated acquisition values (mean over points per line).
        """
        k = x_left.shape[0]  # number of lines

        # Parameter along each segment: t in [0, 1], num_points_per_line values
        t_values = torch.linspace(0, 1, self.num_points_per_line,
                                device=self.device, dtype=torch.float64)  # (num_points_per_line,)

        # Interpolate: (1-t)*x_left + t*x_right for each line and each t
        # x_left.unsqueeze(1) -> (k, 1, d), (1-t).view(1,-1,1) -> (1, num_points_per_line, 1); broadcast -> (k, num_points_per_line, d)
        points = (x_left.unsqueeze(1) * (1 - t_values).view(1, -1, 1) +
                 x_right.unsqueeze(1) * t_values.view(1, -1, 1))  # (k, num_points_per_line, d)

        # Flatten to (k * num_points_per_line, 1, d) for acquisition (often expects batch with last dim d)
        points_flat = points.reshape(-1, 1, self.d)  # (k * num_points_per_line, 1, d)

        batch_size = 500
        num_points = points_flat.shape[0]
        acquisition_values_flat = torch.empty(num_points, device=self.device, dtype=torch.float64)  # (num_points,)

        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                end_idx = min(i + batch_size, num_points)
                batch = points_flat[i:end_idx]  # (batch_len, 1, d)

                try:
                    batch_values = acquisition_function(batch)  # (batch_len, 1) or (batch_len,)
                    acquisition_values_flat[i:end_idx] = batch_values.squeeze()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        smaller_batch_size = batch_size // 4
                        for j in range(i, end_idx, smaller_batch_size):
                            small_end_idx = min(j + smaller_batch_size, end_idx)
                            small_batch = points_flat[j:small_end_idx]
                            small_batch_values = acquisition_function(small_batch)
                            acquisition_values_flat[j:small_end_idx] = small_batch_values.squeeze()
                    else:
                        raise e

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        # Reshape to (k, num_points_per_line) and take mean acquisition per line.
        # Mean (not max) is critical: with max, a line that barely grazes the unpenalized
        # fringe of an already-found needle still scores as high as the pure UCB at that
        # fringe point, because the -lambda*violation^2 penalties on the rest of the line
        # are never seen by argmax. Mean integrates penalties along the full line, so
        # lines that spend most of their length in penalized regions score very negatively.
        acquisition_values = acquisition_values_flat.reshape(k, self.num_points_per_line)  # (k, num_points_per_line)
        integrated = acquisition_values.mean(dim=1)  # (k,) — mean acquisition along each line

        return integrated

    def ranked_line_endpoints(
        self,
        x_tell: torch.Tensor,
        bounds: Optional[torch.Tensor] = None,
        acquisition_function: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return all evaluated line endpoints ranked by integrated acquisition (best first).
        Does not call the objective.

        Parameters
        ----------
        x_tell : torch.Tensor
            Starting point (d,) on simplex.
        bounds : torch.Tensor, optional
            (2, d) tensor: bounds[0] = lower, bounds[1] = upper.
            If None, uses the full simplex [0, 1]^d.
        acquisition_function : nn.Module, optional
            Acquisition function to rank lines. If None, random order.
        """
        x_tell = x_tell.to(device=self.device, dtype=self.dtype)

        assert abs(x_tell.sum().item() - 1.0) < 1e-12, f"x_tell must sum to 1, got {x_tell.sum().item()}"

        # Build effective bounds (full simplex if not provided)
        d = self.d
        if bounds is not None:
            bounds_eff = bounds.to(device=self.device, dtype=self.dtype)
        else:
            bounds_eff = torch.stack([
                torch.zeros(d, device=self.device, dtype=self.dtype),
                torch.ones(d, device=self.device, dtype=self.dtype),
            ], dim=0)

        x_left_parts: List[torch.Tensor] = []
        x_right_parts: List[torch.Tensor] = []

        rand_xl, rand_xr = random_chords_through_simplex(
            self.num_lines, bounds_eff, self.device, self.dtype
        )

        if rand_xl.shape[0] > 0:
            x_left_parts.append(rand_xl)
            x_right_parts.append(rand_xr)

        if not x_left_parts:
            return x_tell.unsqueeze(0), x_tell.unsqueeze(0)

        x_left = torch.cat(x_left_parts, dim=0)
        x_right = torch.cat(x_right_parts, dim=0)

        # Rank all candidates by acquisition (or random order), return top num_lines.
        if acquisition_function is None:
            order = torch.randperm(x_left.shape[0], device=self.device)
        else:
            integrated_values = self._integrate_acquisition_along_lines(
                x_left, x_right, acquisition_function
            )
            if integrated_values.numel() == 0:
                order = torch.arange(x_left.shape[0], device=self.device)
            else:
                order = torch.argsort(integrated_values, descending=True)

        order = order[: self.num_lines]
        return x_left[order], x_right[order]

    def sampler(
        self,
        x_tell: torch.Tensor,
        bounds: Optional[torch.Tensor] = None,
        acquisition_function: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points using LineBO: pick best line (by acquisition or random), evaluate objective, return requested/actual/y.

        Parameters
        ----------
        x_tell : torch.Tensor
            Starting point (d,) on simplex.
        bounds : torch.Tensor, optional
            (2, d) tensor trust region for segment clipping (see ``ranked_line_endpoints``).
        acquisition_function : nn.Module, optional
            Acquisition function to optimize. If None, random selection.

        Returns
        -------
        tuple
            (x_requested, x_actual, y) tensors.
        """
        x_left_ranked, x_right_ranked = self.ranked_line_endpoints(
            x_tell, bounds, acquisition_function
        )
        # (num_lines, 2, d): [i, 0] = left, [i, 1] = right, ranked best first
        endpoints_ranked = torch.stack([x_left_ranked, x_right_ranked], dim=1)
        x_actual, y = self.objective_function(endpoints_ranked)
        # x_actual: (n, d), y: (n,) or (n, 1)

        assert torch.is_tensor(x_actual) and x_actual.shape[1] == self.d
        assert torch.is_tensor(y) and y.shape[0] == x_actual.shape[0]

        x_actual = x_actual.to(device=self.device, dtype=self.dtype)  # (n, d)
        y = y.to(device=self.device, dtype=self.dtype)  # (n,) or (n, 1)

        # Build x_requested: points along the first principal direction of x_actual (or clone if single point)
        if x_actual.shape[0] > 1:
            x_centered = x_actual - x_actual.mean(dim=0, keepdim=True)  # (n, d)
            U, S, V = torch.svd(x_centered)  # U: (n, d), S: (d,), V: (d, d)
            direction = V[:, 0]  # (d,) first principal direction

            projections = torch.matmul(x_centered, direction.unsqueeze(1)).squeeze(1)  # (n,)

            t_vals = torch.linspace(projections.min(), projections.max(),
                                  x_actual.shape[0], device=self.device, dtype=torch.float64)  # (n,)

            x_requested = x_actual.mean(dim=0).unsqueeze(0) + t_vals.unsqueeze(1) * direction.unsqueeze(0)  # (n, d)
        else:
            x_requested = x_actual.clone()  # (1, d)

        return x_requested, x_actual, y
