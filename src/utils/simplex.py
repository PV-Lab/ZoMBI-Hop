"""
Simplex Utilities
=================

Helper functions for working with the simplex constraint
in materials research applications.

The simplex constraint ensures that compositions sum to 1:
    x_1 + x_2 + ... + x_d = 1, x_i >= 0
"""
from __future__ import annotations

import torch
import math
from collections import namedtuple
from typing import Tuple, Optional

Ellipsoid = namedtuple(
    "Ellipsoid",
    [
        "c",  # (d,) centre on Δ
        "M",  # (d-1,d-1) SPD precision — trust region u^T M u <= 1, u=B^T(x-c)
        "B",  # (d,d-1) orthonormal tangent-space columns (fixed for given d,dtype)
    ],
)


def get_tangent_basis(d: int, device, dtype=torch.float64) -> torch.Tensor:
    """
    QR basis for the centred hyperplane Σ v_i = 0: orthonormal columns (d × d−1).

    Matches the construction used historically in GPSimplex._get_tangent_basis.
    """
    eye = torch.eye(d, device=device, dtype=dtype)
    ones = torch.ones(d, device=device, dtype=dtype)
    p = eye - torch.outer(ones, ones) / d
    q, _ = torch.linalg.qr(p)
    return q[:, : d - 1]


def full_simplex_ellipsoid(d: int, device, dtype, ellipsoid_init_radius: float = 0.5) -> Ellipsoid:
    """
    Initial trust region centred at the simplex barycentre with spherical tangent precision.

    u^T M u <= 1 with M = (1/r²) I  corresponds to Euclidean radius r on the centred subspace slice.
    """
    b_mat = get_tangent_basis(d, device, dtype)
    c_vec = torch.ones(d, device=device, dtype=dtype) / d
    rr = float(ellipsoid_init_radius) ** 2
    rr = max(rr, 1e-10)
    m_mat = torch.eye(d - 1, device=device, dtype=dtype) / rr
    return Ellipsoid(c=c_vec, M=m_mat, B=b_mat)


# =============================================================================
# CFS (Conditional Frechet Sampling) for Bounded Simplex
# =============================================================================

def subset_sums_and_signs(caps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute subset sums and inclusion-exclusion signs for CFS algorithm.

    Parameters
    ----------
    caps : torch.Tensor
        Capacity values for each dimension.

    Returns
    -------
    tuple
        (subset_sums, signs) tensors for polytope volume computation.
    """
    m = caps.numel()
    device, dtype = caps.device, caps.dtype
    subset_sums = torch.zeros(1 << m, dtype=dtype, device=device)
    for mask in range(1, 1 << m):
        lsb = mask & -mask
        j = (lsb.bit_length() - 1)
        subset_sums[mask] = subset_sums[mask ^ lsb] + caps[j]
    signs = torch.tensor([1 - 2 * (mask.bit_count() & 1) for mask in range(1 << m)],
                            dtype=dtype, device=device)
    return subset_sums, signs


def polytope_volume(S: torch.Tensor, subset_sums: torch.Tensor,
                    signs: torch.Tensor, power: int, denom: int) -> torch.Tensor:
    """
    Compute polytope volume using inclusion-exclusion principle.

    Parameters
    ----------
    S : torch.Tensor
        Sum values.
    subset_sums : torch.Tensor
        Precomputed subset sums.
    signs : torch.Tensor
        Inclusion-exclusion signs.
    power : int
        Power for the volume computation.
    denom : int
        Denominator (factorial).

    Returns
    -------
    torch.Tensor
        Computed volume values.
    """
    if torch.any(torch.isnan(S)) or torch.any(torch.isinf(S)):
        S = torch.clamp(S, min=0.0, max=1e6)

    shifted = (S.unsqueeze(1) - subset_sums)
    positive = torch.clamp(shifted, min=0.0)

    if power > 50:
        positive = torch.clamp(positive, max=10.0)

    powered = positive.pow(power)

    if torch.any(torch.isnan(powered)) or torch.any(torch.isinf(powered)):
        powered = torch.nan_to_num(powered, nan=0.0, posinf=1e6, neginf=0.0)

    result = (signs * powered).sum(dim=1) / denom

    result = torch.clamp(result, min=0.0)
    if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
        result = torch.nan_to_num(result, nan=1e-15, posinf=1e6, neginf=1e-15)

    return result


def newton_in_bracket(
    sr: torch.Tensor,
    tl: torch.Tensor,
    th: torch.Tensor,
    ss: torch.Tensor,
    si: torch.Tensor,
    m: int,
    denom_int: int,
    denom_vol: int,
    vol_at_tl: torch.Tensor,
    target_mass: torch.Tensor,
    u: torch.Tensor,
    xtol: float = 1e-12,
    max_iter: int = 20,
) -> torch.Tensor:
    """Safe Newton root finder for bounded-simplex CDF inversion."""
    n = tl.shape[0]
    # Warm start: uncapped inverse-CDF formula, with exact linear form for m=1.
    if m == 1:
        t = tl + (th - tl) * u
    else:
        t = sr - (sr - tl) * (1.0 - u).clamp(min=1e-30).pow(1.0 / m)
    t = torch.minimum(torch.maximum(t, tl), th)
    lo = tl.clone()
    hi = th.clone()
    active = torch.ones(n, dtype=torch.bool, device=tl.device)

    for _ in range(max_iter):
        if not active.any():
            break
        ta = t[active]
        sra = sr[active]
        loa = lo[active]
        hia = hi[active]
        vol_t = polytope_volume(sra - ta, ss, si, m, denom_int)
        g_val = (vol_at_tl[active] - vol_t) - target_mass[active]
        if m > 1:
            g_der = polytope_volume(sra - ta, ss, si, m - 1, denom_vol)
        else:
            g_der = torch.ones_like(ta)
        t_newton = ta - g_val / g_der.clamp(min=1e-30)
        lo_new = torch.where(g_val < 0, ta, loa)
        hi_new = torch.where(g_val > 0, ta, hia)
        t_new = torch.minimum(torch.maximum(t_newton, lo_new), hi_new)
        lo[active] = lo_new
        hi[active] = hi_new
        t[active] = t_new
        interval = hi[active] - lo[active]
        newly_done = (interval < xtol) | (g_val.abs() < 1e-13)
        if newly_done.any():
            active_idx = active.nonzero(as_tuple=True)[0]
            active[active_idx[newly_done]] = False

    return torch.minimum(torch.maximum(t, tl), th)


def random_simplex(
    num_samples: int,
    a: torch.Tensor,
    b: torch.Tensor,
    S: float = 1.0,
    max_batch: Optional[int] = None,
    debug: bool = False,
    device: str = 'cuda',
    torch_dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    **ignored,
) -> torch.Tensor:
    """Generate CFS samples uniformly from the bounded simplex."""
    a = a.to(device=device, dtype=torch.float64).flatten()
    b = b.to(device=device, dtype=torch.float64).flatten()
    d = a.numel()
    caps_full = b - a

    if d > 20:
        raise ValueError("Analytic CFS supports dimension <= 20 (2^d cost)")
    if not torch.all(b >= a):
        raise ValueError("All upper bounds must exceed lower bounds")

    s = torch.as_tensor(S, dtype=torch.float64, device=device)
    if not (a.sum() - 1e-10 <= s <= b.sum() + 1e-10):
        raise ValueError(
            f"Sum S={s.item():.6f} outside feasible range "
            f"[{a.sum().item():.6f}, {b.sum().item():.6f}]"
        )
    s0 = s - a.sum()

    if max_batch is None:
        if device == 'cuda':
            estimated_bytes = d * 8 * 32
            gpu_mem = torch.cuda.get_device_properties(device).total_memory
            avail = (gpu_mem - torch.cuda.memory_allocated(device)) * 0.8
            max_batch = max(int(avail / estimated_bytes), 1000)
        else:
            max_batch = min(num_samples, 5_000_000)

    if num_samples > 10_000:
        print(f"Generating {num_samples:,} samples using CFS")
    if debug:
        print(f"CFS: d={d}, num_samples={num_samples:,}, batch={max_batch:,}, S0={s0.item():.4f}, device={device}")

    precomp = []
    for k in range(d - 1):
        ss, si = subset_sums_and_signs(caps_full[k + 1:])
        precomp.append((ss, si))

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    out = torch.empty((num_samples, d), dtype=torch.float64, device=device)
    written = 0
    while written < num_samples:
        batch = min(max_batch, num_samples - written)
        s_rem = s0.expand(batch).clone()
        y = torch.empty((batch, d), dtype=torch.float64, device=device)

        for k in range(d - 1):
            ss, si = precomp[k]
            m = d - k - 1
            sum_tail_caps = caps_full[k + 1:].sum()
            t_low = torch.clamp(s_rem - sum_tail_caps, min=0.0)
            t_high = torch.minimum(caps_full[k].expand_as(s_rem), s_rem)
            t_high = torch.maximum(t_high, t_low)
            det_mask = (t_high - t_low) < 1e-14
            yk = t_low.clone()
            sto_mask = ~det_mask
            n_sto = sto_mask.sum().item()

            if n_sto > 0:
                tl = t_low[sto_mask]
                th = t_high[sto_mask]
                sr = s_rem[sto_mask]
                denom_int = math.factorial(m)
                denom_vol = math.factorial(m - 1) if m > 1 else 1
                vol_at_tl = polytope_volume(sr - tl, ss, si, m, denom_int)
                vol_at_th = polytope_volume(sr - th, ss, si, m, denom_int)
                total_mass = (vol_at_tl - vol_at_th).clamp(min=1e-30)
                u = torch.rand(n_sto, generator=rng, device=device, dtype=torch.float64)
                target_mass = u * total_mass
                t_solved = newton_in_bracket(
                    sr, tl, th, ss, si, m, denom_int, denom_vol,
                    vol_at_tl, target_mass, u, xtol=1e-12, max_iter=20,
                )
                yk[sto_mask] = torch.clamp(t_solved, tl, th)

            bad = torch.isnan(yk) | torch.isinf(yk)
            if bad.any():
                interval = (t_high - t_low).clamp(min=0.0)
                yk[bad] = t_low[bad] + torch.rand(
                    bad.sum(), generator=rng, device=device, dtype=torch.float64
                ) * interval[bad]

            y[:, k] = yk
            s_rem = (s_rem - yk).clamp(min=0.0)

        last = s_rem.clamp(min=0.0, max=caps_full[-1].item())
        if debug:
            violation = (s_rem - caps_full[-1]).clamp(min=0).max().item()
            if violation > 1e-8:
                print(f"  [warn] batch {written}: max last-coord violation = {violation:.2e}")
        y[:, -1] = last
        out[written: written + batch] = y + a.unsqueeze(0)
        written += batch

    if debug:
        sums = out.sum(dim=1)
        err = (sums - s).abs().max().item()
        print(f"CFS done. Max sum error: {err:.2e}")
    return out.to(dtype=torch_dtype)


# =============================================================================
# Simplex Projection
# =============================================================================

def add_composition_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add isotropic Gaussian noise in R^d and project back onto the simplex."""
    if sigma <= 0.0:
        return x
    noisy = x + torch.randn_like(x) * sigma
    return proj_simplex(noisy)


def proj_simplex(X: torch.Tensor) -> torch.Tensor:
    """
    Project points onto the simplex (differentiable).

    Uses the algorithm from "Efficient Projections onto the l1-Ball
    for Learning in High Dimensions" (Duchi et al., 2008).

    Parameters
    ----------
    X : torch.Tensor
        Points to project, shape (..., d) or (n, l, d).

    Returns
    -------
    torch.Tensor
        Projected points on the simplex.
    """
    original_dim = X.dim()

    if original_dim == 3:
        n, l, d = X.shape
        X_2d = X.reshape(-1, d)
    else:
        X_2d = X

    u, _ = torch.sort(X_2d, descending=True, dim=-1)
    css = torch.cumsum(u, dim=-1)
    d = X_2d.size(-1)
    indices = torch.arange(1, d+1, device=X_2d.device, dtype=X_2d.dtype)
    rho = torch.sum((u * indices) > (css - 1), dim=-1) - 1

    batch_indices = torch.arange(X_2d.size(0), device=X_2d.device)
    theta = (torch.gather(css, 1, rho.unsqueeze(-1)).squeeze(-1) - 1) / (rho + 1).to(X_2d.dtype)

    result = torch.maximum(X_2d - theta.unsqueeze(-1), torch.zeros_like(X_2d))

    if original_dim == 3:
        result = result.reshape(n, l, d)

    return result


def sample_ellipsoid(
    n: int,
    ell: Ellipsoid,
    *,
    scale: float = 1.0,
    jitter: float = 1e-8,
) -> torch.Tensor:
    """
    Gaussian proposals in tangent space: u ~ N(0, scale² M^{-1}); x = Proj_Δ(c + B u).
    """
    d = ell.c.shape[0]
    device = ell.c.device
    dtype = ell.c.dtype

    md = ell.M.shape[0]
    m_reg = ell.M + jitter * torch.eye(md, device=device, dtype=dtype)
    cov = (scale ** 2) * torch.linalg.inv(m_reg)
    cov = 0.5 * (cov + cov.mT)
    try:
        l_chol = torch.linalg.cholesky(cov)
    except RuntimeError:
        cov = cov + jitter * torch.eye(md, device=device, dtype=dtype)
        l_chol = torch.linalg.cholesky(cov)

    z = torch.randn(n, md, device=device, dtype=dtype)
    u_rows = z @ l_chol.T

    lifted = proj_simplex(ell.c.unsqueeze(0) + u_rows @ ell.B.T)
    return lifted.view(-1, d)


def clamp_ellipsoid_scale(ell: Ellipsoid, margin: float = 1.05) -> Ellipsoid:
    """
    Widen ambient trust-region after optimisation by dividing tangent-space eigen-values of ``M``.
    """
    w, vec = torch.linalg.eigh(ell.M)
    w = torch.clamp(w.real, min=1e-10)
    w_new = w / (margin ** 2 + 1e-12)
    vec_r = vec.real.to(dtype=ell.M.dtype)
    m_new = (vec_r * w_new.unsqueeze(0)) @ vec_r.mT
    m_new = 0.5 * (m_new + m_new.mT).to(device=ell.c.device)
    return Ellipsoid(c=ell.c, M=m_new, B=ell.B)


def random_simplex_direction(n_directions: int, d: int, device: str = 'cuda',
                              dtype: torch.dtype = torch.float64,
                              seed: Optional[int] = None) -> torch.Tensor:
    """
    Sample ``n_directions`` uniformly random directions tangent to the probability simplex in R^d.

    Each direction lives in the zero-sum hyperplane (sum = 0) and has unit norm,
    so adding a scaled direction to any point on the simplex keeps the sum at 1.

    Algorithm (matches the statistically verified scalar version):
        1. Draw v ~ N(0, I) in R^d
        2. Project onto the zero-sum hyperplane: v -= v.mean()
        3. Normalise: v /= ||v||
        4. Resample rows where ||v|| < 1e-12 (degenerate; astronomically rare)

    Parameters
    ----------
    n_directions : int
        Number of directions to sample.
    d : int
        Dimensionality of each direction.
    device : str
        Torch device.
    dtype : torch.dtype
        Data type.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        (n_directions, d) tensor of zero-sum, unit-norm vectors.
    """
    if seed is not None:
        torch.manual_seed(seed)
    v = torch.randn(n_directions, d, device=device, dtype=dtype)
    v = v - v.mean(dim=1, keepdim=True)
    norms = v.norm(dim=1, keepdim=True)
    mask = norms.squeeze(1) < 1e-12
    while mask.any():
        r = torch.randn(int(mask.sum()), d, device=device, dtype=dtype)
        r = r - r.mean(dim=1, keepdim=True)
        v[mask] = r
        norms[mask] = v[mask].norm(dim=1, keepdim=True)
        mask = norms.squeeze(1) < 1e-12
    return v / norms


# Backward-compatible alias
random_zero_sum_directions = random_simplex_direction


# =============================================================================
# Basic Simplex Utilities (original functions)
# =============================================================================



def is_on_simplex(x: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """
    Check if points are on the simplex.

    Parameters
    ----------
    x : torch.Tensor
        Points to check, shape (..., d).
    tol : float
        Tolerance for checks.

    Returns
    -------
    torch.Tensor
        Boolean tensor indicating if each point is on simplex.
    """
    # Check sum to 1
    sum_check = torch.abs(x.sum(dim=-1) - 1.0) < tol

    # Check non-negative
    nonneg_check = (x >= -tol).all(dim=-1)

    return sum_check & nonneg_check


def simplex_distance(x: torch.Tensor, y: torch.Tensor,
                     metric: str = 'euclidean') -> torch.Tensor:
    """
    Compute distance between points on the simplex.

    Parameters
    ----------
    x : torch.Tensor
        First set of points, shape (n, d) or (d,).
    y : torch.Tensor
        Second set of points, shape (m, d) or (d,).
    metric : str
        Distance metric: 'euclidean', 'aitchison', 'kl'.

    Returns
    -------
    torch.Tensor
        Distance matrix or scalar.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    if metric == 'euclidean':
        return torch.cdist(x, y)

    elif metric == 'aitchison':
        # Aitchison distance (log-ratio distance)
        eps = 1e-10
        log_x = torch.log(x + eps)
        log_y = torch.log(y + eps)

        # Center log-ratios
        clr_x = log_x - log_x.mean(dim=-1, keepdim=True)
        clr_y = log_y - log_y.mean(dim=-1, keepdim=True)

        return torch.cdist(clr_x, clr_y)

    elif metric == 'kl':
        # KL divergence (not symmetric)
        eps = 1e-10
        # Compute pairwise KL: D_KL(x || y)
        n, m = x.shape[0], y.shape[0]
        kl_matrix = torch.zeros(n, m, device=x.device, dtype=x.dtype)

        for i in range(n):
            for j in range(m):
                kl_matrix[i, j] = (x[i] * (torch.log(x[i] + eps) - torch.log(y[j] + eps))).sum()

        return kl_matrix

    else:
        raise ValueError(f"Unknown metric: {metric}")




def composition_to_ilr(x: torch.Tensor) -> torch.Tensor:
    """
    Transform compositions to isometric log-ratio (ILR) coordinates.

    ILR is useful for unconstrained optimization on simplex data.

    Parameters
    ----------
    x : torch.Tensor
        Compositions on simplex, shape (..., d).

    Returns
    -------
    torch.Tensor
        ILR coordinates, shape (..., d-1).
    """
    d = x.shape[-1]
    eps = 1e-10
    log_x = torch.log(x + eps)

    # Helmert-like ILR transformation
    ilr_coords = []
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        term1 = log_x[..., :i+1].sum(dim=-1) / (i + 1)
        term2 = log_x[..., i+1]
        ilr_coords.append(coef * (term1 - term2))

    return torch.stack(ilr_coords, dim=-1)


def ilr_to_composition(ilr: torch.Tensor, d: int) -> torch.Tensor:
    """
    Transform ILR coordinates back to compositions.

    Parameters
    ----------
    ilr : torch.Tensor
        ILR coordinates, shape (..., d-1).
    d : int
        Original dimensionality.

    Returns
    -------
    torch.Tensor
        Compositions on simplex, shape (..., d).
    """
    # Inverse ILR transformation via the transpose of the Helmert contrast matrix:
    # forward: ilr[i] = coef * (mean(log_x[:i+1]) - log_x[i+1])
    # inverse (pseudoinverse = C^T): log_x[j] += coef * ilr[i] / (i+1)  for j <= i
    #                                log_x[i+1] -= coef * ilr[i]
    log_x = torch.zeros(*ilr.shape[:-1], d, device=ilr.device, dtype=ilr.dtype)

    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        contribution = ilr[..., i] * coef  # C^T scaling: coef, not 1/coef

        # Distribute to first i+1 components
        log_x[..., :i+1] += contribution.unsqueeze(-1) / (i + 1)
        # Subtract from component i+1
        log_x[..., i+1] -= contribution

    # Normalize to sum to 1
    x = torch.exp(log_x)
    x = x / x.sum(dim=-1, keepdim=True)

    return x
