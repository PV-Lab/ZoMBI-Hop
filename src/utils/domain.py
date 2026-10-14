"""
Search Domains
==============

Every geometry-specific operation ZoMBI-Hop performs, gathered behind one object
so the optimiser runs on either the probability simplex or an axis-aligned box.

* ``SimplexDomain`` — {x : x_i >= 0, sum(x) = 1}. Each method delegates to the
  exact function the optimiser called before this abstraction existed, in the
  same order and with the same arguments, so simplex runs are bit-for-bit
  unchanged (pinned by ``tests/test_domain_golden.py``).
* ``BoxDomain`` — a hyper-rectangle [lower, upper]. Defaults to the unit cube:
  benchmarks should map their native rectangle onto [0,1]^d with
  ``UnitBoxScaler`` so the length-scale hyperparameters tuned on the simplex
  (``input_noise``, ``needle_repeat_radius_frac``, ``max_penalty_radius``) keep
  roughly their meaning.

The optimiser only ever sees the domain through these methods:

    project(X)                     snap points back into the domain
    sample(n, a, b, ...)           uniform samples from domain ∩ [a, b]
    directions(n, d, ...)          unit directions along the domain (for LineBO chords)
    tangent_basis(d, ...)          orthonormal basis of the domain's affine hull
    ascent_step(x, g, step, lo, hi)  one acquisition-ascent step inside [lo, hi]
    boundary_penalty(X, eps)       repulsion from the domain's faces
    region_jaccard(a, b, ...)      overlap of two zoom boxes *within the domain*
    check_point(x)                 is x a valid LineBO anchor?
    default_bounds(d, ...)         the full search box
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .simplex import (
    proj_simplex,
    random_simplex,
    random_simplex_direction,
    get_tangent_basis,
)


def _bounds_jaccard_simplex(
    bounds_a: torch.Tensor,
    bounds_b: torch.Tensor,
    n_samples: int = 500,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> float:
    """
    Jaccard overlap of two AABB boxes restricted to the simplex, estimated via Monte Carlo.

    Samples from the simplex; counts what fraction fall inside each box;
    returns  |A ∩ B| / |A ∪ B|  (both sets restricted to the simplex).

    Note: normalised i.i.d. uniforms are NOT uniform on the simplex (they are
    biased toward the barycentre). Kept as-is because the zoom guard's thresholds
    were tuned against this estimator.
    """
    d = bounds_a.shape[1]
    kw: dict = {}
    if device is not None:
        kw["device"] = device
    if dtype is not None:
        kw["dtype"] = dtype
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


def _bounds_jaccard_box(bounds_a: torch.Tensor, bounds_b: torch.Tensor) -> float:
    """Exact volume Jaccard of two (2, d) axis-aligned boxes.

    Axes where both boxes are zero-width are projected out, as in
    ``DataHandler._jaccard_box``.
    """
    a = bounds_a.to(torch.float64)
    b = bounds_b.to(torch.float64)
    w_a = a[1] - a[0]
    w_b = b[1] - b[0]
    active = ~((w_a < 1e-12) & (w_b < 1e-12))
    if not bool(active.any()):
        return 1.0
    inter = (torch.minimum(a[1], b[1]) - torch.maximum(a[0], b[0])).clamp(min=0.0)[active]
    vol_inter = float(inter.prod())
    vol_a = float(w_a[active].clamp(min=1e-30).prod())
    vol_b = float(w_b[active].clamp(min=1e-30).prod())
    vol_union = vol_a + vol_b - vol_inter
    return 0.0 if vol_union < 1e-30 else vol_inter / vol_union


class Domain:
    """Interface; see the module docstring. Subclasses implement every method."""

    name: str = "abstract"
    # Whether the acquisition is repelled from the domain's faces
    # (``RepulsiveAcquisition`` boundary term, decay length = input_noise).
    boundary_repulsion: bool = False

    def project(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, num_samples: int, a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def directions(self, n: int, d: int, device: str = "cuda",
                   dtype: torch.dtype = torch.float64) -> torch.Tensor:
        raise NotImplementedError

    def tangent_basis(self, d: int, device, dtype=torch.float64) -> torch.Tensor:
        raise NotImplementedError

    def ascent_step(self, x: torch.Tensor, g: torch.Tensor, step: float,
                    lo: torch.Tensor, hi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def boundary_penalty(self, X: torch.Tensor, eps: float) -> torch.Tensor:
        raise NotImplementedError

    def region_jaccard(self, bounds_a: torch.Tensor, bounds_b: torch.Tensor,
                       device=None, dtype=None) -> float:
        raise NotImplementedError

    def check_point(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def default_bounds(self, d: int, device, dtype) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SimplexDomain(Domain):
    """The probability simplex. Reproduces the pre-abstraction behaviour exactly."""

    name = "simplex"
    boundary_repulsion = True

    def project(self, X: torch.Tensor) -> torch.Tensor:
        return proj_simplex(X)

    def sample(self, num_samples, a, b, S: float = 1.0, max_batch=None, debug: bool = False,
               device: str = "cuda", torch_dtype: torch.dtype = torch.float64, **kwargs):
        return random_simplex(num_samples, a, b, S, max_batch, debug, device, torch_dtype, **kwargs)

    def directions(self, n, d, device="cuda", dtype=torch.float64):
        return random_simplex_direction(n, d, device=device, dtype=dtype)

    def tangent_basis(self, d, device, dtype=torch.float64):
        # The (d, d-1) basis of the zero-sum hyperplane.
        return get_tangent_basis(d, device, dtype).contiguous()

    def ascent_step(self, x, g, step, lo, hi):
        """Exponentiated-gradient (Fisher-Rao natural-gradient) step:
        x ← normalize(x ⊙ exp(α(g − ḡ))), clamped to [lo, hi] and renormalised.
        Returns (x_new, ok) — ok is False for restarts that degenerated to zero mass."""
        g_bar = (x * g).sum(dim=1, keepdim=True)
        shift = torch.clamp(step * (g - g_bar), -10.0, 10.0)
        x_new = x * torch.exp(shift)

        s = x_new.sum(dim=1, keepdim=True)
        ok = s.squeeze(1) >= 1e-12
        x_new = x_new / s.clamp(min=1e-12)
        x_new = torch.clamp(x_new, lo, hi)
        s2 = x_new.sum(dim=1, keepdim=True)
        ok &= s2.squeeze(1) >= 1e-12
        x_new = x_new / s2.clamp(min=1e-12)
        return x_new, ok

    def boundary_penalty(self, X, eps):
        # exp-decay from each simplex face x_i = 0
        return torch.exp(-X / eps).sum(dim=-1)

    def region_jaccard(self, bounds_a, bounds_b, device=None, dtype=None):
        return _bounds_jaccard_simplex(bounds_a, bounds_b, device=device, dtype=dtype)

    def check_point(self, x):
        assert abs(x.sum().item() - 1.0) < 1e-12, f"x_tell must sum to 1, got {x.sum().item()}"

    def default_bounds(self, d, device, dtype):
        b = torch.zeros(2, d, device=device, dtype=dtype)
        b[1] = 1.0
        return b


class BoxDomain(Domain):
    """Axis-aligned box [lower, upper] (default: the unit cube [0,1]^d).

    Parameters
    ----------
    lower, upper : sequence or tensor, optional
        Per-dimension limits. Both None ⇒ unit cube of whatever d the optimiser
        is constructed with. Prefer the unit cube plus ``UnitBoxScaler`` over a
        native rectangle; see the module docstring.
    boundary_repulsion : bool
        Repel the acquisition from the box faces with the same exp-decay law the
        simplex applies to its x_i = 0 faces (both faces per axis here). Off by
        default: on the simplex a face means a missing component, but benchmark
        optima commonly sit on or near a box face, and with the default decay
        length (input_noise) the term would bias the search toward the centre.
    max_step_frac : float
        Per-coordinate cap on one ascent step, as a fraction of the search box
        width. The repulsion term's gradient can be orders of magnitude larger
        than the acquisition's; the simplex step caps it with a ±10 clamp in
        log-space, this is the Euclidean counterpart.
    """

    name = "box"

    def __init__(self, lower=None, upper=None, boundary_repulsion: bool = False,
                 max_step_frac: float = 0.1):
        if (lower is None) != (upper is None):
            raise ValueError("BoxDomain needs both lower and upper, or neither")
        self.lower = None if lower is None else torch.as_tensor(lower, dtype=torch.float64).flatten()
        self.upper = None if upper is None else torch.as_tensor(upper, dtype=torch.float64).flatten()
        if self.lower is not None:
            if self.lower.shape != self.upper.shape:
                raise ValueError("lower and upper must have the same length")
            if not bool((self.upper > self.lower).all()):
                raise ValueError("upper must exceed lower in every dimension")
        self.boundary_repulsion = bool(boundary_repulsion)
        self.max_step_frac = float(max_step_frac)

    def _limits(self, d: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.lower is None:
            return (torch.zeros(d, device=device, dtype=dtype),
                    torch.ones(d, device=device, dtype=dtype))
        if self.lower.numel() != d:
            raise ValueError(f"BoxDomain has {self.lower.numel()} dims, got points with {d}")
        return self.lower.to(device=device, dtype=dtype), self.upper.to(device=device, dtype=dtype)

    def project(self, X):
        lo, hi = self._limits(X.shape[-1], X.device, X.dtype)
        return torch.minimum(torch.maximum(X, lo), hi)

    def sample(self, num_samples, a, b, device: str = "cuda",
               torch_dtype: torch.dtype = torch.float64, seed: Optional[int] = None, **ignored):
        a = a.to(device=device, dtype=torch_dtype).flatten()
        b = b.to(device=device, dtype=torch_dtype).flatten()
        if not bool((b >= a).all()):
            raise ValueError("All upper bounds must be >= lower bounds")
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
        u = torch.rand(num_samples, a.numel(), generator=gen, device=device, dtype=torch_dtype)
        return a + u * (b - a)

    def directions(self, n, d, device="cuda", dtype=torch.float64):
        v = torch.randn(n, d, device=device, dtype=dtype)
        norms = v.norm(dim=1, keepdim=True)
        bad = norms.squeeze(1) < 1e-12
        while bool(bad.any()):
            v[bad] = torch.randn(int(bad.sum()), d, device=device, dtype=dtype)
            norms = v.norm(dim=1, keepdim=True)
            bad = norms.squeeze(1) < 1e-12
        return v / norms

    def tangent_basis(self, d, device, dtype=torch.float64):
        # Full-dimensional domain: the ellipsoid lives in ambient coordinates.
        return torch.eye(d, device=device, dtype=dtype)

    def ascent_step(self, x, g, step, lo, hi):
        """Projected gradient ascent: x ← clip(x + clip(α g, ±cap), lo, hi),
        cap = max_step_frac × (hi − lo). ok is False for non-finite updates."""
        cap = self.max_step_frac * (hi - lo).clamp(min=1e-12)
        delta = torch.maximum(torch.minimum(step * g, cap), -cap)
        x_new = torch.minimum(torch.maximum(x + delta, lo), hi)
        ok = torch.isfinite(x_new).all(dim=1)
        return x_new, ok

    def boundary_penalty(self, X, eps):
        lo, hi = self._limits(X.shape[-1], X.device, X.dtype)
        return (torch.exp(-(X - lo) / eps) + torch.exp(-(hi - X) / eps)).sum(dim=-1)

    def region_jaccard(self, bounds_a, bounds_b, device=None, dtype=None):
        return _bounds_jaccard_box(bounds_a, bounds_b)

    def check_point(self, x):
        lo, hi = self._limits(x.shape[-1], x.device, x.dtype)
        tol = 1e-9
        assert bool(((x >= lo - tol) & (x <= hi + tol)).all()), \
            f"x_tell must lie inside the box, got {x.tolist()}"

    def default_bounds(self, d, device, dtype):
        lo, hi = self._limits(d, device, dtype)
        return torch.stack([lo, hi], dim=0)

    def __repr__(self) -> str:
        box = "unit" if self.lower is None else f"{self.lower.tolist()}–{self.upper.tolist()}"
        return (f"BoxDomain({box}, boundary_repulsion={self.boundary_repulsion}, "
                f"max_step_frac={self.max_step_frac})")


class UnitBoxScaler:
    """Affine map between a native rectangle [lower, upper] and the unit cube.

    Wrap a benchmark so ZoMBI-Hop (with ``BoxDomain()``) works in [0,1]^d while
    the test function is evaluated in its own coordinates::

        scaler = UnitBoxScaler([-5, -5], [10, 15])
        f_unit = lambda U: f(scaler.from_unit(U))
        needles_native = scaler.from_unit(needles)
    """

    def __init__(self, lower, upper):
        self.lower = torch.as_tensor(lower, dtype=torch.float64).flatten()
        self.upper = torch.as_tensor(upper, dtype=torch.float64).flatten()
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same length")
        if not bool((self.upper > self.lower).all()):
            raise ValueError("upper must exceed lower in every dimension")

    @property
    def d(self) -> int:
        return self.lower.numel()

    def _lu(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.lower.to(device=X.device, dtype=X.dtype),
                self.upper.to(device=X.device, dtype=X.dtype))

    def to_unit(self, X: torch.Tensor) -> torch.Tensor:
        lo, hi = self._lu(X)
        return (X - lo) / (hi - lo)

    def from_unit(self, U: torch.Tensor) -> torch.Tensor:
        lo, hi = self._lu(U)
        return lo + U * (hi - lo)


def make_domain(spec) -> Domain:
    """Resolve ``None`` / ``"simplex"`` / ``"box"`` / a ``Domain`` instance."""
    if spec is None or spec == "simplex":
        return SimplexDomain()
    if spec == "box":
        return BoxDomain()
    if isinstance(spec, Domain):
        return spec
    raise ValueError(f"unknown domain {spec!r}; expected 'simplex', 'box' or a Domain")
