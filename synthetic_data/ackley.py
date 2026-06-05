"""
ackley.py
=========
Negated Ackley test functions on the 3-element probability simplex.

These are the analytic synthetic objectives originally defined inline in
``scripts/run_zombi_test.py``, refactored here into a single reusable
``Ackley`` class so they can be shared between the benchmark harness and the
interactive simulator (``interactive_testing/interactive_test_zombi.py``).

Five variants are provided, each a *negated* Ackley function whose global
maximum (value ≈ 0) sits at a canonical simplex location:

    "centroid"    peak at [1/3, 1/3, 1/3]   (simplex centroid)
    "edge"        peak at [0.5, 0.5, 0]     (edge midpoint)
    "vertex"      peak at [1,   0,   0]     (simplex vertex)
    "multimodal"  sum of three skinnier-peaked Ackleys, with maxima at all
                  three locations above
    "realistic"   sum of several Ackleys at scattered interior locations, each
                  with its *own* basin width (the per-peak ``b``): a couple of
                  broad basins and a couple of very pointy/spiky ones. No basin
                  is wider than the widest of the other variants (b ≥ ACKLEY_B).
                  The peak locations are still known analytically (see
                  ``REALISTIC_PEAKS`` / ``known_maxima``).

Because the functions are negated, the maximum is at the centre and the value
decreases (becomes more negative) away from it, so ZoMBI-Hop's internal
maximisation finds the peaks directly.

The ``Ackley`` class exposes a ``predict(X)`` method matching scikit-learn's
``RandomForestRegressor.predict`` signature ``(N, d) -> (N,)``, so an ``Ackley``
instance can be used as a drop-in replacement for a trained RF surrogate.

Example
-------
    from synthetic_data.ackley import Ackley

    fn = Ackley("centroid")
    y = fn.predict(np.array([[1/3, 1/3, 1/3]]))   # ≈ [0.0]
    print(fn.known_maxima)                         # [(array([...]), value), ...]
"""

from __future__ import annotations

import numpy as np

# ── Ackley constants (mirror the originals in scripts/run_zombi_test.py) ───────
ACKLEY_A = 20.0
ACKLEY_B = 0.2
ACKLEY_B_SKINNY = 1.2     # larger b → skinnier peaks (used by "multimodal")
ACKLEY_C = 2.0 * np.pi
ACKLEY_SCALE = 30.0       # vertical scale so Y values are numerically healthy

# ── Dimension-general variant registry ────────────────────────────────────────
# Each variant is defined *generatively* as a function of the simplex dimension
# ``dim``, so the same catalogue works on the 3-simplex (the benchmark default),
# the 4-simplex (the interactive plots in plot_4d.py / point_cloud_4d.py), or any
# higher d.  To add a new mode, add one entry to ``_VARIANT_SPECS`` below: every
# consumer -- the ``Ackley`` class and both visualisers -- picks it up
# automatically, so nothing downstream needs editing.
#
# A spec is ``(peaks_builder, combine)`` where ``peaks_builder(dim)`` returns a
# list of ``(center, b)`` tuples -- one negated-Ackley peak per centre, each with
# its own basin width ``b`` (larger b → skinnier/spikier).  ``combine`` is how
# the peaks are merged (see ``Ackley.predict``): ``"sum"`` adds them (basins
# blend) or ``"max"`` takes the pointwise maximum (every listed centre stays an
# exact global maximum regardless of basin width).


def _uniform_center(dim: int) -> np.ndarray:
    """Simplex centroid [1/dim, ..., 1/dim]."""
    return np.full(dim, 1.0 / dim)


def _edge_center(dim: int) -> np.ndarray:
    """Edge midpoint [0.5, 0.5, 0, ..., 0]."""
    c = np.zeros(dim)
    c[0] = c[1] = 0.5
    return c


def _vertex_center(dim: int) -> np.ndarray:
    """Simplex vertex [1, 0, ..., 0]."""
    c = np.zeros(dim)
    c[0] = 1.0
    return c


# "realistic": scattered interior maxima with *per-peak* basin widths.  Larger b
# → skinnier (more spiky) basin; smaller b → broader.  The defining character is
# a mix of basin widths (moderately wide through very pointy) at scattered,
# analytically-known locations.  No basin may be broader than
# ``_REALISTIC_BROADEST`` -- there is deliberately no very-wide basin.  The number
# and placement of peaks are dimension-specific, so each entry below is an
# explicit ``(center, b)`` list.  Dimensions that aren't listed fall back to a
# deterministic seeded scatter so the variant still works everywhere; either way
# the locations are recoverable via ``Ackley.known_maxima``.
_REALISTIC_BROADEST = 0.60   # broadest allowed basin (smallest permitted b)
_REALISTIC_PEAKS_BY_DIM = {
    3: [
        (np.array([0.20, 0.70, 0.10]), 7),   # least pointy here
        (np.array([0.34, 0.33, 0.33]), 20),   # pointier
        (np.array([0.10, 0.25, 0.65]), 5),   # pointier still
        (np.array([0.45, 0.10, 0.45]), 10),   # very pointy / spiky
    ],
    4: [
        (np.array([0.15, 0.60, 0.15, 0.10]), 7),   # least pointy here
        (np.array([0.25, 0.25, 0.25, 0.25]), 20),   # pointier
        (np.array([0.10, 0.20, 0.55, 0.15]), 5),   # pointier still
        (np.array([0.35, 0.10, 0.15, 0.40]), 10),   # very pointy / spiky
    ],
    # 10-simplex: ten well-separated optima, each dominated by a different
    # component (0.64 on one axis, 0.04 on the other nine), with basin widths
    # spanning moderately wide (0.60) to very pointy (2.50).
    10: [
        (np.array([0.64, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]), 5.60),
        (np.array([0.04, 0.64, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]), 5.81),
        (np.array([0.04, 0.04, 0.64, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]), 5.02),
        (np.array([0.04, 0.04, 0.04, 0.64, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]), 5.23),
        (np.array([0.04, 0.04, 0.04, 0.04, 0.64, 0.04, 0.04, 0.04, 0.04, 0.04]), 10.44),
        (np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.64, 0.04, 0.04, 0.04, 0.04]), 10.65),
        (np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.64, 0.04, 0.04, 0.04]), 10.86),
        (np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.64, 0.04, 0.04]), 15.07),
        (np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.64, 0.04]), 15.28),
        (np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.64]), 15.50),
    ],
}


def _realistic_peaks(dim: int) -> list[tuple[np.ndarray, float]]:
    peaks = _REALISTIC_PEAKS_BY_DIM.get(dim)
    if peaks is None:
        # Deterministic fallback for unlisted dimensions: a fixed-seed Dirichlet
        # scatter of ``dim`` interior points, with basin widths spanning the
        # broadest-allowed (0.60) to very pointy (2.50).
        rng = np.random.default_rng(0)
        centers = rng.dirichlet(np.ones(dim), size=dim)
        widths = np.linspace(_REALISTIC_BROADEST, 2.50, dim)
        peaks = list(zip(centers, widths))
    return [(np.asarray(c, dtype=float).copy(), float(b)) for c, b in peaks]


# variant -> (peaks_builder, combine).  See the block comment above.
_VARIANT_SPECS = {
    "centroid":   (lambda dim: [(_uniform_center(dim), ACKLEY_B)], "sum"),
    "edge":       (lambda dim: [(_edge_center(dim), ACKLEY_B)], "sum"),
    "vertex":     (lambda dim: [(_vertex_center(dim), ACKLEY_B)], "sum"),
    "multimodal": (lambda dim: [(_uniform_center(dim), ACKLEY_B_SKINNY),
                                (_edge_center(dim), ACKLEY_B_SKINNY),
                                (_vertex_center(dim), ACKLEY_B_SKINNY)], "sum"),
    "realistic":  (_realistic_peaks, "max"),
}

VARIANTS = tuple(_VARIANT_SPECS)

# ── Backwards-compatible 3-simplex names (re-exported by run_zombi_test.py) ────
CENTER_CENTROID = _uniform_center(3)
CENTER_EDGE     = _edge_center(3)
CENTER_VERTEX   = _vertex_center(3)
MULTIMODAL_CENTERS = [CENTER_CENTROID, CENTER_EDGE, CENTER_VERTEX]
REALISTIC_PEAKS = _realistic_peaks(3)


def _negated_ackley(
    X: np.ndarray,
    center: np.ndarray,
    *,
    a: float = ACKLEY_A,
    b: float = ACKLEY_B,
    c: float = ACKLEY_C,
    scale: float = ACKLEY_SCALE,
) -> np.ndarray:
    """
    Vectorised negated Ackley centred at ``center``.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Query points on the d-simplex.
    center : np.ndarray, shape (d,)
        Peak location (function maximum, value ≈ 0, sits here).

    Returns
    -------
    np.ndarray, shape (N,)
        Negated Ackley values (≤ 0; equal to 0 at ``center``).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    d = X.shape[1]
    delta = X - np.asarray(center, dtype=float)
    t1 = -a * np.exp(-b * np.sqrt(np.sum(delta ** 2, axis=1) / d))
    t2 = -np.exp(np.sum(np.cos(c * delta), axis=1) / d)
    # Standard Ackley: 0 at the centre, > 0 away (the centre is a *minimum*).
    ackley = t1 + t2 + a + np.e
    # Negate so the centre becomes the maximum (value 0, decreasing away).
    return -scale * ackley


class Ackley:
    """
    A negated Ackley objective on the ``dim``-element probability simplex.

    Parameters
    ----------
    variant : str
        One of ``Ackley.VARIANTS`` (``"centroid"``, ``"edge"``, ``"vertex"``,
        ``"multimodal"``, ``"realistic"`` -- see the module docstring).
        Case-insensitive.
    dim : int, default 3
        Simplex dimensionality.  The variant's peaks are generated for this
        ``dim`` from the shared registry, so ``Ackley(v)`` is the 3-simplex
        benchmark objective and ``Ackley(v, dim=4)`` is its 4-simplex analogue
        (used by plot_4d.py / point_cloud_4d.py).

    Attributes
    ----------
    variant : str
        The (lower-cased) variant name.
    dim : int
        The simplex dimensionality the peaks were generated for.
    centers : list of np.ndarray
        Peak location(s), each of length ``dim``: a single-element list for the
        unimodal variants, all canonical centres for the multi-peak ones.
    basin_widths : list of float
        The per-peak ``b`` parallel to ``centers`` (larger = skinnier).
    combine : str
        How the per-peak Ackleys are merged: ``"sum"`` (peaks add, so centres
        are pulled slightly off the listed locations — used by ``"multimodal"``)
        or ``"max"`` (the pointwise maximum, so every listed centre is exactly a
        global maximum with value 0 — used by ``"realistic"``).
    """

    VARIANTS = VARIANTS

    def __init__(self, variant: str = "centroid", dim: int = 3) -> None:
        variant = str(variant).strip().lower()
        if variant not in _VARIANT_SPECS:
            raise ValueError(
                f"Unknown Ackley variant {variant!r}; expected one of {VARIANTS}."
            )
        if dim < 2:
            raise ValueError(f"dim must be ≥ 2 (got {dim}).")
        self.variant = variant
        self.dim = dim
        peaks_builder, combine = _VARIANT_SPECS[variant]
        peaks = peaks_builder(dim)
        self.centers = [c.copy() for c, _ in peaks]
        self.basin_widths = [b for _, b in peaks]
        self.combine = combine

    # ── sklearn-compatible interface ──────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the objective at one or more simplex points.

        Mirrors ``RandomForestRegressor.predict``: accepts ``(N, d)`` (or a
        single ``(d,)`` point) and returns a 1-D array of shape ``(N,)``.

        Each centre contributes one negated Ackley with its own basin width;
        the contributions are merged per ``self.combine`` (``"sum"`` for the
        unimodal variants and ``"multimodal"``, ``"max"`` for ``"realistic"``).
        For a single centre both reduce to that one negated Ackley.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        terms = np.stack(
            [_negated_ackley(X, center, b=b)
             for center, b in zip(self.centers, self.basin_widths)],
            axis=0,
        )
        return terms.max(axis=0) if self.combine == "max" else terms.sum(axis=0)

    # Allow ``fn(x)`` as a convenient alias for a single point, matching the
    # scalar ``f(x: np.ndarray) -> float`` signature used by run_zombi_test.py.
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate a single point and return a Python float."""
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    @property
    def known_maxima(self) -> list[tuple[np.ndarray, float]]:
        """
        Analytic peak locations and their objective values.

        Returns a list of ``(composition, value)`` tuples — one per peak.
        Useful as ready-made reference extrema for plotting (no interactive
        clicking required).
        """
        return [(c.copy(), float(self.predict(c.reshape(1, -1))[0])) for c in self.centers]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Ackley(variant={self.variant!r})"
