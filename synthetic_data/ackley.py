"""
ackley.py
=========
Negated Ackley test functions on the 3-element probability simplex.

These are the analytic synthetic objectives originally defined inline in
``scripts/run_zombi_test.py``, refactored here into a single reusable
``Ackley`` class so they can be shared between the benchmark harness and the
interactive simulator (``interactive_testing/interactive_test_zombi.py``).

Four variants are provided, each a *negated* Ackley function whose global
maximum (value ≈ 0) sits at a canonical simplex location:

    "centroid"    peak at [1/3, 1/3, 1/3]   (simplex centroid)
    "edge"        peak at [0.5, 0.5, 0]     (edge midpoint)
    "vertex"      peak at [1,   0,   0]     (simplex vertex)
    "multimodal"  sum of three skinnier-peaked Ackleys, with maxima at all
                  three locations above

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

# ── Canonical peak locations on the 3-simplex ─────────────────────────────────
CENTER_CENTROID = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
CENTER_EDGE     = np.array([0.5, 0.5, 0.0])
CENTER_VERTEX   = np.array([1.0, 0.0, 0.0])
MULTIMODAL_CENTERS = [CENTER_CENTROID, CENTER_EDGE, CENTER_VERTEX]

VARIANTS = ("centroid", "edge", "vertex", "multimodal")

# Map each single-peak variant to its centre.
_VARIANT_CENTER = {
    "centroid": CENTER_CENTROID,
    "edge":     CENTER_EDGE,
    "vertex":   CENTER_VERTEX,
}


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
    return scale * (t1 + t2 + a + np.e)


class Ackley:
    """
    A negated Ackley objective on the 3-element simplex.

    Parameters
    ----------
    variant : str
        One of ``"centroid"``, ``"edge"``, ``"vertex"``, ``"multimodal"``
        (see the module docstring).  Case-insensitive.

    Attributes
    ----------
    variant : str
        The (lower-cased) variant name.
    centers : list of np.ndarray
        Peak location(s): a single-element list for the unimodal variants,
        all three canonical centres for ``"multimodal"``.
    """

    VARIANTS = VARIANTS

    def __init__(self, variant: str = "centroid") -> None:
        variant = str(variant).strip().lower()
        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown Ackley variant {variant!r}; expected one of {VARIANTS}."
            )
        self.variant = variant
        if variant == "multimodal":
            self.centers = list(MULTIMODAL_CENTERS)
            self._b = ACKLEY_B_SKINNY
        else:
            self.centers = [_VARIANT_CENTER[variant]]
            self._b = ACKLEY_B

    # ── sklearn-compatible interface ──────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the objective at one or more simplex points.

        Mirrors ``RandomForestRegressor.predict``: accepts ``(N, d)`` (or a
        single ``(d,)`` point) and returns a 1-D array of shape ``(N,)``.

        For ``"multimodal"`` the value is the sum of three skinnier-peaked
        negated Ackleys; for the unimodal variants it is a single negated
        Ackley peaked at ``self.centers[0]``.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.variant == "multimodal":
            total = np.zeros(X.shape[0], dtype=float)
            for center in self.centers:
                total = total + _negated_ackley(X, center, b=self._b)
            return total
        return _negated_ackley(X, self.centers[0], b=self._b)

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
