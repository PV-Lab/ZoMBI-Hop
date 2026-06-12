"""
Dimension scaling of ZoMBI-Hop hyperparameters
==============================================

Single source of truth for the two dimension-scaling factors (see
``DIMENSION_SCALING.md`` for the derivation).  Both factors are exactly **1.0 at
the 3-simplex tuning dimension** (d=3), so a 3-D-tuned configuration is
reproduced unchanged there.

  * ``dimension_radius_scale(d)`` — Group 1 (tangent-space distances / radii),
    ``sqrt((d-1)/2)``.
  * ``dimension_line_scale(d)``   — Group 2 (sampling budgets), ``(d-1)/2``.

A process-wide toggle (default **on**) lets experiments run *with* and *without*
scaling through the very same code paths — disable it for a block of work with
the :func:`dimension_scaling_disabled` context manager (used by
``optimize/test_dim_scale.py``).  When disabled, both factors return 1.0.
"""

import math
from contextlib import contextmanager

_ENABLED = True


def dimension_scaling_enabled() -> bool:
    """True when dimension scaling is currently active."""
    return _ENABLED


def set_dimension_scaling(enabled: bool) -> None:
    """Globally enable/disable dimension scaling (process-wide)."""
    global _ENABLED
    _ENABLED = bool(enabled)


@contextmanager
def dimension_scaling_disabled():
    """Temporarily disable dimension scaling; the previous state is restored on exit.

    Construction-time reads (``ZoMBIHop.__init__``, ``LineBO.__init__``) and the
    initial-seeding budget are all evaluated while this is active, so wrapping a
    whole run disables scaling consistently across every site.
    """
    global _ENABLED
    prev = _ENABLED
    _ENABLED = False
    try:
        yield
    finally:
        _ENABLED = prev


def dimension_line_scale(d: int) -> float:
    """Group-2 linear budget factor ``(d-1)/2`` (1.0 at d=3, or 1.0 when disabled)."""
    return (d - 1) / 2.0 if _ENABLED else 1.0


def dimension_radius_scale(d: int) -> float:
    """Group-1 length/radius factor ``sqrt((d-1)/2)`` (1.0 at d=3, or 1.0 when disabled)."""
    return math.sqrt((d - 1) / 2.0) if _ENABLED else 1.0
