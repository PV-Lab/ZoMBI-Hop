"""Import shim for the ZoMBI-Hop repo modules.

``optimize/run_mobo.py`` cannot be imported as ``optimize.run_mobo``: at line 1696
it does a bare ``from eval_metrics import ...`` that only resolves with
``optimize/`` itself on ``sys.path``. ``optimize/evaluate.py:174`` sets that up
before importing it, so we mirror that here once, in one place.

Import cost is real (~4 s, pulls torch + matplotlib + sklearn), so this module is
imported lazily by callers that actually need the core.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OPTIMIZE_DIR = os.path.join(REPO_ROOT, "optimize")


def _ensure_path() -> None:
    for p in (REPO_ROOT, _OPTIMIZE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


@lru_cache(maxsize=1)
def run_mobo():
    """``optimize/run_mobo.py`` — noise constants, ZOMBI_FIXED, LineBO wrappers."""
    _ensure_path()
    import run_mobo as rm  # noqa: PLC0415
    return rm


@lru_cache(maxsize=1)
def evaluate():
    """``optimize/evaluate.py`` — the reference way to run the core on a landscape."""
    _ensure_path()
    from optimize import evaluate as ev  # noqa: PLC0415
    return ev


@lru_cache(maxsize=1)
def eval_metrics():
    """``optimize/eval_metrics.py`` — Hungarian dist_to_needles, dup fraction, radii."""
    _ensure_path()
    from optimize import eval_metrics as em  # noqa: PLC0415
    return em
