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


class preserve_torch_defaults:
    """Undo the core's import-time mutation of global torch state.

    ``src/core/zombihop.py:29-35`` runs at import time and, whenever CUDA is
    available, calls ``torch.set_default_device("cuda")`` and
    ``torch.set_default_dtype(torch.float32)``. That is reasonable for a dedicated
    ZoMBI-Hop process and wrong for a benchmark process: every BoTorch baseline
    that later builds a tensor without an explicit device/dtype would silently
    land on the GPU in float32, so the baselines would run at a different
    numerical precision from the method they are compared against.

    The branch never fires on a CPU-only box, so this is invisible until the suite
    runs on the cluster, which is exactly the kind of bug that yields a quietly
    wrong result. Snapshot and restore here rather than edit the core.
    """

    def __enter__(self):
        import torch
        self._torch = torch
        self._dtype = torch.get_default_dtype()
        try:
            self._device = torch.get_default_device()
        except AttributeError:      # torch < 2.3
            self._device = None
        return self

    def __exit__(self, *exc):
        self._torch.set_default_dtype(self._dtype)
        if self._device is not None:
            self._torch.set_default_device(self._device)
        return False


@lru_cache(maxsize=1)
def run_mobo():
    """``optimize/run_mobo.py``: noise constants, ZOMBI_FIXED, LineBO wrappers."""
    _ensure_path()
    with preserve_torch_defaults():
        import run_mobo as rm  # noqa: PLC0415
    return rm


@lru_cache(maxsize=1)
def evaluate():
    """``optimize/evaluate.py``: the reference way to run the core on a landscape."""
    _ensure_path()
    with preserve_torch_defaults():
        from optimize import evaluate as ev  # noqa: PLC0415
    return ev


@lru_cache(maxsize=1)
def eval_metrics():
    """``optimize/eval_metrics.py``: Hungarian dist_to_needles, dup fraction, radii."""
    _ensure_path()
    with preserve_torch_defaults():
        from optimize import eval_metrics as em  # noqa: PLC0415
    return em
