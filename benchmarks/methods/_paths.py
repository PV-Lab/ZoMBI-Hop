"""
benchmarks/methods/_paths.py
============================
sys.path bootstrap: the repo root (``src.*``, ``synthetic_data.*``) and ``optimize/``
(``eval_metrics``, which the shared scoring functions live in) go on ``sys.path``,
matplotlib is pinned to a headless backend, and stdio is forced to UTF-8 (the
ZoMBI-Hop internals log non-ASCII; see ``benchmarks/ablations/_paths.py``).

Deliberately NOT delegated to ``benchmarks.ablations._paths``: importing anything
under ``benchmarks.ablations`` runs that package's ``__init__``, which imports
``run_mobo`` and through it ``src.core.zombihop`` — and that module switches torch's
*global* default device to CUDA at import time. A benchmark process running a
BoTorch baseline must not inherit that, so this package imports ZoMBI-Hop only
inside the ZoMBI-Hop adapter, and each sweep cell runs in its own process.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OPTIMIZE_DIR = os.path.join(REPO_ROOT, "optimize")

_READY = False


def ensure_paths(*, headless: bool = True) -> None:
    """Idempotent; safe to call from every module's import block."""
    global _READY
    if _READY:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            pass
    if headless and not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = "Agg"
    for p in (REPO_ROOT, OPTIMIZE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    _READY = True
