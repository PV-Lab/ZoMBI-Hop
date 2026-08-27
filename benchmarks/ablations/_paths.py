"""
benchmarks/ablations/_paths.py
==============================
sys.path bootstrap shared by every module in this package.

``optimize/run_mobo.py`` is imported as a TOP-LEVEL module (``import run_mobo``),
not as ``optimize.run_mobo`` — its own internals and its siblings (``pareto``,
``plot_metrics``, ``coverage_plot``, ``eval_metrics``) import each other that way,
so ``optimize/`` has to be on ``sys.path`` for any of it to resolve. The repo root
goes on too, for ``src.*`` and ``synthetic_data.*``.

``import matplotlib.pyplot`` runs at ``run_mobo`` import time and picks a backend
then; on a headless worker that is a hard failure unless a non-interactive backend
is already selected. So the backend is pinned BEFORE the import, not after.

Stdio is forced to UTF-8 for the same reason ``warm_start/__init__.py`` does it.
The shared code these ablations drive was written on a UTF-8 Linux node and logs
non-ASCII freely — the radius-cap message in
``GPSimplex.determine_penalty_ellipsoid`` contains a U+2192 arrow. On Windows the
default console/redirect encoding is cp1252, so that print raises
``UnicodeEncodeError``, and because it fires *inside* the needle-declaration path
it aborts the ZoMBI run mid-trial. That would hit the ablation arms unevenly (A4
wraps exactly that function, and arms differ in how often they hit the cap), so
left unfixed it would corrupt the comparison rather than merely fail loudly.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OPTIMIZE_DIR = os.path.join(REPO_ROOT, "optimize")

_READY = False


def ensure_paths(*, headless: bool = True) -> None:
    """Put the repo root and ``optimize/`` on ``sys.path``; pin a headless backend.

    Idempotent — safe to call from every module's import block.
    """
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
            # Already-wrapped or non-reconfigurable stream (e.g. a captured pipe).
            pass
    if headless and not os.environ.get("MPLBACKEND"):
        # Set the env var rather than calling matplotlib.use(): this runs before
        # matplotlib is imported at all, and MPLBACKEND is what run_mobo's own
        # _configure_mpl_backend defers to.
        os.environ["MPLBACKEND"] = "Agg"
    for p in (REPO_ROOT, OPTIMIZE_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    _READY = True
