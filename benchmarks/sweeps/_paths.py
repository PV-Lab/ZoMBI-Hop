"""
benchmarks/sweeps/_paths.py
===========================
sys.path bootstrap, delegated to ``benchmarks.methods._paths`` (repo root and
``optimize/`` on the path, headless matplotlib, UTF-8 stdio).

Not ``benchmarks.ablations._paths`` any more: importing anything under
``benchmarks.ablations`` imports ``run_mobo`` and, through it, ZoMBI-Hop's
global-torch-default side effect, which a process running the BoTorch baselines
must not inherit. See ``benchmarks/methods/_paths.py``.
"""

from __future__ import annotations

from benchmarks.methods._paths import (  # noqa: F401
    OPTIMIZE_DIR,
    REPO_ROOT,
    ensure_paths,
)
