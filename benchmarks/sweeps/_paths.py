"""
benchmarks/sweeps/_paths.py
===========================
sys.path bootstrap, delegated to the ablations package.

This sweep drives the same shared code (``optimize/run_mobo.py`` imported as a
TOP-LEVEL module, ``src.*``, ``synthetic_data.*``) and hits the same two Windows
hazards — a matplotlib backend chosen at ``run_mobo`` import time, and cp1252
stdio blowing up on the non-ASCII log lines inside the needle-declaration path.
``benchmarks.ablations._paths`` already solves all of that and is idempotent, so
re-solving it here would only be a second copy to keep in sync.
"""

from __future__ import annotations

from benchmarks.ablations._paths import (  # noqa: F401
    OPTIMIZE_DIR,
    REPO_ROOT,
    ensure_paths,
)
