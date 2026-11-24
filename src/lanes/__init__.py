"""
src.lanes: the two-lane ZoMBI-Hop engine (opt-in; DESIGN.md benchmarks/zoom_fix/lanes/report/review/impl/design).

The validated sandbox arm ``im+cross_cover_frac=1.01__zbudget@z2_min=3+catchup`` ("the leader"), ported
from ``benchmarks/zoom_fix/lanes/``. Nothing in ``src/core``, ``src/utils``, ``src/hparams.py`` changes for
it; ``ZoMBIHop.run()`` never imports this package.

Public names (cheap to import: registry, options, presets) plus a lazy ``build_lane_engine`` that imports
``src.lanes.engine`` on its first call.

No import-time side effects (REVIEW F12): importing ``src.lanes`` (or any of its modules) patches nothing.
In particular Fix 4 (the seeded ``random_simplex`` wrapper, ``src.lanes.rng``) is installed only while an
engine exists (built -> ``engine.close()``).
"""
from __future__ import annotations

from .hparams import (
    LANES_HPARAMS,
    LANES_KNOWN_NAMES,
    LANES_LIVE_EDITABLE,
    check_lanes_conflicts,
    lanes_conflicts,
    lanes_defaults,
    lanes_lines_per_activation,
    lanes_ui_categories,
    lanes_warnings,
)
from .options import PRESETS, LaneOptions


def build_lane_engine(*args, **kwargs):
    """Build a lanes engine (``src.lanes.engine.build_lane_engine``; DESIGN.md 2.1). Imported lazily so
    that importing ``src.lanes`` stays cheap and side-effect free."""
    from .engine import build_lane_engine as _build
    return _build(*args, **kwargs)


__all__ = [
    "PRESETS",
    "LaneOptions",
    "LANES_HPARAMS",
    "LANES_KNOWN_NAMES",
    "LANES_LIVE_EDITABLE",
    "lanes_defaults",
    "lanes_conflicts",
    "check_lanes_conflicts",
    "lanes_warnings",
    "lanes_lines_per_activation",
    "lanes_ui_categories",
    "build_lane_engine",
]
