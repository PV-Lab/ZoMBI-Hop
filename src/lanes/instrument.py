"""Exclusive wall-time per engine component (instrumentation only; never read by a decision).

The sandbox ``LaneStore._install_timers`` (``benchmarks/zoom_fix/lanes/lane_zombihop.py`` LZ 429-460)
wrapped ten bound methods of the GP handler, the LineBO, the DataHandler and the ZoMBIHop instance with a
timing pass-through. The port replaces those instance patches with overrides in the lane subclasses
(``src/lanes/handlers.py``, ``src/lanes/linebo.py``; DESIGN.md 1.3.4):

    def fit(self, X, Y):
        with self._lanes_timers.timed("fit"):
            return <memo>(super().fit ...)

``Timers`` keeps the sandbox's logic exactly: one shared stack of child times, so a component's seconds are
EXCLUSIVE of the timed components it calls (``comp[key] += max(dt - child, 0)``), and ``comp[key + "_n"]``
counts calls; both are updated even when the call raises (the sandbox's ``try/finally``).
``comp`` is a ``defaultdict(float)`` shared with the engine (``engine.comp = engine.timers.comp``), which
also adds its own ``comp["ingest"]`` directly (LZ 539).

``TIMED_METHODS`` lists the sandbox's (owner, method, key) triples (LZ 449-458) so every implementer
uses the same keys. Only ``time.perf_counter`` is used: no RNG, no state a decision reads.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, DefaultDict, List, Optional

# (owner attribute on the engine, method name, timer key): LZ 449-458. Owners: gp = GPSimplex handler,
# linebo = LineBO, dh = DataHandler, z = ZoMBIHop.
TIMED_METHODS = (
    ("gp", "fit", "fit"),
    ("gp", "get_candidate", "candidate"),
    ("gp", "determine_penalty_ellipsoid", "ellipsoid"),
    ("linebo", "ranked_line_endpoints", "lines"),
    ("dh", "add_all_points", "add_points"),
    ("dh", "take_snapshot", "snapshot"),
    ("dh", "_relabel_pared_with_medians", "relabel"),
    ("z", "_check_convergence_to_needle", "convergence"),
    ("z", "_declare_needle_from_point", "declare"),
    ("z", "_penalize_capped_zone", "cap_zone"),
)
TIMER_KEYS = {(owner, meth): key for owner, meth, key in TIMED_METHODS}


class _Timed:
    """One timed call (context manager); the body of the sandbox's ``_timed`` (LZ 434-447)."""
    __slots__ = ("_timers", "_key", "_t")

    def __init__(self, timers: "Timers", key: str):
        self._timers = timers
        self._key = key
        self._t = 0.0

    def __enter__(self) -> "_Timed":
        self._timers._stack.append([0.0])
        self._t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        stack = self._timers._stack
        comp = self._timers.comp
        key = self._key
        dt = time.perf_counter() - self._t
        child = stack.pop()[0]
        if stack:
            stack[-1][0] += dt
        comp[key] += max(dt - child, 0.0)
        comp[key + "_n"] += 1
        return False                                    # never swallow the exception


class Timers:
    """Exclusive seconds per component.

    ``timed(key)`` -> context manager timing one call; nested timed calls are subtracted from their
    parent (one shared stack, as LZ 430). ``wrap(fn, key)`` -> a plain function with the same timing (for
    callables that are not methods of the lane subclasses). ``comp``: ``defaultdict(float)`` of seconds
    per key plus ``<key>_n`` call counts."""

    def __init__(self, comp: Optional[DefaultDict[str, float]] = None):
        self.comp: DefaultDict[str, float] = comp if comp is not None else defaultdict(float)
        self._stack: List[list] = []

    def timed(self, key: str) -> _Timed:
        return _Timed(self, key)

    def wrap(self, fn: Callable, key: str) -> Callable:
        timers = self

        def _timed(*a, **k):
            with timers.timed(key):
                return fn(*a, **k)
        _timed.__wrapped__ = fn
        return _timed

    @property
    def depth(self) -> int:
        """Number of timed calls currently open (0 between decisions)."""
        return len(self._stack)

    def snapshot(self) -> dict:
        """A plain-dict copy of ``comp`` (the sandbox's ``c0 = dict(self.s.comp)``, LZ 684 / 710)."""
        return dict(self.comp)

    def rounded(self, nd: int = 3) -> dict:
        """``comp`` rounded as ``LaneStore.summary`` reports it (LZ 591)."""
        return {k: round(v, nd) for k, v in self.comp.items()}


__all__ = ["Timers", "TIMED_METHODS", "TIMER_KEYS"]
