"""Fix 4, scoped to a lanes engine's lifetime (DESIGN.md section 1.4, N7; REVIEW F2).

``src.utils.simplex.random_simplex(..., seed=None)`` builds a fresh ``torch.Generator`` with its default
seed, so every unseeded call returns the SAME samples (memory "random_simplex unseeded bug"). The sandbox
fixed that process-wide with ``benchmarks/zoom_fix/proposed_fixes.fix_rng`` (PF 406-420): it replaces the
``random_simplex`` attribute of ``src.utils.domain``, ``src.core.zombihop`` and ``src.core.linebo`` with a
wrapper that draws a seed from the global torch RNG when the caller passes none.

The lanes engine was validated with that wrapper, so it needs it; ``ZoMBIHop.run()`` must never see it.
``SeededSimplex.install()`` therefore installs the same wrapper into the same three module attributes
when an engine is built and ``token.uninstall()`` restores them in ``engine.close()``:

* The wrapper body is token-identical to PF 412-415 (REVIEW F2; tests/test_lanes_rng.py compares the ASTs
  with the sandbox file when present). ``torch.randint(0, 2 ** 62, (1,))`` has no ``device=`` and no
  ``generator=``, so the draw comes from the DEFAULT DEVICE's generator: importing ``src.core.zombihop``
  calls ``torch.set_default_device("cuda")`` when CUDA is available (zombihop.py 49-55), so on the 4090 it
  is the CUDA generator, exactly as in the sandbox. One draw per unseeded call.
* Ref-counted: the first install saves the three attributes and sets the wrapper; nested installs only
  count; the last uninstall restores the saved attributes (only where the wrapper is still in place: an
  attribute someone else replaced meanwhile is left alone). ``uninstall()`` is idempotent per token.
* If ``src.utils.domain.random_simplex`` already carries ``_zoomfix`` (the sandbox PF wrapper, permanent in
  a sandbox process) or ``_lanes_seeded``, install does not wrap again: it only counts, and nothing is
  restored for it (as PF 409-410).
* Self-healing: if the attributes were restored behind the counter's back (e.g. a test fixture), the next
  install wraps again.

No import-time side effects (REVIEW F12): nothing is patched until ``install()`` is called.

Since 2026-11-24 ``random_simplex`` itself draws an unseeded call's seed with the same
``torch.randint(0, 2 ** 62, (1,))``, so the bug above is fixed for every caller and this wrapper no
longer changes any samples. It is kept so the engine draws exactly as validated even when
``ZOMBI_LEGACY_UNSEEDED_SIMPLEX=1`` restores the old behaviour for a resumed MOBO run.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional

import torch

from src.core import linebo as _linebo
from src.core import zombihop as _zh
from src.utils import domain as _domain

# PF 418: the three modules whose ``random_simplex`` name the sampling code resolves at call time
# (zombihop.py 15-21, linebo.py 21-25, domain.py 36-41; GPSimplex samples through ``domain.sample``).
_MODULES = (_domain, _zh, _linebo)
_ATTR = "random_simplex"


def _wrapper(orig: Callable) -> Callable:
    """PF 412-415 verbatim (token-identical), marked ``_lanes_seeded``."""

    def random_simplex(*args, seed=None, **kwargs):
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, (1,)).item())
        return orig(*args, seed=seed, **kwargs)

    random_simplex._lanes_seeded = True
    return random_simplex


def is_seeded(fn) -> bool:
    """True for a seeding wrapper (the sandbox's ``_zoomfix`` one or ours)."""
    return bool(getattr(fn, "_zoomfix", False) or getattr(fn, "_lanes_seeded", False))


class SeededSimplex:
    """Token of one ``install()``; ``uninstall()`` it once the engine is closed."""

    _lock = threading.RLock()
    _count = 0
    _saved: Optional[Dict[object, Callable]] = None     # module -> original attribute (None: nothing to restore)
    _installed: Optional[Callable] = None               # the wrapper we set (None: we did not wrap)

    def __init__(self):
        self._active = True

    # ── class API ──
    @classmethod
    def install(cls) -> "SeededSimplex":
        with cls._lock:
            cur = getattr(_domain, _ATTR)
            if cls._count == 0 or not is_seeded(cur):
                if is_seeded(cur):
                    # a seeding wrapper is already there (sandbox PF): count only, restore nothing
                    cls._saved, cls._installed = None, None
                else:
                    saved = {m: getattr(m, _ATTR) for m in _MODULES if hasattr(m, _ATTR)}
                    wrapper = _wrapper(cur)
                    for m in saved:
                        setattr(m, _ATTR, wrapper)
                    cls._saved, cls._installed = saved, wrapper
            cls._count += 1
            return cls()

    @classmethod
    def active_count(cls) -> int:
        return cls._count

    @classmethod
    def installed(cls) -> bool:
        """True while a seeding wrapper is in ``src.utils.domain.random_simplex``."""
        return is_seeded(getattr(_domain, _ATTR))

    @classmethod
    def _reset(cls) -> None:
        """Test helper: restore what we saved and forget every token (never used by the engine)."""
        with cls._lock:
            cls._restore()
            cls._count = 0

    @classmethod
    def _restore(cls) -> None:
        if cls._saved is not None:
            for m, f in cls._saved.items():
                if getattr(m, _ATTR, None) is cls._installed:
                    setattr(m, _ATTR, f)
        cls._saved, cls._installed = None, None

    # ── token API ──
    def uninstall(self) -> None:
        """Idempotent; the last outstanding token restores the original attributes."""
        cls = type(self)
        with cls._lock:
            if not self._active:
                return
            self._active = False
            cls._count = max(0, cls._count - 1)
            if cls._count == 0:
                cls._restore()

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self) -> "SeededSimplex":
        return self

    def __exit__(self, *exc) -> bool:
        self.uninstall()
        return False


@contextmanager
def seeded_simplex() -> Iterator[SeededSimplex]:
    """``with seeded_simplex(): ...`` = install for the block, uninstall after it (even on error)."""
    tok = SeededSimplex.install()
    try:
        yield tok
    finally:
        tok.uninstall()


__all__ = ["SeededSimplex", "seeded_simplex", "is_seeded"]
