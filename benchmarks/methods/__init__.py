"""
benchmarks/methods — black-box optimisers behind one interface, for benchmarking
ZoMBI-Hop against the standard alternatives on identical problems.

Built in
--------
    random      random sampling (uniform, or Sobol' with sampler="sobol")
    gp_bo       standard BO: BoTorch SingleTaskGP + batch qLogNEI
    turbo       TuRBO-m trust-region BO (Eriksson et al. 2019), m=1 by default
    hebo        HEBO (Cowen-Rivers et al. 2022), the authors' package — run
                ``python -m benchmarks.methods install-hebo`` once first
    zombi_hop   ZoMBI-Hop from ``src/``, on the unit box

Every method maximises a noisy objective on ``[0, 1]^dim`` through a
:class:`Problem`, which owns the measurement noise and the point budget, so all of
them are measured identically and none can overspend. See ``README.md``.

Adding a method
---------------
Most optimisers are ask/tell::

    import numpy as np
    from benchmarks.methods import AskTellMethod, register

    @register
    class NelderMead(AskTellMethod):
        name = "nelder_mead"                 # --methods nelder_mead
        description = "one line for `list`"
        defaults = {"n_init": 48}            # every tunable, with its default

        def setup(self, problem):            # once, before the first ask
            self.dim = problem.dim

        def ask(self, n) -> np.ndarray:      # (n, dim) points in [0, 1]^dim
            ...

        def tell(self, X_requested, X_actual, Y):   # Y is noisy; MAXIMISE it
            ...

Put the module in this package and import it at the bottom of this file, or keep it
anywhere and name it by path: ``--methods path/to/file.py:NelderMead``. A method
that owns its loop (ZoMBI-Hop calls the objective itself) subclasses :class:`Method`
and implements ``run(problem)``; a method that declares its own optima sets
``declares_needles = True`` and implements ``declared_needles()``. Everything else is
scored through a needle extractor (``extract.py``).

    python -m benchmarks.methods list
    python -m benchmarks.methods smoke --methods random,turbo
"""

from __future__ import annotations

from ._paths import ensure_paths

ensure_paths()

from .base import (  # noqa: E402
    AskTellMethod,
    BudgetExhausted,
    Method,
    Problem,
    TimeLimitReached,
    sobol_design,
)
from .extract import EXTRACTORS, make_extractor, register_extractor  # noqa: E402
from .registry import (  # noqa: E402
    METHODS,
    available_methods,
    get_method,
    make_method,
    register,
)

# Built-in methods register themselves on import. Each keeps its heavy imports
# (torch, BoTorch, HEBO, ZoMBI-Hop) inside its methods, so this stays cheap.
from . import random_search, gp_bo, turbo, hebo_method, zombihop  # noqa: E402,F401

__all__ = [
    "AskTellMethod",
    "BudgetExhausted",
    "EXTRACTORS",
    "METHODS",
    "Method",
    "Problem",
    "TimeLimitReached",
    "available_methods",
    "get_method",
    "make_extractor",
    "make_method",
    "register",
    "register_extractor",
    "sobol_design",
]
