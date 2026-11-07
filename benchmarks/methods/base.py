"""
benchmarks/methods/base.py
==========================
The contract every benchmarked optimiser satisfies, and the measurement model they
all share.

Two objects, deliberately separate:

* :class:`Problem` — the thing being optimised, as a method is allowed to see it:
  the unit box ``[0, 1]^dim``, a measurement budget in points, a preferred batch
  size, and :meth:`Problem.evaluate`. It owns the noise model and the budget, so
  every method is measured the same way and none of them can overspend. The
  ground truth (the noiseless objective, the true optima) is NOT reachable through
  it — the runner holds that and scores the run afterwards.

* :class:`Method` — an optimiser. It gets a :class:`Problem` and spends the budget
  however it likes: point by point, in batches, or a line at a time. Most methods
  are naturally ask/tell, and :class:`AskTellMethod` turns ``ask``/``tell`` into
  ``run``; ZoMBI-Hop owns its own loop and subclasses :class:`Method` directly.

Stopping
--------
A run ends when :meth:`Problem.evaluate` raises :class:`BudgetExhausted` (the budget
is spent) or :class:`TimeLimitReached` (the wall-clock safety ceiling passed). Both
derive from ``BaseException`` on purpose: optimisers catch broad ``Exception`` in
their inner loops (GP refit retries, failure recovery), and a stop that one of them
swallowed would leave the run spinning against an objective that refuses to
measure. The runner is the only place that catches them.

Measurement model
-----------------
``evaluate(X)`` measures each requested row once::

    x_actual = clip(x_requested + N(0, input_noise^2 I), 0, 1)
    y        = f(x_actual) * (1 + N(0, output_noise_frac^2))

``output_noise_frac`` defaults to 0.045, the multiplicative metrology noise
``run_mobo`` simulates (``OUTPUT_NOISE_FRAC``). ``input_noise`` defaults to 0: the
composition-space actuation noise that ``run_mobo`` models has no counterpart on a
Cartesian benchmark, and a standard black-box benchmark assumes the optimiser gets
the point it asked for. Both are recorded in the manifest either way. Methods are
handed ``x_actual`` back, so with input noise on they still fit their models to
where they actually measured.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

import numpy as np

#: Multiplicative output noise, matching ``run_mobo.OUTPUT_NOISE_FRAC``.
DEFAULT_OUTPUT_NOISE_FRAC = 0.045


class BudgetExhausted(BaseException):
    """The measurement budget is spent. ``BaseException`` on purpose — see above."""


class TimeLimitReached(BudgetExhausted):
    """The wall-clock safety ceiling passed before the budget was spent.

    A subclass of :class:`BudgetExhausted` so one handler ends the run either way;
    the runner tells them apart to record ``budget_hit``.
    """


# ─── The problem ─────────────────────────────────────────────────────────────────

class Problem:
    """A noisy black-box maximisation problem on the unit box ``[0, 1]^dim``.

    Parameters
    ----------
    fn : callable
        ``(n, dim) ndarray -> (n,) ndarray``, the noiseless objective. Private to
        the problem: methods only ever see noisy measurements of it.
    dim : int
    budget : int
        Total points a method may measure, initial design included.
    batch_size : int
        The batch a method *should* measure per call — ZoMBI-Hop's points per line,
        and the ``q`` of the batch baselines. Advisory: ``evaluate`` accepts any
        number of rows, and truncates the call that would overrun the budget.
    input_noise, output_noise_frac : float
        The measurement model (module docstring).
    seed : int
        Seeds the measurement-noise stream. It is set from the landscape, not the
        method, so every method on a cell draws from the same noise stream
        (common random numbers).
    deadline : float, optional
        ``time.time()`` value after which ``evaluate`` raises
        :class:`TimeLimitReached`.
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], dim: int, *,
                 budget: int, batch_size: int, input_noise: float = 0.0,
                 output_noise_frac: float = DEFAULT_OUTPUT_NOISE_FRAC,
                 seed: int = 0, deadline: float | None = None) -> None:
        if budget < 1 or batch_size < 1:
            raise ValueError(f"budget ({budget}) and batch_size ({batch_size}) must be >= 1")
        self._fn = fn
        self.dim = int(dim)
        self.budget = int(budget)
        self.batch_size = int(batch_size)
        self.input_noise = float(input_noise)
        self.output_noise_frac = float(output_noise_frac)
        self.deadline = deadline
        self._rng = np.random.default_rng(int(seed))

        # History, one entry per evaluate() call; concatenated on demand.
        self._X_req: list[np.ndarray] = []
        self._X_act: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []
        self._F: list[np.ndarray] = []
        self._tags: list[dict] = []
        self.n_evaluated = 0
        self.n_clipped = 0          # requested coordinates outside the box, clipped
        self.stop_reason: str | None = None

        #: Called as ``hook(problem)`` at the top of every evaluate(), i.e. with the
        #: state as of the previous batch. The runner snapshots a method's declared
        #: needles this way.
        self.before_batch: list[Callable[["Problem"], None]] = []

    # ── What a method may read ──

    @property
    def bounds(self) -> np.ndarray:
        """``(2, dim)``: row 0 lower, row 1 upper (the unit box)."""
        return np.stack([np.zeros(self.dim), np.ones(self.dim)])

    @property
    def remaining(self) -> int:
        return self.budget - self.n_evaluated

    @property
    def n_batches(self) -> int:
        return len(self._Y)

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def evaluate(self, X, **tags: Any) -> tuple[np.ndarray, np.ndarray]:
        """Measure the rows of ``X``. Returns ``(X_actual, Y)`` as ndarrays.

        ``tags`` are per-batch metadata a method wants recorded beside its points
        (ZoMBI-Hop passes its activation and zoom level); they become columns of
        ``points.csv``.

        Raises :class:`BudgetExhausted` when nothing is left to spend and
        :class:`TimeLimitReached` once the deadline has passed — both BEFORE
        measuring, so a run is never credited with points it was not allowed.
        A call asking for more than :attr:`remaining` measures the first
        ``remaining`` rows and returns only those.
        """
        for hook in self.before_batch:
            hook(self)
        if self.remaining <= 0:
            self.stop_reason = "budget"
            raise BudgetExhausted(f"{self.n_evaluated}/{self.budget} points measured")
        if self.deadline is not None and time.time() >= self.deadline:
            self.stop_reason = "time"
            raise TimeLimitReached(
                f"wall-clock ceiling reached at {self.n_evaluated}/{self.budget} points")

        X = np.atleast_2d(np.asarray(_to_numpy(X), dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f"evaluate expects (n, {self.dim}) points, got {X.shape}")
        X = X[: self.remaining]
        outside = (X < 0.0) | (X > 1.0)
        self.n_clipped += int(outside.sum())
        X_req = np.clip(X, 0.0, 1.0)

        X_act = X_req
        if self.input_noise > 0:
            X_act = np.clip(X_req + self._rng.normal(0.0, self.input_noise, X_req.shape),
                            0.0, 1.0)
        F = np.asarray(self._fn(X_act), dtype=float).reshape(-1)
        Y = F * (1.0 + self._rng.normal(0.0, self.output_noise_frac, F.shape))

        self._X_req.append(X_req)
        self._X_act.append(X_act)
        self._Y.append(Y)
        self._F.append(F)
        self._tags.append(dict(tags))
        self.n_evaluated += len(Y)
        return X_act.copy(), Y.copy()

    # ── History (read by methods that want it, and by the runner) ──

    def _cat(self, parts: list[np.ndarray], width: int | None) -> np.ndarray:
        if not parts:
            return np.empty((0, width) if width else (0,))
        return np.concatenate(parts, axis=0)

    @property
    def X_requested(self) -> np.ndarray:
        return self._cat(self._X_req, self.dim)

    @property
    def X(self) -> np.ndarray:
        """Every measured point (actual, i.e. after input noise)."""
        return self._cat(self._X_act, self.dim)

    @property
    def Y(self) -> np.ndarray:
        """Every noisy measurement, aligned with :attr:`X`."""
        return self._cat(self._Y, None)

    # ── Runner-only (ground truth / bookkeeping). Methods must not read these. ──

    def _true_values(self) -> np.ndarray:
        return self._cat(self._F, None)

    def _batch_index(self) -> np.ndarray:
        if not self._Y:
            return np.empty(0, dtype=int)
        return np.concatenate([np.full(len(y), i) for i, y in enumerate(self._Y)])

    def _tag_columns(self) -> dict[str, np.ndarray]:
        """Per-point columns from the per-batch tags (NaN where a batch lacked one)."""
        keys = sorted({k for t in self._tags for k in t})
        out = {}
        for k in keys:
            out[k] = np.concatenate([
                np.full(len(y), t.get(k, np.nan), dtype=object)
                for t, y in zip(self._tags, self._Y)])
        return out


def _to_numpy(X):
    """Accept ndarrays, lists and torch tensors without importing torch here."""
    if hasattr(X, "detach"):
        return X.detach().cpu().numpy()
    return X


# ─── Methods ─────────────────────────────────────────────────────────────────────

class Method(ABC):
    """An optimiser under benchmark.

    Subclass, set :attr:`name` (the registry key and the directory name its cells
    are written under), list every tunable in :attr:`defaults`, and implement
    :meth:`run`. Register it with :func:`benchmarks.methods.register` and it is
    available to the sweep as ``--methods <name>``.

    Configuration is a flat dict merged over :attr:`defaults`; an unknown key is a
    ``TypeError`` rather than silently ignored, because a typo in a sweep override
    would otherwise run the default and be recorded as the override.

    Methods that *declare* optima as part of the algorithm (ZoMBI-Hop's needles) set
    :attr:`declares_needles` and implement :meth:`declared_needles`. Everything else
    is scored on the needles a :mod:`~benchmarks.methods.extract` extractor finds in
    its samples — see the README for why the extractor is also applied to the
    declaring methods.
    """

    #: Registry key; also the per-method directory name in a campaign.
    name: ClassVar[str] = ""
    #: One line for ``python -m benchmarks.methods list``.
    description: ClassVar[str] = ""
    #: Every configurable key with its default.
    defaults: ClassVar[dict[str, Any]] = {}
    #: True when the algorithm itself outputs a set of optima.
    declares_needles: ClassVar[bool] = False

    def __init__(self, config: dict | None = None, *, seed: int = 0,
                 device: str = "cpu") -> None:
        config = dict(config or {})
        unknown = sorted(set(config) - set(self.defaults))
        if unknown:
            raise TypeError(f"{self.name}: unknown config key(s) {unknown}; "
                            f"known: {sorted(self.defaults)}")
        self.config: dict[str, Any] = {**copy.deepcopy(self.defaults), **config}
        self.seed = int(seed)
        self.device = str(device)
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def run(self, problem: Problem) -> None:
        """Spend ``problem``'s budget. Return normally or let the stop propagate."""

    def declared_needles(self) -> np.ndarray | None:
        """``(k, dim)`` optima this method declares right now (declaring methods only)."""
        return None

    def needle_records(self) -> list[dict]:
        """Per-needle extra columns for ``needles.csv`` (aligned with declared_needles)."""
        return []

    def point_columns(self) -> dict[str, np.ndarray]:
        """Per-sample extra columns for ``points.csv``, aligned with ``problem.X``."""
        return {}

    def summary(self) -> dict:
        """Method-specific scalars merged into ``metrics.json`` (e.g. restart counts)."""
        return {}

    def write_artifacts(self, trial_dir: str) -> None:
        """Method-specific files for the cell directory (e.g. a trust-region log)."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, config={self.config})"


class AskTellMethod(Method):
    """A method expressed as ``ask`` / ``tell``; :meth:`run` drives the loop.

    Implement :meth:`ask` (propose ``n`` points in the unit box) and :meth:`tell`
    (ingest the measurements); optionally :meth:`setup`, called once with the
    problem before the first ask. The loop asks for ``problem.batch_size`` points at
    a time, so every batch method runs at the same ``q`` ZoMBI-Hop measures per line.
    """

    def setup(self, problem: Problem) -> None:
        """Called once before the first :meth:`ask`."""

    @abstractmethod
    def ask(self, n: int) -> np.ndarray:
        """``(n, dim)`` points in ``[0, 1]^dim`` to measure next."""

    @abstractmethod
    def tell(self, X_requested: np.ndarray, X_actual: np.ndarray, Y: np.ndarray) -> None:
        """Ingest one batch of measurements (``Y`` is noisy, to be MAXIMISED)."""

    def run(self, problem: Problem) -> None:
        self.problem = problem
        self.setup(problem)
        while problem.remaining > 0:
            n = min(problem.batch_size, problem.remaining)
            X = np.atleast_2d(np.asarray(self.ask(n), dtype=float))
            X_act, Y = problem.evaluate(X)
            self.tell(X[: len(Y)], X_act, Y)


# ─── Shared helpers ──────────────────────────────────────────────────────────────

def sobol_design(n: int, dim: int, seed: int) -> np.ndarray:
    """``n`` scrambled-Sobol' points in the unit box (the baselines' initial design)."""
    import warnings

    from scipy.stats import qmc

    if n <= 0:
        return np.empty((0, dim))
    with warnings.catch_warnings():
        # "The balance properties of Sobol' points require n to be a power of 2":
        # true, and irrelevant for a 48-point initial design.
        warnings.simplefilter("ignore", UserWarning)
        return qmc.Sobol(d=dim, scramble=True, seed=int(seed) & 0xFFFFFFFF).random(int(n))
