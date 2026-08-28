"""
benchmarks/sweeps/budget.py
===========================
A **measurement budget** instead of a wall-clock budget.

``run_mobo.run_single_trial`` bounds a trial with ``time_limit_hours``, which is
the wrong control for this sweep. The cost of one LineBO iteration is dominated by
the exact-GP refit and the acquisition ascent, both of which grow with the number
of accumulated points and with the dimension — so a wall-clock budget would hand
dim 3 several times as many experiments as dim 10, and the "dimensionality" axis
would be measuring the GP's cost curve rather than the landscape. Capping *lines*
instead gives every cell in the grid the same number of experiments, which is also
the quantity that costs money on real hardware.

The sweep's budget is **125 lines = 3000 measured compositions**, at
``run_mobo.NUM_EXPERIMENTS`` (24) points per line.

How the cap is applied
----------------------
``ZoMBIHop.run`` has no notion of a sample budget and only checks its time limit at
iteration boundaries, so the cap cannot be a parameter — it has to interrupt the
objective. This is the same mechanism ``warm_start/trial.py`` uses for its own
600-point budget, lifted out so it can wrap ``run_single_trial`` (which builds its
objective in a closure this module cannot reach) rather than a bespoke runner:

* ``run_mobo.make_linebo_wrapper`` is patched so the wrapper it returns raises
  :class:`BudgetExhausted` *before* measuring a line that would overrun the budget.
  Stopping before rather than after means a cell is never credited with
  experiments it was not allowed to run.
* ``ZoMBIHop.run`` is patched to catch that one exception and return normally, so
  ``run_single_trial`` sees an ordinary completed run and writes its full artifact
  set. Without this the exception would surface inside ``run_single_trial``'s
  ``except Exception`` guard, which prints "ZoMBI crashed" and a traceback for what
  is actually the designed stopping condition.

:class:`BudgetExhausted` derives from ``BaseException``, not ``Exception``. The
optimiser's inner loops catch broad ``Exception`` in several places (failure
retries, GP refits); a budget stop that any of those swallowed would leave the run
spinning against an objective that refuses to measure, burning the wall-clock
ceiling instead of finishing. Deriving from ``BaseException`` makes the stop
un-swallowable by anything but the handler installed here.

Init lines count
----------------
``run_mobo._gen_init_data`` measures ``N_INIT_LINES`` (2) lines before the
optimiser starts, and those 48 points are real experiments on the same landscape.
They are charged to the budget, so a 125-line cell runs 2 initial lines plus at
most 123 adaptive ones and lands at 3000 points total — not 3048.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from ._paths import ensure_paths

ensure_paths()

import run_mobo as rm  # noqa: E402
from src import ZoMBIHop  # noqa: E402

#: The sweep's budget. 125 lines x 24 points per line = 3000 measured compositions.
DEFAULT_N_LINES = 125


class BudgetExhausted(BaseException):
    """Raised by the patched objective once the line budget is spent.

    ``BaseException`` on purpose — see the module docstring. Never let this become
    an ``Exception`` subclass: the optimiser's broad handlers would eat it.
    """


@dataclass
class BudgetState:
    """What the cap did, for the cell's record.

    ``lines_measured`` counts adaptive lines only; ``n_init_lines`` is what the
    initial design already spent, and the two add up to the budget when the cap
    fired.
    """

    n_lines: int
    n_init_lines: int
    points_per_line: int
    lines_measured: int = 0
    budget_hit: bool = False

    @property
    def max_adaptive_lines(self) -> int:
        """Adaptive lines the cell may run, after the initial design is charged."""
        return max(0, int(self.n_lines) - int(self.n_init_lines))

    def to_dict(self) -> dict:
        return {
            "n_lines_budget": int(self.n_lines),
            "n_init_lines": int(self.n_init_lines),
            "points_per_line": int(self.points_per_line),
            "points_budget": int(self.n_lines) * int(self.points_per_line),
            "adaptive_lines_allowed": self.max_adaptive_lines,
            "adaptive_lines_measured": int(self.lines_measured),
            "budget_hit": bool(self.budget_hit),
        }


@contextlib.contextmanager
def line_budget(n_lines: int = DEFAULT_N_LINES, *, state: BudgetState | None = None):
    """Cap the enclosed trial at ``n_lines`` measured LineBO lines.

    Yields the :class:`BudgetState`, which after the block says how many lines were
    actually measured and whether the cap is what stopped the run. Both patches are
    unwound on exit, so cells can run back-to-back in one worker process and the
    baseline code path is untouched between them.

    ``budget_hit is False`` at the end is a signal worth reading, not a nit: it
    means something *else* stopped the cell — the wall-clock safety ceiling, or the
    optimiser's own termination — and the cell did not spend its full budget.
    """
    st = state or BudgetState(
        n_lines=int(n_lines),
        n_init_lines=int(rm.N_INIT_LINES),
        points_per_line=int(rm.NUM_EXPERIMENTS),
    )
    limit = st.max_adaptive_lines

    orig_make = rm.make_linebo_wrapper
    orig_run = ZoMBIHop.run

    def budgeted_make_linebo_wrapper(*args, **kwargs):
        inner = orig_make(*args, **kwargs)

        def wrapper(x_tell, bounds, acq_fn):
            # Checked BEFORE measuring, so the cell never exceeds its budget by the
            # line that would have crossed it.
            if st.lines_measured >= limit:
                st.budget_hit = True
                raise BudgetExhausted(
                    f"{st.lines_measured}/{limit} adaptive lines measured "
                    f"({st.n_lines}-line budget including {st.n_init_lines} init)")
            out = inner(x_tell, bounds, acq_fn)
            st.lines_measured += 1
            return out

        return wrapper

    def budgeted_run(self, *args, **kwargs):
        try:
            return orig_run(self, *args, **kwargs)
        except BudgetExhausted as exc:
            print(f"    [budget] {exc}", flush=True)
            # The same shape ``ZoMBIHop.run`` returns on a normal exit. Nothing in
            # run_single_trial reads it — it pulls results off the data handler —
            # but returning the right arity keeps any other caller honest.
            dh = self.data_handler
            return (None, dh.needles, dh.needle_vals, dh.X_all_actual, dh.Y_all)

    rm.make_linebo_wrapper = budgeted_make_linebo_wrapper
    ZoMBIHop.run = budgeted_run
    try:
        yield st
    finally:
        rm.make_linebo_wrapper = orig_make
        ZoMBIHop.run = orig_run
