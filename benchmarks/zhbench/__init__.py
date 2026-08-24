"""zhbench — the ZoMBI-Hop multi-optimum benchmark (benchmarking-v2).

Measures what ZoMBI-Hop is built for: finding MANY distinct local optima of a
simplex-constrained objective inside a realistic SDL sample budget, against
standard BO baselines run exactly as their authors intended.

Nothing in this package edits the ZoMBI-Hop core. It imports from it.
"""

from .protocol import BudgetExhausted, Protocol, ObjectiveRun

__all__ = ["BudgetExhausted", "Protocol", "ObjectiveRun"]
