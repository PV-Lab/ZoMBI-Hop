"""Interactive ternary plot — now part of the unified oracle explorer.

The original tunable realistic Ackley app lives in ``plot_oracles_interactive.py``
alongside messy / gaussian / planted_bumps / rastrigin_ilr / Multi-Ackley.

Usage
-----
  python synthetic_data/plot_oracles_interactive.py
  python synthetic_data/plot_3d.py              # opens realistic Ackley preset
"""

from __future__ import annotations

if __name__ == "__main__":
    from synthetic_data.plot_oracles_interactive import main
    main(initial_oracle="realistic_ackley")
