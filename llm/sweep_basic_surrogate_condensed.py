"""
llm/sweep_basic_surrogate_condensed.py
======================================
Condensed-curve twin of ``sweep_basic_surrogate.py``.

Identical in every respect — same generative surrogate, same baseline, same
cold-start → inject-every-k → resume-exact-state continuation, same
common-random-numbers, same metrics / summary / convergence plot, and the SAME
injection prompt — EXCEPT how the measured curve groups (absorption spectrum, the
dark/light stability voltage sweep, the initial/final PL spectra) are shown to the
LLM:

    * ``sweep_basic_surrogate.py``  (CURVE_MODE="full")      — reconstructs and prints
      each curve in FULL on its native wavelength/voltage grid (hundreds of numbers
      per droplet).
    * this script                   (CURVE_MODE="condensed") — shows the curves as
      their compact functional-PCA scores (a few numbers per curve group), carried
      as extra columns in the same scalar tables.

The physical scalars (Bandgap, Photoconductance, Stability, the environment
channels, the degradation kinetics) are shown identically in both. Because the
surrogate objective draws are seeded identically (common random numbers) and every
non-prompt code path is shared, the ONLY difference between the two experiments is
the curve representation in the prompt — a clean A/B on full curves vs. their PCA
compression.

Only the process-wide ``CURVE_MODE`` flag is flipped here (via ``SBS.CURVE_MODE``);
everything else — including the injection prompt template and its feature sections —
is imported from ``sweep_basic_surrogate`` and ``main()`` just points at a distinct
results directory.

Usage:
  # repo-root uv venv (see MEMORY.md), NOT `conda activate zombi-hop`
  python llm/sweep_basic_surrogate_condensed.py
  python llm/sweep_basic_surrogate_condensed.py --plot <sweep_dir>
  python llm/sweep_basic_surrogate_condensed.py --regenerate <sweep_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sweep_basic_surrogate as SBS  # noqa: E402  (reuse all machinery)


def main(resume_dir=None) -> None:
    # Show the measured curves as their compact fPCA scores rather than the full
    # reconstructed spectra/sweeps. This flag is read by every renderer in SBS.
    SBS.CURVE_MODE = "condensed"
    SBS.main(sweep_prefix="sweep_surrogate_condensed",
             plot_title=("Convergence (condensed fPCA curve features): baseline vs "
                         "LLM injection cadences\n(mean ± 95% CI over repeats)"),
             resume_dir=resume_dir)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] in ("--regenerate", "-r"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_surrogate_condensed.py "
                             "--regenerate <sweep_dir>")
        SBS.regenerate_summary(Path(args[1]))
    elif args and args[0] in ("--plot", "-p"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_surrogate_condensed.py "
                             "--plot <sweep_dir>")
        SBS.plot_convergence_comparison(Path(args[1]))
    elif args and args[0] in ("--resume",):
        # Resume an existing sweep: skip reps already finished, run the rest. Pass a
        # sweep dir, or omit it to resume the most recent sweep_surrogate_condensed_*.
        main(resume_dir=SBS.resolve_resume_dir(args[1] if len(args) > 1 else None,
                                               "sweep_surrogate_condensed"))
    else:
        main()
