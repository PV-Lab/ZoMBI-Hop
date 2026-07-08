"""
llm/sweep_volume_RF.py
======================
RF-landscape twin of ``sweep_volume_control.py``.

Same volume-control experiment — every 1 / 5 / 10 ZoMBI-Hop iterations the LLM
places **reward** and/or **penalization volumes** (hyperspheres on the composition
simplex) on top of ZoMBI-Hop's acquisition — but on the same stripped-down problem
as ``sweep_basic_RF.py``:

1.  **Noise-free RF objective landscape** — the surrogate's deterministic RF
    conditional mean, with NO residual draw (see ``sweep_basic_RF``/``surrogate.py``).
2.  **Only Bandgap / Photoconductance / Stability** are surfaced as supplemental
    features (no environment, kinetics, or measured curves), each read from the
    noise-free RF mean; the prompt keeps only a compact GLOBAL stats line for those
    three plus the Objective.

All the volume mechanics (``VolumeControlAcquisition``, ``install_volume_control``,
``validate_volumes``, ``call_volume_llm``, the segment runner, aggregation, and
``main``) are inherited unchanged from ``sweep_volume_control`` (``VC``); the RF
objective / feature config is inherited from ``sweep_basic_RF`` (``RFB``). This
module only swaps in an RF-specific volume-placement prompt and points ``VC.main``
at a distinct results directory with its own baseline-reuse tag (``impl="volume_rf"``)
so an RF baseline is never confused with a noisy-surrogate one of the same seed.

Usage:
  # repo-root uv venv (see MEMORY.md), NOT `conda activate zombi-hop`
  python llm/sweep_volume_RF.py
  python llm/sweep_volume_RF.py --plot <sweep_dir>
  python llm/sweep_volume_RF.py --regenerate <sweep_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import llm_config  # noqa: E402  (kept for API parity; VC/SBS also import it)
import sweep_basic_surrogate as SBS  # noqa: E402  (prompt-section helpers)
import sweep_volume_control as VC  # noqa: E402  (all volume machinery)
import sweep_basic_RF as RFB  # noqa: E402  (noise-free RF objective + feature config)


# ════════════════════════════════════════════════════════════════════════════════
# Injection prompt (RF-specific: trimmed feature apparatus + optima-priority task)
# ════════════════════════════════════════════════════════════════════════════════

VOLUME_RF_PROMPT_TEMPLATE = """\
You are steering a Bayesian-optimization algorithm, ZoMBI-Hop, on the fly. It is
optimizing a perovskite-materials-discovery lab over the 3-simplex composition
(`FAPbI3`, `MAPbI3`, `MAPbBr3`, which always sum to 1); the `Objective` is
MAXIMIZED. ZoMBI-Hop is a zooming multi-basin optimizer that hunts MULTIPLE optima
("needles"): it fits a GP, uses an acquisition function each iteration to pick a
LineBO line of 24 measured droplets, declares a needle when expected improvement
hits the output-noise floor, penalizes that region, and moves on.

Alongside each measured droplet's `Objective`, the lab reports three interpretable
material properties measured behind it:

{system_features}

## How you steer it: reward and penalization volumes

Instead of tuning hyperparameters, you reshape ZoMBI-Hop's acquisition landscape by
placing **volumes** — hyperspheres on the composition simplex. Each volume has a
`center` (a composition, 3 numbers that will be renormalized to sum to 1) and a
`radius` (in composition-distance units; the simplex spans distances up to ~1.4,
the input/measurement noise scale is ~0.06, and a radius near {max_radius} already
covers a large fraction of the space). Radii are clamped to
[{min_radius}, {max_radius}].

Each volume also carries a `strength` in [{min_strength}, {max_strength}] that YOU
choose. For a query composition `x` at Euclidean distance `d = ||x - center||`, a
volume contributes `strength * max(0, 1 - d/radius)**2` to the acquisition — a smooth
bump that is strongest at the center and decays to exactly zero at the edge of the
ball. At the center the bump's force equals `strength` needle-penalties:
**`strength = 1` is exactly as strong as one of ZoMBI-Hop's own needle penalties**,
`strength = {max_strength}` is {max_strength}× that, and `strength = 0` is a no-op.
The PEAK force depends only on `strength`, NOT on the radius — the radius only sets
how much of the simplex the bump covers, not how hard it pushes:

- A **`penalty`** volume SUBTRACTS that bump, so ZoMBI-Hop becomes LESS likely to
  sample inside the ball (just like the penalty regions around declared needles, but
  isotropic and smoothed to the edge — it discourages, it does not hard-forbid).
- A **`reward`** volume ADDS that bump, so ZoMBI-Hop becomes MORE likely to sample
  inside the ball.

Each time you are prompted you specify the COMPLETE set of volumes to be in effect
going forward: the list you return REPLACES the current set entirely (they always
sit on top of ZoMBI-Hop's own needle penalties). So you may add new volumes, keep
existing ones, drop ones that are no longer useful, or resize/move them — you have
full control of the whole set every time.

## Volumes currently in effect

{active_volumes}

## This run so far

This is injection #{injection_idx}. ZoMBI-Hop has completed {iters_done} of
{budget} iterations for this run. Progress:

{progress_summary}

## Change since the last injection

{trend_summary}

Recent measured points — the last 2 LineBO lines (up to 48 droplets), each shown
with its composition, `Objective`, and the three material features:

{history_table}

## Supplemental measured features (GLOBAL summary over all {n_points} droplets)

Global statistics over ALL {n_points} droplets measured so far: the global mean,
the global [min, max] range, and the global Pearson correlation of each feature
with `Objective` (a positive corr means the feature tends to rise where the
objective is high). Use these as the population baseline for the top-k and needle
rows below.

{supplemental_summary}

Best droplet so far — its material features:

{best_point_summary}

## Top {top_k} droplets by Objective (with features)

The {top_k} highest-`Objective` droplets measured so far and their material
features.

{top_k_summary}

## Declared needles (local optima) with features

Each needle ZoMBI-Hop has declared, its composition and declared `Objective`
value, plus the features of the nearest measured droplet. Spread-out needle
compositions indicate healthy multi-basin coverage; needles clustered in one
corner of the simplex indicate over-exploitation of a single region (a cue to
`reward` an under-sampled region or `penalty` the crowded one).

{needle_summary}

## Your task

Specify the COMPLETE set of reward and/or penalization volumes that should be in
effect for the NEXT {interval} iteration(s). ZoMBI-Hop will continue from its exact
current internal state (same data, needles, zoom bounds); the list you return
REPLACES the volumes currently in effect (shown above).

We are prioritizing the GLOBAL optimum together with any secondary local optima
that are ALSO very high-valued (near the global optimum in `Objective`) but sit on
a SEPARATE peak — i.e. genuinely distinct high-value basins, not extra points piled
onto the same peak as the global optimum. Use `reward` volumes to steer toward
promising, under-sampled basins and `penalty` volumes to push the run off a
picked-over peak (or a physically poor region — high `Objective` but bad
stability/bandgap) so it can find those distinct secondary needles rather than
re-measuring one basin. Then answer through the schema:

- `reasoning`: a concise, quantitative justification (2-6 sentences) grounded in the
  specific numbers above (name the compositions, radii, and strengths you chose and why).
- `volumes`: the FULL list of {{"kind", "center", "radius", "strength"}} entries to be
  active now (not just newly added ones). `kind` is `"reward"` or `"penalty"`, `center`
  is a 3-number composition, `radius` is within [{min_radius}, {max_radius}], and
  `strength` is within [{min_strength}, {max_strength}] (1 == the force of one of
  ZoMBI-Hop's own needle penalties). To keep the current landscape unchanged, re-list
  exactly the volumes shown above. Return an EMPTY list to clear all volumes (a valid,
  deliberate choice — e.g. to hand full control back to ZoMBI-Hop's own acquisition).

Reason qualitatively about the trajectory's direction and the feature trade-offs
rather than over-fitting to individual numeric rows; prefer a small, well-justified
set of volumes over many, and leave the landscape unchanged when the trend is
already healthy.
"""


def build_injection_prompt(feature_log, dh, volumes, injection_idx, iters_done,
                           budget, interval, prev_best: Optional[float] = None) -> str:
    """RF-specific volume-placement prompt: same structure as
    ``VC.build_injection_prompt`` but with the trimmed feature apparatus (RF
    system-features table + compact GLOBAL stats line) and the global/distinct-optima
    priority in the task section."""
    return VOLUME_RF_PROMPT_TEMPLATE.format(
        system_features=RFB.RF_SYSTEM_FEATURES,
        active_volumes=VC.format_active_volumes(volumes),
        injection_idx=injection_idx,
        iters_done=iters_done,
        budget=budget,
        progress_summary=SBS.progress_summary(feature_log, dh, iters_done, budget),
        trend_summary=SBS.trend_summary(feature_log, prev_best),
        history_table=SBS.recent_history_table(feature_log),
        n_points=len(feature_log),
        supplemental_summary=RFB.rf_supplemental_summary(feature_log),
        best_point_summary=SBS.best_point_summary(feature_log),
        top_k=SBS.TOP_K_DROPLETS,
        top_k_summary=SBS.top_k_summary(feature_log, SBS.TOP_K_DROPLETS),
        needle_summary=SBS.needle_summary(feature_log, dh),
        interval=interval,
        min_radius=VC.MIN_VOLUME_RADIUS,
        max_radius=VC.MAX_VOLUME_RADIUS,
        min_strength=VC.MIN_VOLUME_STRENGTH,
        max_strength=VC.MAX_VOLUME_STRENGTH,
    )


def install_rf_config() -> None:
    """Apply the RF config (noise-free objective, three material features only, no
    curves, noise-free needle plots) AND route ``VC.run_llm_trial`` — which looks up
    ``build_injection_prompt`` in its own module namespace — to the RF volume prompt.
    Idempotent; scoped to this process."""
    RFB.install_rf_config()
    VC.build_injection_prompt = build_injection_prompt


# ════════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════════

_VOL_RF_PLOT_TITLE = ("Volume control (noise-free RF landscape, material features "
                      "only) — convergence: baseline vs LLM injection cadences\n"
                      "(mean ± 95% CI over repeats)")


def main(resume_dir=None) -> None:
    install_rf_config()
    VC.main(resume_dir=resume_dir, sweep_prefix="sweep_volume_rf",
            baseline_impl="volume_rf", plot_title=_VOL_RF_PLOT_TITLE)


if __name__ == "__main__":
    install_rf_config()
    args = sys.argv[1:]
    if args and args[0] in ("--regenerate", "-r"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_volume_RF.py --regenerate <sweep_dir>")
        SBS.regenerate_summary(Path(args[1]), group_row=VC._group_row,
                               write=VC.write_summary)
    elif args and args[0] in ("--plot", "-p"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_volume_RF.py --plot <sweep_dir>")
        SBS.plot_convergence_comparison(Path(args[1]), title=_VOL_RF_PLOT_TITLE)
        SBS.plot_per_rep_vs_baseline(Path(args[1]))
    elif args and args[0] in ("--needle-plots",):
        if len(args) < 2:
            raise SystemExit("usage: sweep_volume_RF.py --needle-plots "
                             "<sweep_dir> [group ...]")
        SBS.regenerate_needle_plots(Path(args[1]), groups=(args[2:] or None))
    elif args and args[0] in ("--resume",):
        main(resume_dir=SBS.resolve_resume_dir(args[1] if len(args) > 1 else None,
                                               "sweep_volume_rf"))
    else:
        main()
