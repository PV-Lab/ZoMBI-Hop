"""
llm/sweep_basic_surrogate_no_features.py
========================================
Feature-ablation twin of ``sweep_basic_surrogate.py``.

Identical in every respect — same generative surrogate, same trial_112 baseline,
same cold-start → inject-every-k → resume-exact-state continuation, same
common-random-numbers, same metrics / summary / convergence plot — EXCEPT the LLM
injection prompt WITHHOLDS the surrogate's supplemental measured features. The LLM
sees only what a vanilla ZoMBI-Hop run exposes:

    * the composition (FAPbI3, MAPbI3, MAPbBr3),
    * the BO iteration index and remaining budget,
    * the `Objective`,

plus the same *static* context the featured sweep already gave it (the tunable
hyperparameters and their ranges, the current values, and the offline MOBO
optimization history). It never sees Bandgap, Photoconductance, Stability, the
environment channels, the degradation kinetics, or the spectra fPCA scores.

Purpose: an A/B test of whether surfacing the rich per-droplet features actually
helped the LLM tune ZoMBI-Hop. Run this alongside ``sweep_basic_surrogate.py`` and
compare the two convergence plots / summaries. Because the surrogate objective draws
are seeded identically (common random numbers) and every non-prompt code path is
shared, the ONLY difference between the two experiments is the information in the
prompt — so any difference in outcome is attributable to the features.

Only the injection prompt is overridden here (the feature sections are removed);
everything else is imported from ``sweep_basic_surrogate`` and ``main()`` just points
at a distinct results directory.

Usage:
  # repo-root uv venv (see MEMORY.md), NOT `conda activate zombi-hop`
  python llm/sweep_basic_surrogate_no_features.py
  python llm/sweep_basic_surrogate_no_features.py --plot <sweep_dir>
  python llm/sweep_basic_surrogate_no_features.py --regenerate <sweep_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sweep_basic_surrogate as SBS  # noqa: E402  (reuse all machinery)


# ── ablated injection prompt (composition + iteration + Objective only) ──────────
# SBS.SURR_PROMPT_TEMPLATE with the "rich physical features" framing, the
# {system_features} block, the supplemental-feature table, and the best-droplet
# feature line all removed; the task wording no longer refers to stability/bandgap.
NO_FEATURES_PROMPT_TEMPLATE = """\
You are helping tune a Bayesian-optimization algorithm, ZoMBI-Hop, on the fly. It
is optimizing a perovskite-materials-discovery lab over the 3-simplex composition
(`FAPbI3`, `MAPbI3`, `MAPbBr3`, summing to 1); the `Objective` is MAXIMIZED.
ZoMBI-Hop is a zooming multi-basin optimizer that hunts MULTIPLE optima
("needles"): it fits a GP, uses an acquisition function each iteration to pick a
LineBO line of ~24 measured droplets, declares a needle when expected improvement
hits the output-noise floor, penalizes that region, and moves on.

## The ZoMBI-Hop hyperparameters you may tune

(ranges are the allowed bounds; stay inside them)
{hparam_descriptions}

## Current ZoMBI-Hop hyperparameters (in effect right now)

{current_hparams}

## Offline hyperparameter-optimization history

The current values were chosen by a long offline multi-objective Bayesian
optimization run. Each row is one trial: its hyperparameter values followed by
three MINIMIZED objectives — `dist_to_needles` (how well it located the true
needles), `dup_fraction` (fraction of wasted duplicate points), and `runtime_s`.
Use it as a strong prior on which hyperparameter regions are good.

{hparam_search_history}

## This run so far

This is injection #{injection_idx}. ZoMBI-Hop has completed {iters_done} of
{budget} iterations for this run. Progress:

{progress_summary}

Recent measured points (composition → Objective):

{history_table}

## Your task

Decide whether to change any ZoMBI-Hop hyperparameters for the NEXT {interval}
iteration(s). ZoMBI-Hop will continue from its exact current internal state (same
data, needles, zoom bounds) with whatever hyperparameters you specify; anything you
don't specify keeps its current value. Consider the trajectory so far: is the run
over-exploring (many duplicate points) or over-exploiting (collapsing onto a single
needle), and is it still finding new needles? Then answer through the schema:

- `reasoning`: a concise, quantitative justification (2-6 sentences) grounded in
  the specific numbers above.
- `hyperparameter_changes`: a list of {{"name", "value"}} entries — ONLY the
  hyperparameters you want to change, each within its allowed range. Leave the
  list EMPTY if the current settings are already appropriate.
"""


def build_injection_prompt(feature_log, dh, hp, injection_idx, iters_done, budget,
                           interval) -> str:
    """SBS.build_injection_prompt minus every supplemental-feature section — the LLM
    sees only composition, iteration/budget, and Objective (plus the static hparam +
    MOBO context). The signature matches SBS.build_injection_prompt so this drops
    straight into SBS.run_llm_trial as its ``prompt_builder``. ``feature_log`` still
    carries the sampled features, but only its composition/Objective fields are used
    (via progress_summary / recent_history_table)."""
    return NO_FEATURES_PROMPT_TEMPLATE.format(
        hparam_descriptions=SBS.E.hparam_descriptions_block(),
        current_hparams=SBS.E.format_current_hparams(hp),
        hparam_search_history=SBS.E.hparam_optimization_history(),
        injection_idx=injection_idx,
        iters_done=iters_done,
        budget=budget,
        progress_summary=SBS.progress_summary(feature_log, dh, iters_done, budget),
        history_table=SBS.recent_history_table(feature_log),
        interval=interval,
    )


def main() -> None:
    SBS.main(sweep_prefix="sweep_surrogate_no_features",
             prompt_builder=build_injection_prompt,
             plot_title=("Convergence (no supplemental features): baseline vs LLM "
                         "injection cadences\n(mean ± 95% CI over repeats)"))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] in ("--regenerate", "-r"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_surrogate_no_features.py "
                             "--regenerate <sweep_dir>")
        SBS.regenerate_summary(Path(args[1]))
    elif args and args[0] in ("--plot", "-p"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_surrogate_no_features.py "
                             "--plot <sweep_dir>")
        SBS.plot_convergence_comparison(Path(args[1]))
    else:
        main()
