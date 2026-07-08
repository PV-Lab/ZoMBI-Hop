"""
llm/sweep_basic_RF.py
=====================
RF-landscape twin of ``sweep_basic_surrogate.py``.

Same experiment shape — cold-start → inject-every-k → resume-exact-state
continuation, same common-random-numbers baseline (``trial_169`` hyperparameters,
NO LLM), same metrics / summary / convergence plots, same hyperparameter-tuning
task — but stripped down in two deliberate ways so the LLM sees a *clean,
noise-free* problem with only the three interpretable material properties:

1.  **Noise-free RF objective landscape.**  ``sweep_basic_surrogate.py`` measures
    the generative surrogate: each droplet is ``E[Objective | comp]`` PLUS a joint
    residual draw (grainy, correlated noise).  Here the objective is the surrogate's
    *deterministic* RF conditional mean only — ``surr._cond_means`` with NO residual
    added — i.e. exactly the RF reconstruction of the Objective landscape from the
    campaign points that ``visualization/plot_run.py`` / ``_ObjMeanPredictor`` draw
    (see ``llm/surrogate.py``).  So every rep measures the same smooth landscape;
    trajectories differ only through their Sobol init (seed) and the LLM's edits.

2.  **Only Bandgap / Photoconductance / Stability as supplemental features.**  None
    of the environment channels (temperature, pressure, humidity, DMF), degradation
    kinetics, or measured CURVES (absorption / PL spectra, stability voltage sweep)
    are surfaced.  The three retained scalars are likewise read from the RF
    conditional mean (noise-free).  The injection prompt therefore drops the big
    "supplemental measured features" apparatus, keeping only a compact GLOBAL stats
    line (mean, [min, max], corr-with-Objective) for Bandgap, Photoconductance,
    Stability, and the Objective itself.

Everything else is inherited from ``sweep_basic_surrogate`` (``SBS``): the trial
loop, ZoMBI-Hop segment plumbing, baseline reuse cache, aggregation, and plots.
This module just (a) flips ``SBS.SUP_SCALARS`` / ``SBS.CURVE_MODE``, (b) swaps in a
deterministic objective and a noise-free needle-plot background, (c) supplies an
RF-specific injection prompt, and (d) points ``SBS.main`` at a distinct results
directory with its own baseline-reuse tag (``impl="rf"``) so an RF baseline is
never confused with a noisy-surrogate baseline of the same seed.

Usage:
  # repo-root uv venv (see MEMORY.md), NOT `conda activate zombi-hop`
  python llm/sweep_basic_RF.py
  python llm/sweep_basic_RF.py --plot <sweep_dir>
  python llm/sweep_basic_RF.py --regenerate <sweep_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sweep_basic_surrogate as SBS  # noqa: E402  (reuse all machinery)

# The only supplemental features the RF sweep surfaces (no environment, kinetics,
# or curve groups). Read from the surrogate's noise-free RF conditional mean.
RF_SUP_SCALARS: List[str] = ["Bandgap", "Photoconductance", "Stability"]

# A compact stand-in for llm_config.SYSTEM_FEATURES: only the three retained
# scalars plus the Objective, so the prompt does not advertise features the LLM
# never sees.
RF_SYSTEM_FEATURES = """\
| Feature | Range | What it is |
|---|---|---|
| `Bandgap` | 1.45–2.59 eV | Optical bandgap of the deposited perovskite film |
| `Photoconductance` | 0–1 | Light-induced conductance (charge generation / transport) |
| `Stability` | 0.32–1.0 | Retained performance after a degradation study |
| `Objective` | 0.28–0.86 | Combined figure of merit being MAXIMIZED |"""


# ════════════════════════════════════════════════════════════════════════════════
# Noise-free RF objective + supplemental scalars
# ════════════════════════════════════════════════════════════════════════════════

def rf_make_objective(surr, rng):
    """Drop-in replacement for ``SBS.make_surrogate_objective`` that measures the
    surrogate's DETERMINISTIC RF conditional mean — no residual draw, so the
    landscape is exactly the noise-free RF reconstruction of the Objective.

    Records only Bandgap / Photoconductance / Stability (also noise-free means) per
    droplet. ``rng`` is accepted for signature compatibility with the callers in
    ``SBS`` but is unused (there is nothing stochastic left to draw)."""
    oi = surr.obj_index
    t, it = surr.time_ref["max"], surr.iter_ref["max"]
    record_names = list(RF_SUP_SCALARS)
    name_idx = {nm: surr.feat_names.index(nm) for nm in record_names}
    feature_log: List[Dict[str, Any]] = []

    def fn(x) -> float:
        comp = np.asarray(x, float)
        s = comp.sum()
        comp = comp / (s if s != 0 else 1.0)
        Z = np.array([[comp[0], comp[2], t, it]], float)
        row = surr._cond_means(Z).ravel()          # RF mean only — NO residual noise
        rec: Dict[str, Any] = {"FAPbI3": comp[0], "MAPbI3": comp[1], "MAPbBr3": comp[2],
                               "Objective": float(row[oi])}
        for nm in record_names:
            rec[nm] = float(row[name_idx[nm]])
        feature_log.append(rec)
        return float(row[oi])

    return fn, feature_log


def _rf_landscape_noiseless(surr, seed: int = 0):
    """Drop-in replacement for ``SBS._surrogate_landscape_noisy``: the RF objective
    here is noise-free, so the needle-plot background is the deterministic mean
    landscape (ignore the ``seed``) instead of a grainy per-point realization."""
    return SBS._surrogate_landscape(surr)


def rf_supplemental_summary(feature_log: List[Dict[str, Any]]) -> str:
    """GLOBAL stats line for the Objective and the three retained scalars: global
    mean, [global min, global max], and global corr(feature, Objective). Replaces
    ``SBS.supplemental_summary`` (which also renders curve blocks / more columns)."""
    if not feature_log:
        return "(no measured points yet)"
    obj = np.array([r["Objective"] for r in feature_log], float)
    lines = ["feature | global mean [global min, global max] | global corr(Objective)"]
    lines.append(f"Objective | {obj.mean():.3g} [{obj.min():.3g}, {obj.max():.3g}] | —")
    for nm in RF_SUP_SCALARS:
        col = np.array([r.get(nm, np.nan) for r in feature_log], float)
        v = col[np.isfinite(col)]
        if v.size == 0:
            continue
        corr = SBS._pearson(col, obj)
        corr_s = "n/a" if not np.isfinite(corr) else f"{corr:+.2f}"
        lines.append(f"{nm} | {v.mean():.3g} [{v.min():.3g}, {v.max():.3g}] | {corr_s}")
    return "\n".join(lines)


def install_rf_config() -> None:
    """Reconfigure the shared ``SBS`` machinery for the RF sweep: show only the three
    material scalars (no curves), measure the noise-free RF objective, and draw a
    noise-free needle-plot background. Idempotent — safe to call from every entry
    point (each script runs in its own process, so this never leaks to other sweeps)."""
    SBS.SUP_SCALARS = list(RF_SUP_SCALARS)
    SBS.CURVE_MODE = "none"                     # neither "full" nor "condensed" → no curves
    SBS.make_surrogate_objective = rf_make_objective
    SBS._surrogate_landscape_noisy = _rf_landscape_noiseless


# ════════════════════════════════════════════════════════════════════════════════
# Injection prompt (RF-specific: trimmed feature apparatus + optima-priority task)
# ════════════════════════════════════════════════════════════════════════════════

RF_PROMPT_TEMPLATE = """\
You are helping tune a Bayesian-optimization algorithm, ZoMBI-Hop, on the fly. It
is optimizing a perovskite-materials-discovery lab over the 3-simplex composition
(`FAPbI3`, `MAPbI3`, `MAPbBr3`, summing to 1); the `Objective` is MAXIMIZED.
ZoMBI-Hop is a zooming multi-basin optimizer that hunts MULTIPLE optima
("needles"): it fits a GP, uses an acquisition function each iteration to pick a
LineBO line of 24 measured droplets, declares a needle when expected improvement
hits the output-noise floor, penalizes that region, and moves on.

Alongside each measured droplet's `Objective`, the lab reports three interpretable
material properties measured behind it:

{system_features}

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
corner of the simplex indicate over-exploitation of a single region.

{needle_summary}

## Your task

Decide whether to change any ZoMBI-Hop hyperparameters for the NEXT {interval}
iteration(s). ZoMBI-Hop will continue from its exact current internal state (same
data, needles, zoom bounds) with whatever hyperparameters you specify; anything
you don't specify keeps its current value.

We are prioritizing the GLOBAL optimum together with any secondary local optima
that are ALSO very high-valued (near the global optimum in `Objective`) but sit on
a SEPARATE peak — i.e. genuinely distinct high-value basins, not extra points
piled onto the same peak as the global optimum. Favor hyperparameter choices that
help ZoMBI-Hop both climb the global peak and discover these distinct secondary
needles, rather than repeatedly re-sampling one basin or spreading needles densely
across a single peak.

Weigh the material features: do they suggest the run is chasing a region that
looks good on `Objective` but poor on stability/bandgap, or that it is
over-exploring (many duplicates) or over-exploiting (collapsing onto one needle)?
Then answer through the schema:

- `reasoning`: a concise, quantitative justification (2-6 sentences) grounded in
  the specific numbers above.
- `hyperparameter_changes`: a list of {{"name", "value"}} entries — ONLY the
  hyperparameters you want to change, each within its allowed range. Leave the
  list EMPTY if the current settings are already appropriate.

Reason qualitatively about the trajectory's direction and the feature trade-offs
rather than over-fitting to individual numeric rows; prefer a single well-justified
change over many, and keep the current settings when the trend is already healthy.
"""


def build_injection_prompt(feature_log, dh, hp, injection_idx, iters_done, budget,
                           interval, prev_best: Optional[float] = None) -> str:
    """RF-specific injection prompt: same structure as ``SBS.build_injection_prompt``
    but with the trimmed feature apparatus (RF system-features table + compact GLOBAL
    stats line) and the global/distinct-optima priority in the task section."""
    import evaluate_llm as E  # noqa: E402  (same heavy imports SBS already pulled in)
    return RF_PROMPT_TEMPLATE.format(
        system_features=RF_SYSTEM_FEATURES,
        hparam_descriptions=E.hparam_descriptions_block(),
        current_hparams=E.format_current_hparams(hp),
        hparam_search_history=E.hparam_optimization_history(top_k=SBS.MOBO_HISTORY_TOP_K),
        injection_idx=injection_idx,
        iters_done=iters_done,
        budget=budget,
        progress_summary=SBS.progress_summary(feature_log, dh, iters_done, budget),
        trend_summary=SBS.trend_summary(feature_log, prev_best),
        history_table=SBS.recent_history_table(feature_log),
        n_points=len(feature_log),
        supplemental_summary=rf_supplemental_summary(feature_log),
        best_point_summary=SBS.best_point_summary(feature_log),
        top_k=SBS.TOP_K_DROPLETS,
        top_k_summary=SBS.top_k_summary(feature_log, SBS.TOP_K_DROPLETS),
        needle_summary=SBS.needle_summary(feature_log, dh),
        interval=interval,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════════

_RF_PLOT_TITLE = ("Convergence (noise-free RF landscape, material features only): "
                  "baseline vs LLM injection cadences\n(mean ± 95% CI over repeats)")


def main(resume_dir=None) -> None:
    install_rf_config()
    SBS.main(sweep_prefix="sweep_basic_rf", prompt_builder=build_injection_prompt,
             plot_title=_RF_PLOT_TITLE, resume_dir=resume_dir, baseline_impl="rf")


if __name__ == "__main__":
    install_rf_config()
    args = sys.argv[1:]
    if args and args[0] in ("--regenerate", "-r"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_RF.py --regenerate <sweep_dir>")
        SBS.regenerate_summary(Path(args[1]))
    elif args and args[0] in ("--plot", "-p"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_RF.py --plot <sweep_dir>")
        SBS.plot_convergence_comparison(Path(args[1]))
        SBS.plot_per_rep_vs_baseline(Path(args[1]))
    elif args and args[0] in ("--needle-plots",):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_RF.py --needle-plots "
                             "<sweep_dir> [group ...]")
        SBS.regenerate_needle_plots(Path(args[1]), groups=(args[2:] or None))
    elif args and args[0] in ("--resume",):
        main(resume_dir=SBS.resolve_resume_dir(args[1] if len(args) > 1 else None,
                                               "sweep_basic_rf"))
    else:
        main()
