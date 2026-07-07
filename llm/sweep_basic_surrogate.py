"""
llm/sweep_basic_surrogate.py
============================
Surrogate-in-the-loop version of ``sweep_basic_no_surrogate.py``.

Instead of a single LLM call at one injection point on the RF *Objective-only*
surrogate, this experiment runs the LLM **repeatedly during a full ZoMBI-Hop run
on the generative surrogate** (``llm/surrogate.py``). Because that surrogate
emits a whole synthetic measurement per droplet — not just the ``Objective`` but
the interpretable supplemental scalars (Bandgap, Photoconductance, Stability, the
environment channels, and the degradation kinetics) — the LLM can be shown those
extra features at every injection and re-tune the hyperparameters on the fly.

Experiment design
-----------------
Three *injection cadences* are swept: the LLM is prompted to inject a
hyperparameter change **every 1, every 5, or every 10** ZoMBI-Hop iterations.
For a given cadence ``k`` one *trial* is:

    fresh ZoMBI-Hop cold-start on the surrogate (trial_112 hyperparameters)
      → run k iterations
      → show the LLM the run so far *including the surrogate's supplemental
        features* and let it change hyperparameters
      → resume ZoMBI-Hop's EXACT state (a genuine continuation) with the new
        hyperparameters, run k more iterations
      → inject again … until the fixed iteration budget is spent.

The continuation is real: injection #2 continues from the state produced *after*
injection #1 (checkpoint/resume of the same run), so the hyperparameter edits
compound the way they would in a live campaign.

Each trial is repeated ``N_REPEATS`` times *from the start* (independent surrogate
noise draws) to measure variance.

Baseline
--------
The baseline is the ``trial_112`` (run_7eb9 / offline-MOBO) hyperparameters run on
the *same* surrogate for the *same* iteration budget, ``N_REPEATS`` times, with NO
LLM in the loop — ZoMBI-Hop sees only the compositions and the ``Objective`` (the
supplemental features are never surfaced). We deliberately do NOT use the real
campaign trajectory (the data the surrogate was fit on) as a baseline sample.

Common random numbers: repeat ``r`` of every group (baseline and each cadence)
shares the same seed, so they face the same initial design and surrogate-noise
stream — trajectories diverge only because of the LLM's hyperparameter edits.

Output layout (each trial type is distinguishable by its directory title)
------------------------------------------------------------------------
    results/sweep_surrogate_<ts>/
        baseline/rep0 … rep{N-1}/
        inject_every_01/rep0 … rep{N-1}/injections/inj_00 …
        inject_every_05/rep0 … rep{N-1}/…
        inject_every_10/rep0 … rep{N-1}/…
        sweep_summary.csv / .json   (one row per group)

Usage:
  # repo-root uv venv (see MEMORY.md), NOT `conda activate zombi-hop`
  python llm/sweep_basic_surrogate.py

The model / prompt come from llm/llm_config.py (shared with evaluate_llm).
"""

from __future__ import annotations

# ─── HARDCODED CONFIG ──────────────────────────────────────────────────────────
INJECTION_INTERVALS: list[int] = [5, 10]   # LLM injects every k iterations
MAX_ITERS: int = 40                            # total ZoMBI-Hop iterations per trial
N_REPEATS: int = 5                             # trials per group (variance)
# Cost note: cadence k does ~ceil(MAX_ITERS/k)-1 LLM calls per repeat, so with the
# defaults k=1 ≈ 19, k=5 ≈ 3, k=10 ≈ 1 calls/repeat → ~115 LLM calls total across
# the 5 repeats of the three cadences (the baseline calls the LLM zero times).
SURROGATE_PICKLE: str | None = None            # reuse a fitted surrogate if set
# Baseline hyperparameters: point this at a MOBO ``trial_*`` directory (with a
# trial.json holding a "hparams" block) to pin the baseline/starting hyperparameters
# to that trial. A relative path is resolved against the repo root. Set to None to
# instead pick the best-dist_to_needles ensemble trial (falling back to trial_112 /
# run_7eb9).
BASELINE_TRIAL_DIR: str | None = "optimize/runs/mobo_ensemble_3d_job17147229/trial_169"
# ───────────────────────────────────────────────────────────────────────────────

import csv
import datetime
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# evaluate_llm sets up sys.path for optimize/ + repo root and does the heavy
# botorch/run_mobo import; reuse all of its data + ZoMBI plumbing.
import evaluate_llm as E  # noqa: E402
import torch  # noqa: E402
import run_mobo as R  # noqa: E402
import llm_config  # noqa: E402
import sweep_basic_no_surrogate as SW  # noqa: E402  (reuse welch_significance)
from eval_metrics import (  # noqa: E402
    as_numpy,
    metric_avg_pairwise_dist,
    metric_dist_to_needles,
    metric_dup_fraction,
)
from src.core.zombihop import ZoMBIHop  # noqa: E402

from surrogate import Surrogate, SUBTARGETS, ENV, KIN  # noqa: E402

# Interpretable supplemental scalars shown to the LLM. The measured CURVE groups
# (absorption spectrum, dark/light stability voltage sweep, initial/final PL spectra)
# are surfaced separately — see CURVE_MODE below — not lumped in here.
SUP_SCALARS: List[str] = list(SUBTARGETS) + list(ENV) + list(KIN)

# ── Curve (functional) feature rendering ────────────────────────────────────────
# The generative surrogate compresses each measured curve group into a handful of
# functional-PCA scores (see llm/surrogate.py); it never stores the raw curves.
# There are two ways to surface those curves to the LLM, chosen by CURVE_MODE:
#   "full"      — reconstruct the COMPLETE curve from its fPCA scores (via the
#                 surrogate's FunctionalReconstructor) and print it on its native
#                 grid (wavelength / voltage). No PCA numbers shown; token-heavy.
#   "condensed" — show the fPCA scores themselves as extra scalar table columns (the
#                 compact PCA representation). Used by sweep_basic_surrogate_condensed.
# sweep_volume_control reuses these renderers, so it inherits whatever CURVE_MODE is
# set process-wide (default "full"). The no-features ablation withholds all of it.
CURVE_MODE: str = "full"

# curve group key (surrogate reconstructor key) -> (human label, grid unit).
_CURVE_GROUP_LABELS: Dict[str, Tuple[str, str]] = {
    "absorption":      ("absorption spectrum",     "nm"),
    "stability_dark":  ("stability sweep (dark)",  "V"),
    "stability_light": ("stability sweep (light)", "V"),
    "PL_initial":      ("PL spectrum (initial)",   "nm"),
    "PL_final":        ("PL spectrum (final)",     "nm"),
}

# Populated from the shared surrogate on the first make_surrogate_objective() call.
_CURVE_META: Optional[List[Dict[str, Any]]] = None   # per-group reconstruction info
_FPCA_SCORE_NAMES: List[str] = []                     # flat list of fPCA score names


def _init_curve_meta(surr) -> None:
    """Cache each curve group's reconstruction metadata (fPCA score names, native
    grid, reconstructor) and the flat list of fPCA score feature names, both derived
    from the fitted surrogate. Idempotent; the surrogate is shared across all trials."""
    global _CURVE_META, _FPCA_SCORE_NAMES
    if _CURVE_META is not None:
        return
    meta: List[Dict[str, Any]] = []
    for key, recon in surr.reconstructors.items():
        label, unit = _CURVE_GROUP_LABELS.get(key, (key, ""))
        meta.append({"key": key, "label": label, "unit": unit,
                     "score_names": list(recon.score_names),
                     "grid": np.asarray(recon.grid, float), "recon": recon})
    _CURVE_META = meta
    _FPCA_SCORE_NAMES = [nm for m in meta for nm in m["score_names"]]


def _fpca_score_names() -> List[str]:
    return list(_FPCA_SCORE_NAMES)


def _table_scalar_names() -> List[str]:
    """Feature names rendered as scalar table columns/rows. In ``condensed`` mode the
    compact fPCA curve scores ride along as extra scalar columns; in ``full`` mode the
    curves are shown as reconstructed-curve blocks instead, so only the physical
    scalars appear in the scalar tables."""
    if CURVE_MODE == "condensed":
        return list(SUP_SCALARS) + _fpca_score_names()
    return list(SUP_SCALARS)


MAXIMIZE = True
N_REF_OPTIMA = 3
SIG_ALPHA = 0.05
TOP_K_DROPLETS = 16  # how many top-Objective droplets to show with features
# How many offline MOBO hyperparameter-search trials to surface in the injection
# prompt: the top-k Pareto-optimal trials by dist_to_needles (see
# evaluate_llm.hparam_optimization_history). Keeps the prompt from carrying a long
# dense numeric table — LLMs reason better over a short curated set (arXiv 2404.06290).
MOBO_HISTORY_TOP_K = 16

# Split the tunable hyperparameters the same way evaluate_llm does: config-read
# ones are applied by rewriting the resumed run's config.json, the rest are passed
# to the ZoMBIHop constructor on each resume.
_CONFIG_READ = E._CONFIG_READ_HPARAMS
_CONSTRUCTOR = E._CONSTRUCTOR_HPARAMS

_METRIC_KEYS = ["best_objective", "best_needle", "n_needles",
                "dist_to_ref_optima", "dup_fraction", "mean_pairwise_needle_dist"]


# ════════════════════════════════════════════════════════════════════════════════
# Surrogate objective + landscape
# ════════════════════════════════════════════════════════════════════════════════

class _ObjMeanPredictor:
    """``.predict(X)`` → deterministic surrogate-mean ``Objective`` at compositions
    ``X`` (m×3), so run_mobo.auto_detect_rf_optima can pick the true optima the
    same way it does for an RF."""

    def __init__(self, surr: Surrogate):
        self.surr = surr
        self.oi = surr.obj_index
        self.t = surr.time_ref["max"]
        self.it = surr.iter_ref["max"]

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        Z = np.column_stack([X[:, 0], X[:, 2],
                             np.full(len(X), self.t), np.full(len(X), self.it)])
        return self.surr._cond_means(Z)[:, self.oi]


def make_surrogate_objective(surr: Surrogate, rng: np.random.Generator):
    """Build ``(fn_callable, feature_log)``.

    ``fn_callable(x)`` samples one full synthetic row at composition ``x`` from the
    generative surrogate, returns its ``Objective`` (for ZoMBI-Hop), and appends
    the composition + interpretable supplemental scalars to ``feature_log`` so the
    injection prompt can report them. Draws come from ``rng`` (persistent across the
    whole trial → a reproducible but evolving noise stream)."""
    _init_curve_meta(surr)
    oi = surr.obj_index
    t, it = surr.time_ref["max"], surr.iter_ref["max"]
    # Record the physical scalars AND the fPCA curve scores per droplet: the scores
    # are what the "full" renderer reconstructs the curves from and what the
    # "condensed" renderer prints directly.
    record_names = list(SUP_SCALARS) + _fpca_score_names()
    name_idx = {nm: surr.feat_names.index(nm) for nm in ([surr.feat_names[oi]] + record_names)}
    feature_log: List[Dict[str, Any]] = []

    def fn(x) -> float:
        comp = np.asarray(x, float)
        s = comp.sum()
        comp = comp / (s if s != 0 else 1.0)
        Z = np.array([[comp[0], comp[2], t, it]], float)
        row = (surr._cond_means(Z) + surr._draw_resid(1, rng)).ravel()
        rec: Dict[str, Any] = {"FAPbI3": comp[0], "MAPbI3": comp[1], "MAPbBr3": comp[2],
                               "Objective": float(row[oi])}
        for nm in record_names:
            rec[nm] = float(row[name_idx[nm]])
        feature_log.append(rec)
        return float(row[oi])

    return fn, feature_log


# ════════════════════════════════════════════════════════════════════════════════
# Injection prompt (surrogate-specific: adds the supplemental-feature section)
# ════════════════════════════════════════════════════════════════════════════════

SURR_PROMPT_TEMPLATE = """\
You are helping tune a Bayesian-optimization algorithm, ZoMBI-Hop, on the fly. It
is optimizing a perovskite-materials-discovery lab over the 3-simplex composition
(`FAPbI3`, `MAPbI3`, `MAPbBr3`, summing to 1); the `Objective` is MAXIMIZED.
ZoMBI-Hop is a zooming multi-basin optimizer that hunts MULTIPLE optima
("needles"): it fits a GP, uses an acquisition function each iteration to pick a
LineBO line of 24 measured droplets, declares a needle when expected improvement
hits the output-noise floor, penalizes that region, and moves on.

Unlike a normal run, every measured droplet here also reports the rich physical
features the lab records behind each `Objective` value. The feature groups are:

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
with its composition, `Objective`, and supplemental features:

{history_table}

## Supplemental measured features so far (GLOBAL, all {n_points} droplets)

Beyond the composition and `Objective`, these per-droplet supplemental features
were measured across all {n_points} droplets so far. Every number in the scalar
table below is a GLOBAL statistic over ALL {n_points} droplets: the global mean, the global
[min, max] range, and the global Pearson correlation of the feature with
`Objective` (a positive corr means the feature tends to rise where the objective
is high). Use these as the population baseline to compare the top-k and needle
rows below against — e.g. if a feature's value in the top-k droplets sits near a
tail of its global [min, max], that basin is atypical on that feature.

{supplemental_summary}

Best droplet so far — its supplemental features:

{best_point_summary}

## Top {top_k} droplets by Objective (with features)

The {top_k} highest-`Objective` droplets measured so far and their supplemental
features. Compare these against the GLOBAL table above: a feature that is high in
the objective but sits at an unfavourable tail of its global range here signals a
trade-off the run may be walking into.

{top_k_summary}

## Declared needles (local optima) with features

Each needle ZoMBI-Hop has declared, its composition and declared `Objective`
value, plus the supplemental features of the nearest measured droplet. Spread-out
needle compositions indicate healthy multi-basin coverage; needles clustered in
one corner of the simplex indicate over-exploitation of a single region.

{needle_summary}

## Your task

Decide whether to change any ZoMBI-Hop hyperparameters for the NEXT {interval}
iteration(s). ZoMBI-Hop will continue from its exact current internal state (same
data, needles, zoom bounds) with whatever hyperparameters you specify; anything
you don't specify keeps its current value. Weigh the supplemental features: do
they suggest the run is chasing a region that looks good on `Objective` but poor
on stability/bandgap, or that it is over-exploring (many duplicates) or
over-exploiting (collapsing onto one needle)? Then answer through the schema:

- `reasoning`: a concise, quantitative justification (2-6 sentences) grounded in
  the specific numbers above.
- `hyperparameter_changes`: a list of {{"name", "value"}} entries — ONLY the
  hyperparameters you want to change, each within its allowed range. Leave the
  list EMPTY if the current settings are already appropriate.

Reason qualitatively about the trajectory's direction and the feature trade-offs
rather than over-fitting to individual numeric rows; prefer a single well-justified
change over many, and keep the current settings when the trend is already healthy.
"""


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ── reconstructed-curve rendering (CURVE_MODE == "full") ─────────────────────────

def _fmt_grid(g: float) -> str:
    """Compact grid label: integer wavelengths/voltages print without a decimal."""
    return f"{int(round(g))}" if abs(g - round(g)) < 1e-6 else f"{g:.4g}"


def _format_curve_line(grid: np.ndarray, values: np.ndarray, unit: str) -> str:
    """One curve as ``x1unit=y1, x2unit=y2, …`` over its whole native grid."""
    return ", ".join(f"{_fmt_grid(g)}{unit}={v:.3g}" for g, v in zip(grid, values))


def _reconstruct_record_curve(rec: Dict[str, Any], m: Dict[str, Any]) -> Optional[np.ndarray]:
    """Full curve for one droplet, reconstructed from its stored fPCA scores. None if
    that curve group was not measured for this droplet (scores NaN)."""
    scores = np.array([rec.get(nm, np.nan) for nm in m["score_names"]], float)
    if not np.all(np.isfinite(scores)):
        return None
    return np.asarray(m["recon"].reconstruct(scores), float)


def _droplet_curve_block(rec: Dict[str, Any], indent: str = "  ") -> str:
    """Every measured curve for one droplet, each reconstructed onto its native grid.
    Empty string unless CURVE_MODE == 'full' (condensed mode shows fPCA scores in the
    scalar tables instead)."""
    if CURVE_MODE != "full" or not _CURVE_META:
        return ""
    lines = []
    for m in _CURVE_META:
        vals = _reconstruct_record_curve(rec, m)
        if vals is None:
            continue
        lines.append(f"{indent}{m['label']} ({len(m['grid'])} pts): "
                     + _format_curve_line(m["grid"], vals, m["unit"]))
    return "\n".join(lines)


def _global_curve_block(feature_log: List[Dict[str, Any]]) -> str:
    """Population mean curve per group over all droplets. Reconstruction is linear in
    the fPCA scores, so the mean of the reconstructed curves equals the reconstruction
    of the mean scores. Empty unless CURVE_MODE == 'full'."""
    if CURVE_MODE != "full" or not _CURVE_META or not feature_log:
        return ""
    lines = []
    for m in _CURVE_META:
        S = np.array([[r.get(nm, np.nan) for nm in m["score_names"]] for r in feature_log], float)
        ok = np.all(np.isfinite(S), axis=1)
        if not ok.any():
            continue
        vals = np.asarray(m["recon"].reconstruct(S[ok].mean(0)), float)
        lines.append(f"{m['label']} — global mean curve over {int(ok.sum())} droplets "
                     f"({len(m['grid'])} pts): " + _format_curve_line(m["grid"], vals, m["unit"]))
    return "\n".join(lines)


def supplemental_summary(feature_log: List[Dict[str, Any]]) -> str:
    if not feature_log:
        return "(no measured points yet)"
    obj = np.array([r["Objective"] for r in feature_log], float)
    lines = ["feature | global mean [global min, global max] | global corr(Objective)"]
    for nm in _table_scalar_names():
        col = np.array([r.get(nm, np.nan) for r in feature_log], float)
        v = col[np.isfinite(col)]
        if v.size == 0:
            continue
        corr = _pearson(col, obj)
        corr_s = "n/a" if not np.isfinite(corr) else f"{corr:+.2f}"
        lines.append(f"{nm} | {v.mean():.3g} [{v.min():.3g}, {v.max():.3g}] | {corr_s}")
    table = "\n".join(lines)
    curves = _global_curve_block(feature_log)
    return table + ("\n\nFull measured curves (population mean, reconstructed on the "
                    "native grid):\n" + curves if curves else "")


def _informative_scalars(feature_log: List[Dict[str, Any]]) -> List[str]:
    """Scalar-table feature names that actually vary across droplets (drops the
    constant env columns like Temperature_in), so per-droplet feature rows stay
    compact. In condensed mode this also covers the fPCA curve scores."""
    out = []
    for nm in _table_scalar_names():
        v = np.array([r[nm] for r in feature_log if nm in r], float)
        v = v[np.isfinite(v)]
        if v.size and (v.max() - v.min()) > 1e-9:
            out.append(nm)
    return out


def top_k_summary(feature_log: List[Dict[str, Any]], k: int = 8) -> str:
    if not feature_log:
        return "(none yet)"
    scalars = _informative_scalars(feature_log)
    ranked = sorted(feature_log, key=lambda r: r["Objective"], reverse=True)[:k]
    if CURVE_MODE == "full" and _CURVE_META:
        # Per-droplet stanzas: a wide pipe table can't hold full curves.
        blocks = []
        for i, r in enumerate(ranked, 1):
            head = (f"#{i} | FAPbI3={r['FAPbI3']:.3f} | MAPbI3={r['MAPbI3']:.3f} | "
                    f"MAPbBr3={r['MAPbBr3']:.3f} | Objective={r['Objective']:.4f}")
            feats = ", ".join(f"{r[nm]:.3g}" for nm in scalars)
            entry = head + (f"\n  scalars: {feats}" if feats else "")
            cb = _droplet_curve_block(r)
            if cb:
                entry += "\n" + cb
            blocks.append(entry)
        return "\n\n".join(blocks)
    header = "rank | FAPbI3 | MAPbI3 | MAPbBr3 | Objective | " + " | ".join(scalars)
    lines = [header]
    for i, r in enumerate(ranked, 1):
        feats = " | ".join(f"{r.get(nm, float('nan')):.3g}" for nm in scalars)
        lines.append(f"{i} | {r['FAPbI3']:.3f} | {r['MAPbI3']:.3f} | "
                     f"{r['MAPbBr3']:.3f} | {r['Objective']:.4f} | {feats}")
    return "\n".join(lines)


def needle_summary(feature_log: List[Dict[str, Any]], dh) -> str:
    needles = dh_needles(dh)
    if not needles:
        return "(no needles declared yet)"
    if not feature_log:
        return "\n".join(
            f"needle {i} | FAPbI3={n['composition'][0]:.3f}, "
            f"MAPbI3={n['composition'][1]:.3f}, MAPbBr3={n['composition'][2]:.3f} | "
            f"Objective={n['value']:.4f}" for i, n in enumerate(needles, 1))
    scalars = _informative_scalars(feature_log)
    comps = np.array([[r["FAPbI3"], r["MAPbI3"], r["MAPbBr3"]] for r in feature_log], float)
    if CURVE_MODE == "full" and _CURVE_META:
        blocks = []
        for i, n in enumerate(needles, 1):
            c = np.asarray(n["composition"], float)
            j = int(np.argmin(np.linalg.norm(comps - c, axis=1)))  # nearest droplet
            r = feature_log[j]
            head = (f"needle {i} | FAPbI3={c[0]:.3f} | MAPbI3={c[1]:.3f} | "
                    f"MAPbBr3={c[2]:.3f} | declared Objective={n['value']:.4f}")
            feats = ", ".join(f"{r[nm]:.3g}" for nm in scalars)
            entry = head + (f"\n  nearest-droplet scalars: {feats}" if feats else "")
            cb = _droplet_curve_block(r)
            if cb:
                entry += "\n  nearest-droplet curves:\n" + cb
            blocks.append(entry)
        return "\n\n".join(blocks)
    header = ("needle | FAPbI3 | MAPbI3 | MAPbBr3 | declared Objective | "
              + " | ".join(f"{nm} (nearest droplet)" for nm in scalars))
    lines = [header]
    for i, n in enumerate(needles, 1):
        c = np.asarray(n["composition"], float)
        j = int(np.argmin(np.linalg.norm(comps - c, axis=1)))  # nearest measured droplet
        feats = " | ".join(f"{feature_log[j].get(nm, float('nan')):.3g}" for nm in scalars)
        lines.append(f"{i} | {c[0]:.3f} | {c[1]:.3f} | {c[2]:.3f} | "
                     f"{n['value']:.4f} | {feats}")
    return "\n".join(lines)


def best_point_summary(feature_log: List[Dict[str, Any]]) -> str:
    if not feature_log:
        return "(none yet)"
    best = max(feature_log, key=lambda r: r["Objective"])
    comp = f"FAPbI3={best['FAPbI3']:.3f}, MAPbI3={best['MAPbI3']:.3f}, MAPbBr3={best['MAPbBr3']:.3f}"
    feats = ", ".join(f"{nm}={best[nm]:.3g}" for nm in _table_scalar_names() if nm in best)
    out = f"- Objective {best['Objective']:.4f} at {comp}\n- {feats}"
    cb = _droplet_curve_block(best)
    if cb:
        out += "\n- full measured curves:\n" + cb
    return out


def recent_history_table(feature_log: List[Dict[str, Any]], max_rows: int = 48,
                         include_features: bool = True) -> str:
    """The most recent ``max_rows`` droplets (default 48 ≈ the last 2 LineBO lines),
    each shown with its composition and ``Objective``. When ``include_features``
    (default) the interpretable supplemental features are appended too (the constant
    env columns are dropped via ``_informative_scalars`` to keep rows compact); the
    no-features ablation passes ``include_features=False`` so its table stays
    composition→Objective only."""
    if not feature_log:
        return "(no measured points yet)"
    scalars = _informative_scalars(feature_log) if include_features else []
    rows = feature_log[-max_rows:]
    omitted = len(feature_log) - len(rows)
    if include_features and CURVE_MODE == "full" and _CURVE_META:
        # Per-droplet stanzas so each row can carry its full reconstructed curves.
        blocks = []
        if omitted > 0:
            blocks.append(f"... ({omitted} earlier droplets omitted)")
        for r in rows:
            head = (f"FAPbI3={r['FAPbI3']:.3f} | MAPbI3={r['MAPbI3']:.3f} | "
                    f"MAPbBr3={r['MAPbBr3']:.3f} | Objective={r['Objective']:.4f}")
            feats = ", ".join(f"{r[nm]:.3g}" for nm in scalars)
            entry = head + (f"\n  scalars: {feats}" if feats else "")
            cb = _droplet_curve_block(r)
            if cb:
                entry += "\n" + cb
            blocks.append(entry)
        return "\n\n".join(blocks)
    header = "FAPbI3 | MAPbI3 | MAPbBr3 | Objective"
    if scalars:
        header += " | " + " | ".join(scalars)
    lines = [header]
    if omitted > 0:
        lines.append(f"... ({omitted} earlier droplets omitted)")
    for r in rows:
        line = f"{r['FAPbI3']:.3f} | {r['MAPbI3']:.3f} | {r['MAPbBr3']:.3f} | {r['Objective']:.4f}"
        if scalars:
            line += " | " + " | ".join(f"{r.get(nm, float('nan')):.3g}" for nm in scalars)
        lines.append(line)
    return "\n".join(lines)


def dh_needles(dh) -> List[Dict[str, Any]]:
    """(composition, value) of every needle ZoMBI-Hop has declared so far."""
    loc = dh.get_all_needle_locations()
    if loc is None or loc.numel() == 0:
        return []
    locs = as_numpy(loc, dtype=float)
    vals_t = dh.get_all_needle_vals()
    vals = as_numpy(vals_t, dtype=float).ravel() if vals_t.numel() > 0 else np.full(len(locs), np.nan)
    return [{"composition": [float(x) for x in locs[i]],
             "value": float(vals[i]) if i < len(vals) else float("nan")}
            for i in range(len(locs))]


def progress_summary(feature_log: List[Dict[str, Any]], dh, iters_done: int,
                     budget: int) -> str:
    n = len(feature_log)
    if n == 0:
        return "No measured points yet."
    obj = np.array([r["Objective"] for r in feature_log], float)
    bi = int(np.argmax(obj))
    b = feature_log[bi]
    return (
        f"- Measured droplets so far: {n}.\n"
        f"- Objective range found so far (global over all droplets): "
        f"min {obj.min():.4f}, max {obj.max():.4f}.\n"
        f"- Best Objective so far: {obj.max():.4f} at FAPbI3={b['FAPbI3']:.3f}, "
        f"MAPbI3={b['MAPbI3']:.3f}, MAPbBr3={b['MAPbBr3']:.3f}.\n"
        f"- Needles (local optima) declared so far: {len(dh_needles(dh))}.\n"
        f"- Iterations remaining in this run's budget: {budget - iters_done}."
    )


def trend_summary(feature_log: List[Dict[str, Any]], prev_best: Optional[float]) -> str:
    """One-line computed signal on whether the best Objective moved since the last
    injection, so the LLM gets an explicit trend rather than having to infer it from
    the raw rows (LLMs reason poorly over dense numbers; arXiv 2404.06290)."""
    if not feature_log:
        return "(no measured points yet)"
    cur = max(r["Objective"] for r in feature_log)
    if prev_best is None:
        return (f"- This is the first injection; best Objective so far is {cur:.4f}. "
                f"No prior injection to compare against yet.")
    delta = cur - prev_best
    if delta > 1e-9:
        verdict = f"IMPROVED by {delta:+.4f}"
    else:
        verdict = "did NOT improve (plateaued)"
    return (f"- Best Objective at the previous injection: {prev_best:.4f}; now: "
            f"{cur:.4f} → {verdict} since the previous injection.")


def build_injection_prompt(feature_log, dh, hp, injection_idx, iters_done, budget,
                           interval, prev_best: Optional[float] = None) -> str:
    return SURR_PROMPT_TEMPLATE.format(
        system_features=llm_config.SYSTEM_FEATURES,
        hparam_descriptions=E.hparam_descriptions_block(),
        current_hparams=E.format_current_hparams(hp),
        hparam_search_history=E.hparam_optimization_history(top_k=MOBO_HISTORY_TOP_K),
        injection_idx=injection_idx,
        iters_done=iters_done,
        budget=budget,
        progress_summary=progress_summary(feature_log, dh, iters_done, budget),
        trend_summary=trend_summary(feature_log, prev_best),
        history_table=recent_history_table(feature_log),
        n_points=len(feature_log),
        supplemental_summary=supplemental_summary(feature_log),
        best_point_summary=best_point_summary(feature_log),
        top_k=TOP_K_DROPLETS,
        top_k_summary=top_k_summary(feature_log, TOP_K_DROPLETS),
        needle_summary=needle_summary(feature_log, dh),
        interval=interval,
    )


# ════════════════════════════════════════════════════════════════════════════════
# One ZoMBI-Hop segment (fresh cold-start or resume-continuation)
# ════════════════════════════════════════════════════════════════════════════════

def _patch_config(run_dir: Path, hp: Dict[str, Any]) -> None:
    """Rewrite the config-read hyperparameters into a checkpoint's config.json so a
    resume applies them (load_state overwrites constructor values from this file)."""
    cfg_path = run_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    for k, v in hp.items():
        if k in _CONFIG_READ:
            cfg[k] = v
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def run_segment(ckpt_dir: Path, run_uuid: str, fresh: bool, hp: Dict[str, Any],
                fn_callable, stop_at: int, call_counter: List[int],
                payloads: List[dict], snap_records: List[tuple]):
    """Run ZoMBI-Hop until the GLOBAL objective-call counter reaches ``stop_at``.

    ``fresh`` cold-starts a new run (initial design + config.json written from the
    passed hyperparameters); otherwise it resumes the exact state on disk with the
    config-read hyperparameters patched into config.json and the constructor ones
    passed as kwargs. Returns the live DataHandler after the segment stops."""
    run_dir = ckpt_dir / f"run_{run_uuid}"
    constructor_hp = {k: v for k, v in hp.items() if k in _CONSTRUCTOR}

    def obj_wrapper(x_tell, bounds, acq_fn):
        if call_counter[0] >= stop_at:
            raise E.BudgetExhausted()
        x_req, x_act, y = inner(x_tell, bounds, acq_fn)
        call_counter[0] += 1
        dh = dh_ref[0]
        needles = dh.needles
        payloads.append(dict(
            iter_num=call_counter[0],
            needles=(as_numpy(needles) if needles is not None and needles.shape[0] > 0 else None),
            needle_vals=(as_numpy(dh.needle_vals).ravel()
                         if dh.needle_vals is not None and dh.needle_vals.shape[0] > 0 else None),
            line_0=plot_state.get("line_0"), line_1=plot_state.get("line_1"),
            n_points_before=(dh.X_all_actual.shape[0] if dh.X_all_actual is not None else 0),
        ))
        return x_req, x_act, y

    plot_state: Dict[str, Any] = {"line_0": None, "line_1": None}
    sim_obj = R.make_sim_obj(fn_callable, R.DEVICE, R.DTYPE, maximize=MAXIMIZE)
    inner = R.make_linebo_wrapper(sim_obj, 3, R.NUM_LINES, R.DEVICE, R.DTYPE, plot_state)
    dh_ref: List[Any] = [None]

    if fresh:
        X_a, X_e, Y = R._gen_init_data(fn_callable, MAXIMIZE, dim=3)
        optimizer = ZoMBIHop(
            objective=obj_wrapper, X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
            **R.ZOMBI_FIXED, **hp, device=str(R.DEVICE), dtype=R.DTYPE,
            run_uuid=run_uuid, checkpoint_dir=str(ckpt_dir), resume=False,
        )
    else:
        _patch_config(run_dir, hp)
        dummy_Xa = torch.full((1, 3), 1.0 / 3, device=R.DEVICE, dtype=R.DTYPE)
        dummy_Y = torch.zeros(1, 1, device=R.DEVICE, dtype=R.DTYPE)
        optimizer = ZoMBIHop(
            objective=obj_wrapper, X_init_actual=dummy_Xa,
            X_init_expected=dummy_Xa.clone(), Y_init=dummy_Y,
            input_noise=R.NOISE_LEVEL, acquisition_type="ucb", max_gp_points=3000,
            device=str(R.DEVICE), dtype=R.DTYPE, verbose=False,
            run_uuid=run_uuid, resume=True, checkpoint_dir=str(ckpt_dir),
            **constructor_hp,
        )
    dh = optimizer.data_handler
    dh_ref[0] = dh

    orig_snap = dh.take_snapshot

    def snap_wrap(*a, **k):
        orig_snap(*a, **k)
        if dh.X_all_actual is not None:
            czb = dh.current_zoom_bounds if dh.current_zoom_bounds is not None else dh.bounds
            zoom_size = R.zoom_size_fraction(czb) if czb is not None else 1.0
            snap_records.append((dh.X_all_actual.shape[0], dh.current_activation,
                                 dh.current_zoom, zoom_size))
    dh.take_snapshot = snap_wrap

    try:
        # never_terminate: mirror interface/app.py — the optimiser must not stop on
        # its own via any internal pathway (over-penalisation, activation failure,
        # noise-floor exhaustion). It only ends when the objective raises
        # BudgetExhausted at ``stop_at`` — so every segment runs its full iteration
        # budget for any dimensionality, and the baseline curve reaches the last
        # iteration instead of converging short.
        optimizer.run(max_activations=float("inf"), time_limit_hours=None,
                      never_terminate=True)
    except E.BudgetExhausted:
        pass
    except Exception as e:
        print(f"      [segment] ZoMBI-Hop stopped early: {e}")
    return dh


# ════════════════════════════════════════════════════════════════════════════════
# Final metrics + artifacts for a finished trial
# ════════════════════════════════════════════════════════════════════════════════

def finalize_trial(dh, ref_optima, payloads, snap_records, trial_dir: Path) -> Dict[str, Any]:
    """Extract the scalar metrics from the final DataHandler and write run_mobo-style
    artifacts (mirrors evaluate_llm.continue_run's tail)."""
    dim = 3
    needle_t = dh.get_all_needle_locations()
    discovered = as_numpy(needle_t) if needle_t.numel() > 0 else np.empty((0, dim))
    needle_vals_t = dh.get_all_needle_vals()
    best_needle = (float(as_numpy(needle_vals_t).ravel().max())
                   if needle_vals_t.numel() > 0 else float("nan"))
    X_all = as_numpy(dh.X_all_actual) if dh.X_all_actual is not None else np.empty((0, dim))
    Y_all = as_numpy(dh.Y_all).ravel() if dh.Y_all is not None else np.empty((0,))

    # Best Objective over ALL measured points (including points inside declared-
    # needle penalty regions) = the endpoint of the running-best curve, so the
    # significance test and the convergence plot are on the same quantity.
    best_obj = float(Y_all.max()) if Y_all.size else float("nan")
    dist = metric_dist_to_needles(discovered, ref_optima, dim=dim) if len(ref_optima) else float("nan")
    dup = metric_dup_fraction(X_all, dim=dim) if X_all.shape[0] else float("nan")
    # Average pairwise distance between declared needles (how spread out the discovered
    # basins are); nan with fewer than 2 needles (no pair to measure).
    needle_spread = (metric_avg_pairwise_dist(discovered)
                     if discovered.shape[0] >= 2 else float("nan"))

    trial_dir.mkdir(parents=True, exist_ok=True)
    try:
        R.write_points_csv(str(trial_dir / "points.csv"), dh, snap_records, dim=dim)
        R.write_needles_csv(str(trial_dir / "needles.csv"), dh, dim=dim)
        R.write_metrics_over_time_csv(str(trial_dir / "metrics_over_time.csv"),
                                      payloads, X_all, ref_optima, dim=dim)
    except Exception as e:
        print(f"      [finalize] CSV write failed: {e}")
    try:
        R.plot_convergence(str(trial_dir / "convergence.png"), dh, MAXIMIZE)
    except Exception as e:
        print(f"      [finalize] plot failed: {e}")

    return {
        "n_iters": len(payloads),
        "n_points_total": int(X_all.shape[0]),
        "n_needles": int(discovered.shape[0]),
        "best_objective": best_obj,
        "best_needle": best_needle,
        "dist_to_ref_optima": float(dist),
        "dup_fraction": float(dup),
        "mean_pairwise_needle_dist": float(needle_spread),
        "Y_all_running_best": np.maximum.accumulate(Y_all).tolist() if Y_all.size else [],
    }


# ════════════════════════════════════════════════════════════════════════════════
# One trial: LLM cadence run, or baseline
# ════════════════════════════════════════════════════════════════════════════════

def run_baseline_trial(surr, base_hp, ref_optima, seed: int, trial_dir: Path) -> Dict[str, Any]:
    """trial_112 hyperparameters, whole budget in one cold-started run, no LLM."""
    E._seed_everything(seed)
    rng = np.random.default_rng(seed)
    fn_callable, feature_log = make_surrogate_objective(surr, rng)
    tmp = Path(tempfile.mkdtemp(prefix="zombi_surr_"))
    payloads: List[dict] = []
    snap_records: List[tuple] = []
    call_counter = [0]
    try:
        dh = run_segment(tmp, "base", True, base_hp, fn_callable, MAX_ITERS,
                         call_counter, payloads, snap_records)
        metrics = finalize_trial(dh, ref_optima, payloads, snap_records, trial_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    metrics["source"] = "baseline"
    metrics["n_injections"] = 0
    metrics["n_changes"] = 0
    metrics["llm_total_latency_s"] = 0.0
    metrics["llm_latency_per_iter_s"] = 0.0  # no LLM calls → no added latency
    return metrics


# ════════════════════════════════════════════════════════════════════════════════
# Baseline reuse cache (skip recomputing an identical baseline rep)
# ════════════════════════════════════════════════════════════════════════════════
#
# A baseline rep is a deterministic function of (baseline hyperparameters, surrogate,
# iteration budget, seed, fixed ZoMBI config) — there is no LLM in the loop. So a
# baseline rep already computed for that exact fingerprint in ANY prior sweep under
# RESULTS_ROOT can be copied verbatim instead of recomputed. Because reps are keyed
# by seed, a prior sweep that covered only SOME of the seeds still lets us reuse those
# and only recompute the new ones ("part of the seeds or all the seeds").
#
# Each computed baseline rep drops a ``baseline_manifest.json`` recording its
# fingerprint + seed; reuse matches on that. The ``impl`` tag keeps a plain-surrogate
# baseline from being reused by the volume-control sweep (whose metrics.json carries
# extra volume-specific keys), even though the two share the same trajectory.

_BASELINE_MANIFEST = "baseline_manifest.json"


def _canonical_hparams(hp: Dict[str, Any]) -> Dict[str, Any]:
    """Order-independent, float-normalised view of a hyperparameter dict so equal
    settings fingerprint identically regardless of key order or int/float typing."""
    out: Dict[str, Any] = {}
    for k in sorted(hp):
        v = hp[k]
        out[k] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    return out


# Fixed probe compositions for the surrogate fingerprint — deterministic points that
# exercise the conditional-mean landscape without recomputing the full ternary grid.
_SURR_PROBE = np.array([[0.70, 0.20, 0.10], [1 / 3, 1 / 3, 1 / 3],
                        [0.10, 0.10, 0.80], [0.50, 0.25, 0.25],
                        [0.20, 0.50, 0.30], [0.05, 0.90, 0.05]], float)


# Decimals to round surrogate arrays to before hashing. RandomForest.predict with
# n_jobs=-1 is only reproducible to ~1e-12 (thread-reduction order), so an exact byte
# hash of the predictions would differ on every fit. Rounding to 1e-6 absorbs that
# noise while still separating any materially different surrogate (a different DB, fit
# seed, or tree count shifts cond_means far more than 1e-6).
_FP_ROUND = 6


def _surrogate_fingerprint(surr: Surrogate) -> str:
    """Short quantized hash identifying the surrogate's Objective landscape AND
    residual-draw parameters — everything that shapes a baseline trajectory. Robust to
    the ~1e-12 nondeterminism of parallel RF prediction (see ``_FP_ROUND``), so two
    independent fits of the same surrogate on the same data hash identically."""
    import hashlib
    t, it = surr.time_ref["max"], surr.iter_ref["max"]
    Z = np.column_stack([_SURR_PROBE[:, 0], _SURR_PROBE[:, 2],
                         np.full(len(_SURR_PROBE), t), np.full(len(_SURR_PROBE), it)])
    h = hashlib.sha256()
    for arr in (surr._cond_means(Z), surr.resid_mean, surr.resid_std, surr.Sigma):
        q = np.round(np.asarray(arr, float), _FP_ROUND) + 0.0  # +0.0 folds -0.0 → 0.0
        h.update(np.ascontiguousarray(q).tobytes())
    return h.hexdigest()[:16]


def _baseline_spec() -> Dict[str, Any]:
    """The fixed (non-hyperparameter) ZoMBI-Hop configuration a baseline trajectory
    also depends on; drift in any of it must invalidate a cached baseline."""
    return {
        "zombi_fixed": json.loads(json.dumps(R.ZOMBI_FIXED, default=str)),
        "noise_level": float(R.NOISE_LEVEL),
        "num_lines": int(R.NUM_LINES),
        "num_experiments": int(R.NUM_EXPERIMENTS),
        "maximize": bool(MAXIMIZE),
        "dim": 3,
        "n_ref_optima": int(N_REF_OPTIMA),
        "ternary_grid_n": int(R.TERNARY_GRID_N),
    }


def baseline_fingerprint(base_hp: Dict[str, Any], max_iters: int, surr: Surrogate,
                         impl: str) -> Dict[str, Any]:
    """The full identity of a baseline rep (minus its seed). Equal fingerprints ⇒
    interchangeable baseline reps."""
    return {
        "impl": impl,
        "base_hp": _canonical_hparams(base_hp),
        "max_iters": int(max_iters),
        "surrogate": _surrogate_fingerprint(surr),
        "spec": _baseline_spec(),
    }


def _fp_key(fp: Dict[str, Any]) -> str:
    return json.dumps(fp, sort_keys=True)


def find_cached_baseline(fp: Dict[str, Any], seed: int,
                         exclude: Optional[Path] = None) -> Optional[Path]:
    """Search prior sweeps under RESULTS_ROOT for a completed baseline rep whose
    manifest matches ``fp`` and ``seed``. Returns its rep directory, or None. Scoped
    to RESULTS_ROOT with a fixed depth (never a broad filesystem scan)."""
    want = _fp_key(fp)
    for man in sorted(E.RESULTS_ROOT.glob(f"*/baseline*/rep*/{_BASELINE_MANIFEST}")):
        rep_dir = man.parent
        if exclude is not None and rep_dir.resolve() == exclude.resolve():
            continue
        if not (rep_dir / "metrics.json").exists():
            continue
        try:
            m = json.loads(man.read_text())
        except Exception:
            continue
        if int(m.get("seed", -1)) == int(seed) and _fp_key(m.get("fingerprint", {})) == want:
            return rep_dir
    return None


def write_baseline_manifest(trial_dir: Path, fp: Dict[str, Any], seed: int) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / _BASELINE_MANIFEST).write_text(
        json.dumps({"seed": int(seed), "fingerprint": fp}, indent=2))


def _copy_rep_artifacts(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def run_or_reuse_baseline(surr, base_hp, ref_optima, seed: int, trial_dir: Path,
                          max_iters: int, runner, impl: str) -> Tuple[Dict[str, Any], bool]:
    """Return ``(metrics, reused)`` for one baseline rep. If a byte-identical baseline
    (same fingerprint + seed) exists in a prior sweep, copy its artifacts into
    ``trial_dir`` and load its metrics; otherwise run ``runner`` (this or the volume
    sweep's ``run_baseline_trial``) and drop a manifest so later sweeps can reuse it."""
    fp = baseline_fingerprint(base_hp, max_iters, surr, impl)
    src = find_cached_baseline(fp, seed, exclude=trial_dir)
    if src is not None:
        _copy_rep_artifacts(src, trial_dir)
        write_baseline_manifest(trial_dir, fp, seed)  # normalise even if src's differed
        metrics = json.loads((trial_dir / "metrics.json").read_text())
        rel = src.relative_to(E.RESULTS_ROOT) if E.RESULTS_ROOT in src.parents else src
        print(f"    reused cached baseline ← {rel} (seed {seed})")
        return metrics, True
    metrics = runner(surr, base_hp, ref_optima, seed, trial_dir)
    write_baseline_manifest(trial_dir, fp, seed)
    return metrics, False


def backfill_baseline_manifests(sweep_dir: Path, base_hp: Dict[str, Any],
                                max_iters: int, surr: Surrogate, impl: str) -> int:
    """Stamp a ``baseline_manifest.json`` onto each already-computed baseline rep of an
    EXISTING sweep, so future sweeps can reuse it. The manifest is built from the
    CURRENT config's fingerprint, so only call this on a sweep you know was produced
    with the current baseline hyperparameters / budget / surrogate. Rep seeds are
    inferred as ``1000 + <rep index>`` (the sweep convention). Returns the count
    stamped."""
    sweep_dir = Path(sweep_dir)
    base_dir = _find_baseline_group(sweep_dir)
    if base_dir is None:
        print(f"  [backfill] no baseline group under {sweep_dir}")
        return 0
    fp = baseline_fingerprint(base_hp, max_iters, surr, impl)
    print(f"  [backfill] fingerprint surrogate={fp['surrogate']} max_iters={max_iters} "
          f"impl={impl}")
    n = 0
    for rep_dir in sorted(base_dir.glob("rep*")):
        if not (rep_dir / "metrics.json").exists():
            continue
        try:
            seed = 1000 + int(rep_dir.name[3:])
        except ValueError:
            print(f"    [skip] {rep_dir.name}: cannot parse rep index → seed")
            continue
        write_baseline_manifest(rep_dir, fp, seed)
        n += 1
        print(f"    stamped {base_dir.name}/{rep_dir.name} (seed {seed})")
    print(f"  [backfill] wrote {n} manifest(s) under {sweep_dir}")
    return n


def run_llm_trial(surr, base_hp, ref_optima, interval: int, seed: int,
                  trial_dir: Path, prompt_builder=None) -> Dict[str, Any]:
    """One cadence-``interval`` trial: cold-start, then inject the LLM every
    ``interval`` iterations, resuming ZoMBI-Hop's exact state each time.

    ``prompt_builder`` builds each injection prompt; it defaults to this module's
    feature-rich ``build_injection_prompt``. sweep_basic_surrogate_no_features passes
    a feature-ablated builder so only the prompt changes (the surrogate draws, seeds,
    and everything else stay identical → a clean A/B on the supplemental features)."""
    prompt_builder = prompt_builder or build_injection_prompt
    E._seed_everything(seed)
    rng = np.random.default_rng(seed)
    fn_callable, feature_log = make_surrogate_objective(surr, rng)
    tmp = Path(tempfile.mkdtemp(prefix="zombi_surr_"))
    inj_dir = trial_dir / "injections"
    payloads: List[dict] = []
    snap_records: List[tuple] = []
    call_counter = [0]
    hp = dict(base_hp)
    injections: List[Dict[str, Any]] = []
    n_changes = 0
    prev_best: Optional[float] = None  # best Objective at the previous injection

    try:
        fresh = True
        injection_idx = 0
        while call_counter[0] < MAX_ITERS:
            stop_at = min(call_counter[0] + interval, MAX_ITERS)
            before = call_counter[0]
            dh = run_segment(tmp, "llm", fresh, hp, fn_callable, stop_at,
                             call_counter, payloads, snap_records)
            fresh = False
            if call_counter[0] >= MAX_ITERS:
                break  # budget spent → no injection after the final segment
            if call_counter[0] == before:
                # ZoMBI-Hop returned without advancing the objective counter (early
                # stop / caught error); resuming again would spin forever, injecting
                # endlessly. Stop the trial here instead.
                print(f"      [trial] no progress at iter {before}; stopping trial "
                      f"(segment stalled)")
                break

            # Injection: show the LLM the run so far.
            prompt = prompt_builder(feature_log, dh, hp, injection_idx,
                                    call_counter[0], MAX_ITERS, interval, prev_best)
            prev_best = max((r["Objective"] for r in feature_log), default=None)
            this_dir = inj_dir / f"inj_{injection_idx:02d}"
            this_dir.mkdir(parents=True, exist_ok=True)
            (this_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            llm_out = llm_config.call_llm(prompt, R.HPARAM_NAMES)
            decision = llm_out.get("decision") or {}
            raw_changes = decision.get("hyperparameter_changes", []) if isinstance(decision, dict) else []
            changes, warns = E.validate_changes(raw_changes)
            if changes:
                hp.update(changes)
                n_changes += 1
            rec = {
                "injection_idx": injection_idx,
                "iters_done": call_counter[0],
                "latency_s": llm_out["latency_s"],
                "error": llm_out["error"],
                "reasoning": decision.get("reasoning") if isinstance(decision, dict) else None,
                "validated_changes": changes,
                "validation_warnings": warns,
                "hparams_after": dict(hp),
            }
            injections.append(rec)
            (this_dir / "decision.json").write_text(json.dumps(rec, indent=2))
            print(f"      inj {injection_idx} @ iter {call_counter[0]}: "
                  f"{('CHANGE ' + str(changes)) if changes else 'KEEP'} "
                  f"({llm_out['latency_s']:.1f}s)"
                  + (f"  [ERROR: {llm_out['error']}]" if llm_out["error"] else ""))
            # A failed LLM call is not a legitimate "keep" decision — abort the whole
            # run (BaseException escapes the sweep loop's except Exception).
            if llm_out["error"]:
                raise llm_config.LLMCallError(
                    f"LLM call failed at injection {injection_idx} "
                    f"(iter {call_counter[0]}): {llm_out['error']}")
            injection_idx += 1

        metrics = finalize_trial(dh, ref_optima, payloads, snap_records, trial_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    (trial_dir / "injections.json").write_text(json.dumps(injections, indent=2))
    metrics["source"] = f"llm_every_{interval}"
    metrics["n_injections"] = len(injections)
    metrics["n_changes"] = n_changes
    # Latency the LLM calls added, amortized over EVERY ZoMBI-Hop iteration in the
    # trial (including the iterations with no injection) — the per-iteration wall-clock
    # tax of keeping the LLM in the loop.
    total_latency = float(sum((inj.get("latency_s") or 0.0) for inj in injections))
    n_iters = metrics.get("n_iters") or 0
    metrics["llm_total_latency_s"] = total_latency
    metrics["llm_latency_per_iter_s"] = total_latency / n_iters if n_iters else 0.0
    return metrics


# ════════════════════════════════════════════════════════════════════════════════
# Aggregation + summary
# ════════════════════════════════════════════════════════════════════════════════

def aggregate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in _METRIC_KEYS:
        vals = np.array([float(s.get(k, np.nan)) for s in samples], float)
        finite = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "n": int(finite.size),
            "values": [None if not np.isfinite(v) else float(v) for v in vals],
        }
    return out


def _group_row(group: str, interval: Optional[int], stats: Dict[str, Any],
               samples: List[Dict[str, Any]], baseline_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    def m(k, f):
        return (stats.get(k) or {}).get(f)
    row = {
        "group": group,
        "injection_interval": interval,
        "n_repeats": len(samples),
        "best_mean": m("best_objective", "mean"),
        "best_std": m("best_objective", "std"),
        "best_needle_mean": m("best_needle", "mean"),
        "needles_mean": m("n_needles", "mean"),
        "dist_mean": m("dist_to_ref_optima", "mean"),
        "dup_mean": m("dup_fraction", "mean"),
        "needle_spread_mean": m("mean_pairwise_needle_dist", "mean"),
        "needle_spread_std": m("mean_pairwise_needle_dist", "std"),
        "n_injections_mean": float(np.mean([s.get("n_injections", 0) for s in samples])) if samples else None,
        "n_changes_mean": float(np.mean([s.get("n_changes", 0) for s in samples])) if samples else None,
        "llm_latency_per_iter_mean": float(np.mean([s.get("llm_latency_per_iter_s", 0) for s in samples])) if samples else None,
    }
    if baseline_stats is not None:
        diff = (m("best_objective", "mean") or float("nan")) - \
               (baseline_stats["best_objective"]["mean"] or float("nan"))
        sig = SW.welch_significance(baseline_stats["best_objective"]["values"],
                                    stats["best_objective"]["values"], alpha=SIG_ALPHA)
        row.update({
            "diff_best_vs_baseline": diff,
            "diff_best_p_value": sig["p_value"],
            "diff_best_ci95_low": sig["ci95_low"],
            "diff_best_ci95_high": sig["ci95_high"],
        })
    else:
        row.update({"diff_best_vs_baseline": None, "diff_best_p_value": None,
                    "diff_best_ci95_low": None, "diff_best_ci95_high": None})
    return row


_SUMMARY_FIELDS = ["group", "injection_interval", "n_repeats",
                   "best_mean", "best_std", "best_needle_mean", "needles_mean",
                   "dist_mean", "dup_mean", "needle_spread_mean", "needle_spread_std",
                   "n_injections_mean", "n_changes_mean",
                   "llm_latency_per_iter_mean",
                   "diff_best_vs_baseline", "diff_best_p_value",
                   "diff_best_ci95_low", "diff_best_ci95_high"]


def write_summary(sweep_dir: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(sweep_dir / "sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2))


# ════════════════════════════════════════════════════════════════════════════════
# Convergence comparison plot (baseline vs each injection cadence)
# ════════════════════════════════════════════════════════════════════════════════
#
# Reusable by every cadence-style sweep (this one + sweep_volume_control), since all
# of them share the baseline / inject_every_{01,05,10} group layout and
# store a per-repeat ``Y_all_running_best`` in each rep's metrics.json.

# (group directory, legend label, line color) — ordered baseline first.
_CONVERGENCE_GROUPS: List[Tuple[str, str, str]] = [
    ("baseline",          "baseline",             "#4c4c4c"),
    ("inject_every_01",   "LLM every 1",          "#1f77b4"),
    ("inject_every_05",   "LLM every 5",          "#ff7f0e"),
    ("inject_every_10",   "LLM every 10",         "#2ca02c"),
]


def _n_bo_iterations(rep_dir: Path) -> Optional[int]:
    """Number of ZoMBI-Hop BO iterations a rep ran, from its metrics_over_time.csv
    (one row per iteration). None if the file is missing/unreadable."""
    mot = rep_dir / "metrics_over_time.csv"
    if not mot.exists():
        return None
    try:
        with open(mot, newline="", encoding="utf-8") as f:
            return max(sum(1 for _ in csv.reader(f)) - 1, 0)  # minus header
    except Exception:
        return None


def _rep_iteration_curve(rep_dir: Path) -> Optional[np.ndarray]:
    """Running-best Objective sampled at each ZoMBI-Hop *iteration* boundary
    (index 0 = after the Sobol init, index i = after i BO iterations).

    The per-droplet ``Y_all_running_best`` is stored in metrics.json; on the
    surrogate each BO iteration measures a fixed ``NUM_EXPERIMENTS`` droplets after
    an initial design, so iteration i ends at droplet ``len - (n_iter - i)*batch``.
    We sample the (monotone) running-best at those boundaries so the curve lives on
    an iteration axis that is comparable across reps of unequal droplet length."""
    mp = rep_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        rb = np.asarray(json.loads(mp.read_text()).get("Y_all_running_best", []), float)
    except Exception:
        return None
    if rb.size == 0:
        return None
    L = rb.size
    batch = int(R.NUM_EXPERIMENTS)
    n_iter = _n_bo_iterations(rep_dir)
    if not n_iter or n_iter <= 0:
        n_iter = max((L - 48) // batch, 0)  # fallback: 48-droplet init + batch/iter
    idx = L - (n_iter - np.arange(n_iter + 1)) * batch - 1
    idx = np.clip(idx, 0, L - 1)
    return rb[idx]


def _load_running_best_curves(group_dir: Path) -> List[np.ndarray]:
    """Per-iteration running-best curve for every rep under a group directory."""
    curves: List[np.ndarray] = []
    for rep_dir in sorted(group_dir.glob("rep*")):
        c = _rep_iteration_curve(rep_dir)
        if c is not None and c.size:
            curves.append(c)
    return curves


def _ci95_halfwidth(std: np.ndarray, n: int) -> np.ndarray:
    """Half-width of the 95% CI for the mean, using a Student-t multiplier for the
    small repeat counts these sweeps use (falls back to the normal 1.96 without
    scipy)."""
    if n < 2:
        return np.zeros_like(std)
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        t = 1.959963984540054  # z_{0.975}
    return t * std / np.sqrt(n)


def plot_convergence_comparison(sweep_dir: Path, out_png: Optional[Path] = None,
                                title: Optional[str] = None) -> Optional[Path]:
    """Overlay the running-best-Objective convergence of the baseline and each
    injection cadence on one axis: one line per group (its mean over repeats) with a
    shaded 95% confidence interval — lines only, no per-point markers.

    Repeats can run a different number of iterations (early-converging runs stop
    sooner) — both across repeats within a group AND across groups (e.g. every
    baseline repeat may terminate before the longest LLM cadence). Because
    running-best is monotone non-decreasing, each shorter curve is forward-filled
    with its final value up to the GLOBAL iteration count (the longest curve over
    ALL groups), so a converged run — or a whole group that converged early —
    contributes a flat tail that extends to the right edge of the plot rather than
    stopping short. This is what keeps the baseline line running all the way to the
    final iteration even when its trials terminated before the budget was spent."""
    import matplotlib.pyplot as plt  # Agg backend already selected via evaluate_llm

    sweep_dir = Path(sweep_dir)
    if out_png is None:
        out_png = sweep_dir / "convergence_comparison.png"

    # Pass 1: load every group's per-iteration curves so we can forward-fill them all
    # to the same global iteration count.
    loaded: List[Tuple[str, str, List[np.ndarray]]] = []  # (label, color, curves)
    for group, label, color in _CONVERGENCE_GROUPS:
        gdir = sweep_dir / group
        if not gdir.is_dir():
            continue
        curves = _load_running_best_curves(gdir)
        if curves:
            loaded.append((label, color, curves))

    if not loaded:
        print(f"  [plot] no group running-best data under {sweep_dir}; skipped")
        return None

    # Global iteration count across every repeat of every group.
    global_L = max(len(c) for _, _, curves in loaded for c in curves)

    # Pass 2: forward-fill each repeat to global_L and draw one mean±CI line per group.
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(global_L)  # ZoMBI-Hop iteration (0 = after Sobol init)
    for label, color, curves in loaded:
        M = np.vstack([np.concatenate([c, np.full(global_L - len(c), c[-1])])
                       for c in curves])
        mean = M.mean(axis=0)
        std = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(global_L)
        half = _ci95_halfwidth(std, M.shape[0])
        ax.plot(x, mean, color=color, lw=2.0, zorder=5,
                label=f"{label} (n={M.shape[0]})")
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18,
                        linewidth=0, zorder=2)

    ax.set_xlabel("ZoMBI-Hop iteration")
    ax.set_ylabel("running-best Objective")
    ax.set_title(title or ("Convergence: baseline vs LLM injection cadences\n"
                           "(mean ± 95% CI over repeats)"), fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")
    return out_png


def _find_baseline_group(sweep_dir: Path) -> Optional[Path]:
    """Locate the baseline group directory. Newer runs name it ``baseline``; older
    ones name it ``baseline_trial112`` (or another ``baseline_*``). Prefer an exact
    ``baseline`` match, else the first ``baseline*`` directory."""
    exact = sweep_dir / "baseline"
    if exact.is_dir():
        return exact
    cands = sorted(d for d in sweep_dir.glob("baseline*") if d.is_dir())
    return cands[0] if cands else None


def plot_per_rep_vs_baseline(sweep_dir: Path) -> List[Path]:
    """For every rep of every ``inject_every_*`` group, write a 2-line running-best
    convergence plot (NO confidence band): that single LLM rep's curve and the
    baseline rep's curve for the SAME seed (matched by rep index → same ``seed =
    1000 + rep``). Saved as ``convergence_vs_baseline.png`` inside each LLM rep dir.

    This is the CRN-paired per-run view: rep ``r`` of the LLM group and rep ``r`` of
    the baseline share the seed / initial design / surrogate-noise stream, so the two
    lines diverge ONLY because of the LLM's hyperparameter edits — no across-seed
    spread to average over, so no CI is drawn or needed."""
    import matplotlib.pyplot as plt  # Agg backend already selected via evaluate_llm

    sweep_dir = Path(sweep_dir)
    base_dir = _find_baseline_group(sweep_dir)
    if base_dir is None:
        print(f"  [per-rep] no baseline group under {sweep_dir}; skipped")
        return []

    written: List[Path] = []
    for gdir in sorted(sweep_dir.glob("inject_every_*")):
        if not gdir.is_dir():
            continue
        label = gdir.name.replace("inject_every_0", "LLM every ").replace(
            "inject_every_", "LLM every ")
        for rep_dir in sorted(gdir.glob("rep*")):
            llm_curve = _rep_iteration_curve(rep_dir)
            base_rep = base_dir / rep_dir.name          # same rep index = same seed
            base_curve = _rep_iteration_curve(base_rep) if base_rep.is_dir() else None
            if llm_curve is None or base_curve is None:
                print(f"  [per-rep] missing curve for {gdir.name}/{rep_dir.name}; skipped")
                continue
            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            ax.plot(np.arange(len(base_curve)), base_curve, color="#4c4c4c", lw=2.0,
                    label="baseline")
            ax.plot(np.arange(len(llm_curve)), llm_curve, color="#1f77b4", lw=2.0,
                    label=label)
            ax.set_xlabel("ZoMBI-Hop iteration")
            ax.set_ylabel("running-best Objective")
            ax.set_title(f"{gdir.name} / {rep_dir.name} vs baseline "
                         f"(same seed, same CRN stream)", fontsize=10)
            ax.legend(fontsize=9, loc="lower right")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            out = rep_dir / "convergence_vs_baseline.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    print(f"  wrote {len(written)} per-rep vs-baseline plot(s) under {sweep_dir}")
    return written


def regenerate_summary(sweep_dir: Path, group_row=None, write=None,
                       plot: bool = True) -> None:
    """Rebuild sweep_summary.{json,csv} for an existing cadence sweep from each rep's
    metrics.json, WITHOUT re-running any trials. ``best_objective`` is recomputed as
    the max over ALL measured points (= endpoint of the stored per-droplet
    ``Y_all_running_best``) rather than the best un-penalized point, so the summary's
    diff_best / p-values match the convergence plot. ``group_row`` / ``write`` default
    to this module's variants; sweep_volume_control passes its own so its extra
    columns are preserved."""
    group_row = group_row or _group_row
    write = write or write_summary
    sweep_dir = Path(sweep_dir)

    groups: List[Tuple[str, Optional[int]]] = []
    if (sweep_dir / "baseline").is_dir():
        groups.append(("baseline", None))
    for d in sorted(sweep_dir.glob("inject_every_*")):
        if d.is_dir():
            try:
                interval: Optional[int] = int(d.name.rsplit("_", 1)[1])
            except ValueError:
                interval = None
            groups.append((d.name, interval))
    if not groups:
        raise SystemExit(f"No group directories under {sweep_dir}")
    print(f"Regenerating summary for {len(groups)} groups in {sweep_dir}")

    rows: List[dict] = []
    baseline_stats: Optional[Dict[str, Any]] = None
    for group, interval in groups:
        samples: List[Dict[str, Any]] = []
        for rep in sorted((sweep_dir / group).glob("rep*")):
            mp = rep / "metrics.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            rb = m.get("Y_all_running_best")
            if rb:  # best = max over ALL measured points
                m["best_objective"] = float(np.max(np.asarray(rb, float)))
            samples.append(m)
        if not samples:
            print(f"  [skip] {group}: no rep metrics.json")
            continue
        stats = aggregate(samples)
        rows.append(group_row(group, interval, stats, samples, baseline_stats))
        if group == "baseline":
            baseline_stats = stats
        print(f"  {group}: best_mean={rows[-1].get('best_mean')}, "
              f"diff={rows[-1].get('diff_best_vs_baseline')}, "
              f"p={rows[-1].get('diff_best_p_value')}")

    write(sweep_dir, rows)
    print(f"\nWrote {sweep_dir / 'sweep_summary.csv'} and .json")
    if plot:
        plot_convergence_comparison(sweep_dir)
        plot_per_rep_vs_baseline(sweep_dir)


# ════════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════════

def resolve_baseline_hparams() -> Tuple[Dict[str, Any], str]:
    """Baseline/starting hyperparameters for the sweep. If ``BASELINE_TRIAL_DIR`` is
    set, read them from that MOBO ``trial_*`` directory's trial.json; otherwise pick
    the best-dist_to_needles 3d ensemble MOBO trial, falling back to trial_112 /
    run_7eb9 when no ensemble run has been synced yet."""
    if BASELINE_TRIAL_DIR:
        tdir = Path(BASELINE_TRIAL_DIR)
        if not tdir.is_absolute():
            tdir = E._ROOT / tdir
        return E.hparams_from_trial_dir(tdir), f"trial dir {tdir.name} ({tdir})"
    base_hp, base_desc = E.best_dist_ensemble_hparams()
    if base_hp is None:
        with open(E.RUN_DIR / "config.json") as f:
            run_config = json.load(f)
        base_hp = E.current_hparams(run_config)
        base_desc = f"trial_112 fallback ({base_desc})"
    return base_hp, base_desc


def main(sweep_prefix: str = "sweep_surrogate", prompt_builder=None,
         plot_title: Optional[str] = None) -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = E.RESULTS_ROOT / f"{sweep_prefix}_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Surrogate LLM-in-the-loop sweep\n  sweep dir: {sweep_dir}")
    print(f"  cadences: {INJECTION_INTERVALS}   budget: {MAX_ITERS} iters   "
          f"repeats: {N_REPEATS}")

    # Fit (or load) the generative surrogate once; share across all trials.
    if SURROGATE_PICKLE and Path(SURROGATE_PICKLE).exists():
        print(f"  loading surrogate ← {SURROGATE_PICKLE}")
        surr = Surrogate.load(SURROGATE_PICKLE)
    else:
        print("  fitting generative surrogate …")
        surr = Surrogate.fit(verbose=False)

    base_hp, base_desc = resolve_baseline_hparams()
    print(f"  baseline hyperparameters [{base_desc}]: {base_hp}")

    # True optima of the surrogate's deterministic Objective landscape (needle metric).
    predictor = _ObjMeanPredictor(surr)
    grid_pts = R.ternary_grid(R.TERNARY_GRID_N)
    grid_vals = predictor.predict(grid_pts)
    ref_optima = R.auto_detect_rf_optima(predictor, grid_pts, grid_vals,
                                         maximize=MAXIMIZE, n_peaks=N_REF_OPTIMA)

    rows: List[dict] = []

    # ── Baseline group ───────────────────────────────────────────────────────────
    print(f"\n[baseline] {N_REPEATS} repeats")
    baseline_samples: List[Dict[str, Any]] = []
    for rep in range(N_REPEATS):
        seed = 1000 + rep
        tdir = sweep_dir / "baseline" / f"rep{rep}"
        print(f"  rep {rep} (seed {seed}) …")
        try:
            m, reused = run_or_reuse_baseline(surr, base_hp, ref_optima, seed, tdir,
                                              MAX_ITERS, run_baseline_trial,
                                              impl="surrogate")
            print(f"    best={m['best_objective']:.4f}, needles={m['n_needles']}, "
                  f"dup={m['dup_fraction']:.4f}" + ("  [reused]" if reused else ""))
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            m = {"source": "baseline", "n_injections": 0, "n_changes": 0}
        baseline_samples.append(m)
        (tdir / "metrics.json").write_text(json.dumps(m, indent=2))
    baseline_stats = aggregate(baseline_samples)
    rows.append(_group_row("baseline", None, baseline_stats, baseline_samples, None))
    write_summary(sweep_dir, rows)

    # ── LLM cadence groups ───────────────────────────────────────────────────────
    for interval in INJECTION_INTERVALS:
        group = f"inject_every_{interval:02d}"
        print(f"\n[{group}] {N_REPEATS} repeats")
        samples: List[Dict[str, Any]] = []
        for rep in range(N_REPEATS):
            seed = 1000 + rep   # common random numbers with the baseline rep
            tdir = sweep_dir / group / f"rep{rep}"
            print(f"  rep {rep} (seed {seed}) …")
            try:
                m = run_llm_trial(surr, base_hp, ref_optima, interval, seed, tdir,
                                  prompt_builder=prompt_builder)
                print(f"    best={m['best_objective']:.4f}, needles={m['n_needles']}, "
                      f"dup={m['dup_fraction']:.4f}, injections={m['n_injections']}, "
                      f"changes={m['n_changes']}")
            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                m = {"source": group, "n_injections": 0, "n_changes": 0}
            samples.append(m)
            (tdir / "metrics.json").write_text(json.dumps(m, indent=2))
        stats = aggregate(samples)
        rows.append(_group_row(group, interval, stats, samples, baseline_stats))
        write_summary(sweep_dir, rows)  # incremental

    # Overlaid running-best convergence: baseline vs each cadence (mean ± 95% CI).
    print("\n[plot] convergence comparison …")
    plot_convergence_comparison(sweep_dir, title=plot_title)
    # Per-rep seed-matched view: each LLM rep vs its CRN baseline (2 lines, no CI).
    plot_per_rep_vs_baseline(sweep_dir)

    print(f"\nSweep complete → {sweep_dir / 'sweep_summary.csv'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] in ("--regenerate", "-r"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_surrogate.py --regenerate <sweep_dir>")
        regenerate_summary(Path(args[1]))
    elif args and args[0] in ("--plot", "-p"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_surrogate.py --plot <sweep_dir>")
        plot_convergence_comparison(Path(args[1]))
        plot_per_rep_vs_baseline(Path(args[1]))
    elif args and args[0] in ("--per-rep",):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_surrogate.py --per-rep <sweep_dir>")
        plot_per_rep_vs_baseline(Path(args[1]))
    elif args and args[0] in ("--backfill-baselines",):
        if len(args) < 2:
            raise SystemExit("usage: sweep_basic_surrogate.py --backfill-baselines <sweep_dir>")
        _surr = (Surrogate.load(SURROGATE_PICKLE)
                 if SURROGATE_PICKLE and Path(SURROGATE_PICKLE).exists()
                 else Surrogate.fit(verbose=False))
        _bhp, _desc = resolve_baseline_hparams()
        print(f"baseline hyperparameters [{_desc}]: {_bhp}")
        backfill_baseline_manifests(Path(args[1]), _bhp, MAX_ITERS, _surr,
                                    impl="surrogate")
    else:
        main()
