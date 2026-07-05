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
        baseline_trial112/rep0 … rep{N-1}/
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
INJECTION_INTERVALS: list[int] = [1, 5, 10]   # LLM injects every k iterations
MAX_ITERS: int = 40                            # total ZoMBI-Hop iterations per trial
N_REPEATS: int = 3                             # trials per group (variance)
# Cost note: cadence k does ~ceil(MAX_ITERS/k)-1 LLM calls per repeat, so with the
# defaults k=1 ≈ 19, k=5 ≈ 3, k=10 ≈ 1 calls/repeat → ~115 LLM calls total across
# the 5 repeats of the three cadences (the baseline calls the LLM zero times).
SURROGATE_PICKLE: str | None = None            # reuse a fitted surrogate if set
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
    metric_dist_to_needles,
    metric_dup_fraction,
)
from src.core.zombihop import ZoMBIHop  # noqa: E402

from surrogate import Surrogate, SUBTARGETS, ENV, KIN  # noqa: E402

# Interpretable supplemental scalars shown to the LLM (no fPCA curve scores).
SUP_SCALARS: List[str] = list(SUBTARGETS) + list(ENV) + list(KIN)

MAXIMIZE = True
N_REF_OPTIMA = 3
SIG_ALPHA = 0.05

# Split the tunable hyperparameters the same way evaluate_llm does: config-read
# ones are applied by rewriting the resumed run's config.json, the rest are passed
# to the ZoMBIHop constructor on each resume.
_CONFIG_READ = E._CONFIG_READ_HPARAMS
_CONSTRUCTOR = E._CONSTRUCTOR_HPARAMS

_METRIC_KEYS = ["best_objective", "best_needle", "n_needles",
                "dist_to_ref_optima", "dup_fraction"]


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
    oi = surr.obj_index
    t, it = surr.time_ref["max"], surr.iter_ref["max"]
    name_idx = {nm: surr.feat_names.index(nm) for nm in ([surr.feat_names[oi]] + SUP_SCALARS)}
    feature_log: List[Dict[str, Any]] = []

    def fn(x) -> float:
        comp = np.asarray(x, float)
        s = comp.sum()
        comp = comp / (s if s != 0 else 1.0)
        Z = np.array([[comp[0], comp[2], t, it]], float)
        row = (surr._cond_means(Z) + surr._draw_resid(1, rng)).ravel()
        rec: Dict[str, Any] = {"FAPbI3": comp[0], "MAPbI3": comp[1], "MAPbBr3": comp[2],
                               "Objective": float(row[oi])}
        for nm in SUP_SCALARS:
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
LineBO line of ~24 measured droplets, declares a needle when expected improvement
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

Recent measured points (composition → Objective):

{history_table}

## Supplemental measured features so far

Beyond the composition and `Objective`, these interpretable per-droplet scalars
were measured across all {n_points} droplets so far (mean [min, max], and the
Pearson correlation of each with `Objective` — a positive corr means the feature
tends to rise where the objective is high):

{supplemental_summary}

Best droplet so far — its supplemental features:

{best_point_summary}

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
"""


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def supplemental_summary(feature_log: List[Dict[str, Any]]) -> str:
    if not feature_log:
        return "(no measured points yet)"
    obj = np.array([r["Objective"] for r in feature_log], float)
    lines = ["feature | mean [min, max] | corr(Objective)"]
    for nm in SUP_SCALARS:
        v = np.array([r[nm] for r in feature_log], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        corr = _pearson(np.array([r[nm] for r in feature_log], float), obj)
        corr_s = "n/a" if not np.isfinite(corr) else f"{corr:+.2f}"
        lines.append(f"{nm} | {v.mean():.3g} [{v.min():.3g}, {v.max():.3g}] | {corr_s}")
    return "\n".join(lines)


def best_point_summary(feature_log: List[Dict[str, Any]]) -> str:
    if not feature_log:
        return "(none yet)"
    best = max(feature_log, key=lambda r: r["Objective"])
    comp = f"FAPbI3={best['FAPbI3']:.3f}, MAPbI3={best['MAPbI3']:.3f}, MAPbBr3={best['MAPbBr3']:.3f}"
    feats = ", ".join(f"{nm}={best[nm]:.3g}" for nm in SUP_SCALARS if nm in best)
    return f"- Objective {best['Objective']:.4f} at {comp}\n- {feats}"


def recent_history_table(feature_log: List[Dict[str, Any]], max_rows: int = 40) -> str:
    if not feature_log:
        return "(no measured points yet)"
    rows = feature_log[-max_rows:]
    lines = ["FAPbI3 | MAPbI3 | MAPbBr3 | Objective"]
    if len(feature_log) > max_rows:
        lines.append(f"... ({len(feature_log) - max_rows} earlier droplets omitted)")
    for r in rows:
        lines.append(f"{r['FAPbI3']:.3f} | {r['MAPbI3']:.3f} | {r['MAPbBr3']:.3f} | {r['Objective']:.4f}")
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
        f"- Best Objective so far: {obj.max():.4f} at FAPbI3={b['FAPbI3']:.3f}, "
        f"MAPbI3={b['MAPbI3']:.3f}, MAPbBr3={b['MAPbBr3']:.3f}.\n"
        f"- Needles (local optima) declared so far: {len(dh_needles(dh))}.\n"
        f"- Iterations remaining in this run's budget: {budget - iters_done}."
    )


def build_injection_prompt(feature_log, dh, hp, injection_idx, iters_done, budget,
                           interval) -> str:
    return SURR_PROMPT_TEMPLATE.format(
        system_features=llm_config.SYSTEM_FEATURES,
        hparam_descriptions=E.hparam_descriptions_block(),
        current_hparams=E.format_current_hparams(hp),
        hparam_search_history=E.hparam_optimization_history(),
        injection_idx=injection_idx,
        iters_done=iters_done,
        budget=budget,
        progress_summary=progress_summary(feature_log, dh, iters_done, budget),
        history_table=recent_history_table(feature_log),
        n_points=len(feature_log),
        supplemental_summary=supplemental_summary(feature_log),
        best_point_summary=best_point_summary(feature_log),
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
        optimizer.run(max_activations=float("inf"), time_limit_hours=None)
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

    bx, by, _ = dh.get_best_unpenalized()
    best_obj = float(by.item()) if by is not None else (float(Y_all.max()) if Y_all.size else float("nan"))
    dist = metric_dist_to_needles(discovered, ref_optima, dim=dim) if len(ref_optima) else float("nan")
    dup = metric_dup_fraction(X_all, dim=dim) if X_all.shape[0] else float("nan")

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
    metrics["source"] = "baseline_trial112"
    metrics["n_injections"] = 0
    metrics["n_changes"] = 0
    return metrics


def run_llm_trial(surr, base_hp, ref_optima, interval: int, seed: int,
                  trial_dir: Path) -> Dict[str, Any]:
    """One cadence-``interval`` trial: cold-start, then inject the LLM every
    ``interval`` iterations, resuming ZoMBI-Hop's exact state each time."""
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

    try:
        fresh = True
        injection_idx = 0
        while call_counter[0] < MAX_ITERS:
            stop_at = min(call_counter[0] + interval, MAX_ITERS)
            dh = run_segment(tmp, "llm", fresh, hp, fn_callable, stop_at,
                             call_counter, payloads, snap_records)
            fresh = False
            if call_counter[0] >= MAX_ITERS:
                break  # budget spent → no injection after the final segment

            # Injection: show the LLM the run so far (incl. supplemental features).
            prompt = build_injection_prompt(feature_log, dh, hp, injection_idx,
                                            call_counter[0], MAX_ITERS, interval)
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
            injection_idx += 1

        metrics = finalize_trial(dh, ref_optima, payloads, snap_records, trial_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    (trial_dir / "injections.json").write_text(json.dumps(injections, indent=2))
    metrics["source"] = f"llm_every_{interval}"
    metrics["n_injections"] = len(injections)
    metrics["n_changes"] = n_changes
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
        "n_injections_mean": float(np.mean([s.get("n_injections", 0) for s in samples])) if samples else None,
        "n_changes_mean": float(np.mean([s.get("n_changes", 0) for s in samples])) if samples else None,
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
                   "dist_mean", "dup_mean", "n_injections_mean", "n_changes_mean",
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
# Orchestration
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = E.RESULTS_ROOT / f"sweep_surrogate_{ts}"
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

    # trial_112 hyperparameters = the values run_7eb9 actually used (offline-MOBO pick).
    with open(E.RUN_DIR / "config.json") as f:
        run_config = json.load(f)
    base_hp = E.current_hparams(run_config)
    print(f"  trial_112 hyperparameters: {base_hp}")

    # True optima of the surrogate's deterministic Objective landscape (needle metric).
    predictor = _ObjMeanPredictor(surr)
    grid_pts = R.ternary_grid(R.TERNARY_GRID_N)
    grid_vals = predictor.predict(grid_pts)
    ref_optima = R.auto_detect_rf_optima(predictor, grid_pts, grid_vals,
                                         maximize=MAXIMIZE, n_peaks=N_REF_OPTIMA)

    rows: List[dict] = []

    # ── Baseline group ───────────────────────────────────────────────────────────
    print(f"\n[baseline_trial112] {N_REPEATS} repeats")
    baseline_samples: List[Dict[str, Any]] = []
    for rep in range(N_REPEATS):
        seed = 1000 + rep
        tdir = sweep_dir / "baseline_trial112" / f"rep{rep}"
        print(f"  rep {rep} (seed {seed}) …")
        try:
            m = run_baseline_trial(surr, base_hp, ref_optima, seed, tdir)
            print(f"    best={m['best_objective']:.4f}, needles={m['n_needles']}, "
                  f"dup={m['dup_fraction']:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            m = {"source": "baseline_trial112", "n_injections": 0, "n_changes": 0}
        baseline_samples.append(m)
        (tdir / "metrics.json").write_text(json.dumps(m, indent=2))
    baseline_stats = aggregate(baseline_samples)
    rows.append(_group_row("baseline_trial112", None, baseline_stats, baseline_samples, None))
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
                m = run_llm_trial(surr, base_hp, ref_optima, interval, seed, tdir)
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

    print(f"\nSweep complete → {sweep_dir / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
