"""
llm/evaluate_llm.py
===================
Run the LLM-in-the-loop ZoMBI-Hop hyperparameter-tuning experiment on a single
injection point.

Pipeline (one LLM call per run):
  1. Build a Random-Forest reconstruction of the campaign2 objective (exactly like
     visualization/plot_run.py) — this is the surrogate the *continued* run is
     evaluated on.
  2. Take the real run's history (llm/data/campaign2_all.db) up to and including
     ``INJECTION_ITER``, plus ZoMBI-Hop's true internal state at that point
     (reconstructed from the delta snapshots in runs/run_7eb9, which is the same
     run logged with full algorithm state).
  3. Ask the LLM (see llm/llm_config.py) once whether it wants to change any
     ZoMBI-Hop hyperparameters. Time the response.
  4. If YES: resume vanilla ZoMBI-Hop from the exact state at INJECTION_ITER with
     the new hyperparameters, running on the RF objective for the SAME number of
     additional ZoMBI iterations the real run used after the injection point
     ("equal budget"). Log everything run_mobo.py logs, plus the difference vs the
     baseline (the real campaign2 trajectory) and the LLM latency.
     If NO: no rerun is needed — the outcome equals the baseline; log the decision
     and latency.

Usage:
  conda activate zombi-hop
  python llm/evaluate_llm.py            # uses the INJECTION_ITER constant below

Change the injection point via the INJECTION_ITER constant at the top, the model /
prompt via llm/llm_config.py, and sweep many injection points via
llm/evaluate_llm_sweep.py.
"""

from __future__ import annotations

# ─── HARDCODED CONFIG ──────────────────────────────────────────────────────────
INJECTION_ITER: int = 15          # db iteration (0..40) at which the LLM intervenes

# Repeats to average out the stochasticity of ZoMBI-Hop + the noisy RF objective.
# The LLM is still called only ONCE per injection point; only the continuation is
# repeated.
N_LLM_REPEATS: int = 5            # RF continuations with the LLM's hyperparameters
# Baseline variance: the real campaign2 run counts as sample #1, plus
# (N_BASELINE_REPEATS - 1) RF continuations with the ORIGINAL (run_7eb9 / trial_112)
# hyperparameters, resumed from the same injection state. Mean/std use all of them.
N_BASELINE_REPEATS: int = 5
# ───────────────────────────────────────────────────────────────────────────────

import datetime
import json
import os
import re
import shutil
import sqlite3
import tempfile
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Windows consoles default to cp1252; keep pretty console output from crashing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# ── project paths: put repo root AND optimize/ on sys.path so `run_mobo` imports
#    resolve the same way they do when run as a script (from optimize/). ──────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_OPT = _ROOT / "optimize"
for _p in (str(_OPT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import run_mobo as R  # noqa: E402  (heavy import: botorch/eval_metrics; see path hack above)
from eval_metrics import (  # noqa: E402
    as_numpy,
    metric_dist_to_needles,
    metric_dup_fraction,
)
from src.core.zombihop import ZoMBIHop  # noqa: E402
from src.utils.datahandler import reconstruct_snapshot_tensors  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import llm_config  # noqa: E402  (llm/ is the script dir → on sys.path)


# ── fixed inputs ───────────────────────────────────────────────────────────────
DB_PATH = _HERE / "data" / "campaign2_all.db"
RUN_DIR = _ROOT / "runs" / "run_7eb9"
# Offline MOBO hyperparameter-search run whose per-trial results (trial_*/trial.json)
# are shown to the LLM as the hyperparameter-optimization history. Point this at the
# full optimization run (hundreds of trials); the example below has just one.
MOBO_RUN_DIR = _ROOT / "optimize" / "runs" / "mobo_05_06_15_32"
COMP_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
VALUE_COL = "Objective"
MAXIMIZE = True                     # Objective is maximized
N_REF_OPTIMA = 3                    # RF peaks used as reference "true optima" for needle metrics
RESULTS_ROOT = _HERE / "results"

# Hyperparameters that ZoMBI-Hop's config.json (and DataHandler.load_state) reads.
# Overriding these is done by rewriting the resumed run's config.json.
_CONFIG_READ_HPARAMS = {
    "max_zooms", "max_iterations", "top_m_points", "n_restarts", "raw",
    "input_noise_threshold_mult", "output_noise_threshold_mult",
    "n_consecutive_converged", "ucb_beta", "nat_grad_step", "nat_grad_max_steps",
}
# HPARAM_SPACE keys NOT read from config.json — passed to the ZoMBIHop constructor.
_CONSTRUCTOR_HPARAMS = {
    "max_penalty_radius", "needle_shrink_factor", "needle_stop_noise_multiplier",
    "paring_spatial_halfnoise", "paring_y_noise_multiplier",
}

# One-line descriptions of each tunable hyperparameter for the LLM prompt.
HPARAM_DESC: Dict[str, str] = {
    "nat_grad_step": "step size for the natural-gradient acquisition optimizer on the simplex (larger = bolder moves).",
    "nat_grad_max_steps": "max natural-gradient steps when optimizing the acquisition (more = finer candidate search, slower).",
    "n_restarts": "number of random restarts for acquisition optimization (more = less likely to miss the acquisition optimum).",
    "raw": "number of raw random samples seeding acquisition optimization / penalty-coverage checks.",
    "ucb_beta": "UCB exploration weight: higher = more exploration of uncertain regions, lower = more exploitation of the current best.",
    "max_zooms": "max number of zoom-ins (trust-region shrinks) per activation before declaring a needle.",
    "max_iterations": "max acquisition iterations per zoom level before moving on.",
    "top_m_points": "how many top points define the next zoom's bounding box (larger = wider zoom regions).",
    "n_consecutive_converged": "consecutive low-improvement iterations required to declare a needle (higher = more reluctant to call a needle).",
    "input_noise_threshold_mult": "multiplier on input noise controlling how aggressively nearby points are treated as duplicates when zooming/stopping.",
    "output_noise_threshold_mult": "multiplier on output noise for the convergence gate: lower = converges sooner (declares needles faster).",
    "max_penalty_radius": "max radius of the ellipsoid that penalizes the region around a found needle (larger = more area excluded from future search).",
    "needle_shrink_factor": "factor by which needle trust regions shrink on retry (closer to 1 = shrink slowly).",
    "needle_stop_noise_multiplier": "noise multiplier controlling when a needle's trust region is considered collapsed (stop).",
    "paring_spatial_halfnoise": "spatial radius (in input-noise units) for deduplicating/averaging nearby measurements before the GP.",
    "paring_y_noise_multiplier": "objective-noise multiplier for the point-paring deduplication.",
}


class BudgetExhausted(Exception):
    """Raised inside the objective wrapper once the equal-budget iteration cap is hit."""


# ════════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════════

def load_db_rows() -> List[Dict[str, Any]]:
    """Every campaign2 row in chronological (rowid) order with comps + Objective."""
    con = sqlite3.connect(str(DB_PATH))
    try:
        cols = ["Iteration"] + COMP_COLS + [VALUE_COL]
        sel = ", ".join(f'"{c}"' for c in cols)
        rows = con.execute(f"SELECT {sel} FROM results ORDER BY rowid").fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        out.append({
            "Iteration": None if r[0] is None else int(round(r[0])),
            "FAPbI3": r[1], "MAPbI3": r[2], "MAPbBr3": r[3],
            "Objective": r[4],
        })
    return out


def measured_points(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """(X (N,3), Y (N,), iteration_per_point) for rows with a non-null Objective,
    composition renormalised to sum 1, in chronological order."""
    X, Y, its = [], [], []
    for r in rows:
        if r["Objective"] is None:
            continue
        comp = np.array([r["FAPbI3"], r["MAPbI3"], r["MAPbBr3"]], dtype=float)
        if not np.all(np.isfinite(comp)):
            continue
        s = comp.sum()
        comp = comp / (s if s != 0 else 1.0)
        X.append(comp)
        Y.append(float(r["Objective"]))
        its.append(r["Iteration"] if r["Iteration"] is not None else -1)
    return np.asarray(X, float), np.asarray(Y, float), its


def build_rf(X: np.ndarray, Y: np.ndarray):
    """RF surrogate + ternary grid, exactly like plot_run.py / run_mobo.build_rf_and_grid.
    Returns (rf, fn_callable, grid_pts, grid_vals)."""
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=R.RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X, Y)
    grid_pts = R.ternary_grid(R.TERNARY_GRID_N)
    grid_vals = rf.predict(grid_pts)
    fn_callable = lambda x, _rf=rf: float(_rf.predict(np.asarray(x, float).reshape(1, -1))[0])
    fn_callable.predict = rf.predict  # so make_sim_obj / any predict-based code works
    return rf, fn_callable, grid_pts, grid_vals


# ════════════════════════════════════════════════════════════════════════════════
# Snapshot ↔ injection-iteration mapping
# ════════════════════════════════════════════════════════════════════════════════

_SNAP_OBJCALL_RE = re.compile(r"_act\d+_z\d+_i\d+$")


def snapshot_index(name: str) -> int:
    """Leading zero-padded counter of a snapshot dir name, e.g. '0050_act13_z0_i5' -> 50."""
    try:
        return int(name.split("_", 1)[0])
    except ValueError:
        return -1


def scan_snapshots() -> List[Dict[str, Any]]:
    """Chronological list of snapshots with cumulative measured-point count + flags.

    Reconstruction replays deltas once (cheap for ~50 snapshots). Each entry:
      {name, idx, n_points, is_objcall, is_needle}
    """
    snap_dir = RUN_DIR / "snapshots"
    names = sorted(p.name for p in snap_dir.iterdir() if p.is_dir())
    out = []
    for nm in names:
        t = reconstruct_snapshot_tensors(RUN_DIR, nm, device="cpu")
        X = t.get("X_all_actual")
        n = int(X.shape[0]) if (X is not None and X.ndim == 2) else 0
        out.append({
            "name": nm,
            "idx": snapshot_index(nm),
            "n_points": n,
            "is_objcall": bool(_SNAP_OBJCALL_RE.search(nm)),
            "is_needle": nm.endswith("_needle"),
        })
    return out


def map_injection(injection_iter: int, rows: List[Dict[str, Any]],
                  snaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map a db injection iteration to a resume snapshot + equal-budget iteration cap.

    The db 'Iteration' segmentation differs from ZoMBI-Hop's internal iterations,
    but both are the same measured stream, so we align by cumulative measured-point
    count: N = #(non-null-Objective rows with Iteration <= injection_iter). The
    resume snapshot is the one whose reconstructed data count is closest to N.
    The equal budget = number of objective-call snapshots after that snapshot.
    """
    n_target = sum(1 for r in rows
                   if r["Objective"] is not None and r["Iteration"] is not None
                   and r["Iteration"] <= injection_iter)

    # Closest snapshot by cumulative point count (prefer objective-call snapshots).
    best = min(snaps, key=lambda s: (abs(s["n_points"] - n_target), 0 if s["is_objcall"] else 1))
    inj_idx = best["idx"]

    budget = sum(1 for s in snaps if s["is_objcall"] and s["idx"] > inj_idx)

    return {
        "injection_iter": injection_iter,
        "n_points_at_injection": n_target,
        "snapshot_name": best["name"],
        "snapshot_n_points": best["n_points"],
        "budget_iterations": budget,
    }


def needles_at_snapshot(snapshot_name: str) -> List[Dict[str, Any]]:
    """Needles ZoMBI-Hop had already declared as of ``snapshot_name``."""
    t = reconstruct_snapshot_tensors(RUN_DIR, snapshot_name, device="cpu")
    needles = t.get("needles")
    vals = t.get("needle_vals")
    idxs = t.get("needle_indices")
    if needles is None or (hasattr(needles, "shape") and needles.shape[0] == 0):
        return []
    needles = as_numpy(needles, dtype=float)
    vals = as_numpy(vals, dtype=float).ravel() if vals is not None else np.full(len(needles), np.nan)
    idxs = as_numpy(idxs).ravel().tolist() if idxs is not None else [None] * len(needles)
    out = []
    for i in range(needles.shape[0]):
        out.append({
            "composition": [float(x) for x in needles[i]],
            "value": float(vals[i]) if i < len(vals) else float("nan"),
            "found_at_point_index": (int(idxs[i]) if i < len(idxs) and idxs[i] is not None else None),
        })
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Prompt construction
# ════════════════════════════════════════════════════════════════════════════════

def hparam_descriptions_block() -> str:
    lines = []
    for name in R.HPARAM_NAMES:
        lo, hi, tfm = R.HPARAM_SPACE[name]
        kind = "integer" if tfm == "int" else "float"
        desc = HPARAM_DESC.get(name, "")
        lines.append(f"- `{name}` ({kind}, range [{lo}, {hi}]): {desc}")
    return "\n".join(lines)


def current_hparams(run_config: Dict[str, Any]) -> Dict[str, Any]:
    """The hyperparameter values in effect at the injection point.

    Config-read hyperparameters come from run_7eb9/config.json; the rest used
    ZoMBIHop's constructor defaults during the original run.
    """
    import inspect
    defaults = {}
    sig = inspect.signature(ZoMBIHop.__init__)
    for name in _CONSTRUCTOR_HPARAMS:
        p = sig.parameters.get(name)
        if p is not None and p.default is not inspect.Parameter.empty:
            defaults[name] = p.default
    hp = {}
    for name in R.HPARAM_NAMES:
        if name in run_config:
            hp[name] = run_config[name]
        elif name in defaults:
            hp[name] = defaults[name]
    return hp


def format_current_hparams(hp: Dict[str, Any]) -> str:
    return "\n".join(f"- `{k}` = {v}" for k, v in hp.items())


def _fmt_num(v: Any) -> str:
    """Compact numeric formatting for the trial table."""
    if v is None:
        return ""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.5g}"
    return str(v)


# The three MOBO objectives recorded per trial (all MINIMIZED, lower is better).
_MOBO_METRIC_KEYS = ("dist_to_needles", "dup_fraction", "runtime_s")


def hparam_optimization_history(mobo_run_dir: Path = MOBO_RUN_DIR,
                                max_trials: int = 1000) -> str:
    """Full offline hyperparameter-search history as a compact markdown table.

    Reads every ``trial_*/trial.json`` under ``mobo_run_dir`` (each holds a tried
    hyperparameter config plus its three MOBO objective scores) and renders one
    row per trial. The hyperparameter column order follows the run's
    ``run_config.json['hparam_names']`` when available.
    """
    trial_files = sorted(
        mobo_run_dir.glob("trial_*/trial.json"),
        key=lambda p: int(re.search(r"trial_(\d+)", p.parent.name).group(1)),
    )
    if not trial_files:
        return f"(no trials found under {mobo_run_dir})"

    trials: List[Dict[str, Any]] = []
    for tf in trial_files:
        try:
            trials.append(json.loads(tf.read_text()))
        except Exception:
            continue
    if not trials:
        return f"(could not read any trials under {mobo_run_dir})"

    # Hyperparameter column order: prefer the run's declared order.
    hp_order: List[str] = []
    cfg = mobo_run_dir / "run_config.json"
    if cfg.exists():
        try:
            hp_order = list(json.loads(cfg.read_text()).get("hparam_names", []))
        except Exception:
            hp_order = []
    if not hp_order:
        hp_order = list(trials[0].get("hparams", {}).keys())

    header = ["trial"] + hp_order + list(_MOBO_METRIC_KEYS)
    lines = [" | ".join(header), " | ".join("---" for _ in header)]

    truncated = False
    for t in trials[:max_trials]:
        hp = t.get("hparams", {})
        metrics = t.get("metrics", {})
        row = [str(t.get("trial", ""))]
        row += [_fmt_num(hp.get(name)) for name in hp_order]
        row += [_fmt_num(metrics.get(k)) for k in _MOBO_METRIC_KEYS]
        lines.append(" | ".join(row))
    if len(trials) > max_trials:
        truncated = True

    table = "\n".join(lines)
    prefix = f"{len(trials)} offline trials"
    if truncated:
        prefix += f" ({max_trials} shown)"
    return f"{prefix}:\n\n{table}"


def history_table(rows: List[Dict[str, Any]], injection_iter: int, max_rows: int = 800) -> str:
    """Compact table of measured points up to and including ``injection_iter``."""
    lines = ["iteration | FAPbI3 | MAPbI3 | MAPbBr3 | Objective"]
    shown = 0
    for r in rows:
        if r["Objective"] is None or r["Iteration"] is None or r["Iteration"] > injection_iter:
            continue
        comp = np.array([r["FAPbI3"], r["MAPbI3"], r["MAPbBr3"]], float)
        s = comp.sum()
        comp = comp / (s if s != 0 else 1.0)
        lines.append(f"{r['Iteration']:>3d} | {comp[0]:.3f} | {comp[1]:.3f} | "
                     f"{comp[2]:.3f} | {r['Objective']:.4f}")
        shown += 1
        if shown >= max_rows:
            lines.append(f"... ({shown} rows shown)")
            break
    return "\n".join(lines)


def progress_summary(Xm: np.ndarray, Ym: np.ndarray, its: List[int],
                     injection_iter: int, mapping: Dict[str, Any],
                     needles: List[Dict[str, Any]]) -> str:
    mask = np.array([it != -1 and it <= injection_iter for it in its])
    Yup = Ym[mask]
    Xup = Xm[mask]
    if Yup.size == 0:
        return "No measured points yet."
    best_i = int(np.argmax(Yup))
    best_comp = Xup[best_i]
    return (
        f"- Measured points so far: {int(mask.sum())} (through iteration {injection_iter}).\n"
        f"- Best Objective so far: {Yup.max():.4f} at composition "
        f"FAPbI3={best_comp[0]:.3f}, MAPbI3={best_comp[1]:.3f}, MAPbBr3={best_comp[2]:.3f}.\n"
        f"- Needles (local optima) found so far: {len(needles)}.\n"
        f"- Equal-budget iterations remaining for the continued run: "
        f"{mapping['budget_iterations']}."
    )


def needle_summary(needles: List[Dict[str, Any]]) -> str:
    if not needles:
        return "None yet."
    lines = []
    for i, n in enumerate(needles):
        c = n["composition"]
        at = n["found_at_point_index"]
        lines.append(
            f"- Needle {i + 1}: value {n['value']:.4f} at FAPbI3={c[0]:.3f}, "
            f"MAPbI3={c[1]:.3f}, MAPbBr3={c[2]:.3f}"
            + (f" (found around measured point #{at})." if at is not None else ".")
        )
    return "\n".join(lines)


def build_prompt(rows, Xm, Ym, its, injection_iter, mapping, run_config, needles) -> str:
    hp = current_hparams(run_config)
    return llm_config.PROMPT_TEMPLATE.format(
        system_features=llm_config.SYSTEM_FEATURES,
        hparam_descriptions=hparam_descriptions_block(),
        current_hparams=format_current_hparams(hp),
        hparam_search_history=hparam_optimization_history(),
        injection_iter=injection_iter,
        history_table=history_table(rows, injection_iter),
        progress_summary=progress_summary(Xm, Ym, its, injection_iter, mapping, needles),
        needle_summary=needle_summary(needles),
    )


# ════════════════════════════════════════════════════════════════════════════════
# LLM decision parsing / validation
# ════════════════════════════════════════════════════════════════════════════════

def validate_changes(changes: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Clamp + type-cast proposed hyperparameters to HPARAM_SPACE. Returns
    (validated_dict, warnings)."""
    out: Dict[str, Any] = {}
    warnings: List[str] = []
    for ch in changes or []:
        name = ch.get("name")
        val = ch.get("value")
        if name not in R.HPARAM_SPACE:
            warnings.append(f"ignored unknown hyperparameter '{name}'")
            continue
        lo, hi, tfm = R.HPARAM_SPACE[name]
        try:
            v = float(val)
        except (TypeError, ValueError):
            warnings.append(f"ignored non-numeric value for '{name}': {val!r}")
            continue
        if v < lo or v > hi:
            warnings.append(f"clamped '{name}' from {v} into [{lo}, {hi}]")
            v = min(max(v, lo), hi)
        out[name] = int(round(v)) if tfm == "int" else v
    return out, warnings


# ════════════════════════════════════════════════════════════════════════════════
# Resume + continue ZoMBI-Hop on the RF objective
# ════════════════════════════════════════════════════════════════════════════════

def prepare_resume_dir(ckpt_root: Path, snapshot_name: str,
                       config_overrides: Dict[str, Any]) -> str:
    """Copy run_7eb9 into a scratch checkpoint dir truncated to ``snapshot_name``,
    with config.json patched to carry the LLM's config-read hyperparameters.
    Returns the checkpoint_dir (parent of run_7eb9) as a string.
    """
    ckpt_root.mkdir(parents=True, exist_ok=True)
    dst = ckpt_root / "run_7eb9"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "snapshots").mkdir(parents=True, exist_ok=True)

    # Copy config.json with hyperparameter overrides applied.
    with open(RUN_DIR / "config.json") as f:
        cfg = json.load(f)
    for k, v in config_overrides.items():
        cfg[k] = v
    with open(dst / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Copy snapshots up to and including the injection snapshot.
    inj_idx = snapshot_index(snapshot_name)
    for sd in sorted((RUN_DIR / "snapshots").iterdir()):
        if sd.is_dir() and snapshot_index(sd.name) <= inj_idx:
            shutil.copytree(sd, dst / "snapshots" / sd.name)

    (dst / "latest.txt").write_text(snapshot_name)
    return str(ckpt_root)


def _seed_everything(seed: int) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def continue_run(ckpt_dir: str, fn_callable, ref_optima,
                 constructor_overrides: Dict[str, Any], budget: int,
                 trial_dir: Path, seed: Optional[int] = None) -> Dict[str, Any]:
    """Resume vanilla ZoMBI-Hop from the injection state and run for ``budget``
    additional objective iterations on the RF objective. Writes run_mobo-style
    artifacts into ``trial_dir``. Returns a metrics dict. ``seed`` makes a single
    repeat reproducible while still giving distinct draws across repeats.
    """
    import inspect
    if seed is not None:
        _seed_everything(seed)
    device, dtype = R.DEVICE, R.DTYPE
    dim = 3

    # Objective: identical machinery to a run_mobo trial (LineBO over the RF).
    plot_state: Dict[str, Any] = {"line_0": None, "line_1": None}
    sim_obj = R.make_sim_obj(fn_callable, device, dtype, maximize=MAXIMIZE)
    inner = R.make_linebo_wrapper(sim_obj, dim, R.NUM_LINES, device, dtype, plot_state)

    payloads: List[dict] = []
    snap_records: List[tuple] = []
    call_counter = [0]
    dh_ref = [None]

    def obj_wrapper(x_tell, bounds, acq_fn):
        if call_counter[0] >= budget:            # equal-budget stop
            raise BudgetExhausted()
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

    # Constructor: pass the non-config hyperparameters (config-read ones are baked
    # into the patched config.json and applied by DataHandler.load_state).
    sig = inspect.signature(ZoMBIHop.__init__)
    extra = {k: v for k, v in constructor_overrides.items() if k in sig.parameters}

    dummy_Xa = torch.full((1, dim), 1.0 / dim, device=device, dtype=dtype)
    dummy_Y = torch.zeros(1, 1, device=device, dtype=dtype)

    optimizer = ZoMBIHop(
        objective=obj_wrapper,
        X_init_actual=dummy_Xa, X_init_expected=dummy_Xa.clone(), Y_init=dummy_Y,
        input_noise=R.NOISE_LEVEL, acquisition_type="ucb", max_gp_points=3000,
        device=str(device), dtype=dtype, verbose=False,
        run_uuid="7eb9", resume=True, checkpoint_dir=ckpt_dir,
        **extra,
    )
    dh = optimizer.data_handler
    dh_ref[0] = dh

    # Record cumulative-point / activation / zoom at each snapshot (for points.csv).
    orig_snap = dh.take_snapshot

    def snap_wrap(*a, **k):
        orig_snap(*a, **k)
        if dh.X_all_actual is not None:
            czb = dh.current_zoom_bounds if dh.current_zoom_bounds is not None else dh.bounds
            zoom_size = R.zoom_size_fraction(czb) if czb is not None else 1.0
            snap_records.append((dh.X_all_actual.shape[0], dh.current_activation,
                                 dh.current_zoom, zoom_size))
    dh.take_snapshot = snap_wrap

    t0 = time.time()
    try:
        optimizer.run(max_activations=float("inf"), time_limit_hours=None)
    except BudgetExhausted:
        pass
    except Exception as e:  # match run_single_trial tolerance
        print(f"    [continue] ZoMBI-Hop stopped early: {e}")
    runtime = time.time() - t0
    n_iters = call_counter[0]

    needle_t = dh.get_all_needle_locations()
    discovered = as_numpy(needle_t) if needle_t.numel() > 0 else np.empty((0, dim))
    needle_vals_t = dh.get_all_needle_vals()
    best_needle = (float(as_numpy(needle_vals_t).ravel().max())
                   if needle_vals_t.numel() > 0 else float("nan"))
    X_all = as_numpy(dh.X_all_actual) if dh.X_all_actual is not None else np.empty((0, dim))
    Y_all = as_numpy(dh.Y_all).ravel() if dh.Y_all is not None else np.empty((0,))

    # Best Objective over ALL measured points (including points inside declared-
    # needle penalty regions) = the endpoint of the running-best curve. This matches
    # the real-run baseline (baseline_metrics uses the same max-over-all definition)
    # and the convergence plot, so the significance test is on the same quantity.
    best_obj = float(Y_all.max()) if Y_all.size else float("nan")

    dist = metric_dist_to_needles(discovered, ref_optima, dim=dim) if len(ref_optima) else float("nan")
    dup = metric_dup_fraction(X_all, dim=dim) if X_all.shape[0] else float("nan")

    # run_mobo-style artifacts.
    trial_dir.mkdir(parents=True, exist_ok=True)
    try:
        R.write_points_csv(str(trial_dir / "points.csv"), dh, snap_records, dim=dim)
        R.write_needles_csv(str(trial_dir / "needles.csv"), dh, dim=dim)
        R.write_metrics_over_time_csv(str(trial_dir / "metrics_over_time.csv"),
                                      payloads, X_all, ref_optima, dim=dim)
    except Exception as e:
        print(f"    [continue] CSV write failed: {e}")
    try:
        R.plot_convergence(str(trial_dir / "convergence.png"), dh, MAXIMIZE)
        R.plot_dist_from_centre(str(trial_dir / "dist_from_centre.png"), dh, MAXIMIZE)
        R.plot_line_length_hist(str(trial_dir / "line_length_hist.png"), payloads)
    except Exception as e:
        print(f"    [continue] plot failed: {e}")
    try:
        shutil.copy2(Path(ckpt_dir) / "run_7eb9" / "config.json", trial_dir / "config.json")
    except Exception:
        pass

    return {
        "n_iters": n_iters,
        "runtime_s": runtime,
        "avg_time_per_iter_s": runtime / n_iters if n_iters else 0.0,
        "n_points_total": int(X_all.shape[0]),
        "n_needles": int(discovered.shape[0]),
        "best_objective": best_obj,
        "best_needle": best_needle,
        "dist_to_ref_optima": float(dist),
        "dup_fraction": float(dup),
        "Y_all_running_best": np.maximum.accumulate(Y_all).tolist() if Y_all.size else [],
        "ref_optima": [list(map(float, o)) for o in ref_optima],
    }


# ════════════════════════════════════════════════════════════════════════════════
# Repeats: run one continuation in an isolated temp checkpoint dir
# ════════════════════════════════════════════════════════════════════════════════

# The four scalar metrics aggregated across repeats.
_METRIC_KEYS = ["best_objective", "best_needle", "n_needles", "dist_to_ref_optima", "dup_fraction"]


def one_continuation(snapshot_name: str, fn_callable, ref_optima,
                     config_overrides: Dict[str, Any], constructor_overrides: Dict[str, Any],
                     budget: int, trial_dir: Path, seed: int) -> Dict[str, Any]:
    """Prepare an isolated (temp) truncated checkpoint, resume + run one repeat,
    then clean up the temp checkpoint. Artifacts land in ``trial_dir``."""
    tmp = Path(tempfile.mkdtemp(prefix="zombi_ckpt_"))
    try:
        ckpt = prepare_resume_dir(tmp, snapshot_name, config_overrides)
        return continue_run(ckpt, fn_callable, ref_optima, constructor_overrides,
                            budget, trial_dir, seed=seed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def aggregate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean/std/n and raw values for each scalar metric across repeats (nan-aware)."""
    out: Dict[str, Any] = {}
    for k in _METRIC_KEYS:
        vals = np.array([float(s.get(k, np.nan)) for s in samples], dtype=float)
        finite = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "n": int(finite.size),
            "values": [None if not np.isfinite(v) else float(v) for v in vals],
        }
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Baseline (the real campaign2 trajectory) metrics
# ════════════════════════════════════════════════════════════════════════════════

def baseline_metrics(Xm: np.ndarray, Ym: np.ndarray, mapping: Dict[str, Any],
                     ref_optima) -> Dict[str, Any]:
    n_inj = mapping["n_points_at_injection"]
    running_best = np.maximum.accumulate(Ym) if Ym.size else np.array([])
    best_at_injection = float(running_best[min(n_inj, len(running_best)) - 1]) if len(running_best) else float("nan")
    final_best = float(running_best[-1]) if len(running_best) else float("nan")

    # Final real-run needles (full run_7eb9).
    final_needles = needles_at_snapshot(_default_final_snapshot())
    disc = np.array([n["composition"] for n in final_needles], float) if final_needles else np.empty((0, 3))
    needle_vals = np.array([n.get("value", np.nan) for n in final_needles], float)
    best_needle = (float(np.nanmax(needle_vals))
                   if needle_vals.size and np.isfinite(needle_vals).any() else float("nan"))
    dist = metric_dist_to_needles(disc, ref_optima, dim=3) if len(ref_optima) and len(disc) else float("nan")
    dup = metric_dup_fraction(Xm, dim=3) if Xm.shape[0] else float("nan")

    return {
        "source": "campaign2_db_real",
        "n_points_total": int(Xm.shape[0]),
        "n_points_at_injection": int(n_inj),
        "best_objective_at_injection": best_at_injection,
        # Keys aligned with continue_run()'s metrics so this counts as a sample.
        "best_objective": final_best,
        "best_needle": best_needle,
        "n_needles": int(disc.shape[0]),
        "dist_to_ref_optima": float(dist),
        "dup_fraction": float(dup),
        "running_best": running_best.tolist(),
    }


def _default_final_snapshot() -> str:
    latest = RUN_DIR / "latest.txt"
    if latest.exists():
        nm = latest.read_text().strip()
        if nm:
            return nm
    snaps = sorted(p.name for p in (RUN_DIR / "snapshots").iterdir() if p.is_dir())
    return snaps[-1]


# ════════════════════════════════════════════════════════════════════════════════
# Comparison plot
# ════════════════════════════════════════════════════════════════════════════════

def plot_comparison(out_png: Path, base_db: Dict[str, Any],
                    baseline_rf_samples: List[Dict[str, Any]],
                    llm_samples: List[Dict[str, Any]], mapping: Dict[str, Any]) -> None:
    """Running-best objective vs measured points: the real campaign2 baseline,
    every original-hyperparameter RF repeat, and every LLM-hyperparameter RF
    repeat, with the injection point marked."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    n_inj = mapping["n_points_at_injection"]

    base_rb = np.asarray(base_db["running_best"], float)
    ax.plot(np.arange(len(base_rb)), base_rb, color="steelblue", lw=2.2,
            label="baseline: real campaign2", zorder=5)

    for i, s in enumerate(baseline_rf_samples):
        rb = np.asarray(s.get("Y_all_running_best", []), float)
        if rb.size:
            ax.plot(np.arange(len(rb)), rb, color="steelblue", lw=1.0, alpha=0.35,
                    zorder=3, label="baseline: original HPs on RF" if i == 0 else None)
    for i, s in enumerate(llm_samples):
        rb = np.asarray(s.get("Y_all_running_best", []), float)
        if rb.size:
            ax.plot(np.arange(len(rb)), rb, color="darkorange", lw=1.0, alpha=0.5,
                    zorder=4, label="LLM HPs on RF" if i == 0 else None)

    ax.axvline(n_inj, color="crimson", ls="--", lw=1.0, alpha=0.7,
               label=f"injection (iter {mapping['injection_iter']}, ~{n_inj} pts)")
    ax.set_xlabel("cumulative measured points")
    ax.set_ylabel("running-best Objective")
    ax.set_title("Baseline vs LLM-tuned ZoMBI-Hop (repeats overlaid)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
# Orchestration
# ════════════════════════════════════════════════════════════════════════════════

def _sample_scalars(s: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the fields worth serialising per repeat (drop the big curve)."""
    keep = _METRIC_KEYS + ["n_iters", "runtime_s", "n_points_total", "source"]
    return {k: s[k] for k in keep if k in s}


def run_evaluation(injection_iter: int, out_root: Optional[Path] = None,
                   make_plots: bool = True,
                   n_llm_repeats: int = N_LLM_REPEATS,
                   n_baseline_repeats: int = N_BASELINE_REPEATS) -> Dict[str, Any]:
    """Full pipeline for one injection point. The LLM is called ONCE; the
    continuation is repeated ``n_llm_repeats`` times (LLM hyperparameters) and the
    baseline is estimated from ``n_baseline_repeats`` samples — the real campaign2
    run plus (n_baseline_repeats - 1) RF continuations with the ORIGINAL
    (run_7eb9 / trial_112) hyperparameters. Mean/std are reported across repeats.
    Writes all artifacts under ``out_root`` (default: llm/results/inj_XXX_<ts>/)."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root or (RESULTS_ROOT / f"inj_{injection_iter:03d}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\nLLM tuning evaluation — injection iteration {injection_iter}\n{'='*70}")
    print(f"  output dir: {out_dir}")

    # 1. Data + RF objective.
    rows = load_db_rows()
    Xm, Ym, its = measured_points(rows)
    rf, fn_callable, grid_pts, grid_vals = build_rf(Xm, Ym)
    ref_optima = R.auto_detect_rf_optima(rf, grid_pts, grid_vals, maximize=MAXIMIZE,
                                         n_peaks=N_REF_OPTIMA)

    # 2. State mapping + run config.
    snaps = scan_snapshots()
    mapping = map_injection(injection_iter, rows, snaps)
    with open(RUN_DIR / "config.json") as f:
        run_config = json.load(f)
    needles = needles_at_snapshot(mapping["snapshot_name"])
    budget = mapping["budget_iterations"]
    snap = mapping["snapshot_name"]
    print(f"  injection snapshot: {snap} (~{mapping['snapshot_n_points']} pts); "
          f"equal budget = {budget} iterations")

    # 3. LLM call (exactly once).
    prompt = build_prompt(rows, Xm, Ym, its, injection_iter, mapping, run_config, needles)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"  calling {llm_config.MODEL} (effort={llm_config.EFFORT}) …")
    llm_out = llm_config.call_llm(prompt, R.HPARAM_NAMES)
    print(f"  LLM responded in {llm_out['latency_s']:.2f}s"
          + (f"  [ERROR: {llm_out['error']}]" if llm_out["error"] else ""))

    decision = llm_out.get("decision") or {}
    raw_changes = decision.get("hyperparameter_changes", []) if isinstance(decision, dict) else []
    changes, warns = validate_changes(raw_changes)
    for w in warns:
        print(f"    [validate] {w}")
    changed = len(changes) > 0

    decision_record = {
        "injection_iter": injection_iter,
        "model": llm_out["model"],
        "effort": llm_out["effort"],
        "latency_s": llm_out["latency_s"],
        "usage": llm_out["usage"],
        "error": llm_out["error"],
        "reasoning": decision.get("reasoning") if isinstance(decision, dict) else None,
        "raw_hyperparameter_changes": raw_changes,
        "validated_changes": changes,
        "validation_warnings": warns,
        "changed_hyperparameters": changed,
        "raw_response": llm_out["raw_text"],
    }
    (out_dir / "llm_decision.json").write_text(json.dumps(decision_record, indent=2))
    print(f"  LLM chose to {'CHANGE' if changed else 'KEEP'} hyperparameters"
          + (f": {changes}" if changed else ""))
    if decision_record["reasoning"]:
        print(f"  reasoning: {decision_record['reasoning']}")

    # 4. Baseline sample #1: the real campaign2 trajectory.
    base_db = baseline_metrics(Xm, Ym, mapping, ref_optima)

    # 5. Baseline samples #2..N: RF continuations with the ORIGINAL hyperparameters.
    baseline_rf: List[Dict[str, Any]] = []
    n_rf_baseline = max(0, n_baseline_repeats - 1)
    for rep in range(n_rf_baseline):
        seed = injection_iter * 10_000 + 1_000 + rep
        print(f"  [baseline RF rep {rep + 1}/{n_rf_baseline}] original HPs, "
              f"budget {budget}, seed {seed} …")
        m = one_continuation(snap, fn_callable, ref_optima, {}, {}, budget,
                             out_dir / "baseline_rf" / f"rep{rep}", seed)
        m["source"] = "rf_original_hparams"
        baseline_rf.append(m)
        print(f"      best={m['best_objective']:.4f}, needles={m['n_needles']}, "
              f"dup={m['dup_fraction']:.4f}")

    baseline_samples = [base_db] + baseline_rf
    baseline_stats = aggregate_samples(baseline_samples)

    # 6. LLM continuation samples (only if the LLM changed something).
    llm_samples: List[Dict[str, Any]] = []
    if changed:
        config_overrides = {k: v for k, v in changes.items() if k in _CONFIG_READ_HPARAMS}
        constructor_overrides = {k: v for k, v in changes.items() if k in _CONSTRUCTOR_HPARAMS}
        for rep in range(n_llm_repeats):
            seed = injection_iter * 10_000 + 2_000 + rep
            print(f"  [LLM rep {rep + 1}/{n_llm_repeats}] new HPs, budget {budget}, "
                  f"seed {seed} …")
            m = one_continuation(snap, fn_callable, ref_optima, config_overrides,
                                 constructor_overrides, budget,
                                 out_dir / "continuation" / f"rep{rep}", seed)
            m["source"] = "rf_llm_hparams"
            llm_samples.append(m)
            print(f"      best={m['best_objective']:.4f}, needles={m['n_needles']}, "
                  f"dup={m['dup_fraction']:.4f}")
    else:
        print("  no change → no LLM rerun (outcome equals the original hyperparameters).")

    llm_stats = aggregate_samples(llm_samples) if llm_samples else None

    # 7. Diff (LLM mean − baseline mean) + summary.
    diff = None
    if llm_stats is not None:
        diff = {k: llm_stats[k]["mean"] - baseline_stats[k]["mean"] for k in _METRIC_KEYS}

    comparison = {
        "injection_iter": injection_iter,
        "mapping": mapping,
        "n_llm_repeats": len(llm_samples),
        "n_baseline_repeats": len(baseline_samples),
        "llm_decision": {k: decision_record[k] for k in
                         ("model", "effort", "latency_s", "changed_hyperparameters",
                          "validated_changes", "reasoning")},
        "baseline_stats": baseline_stats,
        "baseline_samples": [_sample_scalars(s) for s in baseline_samples],
        "llm_stats": llm_stats,
        "llm_samples": [_sample_scalars(s) for s in llm_samples],
        "difference_llm_minus_baseline_mean": diff,
    }
    (out_dir / "baseline_vs_llm.json").write_text(json.dumps(comparison, indent=2))

    if make_plots:
        try:
            plot_comparison(out_dir / "comparison.png", base_db, baseline_rf,
                            llm_samples, mapping)
        except Exception as e:
            print(f"  comparison plot failed: {e}")

    b = baseline_stats["best_objective"]
    print(f"  baseline best objective: {b['mean']:.4f} ± {b['std']:.4f} (n={b['n']})")
    if llm_stats is not None:
        l = llm_stats["best_objective"]
        print(f"  LLM best objective:      {l['mean']:.4f} ± {l['std']:.4f} (n={l['n']})")
        print(f"  Δ(best objective) = {diff['best_objective']:+.4f}, "
              f"Δ(needles) = {diff['n_needles']:+.2f}, "
              f"Δ(dup) = {diff['dup_fraction']:+.4f}")
    print(f"  wrote summary → {out_dir / 'baseline_vs_llm.json'}")

    comparison["out_dir"] = str(out_dir)
    return comparison


def main() -> None:
    run_evaluation(INJECTION_ITER)


if __name__ == "__main__":
    main()
