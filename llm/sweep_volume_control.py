"""
llm/sweep_volume_control.py
===========================
Volume-control variant of ``sweep_basic_surrogate.py``.

Instead of letting the LLM re-tune ZoMBI-Hop's *hyperparameters* on the fly, this
experiment gives the LLM direct control over the **acquisition landscape**: every
1 / 5 / 10 ZoMBI-Hop iterations the LLM is prompted to place **reward** and/or
**penalization volumes** (hyperspheres on the composition simplex). It sees the
same rich surrogate features as ``sweep_basic_surrogate.py`` and decides where to
steer — or nudge away from — the search.

How the volumes work
--------------------
The volumes are added on top of ZoMBI-Hop's existing acquisition (its
``RepulsiveAcquisition``; see ``src/utils/gp_simplex.py``). For a volume centred at
``c`` with radius ``r`` and a query point ``x`` at Euclidean distance
``dist = ||x - c||``:

    violation = max(0, r - dist)          # smoothed: 0 at the edge, r at the centre
    term      = strength * violation**2

* **Penalization volume** — SUBTRACTS ``term`` from the acquisition, exactly like a
  needle penalty ellipsoid, but constrained to a hypersphere (isotropic) so the LLM
  can reason about it as "a ball of radius r at composition c". The penalty is
  smoothed up to the edge of the ball (it decays to 0 at ``dist == r``), so it does
  NOT hard-exclude the region — it just makes it less attractive.
* **Reward volume** — ADDS ``term`` instead, making the algorithm MORE likely to
  sample inside the ball.

``strength`` is tied to ZoMBI-Hop's own auto-computed ``repulsion_lambda`` (the same
scale it uses to penalize needles), times ``VOLUME_STRENGTH_MULT``, so a volume is
comparable in force to a needle penalty.

The LLM only chooses, per volume, its ``kind`` (reward / penalty), ``center`` (a
composition), and ``radius``. Each injection it specifies the COMPLETE set of
volumes that should be in effect going forward — the returned list REPLACES the
previous set (so it can add, remove, resize, or move volumes freely). Every
injection prompt shows the LLM the volumes currently in effect; to leave the
landscape unchanged it re-lists them, and to clear the landscape it returns an
empty list.

Everything else mirrors ``sweep_basic_surrogate.py``: same generative surrogate,
same cold-start → inject-every-k → resume-exact-state continuation, same
common-random-numbers baseline (``trial_112`` hyperparameters, NO volumes, NO LLM),
same output layout and Welch significance test.

Usage:
  # repo-root uv venv (see MEMORY.md), NOT `conda activate zombi-hop`
  python llm/sweep_volume_control.py

The model / effort come from llm/llm_config.py (shared with evaluate_llm).
"""

from __future__ import annotations

# ─── HARDCODED CONFIG ──────────────────────────────────────────────────────────
INJECTION_INTERVALS: list[int] = [1, 5, 10]   # LLM places volumes every k iterations
MAX_ITERS: int = 40                            # total ZoMBI-Hop iterations per trial
N_REPEATS: int = 3                             # trials per group (variance)
SURROGATE_PICKLE: str | None = None            # reuse a fitted surrogate if set

# Volume-control knobs.
VOLUME_STRENGTH_MULT: float = 1.0   # volume force = this × ZoMBI-Hop's repulsion_lambda
MIN_VOLUME_RADIUS: float = 0.02     # composition-distance units (≈ input noise scale)
MAX_VOLUME_RADIUS: float = 0.60     # keep a single ball from covering the whole simplex
MAX_VOLUMES_PER_INJECTION: int = 8  # guard against a runaway list in one call
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

# Reuse all of sweep_basic_surrogate's data + ZoMBI plumbing (which in turn reuses
# evaluate_llm / run_mobo / surrogate). We only swap "tune hyperparameters" for
# "place volumes".
import evaluate_llm as E  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import run_mobo as R  # noqa: E402
import llm_config  # noqa: E402
import sweep_basic_no_surrogate as SW  # noqa: E402  (welch_significance)
import sweep_basic_surrogate as SBS  # noqa: E402  (surrogate objective + summaries)
from eval_metrics import as_numpy  # noqa: E402
from src.core.zombihop import ZoMBIHop  # noqa: E402
from src.utils.simplex import proj_simplex  # noqa: E402

MAXIMIZE = SBS.MAXIMIZE
N_REF_OPTIMA = SBS.N_REF_OPTIMA
SIG_ALPHA = SBS.SIG_ALPHA
DIM = 3

_METRIC_KEYS = SBS._METRIC_KEYS


# ════════════════════════════════════════════════════════════════════════════════
# Reward / penalty volumes: acquisition wrapper + installer
# ════════════════════════════════════════════════════════════════════════════════

class VolumeControlAcquisition(nn.Module):
    """Wrap ZoMBI-Hop's acquisition and add the LLM's reward/penalty hyperspheres.

    For each volume (centre ``c``, radius ``r``, sign ``s`` = +1 reward / −1 penalty):
        violation = max(0, r − ||x − c||)     # smoothed to 0 at the ball's edge
        acq(x)   += s · strength · violation²

    ``strength`` is ZoMBI-Hop's own ``repulsion_lambda`` (matched to how hard it
    penalizes needles) scaled by ``strength_mult``. ``.base`` is exposed as the wrapped
    acquisition's own clean base so ``determine_penalty_ellipsoid`` (which reads
    ``acq_fn.base`` for uncontaminated needle curvature) is unaffected by the volumes.
    """

    def __init__(self, wrapped: nn.Module, proj_fn, volumes: List[Dict[str, Any]],
                 strength_mult: float, device, dtype):
        super().__init__()
        self.wrapped = wrapped
        # Clean base for curvature estimates (fall through the wrapper chain).
        self.base = getattr(wrapped, "base", wrapped)
        self.proj_fn = proj_fn
        base_lambda = float(getattr(wrapped, "repulsion_lambda", 1000.0) or 1000.0)
        self.strength = base_lambda * float(strength_mult)

        centers = torch.tensor([v["center"] for v in volumes], device=device, dtype=dtype)
        self.register_buffer("centers", centers)                                  # (K, d)
        self.register_buffer(
            "radii", torch.tensor([v["radius"] for v in volumes], device=device, dtype=dtype))
        self.register_buffer(
            "signs",
            torch.tensor([1.0 if v["kind"] == "reward" else -1.0 for v in volumes],
                         device=device, dtype=dtype))

    def forward(self, Xq: torch.Tensor) -> torch.Tensor:
        base_val = self.wrapped(Xq)
        if self.centers.shape[0] == 0:
            return base_val

        X_proj = self.proj_fn(Xq)
        X_flat = X_proj.reshape(-1, X_proj.shape[-1])                             # (B, d)

        # (B, K) Euclidean distance to every volume centre.
        dist = torch.cdist(X_flat, self.centers)                                 # (B, K)
        violation = torch.clamp(self.radii.unsqueeze(0) - dist, min=0.0)         # (B, K)
        contrib = (self.signs.unsqueeze(0) * violation ** 2).sum(dim=1)          # (B,)
        extra = self.strength * contrib
        return base_val + extra.view(base_val.shape)


def install_volume_control(optimizer: ZoMBIHop, volumes: List[Dict[str, Any]],
                           strength_mult: float) -> None:
    """Monkeypatch the optimizer's GP handler so every acquisition it builds is
    wrapped with the (shared, mutable) ``volumes`` list. Because ``volumes`` is
    captured by reference, later injections that append to it take effect on the
    next iteration's acquisition without re-patching."""
    gp = optimizer.gp_handler
    orig_create = gp.create_acquisition

    def patched_create(*args, **kwargs):
        acq = orig_create(*args, **kwargs)   # RepulsiveAcquisition; also sets gp.acq_fn
        if not volumes:
            return acq
        wrapped = VolumeControlAcquisition(
            acq, gp.proj_fn, volumes, strength_mult, device=gp.device, dtype=gp.dtype)
        gp.acq_fn = wrapped
        return wrapped

    gp.create_acquisition = patched_create


# ════════════════════════════════════════════════════════════════════════════════
# LLM volume decision: schema, call, validation
# ════════════════════════════════════════════════════════════════════════════════

def build_volume_schema(dim: int = DIM) -> Dict[str, Any]:
    """JSON schema constraining the LLM's answer to a list of typed hyperspheres."""
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "volumes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["reward", "penalty"]},
                        # NOTE: the Anthropic structured-output API rejects array
                        # minItems/maxItems other than 0 or 1, so the exact length
                        # (== dim) is enforced downstream in validate_volumes()
                        # rather than in the schema.
                        "center": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "radius": {"type": "number"},
                    },
                    "required": ["kind", "center", "radius"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reasoning", "volumes"],
        "additionalProperties": False,
    }


VOLUME_SYSTEM_PROMPT = (
    "You are an expert Bayesian-optimization engineer steering the ZoMBI-Hop "
    "algorithm on the fly by reshaping its acquisition landscape with reward and "
    "penalization volumes. You are precise, quantitative, and deliberate: you only "
    "place a volume when the run history gives you a concrete reason. You always "
    "answer through the provided structured-output schema."
)


def call_volume_llm(user_prompt: str, dim: int = DIM) -> Dict[str, Any]:
    """Single volume-decision call. Mirrors ``llm_config.call_llm`` but returns a
    ``{"reasoning", "volumes": [{"kind","center","radius"}, ...]}`` decision."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The `anthropic` package is required. Install it with "
            "`pip install anthropic` and set ANTHROPIC_API_KEY."
        ) from e

    client = anthropic.Anthropic()
    output_config: Dict[str, Any] = {
        "effort": llm_config.EFFORT,
        "format": {"type": "json_schema", "schema": build_volume_schema(dim)},
    }
    kwargs: Dict[str, Any] = dict(
        model=llm_config.MODEL,
        max_tokens=llm_config.MAX_TOKENS,
        system=VOLUME_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config=output_config,
    )
    if llm_config.USE_THINKING:
        kwargs["thinking"] = {"type": "adaptive"}

    t0 = time.time()
    error, raw_text, decision, usage = None, "", None, {}
    try:
        resp = client.messages.create(**kwargs)
        latency_s = time.time() - t0
        raw_text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            usage = (resp.usage.model_dump() if hasattr(resp.usage, "model_dump")
                     else dict(resp.usage))
        except Exception:
            usage = {}
        try:
            decision = json.loads(raw_text)
        except json.JSONDecodeError as e:
            error = f"could not parse model JSON: {e}"
    except Exception as e:
        latency_s = time.time() - t0
        error = f"LLM call failed: {e}"

    return {
        "decision": decision, "latency_s": latency_s, "raw_text": raw_text,
        "model": llm_config.MODEL, "effort": llm_config.EFFORT,
        "usage": usage, "error": error,
    }


def validate_volumes(raw_volumes: List[Dict[str, Any]], dim: int = DIM
                     ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Sanitize the LLM's proposed volumes: project each centre onto the simplex,
    clamp radii into [MIN_VOLUME_RADIUS, MAX_VOLUME_RADIUS], drop malformed entries,
    and cap the count. Returns (validated, warnings)."""
    out: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for i, v in enumerate(raw_volumes or []):
        if len(out) >= MAX_VOLUMES_PER_INJECTION:
            warnings.append(f"dropped volume #{i}: exceeds cap of {MAX_VOLUMES_PER_INJECTION}")
            continue
        kind = v.get("kind")
        if kind not in ("reward", "penalty"):
            warnings.append(f"dropped volume #{i}: bad kind {kind!r}")
            continue
        center = v.get("center")
        try:
            c = np.asarray(center, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            warnings.append(f"dropped volume #{i}: non-numeric center {center!r}")
            continue
        if c.shape[0] != dim or not np.all(np.isfinite(c)):
            warnings.append(f"dropped volume #{i}: center must be {dim} finite numbers")
            continue
        # Project onto the simplex (clamp negatives, renormalise to sum 1).
        c_proj = proj_simplex(torch.tensor(c).unsqueeze(0)).squeeze(0).numpy()
        try:
            r = float(v.get("radius"))
        except (TypeError, ValueError):
            warnings.append(f"dropped volume #{i}: non-numeric radius {v.get('radius')!r}")
            continue
        if not np.isfinite(r) or r <= 0:
            warnings.append(f"dropped volume #{i}: radius must be positive")
            continue
        r_clamped = float(min(max(r, MIN_VOLUME_RADIUS), MAX_VOLUME_RADIUS))
        if r_clamped != r:
            warnings.append(
                f"clamped volume #{i} radius {r} → {r_clamped} "
                f"[{MIN_VOLUME_RADIUS}, {MAX_VOLUME_RADIUS}]")
        out.append({"kind": kind, "center": [float(x) for x in c_proj],
                    "radius": r_clamped})
    return out, warnings


# ════════════════════════════════════════════════════════════════════════════════
# Injection prompt (volume-placement task on the surrogate)
# ════════════════════════════════════════════════════════════════════════════════

VOLUME_PROMPT_TEMPLATE = """\
You are steering a Bayesian-optimization algorithm, ZoMBI-Hop, on the fly. It is
optimizing a perovskite-materials-discovery lab over the 3-simplex composition
(`FAPbI3`, `MAPbI3`, `MAPbBr3`, which always sum to 1); the `Objective` is
MAXIMIZED. ZoMBI-Hop is a zooming multi-basin optimizer that hunts MULTIPLE optima
("needles"): it fits a GP, uses an acquisition function each iteration to pick a
LineBO line of ~24 measured droplets, declares a needle when expected improvement
hits the output-noise floor, penalizes that region, and moves on.

Unlike a normal run, every measured droplet here also reports the rich physical
features the lab records behind each `Objective` value. The feature groups are:

{system_features}

## How you steer it: reward and penalization volumes

Instead of tuning hyperparameters, you reshape ZoMBI-Hop's acquisition landscape by
placing **volumes** — hyperspheres on the composition simplex. Each volume has a
`center` (a composition, 3 numbers that will be renormalized to sum to 1) and a
`radius` (in composition-distance units; the simplex spans distances up to ~1.4,
the input/measurement noise scale is ~0.06, and a radius near {max_radius} already
covers a large fraction of the space). Radii are clamped to
[{min_radius}, {max_radius}].

For a query composition `x` at Euclidean distance `d = ||x - center||`, a volume
contributes `strength * max(0, radius - d)**2` to the acquisition — a smooth bump
that is strongest at the center and decays to exactly zero at the edge of the ball:

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

Specify the COMPLETE set of reward and/or penalization volumes that should be in
effect for the NEXT {interval} iteration(s). ZoMBI-Hop will continue from its exact
current internal state (same data, needles, zoom bounds); the list you return
REPLACES the volumes currently in effect (shown above). Think about the measured
points and their supplemental features: is there a promising, under-sampled region
worth a `reward` volume? Is the run wasting budget re-measuring a picked-over or
physically poor region (e.g. high `Objective` but bad stability/bandgap) that
deserves a `penalty` volume? Are any existing volumes no longer useful and worth
dropping or moving? Then answer through the schema:

- `reasoning`: a concise, quantitative justification (2-6 sentences) grounded in the
  specific numbers above (name the compositions and radii you chose and why).
- `volumes`: the FULL list of {{"kind", "center", "radius"}} entries to be active
  now (not just newly added ones). `kind` is `"reward"` or `"penalty"`, `center` is
  a 3-number composition, `radius` is within [{min_radius}, {max_radius}]. To keep
  the current landscape unchanged, re-list exactly the volumes shown above. Return an
  EMPTY list to clear all volumes (a valid, deliberate choice — e.g. to hand full
  control back to ZoMBI-Hop's own acquisition).
"""


def format_active_volumes(volumes: List[Dict[str, Any]]) -> str:
    if not volumes:
        return "(none — the acquisition landscape is currently unmodified)"
    lines = ["kind | center (FAPbI3, MAPbI3, MAPbBr3) | radius"]
    for v in volumes:
        c = v["center"]
        lines.append(f"{v['kind']} | ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) | {v['radius']:.3f}")
    return "\n".join(lines)


def build_injection_prompt(feature_log, dh, volumes, injection_idx, iters_done,
                           budget, interval) -> str:
    return VOLUME_PROMPT_TEMPLATE.format(
        system_features=llm_config.SYSTEM_FEATURES,
        active_volumes=format_active_volumes(volumes),
        injection_idx=injection_idx,
        iters_done=iters_done,
        budget=budget,
        progress_summary=SBS.progress_summary(feature_log, dh, iters_done, budget),
        history_table=SBS.recent_history_table(feature_log),
        n_points=len(feature_log),
        supplemental_summary=SBS.supplemental_summary(feature_log),
        best_point_summary=SBS.best_point_summary(feature_log),
        interval=interval,
        min_radius=MIN_VOLUME_RADIUS,
        max_radius=MAX_VOLUME_RADIUS,
    )


# ════════════════════════════════════════════════════════════════════════════════
# One ZoMBI-Hop segment (cold-start or resume) with volume control installed
# ════════════════════════════════════════════════════════════════════════════════

def run_segment(ckpt_dir: Path, run_uuid: str, fresh: bool, hp: Dict[str, Any],
                volumes: List[Dict[str, Any]], fn_callable, stop_at: int,
                call_counter: List[int], payloads: List[dict], snap_records: List[tuple]):
    """Run ZoMBI-Hop until the global objective-call counter reaches ``stop_at``,
    with the LLM's reward/penalty ``volumes`` applied to the acquisition. Mirrors
    ``sweep_basic_surrogate.run_segment`` (fresh cold-start vs. resume-exact-state),
    but the hyperparameters ``hp`` stay fixed at trial_112 — the LLM steers via
    ``volumes`` instead."""
    run_dir = ckpt_dir / f"run_{run_uuid}"
    constructor_hp = {k: v for k, v in hp.items() if k in SBS._CONSTRUCTOR}

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
    inner = R.make_linebo_wrapper(sim_obj, DIM, R.NUM_LINES, R.DEVICE, R.DTYPE, plot_state)
    dh_ref: List[Any] = [None]

    if fresh:
        X_a, X_e, Y = R._gen_init_data(fn_callable, MAXIMIZE, dim=DIM)
        optimizer = ZoMBIHop(
            objective=obj_wrapper, X_init_actual=X_a, X_init_expected=X_e, Y_init=Y,
            **R.ZOMBI_FIXED, **hp, device=str(R.DEVICE), dtype=R.DTYPE,
            run_uuid=run_uuid, checkpoint_dir=str(ckpt_dir), resume=False,
        )
    else:
        SBS._patch_config(run_dir, hp)
        dummy_Xa = torch.full((1, DIM), 1.0 / DIM, device=R.DEVICE, dtype=R.DTYPE)
        dummy_Y = torch.zeros(1, 1, device=R.DEVICE, dtype=R.DTYPE)
        optimizer = ZoMBIHop(
            objective=obj_wrapper, X_init_actual=dummy_Xa,
            X_init_expected=dummy_Xa.clone(), Y_init=dummy_Y,
            input_noise=R.NOISE_LEVEL, acquisition_type="ucb", max_gp_points=3000,
            device=str(R.DEVICE), dtype=R.DTYPE, verbose=False,
            run_uuid=run_uuid, resume=True, checkpoint_dir=str(ckpt_dir),
            **constructor_hp,
        )

    # Reshape the acquisition landscape with whatever volumes are in effect.
    install_volume_control(optimizer, volumes, VOLUME_STRENGTH_MULT)

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
# One trial: LLM volume-control run, or baseline
# ════════════════════════════════════════════════════════════════════════════════

def run_baseline_trial(surr, base_hp, ref_optima, seed: int, trial_dir: Path) -> Dict[str, Any]:
    """trial_112 hyperparameters, whole budget in one cold-started run, NO volumes,
    NO LLM — identical to sweep_basic_surrogate's baseline (empty volume list)."""
    E._seed_everything(seed)
    rng = np.random.default_rng(seed)
    fn_callable, feature_log = SBS.make_surrogate_objective(surr, rng)
    tmp = Path(tempfile.mkdtemp(prefix="zombi_vol_"))
    payloads: List[dict] = []
    snap_records: List[tuple] = []
    call_counter = [0]
    try:
        dh = run_segment(tmp, "base", True, base_hp, [], fn_callable, MAX_ITERS,
                         call_counter, payloads, snap_records)
        metrics = SBS.finalize_trial(dh, ref_optima, payloads, snap_records, trial_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    metrics["source"] = "baseline_trial112"
    metrics["n_injections"] = 0
    metrics["n_volumes_final"] = 0
    metrics["n_volumes_total"] = 0
    metrics["n_reward_final"] = 0
    metrics["n_penalty_final"] = 0
    return metrics


def run_llm_trial(surr, base_hp, ref_optima, interval: int, seed: int,
                  trial_dir: Path) -> Dict[str, Any]:
    """One cadence-``interval`` trial: cold-start, then every ``interval`` iterations
    ask the LLM for the COMPLETE set of reward/penalty volumes, resuming ZoMBI-Hop's
    exact state each time with that set applied to its acquisition. The returned set
    replaces the previous one, so the LLM has full control every injection."""
    E._seed_everything(seed)
    rng = np.random.default_rng(seed)
    fn_callable, feature_log = SBS.make_surrogate_objective(surr, rng)
    tmp = Path(tempfile.mkdtemp(prefix="zombi_vol_"))
    inj_dir = trial_dir / "injections"
    payloads: List[dict] = []
    snap_records: List[tuple] = []
    call_counter = [0]
    volumes: List[Dict[str, Any]] = []          # the active set (replaced each injection)
    total_volumes_placed = 0                     # cumulative count across all injections
    injections: List[Dict[str, Any]] = []

    try:
        fresh = True
        injection_idx = 0
        while call_counter[0] < MAX_ITERS:
            stop_at = min(call_counter[0] + interval, MAX_ITERS)
            dh = run_segment(tmp, "llm", fresh, base_hp, volumes, fn_callable, stop_at,
                             call_counter, payloads, snap_records)
            fresh = False
            if call_counter[0] >= MAX_ITERS:
                break  # budget spent → no injection after the final segment

            prompt = build_injection_prompt(feature_log, dh, volumes, injection_idx,
                                            call_counter[0], MAX_ITERS, interval)
            this_dir = inj_dir / f"inj_{injection_idx:02d}"
            this_dir.mkdir(parents=True, exist_ok=True)
            (this_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            llm_out = call_volume_llm(prompt)
            decision = llm_out.get("decision") or {}
            raw_volumes = decision.get("volumes", []) if isinstance(decision, dict) else []
            new_set, warns = validate_volumes(raw_volumes)
            # Replace the active set in place (shared list → next segment sees it live).
            volumes[:] = new_set
            total_volumes_placed += len(new_set)

            rec = {
                "injection_idx": injection_idx,
                "iters_done": call_counter[0],
                "latency_s": llm_out["latency_s"],
                "error": llm_out["error"],
                "reasoning": decision.get("reasoning") if isinstance(decision, dict) else None,
                "active_volumes": list(volumes),
                "validation_warnings": warns,
            }
            injections.append(rec)
            (this_dir / "decision.json").write_text(json.dumps(rec, indent=2))
            n_set = len(new_set)
            summary = (f"set {n_set} volume(s) "
                       f"[{sum(v['kind']=='reward' for v in new_set)}R/"
                       f"{sum(v['kind']=='penalty' for v in new_set)}P]") if n_set else "CLEAR"
            print(f"      inj {injection_idx} @ iter {call_counter[0]}: {summary} "
                  f"({llm_out['latency_s']:.1f}s)"
                  + (f"  [ERROR: {llm_out['error']}]" if llm_out["error"] else ""))
            injection_idx += 1

        metrics = SBS.finalize_trial(dh, ref_optima, payloads, snap_records, trial_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    (trial_dir / "injections.json").write_text(json.dumps(injections, indent=2))
    metrics["source"] = f"llm_every_{interval}"
    metrics["n_injections"] = len(injections)
    metrics["n_volumes_final"] = len(volumes)   # active set at the end of the run
    metrics["n_volumes_total"] = total_volumes_placed  # summed over all injections
    metrics["n_reward_final"] = sum(v["kind"] == "reward" for v in volumes)
    metrics["n_penalty_final"] = sum(v["kind"] == "penalty" for v in volumes)
    return metrics


# ════════════════════════════════════════════════════════════════════════════════
# Aggregation + summary
# ════════════════════════════════════════════════════════════════════════════════

def _group_row(group: str, interval: Optional[int], stats: Dict[str, Any],
               samples: List[Dict[str, Any]], baseline_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    def m(k, f):
        return (stats.get(k) or {}).get(f)

    def mean_of(key):
        return float(np.mean([s.get(key, 0) for s in samples])) if samples else None

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
        "n_injections_mean": mean_of("n_injections"),
        "n_volumes_final_mean": mean_of("n_volumes_final"),
        "n_volumes_total_mean": mean_of("n_volumes_total"),
        "n_reward_final_mean": mean_of("n_reward_final"),
        "n_penalty_final_mean": mean_of("n_penalty_final"),
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
                   "dist_mean", "dup_mean", "n_injections_mean",
                   "n_volumes_final_mean", "n_volumes_total_mean",
                   "n_reward_final_mean", "n_penalty_final_mean",
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
    sweep_dir = E.RESULTS_ROOT / f"sweep_volume_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Volume-control LLM-in-the-loop sweep\n  sweep dir: {sweep_dir}")
    print(f"  cadences: {INJECTION_INTERVALS}   budget: {MAX_ITERS} iters   "
          f"repeats: {N_REPEATS}")
    print(f"  volume strength ×{VOLUME_STRENGTH_MULT}, radius "
          f"[{MIN_VOLUME_RADIUS}, {MAX_VOLUME_RADIUS}]")

    # Fit (or load) the generative surrogate once; share across all trials.
    if SURROGATE_PICKLE and Path(SURROGATE_PICKLE).exists():
        print(f"  loading surrogate ← {SURROGATE_PICKLE}")
        surr = SBS.Surrogate.load(SURROGATE_PICKLE)
    else:
        print("  fitting generative surrogate …")
        surr = SBS.Surrogate.fit(verbose=False)

    # trial_112 hyperparameters = the values run_7eb9 actually used (offline-MOBO pick).
    with open(E.RUN_DIR / "config.json") as f:
        run_config = json.load(f)
    base_hp = E.current_hparams(run_config)
    print(f"  trial_112 hyperparameters: {base_hp}")

    # True optima of the surrogate's deterministic Objective landscape (needle metric).
    predictor = SBS._ObjMeanPredictor(surr)
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
            m = {"source": "baseline_trial112", "n_injections": 0, "n_volumes_final": 0,
                 "n_volumes_total": 0, "n_reward_final": 0, "n_penalty_final": 0}
        baseline_samples.append(m)
        (tdir / "metrics.json").write_text(json.dumps(m, indent=2))
    baseline_stats = SBS.aggregate(baseline_samples)
    rows.append(_group_row("baseline_trial112", None, baseline_stats, baseline_samples, None))
    write_summary(sweep_dir, rows)

    # ── LLM volume-control cadence groups ────────────────────────────────────────
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
                      f"final volumes={m['n_volumes_final']} "
                      f"({m['n_reward_final']}R/{m['n_penalty_final']}P), "
                      f"total placed={m['n_volumes_total']}")
            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                m = {"source": group, "n_injections": 0, "n_volumes_final": 0,
                     "n_volumes_total": 0, "n_reward_final": 0, "n_penalty_final": 0}
            samples.append(m)
            (tdir / "metrics.json").write_text(json.dumps(m, indent=2))
        stats = SBS.aggregate(samples)
        rows.append(_group_row(group, interval, stats, samples, baseline_stats))
        write_summary(sweep_dir, rows)  # incremental

    # Overlaid running-best convergence: baseline vs each cadence (mean ± 95% CI).
    print("\n[plot] convergence comparison …")
    SBS.plot_convergence_comparison(
        sweep_dir,
        title=("Volume control — convergence: baseline vs LLM injection cadences\n"
               "(mean ± 95% CI over repeats)"))

    print(f"\nSweep complete → {sweep_dir / 'sweep_summary.csv'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] in ("--plot", "-p"):
        if len(args) < 2:
            raise SystemExit("usage: sweep_volume_control.py --plot <sweep_dir>")
        SBS.plot_convergence_comparison(
            Path(args[1]),
            title=("Volume control — convergence: baseline vs LLM injection cadences\n"
                   "(mean ± 95% CI over repeats)"))
    else:
        main()
