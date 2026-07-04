"""
llm/llm_config.py
=================
Single place to change (a) which Anthropic model tunes ZoMBI-Hop, (b) how the
prompt is written, and (c) how the LLM's answer is parsed.

The LLM is asked *once per run* whether it wants to change ZoMBI-Hop's
hyperparameters at the injection iteration.  It answers through a **structured
JSON schema** (Anthropic ``output_config.format``) so we get a validated
``{"reasoning": ..., "hyperparameter_changes": [{"name", "value"}, ...]}`` back
instead of having to parse free text.  An empty ``hyperparameter_changes`` list
means "leave the hyperparameters as they are".

Edit ``MODEL``, ``EFFORT``, and ``PROMPT_TEMPLATE`` below to retune.

Auth: the Anthropic SDK reads ``ANTHROPIC_API_KEY`` from the environment (or an
``ant auth login`` profile).  Nothing is hard-coded here.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ── .env loading ───────────────────────────────────────────────────────────────
# So you never have to export ANTHROPIC_API_KEY by hand: on import we load a .env
# file (repo root, then llm/) into the environment. A real environment variable
# always wins over the file. Uses python-dotenv if installed, else a tiny built-in
# parser (no dependency). See .env.example for the expected keys.
def _load_dotenv() -> None:
    candidates = [Path(__file__).resolve().parent.parent / ".env",   # repo root
                  Path(__file__).resolve().parent / ".env"]          # llm/.env
    try:
        from dotenv import load_dotenv  # type: ignore
        for p in candidates:
            if p.exists():
                load_dotenv(p, override=False)
        return
    except ImportError:
        pass
    # Fallback: minimal KEY=VALUE parser (no python-dotenv installed).
    for p in candidates:
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv()


# ── Model selection ────────────────────────────────────────────────────────────
# Any Anthropic model works.  The two the project cares about:
#   • Opus 4.8 at high effort   (default)
#   • Sonnet 5 at medium effort
# Swap the two lines below to switch.
MODEL: str = "claude-opus-4-8"
EFFORT: str = "high"          # "low" | "medium" | "high" | "xhigh" | "max"

# Alternative (uncomment to use Sonnet 5 medium instead):
# MODEL = "claude-sonnet-5"
# EFFORT = "medium"

# Adaptive thinking is the only supported thinking mode on Opus 4.8 / Sonnet 5.
USE_THINKING: bool = True
MAX_TOKENS: int = 8000        # generous headroom for reasoning + the JSON answer


# ── Static system-features overview ────────────────────────────────────────────
# The per-sample feature groups Archerfish measures, mirroring the "column groups"
# table in llm/data/campaign2_all.md. Injected into the prompt as {system_features}.
SYSTEM_FEATURES = """\
| Group | Fields | What it is |
|---|---|---|
| Composition inputs | `FAPbI3`, `MAPbI3`, `MAPbBr3`, `Module1`-`Module7` | Perovskite precursor / module fractions (the knobs being optimized) |
| Position | `X`, `Y` | Stage coordinates of the droplet on the substrate |
| Objectives / properties | `Bandgap` (1.45-2.59 eV), `Photoconductance` (0-1), `Stability` (0.32-1.0), `Objective` (0.28-0.86) | Measured targets and the combined `Objective` being maximized |
| Environment | `Temperature_in/out`, `Humidity_in/out`, `Pressure_in/out`, `DMF_ppm_in/out` | Glovebox/chamber conditions & solvent vapor at start vs. end of measurement |
| Absorption/reflectance spectrum | 462 columns `387`-`1003` | Intensity per wavelength (nm), ~UV-NIR |
| Stability sweep | `-40_dark_1` - `40_light_2/3` (120 cols) | Signals at voltages -40..+40, dark vs. light, 3 repeats |
| Kinetics | `k1`, `k2`, `k3` (+ `_var`) | Degradation rate constants and their variances |
| PL spectra | `450_i_pl`-`1042_i_pl` (initial) and `..._f_pl` (final), 30 each | Photoluminescence spectra before/after a degradation study |
| Metadata | `Timestamp` | Measurement time |"""


# ── The prompt ─────────────────────────────────────────────────────────────────
# Everything in {curly_braces} is filled in by evaluate_llm.build_prompt().
SYSTEM_PROMPT = (
    "You are an expert Bayesian-optimization engineer helping tune the "
    "ZoMBI-Hop algorithm on the fly. You are precise, quantitative, and "
    "conservative: you only change a hyperparameter when the run history gives "
    "you a concrete reason to expect it will help. You always answer through the "
    "provided structured-output schema."
)

PROMPT_TEMPLATE = """\
You are helping tune a Bayesian-optimization algorithm, ZoMBI-Hop, that is
optimizing a high-throughput autonomous perovskite-materials-discovery lab,
Archerfish, by tuning its hyperparameters on the fly. Archerfish continuously
dispenses precursor fluids from 3 syringe pumps at variable rates to deposit
gradients of material compounds as individual droplets.

## How ZoMBI-Hop works

ZoMBI-Hop is a "zooming multi-basin" Bayesian optimizer for finding MULTIPLE
optima ("needles") on a simplex-constrained search space. The composition inputs
are the mole fractions of `FAPbI3`, `MAPbI3`, and `MAPbBr3`, which always sum to
1 (a 3-simplex / ternary diagram). The objective (`Objective`) is MAXIMIZED.

Each outer loop is an "activation": ZoMBI-Hop fits a Gaussian-process (GP)
surrogate to the data, then repeatedly (each "iteration") uses an acquisition
function to pick a candidate composition. Archerfish measures a whole LINE of
~24 droplets through that candidate (LineBO), so every iteration adds a batch of
measured points. When the GP's expected improvement drops to the output-noise
floor for a few consecutive iterations, ZoMBI-Hop declares a "needle" (a local
optimum), penalizes the region around it with an ellipsoid so future search
avoids it, and starts a new activation to hunt the next needle. Within an
activation it also "zooms": it shrinks the search bounds around the best region
to refine the optimum. The algorithm stops when the search space is mostly
penalized or the trust regions collapse below the input-noise floor.

## What the Archerfish system measures

ZoMBI-Hop only optimizes over the 3 composition inputs and the `Objective`, but
each measured droplet is a rich physical sample: Archerfish records the full
optical spectra, environmental conditions, and degradation signals behind every
`Objective` value. For context, these are the feature groups captured per sample
(from the campaign database):

{system_features}

## The ZoMBI-Hop hyperparameters you may tune

(ranges are the allowed bounds; stay inside them)
{hparam_descriptions}

## The current ZoMBI-Hop hyperparameters (at the injection point)

{current_hparams}

## Offline hyperparameter-optimization history

These current defaults were chosen by a long offline multi-objective Bayesian
optimization (MOBO) run that tuned ZoMBI-Hop's hyperparameters on a surrogate of
this problem. Below is the FULL history of every hyperparameter configuration
that offline search evaluated and the performance it achieved. Each row is one
trial: its hyperparameter values followed by the three objectives it scored, all
of which are MINIMIZED (lower is better):

- `dist_to_needles`: mean distance from the needles ZoMBI-Hop found to the true
  optima — how well it located the real needles (lower = found them better).
- `dup_fraction`: fraction of measured points that were near-duplicates — wasted
  budget from re-measuring the same place (lower = more efficient).
- `runtime_s`: wall-clock seconds the run took (lower = faster).

Use this to understand which hyperparameter regions the offline search found
good vs. bad, and how each hyperparameter trades off the objectives. Note the
offline objectives differ from your online goal (you care about the continued
run's outcome on THIS specific real trajectory), so treat this as a strong prior,
not a rule.

{hparam_search_history}

## The measured campaign data so far

Columns: `iteration` is the ZoMBI-Hop iteration index (each iteration is one
LineBO line ≈ 24 measured droplets); `FAPbI3`,`MAPbI3`,`MAPbBr3` are the measured
composition (sum to 1); `Objective` is the measured objective (higher is better).
This is the real run's history up to and including injection iteration
{injection_iter}.

{history_table}

## Progress summary

{progress_summary}

## Needles found so far (local optima ZoMBI-Hop has already located)

{needle_summary}

## Your task

Decide whether to change any ZoMBI-Hop hyperparameters for the REST of this run
(from the injection point onward). The algorithm will continue from its exact
current internal state (same data, needles, zoom bounds) using whatever
hyperparameters you specify; anything you don't specify keeps its current value.

Think about what the history tells you: Is the run converging too slowly or too
fast? Is it over-exploring (many wasted duplicate points) or over-exploiting
(collapsing onto one region and missing needles)? Are needles being declared too
eagerly or too reluctantly? Then answer through the structured schema:

- `reasoning`: a concise, quantitative justification (2-6 sentences) grounded in
  the specific numbers above.
- `hyperparameter_changes`: a list of {{"name", "value"}} entries — ONLY the
  hyperparameters you want to change, each within its allowed range. Leave the
  list EMPTY if you judge the current settings are already appropriate.
"""


# ── Structured-output schema (validated by the Anthropic API) ──────────────────
def build_output_schema(hparam_names: List[str]) -> Dict[str, Any]:
    """JSON schema constraining the LLM's answer.

    ``hparam_names`` restricts the ``name`` field to the real tunable set so the
    model cannot invent a hyperparameter.
    """
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "hyperparameter_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": list(hparam_names)},
                        "value": {"type": "number"},
                    },
                    "required": ["name", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reasoning", "hyperparameter_changes"],
        "additionalProperties": False,
    }


# ── The one LLM call ───────────────────────────────────────────────────────────
def call_llm(user_prompt: str, hparam_names: List[str]) -> Dict[str, Any]:
    """Make the single hyperparameter-tuning call.

    Returns a dict with:
      ``decision``    – parsed {"reasoning", "hyperparameter_changes"} (or None on failure)
      ``latency_s``   – wall-clock seconds the API call took
      ``raw_text``    – the raw text block returned by the model
      ``model``       – the model id used
      ``effort``      – the effort level used
      ``usage``       – token usage dict
      ``error``       – error string if the call/parse failed, else None
    """
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - guidance for the user
        raise ImportError(
            "The `anthropic` package is required. Install it with "
            "`pip install anthropic` and set ANTHROPIC_API_KEY."
        ) from e

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / ant profile

    output_config: Dict[str, Any] = {
        "effort": EFFORT,
        "format": {"type": "json_schema", "schema": build_output_schema(hparam_names)},
    }
    kwargs: Dict[str, Any] = dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config=output_config,
    )
    if USE_THINKING:
        kwargs["thinking"] = {"type": "adaptive"}

    t0 = time.time()
    error = None
    raw_text = ""
    decision = None
    usage: Dict[str, Any] = {}
    try:
        resp = client.messages.create(**kwargs)
        latency_s = time.time() - t0
        # The structured-output guarantee: the first text block is valid JSON.
        raw_text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            if hasattr(resp.usage, "model_dump"):
                usage = resp.usage.model_dump()
            elif hasattr(resp.usage, "to_dict"):
                usage = resp.usage.to_dict()
            else:
                usage = dict(resp.usage)
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
        "decision": decision,
        "latency_s": latency_s,
        "raw_text": raw_text,
        "model": MODEL,
        "effort": EFFORT,
        "usage": usage,
        "error": error,
    }
