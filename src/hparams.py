"""
src/hparams.py
==============
The ONE registry of ZoMBI-Hop hyperparameters.

Every knob the optimiser reads is declared here exactly once, with its type, its
validation rule, where it lives on the running objects, and — if MOBO tunes it —
its search range. Everything else derives from this table:

* ``ZoMBIHop`` / ``DataHandler`` constructor defaults  -> ``ALL_DEFAULTS``
* the MOBO search space (``optimize/hparam_space.py``) -> ``HPARAM_SPACE``
* the GUI's tabs, labels and tooltips (``interface/app.py``) -> ``ui_categories()``
* the live-override channel (``src/core/hparam_live.py``) -> ``live_targets()``
* which keys a hyperparameter JSON may set (GUI, hardware runner, evaluate,
  apply_correction, retro_needles) -> ``KNOWN_NAMES`` / ``extract_hparams()``

Where values come from
----------------------
Tuned knobs take their shipped value from ``optimize/hparams/6d_ensemble.json``
(``DEFAULT_HPARAMS_PROVENANCE``); that file must set every tuned knob and nothing
else. Untuned knobs (physical constants, compute budgets, numerical guards) take
the ``default`` declared below. There is no third place: if a value is not in one
of those two, it does not exist.

Conflicts
---------
Some combinations make a behaviour impossible rather than merely bad — e.g.
``min_iters_per_zoom`` > ``max_iterations`` means no zoom level ever samples
enough lines to declare a needle. ``conflicts()`` names every such combination;
``ZoMBIHop`` refuses to construct (and the live-override channel refuses to apply)
a configuration that has any. MOBO samples a box, so ``repair_conflicts()`` maps a
sampled point onto the nearest feasible configuration instead.

No torch import, so the GUI, ``pareto.py`` and the planners can load this cheaply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Shipped values for every TUNED knob. Used by the GUI, the hardware runner and the
# ablation harness when no hyperparameter file is supplied, and as the ZoMBIHop
# constructor defaults. To adopt a different trial, replace that file's "hparams".
DEFAULT_HPARAMS_PROVENANCE = "optimize/hparams/6d_ensemble.json"


class HyperparameterConflict(ValueError):
    """A hyperparameter value or combination that makes ZoMBI-Hop unable to work."""


# ─── Value validators ─────────────────────────────────────────────────────────

def _pos_int(v: Any) -> int:
    if isinstance(v, float) and not v.is_integer():
        raise ValueError(f"must be an integer, got {v}")
    i = int(v)
    if i < 1:
        raise ValueError(f"must be >= 1, got {i}")
    return i


def _nonneg_int(v: Any) -> int:
    if isinstance(v, float) and not v.is_integer():
        raise ValueError(f"must be an integer, got {v}")
    i = int(v)
    if i < 0:
        raise ValueError(f"must be >= 0, got {i}")
    return i


def _pos_float(v: Any) -> float:
    f = float(v)
    if not f > 0:
        raise ValueError(f"must be > 0, got {f}")
    return f


def _nonneg_float(v: Any) -> float:
    f = float(v)
    if not f >= 0:
        raise ValueError(f"must be >= 0, got {f}")
    return f


def _open_unit(v: Any) -> float:
    f = float(v)
    if not 0.0 < f < 1.0:
        raise ValueError(f"must be strictly between 0 and 1, got {f}")
    return f


def _closed_unit(v: Any) -> float:
    f = float(v)
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"must be between 0 and 1, got {f}")
    return f


def _acq_type(v: Any) -> str:
    s = str(v).strip().lower()
    if s not in ("ucb", "ei"):
        raise ValueError(f"must be 'ucb' or 'ei', got {v!r}")
    return s


def _optional(cast: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def _cast(v: Any) -> Any:
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "none", "auto")):
            return None
        return cast(v)
    return _cast


# ─── The registry ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HParam:
    name: str
    kind: str                       # "int" | "float" | "str" — GUI entry type
    cast: Callable[[Any], Any]      # validate + coerce one value; raises ValueError
    category: str                   # GUI tab
    label: str                      # GUI label
    help: str                       # GUI tooltip / docs
    # Every live object attribute holding this value: ("dh"|"gp"|"zombi", attr).
    # The first entry is canonical. GPSimplex keeps renamed copies of a few.
    targets: tuple[tuple[str, str], ...]
    # (lo, hi, transform) with transform in {"log", "linear", "int"}; None = not tuned.
    tune: Optional[tuple] = None
    # Value for an UNTUNED knob. Tuned knobs get theirs from the shipped JSON.
    default: Any = None
    # Can the operator change it while a run is in flight (hparam_live)?
    live: bool = True


_ACQ_OPT = "Acquisition Opt"
_ACQ_FN = "Acq Function"
_ZOOM = "Zoom / Convergence"
_NEEDLE = "Penalty & Needle"
_PARING = "Point Paring"

_REGISTRY: tuple[HParam, ...] = (
    # ── acquisition optimisation ──────────────────────────────────────────────
    HParam("nat_grad_step", "float", _pos_float, _ACQ_OPT, "Nat-grad step",
           "Step size of the natural-gradient ascent that optimises the acquisition "
           "function on the simplex.",
           (("dh", "nat_grad_step"), ("gp", "nat_grad_step")),
           tune=(0.01, 0.1, "log")),
    HParam("nat_grad_max_steps", "int", _pos_int, _ACQ_OPT, "Nat-grad max steps",
           "Maximum natural-gradient steps per acquisition optimisation.",
           (("dh", "nat_grad_max_steps"), ("gp", "nat_grad_max_steps")),
           tune=(10, 80, "int")),
    HParam("n_restarts", "int", _pos_int, _ACQ_OPT, "Restarts",
           "Random restarts for multi-start acquisition optimisation.",
           (("dh", "n_restarts"), ("gp", "num_restarts")),
           tune=(100, 300, "int")),
    HParam("raw", "int", _pos_int, _ACQ_OPT, "Raw samples",
           "Raw random candidates scored before gradient refinement; also the sample "
           "count of the over-penalisation check.",
           (("dh", "raw"), ("gp", "raw_samples")),
           tune=(150, 400, "int")),

    # ── acquisition function ──────────────────────────────────────────────────
    HParam("ucb_beta", "float", _nonneg_float, _ACQ_FN, "UCB β",
           "UCB exploration weight: higher = more exploration. Ignored when "
           "acquisition_type is 'ei'.",
           (("dh", "ucb_beta"), ("gp", "ucb_beta")),
           tune=(0.001, 3.0, "linear")),
    HParam("acquisition_type", "str", _acq_type, _ACQ_FN, "Acq. type",
           "Acquisition function used to pick the next query point.\n"
           "  ucb – Upper Confidence Bound (exploration via β)\n"
           "  ei  – Expected Improvement",
           (("dh", "acquisition_type"), ("gp", "acquisition_type")),
           default="ucb"),
    HParam("input_noise", "float", _pos_float, _ACQ_FN, "Input noise (composition)",
           "Known per-component composition noise std of the apparatus. A physical "
           "measurement, not a tuned knob: 0.128 is the pooled std from "
           "runs/run_39af/composition_log.jsonl (6-dim hardware run of 2026-08-12), "
           "which logs the SENT composition next to the realised one. The earlier "
           "0.064 came from data/2nd_real_run.db, where the sent composition had to be "
           "inferred by a line fit that absorbs the systematic offset and under-reports "
           "~3x. Scales the paring radius, the minimum zoom-box width and the needle "
           "stop floor.",
           (("dh", "input_noise"),),
           default=0.128),
    HParam("repulsion_lambda", "float", _optional(_pos_float), _ACQ_FN,
           "Boundary repulsion λ",
           "Strength of the acquisition penalty for leaving the search box. None = "
           "auto-scaled to the acquisition's magnitude.",
           (("dh", "repulsion_lambda"), ("gp", "repulsion_lambda")),
           default=None, live=False),

    # ── zoom / convergence ────────────────────────────────────────────────────
    HParam("max_zooms", "int", _pos_int, _ZOOM, "Max zooms",
           "Zoom levels per activation. Must exceed min_zoom_for_needle, or no "
           "activation ever gets deep enough to declare a needle.",
           (("dh", "max_zooms"),),
           tune=(3, 6, "int")),
    HParam("max_iterations", "int", _pos_int, _ZOOM, "Max iters / zoom",
           "Measured lines per zoom level. Must be >= min_iters_per_zoom and >= "
           "n_consecutive_converged. max_zooms × max_iterations is also the line "
           "budget of one activation.",
           (("dh", "max_iterations"),),
           tune=(3, 12, "int")),
    HParam("top_m_points", "int", _optional(_pos_int), _ZOOM, "Top-M points",
           "Best points whose bounding box becomes the next zoom region. None = "
           "auto, max(d + 1, 4).",
           (("dh", "top_m_points"),),
           tune=(4, 16, "int")),
    HParam("n_consecutive_converged", "int", _pos_int, _ZOOM, "Consec. converged",
           "Consecutive converged lines (EI and realised improvement both below "
           "output_noise_threshold_mult × GP noise) needed before a needle may be "
           "declared. Must be <= max_iterations.",
           (("dh", "n_consecutive_converged"),),
           tune=(2, 5, "int")),
    HParam("output_noise_threshold_mult", "float", _nonneg_float, _ZOOM,
           "Output noise mult",
           "Convergence floor: a line converges when EI and the realised best-Y "
           "improvement are both below mult × GP output-noise std. Below ~0.1 the "
           "test is effectively unsatisfiable.",
           (("dh", "output_noise_threshold_mult"),),
           tune=(0.1, 2.0, "linear")),
    HParam("input_noise_threshold_mult", "float", _nonneg_float, _ZOOM,
           "Min zoom-box width",
           "Floor on every zoom box's per-axis width, as a multiple of input_noise "
           "(3 × 0.128 = 0.384). 0 disables the floor.",
           (("dh", "input_noise_threshold_mult"),),
           tune=(0.5, 4.0, "linear")),
    HParam("zoom_jaccard_threshold", "float", _closed_unit, _ZOOM,
           "Zoom Jaccard thresh",
           "If the next zoom box overlaps a box this activation already searched by "
           "more than this Jaccard fraction, there is no novel window left: the "
           "activation ends and its region is penalised.",
           (("zombi", "zoom_jaccard_threshold"),),
           tune=(0.5, 0.95, "linear")),
    HParam("jaccard_window", "int", _pos_int, _ZOOM, "Jaccard window",
           "How many recent zoom boxes a candidate box is compared against when "
           "sliding the top-M window.",
           (("dh", "jaccard_window"),),
           tune=(1, 5, "int")),
    HParam("jaccard_threshold", "float", _closed_unit, _ZOOM, "Jaccard slide thresh",
           "A candidate zoom box overlapping a recent one by more than this is "
           "rejected and the top-M window slides to the next-best points.",
           (("dh", "jaccard_threshold"),),
           tune=(0.5, 0.99, "linear")),
    HParam("max_gp_points", "int", _pos_int, _ZOOM, "Max GP points",
           "Cap on points passed to the GP fit (compute budget).",
           (("dh", "max_gp_points"),),
           default=3000),

    # ── penalty & needle ──────────────────────────────────────────────────────
    HParam("max_penalty_radius", "float", _pos_float, _NEEDLE, "Max penalty radius",
           "Cap on each semi-axis of a needle's (or exclusion zone's) penalty "
           "ellipsoid.",
           (("zombi", "max_penalty_radius"),),
           tune=(0.15, 1.5, "linear")),
    HParam("max_penalty_fraction", "float", _open_unit, _NEEDLE, "Max penalty fraction",
           "Dimension-aware cap: no single needle or exclusion zone may cover more "
           "than this fraction of the simplex's volume. Each semi-axis is capped at "
           "the radius of a ball of that volume (0.208 in 6-d for 0.10), or at "
           "max_penalty_radius if that is smaller.",
           (("zombi", "max_penalty_fraction"),),
           default=0.10),
    HParam("ellipsoid_drop_fraction", "float", _open_unit, _NEEDLE,
           "Ellipsoid drop frac",
           "Penalty ellipsoid boundary: where the acquisition has dropped by this "
           "fraction of its peak value at the needle.",
           (("zombi", "ellipsoid_drop_fraction"),),
           tune=(0.05, 0.5, "linear")),
    HParam("ellipsoid_eigenvalue_floor", "float", _pos_float, _NEEDLE,
           "Eigenvalue floor",
           "Minimum eigenvalue of a penalty ellipsoid matrix (numerical guard).",
           (("zombi", "ellipsoid_eigenvalue_floor"),),
           default=1e-6),
    HParam("needle_shrink_factor", "float", _open_unit, _NEEDLE, "Needle shrink factor",
           "When no candidate can be produced and no new data arrived, every needle "
           "ellipsoid's radii are multiplied by this to hand territory back.",
           (("zombi", "needle_shrink_factor"),),
           tune=(0.5, 0.95, "linear")),
    HParam("needle_stop_noise_multiplier", "float", _pos_float, _NEEDLE,
           "Needle stop mult",
           "Shrinking stops (and the run ends, unless never_terminate) once every "
           "needle semi-axis is below mult × input_noise.",
           (("zombi", "needle_stop_noise_multiplier"),),
           tune=(1.0, 8.0, "linear")),
    HParam("min_zoom_for_needle", "int", _nonneg_int, _NEEDLE, "Min zoom for needle",
           "A needle may only be declared at this 0-indexed zoom level or deeper "
           "(2 ⇒ after at least two zoom-ins). Must be < max_zooms.",
           (("zombi", "min_zoom_for_needle"),),
           tune=(0, 2, "int")),
    HParam("min_iters_per_zoom", "int", _nonneg_int, _NEEDLE, "Min iters per zoom",
           "Lines that must be measured at a zoom level before it may declare a "
           "needle or zoom in early. Must be <= max_iterations.",
           (("zombi", "min_iters_per_zoom"),),
           tune=(1, 3, "int")),
    HParam("needle_min_repeats", "int", _nonneg_int, _NEEDLE, "Needle min repeats",
           "Repeatability gate: a needle needs at least this many OTHER unpenalised "
           "measurements within needle_repeat_radius_frac of it. 0 disables the gate.",
           (("zombi", "needle_min_repeats"),),
           tune=(0, 10, "int")),
    HParam("needle_repeat_radius_frac", "float", _pos_float, _NEEDLE,
           "Needle repeat radius",
           "Composition-L2 radius within which measurements count as repeats for "
           "the repeatability gate.",
           (("zombi", "needle_repeat_radius_frac"),),
           tune=(0.03, 0.2, "linear")),

    # ── point paring ──────────────────────────────────────────────────────────
    HParam("paring_spatial_halfnoise", "float", _nonneg_float, _PARING,
           "Spatial halfnoise",
           "Points within mult × input_noise (scaled by zoom size) are candidates "
           "for merging when deduplicating the GP training set.",
           (("dh", "paring_spatial_halfnoise"),),
           tune=(0.1, 2.0, "linear")),
    HParam("paring_y_noise_multiplier", "float", _nonneg_float, _PARING,
           "Y-noise mult",
           "Spatial neighbours whose Y differ by less than mult × output noise are "
           "averaged into one representative point.",
           (("dh", "paring_y_noise_multiplier"),),
           tune=(0.1, 5.0, "linear")),
)

HPARAMS: dict[str, HParam] = {p.name: p for p in _REGISTRY}
KNOWN_NAMES: frozenset[str] = frozenset(HPARAMS)

# MOBO search space, in registry order. run_mobo / pareto read it via
# optimize/hparam_space.py.
HPARAM_SPACE: dict[str, tuple] = {p.name: p.tune for p in _REGISTRY if p.tune is not None}
TUNED_NAMES: list[str] = list(HPARAM_SPACE)

LIVE_EDITABLE_NAMES: frozenset[str] = frozenset(p.name for p in _REGISTRY if p.live)

FIXED_DEFAULTS: dict[str, Any] = {p.name: p.default for p in _REGISTRY if p.tune is None}
DEFAULT_INPUT_NOISE: float = FIXED_DEFAULTS["input_noise"]


# ─── Legacy trial records ─────────────────────────────────────────────────────
#
# Trials recorded before 2026-10-12 (and every hparams JSON written from them) carry
# the 16-knob space. Two of those knobs never reached the optimiser — the code read
# bounds_shrink_factor (0.8) and min_axis_noise_mult (2.0) instead — and the knobs
# added to the space that day were hidden constructor defaults. So every legacy
# trial actually ran with the values below, whatever its record says. Rewriting
# them makes old records reproducible and lets old MOBO history keep pooling
# without teaching the GP a fake response on axes that were never wired up.
LEGACY_NAMES: frozenset[str] = frozenset({
    "nat_grad_step", "nat_grad_max_steps", "n_restarts", "raw", "ucb_beta",
    "max_zooms", "max_iterations", "top_m_points", "n_consecutive_converged",
    "output_noise_threshold_mult", "max_penalty_radius", "needle_shrink_factor",
    "needle_stop_noise_multiplier", "paring_spatial_halfnoise",
    "paring_y_noise_multiplier",
})
LEGACY_EFFECTIVE: dict[str, Any] = {
    "needle_shrink_factor": 0.8,
    "needle_stop_noise_multiplier": 2.0,
    # Retired from the space in 56dd11d; every run since used the constructor's 3.0
    # (the values older records carry were inert at the time).
    "input_noise_threshold_mult": 3.0,
    "zoom_jaccard_threshold": 0.75,
    "jaccard_window": 3,
    "jaccard_threshold": 0.9,
    "ellipsoid_drop_fraction": 0.25,
    "min_zoom_for_needle": 2,
    "min_iters_per_zoom": 3,
    "needle_min_repeats": 5,
    "needle_repeat_radius_frac": 0.10,
}
# Knobs no pre-2026-10-12 record can contain: they were neither tuned nor live-
# editable, so they appear in no trial record and no hparams_effective.json.
_NEW_MARKERS = frozenset({"min_zoom_for_needle", "min_iters_per_zoom",
                          "needle_min_repeats", "needle_repeat_radius_frac"})
# Retired names a live run's hparams_effective.json / config dump may carry, mapped
# to the knob that now does their job. Their recorded value is the one in force.
_RETIRED_ALIASES = {"bounds_shrink_factor": "needle_shrink_factor",
                    "min_axis_noise_mult": "needle_stop_noise_multiplier"}


def is_legacy(hp: dict) -> bool:
    """True for a full pre-2026-10-12 record: every old axis, none of the new gates.

    A hand-written partial config never has every old axis, so it is never
    mistaken for one.
    """
    return LEGACY_NAMES <= set(hp) and not (_NEW_MARKERS & set(hp))


def upgrade_legacy(hp: dict) -> tuple[dict, list[str]]:
    """Rewrite a legacy record to the values it actually ran with.

    Returns ``(hp, notes)``; ``notes`` is empty (and ``hp`` an unchanged copy) for
    anything that is not a legacy record.
    """
    out = dict(hp)
    if not is_legacy(hp):
        return out, []
    # A live run's own dump (it carries the retired names) recorded the values that
    # were really in force for every live-editable knob; a MOBO record did not.
    live_dump = any(alias in hp for alias in _RETIRED_ALIASES)
    alias_of = {new: old for old, new in _RETIRED_ALIASES.items()}
    notes = []
    for name, value in LEGACY_EFFECTIVE.items():
        if name in alias_of:
            value = hp.get(alias_of[name], value)
        elif live_dump and name in hp:
            continue
        if name in out and out[name] != value:
            notes.append(f"{name} {out[name]} -> {value} (what legacy code used)")
        out[name] = value
    for alias in _RETIRED_ALIASES:
        out.pop(alias, None)
    return out, notes


# ─── Shipped defaults ─────────────────────────────────────────────────────────

def _load_shipped() -> dict[str, Any]:
    path = _REPO_ROOT / DEFAULT_HPARAMS_PROVENANCE
    with open(path) as f:
        blob = json.load(f)
    hp = blob.get("hparams", blob)
    missing = [n for n in TUNED_NAMES if n not in hp]
    extra = sorted(set(hp) - set(TUNED_NAMES))
    if missing or extra:
        raise RuntimeError(
            f"{DEFAULT_HPARAMS_PROVENANCE} must set exactly the tuned hyperparameters; "
            f"missing {missing}, unexpected {extra}")
    return {n: HPARAMS[n].cast(hp[n]) for n in TUNED_NAMES}


# Shipped value of every TUNED knob (the shape pre-registry callers expect).
DEFAULT_HPARAMS: dict[str, Any] = _load_shipped()
# Every knob — what a ZoMBIHop constructed with no hyperparameter arguments uses.
ALL_DEFAULTS: dict[str, Any] = {**FIXED_DEFAULTS, **DEFAULT_HPARAMS}


# ─── Validation ───────────────────────────────────────────────────────────────

def coerce(name: str, value: Any) -> Any:
    """Validate/cast one hyperparameter. Raises ValueError with the name in it."""
    if name not in HPARAMS:
        raise ValueError(f"unknown hyperparameter {name!r}")
    try:
        return HPARAMS[name].cast(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {exc}") from None


def lines_per_activation(hp: dict) -> int:
    """Line budget of one activation: every zoom level run to max_iterations.

    Derived rather than set: any smaller cap would cut zoom levels short, and any
    larger one could only be reached by the failure-retry path re-entering a zoom.
    """
    return int(hp["max_zooms"]) * int(hp["max_iterations"])


def conflicts(hp: dict) -> list[str]:
    """Every value or combination in ``hp`` that makes ZoMBI-Hop unable to work.

    Missing keys are filled from ``ALL_DEFAULTS``, so a partial dict is checked as
    the configuration it would actually produce.
    """
    problems: list[str] = []
    full = {**ALL_DEFAULTS, **hp}
    for name in KNOWN_NAMES & set(hp):
        try:
            HPARAMS[name].cast(hp[name])
        except (TypeError, ValueError) as exc:
            problems.append(f"{name}={hp[name]!r}: {exc}")
    if problems:
        return problems

    max_zooms = int(full["max_zooms"])
    max_iter = int(full["max_iterations"])
    min_zoom = int(full["min_zoom_for_needle"])
    min_iters = int(full["min_iters_per_zoom"])
    n_cc = int(full["n_consecutive_converged"])

    if max_zooms <= min_zoom:
        problems.append(
            f"max_zooms={max_zooms} <= min_zoom_for_needle={min_zoom}: zoom levels run "
            f"0..{max_zooms - 1}, but a needle needs level {min_zoom} or deeper, so "
            "no needle can ever be declared")
    if min_iters > max_iter:
        problems.append(
            f"min_iters_per_zoom={min_iters} > max_iterations={max_iter}: no zoom "
            "level ever measures enough lines to declare a needle")
    if n_cc > max_iter:
        problems.append(
            f"n_consecutive_converged={n_cc} > max_iterations={max_iter}: a zoom level "
            "ends before it can see that many converged lines in a row, so a needle "
            "needs a convergence streak carried over from a shallower zoom")
    return problems


def check_conflicts(hp: dict, *, context: str = "") -> None:
    """Raise ``HyperparameterConflict`` listing every problem ``conflicts`` finds."""
    problems = conflicts(hp)
    if problems:
        where = f" ({context})" if context else ""
        raise HyperparameterConflict(
            f"Hyperparameter conflict{where}:\n  - " + "\n  - ".join(problems))


def resolve_for_run(passed: dict, run_config: Optional[dict] = None) -> dict:
    """The configuration a ZoMBIHop built with ``passed`` would actually run.

    For a resume, ``run_config`` is the run's config.json: its recorded knobs win
    over ``passed`` (that is what the resume does), except ``input_noise``, a
    calibration of the apparatus for which the current measurement wins. Lets a
    launcher check for conflicts BEFORE committing hardware to a run.
    """
    out = {**ALL_DEFAULTS, **passed}
    if run_config:
        out.update({k: v for k, v in run_config.items()
                    if k in KNOWN_NAMES and k != "input_noise"})
    return out


def repair_conflicts(hp: dict) -> tuple[dict, list[str]]:
    """Move ``hp`` onto the nearest configuration ``conflicts`` accepts.

    ONLY for MOBO, which samples a box that contains infeasible corners and records
    the repaired point it actually ran. Anything a person supplies (a config file,
    a showdown, a GUI launch or live edit) must instead be refused with
    ``check_conflicts`` — running a config other than the one asked for, however
    close, is a silent substitution. Each fix lowers the requirement rather than
    raising the budget, so a repaired config never costs more lines than the
    sampled one. Returns ``(hp, notes)``.
    """
    out = dict(hp)
    notes: list[str] = []
    full = {**ALL_DEFAULTS, **out}
    max_iter = int(full["max_iterations"])
    max_zooms = int(full["max_zooms"])
    for name, cap in (("n_consecutive_converged", max_iter),
                      ("min_iters_per_zoom", max_iter),
                      ("min_zoom_for_needle", max_zooms - 1)):
        if int(full[name]) > cap:
            notes.append(f"{name} {full[name]} -> {cap}")
            out[name] = cap
    return out, notes


# ─── Reading hyperparameter files ─────────────────────────────────────────────

def extract_hparams(blob: dict, *, source: str = "hparams",
                    log: Optional[Callable[[str], None]] = print) -> dict:
    """Known hyperparameters from a trial.json-style blob or a flat dict.

    Unknown keys (metrics, provenance, retired knobs) are dropped and reported, and
    a legacy trial record is upgraded to the values it actually ran with.
    """
    if not isinstance(blob, dict):
        raise ValueError(f"{source}: hyperparameter JSON must be an object")
    hp = blob.get("hparams", blob)
    if not isinstance(hp, dict):
        raise ValueError(f"{source}: 'hparams' must be an object")
    hp, notes = upgrade_legacy(hp)
    if notes and log:
        log(f"[hparams] {source}: legacy record — " + "; ".join(notes))
    known = {k: v for k, v in hp.items() if k in KNOWN_NAMES}
    dropped = sorted(set(hp) - KNOWN_NAMES)
    if dropped and log:
        log(f"[hparams] {source}: ignoring unknown/retired key(s) {dropped}")
    return known


def load_hparams_file(path, *, log: Optional[Callable[[str], None]] = print) -> dict:
    """``extract_hparams`` on a JSON file."""
    with open(path) as f:
        return extract_hparams(json.load(f), source=str(path), log=log)


# ─── Consumers' views ─────────────────────────────────────────────────────────

def live_targets() -> dict[str, tuple[tuple[str, str], ...]]:
    """name -> live attribute locations, for every live-editable knob."""
    return {p.name: p.targets for p in _REGISTRY if p.live}


def owned_by(owner: str) -> list[str]:
    """Names whose canonical home is ``owner`` ("dh" or "zombi")."""
    return [p.name for p in _REGISTRY if p.targets[0][0] == owner]


def ui_categories() -> dict[str, list[tuple]]:
    """``{tab: [(name, kind, default, label, help), ...]}`` for the GUI."""
    out: dict[str, list[tuple]] = {}
    for p in _REGISTRY:
        out.setdefault(p.category, []).append(
            (p.name, p.kind, ALL_DEFAULTS[p.name], p.label, p.help))
    return out
