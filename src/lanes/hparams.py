"""
src/lanes/hparams.py
====================
The registry of the lanes knobs (DESIGN.md section 2.2, N3; reviewer correction R2 item 5).

Same ``HParam`` dataclass, validators and conventions as ``src/hparams.py``, but a SEPARATE registry: the
lanes knobs are deliberately NOT in ``src.hparams._REGISTRY`` / ``ALL_DEFAULTS`` / ``KNOWN_NAMES``.
``ALL_DEFAULTS`` feeds ``ZoMBIHop(**hp)`` on hardware (an unknown kwarg raises TypeError), and
``owned_by()`` / ``ui_categories()`` feed ``config.json`` and the GUI tabs, so adding names there would
change flag-off artefacts. Importing this module changes nothing in ``src.hparams``.

Conventions
-----------
* Every name starts with ``lanes_``. ``targets[0]`` is ``("lanes", <LaneOptions field>)`` (the engine,
  ``src/lanes/options.py``) or ``("controller", <field>)`` (the Phase-2 controller / driver config).
* ``tune=None`` (nothing here is tuned by MOBO). Bools have GUI kind ``"str"`` and a strict
  ``true`` / ``false`` cast (the GUI entry kinds are int / float / str).
* Defaults ARE the validated leader ``im+cross_cover_frac=1.01__zbudget@z2_min=3+catchup``.
* ``live=True`` only for the controller timing knobs (``LANES_LIVE_EDITABLE``): they may change in a running
  lanes run through ``<run_dir>/lanes/override.json`` (Phase 2) because the emulator constants must be
  calibrated on hardware (critic G7). The dive rules were validated as one set and shape per-dive
  counters, so they are not live; core ZoMBIHop knobs are not live in lanes runs either.

Conflicts (``lanes_conflicts``; refused like ``src.hparams.check_conflicts``): per-zoom budget rules of the
sandbox ``validate_zbudget_params`` / ``validate_against_stock`` (ZB 103-124), the zoom floor reachable within
the extensions (critic G6), catch-up zooms inside the zoom range, ``lead_s < period0_s``. Core-knob values
outside the validated set are warnings only (``lanes_warnings``). Core conflicts themselves stay with
``src.hparams.conflicts``.

Pure module: no torch use, no side effects.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from src.hparams import (
    ALL_DEFAULTS,
    HParam,
    HyperparameterConflict,
    _nonneg_float,
    _nonneg_int,
    _open_unit,
    _pos_float,
)
from src.lanes.budget import ZOOMS, activation_budget, zbudget_hp

# ─── Value validators (src/hparams.py conventions) ────────────────────────────


def _bool_strict(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ValueError(f"must be true or false, got {v!r}")


def _floor_rule(v: Any) -> str:
    s = str(v).strip().lower()
    if s not in ("graded", "volume"):
        raise ValueError(f"must be 'graded' or 'volume', got {v!r}")
    return s


def _half_open_half(v: Any) -> float:
    f = float(v)
    if not 0.0 <= f < 0.5:
        raise ValueError(f"must be in [0, 0.5), got {f}")
    return f


def _unit_left_open(v: Any) -> float:
    f = float(v)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"must be in (0, 1], got {f}")
    return f


def _int_ge2(v: Any) -> int:
    i = _nonneg_int(v)
    if i < 2:
        raise ValueError(f"must be >= 2, got {i}")
    return i


def _zero_or_ge2(v: Any) -> int:
    """ZB 108-112: 0 (= the core value) or >= 2."""
    i = _nonneg_int(v)
    if 0 < i < 2:
        raise ValueError(f"must be 0 (= the core value) or >= 2, got {i}")
    return i


def _catchup_zooms(v: Any) -> str:
    """CU 92-96 (``parse_zooms``): digits from '12' (k >= 1: T_k vs T_(k-1)); canonical sorted digits."""
    z = str(v).strip()
    if not z or any(c not in "12" for c in z):
        raise ValueError(f"must be digits from '12' (zooms k >= 1 whose exit is tested), got {v!r}")
    return "".join(sorted(set(z)))


def _pickup_source(v: Any) -> str:
    s = str(v).strip().lower()
    if s != "infer":
        raise ValueError(f"only 'infer' is accepted in v1 (pick-ups inferred from timing; acks are logged "
                         f"only), got {v!r}")
    return s


# ─── The registry ─────────────────────────────────────────────────────────────

_DIVE = "Lanes: dive"
_ZB = "Lanes: zoom budget"
_CU = "Lanes: catch-up"
_TIMING = "Lanes: timing"


def _ln(name: str, kind: str, cast: Callable[[Any], Any], cat: str, label: str, help_: str, field: str,
        default: Any, *, owner: str = "lanes", live: bool = False) -> HParam:
    return HParam(name, kind, cast, cat, label, help_, ((owner, field),), tune=None, default=default, live=live)


def _zoom_knobs() -> tuple:
    out = []
    what = {
        "max": ("int", _zero_or_ge2, "sends",
                "Lines sent at zoom {k} before the budget transition (zoom-in, or the 'zooms_exhausted' zone at the "
                "last zoom), plus up to lanes_ext_cap extensions while fewer than the zoom's minimum results are "
                "back. 0 = the core max_iterations."),
        "min": ("int", _zero_or_ge2, "results",
                "Own counted zoom-{k} results needed before the zoom-in (or, at the last zoom, before the needle "
                "is tried). 0 = the core min_iters_per_zoom; else >= max(2, min_iters_per_zoom)."),
        "nc": ("int", _nonneg_int, "converged checks",
               "Consecutive converged checks needed at zoom {k} before the zoom-in / needle try. 0 = the core "
               "n_consecutive_converged."),
    }
    for p in ("max", "min", "nc"):
        kind, cast, short, help_ = what[p]
        for k in ZOOMS:
            default = 3 if (p == "min" and k == 2) else 0          # the leader: zbudget@z2_min=3
            out.append(_ln(f"lanes_z{k}_{p}", kind, cast, _ZB, f"Zoom {k} {short}", help_.format(k=k),
                           f"z{k}_{p}", default))
    return tuple(out)


_REGISTRY: tuple[HParam, ...] = (
    # ── dive rules (LaneOptions core knobs, LO 60-95) ────────────────────────
    _ln("lanes_rho", "float", _open_unit, _DIVE, "Box shrink ρ",
        "Every axis of the zoom-(k+1) box is at most rho x the zoom-k box's width (nested, strictly "
        "shrinking boxes).", "rho", 0.7),
    _ln("lanes_floor_rule", "str", _floor_rule, _DIVE, "Box width floor rule",
        "Input-noise width floor of a zoom box: 'graded' = min_w / rho^(max_zooms-1-k) (min_w at the last "
        "zoom), so every axis can still shrink by rho; 'volume' = min_w at every zoom.", "floor_rule", "graded"),
    _ln("lanes_inner_margin", "float", _nonneg_float, _DIVE, "Line inner margin",
        "Zoom >= 1 lines stay this far inside the zoom box (absolute, per side).", "inner_margin", 0.02),
    _ln("lanes_inner_margin_frac", "float", _half_open_half, _DIVE, "Line inner margin (max share)",
        "... but never more than this share of the axis width.", "inner_margin_frac", 0.25),
    _ln("lanes_count_frac", "float", _unit_left_open, _DIVE, "Count share in box",
        "A zoom >= 1 packet counts as a result of its zoom when at least this share of its rows lies inside the "
        "zoom box.", "count_frac", 0.25),
    _ln("lanes_cross_cover_frac", "float", _pos_float, _DIVE, "Cross-lane cover end",
        "The other lane's needles / zones covering at least this share of the box end this lane's dive. "
        "> 1 = the rule is off (the leader: 1.01).", "cross_cover_frac", 1.01),
    _ln("lanes_cross_overlap_frac", "float", _unit_left_open, _DIVE, "Cross-lane overlap end",
        "A new box overlapping the other lane's active box by at least this share of its volume ends the dive.",
        "cross_overlap_frac", 0.5),
    _ln("lanes_ext_cap", "int", _nonneg_int, _DIVE, "Extensions per zoom",
        "Extra lines at a zoom whose sends ran out while fewer than its minimum own results are back.",
        "ext_cap", 3),
    _ln("lanes_incumbent_lines", "str", _bool_strict, _DIVE, "Incumbent-mean lines",
        "Zoom >= 1 lines are chords through the GP-mean incumbent of the lane's own rows (Fix 8).", "incmean", True),
    _ln("lanes_num_lines", "int", _int_ge2, _DIVE, "Ranked lines",
        "LineBO ranked lines per decision (main = rank 1, backup = rank 2).", "num_lines", 10),
    _ln("lanes_refresh_relevance", "str", _bool_strict, _DIVE, "Refresh only when relevant",
        "A refresh recomputes only when new rows fall in the lane's box (zoom >= 1), any new rows (zoom 0), or "
        "zones changed.", "refresh_relevance", True),
    _ln("lanes_refresh_full", "str", _bool_strict, _DIVE, "Full refresh",
        "False: a refresh refits and re-ranks but keeps the queued line's GP candidate; True: also re-runs "
        "get_candidate.", "refresh_full", False),

    # ── per-zoom budget (ZB 79; the leader's zoom-2 floor z2_min = 3) ───────
    *_zoom_knobs(),

    # ── catch-up (CU 384) ─────────────────────────────────────────────────────
    _ln("lanes_catchup_zooms", "str", _catchup_zooms, _CU, "Catch-up zooms",
        "Zooms k whose exit is tested (T_k vs T_(k-1), the dive's measured top-m levels): '1' = the 1 -> 2 "
        "zoom-in, '2' = the needle, '12' = both.", "catchup_zooms", "12"),
    _ln("lanes_catchup_eps", "float", _nonneg_float, _CU, "Catch-up tolerance",
        "Hold a zoom exit while T_k < T_(k-1) - eps.", "catchup_eps", 0.01),
    _ln("lanes_catchup_max", "int", _nonneg_int, _CU, "Catch-up extra lines",
        "Extra zoom-k lines per activation and zoom while held (0 = never hold).", "catchup_max", 2),
    _ln("lanes_catchup_top_m", "int", _nonneg_int, _CU, "Catch-up top-m",
        "Droplets averaged into a level T_k (the m best of the dive's packets sent at zoom k). 0 = 8 (6-D) / "
        "11 (10-D) / 8 (other).", "catchup_top_m", 0),

    # ── controller / timing (LH 857-858, 883; PR 450-453) ────────────────────
    _ln("lanes_refresh", "str", _bool_strict, _TIMING, "Refresh",
        "Refresh the other lane's queued line before its rail is predicted to read it.", "refresh", True,
        owner="controller"),
    _ln("lanes_refresh_margin_s", "float", _nonneg_float, _TIMING, "Refresh margin (s)",
        "A refresh is requested only when at least this many seconds remain before the predicted pick-up.",
        "refresh_margin_s", 240.0, owner="controller", live=True),
    _ln("lanes_lead_s", "float", _pos_float, _TIMING, "Pick-up lead (s)",
        "Seconds between a rail reading its recipe and its result returning (emulator 'typical' profile: "
        "8.57 min + 685.8 s). Calibrate on hardware.", "lead_s", 1200.0, owner="controller", live=True),
    _ln("lanes_period0_s", "float", _pos_float, _TIMING, "Initial period (s)",
        "Result period assumed before results give a measured spacing (must exceed the lead).", "period0_s",
        3000.0, owner="controller", live=True),
    _ln("lanes_replenish_delay_s", "float", _nonneg_float, _TIMING, "Replenish delay (s)",
        "Extra wait after two periods without a result before a lane's backup is replenished.",
        "replenish_delay_s", 300.0, owner="controller", live=True),
    _ln("lanes_max_replenish", "int", _nonneg_int, _TIMING, "Max replenish",
        "Replenishes in a row without a result before the lane waits.", "max_replenish", 4,
        owner="controller", live=True),
    _ln("lanes_pickup_source", "str", _pickup_source, _TIMING, "Pick-up source",
        "How pick-ups are known: 'infer' (timing, validated) is the only value in v1; DiSCO read acks are "
        "logged only.", "pickup_source", "infer", owner="controller"),
    _ln("lanes_ingest_unattributed", "str", _bool_strict, _TIMING, "Ingest unattributed results",
        "lanes4: results whose recipe id is unknown or missing are logged and NOT ingested (false, default); "
        "true ingests them as shared data when they are newer than the run's first write (reviewer R2).",
        "ingest_unattributed", False, owner="controller"),
)

LANES_HPARAMS: dict[str, HParam] = {p.name: p for p in _REGISTRY}
LANES_KNOWN_NAMES: frozenset[str] = frozenset(LANES_HPARAMS)
LANES_LIVE_EDITABLE: frozenset[str] = frozenset(p.name for p in _REGISTRY if p.live)

# name -> LaneOptions field / controller field, and back
LANES_OPTION_FIELDS: dict[str, str] = {p.name: p.targets[0][1] for p in _REGISTRY if p.targets[0][0] == "lanes"}
CONTROLLER_FIELDS: dict[str, str] = {p.name: p.targets[0][1] for p in _REGISTRY
                                     if p.targets[0][0] == "controller"}
FIELD_TO_NAME: dict[str, str] = {v: k for k, v in LANES_OPTION_FIELDS.items()}

# Core knobs of the validated leader set (warnings, not refusals; DESIGN 2.2).
VALIDATED_CORE: dict[str, int] = {"max_zooms": 3, "min_zoom_for_needle": 2, "min_iters_per_zoom": 2,
                                  "max_iterations": 5, "n_consecutive_converged": 2}


def lanes_defaults() -> dict[str, Any]:
    """Every lanes knob at its default (= the validated leader). A fresh dict."""
    return {p.name: p.default for p in _REGISTRY}


def lanes_coerce(name: str, value: Any) -> Any:
    """Validate / cast one lanes knob. Raises ValueError with the name in it."""
    if name not in LANES_HPARAMS:
        raise ValueError(f"unknown lanes hyperparameter {name!r}")
    try:
        return LANES_HPARAMS[name].cast(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {exc}") from None


def resolve_lanes_hparams(lanes_hp: Optional[Mapping] = None) -> dict[str, Any]:
    """Defaults overlaid with ``lanes_hp`` (every value cast). Raises ValueError on an unknown name or a bad
    value (use ``lanes_conflicts`` to collect every problem instead)."""
    out = lanes_defaults()
    for k, v in dict(lanes_hp or {}).items():
        out[k] = lanes_coerce(k, v)
    return out


def lanes_option_fields(lanes_hp: Optional[Mapping] = None) -> dict[str, Any]:
    """``{LaneOptions field: value}`` of the engine knobs (defaults overlaid with ``lanes_hp``)."""
    vals = resolve_lanes_hparams(lanes_hp)
    return {f: vals[n] for n, f in LANES_OPTION_FIELDS.items()}


def controller_fields(lanes_hp: Optional[Mapping] = None) -> dict[str, Any]:
    """``{controller field: value}`` of the controller / driver knobs (defaults overlaid with ``lanes_hp``)."""
    vals = resolve_lanes_hparams(lanes_hp)
    return {f: vals[n] for n, f in CONTROLLER_FIELDS.items()}


def core_stock_hp(core_hp: Optional[Mapping] = None) -> dict[str, int]:
    """The lane's stock ``_hp()`` (LZ 626-630) computed from core hyperparameter NAMES (missing ones from
    ``src.hparams.ALL_DEFAULTS``)."""
    full = {**ALL_DEFAULTS, **dict(core_hp or {})}
    return dict(max_zooms=int(full["max_zooms"]), max_iter=int(full["max_iterations"]),
                min_iters=int(full["min_iters_per_zoom"]), n_consec=int(full["n_consecutive_converged"]),
                min_zoom_needle=int(full["min_zoom_for_needle"]))


def zbudget_params(lanes_hp: Optional[Mapping] = None) -> dict[str, int]:
    """``{"z0_max": .., ..., "z2_nc": ..}``: the ``P`` of ``src.lanes.budget.zbudget_hp``."""
    vals = resolve_lanes_hparams(lanes_hp)
    return {f"z{k}_{p}": int(vals[f"lanes_z{k}_{p}"]) for p in ("max", "min", "nc") for k in ZOOMS}


def lanes_lines_per_activation(core_hp: Optional[Mapping] = None, lanes_hp: Optional[Mapping] = None) -> int:
    """Derived activation line budget ``sum_k (max_iter_k + ext_cap)`` (ZB 127-129, 223-225): 24 for the
    leader. (Catch-up can raise ``ext[k]`` further; the activation budget still binds.)"""
    stock = core_stock_hp(core_hp)
    P = zbudget_params(lanes_hp)
    ext_cap = int(resolve_lanes_hparams(lanes_hp)["lanes_ext_cap"])
    return activation_budget([zbudget_hp(stock, j, P) for j in range(int(stock["max_zooms"]))], ext_cap)


def lanes_conflicts(core_hp: Optional[Mapping] = None, lanes_hp: Optional[Mapping] = None) -> list[str]:
    """Every lanes value or combination that the validated lanes engine cannot run.

    ``core_hp``: core ZoMBIHop knobs (missing ones from ``ALL_DEFAULTS``); ``lanes_hp``: ``lanes_*`` knobs
    (missing ones from ``lanes_defaults()``). Returns [] when the configuration is accepted."""
    problems: list[str] = []
    given = dict(lanes_hp or {})
    for n in sorted(set(given) - LANES_KNOWN_NAMES):
        problems.append(f"unknown lanes hyperparameter {n!r}")
    vals = lanes_defaults()
    for n in sorted(LANES_KNOWN_NAMES & set(given)):
        try:
            vals[n] = LANES_HPARAMS[n].cast(given[n])
        except (TypeError, ValueError) as exc:
            problems.append(f"{n}={given[n]!r}: {exc}")
    try:
        stock = core_stock_hp(core_hp)
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"core hyperparameters unreadable: {exc!r}")
        return problems
    if problems:
        return problems

    P = {f"z{k}_{p}": int(vals[f"lanes_z{k}_{p}"]) for p in ("max", "min", "nc") for k in ZOOMS}
    ext_cap = int(vals["lanes_ext_cap"])
    min_iters, max_zooms = int(stock["min_iters"]), int(stock["max_zooms"])
    for k in ZOOMS:                                                    # ZB 115-124
        m = P[f"z{k}_min"]
        if 0 < m < max(2, min_iters):
            problems.append(f"lanes_z{k}_min={m} is below max(2, min_iters_per_zoom={min_iters}): every zoom-in "
                            f"and needle needs at least that many own counted results (auditor A1)")
        hk = zbudget_hp(stock, k, P)
        if hk["max_iter"] < hk["min_iters"]:
            problems.append(f"zoom {k} would need {hk['min_iters']} results from {hk['max_iter']} sends "
                            f"(lanes_z{k}_max / max_iterations < lanes_z{k}_min); raise lanes_z{k}_max")
        if hk["min_iters"] > hk["max_iter"] + ext_cap:                # critic G6
            problems.append(f"zoom {k} floor {hk['min_iters']} results > {hk['max_iter']} sends + lanes_ext_cap "
                            f"{ext_cap}: the floor can never be reached")
    bad = [int(c) for c in vals["lanes_catchup_zooms"] if not 1 <= int(c) <= max_zooms - 1]
    if bad:
        problems.append(f"lanes_catchup_zooms={vals['lanes_catchup_zooms']!r}: zooms {bad} are outside 1.."
                        f"{max_zooms - 1} (max_zooms={max_zooms})")
    if not float(vals["lanes_lead_s"]) < float(vals["lanes_period0_s"]):
        problems.append(f"lanes_lead_s={vals['lanes_lead_s']} must be < lanes_period0_s={vals['lanes_period0_s']}")
    return problems


def check_lanes_conflicts(core_hp: Optional[Mapping] = None, lanes_hp: Optional[Mapping] = None, *,
                          context: str = "") -> None:
    """Raise ``src.hparams.HyperparameterConflict`` listing every problem ``lanes_conflicts`` finds."""
    problems = lanes_conflicts(core_hp, lanes_hp)
    if problems:
        where = f" ({context})" if context else ""
        raise HyperparameterConflict(
            f"Lanes hyperparameter conflict{where}:\n  - " + "\n  - ".join(problems))


def lanes_warnings(core_hp: Optional[Mapping] = None, lanes_hp: Optional[Mapping] = None) -> list[str]:
    """Values outside the validated leader set (logged, not refused): the core knobs of ``VALIDATED_CORE``
    and every non-live lanes knob that differs from its default."""
    out: list[str] = []
    full = {**ALL_DEFAULTS, **dict(core_hp or {})}
    for name, v in VALIDATED_CORE.items():
        try:
            same = int(full[name]) == v
        except (KeyError, TypeError, ValueError):
            same = False
        if not same:
            out.append(f"{name}={full.get(name)!r} is outside the validated lanes set ({name}={v})")
    dflt = lanes_defaults()
    for n, v in dict(lanes_hp or {}).items():
        if n in LANES_HPARAMS and not LANES_HPARAMS[n].live:
            try:
                cv = LANES_HPARAMS[n].cast(v)
            except (TypeError, ValueError):
                continue                                               # a conflict, reported there
            if cv != dflt[n]:
                out.append(f"{n}={cv!r} differs from the validated leader ({dflt[n]!r})")
    return out


def extract_lanes_hparams(blob: Mapping, *, source: str = "lanes config") -> dict[str, Any]:
    """The ``lanes_*`` knobs of a JSON object: flat, or wrapped as ``{"lanes_hparams": {...}}`` (the
    ``lanes/config.json`` layout). Strict: an unknown key raises ValueError. Values are NOT cast here
    (``lanes_conflicts`` reports bad values)."""
    if not isinstance(blob, Mapping):
        raise ValueError(f"{source}: lanes hyperparameter JSON must be an object")
    hp = blob.get("lanes_hparams", blob)
    if not isinstance(hp, Mapping):
        raise ValueError(f"{source}: 'lanes_hparams' must be an object")
    unknown = sorted(set(hp) - LANES_KNOWN_NAMES)
    if unknown:
        raise ValueError(f"{source}: unknown lanes hyperparameter(s) {unknown}")
    return dict(hp)


def lanes_ui_categories() -> dict[str, list[tuple]]:
    """``{tab: [(name, kind, default, label, help), ...]}`` for the GUI's read-only Lanes tab (same shape as
    ``src.hparams.ui_categories``)."""
    out: dict[str, list[tuple]] = {}
    for p in _REGISTRY:
        out.setdefault(p.category, []).append((p.name, p.kind, p.default, p.label, p.help))
    return out


# Registry self-check at import (pure; mirrors src/hparams.py's DIM_DEFAULTS check): every default casts
# to itself and the defaults are conflict-free against the core defaults.
for _p in _REGISTRY:
    if _p.cast(_p.default) != _p.default:
        raise RuntimeError(f"lanes registry: default of {_p.name} does not survive its own cast")
_bad = lanes_conflicts(None, None)
if _bad:
    raise RuntimeError(f"lanes registry defaults conflict: {_bad}")
del _p, _bad

__all__ = ["LANES_HPARAMS", "LANES_KNOWN_NAMES", "LANES_LIVE_EDITABLE", "LANES_OPTION_FIELDS", "CONTROLLER_FIELDS",
           "FIELD_TO_NAME", "VALIDATED_CORE", "lanes_defaults", "lanes_coerce", "resolve_lanes_hparams",
           "lanes_option_fields", "controller_fields", "core_stock_hp", "zbudget_params",
           "lanes_lines_per_activation", "lanes_conflicts", "check_lanes_conflicts", "lanes_warnings",
           "extract_lanes_hparams", "lanes_ui_categories"]
