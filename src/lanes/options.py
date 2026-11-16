"""Frozen options of a lanes engine (DESIGN.md section 1.2, 2.2, N4).

``LaneOptions`` is the sandbox's ``benchmarks/zoom_fix/lanes/lane_options.py`` dataclass (LO 38-147:
fields, ``to_dict``, ``label``, ``switches``, value coercion) with these changes:

* Built from the lanes registry (``src/lanes/hparams.py``): ``LaneOptions.from_hparams({lanes_*: ...})``
  / ``from_preset("leader", overrides)``. Every registry-backed field is cast by its registry validator in
  ``__post_init__`` (replaces LO's ``_coerce``). The string spec parser ``from_spec`` and the study presets
  (LO 117-192) are not moved.
* The dataclass defaults ARE the validated leader (``im+cross_cover_frac=1.01__zbudget@z2_min=3+catchup``):
  ``incmean=True``, ``cross_cover_frac=1.01``, ``z2_min=3``, catch-up ``"12"`` / 0.01 / 2.
  ``PRESETS = {"leader": {}}``.
* Added fields: the per-zoom budget ``z0_max .. z2_nc`` (ZB 79) and catch-up ``catchup_zooms / catchup_eps /
  catchup_max / catchup_top_m`` (CU 384, 88-89).
* The unvalidated switches ``bridge, confirm_revise, noface, gp_box, gate`` are NOT ported (N4). They stay
  as fields fixed to ``False`` and ``input_noise`` fixed to ``None`` (a core hparam now), because the auditor
  and ``lane_system.ZoMBILaneSystem.finish`` read them; setting any of them raises ``ValueError("... not
  ported ...")``. Their knobs (``bridge_predict``, ``revise_threshold``, ``gate_*``) are dropped.
* ``snapshots`` (engine-internal, not a registry knob) keeps its sandbox meaning and default True.

Pure module: no torch use, no side effects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Mapping, Optional

from src.lanes.hparams import (
    FIELD_TO_NAME,
    LANES_HPARAMS,
    LANES_KNOWN_NAMES,
    LANES_OPTION_FIELDS,
    lanes_coerce,
)

# LO 35 minus incmean (ported and ON in the leader): the switches that are NOT ported.
NOT_PORTED_SWITCHES = ("bridge", "confirm_revise", "noface", "gp_box", "gate")
SWITCHES = ("bridge", "confirm_revise", "noface", "gp_box", "incmean", "gate")     # LO 35 (for switches())

# lanes_* presets on top of the registry defaults; the defaults ARE the leader.
PRESETS: dict = {"leader": {}}


@dataclass(frozen=True)
class LaneOptions:
    # ── switches of the sandbox (LO 41-47): only incmean is ported ──
    bridge: bool = False
    confirm_revise: bool = False
    noface: bool = False
    gp_box: bool = False
    input_noise: Optional[float] = None
    incmean: bool = True                  # Fix 8: zoom >= 1 lines are chords through the GP-mean incumbent
    gate: bool = False

    # ── lanes core knobs (LO 60-95; defaults = the leader) ──
    rho: float = 0.7
    floor_rule: str = "graded"
    inner_margin: float = 0.02
    inner_margin_frac: float = 0.25
    count_frac: float = 0.25
    cross_cover_frac: float = 1.01        # > 1: the cross-lane cover rule never fires (the leader)
    cross_overlap_frac: float = 0.5
    ext_cap: int = 3
    refresh_relevance: bool = True
    refresh_full: bool = False
    snapshots: bool = True                # dh.take_snapshot per ingested packet / committed decision
    num_lines: int = 10

    # ── per-zoom budget (ZB 79; 0 = the core value). The leader: z2_min = 3 ──
    z0_max: int = 0
    z1_max: int = 0
    z2_max: int = 0
    z0_min: int = 0
    z1_min: int = 0
    z2_min: int = 3
    z0_nc: int = 0
    z1_nc: int = 0
    z2_nc: int = 0

    # ── catch-up (CU 384, 88-89) ──
    catchup_zooms: str = "12"
    catchup_eps: float = 0.01
    catchup_max: int = 2
    catchup_top_m: int = 0                # 0 = TOP_M by dimension (8 at 6-D, 11 at 10-D, else 8)

    def __post_init__(self):
        for f in NOT_PORTED_SWITCHES:
            if getattr(self, f) is not False:
                raise ValueError(f"LaneOptions.{f}={getattr(self, f)!r}: not ported (only the validated leader "
                                 f"is; DESIGN.md N4). It must stay False.")
        if self.input_noise is not None:
            raise ValueError(f"LaneOptions.input_noise={self.input_noise!r}: not ported (input_noise is a core "
                             f"hyperparameter; pass it in the engine's hparams)")
        for f, name in FIELD_TO_NAME.items():
            try:
                v = LANES_HPARAMS[name].cast(getattr(self, f))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"LaneOptions.{f} ({name}): {exc}") from None
            object.__setattr__(self, f, v)
        if not isinstance(self.snapshots, bool):
            raise ValueError(f"LaneOptions.snapshots must be a bool, got {self.snapshots!r}")

    # ── construction ──
    @classmethod
    def from_hparams(cls, lanes_hp: Optional[Mapping] = None, **extra) -> "LaneOptions":
        """Options from ``lanes_*`` knobs (missing ones = the leader defaults). Controller knobs are accepted
        and ignored (they belong to the Phase-2 controller config). ``extra``: non-registry fields
        (``snapshots``). Raises ValueError on an unknown name or a bad value."""
        hp = dict(lanes_hp or {})
        unknown = sorted(set(hp) - LANES_KNOWN_NAMES)
        if unknown:
            raise ValueError(f"unknown lanes hyperparameter(s) {unknown}")
        kw = {f: lanes_coerce(n, hp[n]) for n, f in LANES_OPTION_FIELDS.items() if n in hp}
        bad = sorted(set(extra) & set(FIELD_TO_NAME))
        if bad:
            raise ValueError(f"set {bad} through their lanes_* names, not as extra fields")
        kw.update(extra)
        return cls(**kw)

    @classmethod
    def from_preset(cls, preset: str = "leader", overrides: Optional[Mapping] = None, **extra) -> "LaneOptions":
        """``PRESETS[preset]`` overlaid with ``overrides`` (``lanes_*`` knobs)."""
        if preset not in PRESETS:
            raise ValueError(f"unknown lanes preset {preset!r} (known: {sorted(PRESETS)})")
        return cls.from_hparams({**PRESETS[preset], **dict(overrides or {})}, **extra)

    def replace(self, **changes) -> "LaneOptions":
        """``dataclasses.replace`` (field names), validated again."""
        return replace(self, **changes)

    # ── views ──
    def to_dict(self) -> dict:
        return asdict(self)

    def to_hparams(self) -> dict:
        """``{lanes_* name: value}`` of the registry-backed fields (engine knobs only)."""
        return {n: getattr(self, f) for n, f in LANES_OPTION_FIELDS.items()}

    def switches(self) -> dict:
        d = {k: getattr(self, k) for k in SWITCHES}
        d["input_noise"] = self.input_noise
        return d

    def zbudget_params(self) -> dict:
        """``{"z0_max": .., ..., "z2_nc": ..}`` = the ``P`` of ``src.lanes.budget.zbudget_hp``."""
        return {f"z{k}_{p}": int(getattr(self, f"z{k}_{p}")) for p in ("max", "min", "nc") for k in (0, 1, 2)}

    def label(self) -> str:
        """'leader' for the validated leader; else 'leader+<field>=<value>+...' for every field that differs
        (field order). Logging only (LO 106-112 listed the switches that were on)."""
        base = type(self)()
        diff = [f"{fl.name}={getattr(self, fl.name)!r}" for fl in fields(self)
                if getattr(self, fl.name) != getattr(base, fl.name)]
        return "leader" if not diff else "leader+" + "+".join(diff)

    def is_leader(self) -> bool:
        return self == type(self)()


__all__ = ["LaneOptions", "PRESETS", "NOT_PORTED_SWITCHES", "SWITCHES"]
