"""Building a lanes engine (DESIGN.md 2.1): ``build_lane_engine(...)`` -> ``LaneEngine`` with two ``LeaderLane``s.

    from src.lanes import build_lane_engine          # lazy import of this module
    engine = build_lane_engine(X_init_actual, X_init_expected, Y_init, bounds, preset="leader", seed=0, ...)
    engine.lanes["front"].decide(now); engine.lanes["back"].refresh(now); engine.on_result(packet, tags, now)
    engine.close()

Replaces the sandbox's ``default_hparams`` (``benchmarks/zoom_fix/lanes/lane_zombihop.py`` = LZ 178-182) and
the explore framework's lane-class assembly (``explore/xruntime.py`` = XR 55-60, 137-153: ``XLane =
FrameworkLaneMixin + ZBudgetMixin + CatchupMixin + ZoMBILane``, lanes rebuilt after the store) by a static
class built once:

    LeaderLane(CatchupMixin, DiveLane)
        MRO: LeaderLane -> CatchupMixin -> DiveLane (zbudget folded in)
        = the sandbox MRO FrameworkLaneMixin -> ZBudgetMixin -> CatchupMixin -> ZoMBILane for every method
          the leader reaches: ``_choose`` (catch-up tick, then the base), ``_zoom_in`` / ``_try_needle``
          (catch-up hold check, then the base), ``_verdict`` / ``_budget_transition`` (the per-zoom copies,
          no super). ``FrameworkLaneMixin`` only reset ``_x_pred`` (read by the bridge-aware line gate; no
          bridges here), so it has no counterpart.

The sandbox built the stock lanes first (activation ids 0, 1), reset the counter and rebuilt them as XLanes
(ids 0, 1 again; XR 143-150): lane construction only allocates activation ids, so building the leader lanes
once, front then back, is the same state.

Rules (REVIEW F3, F11): ``build_lane_engine`` never calls ``torch.manual_seed`` (in the emulator the harness
seeded and consumed RNG before the build; on hardware only ``run_lanes_main`` seeds). Core hyperparameters
default to ``engine_hparams(d)`` = the hardware convention ``apply_dim_rules(defaults_for_dim(d), d)``; a
core knob outside that validated set is logged (stderr), never refused; lanes knobs go through
``check_lanes_conflicts`` (refused).
"""
from __future__ import annotations

import sys
from typing import Mapping, Optional

from src.hparams import apply_dim_rules, defaults_for_dim
from src.lanes.budget import validate_against_stock, validate_zbudget_params
from src.lanes.catchup import CatchupMixin, catchup_top_m, parse_zooms
from src.lanes.hparams import check_lanes_conflicts, lanes_warnings
from src.lanes.lane import DiveLane
from src.lanes.options import LaneOptions
from src.lanes.store import LaneEngine

__all__ = ["LeaderLane", "engine_hparams", "resolve_options", "build_lane_engine", "core_hparam_diffs"]


class LeaderLane(CatchupMixin, DiveLane):
    """The validated leader's lane: per-zoom budget with the zoom-2 floor (``DiveLane``) + catch-up."""


def engine_hparams(d: int) -> dict:
    """Core ZoMBIHop hyperparameters of a lanes engine for ``d`` optimizing dims: the hardware convention
    ``apply_dim_rules(defaults_for_dim(d), d)`` (scripts/run_zombi_main.py 1532 + 1681). Equal in force to the
    sandbox recipe ``apply_dim_rules(load_hparams_file(dup1c_custom.json), d)`` + ``input_noise`` default (LZ
    178-182) for d = 6 and 10 (tests/test_lanes_hparams_engine.py)."""
    return apply_dim_rules(defaults_for_dim(int(d)), int(d))


def resolve_options(preset: str = "leader", overrides: Optional[Mapping] = None, **extra) -> LaneOptions:
    """``LaneOptions`` of ``PRESETS[preset]`` overlaid with ``overrides`` (``lanes_*`` knobs; controller knobs
    are accepted and ignored here). ``extra``: non-registry fields (``snapshots``)."""
    return LaneOptions.from_preset(preset, overrides, **extra)


def core_hparam_diffs(hp: Mapping, d: int) -> list:
    """Core knobs in ``hp`` whose value differs from ``engine_hparams(d)`` (the validated set), as messages."""
    ref = engine_hparams(d)
    out = []
    for k in sorted(hp):
        if k in ref and hp[k] != ref[k]:
            out.append(f"core hyperparameter {k}={hp[k]!r} differs from the validated lanes set ({ref[k]!r})")
        elif k not in ref:
            out.append(f"core hyperparameter {k}={hp[k]!r} is not in the validated lanes set")
    return out


def build_lane_engine(X_init_actual, X_init_expected, Y_init, bounds, *, preset: str = "leader",
                      overrides: Optional[Mapping] = None, hparams: Optional[Mapping] = None, seed: int = 0,
                      checkpoint_dir: Optional[str] = None, run_uuid: Optional[str] = None, resume: bool = False,
                      predict=None, verbose: bool = False, defer_decision_commit: bool = False,
                      snapshots: bool = True) -> LaneEngine:
    """Build the lanes engine (DESIGN 2.1).

    ``X_init_actual`` / ``X_init_expected`` (n, d) and ``Y_init`` (n, 1) tensors and ``bounds`` (2, d), as the
    sandbox ``LaneStore`` took them (LZ 268). ``preset``: the only one is ``"leader"`` (the registry defaults
    ARE the leader). ``overrides``: ``lanes_*`` knobs (validated by ``check_lanes_conflicts``). ``hparams``: core
    ZoMBIHop knobs (None -> ``engine_hparams(d)``). ``seed``: the cover-rule RNG seed (LZ 393); never a
    ``torch.manual_seed`` here (REVIEW F3). ``checkpoint_dir=None`` with snapshots on -> a scratch dir removed by
    ``close()`` (LZ 281-287). ``defer_decision_commit``: the Phase-2 controller commits decisions itself.
    ``snapshots``: one DataHandler snapshot per ingested packet and per decision (N6; the leader ran with it).
    """
    d = int(X_init_actual.shape[1])
    hp = engine_hparams(d) if hparams is None else dict(hparams)
    lanes_hp = dict(overrides or {})
    check_lanes_conflicts(hp, lanes_hp, context="build_lane_engine")
    opts = resolve_options(preset, lanes_hp, snapshots=bool(snapshots))
    parse_zooms(opts.catchup_zooms)                       # CU 387 (validated once, at build)
    validate_zbudget_params(opts.zbudget_params())        # ZB 413
    warnings = lanes_warnings(hp, lanes_hp) + core_hparam_diffs(hp, d)
    engine = LaneEngine(X_init_actual, X_init_expected, Y_init, bounds, options=opts, hparams=hp, seed=seed,
                        checkpoint_dir=checkpoint_dir, run_uuid=run_uuid, resume=resume, predict=predict,
                        verbose=verbose, defer_decision_commit=defer_decision_commit, lane_cls=LeaderLane)
    try:
        lane = next(iter(engine.lanes.values()))
        stock = lane._hp()
        validate_against_stock(opts.zbudget_params(), stock)   # ZB 430 (on_store), on the built ZoMBIHop
        # the on_store stats of the two explore methods (ZB 423-434, CU 393-396), for the run log only; kept
        # out of engine.stats (whose 'zbudget.' / 'catchup.' keys are the counters)
        engine.build_info.update(
            preset=str(preset), overrides=lanes_hp, warnings=list(warnings), hp_stock=dict(stock),
            hp_z={f"z{k}": lane._hp_z(k) for k in range(int(stock["max_zooms"]))},
            activation_budget=int(lane._act_budget()), catchup_top_m=catchup_top_m(opts, d),
            mro=[c.__name__ for c in type(lane).__mro__])
    except BaseException:
        engine.close()
        raise
    if warnings:
        print("[lanes] build_lane_engine: outside the validated leader set (logged, not refused):\n  - "
              + "\n  - ".join(warnings), file=sys.stderr, flush=True)
    return engine
