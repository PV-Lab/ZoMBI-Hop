"""
warm_start/dynamic_hparams.py
=============================
The DYNAMIC warm-start method: alternate between two hyperparameter sets every
``period`` ZoMBI-Hop iterations *inside a single run*.

The static method deploys the warm-start-tuned hyperparameters and never changes
them. The dynamic method starts from those same tuned hyperparameters but every
``period`` (default 10) iterations swaps to the current production hyperparameters
(``BEST_HPARAMS.md``) and back, so the run spends alternating blocks exploring with
the warm-tuned settings and consolidating with the production settings.

How it hooks in
---------------
``ZoMBIHop.run`` already calls ``apply_pending_hparams(zombi=self, ...)`` at the top
of every iteration (the live operator-override channel in
``src/core/hparam_live.py``). We monkeypatch that module-level name so our wrapper
runs first each iteration: it counts iterations, and at each ``period`` boundary
applies the block's hyperparameter set to the live optimiser using the *same*
``_TARGETS`` registry the override channel uses (DataHandler / GPSimplex / ZoMBIHop
copies are all updated). Then it delegates to the original function so a genuine
operator override still works.

Applying directly (not via the override JSON file) means it works whether or not
the run has a ``run_dir`` — the fixed-hparam re-evaluation in ``optimize/evaluate.py``
constructs ZoMBIHop with ``checkpoint_dir=None``, so there is no run dir to drop a
file into.

Usage
-----
    from warm_start import dynamic_hparams as dyn
    dyn.arm(warm_hp, best_hp, period=10)   # block 0 -> warm, block 1 -> best, ...
    optimizer.run(...)                     # any ZoMBIHop run in this process
    dyn.disarm()

Only one schedule is active per process at a time; ``arm`` resets the per-run
iteration counters, so re-arming before each repeat is correct.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from weakref import WeakKeyDictionary

# Per-ZoMBIHop iteration bookkeeping: {zombi -> {"count": int, "block_parity": int}}.
# WeakKey so finished runs are collected and a fresh run starts clean.
_STATE: "WeakKeyDictionary[Any, dict]" = WeakKeyDictionary()

# (warm_hp, best_hp, period) or None when disarmed.
_SCHEDULE: Optional[tuple[dict, dict, int]] = None

# The original hparam_live.apply_pending, captured on first arm().
_ORIG_APPLY: Optional[Callable] = None


def _apply_set(zombi, hp: dict, log: Optional[Callable[[str], None]]) -> list[str]:
    """Push every live-editable key of ``hp`` onto the live optimiser objects.

    Mirrors the write path of ``hparam_live.apply_pending`` but takes the values
    from ``hp`` directly instead of a JSON file. Non-live-editable keys (none of
    the tuned set, in practice) are skipped; a bad value raises, since ``arm``
    already validated both sets and a silent skip would run a mixed schedule.
    """
    from src.core import hparam_live as hl

    owners = hl._owners(zombi)
    changes: list[str] = []
    for name, value in hp.items():
        targets = hl._TARGETS.get(name)
        if not targets:
            continue
        new = hl.coerce(name, value)
        applied = False
        for owner_key, attr in targets:
            obj = owners.get(owner_key)
            if obj is None:
                continue
            setattr(obj, attr, new)
            applied = True
        if applied:
            changes.append(f"{name}={new}")
    # Persist so a resume restores the block's values (best-effort; never fatal).
    try:
        zombi.data_handler._save_config()
        hl.write_effective(zombi)
    except Exception:
        pass
    return changes


def _wrapper(zombi, log: Optional[Callable[[str], None]] = None) -> list[str]:
    """Iteration-top hook: run the operator-override channel, then the schedule."""
    changes = list(_ORIG_APPLY(zombi, log=log)) if _ORIG_APPLY is not None else []
    sched = _SCHEDULE
    if sched is not None:
        warm_hp, best_hp, period = sched
        st = _STATE.get(zombi)
        if st is None:
            st = _STATE[zombi] = {"count": 0, "block_parity": None}
        parity = (st["count"] // period) % 2  # 0 -> warm block, 1 -> best block
        if parity != st["block_parity"]:
            hp = warm_hp if parity == 0 else best_hp
            applied = _apply_set(zombi, hp, log)
            if log:
                which = "WARM-tuned" if parity == 0 else "BEST_HPARAMS"
                log(f"  [dynamic] iter {st['count']}: block -> {which} "
                    f"({len(applied)} hparam(s))")
            st["block_parity"] = parity
            changes.append(f"dynamic->{'warm' if parity == 0 else 'best'}")
        st["count"] += 1
    return changes


def _check_schedule(warm_hp: dict, best_hp: dict) -> None:
    """Raise ``HyperparameterConflict`` if either block would be impossible.

    The run is constructed with ``warm_hp``; each block then overwrites only the
    live-editable keys of its set, so a value from the other set can persist into
    it. Check the configuration in force during each block, not the sets alone.
    """
    from src.core import hparam_live as hl
    from src.hparams import check_conflicts

    live = lambda hp: {k: v for k, v in hp.items() if k in hl._TARGETS}
    best_block = {**warm_hp, **live(best_hp)}
    check_conflicts(warm_hp, context="dynamic schedule, WARM block")
    check_conflicts(best_block, context="dynamic schedule, BEST_HPARAMS block")
    check_conflicts({**best_block, **live(warm_hp)},
                    context="dynamic schedule, WARM block after a BEST block")


def arm(warm_hp: dict, best_hp: dict, period: int = 10) -> None:
    """Install the alternating schedule for every ZoMBIHop run in this process."""
    global _SCHEDULE, _ORIG_APPLY
    import src.core.zombihop as zh

    if int(period) < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    _check_schedule(warm_hp, best_hp)
    if _ORIG_APPLY is None:
        _ORIG_APPLY = zh.apply_pending_hparams
    _SCHEDULE = (dict(warm_hp), dict(best_hp), int(period))
    _STATE.clear()
    zh.apply_pending_hparams = _wrapper


def disarm() -> None:
    """Remove the schedule and restore the original override channel."""
    global _SCHEDULE
    import src.core.zombihop as zh

    if _ORIG_APPLY is not None:
        zh.apply_pending_hparams = _ORIG_APPLY
    _SCHEDULE = None
    _STATE.clear()


def is_armed() -> bool:
    return _SCHEDULE is not None
