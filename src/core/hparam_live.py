"""
src/core/hparam_live.py
=======================
Manual hyperparameter adjustment for an in-flight ZoMBI-Hop run.

This is an operator override channel, not an automated tuner: a human watching a
run decides to change a value, and it takes effect without stopping the run.
(Automated hyperparameter search lives in ``optimize/`` — unrelated to this.)

The GUI (or any other process) drops a JSON file into the run directory; the
optimiser picks it up at the top of its next iteration — i.e. between measured
LineBO lines — and applies it to the live objects. This works identically for a
synthetic run (GUI worker thread) and a hardware run (``scripts/main.py``
subprocess), because both drive the same ``ZoMBIHop.run`` loop and share the run
directory. No IPC, no restart, no lost data.

Protocol
--------
``<run_dir>/hparams_override.json``   written by the UI  → consumed & deleted by
                                     the optimiser (a request, applied once).
``<run_dir>/hparams_effective.json``  written by the optimiser → the values
                                     currently in force (what the UI prefills).
``<run_dir>/hparams_override_rejected.json``  written by the optimiser when it
                                     refuses a request → read & deleted by the UI,
                                     which must surface it as an error. A refused
                                     request is never applied in part.

Why a registry
--------------
The editable values do not live in one place: ``DataHandler`` owns most of them,
``GPSimplex`` keeps *its own copies* of the acquisition settings (taken at
construction, under different names), and a handful are plain ``ZoMBIHop``
attributes. Writing only to the DataHandler would silently do nothing for e.g.
``ucb_beta``, so each name maps to *every* live location that must be updated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from ..hparams import coerce as _coerce_registered, conflicts, live_targets

OVERRIDE_FILENAME  = "hparams_override.json"
EFFECTIVE_FILENAME = "hparams_effective.json"
REJECTED_FILENAME  = "hparams_override_rejected.json"

# name → ((owner, attribute), …). owner: "dh" = DataHandler, "gp" = GPSimplex,
# "zombi" = the ZoMBIHop instance. The first entry is the canonical source when
# reporting the effective value. Declared per knob in src/hparams.py.
_TARGETS: dict[str, tuple[tuple[str, str], ...]] = live_targets()

#: Every hyperparameter that can be changed while a run is in flight.
LIVE_EDITABLE_NAMES: frozenset[str] = frozenset(_TARGETS)


def coerce(name: str, value: Any) -> Any:
    """Validate/cast one hyperparameter. Raises ValueError if unusable."""
    if name not in _TARGETS:
        raise ValueError(f"{name!r} cannot be changed while a run is in flight")
    return _coerce_registered(name, value)


def _owners(zombi) -> dict[str, Any]:
    return {
        "zombi": zombi,
        "dh":    getattr(zombi, "data_handler", None),
        "gp":    getattr(zombi, "gp_handler", None),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(str(tmp), str(path))  # atomic — readers never see a partial file


def _run_dir(zombi) -> Optional[Path]:
    rd = getattr(getattr(zombi, "data_handler", None), "run_dir", None)
    return Path(rd) if rd is not None else None


def current_values(zombi) -> dict[str, Any]:
    """The live value of every editable hyperparameter, from its canonical owner."""
    owners = _owners(zombi)
    out: dict[str, Any] = {}
    for name, targets in _TARGETS.items():
        owner_key, attr = targets[0]
        obj = owners.get(owner_key)
        if obj is not None and hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (int, float, str, bool)):
                out[name] = val
    return out


def write_effective(zombi) -> None:
    """Publish the values currently in force so the UI can prefill from them."""
    rd = _run_dir(zombi)
    if rd is None:
        return
    try:
        _atomic_write_json(rd / EFFECTIVE_FILENAME, current_values(zombi))
    except Exception:
        pass  # never let bookkeeping kill a run


def read_effective(run_dir) -> dict:
    """Read a run's in-force hyperparameters; {} if not published yet."""
    try:
        data = json.loads((Path(run_dir) / EFFECTIVE_FILENAME).read_text())
        return {k: v for k, v in data.items() if k in _TARGETS} if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_override(run_dir, hp: dict) -> dict:
    """
    Request that ``hp`` be applied to the run in ``run_dir`` at its next
    iteration. Values are validated here so the UI reports errors immediately
    rather than the optimiser discovering them mid-run. Returns the coerced dict.
    """
    clean = {name: coerce(name, val) for name, val in hp.items()}
    # Check the configuration the run would END UP with, not the edit alone:
    # lowering max_iterations is fine on its own and fatal next to
    # min_iters_per_zoom=3. The optimiser re-checks on apply (its own values are
    # authoritative); this is so the UI can refuse before writing anything.
    # Without the run's published values there is nothing to check the edit
    # against (conflicts() would fall back to the shipped defaults, which are not
    # what this run is using), so refuse rather than pass it unchecked.
    effective = read_effective(run_dir)
    if not effective:
        raise ValueError(f"the run has not published its in-force hyperparameters "
                         f"({EFFECTIVE_FILENAME}) yet, so this change cannot be "
                         f"checked; wait for its first iteration and try again")
    problems = conflicts({**effective, **clean})
    if problems:
        raise ValueError("this change would make the run unable to declare "
                         "needles:\n  - " + "\n  - ".join(problems))
    if clean:
        # A stale refusal belongs to an earlier request; drop it so the UI
        # doesn't report it against this one.
        _unlink(Path(run_dir) / REJECTED_FILENAME)
        _atomic_write_json(Path(run_dir) / OVERRIDE_FILENAME, clean)
    return clean


def read_rejection(run_dir) -> Optional[dict]:
    """Consume the optimiser's refusal of the last request, or None if there is none.

    Returns ``{"requested": {...}, "problems": [...]}``. Deleted once read so each
    refusal is reported exactly once.
    """
    path = Path(run_dir) / REJECTED_FILENAME
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    _unlink(path)
    return data if isinstance(data, dict) else {"requested": {}, "problems": [str(data)]}


def _reject(zombi, requested: dict, problems: list[str],
            log: Optional[Callable[[str], None]]) -> None:
    """Refuse a whole override: log it and tell the requester (the UI) why."""
    if log:
        log("  [hparams] override REJECTED, nothing was applied:")
        for p in problems:
            log(f"  [hparams]   - {p}")
    rd = _run_dir(zombi)
    if rd is not None:
        try:
            _atomic_write_json(rd / REJECTED_FILENAME,
                               {"requested": requested, "problems": problems})
        except Exception as exc:
            if log:
                log(f"  [hparams] could not record the rejection for the UI: {exc}")


def apply_pending(zombi, log: Optional[Callable[[str], None]] = None) -> list[str]:
    """
    Consume ``hparams_override.json`` if present and apply it to the live
    optimiser. Called at the top of each iteration; returns a human-readable
    description of each change (empty when there was nothing to do).

    The override file is deleted once read, so a request applies exactly once.
    A request is applied whole or not at all: an unknown name, a bad value or a
    resulting conflict refuses all of it (the run keeps going on its current
    values) and the reason goes to ``REJECTED_FILENAME`` for the UI to report.
    """
    rd = _run_dir(zombi)
    if rd is None:
        return []
    path = rd / OVERRIDE_FILENAME
    try:
        if not path.exists():
            return []
        raw = json.loads(path.read_text())
    except Exception as exc:
        _unlink(path)
        _reject(zombi, {}, [f"could not read {OVERRIDE_FILENAME}: {exc}"], log)
        return []

    # Remove first: if applying raises, we still must not re-apply on every
    # iteration forever.
    _unlink(path)

    if not isinstance(raw, dict):
        _reject(zombi, {}, [f"{OVERRIDE_FILENAME} is not a JSON object"], log)
        return []

    # All-or-nothing: applying half of an edit that only makes sense as a whole
    # (e.g. max_iterations and min_iters_per_zoom lowered together) could strand
    # the run in an impossible state between lines, and silently dropping one
    # value leaves the operator believing a change took effect when it did not.
    clean: dict[str, Any] = {}
    problems: list[str] = []
    for name, value in raw.items():
        try:
            clean[name] = coerce(name, value)
        except (TypeError, ValueError) as exc:
            problems.append(f"{name}={value!r}: {exc}")
    if not problems:
        problems = conflicts({**current_values(zombi), **clean})
    if problems:
        _reject(zombi, raw, problems, log)
        return []

    owners = _owners(zombi)
    changes: list[str] = []
    for name, new in clean.items():
        old = None
        applied = False
        for owner_key, attr in _TARGETS[name]:
            obj = owners.get(owner_key)
            if obj is None:
                continue
            if not applied:
                old = getattr(obj, attr, None)
            setattr(obj, attr, new)
            applied = True
        if applied and old != new:
            changes.append(f"{name}: {old} → {new}")
        elif applied:
            changes.append(f"{name}: {new} (unchanged)")

    if changes:
        # Persist so the new values survive a resume: DataHandler.load_state()
        # restores its hyperparameters from config.json.
        try:
            zombi.data_handler._save_config()
        except Exception as exc:
            if log:
                log(f"  [hparams] warning: could not persist to config.json: {exc}")
        write_effective(zombi)

    return changes


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
