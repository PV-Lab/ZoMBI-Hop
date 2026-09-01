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

OVERRIDE_FILENAME  = "hparams_override.json"
EFFECTIVE_FILENAME = "hparams_effective.json"

# name → ((owner, attribute), …). owner: "dh" = DataHandler, "gp" = GPSimplex,
# "zombi" = the ZoMBIHop instance. Order matters only for reads (first entry is
# the canonical source when reporting the effective value).
_TARGETS: dict[str, tuple[tuple[str, str], ...]] = {
    # ── acquisition optimisation (GPSimplex renames these) ────────────────
    "nat_grad_step":      (("dh", "nat_grad_step"),      ("gp", "nat_grad_step")),
    "nat_grad_max_steps": (("dh", "nat_grad_max_steps"), ("gp", "nat_grad_max_steps")),
    "n_restarts":         (("dh", "n_restarts"),         ("gp", "num_restarts")),
    "raw":                (("dh", "raw"),                ("gp", "raw_samples")),
    # ── acquisition function ──────────────────────────────────────────────
    "acquisition_type":   (("dh", "acquisition_type"),   ("gp", "acquisition_type")),
    "ucb_beta":           (("dh", "ucb_beta"),           ("gp", "ucb_beta")),
    "input_noise":        (("dh", "input_noise"),),
    # ── zoom resolution floor (DataHandler-owned) ─────────────────────────
    # input_noise_threshold_mult sets the minimum width of any zoom box, and so
    # bounds how deep the basin flood fill can zoom before the printer's own
    # resolution makes a tighter box meaningless.
    "input_noise_threshold_mult":  (("dh", "input_noise_threshold_mult"),),
    "max_gp_points":               (("dh", "max_gp_points"),),
    # ── point paring (DataHandler-owned) ──────────────────────────────────
    "paring_spatial_halfnoise":  (("dh", "paring_spatial_halfnoise"),),
    "paring_y_noise_multiplier": (("dh", "paring_y_noise_multiplier"),),
    # ── basin flood fill / zoom / needle (ZoMBIHop-owned) ─────────────────
    "max_lines_per_activation":     (("zombi", "max_lines_per_activation"),),
    "flood_ci_z":                   (("zombi", "flood_ci_z"),),
    "flood_k":                      (("zombi", "flood_k"),),
    "zoom_volume_fraction":         (("zombi", "zoom_volume_fraction"),),
    "needle_move_tol":              (("zombi", "needle_move_tol"),),
    "needle_ci_tol":                (("zombi", "needle_ci_tol"),),
    "needle_ci_bootstrap":          (("zombi", "needle_ci_bootstrap"),),
    # ── penalty (ZoMBIHop-owned) ──────────────────────────────────────────
    "max_penalty_radius":           (("zombi", "max_penalty_radius"),),
    "ellipsoid_drop_fraction":      (("zombi", "ellipsoid_drop_fraction"),),
    "ellipsoid_eigenvalue_floor":   (("zombi", "ellipsoid_eigenvalue_floor"),),
    "bounds_shrink_factor":         (("zombi", "bounds_shrink_factor"),),
    "min_axis_noise_mult":          (("zombi", "min_axis_noise_mult"),),
}


def _acq_cast(v: Any) -> str:
    s = str(v).strip().lower()
    if s not in ("ucb", "ei"):
        raise ValueError(f"acquisition_type must be 'ucb' or 'ei', got {v!r}")
    return s


def _positive_int(v: Any) -> int:
    i = int(v)
    if i < 1:
        raise ValueError(f"must be >= 1, got {i}")
    return i


def _positive_float(v: Any) -> float:
    f = float(v)
    if not f > 0:
        raise ValueError(f"must be > 0, got {f}")
    return f


def _nonneg_float(v: Any) -> float:
    f = float(v)
    if f < 0:
        raise ValueError(f"must be >= 0, got {f}")
    return f


_CASTS: dict[str, Callable[[Any], Any]] = {
    "nat_grad_step":      _positive_float,
    "nat_grad_max_steps": _positive_int,
    "n_restarts":         _positive_int,
    "raw":                _positive_int,
    "acquisition_type":   _acq_cast,
    "ucb_beta":           _nonneg_float,
    "input_noise":        _positive_float,
    "input_noise_threshold_mult":  _nonneg_float,
    "max_gp_points":               _positive_int,
    "paring_spatial_halfnoise":    _nonneg_float,
    "paring_y_noise_multiplier":   _nonneg_float,
    "max_lines_per_activation":     _positive_int,
    "flood_ci_z":                   _positive_float,
    "flood_k":                      _positive_int,
    "zoom_volume_fraction":         _positive_float,
    "needle_move_tol":              _positive_float,
    "needle_ci_tol":                _nonneg_float,
    "needle_ci_bootstrap":          _positive_int,
    "max_penalty_radius":           _positive_float,
    "ellipsoid_drop_fraction":      _nonneg_float,
    "ellipsoid_eigenvalue_floor":   _positive_float,
    "bounds_shrink_factor":         _positive_float,
    "min_axis_noise_mult":          _nonneg_float,
}

#: Every hyperparameter that can be changed while a run is in flight.
LIVE_EDITABLE_NAMES: frozenset[str] = frozenset(_TARGETS)


def coerce(name: str, value: Any) -> Any:
    """Validate/cast one hyperparameter. Raises ValueError if unusable."""
    if name not in _TARGETS:
        raise ValueError(f"{name!r} cannot be changed while a run is in flight")
    return _CASTS[name](value)


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
    if clean:
        _atomic_write_json(Path(run_dir) / OVERRIDE_FILENAME, clean)
    return clean


def apply_pending(zombi, log: Optional[Callable[[str], None]] = None) -> list[str]:
    """
    Consume ``hparams_override.json`` if present and apply it to the live
    optimiser. Called at the top of each iteration; returns a human-readable
    description of each change (empty when there was nothing to do).

    The override file is deleted once read, so a request applies exactly once.
    Bad values are skipped individually — one typo must not abort the run.
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
        if log:
            log(f"  [hparams] could not read override file: {exc}")
        _unlink(path)
        return []

    # Remove first: if applying raises, we still must not re-apply on every
    # iteration forever.
    _unlink(path)

    if not isinstance(raw, dict):
        if log:
            log("  [hparams] override file is not a JSON object; ignored.")
        return []

    owners = _owners(zombi)
    changes: list[str] = []
    for name, value in raw.items():
        if name not in _TARGETS:
            if log:
                log(f"  [hparams] ignoring unknown/not-live-tunable {name!r}")
            continue
        try:
            new = coerce(name, value)
        except (TypeError, ValueError) as exc:
            if log:
                log(f"  [hparams] ignoring {name}={value!r}: {exc}")
            continue
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
