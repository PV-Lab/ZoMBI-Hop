"""
benchmarks/sweeps/hparams.py
============================
Which ZoMBI-Hop configuration each dimension of the sweep runs.

The sweep asks how robust the optimiser is to the *landscape*, so within a
dimension the configuration is held fixed — a difference between two grid cells at
the same dim is then attributable to the needle count and the basin width and
nothing else. Across dimensions the configuration changes, because a single config
would hand one dimension its own tuning and the rest somebody else's; the price is
that dim-to-dim comparisons vary the optimiser as well as the landscape, and the
summary says so.

The map
-------
============ =============================================== =========================
dim          file                                            provenance
============ =============================================== =========================
3            ``optimize/hparams/trial_112_composition.json``  archived 3d MOBO winner
                                                             (``mobo_3d_05_06_15_32``
                                                             trial 112), re-expressed
                                                             for composition space.
                                                             Seeds both
                                                             ``ensemble_mobo_3d.sbatch``
                                                             and
                                                             ``ensemble_mobo_4d.sbatch``,
                                                             and is where
                                                             ``warm_start``'s
                                                             ``REFERENCE_HPARAMS``
                                                             comes from.
4, 6, 10     ``optimize/hparams/clamped_6d/dist1c.json``      best ``dist_to_needles``
                                                             trial of the 6d ensemble
                                                             pool
                                                             (``mobo_ensemble_6d_job19202380``
                                                             trial 23), clamped into
                                                             ``HPARAM_SPACE``.
============ =============================================== =========================

Two of those assignments are stand-ins and are labelled as such in the manifest,
so nobody reads a dim-4 or dim-10 result as "the tuned configuration for that
dimension":

* **dim 4** has no tuned file in the repo. ``ensemble_mobo_4d.sbatch`` seeds its
  own search from the dim-3 trial 112, so either neighbour was defensible; the 6d
  config is used because 4 and 6 sit on the same side of the dim-3 special case
  (a 3-simplex is a triangle, and the dim-3 config is tuned against a ternary
  render grid the others do not have).
* **dim 10** has ``optimize/hparams/10d_ensemble.json``, but that file records
  ``"phase": "sobol"`` — trial 3 of the initial quasi-random sweep, not a tuned
  winner — so the 6d configuration is used instead.

``optimize/hparams/tight_6d/`` holds the same five 6d configurations re-projected
into the ``HPARAM_SPACE`` that was re-tightened on 2026-08-12, and
``ensemble_mobo_10d.sbatch`` warns against seeding a *search* from the older
``clamped_6d/`` files for exactly that reason. It does not apply here: this sweep
re-evaluates a fixed configuration rather than seeding a search, so no coordinate
is ever mapped through the space's bounds and ``dist1c.json`` runs as the numbers
it literally contains.
"""

from __future__ import annotations

import json
import os

from ._paths import REPO_ROOT

#: dim -> (path relative to the repo root, one-line provenance, is it a stand-in).
HPARAM_MAP: dict[int, tuple[str, str, bool]] = {
    3: ("optimize/hparams/trial_112_composition.json",
        "archived 3d MOBO winner (mobo_3d_05_06_15_32 trial 112), composition-space",
        False),
    4: ("optimize/hparams/clamped_6d/dist1c.json",
        "6d dist_to_needles winner (job19202380 trial 23) — no tuned 4d config exists",
        True),
    6: ("optimize/hparams/clamped_6d/dist1c.json",
        "6d dist_to_needles winner (job19202380 trial 23), clamped to HPARAM_SPACE",
        False),
    10: ("optimize/hparams/clamped_6d/dist1c.json",
         "6d config: 10d_ensemble.json is an untuned Sobol-phase trial, not a winner",
         True),
}


def load_hparams(path: str) -> dict:
    """Hyperparameters from a bare dict or a ``trial.json``-style blob.

    Same two shapes ``optimize/evaluate.py --hparams-json`` and
    ``optimize/showdown.py --configs`` already accept, so any file that works there
    works here.
    """
    with open(path) as f:
        blob = json.load(f)
    return dict(blob.get("hparams", blob))


def hparams_for_dim(dim: int, overrides: dict[int, str] | None = None) -> dict:
    """The configuration dim ``dim`` runs, with its provenance attached.

    Returns ``{"dim", "path", "provenance", "is_stand_in", "hparams"}`` — the whole
    record, not the bare dict, because the manifest has to say *why* a dimension
    ran what it ran, and a stand-in has to be visible in the summary rather than
    buried in this file.
    """
    dim = int(dim)
    overrides = overrides or {}
    if dim in overrides:
        path = overrides[dim]
        provenance, stand_in = "explicit --hparams override", False
    elif dim in HPARAM_MAP:
        rel, provenance, stand_in = HPARAM_MAP[dim]
        path = os.path.join(REPO_ROOT, rel)
    else:
        raise KeyError(
            f"no hyperparameters mapped for dim {dim}; known dims "
            f"{sorted(HPARAM_MAP)} — pass --hparams {dim}=path/to/config.json")
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"hyperparameters for dim {dim} not found: {path}")
    return {
        "dim": dim,
        "path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
        "provenance": provenance,
        "is_stand_in": bool(stand_in),
        "hparams": load_hparams(path),
    }


def parse_hparam_overrides(pairs: list[str] | None) -> dict[int, str]:
    """``["3=my3d.json", "10=my10d.json"]`` -> ``{3: "my3d.json", 10: "my10d.json"}``."""
    out: dict[int, str] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise ValueError(
                f"--hparams {raw!r} is not of the form DIM=path/to/config.json")
        key, _, value = raw.partition("=")
        out[int(key.strip())] = value.strip()
    return out


def resolve_all(dims, overrides: dict[int, str] | None = None) -> dict[int, dict]:
    """The full per-dimension map for a campaign, recorded in the manifest."""
    return {int(d): hparams_for_dim(int(d), overrides) for d in sorted(set(dims))}
