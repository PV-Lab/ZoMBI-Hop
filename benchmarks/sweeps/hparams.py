"""
benchmarks/sweeps/hparams.py
============================
Which ZoMBI-Hop configuration each dimension of the sweep runs. (The other methods
take their configuration from their own ``defaults`` plus any ``--method-config`` /
``--method-set`` overrides — see :mod:`benchmarks.sweeps.configs`.)

Within a dimension the configuration is held fixed, so a difference between two
cells at the same dim is attributable to the landscape and nothing else. Across
dimensions it changes, because one config would hand one dimension its own tuning
and the rest somebody else's; dim-to-dim comparisons of ZoMBI-Hop therefore vary
its hyperparameters as well as the landscape, and the summary says so.

Tuned on the simplex, run on the cube
-------------------------------------
Every file below was tuned by MOBO on the *simplex* ensemble landscapes; the sweep
now runs on the unit cube. ZoMBI-Hop's length-scale hyperparameters are
dimensionless fractions of a unit-extent domain on both (``BoxDomain`` defaults to
the unit cube for exactly this reason — see ``src/utils/domain.py``), so they carry
over, but they are a transfer, not a cube-tuned optimum. The baselines run their
published defaults with no tuning at all, so neither side of the comparison was
tuned on these landscapes.

Cube dimension = simplex free dimensions
----------------------------------------
A D-component simplex has D - 1 free coordinates, so a config tuned on the
D-simplex is run on the (D - 1)-cube: the 3-simplex ("3d") config at cube dim 2,
the 6-simplex ("6d") config at cube dim 5, and so on. The default grid (2, 3, 5, 9)
is the simplex grid (3, 4, 6, 10) in free dimensions.

The map
-------
============ =============================================== =========================
cube dim     file                                            provenance
(simplex)
============ =============================================== =========================
2 (3)        ``optimize/hparams/trial_112_composition.json``  archived 3d MOBO winner
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
3, 5, 9      ``optimize/hparams/clamped_6d/dist1c.json``      best ``dist_to_needles``
(4, 6, 10)
                                                             trial of the 6d ensemble
                                                             pool
                                                             (``mobo_ensemble_6d_job19202380``
                                                             trial 23), clamped into
                                                             ``HPARAM_SPACE``.
============ =============================================== =========================

Two of those assignments are stand-ins and are labelled as such in the manifest,
so nobody reads a dim-3 or dim-9 result as "the tuned configuration for that
dimension":

* **dim 3 (4-simplex)** has no tuned file in the repo. ``ensemble_mobo_4d.sbatch`` seeds its
  own search from the dim-3 trial 112, so either neighbour was defensible; the 6d
  config is used because 4 and 6 sit on the same side of the 3-simplex special case
  (a 3-simplex is a triangle, and the dim-3 config is tuned against a ternary
  render grid the others do not have).
* **dim 9 (10-simplex)** has ``optimize/hparams/10d_ensemble.json``, but that file records
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

#: Cube dim -> (path relative to the repo root, one-line provenance, is it a
#: stand-in). Keyed by free dimensions: the D-simplex config runs at cube dim D - 1.
HPARAM_MAP: dict[int, tuple[str, str, bool]] = {
    2: ("optimize/hparams/trial_112_composition.json",
        "3-simplex MOBO winner (mobo_3d_05_06_15_32 trial 112), composition-space",
        False),
    3: ("optimize/hparams/clamped_6d/dist1c.json",
        "6-simplex dist_to_needles winner (job19202380 trial 23) — no tuned "
        "4-simplex config exists",
        True),
    5: ("optimize/hparams/clamped_6d/dist1c.json",
        "6-simplex dist_to_needles winner (job19202380 trial 23), clamped to "
        "HPARAM_SPACE",
        False),
    9: ("optimize/hparams/clamped_6d/dist1c.json",
        "6-simplex config: 10d_ensemble.json is an untuned Sobol-phase trial, not a "
        "winner",
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
    """``["2=my2d.json", "9=my9d.json"]`` -> ``{2: "my2d.json", 9: "my9d.json"}``."""
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
