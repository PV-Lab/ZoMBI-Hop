"""
benchmarks/ablations/landscapes.py
==================================
Where the ablation campaign gets its objective functions.

The ablations are about the OPTIMISER, not about any one dataset, so nothing here
is dimensionality- or dataset-specific. A campaign names a factory by string
(``--landscape ensemble``, ``--landscape module:path/to/mine.py:build``) and the
harness asks it for landscape *i*; the factory decides what that means.

The contract
------------
A factory is any object satisfying :class:`LandscapeFactory`:

* ``spec()`` — a JSON-serialisable description, recorded in the manifest so a
  finished campaign says exactly what it ran on.
* ``build(index)`` — ``(LandscapeSpec, ensemble_config | None)`` for landscape
  ``index``. The second element is handed straight to
  ``run_mobo.run_single_trial(ensemble_config=…)``; it is not None only for the
  re-randomised ``Ensemble`` objective, whose ``LandscapeSpec`` is a template that
  gets reseeded per landscape.
* ``dim`` / ``n_available`` — the simplex dimension, and how many DISTINCT
  landscapes the factory can produce (``None`` = unbounded). A fixed surface such
  as ``warmgp`` reports 1, and the campaign refuses to plan five "different"
  landscapes that are all the same surface; repeats are the right knob there.

Adding your own
---------------
When the real dataset arrives, write a module exposing a zero-argument builder and
point the campaign at it — no edit to this file is needed::

    # my_landscapes.py
    def build(dim: int = 6, seed: int = 0, time_limit_hours: float = 0.5):
        return MyFactory(...)          # anything satisfying LandscapeFactory

    $ python -m benchmarks.ablations plan \\
          --landscape module:my_landscapes.py:build --dim 6 ...

Keyword arguments come from ``--landscape-arg k=v`` and are parsed as JSON when
possible, so ``--landscape-arg seed=3`` arrives as an int.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._paths import REPO_ROOT, ensure_paths

ensure_paths()

from mobo_landscapes import LandscapeSpec  # noqa: E402


@runtime_checkable
class LandscapeFactory(Protocol):
    """What the campaign needs from a source of objective functions."""

    dim: int
    kind: str
    n_available: int | None

    def spec(self) -> dict: ...

    def build(self, index: int) -> tuple[LandscapeSpec, dict | None]: ...


# ─── Ensemble: the re-randomised default ─────────────────────────────────────────

@dataclass
class EnsembleFactory:
    """The layered ``synthetic_data.ensemble.Ensemble`` objective, one draw per index.

    This is the campaign default because it is the only source in the repo that
    yields an unbounded supply of *statistically comparable but distinct* multi-optima
    landscapes at any dimension, which is what an ablation needs: a difference
    between two arms should be attributable to the arms, not to one arm drawing an
    easier surface. ``(seed, index)`` fixes a landscape exactly, so every arm is
    handed a byte-identical objective for a given index and the comparison is paired.

    The heavy ``LandscapeSpec`` (including the dim-3 render grid) is built once and
    reused; only the cheap per-index config varies, and ``run_single_trial``
    reseeds the objective from it.
    """

    dim: int
    seed: int = 0
    optima_margin: float = 0.2
    time_limit_hours: float | None = 0.5
    kind: str = "ensemble"
    n_available: int | None = None  # Sobol' sequence: effectively unbounded

    def __post_init__(self) -> None:
        self._base: LandscapeSpec | None = None

    def spec(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "seed": self.seed,
                "optima_margin": self.optima_margin,
                "time_limit_hours": self.time_limit_hours}

    def _base_spec(self) -> LandscapeSpec:
        if self._base is None:
            import run_mobo as rm

            self._base = rm.build_ensemble_landscape(
                self.dim, optima_margin=self.optima_margin, seed=self.seed,
                time_limit_hours=self.time_limit_hours)
        return self._base

    def build(self, index: int) -> tuple[LandscapeSpec, dict]:
        from synthetic_data.ensemble import random_ensemble_config

        cfg = random_ensemble_config(self.dim, index=int(index), seed=int(self.seed),
                                     optima_margin=self.optima_margin)
        return self._base_spec(), cfg


# ─── Synthetic oracles ───────────────────────────────────────────────────────────

@dataclass
class SyntheticFactory:
    """A fixed analytic oracle from ``synthetic_data/oracles.py``.

    ``index`` selects the oracle's own seed, so successive indices are genuinely
    different surfaces of the same family (unlike ``warmgp``/``fullgp`` below).
    """

    dim: int
    oracle: str = "ackley"
    layout: str = "2"
    seed: int = 42
    variant: str = "layout"
    time_limit_hours: float | None = 0.5
    kind: str = "synthetic"
    n_available: int | None = None

    def spec(self) -> dict:
        return {"kind": self.kind, "dim": self.dim, "oracle": self.oracle,
                "layout": self.layout, "seed": self.seed, "variant": self.variant,
                "time_limit_hours": self.time_limit_hours}

    def build(self, index: int) -> tuple[LandscapeSpec, None]:
        import run_mobo as rm

        spec = rm.build_synthetic_landscape(
            self.oracle, self.dim, self.layout,
            seed=int(self.seed) + int(index), variant=self.variant,
            time_limit_hours=self.time_limit_hours)
        return spec, None


# ─── Warm-start / full-run GP surfaces (fixed; one landscape each) ───────────────

@dataclass
class GPSurfaceFactory:
    """The real-campaign GP surfaces: ``warmgp`` (warm-start lines) or ``fullgp``.

    These are DETERMINISTIC — one campaign, one surface — so ``n_available`` is 1.
    Variation for a CI band has to come from repeats, which is the honest thing
    anyway: on a fixed landscape the spread is purely the optimiser's.
    """

    dim: int
    which: str = "warmgp"  # "warmgp" | "fullgp"
    seed: int = 0
    time_limit_hours: float | None = 0.5
    n_available: int | None = 1

    @property
    def kind(self) -> str:
        return self.which

    def spec(self) -> dict:
        return {"kind": self.which, "dim": self.dim, "seed": self.seed,
                "time_limit_hours": self.time_limit_hours}

    def build(self, index: int) -> tuple[LandscapeSpec, None]:
        import run_mobo as rm

        builder = (rm.build_warmgp_landscape if self.which == "warmgp"
                   else rm.build_fullgp_landscape)
        return builder(self.dim, seed=int(self.seed),
                       time_limit_hours=self.time_limit_hours), None


# ─── Resolution from a campaign's ``--landscape`` string ─────────────────────────

BUILTIN_KINDS = ("ensemble", "synthetic", "warmgp", "fullgp")


def _load_module_factory(ref: str, kwargs: dict[str, Any]) -> LandscapeFactory:
    """``module:<path-or-dotted>:<attr>`` → the factory that ``<attr>(**kwargs)`` returns.

    The escape hatch for the eventual real dataset: a campaign can point at any
    module without this package learning about it. A dotted module name is imported
    normally; a path ending in ``.py`` is loaded from the file, so a one-off script
    outside the import path still works.
    """
    parts = ref.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"--landscape {ref!r} is malformed; expected "
            "'module:<path.py or dotted.module>:<factory_attr>'")
    _, target, attr = parts

    if target.endswith(".py"):
        import importlib.util

        path = target if os.path.isabs(target) else os.path.join(REPO_ROOT, target)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"landscape module not found: {path}")
        spec = importlib.util.spec_from_file_location("_ablation_landscape_mod", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    else:
        import importlib

        module = importlib.import_module(target)

    try:
        builder = getattr(module, attr)
    except AttributeError:
        raise AttributeError(
            f"{target} has no attribute {attr!r} (available: "
            f"{[a for a in dir(module) if not a.startswith('_')]})") from None

    factory = builder(**kwargs)
    for required in ("dim", "spec", "build"):
        if not hasattr(factory, required):
            raise TypeError(
                f"{ref} returned {type(factory).__name__}, which has no {required!r}; "
                "see LandscapeFactory in benchmarks/ablations/landscapes.py")
    if not hasattr(factory, "n_available"):
        # Unbounded is the safe assumption: the campaign only uses this to refuse
        # planning N distinct landscapes on a surface that has fewer.
        factory.n_available = None
    return factory


def parse_landscape_args(pairs: list[str] | None) -> dict[str, Any]:
    """``["seed=3", "oracle=ackley"]`` → ``{"seed": 3, "oracle": "ackley"}``.

    Values are parsed as JSON where possible so numbers and booleans arrive typed;
    anything that is not valid JSON stays a string, which is what makes bare words
    like ``ackley`` work without quoting them past two shells.
    """
    out: dict[str, Any] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise ValueError(f"--landscape-arg {raw!r} is not of the form key=value")
        key, _, value = raw.partition("=")
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def resolve_landscape(
    ref: str,
    *,
    dim: int,
    time_limit_hours: float | None,
    extra: dict[str, Any] | None = None,
) -> LandscapeFactory:
    """Build the factory named by a campaign's ``--landscape`` string.

    ``ref`` is one of :data:`BUILTIN_KINDS`, or ``module:<target>:<attr>`` for a
    user-supplied one. ``extra`` are the ``--landscape-arg`` overrides; for the
    module form they are passed to the builder verbatim, so a custom factory
    controls its own signature entirely.
    """
    extra = dict(extra or {})
    if ref.startswith("module:"):
        extra.setdefault("dim", dim)
        extra.setdefault("time_limit_hours", time_limit_hours)
        return _load_module_factory(ref, extra)

    if ref == "ensemble":
        return EnsembleFactory(dim=dim, time_limit_hours=time_limit_hours, **extra)
    if ref == "synthetic":
        return SyntheticFactory(dim=dim, time_limit_hours=time_limit_hours, **extra)
    if ref in ("warmgp", "fullgp"):
        return GPSurfaceFactory(dim=dim, which=ref,
                                time_limit_hours=time_limit_hours, **extra)
    raise ValueError(
        f"unknown --landscape {ref!r}; expected one of {BUILTIN_KINDS} "
        "or 'module:<path.py or dotted.module>:<factory_attr>'")
