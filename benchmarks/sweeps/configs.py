"""
benchmarks/sweeps/configs.py
============================
The configuration every (method, dimension) of a campaign runs, resolved once at
plan time and frozen into the manifest.

* ``zombi_hop`` gets ``{"hparams": <file>}`` per dimension from
  :mod:`benchmarks.sweeps.hparams` (overridable with ``--hparams DIM=path``), and
  ``sampling="point"``: in a sweep every method measures one point per call (see
  ``POINTWISE.md``). The manifest also records ``resolved_hparams``, the values it
  actually runs (the hparams file plus the ``top_m_points`` floor).
* Every other method gets its class ``defaults``, the same at every dimension.

Either can be overridden for the whole campaign:

    --method-config turbo=path/to/turbo.json      a JSON dict of config keys
    --method-set    turbo.n_trust_regions=5       one key; value parsed as JSON
    --method-set    gp_bo.acquisition=qucb        (bare words stay strings)

Overrides are validated by constructing the method at plan time, so a typo in a key
stops the plan instead of silently running the default on a worker hours later.
The manifest stores the FULL merged config — defaults included — so what a cell ran
is readable without knowing what the defaults were when it was planned.
"""

from __future__ import annotations

import json
import os

from ._paths import REPO_ROOT, ensure_paths

ensure_paths()

from benchmarks.methods import get_method  # noqa: E402

from .hparams import hparams_for_dim  # noqa: E402

ZOMBI = "zombi_hop"


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_method_overrides(config_files: list[str] | None,
                           sets: list[str] | None) -> dict[str, dict]:
    """``--method-config`` / ``--method-set`` -> ``{method: {key: value}}``.

    Files are applied first, then single keys, so ``--method-set`` wins.
    """
    out: dict[str, dict] = {}
    for raw in config_files or []:
        name, sep, path = raw.partition("=")
        if not sep:
            raise ValueError(f"--method-config {raw!r} is not NAME=path.json")
        path = path.strip()
        if not os.path.isabs(path):
            path = os.path.join(REPO_ROOT, path) if not os.path.isfile(path) else path
        with open(path) as f:
            blob = json.load(f)
        if not isinstance(blob, dict):
            raise ValueError(f"--method-config {raw!r}: file must hold a JSON object")
        out.setdefault(name.strip(), {}).update(blob)
    for raw in sets or []:
        lhs, sep, value = raw.partition("=")
        name, dot, key = lhs.partition(".")
        if not (sep and dot):
            raise ValueError(f"--method-set {raw!r} is not NAME.key=value")
        out.setdefault(name.strip(), {})[key.strip()] = _parse_value(value.strip())
    return out


def method_names(refs: list[str]) -> dict[str, str]:
    """``{name: ref}`` for the ``--methods`` entries.

    An entry is a registered name or a ``module:Class`` / ``file.py:Class`` ref; a
    campaign files every cell under the class's ``name`` (a ref would put slashes
    in directory names) and the manifest keeps the ref so a worker can import it.
    A relative ``.py`` path is made absolute here, since workers run from the repo
    root and not from wherever ``plan`` was invoked.
    """
    out: dict[str, str] = {}
    for ref in refs:
        target, sep, attr = ref.rpartition(":")
        if sep and target.endswith(".py") and not os.path.isabs(target) \
                and os.path.isfile(target):
            ref = f"{os.path.abspath(target)}:{attr}"
        name = get_method(ref).name
        if name in out:
            raise ValueError(f"two --methods entries resolve to the name {name!r}")
        out[name] = ref
    return out


def resolve_method_configs(refs: list[str], dims: list[int], *,
                           zombi_hparam_files: dict[int, str] | None = None,
                           overrides: dict[str, dict] | None = None) -> dict:
    """``{name: {str(dim): {"config", "source", "is_stand_in"}}}`` for the manifest.

    ``refs`` are ``--methods`` entries; overrides are keyed by method NAME.
    """
    overrides = overrides or {}
    names = method_names(refs)
    stray = sorted(set(overrides) - set(names))
    if stray:
        raise ValueError(f"config overrides given for method(s) not in --methods: {stray}")
    out: dict[str, dict] = {}
    for name, ref in names.items():
        cls = get_method(ref)
        per_dim = {}
        for dim in sorted(set(int(d) for d in dims)):
            override = dict(overrides.get(name, {}))
            if name == ZOMBI:
                override.setdefault("sampling", "point")
            if name == ZOMBI and "hparams" not in override:
                rec = hparams_for_dim(dim, zombi_hparam_files)
                override["hparams"] = rec["hparams"]
                source = f"{rec['path']} ({rec['provenance']})"
                stand_in = rec["is_stand_in"]
            else:
                source = "defaults" + (" + overrides" if override else "")
                stand_in = False
            # Validates the keys (unknown -> TypeError) and yields the merged config.
            method = cls(override)
            per_dim[str(dim)] = {"config": method.config, "source": source,
                                 "is_stand_in": bool(stand_in)}
            if name == ZOMBI:
                per_dim[str(dim)]["resolved_hparams"] = method.resolved_hparams(dim)
        out[name] = per_dim
    return out
