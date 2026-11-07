"""
benchmarks/methods/registry.py
==============================
Name -> :class:`~benchmarks.methods.base.Method` subclass.

Built-in methods register themselves when :mod:`benchmarks.methods` is imported.
A method that lives outside this package can be named by import path instead of
being registered — ``my_pkg.my_module:MyMethod`` or ``path/to/file.py:MyMethod``
— so trying a new optimiser in a sweep needs no edit here.
"""

from __future__ import annotations

import importlib
import importlib.util
import os

from .base import Method

METHODS: dict[str, type[Method]] = {}


def register(cls: type[Method]) -> type[Method]:
    """Class decorator: make ``cls`` available as ``cls.name``."""
    if not (isinstance(cls, type) and issubclass(cls, Method)):
        raise TypeError(f"@register expects a Method subclass, got {cls!r}")
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no `name`")
    if cls.name in METHODS and METHODS[cls.name] is not cls:
        raise ValueError(f"method name {cls.name!r} is already registered by "
                         f"{METHODS[cls.name].__module__}.{METHODS[cls.name].__name__}")
    METHODS[cls.name] = cls
    return cls


def _load_ref(ref: str) -> type[Method]:
    target, _, attr = ref.rpartition(":")
    if target.endswith(".py"):
        from ._paths import REPO_ROOT

        path = target if os.path.isabs(target) else os.path.join(REPO_ROOT, target)
        spec = importlib.util.spec_from_file_location(
            f"_bench_method_{abs(hash(path))}", path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"cannot load method module {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    cls = getattr(module, attr)
    if not (isinstance(cls, type) and issubclass(cls, Method)):
        raise TypeError(f"{ref} is not a Method subclass")
    return cls


def get_method(name: str) -> type[Method]:
    """The class registered as ``name``, or loaded from ``module:Class``."""
    if name in METHODS:
        return METHODS[name]
    if ":" in name:
        return _load_ref(name)
    raise KeyError(f"unknown method {name!r}; registered: {sorted(METHODS)} "
                   "(or pass 'module.path:ClassName' / 'file.py:ClassName')")


def make_method(name: str, config: dict | None = None, *, seed: int = 0,
                device: str = "cpu") -> Method:
    return get_method(name)(config, seed=seed, device=device)


def available_methods() -> list[str]:
    return sorted(METHODS)
