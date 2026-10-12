"""
Re-export of the shipped hyperparameter defaults, kept for existing imports.

The values live in ``src/hparams.py`` (registry + untuned defaults) and
``optimize/hparams/6d_ensemble.json`` (tuned values). Do not add values here.
"""

from __future__ import annotations

from src.hparams import (  # noqa: F401
    ALL_DEFAULTS,
    DEFAULT_HPARAMS,
    DEFAULT_HPARAMS_PROVENANCE,
    DEFAULT_INPUT_NOISE,
)
