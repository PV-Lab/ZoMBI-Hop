"""Synthetic test objectives for ZoMBI-Hop benchmarking and visualisation."""

from .ackley import (
    Ackley,
    TunableRealistic,
    VARIANTS as ACKLEY_VARIANTS,
    load_config,
    save_config,
)
from .bumps import Bumps

__all__ = [
    "Ackley", "TunableRealistic", "ACKLEY_VARIANTS",
    "load_config", "save_config",
    "Bumps",
]
