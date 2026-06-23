"""Synthetic test objectives for ZoMBI-Hop benchmarking and visualisation."""

from .ackley import (
    Ackley,
    TunableRealistic,
    VARIANTS as ACKLEY_VARIANTS,
    load_config,
    save_config,
)
from .bumps import Bumps
from .ensemble import Ensemble
from .oracles import ORACLE_CHOICES, ORACLE_EXPRESSIONS, build_oracle
from .campaign_datasets import (
    SYNTHETIC_3D_PRESETS,
    generate_campaign_files,
    load_metadata,
)

__all__ = [
    "Ackley", "TunableRealistic", "ACKLEY_VARIANTS",
    "load_config", "save_config",
    "Bumps",
    "Ensemble",
    "ORACLE_CHOICES", "ORACLE_EXPRESSIONS", "build_oracle",
    "SYNTHETIC_3D_PRESETS", "generate_campaign_files", "load_metadata",
]
