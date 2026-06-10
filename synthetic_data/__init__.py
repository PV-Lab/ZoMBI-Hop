"""Synthetic test objectives for ZoMBI-Hop benchmarking and visualisation."""

from .ackley import (
    Ackley,
    TunableRealistic,
    VARIANTS as ACKLEY_VARIANTS,
    load_config,
    save_config,
)
from .oracles import ORACLE_CHOICES, build_oracle
from .campaign_datasets import (
    SYNTHETIC_3D_PRESETS,
    generate_campaign_files,
    load_metadata,
)

__all__ = [
    "Ackley", "TunableRealistic", "ACKLEY_VARIANTS",
    "load_config", "save_config",
    "ORACLE_CHOICES", "build_oracle",
    "SYNTHETIC_3D_PRESETS", "generate_campaign_files", "load_metadata",
]
