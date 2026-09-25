"""Public experimental datasets, pulled from their upstream repositories.

Currently exposes the Olympus (https://github.com/the-matter-lab/olympus)
datasets used as real-data benchmarks for ZoMBI-Hop: ``photo_pce10``,
``photo_wf3``, ``hplc`` and ``crossed_barrel``. See ``olympus.py``.
"""
from .olympus import (  # noqa: F401
    CURATED,
    PRETTY_LABELS,
    OlympusDataset,
    available,
    fetch,
    load,
    summary,
)

__all__ = ["CURATED", "PRETTY_LABELS", "OlympusDataset", "available", "fetch", "load", "summary"]
