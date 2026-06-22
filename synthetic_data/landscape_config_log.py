"""Snapshot ``defaults.json`` and resolved oracle parameters for run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SYNTH_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SYNTH_ROOT.parent

DEFAULTS_REL_PATHS: dict[str, str] = {
    "ackley": "synthetic_data/defaults/ackley.json",
    "gaussian": "synthetic_data/gaussian/defaults.json",
    "rastrigin_ilr": "synthetic_data/rastrigin_ilr/defaults.json",
}

_DATASET_DEFAULTS_KEY: dict[str, str] = {
    "ackley3d": "ackley",
    "ackley4d": "ackley",
    "ackley10d": "ackley",
    "gaussian3d": "gaussian",
    "gaussian4d": "gaussian",
    "gaussian10d": "gaussian",
    "gaussian": "gaussian",
    "rastrigin_ilr": "rastrigin_ilr",
}


def load_defaults_file(oracle_key: str) -> tuple[str | None, dict[str, Any] | None]:
    rel = DEFAULTS_REL_PATHS.get(oracle_key)
    if rel is None:
        return None, None
    path = _REPO_ROOT / rel
    if not path.is_file():
        return rel, None
    with open(path, encoding="utf-8") as f:
        return rel, json.load(f)


def _read_json_path(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def metadata_for_csv(csv_path: str | None, metadata_path: str | None = None) -> dict[str, Any] | None:
    if metadata_path:
        meta = _read_json_path(metadata_path)
        if meta is not None:
            return meta
    if not csv_path:
        return None
    sidecar = Path(csv_path).with_name(Path(csv_path).stem + "_meta.json")
    return _read_json_path(sidecar)


def resolved_from_fn(fn: Any) -> dict[str, Any]:
    """Extract runtime oracle parameters from a built landscape callable."""
    out: dict[str, Any] = {}
    cls = type(fn).__name__

    if hasattr(fn, "dim"):
        out["dim"] = int(fn.dim)
    if hasattr(fn, "variant"):
        out["variant"] = fn.variant
    if hasattr(fn, "centers"):
        out["n_peaks"] = len(fn.centers)
    if hasattr(fn, "basin_widths") and fn.basin_widths:
        out["basin_width"] = float(fn.basin_widths[0])
    if hasattr(fn, "sigmas") and fn.sigmas:
        out["sigma"] = float(fn.sigmas[0])
        if len(set(round(s, 8) for s in fn.sigmas)) > 1:
            out["sigma_spread"] = True
    if hasattr(fn, "amplitude"):
        out["amplitude"] = float(fn.amplitude)
    if hasattr(fn, "n_optima"):
        out["n_optima"] = int(fn.n_optima)
    if hasattr(fn, "_noise_amp"):
        out["noise_amp"] = float(fn._noise_amp)
    if hasattr(fn, "_noise_freq"):
        out["noise_freq"] = float(fn._noise_freq)
    if hasattr(fn, "true_optima"):
        out["n_true_optima"] = len(fn.true_optima)
    elif hasattr(fn, "_optima"):
        out["n_true_optima"] = len(fn._optima)

    out["oracle_class"] = cls
    return out


def _defaults_key(*, dataset: str, ds: dict[str, Any]) -> str | None:
    oracle = ds.get("oracle")
    if isinstance(oracle, str) and oracle in DEFAULTS_REL_PATHS:
        return oracle
    return _DATASET_DEFAULTS_KEY.get(dataset)


def build_landscape_config_log(*, dataset: str, ds: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable config snapshot for ``rerun_config.json`` / ``metrics.json``."""
    defaults_key = _defaults_key(dataset=dataset, ds=ds)
    defaults_path, defaults = (None, None)
    if defaults_key:
        defaults_path, defaults = load_defaults_file(defaults_key)

    fn = ds.get("fn") or ds.get("fn_callable")
    resolved = resolved_from_fn(fn) if fn is not None else {}
    if ds.get("seed") is not None:
        resolved.setdefault("seed", ds["seed"])
    if ds.get("layout") is not None:
        resolved.setdefault("layout", ds["layout"])
    if ds.get("variant") is not None:
        resolved.setdefault("variant", ds["variant"])

    csv_meta = metadata_for_csv(ds.get("csv_path"), ds.get("metadata_path"))
    dataset_generation = None
    if csv_meta:
        dataset_generation = {
            k: csv_meta[k]
            for k in (
                "oracle", "dim", "layout", "seed", "maximize",
                "noise_std", "outlier_frac", "sampling", "sampling_desc",
                "n_samples", "oracle_label",
            )
            if k in csv_meta
        }

    log: dict[str, Any] = {
        "dataset": dataset,
        "defaults_oracle": defaults_key,
        "defaults_path": defaults_path,
        "defaults": defaults,
        "resolved": resolved or None,
    }
    if dataset_generation:
        log["dataset_generation"] = dataset_generation
    if csv_meta and "true_optima" in csv_meta:
        log["dataset_meta_true_optima_count"] = len(csv_meta["true_optima"])
    return log


def dataset_label_for_landscape(landscape: str, *, dim: int | None = None, oracle: str | None = None) -> str:
    if landscape == "ackley" and dim is not None:
        return f"ackley{dim}d"
    if landscape == "synthetic" and oracle:
        return oracle
    if landscape == "rf":
        return "RF"
    return oracle or landscape


def format_landscape_config_summary(log: dict[str, Any]) -> str:
    """One-line summary for console logging."""
    parts = [log.get("dataset", "?")]
    if log.get("defaults_path"):
        parts.append(f"defaults={log['defaults_path']}")
    resolved = log.get("resolved") or {}
    if resolved.get("n_peaks") is not None:
        parts.append(f"n_peaks={resolved['n_peaks']}")
    if resolved.get("n_true_optima") is not None:
        parts.append(f"n_optima={resolved['n_true_optima']}")
    if resolved.get("amplitude") is not None:
        parts.append(f"A={resolved['amplitude']:g}")
    if resolved.get("noise_amp") is not None:
        parts.append(f"noise_amp={resolved['noise_amp']:g}")
    if resolved.get("basin_width") is not None:
        parts.append(f"basin_width={resolved['basin_width']:g}")
    if resolved.get("sigma") is not None:
        parts.append(f"σ={resolved['sigma']:g}")
    gen = log.get("dataset_generation") or {}
    if gen.get("noise_std") is not None:
        parts.append(f"csv_noise_std={gen['noise_std']}")
    return "  [landscape_config] " + ", ".join(parts)
