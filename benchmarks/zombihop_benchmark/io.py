from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .types import BatchObservation


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def make_run_dir(output_root: str | Path, experiment_name: str, optimizer: str, objective: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / experiment_name / f"{stamp}_{optimizer}_{objective}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def git_state(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)

    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    return {
        "branch": _run(["git", "branch", "--show-current"]),
        "head": _run(["git", "rev-parse", "HEAD"]),
        "status_short": _run(["git", "status", "--short"]),
        "remote_v": _run(["git", "remote", "-v"]),
    }


def observation_rows(
    obs: BatchObservation,
    step: int,
    elapsed_s: float,
    optimizer: str,
    objective: str,
    seed: int,
    line_index: int | None = None,
    mode: str = "point",
    line_metadata: dict[str, Any] | None = None,
    is_initial_point: bool = False,
    point_offset: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    line_metadata = line_metadata or {}
    for i, (x_exp, x_act, y_val) in enumerate(zip(obs.X_expected, obs.X_actual, obs.y)):
        rows.append(
            {
                "step": step,
                "point_index": point_offset + i,
                "mode": mode,
                "line_index": "" if line_index is None else line_index,
                "line_id": line_metadata.get("line_id", ""),
                "point_index_in_line": "" if line_index is None else i,
                "line_num_points": line_metadata.get("line_num_points", ""),
                "line_score": line_metadata.get("line_score", ""),
                "line_score_method": line_metadata.get("line_score_method", ""),
                "line_length_l2": line_metadata.get("line_length_l2", ""),
                "line_length_ilr": line_metadata.get("line_length_ilr", ""),
                "is_initial_point": bool(is_initial_point),
                "x_expected": json.dumps(np.asarray(x_exp, dtype=float).tolist()),
                "x_actual": json.dumps(np.asarray(x_act, dtype=float).tolist()),
                "y": float(y_val),
                "elapsed_s": float(elapsed_s),
                "optimizer": optimizer,
                "objective": objective,
                "seed": seed,
                "metadata": json.dumps(_jsonable(obs.metadata), sort_keys=True),
            }
        )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

