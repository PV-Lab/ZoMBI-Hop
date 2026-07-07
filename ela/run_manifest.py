"""Run manifest I/O for ELA S1 pilot jobs."""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def slurm_env() -> dict[str, str | None]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_SUBMIT_DIR",
        "SLURM_JOB_NAME",
    )
    return {k: os.environ.get(k) for k in keys}


def thread_env() -> dict[str, str | None]:
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


def resolve_gp_seed(explicit: int | None) -> tuple[int, str]:
    """Choose GP evolution seed. λ_T sample_seed stays fixed in target JSON."""
    if explicit is not None:
        return int(explicit), "cli"
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        return int(job) % 2_147_483_647, "slurm_job_id"
    import secrets

    return secrets.randbelow(2_147_483_647), "os_random"


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def append_manifest(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    data = load_manifest(path) if path.is_file() else {}
    data.update(updates)
    write_manifest(path, data)
    return data


def build_start_manifest(
    *,
    gp_seed: int,
    gp_seed_source: str,
    run_dir: Path,
    args: Any,
    paper_mode: bool,
    population: int,
    generations: int,
    n_dense: int | None,
    eval_workers: int,
    alpha: float,
    tier1_gamma: float,
    linearity_penalty: float,
    sample_seed: int,
) -> dict[str, Any]:
    return {
        "schema": "ela_s1_run_manifest_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "gp_seed": gp_seed,
        "gp_seed_source": gp_seed_source,
        "sample_seed": sample_seed,
        "sample_seed_note": "fixed λ_T dense Sobol seed (do not randomize)",
        "paper_mode": paper_mode,
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "db": str(Path(args.db).resolve()),
            "target": str(Path(args.target).resolve()),
            "pilot_log": str((run_dir / "pilot.log").resolve()),
            "config": str((run_dir / "config.json").resolve()),
            "manifest": str((run_dir / "run_manifest.json").resolve()),
        },
        "hyperparameters": {
            "population": population,
            "generations": generations,
            "n_dense": n_dense,
            "eval_workers": eval_workers,
            "alpha_subspace": alpha,
            "beta_complexity": args.beta,
            "tier1_gamma": tier1_gamma,
            "linearity_penalty_gamma": linearity_penalty,
            "snapshot_every": args.snapshot_every,
            "landscape_every": args.landscape_every,
            "landscape_grid_n": args.landscape_grid_n,
            "grid_n": args.grid_n,
            "early_reject_mult": args.early_reject_mult,
            "quick": bool(args.quick),
            "no_landscape_viz": bool(args.no_landscape_viz),
            "no_viz": bool(args.no_viz),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "slurm": slurm_env(),
            "threads": thread_env(),
        },
        "command_argv": sys.argv,
    }


def build_finish_manifest(
    *,
    best_fitness: float,
    best_tier1_loss: float,
    best_subspace_rmse: float,
    accepted: bool,
    wall_clock_s: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "result": {
            "best_fitness": best_fitness,
            "best_tier1_loss": best_tier1_loss,
            "best_subspace_rmse": best_subspace_rmse,
            "accepted": accepted,
        },
    }
    if wall_clock_s is not None:
        out["wall_clock_s"] = wall_clock_s
    return out
