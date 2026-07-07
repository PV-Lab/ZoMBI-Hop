"""Run directory naming — matches ``optimize/run_mobo.unique_run_dir`` conventions."""
from __future__ import annotations

import datetime
import os
from pathlib import Path

# Same stamp as run_mobo.unique_run_dir: day_month_hour_min_sec + pid
RUN_DIR_STAMP_FMT = "%d_%m_%H_%M_%S"


def unique_run_dir(parent: str | Path, prefix: str) -> Path:
    """Create ``{prefix}_{dd}_{mm}_{HH}_{MM}_{SS}_{pid}`` under *parent* (atomic).

    Mirrors ``optimize.run_mobo.unique_run_dir`` so local ELA and MOBO runs sort
    and collide the same way.
    """
    parent = Path(parent)
    base = (
        datetime.datetime.now().strftime(f"{prefix}_{RUN_DIR_STAMP_FMT}")
        + f"_{os.getpid()}"
    )
    n = 1
    while True:
        name = base if n == 1 else f"{base}_{n}"
        candidate = parent / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            n += 1


def pilot_prefix(*, seed: int = 0, quick: bool = False) -> str:
    """Folder prefix for a 3D S1 pilot run."""
    if quick:
        return f"pilot_3d_quick_seed{seed}"
    return f"pilot_3d_seed{seed}"


def slurm_job_run_dir(
    parent: str | Path,
    *,
    seed: int = 0,
    job_id: int | str,
    quick: bool = False,
) -> Path:
    """Stable Slurm run dir keyed on job id (requeue-safe), like ``mobo_*_job${SLURM_JOB_ID}``."""
    tag = "quick" if quick else f"seed{seed}"
    return Path(parent) / f"pilot_3d_{tag}_job{job_id}"


def slurm_array_run_dir(
    parent: str | Path,
    *,
    seed: int,
    array_job_id: int | str,
    task_id: int | str,
    quick: bool = False,
) -> Path:
    """Run dir for ``#SBATCH --array`` tasks."""
    tag = "quick" if quick else f"seed{seed}"
    return Path(parent) / f"pilot_3d_{tag}_job{array_job_id}_task{task_id}"


def default_runs_root(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent.parent
    return root / "ela" / "runs"
