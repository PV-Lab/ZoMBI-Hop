"""Cap on how many MOBO workers of a campaign run at once (``<campaign>/max_workers``).

Lowering a running campaign's GPU count without killing work: write the new cap
into ``<share_dir>/max_workers`` (one integer). Before a worker starts a NEW trial
(never mid-trial, never on an interrupted trial it is resuming) it calls
``should_retire``; while more of the campaign's SLURM jobs are queued or running
than the cap allows, the worker that asks retires and its job ends. The first
workers to reach a trial boundary retire, so nothing is thrown away.

Why here and not in the sbatch script: a job resubmits itself with ``sbatch "$0"``,
and inside a job ``$0`` is SLURM's spooled copy of the script as first submitted,
so later edits to the .sbatch file never reach the running chain. run_mobo.py,
by contrast, is re-read from disk every time a job (re)starts.

Workers are counted with ``squeue`` (this user's jobs with this job's name, any
active state), minus the ids already listed in ``<share_dir>/retired_jobs`` that
have not left the queue yet. A file lock serialises the decision, so two workers
at a boundary at the same moment cannot both retire when only one should.
Anything unexpected (no file, not under SLURM, squeue failing) means: keep going.
"""
from __future__ import annotations

import fcntl
import getpass
import os
import subprocess

CAP_FILE = "max_workers"
RETIRED_FILE = "retired_jobs"
LOCK_FILE = ".max_workers.lock"
RETIRE_EXIT_CODE = 76


def _read_cap(share_dir: str) -> int | None:
    try:
        with open(os.path.join(share_dir, CAP_FILE)) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _active_job_ids(job_name: str) -> set[str] | None:
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", getpass.getuser(), "-n", job_name, "-o", "%i"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # Array/het jobs print "123_4" / "123+0"; plain campaign jobs print "123".
    return {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}


def should_retire(share_dirs: list[str] | None) -> str | None:
    """A reason string when this worker should retire now, else None."""
    job_id = os.environ.get("SLURM_JOB_ID")
    job_name = os.environ.get("SLURM_JOB_NAME")
    if not share_dirs or not job_id or not job_name:
        return None
    share_dir = share_dirs[0]
    cap = _read_cap(share_dir)
    if cap is None:
        return None
    try:
        lock = open(os.path.join(share_dir, LOCK_FILE), "a")
    except OSError:
        return None
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = _active_job_ids(job_name)
        if active is None:
            return None
        retired_path = os.path.join(share_dir, RETIRED_FILE)
        try:
            with open(retired_path) as f:
                retired = {ln.strip() for ln in f if ln.strip()}
        except OSError:
            retired = set()
        live = active - retired          # retirees still exiting don't count
        if job_id not in live or len(live) <= cap:
            return None
        with open(retired_path, "a") as f:
            f.write(f"{job_id}\n")
        return f"{len(live)} workers active, cap {cap} ({os.path.join(share_dir, CAP_FILE)})"
