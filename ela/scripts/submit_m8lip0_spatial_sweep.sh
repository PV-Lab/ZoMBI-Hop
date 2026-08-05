#!/usr/bin/env bash
# Feed Muñoz-8+Lip α=0 spatial sweep under a CPU concurrency cap.
#
# Default: keep at most 3×32-CPU ela jobs running (fits a 96-CPU budget).
# Already-running ela jobs (any ela_3d_*) count against the cap; this script
# only submits the next sweep member when a free 32-CPU slot opens.
#
# Usage (from repo root on ORCD login node — leave this shell running / use tmux):
#   bash ela/scripts/submit_m8lip0_spatial_sweep.sh
#
# Options via env:
#   PILOT_MAX_CPUS=96          # your concurrent CPU budget
#   PILOT_JOB_CPUS=32          # CPUs per pilot job (must match #SBATCH)
#   PILOT_POLL_SEC=60          # how often to check for free slots
#   PILOT_COUNT_ALL_USER=0     # if 1, count ALL your Running CPUs (not just ela_3d_*)
#
# Alternative — let Slurm queue everything immediately:
#   PILOT_SUBMIT_MODE=slurm-queue bash ela/scripts/submit_m8lip0_spatial_sweep.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
mkdir -p ela/scripts/logs

MAX_CPUS="${PILOT_MAX_CPUS:-96}"
JOB_CPUS="${PILOT_JOB_CPUS:-32}"
POLL_SEC="${PILOT_POLL_SEC:-60}"
MODE="${PILOT_SUBMIT_MODE:-feed}"   # feed | slurm-queue
COUNT_ALL="${PILOT_COUNT_ALL_USER:-0}"

SBATCHES=(
  run_pilot_3d_m8lip0_base.sbatch
  run_pilot_3d_m8lip0_interior.sbatch
  run_pilot_3d_m8lip0_pct.sbatch
  run_pilot_3d_m8lip0_cv.sbatch
  run_pilot_3d_m8lip0_tile.sbatch
)

_used_cpus() {
  # Sum CPUs of this user's Running jobs (pending do not consume allocation yet).
  if [[ "${COUNT_ALL}" == "1" ]]; then
    squeue -u "${USER}" -h -t R -o '%C' 2>/dev/null | awk '{s+=$1} END{print s+0}'
  else
    # Only ela_3d_* pilots (job name starts with ela_3d)
    squeue -u "${USER}" -h -t R -o '%j %C' 2>/dev/null \
      | awk '$1 ~ /^ela_3d/ {s+=$2} END{print s+0}'
  fi
}

_free_slots() {
  local used free
  used="$(_used_cpus)"
  free=$(( MAX_CPUS - used ))
  if (( free < 0 )); then free=0; fi
  echo $(( free / JOB_CPUS ))
}

_submit_one() {
  local script="$1"
  local out job_id
  echo "[$(date -Is)] submitting ${script}  (used_cpus=$(_used_cpus)/${MAX_CPUS})"
  out="$(sbatch "ela/scripts/${script}")"
  echo "  ${out}"
  job_id="$(echo "${out}" | awk '{print $NF}')"
  echo "${job_id}  ${script}" >> ela/scripts/logs/m8lip0_spatial_sweep_submitted.txt
}

if [[ "${MODE}" == "slurm-queue" ]]; then
  echo "Mode=slurm-queue: submitting all ${#SBATCHES[@]} jobs; Slurm holds Pending until CPUs free."
  : > ela/scripts/logs/m8lip0_spatial_sweep_submitted.txt
  for s in "${SBATCHES[@]}"; do
    _submit_one "${s}"
  done
  echo "Done. Watch with: squeue -u \$USER"
  exit 0
fi

echo "Mode=feed: max ${MAX_CPUS} CPUs → at most $(( MAX_CPUS / JOB_CPUS )) concurrent ${JOB_CPUS}-CPU jobs."
echo "Counting Running CPUs for: $([[ ${COUNT_ALL} == 1 ]] && echo 'ALL your jobs' || echo 'ela_3d_* only')"
echo "Poll every ${POLL_SEC}s. Leave this process running (tmux/screen recommended)."
echo

: > ela/scripts/logs/m8lip0_spatial_sweep_submitted.txt
idx=0
n=${#SBATCHES[@]}

while (( idx < n )); do
  slots="$(_free_slots)"
  used="$(_used_cpus)"
  if (( slots >= 1 )); then
    _submit_one "${SBATCHES[$idx]}"
    idx=$(( idx + 1 ))
    # Brief pause so squeue reflects the new job before we fill another slot.
    sleep 3
    continue
  fi
  echo "[$(date -Is)] waiting: ${idx}/${n} submitted, used_cpus=${used}/${MAX_CPUS}, free_slots=0 — sleep ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

echo
echo "All ${n} sweep jobs submitted. Log: ela/scripts/logs/m8lip0_spatial_sweep_submitted.txt"
echo "squeue -u \$USER"
