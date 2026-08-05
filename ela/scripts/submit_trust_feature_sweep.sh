#!/usr/bin/env bash
# Batch-submit the three trusted-feature pure-paper-rf-g configs
# (same machinery as ela_3d_18497187; fitness set varies).
#
# Usage (from repo root on ORCD login node):
#   bash ela/scripts/submit_trust_feature_sweep.sh
#
# Or queue all immediately:
#   PILOT_SUBMIT_MODE=slurm-queue bash ela/scripts/submit_trust_feature_sweep.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
mkdir -p ela/scripts/logs

MAX_CPUS="${PILOT_MAX_CPUS:-96}"
JOB_CPUS="${PILOT_JOB_CPUS:-32}"
POLL_SEC="${PILOT_POLL_SEC:-60}"
MODE="${PILOT_SUBMIT_MODE:-feed}"
COUNT_ALL="${PILOT_COUNT_ALL_USER:-0}"

SBATCHES=(
  run_pilot_3d_pure_paper_rf_g_trust_core.sbatch
  run_pilot_3d_pure_paper_rf_g_trust_lip.sbatch
  run_pilot_3d_pure_paper_rf_g_trust_lip_nbc.sbatch
)

_used_cpus() {
  if [[ "${COUNT_ALL}" == "1" ]]; then
    squeue -u "${USER}" -h -t R -o '%C' 2>/dev/null | awk '{s+=$1} END{print s+0}'
  else
    squeue -u "${USER}" -h -t R -o '%j %C' 2>/dev/null \
      | awk '$1 ~ /^ela_/ {s+=$2} END{print s+0}'
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
  echo "${job_id}  ${script}" >> ela/scripts/logs/trust_feature_sweep_submitted.txt
}

if [[ "${MODE}" == "slurm-queue" ]]; then
  echo "Mode=slurm-queue: submitting all ${#SBATCHES[@]} jobs."
  : > ela/scripts/logs/trust_feature_sweep_submitted.txt
  for s in "${SBATCHES[@]}"; do
    _submit_one "${s}"
  done
  echo "Done. Watch with: squeue -u \$USER"
  exit 0
fi

echo "Mode=feed: max ${MAX_CPUS} CPUs → at most $(( MAX_CPUS / JOB_CPUS )) concurrent ${JOB_CPUS}-CPU jobs."
echo "Poll every ${POLL_SEC}s."
echo

: > ela/scripts/logs/trust_feature_sweep_submitted.txt
idx=0
n=${#SBATCHES[@]}

while (( idx < n )); do
  slots="$(_free_slots)"
  used="$(_used_cpus)"
  if (( slots >= 1 )); then
    _submit_one "${SBATCHES[$idx]}"
    idx=$(( idx + 1 ))
    sleep 3
    continue
  fi
  echo "[$(date -Is)] waiting: ${idx}/${n} submitted, used_cpus=${used}/${MAX_CPUS}, free_slots=0 — sleep ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

echo
echo "All ${n} sweep jobs submitted. Log: ela/scripts/logs/trust_feature_sweep_submitted.txt"
echo "squeue -u \$USER"
