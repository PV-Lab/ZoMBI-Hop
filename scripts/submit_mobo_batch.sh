#!/usr/bin/env bash
# Submit a Slurm array job — one optimize/run_mobo.py run per manifest entry.
#
# Usage (from repo root on MIT ORCD):
#   bash scripts/submit_mobo_batch.sh
#   MOBO_MANIFEST=optimize/mobo_batch_manifest.json bash scripts/submit_mobo_batch.sh
#
# Add a dataset: drop a JSON in optimize/mobo_batch_configs/ and append its path
# to optimize/mobo_batch_manifest.json, then re-run this script.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MOBO_MANIFEST:-optimize/mobo_batch_manifest.json}"
SBATCH="${REPO}/slurm/run_mobo_array.sbatch"

cd "${REPO}"
mkdir -p logs

N=$(python - <<PY
import json
with open("${MANIFEST}") as f:
    print(len(json.load(f)["configs"]))
PY
)

if [[ "${N}" -lt 1 ]]; then
  echo "No configs in ${MANIFEST}" >&2
  exit 1
fi

LAST=$(( N - 1 ))
echo "Submitting MOBO array: ${N} task(s) (0-${LAST}) from ${MANIFEST}"

MOBO_MANIFEST="${MANIFEST}" sbatch \
  --array="0-${LAST}" \
  --export=ALL,MOBO_MANIFEST="${MANIFEST}" \
  "${SBATCH}"
