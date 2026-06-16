#!/usr/bin/env bash
# Submit a Slurm array job — one optimize/run_mobo.py run per manifest entry.
#
# Usage (from repo root on MIT Engaging):
#   bash scripts/submit_mobo_batch.sh
#   MOBO_MANIFEST=optimize/mobo_batch_manifest.json bash scripts/submit_mobo_batch.sh
#   MOBO_DEVICE=cuda MOBO_MANIFEST=optimize/mobo_batch_manifest_synthetic_3d.json bash scripts/submit_mobo_batch.sh
#
# Add a dataset: drop a JSON in optimize/mobo_batch_configs/ and append its path
# to a manifest JSON, then re-run this script.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MOBO_MANIFEST:-optimize/mobo_batch_manifest.json}"
SBATCH="${REPO}/slurm/run_mobo_array.sbatch"
DEVICE="${MOBO_DEVICE:-cpu}"

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

extra=()
case "${DEVICE}" in
  cuda)
    extra+=(
      --partition=mit_normal_gpu
      --gres=gpu:1
      --cpus-per-task=8
      --mem=64G
      --time=6:00:00
    )
    ;;
  cpu)
    extra+=(
      --partition=mit_normal
      --cpus-per-task=16
      --mem=32G
      --time=12:00:00
    )
    ;;
  *)
    echo "MOBO_DEVICE must be cpu or cuda (got ${DEVICE})" >&2
    exit 1
    ;;
esac

echo "Submitting MOBO array: ${N} task(s) (0-${LAST}) device=${DEVICE} manifest=${MANIFEST}"

MOBO_DEVICE="${DEVICE}" MOBO_MANIFEST="${MANIFEST}" sbatch \
  "${extra[@]}" \
  --array="0-${LAST}" \
  --export=ALL,MOBO_DEVICE="${DEVICE}",MOBO_MANIFEST="${MANIFEST}" \
  "${SBATCH}"
