#!/usr/bin/env bash
# Generate all standard 3D synthetic campaign CSVs + metadata for MOBO benchmarking.
#
# Usage (from repo root):
#   bash scripts/generate_synthetic_mobo_data.sh
#   bash scripts/generate_synthetic_mobo_data.sh --dim 10 --oracle messy

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

python synthetic_data/generate_synthetic_campaign.py --no-show "$@"

echo ""
echo "MOBO batch configs: optimize/mobo_batch_configs/synthetic_3d_*.json"
echo "Submit parallel GPU jobs:"
echo "  MOBO_MANIFEST=optimize/mobo_batch_manifest_synthetic_3d.json bash scripts/submit_mobo_batch.sh"
echo "Or single synthetic run:"
echo "  MOBO_CONFIG=optimize/mobo_batch_configs/synthetic_3d_messy_rf_max.json sbatch slurm/run_mobo_gpu.sbatch"
