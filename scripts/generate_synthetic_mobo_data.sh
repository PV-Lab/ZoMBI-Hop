#!/usr/bin/env bash
# Generate synthetic campaign CSVs + metadata (for RF comparison / analysis).
# MOBO on synthetic benchmarks uses direct oracles — CSVs are optional.
#
# Usage (from repo root):
#   bash scripts/generate_synthetic_mobo_data.sh
#   bash scripts/generate_synthetic_mobo_data.sh --dim 10 --oracle messy

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

python synthetic_data/generate_synthetic_campaign.py --no-show "$@"

echo ""
echo "MOBO (direct oracle): optimize/mobo_batch_configs/synthetic_3d_*.json"
echo "Submit parallel GPU jobs:"
echo "  MOBO_DEVICE=cuda MOBO_MANIFEST=optimize/mobo_batch_manifest_synthetic_3d.json bash scripts/submit_mobo_batch.sh"
echo "Or single synthetic run:"
echo "  MOBO_DEVICE=cuda MOBO_CONFIG=optimize/mobo_batch_configs/synthetic_3d_messy.json bash scripts/submit_mobo.sh"
echo "RF comparison runs (optional): mobo_batch_manifest_synthetic_3d_rf.json"
echo ""
echo "Compare vs campaign1a:"
echo "  python synthetic_data/compare_campaign_datasets.py --no-show"
