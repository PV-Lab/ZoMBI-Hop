#!/usr/bin/env bash
# Submit ELA S1 3D pilot jobs to ORCD / Engaging Slurm.
#
# Intended layout on ORCD scratch:
#   ~/orcd/scratch/ZoMBI-Hop/          ← repo clone (REPO)
#   ~/orcd/scratch/ZoMBI-Hop/ela/runs/ ← pilot outputs (default)
#
# Usage (from repo root on login node):
#   bash ela/scripts/submit_pilot_3d.sh
#   bash ela/scripts/submit_pilot_3d.sh probe
#   bash ela/scripts/submit_pilot_3d.sh array
#
# Or submit directly:
#   sbatch ela/scripts/run_pilot_3d.sbatch
#   sbatch ela/scripts/run_pilot_3d_probe.sbatch
#   sbatch ela/scripts/run_pilot_3d_array.sbatch
#
# Edit parameters in ela/pilot_config.json (default), then submit:
#   bash ela/scripts/submit_pilot_3d.sh
#   sbatch ela/scripts/run_pilot_3d.sbatch
#
# Preset configs:
#   PILOT_CONFIG=ela/pilot_config_campaign.json bash ela/scripts/submit_pilot_3d.sh
#   PILOT_CONFIG=ela/pilot_config_pure_paper.json sbatch ela/scripts/run_pilot_3d.sbatch
#   sbatch ela/scripts/run_pilot_3d_pure_paper.sbatch   # exact Muñoz S1 paper preset
#
# Parallel pair (submit individually on ORCD):
#   bash ela/scripts/submit_pilot_3d_munoz8_lip.sh
#   bash ela/scripts/submit_pilot_3d_set_b_v2.sh
#
# Next batch (sbatch individually):
#   sbatch ela/scripts/run_pilot_3d_set_b_v11.sbatch
#   sbatch ela/scripts/run_pilot_3d_munoz8_lip_v2.sbatch
#   sbatch ela/scripts/run_pilot_3d_set_b_rq.sbatch
#
# Follow-up after munoz8-lip-v2 breakthrough (17591803):
#   sbatch ela/scripts/run_pilot_3d_munoz8_lip_fdc.sbatch
#   sbatch ela/scripts/run_pilot_3d_set_b_r2q.sbatch
#   sbatch ela/scripts/run_pilot_3d_set_b_rq_v2.sbatch
#
# Spatial roughness (interior / percentiles / uniform):
#   sbatch ela/scripts/run_pilot_3d_spatial_interior.sbatch
#   sbatch ela/scripts/run_pilot_3d_spatial_pct.sbatch
#   sbatch ela/scripts/run_pilot_3d_spatial_uniform.sbatch
#
# Spatial v2 (MA lobe / blend / CV uniformity):
#   sbatch ela/scripts/run_pilot_3d_spatial_blend.sbatch
#   sbatch ela/scripts/run_pilot_3d_spatial_ma.sbatch
#   sbatch ela/scripts/run_pilot_3d_spatial_cv.sbatch
#
# Two-track next (peaks vs scattered modes; no MA-only):
#   sbatch ela/scripts/run_pilot_3d_multimodal_pks.sbatch
#   sbatch ela/scripts/run_pilot_3d_spatial_modes.sbatch
#
# Muñoz-8+Lip α=0 spatial sweep (paper_ga, no FDC/R2_QI; ~5 jobs):
#   bash ela/scripts/submit_m8lip0_spatial_sweep.sh
#
# Bump × Lipschitz sweep (RBF bump primitive, paper-like GA; base + 4 Lip
# variants + spatial-modes kitchen sink; ~6 jobs). Same as m8lip0 sweep but
# with localized ILR Gaussian bumps enabled (paper_ga=false + allow_rbf=true):
#   bash ela/scripts/submit_bump_lip_sweep.sh
#   # or individually:
#   sbatch ela/scripts/run_pilot_3d_bump_base.sbatch
#   sbatch ela/scripts/run_pilot_3d_bump_interior.sbatch
#   sbatch ela/scripts/run_pilot_3d_bump_pct.sbatch
#   sbatch ela/scripts/run_pilot_3d_bump_cv.sbatch
#   sbatch ela/scripts/run_pilot_3d_bump_tile.sbatch
#
# Environment overrides (optional; beat config file at runtime):
#   PILOT_CONFIG=ela/pilot_config.json
#   PILOT_POPULATION=400  PILOT_GENERATIONS=100
#   PILOT_ALPHA=3  PILOT_TIER1_GAMMA=5  PILOT_BETA=0.001
#   PILOT_CAMPAIGN_MODE=1  PILOT_PURE_PAPER=1
#   (omit PILOT_SEED for auto: SLURM_JOB_ID on cluster, os_random locally)
#   PILOT_EVAL_WORKERS=8  PILOT_OMP_THREADS=4
#   PILOT_QUICK=1  PILOT_NO_LANDSCAPE_VIZ=1  PILOT_NO_VIZ=1
#   PILOT_RUN_ROOT=...   (default: ${REPO}/ela/runs)
#   PILOT_PARTITION=mit_normal  PILOT_TIME=12:00:00

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-single}"
SCRIPTS="${REPO}/ela/scripts"

cd "${REPO}"
mkdir -p "${SCRIPTS}/logs" "${REPO}/ela/runs"

# shellcheck source=/dev/null
source "${SCRIPTS}/pilot_job_common.sh"
export REPO
_pilot_configure_parallelism
_pilot_setup_dirs
_pilot_check_data

PARTITION="${PILOT_PARTITION:-mit_normal}"
TIME_LIMIT="${PILOT_TIME:-12:00:00}"
CPUS="${PILOT_CPUS:-32}"
MEM="${PILOT_MEM:-128G}"

sbatch_common=(
  --partition="${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
)

case "${MODE}" in
  single|"")
    SBATCH="${SCRIPTS}/run_pilot_3d.sbatch"
    sbatch_common+=(--time="${TIME_LIMIT}" --job-name=ela_3d)
    ;;
  probe)
    SBATCH="${SCRIPTS}/run_pilot_3d_probe.sbatch"
    sbatch_common+=(--time=00:30:00 --job-name=ela_3d_probe --cpus-per-task=4 --mem=16G)
    ;;
  array)
    SBATCH="${SCRIPTS}/run_pilot_3d_array.sbatch"
    sbatch_common+=(--time="${TIME_LIMIT}" --job-name=ela_3d_arr)
    if [[ -n "${PILOT_ARRAY:-}" ]]; then
      sbatch_common+=(--array="${PILOT_ARRAY}")
    fi
    ;;
  *)
    echo "Usage: bash ela/scripts/submit_pilot_3d.sh [single|probe|array]" >&2
    exit 1
    ;;
esac

echo "Submitting pilot (${MODE}) from ${REPO}"
_pilot_log_config_summary
echo "  eval_workers: ${PILOT_EVAL_WORKERS} (OMP=${OMP_NUM_THREADS}/worker)"
echo "  sbatch: ${SBATCH}"

job_id=$(sbatch --export=ALL "${sbatch_common[@]}" "${SBATCH}" | awk '{print $NF}')
echo "Submitted job ${job_id}"
echo "  logs: ${REPO}/ela/scripts/logs/ela_3d_${job_id}.{out,err}"
echo "  runs: ${PILOT_RUN_ROOT}/ela_3d_${job_id}"
echo "  squeue -j ${job_id}"
