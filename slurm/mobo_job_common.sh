# Shared conda + thread env for MOBO Slurm jobs.
# Source from slurm/*.sbatch after cd to REPO.

_mobo_setup_env() {
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate zombi-hop-linebo

  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
  export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
  export MPLBACKEND=Agg
  export PYTHONUNBUFFERED=1
}
