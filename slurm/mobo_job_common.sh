# Shared conda + thread env for MOBO Slurm jobs.
# Source from slurm/*.sbatch after cd to REPO.

_mobo_setup_env() {
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate zombi-hop-linebo

  # ORCD compute nodes ship an older system libstdc++. BoTorch's JIT logei
  # extension needs GLIBCXX_3.4.29+; prefer conda's libstdc++ at load/compile.
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi

  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
  export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
  export MPLBACKEND=Agg
  export PYTHONUNBUFFERED=1
}
