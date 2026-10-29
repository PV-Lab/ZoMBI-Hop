#!/bin/bash
#SBATCH --job-name=needles_rf_points
#SBATCH --output=optimize/scripts/logs/needles_rf_points_%j.out
#SBATCH --error=optimize/scripts/logs/needles_rf_points_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=sched_mit_sloan_interactive_r8,sched_mit_sloan_gpu_r8,mit_preemptable,mit_normal

# Uniform random samples (as many as run_69af has scored points) of the needles
# n2/bw2.2, n3/bw6 and n10/bw6 landscapes, for the runrf landscapes of the mixed showdown
# optimize/runs/showdown_6d_dup1c_mix_rf69af_needles_rf_unit_v2 (unit_range: plain 0, peaks 1). CPU only. Submit from the repo root.

uv run python optimize/scripts/needles_rf_points.py --match-run runs/run_69af \
    --needles n_optima=2,basin_width=2.2,unit_range=true --needles n_optima=3,basin_width=6,unit_range=true \
    --needles n_optima=10,basin_width=6,unit_range=true \
    --out analysis/needles_rf_points_unit_v2_20261026 < /dev/null
