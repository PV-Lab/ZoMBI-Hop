#!/bin/bash
#SBATCH --job-name=ensemble_mobo_4d
#SBATCH --output=optimize/scripts/logs/ensemble_mobo_4d_%j.out
#SBATCH --error=optimize/scripts/logs/ensemble_mobo_4d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1

uv run python optimize/run_mobo.py --dataset ensemble --dim 4 --time-limit 0.3 --start-from-best optimize/runs/mobo_3d_05_06_15_32/trial_112
