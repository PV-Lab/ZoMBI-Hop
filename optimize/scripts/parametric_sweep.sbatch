#!/bin/bash
#SBATCH --job-name=param_sweep
#SBATCH --output=optimize/scripts/logs/param_sweep_%j.out
#SBATCH --error=optimize/scripts/logs/param_sweep_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1

uv run python optimize/parametric_sweep.py
