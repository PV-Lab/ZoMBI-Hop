#!/bin/bash
#SBATCH --job-name=mobo_10d
#SBATCH --output=optimize/scripts/logs/mobo_10d_%j.out
#SBATCH --error=optimize/scripts/logs/mobo_10d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1

uv run python optimize/run_mobo.py --resume-dim --dataset ackley10d --time-limit 2
