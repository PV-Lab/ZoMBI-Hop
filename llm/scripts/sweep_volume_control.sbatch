#!/bin/bash
#SBATCH --job-name=sweep_volume_control
#SBATCH --output=llm/scripts/logs/sweep_volume_control_%j.out
#SBATCH --error=llm/scripts/logs/sweep_volume_control_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1

# Single 12-hour run — no requeue, no auto-continue. The sweep must finish within
# the wall-time limit; if it doesn't, the job is killed and NOT resubmitted.
uv run python llm/sweep_volume_control.py
