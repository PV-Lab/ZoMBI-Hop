#!/bin/bash
#SBATCH --job-name=sweep_catastrophic_llm
#SBATCH --output=llm/scripts/logs/sweep_catastrophic_llm_%j.out
#SBATCH --error=llm/scripts/logs/sweep_catastrophic_llm_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1

uv run python -u llm/sweep_catastrophic.py
