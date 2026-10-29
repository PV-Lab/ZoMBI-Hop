#!/bin/bash
#SBATCH --job-name=landscape_hist
#SBATCH --output=optimize/scripts/logs/landscape_hist_%j.out
#SBATCH --error=optimize/scripts/logs/landscape_hist_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_interactive_r8,sched_mit_sloan_gpu_r8,mit_preemptable,mit_normal

# Objective-value histograms of the showdown landscapes (synthetic_data/histogram.py
# --mode showdowns): the run_69af RF500 landscape vs needles bw2.2 / bw6 / bw10, plus a
# second row for the mixed showdown (RF 69af, needles n2/bw2.2 + n3/bw6, and RFs of
# uniform samples of those two). CPU
# only; the RF is refit per invocation (seconds), the 200k-point predictions dominate.
# Submit from the repo root.

OUT="analysis/landscape_hist_20261026_unit_v2"
mkdir -p "$OUT"
for norm in "" "--normalize"; do
    tag=${norm:+_norm}
    uv run python synthetic_data/histogram.py --mode showdowns --log $norm \
        --n-samples 200000 --save "$OUT/landscape_hist${tag}.png" < /dev/null
done
uv run python synthetic_data/histogram.py --mode showdowns --log --normalize \
    --n-samples 200000 --save "$OUT/landscape_hist_norm.html" < /dev/null
