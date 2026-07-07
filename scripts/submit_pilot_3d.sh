#!/usr/bin/env bash
# Wrapper — pilot Slurm scripts live under ela/scripts/ (scratch repo layout).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ela/scripts/submit_pilot_3d.sh" "$@"
