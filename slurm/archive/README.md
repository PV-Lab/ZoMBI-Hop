# Archived Slurm scripts

These jobs target the **legacy v2 MOBO stack** (`scripts/run_zombi_test_v2*.py --mobo`), not the current `optimize/run_mobo.py` pipeline.

| Script | Entry point | Output |
|--------|-------------|--------|
| `mobo_cpu_v2.sbatch` | `run_zombi_test_v2.py --mobo` | `runs/hyperparam_cpu_results_v2.json` |
| `mobo_cpu_parallel_v2.sbatch` | `run_zombi_test_v2_parallel.py --mobo` | `runs/hyperparam_cpu_results_v2_parallel.json` |

For current MOBO batch runs on Engaging, use:

- `scripts/submit_mobo.sh` — single job (`MOBO_DEVICE=cpu|cuda`)
- `scripts/submit_mobo_batch.sh` — Slurm array from a manifest
- `scripts/probe_cluster.sh` — short runtime probes
