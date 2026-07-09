# Step 9 Branch-Sync Review

Date: 2026-07-09

## Branch State

Local working branch for Step 9:

- `benchmarking-milestone1a-consolidation`
- HEAD: `e3d419cb6e237f32618103e4ba89cb93cc0c96e5`

The branch was created from local `benchmarking`, which was up to date with the cached `origin/benchmarking` ref before the safety branch was created.

Remote refs available locally:

| Branch | SHA |
| --- | --- |
| `origin/brianna` | `06dbb625a9fcb7eefe7a70f413b9878bf3320aff` |
| `origin/benchmarking` | `e3d419cb6e237f32618103e4ba89cb93cc0c96e5` |
| `origin/brianna-compositional` | `56062505ac173325e23651bc78098bf273309dc6` |
| `origin/evelyn-compositional` | `79312be42218ad48b660dfe8f2f5c881c0607665` |

Note: `git fetch origin --prune` was attempted for this audit, but this local environment could not refresh the HTTPS remote. The audit therefore uses the local cached remote refs listed above.

## High-Level Diff Summary

`origin/benchmarking...origin/brianna-compositional`:

- 73 files changed
- 9844 insertions
- 1390 deletions

`origin/benchmarking...origin/evelyn-compositional`:

- 69 files changed
- 7426 insertions
- 1428 deletions

## Files Relevant to Step 9

Already present and directly used for Step 9:

- `synthetic_data/ackley.py`
- `synthetic_data/defaults/ackley.json`

The local defaults match the Step 9 realistic Ackley values:

- `n_optima: 20`
- `basin_width: 86.0`
- `noise_freq: 9.0`
- `noise_amp: 400.0`

Changed between `origin/benchmarking` and `origin/brianna-compositional`:

- `synthetic_data/oracles.py`
- `optimize/eval_metrics.py`
- `src/core/zombihop.py`
- `src/utils/datahandler.py`
- `src/utils/simplex.py`
- `src/utils/gp_simplex.py`

Changed between `origin/benchmarking` and `origin/evelyn-compositional`:

- `synthetic_data/oracles.py`
- `optimize/eval_metrics.py`
- `src/core/zombihop.py`
- `src/utils/datahandler.py`
- `src/utils/simplex.py`
- `src/utils/gp_simplex.py`

Interpretation for Step 9:

- The realistic Ackley generator and defaults can be used now without a full branch merge.
- The newer metric work is relevant, especially composition-L2 defaults for match and duplicate metrics. Step 9 adds composition-L2 benchmark metrics alongside the existing ILR metrics.
- The core ZoMBI-Hop drift is scientifically relevant but not merge-trivial.

## Merge Preview

A non-mutating merge preview against `origin/brianna-compositional` shows conflicts in files that are outside the benchmark report path plus a direct conflict in:

- `src/core/zombihop.py`

The `src/core/zombihop.py` conflict is around the benchmark-specific Step 4 additions:

- `max_objective_calls`
- `objective_call_callback`

Because these hooks are needed for fair line-mode budget stopping and callback logging, Step 9 keeps the current benchmarking-pinned ZoMBI-Hop core and defers full core sync to a later dedicated step.

## Later Work

Not needed for the Milestone 1A consolidation report:

- Visualization utilities, including input-noise and run plotting helpers.
- ORCD/SLURM fleet scripts and MOBO orchestration scripts.
- Broader MOBO hyperparameter infrastructure.
- Any ELA/digital-twin workflow should remain a later 10D-oriented workstream.

## Step 9 Decision

Proceed without a full merge. Import the benchmark-relevant behavior by:

- Adding a benchmark objective wrapper for `Ackley("realistic", dim=3)`.
- Preserving the old `synthetic_3d_planted` objective as a smoke fixture only.
- Adding explicit ILR and composition-L2 metrics to benchmark outputs.
- Running the external-baseline Milestone 1A report on the current benchmark-pinned core.
