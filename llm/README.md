# LLM-in-the-loop ZoMBI-Hop hyperparameter tuning

Integrates an Anthropic LLM into ZoMBI-Hop so it can retune the algorithm's
hyperparameters **once, on the fly**, at a chosen iteration of a real campaign,
and measures whether that intervention helps versus the real run's own
trajectory.

Everything is evaluated on a **Random-Forest reconstruction** of the campaign2
objective (`llm/data/campaign2_all.db`), built exactly like
`visualization/plot_run.py`. The composition space is the `FAPbI3 / MAPbI3 /
MAPbBr3` 3-simplex, and `Objective` is maximized.

## Files

| File | What it is |
|---|---|
| `llm_config.py` | **Edit here** to change the model (`MODEL`/`EFFORT`, default Opus 4.8 high; Sonnet 5 medium is a two-line swap), the prompt (`PROMPT_TEMPLATE`), and the structured-output schema. The LLM answers through a validated JSON schema (`output_config.format`), so no free-text parsing. |
| `evaluate_llm.py` | One injection point. Set `INJECTION_ITER` at the top. |
| `evaluate_llm_sweep.py` | Sweeps `INJECTION_ITER` in steps of `INJECTION_INTERVAL`. |
| `requirements.txt` | The one extra dependency (`anthropic`). |

## How it works

1. Build the RF objective from `campaign2_all.db`.
2. Reconstruct ZoMBI-Hop's **exact internal state** at the injection point
   (data, needles, zoom bounds, activation/zoom/iteration) from the delta
   snapshots in `runs/run_7eb9` — the same run logged with full algorithm state.
   The db "Iteration" is aligned to a snapshot by cumulative measured-point
   count (the two count almost identically: 644 vs 643 points).
3. Show the LLM the history up to `INJECTION_ITER`, the current hyperparameters
   (and their ranges from `optimize/run_mobo.py`), a progress summary, and the
   needles found so far. Ask it once whether to change any hyperparameters, and
   **time the response**.
4. If it changes anything: resume vanilla ZoMBI-Hop from the injection state with
   the new hyperparameters, running on the RF objective for the **same number of
   additional iterations the real run used** after the injection point
   ("equal budget"). If it changes nothing: no LLM rerun (the outcome equals the
   original hyperparameters).

## Repeats & variance

ZoMBI-Hop + the noisy RF objective are stochastic, so metrics are averaged over
repeats (the LLM is still called **once** per injection point — only the
continuation is repeated):

- **LLM** — `N_LLM_REPEATS` continuations with the LLM's hyperparameters (default 5).
- **Baseline** — `N_BASELINE_REPEATS` samples (default 5): the **real campaign2
  run** (sample #1) plus `N_BASELINE_REPEATS - 1` RF continuations with the
  **original (trial_112) hyperparameters**, all pooled for mean/variance.

Set these constants at the top of `evaluate_llm.py` (single run) and
`evaluate_llm_sweep.py` (sweep). Runtime scales with
`(N_LLM_REPEATS + N_BASELINE_REPEATS − 1) × equal-budget iterations` per injection
point — lower the repeats or budget for a quick check.

## Outputs (per injection point, under `llm/results/`)

- `prompt.txt` — the exact prompt sent.
- `llm_decision.json` — model, effort, **latency**, token usage, the LLM's
  **`reasoning`** and full `raw_response`, and the proposed/validated changes.
- `baseline_vs_llm.json` — `baseline_stats` and `llm_stats` (mean/std/n/values
  for best objective, needles found, needle distance to RF optima, and sampling
  duplicate-fraction), every per-repeat sample, and the LLM−baseline difference
  of the means.
- `comparison.png` — running-best objective vs measured points, with the real
  campaign2 baseline plus every baseline (original-HP) and LLM (new-HP) RF repeat
  overlaid and the injection point marked.
- `continuation/rep{k}/` (only if hyperparameters changed) and
  `baseline_rf/rep{k}/` — everything `run_mobo.py` logs, per repeat: `points.csv`,
  `needles.csv`, `metrics_over_time.csv`, `convergence.png`,
  `dist_from_centre.png`, `line_length_hist.png`, and the (patched) `config.json`.

The sweep additionally writes `sweep_summary.csv` / `.json` tabulating every
injection point with mean±std columns and the LLM's reasoning.

## Running

```bash
conda activate zombi-hop
pip install -r llm/requirements.txt
export ANTHROPIC_API_KEY=...        # or `ant auth login`

python llm/evaluate_llm.py          # single injection (INJECTION_ITER in the file)
python llm/evaluate_llm_sweep.py    # sweep (INJECTION_INTERVAL in the file)
```

If `anthropic` is missing or no credentials are set, each run logs the error and
falls back to reporting the baseline only (no rerun).

> Note on the baseline: it is the real campaign2 measured trajectory (per the
> experiment design), while the LLM continuation is evaluated on the RF
> surrogate. The two share the pre-injection prefix exactly (the continuation
> resumes from that state), so the curves diverge only after the injection point.
