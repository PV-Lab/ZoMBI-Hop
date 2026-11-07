#!/bin/bash
#SBATCH --job-name=smoke_mobo_parity
#SBATCH --output=optimize/scripts/logs/smoke_mobo_parity_%j.out
#SBATCH --error=optimize/scripts/logs/smoke_mobo_parity_%j.err
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1

# Smoke test for the 2026-11-07 MOBO/showdown parity changes (run_mobo.py):
# never_terminate, --budget-points line cap, axial init, apply_dim_rules, 19-axis
# space, dup1c_custom.json as the start-from-best seed.
#
# Small on purpose: 6d, 600-point budget (600//24 - 6 axial init = 19 optimizer
# lines), 1 landscape per trial, 2 trials (the dup1c_custom seed + 1 Sobol point).
# No --share-history, and the run dir sits under runs/smoke/, which no fleet run's
# runs/mobo_* glob scans, so nothing here pools into real history.

cd /orcd/scratch/orcd/013/adewinmb/ZoMBI-Hop
RUN_DIR="optimize/runs/smoke/mobo_ensemble_6d_smoke_job${SLURM_JOB_ID}"
mkdir -p optimize/runs/smoke

echo "[$(date)] smoke on $(hostname), gpu=${CUDA_VISIBLE_DEVICES:-?}"
uv run python optimize/run_mobo.py --dataset ensemble --dim 6 --time-limit 0.25 \
    --init axial --budget-points 600 \
    --ensemble-repeats 1 --n-init-trials 1 --max-trials 2 \
    --ensemble-seed 12345 \
    --start-from-best optimize/hparams/dup1c_custom.json \
    --runs-dir optimize/runs/smoke \
    --run-dir "$RUN_DIR" < /dev/null
rc=$?
echo "[$(date)] run_mobo rc=$rc"

# Checks — each prints PASS/FAIL so the log is readable at a glance.
uv run python - "$RUN_DIR" <<'EOF'
import glob, json, os, sys
run = sys.argv[1]
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
cfg = json.load(open(os.path.join(run, "run_config.json")))
check("run_config never_terminate", cfg.get("never_terminate") is True, cfg.get("never_terminate"))
check("run_config budget_points", cfg.get("budget_points") == 600, cfg.get("budget_points"))
check("run_config init_design", cfg.get("init_design") == "axial", cfg.get("init_design"))
check("19 hparam axes", len(cfg.get("hparam_names", [])) == 19, len(cfg.get("hparam_names", [])))
check("inert axes dropped",
      not {"needle_shrink_factor", "needle_stop_noise_multiplier"} & set(cfg["hparam_names"]))
prog = json.load(open(os.path.join(run, "mobo_progress.json")))
trials = prog.get("trials", [])
check("2 trials recorded", len(trials) == 2, len(trials))
seed = json.load(open("optimize/hparams/dup1c_custom.json"))["hparams"]
if trials:
    hp = trials[0]["hparams"]
    diff = {k: (hp.get(k), v) for k, v in seed.items()
            if not (isinstance(v, (int, float)) and abs(float(hp.get(k, -1e9)) - v) < 1e-6 * max(1, abs(v)))}
    check("trial 0 = dup1c_custom seed", not diff, diff)
    check("seed nat_grad_max_steps 400 survives", hp.get("nat_grad_max_steps") == 400, hp.get("nat_grad_max_steps"))
check("trial 1 phase is seed", bool(trials) and trials[0].get("phase") == "seed",
      trials[0].get("phase") if trials else None)
for t in trials:
    print(f"    trial {t.get('trial')} ({t.get('phase')}): {t.get('metrics')}")
print("SMOKE", "PASS" if ok else "FAIL")
EOF
echo "--- budget / never_terminate / crash lines from this job's log ---"
grep -E "line budget reached|\[no-stop\]|\[rules\]|\[seed\]|ZoMBI crashed|Traceback|top_m" \
    "optimize/scripts/logs/smoke_mobo_parity_${SLURM_JOB_ID}.out" | head -40
