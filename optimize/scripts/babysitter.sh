#!/bin/bash
# =============================================================================
# babysitter.sh  —  keep the ensemble MOBO fleet alive, diagnose real breaks.
# -----------------------------------------------------------------------------
# ONE idempotent pass. Safe to run from cron (every ~10 min). Does NOT loop
# itself — the scheduler provides the cadence, so a hung pass can never wedge.
#
# Target fleet:  see the TARGET map below (currently 4x ensemble_mobo_10d; 3d/4d
#                are listed but paused). Kept alive at all times.
#
# The .sbatch scripts already self-heal the common cases: `scontrol requeue`
# across the 12h wall-time (same job id, stays in the queue) and an sbatch of a
# replacement on a CLEAN (rc=0) exit. This babysitter covers the gaps those
# leave:
#   * fatal rc!=0 exits  — the script deliberately does NOT resubmit (a crash
#     loop would spawn a dead run_dir every couple minutes). We escalate to
#     `claude -p` to DIAGNOSE (never edit code) and resubmit only if transient.
#   * node failure / preemption / TIMEOUT with requeue refused — environmental;
#     we just resubmit.
#   * a self-resubmit that silently failed to land — we top the fleet back up.
#
# Autonomy: DIAGNOSE + FIX + RESUBMIT. Claude reads the failure, writes a report
# under .babysitter/reports/, and then:
#   * transient/environmental cause  -> just resubmit.
#   * genuine CODE BUG               -> edit the repo to fix it, commit the fix
#     locally (specific files, no push), SMOKE-TEST the fix (import + a short
#     run), and resubmit ONLY if the smoke test passes. If it can't produce a
#     passing fix, it leaves the fleet short and says so in the report.
# Guardrails: one code-editing escalation at a time (fix-lock, so two crashes
# can't corrupt the same edit/commit); the 3-strike circuit breaker still halts a
# type whose last 3 jobs all crashed fatally; Claude may not push, scancel, or
# touch git history beyond a single local commit of the files it changed.
# =============================================================================
set -u

REPO="/orcd/scratch/orcd/013/adewinmb/ZoMBI-Hop"
# cron runs with a minimal environment — pin PATH so sbatch/squeue/claude/uv resolve.
export PATH="/home/adewinmb/.local/bin:/usr/bin:/bin:$PATH"
export HOME="${HOME:-/home/adewinmb}"

SCRIPTS="$REPO/optimize/scripts"
STATE="$SCRIPTS/.babysitter"
REPORTS="$STATE/reports"
LOG="$STATE/babysitter.log"
LOCK="$STATE/babysitter.lock"
mkdir -p "$REPORTS"

# sbatch paths in the .sbatch files are RELATIVE to the submit dir (logs/, run_mobo.py).
cd "$REPO" || { echo "cannot cd $REPO" >&2; exit 1; }

# Desired running+pending count, PER TYPE. Was a single global TARGET=2 back when
# every managed type wanted the same count; 10d wants 4 GPUs and 3d/4d want 2 each,
# so the target has to vary per type. The keys of this map are also the list of types
# the main loop walks — adding a type here (with an ensemble_mobo_<type>.sbatch beside
# this script) is all it takes to bring it under the babysitter.
#
# 3d and 4d are kept in the map at their old target but are currently PAUSED via
# fleet.sh (see .babysitter/paused_3d, paused_4d): their pooled history predates the
# 2026-08-11 dist_to_needles change and would mix metric scales in one GP. Leaving
# them listed-but-paused means `fleet.sh resume 3d` is all that is needed once their
# history is sorted, rather than another edit here.
declare -A TARGET=( [10d]=4 [4d]=2 [3d]=2 )
COOLDOWN=600             # min seconds between environmental resubmits of a type
CIRCUIT_BREAK_FAILS=3    # N consecutive FAILED for a type -> halt, stop escalating

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$LOG"; }

# --- single-flight: never let two passes overlap (claude can take minutes) ----
exec 9>"$LOCK"
if ! flock -n 9; then
    log "another pass is still running; skipping this tick."
    exit 0
fi

# IMPORTANT: sacct State is an unreliable "did the run crash?" signal here. The
# .sbatch wrapper CATCHES a non-zero run_mobo.py rc, prints a message, and then
# exits 0 itself — so an app crash shows up in sacct as COMPLETED, not FAILED.
# The trustworthy signal is the marker the wrapper writes to its .out log:
#   "fatal abort/crash) — NOT resubmitting"  <- real crash; wrapper gave up.
#   "exited cleanly (rc=0)" / "requeuing"    <- self-heal path (replacement, if
#                                                any, shows as PENDING in squeue).
# So we classify off the .out of the most-recent ended job, not off sacct state.

# Job IDs of the last N ended (non-active) jobs of a name, oldest..newest.
recent_ended_ids() {
    local name="$1" n="$2"
    sacct -u adewinmb -n -X --name "$name" --starttime now-3days \
          -o JobID,State -P 2>/dev/null \
        | awk -F'|' '$2!="RUNNING" && $2!="PENDING" && $2!="REQUEUED" \
                     && $2!~/^COMPLETING/ && $2!~/^CONFIGURING/ {print $1}' \
        | sort -n | tail -"$n"
}

# Did this job's wrapper report a fatal app crash?  crashed <name> <jobid>
crashed() {
    local f="$SCRIPTS/logs/${1}_${2}.out"
    [ -f "$f" ] && grep -q "fatal abort/crash" "$f"
}

resubmit() {                       # resubmit <script> <n>
    local script="$1" n="$2" i
    for ((i=0; i<n; i++)); do
        local out; out="$(sbatch "$script" 2>&1)"
        log "  sbatch $script -> $out"
    done
}

# Fire claude ONCE per dead job id, detached, so cron returns immediately and the
# next tick doesn't double-diagnose. The .pending marker is the "already handled"
# flag; claude turns it into <jobid>.md and removes .pending when done.
escalate() {                       # escalate <name> <script> <jobid> <need>
    local name="$1" script="$2" jobid="$3" need="$4"
    local pend="$REPORTS/job${jobid}.pending" rep="$REPORTS/job${jobid}.md"
    [ -e "$pend" ] || [ -e "$rep" ] && { log "  job $jobid already escalated; skip."; return; }
    : >"$pend"
    log "  ESCALATING job $jobid ($name) to claude (diagnose + fix + verify + resubmit)."

    local errf outf dim
    errf="$(ls -t "$SCRIPTS"/logs/${name}_${jobid}.err 2>/dev/null | head -1)"
    outf="$(ls -t "$SCRIPTS"/logs/${name}_${jobid}.out 2>/dev/null | head -1)"
    dim="${name##*_}"; dim="${dim%d}"    # ensemble_mobo_4d -> 4  (don't lean on the loop var)

    # Your working directory is already $REPO (this script cd'd there and the
    # detached subshell inherits it), so relative paths and a bare `sbatch` work.
    local prompt
    prompt="You are babysitting SLURM MOBO runs. Your working directory is the git
repo $REPO (branch is whatever is currently checked out). Job $jobid (name '$name',
submitted via $script) exited FATALLY with a non-zero return code. The fleet is now
SHORT by $need job(s) of this type; target is $target for this type.

Do ALL of this in order, then stop:
1. Read the failure. stderr: ${errf:-<none>}  stdout: ${outf:-<none>}.
   Find the Python traceback / exit reason.
2. Decide the root cause: TRANSIENT/environmental (GPU OOM, CUDA hiccup, NaN,
   filesystem/NFS blip, a shared-history race, a bad resumed run_dir, preemption)
   OR a genuine CODE BUG anywhere in the repo (optimize/*.py, src/utils/*.py, ...).
3. Write a concise markdown report to $rep: the exit signature, the root-cause
   diagnosis, the exact file:line, and what you did.
4. THEN act on the diagnosis:
   A) TRANSIENT/environmental (NOT a code bug): do NOT edit code. Restore the
      fleet by running exactly '$script' via sbatch, $need time(s). Note it in
      the report and stop.
   B) CODE BUG: fix it directly.
      - Edit the offending file(s) with the smallest correct change. Match the
        surrounding style. Do NOT reformat or refactor unrelated code.
      - VERIFY before you trust it — this is mandatory:
          * compile check: 'uv run python -m py_compile <each file you changed>'
            must succeed (the repo runs run_mobo.py as a script, not an installed
            package, so do NOT rely on 'import optimize.run_mobo');
          * smoke run: a short real run must exit 0. Start from
            'uv run python optimize/run_mobo.py --dataset ensemble --dim ${dim}
            --time-limit 0.05 --run-dir optimize/runs/_babysit_smoke_${jobid}'
            (bump the time-limit slightly if 0.05 is too short to initialize). If
            the crash was specific to a flag in '$script' (e.g. --share-history or
            --start-from-best), MIRROR those flags so the smoke run exercises the
            same code path. Read its output; a non-zero exit or traceback means
            your fix is wrong.
      - If the smoke test PASSES: commit ONLY the files you changed with an
        explicit 'git add <paths>' (never 'git add -A' / '.') and a message like
        'babysitter: fix <bug> causing $name fatal crash (job $jobid)'. Do NOT
        push. Do NOT commit unrelated working-tree changes. Then resubmit the
        fleet: run '$script' via sbatch, $need time(s). Record the commit hash and
        smoke result in the report.
      - If you CANNOT produce a fix that passes the smoke test: revert your edits
        ('git checkout -- <paths>'), do NOT resubmit, leave the fleet short so the
        user notices, and explain in the report what you tried and what is needed.
Constraints: never push, never scancel any job, never rewrite git history beyond
one local commit of the files you changed, never touch .git config or remotes.
Delete the smoke run dir (optimize/runs/_babysit_smoke_${jobid}) when done."

    # No --dangerously-skip-permissions. The headless run is confined to an
    # explicit allowlist covering exactly the diagnose+fix+verify+commit+resubmit
    # workflow: read-only inspection, editing/writing source, the scoped `uv run`
    # smoke test, a local `git add/commit` (NO push / rebase / reset), the
    # read-only SLURM queries, and the `sbatch` used to refill. `scancel`, `git
    # push`, and arbitrary shell are simply refused.
    #
    # fix-lock: serialize code-editing escalations so two simultaneous crashes
    # can't interleave edits/commits on the same tree. Read-only+sbatch actions
    # would be safe concurrently, but grabbing it unconditionally is simplest and
    # crashes are rare; a stuck holder is bounded by claude's own runtime.
    ( exec 8>"$STATE/fix.lock"
      flock 8
      claude -p "$prompt" \
        --allowedTools \
            "Read" "Write" "Edit" "Glob" "Grep" \
            "Bash(uv run:*)" \
            "Bash(git add:*)" "Bash(git commit:*)" "Bash(git checkout --:*)" \
            "Bash(git diff:*)" "Bash(git status:*)" "Bash(git log:*)" \
            "Bash(sbatch:*)" "Bash(squeue:*)" "Bash(sacct:*)" \
            "Bash(scontrol show:*)" "Bash(cat:*)" "Bash(tail:*)" \
            "Bash(head:*)" "Bash(grep:*)" "Bash(ls:*)" "Bash(rm -rf optimize/runs/_babysit_smoke_:*)" \
        >>"$REPORTS/job${jobid}.claude.log" 2>&1
      rm -f "$pend"
      log "  claude escalation for job $jobid finished (see job${jobid}.md)."
    ) </dev/null &
    disown
}

# ----------------------------------------------------------------------------
# Main: one pass over each type.
# ----------------------------------------------------------------------------
for type in "${!TARGET[@]}"; do
    name="ensemble_mobo_${type}"
    script="optimize/scripts/ensemble_mobo_${type}.sbatch"
    stamp="$STATE/last_submit_${type}"
    target="${TARGET[$type]}"

    # Manual pause (set by fleet.sh kill): a global "paused" flag halts every type,
    # a per-type "paused_<type>" halts just this one. Honor it so the babysitter
    # never quietly refills something the user deliberately took down.
    if [ -e "$STATE/paused" ] || [ -e "$STATE/paused_${type}" ]; then
        log "$name: paused by fleet.sh — skipping (fleet.sh resume $type to re-enable)."
        continue
    fi

    # squeue lists only active jobs (R/PD/CF/CG) -> that IS the live fleet count.
    active="$(squeue -u adewinmb -n "$name" -h -o '%i' 2>/dev/null | wc -l)"
    if [ "$active" -ge "$target" ]; then
        continue                                   # healthy; stay quiet
    fi
    need=$(( target - active ))
    log "$name: $active/$target active — short by $need. Investigating."

    mapfile -t ids < <(recent_ended_ids "$name" "$CIRCUIT_BREAK_FAILS")
    if [ "${#ids[@]}" -eq 0 ]; then
        log "  no ended $name jobs on record — cold start; resubmitting $need."
        resubmit "$script" "$need"; echo "$(date +%s)" >"$stamp"; continue
    fi

    # Circuit breaker: last N ended jobs ALL crashed the same fatal way -> stop.
    if [ "${#ids[@]}" -ge "$CIRCUIT_BREAK_FAILS" ]; then
        allfatal=1
        for id in "${ids[@]}"; do crashed "$name" "$id" || allfatal=0; done
        if [ "$allfatal" -eq 1 ]; then
            log "  CIRCUIT BREAKER: last $CIRCUIT_BREAK_FAILS $name jobs all crashed fatally."
            log "  Halting auto-recovery for $name. See .babysitter/reports/; fix + resubmit manually."
            continue
        fi
    fi

    jobid="${ids[-1]}"                              # most recent ended job
    if crashed "$name" "$jobid"; then
        # Real app crash -> Claude diagnoses (never resubmit blindly from bash).
        log "  most recent ended job $jobid crashed fatally -> escalating."
        escalate "$name" "$script" "$jobid" "$need"
    else
        # Clean exit whose replacement didn't land, node failure, preemption,
        # cancel — all environmental. Refill under a cooldown so we can't spam.
        now="$(date +%s)"
        last="$(cat "$stamp" 2>/dev/null || echo 0)"
        if [ $(( now - last )) -lt "$COOLDOWN" ]; then
            log "  shortfall but in cooldown ($(( COOLDOWN - (now-last) ))s left); waiting."
        else
            log "  environmental shortfall (last job $jobid exited non-fatally) -> resubmitting $need."
            resubmit "$script" "$need"
            echo "$now" >"$stamp"
        fi
    fi
done
