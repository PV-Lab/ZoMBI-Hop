# Ensemble MOBO fleet: babysitter & `fleet.sh`

How the `ensemble_mobo_*` SLURM runs stay alive, recover from crashes, and how
to drive them by hand. Everything lives in `optimize/scripts/`.

## The three layers

Keeping the fleet running is split across three mechanisms so no single one has
to do everything:

1. **The `.sbatch` scripts self-heal the common cases.** Each
   `ensemble_mobo_<type>.sbatch`:
   - **Wall-time (12 h):** 120 s before the limit SLURM sends `USR1`; the script
     `scontrol requeue`s **itself** (same job id → same `run_dir` → resumes the
     interrupted trial). If requeue is refused it submits a fresh job instead.
   - **Clean exit (`rc=0`):** submits one replacement (a genuinely fresh run,
     new job id, new landscape seed).
   - **Fatal exit (`rc≠0`):** deliberately does **not** resubmit — a crash loop
     would spawn a dead `run_dir` every couple minutes. It prints
     `fatal abort/crash` to its `.out` and stops. This is the gap the babysitter
     covers.

2. **`babysitter.sh` — the cron watchdog.** One idempotent pass every ~10 min
   (cron on the login node). It tops the fleet back up to target and, on a real
   crash, sends a headless Claude to **diagnose and fix** it. Details below.

3. **`fleet.sh` — your manual control.** Start / stop / restart the fleet by hand
   without racing the babysitter. Use this instead of raw `sbatch`/`scancel`.

**Target fleet:** `2× ensemble_mobo_4d` + `2× ensemble_mobo_3d` running at all
times (`TARGET=2` per type in `babysitter.sh`).

---

## `fleet.sh` — manual control

Always prefer this over hand `sbatch`/`scancel`: every mutating command takes the
**same `flock` the babysitter uses**, so a manual action and a cron tick can
never race and double-submit a type. (That race is exactly what once left 4× 4d
running: a hand restart landed at the same second as the 10-min tick, which
sampled the fleet before the manual jobs registered and added two more.)

```
fleet.sh status                 # live counts per type vs target, + pause state
fleet.sh restart [SPEC ...]     # scancel ALL ensemble jobs, then bring up SPECs
fleet.sh kill  <TYPE|all>       # scancel a type (or all) AND pause the babysitter
fleet.sh resume [TYPE|all]      # clear the pause so the babysitter refills again
```

- **SPEC / TYPE** is any type with an `ensemble_mobo_<type>.sbatch` here (`4d`,
  `3d`, `10d`). A restart SPEC may carry a count: `4d:3` = three 4d jobs; a bare
  `4d` = one.
- **`restart` with no args** brings up the standard `4d:2 3d:2` fleet and clears
  every pause flag.
- **`kill` sets a pause flag** (`kill all` → global; `kill 4d` → just 4d) so the
  babysitter won't quietly bring back what you deliberately took down. It stays
  down until you `resume`.

Examples:

```
fleet.sh restart                # reset to the standard 2×4d + 2×3d
fleet.sh restart 4d:3 3d:1      # 3×4d + 1×3d
fleet.sh kill 4d                # take 4d down and KEEP it down
fleet.sh resume 4d              # let the babysitter bring 4d back to target
fleet.sh kill all               # stop the whole fleet and pause the watchdog
```

> **Note:** a plain `scancel` on its own does **not** stop the babysitter — the
> next tick will refill within ~10 min. If you want jobs to stay down, use
> `fleet.sh kill` (or `resume` afterward to re-enable).

---

## What the babysitter does each tick

For each type (`4d`, `3d`):

1. **Paused?** If `fleet.sh kill` set a pause flag for this type (or globally),
   skip it.
2. **At target?** `squeue` count ≥ 2 → stay quiet.
3. **Short?** Look at the most-recent **ended** job of that type and classify off
   its `.out` marker (not `sacct` state — the wrapper catches the crash and exits
   0 itself, so `sacct` shows `COMPLETED` even for an app crash):
   - **Environmental** (clean-exit replacement that didn't land, node failure,
     preemption, cancel): resubmit, under a 10-min cooldown so it can't spam.
   - **Fatal `fatal abort/crash`:** escalate to Claude (see below).
4. **Circuit breaker:** if the last **3** ended jobs of a type all crashed
   fatally, halt auto-recovery for that type and leave it down — a real bug that
   the auto-fix didn't resolve. Fix it yourself, then `fleet.sh restart`.

## Autonomy: diagnose **and fix**

On a fatal crash the babysitter fires one headless `claude -p` per dead job id
(detached; the `.pending` marker prevents double-diagnosing). Claude:

1. Reads the failure (`.err`/`.out`), finds the traceback.
2. Writes a report to `.babysitter/reports/job<jobid>.md`.
3. Acts on the diagnosis:
   - **Transient/environmental** → does *not* edit code; just resubmits.
   - **Code bug** → **edits the repo to fix it** (anywhere: `optimize/*.py`,
     `src/utils/*.py`, …), then **verifies**: an import check *and* a short real
     smoke run must exit 0. If it passes, it **commits only the changed files
     locally** (`babysitter: fix … (job <jobid>)`, **no push**) and resubmits. If
     it can't get a passing fix, it reverts, leaves the fleet short, and explains
     what's needed in the report.

**Guardrails.** No `--dangerously-skip-permissions`; the headless run is confined
to an explicit allowlist (read/edit source, scoped `uv run` smoke test, local
`git add`/`commit`, read-only SLURM queries, `sbatch` to refill). It **cannot**
`git push`, `scancel`, or rewrite history beyond one local commit. A **fix-lock**
serializes code-editing escalations so two simultaneous crashes can't corrupt the
same edit/commit. The circuit breaker still halts a type after 3 straight fatal
crashes.

> **After the babysitter auto-fixes something:** it committed to the current
> branch but did **not** push. Review with `git log --oneline` / `git show`, then
> push (or revert) yourself. Read `.babysitter/reports/job<jobid>.md` for the
> diagnosis, the commit hash, and the smoke-test result.

---

## Files & state (`optimize/scripts/.babysitter/`)

| Path | What it is |
|------|-----------|
| `babysitter.log` | Every pass's actions (submissions, escalations, skips). Start here. |
| `reports/job<id>.md` | Claude's diagnosis + fix report for a crashed job. |
| `reports/job<id>.claude.log` | Raw headless-Claude transcript for that escalation. |
| `reports/job<id>.pending` | "Escalation in flight" marker (auto-removed when done). |
| `babysitter.lock` | The `flock` shared by the babysitter and `fleet.sh`. |
| `fix.lock` | Serializes code-editing escalations. |
| `last_submit_<type>` | Timestamp of the last environmental resubmit (cooldown). |
| `paused`, `paused_<type>` | Pause flags set by `fleet.sh kill`, cleared by `resume`/`restart`. |

## Troubleshooting

- **A type is stuck down.** Check `paused`/`paused_<type>` (`fleet.sh status`
  shows them) — you may have `kill`ed it; `fleet.sh resume`. Otherwise the circuit
  breaker likely tripped: read the latest `reports/job<id>.md`, fix the bug, then
  `fleet.sh restart`.
- **Too many jobs of a type.** Almost always a manual `sbatch` that raced the
  cron tick — use `fleet.sh` next time. Trim with `fleet.sh restart` (resets to
  target).
- **Watch a pass live.** `tail -f optimize/scripts/.babysitter/babysitter.log`.
- **Did an auto-fix happen?** `git log --oneline --grep babysitter`.
