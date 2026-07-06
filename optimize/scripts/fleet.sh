#!/bin/bash
# =============================================================================
# fleet.sh  —  manual control of the ensemble MOBO SLURM fleet.
# -----------------------------------------------------------------------------
# One tool to start / stop / restart the ensemble_mobo_* runs by hand WITHOUT
# fighting the cron babysitter. Every mutating command takes the SAME flock the
# babysitter uses, so a `fleet.sh` action and a babysitter tick can never race
# and double-submit a type (the bug that once left 4x 4d running).
#
# It also drives a pause flag the babysitter honors: a hand `kill` sets it so the
# babysitter won't quietly refill what you deliberately took down. `restart`
# clears it; `resume` clears it explicitly.
#
#   fleet.sh status                 show live counts per type vs target
#   fleet.sh restart [SPEC ...]     scancel ALL ensemble jobs, then bring up SPECs
#                                   (default: 4d:2 3d:2). Clears any pause.
#   fleet.sh kill  <TYPE|all>       scancel a type (or all) and PAUSE the
#                                   babysitter for it so it stays down.
#   fleet.sh resume [TYPE|all]      clear the pause so the babysitter refills again
#
# SPEC / TYPE is a job type that has an ensemble_mobo_<type>.sbatch here, e.g.
# 4d, 3d, 10d. A restart SPEC may carry a count: `4d:3` submits three 4d jobs.
# A bare `4d` means one. Examples:
#   fleet.sh restart                # 2x4d + 2x3d, the standard fleet
#   fleet.sh restart 4d:3 3d:1      # 3x4d + 1x3d
#   fleet.sh kill 4d                # take 4d down and keep it down
#   fleet.sh resume 4d              # let the babysitter bring 4d back
# =============================================================================
set -u

REPO="/orcd/scratch/orcd/013/adewinmb/ZoMBI-Hop"
export PATH="/home/adewinmb/.local/bin:/usr/bin:/bin:$PATH"
export HOME="${HOME:-/home/adewinmb}"

SCRIPTS="$REPO/optimize/scripts"
STATE="$SCRIPTS/.babysitter"
LOCK="$STATE/babysitter.lock"
LOG="$STATE/babysitter.log"
mkdir -p "$STATE"
cd "$REPO" || { echo "cannot cd $REPO" >&2; exit 1; }

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] fleet.sh: $*" | tee -a "$LOG"; }
die() { echo "fleet.sh: $*" >&2; exit 1; }

# Types we know how to launch = the ensemble_mobo_<type>.sbatch files present.
known_types() {
    ls "$SCRIPTS"/ensemble_mobo_*.sbatch 2>/dev/null \
        | sed -E 's#.*/ensemble_mobo_(.+)\.sbatch#\1#'
}
is_known()   { known_types | grep -qx "$1"; }
script_for() { echo "optimize/scripts/ensemble_mobo_${1}.sbatch"; }
active_count() { squeue -u adewinmb -n "ensemble_mobo_${1}" -h -o '%i' 2>/dev/null | wc -l; }

# --- take the babysitter's lock so no tick runs while we mutate the fleet ------
# Blocking with a timeout: a babysitter pass can take minutes if it's escalating.
grab_lock() {
    exec 9>"$LOCK"
    if ! flock -w 180 9; then
        die "could not acquire babysitter lock within 180s (a pass may be escalating); try again shortly."
    fi
}

cmd_status() {
    printf '%-20s %-8s %s\n' "TYPE" "ACTIVE" "PAUSED?"
    local paused_all=""
    [ -e "$STATE/paused" ] && paused_all="ALL-PAUSED"
    for t in $(known_types); do
        local p=""
        { [ -n "$paused_all" ] || [ -e "$STATE/paused_${t}" ]; } && p="paused"
        printf '%-20s %-8s %s\n' "ensemble_mobo_${t}" "$(active_count "$t")" "$p"
    done
    [ -n "$paused_all" ] && echo "NOTE: global pause is set — the babysitter is refilling nothing until 'fleet.sh resume'."
    return 0
}

# scancel every ensemble type and wait until squeue shows them gone.
scancel_all_types() {
    local t any=0
    for t in $(known_types); do
        if [ "$(active_count "$t")" -gt 0 ]; then
            scancel -u adewinmb --name "ensemble_mobo_${t}" 2>/dev/null && any=1
            log "scancel'd all ensemble_mobo_${t}"
        fi
    done
    [ "$any" -eq 1 ] || return 0
    local waited=0
    while [ "$waited" -lt 60 ]; do
        local n=0
        for t in $(known_types); do n=$(( n + $(active_count "$t") )); done
        [ "$n" -eq 0 ] && return 0
        sleep 2; waited=$(( waited + 2 ))
    done
    log "WARNING: some jobs still draining after 60s; proceeding anyway."
}

cmd_restart() {
    local -a specs=("$@")
    [ "${#specs[@]}" -eq 0 ] && specs=(4d:2 3d:2)   # the standard fleet
    # Validate before we cancel anything.
    local s type count
    for s in "${specs[@]}"; do
        type="${s%%:*}"; count="${s##*:}"; [ "$type" = "$count" ] && count=1
        is_known "$type" || die "unknown type '$type' (known: $(known_types | tr '\n' ' '))"
        [[ "$count" =~ ^[0-9]+$ ]] || die "bad count in spec '$s'"
    done
    grab_lock
    log "restart requested: ${specs[*]}"
    scancel_all_types
    rm -f "$STATE"/paused "$STATE"/paused_* 2>/dev/null   # a restart un-pauses everything
    for s in "${specs[@]}"; do
        type="${s%%:*}"; count="${s##*:}"; [ "$type" = "$count" ] && count=1
        local i out
        for ((i=0; i<count; i++)); do
            out="$(sbatch "$(script_for "$type")" 2>&1)"
            log "  sbatch $(script_for "$type") -> $out"
        done
        # Stamp last_submit so the babysitter's environmental cooldown also
        # counts from now — no redundant top-up right after we launch.
        echo "$(date +%s)" >"$STATE/last_submit_${type}"
    done
    echo; cmd_status
}

cmd_kill() {
    local target="${1:-}"
    [ -n "$target" ] || die "usage: fleet.sh kill <type|all>"
    grab_lock
    if [ "$target" = "all" ]; then
        : >"$STATE/paused"                 # global pause: babysitter refills nothing
        log "kill all: pausing babysitter globally + cancelling every ensemble type"
        scancel_all_types
    else
        is_known "$target" || die "unknown type '$target' (known: $(known_types | tr '\n' ' '))"
        : >"$STATE/paused_${target}"       # per-type pause
        scancel -u adewinmb --name "ensemble_mobo_${target}" 2>/dev/null
        log "kill $target: paused + cancelled ensemble_mobo_${target}"
    fi
    echo "Paused. The babysitter will NOT refill this until you run: fleet.sh resume ${target}"
    echo; cmd_status
}

cmd_resume() {
    local target="${1:-all}"
    grab_lock
    if [ "$target" = "all" ]; then
        rm -f "$STATE"/paused "$STATE"/paused_* 2>/dev/null
        log "resume all: cleared every pause flag"
    else
        is_known "$target" || die "unknown type '$target'"
        rm -f "$STATE/paused_${target}" 2>/dev/null
        # If a global pause is set, resuming one type means lifting global and
        # re-pausing the others, so the intent ("bring 4d back") is honored.
        if [ -e "$STATE/paused" ]; then
            local t
            for t in $(known_types); do [ "$t" = "$target" ] || : >"$STATE/paused_${t}"; done
            rm -f "$STATE/paused" 2>/dev/null
        fi
        log "resume $target: cleared its pause flag"
    fi
    echo "Resumed. The babysitter will top this back up to target on its next tick."
    echo; cmd_status
}

case "${1:-status}" in
    status)  cmd_status ;;
    restart) shift; cmd_restart "$@" ;;
    kill)    shift; cmd_kill "$@" ;;
    resume)  shift; cmd_resume "$@" ;;
    -h|--help|help) sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown command '$1' (try: status | restart | kill | resume | help)" ;;
esac
