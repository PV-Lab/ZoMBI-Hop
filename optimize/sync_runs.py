"""Sync run directories from ORCD, skipping directories that already exist locally.

Everything happens over a single SSH connection, so you are prompted for your
password exactly once (Windows OpenSSH does not support ControlMaster
connection multiplexing, so that approach can't work here).

How it works:
  1. We collect the names of the run directories we already have locally.
  2. We open ONE ssh connection and hand the remote that list plus the glob
     pattern. The remote figures out which matching directories are new and
     streams them back as a gzip'd tar on stdout.
  3. We extract that stream locally with Python's tarfile (no local tar/scp
     needed). Progress messages come back on the connection's stderr.
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

from tqdm import tqdm

REMOTE = "adewinmb@orcd-login.mit.edu"
LOCAL_DIR = Path(__file__).parent / "runs"

# Shared-history sync (mirrors --share-history in run_mobo.py): when
# SHARE_COLLABORATOR_HISTORY is set, ALSO pull run directories from the
# collaborator's runs dir so this checkout accumulates both users' history
# locally; otherwise only the current user's own remote runs dir is synced.
#
# The dir list and toggle come from collab_dirs (the single definition shared with
# run_mobo and pareto) so the paths can't drift. These are evaluated on the ORCD
# login node: each user's ~/orcd/scratch runs dir, reachable as adewinmb over ssh
# via the sched_mit_hill group grant. Always sync the first (this account's own
# dir, since we ssh in as adewinmb); the rest are added only when sharing is on.
try:
    from optimize.collab_dirs import (
        SHARE_COLLABORATOR_HISTORY,
        COLLABORATOR_RUNS_DIRS as _COLLABORATOR_RUNS_DIRS,
    )
except ImportError:
    from collab_dirs import (
        SHARE_COLLABORATOR_HISTORY,
        COLLABORATOR_RUNS_DIRS as _COLLABORATOR_RUNS_DIRS,
    )
REMOTE_DIRS = (
    _COLLABORATOR_RUNS_DIRS if SHARE_COLLABORATOR_HISTORY
    else _COLLABORATOR_RUNS_DIRS[:1]
)

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "LogLevel=ERROR",
]


def _remote_script(pattern: str, have: list[str], dirs: list[str],
                   dry_run: bool, only: list[str] | None = None) -> str:
    """Build the shell snippet that runs on ORCD over the single connection.

    Scans each directory in *dirs* in order, collecting matching run dirs we
    don't already have. A name seen in an earlier dir wins, so a run present in
    more than one dir is fetched only once (and extracts flat into LOCAL_DIR).

    When *only* is given (an allowlist of run names, e.g. from --pareto/--top-k),
    any matching dir whose name is not in the list is skipped.
    """
    have_str = " ".join(have)
    dirs_str = " ".join(f"'{d}'" for d in dirs)
    only_str = " ".join(only) if only else ""
    dry = "1" if dry_run else "0"
    # `pattern` is interpolated unquoted on purpose so the remote shell expands
    # the glob (relative to each base dir). `have` names are validated to be
    # glob-safe before we get here; `dirs` are our own trusted constants.
    return f"""
have="{have_str}"
only="{only_str}"
dry={dry}
got_names="$have"   # names already chosen (or local) — used to dedupe
tar_args=""         # "-C <base> <name> ..." pairs for a single flat tar
du_paths=""         # full paths for the size header
found=0
for base in {dirs_str}; do
  [ -d "$base" ] || {{ echo "Skipping missing dir: $base" >&2; continue; }}
  for d in "$base"/{pattern}; do
    [ -e "$d" ] || continue
    name=$(basename "$d")
    if [ -n "$only" ]; then
      keep=0
      for o in $only; do [ "$o" = "$name" ] && {{ keep=1; break; }}; done
      [ "$keep" -eq 0 ] && continue
    fi
    skip=0
    for h in $got_names; do [ "$h" = "$name" ] && {{ skip=1; break; }}; done
    [ "$skip" -eq 1 ] && continue
    got_names="$got_names $name"
    tar_args="$tar_args -C $base $name"
    du_paths="$du_paths $base/$name"
    found=1
    echo "  $name  ($base)" >&2
  done
done
if [ "$found" -eq 0 ]; then
  echo "Everything is already synced (or nothing matched the pattern)." >&2
  [ "$dry" -eq 0 ] && {{ printf 'SIZE 0\n'; tar czf - -T /dev/null; }}
  exit 0
fi
if [ "$dry" -eq 1 ]; then
  echo "(above is what would be downloaded)" >&2
  exit 0
fi
echo "Downloading the directories listed above." >&2
# Emit a parseable size header (total uncompressed bytes) before the tar so the
# client can render a determinate progress bar, then stream the archive.
size=$(du -scb $du_paths | tail -n1 | cut -f1)
printf 'SIZE %s\n' "$size"
tar czf - $tar_args
"""


def _read_size_header(stream) -> int | None:
    """Read the leading ``SIZE <bytes>\\n`` line from the remote's stdout.

    Returns the total byte count, or None if the stream ended before a valid
    header arrived (which means ssh failed before streaming anything).
    """
    line = bytearray()
    while not line.endswith(b"\n"):
        b = stream.read(1)
        if not b:
            return None
        line += b
    text = line.decode("utf-8", "replace").strip()
    if not text.startswith("SIZE "):
        return None
    try:
        return int(text[len("SIZE "):])
    except ValueError:
        return None


# ─── Pareto / top-k run selection ────────────────────────────────────────────────
# Selection mirrors optimize/pareto.py (kept in sync with it), reimplemented with
# the standard library so this sync utility needs neither numpy nor matplotlib. The
# front is over three MINIMISED objectives — dist_to_needles, dup_fraction, and a
# time metric whose key varies by run age (avg_time_per_iter_s preferred, legacy
# runtime_s fallback). Trials missing any objective, or with time <= 0 (a failed
# 0-iteration run whose failure-sentinel scores masquerade as Pareto-optimal), are
# dropped before the front is computed — exactly as pareto.py excludes them.
_DIST_KEY = "dist_to_needles"
_DUP_KEY = "dup_fraction"
_TIME_KEYS = ("avg_time_per_iter_s", "runtime_s")


def _time_metric(metrics: dict) -> float | None:
    for k in _TIME_KEYS:
        if k in metrics:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                return None
    return None


def _pareto_mask_min(M: list[list[float]]) -> list[bool]:
    """Boolean mask of non-dominated rows of *M* (all columns minimised); a row is
    kept iff nothing dominates it. Mirrors optimize/pareto.py:pareto_mask_min."""
    n = len(M)
    keep = [True] * n
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            if (all(M[j][k] <= M[i][k] for k in range(len(M[i])))
                    and any(M[j][k] < M[i][k] for k in range(len(M[i])))):
                keep[i] = False
                break
    return keep


def _metrics_script(pattern: str, dirs: list[str]) -> str:
    """Shell snippet that streams each matching run's mobo_progress.json back,
    delimited by ``@@@RUN <name>`` / ``@@@END`` so the client can parse per run."""
    dirs_str = " ".join(f"'{d}'" for d in dirs)
    return f"""
for base in {dirs_str}; do
  [ -d "$base" ] || continue
  for d in "$base"/{pattern}; do
    [ -e "$d" ] || continue
    prog="$d/mobo_progress.json"
    [ -f "$prog" ] || continue
    printf '@@@RUN %s\\n' "$(basename "$d")"
    cat "$prog"
    printf '\\n@@@END\\n'
  done
done
"""


def fetch_run_metrics(pattern: str) -> dict[str, dict]:
    """One SSH round-trip: return {run_name: mobo_progress.json dict} for every
    matching remote run. Used only when a selection flag is set (so filtering costs
    one extra connection / password prompt on top of the download)."""
    remote_cmd = _metrics_script(pattern, REMOTE_DIRS)
    ssh_cmd = ["ssh", *SSH_OPTS, REMOTE, remote_cmd]
    print("Fetching run metrics from ORCD to plan the download "
          "(you'll be prompted for your password)...")
    proc = subprocess.run(ssh_cmd, stdout=subprocess.PIPE, text=True)
    out = proc.stdout or ""
    metrics: dict[str, dict] = {}
    name = None
    buf: list[str] = []
    for line in out.splitlines():
        if line.startswith("@@@RUN "):
            name, buf = line[len("@@@RUN "):].strip(), []
        elif line == "@@@END":
            if name is not None:
                try:
                    metrics[name] = json.loads("\n".join(buf))
                except json.JSONDecodeError:
                    print(f"  [metrics] {name}: unreadable mobo_progress.json; skipping.",
                          file=sys.stderr)
            name, buf = None, []
        elif name is not None:
            buf.append(line)
    return metrics


def select_runs(metrics: dict[str, dict], pareto: bool, top_k: int | None) -> list[str]:
    """Run names to download, applying --pareto then --top-k (they stack).

    Builds one record per usable trial (tagged with its run), drops failed/malformed
    trials, then: with *pareto*, keeps runs contributing >=1 Pareto-optimal trial to
    the pooled front; with *top_k*, keeps the K runs with the best (lowest)
    dist_to_needles among whichever runs survive the pareto step.
    """
    recs: list[tuple[str, float, float, float]] = []   # (run, dist, dup, time)
    for name, prog in metrics.items():
        for t in prog.get("trials", []):
            m = t.get("metrics", {})
            if _DIST_KEY not in m or _DUP_KEY not in m:
                continue
            tv = _time_metric(m)
            if tv is None or tv <= 0:
                continue
            try:
                recs.append((name, float(m[_DIST_KEY]), float(m[_DUP_KEY]), tv))
            except (TypeError, ValueError):
                continue
    if not recs:
        return []

    keep_names = {r[0] for r in recs}
    if pareto:
        mask = _pareto_mask_min([[r[1], r[2], r[3]] for r in recs])
        keep_names = {recs[i][0] for i, k in enumerate(mask) if k}
        print(f"  --pareto: {sum(mask)} Pareto-optimal trial(s) across "
              f"{len(recs)} usable -> {len(keep_names)} run(s).")

    if top_k is not None:
        best_dist: dict[str, float] = {}
        for name, dist, _dup, _t in recs:
            if name in keep_names:
                best_dist[name] = min(best_dist.get(name, float("inf")), dist)
        ranked = sorted(best_dist, key=lambda n: best_dist[n])[:top_k]
        keep_names = set(ranked)
        print(f"  --top-k {top_k}: kept "
              + ", ".join(f"{n} (dist={best_dist[n]:.4f})" for n in ranked))

    return sorted(keep_names)


def sync(pattern: str, dry_run: bool = False, only_names: list[str] | None = None):
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(p.name for p in LOCAL_DIR.iterdir() if p.is_dir())
    # Names with whitespace would break the simple space-separated list we send
    # to the remote; such names never occur for run dirs, but guard anyway.
    safe = [n for n in existing if not any(c.isspace() for c in n)]
    if existing:
        print(f"{len(safe)} director(ies) already local; they'll be skipped.")

    if len(REMOTE_DIRS) > 1:
        print(f"Sharing collaborator history; scanning: {', '.join(REMOTE_DIRS)}")

    remote_cmd = _remote_script(pattern, safe, REMOTE_DIRS, dry_run, only=only_names)
    ssh_cmd = ["ssh", *SSH_OPTS, REMOTE, remote_cmd]

    print("Connecting to ORCD (you'll be prompted for your password once)...")

    if dry_run:
        # No tar stream to extract; let ssh inherit stdout/stderr.
        subprocess.run(ssh_cmd)
        return

    # stdout = "SIZE <bytes>\n" header followed by the tar stream (binary).
    # stderr inherits the terminal so the password prompt and progress messages
    # show up; stdin inherits so ssh can read the password from the console.
    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE)
    stream = proc.stdout
    try:
        total = _read_size_header(stream)
        if total is None:
            # No header => ssh failed before producing output (bad password,
            # connection refused, ...). stderr already showed the reason.
            proc.wait()
            print("Download failed: no data received from ORCD.", file=sys.stderr)
            sys.exit(proc.returncode or 1)

        with tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc="Syncing", disable=(total == 0),
        ) as bar:
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                for member in tar:
                    try:
                        try:
                            tar.extract(member, LOCAL_DIR, filter="data")
                        except TypeError:
                            # `filter=` added in Python 3.12; fall back for older.
                            tar.extract(member, LOCAL_DIR)
                    except (OSError, KeyError) as e:
                        # A file that changed/vanished on the remote mid-read can
                        # produce a truncated or unreadable member. Skip it
                        # rather than aborting the whole sync.
                        print(f"Skipping {member.name}: {e}", file=sys.stderr)
                    bar.update(member.size)
    except tarfile.ReadError:
        proc.wait()
        print("Download failed: archive stream was incomplete.", file=sys.stderr)
        sys.exit(proc.returncode or 1)
    finally:
        if stream:
            stream.close()

    ret = proc.wait()
    if ret == 1:
        # tar returns 1 for non-fatal warnings (e.g. "file changed as we read
        # it" when a run is still being written). The archive is complete, so
        # treat this as a warning and finish normally.
        print("Some files changed during transfer (warning); archive is "
              "otherwise complete.", file=sys.stderr)
    elif ret != 0:
        print(f"ssh exited with status {ret}.", file=sys.stderr)
        sys.exit(ret)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default=None,
                        help="Glob pattern for directory names (default: mobo_ensemble_*)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ensemble", action="store_true",
                       help="Sync the ensemble runs (mobo_ensemble_*).")
    group.add_argument("--hebo", action="store_true",
                       help="Sync the HEBO runs (mobo_hebo_*).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--pareto", action="store_true",
                        help="Only download runs that contribute a Pareto-optimal trial "
                             "(over dist_to_needles, dup_fraction, time); mirrors "
                             "optimize/pareto.py. Costs one extra connection to fetch metrics.")
    parser.add_argument("--top-k", type=int, default=None, metavar="K",
                        help="Only download the K runs with the best (lowest) "
                             "dist_to_needles. Stacks with --pareto (ranks among the "
                             "Pareto set). Also costs the extra metrics connection.")
    args = parser.parse_args()

    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be a positive integer")

    # A positional pattern always wins; otherwise the convenience flags pick a
    # preset, falling back to the historical ensemble default.
    if args.pattern is not None:
        pattern = args.pattern
    elif args.hebo:
        pattern = "mobo_hebo_*"
    elif args.ensemble:
        pattern = "mobo_ensemble_*"
    else:
        pattern = "mobo_ensemble_*"

    # Selection flags: fetch metrics first (one extra connection), decide which runs
    # to pull, then hand that allowlist to the normal download.
    only_names = None
    if args.pareto or args.top_k is not None:
        metrics = fetch_run_metrics(pattern)
        if not metrics:
            print("No run metrics found on the remote for that pattern; nothing to sync.")
            sys.exit(0)
        only_names = select_runs(metrics, args.pareto, args.top_k)
        if not only_names:
            print("No runs selected after filtering; nothing to sync.")
            sys.exit(0)
        print(f"Selected {len(only_names)} run(s) to sync: {', '.join(only_names)}")

    sync(pattern, args.dry_run, only_names=only_names)
