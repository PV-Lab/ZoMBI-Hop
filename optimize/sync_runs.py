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
REMOTE_DIR = "/home/adewinmb/ZoMBI-Hop/optimize/runs"
LOCAL_DIR = Path(__file__).parent / "runs"

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "LogLevel=ERROR",
]


def _remote_script(pattern: str, have: list[str], dry_run: bool) -> str:
    """Build the shell snippet that runs on ORCD over the single connection."""
    have_str = " ".join(have)
    dry = "1" if dry_run else "0"
    # `pattern` is interpolated unquoted on purpose so the remote shell expands
    # the glob. `have` names are validated to be glob-safe before we get here.
    return f"""
cd '{REMOTE_DIR}' || {{ echo "Cannot cd to {REMOTE_DIR}" >&2; exit 1; }}
have="{have_str}"
dry={dry}
to_get=""
for d in {pattern}; do
  [ -e "$d" ] || continue
  skip=0
  for h in $have; do [ "$h" = "$d" ] && {{ skip=1; break; }}; done
  [ "$skip" -eq 0 ] && to_get="$to_get $d"
done
if [ -z "$to_get" ]; then
  echo "Everything is already synced (or nothing matched the pattern)." >&2
  [ "$dry" -eq 0 ] && {{ printf 'SIZE 0\n'; tar czf - -T /dev/null; }}
  exit 0
fi
if [ "$dry" -eq 1 ]; then
  echo "Would download:" >&2
  for d in $to_get; do echo "  $d" >&2; done
  exit 0
fi
echo "Downloading:$to_get" >&2
# Emit a parseable size header (total uncompressed bytes) before the tar so the
# client can render a determinate progress bar, then stream the archive.
size=$(du -scb $to_get | tail -n1 | cut -f1)
printf 'SIZE %s\n' "$size"
tar czf - $to_get
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


def sync(pattern: str, dry_run: bool = False):
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(p.name for p in LOCAL_DIR.iterdir() if p.is_dir())
    # Names with whitespace would break the simple space-separated list we send
    # to the remote; such names never occur for run dirs, but guard anyway.
    safe = [n for n in existing if not any(c.isspace() for c in n)]
    if existing:
        print(f"{len(safe)} director(ies) already local; they'll be skipped.")

    remote_cmd = _remote_script(pattern, safe, dry_run)
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
                        tar.extract(member, LOCAL_DIR, filter="data")
                    except TypeError:
                        # `filter=` added in Python 3.12; fall back for older.
                        tar.extract(member, LOCAL_DIR)
                    bar.update(member.size)
    except tarfile.ReadError:
        proc.wait()
        print("Download failed: archive stream was incomplete.", file=sys.stderr)
        sys.exit(proc.returncode or 1)
    finally:
        if stream:
            stream.close()

    ret = proc.wait()
    if ret != 0:
        print(f"ssh exited with status {ret}.", file=sys.stderr)
        sys.exit(ret)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default="mobo_ensemble_*",
                        help="Glob pattern for directory names (default: mobo_ensemble_*)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()
    sync(args.pattern, args.dry_run)
