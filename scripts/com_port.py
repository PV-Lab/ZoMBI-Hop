"""
com_port.py
===========
COM-port ownership tools — the software replacement for physically unplugging
the USB-serial adapter.

Background
----------
A Windows serial port is exclusive: exactly one process may hold it. When a
ZoMBI serial process is killed in a way that skips its cleanup (hard
``TerminateProcess``, GUI closed while a serial child is running, terminal
window closed), the process may survive as an *orphan* that keeps the port
open forever. The next run then fails with::

    could not open port 'COM5': PermissionError(13, 'Access is denied.', None, 5)

Physically replugging the adapter is the brute-force cure: device removal
invalidates the orphan's handle. This module does the same thing in software —
find the stale ZoMBI process, kill it, confirm the port is free — plus, as a
last resort, a real device restart (``pnputil /restart-device``), which is
literally an unplug/replug without touching the cable.

Ownership marking
-----------------
Anything that opens a ZoMBI serial port should be launched with
``OWNER_ENV_VAR`` set to the port name (see :func:`owner_env`). The variable is
inherited by ``multiprocessing`` children, so a ``scripts/main.py`` tree is
identified even though its serial worker's command line is just the generic
``spawn_main`` bootstrap. Processes started before this convention existed are
still matched by command line.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes
from typing import Iterable, NamedTuple

_IS_WINDOWS = os.name == "nt"

#: Set on every process that may open a ZoMBI serial port; value is the port name.
OWNER_ENV_VAR = "ZOMBI_COM_OWNER"

#: Command-line fragments that identify a ZoMBI serial-owning process. Used as a
#: fallback for processes launched without OWNER_ENV_VAR (e.g. started by an
#: older build, or by hand from a terminal).
_OWNER_CMDLINE_HINTS = (
    "serial_only.py",
    "scripts/main.py",
    "scripts\\main.py",
    "-m scripts.main",
    "-m scripts.serial_only",
)

ERROR_FILE_NOT_FOUND = 2
ERROR_ACCESS_DENIED = 5


class Holder(NamedTuple):
    pid: int
    name: str
    cmdline: str
    why: str          # how we identified it: "env" or "cmdline"


def owner_env(port: str, base: dict | None = None) -> dict:
    """Return an environment dict tagged as owning ``port``.

    Launch every serial-owning subprocess with this so :func:`find_holders` can
    identify it later, including its ``multiprocessing`` descendants.
    """
    env = dict(os.environ if base is None else base)
    env[OWNER_ENV_VAR] = port
    return env


# ── port state ────────────────────────────────────────────────────────────────

def port_is_free(port: str) -> bool | None:
    """Is ``port`` openable right now?

    ``True``  — free.
    ``False`` — held by another process (ERROR_ACCESS_DENIED).
    ``None``  — port does not exist (adapter unplugged), or the check is not
                supported on this platform.

    Uses ``CreateFile`` with ``dwDesiredAccess=0``: serial drivers still enforce
    exclusive access on open, but with no access requested no DCB is applied, so
    a *successful* probe does not disturb DTR/RTS or the port's settings.
    """
    if not _IS_WINDOWS:
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                wintypes.HANDLE]
    invalid = wintypes.HANDLE(-1).value
    OPEN_EXISTING = 3
    h = k32.CreateFileW(rf"\\.\{port}", 0, 0, None, OPEN_EXISTING, 0, None)
    if h == invalid:
        err = ctypes.get_last_error()
        if err == ERROR_FILE_NOT_FOUND:
            return None
        return False
    k32.CloseHandle(h)
    return True


def _self_and_ancestors() -> set[int]:
    """PIDs we must never kill: this process and everything above it."""
    pids = {os.getpid()}
    try:
        import psutil
        p = psutil.Process()
        for anc in p.parents():
            pids.add(anc.pid)
    except Exception:
        pass
    return pids


def find_holders(port: str) -> list[Holder]:
    """ZoMBI serial processes that are candidates for holding ``port``.

    Matches on the ``OWNER_ENV_VAR`` environment marker first (this also catches
    ``multiprocessing`` spawn children, whose command line carries no script
    name), then on known script names in the command line. Never returns this
    process or any of its ancestors.
    """
    try:
        import psutil
    except ImportError:
        return []

    protected = _self_and_ancestors()
    found: list[Holder] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        pid = proc.info.get("pid")
        if pid in protected:
            continue
        name = (proc.info.get("name") or "").lower()
        if "python" not in name:
            continue
        cmdline = " ".join(proc.info.get("cmdline") or [])

        why = ""
        try:
            if (proc.environ() or {}).get(OWNER_ENV_VAR, "").upper() == port.upper():
                why = "env"
        except Exception:
            pass  # access denied / process gone — fall through to cmdline match
        if not why and any(h in cmdline for h in _OWNER_CMDLINE_HINTS) \
                and _cmdline_owns_port(cmdline, port):
            why = "cmdline"
        if why:
            found.append(Holder(pid, proc.info.get("name") or "?", cmdline, why))
    return found


def _cmdline_owns_port(cmdline: str, port: str) -> bool:
    """Could a ZoMBI process with this command line be holding ``port``?

    A ZoMBI serial process names its port on the command line (``--com COM5``);
    when it does, only that port counts — otherwise releasing COM1 could kill the
    process that legitimately owns COM5. When no port appears (``scripts/main.py``
    compiles its port in) the process owns exactly one port, so treat it as a
    candidate and let the caller's post-kill port re-probe be the real check.
    """
    import re
    named = {m.upper() for m in re.findall(r"\bCOM\d+\b", cmdline, re.IGNORECASE)}
    return port.upper() in named if named else True


# ── releasing ─────────────────────────────────────────────────────────────────

def release_port(port: str, timeout: float = 8.0, log=print) -> bool:
    """Make ``port`` openable again by killing stale ZoMBI serial processes.

    Returns True when the port is free (or already was). Only ever kills
    processes identified by :func:`find_holders` — never an unrelated program
    that happens to hold the port (a terminal emulator, Arduino IDE, …); those
    are reported so the user can close them.
    """
    state = port_is_free(port)
    if state is True:
        return True
    if state is None:
        log(f"[com_port] {port} is not present — is the USB adapter plugged in?")
        return False

    holders = find_holders(port)
    if not holders:
        log(f"[com_port] {port} is held, but no stale ZoMBI serial process was found. "
            f"Another program (terminal, Arduino IDE, another user session) has it open.")
        return False

    for h in holders:
        log(f"[com_port] Killing stale serial process pid={h.pid} ({h.why}): {h.cmdline[:140]}")
        _kill_tree(h.pid, log=log)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_free(port) is True:
            log(f"[com_port] ✅ {port} released.")
            return True
        time.sleep(0.25)

    log(f"[com_port] ⚠️ {port} still held {timeout:.0f}s after killing "
        f"{len(holders)} process(es).")
    return False


def _kill_tree(pid: int, log=print) -> None:
    """Kill ``pid`` and its descendants. Descendants first, so a parent cannot
    respawn or re-parent them mid-kill."""
    try:
        import psutil
        proc = psutil.Process(pid)
        victims = proc.children(recursive=True) + [proc]
    except Exception:
        victims = []

    if victims:
        for v in victims:
            try:
                v.kill()
            except Exception:
                pass
        try:
            import psutil
            psutil.wait_procs(victims, timeout=3)
        except Exception:
            pass
        return

    # psutil unavailable or the process is an orphan we could not open: fall back
    # to taskkill. /T only reaches live children of a live parent, so this is a
    # best-effort path.
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, check=False)
    except Exception as e:
        log(f"[com_port] taskkill for pid={pid} failed: {e}")


# ── last resort: software replug ──────────────────────────────────────────────

def device_instance_id(port: str) -> str | None:
    """The PnP instance id backing ``port`` (e.g.
    ``FTDIBUS\\VID_0403+PID_6001+BG00KEFWA\\0000``), or None."""
    if not _IS_WINDOWS:
        return None
    ps = (
        "Get-PnpDevice -Class Ports -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.FriendlyName -like '*({port})' }} | "
        "Select-Object -First 1 -ExpandProperty InstanceId"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
        inst = (out.stdout or "").strip()
        return inst or None
    except Exception:
        return None


def restart_device(port: str, log=print) -> bool:
    """Software unplug/replug of the adapter behind ``port``.

    This is exactly what pulling the USB cable does — the device is removed and
    re-enumerated, which invalidates every open handle. Needs Administrator; if
    it is not available the caller should fall back to asking the user to
    replug. Only worth reaching for when :func:`release_port` could not identify
    the holder, or when the port opens but the link is genuinely dead.
    """
    inst = device_instance_id(port)
    if not inst:
        log(f"[com_port] Could not resolve a device instance id for {port}.")
        return False
    log(f"[com_port] Restarting device {inst} (software replug of {port})…")
    try:
        out = subprocess.run(["pnputil", "/restart-device", inst],
                             capture_output=True, text=True, timeout=60)
    except Exception as e:
        log(f"[com_port] pnputil failed: {e}")
        return False
    if out.returncode != 0:
        log(f"[com_port] pnputil returned {out.returncode}: "
            f"{(out.stdout or '').strip()} {(out.stderr or '').strip()} "
            f"(this usually means it needs to run as Administrator)")
        return False
    for _ in range(20):                       # re-enumeration takes a moment
        time.sleep(0.5)
        if port_is_free(port) is True:
            log(f"[com_port] ✅ {port} restarted and free.")
            return True
    log(f"[com_port] Device restarted but {port} is still not openable.")
    return False


def ensure_port_available(port: str, allow_device_restart: bool = False,
                          log=print) -> bool:
    """Best-effort: make ``port`` openable. Kill stale ZoMBI holders, and only
    if that fails and ``allow_device_restart`` is set, do a software replug."""
    if release_port(port, log=log):
        return True
    if allow_device_restart:
        return restart_device(port, log=log)
    return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m scripts.com_port",
        description="Inspect and release a COM port held by a stale ZoMBI serial process.")
    ap.add_argument("port", nargs="?", default="COM5", help="Port name (default COM5)")
    ap.add_argument("--status", action="store_true",
                    help="Only report who holds the port; change nothing.")
    ap.add_argument("--restart-device", action="store_true",
                    help="If killing stale processes is not enough, do a software "
                         "unplug/replug of the adapter (needs Administrator).")
    args = ap.parse_args(argv)

    state = port_is_free(args.port)
    label = {True: "FREE", False: "HELD", None: "NOT PRESENT"}[state]
    print(f"{args.port}: {label}")
    holders = find_holders(args.port)
    for h in holders:
        print(f"  candidate holder pid={h.pid} ({h.why}) {h.cmdline[:160]}")
    if not holders:
        print("  no ZoMBI serial process identified")

    if args.status:
        return 0 if state is True else 1
    ok = ensure_port_available(args.port, allow_device_restart=args.restart_device)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
