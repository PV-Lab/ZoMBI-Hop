"""
ZoMBI-Hop v2 Main Launcher with Database Communication

This script starts two parallel processes:
1. Serial communication process (Archerfish interface)
2. ZoMBI-Hop v2 optimization process

Usage:
    python -m scripts.main              # Start new trial
    python -m scripts.main <uuid>       # Resume trial from UUID
    python -m scripts.main list         # List available trials

Examples:
    python -m scripts.main              # New trial
    python -m scripts.main a2fe         # Resume trial with UUID 'a2fe'
    python -m scripts.main list         # Show all available trials

The trial UUID will be printed when starting. Use Ctrl+C to stop gracefully.
Checkpoints are saved automatically to: actual_runs/checkpoints/run_<uuid>/
"""

import os
import multiprocessing
import signal
import threading
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.initialize_databases import initialize_db
from scripts.communication import start_serial_dual_io_shared_port
from scripts import communication
from scripts.run_zombi_main import run_zombi_main


def list_runs_and_exit():
    base = Path('actual_runs')
    print("Available trials in 'actual_runs':")
    print("="*80)
    if not base.exists():
        print("No trials found.")
        return
    trials = [d for d in base.iterdir() if d.is_dir()]
    if not trials:
        print("No trials found.")
    else:
        for td in sorted(trials):
            print(f"\nTrial directory: {td.name}")
            checkpoints_dir = td / 'checkpoints'
            if checkpoints_dir.exists():
                for run_dir in sorted(checkpoints_dir.iterdir()):
                    if run_dir.is_dir() and run_dir.name.startswith('run_'):
                        uuid = run_dir.name.replace('run_', '')
                        meta = td / 'trial_metadata.json'
                        if meta.exists():
                            import json
                            with open(meta, 'r') as f:
                                m = json.load(f)
                            print(f"  UUID: {uuid} ({m.get('num_minima','?')} minima, {m.get('dimensions','?')}D, {m.get('time_limit_hours','?')}h)")
                        else:
                            print(f"  UUID: {uuid}")
            else:
                print("  No checkpoints found")


def start_serial(parent_shutdown: "multiprocessing.synchronize.Event"):
    """Child process: opens COM; parent sets parent_shutdown to release the port."""
    try:
        start_serial_dual_io_shared_port(
            COM="COM5",
            baud=115200,
            obj_hz=1.0,
            comp_hz=1.0,
            chaos=False,
            parent_shutdown=parent_shutdown,
        )
    except Exception as e:
        print(f"[Serial Process] Error: {e}")
        sys.exit(1)


def start_zombi(resume_uuid=None, optimizing_dims=None, checkpoint_dir=None,
                hparams_path=None, new_run_uuid=None):
    try:
        time.sleep(2)
        if resume_uuid:
            print(f"[ZoMBI Process] Resuming ZoMBI-Hop v2 with UUID: {resume_uuid}...")
        else:
            print("[ZoMBI Process] Starting ZoMBI-Hop v2 (DB-driven)...")
        run_zombi_main(resume_uuid=resume_uuid, optimizing_dims=optimizing_dims,
                       checkpoint_dir=checkpoint_dir, hparams_path=hparams_path,
                       new_run_uuid=new_run_uuid)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ZoMBI Process] Error: {e}")
        # Emit the full stack so the failing file:line reaches the GUI log panel
        # (Popen pipes stdout->the app). Without this only str(e) was visible.
        print(tb)
        # ALSO persist the traceback to the run's run.log directly. start_zombi runs
        # as a spawned multiprocessing child (main.py) whose stderr is not reliably
        # captured by the GUI's stdout pipe, so the tee could miss the crash. We own
        # the run dir path here, so write it where it will always survive.
        try:
            _uuid = resume_uuid or new_run_uuid
            if checkpoint_dir and _uuid:
                _rp = Path(checkpoint_dir) / f"run_{_uuid}" / "run.log"
                _rp.parent.mkdir(parents=True, exist_ok=True)
                with open(_rp, "a", encoding="utf-8") as _f:
                    _f.write(f"\n[ZoMBI Process] Error: {e}\n{tb}\n")
        except Exception:
            pass
        sys.exit(1)


def main():
    import argparse
    Path('actual_runs').mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="ZoMBI-Hop Hardware Launcher")
    parser.add_argument("resume_uuid", nargs="?", default=None,
                        help="Resume UUID (omit for new run, 'list' to show runs)")
    parser.add_argument("--dims", default=None,
                        help="Comma-separated dimension indices to optimise, e.g. 0,8,9")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory to save run checkpoints")
    parser.add_argument("--hparams", default=None,
                        help="Path to a trial.json-style JSON file whose 'hparams' "
                             "override the built-in ZoMBI-Hop defaults")
    parser.add_argument("--run-uuid", default=None, dest="run_uuid",
                        help="Caller-provided UUID for a NEW run (lets the GUI "
                             "pre-create/display the run dir immediately). Ignored "
                             "when a resume UUID is given.")
    args = parser.parse_args()

    resume_uuid = args.resume_uuid
    optimizing_dims = None
    if args.dims:
        try:
            optimizing_dims = [int(x.strip()) for x in args.dims.split(",")]
        except ValueError:
            print(f"[Main] Invalid --dims value: {args.dims!r}. Expected comma-separated ints.")
            sys.exit(1)
    checkpoint_dir = args.checkpoint_dir
    hparams_path = args.hparams
    new_run_uuid = args.run_uuid if resume_uuid is None else None

    if resume_uuid is not None and resume_uuid.lower() == 'list':
        list_runs_and_exit()
        sys.exit(0)

    if resume_uuid:
        print(f"[Main2] Resume UUID provided: {resume_uuid}")
    if optimizing_dims:
        print(f"[Main] Optimizing dims: {optimizing_dims}")

    # Hard reset all DBs and communication state ONLY if starting new trial
    if resume_uuid is None:
        initialize_db()
        communication.reset_objective()
        communication.reset_compositions()
        print("[Main] Database reset complete (new trial)")
    else:
        print("[Main] Skipping database reset (resuming trial)")

    shutdown = multiprocessing.Event()
    p_serial: multiprocessing.Process | None = None
    p_zombi: multiprocessing.Process | None = None

    _interrupt_lock = threading.Lock()
    _interrupt_seen = False

    def _on_interrupt(signum, frame):
        nonlocal _interrupt_seen
        with _interrupt_lock:
            if not _interrupt_seen:
                _interrupt_seen = True
                print(
                    f"\n[Main] Signal {signum} — stopping ZoMBI, then releasing COM5…"
                )
            shutdown.set()

    signal.signal(signal.SIGINT, _on_interrupt)
    if hasattr(signal, "SIGBREAK"):
        # Windows: Ctrl+Break
        try:
            signal.signal(signal.SIGBREAK, _on_interrupt)
        except (AttributeError, OSError):
            pass
    signal.signal(signal.SIGTERM, _on_interrupt)

    try:
        initialize_db()
        print("[Main] Databases initialized successfully")
    except Exception as e:
        print(f"[Main] Error initializing databases: {e}")
        sys.exit(1)

    multiprocessing.set_start_method("spawn", force=True)

    p_serial = multiprocessing.Process(
        target=start_serial,
        args=(shutdown,),
        name="SerialIO",
    )
    p_zombi = multiprocessing.Process(target=start_zombi,
                                      args=(resume_uuid, optimizing_dims, checkpoint_dir,
                                            hparams_path, new_run_uuid),
                                      name="ZoMBI")

    zombi_finished_normally = False
    serial_died_unexpectedly = False
    try:
        print("[Main] Starting serial communication process...")
        p_serial.start()
        time.sleep(3)
        if not p_serial.is_alive():
            print("[Main] Serial process failed to start or died immediately")
            sys.exit(1)

        print("[Main] Starting ZoMBI-Hop optimization process...")
        p_zombi.start()

        while not shutdown.is_set():
            if not p_serial.is_alive():
                print("[Main] Serial process died unexpectedly")
                serial_died_unexpectedly = True
                if p_zombi.is_alive():
                    print("[Main] Terminating ZoMBI process...")
                    p_zombi.terminate()
                    p_zombi.join(timeout=5)
                break

            if not p_zombi.is_alive():
                ex = p_zombi.exitcode
                if ex == 0:
                    zombi_finished_normally = True
                print(
                    f"[Main] ZoMBI process completed or died (exitcode={ex})"
                )
                if p_serial.is_alive():
                    print("[Main] Stopping serial process (release COM)...")
                    shutdown.set()
                    p_serial.join(timeout=12)
                break

            shutdown.wait(timeout=0.5)

    except KeyboardInterrupt:
        with _interrupt_lock:
            if not _interrupt_seen:
                _interrupt_seen = True
                print("\n[Main] KeyboardInterrupt — stopping ZoMBI, then releasing COM5…")
        shutdown.set()
    except Exception as e:
        print(f"[Main] Unexpected error: {e}")
    finally:
        print("[Main] Cleaning up processes...")

        if not shutdown.is_set():
            shutdown.set()

        # 1) Stop ZoMBI first: it can block in CUDA / LineBO / sqlite while the serial
        #    process is light. Waiting on serial before killing ZoMBI caused hangs.
        if p_zombi is not None and p_zombi.is_alive():
            print("[Main] Terminating ZoMBI process…")
            p_zombi.terminate()
            p_zombi.join(timeout=5)
        if p_zombi is not None and p_zombi.is_alive():
            print("[Main] Force killing ZoMBI process…")
            p_zombi.kill()
            p_zombi.join(timeout=3)

        # 2) Serial child releases COM (parent must not open the port)
        if p_serial is not None and p_serial.is_alive():
            print("[Main] Waiting for serial process to release COM5…")
            p_serial.join(timeout=15)
        if p_serial is not None and p_serial.is_alive():
            print("[Main] Serial did not exit in time — terminating…")
            p_serial.terminate()
            p_serial.join(timeout=4)
        if p_serial is not None and p_serial.is_alive():
            p_serial.kill()
            p_serial.join(timeout=2)

        # 2) If we did not finish a full ZoMBI run, clear handshake + stale objective rows so
        #    the next `python -m scripts.main` can send a new proposed line and not block on
        #    a half-finished get_y wait (skip this after a normal zombi completion).
        if not zombi_finished_normally:
            try:
                communication.clear_in_flight_objective_state()
                print("[Main] Objective DB + handshake reset for a clean next run (new line proposal).")
            except Exception as e:
                print(f"[Main] Note: could not clear objective handshake: {e}")

        print("[Main] Cleanup complete")
        if serial_died_unexpectedly:
            sys.exit(1)


if __name__ == "__main__":
    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        print("\nCurrent configuration:")
        print(f"  Serial port: COM5")
        print(f"  Checkpoint directory: actual_runs/checkpoints/")
        print(f"  Device: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
        sys.exit(0)

    main()
