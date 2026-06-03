"""
serial_only.py
==============
Standalone serial IO process — starts communication with the hardware device
without running any optimization.  Launched by the GUI's Manual Control tab.

Usage (normally invoked by the GUI, not directly):
    python scripts/serial_only.py --com COM5 --baud 9600 \\
        --comp-db ./sql/compositions.db \\
        --obj-db  ./sql/objective.db \\
        --mem-db  ./sql/objective_memory.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.communication import start_serial_dual_io_shared_port


def main():
    p = argparse.ArgumentParser(description="Serial-only IO process")
    p.add_argument("--com",      default="COM5",
                   help="Serial port name (e.g. COM5 or /dev/ttyUSB0)")
    p.add_argument("--baud",     type=int, default=9600, help="Baud rate")
    p.add_argument("--comp-db",  default="./sql/compositions.db",
                   help="Path to compositions.db")
    p.add_argument("--obj-db",   default="./sql/objective.db",
                   help="Path to objective.db")
    p.add_argument("--mem-db",   default="./sql/objective_memory.db",
                   help="Path to objective_memory.db")
    p.add_argument("--obj-hz",   type=float, default=1.0,
                   help="Objective-receiver polling rate (Hz)")
    p.add_argument("--comp-hz",  type=float, default=1.0,
                   help="Composition-sender rate (Hz)")
    args = p.parse_args()

    print(f"[serial_only] Starting serial IO: {args.com} @ {args.baud} baud")
    print(f"[serial_only]   comp-db : {args.comp_db}")
    print(f"[serial_only]   obj-db  : {args.obj_db}")
    print(f"[serial_only]   mem-db  : {args.mem_db}")
    print(f"[serial_only] Ctrl-C to stop.", flush=True)

    start_serial_dual_io_shared_port(
        COM=args.com,
        baud=args.baud,
        obj_hz=args.obj_hz,
        comp_hz=args.comp_hz,
        obj_db=args.obj_db,
        mem_db=args.mem_db,
        comp_db=args.comp_db,
        verbose=True,
    )


if __name__ == "__main__":
    main()
