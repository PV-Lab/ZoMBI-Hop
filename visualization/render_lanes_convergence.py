"""Render the lanes convergence plot of a run folder to a PNG: the figure the GUI's Convergence tab draws
for a lanes run, or for a run still in its axial initialization (``visualization/lanes_timeline.py``).

    CUDA_VISIBLE_DEVICES=-1 python visualization/render_lanes_convergence.py --run runs/run_1d1f --out <png> [--dpi 150]

Draws in every phase of a run: the init view while it initialises (init_state.json, no lanes/ data yet),
the dive plot once lanes/ has data. Agg backend, no window, CPU only (torch files load with
map_location='cpu'; run as a script it sets CUDA_VISIBLE_DEVICES=-1 for its own process, so an empty value
in the caller's environment is fine too, and drawing skips src's import-time CUDA setup). Reads the run
folder only and never writes into it (an --out inside the run folder is refused). The PNG is rendered to
memory and then written in place, retried briefly while the file is locked: the repo lives in Dropbox,
where a rename over an existing file fails. Prints one line (its warnings, e.g. a torn file, included).

Exit codes:
    0  PNG written
    3  PNG written, but it is only the INIT VIEW of a snapshot-optimizer run past its init (init_state.json
       + snapshots; its later progress is in its snapshots: use visualization/plot_convergence.py)
    2  nothing written: nothing to draw (neither lanes/ data nor init_state.json) or a bad --out (inside the
       run folder, or a directory)
    1  nothing written: any other error (a bug or unexpected data while loading / drawing / writing)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
if __name__ == "__main__":
    # This tool never needs a GPU: hide every CUDA device from its own process before anything can touch one.
    # An EMPTY CUDA_VISIBLE_DEVICES does not hide the GPU on this Windows driver; '-1' (no such device) does.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIGSIZE = (11.0, 6.2)            # in; about the GUI tab's proportions, wider for the saved copy
EXIT_OK, EXIT_ERROR, EXIT_NOTHING, EXIT_INIT_ONLY = 0, 1, 2, 3


class NothingToDraw(Exception):
    """The two expected refusals (exit 2): no drawable data in the run folder, or a bad --out."""


def _write_in_place(out: Path, data: bytes, tries: int = 10) -> None:
    """Write ``data`` to ``out`` in place; a PermissionError (Dropbox or a viewer holding the file) is retried
    briefly, any other error is raised at once."""
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            with open(out, "wb") as fh:
                fh.write(data)
            return
        except PermissionError:
            if attempt == tries - 1 or out.is_dir():
                raise
            time.sleep(0.2 * (attempt + 1))


def render(run_dir, out, dpi: int = 150, figsize=FIGSIZE) -> dict:
    """Render ``run_dir``'s lanes convergence plot to ``out``; returns the plot's checks dict. Raises
    ``NothingToDraw`` for a run folder with nothing to draw or a bad ``out``."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from visualization.lanes_timeline import draw_lanes_convergence, load_lanes_run, run_kind

    run_dir = Path(run_dir).resolve()
    out = Path(out).resolve()
    if out == run_dir or run_dir in out.parents:
        raise NothingToDraw(f"--out {out} is inside the run folder {run_dir}; this tool never writes there")
    if out.is_dir():
        raise NothingToDraw(f"--out {out} is a directory; give a .png file path")
    if run_kind(run_dir) is None:
        raise NothingToDraw(f"{run_dir}: nothing to draw (no lanes/rows.jsonl, lanes/init.pt or "
                            f"init_state.json); use visualization/plot_convergence.py for a snapshot run")
    data = load_lanes_run(run_dir)
    fig = Figure(figsize=figsize, dpi=96)
    FigureCanvasAgg(fig)
    chk = draw_lanes_convergence(fig, data)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi), facecolor="white")
    _write_in_place(out, buf.getvalue())
    chk["out"] = str(out)
    return chk


def summary_line(chk: dict) -> str:
    w = chk.get("warnings") or []
    warn = f", {len(w)} warning(s): {w[0]}" if w else ""
    if chk.get("kind") in ("init", "legacy_init"):
        pend = chk.get("pending") or []
        n_empty = chk.get("n_empty") or 0
        n_req = chk.get("n_requeued") or 0
        if chk.get("kind") == "legacy_init":
            tail = "; snapshot run past init (init view only)"
        elif chk.get("optimizer") not in (None, "lanes"):
            tail = f"; optimizer {chk.get('optimizer')}"
        else:
            tail = "; lanes not started"
        return (f"wrote {chk.get('out')}: {chk.get('run')} initialization, {chk.get('n_done')} of "
                f"{chk.get('n_traces')} traces " + (f"reported ({n_empty} empty, " if n_empty else "measured (")
                + f"{chk.get('n_init')} points)"
                + (f", pending: {' '.join(pend)}" if pend else "")
                + (f" ({n_req} re-queued)" if n_req else "") + tail + warn)
    dives = [b for b in chk.get("blocks", []) if b.get("key") is not None]
    nf = sum(b["lane"] == "front" for b in dives)
    nb = sum(b["lane"] == "back" for b in dives)
    n_open = sum(bool(b.get("open")) for b in dives)
    late = chk.get("n_late_init") or 0
    n_rows = chk.get("n_rows") or 0                    # includes the late init rows: they are rows.jsonl rows
    return (f"wrote {chk.get('out')}: {chk.get('run')} {chk.get('n_init')} init"
            + (f" (+{late} late)" if late else "") + f" + {n_rows - late} rows, "
            f"{nf} front + {nb} back dives ({n_open} running), {len(chk.get('needles') or [])} needles, "
            f"clean {chk.get('clean')}/{chk.get('finished')} finished, {chk.get('red_bars')} red zoom bars" + warn)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder (runs/run_XXXX): a lanes run or one initialising")
    ap.add_argument("--out", required=True, help="PNG to write (outside the run folder)")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):        # a cp1252 console / log file must not turn a '…' into exit 1
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 - not a TextIOWrapper (pytest capture)
            pass
    try:
        chk = render(a.run, a.out, dpi=a.dpi)
    except NothingToDraw as e:
        print(f"render_lanes_convergence: {e}", file=sys.stderr)
        return EXIT_NOTHING
    except Exception as e:  # noqa: BLE001
        print(f"render_lanes_convergence: failed: {e!r}", file=sys.stderr)
        return EXIT_ERROR
    print(summary_line(chk))
    return EXIT_INIT_ONLY if chk.get("kind") == "legacy_init" else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
