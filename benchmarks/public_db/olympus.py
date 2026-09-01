"""
benchmarks/public_db/olympus.py
===============================
Pull and load public experimental datasets from **Olympus**
(https://github.com/the-matter-lab/olympus) without taking on Olympus itself as
a dependency.

Olympus ships each dataset as three plain files under
``src/olympus/datasets/dataset_<name>/``:

  * ``config.json``      -- parameter names/types/bounds, measurement names, the
                           default optimisation goal, and the *parameter
                           constraint* (``"simplex"`` or ``"none"``), which is
                           what decides whether the inputs live on a simplex.
  * ``data.csv``         -- headerless; ``n_params`` input columns followed by
                           ``n_measurements`` target columns.
  * ``description.txt``  -- the human-readable summary and source publication.

Those are fetched verbatim into ``benchmarks/public_db/data/<name>/`` and cached
there, so every load after the first is offline and byte-identical to upstream.

The four curated datasets
-------------------------
=============  ===  ==========  ========  ============================================
name            d   constraint  goal      what it measures
=============  ===  ==========  ========  ============================================
photo_pce10     4   simplex     minimize  photodegradation of a PCE10/P3HT/PCBM/oIDTBR
                                          organic-solar-cell blend
photo_wf3       4   simplex     minimize  the same, with WF3 in place of PCE10
hplc            6   none        maximize  HPLC peak area vs six process parameters
crossed_barrel  4   none        maximize  toughness of a 3D-printed crossed-barrel
                                          structure vs four geometry parameters
                                          (Gongora et al., Sci. Adv. 6:eaaz1708, 2020)
=============  ===  ==========  ========  ============================================

Properties of these four worth knowing before plotting them (see
``visualization/plot_run.py``, which dispatches on exactly these):

  * **Both ``photo_*`` sets are 4-component simplices, not 3.** Their rows sum to
    exactly 1.0 and the design matrix has rank 3, so they are *quaternary* blends
    and belong on a tetrahedron, never on a ternary triangle. (The source paper is
    "Beyond Ternary OPV" -- quaternary is the whole point of it.) They also share
    an identical 1040-point design grid; only the measured degradation differs,
    which makes them a clean paired comparison.
  * **``hplc`` is the only non-simplex set, and its columns span four orders of
    magnitude** (``sample_loop`` 0-0.08 ml vs ``push_speed`` 80-150 Hz). Any
    distance-based view of it -- UMAP, CoNet, a GP with one shared length scale --
    must use ``unit_X`` rather than the raw columns, or ``push_speed`` alone
    decides every distance.
  * **``crossed_barrel`` is the other non-simplex set, and the only curated one
    that lands on the scatter-plot matrix.** Four geometry parameters, so it has
    the same ``d=4`` as the ``photo_*`` pair but none of their sum-to-one
    structure -- which is precisely the pair (constraint, d) that separates the
    tetrahedron from the SPLOM. Its columns are declared ``continuous`` but only
    ever take a handful of levels each (``n``: 4, ``theta``: 9, ``r``: 11,
    ``t``: 3), and its 600 rows are a ~50% subset of that 1188-point lattice with
    no replicates, so a surrogate over it is interpolating a grid, not a
    continuum. ``theta`` (0-200) is 77% of every raw row, so it needs ``unit_X``
    for the same reason ``hplc`` does.
  * **The goals differ.** ``photo_*`` degradation is *minimised*; ``hplc`` peak
    area and ``crossed_barrel`` toughness are *maximised*. A best-so-far envelope
    hard-coded to ``max`` is wrong for two of the four, so ``goal`` is carried
    through to the plotting layer.

Usage
-----
    python -m benchmarks.public_db.olympus --list
    python -m benchmarks.public_db.olympus --fetch all
    python -m benchmarks.public_db.olympus --info all

    from benchmarks.public_db import load
    ds = load("hplc")
    ds.X, ds.Y, ds.param_names, ds.simplex, ds.goal
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# -- upstream layout ----------------------------------------------------------

RAW_BASE = ("https://raw.githubusercontent.com/the-matter-lab/olympus/main"
            "/src/olympus/datasets")
FILES = ("config.json", "data.csv", "description.txt")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

#: The datasets this module is curated for. Any other Olympus dataset name works
#: too -- nothing below is hard-coded to these four -- but these are the ones
#: whose shape is documented above and covered by the tests.
CURATED: tuple[str, ...] = ("photo_pce10", "photo_wf3", "hplc", "crossed_barrel")

#: Readable component names for datasets whose ``config.json`` uses placeholders.
#: Both ``photo_*`` sets name their parameters ``mat_1..mat_4``, but their
#: ``description.txt`` says which polymer each one is; a tetrahedron labelled
#: PCE10/P3HT/PCBM/oIDTBR is worth a great deal more than one labelled
#: mat_1..mat_4. Datasets absent from here (``hplc``, whose parameters are
#: already named) simply keep their config names.
#:
#: ``crossed_barrel`` is here for the same reason in a different form: its config
#: names the four geometry parameters ``n``/``theta``/``r``/``t``, which are the
#: paper's symbols and say nothing on an axis. They are the number of hollow
#: columns, the twist angle in degrees, the outer radius in mm and the column
#: thickness in mm. Kept ASCII (no theta glyph) because ``summary()`` prints
#: straight to a console that is still cp1252 on Windows.
PRETTY_LABELS: dict[str, tuple[str, ...]] = {
    "photo_pce10": ("PCE10", "P3HT", "PCBM", "oIDTBR"),
    "photo_wf3": ("WF3", "P3HT", "PCBM", "oIDTBR"),
    "crossed_barrel": ("columns n", "twist (deg)", "outer radius (mm)",
                       "thickness (mm)"),
}


# -- the loaded dataset -------------------------------------------------------

@dataclass(frozen=True)
class OlympusDataset:
    """One Olympus dataset, loaded from the local cache.

    ``X`` is (N, d) in the parameters' own physical units -- deliberately *not*
    rescaled -- and ``Y`` is (N,) the selected measurement column. ``bounds`` is
    (d, 2) of the declared ``[low, high]`` per parameter, which is what
    ``unit_X`` normalises against.

    ``simplex`` reports the upstream ``constraints.parameters == "simplex"``. It
    is a statement about the *design space*, and ``load`` verifies it against the
    data (rows summing to 1) rather than trusting the flag on its own.
    """

    name: str
    X: np.ndarray
    Y: np.ndarray
    param_names: tuple[str, ...]
    bounds: np.ndarray
    target_name: str
    goal: str
    simplex: bool
    description: str

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def labels(self) -> tuple[str, ...]:
        """Display names for the columns: ``PRETTY_LABELS`` if any, else config.

        Falls back to ``param_names`` whenever no override is registered *or* the
        registered one has the wrong width, so an upstream change to a dataset's
        dimensionality degrades to placeholder names rather than mislabelling
        axes with names from a different set of components.
        """
        pretty = PRETTY_LABELS.get(self.name)
        if pretty is not None and len(pretty) == self.d:
            return pretty
        return self.param_names

    @property
    def unit_X(self) -> np.ndarray:
        """``X`` min-max scaled into [0, 1] against the declared ``bounds``.

        Use this, not ``X``, for anything metric -- a UMAP/CoNet embedding, a GP
        with a shared length scale, a nearest-neighbour search. On ``hplc`` the
        raw columns differ by ~4 orders of magnitude, so on raw units Euclidean
        distance is very nearly ``push_speed`` alone.

        Degenerate bounds (high == low) map to 0. Values are clipped, so a row
        marginally outside its declared range cannot push the scale past [0, 1].
        """
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        span = np.where(hi > lo, hi - lo, 1.0)
        return np.clip((self.X - lo) / span, 0.0, 1.0)

    def summary(self) -> str:
        """Plain-ASCII shape report -- the text ``--info`` prints."""
        kind = "simplex" if self.simplex else "unconstrained box"
        out = [
            f"{self.name}: n={self.n}, d={self.d}  ({kind}, goal={self.goal})",
            f"  target: {self.target_name}   "
            f"range [{self.Y.min():.6g}, {self.Y.max():.6g}]",
            "  parameters:",
        ]
        for i, (nm, lab) in enumerate(zip(self.param_names, self.labels)):
            lo, hi = self.bounds[i]
            col = self.X[:, i]
            shown = nm if lab == nm else f"{nm} ({lab})"
            out.append(f"    {shown:<26s} declared [{lo:g}, {hi:g}]"
                       f"   observed [{col.min():.6g}, {col.max():.6g}]")
        return "\n".join(out)


# -- fetching -----------------------------------------------------------------

def _dataset_dir(name: str) -> Path:
    return DATA_DIR / name


def _download(url: str, dest: Path, timeout: float) -> None:
    """Fetch one upstream file to ``dest``, writing only on a complete read."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            payload = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url} -> {e.reason}") from e
    if not payload:
        raise RuntimeError(f"{url} -> empty response")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)


def fetch(name: str, *, force: bool = False, timeout: float = 60.0) -> Path:
    """Ensure ``name``'s three upstream files are cached locally; return its dir.

    A no-op when all three are already present, unless ``force`` re-downloads
    them. This is the only function here that touches the network.
    """
    d = _dataset_dir(name)
    missing = [f for f in FILES if not (d / f).is_file()]
    if not missing and not force:
        return d
    for f in (FILES if force else missing):
        _download(f"{RAW_BASE}/dataset_{name}/{f}", d / f, timeout)
    return d


def available() -> list[str]:
    """Dataset names already cached under ``benchmarks/public_db/data/``."""
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir()
                  if p.is_dir() and (p / "data.csv").is_file())


# -- loading ------------------------------------------------------------------

#: Row sums of a simplex dataset must match 1.0 to within this before the
#: ``constraints.parameters == "simplex"`` claim is accepted.
SIMPLEX_TOL = 1e-6


def load(name: str, *, download: bool = True,
         target: str | int = 0) -> OlympusDataset:
    """Load a cached Olympus dataset, fetching it first if needed.

    ``target`` selects the measurement column, by name or index; all four
    curated datasets have exactly one.

    The ``simplex`` flag from ``config.json`` is cross-checked against the data.
    If a dataset claims a simplex but its rows do not sum to 1, that is an
    upstream inconsistency and raises -- better than silently drawing a
    composition diagram of non-composition data.
    """
    d = fetch(name) if download else _dataset_dir(name)
    if not (d / "config.json").is_file():
        raise FileNotFoundError(
            f"{name} is not cached in {DATA_DIR} -- load with download=True, or "
            f"run `python -m benchmarks.public_db.olympus --fetch {name}`")

    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    params = cfg["parameters"]
    meas = cfg["measurements"]
    param_names = tuple(str(p["name"]) for p in params)
    bounds = np.array([[float(p["low"]), float(p["high"])] for p in params],
                      dtype=float)
    goal = str(cfg.get("default_goal", "maximize")).lower()
    simplex = str(cfg.get("constraints", {}).get("parameters", "none")) == "simplex"

    raw = np.loadtxt(d / "data.csv", delimiter=",", ndmin=2)
    want = len(params) + len(meas)
    if raw.shape[1] != want:
        raise ValueError(
            f"{name}/data.csv has {raw.shape[1]} columns, but config.json "
            f"declares {len(params)} parameters + {len(meas)} measurements")

    tgt = (int(target) if isinstance(target, int)
           else [str(m["name"]) for m in meas].index(target))
    X = raw[:, :len(params)]
    Y = raw[:, len(params) + tgt]

    if simplex:
        s = X.sum(axis=1)
        if not np.allclose(s, 1.0, atol=SIMPLEX_TOL):
            raise ValueError(
                f"{name} declares constraints.parameters='simplex' but its rows "
                f"sum to [{s.min():.6g}, {s.max():.6g}], not 1.0")

    return OlympusDataset(
        name=name, X=X, Y=Y, param_names=param_names, bounds=bounds,
        target_name=str(meas[tgt]["name"]), goal=goal, simplex=simplex,
        description=(d / "description.txt").read_text(
            encoding="utf-8", errors="replace"),
    )


def summary(name: str) -> str:
    """``load(name).summary()`` -- the shape report for one dataset."""
    return load(name).summary()


# -- CLI ----------------------------------------------------------------------

def _safe_print(text: str) -> None:
    """Print text that may hold non-ASCII (author names) on a cp1252 console.

    ``description.txt`` carries names like "Hase" with an umlaut. On a Windows
    console still defaulting to cp1252 a bare ``print`` of that raises
    ``UnicodeEncodeError``, so it is re-encoded through the real stdout encoding
    with replacement instead.
    """
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc))
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch and inspect public Olympus datasets.")
    ap.add_argument("--fetch", metavar="NAME",
                    help="dataset to download ('all' for every curated one)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even when already cached")
    ap.add_argument("--info", metavar="NAME",
                    help="print a shape report ('all' for every cached one)")
    ap.add_argument("--describe", metavar="NAME",
                    help="print the upstream description.txt")
    ap.add_argument("--list", action="store_true",
                    help="list curated and cached dataset names")
    args = ap.parse_args(argv)

    if not any((args.fetch, args.info, args.describe, args.list)):
        ap.print_help()
        return 0

    if args.list:
        have = set(available())
        print("curated:")
        for n in CURATED:
            print(f"  {n:<14s} {'cached' if n in have else 'not cached'}")
        extra = sorted(have - set(CURATED))
        if extra:
            print("also cached:")
            for n in extra:
                print(f"  {n}")

    if args.fetch:
        for n in (CURATED if args.fetch == "all" else (args.fetch,)):
            print(f"fetched {n} -> {fetch(n, force=args.force)}")

    if args.info:
        for n in (available() if args.info == "all" else (args.info,)):
            print(load(n).summary())
            print()

    if args.describe:
        _safe_print(load(args.describe).description)
    return 0


if __name__ == "__main__":
    sys.exit(main())
