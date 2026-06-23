#!/usr/bin/env python3
"""
build_datasets.py — pull selected HTEM-DB combinatorial libraries and write one
CSV per library in a "campaign"-style layout: one row per sample (a position on
the combinatorial wafer), with the mixed-compound composition fractions as the
leading columns (the optimization landscape), followed by position and the real
measured scalar properties.

No data is synthesized or interpolated. Every column is a real per-sample
measurement returned by the public HTEM-DB API. Spectra (XRD / optical curves)
are intentionally NOT flattened into columns: each sample uses a different
wavelength / angle grid, so aligning them would require interpolation, which we
avoid. Only genuinely measured scalar fields are kept.

Composition note: HTEM reports `xrf_concentration` per *compound* (the sputter
targets), summing to ~100% -- directly analogous to a campaign that mixes a few
compounds in fractions. Those compound fractions become the landscape columns.

Usage:
  python htem/build_datasets.py
  python htem/build_datasets.py --libs 6658 7707 --outdir htem/datasets
"""

import argparse
import csv
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("Needs `requests`:  pip install requests")

BASE = "https://htem-api.nlr.gov"
TIMEOUT = 60

# (library_id, short human label) — chemically diverse, well-measured systems.
DEFAULT_LIBS = [
    (6658,  "ZnNiCo-oxide"),
    (7707,  "SnZnTi-oxide"),
    (6750,  "CuZnSe-chalcogenide"),
    (7079,  "SnInZn-oxide"),          # transparent conductor (ITO-like)
    (7030,  "MnCrZn-oxide"),
    (6988,  "CuZn-oxide-sulfide"),
    (10661, "ZnMnTeSe-chalcogenide"), # true 4-component mixture, with bandgaps
    (7837,  "SnTeMnSe-chalcogenide"), # 4-component; note: no optical bandgap
]

# Real per-sample scalar measurements -> output column name.
SCALAR_FIELDS = [
    ("opt_direct_bandgap",        "Bandgap_eV"),
    ("opt_average_vis_trans",     "AvgVisTransmittance"),
    ("fpm_conductivity",          "Conductivity_S_per_cm"),
    ("fpm_resistivity",           "Resistivity_ohm_cm"),
    ("fpm_sheet_resistance",      "SheetResistance_ohm_sq"),
    ("fpm_standard_deviation",    "Conductivity_std"),
    ("thickness",                 "Thickness_um"),
    ("absolute_temp_c",           "Temperature_C"),
]

_session = requests.Session()
_session.headers.update({"User-Agent": "htem-build-datasets/1.0 (+public-api)"})


def get(path):
    r = _session.get(f"{BASE}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def build_library_csv(lib_id, label, outdir):
    lib = get(f"/api/sample_library/{lib_id}")
    if not lib:
        print(f"  [skip] library {lib_id} not found")
        return
    sids = lib.get("sample_ids") or []
    elements = "".join(lib.get("elements") or [])
    print(f"Library {lib_id} ({label}, {elements}): {len(sids)} samples")

    samples = []
    compound_set = []  # preserve first-seen order
    for sid in sids:
        s = get(f"/api/sample/{sid}")
        if not s:
            continue
        compounds = s.get("xrf_compounds") or []
        for c in compounds:
            if c not in compound_set:
                compound_set.append(c)
        samples.append(s)

    if not samples:
        print(f"  [skip] no samples returned for {lib_id}")
        return

    # Header: composition (one col per mixed compound) + position + scalars + ids
    header = (
        ["Iteration"]
        + [f"{c}_frac_pct" for c in compound_set]
        + ["X_mm", "Y_mm", "Z_mm"]
        + [name for _, name in SCALAR_FIELDS]
        + ["sample_id", "position"]
    )

    rows = []
    for i, s in enumerate(samples):
        # Accumulate by compound name: some libraries report a compound more
        # than once (e.g. two Te targets), so summing recovers the true total
        # instead of letting a later 0 overwrite the real value.
        comp = {}
        for c, v in zip(s.get("xrf_compounds") or [], s.get("xrf_concentration") or []):
            comp[c] = comp.get(c, 0.0) + (v or 0.0)
        xyz = s.get("xyz_mm") or [None, None, None]
        row = [i]
        row += [comp.get(c, "") for c in compound_set]
        row += [xyz[0] if len(xyz) > 0 else "",
                xyz[1] if len(xyz) > 1 else "",
                xyz[2] if len(xyz) > 2 else ""]
        row += [s.get(field, "") for field, _ in SCALAR_FIELDS]
        row += [s.get("id", ""), s.get("position", "")]
        rows.append(row)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"htem_{lib_id}_{label}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  -> wrote {len(rows)} rows x {len(header)} cols  ({', '.join(compound_set)})  {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libs", nargs="*", type=int, help="library ids (default: curated set)")
    p.add_argument("--outdir", default=os.path.join("htem", "datasets"))
    args = p.parse_args(argv)

    libs = ([(lid, str(lid)) for lid in args.libs] if args.libs else DEFAULT_LIBS)
    for lib_id, label in libs:
        build_library_csv(lib_id, label, args.outdir)


if __name__ == "__main__":
    main()
