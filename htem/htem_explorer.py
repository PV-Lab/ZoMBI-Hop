#!/usr/bin/env python3
"""
htem_explorer.py — a terminal client for the HTEM-DB (High-Throughput
Experimental Materials Database), built to replace the web UI when it's
unavailable.

Background: NREL became the "National Laboratory of the Rockies" (NLR) and
retired the nrel.gov domain on 2026-05-29. The database moved from
htem-api.nrel.gov  ->  htem-api.nlr.gov . This tool defaults to the new
domain and auto-detects which one is actually live.

The HTEM-DB read API is PUBLIC and requires NO API key.

Endpoints used:
  GET /api/sample_library?element=Zn,Sn   -> list of libraries (id, elements, ...)
  GET /api/sample_library/<id>            -> one library; includes sample_ids[]
  GET /api/sample/<id>                     -> one sample/position; props + spectra

Requires: requests   (pip install requests)
Optional: pandas, matplotlib  (nicer tables / plotting; not required)

Usage examples:
  python htem_explorer.py doctor
  python htem_explorer.py search Zn Sn
  python htem_explorer.py search Zn Sn --exclude O
  python htem_explorer.py library 1234
  python htem_explorer.py sample 56789
  python htem_explorer.py sample 56789 --spectrum xrd --save xrd.csv
"""

import argparse
import json
import sys
import time
from urllib.parse import quote  # noqa: F401  (kept for ad-hoc use)

try:
    import requests
except ImportError:
    sys.exit("This tool needs `requests`. Install with:  pip install requests")

# New domain first, old domain as a fallback probe.
CANDIDATE_HOSTS = [
    "https://htem-api.nlr.gov",
    "https://htem-api.nrel.gov",
]
TIMEOUT = 30
_session = requests.Session()
_session.headers.update({"User-Agent": "htem-explorer/1.0 (+public-api)"})
_BASE = None  # resolved lazily


def resolve_base(verbose=False):
    """Find the first reachable API host and cache it."""
    global _BASE
    if _BASE:
        return _BASE
    errors = []
    for host in CANDIDATE_HOSTS:
        url = f"{host}/api/sample_library?element=Zn"
        try:
            r = _session.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                if verbose:
                    print(f"[ok] reachable: {host}")
                _BASE = host
                return host
            errors.append(f"{host} -> HTTP {r.status_code}")
        except requests.RequestException as e:
            errors.append(f"{host} -> {type(e).__name__}: {e}")
    raise SystemExit(
        "Could not reach any HTEM-DB API host.\n  "
        + "\n  ".join(errors)
        + "\n\nIf this persists, the new web UI is https://htem.nlr.gov/ — "
          "check it in a browser. A stale DNS cache for the old nrel.gov "
          "domain is a common cause; try flushing DNS."
    )


def _get(path, params=None, retries=3):
    base = resolve_base()
    url = f"{base}{path}"
    last = None
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            last = f"HTTP {r.status_code} for {r.url}"
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (attempt + 1))  # simple backoff; be polite
    raise RuntimeError(f"Request failed after {retries} tries: {last}")


# ----------------------------- API wrappers -----------------------------

def search_libraries(only=None, exclude=None):
    """Libraries containing ALL of `only` and NONE of `exclude`.

    The API's element filter is permissive, so we refine client-side using
    each library's `elements` field (matching the official helper's logic).
    """
    only = [e.strip().capitalize() for e in (only or []) if e.strip()]
    exclude = [e.strip().capitalize() for e in (exclude or []) if e.strip()]
    # The API expects a literal comma-separated list (no percent-encoding of
    # the comma), matching the official helper.
    query = ",".join(only) if only else ""
    data = _get("/api/sample_library", params={"element": query}) or []

    def elements_of(lib):
        raw = lib.get("elements", [])
        if isinstance(raw, str):
            return [x.strip().capitalize() for x in raw.replace(";", ",").split(",") if x.strip()]
        return [str(x).capitalize() for x in raw]

    out = []
    for lib in data:
        els = elements_of(lib)
        if all(e in els for e in only) and not any(e in els for e in exclude):
            out.append(lib)
    return out


def get_library(library_id):
    return _get(f"/api/sample_library/{library_id}")


def get_sample(sample_id):
    return _get(f"/api/sample/{sample_id}")


def extract_spectrum(sample, which):
    """Return list-of-rows for 'xrd' or 'optical' from a sample dict."""
    if which == "xrd":
        ang = sample.get("xrd_angle") or []
        bg = sample.get("xrd_background") or []
        inten = sample.get("xrd_intensity") or []
        rows = [["angle", "background", "intensity"]]
        for i in range(len(ang)):
            rows.append([
                ang[i],
                bg[i] if i < len(bg) else "",
                inten[i] if i < len(inten) else "",
            ])
        return rows
    if which == "optical":
        oo = sample.get("oo") or {}
        rows = [["channel", "wavelength", "response"]]
        for chan in ("uvit", "uvir", "nirt", "nirr"):
            block = oo.get(chan) or {}
            waves = block.get("wavelength") or []
            resp = block.get("response") or []
            for i in range(len(waves)):
                rows.append([chan, waves[i], resp[i] if i < len(resp) else ""])
        return rows
    raise ValueError("spectrum must be 'xrd' or 'optical'")


# ----------------------------- pretty printing -----------------------------

def _short(v, width=60):
    s = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
    return s if len(s) <= width else s[: width - 1] + "…"


def print_kv(d, skip_spectra=True):
    spectra_keys = {"xrd_angle", "xrd_background", "xrd_intensity", "oo"}
    for k in sorted(d.keys()):
        if skip_spectra and k in spectra_keys:
            v = d[k]
            n = len(v) if isinstance(v, (list, dict)) else "?"
            print(f"  {k:<26} <{type(v).__name__}, len={n}> (use --spectrum to view)")
        else:
            print(f"  {k:<26} {_short(d[k])}")


def write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  saved {len(rows)-1} rows -> {path}")


# ----------------------------- CLI commands -----------------------------

def cmd_doctor(args):
    print("Probing HTEM-DB API hosts...")
    base = resolve_base(verbose=True)
    print(f"Active base URL: {base}")
    sample = _get("/api/sample_library?element=Zn") or []
    print(f"Smoke test: 'element=Zn' returned {len(sample)} libraries.")
    if sample:
        ex = sample[0]
        print("Example library keys:", ", ".join(sorted(ex.keys())))
    print("API key required: NO (this is a public read API).")


def cmd_search(args):
    libs = search_libraries(only=args.elements, exclude=args.exclude)
    print(f"{len(libs)} libraries match "
          f"only={args.elements or '[]'} exclude={args.exclude or '[]'}\n")
    for lib in libs[: args.limit]:
        lid = lib.get("id", "?")
        els = lib.get("elements", "")
        n = len(lib.get("sample_ids", []) or [])
        print(f"  id={lid:<8} elements={_short(els, 40):<42} samples={n}")
    if len(libs) > args.limit:
        print(f"  ... +{len(libs) - args.limit} more (raise --limit to see all)")


def cmd_library(args):
    lib = get_library(args.id)
    if not lib:
        sys.exit(f"No library with id {args.id}")
    print(f"Library {args.id}\n" + "-" * 40)
    print_kv(lib)
    sids = lib.get("sample_ids") or []
    if sids:
        preview = ", ".join(str(s) for s in sids[:12])
        print(f"\n  {len(sids)} samples. First few ids: {preview}"
              + (" ..." if len(sids) > 12 else ""))
        print(f"  Inspect one with:  python htem_explorer.py sample {sids[0]}")


def cmd_sample(args):
    s = get_sample(args.id)
    if not s:
        sys.exit(f"No sample with id {args.id}")
    if args.spectrum:
        rows = extract_spectrum(s, args.spectrum)
        if len(rows) <= 1:
            print(f"No '{args.spectrum}' spectrum on sample {args.id}.")
            return
        if args.save:
            write_csv(rows, args.save)
        else:
            for row in rows[: args.limit + 1]:
                print("  " + "\t".join(str(c) for c in row))
            if len(rows) - 1 > args.limit:
                print(f"  ... {len(rows)-1-args.limit} more points "
                      f"(use --save FILE.csv for all, or raise --limit)")
        return
    print(f"Sample {args.id}\n" + "-" * 40)
    print_kv(s)


def build_parser():
    p = argparse.ArgumentParser(
        description="Terminal explorer for the HTEM-DB (public, no API key).")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="check connectivity / which host is live")
    d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("search", help="find libraries by element composition")
    s.add_argument("elements", nargs="*", help="elements that MUST be present, e.g. Zn Sn")
    s.add_argument("--exclude", nargs="*", default=[], help="elements that must be ABSENT")
    s.add_argument("--limit", type=int, default=25)
    s.set_defaults(func=cmd_search)

    li = sub.add_parser("library", help="show one library and its sample ids")
    li.add_argument("id")
    li.set_defaults(func=cmd_library)

    sa = sub.add_parser("sample", help="show one sample, or dump a spectrum")
    sa.add_argument("id")
    sa.add_argument("--spectrum", choices=["xrd", "optical"])
    sa.add_argument("--save", help="write spectrum to this CSV path")
    sa.add_argument("--limit", type=int, default=20, help="rows to print inline")
    sa.set_defaults(func=cmd_sample)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
    