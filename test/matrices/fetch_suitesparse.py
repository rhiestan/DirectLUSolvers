#!/usr/bin/env python3
"""Fetch the curated SuiteSparse corpus for the DirectLUSolvers test suites.

The corpus is defined by `suitesparse.manifest` (checked in, human-curated) and
downloaded into `cache/` (git-ignored). Nothing here is automatic: adding a
matrix to the suite means adding a line to the manifest, deliberately.

    python fetch_suitesparse.py                # download everything missing
    python fetch_suitesparse.py --tier quick   # only the small/fast tier
    python fetch_suitesparse.py --list         # show the manifest, mark cached
    python fetch_suitesparse.py --verify       # check manifest vs the live index
    python fetch_suitesparse.py --propose 20   # suggest new candidates to add

DEPENDENCIES: none. Downloading uses only the standard library, because the
SuiteSparse URL scheme is stable and a test corpus should not need a pip install
to reproduce. `ssgetpy` is used only by --propose, and only if it is installed;
without it --propose falls back to the collection's own ssstats.csv index, which
is what the curation was actually done from.
"""

import argparse
import csv
import io
import os
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "suitesparse.manifest")
CACHE = os.path.join(HERE, "cache")

INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"
MM_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"

TIERS = ("quick", "standard", "large")


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

class Entry:
    __slots__ = ("group", "name", "tier", "n", "nnz", "psym", "spd", "kind")

    def __init__(self, group, name, tier, n, nnz, psym, spd, kind):
        self.group, self.name, self.tier = group, name, tier
        self.n, self.nnz, self.psym, self.spd, self.kind = n, nnz, psym, spd, kind

    @property
    def label(self):
        return f"{self.group}/{self.name}"

    @property
    def mtx_path(self):
        # The tarball expands to <Name>/<Name>.mtx; we extract under cache/<Group>/.
        return os.path.join(CACHE, self.group, self.name, self.name + ".mtx")

    @property
    def cached(self):
        return os.path.isfile(self.mtx_path)


def read_manifest(path=MANIFEST):
    entries = []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 7)
            if len(parts) < 8:
                sys.stderr.write(f"skipping malformed manifest line: {line}\n")
                continue
            group, name, tier, n, nnz, psym, spd, kind = parts
            entries.append(Entry(group, name, tier, int(n), int(nnz), float(psym),
                                 spd == "y", kind))
    return entries


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download(entry, force=False):
    if entry.cached and not force:
        return "cached"
    url = MM_URL.format(group=entry.group, name=entry.name)
    dest_dir = os.path.join(CACHE, entry.group)
    os.makedirs(dest_dir, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as resp:
        blob = resp.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".mtx")]
        if not members:
            raise RuntimeError(f"no .mtx inside {url}")
        for m in members:
            # Refuse absolute paths and parent traversal before extracting.
            norm = os.path.normpath(m.name)
            if norm.startswith("..") or os.path.isabs(norm):
                raise RuntimeError(f"unsafe path in archive: {m.name}")
        tf.extractall(dest_dir, members=members)
    if not entry.cached:
        raise RuntimeError(f"expected {entry.mtx_path} after extracting {url}")
    return f"{len(blob) / 1e6:.1f} MB"


# --------------------------------------------------------------------------
# Index (for --verify and --propose)
# --------------------------------------------------------------------------

def load_index():
    with urllib.request.urlopen(INDEX_URL, timeout=120) as r:
        raw = r.read().decode("utf8", "replace")
    out = {}
    for f in csv.reader(raw.splitlines()[2:]):
        if len(f) < 12:
            continue
        try:
            rows, cols, nnz = int(f[2]), int(f[3]), int(f[4])
            psym = float(f[9])
        except ValueError:
            continue
        out[(f[0], f[1])] = dict(n=rows, cols=cols, nnz=nnz, psym=psym,
                                 real=f[5] == "1", binary=f[6] == "1",
                                 spd=f[8] == "1", kind=f[11])
    return out


def cmd_verify(entries):
    index = load_index()
    bad = 0
    for e in entries:
        info = index.get((e.group, e.name))
        if info is None:
            print(f"  MISSING FROM INDEX  {e.label}")
            bad += 1
            continue
        drift = []
        if info["n"] != e.n:
            drift.append(f"n {e.n} -> {info['n']}")
        if info["nnz"] != e.nnz:
            drift.append(f"nnz {e.nnz} -> {info['nnz']}")
        if abs(info["psym"] - e.psym) > 0.01:
            drift.append(f"psym {e.psym} -> {info['psym']:.2f}")
        if drift:
            print(f"  DRIFT  {e.label}: {', '.join(drift)}")
            bad += 1
    print(f"\n{len(entries)} entries checked, {bad} problem(s).")
    return 1 if bad else 0


def cmd_propose(entries, count):
    """Suggest matrices not yet in the manifest, keeping the symmetry spread."""
    try:
        import ssgetpy  # noqa: F401
        print("(ssgetpy is installed but the collection's own ssstats.csv index is\n"
              " richer for this purpose, so it is used instead.)\n")
    except ImportError:
        pass
    index = load_index()
    have = {(e.group, e.name) for e in entries}

    def band(psym):
        return "sym" if psym >= 0.999 else ("partial" if psym >= 0.5 else "unsym")

    have_bands = {}
    for e in entries:
        have_bands[band(e.psym)] = have_bands.get(band(e.psym), 0) + 1

    cands = [dict(group=g, name=nm, **info) for (g, nm), info in index.items()
             if info["real"] and not info["binary"] and info["n"] == info["cols"]
             and 1000 <= info["n"] <= 60000 and info["nnz"] >= 3 * info["n"]
             and (g, nm) not in have]
    cands.sort(key=lambda c: c["nnz"])
    print(f"current spread: {have_bands}")
    print(f"{'group/name':<34} {'n':>8} {'nnz':>10} {'psym':>6}  kind")
    seen_kind = set()
    shown = 0
    for c in cands:
        key = (band(c["psym"]), c["kind"])
        if key in seen_kind:
            continue
        seen_kind.add(key)
        print(f"{c['group'] + '/' + c['name']:<34} {c['n']:>8} {c['nnz']:>10} "
              f"{c['psym']:>6.2f}  {c['kind']}")
        shown += 1
        if shown >= count:
            break
    print("\nAdd a line to suitesparse.manifest to adopt one; then re-run --verify.")
    return 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=TIERS, help="only this tier and smaller")
    ap.add_argument("--list", action="store_true", help="show the manifest and exit")
    ap.add_argument("--verify", action="store_true", help="check manifest against the live index")
    ap.add_argument("--propose", type=int, metavar="N", help="suggest N candidates to add")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    entries = read_manifest()

    if args.tier:
        limit = TIERS.index(args.tier)
        entries = [e for e in entries if TIERS.index(e.tier) <= limit]

    if args.list:
        print(f"{'group/name':<34} {'tier':<9} {'n':>8} {'nnz':>10} {'psym':>6} {'cached':>7}")
        for e in entries:
            print(f"{e.label:<34} {e.tier:<9} {e.n:>8} {e.nnz:>10} {e.psym:>6.2f} "
                  f"{'yes' if e.cached else 'no':>7}")
        return 0
    if args.verify:
        return cmd_verify(entries)
    if args.propose:
        return cmd_propose(entries, args.propose)

    missing = [e for e in entries if args.force or not e.cached]
    if not missing:
        print(f"All {len(entries)} matrices already cached in {CACHE}")
        return 0

    print(f"Downloading {len(missing)} of {len(entries)} matrices into {CACHE}")
    failed = []
    for i, e in enumerate(missing, 1):
        sys.stdout.write(f"  [{i}/{len(missing)}] {e.label} ... ")
        sys.stdout.flush()
        try:
            print(download(e, force=args.force))
        except Exception as exc:  # network, archive layout, disk
            print(f"FAILED: {type(exc).__name__}: {exc}")
            failed.append(e.label)
    if failed:
        print(f"\n{len(failed)} download(s) failed: {', '.join(failed)}")
        return 1
    print("\nDone. The C++ suites pick these up automatically; run:")
    print("  ctest --test-dir build -R test_suitesparse --output-on-failure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
