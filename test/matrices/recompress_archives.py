#!/usr/bin/env python3
"""Recompress the checked-in SuiteSparse archives with AdvanceCOMP's advdef.

    python recompress_archives.py                      # every suitesparse/*/*.tar.gz
    python recompress_archives.py suitesparse/HB/nnc1374.tar.gz ...
    python recompress_archives.py --advdef /path/to/advdef --jobs 8

Runs `advdef -z -4` (zopfli, the strongest level) on each archive. That is still
plain gzip, so CMake and tarfile read the result as before; it is just smaller.
advdef rewrites the file in place and only when the result is smaller. Each
archive's decompressed bytes are compared before and after, so a recompression
that changed the content fails loudly instead of being committed.

Slow by design (zopfli): run it once after `fetch_suitesparse.py --pack` adds or
changes an archive, not as part of a build.
"""

import argparse
import concurrent.futures
import glob
import gzip
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADVDEF = r"C:\Tools\AdvanceCOMP\advdef.exe"


def content_digest(path):
    with gzip.open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def recompress(advdef, path):
    before_size = os.path.getsize(path)
    before = content_digest(path)
    proc = subprocess.run([advdef, "-z", "-4", path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"advdef failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    if content_digest(path) != before:
        raise RuntimeError("decompressed content changed")
    return before_size, os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archives", nargs="*", help="archives to recompress (default: all)")
    ap.add_argument("--advdef", default=DEFAULT_ADVDEF, help=f"advdef executable (default {DEFAULT_ADVDEF})")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel advdef processes")
    args = ap.parse_args()

    archives = args.archives or sorted(glob.glob(os.path.join(HERE, "suitesparse", "*", "*.tar.gz")))
    if not archives:
        print("no archives found")
        return 1
    if not os.path.isfile(args.advdef):
        print(f"advdef not found at {args.advdef} (use --advdef)")
        return 1

    total_before = total_after = 0
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(recompress, args.advdef, a): a for a in archives}
        for fut in concurrent.futures.as_completed(futures):
            path = os.path.abspath(futures[fut])
            name = os.path.relpath(path, HERE) if path.startswith(HERE + os.sep) else path
            try:
                before, after = fut.result()
            except Exception as exc:
                print(f"  FAILED  {name}: {exc}")
                failed.append(name)
                continue
            total_before += before
            total_after += after
            print(f"  {name:<46} {before / 1e6:7.2f} -> {after / 1e6:7.2f} MB ({100.0 * after / before:5.1f}%)")
            sys.stdout.flush()

    if total_before:
        print(f"\n{len(archives) - len(failed)} archives: {total_before / 1e6:.2f} -> {total_after / 1e6:.2f} MB "
              f"({100.0 * total_after / total_before:.1f}%)")
    if failed:
        print(f"{len(failed)} failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
