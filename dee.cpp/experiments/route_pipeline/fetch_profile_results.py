#!/usr/bin/env python3
"""Fetch host-sync profile kernel output and stage it for ingestion.

Downloads the KAGGLE output bundle for the pushed session kernel,
verifies the session-summary, and lays out:
    fetched/profile-run-{off,on,off2}/
    fetched/abc-mechanics.json
    fetched/session-summary.json
for run_evidence-style ingestion via evidence.py (pair off/on + order
analysis with off2). Read-only against the remote; never modifies evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

KERNEL = "nivind/dee-cpp-host-sync-profile-20260905"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernel", default=KERNEL)
    args = parser.parse_args()

    import os
    os.environ.setdefault("KAGGLE_CONFIG_DIR", r"C:\Users\carth\.kaggle")
    import kaggle  # noqa: PLC0415

    args.out.mkdir(parents=True, exist_ok=True)
    bundle = args.out / "kernel-output.zip"
    print(f"downloading {args.kernel} output -> {bundle}", flush=True)
    kaggle.api.kernels_output(args.kernel, path=str(args.out),
                              force=True)
    zips = sorted(args.out.glob("*.zip"))
    print(f"zips: {[z.name for z in zips]}")
    for zip_path in zips:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(args.out / "unpacked")
    summary = args.out / "unpacked" / "session-summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        print(json.dumps({k: (v if k != "arms" else list(v))
                          for k, v in data.items()}, indent=1)[:2000])
    else:
        print("WARNING: session-summary.json absent from output")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
