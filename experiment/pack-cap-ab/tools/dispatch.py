#!/usr/bin/env python3
"""Dispatch + poll tooling for the pack-cap A/B session kernels.

Usage:
  python tools/dispatch.py dispatch --dir kernels/session1 --label s1
  python tools/dispatch.py status   --kernel nivind/dee-cpp-dsv4-pack-cap-s1-20260905
  python tools/dispatch.py fetch    --kernel nivind/dee-cpp-dsv4-pack-cap-s1-20260905 \
                                    --out results/live/s1

Matches the seal-era launch recipe exactly (recorded in
dee.cpp/tmp/host-reuse-217-ab-20260904/run_control.py):
  api.kernels_push(folder, timeout="14400", acc="NvidiaTeslaT4")

One authorized dispatch per session; NO automatic retries. Dispatch and
fetch actions are recorded to results/live/dispatch-log.jsonl.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RESULTS = PKG / "results"
LIVE = RESULTS / "live"
LOG = LIVE / "dispatch-log.jsonl"


def now_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def record(entry: dict) -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    entry = {"recorded_at_utc": now_utc(), **entry}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def dispatch(dir_path: Path) -> int:
    import kaggle
    api = kaggle.api
    meta = json.loads((dir_path / "kernel-metadata.json").read_text())
    slug = meta["id"]
    print(f"dispatching {slug} from {dir_path} "
          f"(timeout=14400, acc=NvidiaTeslaT4)")
    resp = api.kernels_push(str(dir_path), timeout="14400",
                            acc="NvidiaTeslaT4")
    d = resp if isinstance(resp, dict) else {
        k: getattr(resp, k) for k in dir(resp)
        if not k.startswith("_") and not callable(getattr(resp, k))}
    print(json.dumps(d, indent=1, default=str)[:1500])
    record({"action": "dispatch", "kernel": slug,
            "dir": str(dir_path), "response": d})
    if d.get("error"):
        print(f"DISPATCH ERROR: {d['error']}")
        return 1
    return 0


def status(kernel: str) -> int:
    import kaggle
    api = kaggle.api
    r = api.kernels_status(kernel)
    d = r if isinstance(r, dict) else {
        k: getattr(r, k) for k in dir(r)
        if not k.startswith("_") and not callable(getattr(r, k))}
    print(json.dumps(d, indent=1, default=str))
    return 0


def fetch(kernel: str, out: Path) -> int:
    import kaggle
    api = kaggle.api
    out.mkdir(parents=True, exist_ok=True)
    print(f"fetching {kernel} output -> {out}")
    api.kernels_output(kernel, path=str(out))
    files = sorted(p.name for p in out.iterdir())
    print(json.dumps({"files": files}, indent=1))
    record({"action": "fetch", "kernel": kernel, "out": str(out),
            "files": files})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dispatch")
    d.add_argument("--dir", type=Path, required=True)
    s = sub.add_parser("status")
    s.add_argument("--kernel", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--kernel", required=True)
    f.add_argument("--out", type=Path, default=LIVE)
    args = ap.parse_args()
    if args.cmd == "dispatch":
        return dispatch(args.dir)
    if args.cmd == "status":
        return status(args.kernel)
    if args.cmd == "fetch":
        return fetch(args.kernel, args.out)
    return 2


if __name__ == "__main__":
    sys.exit(main())
