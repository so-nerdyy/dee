#!/usr/bin/env python3
"""dee4 lossless full-bank scan session driver (research only, CPU-only).

Runs on Kaggle with dataset nivind/deepseek-v4-flash-0731-shards mounted:
reconstructs all 2364 trace-bank records byte-exactly (verified against
integrity sidecar), runs the sweep codec tier on all + full tier on a
stratified 96 + a zstd-dictionary study, writes CSV + summary to
/kaggle/working. Always exits 0 (validity from evidence, pack-cap pattern).
No GPU, no model execution, no production code touched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Pinned harness source (commit B fills these in; shas enforced at fetch).
HARNESS_COMMIT = "ec31f174c14caf25fd92214060d50cabd05a240c"
HARNESS_BASE = ("https://raw.githubusercontent.com/so-nerdyy/dee/"
                + HARNESS_COMMIT + "/research/dee4-lossless")
HARNESS_FILES = {
    "records.py": "4266a1e0bf2468759209fd1da8a69a42d69331c7045d7563a7fbb7f6d1684d1f",
    "dq_codecs.py": "4347c34b0c215f160593c6b89f8874329195f6a983fd7451b58d4ff5a44687f7",
    "scan.py": "c5318a950aaeb28409bb2058c065a876be9e0bf3fa59be3d29263c720298b06b",
}
INPUT_FILES = {
    "model.safetensors.index.json": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
    "journal.txt": "a1c63d022aa55e5552bf7220bf7e0ec6eade4682fddfdb26a938d54c0caf57c6",
    "dee4-integrity.jsonl": "55129bc080555d57caff1d89899de0e3cd85a8f5cfa57668b1b4ce6b1545fea4",
}
NWORKERS = 4
FULL_EVERY = 25  # stratified full-tier subset: every 25th record
DICT_TRAIN = 256
DICT_TEST = 64

WORK = Path("/kaggle/working")
TMP = Path("/tmp/dee4-scan")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [scan] {msg}", flush=True)


def fetch(url: str, dst: Path, sha: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "dee-scan/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    dst.write_bytes(data)
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError(f"sha mismatch for {url}")


def find_shards() -> dict:
    found: dict[str, str] = {}
    for base in (Path("/kaggle/input"),):
        for p in base.rglob("model-*-of-00048.safetensors"):
            found[p.name] = str(p.parent)
    if len(found) != 48:
        raise RuntimeError(f"need 48 shards, found {len(found)}")
    return found


def _worker_init(shard_dirs: dict, wmap: dict):
    global _BE
    sys.path.insert(0, str(TMP))
    from records import MountBackend
    # MountBackend takes one root; shards share one dir in practice.
    roots = set(shard_dirs.values())
    if len(roots) != 1:
        raise RuntimeError(f"shards span dirs: {roots}")
    _BE = MountBackend(wmap, next(iter(roots)))


def _worker_job(job: tuple) -> dict:
    layer, expert, tier, want_sha = job
    sys.path.insert(0, str(TMP))
    from records import reconstruct, sha256_hex
    from scan import scan_record
    t0 = time.time()
    rec = reconstruct(layer, expert, _BE)
    got = sha256_hex(rec)
    if got != want_sha:
        return {"record": (layer, expert), "exact": False,
                "rows": [], "fetch_s": time.time() - t0}
    t1 = time.time()
    rows = scan_record(layer, expert, rec, tier)
    return {"record": (layer, expert), "exact": True, "rows": rows,
            "fetch_s": t1 - t0, "scan_s": time.time() - t1}


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    summary: dict = {"session": "dee4-lossless-scan", "exact_records": 0,
                     "mismatch": []}
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "lz4", "zstandard", "constriction"],
                       check=True, timeout=900)
        for name, sha in list(HARNESS_FILES.items()) + list(INPUT_FILES.items()):
            fetch(f"{HARNESS_BASE}/{name}" if name in HARNESS_FILES
                  else f"{HARNESS_BASE}/inputs/{name}",
                  TMP / name, sha)
        log("harness fetched + sha-verified")
        wmap = json.loads((TMP / "model.safetensors.index.json").read_text())
        wmap = wmap["weight_map"]
        pairs = set()
        for line in (TMP / "journal.txt").read_text().splitlines():
            p = line.split()
            if p:
                pairs.add((int(p[0]), int(p[1])))
        integ = {}
        for line in (TMP / "dee4-integrity.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                integ[(r["layer"], r["expert"])] = r["record_sha256"]
        sel = sorted(pairs)
        summary["selection"] = len(sel)
        if len(sel) != 2364 or set(sel) != set(integ):
            raise RuntimeError("selection != integrity set")
        shards = find_shards()
        summary["shards"] = len(shards)
        jobs = [(l, e, "full" if i % FULL_EVERY == 0 else "sweep",
                 integ[(l, e)]) for i, (l, e) in enumerate(sel)]
        summary["full_tier_n"] = sum(1 for j in jobs if j[2] == "full")
        all_rows: list[dict] = []
        n_exact = 0
        with ProcessPoolExecutor(
                max_workers=NWORKERS,
                initializer=_worker_init, initargs=(shards, wmap)) as ex:
            for i, res in enumerate(ex.map(_worker_job, jobs)):
                if res["exact"]:
                    n_exact += 1
                    all_rows.extend(res["rows"])
                else:
                    summary["mismatch"].append(res["record"])
                if (i + 1) % 200 == 0:
                    log(f"{i + 1}/{len(jobs)} records, "
                        f"{n_exact} exact")
        summary["exact_records"] = n_exact
        with open(WORK / "scan_out.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        summary["csv_rows"] = len(all_rows)
        # zstd-dictionary study (global-state, labeled separately).
        sys.path.insert(0, str(TMP))
        from records import MountBackend
        import zstandard as zstd
        roots = set(shards.values())
        be = MountBackend(wmap, next(iter(roots)))
        from records import reconstruct as _rec
        train = [_rec(l, e, be) for (l, e) in sel[:DICT_TRAIN]]
        tests = [_rec(l, e, be)
                 for (l, e) in sel[DICT_TRAIN:DICT_TRAIN + DICT_TEST]]
        d = zstd.train_dictionary(1 << 20, train)
        cc = zstd.ZstdCompressor(level=3, dict_data=d)
        ratios = []
        for t in tests:
            c = cc.compress(t)
            assert zstd.ZstdDecompressor(
                dict_data=d).decompress(c) == t
            ratios.append(len(c) / len(t))
        summary["dict_study"] = {
            "train": DICT_TRAIN, "test": DICT_TEST,
            "mean_ratio": sum(ratios) / len(ratios),
            "min_ratio": min(ratios), "max_ratio": max(ratios),
            "note": "global-state; NOT per-record independent",
        }
        summary["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - evidence-first
        summary["status"] = "failed"
        summary["error"] = repr(exc)[:1000]
        log("FAILED: " + repr(exc)[:500])
    summary["wall_s"] = time.time() - t_start
    (WORK / "scan_summary.json").write_text(json.dumps(summary, indent=2))
    log("summary: " + json.dumps(summary)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
