#!/usr/bin/env python3
"""P2.4 storage-decision probe (CPU-only, no GPU).

The v15/v16 decode reads experts from the Kaggle dataset mount (~13 MB/s loop
device) = 95.7% of wall time.  The harness prefers the mount over the /tmp
download path, but the /tmp (root overlay) READ speed was never measured.
This kernel measures it on a CPU-only worker (instant allocation, no
dual-GPU pool contention) so we can decide where the runtime should read
expert bytes from.

Tests:
  1. /tmp  pread + mmap (16/64/256 MiB)
  2. /kaggle/working pread (sanity vs 126 MB/s roofline)
  3. dataset mount: if present, read one real shard + measure MB/s
  4. HF range download of one shard to /tmp (resume-capable) + re-read speed
  5. DEE4 repack of layers 0-1 into /tmp, checksum vs source, contiguous read
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

REV = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
SHARD = "model-00002-of-00048.safetensors"  # contains early routed experts
WORK = Path("/kaggle/working")
TMP = Path("/tmp/dee-storage-decision")
DATASET = Path("/kaggle/input/deepseek-v4-flash-0731-shards")
REPORT = WORK / "storage_decision.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pread_bench(name: str, path: Path, size_mib: int, n_iter: int = 5) -> dict:
    sz = size_mib << 20
    fd = os.open(str(path), os.O_RDONLY)
    try:
        fsize = os.fstat(fd).st_size
        sz = min(sz, fsize)
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            data = os.pread(fd, sz, 0)
            _ = data[0] + data[-1]
            times.append(time.perf_counter() - t0)
        times.sort()
        med = times[len(times) // 2]
        return {"name": name, "size_mib": sz >> 20, "p50_s": round(med, 4),
                "mbps": round(sz / (1 << 20) / med, 1)}
    finally:
        os.close(fd)


def mmap_bench(name: str, path: Path, size_mib: int, n_iter: int = 3) -> dict:
    sz = size_mib << 20
    fd = os.open(str(path), os.O_RDONLY)
    try:
        fsize = os.fstat(fd).st_size
        sz = min(sz, fsize)
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            with mmap.mmap(fd, sz, access=mmap.ACCESS_READ) as m:
                acc = 0
                step = 4096
                for off in range(0, sz, step):
                    acc += m[off]
            times.append(time.perf_counter() - t0)
        times.sort()
        med = times[len(times) // 2]
        return {"name": name, "size_mib": sz >> 20, "p50_s": round(med, 4),
                "mbps": round(sz / (1 << 20) / med, 1)}
    finally:
        os.close(fd)


def download_shard(dest: Path, url: str) -> float:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        cr = resp.headers.get("Content-Range", "")
    want = int(cr.split("/")[1])
    have = dest.stat().st_size if dest.is_file() else 0
    if have == want:
        log(f"  shard already complete ({want} bytes)")
        return 0.0
    chunk = 32 << 20
    t0 = time.perf_counter()
    with open(dest, "ab") as fh:
        while have < want:
            end = min(have + chunk - 1, want - 1)
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={have}-{end}"})
            for attempt in range(8):
                try:
                    with urllib.request.urlopen(req, timeout=600) as r:
                        data = r.read(chunk + 1)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError,
                        ConnectionError, TimeoutError) as e:
                    last = e
                    time.sleep(2.0 * (2 ** attempt))
            else:
                raise ConnectionError(f"download failed: {last!r}")
            fh.write(data)
            have += len(data)
    dt = time.perf_counter() - t0
    return dt


def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hlen))


def read_tensor_bytes(path: Path, header: dict, name: str) -> bytes:
    meta = header[name]
    off = meta["data_offsets"]
    length = off[1] - off[0]
    with open(path, "rb") as f:
        f.seek(8 + off[0])
        return f.read(length)


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    report: dict = {"schema": "p2.4-storage-decision-v1",
                    "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    results: list[dict] = []

    log("=== 1. /tmp root-overlay read speed ===")
    tmpf = TMP / "probe.bin"
    tmpf.write_bytes(os.urandom(256 << 20))
    for sz in (16, 64, 256):
        results.append(pread_bench(f"tmp-pread-{sz}MiB", tmpf, sz))
        results.append(mmap_bench(f"tmp-mmap-{sz}MiB", tmpf, sz))
    log("  /tmp probes done")

    log("=== 2. /kaggle/working read speed (sanity) ===")
    wf = WORK / "probe.bin"
    wf.write_bytes(os.urandom(256 << 20))
    results.append(pread_bench("working-pread-64MiB", wf, 64))
    wf.unlink(missing_ok=True)

    log("=== 3. dataset mount present? ===")
    mount_ok = DATASET.is_dir() and (DATASET / SHARD).is_file()
    report["dataset_mount_ok"] = mount_ok
    if mount_ok:
        sp = DATASET / SHARD
        results.append(pread_bench(f"dataset-pread-{SHARD[:9]}", sp, 16))
        # Random expert-sized gathers (simulate the runtime's mmap scatter)
        header = read_safetensors_header(sp)
        names = [n for n in header if n.startswith("layers.")]
        t0 = time.perf_counter(); total = 0
        for nm in names[:64]:
            b = read_tensor_bytes(sp, header, nm)
            total += len(b)
        dt = time.perf_counter() - t0
        results.append({"name": "dataset-expert-scatter-64",
                        "bytes": total, "seconds": round(dt, 3),
                        "mbps": round(total / (1 << 20) / max(dt, 1e-9), 1)})
    else:
        log(f"  dataset mount NOT present ({DATASET})")

    log("=== 4. HF download of one shard to /tmp ===")
    url = (f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/"
           f"resolve/{REV}/{SHARD}")
    dest = TMP / SHARD
    dl_s = download_shard(dest, url)
    dl_mbps = (dest.stat().st_size / (1 << 20) / max(dl_s, 1e-9)
               if dl_s > 0 else None)
    results.append({"name": "hf-download-shard", "bytes": dest.stat().st_size,
                    "seconds": round(dl_s, 1), "mbps": round(dl_mbps, 1)
                    if dl_mbps else None})
    results.append(pread_bench("tmp-shard-pread-16MiB", dest, 16))
    results.append(mmap_bench("tmp-shard-mmap-16MiB", dest, 16))

    log("=== 5. DEE4 repack layers 0-1 to /tmp + checksum ===")
    try:
        sys.path.insert(0, str(Path("/tmp").parent))  # noop guard
        from pathlib import Path as _P
        # inline minimal repack: w1/w2/w3 + scales for layer 0, experts 0-3
        idx = TMP / "index.json"
        if not idx.is_file():
            idx_url = (f"https://huggingface.co/deepseek-ai/"
                       f"DeepSeek-V4-Flash-0731/resolve/{REV}/"
                       f"model.safetensors.index.json")
            idx.write_bytes(urllib.request.urlopen(idx_url, timeout=300).read())
        wm = json.loads(idx.read_text("utf-8"))["weight_map"]
        hdr_cache: dict[str, dict] = {}
        bank = TMP / "dee4"
        bank.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter(); nbytes = 0; nchk = 0
        for L in range(2):
            for e in range(4):
                for proj in ("w1", "w2", "w3"):
                    for kind in ("weight", "scale"):
                        nm = f"layers.{L}.ffn.experts.{e}.{proj}.{kind}"
                        if nm not in wm:
                            continue
                        sn = wm[nm]
                        sp = TMP / sn if (TMP / sn).is_file() else None
                        if sp is None:
                            # fetch from HF (already downloaded shard 2 covers
                            # some early layers; otherwise skip)
                            continue
                        hdr = hdr_cache.get(sn)
                        if hdr is None:
                            hdr = read_safetensors_header(sp)
                            hdr_cache[sn] = hdr
                        data = read_tensor_bytes(sp, hdr, nm)
                        out = bank / f"L{L:03d}_e{e}_{proj}_{kind}.bin"
                        out.write_bytes(data)
                        nbytes += len(data)
                        nchk += 1
        repack_s = time.perf_counter() - t0
        # checksum: re-read bank and compare to source for the first tensor
        chk_ok = True
        for L in range(2):
            for e in range(4):
                for proj in ("w1", "w2", "w3"):
                    for kind in ("weight", "scale"):
                        nm = f"layers.{L}.ffn.experts.{e}.{proj}.{kind}"
                        if nm not in wm:
                            continue
                        sn = wm[nm]
                        sp = TMP / sn
                        if not sp.is_file():
                            continue
                        hdr = hdr_cache[sn]
                        src = read_tensor_bytes(sp, hdr, nm)
                        dst = (bank / f"L{L:03d}_e{e}_{proj}_{kind}.bin").read_bytes()
                        if src != dst:
                            chk_ok = False
                            log(f"  CHECKSUM MISMATCH {nm}")
        results.append({"name": "dee4-repack-tmp", "tensors": nchk,
                        "bytes": nbytes, "seconds": round(repack_s, 3),
                        "mbps": round(nbytes / (1 << 20) / max(repack_s, 1e-9), 1),
                        "checksum_ok": chk_ok})
        # contiguous bank read
        big = TMP / "bank.bin"
        with open(big, "wb") as f:
            for p in sorted(bank.glob("*.bin")):
                f.write(p.read_bytes())
        results.append(pread_bench("tmp-dee4-contig-256MiB", big, 256))
        log(f"  repack {nchk} tensors {nbytes/(1<<20):.0f} MiB in "
            f"{repack_s:.1f}s, checksum_ok={chk_ok}")
    except Exception as exc:  # noqa: BLE001
        log(f"  DEE4 repack failed: {exc}")
        import traceback
        traceback.print_exc()
        results.append({"name": "dee4-repack-tmp", "error": str(exc)})

    report["results"] = results
    REPORT.write_text(json.dumps(report, indent=2), "utf-8")
    log("RESULT " + json.dumps(report, indent=1))
    log("=== DONE ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        tb = traceback.format_exc()
        print("FATAL " + tb, flush=True)
        (WORK / "storage-decision-error.txt").write_text(tb, "utf-8")
        sys.exit(0)
