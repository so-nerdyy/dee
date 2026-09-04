#!/usr/bin/env python3
"""Kaggle A/B/C mechanics runner (route-pipeline track).

Measures the MECHANICS behind candidates A/B/C on a T4 host WITHOUT the
campaign checkout and WITHOUT the checkpoint. All bytes are synthetic and
labeled synthetic; shapes for the shared-expert probe are ASSUMED
(torch linear 4096x2048 fp16) and labeled as such. No timing here proves an
end-to-end win — it bounds the overlap windows the host sim needs.

Cases (independent, each emits JSON):
  A. hash staging mechanics: 18 x 12.75 MiB file reads + pinned H2D,
     serial vs batched-submit; reports hidden bytes/ms.
  B. contention probe: shared-shaped GEMM alone vs under H2D-stream
     saturation; reports efficiency ratio c in [0,1] for shared_overlap.py.
  C. barrier cost: cudaStreamSynchronize vs event record/wait/query;
     bounds the prize of narrowing the per-layer sync and route-D2H sync.

Usage (fresh notebook, torch with CUDA required):
    python3 kaggle_runner_abc.py --workdir /tmp/rpabc --reps 5
    python3 kaggle_runner_abc.py --workdir /tmp/rpabc --allow-non-sm75

Refuses hosts without CUDA. Refuses non-SM75 unless --allow-non-sm75
(development only; not T4 evidence).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPERT_BYTES = 13369344  # 12.75 MiB packed record
HASH_READS = 18          # 3 layers x top-6


def need_torch():
    try:
        import torch
    except ImportError:
        raise SystemExit("REFUSE: torch not importable on this host.")
    if not torch.cuda.is_available():
        raise SystemExit("REFUSE: torch.cuda unavailable; no GPU here.")
    return torch


def check_gpu(torch, allow_other: bool) -> dict:
    props = torch.cuda.get_device_properties(0)
    sm = f"{props.major}.{props.minor}"
    ok = sm == "7.5"
    if not ok and not allow_other:
        raise SystemExit(f"REFUSE: SM{sm} is not SM75; pass --allow-non-sm75 "
                         "for development (not T4 evidence).")
    return {"name": props.name, "sm": sm, "sm75": ok,
            "override": bool(not ok and allow_other),
            "total_vram_bytes": props.total_memory}


def stats_ms(samples: list[float]) -> dict:
    return {"mean": statistics.fmean(samples), "p50": statistics.median(samples),
            "min": min(samples), "n": len(samples), "samples": samples}


def case_a(torch, workdir: Path, reps: int) -> dict:
    f = workdir / "hash18.bin"
    if not f.is_file() or f.stat().st_size != HASH_READS * EXPERT_BYTES:
        with open(f, "wb") as fh:
            fh.write(os.urandom(1 << 20))
            fh.seek(HASH_READS * EXPERT_BYTES - 1)
            fh.write(b"\x00")
    s = torch.cuda.Stream()
    serial, batched = [], []
    for _ in range(reps):
        # Serial: read chunk -> H2D -> sync, one expert at a time.
        t0 = time.perf_counter()
        with open(f, "rb") as fh:
            for _ in range(HASH_READS):
                buf = fh.read(EXPERT_BYTES)
                d = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device="cuda")
                d.copy_(torch.from_numpy(
                    __import__("numpy").frombuffer(buf, dtype="uint8")),
                    non_blocking=False)
        torch.cuda.synchronize()
        serial.append((time.perf_counter() - t0) * 1000.0)
        # Batched submit: all reads queued, async H2Ds on one stream, one sync.
        t0 = time.perf_counter()
        with open(f, "rb") as fh:
            blobs = [fh.read(EXPERT_BYTES) for _ in range(HASH_READS)]
        import numpy as np
        with torch.cuda.stream(s):
            devs = []
            for b in blobs:
                d = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device="cuda")
                d.copy_(torch.from_numpy(np.frombuffer(b, dtype="uint8")),
                        non_blocking=True)
                devs.append(d)
        torch.cuda.synchronize()
        batched.append((time.perf_counter() - t0) * 1000.0)
        del devs
    return {"fixture": "synthetic", "reads": HASH_READS,
            "bytes": HASH_READS * EXPERT_BYTES,
            "serial_ms": stats_ms(serial), "batched_ms": stats_ms(batched),
            "status": "ok"}


def case_b(torch, reps: int) -> dict:
    import torch.nn.functional as Fn
    H, I = 4096, 2048  # ASSUMED shared-expert-ish shape, labeled
    w = torch.randn(I, H, dtype=torch.float16, device="cuda")
    x = torch.randn(H, dtype=torch.float16, device="cuda")
    s = torch.cuda.Stream()
    alone, loaded = [], []
    big = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")  # 256 MiB sink
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y = Fn.linear(x, w)
        torch.cuda.synchronize()
        alone.append((time.perf_counter() - t0) * 1000.0 / 20)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(20):
                big.copy_(torch.zeros_like(big))
        for _ in range(20):
            y = Fn.linear(x, w)
        torch.cuda.synchronize()
        loaded.append((time.perf_counter() - t0) * 1000.0 / 20)
    a, b = statistics.median(alone), statistics.median(loaded)
    return {"fixture": "synthetic", "shape_note": "ASSUMED 4096x2048 fp16",
            "alone_ms": stats_ms(alone), "loaded_ms": stats_ms(loaded),
            "efficiency_c": round(a / b, 4) if b else None,
            "status": "ok"}


def case_c(torch, reps: int) -> dict:
    s = torch.cuda.current_stream()
    syncs, events = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        syncs.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        e = torch.cuda.Event()
        e.record(s)
        e.synchronize()
        events.append((time.perf_counter() - t0) * 1000.0)
    return {"sync_ms": stats_ms(syncs), "event_ms": stats_ms(events),
            "status": "ok"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/rpabc"))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--allow-non-sm75", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    torch = need_torch()
    args.workdir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "package": "route_pipeline/abc-mechanics",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": check_gpu(torch, args.allow_non_sm75),
        "torch": torch.__version__,
    }
    try:
        report["A_hash_staging"] = case_a(torch, args.workdir, args.reps)
        report["B_contention"] = case_b(torch, args.reps)
        report["C_barrier"] = case_c(torch, args.reps)
        report["status"] = "ok"
        code = 0
    except Exception as exc:  # noqa: BLE001
        report["status"] = f"error: {exc}"
        code = 1
    out = args.out_json or (args.workdir / "abc_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nreport: {out}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
