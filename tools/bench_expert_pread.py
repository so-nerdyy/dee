#!/usr/bin/env python3
"""Concurrent ExpertStore pread microbench (Phase A, exact-staging track).

Measures the real concurrent `pread` behavior of the backing expert store at
queue depths 1..16 using the official record size (13,369,344 B). Runs on the
Kaggle host against the /tmp trace store; a Windows fallback exists ONLY to
let the test suite validate the bookkeeping locally (clearly labeled, not for
measurements).

Patterns:
  seq      : sequential-ish records (record-index order)
  dispersed: shuffled record order (random expert identity, same sizes)
  journal  : sealed route-journal demand order (per-step, exact identities)

Cache states (reported separately, never mixed):
  coldish : posix_fadvise(DONTNEED) over the store before the pass; page-cache
            ground truth afterward is UNKNOWN (reported as such)
  warm    : same pass repeated immediately (page cache hot)

I/O mechanism: threads + os.pread. NOTE (measured, not assumed): one fd shared
by all threads serializes at the fd's file position — real concurrency needs
one fd per lane (or O_DIRECT/aio/io_uring, which this bench does NOT use and
does not claim). A shared-fd variant can be measured for the contrast.

Output JSON is shaped for ingestion by research/exact-critical-path
(see research/exact-staging/PREAD_BENCH_PROTOCOL.md for the contract).

Measured I/O only. No TPS claims.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from pathlib import Path

RECORD_BYTES = 13_369_344
QUEUE_DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16]
FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
HAS_PREAD = hasattr(os, "pread")
O_BINARY = getattr(os, "O_BINARY", 0)   # Windows: avoid text-mode 0x1A EOF


def _pread(fd: int, nbytes: int, offset: int) -> bytes:
    if HAS_PREAD:
        return os.pread(fd, nbytes, offset)
    # Windows/local-validation fallback: lseek+read on a per-lane fd.
    # NOT position-safe across shared fds; bench always uses per-lane fds.
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, nbytes)


def _posix_fadvise_safe(fd: int, off: int, n: int, advice: int) -> bool:
    try:
        os.posix_fadvise(fd, off, n, advice)
        return True
    except (AttributeError, OSError):
        return False


def _one_read(fd: int, offset: int, nbytes: int, lane: int, idx: int,
              t0: float) -> dict:
    t_submit = time.perf_counter()
    buf = _pread(fd, nbytes, offset)
    t_complete = time.perf_counter()
    return {
        "idx": idx,
        "record_index": offset // nbytes,
        "offset": offset,
        "nbytes": len(buf),
        "t_submit": round(t_submit - t0, 6),
        "t_complete": round(t_complete - t0, 6),
        "lane": lane,
        "checksum8": int.from_bytes(buf[:8], "little") if buf else -1,
        "ms": round((t_complete - t_submit) * 1e3, 3),
    }


def _run_pass(store_path: Path, order: list[int], depth: int, per_lane_fds: bool,
              target_records: int) -> dict:
    """One pass: read `target_records` records in `order` with `depth` lanes."""
    n = min(target_records, len(order))
    offsets = [idx * RECORD_BYTES for idx in order[:n]]
    fds = [os.open(store_path, os.O_RDONLY | O_BINARY)
           for _ in range(depth if per_lane_fds else 1)]
    outcomes: list[dict] = []
    lock = threading.Lock()
    next_idx = 0
    t0 = time.perf_counter()
    cpu0 = time.process_time()

    def worker(lane: int):
        nonlocal next_idx
        fd = fds[lane % len(fds)]
        while True:
            with lock:
                if next_idx >= n:
                    return
                i = next_idx
                next_idx += 1
            outcomes.append(_one_read(fd, offsets[i], RECORD_BYTES, lane, i, t0))

    threads = [threading.Thread(target=worker, args=(l,)) for l in range(depth)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    cpu_s = time.process_time() - cpu0
    for fd in fds:
        os.close(fd)

    outcomes.sort(key=lambda o: o["idx"])
    lat = sorted(o["ms"] for o in outcomes)

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(len(lat) * p))]

    bytes_done = sum(o["nbytes"] for o in outcomes)
    busy = sum(o["ms"] for o in outcomes) / 1e3
    zero_suspicious = sum(1 for o in outcomes if o["checksum8"] == 0)
    empty = sum(1 for o in outcomes if o["nbytes"] != RECORD_BYTES)
    return {
        "queue_depth_requested": depth,
        "fds": "per_lane" if per_lane_fds else "shared",
        "reads": n,
        "bytes_requested": n * RECORD_BYTES,
        "bytes_completed": bytes_done,
        "short_or_empty_reads": empty,
        "zero_prefix_records": zero_suspicious,
        "wall_s": round(wall, 4),
        "cpu_s": round(cpu_s, 4),
        "aggregate_mb_s": round(bytes_done / wall / 1e6, 2) if wall > 0 else 0.0,
        "p50_ms": round(pct(0.50), 3),
        "p90_ms": round(pct(0.90), 3),
        "p99_ms": round(pct(0.99), 3),
        "max_ms": round(lat[-1], 3),
        "concurrency_achieved_est": round(busy / wall, 2) if wall > 0 else 0.0,
    }


def _fadvise_drop(store_path: Path, nbytes: int) -> bool:
    """Best-effort cache resistance. Returns whether fadvise was AVAILABLE —
    not whether the cache actually dropped (unknowable from userspace here)."""
    fd = os.open(store_path, os.O_RDONLY | O_BINARY)
    ok = _posix_fadvise_safe(fd, 0, nbytes, FADV_DONTNEED)
    os.close(fd)
    return ok


def load_meta_index(meta_path: Path) -> dict:
    """Load dee4 metadata.json and invert to 'layer,expert' -> record_index.
    Tolerant to schema variants; raises with observed keys otherwise."""
    data = json.loads(meta_path.read_text())
    if isinstance(data, dict):
        for key in ("records", "experts", "entries", "index"):
            if key in data and isinstance(data[key], list):
                idx = {}
                for i, ent in enumerate(data[key]):
                    layer = ent.get("layer", ent.get("expert_layer"))
                    expert = ent.get("expert", ent.get("expert_id"))
                    if layer is None or expert is None:
                        raise SystemExit(f"metadata entry missing layer/expert: {ent}")
                    idx[f"{layer},{expert}"] = ent.get("record_index", i)
                return idx
        raise SystemExit(f"metadata keys not recognized: {list(data)[:8]}")
    raise SystemExit("metadata.json: unexpected top-level type")


def build_journal_order(journal: Path, meta: dict) -> dict:
    """Sealed route-journal demand order: per forward step, the deduped
    sequence of record indices demanded, in execution order."""
    per_step: dict = {}
    with journal.open() as fh:
        for line in fh:
            rec = json.loads(line)
            fs, L = rec["forward_step"], rec["layer"]
            for row in rec["expert_ids_rank_order"]:
                for e in row:
                    ri = meta.get(f"{L},{e}")
                    if ri is not None:
                        per_step.setdefault(fs, {}).setdefault(L, []).append(ri)
    order = {}
    for fs, layers in per_step.items():
        seen, seq = set(), []
        for L in sorted(layers):
            for ri in layers[L]:
                if ri not in seen:
                    seen.add(ri)
                    seq.append(ri)
        order[f"fwd{fs}"] = seq
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True,
                    help="Path to experts.dee4 (or any multi-record file).")
    ap.add_argument("--record-bytes", type=int, default=RECORD_BYTES)
    ap.add_argument("--records", type=int, default=96,
                    help="Records per pass (default: one decode step's reads).")
    ap.add_argument("--depths", type=int, nargs="+", default=QUEUE_DEPTHS)
    ap.add_argument("--patterns", nargs="+", default=["seq", "dispersed"],
                    choices=["seq", "dispersed", "journal"])
    ap.add_argument("--journal", type=Path, default=None,
                    help="routed_experts.jsonl for pattern=journal")
    ap.add_argument("--journal-meta", type=Path, default=None,
                    help="dee4 metadata.json (for record identity mapping)")
    ap.add_argument("--repeat", type=int, default=2,
                    help="passes per (pattern, depth); 1st=coldish, 2nd=warm")
    ap.add_argument("--shared-fd-variant", action="store_true",
                    help="also measure one shared fd across lanes (warm only)")
    ap.add_argument("--label", default="", help="free-form host label")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not HAS_PREAD:
        print("WARNING: os.pread unavailable on this platform; using an "
              "lseek+read fallback. Valid for bookkeeping tests ONLY — do not "
              "report measurements from this platform.")

    rb = args.record_bytes
    nbytes_total = args.store.stat().st_size
    record_count = nbytes_total // rb
    if record_count < 1:
        raise SystemExit("store too small for one record")
    for d in args.depths:
        if d < 1 or d > 64:
            raise SystemExit("depths must be in [1, 64]")

    rng = random.Random(20260904)
    seq = list(range(record_count))
    dispersed = seq[:]
    rng.shuffle(dispersed)
    journal_orders = None
    if "journal" in args.patterns:
        if not (args.journal and args.journal.exists()
                and args.journal_meta and args.journal_meta.exists()):
            raise SystemExit("pattern=journal requires --journal and --journal-meta")
        journal_orders = build_journal_order(args.journal,
                                             load_meta_index(args.journal_meta))

    results = []

    def do_pass(pattern: str, order: list[int], depth: int, warm: bool):
        if not warm:
            fadvise_available = _fadvise_drop(args.store, nbytes_total)
        else:
            fadvise_available = True
        r = _run_pass(args.store, order, depth, per_lane_fds=True,
                      target_records=args.records)
        r.update({"pattern": pattern,
                  "cache_state": "warm" if warm else "coldish",
                  "fadvise_available": bool(fadvise_available),
                  "page_cache_ground_truth": "unknown"})
        results.append(r)

    for pattern in args.patterns:
        if pattern == "journal":
            for key, order in sorted(journal_orders.items()):
                if len(order) < 8:
                    continue
                for depth in args.depths:
                    for rep in range(args.repeat):
                        do_pass(f"journal/{key}", order, depth, rep > 0)
                        if args.shared_fd_variant and rep > 0:
                            r = _run_pass(args.store, order, depth,
                                          per_lane_fds=False,
                                          target_records=args.records)
                            r.update({"pattern": f"journal/{key}",
                                      "cache_state": "warm",
                                      "fadvise_available": True,
                                      "page_cache_ground_truth": "unknown"})
                            results.append(r)
            continue
        base = seq if pattern == "seq" else dispersed
        for depth in args.depths:
            for rep in range(args.repeat):
                do_pass(pattern, base, depth, rep > 0)
                if args.shared_fd_variant and rep > 0:
                    r = _run_pass(args.store, base, depth, per_lane_fds=False,
                                  target_records=args.records)
                    r.update({"pattern": pattern, "cache_state": "warm",
                              "fadvise_available": True,
                              "page_cache_ground_truth": "unknown"})
                    results.append(r)

    payload = {
        "schema": "exact-staging/pread-bench-v1",
        "label": args.label,
        "store": str(args.store),
        "store_bytes": nbytes_total,
        "record_bytes": rb,
        "record_count": record_count,
        "io_mechanism": ("threads+os.pread, one fd per lane"
                         if HAS_PREAD else
                         "threads+lseek/read fallback (LOCAL VALIDATION ONLY)"),
        "concurrency_caveat": "per-lane fds give independent file positions; "
                              "kernel-level async (aio/io_uring) NOT used and "
                              "not claimed; shared-fd variant measured on request",
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"store={args.store} records={record_count} record_bytes={rb}")
    for r in results:
        print(f"{r['pattern']:<14} d={r['queue_depth_requested']:>2} "
              f"{r['cache_state']:<7} {r['aggregate_mb_s']:>8.1f} MB/s "
              f"p50={r['p50_ms']:>7.1f} p90={r['p90_ms']:>7.1f} "
              f"p99={r['p99_ms']:>8.1f} conc~{r['concurrency_achieved_est']}")


if __name__ == "__main__":
    main()
