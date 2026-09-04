#!/usr/bin/env python3
"""Storage / checkpoint-path benchmark (measures, never fabricates).

Measures for a selected --storage-path:
- filesystem + free/total disk,
- sequential read throughput (configurable block sizes),
- random-read throughput (configurable block sizes),
- cold (first read after flush) vs warm (subsequent reads) behavior.

Cold-cache honesty: portable user-space code cannot reliably drop the OS page
cache. Cold = first read after write+flush (+ posix_fadvise DONTNEED on Linux
where available). The report labels exactly what was done; where the cache
could not be dropped it says so instead of claiming a true cold read.

No checkpoint is required: the tool writes its own temp file (default 128 MiB,
configurable) and deletes it afterwards (unless --keep-file).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.0.0"
MIB = 1024 * 1024


def parse_bytes(value: str) -> int:
    text = value.strip().lower().replace(" ", "").replace("_", "")
    mults = {"gib": 1024 ** 3, "gb": 1000 ** 3, "mib": MIB, "mb": 1000 ** 2,
             "kib": 1024, "kb": 1000, "b": 1, "": 1}
    for suffix in sorted(mults, key=len, reverse=True):
        if suffix and text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * mults[suffix])
        if not suffix:
            return int(float(text) * mults[suffix])
    raise ValueError(f"unparseable byte quantity: {value!r}")


def parse_block_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(parse_bytes(part))
    if not out:
        raise ValueError("empty block list")
    return out


def detect_filesystem(path: Path) -> dict:
    system = platform.system()
    if system == "Windows":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            vol = ctypes.create_unicode_buffer(261)
            fs = ctypes.create_unicode_buffer(261)
            root = str(Path(path.resolve().anchor)) + "\\"
            ok = kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(root), vol, 261, None, None, None, fs, 261)
            if ok:
                return {"filesystem": fs.value or "UNKNOWN",
                        "method": "GetVolumeInformationW", "status": "measured"}
        except Exception as exc:  # noqa: BLE001
            return {"filesystem": "UNKNOWN", "method": "GetVolumeInformationW",
                    "status": "unknown", "reason": str(exc)}
        return {"filesystem": "UNKNOWN", "method": "GetVolumeInformationW",
                "status": "unknown", "reason": "call failed"}
    # POSIX: statvfs has no fs name; try `df -T` / `stat -f`.
    import subprocess
    for cmd in (["df", "-T", str(path)], ["stat", "-f", "-c", "%T", str(path)]):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if proc.returncode == 0 and proc.stdout.strip():
                return {"filesystem": proc.stdout.strip().splitlines()[-1][:120],
                        "method": " ".join(cmd[:2]), "status": "measured"}
        except Exception:  # noqa: BLE001
            continue
    try:
        st = os.statvfs(path)  # noqa: PTH118
        return {"filesystem": "UNKNOWN", "method": "statvfs",
                "status": "unknown",
                "reason": "fs type not exposed portably",
                "f_bsize": st.f_bsize}
    except Exception as exc:  # noqa: BLE001
        return {"filesystem": "UNKNOWN", "method": "none",
                "status": "unknown", "reason": str(exc)}


def disk_stats(path: Path) -> dict:
    try:
        usage = shutil.disk_usage(path)
        return {"total_bytes": usage.total, "free_bytes": usage.free,
                "used_bytes": usage.used, "status": "measured"}
    except Exception as exc:  # noqa: BLE001
        return {"total_bytes": None, "free_bytes": None, "used_bytes": None,
                "status": "unknown", "reason": str(exc)}


def drop_cache_hint(fd: int, path: Path) -> str:
    """Best-effort cache invalidation; returns a label of what was done."""
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]
            return "posix_fadvise(DONTNEED) issued; cold is first-read-after-invalidate"
        except Exception as exc:  # noqa: BLE001
            return f"posix_fadvise failed ({exc}); cold is first-read-after-flush only"
    return ("no portable cache-drop on this OS; cold is first-read-after-flush "
            "only, NOT a guaranteed cold read")


def write_test_file(path: Path, size_bytes: int, chunk_bytes: int = 4 * MIB) -> float:
    """Write `size_bytes` of incompressible-ish data; return write GiB/s."""
    rng = random.Random(0xDEE5)
    started = time.perf_counter()
    with open(path, "wb") as fh:
        remaining = size_bytes
        while remaining > 0:
            n = min(chunk_bytes, remaining)
            fh.write(rng.randbytes(n))
            remaining -= n
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    elapsed = max(time.perf_counter() - started, 1e-9)
    return (size_bytes / elapsed) / (1024 ** 3)


def sequential_read(path: Path, block_bytes: int, iterations: int,
                    invalidate_first: bool) -> dict:
    size = path.stat().st_size
    samples = []
    cold_note = ""
    for it in range(iterations):
        if it == 0 and invalidate_first:
            with open(path, "rb") as fh:
                try:
                    cold_note = drop_cache_hint(fh.fileno(), path)
                except OSError as exc:
                    cold_note = f"cache-drop unavailable ({exc})"
        started = time.perf_counter()
        with open(path, "rb") as fh:
            while fh.read(block_bytes):
                pass
        elapsed = max(time.perf_counter() - started, 1e-9)
        samples.append(size / elapsed)
    return {
        "block_bytes": block_bytes,
        "file_bytes": size,
        "iterations": iterations,
        "cold_note": cold_note or "no invalidation attempted",
        "throughput_bps": {
            "median": statistics.median(samples),
            "mean": statistics.fmean(samples),
            "min": min(samples),
            "max": max(samples),
            "samples": samples,
        },
        "cold_bps": samples[0],
        "warm_bps_median": statistics.median(samples[1:]) if len(samples) > 1 else None,
    }


def random_read(path: Path, block_bytes: int, ops: int, seed: int = 1234) -> dict:
    size = path.stat().st_size
    if block_bytes > size:
        return {"block_bytes": block_bytes, "status": "skipped",
                "reason": "block larger than test file"}
    rng = random.Random(seed)
    max_off = size - block_bytes
    offsets = [rng.randint(0, max_off) for _ in range(ops)]
    started = time.perf_counter()
    with open(path, "rb") as fh:
        for off in offsets:
            fh.seek(off)
            fh.read(block_bytes)
    elapsed = max(time.perf_counter() - started, 1e-9)
    total = ops * block_bytes
    return {
        "block_bytes": block_bytes,
        "ops": ops,
        "total_bytes": total,
        "elapsed_s": elapsed,
        "throughput_bps": total / elapsed,
        "iops": ops / elapsed,
        "status": "measured",
    }


def bench_storage(storage_path: Path, file_mib: int, blocks: list[int],
                  iterations: int, random_ops: int,
                  keep_file: bool = False) -> dict:
    storage_path.mkdir(parents=True, exist_ok=True)
    if not os.access(storage_path, os.R_OK | os.W_OK):
        raise PermissionError(f"storage path not readable/writable: {storage_path}")
    report: dict = {
        "tool": "bench_storage",
        "version": TOOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "storage_path": str(storage_path.resolve()),
        "filesystem": detect_filesystem(storage_path),
        "disk": disk_stats(storage_path),
    }
    size_bytes = file_mib * MIB
    disk_free = (report["disk"].get("free_bytes") or 0)
    if disk_free and size_bytes > disk_free:
        raise ValueError(f"test file {file_mib} MiB exceeds free disk {disk_free} B")
    tmp = storage_path / f".dee_storage_probe_{os.getpid()}.bin"
    try:
        report["write_gbps"] = write_test_file(tmp, size_bytes)
        report["test_file_bytes"] = size_bytes
        seq = []
        for blk in blocks:
            try:
                seq.append(sequential_read(tmp, blk, iterations, invalidate_first=True))
            except OSError as exc:
                seq.append({"block_bytes": blk, "status": "error", "reason": str(exc)})
        report["sequential_read"] = seq
        rnd = []
        for blk in blocks:
            try:
                rnd.append(random_read(tmp, blk, random_ops))
            except OSError as exc:
                rnd.append({"block_bytes": blk, "status": "error", "reason": str(exc)})
        report["random_read"] = rnd
    finally:
        if not keep_file:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                report["cleanup_warning"] = str(tmp)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--file-mib", type=int, default=128)
    parser.add_argument("--block-sizes", default="256KiB,1MiB,4MiB",
                        help="comma list, e.g. 256KiB,1MiB,4MiB")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--random-ops", type=int, default=256)
    parser.add_argument("--keep-file", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        blocks = parse_block_list(args.block_sizes)
        report = bench_storage(args.storage_path, args.file_mib, blocks,
                               args.iterations, args.random_ops, args.keep_file)
    except (ValueError, PermissionError, OSError) as exc:
        print(f"bench_storage: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    if args.json_only:
        print(text)
    else:
        print_human(report)
        print(f"\nJSON: {args.output}" if args.output else "\n(use --output to save JSON)")
    return 0


def print_human(report: dict) -> None:
    print(f"storage probe (MEASURED): {report['storage_path']}")
    print(f"  filesystem: {report['filesystem'].get('filesystem')} "
          f"[{report['filesystem'].get('method')}]")
    disk = report["disk"]
    if disk.get("status") == "measured":
        print(f"  disk total {disk['total_bytes'] / 1024**3:.1f} GiB, "
              f"free {disk['free_bytes'] / 1024**3:.1f} GiB")
    else:
        print(f"  disk: UNKNOWN ({disk.get('reason')})")
    print(f"  write: {report.get('write_gbps', float('nan')):.2f} GiB/s (test-file creation)")
    for entry in report.get("sequential_read", []):
        if "throughput_bps" not in entry:
            print(f"  seq {entry['block_bytes']}: {entry.get('status')}: {entry.get('reason')}")
            continue
        med = entry["throughput_bps"]["median"] / 1024 ** 3
        print(f"  seq block {entry['block_bytes'] // 1024} KiB: median {med:.2f} GiB/s "
              f"(cold {entry['cold_bps'] / 1024**3:.2f}, "
              f"warm {entry['warm_bps_median'] / 1024**3 if entry['warm_bps_median'] else float('nan'):.2f})")
    for entry in report.get("random_read", []):
        if entry.get("status") == "measured":
            print(f"  rand block {entry['block_bytes'] // 1024} KiB: "
                  f"{entry['throughput_bps'] / 1024**3:.2f} GiB/s, {entry['iops']:.0f} IOPS")
    seq0 = (report.get("sequential_read") or [{}])[0]
    if isinstance(seq0, dict) and seq0.get("cold_note"):
        print(f"  cache note: {seq0['cold_note']}")


if __name__ == "__main__":
    raise SystemExit(main())
