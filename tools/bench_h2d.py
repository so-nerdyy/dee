#!/usr/bin/env python3
"""Host<->GPU transfer benchmark (measures with torch.cuda; never fabricates).

Sweeps representative expert-sized transfers (default 1/4/8/13/16/32 MiB;
13 MiB ~ the 12.75 MiB packed MXFP4 expert record) and reports, per size:

- pageable host -> GPU bandwidth + latency,
- pinned host -> GPU bandwidth + latency,
- GPU -> host bandwidth + latency.

Without a CUDA GPU (or without torch) every transfer entry is reported as
UNKNOWN with the reason — the tool exits 0 and never invents numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.0.0"
MIB = 1024 * 1024
DEFAULT_SIZES_MIB = (1, 4, 8, 13, 16, 32)


def parse_size_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip().lower().replace(" ", "")
        if not part:
            continue
        if part.endswith("mib"):
            out.append(int(float(part[:-3]) * MIB))
        elif part.endswith("mb"):
            out.append(int(float(part[:-2]) * 1000 ** 2))
        elif part.endswith("kib"):
            out.append(int(float(part[:-3]) * 1024))
        else:
            out.append(int(float(part)))
    if not out:
        raise ValueError("empty size list")
    return out


def torch_cuda_info() -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "reason": f"torch not importable: {exc}"}
    try:
        if not torch.cuda.is_available():
            return {"available": False, "torch_version": torch.__version__,
                    "reason": "torch.cuda.is_available() is False"}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"cuda query failed: {exc}"}
    devices = []
    try:
        count = torch.cuda.device_count()
        for idx in range(count):
            props = torch.cuda.get_device_properties(idx)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            devices.append({
                "index": idx,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_bytes": props.total_memory,
                "free_vram_bytes": free_b,
                "cuda_runtime": torch.version.cuda,
                "multiprocessor_count": props.multi_processor_count,
            })
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "reason": f"device query failed: {exc}", "devices": []}
    return {"available": True, "torch_version": torch.__version__,
            "device_count": len(devices), "devices": devices}


def nvidia_smi_pcie() -> dict:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,"
             "pcie.link.gen.current,pcie.link.gen.max,"
             "pcie.link.width.current,pcie.link.width.max",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return {"status": "unknown", "reason": "nvidia-smi not found"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "unknown", "reason": str(exc)}
    if proc.returncode != 0:
        return {"status": "unknown", "reason": proc.stderr.strip()[:300]}
    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            rows.append({
                "index": parts[0], "name": parts[1], "driver": parts[2],
                "pcie_gen_current": parts[3], "pcie_gen_max": parts[4],
                "pcie_width_current": parts[5], "pcie_width_max": parts[6],
            })
    return {"status": "measured", "gpus": rows}


def _sync():
    import torch
    torch.cuda.synchronize()


def _timed_copy(src, dst, iterations: int, warmup: int) -> dict:
    import torch
    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
    _sync()
    samples_ms = []
    nbytes = src.nelement() * src.element_size()
    for _ in range(iterations):
        start = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        _sync()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    median_ms = statistics.median(samples_ms)
    gbps = (nbytes / 1e9) / (median_ms / 1000.0) if median_ms > 0 else 0.0
    return {"latency_ms_median": median_ms,
            "latency_ms_mean": statistics.fmean(samples_ms),
            "latency_ms_min": min(samples_ms),
            "latency_ms_max": max(samples_ms),
            "bandwidth_gbps": gbps,
            "iterations": iterations, "warmup": warmup,
            "status": "measured"}


def bench_all_sizes(sizes: list[int], iterations: int, warmup: int,
                    device_index: int = 0) -> dict:
    import torch
    device = torch.device(f"cuda:{device_index}")
    results: dict[str, list] = {"pageable_h2d": [], "pinned_h2d": [], "d2h": []}
    for nbytes in sizes:
        entry_base = {"size_bytes": nbytes, "size_mib": nbytes / MIB}
        # Pageable host -> device.
        try:
            host_page = torch.empty(nbytes, dtype=torch.uint8)
            dev = torch.empty(nbytes, dtype=torch.uint8, device=device)
            m = _timed_copy(host_page, dev, iterations, warmup)
            results["pageable_h2d"].append({**entry_base, **m})
        except Exception as exc:  # noqa: BLE001
            results["pageable_h2d"].append(
                {**entry_base, "status": "error", "reason": str(exc)[:300]})
        # Pinned host -> device.
        try:
            host_pin = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            dev2 = torch.empty(nbytes, dtype=torch.uint8, device=device)
            m = _timed_copy(host_pin, dev2, iterations, warmup)
            results["pinned_h2d"].append({**entry_base, **m})
            del host_pin, dev2
        except Exception as exc:  # noqa: BLE001
            results["pinned_h2d"].append(
                {**entry_base, "status": "error", "reason": str(exc)[:300]})
        # Device -> host (pinned sink).
        try:
            dev3 = torch.empty(nbytes, dtype=torch.uint8, device=device)
            host_sink = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            m = _timed_copy(dev3, host_sink, iterations, warmup)
            results["d2h"].append({**entry_base, **m})
            del dev3, host_sink
        except Exception as exc:  # noqa: BLE001
            results["d2h"].append(
                {**entry_base, "status": "error", "reason": str(exc)[:300]})
        finally:
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    return results


def unknown_results(sizes: list[int], reason: str) -> dict:
    return {key: [{"size_bytes": n, "size_mib": n / MIB, "status": "unknown",
                   "reason": reason} for n in sizes]
            for key in ("pageable_h2d", "pinned_h2d", "d2h")}


def run_bench(sizes_mib=None, iterations: int = 20, warmup: int = 5,
              device_index: int = 0) -> dict:
    sizes = [int(s * MIB) for s in (sizes_mib or DEFAULT_SIZES_MIB)]
    report: dict = {
        "tool": "bench_h2d",
        "version": TOOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sizes_bytes": sizes,
        "iterations": iterations,
        "warmup": warmup,
        "cuda": torch_cuda_info(),
        "pcie": nvidia_smi_pcie(),
    }
    if not report["cuda"].get("available"):
        reason = report["cuda"].get("reason", "CUDA unavailable")
        report["results"] = unknown_results(sizes, reason)
        report["note"] = ("No CUDA GPU accessible: transfer entries are UNKNOWN. "
                          "This is not a failure — run on the target host.")
        return report
    try:
        report["results"] = bench_all_sizes(sizes, iterations, warmup, device_index)
    except Exception as exc:  # noqa: BLE001
        report["results"] = unknown_results(sizes, f"benchmark failed: {exc}")
        report["note"] = str(exc)[:300]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES_MIB),
                        help="MiB list, e.g. 1,4,8,13,16,32")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        sizes = parse_size_list(args.sizes)
    except ValueError as exc:
        print(f"bench_h2d: {exc}", file=sys.stderr)
        return 2
    report = run_bench([s / MIB for s in sizes], args.iterations, args.warmup, args.device)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    if args.json_only:
        print(text)
    else:
        print_human(report)
    return 0


def print_human(report: dict) -> None:
    cuda = report["cuda"]
    if not cuda.get("available"):
        print(f"h2d probe: CUDA unavailable ({cuda.get('reason')}) — all entries UNKNOWN")
        return
    for dev in cuda.get("devices", []):
        print(f"h2d probe (MEASURED): {dev['name']} CC {dev['compute_capability']} "
              f"VRAM free {dev['free_vram_bytes'] / 1024**3:.1f}/"
              f"{dev['total_vram_bytes'] / 1024**3:.1f} GiB")
    for key, label in (("pageable_h2d", "pageable H2D"), ("pinned_h2d", "pinned H2D"),
                       ("d2h", "D2H")):
        print(f"  {label}:")
        for entry in report["results"][key]:
            if entry.get("status") != "measured":
                print(f"    {entry['size_mib']:g} MiB: {entry.get('status')}: {entry.get('reason')}")
            else:
                print(f"    {entry['size_mib']:g} MiB: {entry['latency_ms_median']:.3f} ms med, "
                      f"{entry['bandwidth_gbps']:.2f} GB/s")


if __name__ == "__main__":
    raise SystemExit(main())
