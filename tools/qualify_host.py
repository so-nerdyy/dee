#!/usr/bin/env python3
"""One-command host qualification for dee consumer-GPU migration.

Usage:
    python tools/qualify_host.py --storage-path <path> --output host.json

Measures the machine (system / GPU / storage / H2D transfer), derives memory
arithmetic for DSV4-Flash-0731 geometry, and answers:

- can dee reasonably run here?
- what fits in VRAM / RAM?
- what storage bandwidth is available?
- what H2D cost should the scheduler expect?
- which features are supported?

Output separates MEASURED / DERIVED / UNKNOWN. No throughput (tok/s) is
predicted: raw bandwidth alone cannot prove end-to-end decode speed, native
FP4 execution is NOT inferred from 4-bit weights, and no per-GPU-model
(5070 Ti / 4090 / 5090 / etc.) result is claimed without measuring that host.

Degrades gracefully on: no CUDA, 1..N GPUs, Windows, Linux. Never requires
the DeepSeek checkpoint merely to qualify a host.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.0.0"
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

GIB = 1024 ** 3
MIB = 1024 * 1024

CLAIM_LEGEND = {
    "MEASURED": "observed on this host by this run",
    "DERIVED": "arithmetic from measured inputs + stated assumptions (not measured)",
    "UNKNOWN": "not detectable with available APIs; never fabricated",
    "THEORETICAL_CEILING": "upper bound only; not a runtime expectation",
    "UNPROVEN": "requires a real model run; explicitly not claimed",
}

# ---------------------------------------------------------------------------
# System probe
# ---------------------------------------------------------------------------

def probe_cpu_model() -> dict:
    system = platform.system()
    if system == "Windows":
        try:
            import winreg
            with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if name:
                    return {"cpu_model": name.strip(), "method": "registry",
                            "status": "measured"}
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
        else:
            detail = "empty"
        fallback = (platform.processor() or platform.machine() or "").strip()
        if fallback:
            return {"cpu_model": fallback, "method": "platform.processor",
                    "status": "measured", "note": f"registry unavailable ({detail})"}
        return {"cpu_model": "UNKNOWN", "method": "registry+platform",
                "status": "unknown", "reason": detail}
    # Linux / other POSIX.
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            count = len(re.findall(r"^processor\s*:", text, re.MULTILINE))
            return {"cpu_model": match.group(1).strip(), "method": "/proc/cpuinfo",
                    "status": "measured", "logical_listed": count or None}
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
    else:
        detail = "model name not found"
    fallback = (platform.processor() or platform.machine() or "").strip()
    if fallback:
        return {"cpu_model": fallback, "method": "platform.processor",
                "status": "measured", "note": detail}
    return {"cpu_model": "UNKNOWN", "method": "cpuinfo+platform",
            "status": "unknown", "reason": detail}


def probe_cpu_counts() -> dict:
    out: dict = {"logical": os.cpu_count(), "logical_status": "measured"
                 if os.cpu_count() else "unknown"}
    try:
        import psutil  # type: ignore
        out["logical"] = psutil.cpu_count(logical=True)
        out["physical_cores"] = psutil.cpu_count(logical=False)
        out["method"] = "psutil"
        out["status"] = "measured"
        return out
    except ImportError:
        pass
    if platform.system() == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            phys = set(re.findall(r"^physical id\s*:\s*(\d+)", text, re.MULTILINE))
            cores = set(re.findall(r"^core id\s*:\s*(\d+)", text, re.MULTILINE))
            if phys and cores:
                out["physical_cores"] = len(phys) * (len(cores) // max(len(phys), 1))
                out["method"] = "/proc/cpuinfo"
                out["status"] = "measured"
                return out
        except Exception:  # noqa: BLE001
            pass
    out["physical_cores"] = "UNKNOWN"
    out["method"] = "os.cpu_count (+psutil if installed)"
    out["status"] = "measured" if out["logical"] else "unknown"
    return out


def probe_isa() -> dict:
    """Detect AVX2 / AVX512 / BF16 / AMX where the OS exposes flags."""
    flags: set[str] = set()
    method = "none"
    if Path("/proc/cpuinfo").is_file():
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            match = re.search(r"^flags\s*:\s*(.+)$", text, re.MULTILINE)
            if match:
                flags = set(match.group(1).split())
                method = "/proc/cpuinfo flags"
        except Exception:  # noqa: BLE001
            pass
    if not flags:
        try:
            import cpuinfo  # type: ignore  # py-cpuinfo, optional
            info = cpuinfo.get_cpu_info() or {}
            flags = set(info.get("flags", []))
            if flags:
                method = "py-cpuinfo"
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            pass
    if not flags:
        return {
            "avx2": "UNKNOWN", "avx512f": "UNKNOWN",
            "bf16": "UNKNOWN", "amx": "UNKNOWN",
            "method": method, "status": "unknown",
            "reason": ("ISA flags not exposed portably on this OS without "
                       "py-cpuinfo; install py-cpuinfo or check on Linux. "
                       "Nothing inferred."),
        }
    avx512 = any(f.startswith("avx512") for f in flags)
    bf16 = ("avx512_bf16" in flags or "avx512bf16" in flags or "bf16" in flags)
    amx = any(f.startswith("amx_") for f in flags)
    return {
        "avx2": "avx2" in flags, "avx512f": "avx512f" in flags,
        "avx512_any": avx512, "bf16": bf16, "amx": amx,
        "method": method, "status": "measured",
    }


def probe_ram() -> dict:
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        return {"total_bytes": int(mem.total), "available_bytes": int(mem.available),
                "method": "psutil", "status": "measured"}
    except ImportError:
        pass
    if platform.system() == "Windows":
        try:
            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MemStatus()
            stat.dwLength = ctypes.sizeof(MemStatus)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return {"total_bytes": int(stat.ullTotalPhys),
                        "available_bytes": int(stat.ullAvailPhys),
                        "method": "GlobalMemoryStatusEx", "status": "measured"}
        except Exception as exc:  # noqa: BLE001
            return {"total_bytes": "UNKNOWN", "available_bytes": "UNKNOWN",
                    "method": "GlobalMemoryStatusEx", "status": "unknown",
                    "reason": str(exc)}
    if Path("/proc/meminfo").is_file():
        try:
            kv: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                m = re.match(r"(\w+):\s+(\d+)", line)
                if m:
                    kv[m.group(1)] = int(m.group(2)) * 1024
            if "MemTotal" in kv:
                return {"total_bytes": kv["MemTotal"],
                        "available_bytes": kv.get("MemAvailable", "UNKNOWN"),
                        "method": "/proc/meminfo", "status": "measured"}
        except Exception as exc:  # noqa: BLE001
            return {"total_bytes": "UNKNOWN", "available_bytes": "UNKNOWN",
                    "method": "/proc/meminfo", "status": "unknown",
                    "reason": str(exc)}
    return {"total_bytes": "UNKNOWN", "available_bytes": "UNKNOWN",
            "method": "none", "status": "unknown",
            "reason": "install psutil for portable RAM accounting"}


def probe_system() -> dict:
    return {
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "machine": platform.machine(),
               "status": "measured"},
        "cpu": probe_cpu_model(),
        "counts": probe_cpu_counts(),
        "isa": probe_isa(),
        "ram": probe_ram(),
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation(),
                   "status": "measured"},
    }


# ---------------------------------------------------------------------------
# GPU probe
# ---------------------------------------------------------------------------

def cuda_feature_availability(compute_capability: str | None) -> dict:
    """Capability features DERIVED from compute capability, not measured exec."""
    if not compute_capability:
        return {"status": "unknown", "reason": "no compute capability detected"}
    try:
        major, minor = (int(x) for x in compute_capability.split(".")[:2])
    except ValueError:
        return {"status": "unknown", "reason": f"unparseable CC {compute_capability!r}"}
    cc = major + minor / 10.0
    return {
        "status": "derived",
        "compute_capability": compute_capability,
        "bf16_cuda": cc >= 8.0,
        "fp8_tensor_cores": cc >= 9.0,
        "fp4_tensor_cores": cc >= 10.0,
        "note": ("Derived from compute capability only. 4-bit weights do NOT "
                 "imply native FP4 execution; unpack/dequantize cost is real "
                 "and unmeasured by this probe."),
    }


def probe_gpu() -> dict:
    try:
        from bench_h2d import nvidia_smi_pcie, torch_cuda_info  # type: ignore
    except ImportError as exc:
        return {"status": "unknown", "reason": f"probe import failed: {exc}"}
    cuda = torch_cuda_info()
    pcie = nvidia_smi_pcie()
    # nvidia-smi driver fallback when torch is absent.
    driver = "UNKNOWN"
    if pcie.get("status") == "measured" and pcie.get("gpus"):
        driver = pcie["gpus"][0].get("driver", "UNKNOWN")
    out: dict = {"cuda": cuda, "pcie": pcie, "driver_version": driver}
    devices = cuda.get("devices", []) if cuda.get("available") else []
    out["count"] = len(devices) if cuda.get("available") else (
        len(pcie.get("gpus", [])) if pcie.get("status") == "measured" else 0)
    out["features_by_device"] = [
        {"index": d.get("index"), "name": d.get("name"),
         **cuda_feature_availability(d.get("compute_capability"))}
        for d in devices
    ]
    if not devices:
        out["features_by_device"] = []
        out["features_note"] = ("UNKNOWN: no CUDA device enumerated; feature "
                                "availability cannot be derived without a CC.")
    out["status"] = "measured" if cuda.get("available") or pcie.get("status") == "measured" \
        else "unknown"
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_storage_quick(storage_path: Path, file_mib: int, blocks: str,
                      iterations: int, random_ops: int) -> dict:
    from bench_storage import bench_storage, parse_block_list  # type: ignore
    return bench_storage(storage_path, file_mib, parse_block_list(blocks),
                         iterations, random_ops)


def run_h2d_quick(sizes_mib: list[float], iterations: int, warmup: int) -> dict:
    from bench_h2d import run_bench  # type: ignore
    return run_bench(sizes_mib, iterations, warmup)


def run_memory_budget(vram_total_bytes: int | None, reserved_bytes: int | None,
                      ram_cache_bytes: int | None) -> dict:
    from memory_budget import (DEFAULT_DENSE_RESERVED_BYTES,  # type: ignore
                               EXPERT_RECORD_BYTES, compute_budget)
    from memory_budget import DSV4_FLASH_0731 as GEO
    expert = EXPERT_RECORD_BYTES["mxfp4"]
    total_experts = GEO["n_routed_experts"] * GEO["moe_layers_default"]
    budget = compute_budget(
        vram_total_bytes=vram_total_bytes or 16 * GIB,
        reserved_bytes=(reserved_bytes if reserved_bytes is not None
                        else DEFAULT_DENSE_RESERVED_BYTES),
        expert_bytes=expert,
        ram_cache_bytes=(ram_cache_bytes if ram_cache_bytes is not None
                         else 24 * GIB),
        total_experts=total_experts,
        topk=GEO["num_experts_per_tok"],
        moe_layers=GEO["moe_layers_default"],
    )
    budget["expert_bytes_mxfp4"] = expert
    budget["vram_total_assumed"] = vram_total_bytes is None
    budget["geometry"] = GEO
    return budget


def build_cost_model(transfer: dict | None, storage: dict | None,
                     system: dict, gpu: dict) -> dict:
    """Phase F export: consumable later by dee's scheduler / KT q* cost model.

    Not integrated into production scheduling. dee kernel execution is NOT
    benchmarked here (gpu_execution: null with reason).
    """
    t_h2d_ms: dict[str, float | str] = {}
    h2d_source = "UNKNOWN"
    if transfer and transfer.get("results", {}).get("pinned_h2d"):
        measured = [e for e in transfer["results"]["pinned_h2d"]
                    if e.get("status") == "measured"]
        if measured:
            t_h2d_ms = {str(e["size_bytes"]): e["latency_ms_median"] for e in measured}
            h2d_source = "measured pinned_h2d median"
        else:
            t_h2d_ms = {"note": "UNKNOWN: no measured pinned_h2d entries"}
    ssd: dict = {"status": "UNKNOWN"}
    if storage:
        seq = storage.get("sequential_read", [])
        med = {str(e["block_bytes"]): e["throughput_bps"]["median"]
               for e in seq if "throughput_bps" in e}
        if med:
            ssd = {"status": "measured", "seq_read_bps_by_block": med,
                   "filesystem": storage.get("filesystem", {}).get("filesystem"),
                   "cold_note": (seq[0].get("cold_note") if seq else None)}
    return {
        "t_h2d_ms_pinned_by_size_bytes": t_h2d_ms,
        "t_h2d_source": h2d_source,
        "ssd": ssd,
        "cpu": {"model": system.get("cpu", {}).get("cpu_model"),
                "counts": system.get("counts", {}),
                "isa": system.get("isa", {})},
        "gpu_execution": None,
        "gpu_execution_note": ("dee kernels not benchmarked by this harness; "
                               "integrate only after a correctness-gated kernel "
                               "benchmark exists. Not wired into scheduling."),
        "integration_status": "EXPORT_ONLY_NOT_INTEGRATED",
    }


def build_verdicts(system: dict, gpu: dict, storage: dict | None,
                   transfer: dict | None, budget: dict,
                   checkpoint_bytes: int = 166878536440) -> tuple[dict, list[str]]:
    unknown: list[str] = []
    cuda_avail = bool(gpu.get("cuda", {}).get("available"))
    if not cuda_avail:
        unknown.append("GPU transfer/feature verdicts: no CUDA device enumerated")
    devices = gpu.get("cuda", {}).get("devices", []) if cuda_avail else []
    vram_bytes = None
    if devices:
        # Conservative: min free across visible devices.
        vram_bytes = min(d.get("free_vram_bytes", 0) for d in devices)
    else:
        unknown.append("VRAM totals: no CUDA device; budget uses assumed 16 GiB target")

    ram = system.get("ram", {})
    ram_avail = ram.get("available_bytes")
    if not isinstance(ram_avail, int):
        unknown.append("RAM fit: available RAM UNKNOWN")

    disk_free = None
    fits_checkpoint = "UNKNOWN"
    if storage and isinstance(storage.get("disk", {}).get("free_bytes"), int):
        disk_free = storage["disk"]["free_bytes"]
        fits_checkpoint = bool(disk_free >= checkpoint_bytes)
    else:
        unknown.append("checkpoint fit: free disk UNKNOWN (storage probe skipped/failed)")

    h2d_expected = "UNKNOWN"
    if transfer and transfer.get("results", {}).get("pinned_h2d"):
        measured = [e for e in transfer["results"]["pinned_h2d"]
                    if e.get("status") == "measured"]
        if measured:
            by_size = {e["size_bytes"]: e for e in measured}
            ref = by_size.get(13 * MIB) or by_size.get(12 * MIB) or measured[len(measured) // 2]
            h2d_expected = (f"measured pinned-H2D ~{ref['latency_ms_median']:.3f} ms / "
                            f"{ref['bandwidth_gbps']:.2f} GB/s at {ref['size_mib']:g} MiB "
                            f"(this host, this run; scheduler input only, not a TPS claim)")
        else:
            unknown.append("H2D cost: transfer entries UNKNOWN (no CUDA or benchmark failed)")
    else:
        unknown.append("H2D cost: transfer probe skipped/failed")

    verdicts = {
        "can_run": {
            "cuda_present": cuda_avail,
            "assessment": ("DERIVED: host is measurable for dee planning; "
                           "actual execution requires a correctness-gated model run. "
                           "No TPS claimed.") if cuda_avail else
                          ("DERIVED: no CUDA GPU here — host can stage/measure CPU+storage, "
                           "but GPU-gated dee runs need the target machine."),
        },
        "fits_in_vram_mxfp4": {
            "vram_slots_assumed_budget": budget["vram_slots"],
            "vram_slots_this_host_free": ("UNKNOWN (no CUDA)" if vram_bytes is None
                                          else int(vram_bytes // budget["expert_bytes_mxfp4"])),
            "coverage_note": (f"DERIVED: {budget['vram_slots']} slots per 16 GiB-style budget "
                              f"vs {budget['total_experts']} routed experts "
                              f"({100.0 * budget['cache_coverage_fraction']:.1f}% cached with RAM)."),
        },
        "fits_in_ram_mxfp4": {
            "ram_slots_assumed_budget": budget["ram_slots"],
            "note": "DERIVED from configured RAM cache budget, not a residency measurement.",
        },
        "storage_bandwidth": ("DERIVED from storage probe; see measured.storage"
                              if storage else "UNKNOWN (storage probe skipped)"),
        "h2d_cost_scheduler_input": h2d_expected,
        "features_supported": [
            {"device": f.get("index"), "name": f.get("name"),
             "bf16_cuda": f.get("bf16_cuda"), "fp8": f.get("fp8_tensor_cores"),
             "fp4": f.get("fp4_tensor_cores"),
             "note": "DERIVED from CC; weights-being-4-bit does not imply native FP4 exec."}
            for f in gpu.get("features_by_device", [])
        ] or "UNKNOWN (no CUDA device)",
        "no_performance_prediction": ("UNPROVEN: end-to-end tok/s, TTFT, and any "
                                      "5070 Ti / 4090 / 5090 expectation require a real, "
                                      "correctness-gated run on that host. Not claimed here."),
    }
    return verdicts, unknown


def qualify(storage_path: Path, run_storage: bool, run_h2d: bool,
            file_mib: int, blocks: str, storage_iters: int, random_ops: int,
            h2d_sizes: list[float], h2d_iters: int, h2d_warmup: int,
            vram_override: int | None, reserved_override: int | None,
            ram_cache_override: int | None) -> dict:
    started = time.perf_counter()
    system = probe_system()
    gpu = probe_gpu()

    storage = None
    storage_error = None
    if run_storage:
        try:
            storage = run_storage_quick(storage_path, file_mib, blocks,
                                        storage_iters, random_ops)
        except Exception as exc:  # noqa: BLE001
            storage_error = str(exc)

    transfer = None
    if run_h2d:
        try:
            transfer = run_h2d_quick(h2d_sizes, h2d_iters, h2d_warmup)
        except Exception as exc:  # noqa: BLE001
            transfer = {"tool": "bench_h2d", "status": "error", "reason": str(exc),
                        "results": {}}

    # VRAM default: measured min-total if exactly one GPU, else 16 GiB target.
    vram_total = vram_override
    if vram_total is None:
        devices = gpu.get("cuda", {}).get("devices", []) if gpu.get("cuda", {}).get("available") else []
        if len(devices) == 1 and isinstance(devices[0].get("total_vram_bytes"), int):
            vram_total = int(devices[0]["total_vram_bytes"])
        else:
            vram_total = 16 * GIB
    budget = run_memory_budget(vram_total, reserved_override, ram_cache_override)
    verdicts, unknown = build_verdicts(system, gpu, storage, transfer, budget)
    if storage_error:
        unknown.append(f"storage probe failed: {storage_error}")
    isa = system.get("isa", {})
    if isa.get("status") == "unknown":
        unknown.append(f"CPU ISA flags: {isa.get('reason')}")
    if system.get("ram", {}).get("status") == "unknown":
        unknown.append(f"RAM: {system['ram'].get('reason')}")
    if gpu.get("pcie", {}).get("status") == "unknown":
        unknown.append(f"PCIe link: {gpu['pcie'].get('reason')}")
    for key in ("measured",):
        _ = key

    cost_model = build_cost_model(transfer, storage, system, gpu)
    elapsed = time.perf_counter() - started
    return {
        "tool": "qualify_host",
        "version": TOOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "storage_path": str(storage_path.resolve()) if storage_path.exists() else str(storage_path),
        "claim_tiers": CLAIM_LEGEND,
        "measured": {"system": system, "gpu": gpu, "storage": storage,
                     "transfer": transfer},
        "derived": {"memory_budget_mxfp4": budget, "verdicts": verdicts},
        "unknown": unknown,
        "cost_model": cost_model,
    }


def print_human(report: dict) -> None:
    system = report["measured"]["system"]
    gpu = report["measured"]["gpu"]
    print("=" * 68)
    print("dee host qualification (do NOT read as a performance claim)")
    print("=" * 68)
    print("[MEASURED — system]")
    print(f"  OS: {system['os']['system']} {system['os']['release']} ({system['os']['machine']})")
    print(f"  CPU: {system['cpu'].get('cpu_model')} [{system['cpu'].get('method')}]")
    counts = system["counts"]
    print(f"  cores/threads: physical={counts.get('physical_cores')} logical={counts.get('logical')}")
    isa = system["isa"]
    if isa.get("status") == "measured":
        print(f"  ISA: AVX2={isa.get('avx2')} AVX512F={isa.get('avx512f')} "
              f"BF16={isa.get('bf16')} AMX={isa.get('amx')} [{isa.get('method')}]")
    else:
        print(f"  ISA: UNKNOWN ({isa.get('reason')})")
    ram = system["ram"]
    if ram.get("status") == "measured" and isinstance(ram.get("total_bytes"), int):
        avail = ram.get("available_bytes")
        avail_s = f"{avail / GIB:.1f} GiB" if isinstance(avail, int) else "UNKNOWN"
        print(f"  RAM: total {ram['total_bytes'] / GIB:.1f} GiB, available {avail_s}")
    else:
        print(f"  RAM: UNKNOWN ({ram.get('reason')})")
    print("[MEASURED — gpu]")
    cuda = gpu.get("cuda", {})
    if cuda.get("available"):
        for d in cuda["devices"]:
            print(f"  GPU{d['index']}: {d['name']} CC {d['compute_capability']} "
                  f"free {d['free_vram_bytes'] / GIB:.1f}/{d['total_vram_bytes'] / GIB:.1f} GiB "
                  f"(CUDA {d.get('cuda_runtime')})")
        pcie = gpu.get("pcie", {})
        if pcie.get("status") == "measured":
            for row in pcie["gpus"]:
                print(f"  PCIe GPU{row['index']}: gen {row['pcie_gen_current']}/{row['pcie_gen_max']} "
                      f"x{row['pcie_width_current']}/{row['pcie_width_max']} driver {row['driver']}")
        else:
            print(f"  PCIe: UNKNOWN ({pcie.get('reason')})")
    else:
        print(f"  no CUDA GPU ({cuda.get('reason')}); driver {gpu.get('driver_version')}")
    storage = report["measured"]["storage"]
    print("[MEASURED — storage]")
    if storage:
        seq = storage.get("sequential_read", [])
        for e in seq[:4]:
            if "throughput_bps" in e:
                print(f"  seq {e['block_bytes'] // 1024} KiB: "
                      f"{e['throughput_bps']['median'] / 1024**3:.2f} GiB/s med")
        rnd = storage.get("random_read", [])
        for e in rnd[:4]:
            if e.get("status") == "measured":
                print(f"  rand {e['block_bytes'] // 1024} KiB: "
                      f"{e['throughput_bps'] / 1024**3:.2f} GiB/s, {e['iops']:.0f} IOPS")
    else:
        print("  storage probe skipped/failed (see UNKNOWN)")
    transfer = report["measured"]["transfer"]
    print("[MEASURED — transfer]")
    if transfer and transfer.get("results", {}).get("pinned_h2d"):
        for e in transfer["results"]["pinned_h2d"]:
            if e.get("status") == "measured":
                print(f"  pinned H2D {e['size_mib']:g} MiB: {e['latency_ms_median']:.3f} ms, "
                      f"{e['bandwidth_gbps']:.2f} GB/s")
            else:
                print(f"  pinned H2D {e['size_mib']:g} MiB: {e.get('status')}: {e.get('reason')}")
                break
    else:
        print("  transfer probe skipped/failed (see UNKNOWN)")
    print("[DERIVED — verdicts]")
    verdicts = report["derived"]["verdicts"]
    print(f"  can_run: {verdicts['can_run']['assessment']}")
    print(f"  VRAM: {verdicts['fits_in_vram_mxfp4']['coverage_note']}")
    print(f"  H2D: {verdicts['h2d_cost_scheduler_input']}")
    print(f"  perf: {verdicts['no_performance_prediction']}")
    print("[UNKNOWN]")
    if report["unknown"]:
        for item in report["unknown"]:
            print(f"  - {item}")
    else:
        print("  (none — every probe returned a measurement)")
    print(f"[cost_model] export-only ({report['cost_model']['integration_status']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-storage", action="store_true")
    parser.add_argument("--skip-h2d", action="store_true")
    parser.add_argument("--storage-file-mib", type=int, default=64)
    parser.add_argument("--storage-blocks", default="256KiB,1MiB,4MiB")
    parser.add_argument("--storage-iters", type=int, default=2)
    parser.add_argument("--storage-random-ops", type=int, default=64)
    parser.add_argument("--h2d-sizes", default="1,4,8,13,16,32")
    parser.add_argument("--h2d-iters", type=int, default=10)
    parser.add_argument("--h2d-warmup", type=int, default=3)
    parser.add_argument("--vram-total", default=None, help="e.g. 16GiB")
    parser.add_argument("--reserved", default=None, help="e.g. 8.84GiB")
    parser.add_argument("--ram-cache", default=None, help="e.g. 24GiB")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    from memory_budget import parse_bytes as parse_mem_bytes  # type: ignore

    def opt_bytes(text):
        return parse_mem_bytes(text) if text is not None else None

    try:
        h2d_sizes = [float(s.strip()) for s in args.h2d_sizes.split(",") if s.strip()]
    except ValueError:
        print("qualify_host: unparseable --h2d-sizes", file=sys.stderr)
        return 2
    try:
        report = qualify(
            args.storage_path, not args.skip_storage, not args.skip_h2d,
            args.storage_file_mib, args.storage_blocks, args.storage_iters,
            args.storage_random_ops, h2d_sizes, args.h2d_iters, args.h2d_warmup,
            opt_bytes(args.vram_total), opt_bytes(args.reserved),
            opt_bytes(args.ram_cache))
    except Exception as exc:  # noqa: BLE001
        print(f"qualify_host: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.json_only:
        print(text)
    else:
        print_human(report)
        if args.output is not None:
            print(f"\nJSON artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
