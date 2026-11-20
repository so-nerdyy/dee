import gc
import json
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.milestone25_memory import (
    MemoryProbe,
    collect_cuda_memory,
    collect_memory_snapshot,
    collect_nvml_memory,
    collect_page_faults,
    collect_proc_memory,
    collect_psutil_memory,
    inventory_cpu_tensors,
    json_safe,
    parse_proc_key_values,
    summarize_maps,
    summarize_smaps,
)


MAP_A = "1000-3000 r--p 00000000 08:01 1 /models/model-00001.safetensors"
MAP_B = "4000-5000 rw-p 00000000 00:00 0 [heap]"


class FakeStorage:
    def __init__(self, pointer, byte_count):
        self.pointer = pointer
        self.byte_count = byte_count

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.byte_count


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type

    def __str__(self):
        return self.type


class FakeTensor:
    def __init__(
        self, storage, shape, *, dtype="torch.float16", pinned=False, device="cpu"
    ):
        self._storage = storage
        self.shape = shape
        self.dtype = dtype
        self.device = FakeDevice(device)
        self._pinned = pinned
        self.requires_grad = False

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self):
        return 2

    def untyped_storage(self):
        return self._storage

    def is_pinned(self):
        return self._pinned

    def storage_offset(self):
        return 0


class FakeTorchForTensors:
    Tensor = FakeTensor


class MemoryParsingTests(unittest.TestCase):
    def test_proc_key_values_normalize_units(self):
        parsed = parse_proc_key_values("Rss: 12 kB\nThreads: 7\nName: python\n")
        self.assertEqual(parsed["Rss_bytes"], 12 * 1024)
        self.assertEqual(parsed["Threads"], 7)
        self.assertEqual(parsed["Name"], "python")

    def test_smaps_attribution_is_categorized_and_bounded(self):
        fixture = (
            f"{MAP_A}\n"
            "Size: 8 kB\nRss: 6 kB\nPss: 5 kB\nPrivate_Clean: 4 kB\nLocked: 2 kB\n"
            f"{MAP_B}\n"
            "Size: 4 kB\nRss: 4 kB\nPss: 4 kB\nAnonymous: 4 kB\nPrivate_Dirty: 4 kB\n"
        )
        summary = summarize_smaps(fixture.splitlines(True), max_regions=1)
        self.assertEqual(summary["region_count"], 2)
        self.assertEqual(summary["categories"]["checkpoint"]["rss_bytes"], 6 * 1024)
        self.assertEqual(
            summary["categories"]["anonymous"]["anonymous_bytes"], 4 * 1024
        )
        self.assertEqual(summary["categories"]["checkpoint"]["locked_bytes"], 2 * 1024)
        self.assertEqual(len(summary["top_regions_by_rss"]), 1)
        self.assertTrue(summary["top_regions_truncated"])
        json.dumps(summary, allow_nan=False)

    def test_maps_summary_attributes_virtual_bytes(self):
        summary = summarize_maps([MAP_A + "\n", MAP_B + "\n"], max_paths=1)
        self.assertEqual(summary["region_count"], 2)
        self.assertEqual(summary["categories"]["checkpoint"]["virtual_bytes"], 0x2000)
        self.assertEqual(summary["categories"]["anonymous"]["virtual_bytes"], 0x1000)
        self.assertEqual(len(summary["top_paths_by_virtual_bytes"]), 1)

    def test_proc_collection_reads_all_requested_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = root / "77"
            process.mkdir()
            (process / "smaps_rollup").write_text(
                "Rss: 20 kB\nPss: 18 kB\n", encoding="utf-8"
            )
            (process / "status").write_text(
                "VmRSS: 20 kB\nRssAnon: 4 kB\n", encoding="utf-8"
            )
            (process / "io").write_text(
                "read_bytes: 123\nwrite_bytes: 45\n", encoding="utf-8"
            )
            # fields 3 onward; minflt/cminflt/majflt/cmajflt are 10-13 overall.
            (process / "stat").write_text(
                "77 (worker name) S 1 1 1 0 0 0 11 12 13 14 0 0 0 0 0 0 1 0 0 0 0\n",
                encoding="utf-8",
            )
            (process / "maps").write_text(MAP_A + "\n" + MAP_B + "\n", encoding="utf-8")
            (process / "smaps").write_text(
                MAP_A
                + "\nRss: 6 kB\nPss: 5 kB\n"
                + MAP_B
                + "\nRss: 4 kB\nAnonymous: 4 kB\n",
                encoding="utf-8",
            )
            result = collect_proc_memory(
                77, proc_root=root, include_maps=True, include_smaps=True, max_regions=2
            )
            self.assertTrue(result["available"])
            self.assertEqual(result["smaps_rollup"]["Rss_bytes"], 20 * 1024)
            self.assertEqual(result["io"]["read_bytes"], 123)
            self.assertEqual(result["page_faults"]["minor_faults"], 11)
            self.assertEqual(result["maps"]["region_count"], 2)
            self.assertEqual(result["smaps_attribution"]["region_count"], 2)
            self.assertEqual(result["errors"], [])

    def test_proc_and_fault_collectors_degrade_without_proc(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            proc = collect_proc_memory(999, proc_root=missing)
            faults = collect_page_faults(999, proc_root=missing)
            self.assertFalse(proc["available"])
            self.assertFalse(faults["available"])
            json.dumps({"proc": proc, "faults": faults}, allow_nan=False)


class TensorInventoryTests(unittest.TestCase):
    def test_inventory_groups_owners_pinning_and_shared_storage(self):
        shared = FakeStorage(0x1234, 32)
        dense = FakeTensor(shared, (8,), pinned=True)
        alias = FakeTensor(shared, (8,), pinned=True)
        expert = FakeTensor(FakeStorage(0x5678, 16), (8,))
        report = inventory_cpu_tensors(
            {"dense": {"weight": dense}, "expert_cache": [alias, expert]},
            owner_metadata={
                "dense": {
                    "source_component": "transformers",
                    "purpose": "dense_weights",
                },
                "expert_cache": {
                    "source_component": "dee.cpp",
                    "purpose": "expert_cache",
                },
            },
            torch_module=FakeTorchForTensors,
        )
        self.assertTrue(report["available"])
        self.assertEqual(report["tensor_count"], 3)
        self.assertEqual(report["unique_storage_count"], 2)
        self.assertEqual(report["unique_storage_bytes"], 48)
        self.assertEqual(report["referenced_storage_bytes"], 80)
        self.assertEqual(report["alias_storage_overcount_bytes"], 32)
        self.assertEqual(report["pinned_logical_bytes"], 32)
        self.assertEqual(len(report["duplicate_storages"]), 1)
        self.assertEqual(
            report["duplicate_storages"][0]["owners"], ["dense", "expert_cache"]
        )
        self.assertEqual(
            report["owners"]["dense"]["metadata"]["source_component"], "transformers"
        )
        json.dumps(report, allow_nan=False)

    def test_inventory_does_not_retain_tensor_objects(self):
        tensor = FakeTensor(FakeStorage(0x99, 8), (4,))
        reference = weakref.ref(tensor)
        owners = {"temporary": tensor}
        report = inventory_cpu_tensors(owners, torch_module=FakeTorchForTensors)
        self.assertEqual(report["tensor_count"], 1)
        del owners
        del tensor
        gc.collect()
        self.assertIsNone(reference())

    def test_inventory_is_bounded(self):
        tensors = [
            FakeTensor(FakeStorage(0x100 + index, 2), (1,)) for index in range(10)
        ]
        report = inventory_cpu_tensors(
            {"many": tensors},
            torch_module=FakeTorchForTensors,
            max_tensors=4,
            max_tensor_details=2,
        )
        self.assertEqual(report["tensor_count"], 4)
        self.assertEqual(len(report["tensor_details"]), 2)
        self.assertTrue(report["tensor_details_truncated"])
        self.assertTrue(report["scan_truncated"])

    def test_missing_torch_is_a_supported_state(self):
        report = inventory_cpu_tensors({}, torch_module=None)
        self.assertFalse(report["available"])


class FakeCuda:
    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 2

    @staticmethod
    def memory_stats(device):
        return {
            "active_bytes.all.current": 200 + device,
            "inactive_split_bytes.all.current": 30 + device,
        }

    @staticmethod
    def get_device_properties(device):
        return SimpleNamespace(name=f"Fake T4 {device}", total_memory=16 * 1024**3)

    @staticmethod
    def memory_allocated(device):
        return 1000 + device

    @staticmethod
    def memory_reserved(device):
        return 2000 + device

    @staticmethod
    def max_memory_allocated(device):
        return 3000 + device

    @staticmethod
    def max_memory_reserved(device):
        return 4000 + device

    @staticmethod
    def mem_get_info(device):
        return (5000 + device, 6000 + device)

    @staticmethod
    def memory_snapshot():
        return [
            {
                "device": 0,
                "total_size": 100,
                "allocated_size": 60,
                "active_size": 70,
                "requested_size": 55,
                "blocks": [
                    {"state": "active_allocated", "size": 60, "requested_size": 55}
                ],
            }
        ]


class FakeTorchForCuda:
    __version__ = "test"
    version = SimpleNamespace(cuda="12.test")
    cuda = FakeCuda()


class FakeNvml:
    def __init__(self):
        self.initialized = False
        self.shutdown = False

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        self.shutdown = True

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=1000, used=400, free=600)

    def nvmlDeviceGetName(self, handle):
        return b"Tesla T4"

    def nvmlDeviceGetUUID(self, handle):
        return b"GPU-test"

    def nvmlDeviceGetPciInfo(self, handle):
        return SimpleNamespace(busId=b"00000000:00:1e.0")

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=75, memory=25)

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle):
        return [SimpleNamespace(pid=42, usedGpuMemory=123)]

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle):
        # Same process/table residency must not be counted twice.
        return [SimpleNamespace(pid=42, usedGpuMemory=123)]

    def nvmlSystemGetDriverVersion(self):
        return b"555.55"


class GpuAndTimelineTests(unittest.TestCase):
    def test_cuda_summary_has_allocated_reserved_peaks_and_snapshot(self):
        report = collect_cuda_memory(
            torch_module=FakeTorchForCuda, include_snapshot=True
        )
        self.assertTrue(report["available"])
        self.assertEqual(len(report["devices"]), 2)
        self.assertEqual(report["devices"][0]["allocated_bytes"], 1000)
        self.assertEqual(report["devices"][0]["reserved_bytes"], 2000)
        self.assertEqual(report["devices"][0]["peak_allocated_bytes"], 3000)
        snapshot = report["allocator_snapshot_summary"]["devices"]["0"]
        self.assertEqual(snapshot["total_size_bytes"], 100)
        self.assertTrue(report["allocator_snapshot_summary"]["raw_snapshot_omitted"])
        json.dumps(report, allow_nan=False)

    def test_cuda_and_nvml_gracefully_report_unavailable(self):
        no_cuda = SimpleNamespace(
            __version__="test",
            cuda=SimpleNamespace(
                is_initialized=lambda: True, is_available=lambda: False
            ),
        )
        cuda_report = collect_cuda_memory(torch_module=no_cuda)
        nvml_report = collect_nvml_memory(pynvml_module=None, nvidia_smi_path="")
        self.assertFalse(cuda_report["available"])
        self.assertFalse(nvml_report["available"])

    def test_cuda_probe_does_not_initialize_cuda_by_default(self):
        cuda = SimpleNamespace(
            is_initialized=lambda: False,
            is_available=lambda: self.fail("is_available would touch the CUDA driver"),
        )
        report = collect_cuda_memory(
            torch_module=SimpleNamespace(__version__="test", cuda=cuda)
        )
        self.assertFalse(report["available"])
        self.assertFalse(report["initialized"])
        self.assertIn("avoid perturbing", report["reason"])

    def test_nvml_summary_captures_driver_device_and_process_memory(self):
        module = FakeNvml()
        report = collect_nvml_memory(42, pynvml_module=module, nvidia_smi_path="")
        self.assertTrue(report["available"])
        self.assertEqual(report["method"], "pynvml")
        self.assertEqual(report["driver_version"], "555.55")
        self.assertEqual(report["devices"][0]["pci_bus_id"], "00000000:00:1e.0")
        self.assertEqual(report["devices"][0]["process_used_bytes"], 123)
        self.assertTrue(module.initialized)
        self.assertTrue(module.shutdown)

    def test_psutil_summary_includes_full_memory_info(self):
        Info = __import__("collections").namedtuple("Info", "rss vms")
        Full = __import__("collections").namedtuple("Full", "rss vms uss pss swap")

        class Process:
            def __init__(self, pid):
                self.pid = pid

            def memory_info(self):
                return Info(10, 20)

            def memory_full_info(self):
                return Full(10, 20, 7, 8, 1)

            def memory_percent(self):
                return 0.5

        report = collect_psutil_memory(
            42, psutil_module=SimpleNamespace(Process=Process)
        )
        self.assertTrue(report["available"])
        self.assertEqual(report["memory_full_info"]["uss"], 7)
        self.assertEqual(report["memory_full_info"]["pss"], 8)

    @mock.patch("scripts.milestone25_memory.collect_memory_snapshot")
    def test_probe_preserves_first_checkpoint_and_bounds_timeline(self, collect):
        counter = iter([0.0, 1.0, 1.1, 2.0, 2.2, 3.0, 3.3])
        collect.side_effect = lambda **kwargs: {
            "schema_version": 1,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "pid": kwargs["pid"],
        }
        probe = MemoryProbe(pid=42, max_checkpoints=2, clock=lambda: next(counter))
        probe.checkpoint("start", include_cuda=False, include_nvml=False)
        probe.checkpoint("middle", include_cuda=False, include_nvml=False)
        probe.checkpoint("end", include_cuda=False, include_nvml=False)
        report = probe.as_dict()
        self.assertEqual(
            [row["label"] for row in report["checkpoints"]], ["start", "end"]
        )
        self.assertEqual(report["dropped_checkpoint_count"], 1)
        self.assertAlmostEqual(report["checkpoints"][1]["collection_seconds"], 0.3)
        json.dumps(report, allow_nan=False)

    def test_real_local_snapshot_is_json_serializable(self):
        snapshot = collect_memory_snapshot(
            include_maps=False,
            include_smaps=False,
            include_cuda=True,
            include_nvml=False,
            metadata={"nonfinite": float("inf"), "payload": b"abc"},
        )
        json.dumps(snapshot, allow_nan=False)
        self.assertEqual(snapshot["metadata"]["nonfinite"], "inf")

    def test_json_safe_bounds_user_metadata(self):
        safe = json_safe(
            {"items": list(range(20)), "long": "x" * 100}, max_items=3, max_string=12
        )
        self.assertEqual(len(safe["items"]), 4)
        self.assertLessEqual(len(safe["long"]), 12)
        json.dumps(safe, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
