"""Byte-exact tests for the DEE4 v2 serving repacker."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "kaggle/deepseek-v4-flash-0731/repack_to_dee4.py"
SPEC = importlib.util.spec_from_file_location("repack_to_dee4", MODULE_PATH)
assert SPEC and SPEC.loader
dee4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dee4)


def _write_safetensors(
    path: Path, tensors: list[tuple[str, str, list[int], bytes]]
) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        handle.write(payload)


class Dee4RepackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.output = self.base / "dee4"
        self.source.mkdir()
        self.canonical: dict[str, bytes] = {}
        shards: list[list[tuple[str, str, list[int], bytes]]] = [[], [], []]
        weight_map: dict[str, str] = {}
        shapes = {
            ("w1", "weight"): ("I8", [2, 2]),
            ("w3", "weight"): ("I8", [2, 2]),
            ("w2", "weight"): ("I8", [4, 1]),
            ("w1", "scale"): ("F8_E8M0", [2, 1]),
            ("w3", "scale"): ("F8_E8M0", [2, 1]),
            ("w2", "scale"): ("F8_E8M0", [4, 1]),
        }
        for layer in range(3):
            for expert in range(3):
                for component_index, (projection, kind) in enumerate(dee4.COMPONENTS):
                    dtype, shape = shapes[(projection, kind)]
                    nbytes = shape[0] * shape[1]
                    base = layer * 67 + expert * 19 + component_index * 7
                    raw = bytes((base + index) & 0xFF for index in range(nbytes))
                    name = dee4._tensor_name(layer, expert, projection, kind)
                    shard_index = (layer + expert + component_index) % len(shards)
                    shard_name = f"model-{shard_index + 1:05d}-of-00003.safetensors"
                    shards[shard_index].append((name, dtype, shape, raw))
                    weight_map[name] = shard_name
                    self.canonical[name] = raw
        for index, tensors in enumerate(shards, 1):
            _write_safetensors(
                self.source / f"model-{index:05d}-of-00003.safetensors",
                tensors,
            )
        (self.source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), "utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_repack_is_fixed_stride_and_byte_exact(self) -> None:
        report = dee4.repack(
            self.source,
            self.output,
            start_layer=0,
            end_layer=3,
            experts_per_layer=3,
        )
        self.assertTrue(report["success"])
        metadata = json.loads((self.output / "metadata.json").read_text("utf-8"))
        self.assertEqual(metadata["format"], "dee4-v2")
        self.assertEqual(metadata["record_bytes"], 20)
        self.assertEqual(metadata["total_experts"], 9)
        self.assertEqual((self.output / "experts.dee4").stat().st_size, 180)

        record_index = 1 * 3 + 2
        with (self.output / "experts.dee4").open("rb") as handle:
            handle.seek(record_index * metadata["record_bytes"])
            record = handle.read(metadata["record_bytes"])
        expected = b"".join(
            self.canonical[dee4._tensor_name(1, 2, projection, kind)]
            for projection, kind in dee4.COMPONENTS
        )
        self.assertEqual(record, expected)

        samples = [(layer, expert) for layer in range(3) for expert in range(3)]
        validation = dee4.validate_dee4_against_safetensors(
            self.source, self.output, samples=samples
        )
        self.assertTrue(validation["success"])
        self.assertEqual(validation["sample_count"], 9)
        self.assertEqual(len(validation["source_shards_covered"]), 3)
        self.assertTrue(
            all(
                component["exact_match"]
                for record_result in validation["records"]
                for component in record_result["components"]
            )
        )

        benchmark = dee4.benchmark_dee4_read(self.output, n_experts=3)
        self.assertEqual(benchmark["n_experts"], 3)
        self.assertEqual(benchmark["record_bytes"], 20)

        serving = dee4.benchmark_dee4_serving_access(
            self.output, groups=2, topk=2, queue_depths=(2,)
        )
        self.assertEqual(serving["schema"], "dee4-v2-serving-access-benchmark")
        self.assertEqual(serving["request_count"], 12)
        self.assertEqual(serving["bytes_requested_per_sweep"], 240)
        self.assertEqual(serving["record_bytes"], 20)
        self.assertTrue(serving["modes"])
        self.assertIsNotNone(serving["winner"])
        for mode in serving["modes"]:
            self.assertEqual(mode["bytes_requested"], mode["bytes_returned"])
            self.assertEqual(mode["requests"], 12)
            self.assertEqual(len(mode["checksum_accumulator"].to_bytes(8, "little")), 8)

    def test_validation_detects_first_corrupt_component(self) -> None:
        dee4.repack(
            self.source,
            self.output,
            start_layer=0,
            end_layer=3,
            experts_per_layer=3,
        )
        with (self.output / "experts.dee4").open("r+b") as handle:
            handle.seek(20 * 4 + 1)
            original = handle.read(1)
            handle.seek(20 * 4 + 1)
            handle.write(bytes([original[0] ^ 0xFF]))
        validation = dee4.validate_dee4_against_safetensors(
            self.source, self.output, samples=[(1, 1)]
        )
        self.assertFalse(validation["success"])
        self.assertFalse(validation["records"][0]["components"][0]["exact_match"])

    def test_nonempty_output_is_not_overwritten(self) -> None:
        self.output.mkdir()
        (self.output / "prior-evidence.json").write_text("{}", "utf-8")
        with self.assertRaises(FileExistsError):
            dee4.repack(
                self.source,
                self.output,
                start_layer=0,
                end_layer=1,
                experts_per_layer=1,
            )
        self.assertEqual(
            (self.output / "prior-evidence.json").read_text("utf-8"), "{}"
        )


if __name__ == "__main__":
    unittest.main()
