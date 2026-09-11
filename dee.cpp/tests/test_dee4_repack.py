"""Byte-exact tests for the DEE4 v2 serving repacker."""

from __future__ import annotations

import importlib.util
import hashlib
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

    def _write_trace(self, path: Path) -> tuple[str, str]:
        selections = ((2, 0), (1, 2), (0, 1), (0, 2), (2, 1), (1, 0))
        previous = dee4.TRACE_GENESIS_SHA256
        lines = []
        for record_index, selected in enumerate(selections):
            step, layer = divmod(record_index, 3)
            payload = {
                "canonical_order": dee4.TRACE_CANONICAL_ORDER,
                "forward_step": step,
                "layer": layer,
                "previous_chain_sha256": previous,
                "record_index": record_index,
                "run_id": "synthetic-v50",
                "schema_version": 1,
                "token_rows": 1,
                "topk": 2,
                ("expert_ids_rank_order" if record_index % 2 == 0
                 else "selected_experts"): [list(selected)],
            }
            chain = hashlib.sha256(dee4._canonical_json_bytes(payload)).hexdigest()
            lines.append(dee4._canonical_json_bytes({
                **payload, "chain_sha256": chain
            }) + b"\n")
            previous = chain
        raw = b"".join(lines)
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest(), previous

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
        self.assertEqual(
            [(component["projection"], component["kind"])
             for component in metadata["components"]],
            list(dee4.COMPONENTS),
        )
        self.assertEqual(
            [component["offset"] for component in metadata["components"]],
            [0, 4, 8, 12, 14, 16],
        )
        integrity = [
            json.loads(line)
            for line in (self.output / "integrity.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        self.assertEqual(len(integrity), 9)
        self.assertEqual(
            [row["record_index"] for row in integrity], list(range(9))
        )
        self.assertTrue(
            all(len(row["record_sha256"]) == 64 for row in integrity)
        )
        self.assertTrue(
            all(
                set(row["component_sha256"]) == {
                    f"{projection}.{kind}" for projection, kind in dee4.COMPONENTS
                }
                for row in integrity
            )
        )

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

    def test_dry_run_validates_layout_without_creating_artifacts(self) -> None:
        report = dee4.repack(
            self.source,
            self.output,
            start_layer=0,
            end_layer=3,
            experts_per_layer=3,
            dry_run=True,
        )
        self.assertTrue(report["success"])
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["total_experts_repacked"], 0)
        self.assertFalse(self.output.exists())

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

    def test_trace_repack_is_sparse_sorted_byte_exact_and_linked(self) -> None:
        trace_path = self.base / "routed_experts.jsonl"
        journal_sha256, final_chain = self._write_trace(trace_path)
        report = dee4.repack_trace(
            self.source,
            self.output,
            trace_path,
            n_layers=3,
            experts_per_layer=3,
            expected_journal_sha256=journal_sha256,
            expected_final_chain_sha256=final_chain,
        )
        self.assertTrue(report["success"])
        metadata = json.loads((self.output / "metadata.json").read_text("utf-8"))
        expected_pairs = [(0, 0), (0, 2), (1, 1), (1, 2), (2, 0), (2, 1)]
        self.assertEqual(metadata["format"], "dee4-v3-trace")
        self.assertEqual(metadata["trace_journal_sha256"], journal_sha256)
        self.assertEqual(metadata["trace_final_chain_sha256"], final_chain)
        self.assertEqual(metadata["total_experts"], len(expected_pairs))
        self.assertEqual(
            [(row["layer"], row["expert"], row["record_index"])
             for row in metadata["records"]],
            [(layer, expert, index)
             for index, (layer, expert) in enumerate(expected_pairs)],
        )
        expected_data = b"".join(
            self.canonical[dee4._tensor_name(layer, expert, projection, kind)]
            for layer, expert in expected_pairs
            for projection, kind in dee4.COMPONENTS
        )
        self.assertEqual((self.output / "experts.dee4").read_bytes(), expected_data)
        validation = dee4.validate_dee4_trace_store(
            self.output, trace_path, n_layers=3, experts_per_layer=3,
            expected_journal_sha256=journal_sha256,
            expected_final_chain_sha256=final_chain,
        )
        self.assertTrue(validation["success"])
        self.assertTrue(validation["record_indices_contiguous"])
        canonical = dee4.validate_dee4_against_safetensors(
            self.source, self.output, samples=expected_pairs)
        self.assertTrue(canonical["success"])

    def test_trace_parser_rejects_noncanonical_or_broken_records(self) -> None:
        trace_path = self.base / "routed_experts.jsonl"
        _, _ = self._write_trace(trace_path)
        rows = trace_path.read_text("utf-8").splitlines()

        missing = self.base / "missing.jsonl"
        missing.write_bytes(
            ("\n".join(rows[:1] + rows[2:]) + "\n").encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "record_index"):
            dee4.load_trace_selection(
                missing, n_layers=3, experts_per_layer=3)

        duplicate = json.loads(rows[0])
        duplicate["selected_experts"] = [[0, 2]]
        payload = dict(duplicate)
        payload.pop("chain_sha256")
        duplicate["chain_sha256"] = hashlib.sha256(
            dee4._canonical_json_bytes(payload)).hexdigest()
        ambiguous = self.base / "ambiguous.jsonl"
        ambiguous.write_bytes(
            dee4._canonical_json_bytes(duplicate) + b"\n"
            + b"\n".join(line.encode("utf-8") for line in rows[1:]) + b"\n")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            dee4.load_trace_selection(
                ambiguous, n_layers=3, experts_per_layer=3)

        broken = json.loads(rows[1])
        broken["previous_chain_sha256"] = "f" * 64
        rows[1] = dee4._canonical_json_bytes(broken).decode("utf-8")
        bad_chain = self.base / "bad-chain.jsonl"
        bad_chain.write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "predecessor chain"):
            dee4.load_trace_selection(
                bad_chain, n_layers=3, experts_per_layer=3)


if __name__ == "__main__":
    unittest.main()
