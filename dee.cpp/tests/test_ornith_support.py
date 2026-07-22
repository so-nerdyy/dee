import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.ornith_support import (
    advance_decode_state,
    build_complete_tensor_map,
    causal_position_ids,
    layer_device,
    read_checkpoint_index,
    shard_paths_for_layer,
    target_device_for_tensor,
    tiny_greedy_decode,
    validate_expert_cache_budget,
)


def write_fixture(root: Path):
    names = {
        "model.language_model.layers.0.mlp.gate.weight": ([2, 2], b"\x00\x00" * 4),
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": ([1, 2], b"\x00\x00" * 2),
    }
    data = bytearray()
    header = {}
    for name, (shape, payload) in names.items():
        start = len(data)
        data += payload
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [start, len(data)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    index = {
        "metadata": {"total_size": len(data)},
        "weight_map": {name: shard.name for name in names},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return index


class OrnithSupportTests(unittest.TestCase):
    def test_checkpoint_map_and_index_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = write_fixture(root)
            rows, summary = build_complete_tensor_map(root)
            self.assertEqual(summary["tensor_count"], 2)
            self.assertEqual(summary["validated_tensor_bytes"], 12)
            self.assertEqual({row["dtype"] for row in rows}, {"BF16"})
            self.assertEqual(shard_paths_for_layer(index, 0),
                             ["model-00001-of-00001.safetensors"])

    def test_missing_and_malformed_checkpoint_fail_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                read_checkpoint_index(root)
            (root / "model.safetensors.index.json").write_text("not json")
            with self.assertRaises(ValueError):
                read_checkpoint_index(root)

    def test_explicit_single_and_dual_gpu_placement(self):
        self.assertEqual([layer_device(layer, 2) for layer in range(40)],
                         [0] * 20 + [1] * 20)
        self.assertEqual([layer_device(layer, 1) for layer in range(40)], [0] * 40)
        self.assertEqual(target_device_for_tensor("model.language_model.embed_tokens.weight", 2),
                         "cuda:0")
        self.assertEqual(target_device_for_tensor("lm_head.weight", 2), "cuda:1")

    def test_memory_budget_rejection_and_capacity(self):
        expert_bytes = 3 * 2048 * 512 * 2
        with self.assertRaises(ValueError):
            validate_expert_cache_budget(expert_bytes - 1, 2048, 512, 8)
        with self.assertRaises(ValueError):
            validate_expert_cache_budget(7 * expert_bytes, 2048, 512, 8)
        self.assertEqual(validate_expert_cache_budget(8 * expert_bytes, 2048, 512, 8), 8)

    def test_causal_positions_and_recurrent_decode_state(self):
        self.assertEqual(causal_position_ids(0, 3), [0, 1, 2])
        self.assertEqual(causal_position_ids(3, 1), [3])
        state = {"all_ids": [10, 11, 12], "attention_length": 3,
                 "cache_seen_tokens": 3}
        state = advance_decode_state(state, 13)
        self.assertEqual(state, {"all_ids": [10, 11, 12, 13],
                                 "attention_length": 4, "cache_seen_tokens": 4})

    def test_tiny_end_to_end_greedy_eos_and_max_stop(self):
        def step(sequence):
            return [0.0, 2.0, 1.0] if len(sequence) < 3 else [0.0, 1.0, 3.0]

        eos = tiny_greedy_decode([7], step, 8, {2})
        self.assertEqual(eos["generated_ids"], [1, 1, 2])
        self.assertEqual(eos["stop_reason"], "eos")
        capped = tiny_greedy_decode([7], lambda _: [0.0, 1.0], 2, {99})
        self.assertEqual(capped["generated_ids"], [1, 1])
        self.assertEqual(capped["stop_reason"], "max_new_tokens")


if __name__ == "__main__":
    unittest.main()

