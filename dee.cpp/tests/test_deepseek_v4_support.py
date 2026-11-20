from __future__ import annotations

import json
import sys
from pathlib import Path

# pytest is required to run this file (pytest.raises / pytest.skip fixtures
# and the __main__ runner below). Guard the import so the CTest entry
# (`python tests/test_deepseek_v4_support.py`) fails with a clear message
# instead of an opaque traceback when pytest is missing.
try:
    import pytest
except ImportError:
    sys.stderr.write("tests/test_deepseek_v4_support.py requires pytest: pip install pytest\n")
    sys.exit(2)

# Make `scripts` importable regardless of how this file is invoked (pytest -m
# from the repo root, or `python tests/test_deepseek_v4_support.py` via CTest,
# where sys.path[0] is the script's own directory).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import deepseek_v4_support as v4  # noqa: E402


ROOT = Path(__file__).parents[1]


def _index_with(shards: dict[str, list[tuple[str, str, list[int]]]]) -> dict:
    """Build a minimal index + headers pair.

    shards maps shard name -> list of (tensor_name, dtype, shape).
    """
    weight_map = {}
    headers = {}
    offset = 0
    for shard, entries in shards.items():
        header: dict = {}
        for name, dtype, shape in entries:
            nbytes = v4.DTYPE_BYTES[dtype]
            for dim in shape:
                nbytes *= dim
            header[name] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + nbytes],
            }
            offset += nbytes
            weight_map[name] = shard
        headers[shard] = header
    index = {"metadata": {"total_size": offset}, "weight_map": weight_map}
    return index, headers


def test_component_and_module_classification() -> None:
    assert v4.component_for_tensor("layers.6.ffn.experts.12.w1.weight") == "routed_expert"
    assert v4.component_for_tensor("layers.6.ffn.experts.12.w1.scale") == "routed_expert"
    assert v4.component_for_tensor("layers.6.ffn.shared_experts.w2.weight") == "shared_expert"
    assert v4.component_for_tensor("layers.6.ffn.gate.bias") == "router"
    assert v4.component_for_tensor("layers.6.hc_attn_fn") == "hash_compress"
    assert v4.component_for_tensor("layers.6.attn.indexer.compressor.ape") == "hash_compress"
    assert v4.component_for_tensor("mtp.2.dspark_markov_w1.weight") == "dspark"
    assert v4.component_for_tensor("mtp.0.main_proj.weight") == "dspark"
    assert v4.component_for_tensor("layers.6.attn.wq_b.weight") == "attention_dense"
    assert v4.component_for_tensor("embed.weight") == "embedding"
    assert v4.component_for_tensor("head.weight") == "lm_head"
    assert v4.layer_from_tensor_name("layers.6.ffn.experts.12.w1.weight") == 6
    assert v4.layer_from_tensor_name("mtp.2.dspark_markov_w1.weight") == 2
    assert v4.layer_from_tensor_name("embed.weight") is None


def test_storage_plan_packed_fp4_expert() -> None:
    plan = v4.storage_plan_for_tensor(
        "layers.6.ffn.experts.0.w1.weight", "I8", [2048, 2048]
    )
    assert plan["kind"] == "packed_fp4_expert"
    assert plan["compressed_bytes"] == 2048 * 2048
    assert plan["expanded_fp16_bytes"] == 2048 * 2048 * 2 * 2
    assert plan["expanded_int8_bytes"] == 2048 * 2048 * 2
    assert plan["scale_tensor"] == "layers.6.ffn.experts.0.w1.scale"
    assert plan["scale_block_elements"] == 32
    assert plan["block_shape"] == [2048, 128]


def test_storage_plan_fp8_dense_and_plain() -> None:
    fp8 = v4.storage_plan_for_tensor("layers.6.attn.wq_b.weight", "F8_E4M3", [32768, 1024])
    assert fp8["kind"] == "fp8_tensor"
    assert fp8["compressed_bytes"] == 32768 * 1024
    assert fp8["expanded_fp16_bytes"] == 32768 * 1024 * 2
    assert fp8["scale_tensor"] == "layers.6.attn.wq_b.scale"
    plain = v4.storage_plan_for_tensor("layers.6.attn.attn_sink", "F32", [64])
    assert plain["kind"] == "plain_tensor"
    assert plain["compressed_bytes"] == 64 * 4


def test_ledger_validation_rejects_missing_or_extra_tensors() -> None:
    index, headers = _index_with({
        "s1.safetensors": [("a", "F32", [4])],
        "s2.safetensors": [("b", "I8", [4])],
    })
    rows, summary = v4.build_complete_tensor_ledger(
        index, headers, expected_shard_count=2, expected_tensor_count=2
    )
    assert summary["compressed_bytes"] == 4 * 4 + 4
    assert rows[0]["component"] == "other"

    # missing shard header
    with pytest.raises(ValueError, match="missing shard headers"):
        v4.build_complete_tensor_ledger(index, {}, expected_shard_count=2, expected_tensor_count=2)

    # unexpected tensor in header
    bad = {k: dict(v) for k, v in headers.items()}
    bad["s1.safetensors"] = {**headers["s1.safetensors"], "extra": {
        "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
    }}
    with pytest.raises(ValueError, match="unexpected tensor"):
        v4.build_complete_tensor_ledger(index, bad, expected_shard_count=2, expected_tensor_count=2)


def test_fp8_dense_scale_shape_validation() -> None:
    # wq_b.weight is F8_E4M3 [out, in]; its scale must be [ceil(out/128), ceil(in/128)].
    plan = v4.storage_plan_for_tensor(
        "layers.6.attn.wq_b.weight", "F8_E4M3", [32768, 1024]
    )
    assert plan["kind"] == "fp8_tensor"
    assert plan["block_shape"] == [256, 8]

    # Matching scale shape passes validation.
    index, headers = _index_with({
        "s1.safetensors": [
            ("layers.6.attn.wq_b.weight", "F8_E4M3", [32768, 1024]),
            ("layers.6.attn.wq_b.scale", "F8_E8M0", [256, 8]),
        ],
    })
    rows, _summary = v4.build_complete_tensor_ledger(
        index, headers, expected_shard_count=1, expected_tensor_count=2
    )
    weight_row = next(r for r in rows if r["tensor_name"].endswith(".wq_b.weight"))
    assert weight_row["scale_tensor_shape"] == [256, 8]

    # A mismatched scale shape fails closed.
    index, headers = _index_with({
        "s1.safetensors": [
            ("layers.6.attn.wq_b.weight", "F8_E4M3", [32768, 1024]),
            ("layers.6.attn.wq_b.scale", "F8_E8M0", [128, 8]),
        ],
    })
    with pytest.raises(ValueError, match="scale shape"):
        v4.build_complete_tensor_ledger(
            index, headers, expected_shard_count=1, expected_tensor_count=2
        )


def test_scale_tensor_existence_validation_rejects_missing_scale() -> None:
    # w1.weight is a packed FP4 expert weight, so storage_plan_for_tensor
    # references w1.scale, but the index contains no such tensor.
    index, headers = _index_with({
        "s1.safetensors": [("layers.6.ffn.experts.0.w1.weight", "I8", [16, 16])],
    })
    with pytest.raises(ValueError, match="references missing scale tensor"):
        v4.build_complete_tensor_ledger(
            index, headers, expected_shard_count=1, expected_tensor_count=1
        )

    # Adding the matching scale tensor makes validation pass.
    index, headers = _index_with({
        "s1.safetensors": [
            ("layers.6.ffn.experts.0.w1.weight", "I8", [16, 16]),
            ("layers.6.ffn.experts.0.w1.scale", "F8_E8M0", [16, 1]),
        ],
    })
    rows, _summary = v4.build_complete_tensor_ledger(
        index, headers, expected_shard_count=1, expected_tensor_count=2
    )
    weight_row = next(row for row in rows if row["tensor_name"].endswith(".w1.weight"))
    assert weight_row["scale_tensor"] == "layers.6.ffn.experts.0.w1.scale"
    assert weight_row["shared"] is False
    assert weight_row["cache_class"] == "mmap_checkpoint_plus_bounded_gpu_lru"
    assert weight_row["active_every_token"] is False


def test_real_index_and_cached_headers_invariants() -> None:
    """Guard against drift in the pinned official checkpoint (no network)."""
    # Prefer the pinned copy committed with the campaign; fall back to the
    # local audit checkout under tmp/.
    index_path = ROOT / "benchmark_reports/deepseek-v4-flash-0731-t4/official-source/model.safetensors.index.json"
    if not index_path.is_file():
        index_path = ROOT / "tmp/dsv4-official-audit/model.safetensors.index.json"
    headers_dir = ROOT / "benchmark_reports/deepseek-v4-flash-0731-t4/shard-headers"
    if not index_path.is_file() or not headers_dir.is_dir():
        pytest.skip("pinned official audit files not present locally")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    headers = {}
    for shard_path in sorted(headers_dir.glob("model-*.safetensors.json")):
        headers[shard_path.name[:-5]] = json.loads(shard_path.read_text(encoding="utf-8"))
    rows, summary = v4.build_complete_tensor_ledger(index, headers)
    assert summary["tensor_count"] == 72317
    assert len(summary["shards"]) == 48
    assert summary["compressed_bytes"] == 166878536440
    assert summary["declared_total_size"] == 166878536440
    expert_rows = [row for row in rows if row["routed"]]
    assert len(expert_rows) == 66048
    assert all(row["stored_dtype"] == "I8" or row["tensor_name"].endswith(".scale")
               for row in expert_rows)


def _real_ledger() -> tuple[dict, list]:
    """Load the committed validated ledger rows (or skip if absent)."""
    ledger_path = ROOT / "benchmark_reports/deepseek-v4-flash-0731-t4/MODEL_LEDGER.json"
    if not ledger_path.is_file():
        pytest.skip("MODEL_LEDGER.json not present locally")
    with open(ledger_path, encoding="utf-8") as fh:
        ledger = json.load(fh)
    return v4.index_ledger_rows(ledger["tensors"]), ledger["tensors"]


def test_ds8_routed_expert_resolution_across_matrix() -> None:
    """Arbitrary (layer, expert) resolves against the real official ledger."""
    by_name, _rows = _real_ledger()
    # early / middle / late layer x low / middle / high expert id
    matrix = [(3, 0), (3, 7), (3, 255), (20, 0), (20, 128), (41, 255)]
    seen_shards = set()
    for layer, expert in matrix:
        resolved = v4.resolve_expert_tensors(by_name, layer, expert)
        assert len(resolved) == 6
        plan = v4.expert_load_plan(resolved)
        assert plan["tensor_count"] == 6
        # official shard naming: model-000{N:02d} (5-digit, 48 shards)
        assert plan["shard"] == f"model-000{layer + 2:02d}-of-00048.safetensors"
        assert plan["compressed_bytes"] == 13369344  # all routed experts same size
        # every weight/scale pair co-located
        shards = {row["source_shard"] for row in resolved.values()}
        assert len(shards) == 1
        seen_shards.add(plan["shard"])
    assert len(seen_shards) == 3  # three distinct shards exercised


def test_ds8_shared_expert_resolution_fp8_storage() -> None:
    by_name, _rows = _real_ledger()
    for layer in (3, 20, 41):
        resolved = v4.resolve_shared_expert_tensors(by_name, layer)
        assert len(resolved) == 6
        plan = v4.expert_load_plan(resolved)
        assert plan["tensor_count"] == 6
        assert plan["shard"] == f"model-000{layer + 2:02d}-of-00048.safetensors"
        # Shared expert weights are F8_E4M3 (NOT packed I8 FP4)
        for name in ("w1", "w2", "w3"):
            assert resolved[f"layers.{layer}.ffn.shared_experts.{name}.weight"]["stored_dtype"] == "F8_E4M3"
            assert resolved[f"layers.{layer}.ffn.shared_experts.{name}.scale"]["stored_dtype"] == "F8_E8M0"
        # block-128 tiling for the FP8 scales
        assert resolved[f"layers.{layer}.ffn.shared_experts.w1.weight"]["block_shape"] == [16, 32]
        assert resolved[f"layers.{layer}.ffn.shared_experts.w1.scale"]["shape"] == [16, 32]


def test_ds8_router_resolution_score_and_hash_layers() -> None:
    by_name, _rows = _real_ledger()
    # score layer (>=3): weight BF16 + bias F32
    score = v4.resolve_router_tensors(by_name, 3)
    assert set(score) == {"layers.3.ffn.gate.weight", "layers.3.ffn.gate.bias"}
    assert score["layers.3.ffn.gate.weight"]["stored_dtype"] == "BF16"
    assert score["layers.3.ffn.gate.weight"]["shape"] == [256, 4096]
    assert score["layers.3.ffn.gate.bias"]["stored_dtype"] == "F32"
    # hash layer (<3): tid2eid only
    hash_layer = v4.resolve_router_tensors(by_name, 0)
    assert set(hash_layer) == {"layers.0.ffn.gate.tid2eid"}
    assert hash_layer["layers.0.ffn.gate.tid2eid"]["stored_dtype"] == "I64"
    assert hash_layer["layers.0.ffn.gate.tid2eid"]["shape"] == [129280, 6]


def test_ds8_resolution_fails_closed_on_missing_tensor_and_cross_shard() -> None:
    by_name, _rows = _real_ledger()
    with pytest.raises(ValueError, match="missing from ledger"):
        v4.resolve_expert_tensors(by_name, 999, 0)  # no such layer
    # corrupt one row's shard -> cross-shard rejection
    good = v4.resolve_expert_tensors(by_name, 3, 0)
    corrupted = dict(good)
    first = next(iter(corrupted))
    row = dict(corrupted[first])
    row["source_shard"] = "model-00099-of-00048.safetensors"
    corrupted[first] = row
    with pytest.raises(ValueError, match="exactly one"):
        v4._assert_same_shard(corrupted)


def test_ds8_expert_load_plan_budget_bytes() -> None:
    by_name, _rows = _real_ledger()
    plan = v4.expert_load_plan(v4.resolve_expert_tensors(by_name, 6, 0))
    # DS7 measured: one packed routed expert = 12.75 MiB compressed
    assert plan["compressed_bytes"] == 13369344
    assert plan["packed_weight_bytes"] == 3 * 4194304  # w1+w2+w3 packed
    assert plan["scale_bytes"] == 3 * 262144  # three F8_E8M0 scale tensors


if __name__ == "__main__":
    # Self-executing so the CTest entry (python tests/test_deepseek_v4_support.py)
    # actually runs pytest-collected tests instead of a silent no-op.
    sys.exit(pytest.main([__file__, "-q"]))
