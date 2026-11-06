"""DS5 tests: trace spec (pins, bounded capture) + harness plumbing.

The full reference pipeline needs CUDA + the official checkpoint, so these
tests cover everything that can be proven locally: pinned identities,
bounded-capture semantics, canonical-prompt encoding, subset-shard manifest,
and the name-driven boundary-hook machinery on a synthetic module tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    sys.stderr.write("tests/test_deepseek_v4_trace_spec.py requires pytest\n")
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "kaggle" / "deepseek-v4-flash-0731"))

import torch  # noqa: E402

from deepseek_v4_trace_spec import (  # noqa: E402
    BOUNDARIES,
    CANONICAL_PROMPT,
    CAPTURE_MAX_FEATURES,
    CAPTURE_MAX_TOKENS,
    CONFIG_JSON_SHA256,
    GENERATION_CONFIG_SHA256,
    INFERENCE_CONFIG_SHA256,
    bounded_capture,
    build_subset_config,
    flatten_boundary_keys,
    tensor_sha256,
    verify_pinned_files,
)
import ds5_trace_runtime as harness  # noqa: E402


OFFICIAL_SOURCE = REPO_ROOT / "benchmark_reports" / "deepseek-v4-flash-0731-t4" / "official-source"


# ---------------------------------------------------------------------------
# Pinned identities
# ---------------------------------------------------------------------------

def test_pinned_files_verify_against_checked_in_sources() -> None:
    hashes = verify_pinned_files(OFFICIAL_SOURCE)
    assert hashes["config.json"] == CONFIG_JSON_SHA256
    assert hashes["generation_config.json"] == GENERATION_CONFIG_SHA256
    assert hashes["inference/config.json"] == INFERENCE_CONFIG_SHA256


def test_subset_shard_manifest_pinned() -> None:
    assert set(harness.SUBSET_MANIFEST) == {
        "model-00001-of-00048.safetensors",
        "model-00002-of-00048.safetensors",
        "model-00045-of-00048.safetensors",
    }
    # Full physical sizes: 8 (length prefix) + header_size + max(data_offsets[1]),
    # cross-checked against HF Content-Length of the pinned revision. The v3
    # pins omitted the header overhead and truncated downloads by 96/172240/400
    # bytes, failing the official convert with "incomplete metadata".
    assert harness.SUBSET_MANIFEST["model-00001-of-00048.safetensors"][0] == 1059061856
    assert harness.SUBSET_MANIFEST["model-00002-of-00048.safetensors"][0] == 3566321192
    assert harness.SUBSET_MANIFEST["model-00045-of-00048.safetensors"][0] == 1059332516
    # every pinned size must be exactly covered by its header (self-consistency)
    for shard, (size, _hdr, _cnt) in harness.SUBSET_MANIFEST.items():
        assert size >= 8, shard


def test_build_subset_config_single_layer() -> None:
    cfg = build_subset_config(OFFICIAL_SOURCE / "inference" / "config.json")
    assert cfg["n_layers"] == 1
    assert cfg["n_mtp_layers"] == 0
    assert cfg["dspark_block_size"] == 0
    assert cfg["dspark_target_layer_ids"] == tuple()
    assert cfg["max_batch_size"] == 1
    assert cfg["max_seq_len"] == 4096
    assert cfg["temperature"] == 0.0
    assert cfg["vocab_size"] == 129280
    assert isinstance(cfg["compress_ratios"], list)


def test_boundary_contract_complete() -> None:
    keys = flatten_boundary_keys()
    for needed in ("embed_out", "layer0_attn_norm_in", "layer0_attn_out",
                   "layer0_ffn_norm_in", "layer0_gate_out", "layer0_moe_out",
                   "final_norm_out", "head_logits"):
        assert needed in keys, needed
    required_paths = [b.module_path for b in BOUNDARIES if b.required and b.module_path]
    assert "layers.0.attn_norm" in required_paths


# ---------------------------------------------------------------------------
# Bounded capture
# ---------------------------------------------------------------------------

def test_tensor_sha256_deterministic_and_bitwise() -> None:
    a = torch.randn(16, 64, dtype=torch.float32)
    assert tensor_sha256(a) == tensor_sha256(a.clone())
    b = a.clone()
    b[0, 0] = b[0, 0] + 2.0**-23  # exactly 1 fp32 ULP
    assert tensor_sha256(a) != tensor_sha256(b)


def test_bounded_capture_float() -> None:
    t = torch.randn(32, 128, dtype=torch.float32)
    rec = bounded_capture(t)
    assert rec["shape"] == [32, 128]
    assert rec["finite"] is True
    assert rec["nan_count"] == 0
    assert rec["sha256"] == tensor_sha256(t)
    assert rec["min"] <= rec["max"]
    assert rec["l2norm"] > 0.0
    sl = rec["slice"]
    assert sl["bounds"][0] == [0, CAPTURE_MAX_TOKENS]
    assert sl["bounds"][1] == [0, CAPTURE_MAX_FEATURES]
    assert sl["shape"] == [CAPTURE_MAX_TOKENS, CAPTURE_MAX_FEATURES]


def test_bounded_capture_nan_inf() -> None:
    t = torch.zeros(4, 8)
    t[0, 0] = float("nan")
    t[1, 1] = float("-inf")
    t[2, 2] = float("inf")
    rec = bounded_capture(t)
    assert rec["finite"] is False
    assert rec["nan_count"] == 1
    assert rec["neginf_count"] == 1
    assert rec["posinf_count"] == 1


def test_bounded_capture_int64_and_1d() -> None:
    ids = torch.tensor([0, 128803, 671, 6102, 294, 8760, 344, 128804, 128822],
                       dtype=torch.int64)
    rec = bounded_capture(ids)
    assert rec["dtype"] == "torch.int64"
    # 1-D tensors keep the first max(max_logits, max_features) entries: all 9
    assert rec["slice"]["shape"][0] == min(9, 4096)
    assert rec["slice"]["sha256"] == tensor_sha256(ids)


def test_bounded_capture_empty() -> None:
    rec = bounded_capture(torch.zeros(0, 8))
    assert rec["numel"] == 0
    assert rec["min"] is None and rec["slice"] is None


def test_bounded_capture_3d_limits_all_dims() -> None:
    t = torch.randn(24, 4, 512)  # e.g. hidden [b, hc, d] after HC expansion
    rec = bounded_capture(t)
    assert rec["slice"]["shape"] == [CAPTURE_MAX_TOKENS, 4, CAPTURE_MAX_FEATURES]


# ---------------------------------------------------------------------------
# Canonical prompt (DS4 encoder integration)
# ---------------------------------------------------------------------------

def test_canonical_prompt_exact_ids() -> None:
    prompt = harness.encode_canonical_prompt()
    assert prompt["prompt"] == CANONICAL_PROMPT
    assert prompt["token_ids"] == [0, 128803, 671, 6102, 294, 8760, 344, 128804, 128822]
    assert prompt["n_tokens"] == 9


# ---------------------------------------------------------------------------
# Hook plumbing on a synthetic module tree (CPU)
# ---------------------------------------------------------------------------

class _TinyMoe(torch.nn.Module):
    def __init__(self, n_experts: int = 3) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(8, 8)
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8) for _ in range(n_experts)]
        )
        self.shared_experts = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.gate(x)
        return sum(e(x) for e in self.experts) + self.shared_experts(x)


class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = torch.nn.Linear(8, 8)
        self.attn = torch.nn.Linear(8, 8)
        self.ffn_norm = torch.nn.Linear(8, 8)
        self.ffn = _TinyMoe()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return x


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(8, 8)
        self.layers = torch.nn.ModuleList([_TinyBlock()])
        self.norm = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)


def test_hook_plumbing_synthetic_model() -> None:
    model = _TinyModel()
    handles, registered = harness.register_boundary_hooks(model)
    try:
        with torch.inference_mode():
            _ = model(torch.randn(3, 8))
        captures = harness.take_captures()
    finally:
        for handle in handles:
            handle.handle.remove()
    for key in ("embed_out", "layer0_attn_norm_in", "layer0_attn_norm_out",
                "layer0_attn_out", "layer0_ffn_norm_in", "layer0_ffn_norm_out",
                "layer0_gate_out", "layer0_moe_out",
                "layer0_expert0_out", "layer0_expert1_out", "layer0_expert2_out",
                "layer0_shared_expert_out", "final_norm_out", "head_logits"):
        assert key in captures, key
        assert len(captures[key]) >= 1
        assert captures[key][0]["record"]["sha256"]
    # no expert beyond the 3 present was resolved
    assert "layer0_expert3_out" not in captures


def test_hook_removal_stops_capture() -> None:
    model = _TinyModel()
    handles, _ = harness.register_boundary_hooks(model)
    for handle in handles:
        handle.handle.remove()
    with torch.inference_mode():
        _ = model(torch.randn(2, 8))
    captures = harness.take_captures()
    assert captures == {}


def test_gate_evaluation_pass() -> None:
    result = {
        "shards": {"verified": {"x": 1}},
        "identity": {"files": {"config.json": "a"}, "tokenizer": {"t": 1}},
        "reference": {"missing_params": []},
        "trace": {
            "prefill": {b.key: {"entries": []} for b in BOUNDARIES if b.required},
        },
    }
    out = harness.evaluate_gates(result)
    assert out["verdict"] == "ACCEPT_TRACE_GENERATED"
    assert out["first_failing_gate"] is None


def test_gate_evaluation_fails_on_nan() -> None:
    result = {
        "shards": {"verified": {"x": 1}},
        "identity": {"files": {"config.json": "a"}, "tokenizer": {"t": 1}},
        "reference": {"missing_params": []},
        "trace": {
            "prefill": {
                "embed_out": {"entries": [{
                    "record": {"nan_count": 3, "sha256": "x"},
                }]},
                **{b.key: {"entries": []} for b in BOUNDARIES if b.required
                   and b.key != "embed_out"},
            },
        },
    }
    out = harness.evaluate_gates(result)
    assert out["verdict"] == "INVALID_EXPERIMENT"
    assert out["first_failing_gate"] == "no_nan"


def test_evidence_write(tmp_path) -> None:
    result = {"probe": {}, "identity": {}, "prompt": {}, "reference": {},
              "trace": {}, "gates": {}, "verdict": "INVALID_EXPERIMENT"}
    artifact = harness.write_evidence(result, tmp_path)
    assert artifact["artifact_complete"] is True
    evidence = tmp_path / "ds5-trace-evidence.json"
    manifest = tmp_path / "manifest.json"
    assert evidence.is_file() and manifest.is_file()
    # evidence sha matches the written file
    import hashlib
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == artifact["evidence_sha256"]
