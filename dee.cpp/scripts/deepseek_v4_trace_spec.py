"""DS5: trusted-reference trace specification for DeepSeek-V4-Flash-0731.

The trusted reference is the OFFICIAL inference stack shipped with the pinned
repository revision (``inference/model.py`` + ``inference/kernel.py`` +
``inference/convert.py`` + ``inference/generate.py``), NOT a Freebuff
reimplementation. Its forward path defines the trace boundaries below.

The first DS5 trace target is a bounded single-layer reference: embedding +
layer 0 (a hash-routed, pure sliding-window layer: ``compress_ratios[0] == 0``)
+ final norm/head, run on a canonical prompt with greedy decoding. This needs
only shards 00001 (embedding), 00003 (layer 0), and 00048 (final norm, head,
hc_head) from the 48-shard checkpoint (~6 GB), converted with the official
``convert.py`` semantics (``--expert-dtype fp4``, mp=1).

This module pins identities and defines the machine-readable boundary
contract + bounded-capture policy. It imports nothing from transformers or
torch at module import time (lazy) so it stays importable as a flat copy.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pinned official identity (revision mirrors scripts/deepseek_v4_support and
# scripts/deepseek_v4_encoding; kept local for flat-copy safety).
# ---------------------------------------------------------------------------

OFFICIAL_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

CONFIG_JSON_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
GENERATION_CONFIG_SHA256 = "5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0"
INFERENCE_CONFIG_SHA256 = "c90861f3d10a9e4ef5954f8f1a34c529d480da1c5799f84660028f4e38e14e71"

# Official inference requirements (inference/requirements.txt).
INFERENCE_REQUIREMENTS = {
    "torch": ">=2.10.0",
    "transformers": ">=5.0.0",
    "safetensors": ">=0.7.0",
    "fast_hadamard_transform": "",
    "tilelang": "==0.1.8",
}

# Shards needed for the layer-0 subset reference trace (~6 GB total).
SUBSET_SHARDS = {
    "model-00001-of-00048.safetensors",  # embedding (1.06 GB)
    "model-00003-of-00048.safetensors",  # layer 0 dense + 256 routed + shared
    "model-00048-of-00048.safetensors",  # final norm, head, hc_head
}

# ---------------------------------------------------------------------------
# Canonical prompt (encoded via the official DS4 encoder at runtime)
# ---------------------------------------------------------------------------

CANONICAL_PROMPT = "The capital of France is"
CANONICAL_THINKING_MODE = "chat"

# ---------------------------------------------------------------------------
# Bounded-capture policy
# ---------------------------------------------------------------------------

CAPTURE_MAX_TOKENS = 8     # rows kept in every bounded slice (time dim)
CAPTURE_MAX_FEATURES = 64  # trailing feature dims kept in every bounded slice
CAPTURE_MAX_LOGITS = 4096  # head of the vocab dim kept for logits/score tensors


def tensor_sha256(tensor: Any) -> str:
    """Exact bitwise SHA-256 of a tensor's in-memory bytes (any dtype).

    Uses ``view(torch.uint8)`` so quantized (float8/float4) and bf16 dtypes
    hash by their exact stored bits; no numeric cast is performed.
    """
    import torch
    t = tensor.detach().cpu().contiguous()
    raw = t.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def bounded_capture(tensor: Any, *, max_tokens: int = CAPTURE_MAX_TOKENS,
                    max_features: int = CAPTURE_MAX_FEATURES,
                    max_logits: int = CAPTURE_MAX_LOGITS) -> dict[str, Any]:
    """Produce the bounded JSON-safe record for one boundary tensor.

    Always includes the full-tensor bitwise hash, shape/dtype/stride/device,
    min/max/mean and L2 norm, plus a small bounded slice (head of each dim).
    Never captures a full hidden-state or logits tensor.
    """
    import torch
    t = tensor.detach()
    record: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "stride": list(t.stride()),
        "device": str(t.device),
        "is_contiguous": t.is_contiguous(),
        "numel": int(t.numel()),
        "finite": bool(torch.isfinite(t).all().item()),
        "nan_count": int(torch.isnan(t).sum().item()),
        "posinf_count": int((t == float("inf")).sum().item()),
        "neginf_count": int((t == float("-inf")).sum().item()),
        "min": None, "max": None, "mean": None, "l2norm": None,
        "sha256": tensor_sha256(t),
        "slice": None,
    }
    if t.numel() == 0:
        return record
    try:
        ft = t.to(torch.float64)
        record["min"] = float(ft.min().item())
        record["max"] = float(ft.max().item())
        record["mean"] = float(ft.mean().item())
        record["l2norm"] = float(ft.square().sum().sqrt().item())
    except Exception:  # pragma: no cover - defensive for exotic dtypes
        pass
    # Bounded slice: head of every dim, clamped to the capture limits.
    ndim = t.dim()
    sl = [slice(0, None)] * ndim
    if ndim >= 1:
        sl[0] = slice(0, min(max_tokens, t.shape[0]))
    if ndim >= 2:
        sl[1] = slice(0, min(max_features, t.shape[1]))
    elif ndim == 1:
        # 1-D tensors (scores, ids, weights): keep the first max_logits entries.
        sl[0] = slice(0, min(max(max_logits, max_features), t.shape[0]))
    for d in range(2, ndim):
        sl[d] = slice(0, min(max_features, t.shape[d]))
    sl = tuple(sl)
    try:
        sub = t[sl]
        record["slice"] = {
            "bounds": [[int(s.start or 0), int(s.stop or 0)] for s in sl],
            "shape": list(sub.shape),
            "sha256": tensor_sha256(sub),
            "min": None, "max": None,
        }
        if sub.numel() and sub.dtype not in (torch.int8, torch.uint8, torch.int16,
                                             torch.int32, torch.int64, torch.bool):
            fsub = sub.to(torch.float64)
            record["slice"]["min"] = float(fsub.min().item())
            record["slice"]["max"] = float(fsub.max().item())
    except Exception:  # pragma: no cover - never fail the capture on slicing
        pass
    return record


# ---------------------------------------------------------------------------
# Boundary contract (aligned with official inference/model.py forward path)
# ---------------------------------------------------------------------------

@dataclass
class Boundary:
    """One trace boundary: a (module path, capture kind) pair + gate."""

    key: str
    module_path: str          # dotted path under the Transformer ('' = model)
    kind: str                 # 'output' | 'input'
    required: bool = True


BOUNDARIES: tuple[Boundary, ...] = (
    # embedding + HC-expanded hidden state
    Boundary("embed_out", "embed", "output"),
    # NOTE: ``hidden_after_embed_expand`` (the HC-expanded [B,S,HC,D] state
    # right after embed + unsqueeze/repeat) is NOT hookable via a submodule
    # path (it lives inside Transformer.forward, not a module). It is kept as
    # documented metadata only; capture it by re-deriving it from embed_out
    # in the harness when the expansion semantics are pinned.
    Boundary("hidden_after_embed_expand", "", "output", required=False),
    # layer 0 (hash-routed, pure window attention)
    Boundary("layer0_hc_attn_pre_out", "layers.0", "output", required=False),
    Boundary("layer0_attn_norm_in", "layers.0.attn_norm", "input"),
    Boundary("layer0_attn_norm_out", "layers.0.attn_norm", "output"),
    Boundary("layer0_attn_out", "layers.0.attn", "output"),
    Boundary("layer0_hc_attn_post_out", "layers.0", "output", required=False),
    Boundary("layer0_ffn_norm_in", "layers.0.ffn_norm", "input"),
    Boundary("layer0_ffn_norm_out", "layers.0.ffn_norm", "output"),
    Boundary("layer0_gate_out", "layers.0.ffn.gate", "output"),
    Boundary("layer0_moe_out", "layers.0.ffn", "output"),
    Boundary("layer0_hc_ffn_post_out", "layers.0", "output", required=False),
    # final norm + head
    Boundary("final_norm_out", "norm", "output"),
    Boundary("head_logits", "head", "output"),
)

# Expert-level boundaries captured per selected expert (key pattern filled by
# the harness with the expert index).
EXPERT_BOUNDARY_PATTERNS = (
    ("layer0_expert{eid}_out", "layers.0.ffn.experts.{eid}", "output"),
    ("layer0_shared_expert_out", "layers.0.ffn.shared_experts", "output"),
)

GATES: tuple[str, ...] = (
    "shard_identity",       # subset shards present, sizes + headers verified
    "config_identity",      # config.json SHA matches the pinned hash
    "tokenizer_identity",   # DS4 asset hashes verified (fail-closed)
    "reference_loaded",     # official Transformer(n_layers=1) built + loaded
    "boundary_coverage",    # every required boundary was captured
    "no_nan",               # every captured boundary is finite
    "artifact_complete",    # evidence JSON + manifest written and hashed
)


def build_subset_config(inference_config_path: Path) -> dict[str, Any]:
    """Official ModelArgs for the layer-0 subset reference run.

    Loads inference/config.json and overrides the model to a single layer
    (embedding + layer 0 + final norm/head), disabling DSpark and capping the
    sequence/batch. Returns the kwargs dict for ModelArgs(**kwargs).
    """
    cfg = json.loads(inference_config_path.read_text(encoding="utf-8"))
    cfg = dict(cfg)
    cfg["n_layers"] = 1
    cfg["n_mtp_layers"] = 0
    cfg["dspark_block_size"] = 0
    cfg["dspark_target_layer_ids"] = tuple()
    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = 4096
    cfg["temperature"] = 0.0  # greedy decode: trace is deterministic
    # compress_ratios is a list in JSON; official code indexes per layer.
    return cfg


def verify_pinned_files(assets: Path) -> dict[str, str]:
    """Fail-closed SHA-256 verification of the pinned identity files.

    assets must contain config.json, generation_config.json,
    inference/config.json (and the encoding/tokenizer assets are verified by
    scripts/deepseek_v4_encoding.verify_tokenizer_assets).
    """
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    got = {
        "config.json": _sha256(assets / "config.json"),
        "generation_config.json": _sha256(assets / "generation_config.json"),
        "inference/config.json": _sha256(assets / "inference" / "config.json"),
    }
    expected = {
        "config.json": CONFIG_JSON_SHA256,
        "generation_config.json": GENERATION_CONFIG_SHA256,
        "inference/config.json": INFERENCE_CONFIG_SHA256,
    }
    for name, want in expected.items():
        if got[name] != want:
            raise ValueError(
                f"{name} hash mismatch: expected {want}, got {got[name]}"
            )
    return got


def flatten_boundary_keys() -> list[str]:
    """All fixed boundary keys (expert patterns excluded; they are filled in)."""
    return [b.key for b in BOUNDARIES]


# ---------------------------------------------------------------------------
# Reference-stack availability probe result schema
# ---------------------------------------------------------------------------

@dataclass
class EnvProbe:
    """JSON-safe record of the reference-environment probe."""

    torch_version: str = ""
    transformers_version: str = ""
    tilelang_version: str = ""
    safetensors_version: str = ""
    cuda_available: bool = False
    cuda_device: str = ""
    sm_arch: str = ""
    disk_free_gib: float = 0.0
    hf_reachable: bool = False
    checkpoint_mounted: bool = False
    first_failing_requirement: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "tilelang_version": self.tilelang_version,
            "safetensors_version": self.safetensors_version,
            "cuda_available": self.cuda_available,
            "cuda_device": self.cuda_device,
            "sm_arch": self.sm_arch,
            "disk_free_gib": self.disk_free_gib,
            "hf_reachable": self.hf_reachable,
            "checkpoint_mounted": self.checkpoint_mounted,
            "first_failing_requirement": self.first_failing_requirement,
        }


def parse_sm_arch(device_props: Any) -> str:
    """Best-effort SM architecture string from torch.cuda.get_device_properties."""
    major = getattr(device_props, "major", None)
    minor = getattr(device_props, "minor", None)
    if major is not None and minor is not None:
        return f"sm_{major}{minor}"
    return "unknown"
