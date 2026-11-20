# dee.cpp/pydee/adapter.py
"""DeeMoeAdapter: integrate dee.cpp's MoE expert execution into a Hugging Face
Qwen3_5Moe (Ornith-1.0-35B-compatible) model.

The adapter monkey-patches each decoder layer's MoE block. The routed experts
are executed by dee.cpp via pydee.Engine.moe_forward_experts; the router,
shared expert, and dense code (attention, RMSNorm, residual, LM head) stay on
HF's normal forward path. This preserves HF reference parity for the dense
path while dee.cpp owns the streaming/caching of routed expert weights.

Usage:

    import torch
    import transformers
    import pydee

    cfg = pydee.configure(shard_path="/tmp/.../model-00001-of-00016.safetensors",
                           num_experts=256, num_layers=40, hidden=2048, inter=512,
                           use_cuda=False, topk=8)
    engine = pydee.new_engine(cfg)

    from pydee.adapter import install_dee_moe
    install_dee_moe(model, engine, num_layers_to_patch=1)  # tier-1: layer 0 only

    # ... generate normally

The tree assumed by HF 4.52+ Qwen3-3_5 MoE:

    model.model.layers[i].mlp
        .gate              # nn.Linear(hidden, num_experts)  -> logits
        .experts           # ModuleList of experts each a gate/up/down nn.Linear
        .shared_experts    # "shared_experts" Module: gate/up/down nn.Linear

Each expert forward in Qwen3-3_5:

    out = down_proj(silu(gate_proj(x)) * up_proj(x))

Standard MoE combine (matches HF reference after shared-expert add):

    final = sum_k(routing_weight[k] * expert_k_output) + shared_expert_output
"""
from __future__ import annotations

import contextlib
import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# _SilentDeeEngine removed: dead code path. pydee.Engine now drives
# moe_forward_experts directly, and the adapter allocates its own scratch.


class DeeMoeAdapter(nn.Module):
    """Drop-in replacement for HF's `model.model.layers[i].mlp` module."""

    def __init__(self, hf_mlp: nn.Module, layer_idx: int, dee_engine, top_k: int):
        super().__init__()
        self.hf_mlp = hf_mlp  # keep HF objects for gate + shared_experts
        self.layer_idx = int(layer_idx)
        self.dee_engine = dee_engine
        self.top_k = int(top_k)
        # Discover HF-side attributes with safe fallbacks.
        self._gate = getattr(hf_mlp, "gate", None)
        self._shared = getattr(hf_mlp, "shared_experts", None)
        self._experts = getattr(hf_mlp, "experts", None)  # may be ignored
        if self._shared is not None:
            for p in self._shared.parameters():
                p.requires_grad_(False)
        if self._gate is not None:
            for p in self._gate.parameters():
                p.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """hidden_states shape: (B, S, H).

        Returns shape (B, S, H) after gate-weighted routed sum + shared expert.
        """
        if hidden_states.dim() == 2:  # (S, H) from tests that flatten batch
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        # 1. Router logits (HF router — BF16). Softmax + top-K (32-bit for stability).
        if self._gate is None:
            raise RuntimeError("DeeMoeAdapter: hf_mlp.gate not found")
        logits = self._gate(hidden_states).to(torch.float32)
        # Qwen3_3_5 uses softmax (NOT sigmoid). Don't apply normalization twice.
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_w, topk_idx = torch.topk(routing_weights, k=self.top_k, dim=-1)
        # Match HF Qwen3-MoE top-K renormalization (controlled by
        # `norm_topk_prob` in the Qwen3-MoE config; default is True and Ornith is
        # built on the same convention). Disable with env DEE_DISABLE_TOPK_RENORM=1
        # for models whose block doesn't renormalize.
        if os.environ.get("DEE_DISABLE_TOPK_RENORM") != "1":
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        B, S, _ = hidden_states.shape
        H = hidden_states.shape[-1]

        # 2. Loop over (B, S) tokens; for each, call dee.cpp routed experts.
        # Output buffer (B, S, H), accumulated on the same device/dtype as input.
        routed_out = torch.zeros_like(hidden_states, dtype=torch.float32)

        # Each token may pick different experts. We loop in Python for tier-1.
        out_np = np.empty((self.top_k * H,), dtype=np.float32)
        h_in_np = np.empty((H,), dtype=np.float32)
        for b in range(B):
            for s in range(S):
                k_idx = topk_idx[b, s].cpu().numpy().astype(np.int64).tolist()
                # Convert hidden_states to F32 for dee.cpp (BF16->F32 looses < ULP)
                h_in_fp32 = hidden_states[b, s].to(torch.float32).cpu().numpy()
                # Ensure contiguous w/ right dtype
                if not h_in_fp32.flags["C_CONTIGUOUS"]:
                    h_in_fp32 = np.ascontiguousarray(h_in_fp32)
                if h_in_fp32.dtype != np.float32:
                    h_in_fp32 = h_in_fp32.astype(np.float32, copy=False)
                np.copyto(h_in_np, h_in_fp32)
                # No need for h_in_fp32 after this; let numpy aliasing be safe
                del h_in_fp32

                # Reset scratch
                out_np.fill(0.0)
                # Call dee.cpp
                ok = self.dee_engine.moe_forward_experts(
                    self.layer_idx, h_in_np, out_np, k_idx
                )
                if not ok:
                    # dee.cpp logged the reason; fall back to HF native experts.
                    if self._experts is not None:
                        x = hidden_states[b, s].unsqueeze(0).unsqueeze(0).to(
                            self._gate.weight.dtype
                        )
                        per_exp = []
                        for e in k_idx:
                            if e < len(self._experts):
                                per_exp.append(self._experts[e](x))
                            else:  # pragma: no cover
                                per_exp.append(torch.zeros_like(x))
                        stacked = torch.cat(per_exp, dim=0).reshape(self.top_k, H).to(torch.float32)
                    else:
                        stacked = torch.zeros((self.top_k, H), dtype=torch.float32)
                else:
                    stacked = torch.from_numpy(
                        out_np.reshape(self.top_k, H)
                    ).to(routed_out.dtype)

                # 3. Combine: routed = sum_k(routing_weights[k] * expert_k_output)
                # routing_weights is BF16 in HF reference; multiply on the
                # accumulated F32 buffer for numerical safety.
                rw = topk_w[b, s].to(torch.float32)
                weighted = torch.einsum("k,kh->h", rw, stacked)
                routed_out[b, s] = weighted

        routed_out = routed_out.to(hidden_states.dtype)

        # 4. Shared expert: HF forward uses F.silu(gate(x))*up(x) -> down.
        if self._shared is not None:
            shared_out = self._shared(hidden_states)
        else:
            shared_out = torch.zeros_like(hidden_states)

        result = routed_out + shared_out
        if squeeze_back:
            return result.squeeze(0)
        return result


def _patch_layer_mlp(model, layer_idx: int, engine, top_k: int) -> None:
    layers = _get_layers_container(model)
    layer = layers[layer_idx]
    if not hasattr(layer, "mlp"):
        raise RuntimeError(f"layer {layer_idx} has no .mlp attribute")
    layer.mlp = DeeMoeAdapter(layer.mlp, layer_idx=layer_idx, dee_engine=engine, top_k=top_k)


def _get_layers_container(model):
    """Discover the decoder layer list regardless of model class.

    Resolution order:
      1. model.model.language_model.layers  (Qwen3_5 hybrid w/ vision wrapper)
      2. model.model.layers                  (standard CausalLM)
      3. model.layers                         (bare decoder stack)
    """
    # Qwen3_5 hybrid (model_type=qwen3_5_moe_text): text decoder is nested
    # under model.language_model.
    if (hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")):
        return model.model.language_model.layers
    # Standard HF CausalLM / ConditionalGeneration: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Bare ForCausalLM: model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("cannot find decoder layer list on model")


def install_dee_moe(model, engine, num_layers_to_patch: Optional[int] = None,
                     top_k: int = 8) -> List[int]:
    """Replace `model.model.layers[i].mlp` for i in [0, num_layers_to_patch).

    Returns the list of patched layer indices.
    """
    layers = _get_layers_container(model)
    if num_layers_to_patch is None:
        num_layers_to_patch = len(layers)
    patched: List[int] = []
    for i in range(int(num_layers_to_patch)):
        try:
            _patch_layer_mlp(model, i, engine, top_k=top_k)
            patched.append(i)
        except Exception as exc:  # pragma: no cover
            print(f"[pydee.adapter] layer {i} patch skipped: {exc!r}")
    return patched


@contextlib.contextmanager
def patched_layers(model, engine, num_layers_to_patch: Optional[int] = None,
                    top_k: int = 8):
    """Context manager: install DeeMoeAdapter, restore originals on exit."""
    layers = _get_layers_container(model)
    saved = []
    for i in range(int(num_layers_to_patch or len(layers))):
        try:
            saved.append((i, layers[i].mlp))
            _patch_layer_mlp(model, i, engine, top_k=top_k)
        except Exception:
            pass
    try:
        yield saved
    finally:
        for i, original in reversed(saved):
            layers[i].mlp = original


# --------------------------------------------------------------------------- #
# Tiny self-test: print summary without importing HF (so we can compile/import
# in CI without the heavy stack).
# --------------------------------------------------------------------------- #
def _self_test() -> None:  # pragma: no cover
    print("pydee.adapter: ready (DeeMoeAdapter + install_dee_moe exported)")


if __name__ == "__main__":
    _self_test()
