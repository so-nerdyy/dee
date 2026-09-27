"""MiMo-V2.6 model module — the dee exact-serving adapter for mimo_v2.

Mirrors ``scripts/deepseek_v4_model.py``'s contract (TensorSource → layer
objects → greedy ``generate``) but implements the mimo_v2 architecture
from ``modeling_mimo_v2.py`` (HF revision 5711b268...):

  * standard pre-norm decoder: input_layernorm -> attn -> +res ->
    post_attention_layernorm -> mlp -> +res
  * hybrid attention: ``hybrid_layer_pattern[i] == 1`` -> sliding-window
    (window=128) with per-head attention sinks; ``== 0`` -> full causal.
    SWA and full layers carry their own geometry (swa_* config fields:
    kv heads 8 vs 4, rope theta 1e4 vs 1e7)
  * fused qkv_proj (Flash) or split q/k/v_proj (Pro) — both F8_E4M3 with
    F32 scale_inv 128x128 blocks on Flash; o_proj BF16 (ignored_layers)
  * GQA: 64 q heads, head_dim 192, v_head_dim 128, partial rope on the
    first ``rope_dim`` dims, v_scale=0.707
  * MoE per ``moe_layer_freq`` (Flash: layer 0 dense, 1..47 MoE):
    noaux_tc router — sigmoid scores, selection by sigmoid+bias with
    group-limited top-k (Flash n_group=1 degenerates), weights = raw
    sigmoid gathered, normalized, scaled; 256 routed experts, top-8;
    expert FFN = down(silu(gate)*up) with NO clamp (swiglu_limit=0)
  * routed experts execute through pydee.Engine — mxfp4 records are
    wire-identical to the DSv4 FP4 codec (e2m1 nibbles + ue8m0/32).

Exactness contract identical to DSv4's: checkpoint bytes are authoritative;
dense weights dequantize to fp32 at load then cast to the compute dtype;
expert records feed the engine's native fp4->fp16 SwiGLU path.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from scripts.deepseek_v4_model import (
    DTYPE_MAP,
    CommittedHeaderSource,
    TensorSource,
    _tensor_storage_sha256,
)

OFFICIAL_REPOSITORY = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
OFFICIAL_REVISION = "5711b268169967567844e1e560e8a3966da959b1"


# ---------------------------------------------------------------------------
# Tensor source: local-shard reads keyed by header dump (any shard naming).
# ---------------------------------------------------------------------------


class ScanHeaderSource(CommittedHeaderSource):
    """CommittedHeaderSource generalized to arbitrary shard file names.

    The base class enumerates DSv4's ``model-NNNNN-of-00048`` names; MiMo
    uses EP-sharded names (``model_pp0_epN_shard0.safetensors``), so the
    header directory is scanned instead.  Headers come from
    ``tools/phase3/fetch_headers.py`` (or a committed dump).
    """

    def _load_headers(self) -> None:
        for path in sorted(self.headers_dir.glob("*.json")):
            shard = path.name[: -len(".json")]
            header = json.loads(path.read_text(encoding="utf-8"))
            self._headers[shard] = header
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                self._tensors[name] = {
                    "shard": shard,
                    "offset": int(meta["data_offsets"][0]),
                    "length": int(meta["data_offsets"][1]
                                  - meta["data_offsets"][0]),
                    "dtype": meta["dtype"],
                    "shape": [int(d) for d in meta["shape"]],
                }
        if not self._tensors:
            raise FileNotFoundError(
                f"no tensors discovered under {self.headers_dir}")


class LocalShardSource(ScanHeaderSource):
    """Reads tensor bytes from locally-mounted safetensors shards."""

    def __init__(self, headers_dir: Path | str, shards_dir: Path | str, *,
                 revision: str = OFFICIAL_REVISION):
        self.shards_dir = Path(shards_dir)
        super().__init__(headers_dir, revision)

    def _fetch_prefix_len(self, shard: str) -> int:
        import struct
        with (self.shards_dir / shard).open("rb") as fh:
            return struct.unpack("<Q", fh.read(8))[0]

    def _fetch_bytes(self, name: str) -> bytes:
        shard, start, length = self.absolute_range(name)
        with (self.shards_dir / shard).open("rb") as fh:
            fh.seek(start)
            data = fh.read(length)
        return data


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttnGeometry:
    """Per-layer attention shape — SWA layers may differ from full ones."""
    n_heads: int
    n_kv_heads: int
    head_dim: int
    v_head_dim: int
    rope_dim: int
    rope_theta: float


@dataclass(frozen=True)
class MiMoV2Config:
    vocab_size: int = 152576
    hidden: int = 4096
    n_layers: int = 48
    dense_inter: int = 16384              # intermediate_size (dense MLPs)
    full_attn: AttnGeometry = AttnGeometry(64, 4, 192, 128, 64, 10_000_000.0)
    swa_attn: AttnGeometry = AttnGeometry(64, 8, 192, 128, 64, 10_000.0)
    attention_value_scale: float = 0.707
    sliding_window: int = 128
    hybrid_pattern: tuple[int, ...] = ()  # 1 = sliding-window, 0 = full
    swa_sink: bool = True
    full_sink: bool = False
    n_routed: int = 256
    topk: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling: float = 1.0
    moe_inter: int = 2048
    moe_layer_freq: tuple[int, ...] = ()  # 1 = MoE, 0 = dense MLP
    projection_layout: str = "fused_qkv"  # flash=fused_qkv, pro=split
    norm_topk_prob: bool = True
    layernorm_eps: float = 1e-6
    eos_token_id: int = 151645

    def bucket_of(self, layer: int) -> int:
        """Store bucket index for a MoE layer = count of MoE layers below it."""
        return sum(1 for i in range(layer) if self.moe_layer_freq[i])


def mimo_config_from_official(config_path: Path | str) -> MiMoV2Config:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tc = raw.get("text_config", raw)
    hd = int(tc["head_dim"])
    prf = float(tc.get("partial_rotary_factor", 1.0))
    swa_hd = int(tc.get("swa_head_dim", hd))
    full = AttnGeometry(
        n_heads=int(tc["num_attention_heads"]),
        n_kv_heads=int(tc["num_key_value_heads"]),
        head_dim=hd,
        v_head_dim=int(tc.get("v_head_dim", hd)),
        rope_dim=int(hd * prf),
        rope_theta=float(tc.get("rope_theta", 10_000_000.0)),
    )
    swa = AttnGeometry(
        n_heads=int(tc.get("swa_num_attention_heads", tc["num_attention_heads"])),
        n_kv_heads=int(tc.get("swa_num_key_value_heads",
                             tc["num_key_value_heads"])),
        head_dim=swa_hd,
        v_head_dim=int(tc.get("swa_v_head_dim", tc.get("v_head_dim", hd))),
        rope_dim=int(swa_hd * prf),
        rope_theta=float(tc.get("swa_rope_theta",
                                tc.get("rope_theta", 10_000_000.0))),
    )
    freq = tc.get("moe_layer_freq")
    if freq is None:
        freq = [0] + [1] * (int(tc["num_hidden_layers"]) - 1)
    return MiMoV2Config(
        vocab_size=int(tc["vocab_size"]),
        hidden=int(tc["hidden_size"]),
        n_layers=int(tc["num_hidden_layers"]),
        dense_inter=int(tc["intermediate_size"]),
        full_attn=full, swa_attn=swa,
        attention_value_scale=float(
            tc.get("attention_value_scale") or 1.0),
        sliding_window=int(tc.get("sliding_window") or 0),
        hybrid_pattern=tuple(int(x) for x in tc["hybrid_layer_pattern"]),
        swa_sink=bool(tc.get("add_swa_attention_sink_bias", False)),
        full_sink=bool(tc.get("add_full_attention_sink_bias", False)),
        n_routed=int(tc["n_routed_experts"]),
        topk=int(tc["num_experts_per_tok"]),
        n_group=int(tc.get("n_group", 1)),
        topk_group=int(tc.get("topk_group", tc.get("n_group", 1))),
        routed_scaling=float(tc.get("routed_scaling_factor") or 1.0),
        moe_inter=int(tc["moe_intermediate_size"]),
        moe_layer_freq=tuple(int(x) for x in freq),
        projection_layout=str(tc.get("attention_projection_layout",
                                     "fused_qkv")),
        norm_topk_prob=bool(tc.get("norm_topk_prob", True)),
        layernorm_eps=float(tc.get("layernorm_epsilon", 1e-6)),
        eos_token_id=int(tc.get("eos_token_id") or 151645),
    )


# ---------------------------------------------------------------------------
# Weight decoding helpers
# ---------------------------------------------------------------------------

FP8_BLOCK = 128


def dequantize_fp8_e4m3_block(weight: torch.Tensor,
                              scale_inv: torch.Tensor) -> torch.Tensor:
    """F8_E4M3 [out,in] + F32 scale_inv [ceil(out/128), ceil(in/128)] -> F32.

    ``scale_inv`` is already a float multiplier (DeepSeek-style fp8
    weight_scale_inv), NOT e8m0 — no exponent decode.
    """
    if weight.dtype != torch.float8_e4m3fn:
        weight = weight.view(torch.float8_e4m3fn)
    values = weight.float()
    out, in_dim = values.shape
    expanded = (scale_inv.float()
                .repeat_interleave(FP8_BLOCK, dim=0)
                .repeat_interleave(FP8_BLOCK, dim=1))[:out, :in_dim]
    return values * expanded


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF MiMoV2RMSNorm semantics: fp32 variance, weight * normed -> input dtype."""
    dt = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (weight.float() * (xf * torch.rsqrt(var + eps))).to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryCache:
    """Precomputed cos/sin for positions 0..max_seq on the rope subspace."""

    def __init__(self, rope_dim: int, theta: float, max_seq: int,
                 device: str, dtype: torch.dtype):
        inv = 1.0 / (theta ** (
            torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        pos = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(pos, inv)                    # [seq, rope_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)          # [seq, rope_dim]
        self.cos = emb.cos().to(device=device, dtype=dtype)   # [seq, rope_dim]
        self.sin = emb.sin().to(device=device, dtype=dtype)

    def apply(self, q_or_k: torch.Tensor, start_pos: int) -> torch.Tensor:
        """q_or_k: [b, heads, s, rope_dim]."""
        s = q_or_k.shape[2]
        cos = self.cos[start_pos: start_pos + s].unsqueeze(0).unsqueeze(0)
        sin = self.sin[start_pos: start_pos + s].unsqueeze(0).unsqueeze(0)
        return q_or_k * cos + _rotate_half(q_or_k) * sin


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


class MiMoV2Layer:
    """One decoder layer: attention (full|swa+sink) + (dense MLP | MoE)."""

    def __init__(self, cfg: MiMoV2Config, layer_id: int,
                 weights: dict[str, torch.Tensor], *,
                 engine: Any = None, device: str = "cuda:0",
                 is_moe: bool = True, is_swa: bool = False,
                 geo: Optional[AttnGeometry] = None,
                 rope: Optional[RotaryCache] = None,
                 diagnostics: bool = True):
        self.cfg = cfg
        self.layer_id = layer_id
        self.device = device
        self.is_moe = is_moe
        self.is_swa = is_swa
        self.geo = geo or cfg.full_attn
        self.engine = engine
        self.rope = rope
        self.diagnostics = diagnostics
        self.w = weights
        self.last_route: dict[str, Any] = {}
        # KV cache: post-rope K [kv, s, head_dim], v-scaled V [kv, s, v_head_dim]
        self._k_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None
        self.stats = {"requests": 0, "engine_calls": 0}

    def reset_state(self) -> None:
        self._k_cache = None
        self._v_cache = None
        self.last_route = {}

    # -- attention ----------------------------------------------------------
    def _attention(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        cfg, geo = self.cfg, self.geo
        b, s, _ = x.shape
        dev = x.device
        qkv = x @ self.w["qkv"].T                     # [b,s,q_size+k+v]
        q_size = geo.n_heads * geo.head_dim
        k_size = geo.n_kv_heads * geo.head_dim
        v_size = geo.n_kv_heads * geo.v_head_dim
        q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
        q = q.view(b, s, geo.n_heads, geo.head_dim).transpose(1, 2)
        k = k.view(b, s, geo.n_kv_heads, geo.head_dim).transpose(1, 2)
        v = v.view(b, s, geo.n_kv_heads, geo.v_head_dim).transpose(1, 2)
        if cfg.attention_value_scale != 1.0:
            v = v * cfg.attention_value_scale
        # partial rope on the first rope_dim dims
        rd = geo.rope_dim
        q = torch.cat([self.rope.apply(q[..., :rd], start_pos),
                       q[..., rd:]], dim=-1)
        k = torch.cat([self.rope.apply(k[..., :rd], start_pos),
                       k[..., rd:]], dim=-1)
        # KV cache append
        if self._k_cache is None:
            self._k_cache, self._v_cache = k, v
        else:
            self._k_cache = torch.cat([self._k_cache, k], dim=2)
            self._v_cache = torch.cat([self._v_cache, v], dim=2)
        kc, vc = self._k_cache, self._v_cache
        kv_len = kc.shape[2]
        # SWA slicing is a decode-only optimization: with s>1 (prefill) each
        # query keeps its own window, so the full cache must stay visible and
        # the mask below enforces locality per query position.
        kfirst = 0
        if self.is_swa and s == 1 and kv_len > cfg.sliding_window:
            kfirst = kv_len - cfg.sliding_window
            kc = kc[:, :, kfirst:]
            vc = vc[:, :, kfirst:]
            kv_len = cfg.sliding_window
        # GQA expand
        rep = geo.n_heads // geo.n_kv_heads
        kf = (kc[:, :, None, :, :]
              .expand(b, geo.n_kv_heads, rep, kv_len, geo.head_dim)
              .reshape(b, geo.n_heads, kv_len, geo.head_dim))
        vf = (vc[:, :, None, :, :]
              .expand(b, geo.n_kv_heads, rep, kv_len, geo.v_head_dim)
              .reshape(b, geo.n_heads, kv_len, geo.v_head_dim))
        scores = torch.matmul(q, kf.transpose(2, 3)) * (
            geo.head_dim ** -0.5)                      # [b,h,s,kv]
        # mask: causal (prefill) — decode s=1 rows attend all kv
        if s > 1:
            qpos = torch.arange(start_pos, start_pos + s, device=dev)[:, None]
            kpos = torch.arange(kfirst, kfirst + kv_len, device=dev)[None]
            if self.is_swa:
                mask = (kpos > qpos) | (qpos - kpos >= cfg.sliding_window)
            else:
                mask = kpos > qpos
            scores = scores.masked_fill(mask[None, None], float("-inf"))
        if "sink" in self.w:
            sink = self.w["sink"].float().view(1, geo.n_heads, 1, 1)
            scores = torch.cat([scores.float(), sink.expand(
                b, geo.n_heads, s, 1)], dim=-1)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32)[..., :-1]
        else:
            scores = scores.float()
            probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs.to(vf.dtype), vf)     # [b,h,s,v_d]
        out = out.transpose(1, 2).reshape(b, s, geo.n_heads * geo.v_head_dim)
        return out @ self.w["o"].T

    # -- FFN ----------------------------------------------------------------
    def _dense_mlp(self, x: torch.Tensor) -> torch.Tensor:
        gate = x @ self.w["dense_gate"].T
        up = x @ self.w["dense_up"].T
        return (F.silu(gate) * up) @ self.w["dense_down"].T

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, s, h = x.shape
        xf = x.reshape(-1, h).float()
        logits = xf @ self.w["gate"].float().T         # [t, n_routed]
        scores = torch.sigmoid(logits)
        # noaux_tc: select on sigmoid+correction_bias, group-limited
        chosen = scores + self.w["gate_bias"].float()  # [t, n_routed]
        if cfg.n_group > 1:
            t = chosen.shape[0]
            group_scores = (chosen.view(t, cfg.n_group, -1)
                            .topk(2, dim=-1)[0].sum(dim=-1))
            gidx = torch.topk(group_scores, cfg.topk_group, dim=-1,
                              sorted=False)[1]
            gmask = torch.zeros_like(group_scores).scatter_(1, gidx, 1)
            smask = (gmask.unsqueeze(-1)
                     .expand(t, cfg.n_group, cfg.n_routed // cfg.n_group)
                     .reshape(t, -1)).bool()
            chosen = chosen.masked_fill(~smask, float("-inf"))
        idx = torch.topk(chosen, cfg.topk, dim=-1, sorted=False)[1]
        w = scores.gather(1, idx)
        if cfg.topk > 1 and cfg.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        if cfg.routed_scaling != 1.0:
            w = w * cfg.routed_scaling
        self.last_route = {
            "expert_ids": idx.tolist(),
            "routing_weights": w.tolist(),
        }
        # Engine dispatch: one moe_forward_batch over the token batch.
        import numpy as np
        h_np = xf.detach().to("cpu", torch.float32).numpy()
        ids_np = idx.to(torch.int32).cpu().numpy()
        raw_np = self.engine.moe_forward_batch(
            cfg.bucket_of(self.layer_id), h_np, ids_np)
        self.stats["engine_calls"] += 1
        raw = torch.from_numpy(np.asarray(raw_np)).to(
            x.device)                                # [t, topk, h]
        combined = (raw * w.to(torch.float32)[..., None]).sum(dim=1)
        return combined.to(x.dtype).view(b, s, h)

    def forward(self, x: torch.Tensor, start_pos: int,
                input_ids: Optional[torch.Tensor] = None,
                capture: Optional[dict[str, Any]] = None) -> torch.Tensor:
        h = rms_norm(x, self.w["input_norm"], self.cfg.layernorm_eps)
        h = self._attention(h, start_pos)
        x = x + h
        h = rms_norm(x, self.w["post_norm"], self.cfg.layernorm_eps)
        h = self._moe(h) if self.is_moe else self._dense_mlp(h)
        return x + h


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MiMoV2Model:
    """mimo_v2 forward + greedy decode; layers live on one or two devices."""

    def __init__(self, cfg: MiMoV2Config, *, embed: torch.Tensor,
                 layers: list[MiMoV2Layer], norm_w: torch.Tensor,
                 head: torch.Tensor, split: int = 48,
                 device0: str = "cuda:0", device1: str = "cuda:1",
                 diagnostics: bool = True):
        self.cfg = cfg
        self.embed = embed
        self.layers = layers
        self.norm_w = norm_w
        self.head = head
        self.split = split
        self.device0 = device0
        self.device1 = device1
        self.diagnostics = diagnostics
        self.execution_trace: list[dict[str, Any]] = []

    @classmethod
    def build(cls, cfg: MiMoV2Config, source: TensorSource, *,
              device0: str = "cuda:0", device1: Optional[str] = None,
              engine0: Any = None, engine1: Any = None,
              split: Optional[int] = None,
              dense_dtype: torch.dtype = torch.bfloat16,
              max_seq_len: int = 8192,
              diagnostics: bool = True) -> "MiMoV2Model":
        """Load dense tensors (dequantizing fp8 blocks) and build layers.

        ``source`` resolves raw checkpoint tensors by name; expert bytes are
        NOT touched here — the engine reads them from the DEE4 store.
        """
        n = cfg.n_layers
        dev1 = device1 or device0
        split = split if split is not None else (n + 1) // 2
        names = set(source.tensor_names()) if hasattr(
            source, "tensor_names") else None

        def has(name: str) -> bool:
            if names is not None:
                return name in names
            try:
                source.tensor_identity(name)
                return True
            except KeyError:
                return False

        def linear(prefix: str) -> torch.Tensor:
            """Dense weight: fp8-block-dequant when scale_inv ships, else raw."""
            w = source.get_tensor(prefix + ".weight")
            if has(prefix + ".weight_scale_inv"):
                w = dequantize_fp8_e4m3_block(
                    w, source.get_tensor(prefix + ".weight_scale_inv"))
            return w

        embed = source.get_tensor("model.embed_tokens.weight").to(
            dense_dtype)
        head = source.get_tensor("lm_head.weight").to(dense_dtype)
        norm_w = source.get_tensor("model.norm.weight").to(dense_dtype)
        ropes: dict[tuple[str, bool], RotaryCache] = {}

        def rope_for(device: str, swa: bool) -> RotaryCache:
            key = (device, swa)
            if key not in ropes:
                g = cfg.swa_attn if swa else cfg.full_attn
                ropes[key] = RotaryCache(g.rope_dim, g.rope_theta,
                                         max_seq_len, device, dense_dtype)
            return ropes[key]

        layers: list[MiMoV2Layer] = []
        for layer in range(n):
            device = device0 if layer < split else dev1
            engine = engine0 if layer < split else engine1
            is_moe = bool(cfg.moe_layer_freq[layer]) \
                if cfg.moe_layer_freq else layer > 0
            is_swa = bool(cfg.hybrid_pattern[layer]) if cfg.hybrid_pattern \
                else False
            geo = cfg.swa_attn if is_swa else cfg.full_attn
            p = f"model.layers.{layer}."
            w: dict[str, torch.Tensor] = {
                "input_norm": source.get_tensor(p + "input_layernorm.weight"),
                "post_norm": source.get_tensor(
                    p + "post_attention_layernorm.weight"),
                "o": linear(p + "self_attn.o_proj"),
            }
            if cfg.projection_layout == "split":
                qkv = torch.cat([linear(p + f"self_attn.{n_}proj")
                                 for n_ in ("q_", "k_", "v_")], dim=0)
            else:
                qkv = linear(p + "self_attn.qkv_proj")
            w["qkv"] = qkv
            if (is_swa and cfg.swa_sink) or (not is_swa and cfg.full_sink):
                w["sink"] = source.get_tensor(
                    p + "self_attn.attention_sink_bias").float()
            if is_moe:
                w["gate"] = source.get_tensor(p + "mlp.gate.weight")
                w["gate_bias"] = source.get_tensor(
                    p + "mlp.gate.e_score_correction_bias")
            else:
                w["dense_gate"] = linear(p + "mlp.gate_proj")
                w["dense_up"] = linear(p + "mlp.up_proj")
                w["dense_down"] = linear(p + "mlp.down_proj")
            w = {k: (v.to(dense_dtype).to(device) if v.is_floating_point()
                     else v.to(device)) for k, v in w.items()}
            layers.append(MiMoV2Layer(cfg, layer, w, engine=engine,
                                      device=device, is_moe=is_moe,
                                      is_swa=is_swa, geo=geo,
                                      rope=rope_for(device, is_swa),
                                      diagnostics=diagnostics))
        return cls(cfg, embed=embed.to(device0), layers=layers,
                   norm_w=norm_w.to(dev1), head=head.to(dev1),
                   split=split, device0=device0, device1=dev1,
                   diagnostics=diagnostics)

    # -- runtime ------------------------------------------------------------
    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()
        self.execution_trace = []

    def forward(self, input_ids: torch.Tensor, start_pos: int) \
            -> torch.Tensor:
        cfg = self.cfg
        self.execution_trace = []
        h = F.embedding(input_ids.to(self.device0), self.embed)
        for idx, layer in enumerate(self.layers):
            h = layer.forward(h, start_pos)
            if idx == self.split - 1 and self.device1 != self.device0:
                h = h.to(self.device1)
            if self.diagnostics:
                finite = bool(torch.isfinite(h).all())
                self.execution_trace.append({
                    "layer": idx, "device": str(h.device),
                    "shape": list(h.shape), "finite": finite,
                    "selected_experts": layer.last_route.get("expert_ids"),
                })
                if not finite:
                    raise FloatingPointError(
                        f"non-finite hidden state after layer {idx}")
        h = rms_norm(h, self.norm_w, cfg.layernorm_eps)
        logits = h[:, -1].float() @ self.head.float().T
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite logits")
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, *,
                 eos_id: Optional[int] = None,
                 decode_timings_ms: Optional[list[float]] = None,
                 prompt_sha: bool = True) -> dict[str, Any]:
        """Greedy decode; returns token ids + per-step diagnostics."""
        import time
        cfg_eos = self.cfg.eos_token_id if eos_id is None else eos_id
        seq = input_ids
        logits = self.forward(seq, 0)
        first = int(logits.argmax(-1).item())
        generated = [first]
        total_len = input_ids.shape[1]
        for _ in range(max_new_tokens - 1):
            if generated[-1] == cfg_eos:
                break
            step_ids = torch.tensor([[generated[-1]]],
                                    dtype=input_ids.dtype)
            t0 = time.monotonic()
            logits = self.forward(step_ids, total_len)
            if decode_timings_ms is not None:
                decode_timings_ms.append((time.monotonic() - t0) * 1000.0)
            generated.append(int(logits.argmax(-1).item()))
            total_len += 1
        return {"tokens": generated, "prompt_len": int(input_ids.shape[1])}
