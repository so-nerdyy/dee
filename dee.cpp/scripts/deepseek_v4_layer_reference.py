"""DS9 trusted reference: one complete official DeepSeek-V4-Flash-0731 layer.

Faithful port of the pinned official inference code for layer 20
(``compress_ratios[20] == 4``, so the FULL path is exercised):

- Hyper-Connections (mHC): hc_pre / hc_post with the split-Sinkhorn comb
  matrix (``Block.hc_pre/hc_post`` + ``kernel.hc_split_sinkhorn``).
- Compressor (``Compressor``): gated overlap pooling of the KV cache,
  ratio-4 overlapping windows, learned ape, in-place fp8 act-quant.
- Indexer (``Indexer``): low-rank Q + Hadamard rotation + fp4 act-quant,
  compressed-KV scoring, top-``index_topk`` position selection.
- Attention (``Attention``): low-rank Q, grouped low-rank O, sliding-window
  + compressed sparse attention with the learnable sink (denominator only).
- Router (``Gate``): sqrtsoftplus, bias-shifted selection, non-softmax
  normalization, route_scale.
- FFN: top-6 routed experts (packed FP4 storage) + 1 shared expert
  (F8_E4M3 storage), official weight placement and clamps.

Numerical policy (consistent with DS7/DS8): dense weights are dequantized to
FP32; the reference computes in FP32 with the official dtype discipline
(bf16 activations where the official code stores bf16, fp32 where it computes
in fp32) INCLUDING the in-model QAT simulation points (act_quant / fp4
round trips).  The official fp8 weight-quantization inside ``linear()`` is
NOT re-applied (documented deviation: trusted full-precision-weight
reference, same convention as DS7/DS8).  The FFN executes in FP32 through the
DS8 MoE reference.

The layer is stateful (compressor/indexer incremental state, kv cache); call
``reset_state`` between independent sequences.  ``state_signature`` returns
content hashes of every mutable state tensor for reference-vs-candidate
state agreement checks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from scripts import deepseek_v4_layer_common as common
from scripts import deepseek_v4_moe_reference as moe


@dataclass(frozen=True)
class LayerConfig:
    """Official config values for the DS9 target layer (layer 20)."""

    hidden: int = 4096
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128
    compress_ratio: int = 4
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    n_routed: int = 256
    topk: int = 6
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    norm_eps: float = 1e-6
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    # DS10 full-model variants: per-layer compress_ratio (0 = pure sliding
    # window, 128 = compressor without overlap, 4 = full compressor+indexer)
    # and hash routing (first n_hash_layers).
    hash_layer: bool = False

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def kv_cache_len(self) -> int:
        if self.compress_ratio:
            return self.window_size + self.max_seq_len // self.compress_ratio
        return self.window_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


def tensor_sha256(t: torch.Tensor) -> str:
    """Content hash of a tensor's fp32 bytes (for state agreement checks).

    Moves to CPU first so CUDA-resident state tensors (the remote candidate)
    can be hashed; the implicit synchronize is acceptable for a state gate.
    """
    return hashlib.sha256(
        t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Compressor (official Compressor, ratio-4 overlap path)
# ---------------------------------------------------------------------------


class DeepseekV4Compressor:
    def __init__(self, cfg: LayerConfig, head_dim: int, *, rotate: bool = False,
                 device: str = "cpu", max_batch: int = 1):
        self.cfg = cfg
        self.head_dim = head_dim
        self.ratio = cfg.compress_ratio
        self.overlap = cfg.compress_ratio == 4
        self.rotate = rotate
        self.coff = 1 + self.overlap
        self.device = device
        self.max_batch = max_batch
        self.kv_cache: Optional[torch.Tensor] = None  # assigned by owner
        self.freqs_cis: Optional[torch.Tensor] = None
        rows = self.coff * self.ratio
        cols = self.coff * head_dim
        self.kv_state = torch.zeros(max_batch, rows, cols, dtype=torch.float32,
                                    device=device)
        self.score_state = torch.full((max_batch, rows, cols), float("-inf"),
                                      dtype=torch.float32, device=device)

    def reset_state(self) -> None:
        self.kv_state.zero_()
        self.score_state.fill_(float("-inf"))
        self.kv_cache = None

    def overlap_transform(self, tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        # tensor: [b, g, r, 2d] -> [b, g, 2r, d]
        b, s, _, _ = tensor.size()
        ratio, d = self.ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int,
                w: dict[str, torch.Tensor], capture: Optional[dict[str, Any]] = None) -> Optional[torch.Tensor]:
        if self.kv_cache is None:
            raise RuntimeError("compressor kv_cache not assigned")
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = (self.ratio, self.overlap, self.head_dim,
                                 self.cfg.rope_head_dim)
        dtype = x.dtype
        xf = x.float()
        kv = xf @ w["wkv.weight"].float().transpose(0, 1)
        score = xf @ w["wgate.weight"].float().transpose(0, 1)
        ape = w["ape"]
        should_compress = seqlen >= ratio if start_pos == 0 else (start_pos + 1) % ratio == 0
        # DS9 v7 device-authentic boundary captures: the RAW pre-write
        # projections (kv, score), the input, the ape, and the pre-write
        # state snapshots.  These let the focused diagnostic locate the first
        # divergent intermediate: if raw kv/score are bitwise identical but
        # the written state differs, the bug is in the write/slot/mask path;
        # if raw kv/score already differ, the divergence is in the matmul.
        if capture is not None:
            capture["compressor_input"] = x
            capture["compressor_kv_raw"] = kv.detach().clone()
            capture["compressor_score_raw"] = score.detach().clone()
            capture["compressor_ape"] = ape.detach().clone()
            capture["compressor_kv_state_pre"] = self.kv_state.detach().clone()
            capture["compressor_score_state_pre"] = self.score_state.detach().clone()
            capture["compressor_start_pos"] = int(start_pos)
            capture["compressor_should_compress"] = bool(should_compress)
        if start_pos == 0:
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio: cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio: cutoff] + ape
            if remainder > 0:
                kv_head, kv_tail = kv.split([cutoff, remainder], dim=1)
                kv = kv_head
                self.kv_state[:bsz, offset: offset + remainder] = kv_tail
                self.score_state[:bsz, offset: offset + remainder] = score[:, cutoff:] + ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + ape
            if overlap:
                kv = self.overlap_transform(kv, 0.0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            score = score + ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d],
                                          self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d],
                                             self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        normed = common.rms_norm(kv.to(dtype), w["norm.weight"], self.cfg.norm_eps)
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        common.apply_rotary_emb(normed[..., -rd:], freqs_cis)
        if self.rotate:
            normed = common.hadamard_transform(normed, normed.size(-1) ** -0.5)
            common.fp4_act_quant_inplace(normed, 32)
        else:
            common.act_quant_inplace(normed[..., :-rd], 64, "ue8m0")
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = normed
        else:
            self.kv_cache[:bsz, start_pos // ratio] = normed.squeeze(1)
        if capture is not None:
            capture["kv_compressed"] = normed
        return normed


# ---------------------------------------------------------------------------
# Indexer (official Indexer)
# ---------------------------------------------------------------------------


class DeepseekV4Indexer:
    def __init__(self, cfg: LayerConfig, *, device: str = "cpu", max_batch: int = 1):
        self.cfg = cfg
        self.device = device
        self.max_batch = max_batch
        self.softmax_scale = cfg.index_head_dim ** -0.5
        # official buffer dtype (the runtime sets the default dtype to bf16),
        # matching the bf16 q/kv used by the indexer einsum
        self.kv_cache = torch.zeros(max_batch, cfg.max_seq_len // cfg.compress_ratio,
                                    cfg.index_head_dim, device=device,
                                    dtype=torch.bfloat16)
        self.freqs_cis: Optional[torch.Tensor] = None
        self.compressor = DeepseekV4Compressor(cfg, cfg.index_head_dim, rotate=True,
                                               device=device, max_batch=max_batch)

    def reset_state(self) -> None:
        self.kv_cache.zero_()
        self.compressor.reset_state()

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int,
                w: dict[str, torch.Tensor],
                capture: Optional[dict[str, Any]] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        ratio = self.cfg.compress_ratio
        rd = self.cfg.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = (qr.float() @ w["wq_b.weight"].float().transpose(0, 1)).to(qr.dtype)
        q = q.unflatten(-1, (self.cfg.index_n_heads, self.cfg.index_head_dim))
        common.apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = common.hadamard_transform(q, self.cfg.index_head_dim ** -0.5)
        common.fp4_act_quant_inplace(q, 32)
        # DS9 v7: pass a NESTED capture so the indexer compressor's raw
        # boundary values land under capture["indexer_compressor"] and do not
        # collide with the main attention compressor's captures.
        icap: Optional[dict[str, Any]] = {} if capture is not None else None
        self.compressor.forward(x, start_pos, w["compressor"], capture=icap)
        if capture is not None and icap is not None:
            capture["indexer_compressor"] = icap
        weights = (x.float() @ w["weights_proj.weight"].float().transpose(0, 1)).to(x.dtype)
        weights = weights * (self.softmax_scale * self.cfg.index_n_heads ** -0.5)
        index_score = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio])
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        dev = x.device
        if start_pos == 0:
            mask = (torch.arange(seqlen // ratio, device=dev).repeat(seqlen, 1)
                    >= torch.arange(1, seqlen + 1, device=dev).unsqueeze(1)
                    // ratio)
            index_score = index_score + torch.where(
                mask, float("-inf"), 0.0)
        topk_idxs = index_score.topk(
            min(self.cfg.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(
                1, seqlen + 1, device=dev).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        if capture is not None:
            capture["indexer_scores"] = index_score
            capture["indexer_q"] = q
        return topk_idxs.to(torch.int32)


# ---------------------------------------------------------------------------
# Attention (official Attention, compress_ratio == 4)
# ---------------------------------------------------------------------------


class DeepseekV4Attention:
    def __init__(self, cfg: LayerConfig, *, device: str = "cpu", max_batch: int = 1):
        self.cfg = cfg
        self.device = device
        self.max_batch = max_batch
        self.compress_ratio = cfg.compress_ratio
        # DS10 generalization: the compressor exists only when the layer has
        # a compress_ratio; the indexer only for ratio-4 (official code:
        # ``if self.compress_ratio == 4: self.indexer = Indexer(...)``).
        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = DeepseekV4Compressor(
                cfg, cfg.head_dim, device=device, max_batch=max_batch)
            if self.compress_ratio == 4:
                self.indexer = DeepseekV4Indexer(cfg, device=device,
                                                 max_batch=max_batch)
        # official buffer dtype (runtime default is bf16)
        self.kv_cache = torch.zeros(max_batch, cfg.kv_cache_len, cfg.head_dim,
                                    device=device, dtype=torch.bfloat16)
        # rope variant: ratio-0 layers use base rope_theta WITHOUT YaRN
        # (official Attention: ``if self.compress_ratio: original_seq_len,
        # rope_theta = args.original_seq_len, args.compress_rope_theta else:
        # 0, args.rope_theta``).
        if self.compress_ratio:
            rope_original, rope_theta = cfg.original_seq_len, cfg.compress_rope_theta
        else:
            rope_original, rope_theta = 0, cfg.rope_theta
        # precompute_freqs_cis is lru_cached on CPU; move the result to this
        # layer's device (v4 crashed at common.py apply_rotary_emb with
        # 'found at least two devices, cuda:0 and cpu!' because the cached
        # freqs_cis stayed on CPU while the CUDA candidate's q was on cuda).
        self.freqs_cis = common.precompute_freqs_cis(
            cfg.rope_head_dim, cfg.max_seq_len, rope_original, rope_theta,
            cfg.rope_factor, cfg.beta_fast, cfg.beta_slow).to(device)

    def reset_state(self) -> None:
        self.kv_cache.zero_()
        if self.compressor is not None:
            self.compressor.reset_state()
        if self.indexer is not None:
            self.indexer.reset_state()

    def forward(self, x: torch.Tensor, start_pos: int, w: dict[str, torch.Tensor],
                capture: Optional[dict[str, Any]] = None) -> torch.Tensor:
        cfg = self.cfg
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        win = cfg.window_size
        ratio = self.compress_ratio
        rd = cfg.rope_head_dim
        if ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # q path (official: qr = q_norm(wq_a(x)))
        qr = common.rms_norm(
            (x.float() @ w["wq_a.weight"].float().transpose(0, 1)).to(x.dtype),
            w["q_norm.weight"], cfg.norm_eps)
        q = (qr.float() @ w["wq_b.weight"].float().transpose(0, 1)).to(x.dtype)
        q = q.unflatten(-1, (cfg.n_heads, cfg.head_dim))
        rsqrt = torch.rsqrt(q.float().square().mean(-1, keepdim=True) + cfg.norm_eps)
        q = q * rsqrt.to(q.dtype)
        common.apply_rotary_emb(q[..., -rd:], freqs_cis)
        # kv path
        kv = (x.float() @ w["wkv.weight"].float().transpose(0, 1)).to(x.dtype)
        kv = common.rms_norm(kv, w["kv_norm.weight"], cfg.norm_eps)
        common.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        common.act_quant_inplace(kv[..., :-rd], 64, "ue8m0")
        # indices (get_window_topk_idxs builds on CPU; move to x.device so the
        # torch.cat with the CUDA compress_idxs below stays same-device)
        window_idxs = common.get_window_topk_idxs(
            win, bsz, seqlen, start_pos).to(x.device)
        if ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_idxs = self.indexer.forward(
                    x, qr, start_pos, offset, w["indexer"], capture=capture)
            else:
                compress_idxs = common.get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset).to(x.device)
            topk_idxs = torch.cat([window_idxs, compress_idxs], dim=-1)
        else:
            topk_idxs = window_idxs
        # compress kv
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                kv_tail, kv_head = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
                self.kv_cache[:bsz, cutoff: win] = kv_tail
                self.kv_cache[:bsz, :cutoff] = kv_head
            if capture is not None:
                capture["kv_win"] = kv
            if ratio:
                kv_compress = self.compressor.forward(
                    x, start_pos, w["compressor"], capture)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = common.sparse_attn(q, kv, w["attn_sink"], topk_idxs,
                                   cfg.head_dim ** -0.5)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if ratio:
                self.compressor.forward(x, start_pos, w["compressor"], capture)
            if capture is not None:
                capture["kv"] = self.kv_cache[:bsz]
            o = common.sparse_attn(q, self.kv_cache[:bsz], w["attn_sink"],
                                   topk_idxs, cfg.head_dim ** -0.5)
        common.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        # o projections
        o = o.view(bsz, seqlen, cfg.o_groups, -1)
        wo_a = w["wo_a.weight"].view(cfg.o_groups, cfg.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float())
        o = o.to(x.dtype)
        out = (o.flatten(2).float() @ w["wo_b.weight"].float().transpose(0, 1)).to(x.dtype)
        if capture is not None:
            capture["qr"] = qr
            capture["q"] = q
            capture["kv"] = kv
            capture["attn_window_idxs"] = window_idxs
            if ratio:
                capture["attn_compress_idxs"] = compress_idxs
            capture["attn_topk_idxs"] = topk_idxs
            capture["attn_o"] = o
            capture["attn_out"] = out
        return out


# ---------------------------------------------------------------------------
# Layer (official Block: HC pre -> attn -> HC post -> HC pre -> FFN -> HC post)
# ---------------------------------------------------------------------------


class DeepseekV4Layer:
    """One complete official layer. ``ffn_fn`` must accept (x_bf16 [b,s,d],
    input_ids, capture) and return fp32 [b,s,d]; the fp32 backend is default.
    """

    def __init__(self, cfg: LayerConfig, w: dict[str, torch.Tensor], *,
                 device: str = "cpu", max_batch: int = 1,
                 ffn_fn: Optional[Any] = None, layer_id: int = -1,
                 profile_stages: bool = False):
        self.cfg = cfg
        self.device = device
        self.layer_id = int(layer_id)
        self.profile_stages = bool(profile_stages)
        self._profile_rows: list[dict[str, Any]] = []
        self.w = w
        self.attn = DeepseekV4Attention(cfg, device=device, max_batch=max_batch)
        self.ffn_fn = ffn_fn if ffn_fn is not None else self._ffn_fp32_direct

    def reset_state(self) -> None:
        self.attn.reset_state()
        self._profile_rows = []
        reset_ffn_profile = getattr(self.ffn_fn, "reset_stage_profile", None)
        if reset_ffn_profile is not None:
            reset_ffn_profile()

    def stage_profile_snapshot(self) -> dict[str, Any]:
        """Resolve profiling-only CUDA events after the measured generation."""
        phase_names = (
            "attention_prep", "attention_state", "ffn_prep",
            "routed_and_shared_ffn", "ffn_hc_post")
        rows: list[dict[str, Any]] = []
        totals = {name: 0.0 for name in phase_names}
        per_start_pos: dict[str, dict[str, float]] = {}
        for stored in self._profile_rows:
            events = stored["events"]
            durations = {
                name: float(events[index].elapsed_time(events[index + 1]))
                for index, name in enumerate(phase_names)
            }
            for name, milliseconds in durations.items():
                totals[name] += milliseconds
            start_key = str(stored["start_pos"])
            position_totals = per_start_pos.setdefault(
                start_key, {name: 0.0 for name in phase_names})
            for name, milliseconds in durations.items():
                position_totals[name] += milliseconds
            rows.append({
                "start_pos": stored["start_pos"],
                "sequence_length": stored["sequence_length"],
                "durations_ms": {
                    name: round(milliseconds, 6)
                    for name, milliseconds in durations.items()
                },
            })
        # Merge the native FFN's finer-grained stage events (router, routed
        # dispatch/native, combine, shared expert, output cast) into the same
        # totals so the full-model decomposition separates routed MoE from the
        # shared expert as the storage-matrix contract requires.
        ffn_snapshot_fn = getattr(self.ffn_fn, "stage_profile_snapshot", None)
        if ffn_snapshot_fn is not None:
            ffn_snapshot = ffn_snapshot_fn()
            if ffn_snapshot.get("enabled"):
                for name, milliseconds in ffn_snapshot["totals_ms"].items():
                    totals[name] = totals.get(name, 0.0) + float(milliseconds)
                for start_key, values in ffn_snapshot["per_start_pos_ms"].items():
                    position_totals = per_start_pos.setdefault(
                        start_key, {name: 0.0 for name in phase_names})
                    for name, milliseconds in values.items():
                        position_totals[name] = (
                            position_totals.get(name, 0.0) + float(milliseconds))
        return {
            "layer": self.layer_id,
            "device": str(self.device),
            "calls": len(rows),
            "totals_ms": {
                name: round(milliseconds, 6)
                for name, milliseconds in totals.items()
            },
            "per_start_pos_ms": {
                key: {
                    name: round(milliseconds, 6)
                    for name, milliseconds in values.items()
                }
                for key, values in per_start_pos.items()
            },
            "calls_detail": rows,
        }

    def _ffn_fp32_direct(self, x: torch.Tensor, input_ids: torch.Tensor,
                         capture: Optional[dict[str, Any]]) -> torch.Tensor:
        cfg = self.cfg
        ffn_w = self.w["ffn"]
        # DS10 generalization: hash-routed layers (first n_hash_layers)
        # select experts via the learned tid2eid table instead of the bias.
        tid2eid = ffn_w.get("tid2eid")
        out = moe.moe_layer_forward(
            x.reshape(-1, cfg.hidden).float(), ffn_w["gate_w"], ffn_w["gate_b"],
            ffn_w["routed"], ffn_w["shared"], topk=cfg.topk,
            route_scale=cfg.route_scale, score_func="sqrtsoftplus",
            swiglu_limit=cfg.swiglu_limit, keep_per_expert=True,
            tid2eid=tid2eid, input_ids=input_ids)
        if capture is not None:
            capture["router_scores"] = out["scores"]
            capture["router_bias_scores"] = out["bias_scores"]
            capture["expert_ids"] = out["expert_ids"]
            capture["routing_weights"] = out["routing_weights"]
            capture["moe_out"] = out["moe_output"].view(x.shape)
            capture["shared_out"] = out["shared_output"].view(x.shape)
        # Official MoE.forward returns y.type_as(x): the fp32 accumulation is
        # cast back to the input dtype (bf16) BEFORE hc_post consumes it, so
        # the layer's final output stays bf16.
        return out["moe_output"].view(x.shape).to(x.dtype)

    def _hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                hc_base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.cfg.norm_eps)
        mixes = xf @ hc_fn.float().transpose(0, 1) * rsqrt
        pre, post, comb = common.hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.cfg.hc_mult,
            self.cfg.hc_sinkhorn_iters, self.cfg.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype), post, comb

    def _hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                 comb: torch.Tensor) -> torch.Tensor:
        y = (post.unsqueeze(-1) * x.unsqueeze(-2)
             + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2))
        return y.type_as(x)

    def forward(self, x_hc: torch.Tensor, start_pos: int,
                input_ids: Optional[torch.Tensor] = None,
                capture: Optional[dict[str, Any]] = None) -> torch.Tensor:
        """x_hc: [b, s, hc_mult, hidden]; returns [b, s, hc_mult, hidden]."""
        w = self.w
        c = capture if capture is not None else {}
        profile_events = None
        if self.profile_stages and x_hc.is_cuda:
            stream = torch.cuda.current_stream(x_hc.device)
            profile_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(6)
            ]
            profile_events[0].record(stream)
        c["layer_input"] = x_hc
        residual = x_hc
        x, post, comb = self._hc_pre(x_hc, w["hc_attn_fn"], w["hc_attn_scale"],
                                     w["hc_attn_base"])
        c["attn_norm_in"] = x
        x = common.rms_norm(x, w["attn_norm.weight"], self.cfg.norm_eps)
        c["attn_norm_out"] = x
        if profile_events is not None:
            profile_events[1].record(stream)
        x = self.attn.forward(x, start_pos, w["attn"], capture=c)
        x = self._hc_post(x, residual, post, comb)
        c["attn_hc_out"] = x
        if profile_events is not None:
            profile_events[2].record(stream)

        residual = x
        x, post, comb = self._hc_pre(x, w["hc_ffn_fn"], w["hc_ffn_scale"],
                                     w["hc_ffn_base"])
        c["ffn_norm_in"] = x
        x = common.rms_norm(x, w["ffn_norm.weight"], self.cfg.norm_eps)
        c["ffn_norm_out"] = x
        if profile_events is not None:
            profile_events[3].record(stream)
        if input_ids is None:
            input_ids = torch.zeros(x.size(0) * x.size(1), dtype=torch.long,
                                    device=x.device)
        set_ffn_pos = getattr(self.ffn_fn, "set_profile_start_pos", None)
        if set_ffn_pos is not None:
            set_ffn_pos(int(start_pos))
        x = self.ffn_fn(x, input_ids, c)
        if profile_events is not None:
            profile_events[4].record(stream)
        x = self._hc_post(x, residual, post, comb)
        c["output"] = x
        if profile_events is not None:
            profile_events[5].record(stream)
            self._profile_rows.append({
                "start_pos": int(start_pos),
                "sequence_length": int(x_hc.size(1)),
                "events": profile_events,
            })
        return x

    def state_signature(self) -> dict[str, str]:
        attn = self.attn
        sig = {"attn_kv_cache": tensor_sha256(attn.kv_cache)}
        if attn.compressor is not None:
            sig["compressor_kv_state"] = tensor_sha256(
                attn.compressor.kv_state)
            sig["compressor_score_state"] = tensor_sha256(
                attn.compressor.score_state)
        if attn.indexer is not None:
            sig["indexer_kv_cache"] = tensor_sha256(attn.indexer.kv_cache)
            sig["indexer_compressor_kv_state"] = tensor_sha256(
                attn.indexer.compressor.kv_state)
            sig["indexer_compressor_score_state"] = tensor_sha256(
                attn.indexer.compressor.score_state)
        return sig

    def state_buffers(self) -> dict[str, torch.Tensor]:
        """CPU fp32 snapshots of every mutable state buffer (for cross-device
        bounded state agreement: the reference runs on CPU, the candidate on
        CUDA, so backend ULP drift means the buffers are compared with per-
        buffer relative bounds, not bitwise).

        DS9 v9 fix: every snapshot is CLONED.  ``detach().float().cpu()`` is
        an identity for already-fp32 CPU tensors, so pre-v9 snapshots
        ALIASED the live buffers; the warm-replay ``reset_state()`` then
        zeroed/-inf-filled the aliased snapshots before the per-step gates
        compared them, manufacturing the phantom structural state divergence
        of v6/v8 (max_rel 0.387/0.548 on pristine buffers).  The boundary
        captures proved the real state writes are identical modulo 1-7 ULP
        cross-device reduction-order drift."""
        attn = self.attn
        buffers: dict[str, torch.Tensor] = {
            "attn_kv_cache": attn.kv_cache.detach().float().cpu().clone()}
        if attn.compressor is not None:
            buffers["compressor_kv_state"] = (
                attn.compressor.kv_state.detach().float().cpu().clone())
            buffers["compressor_score_state"] = (
                attn.compressor.score_state.detach().float().cpu().clone())
        if attn.indexer is not None:
            buffers["indexer_kv_cache"] = (
                attn.indexer.kv_cache.detach().float().cpu().clone())
            buffers["indexer_compressor_kv_state"] = (
                attn.indexer.compressor.kv_state.detach().float().cpu().clone())
            buffers["indexer_compressor_score_state"] = (
                attn.indexer.compressor.score_state.detach().float().cpu().clone())
        return buffers


# ---------------------------------------------------------------------------
# DS9.2 weight assembly: raw shard tensors -> the nested layer weight dict.
# ---------------------------------------------------------------------------


def build_layer_weights_from_tensors(raw: dict[str, torch.Tensor], layer: int = 20,
                                     *, hash_layer: bool = False,
                                     compress_ratio: int = 4) -> dict[str, torch.Tensor]:
    """Assemble the nested layer weight dict from raw shard tensors.

    ``raw`` maps official checkpoint names (``layers.N.*``) to tensors as
    loaded from the shard.  Dense F8_E4M3 weights are dequantized to FP32
    (block-128 e8m0 scales) via the DS8 MoE reference; BF16/F32 tensors are
    cast to FP32.  Routed expert tensors stay packed (I8 + F8_E8M0 scales)
    and the shared expert stays F8_E4M3 -- the FFN backends handle those.

    Hash layers (``hash_layer``) load ``gate.tid2eid`` instead of
    ``gate.bias``.  ``compress_ratio`` gates the compressor (ratio != 0) and
    indexer (ratio == 4) sub-modules exactly as the official code does.

    Returns the nested dict consumed by :class:`DeepseekV4Layer`.
    """
    p = f"layers.{layer}"

    def dense(name: str) -> torch.Tensor:
        """Dequantize F8_E4M3 weights with their official ``x.y.scale``;
        BF16/F32 tensors are cast to FP32."""
        t = raw[name]
        if t.dtype == torch.float8_e4m3fn:
            scale_name = name[: -len(".weight")] + ".scale"
            return moe.dequantize_fp8_e4m3(t, raw[scale_name])
        return t.float()

    w: dict[str, torch.Tensor] = {}
    w["attn_norm.weight"] = dense(f"{p}.attn_norm.weight")
    w["ffn_norm.weight"] = dense(f"{p}.ffn_norm.weight")
    for kind in ("attn", "ffn"):
        for part in ("fn", "scale", "base"):
            w[f"hc_{kind}_{part}"] = dense(f"{p}.hc_{kind}_{part}")

    attn = {
        "q_norm.weight": dense(f"{p}.attn.q_norm.weight"),
        "kv_norm.weight": dense(f"{p}.attn.kv_norm.weight"),
        "wq_a.weight": dense(f"{p}.attn.wq_a.weight"),
        "wq_b.weight": dense(f"{p}.attn.wq_b.weight"),
        "wkv.weight": dense(f"{p}.attn.wkv.weight"),
        "wo_a.weight": dense(f"{p}.attn.wo_a.weight"),
        "wo_b.weight": dense(f"{p}.attn.wo_b.weight"),
        "attn_sink": dense(f"{p}.attn.attn_sink"),
    }
    if compress_ratio:
        attn["compressor"] = {
            "wkv.weight": dense(f"{p}.attn.compressor.wkv.weight"),
            "wgate.weight": dense(f"{p}.attn.compressor.wgate.weight"),
            "norm.weight": dense(f"{p}.attn.compressor.norm.weight"),
            "ape": dense(f"{p}.attn.compressor.ape"),
        }
        if compress_ratio == 4:
            attn["indexer"] = {
                "wq_b.weight": dense(f"{p}.attn.indexer.wq_b.weight"),
                "weights_proj.weight": dense(f"{p}.attn.indexer.weights_proj.weight"),
                "compressor": {
                    "wkv.weight": dense(f"{p}.attn.indexer.compressor.wkv.weight"),
                    "wgate.weight": dense(f"{p}.attn.indexer.compressor.wgate.weight"),
                    "norm.weight": dense(f"{p}.attn.indexer.compressor.norm.weight"),
                    "ape": dense(f"{p}.attn.indexer.compressor.ape"),
                },
            }
    w["attn"] = attn
    w["ffn"] = {
        "gate_w": dense(f"{p}.ffn.gate.weight"),
        "gate_b": None,
        "tid2eid": None,
        "routed": {},  # packed FP4 tensors keyed by expert id (harness fills)
        "shared": {},  # F8_E4M3 shared expert tensors (harness fills)
    }
    if hash_layer:
        w["ffn"]["tid2eid"] = raw[f"{p}.ffn.gate.tid2eid"].long()
    else:
        w["ffn"]["gate_b"] = dense(f"{p}.ffn.gate.bias")
    return w


def make_synthetic_layer_weights(cfg: LayerConfig, *, seed: int = 0,
                                 n_experts: int = 8) -> tuple[dict[str, torch.Tensor], dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Deterministic small-scale layer weights for local ladder tests.

    Returns (w, routed_raw, shared_raw) where routed/shared match the
    harness's raw formats (packed FP4 routed, F8_E4M3 shared) so the full
    reference-vs-candidate path is exercised with tiny dimensions.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    d, hd, rd = cfg.hidden, cfg.head_dim, cfg.rope_head_dim
    qlr, olr, og = cfg.q_lora_rank, cfg.o_lora_rank, cfg.o_groups
    mix = cfg.mix_hc
    hcm = cfg.hc_mult
    inter = 2 * d  # small moe_intermediate for tests

    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=g) * 0.1)

    def make_ffn_w(kind: str) -> dict[str, torch.Tensor]:
        return {
            "w1.weight": torch.randint(0, 256, (inter, d // 2), dtype=torch.uint8, generator=g),
            "w1.scale": torch.randint(0x7D, 0x83, (inter, d // 32), dtype=torch.uint8, generator=g),
            "w2.weight": torch.randint(0, 256, (d, inter // 2), dtype=torch.uint8, generator=g),
            "w2.scale": torch.randint(0x7D, 0x83, (d, inter // 32), dtype=torch.uint8, generator=g),
            "w3.weight": torch.randint(0, 256, (inter, d // 2), dtype=torch.uint8, generator=g),
            "w3.scale": torch.randint(0x7D, 0x83, (inter, d // 32), dtype=torch.uint8, generator=g),
        } if kind == "routed" else {
            "w1.weight": rand(inter, d).to(torch.float8_e4m3fn),
            "w1.scale": torch.full(((inter + 127) // 128, (d + 127) // 128), 0x7F, dtype=torch.uint8),
            "w2.weight": rand(d, inter).to(torch.float8_e4m3fn),
            "w2.scale": torch.full(((d + 127) // 128, (inter + 127) // 128), 0x7F, dtype=torch.uint8),
            "w3.weight": rand(inter, d).to(torch.float8_e4m3fn),
            "w3.scale": torch.full(((inter + 127) // 128, (d + 127) // 128), 0x7F, dtype=torch.uint8),
        }

    w: dict[str, torch.Tensor] = {}
    w["attn_norm.weight"] = rand(d)
    w["ffn_norm.weight"] = rand(d)
    for kind in ("attn", "ffn"):
        w[f"hc_{kind}_fn"] = rand(mix, hcm * d)
        w[f"hc_{kind}_scale"] = rand(3)
        w[f"hc_{kind}_base"] = rand(mix)
    attn = {
        "q_norm.weight": rand(qlr),
        "kv_norm.weight": rand(hd),
        "wq_a.weight": rand(qlr, d),
        "wq_b.weight": rand(cfg.n_heads * hd, qlr),
        "wkv.weight": rand(hd, d),
        "wo_a.weight": rand(og * olr, cfg.n_heads * hd // og),
        "wo_b.weight": rand(d, og * olr),
        "attn_sink": rand(cfg.n_heads),
    }
    # The main compressor's output width is coff*head_dim (coff=2 only for
    # the ratio-4 overlap variant; coff=1 for ratio 128).
    coff = 2 if cfg.compress_ratio == 4 else 1
    if cfg.compress_ratio:
        attn["compressor"] = {
            "wkv.weight": rand(coff * hd, d),
            "wgate.weight": rand(coff * hd, d),
            "norm.weight": rand(hd),
            "ape": rand(cfg.compress_ratio, coff * hd),
        }
    if cfg.compress_ratio == 4:
        attn["indexer"] = {
            "wq_b.weight": rand(cfg.index_n_heads * cfg.index_head_dim, qlr),
            "weights_proj.weight": rand(cfg.index_n_heads, d),
            "compressor": {
                "wkv.weight": rand(2 * cfg.index_head_dim, d),
                "wgate.weight": rand(2 * cfg.index_head_dim, d),
                "norm.weight": rand(cfg.index_head_dim),
                "ape": rand(cfg.compress_ratio, 2 * cfg.index_head_dim),
            },
        }
    w["attn"] = attn
    routed_raw = {eid: make_ffn_w("routed") for eid in range(n_experts)}
    shared_raw = make_ffn_w("shared")
    # Rows beyond the provided expert payloads must NEVER be selected.  Pinning
    # them with a constant negative WEIGHT is unsafe: the router logit for such
    # a row is c * sum(x), which becomes hugely POSITIVE whenever a token's
    # sum(x) < 0 (sqrtsoftplus then explodes and the row wins top-k).  Instead
    # zero the unused rows (logit 0 -> constant score sqrt(softplus(0)) ~ 0.83
    # regardless of x) and pin the BIAS strongly negative, since the official
    # bias shifts only the SELECTION (router_scores: scores + bias -> topk).
    gate_w = torch.zeros((cfg.n_routed, d), dtype=torch.float32)
    gate_w[:n_experts] = rand(n_experts, d)
    gate_b = torch.full((cfg.n_routed,), -100.0, dtype=torch.float32)
    gate_b[:n_experts] = rand(n_experts) * 0.1
    w["ffn"] = {
        "gate_w": gate_w,
        "gate_b": gate_b,
        "routed": routed_raw,
        "shared": shared_raw,
    }
    return w, routed_raw, shared_raw


def make_synthetic_hash_layer_weights(cfg: LayerConfig, *, seed: int = 0,
                                      n_experts: int = 8,
                                      vocab_size: int = 32) -> tuple[dict[str, torch.Tensor], dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Hash-routed variant of :func:`make_synthetic_layer_weights`.

    Reuses the same dense/expert payloads but replaces the score gate with
    the official hash gate: ``gate.tid2eid`` (I64 [vocab, topk]) selects
    experts deterministically per token id; ``gate_b`` is None and the gate
    weight still feeds the sqrtsoftplus score function for the routing
    weights (official ``Gate.forward`` for ``layer_id < n_hash_layers``).
    """
    w, routed_raw, shared_raw = make_synthetic_layer_weights(
        cfg, seed=seed, n_experts=n_experts)
    rng = torch.Generator().manual_seed(seed + 1)
    tid2eid = torch.randint(0, n_experts, (vocab_size, cfg.topk),
                            generator=rng).long()
    w["ffn"]["gate_b"] = None
    w["ffn"]["tid2eid"] = tid2eid
    return w, routed_raw, shared_raw
