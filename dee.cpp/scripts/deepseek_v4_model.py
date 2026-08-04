"""DS10: full-model DeepSeek-V4-Flash-0731 execution on dual T4.

Brings the sealed DS9 one-layer discipline to the FULL model:

- every layer variant executes (ratio 0 / 4 / 128 and hash-routed gates
  0..2), each as a :class:`~deepseek_v4_layer_reference.DeepseekV4Layer`
  built on the layer's device;
- the routed + shared FFN runs through the sealed DS8 cache runtime
  (``DeepSeekExpertCache`` + ``DeepSeekExpertLoader``) with FP16-expanded
  payloads; cache misses materialize on demand through an ``ExpertProvider``;
- the model splits across two GPUs (contiguous layer split), with an explicit
  host-staged inter-GPU hidden-state handoff (stream + event + checksum);
- prefill + greedy autoregressive decode, embedding -> 43 layers -> hc_head
  -> final norm -> LM head -> logits, exactly in the official order.

Precision (documented in DS10_FULL_MODEL_CONTRACT.json): dense weights are
dequantized from the official F8/BF16 storage and executed in FP16 on the
GPU; embed/head stay BF16 as stored; experts are FP16-expanded (DS8 family
C); accumulation is FP32 everywhere.  Explicitly NON-bitwise.  The reference
for spot checks is CPU FP32 recomputation of individual components.

No latency interpretation: ``performance_comparable`` stays false.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import torch

from scripts import deepseek_v4_layer_common as common
from scripts import deepseek_v4_layer_reference as layer_ref
from scripts import deepseek_v4_layer_candidate as v4cand
from scripts import deepseek_v4_moe_reference as moe
from scripts import deepseek_v4_expert_reference as ds7
from scripts import deepseek_v4_support as v4support

OFFICIAL_REPOSITORY = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
EXPECTED_SHARD_COUNT = 48

DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.uint8,      # packed-FP4 / scale raw bytes kept as uint8
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.int64,     # torch has no uint64; safetensors I64/U64 both 8B
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": getattr(torch, "float8_e5m2fn", torch.float8_e4m3fn),
    "F8_E8M0": torch.float8_e8m0fnu,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Tensor sources: a checkpoint backend that can load one tensor by name.
# ---------------------------------------------------------------------------


class TensorSource(Protocol):
    """A backend that can load one official checkpoint tensor by name."""

    def get_tensor(self, name: str) -> torch.Tensor:
        ...

    def tensor_identity(self, name: str) -> dict[str, Any]:
        ...


class CommittedHeaderSource:
    """Base: resolves (shard, offset, length, dtype, shape) from the
    committed shard headers (``shard-headers/model-XXXX-of-00048.json``),
    which the DS10 harness verifies against the pinned revision.

    ``absolute_offset`` requires the 8-byte safetensors length prefix per
    shard; ``_fetch_bytes(name)`` is implemented by subclasses.
    """

    def __init__(self, headers_dir: Path | str, revision: str = OFFICIAL_REVISION):
        self.headers_dir = Path(headers_dir)
        self.revision = revision
        self._headers: dict[str, dict[str, Any]] = {}
        self._tensors: dict[str, dict[str, Any]] = {}
        self._prefix_len: dict[str, int] = {}
        self._load_headers()

    def _load_headers(self) -> None:
        for shard in (f"model-{i:05d}-of-00048.safetensors"
                      for i in range(1, EXPECTED_SHARD_COUNT + 1)):
            path = self.headers_dir / f"{shard}.json"
            if not path.is_file():
                raise FileNotFoundError(f"committed shard header missing: {path}")
            header = json.loads(path.read_text(encoding="utf-8"))
            self._headers[shard] = header
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                self._tensors[name] = {
                    "shard": shard,
                    "offset": int(meta["data_offsets"][0]),
                    "length": int(meta["data_offsets"][1] - meta["data_offsets"][0]),
                    "dtype": meta["dtype"],
                    "shape": [int(d) for d in meta["shape"]],
                }

    def tensor_identity(self, name: str) -> dict[str, Any]:
        row = self._tensors.get(name)
        if row is None:
            raise KeyError(f"tensor not in committed headers: {name}")
        return dict(row)

    def tensor_names(self) -> list[str]:
        return sorted(self._tensors)

    def absolute_range(self, name: str) -> tuple[str, int, int]:
        row = self.tensor_identity(name)
        shard = row["shard"]
        if shard not in self._prefix_len:
            self._prefix_len[shard] = self._fetch_prefix_len(shard)
        start = 8 + self._prefix_len[shard] + row["offset"]
        return shard, start, row["length"]

    def _fetch_prefix_len(self, shard: str) -> int:
        raise NotImplementedError

    def _fetch_bytes(self, name: str) -> bytes:
        raise NotImplementedError

    def get_tensor(self, name: str) -> torch.Tensor:
        row = self.tensor_identity(name)
        data = self._fetch_bytes(name)
        expect = row["length"]
        if len(data) != expect:
            raise RuntimeError(
                f"{name}: fetched {len(data)} bytes, expected {expect}")
        try:
            dtype = DTYPE_MAP[row["dtype"]]
        except KeyError:
            raise ValueError(f"{name}: unsupported dtype {row['dtype']}") from None
        t = torch.frombuffer(data, dtype=dtype).reshape(row["shape"])
        return t.clone()


class RemoteTensorSource(CommittedHeaderSource):
    """Per-tensor HTTP Range fetches from the pinned HF revision.

    The 8-byte safetensors length prefixes are fetched once per shard; tensor
    bytes are fetched on demand with a small thread pool and retry/backoff.
    """

    def __init__(self, headers_dir: Path | str, *,
                 revision: str = OFFICIAL_REVISION,
                 repository: str = OFFICIAL_REPOSITORY,
                 max_workers: int = 16,
                 max_attempts: int = 6):
        super().__init__(headers_dir, revision=revision)
        self.repository = repository
        self.max_workers = max_workers
        self.max_attempts = max_attempts
        self.stats: dict[str, int] = {"requests": 0, "bytes": 0, "retries": 0}
        self._fetch_prefix_len = self._remote_prefix_len

    def _url(self, shard: str) -> str:
        return (f"https://huggingface.co/{self.repository}/resolve/"
                f"{self.revision}/{shard}")

    def _remote_prefix_len(self, shard: str) -> int:
        url = self._url(shard)
        data = self._range_request(url, 0, 7)
        if len(data) != 8:
            raise RuntimeError(f"{shard}: bad prefix {len(data)} bytes")
        hlen = int.from_bytes(data[:8], "little")
        if hlen <= 0 or hlen > (1 << 31):
            raise RuntimeError(f"{shard}: implausible header length {hlen}")
        return hlen

    def _range_request(self, url: str, start: int, end: int) -> bytes:
        import time
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    if resp.status != 206:
                        raise RuntimeError(
                            f"server did not honor Range (status {resp.status})")
                    data = resp.read()
                self.stats["requests"] += 1
                self.stats["bytes"] += len(data)
                return data
            except (urllib.error.HTTPError, urllib.error.URLError,
                    ConnectionError, TimeoutError) as exc:
                last_error = exc
                self.stats["retries"] += 1
                time.sleep(1.5 * (2 ** attempt))
        raise ConnectionError(
            f"range fetch failed after {self.max_attempts} attempts: "
            f"{last_error!r}")

    def _fetch_bytes(self, name: str) -> bytes:
        shard, start, length = self.absolute_range(name)
        return self._range_request(self._url(shard), start, start + length - 1)


class LocalDirTensorSource(CommittedHeaderSource):
    """Reads tensors from safetensors shard files already on disk."""

    def __init__(self, headers_dir: Path | str, shards_dir: Path | str, *,
                 revision: str = OFFICIAL_REVISION):
        super().__init__(headers_dir, revision=revision)
        self.shards_dir = Path(shards_dir)
        self._handles: dict[str, Any] = {}
        self._prefix: dict[str, int] = {}

    def _fetch_prefix_len(self, shard: str) -> int:
        if shard not in self._prefix:
            path = self.shards_dir / shard
            if not path.is_file():
                raise FileNotFoundError(f"shard not on disk: {path}")
            with open(path, "rb") as fh:
                raw = fh.read(8)
            self._prefix[shard] = int.from_bytes(raw, "little")
        return self._prefix[shard]

    def _fetch_bytes(self, name: str) -> bytes:
        row = self.tensor_identity(name)
        shard = row["shard"]
        if shard not in self._handles:
            from safetensors import safe_open
            path = self.shards_dir / shard
            if not path.is_file():
                raise FileNotFoundError(f"shard not on disk: {path}")
            self._handles[shard] = safe_open(str(path), framework="pt",
                                             device="cpu")
        return self._handles[shard].get_slice(name)[:].tobytes()


class DictTensorSource:
    """In-memory backend for tests / synthetic models."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = {k: v for k, v in tensors.items()}

    def get_tensor(self, name: str) -> torch.Tensor:
        if name not in self.tensors:
            raise KeyError(f"tensor not in dict source: {name}")
        return self.tensors[name].clone()

    def tensor_identity(self, name: str) -> dict[str, Any]:
        t = self.get_tensor(name)
        return {"shape": list(t.shape), "dtype": str(t.dtype),
                "source": "dict"}


# ---------------------------------------------------------------------------
# Model configuration (official config.json values).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3
    n_heads: int = 64
    n_routed: int = 256
    n_shared: int = 1
    topk: int = 6
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-6
    compress_rope_theta: float = 160000.0
    max_seq_len: int = 4096
    compress_ratios: tuple[int, ...] = field(default_factory=tuple)

    def layer_config(self, layer_id: int) -> layer_ref.LayerConfig:
        if self.compress_ratios:
            ratio = self.compress_ratios[layer_id]
        else:
            ratio = 4
        return layer_ref.LayerConfig(
            hidden=self.dim,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            o_groups=self.o_groups,
            window_size=self.window_size,
            compress_ratio=ratio,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            n_routed=self.n_routed,
            topk=self.topk,
            route_scale=self.route_scale,
            swiglu_limit=self.swiglu_limit,
            norm_eps=self.norm_eps,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            compress_rope_theta=self.compress_rope_theta,
            original_seq_len=self.original_seq_len,
            rope_factor=self.rope_factor,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            hash_layer=layer_id < self.n_hash_layers,
        )


def model_config_from_official(config_path: Path | str) -> ModelConfig:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return ModelConfig(
        vocab_size=int(raw["vocab_size"]),
        dim=int(raw["dim"]),
        moe_inter_dim=int(raw["moe_inter_dim"]),
        n_layers=int(raw["n_layers"]),
        n_hash_layers=int(raw["n_hash_layers"]),
        n_heads=int(raw["n_heads"]),
        n_routed=int(raw["n_routed_experts"]),
        n_shared=int(raw["n_shared_experts"]),
        topk=int(raw["n_activated_experts"]),
        route_scale=float(raw["route_scale"]),
        swiglu_limit=float(raw["swiglu_limit"]),
        q_lora_rank=int(raw["q_lora_rank"]),
        head_dim=int(raw["head_dim"]),
        rope_head_dim=int(raw["rope_head_dim"]),
        o_groups=int(raw["o_groups"]),
        o_lora_rank=int(raw["o_lora_rank"]),
        window_size=int(raw["window_size"]),
        original_seq_len=int(raw["original_seq_len"]),
        rope_theta=float(raw["rope_theta"]),
        rope_factor=float(raw["rope_factor"]),
        beta_fast=int(raw["beta_fast"]),
        beta_slow=int(raw["beta_slow"]),
        index_n_heads=int(raw["index_n_heads"]),
        index_head_dim=int(raw["index_head_dim"]),
        index_topk=int(raw["index_topk"]),
        hc_mult=int(raw["hc_mult"]),
        hc_sinkhorn_iters=int(raw["hc_sinkhorn_iters"]),
        hc_eps=float(raw.get("hc_eps", 1e-6)),
        norm_eps=float(raw.get("norm_eps", 1e-6)),
        compress_rope_theta=float(raw["compress_rope_theta"]),
        max_seq_len=4096,
        compress_ratios=tuple(int(r) for r in raw["compress_ratios"][: int(raw["n_layers"])]),
    )


# ---------------------------------------------------------------------------
# Expert provider: on-demand FP16 payload materialization for cache misses.
# ---------------------------------------------------------------------------


class ExpertProvider:
    """Materializes FP16-expanded routed/shared payloads on demand.

    ``fp16_payloads``/``shared_payload`` caches keep per-layer prebuilt
    payloads (tests / prefill discovery); misses fall through to
    ``source``-based dequantization.
    """

    def __init__(self, source: TensorSource, *,
                 fp16_payloads: Optional[dict[int, dict[int, dict[str, torch.Tensor]]]] = None,
                 shared_payloads: Optional[dict[int, dict[str, torch.Tensor]]] = None):
        self.source = source
        self.fp16_payloads = fp16_payloads or {}
        self.shared_payloads = shared_payloads or {}
        self.fetch_count = 0

    def get_fp16_payload(self, layer: int, expert_id: int) -> dict[str, torch.Tensor]:
        layer_map = self.fp16_payloads.get(layer, {})
        if expert_id in layer_map:
            return layer_map[expert_id]
        names = v4support.routed_expert_tensor_names(layer, expert_id)
        raw = {n[len(f"layers.{layer}.ffn.experts.{expert_id}."):]:
               self.source.get_tensor(n) for n in names}
        payload = v4cand.build_fp16_payloads({expert_id: raw})[expert_id]
        layer_map[expert_id] = payload
        self.fp16_payloads[layer] = layer_map
        self.fetch_count += 1
        return payload

    def get_shared_fp16_payload(self, layer: int) -> dict[str, torch.Tensor]:
        if layer in self.shared_payloads:
            return self.shared_payloads[layer]
        names = v4support.shared_expert_tensor_names(layer)
        raw = {n[len(f"layers.{layer}.ffn.shared_experts."):]:
               self.source.get_tensor(n) for n in names}
        payload = v4cand.build_shared_fp16_payload(raw)
        self.shared_payloads[layer] = payload
        self.fetch_count += 1
        return payload


# ---------------------------------------------------------------------------
# Full model.
# ---------------------------------------------------------------------------


class DeepseekV4Model:
    """Full official model on two GPUs with an explicit handoff."""

    def __init__(self, cfg: ModelConfig, *, embed: torch.Tensor,
                 layers0: list[Any], layers1: list[Any],
                 hc_head_fn: torch.Tensor, hc_head_base: torch.Tensor,
                 hc_head_scale: torch.Tensor, norm_w: torch.Tensor,
                 head: torch.Tensor, device0: str, device1: str):
        self.cfg = cfg
        self.device0 = device0
        self.device1 = device1
        self.embed = embed
        self.layers0 = layers0
        self.layers1 = layers1
        self.split = len(layers0)
        self.hc_head_fn = hc_head_fn
        self.hc_head_base = hc_head_base
        self.hc_head_scale = hc_head_scale
        self.norm_w = norm_w
        self.head = head
        self.handoff_stats: dict[str, Any] = {}
        self.captures: dict[int, dict[str, Any]] = {}

    # -- construction ------------------------------------------------------
    @classmethod
    def build_candidate(cls, cfg: ModelConfig, source: TensorSource, *,
                        device0: str = "cuda:0", device1: str = "cuda:1",
                        cache0: Any, loader0: Any, cache1: Any, loader1: Any,
                        provider: ExpertProvider,
                        dense_dtype: torch.dtype = torch.float16,
                        embed_head_dtype: torch.dtype = torch.bfloat16,
                        split: Optional[int] = None) -> "DeepseekV4Model":
        """Build the dual-GPU candidate.

        Dense weights are dequantized from official storage and cast to
        ``dense_dtype`` (FP16 by default); embed/head/norm stay at
        ``embed_head_dtype`` (BF16 as stored).  Returns the model.
        """
        n_layers = cfg.n_layers
        split = split if split is not None else (n_layers + 1) // 2

        embed = source.get_tensor("embed.weight").to(embed_head_dtype)
        head = source.get_tensor("head.weight").to(embed_head_dtype)
        norm_w = source.get_tensor("norm.weight").to(embed_head_dtype)
        hc_head_fn = source.get_tensor("hc_head_fn").float().to(device1)
        hc_head_base = source.get_tensor("hc_head_base").float().to(device1)
        hc_head_scale = source.get_tensor("hc_head_scale").float().to(device1)

        layers0: list[Any] = []
        layers1: list[Any] = []
        for layer in range(n_layers):
            device = device0 if layer < split else device1
            cache = cache0 if layer < split else cache1
            loader = loader0 if layer < split else loader1
            lcfg = cfg.layer_config(layer)
            names = v4support.layer_dense_tensor_names(
                layer, hash_layer=layer < cfg.n_hash_layers,
                compress_ratio=lcfg.compress_ratio)
            raw = {name: source.get_tensor(name) for name in names}
            w = layer_ref.build_layer_weights_from_tensors(
                raw, layer, hash_layer=layer < cfg.n_hash_layers,
                compress_ratio=lcfg.compress_ratio)
            # Cast the assembled dense tree (already FP32) to the execution
            # dtype; integer tensors (hash tid2eid) stay exact.
            w = _cast_floats(w, dense_dtype)
            w_cuda = _move_to_device(w, device)
            shared_raw = {}
            for name in v4support.shared_expert_tensor_names(layer):
                shared_raw[name[len(f"layers.{layer}.ffn.shared_experts."):]] = (
                    source.get_tensor(name))
            shared_payload = v4cand.build_shared_fp16_payload(shared_raw)
            layer_obj = v4cand.make_candidate_layer(
                lcfg, w_cuda, device=device, max_batch=1, cache=cache,
                loader=loader, layer_id=layer, fp16_payloads={},
                shared_payload=shared_payload, provider=provider)
            (layers0 if layer < split else layers1).append(layer_obj)

        return cls(cfg, embed=embed.to(device0), layers0=layers0,
                   layers1=layers1, hc_head_fn=hc_head_fn,
                   hc_head_base=hc_head_base, hc_head_scale=hc_head_scale,
                   norm_w=norm_w.to(device1), head=head.to(device1),
                   device0=device0, device1=device1)

    # -- runtime -----------------------------------------------------------
    def reset_state(self) -> None:
        for layer in self.layers0 + self.layers1:
            layer.reset_state()
        self.handoff_stats = {}

    def layer(self, layer_id: int) -> Any:
        if layer_id < self.split:
            return self.layers0[layer_id]
        return self.layers1[layer_id - self.split]

    def device_of(self, layer_id: int) -> str:
        return self.device0 if layer_id < self.split else self.device1

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
        mixes = xf @ self.hc_head_fn.transpose(0, 1) * rsqrt
        pre = torch.sigmoid(
            mixes * self.hc_head_scale + self.hc_head_base) + cfg.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype)

    def _handoff(self, h: torch.Tensor, dst: str) -> torch.Tensor:
        """Host-staged inter-GPU hidden-state handoff with checksum.

        CPU/same-device (local tests) is a no-op copy with trivially equal
        checksums; CUDA crosses devices through a host-staged copy guarded by
        stream events.
        """
        h_src = h.detach()
        pre_sha = _sha256_bytes(
            h_src.float().cpu().contiguous().numpy().tobytes())
        cuda_cross = (h.is_cuda and dst != str(h.device)
                      and torch.cuda.is_available())
        if cuda_cross:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            h_dst = h_src.to(dst, non_blocking=True)
            torch.cuda.current_stream(dst).wait_event(ev)
            torch.cuda.synchronize(dst)
        else:
            h_dst = h_src.to(dst)
        post_sha = _sha256_bytes(
            h_dst.float().cpu().contiguous().numpy().tobytes())
        self.handoff_stats = {
            "source_device": str(h.device),
            "destination_device": dst,
            "dtype": str(h.dtype),
            "shape": list(h.shape),
            "bytes": h.numel() * h.element_size(),
            "method": "host-staged D2D copy" if cuda_cross else "same-device copy",
            "pre_checksum": pre_sha,
            "post_checksum": post_sha,
            "checksum_bitwise_equal": pre_sha == post_sha,
        }
        return h_dst

    def forward(self, input_ids: torch.Tensor, start_pos: int,
                captures: Optional[dict[int, dict[str, Any]]] = None) -> torch.Tensor:
        """Full forward; returns logits [b, vocab] for the LAST token."""
        cfg = self.cfg
        b, s = input_ids.shape
        h = torch.nn.functional.embedding(
            input_ids.to(self.device0), self.embed)
        h = h.unsqueeze(2).expand(b, s, cfg.hc_mult, h.size(-1)).contiguous()
        for idx, layer in enumerate(self.layers0):
            cap = captures.setdefault(idx, {}) if captures is not None else None
            h = layer.forward(h, start_pos, input_ids.to(self.device0),
                              capture=cap)
        h = self._handoff(h, self.device1)
        for idx, layer in enumerate(self.layers1):
            cap = (captures.setdefault(self.split + idx, {})
                   if captures is not None else None)
            h = layer.forward(h, start_pos, input_ids.to(self.device1),
                              capture=cap)
        h = self._hc_head(h)
        h = common.rms_norm(h, self.norm_w, cfg.norm_eps)
        logits = h[:, -1].float() @ self.head.float().transpose(0, 1)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, *,
                 eos_id: int = 1, captures: Optional[dict[int, dict[str, Any]]] = None,
                 per_step_captures: Optional[list[dict[int, dict[str, Any]]]] = None,
                 trace: Optional[dict[str, Any]] = None) -> list[int]:
        """Greedy autoregressive decode.  Returns generated token ids."""
        seq_len = input_ids.shape[1]
        logits = self.forward(input_ids, 0, captures=captures)
        tok = int(logits.argmax(-1).item())
        generated = [tok]
        if trace is not None:
            trace["token_0"] = {"start_pos": 0, "token_id": tok,
                                "logits_finite": bool(torch.isfinite(logits).all()),
                                "top5": logits.topk(5, -1)[1][0].tolist()}
        for t in range(1, max_new_tokens):
            if tok == eos_id:
                break
            step_ids = torch.tensor([[tok]], device=input_ids.device,
                                    dtype=input_ids.dtype)
            cap_map = per_step_captures[t] if per_step_captures else None
            logits = self.forward(step_ids, seq_len + t - 1, captures=cap_map)
            tok = int(logits.argmax(-1).item())
            generated.append(tok)
            if trace is not None:
                trace[f"token_{t}"] = {
                    "start_pos": seq_len + t - 1, "token_id": tok,
                    "logits_finite": bool(torch.isfinite(logits).all()),
                    "top5": logits.topk(5, -1)[1][0].tolist()}
        return generated

    def state_buffers(self, layer_ids: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        return {lid: self.layer(lid).state_buffers() for lid in layer_ids}

    def state_signatures(self, layer_ids: list[int]) -> dict[int, dict[str, str]]:
        return {lid: self.layer(lid).state_signature() for lid in layer_ids}

    def per_gpu_memory_plan(self, cache_budgets: dict[str, int]) -> dict[str, Any]:
        """Resident-bytes estimate per GPU (dense + embed/head + cache budget
        + state + activation estimates)."""
        plan: dict[str, dict[str, Any]] = {}
        for dev, layers in ((self.device0, self.layers0),
                            (self.device1, self.layers1)):
            dense = sum(_walk(layer.w) for layer in layers)
            extra = 0
            if dev == self.device0:
                extra += self.embed.numel() * self.embed.element_size()
            else:
                extra += (self.head.numel() * self.head.element_size()
                          + self.norm_w.numel() * self.norm_w.element_size()
                          + self.hc_head_fn.numel() * 4
                          + self.hc_head_base.numel() * 4
                          + self.hc_head_scale.numel() * 4)
            states = 0
            for layer in layers:
                attn = layer.attn
                states += attn.kv_cache.numel() * attn.kv_cache.element_size()
                if attn.compressor is not None:
                    states += (attn.compressor.kv_state.numel() * 4
                               + attn.compressor.score_state.numel() * 4)
                if attn.indexer is not None:
                    states += attn.indexer.kv_cache.numel() * 2
            plan[dev] = {
                "dense_bytes": dense,
                "extra_persistent_bytes": extra,
                "state_bytes": states,
                "cache_budget_bytes": cache_budgets.get(dev, 0),
                "activation_estimate_bytes": 64 << 20,
                "context_estimate_bytes": 512 << 20,
                "total_estimate_bytes": dense + extra + states
                + cache_budgets.get(dev, 0) + (64 << 20) + (512 << 20),
            }
        return plan


# ---------------------------------------------------------------------------
# Module helpers.
# ---------------------------------------------------------------------------


def _cast_floats(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.dtype.is_floating_point else value
    if isinstance(value, dict):
        return {k: _cast_floats(v, dtype) for k, v in value.items()}
    return value


def _move_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _walk(value: Any) -> int:
    total = 0
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        for v in value.values():
            total += _walk(v)
    return total


# ---------------------------------------------------------------------------
# Convenience: build the full tensor-coverage audit report from a source.
# ---------------------------------------------------------------------------


def coverage_audit_report(source: TensorSource, *, n_layers: int = 43,
                          n_hash_layers: int = 3,
                          compress_ratios: Optional[tuple[int, ...]] = None) -> dict[str, Any]:
    """Reconstruct ledger rows from committed headers and run the DS10.1
    full-model coverage audit (identity only, no weight loading)."""
    rows = []
    for name in source.tensor_names():
        ident = source.tensor_identity(name)
        plan = v4support.storage_plan_for_tensor(name, ident["dtype"],
                                                 ident["shape"])
        rows.append({
            "tensor_name": name,
            "layer": v4support.layer_from_tensor_name(name),
            "module": v4support.module_from_tensor_name(name),
            "component": v4support.component_for_tensor(name),
            "source_shard": ident["shard"],
            "shape": ident["shape"],
            "stored_dtype": ident["dtype"],
            "byte_offset": ident["offset"],
            "byte_length": ident["length"],
            # scale linkage comes from storage_plan_for_tensor (F8/packed
            # weights only) -- BF16 norms carry no scale tensor.
            "scale_tensor": plan["scale_tensor"],
        })
    return v4support.full_model_coverage_audit(
        rows, n_layers=n_layers, n_hash_layers=n_hash_layers,
        compress_ratios=compress_ratios)


def static_memory_plan(cfg: ModelConfig, source: TensorSource, *,
                       budgets0: int, budgets1: int, split: Optional[int] = None,
                       dense_bytes_per_elt: int = 2) -> dict[str, Any]:
    """Per-GPU resident-bytes plan computed purely from committed headers
    (identity only -- zero checkpoint bytes downloaded).

    Dense layer weights execute in ``dense_bytes_per_elt`` bytes/element
    (FP16 default, DS10 family C); embed/head/norm stay at storage bytes;
    expert bytes are the cache budget, not the checkpoint size.  State bytes
    come from the exact state shapes (kv_cache, compressor fp32 states,
    indexer bf16 kv).  Activations/context are fixed estimates.
    """
    n_layers = cfg.n_layers
    split = split if split is not None else (n_layers + 1) // 2

    def dense_layer_bytes(layer: int) -> int:
        lcfg = cfg.layer_config(layer)
        names = v4support.layer_dense_tensor_names(
            layer, hash_layer=layer < cfg.n_hash_layers,
            compress_ratio=lcfg.compress_ratio)
        total = 0
        for name in names:
            ident = source.tensor_identity(name)
            if name.endswith(".scale"):
                continue
            # identity shape is the STORED logical shape; execution is FP16
            # so the resident dense cost is numel * 2 bytes for weights
            # (norms/ape stay fp32 as the layer reference keeps them).
            elts = 1
            for d in ident["shape"]:
                elts *= d
            total += elts * dense_bytes_per_elt
        return total

    def state_layer_bytes(layer: int) -> int:
        lcfg = cfg.layer_config(layer)
        ratio = lcfg.compress_ratio
        attn_kv = lcfg.kv_cache_len * cfg.head_dim * 2  # bf16
        total = attn_kv
        if ratio:
            rows = (1 + (ratio == 4)) * ratio
            cols = (1 + (ratio == 4)) * cfg.head_dim
            total += 2 * rows * cols * 4  # compressor fp32 kv+score states
        if ratio == 4:
            total += (cfg.max_seq_len // ratio) * cfg.index_head_dim * 2
        return total

    groups = {"dense": [0] * n_layers, "state": [0] * n_layers}
    for layer in range(n_layers):
        groups["dense"][layer] = dense_layer_bytes(layer)
        groups["state"][layer] = state_layer_bytes(layer)

    def gpu_plan(dev: str, layer_ids: list[int], budget: int,
                 extra: int) -> dict[str, Any]:
        dense = sum(groups["dense"][l] for l in layer_ids)
        state = sum(groups["state"][l] for l in layer_ids)
        act, ctx = 64 << 20, 512 << 20
        total = dense + state + budget + extra + act + ctx
        return {
            "layers": layer_ids,
            "dense_bytes": dense,
            "state_bytes": state,
            "extra_persistent_bytes": extra,
            "cache_budget_bytes": budget,
            "activation_estimate_bytes": act,
            "context_estimate_bytes": ctx,
            "total_estimate_bytes": total,
            "total_estimate_gib": total / (1 << 30),
        }

    embed_extra = (cfg.vocab_size * cfg.dim * 2)  # bf16 embed on GPU 0
    head_extra = (cfg.vocab_size * cfg.dim * 2 + cfg.dim * 2)  # head + norm
    hc_extra = 0
    plan = {
        "split": split,
        "n_layers": n_layers,
        "devices": {
            "cuda:0": gpu_plan("cuda:0", list(range(0, split)), budgets0,
                                embed_extra + hc_extra),
            "cuda:1": gpu_plan("cuda:1", list(range(split, n_layers)), budgets1,
                                head_extra),
        },
        "per_layer_dense_bytes": groups["dense"],
        "per_layer_state_bytes": groups["state"],
    }
    return plan
