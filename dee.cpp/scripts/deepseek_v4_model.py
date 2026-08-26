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

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import torch

from scripts import deepseek_v4_layer_common as common
from scripts import deepseek_v4_layer_reference as layer_ref
from scripts import deepseek_v4_layer_candidate as v4cand
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
    # torch < 2.5 has no float8_e8m0fnu dtype.  The E8M0 decode path
    # (decode_e8m0) always reinterprets the raw bits as uint8 anyway, so
    # loading the scale bytes as uint8 is byte-identical on old torch and
    # required for the P100 repair (torch 2.3.1+cu118).
    "F8_E8M0": getattr(torch, "float8_e8m0fnu", torch.uint8),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_storage_sha256(tensor: torch.Tensor) -> str:
    """Hash the tensor's exact storage bytes without a dtype conversion."""
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return _sha256_bytes(raw)


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
        # Read the raw tensor bytes directly from the committed header's
        # absolute offset/length.  (safe_open(...).get_slice(name)[:] returns
        # a torch.Tensor whose .tobytes() was removed in modern PyTorch, and
        # the byte-level read is both faster and dtype-preserving.)
        shard, start, length = self.absolute_range(name)
        path = self.shards_dir / shard
        if not path.is_file():
            raise FileNotFoundError(f"shard not on disk: {path}")
        with open(path, "rb") as fh:
            fh.seek(start)
            data = fh.read(length)
        if len(data) != length:
            raise RuntimeError(
                f"{name}: short read {len(data)}/{length} from {path}")
        return data


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

    Explicitly supplied ``fp16_payloads``/``shared_payloads`` are used by
    tests and sealed replay.  Dynamic routed experts retain only their compact
    official tensors in a per-layer LRU; retaining every expanded FP16 expert
    reached during prefill would grow host RSS without bound.  Eight compact
    experts per layer cover the six-expert decode set plus two replacements
    while remaining inside the DS10 host budget.
    """

    def __init__(self, source: TensorSource, *,
                 fp16_payloads: Optional[dict[int, dict[int, dict[str, torch.Tensor]]]] = None,
                 shared_payloads: Optional[dict[int, dict[str, torch.Tensor]]] = None,
                 raw_experts_per_layer: int = 8,
                 raw_max_bytes: Optional[int] = None):
        if raw_experts_per_layer < 0:
            raise ValueError("raw_experts_per_layer must be nonnegative")
        if raw_max_bytes is not None and raw_max_bytes <= 0:
            raise ValueError(f"raw_max_bytes must be positive, got {raw_max_bytes}")
        self.source = source
        self.fp16_payloads = fp16_payloads or {}
        self.shared_payloads = shared_payloads or {}
        self.raw_experts_per_layer = int(raw_experts_per_layer)
        # Global byte cap on the retained raw tensors (CACHE1b): the per-layer
        # count LRU alone allowed 16 x 43 = 688 raw experts (9.2 GB host RSS)
        # which, on top of the 6.4 GB base + transient FP16 payload churn,
        # exceeded the 12 GB host ceiling.  When set, the oldest raw entry
        # across ALL layers is evicted until total retained bytes fit.
        self.raw_max_bytes = raw_max_bytes
        self.raw_payloads: dict[
            int, OrderedDict[int, dict[str, torch.Tensor]]] = {}
        self._raw_order: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.fetch_count = 0
        self.raw_hits = 0
        self.raw_misses = 0
        self.raw_evictions = 0

    def _raw_entry_bytes(self, raw: dict[str, torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in raw.values())

    def _evict_raw_order_front(self) -> None:
        """Evict the globally-oldest retained raw expert (count or byte cap)."""
        while self._raw_order:
            (lyr, eid), _ = next(iter(self._raw_order.items()))
            del self._raw_order[(lyr, eid)]
            layer_map = self.raw_payloads.get(lyr)
            if layer_map is not None:
                layer_map.pop(eid, None)
                if not layer_map:
                    self.raw_payloads.pop(lyr, None)
            self.raw_evictions += 1
            return

    def get_fp16_payload(self, layer: int, expert_id: int) -> dict[str, torch.Tensor]:
        layer_map = self.fp16_payloads.get(layer, {})
        if expert_id in layer_map:
            return layer_map[expert_id]
        raw_layer = self.raw_payloads.setdefault(layer, OrderedDict())
        raw = raw_layer.get(expert_id)
        if raw is None:
            self.raw_misses += 1
            names = v4support.routed_expert_tensor_names(layer, expert_id)
            raw = {n[len(f"layers.{layer}.ffn.experts.{expert_id}."):]:
                   self.source.get_tensor(n) for n in names}
            self.fetch_count += 1
            if self.raw_experts_per_layer:
                raw_layer[expert_id] = raw
                raw_layer.move_to_end(expert_id)
                self._raw_order[(layer, expert_id)] = None
                self._raw_order.move_to_end((layer, expert_id))
                while len(raw_layer) > self.raw_experts_per_layer:
                    # evict the oldest entry of THIS layer (per-layer count)
                    victim = next(iter(raw_layer))
                    del raw_layer[victim]
                    self._raw_order.pop((layer, victim), None)
                    self.raw_evictions += 1
                # then enforce the GLOBAL byte cap across all layers
                if self.raw_max_bytes is not None:
                    while self.raw_bytes() > self.raw_max_bytes:
                        self._evict_raw_order_front()
        else:
            self.raw_hits += 1
            raw_layer.move_to_end(expert_id)
            if (layer, expert_id) in self._raw_order:
                self._raw_order.move_to_end((layer, expert_id))
        return v4cand.build_fp16_payloads({expert_id: raw})[expert_id]

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

    def raw_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for layer in self.raw_payloads.values()
                   for payload in layer.values()
                   for t in payload.values())

    def stats(self) -> dict[str, int]:
        return {
            "fetch_count": self.fetch_count,
            "raw_hits": self.raw_hits,
            "raw_misses": self.raw_misses,
            "raw_evictions": self.raw_evictions,
            "raw_entries": sum(len(layer) for layer in self.raw_payloads.values()),
            "raw_bytes": self.raw_bytes(),
            "raw_experts_per_layer": self.raw_experts_per_layer,
            "raw_max_bytes": self.raw_max_bytes or 0,
            "shared_fp16_layers": len(self.shared_payloads),
        }


# ---------------------------------------------------------------------------
# Full model.
# ---------------------------------------------------------------------------


class DeepseekV4Model:
    """Full official model on two GPUs with an explicit handoff."""

    def __init__(self, cfg: ModelConfig, *, embed: torch.Tensor,
                 layers0: list[Any], layers1: list[Any],
                 hc_head_fn: torch.Tensor, hc_head_base: torch.Tensor,
                 hc_head_scale: torch.Tensor, norm_w: torch.Tensor,
                 head: torch.Tensor, device0: str, device1: str,
                 diagnostics: bool = True):
        self.cfg = cfg
        self.device0 = device0
        self.device1 = device1
        self.diagnostics = bool(diagnostics)
        self._handoff_buffers: list[tuple[torch.Tensor, torch.cuda.Event, torch.cuda.Event]] = []
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
        self.execution_trace: list[dict[str, Any]] = []
        self.last_execution: dict[str, Any] = {"phase": "not_started"}

    # -- construction ------------------------------------------------------
    @classmethod
    def build_candidate(cls, cfg: ModelConfig, source: TensorSource, *,
                        device0: str = "cuda:0", device1: str = "cuda:1",
                        cache0: Any, loader0: Any, cache1: Any, loader1: Any,
                        provider: ExpertProvider,
                        dense_dtype: torch.dtype = torch.float16,
                        embed_head_dtype: torch.dtype = torch.bfloat16,
                        ffn_backend: str = "cache_fp16",
                        engine0: Any = None, engine1: Any = None,
                        split: Optional[int] = None,
                        diagnostics: bool = True) -> "DeepseekV4Model":
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
            # CACHE1d: build the shared expert's FP16 payload LAZILY on the
            # first forward (provider.get_shared_fp16_payload) instead of
            # eagerly for all 43 layers here.  Eager building retained 43 x
            # 48 MiB = 2.06 GiB of host FP16 copies for the whole run, which
            # on top of the raw LRU pushed host RSS past the 12 GB ceiling.
            # The layer frees the host copy after the GPU entry is pinned,
            # so only the raw FP8 tensors (13.4 MB x 43) stay resident.
            if ffn_backend == "native":
                engine = engine0 if layer < split else engine1
                layer_obj = v4cand.make_native_candidate_layer(
                    lcfg, w_cuda, engine=engine, layer_id=layer,
                    device=device, max_batch=1, shared_payload=None,
                    provider=provider, diagnostics=diagnostics)
            else:
                layer_obj = v4cand.make_candidate_layer(
                    lcfg, w_cuda, device=device, max_batch=1, cache=cache,
                    loader=loader, layer_id=layer, fp16_payloads={},
                    shared_payload=None, provider=provider,
                    diagnostics=diagnostics)
            (layers0 if layer < split else layers1).append(layer_obj)

        model = cls(cfg, embed=embed.to(device0), layers0=layers0,
                    layers1=layers1, hc_head_fn=hc_head_fn,
                    hc_head_base=hc_head_base, hc_head_scale=hc_head_scale,
                    norm_w=norm_w.to(device1), head=head.to(device1),
                    device0=device0, device1=device1,
                    diagnostics=diagnostics)
        model.provider = provider
        model.cache0 = cache0
        model.cache1 = cache1
        return model

    # -- runtime -----------------------------------------------------------
    def reset_state(self) -> None:
        for layer in self.layers0 + self.layers1:
            layer.reset_state()
        self.handoff_stats = {}
        self.execution_trace = []
        self._handoff_buffers = [
            (host, d2h_event, h2d_event)
            for host, d2h_event, h2d_event in self._handoff_buffers
            if not h2d_event.query()
        ]
        self.last_execution = {"phase": "reset"}

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
        """Move the inter-GPU hidden state through pinned host memory.

        Validation mode uses blocking checksums. Fast mode keeps the same
        pinned-host transfer but chains the D2H and H2D copies with CUDA
        events, so the Python thread does not synchronize twice per forward.
        The host buffer is retained until the H2D completion event fires.
        """
        h_src = h.detach()
        cuda_cross = (h.is_cuda and dst != str(h.device)
                      and torch.cuda.is_available())
        if cuda_cross:
            src_device = h.device
            dst_device = torch.device(dst)
            nbytes = h.numel() * h.element_size()
            host = torch.empty(h.shape, dtype=h.dtype, device="cpu",
                               pin_memory=True)
            src_stream = torch.cuda.current_stream(src_device)
            d2h_done = torch.cuda.Event()
            with torch.cuda.stream(src_stream):
                host.copy_(h_src, non_blocking=True)
                d2h_done.record(src_stream)
            h_dst = torch.empty(h.shape, dtype=h.dtype, device=dst_device)
            dst_stream = torch.cuda.current_stream(dst_device)
            h2d_done = torch.cuda.Event()
            if self.diagnostics:
                # Keep the diagnostic path exactly synchronous and checksum it.
                d2h_done.synchronize()
                pre_sha = _tensor_storage_sha256(host)
                with torch.cuda.stream(dst_stream):
                    h_dst.copy_(host, non_blocking=True)
                    h2d_done.record(dst_stream)
                h2d_done.synchronize()
                post_sha = _tensor_storage_sha256(h_dst)
            else:
                # CUDA permits a stream on the destination device to wait on
                # an event recorded by the source device. This preserves copy
                # ordering without a host-side synchronization.
                dst_stream.wait_event(d2h_done)
                with torch.cuda.stream(dst_stream):
                    h_dst.copy_(host, non_blocking=True)
                    h2d_done.record(dst_stream)
                self._handoff_buffers = [
                    (old_host, old_d2h, old_h2d)
                    for old_host, old_d2h, old_h2d in self._handoff_buffers
                    if not old_h2d.query()
                ]
                self._handoff_buffers.append((host, d2h_done, h2d_done))
                pre_sha = None
                post_sha = None
            p2p_available = bool(torch.cuda.can_device_access_peer(
                src_device.index, dst_device.index))
            method = ("explicit_pinned_host_staging_event_driven"
                      if not self.diagnostics else
                      "explicit_pinned_host_staging")
        else:
            h_dst = h_src.to(dst)
            nbytes = h.numel() * h.element_size()
            if self.diagnostics:
                pre_sha = _tensor_storage_sha256(h_src)
                post_sha = _tensor_storage_sha256(h_dst)
            else:
                pre_sha = None
                post_sha = None
            p2p_available = False
            method = "same_device_copy"
        self.handoff_stats = {
            "source_device": str(h.device),
            "destination_device": dst,
            "dtype": str(h.dtype),
            "shape": list(h.shape),
            "bytes": nbytes,
            "d2h_bytes": nbytes if cuda_cross else 0,
            "h2d_bytes": nbytes if cuda_cross else 0,
            "d2d_bytes": 0,
            "method": method,
            "pinned_host_buffer": bool(cuda_cross),
            "peer_to_peer_available": p2p_available,
            "source_event_recorded": bool(cuda_cross),
            "destination_event_waited": bool(cuda_cross),
            "pre_checksum": pre_sha,
            "post_checksum": post_sha,
            "checksum_bitwise_equal": (
                pre_sha == post_sha if self.diagnostics else None),
        }
        return h_dst

    def forward(self, input_ids: torch.Tensor, start_pos: int,
                 captures: Optional[dict[int, dict[str, Any]]] = None,
                 post_layer_hook: Optional[Any] = None) -> torch.Tensor:
        """Full forward; returns logits [b, vocab] for the LAST token.

        ``post_layer_hook`` (CACHE1g): optional ``callable(layer_id)`` run
        after every layer.  The cache1 stage uses it to bound host RSS DURING
        a forward (gc.collect + malloc_trim per layer).  Without it, the
        transient FP16 payload churn of a full 43-layer prefill accumulates
        until the per-token hygiene, spiking ru_maxrss past the ceiling.
        """
        cfg = self.cfg
        b, s = input_ids.shape
        self.execution_trace = []
        self.last_execution = {"phase": "embedding", "start_pos": start_pos,
                               "sequence_length": s}
        input_ids0 = input_ids.to(self.device0)
        h = torch.nn.functional.embedding(input_ids0, self.embed)
        h = h.unsqueeze(2).expand(b, s, cfg.hc_mult, h.size(-1)).contiguous()
        for idx, layer in enumerate(self.layers0):
            self.last_execution = {"phase": "layer", "layer": idx,
                                   "device": self.device0}
            cap = captures.setdefault(idx, {}) if captures is not None else None
            h = layer.forward(h, start_pos, input_ids0, capture=cap)
            if self.diagnostics:
                finite = bool(torch.isfinite(h).all())
                row = {"layer": idx, "order": len(self.execution_trace),
                       "device": str(h.device), "dtype": str(h.dtype),
                       "shape": list(h.shape), "finite": finite,
                       "selected_experts": layer.ffn_fn.last_route.get("expert_ids"),
                       "routing_weights": layer.ffn_fn.last_route.get("routing_weights"),
                       "ffn_cache_counters": dict(layer.ffn_fn.stats)}
                self.execution_trace.append(row)
                if not finite:
                    raise FloatingPointError(
                        f"non-finite hidden state after layer {idx}")
            if post_layer_hook is not None:
                post_layer_hook(idx)
        self.last_execution = {"phase": "handoff", "after_layer": self.split - 1,
                               "before_layer": self.split}
        h = self._handoff(h, self.device1)
        input_ids1 = input_ids.to(self.device1)
        for idx, layer in enumerate(self.layers1):
            layer_id = self.split + idx
            self.last_execution = {"phase": "layer", "layer": layer_id,
                                   "device": self.device1}
            cap = (captures.setdefault(self.split + idx, {})
                   if captures is not None else None)
            h = layer.forward(h, start_pos, input_ids1, capture=cap)
            if self.diagnostics:
                finite = bool(torch.isfinite(h).all())
                row = {"layer": layer_id, "order": len(self.execution_trace),
                       "device": str(h.device), "dtype": str(h.dtype),
                       "shape": list(h.shape), "finite": finite,
                       "selected_experts": layer.ffn_fn.last_route.get("expert_ids"),
                       "routing_weights": layer.ffn_fn.last_route.get("routing_weights"),
                       "ffn_cache_counters": dict(layer.ffn_fn.stats)}
                self.execution_trace.append(row)
                if not finite:
                    raise FloatingPointError(
                        f"non-finite hidden state after layer {layer_id}")
            if post_layer_hook is not None:
                post_layer_hook(layer_id)
        self.last_execution = {"phase": "hc_head", "after_layer": cfg.n_layers - 1}
        h = self._hc_head(h)
        self.last_execution = {"phase": "final_norm"}
        h = common.rms_norm(h, self.norm_w, cfg.norm_eps)
        self.last_execution = {"phase": "lm_head"}
        logits = h[:, -1].float() @ self.head.float().transpose(0, 1)
        # Keep the fail-closed guarantee in fast mode with one final check per
        # forward, rather than synchronizing after every layer.
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("non-finite logits")
        self.last_execution = {"phase": "complete", "layers_executed":
                               len(self.execution_trace)}
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, *,
                 eos_id: int = 1, captures: Optional[dict[int, dict[str, Any]]] = None,
                 per_step_captures: Optional[list[dict[int, dict[str, Any]]]] = None,
                 trace: Optional[dict[str, Any]] = None,
                 decode_timings_ms: Optional[list[float]] = None,
                 post_step_hook: Optional[Any] = None,
                 post_layer_hook: Optional[Any] = None) -> list[int]:
        """Greedy autoregressive decode.  Returns generated token ids.

        ``decode_timings_ms`` (CACHE1): appends one wall-clock sample per
        generated token (decode step only, trace/serialization excluded),
        enabling an honest decode-only TPS without attributing evidence
        overhead to the cache policy.

        ``post_step_hook`` (CACHE1b): optional ``callable(step_index)`` or
        ``callable(step_index, token_id)`` run after each forward (prefill +
        every decode step).  The cache1 stage uses it to bound process RSS
        DURING decode (gc + cuda empty_cache + malloc_trim), because
        ``ru_maxrss`` is a high-water mark: trimming only before the snapshot
        cannot undo a peak already reached.  The v12 native-generate path
        passes the 2-arg form to checkpoint the token stream so an OOM kill
        (v9/v11 lost ALL tokens) still leaves the exact generated sequence.
        """
        import inspect
        _hook_arity = (len(inspect.signature(post_step_hook).parameters)
                       if post_step_hook is not None else 0)
        def _run_hook(step: int, token_id: int) -> None:
            if post_step_hook is None:
                return
            if _hook_arity >= 2:
                post_step_hook(step, token_id)
            else:
                post_step_hook(step)
        seq_len = input_ids.shape[1]
        t0 = time.monotonic()
        logits = self.forward(input_ids, 0, captures=captures,
                              post_layer_hook=post_layer_hook)
        t1 = time.monotonic()
        if decode_timings_ms is not None:
            decode_timings_ms.append((t1 - t0) * 1000.0)
        tok = int(logits.argmax(-1).item())
        generated = [tok]
        if post_step_hook is not None:
            _run_hook(0, tok)
        if trace is not None:
            trace["token_0"] = {"start_pos": 0, "token_id": tok,
                                 "logits_finite": bool(torch.isfinite(logits).all()),
                                 "top5": logits.topk(5, -1)[1][0].tolist(),
                                 "layers": list(self.execution_trace),
                                 "handoff": dict(self.handoff_stats),
                                 "state_hashes": self.state_signatures(
                                     list(range(self.cfg.n_layers)))}
        for t in range(1, max_new_tokens):
            if tok == eos_id:
                break
            step_ids = torch.tensor([[tok]], device=input_ids.device,
                                    dtype=input_ids.dtype)
            cap_map = per_step_captures[t] if per_step_captures else None
            t0 = time.monotonic()
            logits = self.forward(step_ids, seq_len + t - 1, captures=cap_map,
                                  post_layer_hook=post_layer_hook)
            t1 = time.monotonic()
            if decode_timings_ms is not None:
                decode_timings_ms.append((t1 - t0) * 1000.0)
            tok = int(logits.argmax(-1).item())
            generated.append(tok)
            if post_step_hook is not None:
                _run_hook(t, tok)  # CACHE1b RSS-trim/checkpoint hook
            if trace is not None:
                trace[f"token_{t}"] = {
                    "start_pos": seq_len + t - 1, "token_id": tok,
                    "logits_finite": bool(torch.isfinite(logits).all()),
                    "top5": logits.topk(5, -1)[1][0].tolist(),
                    "layers": list(self.execution_trace),
                    "handoff": dict(self.handoff_stats),
                    "state_hashes": self.state_signatures(
                        list(range(self.cfg.n_layers)))}
        return generated

    def state_buffers(self, layer_ids: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        return {lid: self.layer(lid).state_buffers() for lid in layer_ids}

    def state_signatures(self, layer_ids: list[int]) -> dict[int, dict[str, str]]:
        return {lid: self.layer(lid).state_signature() for lid in layer_ids}

    def bridge_counters(self) -> dict[str, int]:
        """Aggregate Python/native boundary counters for Phase 1 evidence."""
        keys = (
            "full_hidden_d2h_copies", "raw_expert_output_d2h_copies",
            "hidden_h2d_copies", "route_id_d2h_copies", "route_id_d2h_bytes",
            "numpy_bridge_calls", "host_synchronizations",
            "route_id_host_synchronizations", "native_output_synchronizations",
            "native_calls", "native_batch_calls",
        )
        totals = {key: 0 for key in keys}
        for layer in self.layers0 + self.layers1:
            stats = getattr(layer.ffn_fn, "stats", {})
            for key in keys:
                totals[key] += int(stats.get(key, 0))
        totals["inter_gpu_handoff_d2h_copies"] = int(
            self.handoff_stats.get("d2h_bytes", 0) > 0)
        totals["inter_gpu_handoff_h2d_copies"] = int(
            self.handoff_stats.get("h2d_bytes", 0) > 0)
        return totals

    def runtime_snapshot(self) -> dict[str, Any]:
        """Bounded backend/cache/provider evidence for remote DS10 stages."""
        shared_host_tensors = []
        for layer in self.layers0 + self.layers1:
            # CACHE1d: the shared expert's host FP16 copy is freed after the
            # GPU entry is pinned (shared_payload -> None), so a snapshot must
            # tolerate lazily-materialized/None payloads instead of crashing
            # the evidence stage after all correctness gates have passed.
            sp = layer.ffn_fn.shared_payload
            if sp:
                shared_host_tensors.extend(sp.values())
        return {
            "backends": {
                "attention_state": "torch_cuda_hybrid_bringup",
                "router": "torch_cuda_validated_ds9_path",
                "routed_experts": "freebuff_ds8_cache_fp16_cuda",
                "shared_expert": "freebuff_ds8_cache_fp16_cuda",
                "cpu_expert_execution": False,
            },
            "diagnostics": self.diagnostics,
            "bridge_counters": self.bridge_counters(),
            "provider": self.provider.stats(),
            "shared_expert_host": {
                "layers": len(self.layers0) + len(self.layers1),
                "fp16_bytes": sum(t.numel() * t.element_size()
                                  for t in shared_host_tensors),
            },
            "cache0": self._cache_snapshot(self.cache0),
            "cache1": self._cache_snapshot(self.cache1),
        }

    @staticmethod
    def _cache_snapshot(cache: Any) -> Optional[dict[str, Any]]:
        if cache is None:
            return None
        return {**dict(cache.stats),
                "resident_bytes": cache.used_bytes(),
                "peak_resident_bytes": cache.peak_resident_bytes,
                "entries": len(cache.entries)}

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


def coverage_audit_passes(report: dict[str, Any], *, n_layers: int = 43) -> bool:
    """Fail-closed DS10.1 acceptance check for a coverage-audit report.

    ``full_model_coverage_audit`` deliberately uses ``all_resolved`` and
    ``tensor_count`` as its canonical keys.  Keeping this check beside the
    report producer prevents remote harnesses from accidentally accepting or
    rejecting on a differently named summary field.
    """
    layers = report.get("layers")
    return (
        report.get("all_resolved") is True
        and report.get("tensor_count") == v4support.EXPECTED_TENSOR_COUNT
        and report.get("shard_count") == EXPECTED_SHARD_COUNT
        and isinstance(layers, list)
        and len(layers) == n_layers
        and all(row.get("resolved") is True for row in layers)
    )


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
        dense = sum(groups["dense"][layer_id] for layer_id in layer_ids)
        state = sum(groups["state"][layer_id] for layer_id in layer_ids)
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


# ---------------------------------------------------------------------------
# Native routed-expert engine (pydee) construction for the full model.
# ---------------------------------------------------------------------------


def build_native_engine(shard_paths: list[str], *,
                        device_id: int = 0,
                        hidden: int = 4096,
                        inter: int = 2048,
                        num_experts: int = 256,
                        num_layers: int = 43,
                        topk: int = 6,
                        budget_bytes: int = 512 << 20,
                        host_pack_cache_bytes: int = 8 << 30,
                        use_batched_experts: bool = False,
                        profile_stages: bool = False,
                        swiglu_limit: float = 10.0,
                        cache_dtype: str = "fp16") -> Any:
    """Build one pydee.Engine (FP4 transfer, FP16 or packed-FP4 device cache)
    that streams routed experts for the full DeepSeek-V4-Flash-0731 model.

    The engine mmaps every shard in ``shard_paths`` (read-only, lazy); the
    resolver routes ``layers.{L}.ffn.experts.{E}.w*`` lookups to the correct
    shard by tensor name, so one engine per CUDA device serves all 43 layers.
    Routing stays caller-owned (oracle_path empty); ``moe_forward_experts`` is
    the only entry point the harness uses.

    cache_dtype="fp4" selects P2.3 packed residency: the VRAM cache keeps the
    checkpoint's packed e2m1fn bytes + e8m0 scales (12.75 MiB/expert instead of
    48 MiB FP16), expanding into a bounded scratch at compute time.  This is a
    performance experiment; the default "fp16" is the sealed exact path.
    """
    import pydee
    if pydee.Engine is None:
        raise RuntimeError("pydee compiled binding not importable")
    if cache_dtype not in ("fp16", "fp4"):
        raise ValueError(f"cache_dtype must be 'fp16' or 'fp4', got {cache_dtype!r}")
    cfg = pydee.configure(
        shard_path=shard_paths[0], num_experts=num_experts,
        num_layers=num_layers, hidden=hidden, inter=inter,
        use_cuda=True, transfer_dtype="fp4", cache_dtype=cache_dtype,
        topk=topk, budget_bytes=budget_bytes, swiglu_limit=swiglu_limit)
    cfg.shard_paths = [str(p) for p in shard_paths]
    cfg.device_id = device_id
    cfg.base_layer = 0
    cfg.host_pack_cache_bytes = host_pack_cache_bytes
    cfg.use_batched_experts = use_batched_experts
    cfg.profile_stages = profile_stages
    cfg.profile_timeline = False
    return pydee.new_engine(cfg)
