"""DeepSeek-V4-Flash-0731 expert cache + async staging (DS8).

Faithful Python mirror of the C++ ``VramCacheManager`` contract (see
``include/dee/vram_cache.h``) so the DS8 remote harness can exercise the
same LRU + Oracle-priority eviction semantics, pins, and counters on Kaggle
T4, while being device-agnostic and fully unit-testable on the host.

Entry format (per DS8 spec):
  layer, expert_id, expert_type (routed/shared), representation family,
  source shard, source offset, compressed bytes, resident bytes, scale
  bytes, scratch bytes, ready event, last-use sequence, pin count,
  eviction priority.

The cache itself only does accounting / eviction; it never copies weights.
A ``DeepSeekExpertLoader`` builds a resident payload (FP16-expanded active
expert -- the measured DS7 representation, Family C) and stages it with a
bounded pinned buffer + CUDA stream/event when CUDA is available, or
synchronously on the host in tests.

Shared experts use expert_id == -1 within their layer (documented), matching
the official ``shared_experts`` namespace while keeping the (int, int) key
layout of the C++ manager.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - cache tests require torch
    torch = None

PRIORITY_WEIGHT = 1 << 20


@dataclass
class DeepSeekExpertEntry:
    """One resident expert's bookkeeping + payload (per DS8 format)."""

    layer: int
    expert_id: int
    expert_type: str  # "routed" | "shared"
    representation: str  # "fp16_expanded" | "raw_fp4" | "int8" | "prepacked_fp4"
    source_shard: str | None = None
    source_offset: int = 0
    compressed_bytes: int = 0
    scale_bytes: int = 0
    scratch_bytes: int = 0
    resident_bytes: int = 0
    ready_event: Any = None  # torch.cuda.Event or None (host backend)
    last_use_sequence: int = 0
    pin_count: int = 0
    eviction_priority: int = 0
    payload: Any = None  # resident tensors (dict[name, torch.Tensor] for fp16)
    generation: int = 0
    resident: bool = False

    def resident_bytes_total(self) -> int:
        return self.resident_bytes + self.scratch_bytes


class DeepSeekExpertCache:
    """Bounded expert cache: LRU + Oracle-priority eviction with pins.

    Mirrors ``VramCacheManager``:
      - ``ensure(layer, expert, nbytes, priority)`` makes room and reserves a
        slot (evicting lowest-score resident blocks that are not pinned).
      - ``touch`` advances recency; ``pin/unpin`` forbid eviction.
      - ``sync_fallback`` records a stall when compute reaches a miss.
      - Stats counters: ensures, hits, loads, evictions, fallbacks,
        pinned_blocks_skipped; plus DS8 counters (h2d_bytes, prepack_bytes,
        wait_ms, checksum_failures, peak_resident_bytes, requests).
    """

    def __init__(self, budget_bytes: int, *, device: str = "cpu") -> None:
        if budget_bytes <= 0:
            raise ValueError(f"cache budget must be positive, got {budget_bytes}")
        self.budget_bytes = int(budget_bytes)
        self.device = device
        self.entries: dict[tuple[int, int], DeepSeekExpertEntry] = {}
        self.tick = 0
        self.next_generation = 1
        # Independently tracked used bytes (maintained on reserve/evict/clear)
        # so validate_invariants can detect accounting drift -- deriving it
        # from the entries themselves would make the check tautological.
        self._used_bytes = 0
        self.stats: dict[str, Any] = {
            "ensures": 0, "hits": 0, "loads": 0, "evictions": 0,
            "fallbacks": 0, "pinned_blocks_skipped": 0,
            "h2d_bytes": 0, "prepack_bytes": 0, "wait_ms": 0.0,
            "checksum_failures": 0, "requests": 0,
        }
        self.peak_resident_bytes = 0
        self.debug_validation = False

    # ---- helpers ---------------------------------------------------------
    def key(self, layer: int, expert_id: int) -> tuple[int, int]:
        return (layer, expert_id)

    def shared_key(self, layer: int) -> tuple[int, int]:
        return (layer, -1)

    def used_bytes(self) -> int:
        return self._used_bytes

    def free_bytes(self) -> int:
        return self.budget_bytes - self.used_bytes()

    def resident_count(self) -> int:
        return sum(1 for e in self.entries.values() if e.resident)

    def pinned_count(self) -> int:
        return sum(1 for e in self.entries.values()
                   if e.resident and e.pin_count > 0)

    def _score(self, entry: DeepSeekExpertEntry) -> int:
        # Higher score => kept longer (recency + Oracle priority boost).
        return entry.last_use_sequence + entry.eviction_priority * PRIORITY_WEIGHT

    # ---- core API --------------------------------------------------------
    def is_resident(self, layer: int, expert_id: int) -> bool:
        entry = self.entries.get(self.key(layer, expert_id))
        return entry is not None and entry.resident

    def get(self, layer: int, expert_id: int) -> DeepSeekExpertEntry | None:
        entry = self.entries.get(self.key(layer, expert_id))
        return entry if entry is not None and entry.resident else None

    def touch(self, layer: int, expert_id: int) -> None:
        entry = self.entries.get(self.key(layer, expert_id))
        if entry is not None and entry.resident:
            entry.last_use_sequence = self.tick
            self.tick += 1

    def reserve(self, layer: int, expert_id: int, nbytes: int, *,
                priority: int = 0,
                expert_type: str = "routed",
                representation: str = "fp16_expanded",
                metadata: dict[str, Any] | None = None) -> DeepSeekExpertEntry:
        """Reserve a resident slot of ``nbytes`` (evicting if needed).

        Returns the entry (resident=True, empty payload) or raises on
        failure-to-allocate. Caller fills ``entry.payload`` afterwards.
        """
        self.stats["ensures"] += 1
        self.stats["requests"] += 1
        k = self.key(layer, expert_id)
        existing = self.entries.get(k)
        if existing is not None and existing.resident:
            self.stats["hits"] += 1
            existing.last_use_sequence = self.tick
            self.tick += 1
            existing.eviction_priority = priority
            return existing

        self._evict_until_free(nbytes)
        used = self.used_bytes()
        if used + nbytes > self.budget_bytes:
            self._raise_no_victim(layer, expert_id, nbytes)

        meta = metadata or {}
        entry = DeepSeekExpertEntry(
            layer=layer, expert_id=expert_id, expert_type=expert_type,
            representation=representation,
            source_shard=meta.get("source_shard"),
            source_offset=meta.get("source_offset", 0),
            compressed_bytes=meta.get("compressed_bytes", 0),
            scale_bytes=meta.get("scale_bytes", 0),
            scratch_bytes=meta.get("scratch_bytes", 0),
            resident_bytes=int(nbytes),
            last_use_sequence=self.tick,
            eviction_priority=priority,
            generation=self.next_generation,
            resident=True,
        )
        self.next_generation += 1
        self.tick += 1
        self.entries[k] = entry
        self._used_bytes += entry.resident_bytes_total()
        self.stats["loads"] += 1
        self.peak_resident_bytes = max(self.peak_resident_bytes,
                                       self.used_bytes())
        if self.debug_validation:
            self.validate_invariants()
        return entry

    def load(self, layer: int, expert_id: int, payload: Any,
             resident_bytes: int) -> DeepSeekExpertEntry:
        """Attach a resident payload to an already-reserved entry.

        Convenience for the synchronous host/test path.  Reconciles the
        tracked byte counter against the entries (same policy as the loader's
        ``stage``) so accounting stays consistent even if the entry carries
        non-zero scratch bytes.
        """
        entry = self.get(layer, expert_id)
        if entry is None:
            raise KeyError(f"no resident slot for ({layer}, {expert_id})")
        entry.payload = payload
        entry.resident_bytes = int(resident_bytes)
        self._reconcile_used_bytes()
        self.peak_resident_bytes = max(self.peak_resident_bytes,
                                       self.used_bytes())
        return entry

    def _reconcile_used_bytes(self) -> None:
        """Re-sync the tracked byte counter with the resident entries.

        Used by the loader paths that attach payloads/scratch after
        ``reserve``.  Independence is maintained across the core operations
        (reserve/evict/clear); this reconciles the one-shot attach step.
        """
        self._used_bytes = sum(
            e.resident_bytes_total() for e in self.entries.values()
            if e.resident)

    def pin(self, layer: int, expert_id: int) -> bool:
        entry = self.entries.get(self.key(layer, expert_id))
        if entry is None or not entry.resident:
            return False
        entry.pin_count += 1
        return True

    def unpin(self, layer: int, expert_id: int) -> bool:
        entry = self.entries.get(self.key(layer, expert_id))
        if entry is None or not entry.resident or entry.pin_count == 0:
            return False
        entry.pin_count -= 1
        return True

    def sync_fallback(self, layer: int, expert_id: int, nbytes: int, *,
                      priority: int = 0) -> bool:
        if self.is_resident(layer, expert_id):
            return True
        self.stats["fallbacks"] += 1
        try:
            self.reserve(layer, expert_id, nbytes, priority=priority)
            return True
        except RuntimeError:
            return False

    def clear(self) -> None:
        self.entries.clear()
        self.tick = 0
        self.next_generation = 1
        self._used_bytes = 0

    # ---- eviction --------------------------------------------------------
    def _evict_until_free(self, need: int) -> None:
        while self.free_bytes() < need:
            victim: DeepSeekExpertEntry | None = None
            worst = -1
            first = True
            for entry in self.entries.values():
                if not entry.resident:
                    continue
                if entry.pin_count > 0:
                    self.stats["pinned_blocks_skipped"] += 1
                    continue
                score = self._score(entry)
                if first or score < worst:
                    worst = score
                    victim = entry
                    first = False
            if victim is None:
                return  # caller raises with full diagnostics
            self._used_bytes -= victim.resident_bytes_total()
            self.entries.pop(self.key(victim.layer, victim.expert_id))
            self.stats["evictions"] += 1
        if self.debug_validation:
            self.validate_invariants()

    def _raise_no_victim(self, layer: int, expert_id: int, nbytes: int) -> None:
        pinned = [(k, e.pin_count) for k, e in self.entries.items()
                  if e.resident and e.pin_count > 0]
        raise RuntimeError(
            f"DeepSeekExpertCache::reserve(layer={layer} expert={expert_id} "
            f"nbytes={nbytes}) failed: no evictable victim "
            f"(budget={self.budget_bytes} used={self.used_bytes()} "
            f"free={self.free_bytes()} pinned={pinned})"
        )

    # ---- invariants ------------------------------------------------------
    def validate_invariants(self, error: list[str] | None = None) -> bool:
        used = 0
        for entry in self.entries.values():
            if not entry.resident:
                if error is not None:
                    error.append("non-resident entry retained")
                return False
            if entry.resident_bytes <= 0:
                if error is not None:
                    error.append(f"entry ({entry.layer},{entry.expert_id}) "
                                 "has non-positive resident bytes")
                return False
            used += entry.resident_bytes_total()
        # Compare the entry sum against the independently maintained counter:
        # a corruption of either side is then detectable.
        if used != self._used_bytes:
            if error is not None:
                error.append(f"accounted {used} != tracked {self._used_bytes}")
            return False
        if used > self.budget_bytes:
            if error is not None:
                error.append("used exceeds budget")
            return False
        return True


class DeepSeekExpertLoader:
    """Builds and stages FP16-expanded expert payloads (Family C).

    CUDA path: host dequant -> pinned staging -> async H2D on a side stream,
    completion guarded by a per-expert ``torch.cuda.Event``.  Host path
    (tests / no GPU): synchronous CPU tensors with ``ready_event=None``.

    The unpack/prepack happens on the host for DS8 (the fused SM75 unpack
    kernel is a later milestone); this is measured separately.
    """

    def __init__(self, cache: DeepSeekExpertCache, *, max_staging_bytes: int | None = None) -> None:
        self.cache = cache
        self.has_cuda = bool(torch is not None and torch.cuda.is_available())
        self.device = cache.device
        if self.has_cuda:
            self.staging_stream = torch.cuda.Stream(device=self.device)
        else:
            self.staging_stream = None
        self.max_staging_bytes = max_staging_bytes or (256 << 20)

    def _dequant_expert(self, tensors: dict[str, Any], ref: Any) -> dict[str, Any]:
        """FP16-expand one routed expert's packed FP4 weights + scales."""
        w1 = ref.dequantize_expert_weight(tensors["w1.weight"], tensors["w1.scale"])
        w2 = ref.dequantize_expert_weight(tensors["w2.weight"], tensors["w2.scale"])
        w3 = ref.dequantize_expert_weight(tensors["w3.weight"], tensors["w3.scale"])
        return {
            "w1.weight": w1.half(),
            "w2.weight": w2.half(),
            "w3.weight": w3.half(),
        }

    def stage(self, layer: int, expert_id: int, payload: dict[str, Any],
              *, metadata: dict[str, Any] | None = None) -> DeepSeekExpertEntry:
        """Reserve + populate + (async) stage an expert in the cache.

        Returns the resident entry whose ``ready_event`` (if any) the
        consuming stream must wait on before compute.
        """
        meta = metadata or {}
        resident_bytes = int(sum(v.numel() * v.element_size()
                                 for v in payload.values()))
        entry = self.cache.reserve(
            layer, expert_id, resident_bytes,
            priority=meta.get("priority", 0),
            expert_type=meta.get("expert_type", "routed"),
            representation=meta.get("representation", "fp16_expanded"),
            metadata=meta,
        )
        scratch = int(meta.get("scratch_bytes", 0))
        if self.has_cuda and self.staging_stream is not None:
            device_tensors = {}
            t0 = time.time()
            with torch.cuda.stream(self.staging_stream):
                for name, host_tensor in payload.items():
                    pinned = host_tensor.pin_memory()
                    device_tensors[name] = pinned.to(
                        self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.staging_stream)
            entry.ready_event = event
            entry.payload = device_tensors
            entry.scratch_bytes = scratch
            self.cache.stats["h2d_bytes"] += resident_bytes
            self.cache.stats["wait_ms"] += (time.time() - t0) * 1000.0
        else:
            entry.payload = payload
            entry.ready_event = None
            entry.scratch_bytes = scratch
        # Reconcile the independently tracked byte counter with any scratch
        # attached after reserve() (reserve already accounted resident_bytes).
        entry.resident_bytes = resident_bytes
        self.cache._reconcile_used_bytes()
        self.cache.peak_resident_bytes = max(self.cache.peak_resident_bytes,
                                             self.cache.used_bytes())
        return entry

    def wait(self, entry: DeepSeekExpertEntry) -> None:
        """Make the current stream wait on the entry's completion event."""
        if entry is not None and entry.ready_event is not None and self.has_cuda:
            entry.ready_event.wait(torch.cuda.current_stream(self.device))
