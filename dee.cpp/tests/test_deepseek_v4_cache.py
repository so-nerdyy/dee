"""DS8 expert-cache test matrix (host backend, no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    sys.stderr.write("tests/test_deepseek_v4_cache.py requires pytest: pip install pytest\n")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.deepseek_v4_cache import (  # noqa: E402
    PRIORITY_WEIGHT,
    DeepSeekExpertCache,
    DeepSeekExpertLoader,
    EvictedVictim,
    EvictionPolicy,
)


def _entry_bytes(expert_id: int) -> int:
    return 100 + expert_id  # distinct sizes per expert


def test_miss_hit_reserve_load() -> None:
    cache = DeepSeekExpertCache(10_000)
    assert cache.stats["hits"] == 0
    entry = cache.reserve(3, 0, _entry_bytes(0))
    assert entry.resident and entry.layer == 3 and entry.expert_id == 0
    assert cache.is_resident(3, 0)
    assert cache.stats["loads"] == 1
    assert cache.stats["hits"] == 0
    # second reserve = hit
    again = cache.reserve(3, 0, _entry_bytes(0))
    assert again is entry
    assert cache.stats["hits"] == 1
    assert cache.stats["loads"] == 1


def test_eviction_lru_and_priority() -> None:
    cache = DeepSeekExpertCache(1_000)
    # Fill to capacity with three experts, then force eviction.
    cache.reserve(1, 0, 300)
    cache.reserve(1, 1, 300)
    cache.reserve(1, 2, 300)  # 900/1000 used
    cache.touch(1, 0)  # recency: 0 is most recent now
    cache.reserve(1, 3, 300)  # needs 1200 -> evict one (300 freed)
    # LRU victim is expert 1 (oldest: last touched at ensure order).
    assert not cache.is_resident(1, 1)
    assert cache.is_resident(1, 0)
    assert cache.stats["evictions"] == 1
    assert cache.used_bytes() <= 1000


def test_priority_keeps_oracle_predicted() -> None:
    # RankPriority (the default, legacy arm): priority protects the oldest
    # block from eviction. The Recency variant below asserts the opposite.
    cache = DeepSeekExpertCache(1_000, eviction_policy=EvictionPolicy.RankPriority)
    cache.reserve(1, 0, 300, priority=0)
    cache.reserve(1, 1, 300, priority=0)
    cache.reserve(1, 2, 300, priority=100)  # Oracle-predicted: high priority
    # All same recency initially; touch 0 and 1 so 2 becomes oldest but high-prio.
    cache.touch(1, 0)
    cache.touch(1, 1)
    cache.reserve(1, 3, 300)  # must evict one of {0,1,2}
    # High priority keeps 2; victim is expert 0 (older than 1 by touch order).
    assert cache.is_resident(1, 2)
    assert not cache.is_resident(1, 0)
    assert cache.is_resident(1, 1)
    assert cache.stats["pinned_blocks_skipped"] == 0


def test_recency_policy_ignores_priority() -> None:
    # Under Recency the same script evicts strict-LRU: the high-priority but
    # oldest expert 2 is the victim (mirrors the C++ Recency arm).
    cache = DeepSeekExpertCache(1_000, eviction_policy=EvictionPolicy.Recency)
    cache.reserve(1, 0, 300, priority=0)
    cache.reserve(1, 1, 300, priority=0)
    cache.reserve(1, 2, 300, priority=100)
    cache.touch(1, 0)
    cache.touch(1, 1)
    cache.reserve(1, 3, 300)
    assert not cache.is_resident(1, 2)  # priority 100 does NOT protect
    assert cache.is_resident(1, 0)
    assert cache.is_resident(1, 1)
    assert cache.is_resident(1, 3)


def test_eviction_policy_default_and_setter() -> None:
    cache = DeepSeekExpertCache(1_000)
    assert cache.eviction_policy is EvictionPolicy.RankPriority  # legacy default
    cache.set_eviction_policy(EvictionPolicy.Recency)
    assert cache.eviction_policy is EvictionPolicy.Recency
    cache.set_eviction_policy("lru")  # config-string alias
    assert cache.eviction_policy is EvictionPolicy.Recency
    cache.set_eviction_policy("rank_priority")
    assert cache.eviction_policy is EvictionPolicy.RankPriority
    with pytest.raises(ValueError):
        cache.set_eviction_policy("bogus")


@pytest.mark.parametrize(
    "policy,victim,survivor",
    [
        # Same script as the C++ A/B test: E0..E3 loaded, E3 re-reserved
        # with priority=100 (newest tick), then E0,E1,E2 touched -> E3 is
        # oldest. RankPriority protects E3 (victim E0); Recency evicts E3.
        (EvictionPolicy.RankPriority, 0, 3),
        (EvictionPolicy.Recency, 3, 0),
    ],
)
def test_mirror_policy_victim_selection_mirrors_cpp(
        policy: EvictionPolicy, victim: int, survivor: int) -> None:
    cache = DeepSeekExpertCache(4 * 256, eviction_policy=policy)
    for e in range(4):
        cache.reserve(0, e, 256)
    cache.reserve(0, 3, 256, priority=100)  # hit: newest last_use + prio 100
    for e in range(3):
        cache.touch(0, e)
    cache.reserve(0, 4, 256)
    assert not cache.is_resident(0, victim)
    assert cache.is_resident(0, survivor)
    assert [v.expert_id for v in cache.last_evicted] == [victim]
    if policy is EvictionPolicy.Recency:
        # Strict LRU continuation: next miss evicts E0 (now oldest).
        cache.reserve(0, 5, 256)
        assert not cache.is_resident(0, 0)


def test_pins_prevent_eviction() -> None:
    cache = DeepSeekExpertCache(600)
    cache.reserve(1, 0, 300)
    cache.reserve(1, 1, 300)
    assert cache.pin(1, 0)
    cache.reserve(1, 2, 300)  # needs 900, free=0, expert 0 pinned -> skip
    assert cache.stats["pinned_blocks_skipped"] >= 1
    # expert 1 evicted to fit expert 2.
    assert not cache.is_resident(1, 1)
    assert cache.is_resident(1, 2)
    assert cache.is_resident(1, 0)
    # unpin then next pressure evicts 0.
    assert cache.unpin(1, 0)
    cache.reserve(1, 3, 300)
    assert not cache.is_resident(1, 0)


def test_no_evictable_victim_fails_closed() -> None:
    cache = DeepSeekExpertCache(600)
    cache.reserve(1, 0, 600)
    cache.pin(1, 0)
    with pytest.raises(RuntimeError, match="no evictable victim"):
        cache.reserve(1, 1, 100)


def test_shared_expert_key_convention() -> None:
    cache = DeepSeekExpertCache(10_000)
    entry = cache.reserve(6, -1, 500, expert_type="shared",
                          representation="fp16_expanded")
    assert entry.expert_type == "shared"
    assert cache.is_resident(6, -1)
    assert cache.get(6, -1) is entry
    # Routed expert 0 in same layer is a different key.
    assert not cache.is_resident(6, 0)


def test_same_expert_across_tokens_and_layers() -> None:
    cache = DeepSeekExpertCache(50_000)
    # Same expert across tokens = repeated hit (recency refresh).
    for _ in range(5):
        cache.reserve(3, 7, 200)
        cache.touch(3, 7)
    assert cache.stats["hits"] == 4
    # Same expert id across layers = distinct keys.
    cache.reserve(4, 7, 200)
    assert cache.is_resident(3, 7) and cache.is_resident(4, 7)


def test_more_experts_than_capacity_eviction_under_pressure() -> None:
    cache = DeepSeekExpertCache(1_000)
    n = 20
    for expert in range(n):
        cache.reserve(0, expert, 300)
        cache.touch(0, expert)
    assert cache.used_bytes() <= 1000
    assert cache.resident_count() <= 3
    assert cache.stats["evictions"] >= n - 3
    assert cache.stats["loads"] == n


def test_sync_fallback_counts_stall() -> None:
    cache = DeepSeekExpertCache(10_000)
    # first call on an empty cache is a miss -> records a stall
    assert cache.sync_fallback(0, 0, 100)
    assert cache.stats["fallbacks"] == 1
    # now resident -> no new stall
    assert cache.sync_fallback(0, 0, 100)
    assert cache.stats["fallbacks"] == 1
    # clear() keeps cumulative stats (mirrors C++ VramCacheManager.clear),
    # and the next miss counts again
    cache.clear()
    assert cache.sync_fallback(0, 0, 100)
    assert cache.stats["fallbacks"] == 2


def test_counters_h2d_prepack_and_peak() -> None:
    cache = DeepSeekExpertCache(100_000)
    cache.reserve(0, 0, 100, metadata={"source_shard": "s.safetensors",
                                       "source_offset": 64,
                                       "compressed_bytes": 50,
                                       "scale_bytes": 10})
    cache.stats["h2d_bytes"] += 100
    cache.stats["prepack_bytes"] += 50
    assert cache.peak_resident_bytes == 100
    assert cache.used_bytes() == 100


def test_invariants_validation() -> None:
    cache = DeepSeekExpertCache(10_000)
    cache.debug_validation = True
    cache.reserve(0, 0, 100)
    cache.reserve(0, 1, 100)
    assert cache.validate_invariants() is True
    errors: list[str] = []
    assert cache.validate_invariants(errors)
    # corrupt: force accounting mismatch
    entry = cache.get(0, 0)
    entry.resident_bytes = 999
    assert cache.validate_invariants(errors) is False
    assert any("accounted" in e for e in errors)


def test_reserve_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError):
        DeepSeekExpertCache(0)


def test_staging_bound_fails_closed() -> None:
    import torch

    cache = DeepSeekExpertCache(1 << 30)
    loader = DeepSeekExpertLoader(cache, max_staging_bytes=10)
    # A payload larger than the bounded staging limit fails closed before any
    # cache-slot reservation or H2D (no partial allocation, no eviction, no
    # counter side effects).
    with pytest.raises(RuntimeError, match="exceeds bounded staging limit"):
        loader.stage(0, 0, {"w": torch.ones(64, 64, dtype=torch.float16)})
    assert not cache.is_resident(0, 0)
    assert cache.stats["loads"] == 0
    assert cache.stats["evictions"] == 0
    assert cache.stats["ensures"] == 0
    assert cache.stats["requests"] == 0
    # A small payload under the bound stages fine.
    ok = loader.stage(0, 0, {"w": torch.ones(2, 2, dtype=torch.float16)},
                      metadata={"source_shard": "s.safetensors"})
    assert ok is not None and cache.is_resident(0, 0)


def test_multi_victim_list_order() -> None:
    # Mirror of the C++ multi-victim test: one reserve() evicts 3 blocks and
    # last_evicted exposes ALL of them in eviction order (E3, E2, E1), with
    # the scalar last-victim fields matching EnsureInfo's back-compat contract.
    cache = DeepSeekExpertCache(2 * 256)
    for e in range(4):
        cache.reserve(0, e, 128)
    cache.touch(0, 2)
    cache.touch(0, 1)
    cache.touch(0, 0)  # recency: E3 < E2 < E1 < E0
    cache.reserve(0, 4, 384)
    assert cache.stats["evictions"] == 3
    assert [(v.layer, v.expert_id) for v in cache.last_evicted] == [
        (0, 3), (0, 2), (0, 1)]
    assert all(isinstance(v, EvictedVictim) and v.generation != 0
               for v in cache.last_evicted)
    # Last-victim scalars (the C++ evicted_key/evicted_generation).
    assert cache.last_evicted_key == (0, 1)
    assert cache.last_evicted_generation == cache.last_evicted[-1].generation
    # A subsequent hit pass clears the victim list (EnsureInfo reset).
    cache.reserve(0, 4, 384)
    assert cache.last_evicted == []
    assert cache.last_evicted_key is None


def test_oversize_reserve_rejects_before_evicting() -> None:
    cache = DeepSeekExpertCache(4 * 256)
    cache.reserve(0, 0, 256)
    cache.reserve(0, 1, 256)
    with pytest.raises(RuntimeError, match="exceeds budget"):
        cache.reserve(0, 2, 4 * 256 + 1)
    assert cache.stats["budget_rejections"] == 1
    assert cache.stats["evictions"] == 0
    # Residents survive the reject untouched.
    assert cache.is_resident(0, 0)
    assert cache.is_resident(0, 1)
    assert cache.used_bytes() == 2 * 256
    assert cache.last_evicted == []
    # Cache remains fully usable afterwards.
    cache.reserve(0, 2, 256)
    assert cache.is_resident(0, 2)


def test_same_key_different_size_fails_closed() -> None:
    cache = DeepSeekExpertCache(4 * 256)
    cache.reserve(0, 0, 256)
    with pytest.raises(RuntimeError, match="hit-size mismatch"):
        cache.reserve(0, 0, 512)
    assert cache.stats["size_mismatches"] == 1
    assert cache.stats["hits"] == 0  # reject is not a hit
    assert cache.is_resident(0, 0)
    with pytest.raises(RuntimeError, match="hit-size mismatch"):
        cache.reserve(0, 0, 128)  # smaller also fails
    assert cache.stats["size_mismatches"] == 2
    # Same-key same-size still hits.
    assert cache.reserve(0, 0, 256) is not None
    assert cache.stats["hits"] == 1
    # sync_fallback's resident hit path is fail-closed on size too, and does
    # not record a fallback stall for the reject.
    assert cache.sync_fallback(0, 0, 999) is False
    assert cache.stats["size_mismatches"] == 3
    assert cache.stats["fallbacks"] == 0


def test_evict_key_invalidation() -> None:
    cache = DeepSeekExpertCache(4 * 256)
    cache.debug_validation = True
    cache.reserve(0, 0, 256)
    cache.reserve(0, 1, 256)
    assert cache.evict_key(0, 0) is True
    assert not cache.is_resident(0, 0)
    assert cache.stats["invalidations"] == 1
    assert cache.stats["evictions"] == 0  # invalidation is NOT an eviction
    assert cache.used_bytes() == 256
    assert cache.evict_key(0, 0) is False  # absent -> refuse, count nothing
    cache.pin(0, 1)
    assert cache.evict_key(0, 1) is False  # pinned -> refuse, no partial
    assert cache.is_resident(0, 1)
    assert cache.stats["invalidations"] == 1
    assert cache.used_bytes() == 256
    cache.unpin(0, 1)
    assert cache.evict_key(0, 1) is True
    assert cache.stats["invalidations"] == 2
    assert cache.validate_invariants()


def test_pinned_blocks_skipped_once_per_reserve() -> None:
    # Two pinned + two unpinned blocks; a reserve needing TWO evictions scans
    # the pinned pair in two iterations but must count each block once.
    cache = DeepSeekExpertCache(4 * 256)
    for e in range(4):
        cache.reserve(0, e, 256)
    cache.pin(0, 0)
    cache.pin(0, 1)
    cache.reserve(0, 4, 512)
    assert cache.stats["evictions"] == 2
    assert not cache.is_resident(0, 2) and not cache.is_resident(0, 3)
    assert cache.is_resident(0, 4)
    assert cache.stats["pinned_blocks_skipped"] == 2  # not 4 (once per block)
    # Per-reserve semantics: a second pass counts each pinned block once more.
    cache.reserve(0, 5, 512)
    assert cache.stats["pinned_blocks_skipped"] == 4
    assert not cache.is_resident(0, 4)


def test_priority_weight_mirrors_cpp() -> None:
    # The C++ VramCacheManager uses PRIORITY_WEIGHT = 1 << 20; the Python
    # mirror must match so DS8 measured behavior transfers to the engine.
    assert PRIORITY_WEIGHT == (1 << 20)
