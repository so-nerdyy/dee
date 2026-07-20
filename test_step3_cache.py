"""
Pure-Python unit tests for ExpertCache (Step 3). No torch, no Modal, no GPU.
Run:  python3 test_step3_cache.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from modal_step3_cache_manager import ExpertCache


def test_basic_hit_miss():
    c = ExpertCache(capacity=3)
    assert c.request_expert(1, token=0) is False   # miss -> load
    assert c.request_expert(1, token=1) is True    # hit
    assert c.contains(1)
    assert not c.contains(99)
    print("PASS test_basic_hit_miss")


def test_capacity_enforced():
    c = ExpertCache(capacity=2)
    c.request_expert(1, token=0)
    c.request_expert(2, token=1)
    assert len(c.vram) == 2
    c.request_expert(3, token=2)   # should evict one, stay at capacity
    assert len(c.vram) == 2
    assert 3 in c.vram
    print("PASS test_capacity_enforced")


def test_lookahead_keeps_predicted():
    # Capacity 2. Cache has {1,2}. Look-ahead predicts {1,4}.
    # Requesting 3 (not predicted) should evict 2 (not predicted), keep 1.
    c = ExpertCache(capacity=2)
    c.request_expert(1, token=0)
    c.request_expert(2, token=1)
    c.set_lookahead_ranked([1, 4])          # 1 is wanted soon, 2 is not
    c.request_expert(3, token=2)     # miss -> evict; should drop 2 not 1
    assert 1 in c.vram, "look-ahead should have protected predicted expert 1"
    assert 3 in c.vram
    assert 2 not in c.vram
    print("PASS test_lookahead_keeps_predicted")


def test_blind_lru_would_differ():
    # Demonstrate the policy is NOT blind LRU: with look-ahead set to the most
    # recently used non-predicted, the eviction still respects prediction.
    c = ExpertCache(capacity=2)
    c.request_expert(5, token=0)     # most recent
    c.request_expert(6, token=1)
    c.set_lookahead_ranked([6])             # 6 predicted, 5 not
    c.request_expert(7, token=2)     # miss -> evict; 5 dropped despite being LRU-recent
    assert 6 in c.vram and 7 in c.vram and 5 not in c.vram
    print("PASS test_blind_lru_would_differ")


def test_prefetch_does_not_double_count():
    c = ExpertCache(capacity=5)
    c.prefetch(10, token=0)
    assert c.contains(10)
    before = len(c.vram)
    c.prefetch(10, token=1)          # already present -> no new load
    assert len(c.vram) == before
    assert c.prefetch_count[10] == 2  # counted twice as prefetch
    assert c.load_count[10] == 1      # but loaded once
    print("PASS test_prefetch_does_not_double_count")


def test_evict_logged():
    c = ExpertCache(capacity=1)
    c.request_expert(1, token=0)
    c.set_lookahead_ranked([])
    c.request_expert(2, token=1)     # evicts 1
    kinds = [op[0] for op in c.async_ops]
    assert "evict" in kinds
    assert "fetch_on_miss" in kinds
    print("PASS test_evict_logged")


def test_delta_prefetch_only_loads_missing():
    # Capacity 5. Pre-load {1,2,3} via full prefetch, then delta-prefetch a
    # predicted list [1,2,4,5]; only 4,5 should be newly loaded.
    c = ExpertCache(capacity=5, top_k_pred=4)
    for e in (1, 2, 3):
        c.prefetch(e, token=0)
    missing = c.prefetch_delta([1, 2, 4, 5], token=1)
    assert set(missing) == {4, 5}, f"delta should only return missing: {missing}"
    assert c.contains(1) and c.contains(4) and c.contains(5)
    assert c.load_count[1] == 1 and c.load_count[4] == 1
    assert len(c.vram) == 5
    print("PASS test_delta_prefetch_only_loads_missing")


def test_rank_weighted_eviction_keeps_top_rank():
    # Capacity 2. VRAM={1,2}. Look-ahead ranked [2, 9] (2 is rank0/top).
    # Requesting 3 (miss) must evict 1 (NOT in look-ahead), keep 2 (top rank).
    c = ExpertCache(capacity=2, top_k_pred=2)
    c.prefetch(1, token=0)
    c.prefetch(2, token=1)
    c.set_lookahead_ranked([2, 9])      # 2 rank0 (keep), 1 not predicted
    c.request_expert(3, token=2)        # miss -> evict 1
    assert 2 in c.vram and 3 in c.vram and 1 not in c.vram
    print("PASS test_rank_weighted_eviction_keeps_top_rank")


def test_rank_weighted_eviction_lowers_low_rank():
    # Capacity 3. VRAM={1,2,3}. Look-ahead ranked [3(rank0), 1(rank1), 7].
    # Both 1 and 3 predicted (kept); 2 is not predicted -> evicted first.
    c = ExpertCache(capacity=3, top_k_pred=3)
    c.prefetch(1, token=0); c.prefetch(2, token=1); c.prefetch(3, token=2)
    c.set_lookahead_ranked([3, 1, 7])
    c.request_expert(8, token=3)        # miss -> 2 (unpredicted) evicted
    assert 2 not in c.vram and 1 in c.vram and 3 in c.vram and 8 in c.vram
    print("PASS test_rank_weighted_eviction_lowers_low_rank")


if __name__ == "__main__":
    test_basic_hit_miss()
    test_capacity_enforced()
    test_lookahead_keeps_predicted()
    test_blind_lru_would_differ()
    test_prefetch_does_not_double_count()
    test_evict_logged()
    test_delta_prefetch_only_loads_missing()
    test_rank_weighted_eviction_keeps_top_rank()
    test_rank_weighted_eviction_lowers_low_rank()
    print("\nALL STEP 3 CACHE UNIT TESTS PASSED")
