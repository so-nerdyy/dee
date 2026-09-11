// tests/test_legacy_submit_event_leak.cpp
//
// Regression coverage for the legacy AsyncPrefetcher::cuda_submit failure
// paths (fix/legacy-submit-event-leak, fix/legacy-submit-resident-garbage).
// The Phase-2 audit L4 fix in T1 (688bd98) hardened cuda_submit_host; the
// legacy path had the same defect and then some: every failure return after
// cudaEventCreateWithFlags left the per-transfer event on the abandoned
// transfer until reset() — a bounded event-object leak per failed submit —
// AND released the pinned staging slot + cache pin without draining the
// stream, so enqueued work could still be touching the recycled memory.
//
// A second, correctness-class defect lived one frame up: prefetch_impl's
// cold-submit failure path discarded the freshly ensured cache block only on
// the managed (host_lease) path. On the legacy path the block stayed
// resident holding stale/partial bytes, so the next request for the same key
// took the ResidentHit path and served garbage to compute. Section 5 pins
// the repaired invariant: a failed cold submit leaves NO resident block for
// the failed generation, on either submit path (one call site, one rule).
//
// This target compiles the REAL async_prefetcher.cpp with DEE_CUDA defined
// against the mock runtime in tests/cuda_stub/ (per-call fault injection +
// stream/event accounting), so each post-create failure return is executed
// rather than asserted by contract. Oracles:
//   - dee_stub::g.live_events / trace_alloc::live_count: no leaked event;
//     a double-destroy would trip the sentinel's abort instead.
//   - g.stream_syncs / g.pending: the mandatory drain ran before the
//     caller's release_transfer recycles the staging slot + cache pin.
//   - a clean re-submit afterwards: the failed transfer's resources were
//     really released, not just hidden.
//
// The success path is asserted to keep its event until reset() — the fix
// changes only failure-path cleanup, not normal lifecycle.

#include "dee/async_prefetcher.h"
#include "dee/trace_alloc.h"
#include "dee/vram_cache.h"

#include "cuda_stub_control.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {
using namespace dee;

int failures = 0;
void check(bool condition, const char* text) {
    std::cout << (condition ? "PASS " : "FAIL ") << text << '\n';
    if (!condition) ++failures;
}

Arena::Backend mock_device() {
    return {[](size_t b) { return std::malloc(b); },
            [](void* p) { std::free(p); }, "host"};
}

struct Rig {
    VramCacheManager cache;
    AsyncPrefetcher  pf;
    Rig() : pf(cache) {}
    bool init(size_t arena_bytes) {
        return cache.init(arena_bytes, mock_device()) && pf.init(true);
    }
};

// Every post-event-create failure path must leave the same post-conditions:
// prefetch rejected, stream drained (pending work the failing call left
// behind is gone), event destroyed rather than retained on the abandoned
// transfer, and the transfer itself unreachable via wait().
void check_failed_submit(Rig& rig, int layer, int expert, int syncs_before,
                         const char* tag) {
    check(dee_stub::g.stream_syncs == syncs_before + 1,
          (std::string(tag) + ": stream drained before resource release").c_str());
    check(dee_stub::g.pending == 0,
          (std::string(tag) + ": drain cleared the in-flight work").c_str());
    check(dee_stub::g.live_events == 0,
          (std::string(tag) + ": event destroyed on failure path (no leak)").c_str());
    check(!rig.pf.wait(layer, expert),
          (std::string(tag) + ": abandoned transfer fails wait()").c_str());
}

// After any failure, a different expert must submit + complete cleanly:
// proves the staging slot and cache pin were actually recycled.
void check_recovery(Rig& rig, int layer, int expert, const char* tag) {
    const uint8_t src[64] = {0xAB};
    check(rig.pf.prefetch(layer, expert, src, sizeof(src), 0) >= 0,
          (std::string(tag) + ": recovery prefetch issues").c_str());
    check(rig.pf.wait(layer, expert),
          (std::string(tag) + ": recovery transfer completes").c_str());
    check(dee_stub::g.live_events == 1,
          (std::string(tag) + ": success path keeps its event until reset()").c_str());
}

// ===========================================================================
// 1. Baseline: successful submit retains its event for the in-flight wait
//    and wait_on_stream paths; reset() is still its only destroy point.
// ===========================================================================
void success_baseline() {
    std::cout << "-- success path keeps event until reset --\n";
    dee_stub::g.reset();
    const size_t ta_before = dee::trace_alloc::live_count();
    {
        Rig rig;
        check(rig.init(4096), "baseline: init on mock cuda");
        const uint8_t src[64] = {7};
        check(rig.pf.prefetch(0, 1, src, sizeof(src), 0) >= 0,
              "baseline: prefetch issued");
        check(dee_stub::g.live_events == 1, "baseline: one live event");
        check(rig.pf.wait(0, 1), "baseline: wait completes");
        check(std::memcmp(rig.cache.data(0, 1), src, sizeof(src)) == 0,
              "baseline: mock H2D delivered exact bytes");
        check(dee_stub::g.live_events == 1,
              "baseline: event survives wait() (still owned by the transfer)");
        rig.pf.reset();
        check(dee_stub::g.live_events == 0, "baseline: reset() destroys the event");
    }
    check(dee::trace_alloc::live_count() == ta_before,
          "baseline: sentinel table balanced after teardown");
}

// ===========================================================================
// 2. Failures BEFORE event creation need no drain: nothing was enqueued for
//    this transfer. (Contract parity with cuda_submit_host's early returns.)
// ===========================================================================
void pre_event_failures() {
    std::cout << "-- pre-event failures return without drain or event --\n";
    dee_stub::g.reset();
    const size_t ta_before = dee::trace_alloc::live_count();
    {
        Rig rig;
        check(rig.init(4096), "pre-event: init");
        const int syncs_before = dee_stub::g.stream_syncs;
        const uint8_t src[64] = {3};
        dee_stub::g.fail_next("cudaEventCreateWithFlags");
        check(rig.pf.prefetch(0, 1, src, sizeof(src), 0) < 0,
              "pre-event: create failure rejects the prefetch");
        check(dee_stub::g.live_events == 0, "pre-event: no event exists to leak");
        check(dee_stub::g.stream_syncs == syncs_before && dee_stub::g.pending == 0,
              "pre-event: no drain needed — nothing enqueued");
        check_recovery(rig, 0, 2, "pre-event");
    }
    check(dee::trace_alloc::live_count() == ta_before,
          "pre-event: sentinel table balanced");
}

// ===========================================================================
// 3. The leak fix: every failure AFTER the event exists drains the stream,
//    destroys the event, and only then lets the caller recycle resources.
//    Each case is exactly one of the fault-injection probes the GPU
//    campaign's baseline arm would exercise.
// ===========================================================================
void memcpy_failure() {
    std::cout << "-- cudaMemcpyAsync failure --\n";
    dee_stub::g.reset();
    const size_t ta_before = dee::trace_alloc::live_count();
    {
        Rig rig;
        check(rig.init(4096), "memcpy: init");
        const int syncs_before = dee_stub::g.stream_syncs;
        const int drains_before = dee_stub::g.syncs_with_pending;
        const uint8_t src[64] = {5};
        dee_stub::g.fail_next("cudaMemcpyAsync");
        check(rig.pf.prefetch(0, 1, src, sizeof(src), 0) < 0,
              "memcpy: injected failure rejects prefetch");
        check_failed_submit(rig, 0, 1, syncs_before, "memcpy");
        check(dee_stub::g.syncs_with_pending == drains_before + 1,
              "memcpy: drain ran while work was still in flight");
        check_recovery(rig, 0, 2, "memcpy");
    }
    check(dee::trace_alloc::live_count() == ta_before,
          "memcpy: sentinel table balanced");
}

void event_record_failure() {
    std::cout << "-- cudaEventRecord failure --\n";
    dee_stub::g.reset();
    const size_t ta_before = dee::trace_alloc::live_count();
    {
        Rig rig;
        check(rig.init(4096), "record: init");
        const int syncs_before = dee_stub::g.stream_syncs;
        const int drains_before = dee_stub::g.syncs_with_pending;
        const uint8_t src[64] = {9};
        dee_stub::g.fail_next("cudaEventRecord");
        check(rig.pf.prefetch(0, 1, src, sizeof(src), 0) < 0,
              "record: injected failure rejects prefetch");
        check_failed_submit(rig, 0, 1, syncs_before, "record");
        check(dee_stub::g.syncs_with_pending == drains_before + 1,
              "record: drain ran with the H2D copy still in flight");
        check_recovery(rig, 0, 2, "record");
    }
    check(dee::trace_alloc::live_count() == ta_before,
          "record: sentinel table balanced");
}

void conversion_failures() {
    std::cout << "-- conversion-kernel failures --\n";
    const size_t ta_before = dee::trace_alloc::live_count();
    {
        dee_stub::g.reset();
        Rig rig;
        check(rig.init(8192), "convert: init");
        const int syncs_at_start = dee_stub::g.stream_syncs;
        const uint16_t src16[32] = {0};
        const float scales[3] = {1.f, 1.f, 1.f};
        const int8_t src8[24] = {0};
        const uint8_t src4[12] = {0};

        int syncs_before = dee_stub::g.stream_syncs;
        dee_stub::g.fail_next("bf16_to_f32_cuda");
        check(rig.pf.prefetch_bf16_to_f32(0, 1, src16, 32, 0) < 0,
              "convert: bf16 failure rejects prefetch");
        check_failed_submit(rig, 0, 1, syncs_before, "bf16");

        syncs_before = dee_stub::g.stream_syncs;
        dee_stub::g.fail_next("int8_to_f16_cuda");
        check(rig.pf.prefetch_int8_to_f16(0, 2, src8, 24, 8, scales, 0) < 0,
              "convert: int8 failure rejects prefetch");
        check_failed_submit(rig, 0, 2, syncs_before, "int8");

        syncs_before = dee_stub::g.stream_syncs;
        dee_stub::g.fail_next("int4_to_f16_cuda");
        check(rig.pf.prefetch_int4_to_f16(0, 3, src4, 24, 8, scales, 0) < 0,
              "convert: int4 failure rejects prefetch");
        check_failed_submit(rig, 0, 3, syncs_before, "int4");

        // FP4 decodes three projections in a loop — fail the SECOND kernel so
        // the first projection's enqueued work is still in flight at failure.
        const size_t packed_off[3] = {0, 4, 8};
        const size_t scale_off[3] = {12, 13, 14};
        const size_t out[3] = {2, 2, 2};
        const size_t in[3] = {4, 4, 4};
        const uint8_t src_fp4[15] = {0};
        syncs_before = dee_stub::g.stream_syncs;
        dee_stub::g.fail_next("fp4_e2m1_to_f16_cuda", 2);
        check(rig.pf.prefetch_fp4_to_f16(0, 4, src_fp4, sizeof(src_fp4),
                                         packed_off, scale_off, out, in, 0) < 0,
              "convert: mid-loop fp4 failure rejects prefetch");
        check_failed_submit(rig, 0, 4, syncs_before, "fp4");
        check(dee_stub::g.calls["fp4_e2m1_to_f16_cuda"] == 2,
              "convert: loop stopped at the failed projection");

        check(dee_stub::g.stream_syncs == syncs_at_start + 4,
              "convert: each failure drained exactly once");
        check_recovery(rig, 0, 5, "convert");
    }
    check(dee::trace_alloc::live_count() == ta_before,
          "convert: sentinel table balanced");
}

// ===========================================================================
// 4. Failure storm: the campaign's probes fail submits repeatedly. Bounded
//    per-failure leak must stay bounded — N failures, zero live events, and
//    reset() must not trip the sentinel's double-free abort on the
//    already-destroyed events.
// ===========================================================================
void failure_storm() {
    std::cout << "-- repeated failures do not accumulate events --\n";
    dee_stub::g.reset();
    const size_t ta_before = dee::trace_alloc::live_count();
    {
        Rig rig;
        check(rig.init(8192), "storm: init");
        const uint8_t src[64] = {1};
        constexpr int kFailures = 8;
        for (int i = 0; i < kFailures; ++i) {
            dee_stub::g.fail_next("cudaEventRecord");
            check(rig.pf.prefetch(0, 10 + i, src, sizeof(src), 0) < 0,
                  "storm: each injected failure rejects prefetch");
        }
        check(dee_stub::g.live_events == 0,
              "storm: N failed submits leaked zero events");
        check(rig.pf.in_flight() == kFailures,
              "storm: abandoned transfers retained for accounting until reset");
        rig.pf.reset();
        check(dee_stub::g.live_events == 0 && rig.pf.in_flight() == 0,
              "storm: reset() clean — no double-destroy, no residue");
        check_recovery(rig, 0, 99, "storm");
    }
    check(dee::trace_alloc::live_count() == ta_before,
          "storm: sentinel table balanced");
}

// ===========================================================================
// 5. Resident-garbage defect (the correctness half of a failed cold submit).
//    prefetch_impl ensures a fresh cache block BEFORE cuda_submit; when the
//    submit fails the block's bytes are stale/partial. The generation-matched
//    discard must run on the legacy path exactly as on the managed
//    (host_lease) path — otherwise is_resident() answers true and the next
//    request for the key takes ResidentHit, serving garbage to compute.
//
//    Byte oracle is made deterministic by seeding the arena: the first
//    transfer leaves src_a's bytes at offset 0; cache.clear() frees the
//    block without touching arena memory; the failed submit's ensure()
//    first-fit reuses offset 0, so a stale block would read back as src_a —
//    never the retry's src_b — on both stub and real allocators.
// ===========================================================================
void failed_submit_leaves_no_resident_block() {
    std::cout << "-- failed legacy cold submit discards the ensured block --\n";
    dee_stub::g.reset();
    const size_t ta_before = dee::trace_alloc::live_count();
    {
        Rig rig;
        check(rig.init(4096), "garbage: init");
        uint8_t src_a[64], src_b[64];
        std::memset(src_a, 0xAA, sizeof(src_a));
        std::memset(src_b, 0x55, sizeof(src_b));

        check(rig.pf.prefetch(0, 1, src_a, sizeof(src_a), 0) >= 0 &&
              rig.pf.wait(0, 1), "garbage: seed transfer completes");
        check(std::memcmp(rig.cache.data(0, 1), src_a, sizeof(src_a)) == 0,
              "garbage: seed bytes landed at the first arena offset");
        rig.pf.reset();   // retire the seed transfer + its event
        rig.cache.clear(); // free the block; arena bytes are left behind
        check(!rig.cache.is_resident(0, 1) && dee_stub::g.live_events == 0,
              "garbage: clean slate — no resident block, no live events");

        // Injected H2D failure: the stub counts the enqueue then reports the
        // error, matching the drain-first contract. prefetch must reject AND
        // the ensured block must not survive.
        dee_stub::g.fail_next("cudaMemcpyAsync");
        check(rig.pf.prefetch(0, 1, src_b, sizeof(src_b), 0) < 0,
              "garbage: injected failure rejects the cold prefetch");
        check(!rig.cache.is_resident(0, 1),
              "garbage: failed generation is NOT resident");
        check(rig.cache.generation_of(0, 1) == 0 && rig.cache.data(0, 1) == nullptr,
              "garbage: no block survives for the failed key");
        check(rig.cache.pinned_count() == 0,
              "garbage: no leaked cache pin");
        check(dee_stub::g.live_events == 0,
              "garbage: no leaked event");
        const size_t slots_after_fail = rig.pf.staging_slot_count();

        // The next request for the same key must be a ColdLoad — never a
        // ResidentHit on the failed generation's stale bytes.
        const uint64_t hits_before = rig.pf.stats().resident_hits;
        const uint64_t colds_before = rig.pf.stats().cold_loads;
        check(rig.pf.prefetch(0, 1, src_b, sizeof(src_b), 0) >= 0,
              "garbage: re-request issues");
        check(rig.pf.stats().cold_loads == colds_before + 1 &&
              rig.pf.stats().resident_hits == hits_before,
              "garbage: re-request took ColdLoad, not ResidentHit");
        check(rig.pf.staging_slot_count() == slots_after_fail,
              "garbage: staging slot recycled, not leaked");
        check(rig.pf.wait(0, 1), "garbage: re-request completes");
        check(std::memcmp(rig.cache.data(0, 1), src_b, sizeof(src_b)) == 0,
              "garbage: retry delivers the NEW bytes, not the stale block");
        check(dee_stub::g.live_events == 1,
              "garbage: successful retry owns exactly one event");
        std::string err;
        check(rig.pf.validate_invariants(&err),
              "garbage: prefetcher invariants hold after failure + retry");
    }
    check(dee::trace_alloc::live_count() == ta_before,
          "garbage: sentinel table balanced");
}

}  // namespace

int main() {
    success_baseline();
    pre_event_failures();
    memcpy_failure();
    event_record_failure();
    conversion_failures();
    failure_storm();
    failed_submit_leaves_no_resident_block();
    std::cout << "Legacy submit event-leak failures: " << failures << '\n';
    return failures ? 1 : 0;
}
