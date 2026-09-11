#include "dee/async_prefetcher.h"
#include "dee/trace_alloc.h"  // Milestone 3 v5 teardown-forensics sentinel

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_set>

#ifdef DEE_CUDA
#include "dee/cuda_check.h"
#include "dee/cuda_convert.h"
#include <cuda_runtime.h>
#endif

namespace dee {
namespace {

uint64_t key_id(int layer, int expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           static_cast<uint32_t>(expert);
}

}  // namespace

AsyncPrefetcher::AsyncPrefetcher(VramCacheManager& cache, size_t ring_size)
    : cache_(cache), ring_size_(ring_size ? ring_size : 1) {}

AsyncPrefetcher::~AsyncPrefetcher() {
    reset();
#ifdef DEE_CUDA
    for (auto& slot : staging_slots_) {
        if (slot.ptr) DEE_CUDA_CHECK_NAMED(DEE_TA_FREE_HOST(slot.ptr, "slot"), "cudaFreeHost(pinned staging slot)");
        if (slot.device_ptr) {
            DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(slot.device_ptr, "slot"), "cudaFree(BF16 device staging slot)");
        }
    }
    if (stream_) DEE_CUDA_CHECK_NAMED(DEE_TA_STREAM_DESTROY(static_cast<cudaStream_t>(stream_), "stream_"),
                                       "cudaStreamDestroy(prefetch)");
#endif
}

bool AsyncPrefetcher::init(bool use_cuda) {
    if (!inflight_.empty()) reset();
    use_cuda_ = use_cuda;
    return !use_cuda_ || cuda_init();
}

long AsyncPrefetcher::find_inflight(int layer, int expert) const {
    const auto it = key_to_idx_.find(map_key(layer, expert));
    return it == key_to_idx_.end() ? -1 : it->second;
}

uint64_t AsyncPrefetcher::map_key(int layer, int expert) const {
    // Phase-2 audit M1 hardening: always the full 64-bit key_id, on every
    // path. The former conditional `experimental ? key : (uint64_t)(long)key`
    // collapsed all layers sharing an expert id on LLP64 (32-bit long), so
    // the legacy OFF path could never stage layer>0 experts on Windows — it
    // failed closed, but the "default-OFF equivalence" it preserved only ever
    // existed on LP64, where the cast is the identity. The map is internal
    // state (no wire or persistent format), so the full key is safe for the
    // legacy path and mandatory for the experimental path; making it
    // unconditional also removes the coupling where flag persistence was
    // load-bearing for key width.
    return key_id(layer, expert);
}

void AsyncPrefetcher::release_staging(Transfer& transfer) {
    if (transfer.managed_source && transfer.host_lease) {
#ifdef DEE_CUDA
        if (use_cuda_ && !transfer.dma_complete && transfer.event)
            transfer.dma_complete = cudaEventQuery(static_cast<cudaEvent_t>(transfer.event)) == cudaSuccess;
#endif
        if (transfer.dma_complete) transfer.host_lease.reset();
    }
    if (transfer.active_counted) {
        if (active_transfers_ > 0) --active_transfers_;
        transfer.active_counted = false;
    }
#ifdef DEE_CUDA
    if (transfer.staging_slot < staging_slots_.size()) {
        staging_slots_[transfer.staging_slot].busy = false;
        transfer.staging_slot = static_cast<size_t>(-1);
    }
#endif
}

bool AsyncPrefetcher::release_transfer(Transfer& transfer) {
    release_staging(transfer);
    if (transfer.cache_pin_held) {
        cache_.unpin(transfer.key.layer, transfer.key.expert);
        transfer.cache_pin_held = false;
    }
    return true;
}

bool AsyncPrefetcher::validate_invariants(std::string* error) const {
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    std::unordered_set<int> mapped_indices;
    mapped_indices.reserve(key_to_idx_.size());
    size_t active_count = 0;
    for (size_t index = 0; index < inflight_.size(); ++index) {
        const Transfer& transfer = inflight_[index];
        if (transfer.active_counted) ++active_count;
    }
    if (active_count != active_transfers_) {
        return fail("active transfer count does not match transfer metadata");
    }
    for (const auto& entry : key_to_idx_) {
        const int index = entry.second;
        if (index < 0 || index >= static_cast<int>(inflight_.size())) {
            return fail("transfer key maps outside the in-flight table");
        }
        if (!mapped_indices.insert(index).second) {
            return fail("multiple expert keys map to the same transfer");
        }
        const Transfer& transfer = inflight_[static_cast<size_t>(index)];
        if (entry.first != map_key(transfer.key.layer, transfer.key.expert)) {
            return fail("transfer key does not match mapped transfer expert");
        }
        if (transfer.abandoned) {
            return fail("abandoned transfer remains in the current-key map");
        }
        if (!cache_.is_resident(transfer.key.layer, transfer.key.expert)) {
            return fail("current transfer key has no resident cache block");
        }
        if (transfer.dst != cache_.data(
                transfer.key.layer, transfer.key.expert)) {
            return fail("transfer destination is stale for resident cache block");
        }
        if (transfer.nbytes != cache_.size_of(
                transfer.key.layer, transfer.key.expert)) {
            return fail("transfer byte count does not match resident cache block");
        }
        if (transfer.generation == 0 ||
            transfer.generation != cache_.generation_of(
                transfer.key.layer, transfer.key.expert)) {
            return fail("transfer generation does not match resident cache generation");
        }
        if (transfer.cache_pin_held &&
            cache_.pin_count(transfer.key.layer, transfer.key.expert) == 0) {
            return fail("transfer claims a cache pin but resident pin count is zero");
        }
        if (!transfer.done && !transfer.cache_pin_held) {
            return fail("in-flight transfer does not hold its cache block pinned");
        }
    }
    if (error) error->clear();
    return true;
}

long AsyncPrefetcher::validate_request_result(long transfer_id,
                                              const char* context) {
    if (!cache_.debug_validation_enabled()) return transfer_id;
    std::string error;
    if (validate_invariants(&error)) return transfer_id;
    std::ostringstream message;
    message << "prefetch invariant failure after " << context << ": " << error;
    cache_.set_last_error(message.str());
    std::fprintf(stderr, "[dee-cache] %s\n", message.str().c_str());
    return -1;
}

void AsyncPrefetcher::record_request(RequestKind kind, int token, int logical_layer,
                                     int resolved_layer, int expert, int priority,
                                     int evicted_layer, int evicted_expert,
                                     size_t cache_bytes_before,
                                     size_t cache_entries_before,
                                     size_t cache_bytes_after,
                                     size_t cache_entries_after,
                                     size_t source_bytes,
                                     size_t destination_bytes,
                                     uint64_t transfer_id,
                                     bool source_pinned,
                                     uint64_t generation,
                                     uint32_t pin_count,
                                     bool transfer_launched,
                                     uint64_t evicted_generation) {
    ++stats_.requests;
    ++stats_.issued;
    switch (kind) {
        case RequestKind::ResidentHit: ++stats_.resident_hits; break;
        case RequestKind::InflightHit: ++stats_.inflight_hits; break;
        case RequestKind::ColdLoad: ++stats_.cold_loads; break;
    }
    if (profiler_) {
        profiler_->note_request(token, logical_layer, resolved_layer, expert, kind,
                                evicted_layer, evicted_expert, priority,
                                cache_bytes_before, cache_entries_before,
                                cache_bytes_after, cache_entries_after,
                                source_bytes, destination_bytes, transfer_id,
                                source_pinned, generation, pin_count,
                                transfer_launched, evicted_generation);
    }
}

long AsyncPrefetcher::prefetch(int layer, int expert, const void* src, size_t nbytes,
                               int priority, int token, int logical_layer) {
    return prefetch_impl(layer, expert, src, nbytes, nbytes, false, false,
                         false, false, false, nullptr, nullptr, nullptr, nullptr,
                         0, nullptr, false, priority, token, logical_layer);
}

long AsyncPrefetcher::prefetch_bf16_to_f32(int layer, int expert, const uint16_t* src,
                                           size_t elements, int priority,
                                           int token, int logical_layer,
                                           bool source_pinned) {
    if (!use_cuda_) {
        std::fprintf(stderr, "AsyncPrefetcher: BF16 device expansion requires CUDA\n");
        return -1;
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) return -1;
    return prefetch_impl(layer, expert, src, elements * sizeof(uint16_t),
                         elements * sizeof(float), true, false, false, false,
                         false, nullptr, nullptr, nullptr, nullptr,
                         0, nullptr, source_pinned, priority, token, logical_layer);
}

long AsyncPrefetcher::prefetch_bf16_to_f16(int layer, int expert, const uint16_t* src,
                                           size_t elements, int priority,
                                           int token, int logical_layer,
                                           bool source_pinned) {
    if (!use_cuda_) {
        std::fprintf(stderr, "AsyncPrefetcher: BF16-to-FP16 conversion requires CUDA\n");
        return -1;
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return -1;
    return prefetch_impl(layer, expert, src, elements * sizeof(uint16_t),
                         elements * sizeof(uint16_t), true, true, false, false,
                         false, nullptr, nullptr, nullptr, nullptr,
                         0, nullptr, source_pinned, priority, token, logical_layer);
}

long AsyncPrefetcher::prefetch_int8_to_f16(int layer, int expert, const int8_t* src,
                                           size_t elements, size_t projection_elements,
                                           const float scales[3], int priority,
                                           int token, int logical_layer,
                                           bool source_pinned) {
    if (!use_cuda_) {
        std::fprintf(stderr, "AsyncPrefetcher: INT8-to-FP16 conversion requires CUDA\n");
        return -1;
    }
    if (!scales || projection_elements == 0 || elements != 3 * projection_elements ||
        elements > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return -1;
    return prefetch_impl(layer, expert, src, elements, elements * sizeof(uint16_t),
                         false, true, true, false, false, nullptr, nullptr,
                         nullptr, nullptr, projection_elements, scales,
                         source_pinned, priority, token, logical_layer);
}

long AsyncPrefetcher::prefetch_int4_to_f16(int layer, int expert, const uint8_t* src,
                                           size_t elements, size_t projection_elements,
                                           const float scales[3], int priority,
                                           int token, int logical_layer,
                                           bool source_pinned) {
    if (!use_cuda_) {
        std::fprintf(stderr, "AsyncPrefetcher: INT4-to-FP16 conversion requires CUDA\n");
        return -1;
    }
    if (!scales || projection_elements == 0 || elements != 3 * projection_elements ||
        elements > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return -1;
    return prefetch_impl(layer, expert, src, (elements + 1) / 2,
                         elements * sizeof(uint16_t), false, true, false, true,
                         false, nullptr, nullptr, nullptr, nullptr,
                         projection_elements, scales, source_pinned, priority,
                         token, logical_layer);
}

long AsyncPrefetcher::prefetch_fp4_to_f16(int layer, int expert, const uint8_t* src,
                                          size_t source_nbytes,
                                          const size_t packed_offsets[3],
                                          const size_t scale_offsets[3],
                                          const size_t out[3], const size_t in[3],
                                          int priority, int token, int logical_layer,
                                          bool source_pinned) {
    if (!use_cuda_) {
        std::fprintf(stderr, "AsyncPrefetcher: FP4-e2m1 conversion requires CUDA\n");
        return -1;
    }
    if (!src || !packed_offsets || !scale_offsets || !out || !in) return -1;
    size_t destination_elems = 0;
    for (int p = 0; p < 3; ++p) {
        if (out[p] == 0 || in[p] == 0) return -1;
        destination_elems += out[p] * in[p];
    }
    if (destination_elems > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return -1;
    return prefetch_impl(layer, expert, src, source_nbytes,
                         destination_elems * sizeof(uint16_t),
                         false, true, false, false, true,
                         packed_offsets, scale_offsets, out, in,
                         0, nullptr, source_pinned, priority, token, logical_layer);
}

long AsyncPrefetcher::prefetch_fp4_regions_to_f16(
        int layer, int expert, const void* const region_src[6],
        const size_t region_nbytes[6], size_t source_nbytes,
        const size_t packed_offsets[3], const size_t scale_offsets[3],
        const size_t out[3], const size_t in[3],
        int priority, int token, int logical_layer) {
    if (!use_cuda_) {
        std::fprintf(stderr, "AsyncPrefetcher: FP4-e2m1 regions conversion requires CUDA\n");
        return -1;
    }
    if (!region_src || !region_nbytes || !packed_offsets || !scale_offsets || !out || !in) return -1;
    size_t destination_elems = 0;
    for (int p = 0; p < 3; ++p) {
        if (out[p] == 0 || in[p] == 0) return -1;
        destination_elems += out[p] * in[p];
    }
    if (destination_elems > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return -1;
    // src is nullptr; the prefetcher gathers the six regions into its pinned
    // slot. source_pinned=false keeps the existing single-copy H2D path.
    return prefetch_impl(layer, expert, nullptr, source_nbytes,
                         destination_elems * sizeof(uint16_t),
                         false, true, false, false, true,
                         packed_offsets, scale_offsets, out, in,
                         0, nullptr, false, priority, token, logical_layer,
                         region_src, region_nbytes);
}

long AsyncPrefetcher::prefetch_fp4_regions_packed(
        int layer, int expert, const void* const region_src[6],
        const size_t region_nbytes[6], size_t source_nbytes,
        const size_t packed_offsets[3], const size_t scale_offsets[3],
        int priority, int token, int logical_layer) {
    if (!use_cuda_) {
        std::fprintf(stderr, "AsyncPrefetcher: packed FP4 residency requires CUDA\n");
        return -1;
    }
    if (!region_src || !region_nbytes || source_nbytes == 0 ||
        !packed_offsets || !scale_offsets) return -1;
    // Destination block = packed bytes verbatim (no FP16 expansion).  The
    // engine decodes at compute time; the six-region gather + single H2D is
    // identical to the decode-on-transfer path.  The packed/scale offsets are
    // still carried so cuda_submit's pinned-slot gather places each region at
    // its staging-buffer offset ([gate][up][down][g_s][u_s][d_s]).
    return prefetch_impl(layer, expert, nullptr, source_nbytes,
                         source_nbytes,
                         false, false, false, false, false,
                         packed_offsets, scale_offsets, nullptr, nullptr,
                         0, nullptr, false, priority, token, logical_layer,
                         region_src, region_nbytes);
}

long AsyncPrefetcher::prefetch_impl(int layer, int expert, const void* src,
                                    size_t source_nbytes, size_t destination_nbytes,
                                    bool expand_bf16, bool cache_fp16, bool dequantize_int8,
                                    bool dequantize_int4, bool dequantize_fp4,
                                    const size_t* fp4_packed_offsets,
                                    const size_t* fp4_scale_offsets,
                                    const size_t* fp4_out, const size_t* fp4_in,
                                    size_t projection_elements, const float* quant_scales,
                                    bool source_pinned,
                                    int priority, int token,
                                    int logical_layer,
                                    const void* const* fp4_region_src,
                                    const size_t* fp4_region_nbytes,
                                    const HostExpertLease* host_lease) {
    const bool has_regions = fp4_region_src != nullptr && fp4_region_nbytes != nullptr;
    if ((!src && !has_regions) || source_nbytes == 0 || destination_nbytes == 0) {
        std::fprintf(stderr, "AsyncPrefetcher: invalid source for expert (%d,%d)\n", layer, expert);
        return -1;
    }
    const size_t cache_bytes_before = cache_.used_bytes();
    const size_t cache_entries_before = profiler_ ? cache_.resident_count() : 0;
    const auto request_key = map_key(layer, expert);
    if (std::find(batch_keys_.begin(), batch_keys_.end(), request_key) != batch_keys_.end()) {
        ++stats_.duplicate_requests;
        if (profiler_) profiler_->note_duplicate_request();
    } else {
        batch_keys_.push_back(request_key);
    }
    const long existing = find_inflight(layer, expert);
    if (existing >= 0 && existing < static_cast<long>(inflight_.size())) {
        Transfer& prior = inflight_[existing];
        if (host_lease && (prior.key.layer != layer || prior.key.expert != expert ||
            prior.nbytes != destination_nbytes)) return -1;
        if (!prior.done && !prior.abandoned) {
            record_request(RequestKind::InflightHit, token, logical_layer, layer,
                           expert, priority, -1, -1, cache_bytes_before,
                           cache_entries_before, cache_.used_bytes(),
                           cache_.resident_count(), prior.source_nbytes, prior.nbytes,
                           static_cast<uint64_t>(prior.id), prior.source_pinned,
                           prior.generation, cache_.pin_count(layer, expert), false);
            return validate_request_result(
                prior.id, "coalesced in-flight request");
        }
        if (prior.done && cache_.is_resident(layer, expert)) {
            // A resident hit must remain protected while the caller stages
            // other experts in the same batch.  Otherwise a later cold load
            // can evict this hit before wait() consumes it.
            // Route the hit through ensure() so cache-hit accounting, recency,
            // and Oracle priority stay correct on the fast path.
            if (!cache_.ensure(layer, expert, destination_nbytes, priority)) return -1;
            const auto pin_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
            if (!prior.cache_pin_held && !cache_.pin(layer, expert)) return -1;
            if (profiler_ && profiler_->enabled()) profiler_->add_cpu(CpuStage::CacheHitPinning, pin_begin);
            prior.cache_pin_held = true;
            record_request(RequestKind::ResidentHit, token, logical_layer, layer,
                           expert, priority, -1, -1, cache_bytes_before,
                           cache_entries_before, cache_.used_bytes(),
                           cache_.resident_count(), prior.source_nbytes, prior.nbytes,
                           static_cast<uint64_t>(prior.id), prior.source_pinned,
                           prior.generation, cache_.pin_count(layer, expert), false);
            return validate_request_result(prior.id, "resident transfer reuse");
        }
        key_to_idx_.erase(map_key(layer, expert));
    }
    if (cache_.is_resident(layer, expert)) {
        if (!cache_.ensure(layer, expert, destination_nbytes, priority)) return -1;
        const auto pin_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (!cache_.pin(layer, expert)) return -1;
        if (profiler_ && profiler_->enabled()) profiler_->add_cpu(CpuStage::CacheHitPinning, pin_begin);

        Transfer transfer;
        transfer.key = ExpertKey{layer, expert};
        transfer.dst = cache_.data(layer, expert);
        transfer.src = src;
        transfer.nbytes = destination_nbytes;
        transfer.source_nbytes = source_nbytes;
        transfer.expand_bf16 = expand_bf16;
        transfer.cache_fp16 = cache_fp16;
        transfer.dequantize_int8 = dequantize_int8;
        transfer.dequantize_int4 = dequantize_int4;
        transfer.dequantize_fp4 = dequantize_fp4;
        if (fp4_packed_offsets) std::copy(fp4_packed_offsets, fp4_packed_offsets + 3, transfer.fp4_packed_offsets);
        if (fp4_scale_offsets) std::copy(fp4_scale_offsets, fp4_scale_offsets + 3, transfer.fp4_scale_offsets);
        if (fp4_out) std::copy(fp4_out, fp4_out + 3, transfer.fp4_out);
        if (fp4_in) std::copy(fp4_in, fp4_in + 3, transfer.fp4_in);
        if (has_regions) {
            for (int r = 0; r < 6; ++r) {
                transfer.fp4_region_src[r] = fp4_region_src[r];
                transfer.fp4_region_nbytes[r] = fp4_region_nbytes[r];
            }
        }
        transfer.projection_elements = projection_elements;
        if (quant_scales) std::copy(quant_scales, quant_scales + 3, transfer.quant_scales);
        transfer.source_pinned = source_pinned;
        transfer.done = true;
        transfer.id = next_id_++;
        transfer.cache_pin_held = true;
        transfer.token = token;
        transfer.logical_layer = logical_layer;
        transfer.generation = cache_.generation_of(layer, expert);
        inflight_.push_back(transfer);
        const long resident_index = static_cast<long>(inflight_.size() - 1);
        key_to_idx_[request_key] = static_cast<int>(resident_index);
        record_request(RequestKind::ResidentHit, token, logical_layer, layer,
                       expert, priority, -1, -1, cache_bytes_before,
                       cache_entries_before, cache_.used_bytes(),
                       cache_.resident_count(), source_nbytes, destination_nbytes,
                       static_cast<uint64_t>(transfer.id), source_pinned,
                       transfer.generation, cache_.pin_count(layer, expert), false);
        return validate_request_result(transfer.id, "resident cache hit");
    }
    if (!cache_.ensure(layer, expert, destination_nbytes, priority)) return -1;
    const VramCacheManager::EnsureInfo ensure_info = cache_.last_ensure_info();
    if (ensure_info.evicted) {
        key_to_idx_.erase(map_key(ensure_info.evicted_key.layer, ensure_info.evicted_key.expert));
    }
    void* dst = cache_.data(layer, expert);
    if (!dst || !cache_.pin(layer, expert)) return -1;

    Transfer transfer;
    transfer.key = ExpertKey{layer, expert};
    transfer.dst = dst;
    transfer.src = src;
    transfer.nbytes = destination_nbytes;
    transfer.source_nbytes = source_nbytes;
    transfer.expand_bf16 = expand_bf16;
    transfer.cache_fp16 = cache_fp16;
    transfer.dequantize_int8 = dequantize_int8;
    transfer.dequantize_int4 = dequantize_int4;
    transfer.dequantize_fp4 = dequantize_fp4;
    if (fp4_packed_offsets) std::copy(fp4_packed_offsets, fp4_packed_offsets + 3, transfer.fp4_packed_offsets);
    if (fp4_scale_offsets) std::copy(fp4_scale_offsets, fp4_scale_offsets + 3, transfer.fp4_scale_offsets);
    if (fp4_out) std::copy(fp4_out, fp4_out + 3, transfer.fp4_out);
    if (fp4_in) std::copy(fp4_in, fp4_in + 3, transfer.fp4_in);
    if (has_regions) {
        for (int r = 0; r < 6; ++r) {
            transfer.fp4_region_src[r] = fp4_region_src[r];
            transfer.fp4_region_nbytes[r] = fp4_region_nbytes[r];
        }
    }
    transfer.projection_elements = projection_elements;
    if (quant_scales) std::copy(quant_scales, quant_scales + 3, transfer.quant_scales);
    transfer.source_pinned = source_pinned;
    transfer.id = next_id_++;
    transfer.cache_pin_held = true;
    transfer.token = token;
    transfer.logical_layer = logical_layer;
    transfer.generation = ensure_info.generation;
    if (host_lease) {
        transfer.host_lease = *host_lease;
        transfer.managed_source = true;
    }
    inflight_.push_back(transfer);
    const long index = static_cast<long>(inflight_.size() - 1);
    key_to_idx_[map_key(layer, expert)] = static_cast<int>(index);

    if (use_cuda_ && !(host_lease ? cuda_submit_host(index) : cuda_submit(index))) {
        release_transfer(inflight_[index]);
        inflight_[index].abandoned = true;
        key_to_idx_.erase(map_key(layer, expert));
        if (host_lease) cache_.discard_unpinned(layer, expert, transfer.generation);
        return -1;
    }
    record_request(RequestKind::ColdLoad, token, logical_layer, layer, expert, priority,
                   ensure_info.evicted ? ensure_info.evicted_key.layer : -1,
                   ensure_info.evicted ? ensure_info.evicted_key.expert : -1,
                   cache_bytes_before, cache_entries_before,
                   cache_.used_bytes(), cache_.resident_count(),
                   source_nbytes, destination_nbytes,
                   static_cast<uint64_t>(transfer.id), source_pinned,
                   transfer.generation, cache_.pin_count(layer, expert), true,
                   ensure_info.evicted_generation);
    return validate_request_result(transfer.id, "cold transfer launch");
}

void AsyncPrefetcher::drain_until(int index) {
    for (int i = 0; i <= index && i < static_cast<int>(inflight_.size()); ++i) {
        Transfer& t = inflight_[i];
        if (!t.done && !t.abandoned) {
            std::memcpy(t.dst, t.src, t.nbytes);
            t.done = true;
            t.dma_complete = true;
            if (t.managed_source) release_staging(t);
            else release_transfer(t);
        }
    }
}

bool AsyncPrefetcher::wait(int layer, int expert) {
    ++stats_.waited;
    const long index = find_inflight(layer, expert);
    if (index < 0 || index >= static_cast<long>(inflight_.size())) {
        ++stats_.fallbacks;
        return false;
    }
    Transfer& transfer = inflight_[index];
    if (transfer.abandoned) return false;
    if (transfer.managed_source && use_cuda_ && !transfer.dma_complete)
        return cuda_wait(index, HostWaitReason::CacheReadiness);
    if (transfer.done) {
        release_transfer(transfer);
        return cache_.is_resident(layer, expert);
    }
    if (use_cuda_) return cuda_wait(index, HostWaitReason::CacheReadiness);
    const auto wait_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    drain_until(static_cast<int>(index));
    if (transfer.managed_source) release_transfer(transfer);
    if (profiler_ && profiler_->enabled()) profiler_->add_cpu(CpuStage::HostWaiting, wait_begin);
    return transfer.done && cache_.is_resident(layer, expert);
}

bool AsyncPrefetcher::wait_on_stream(int layer, int expert, void* compute_stream) {
#ifdef DEE_CUDA
    const long index = find_inflight(layer, expert);
    if (index < 0 || index >= static_cast<long>(inflight_.size())) return false;
    Transfer& transfer = inflight_[index];
    if (transfer.abandoned) return false;
    if (transfer.done && (!transfer.managed_source || transfer.dma_complete)) {
        // Already complete: no device wait needed, just release ownership.
        release_transfer(transfer);
        return cache_.is_resident(layer, expert);
    }
    if (!use_cuda_ || !transfer.event || !compute_stream) {
        // No event/stream to arm against: fall back to the host-blocking wait.
        return wait(layer, expert);
    }
    // Arm the device-side dependency: the compute stream will not advance past
    // the next cuBLAS GEMM until this expert's H2D + dtype-convert has landed
    // in the cache arena. The HOST does not block here, so it can keep issuing
    // subsequent experts' H2D / routing while the previous expert's compute
    // waits on-device. This is the Milestone 2.5 fix for defect #2
    // (missing compute/transfer overlap). Record GpuStage::StreamWait so the
    // profiler can attribute device-side waits separately from host stalls.
    if (profiler_ && profiler_->enabled()) {
        profiler_->set_cuda_context(transfer.token, transfer.logical_layer,
                                    transfer.key.expert, 0,
                                    static_cast<uint64_t>(transfer.id),
                                    active_transfers_, transfer.staging_slot);
        auto ticket = profiler_->cuda_begin(GpuStage::StreamWait, compute_stream);
        profiler_->cuda_end(ticket, compute_stream);
    }
    if (!DEE_CUDA_CHECK_NAMED(
            cudaStreamWaitEvent(static_cast<cudaStream_t>(compute_stream),
                                static_cast<cudaEvent_t>(transfer.event), 0),
            "cudaStreamWaitEvent(prefetch completion on compute stream)")) {
        return false;
    }
    // The H2D/conversion is in flight on the prefetch stream; once it signals,
    // the staging slot's host source can be recycled (the data lives in the
    // device cache arena, independent of the staging slot).
    //
    // Milestone 3 fix: drop the prefetch transfer's cache pin here.
    // Earlier design assumed a later wait()/synchronize_all() would release;
    // commits 6ed324f + e5a610a removed those teardowns on the device path,
    // leaking pins=1 (field-measured at 992s on dual-T4). Engine re-pins the
    // block before launching cuBLAS, so transient pins==0 is race-free.
    transfer.done = true;
    release_staging(transfer);
    if (transfer.cache_pin_held) {
        cache_.unpin(transfer.key.layer, transfer.key.expert);
        transfer.cache_pin_held = false;
    }
    return cache_.is_resident(layer, expert);
#else
    (void)layer; (void)expert; (void)compute_stream;
    return false;
#endif
}

void AsyncPrefetcher::mark_consumed(int layer, int expert) {
    if (!profiler_) return;
    profiler_->note_transfer_consumed(
        layer, expert, cache_.generation_of(layer, expert));
}

void AsyncPrefetcher::synchronize_all() {
    if (use_cuda_) {
#ifdef DEE_CUDA
        const auto wait_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (stream_ && !DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)),
                                              "cudaStreamSynchronize(prefetch)")) {
            // Phase-2 audit L3 hardening: fail-stop only while leased host DMA
            // sources are actually in flight — an unprovable drain means their
            // memory cannot be freed safely. A stale armed flag with no managed
            // transfers keeps the legacy return path instead of upgrading this
            // failure to a process abort.
            for (const auto& transfer : inflight_) {
                if (transfer.managed_source && transfer.host_lease) {
                    std::terminate(); // cannot free leased DMA sources safely
                }
            }
            return;
        }
        if (stream_ && profiler_ && profiler_->enabled()) {
            const auto wait_end = StageProfiler::now();
            const double wait_ms = std::chrono::duration<double, std::milli>(
                wait_end - wait_begin).count();
            profiler_->add_cpu_ms(CpuStage::Synchronization, wait_ms);
            profiler_->note_host_wait(HostWaitReason::PrefetchDrain, wait_begin, wait_end);
            profiler_->note_host_synchronization();
            profiler_->cuda_collect_ready();
        }
#endif
        for (auto& transfer : inflight_) {
            if (transfer.managed_source) {
                transfer.dma_complete = true;
                release_transfer(transfer);
            }
            if (!transfer.done && !transfer.abandoned) {
                transfer.done = true;
                release_transfer(transfer);
            }
        }
    } else {
        drain_until(static_cast<int>(inflight_.size()) - 1);
        for (auto& transfer : inflight_)
            if (transfer.managed_source) release_transfer(transfer);
    }
}

void AsyncPrefetcher::reset() {
    synchronize_all();
#ifdef DEE_CUDA
    for (auto& transfer : inflight_) {
        if (transfer.event) {
            DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_DESTROY(static_cast<cudaEvent_t>(transfer.event), "transfer"),
                                 "cudaEventDestroy(prefetch completion)");
            transfer.event = nullptr;
        }
        release_transfer(transfer);
    }
#endif
    inflight_.clear();
    key_to_idx_.clear();
    active_transfers_ = 0;
}

bool AsyncPrefetcher::cuda_init() {
#ifdef DEE_CUDA
    cudaStream_t stream = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(DEE_TA_STREAM_CREATE_FLAGS(&stream, cudaStreamNonBlocking, "stream"),
                              "cudaStreamCreateWithFlags(prefetch)")) return false;
    stream_ = static_cast<void*>(stream);
    return true;
#else
    std::fprintf(stderr, "AsyncPrefetcher: --cuda requested but DEE_CUDA=OFF\n");
    return false;
#endif
}

long AsyncPrefetcher::prefetch_host_lease(const HostExpertLease& lease,
        int priority, int token, int logical_layer) {
    if (!experimental_host_tier_ || !lease || !lease.key().valid() ||
        lease.key().model != experimental_scope_.model ||
        lease.key().representation != experimental_scope_.representation) return -1;
    return prefetch_impl(lease.key().layer, lease.key().expert, lease.data(),
        lease.size(), lease.size(), false, false, false, false, false,
        nullptr, nullptr, nullptr, nullptr, 0, nullptr, lease.pinned(),
        priority, token, logical_layer, nullptr, nullptr, &lease);
}

bool AsyncPrefetcher::collect_host_sources(bool wait_one) {
    bool collected = false;
    for (auto& transfer : inflight_) {
        if (!transfer.host_lease) continue;
#ifdef DEE_CUDA
        if (use_cuda_) {
            if (!transfer.event) continue;
            const auto event = static_cast<cudaEvent_t>(transfer.event);
            auto status = cudaEventQuery(event);
            if (status == cudaErrorNotReady && wait_one && !collected)
                status = cudaEventSynchronize(event);
            if (status == cudaErrorNotReady) continue;
            if (!DEE_CUDA_CHECK_NAMED(status, "phase2 host lease completion")) return false;
            transfer.dma_complete = true;
        } else
#endif
        {
            if (!wait_one) continue;
            std::memcpy(transfer.dst, transfer.src, transfer.nbytes);
            transfer.dma_complete = true;
            transfer.done = true;
        }
        release_staging(transfer);
        collected = true;
        if (wait_one) break;
    }
    return collected;
}

bool AsyncPrefetcher::cuda_submit_host(long index) {
#ifdef DEE_CUDA
    auto& transfer = inflight_[index];
    if (!stream_ || !transfer.host_lease) {
        transfer.dma_complete = true;
        return false;
    }
    const auto stream = static_cast<cudaStream_t>(stream_);
    cudaEvent_t event = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_CREATE_FLAGS(&event, cudaEventDisableTiming, "phase2_event"),
                              "phase2 completion event")) {
        transfer.dma_complete = true; // nothing submitted
        return false;
    }
    transfer.event = event;
    const auto begin = StageProfiler::now();
    const bool profiling = profiler_ && profiler_->enabled();
    if (profiling) profiler_->set_cuda_context(transfer.token, transfer.logical_layer,
        transfer.key.expert, transfer.source_nbytes, transfer.id, active_transfers_ + 1,
        static_cast<size_t>(-1));
    const auto ticket = profiling ? profiler_->cuda_begin(GpuStage::H2D, stream_) : static_cast<size_t>(-1);
    bool ok = DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(transfer.dst, transfer.src,
        transfer.source_nbytes, cudaMemcpyHostToDevice, stream), "phase2 packed H2D");
    if (ok && profiling) ok = profiler_->cuda_end(ticket, stream_);
    if (ok) ok = DEE_CUDA_CHECK_NAMED(cudaEventRecord(event, stream), "phase2 record H2D completion");
    if (!ok) {
        // Error only: DMA may have launched before event recording failed.
        // Do not recycle either source or destination until this stream drains.
        if (cudaStreamSynchronize(stream) != cudaSuccess) std::terminate();
        // Phase-2 audit L4 hardening: the drain proved no in-flight work can
        // still signal this event, so destroy it now. Otherwise it would sit
        // on the abandoned transfer until reset() — a bounded event-object
        // leak per failed submit.
        DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_DESTROY(event, "phase2_event"),
                             "cudaEventDestroy(phase2 failed submit)");
        transfer.event = nullptr;
        transfer.dma_complete = true;
        return false;
    }
    if (!transfer.source_pinned) {
        // Registration failure keeps the same aligned host slot. CUDA may stage
        // pageable input internally; wait only for this copy, never the device.
        const auto fallback_begin = std::chrono::steady_clock::now();
        if (!DEE_CUDA_CHECK_NAMED(cudaEventSynchronize(event), "phase2 pageable H2D fallback"))
            std::terminate();
        experimental_pageable_wait_ms_ += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fallback_begin).count();
        transfer.dma_complete = true;
        transfer.host_lease.reset();
    }
    if (profiling) {
        profiler_->add_cpu(CpuStage::TransferSubmission, begin);
        profiler_->note_h2d_copy(transfer.source_nbytes);
        profiler_->note_cpu_timeline(CpuTimelineKind::TransferSubmit, begin,
            transfer.token, transfer.logical_layer, transfer.key.expert,
            transfer.source_nbytes, transfer.id, active_transfers_ + 1, static_cast<size_t>(-1));
    }
    stats_.h2d_bytes += transfer.source_nbytes;
    ++stats_.h2d_copies;
    transfer.active_counted = true; ++active_transfers_;
    return true;
#else
    (void)index;
    return false;
#endif
}

bool AsyncPrefetcher::cuda_submit(long index) {
#ifdef DEE_CUDA
    if (!stream_ || index < 0 || index >= static_cast<long>(inflight_.size())) return false;
    Transfer& transfer = inflight_[index];

    size_t chosen = static_cast<size_t>(-1);
    for (size_t attempt = 0; attempt < staging_slots_.size(); ++attempt) {
        const size_t slot = (next_staging_slot_ + attempt) % staging_slots_.size();
        if (!staging_slots_[slot].busy &&
            (transfer.source_pinned || staging_slots_[slot].bytes >= transfer.source_nbytes)) {
            chosen = slot;
            break;
        }
    }
    if (chosen == static_cast<size_t>(-1) && staging_slots_.size() < ring_size_) {
        PinnedStagingSlot slot;
        slot.bytes = transfer.source_nbytes;
        if (!DEE_CUDA_CHECK_NAMED(DEE_TA_MALLOC_HOST(&slot.ptr, slot.bytes, "slot"),
                                  "cudaMallocHost(pinned staging slot)")) return false;
        staging_slots_.push_back(slot);
        chosen = staging_slots_.size() - 1;
    }
    if (chosen == static_cast<size_t>(-1)) {
        // A bounded ring is full. Complete the oldest outstanding transfer
        // before reusing its host memory; no CUDA copy can retain that slot.
        for (long pending_index = 0; pending_index < static_cast<long>(inflight_.size()); ++pending_index) {
            Transfer& pending = inflight_[pending_index];
            if (!pending.done && pending.staging_slot < staging_slots_.size()) {
                if (!cuda_wait(pending_index, HostWaitReason::StagingSlot)) return false;
                break;
            }
        }
        for (size_t attempt = 0; attempt < staging_slots_.size(); ++attempt) {
            const size_t slot = (next_staging_slot_ + attempt) % staging_slots_.size();
            if (!staging_slots_[slot].busy &&
                (transfer.source_pinned || staging_slots_[slot].bytes >= transfer.source_nbytes)) {
                chosen = slot;
                break;
            }
        }
    }
    if (chosen == static_cast<size_t>(-1)) {
        std::fprintf(stderr, "AsyncPrefetcher: pinned staging ring has no slot for %zu bytes\n",
                     transfer.source_nbytes);
        return false;
    }

    PinnedStagingSlot& slot = staging_slots_[chosen];
    if ((transfer.expand_bf16 || transfer.dequantize_int8 || transfer.dequantize_int4 ||
         transfer.dequantize_fp4) &&
        slot.device_bytes < transfer.source_nbytes) {
        if (slot.device_ptr &&
            !DEE_CUDA_CHECK_NAMED(DEE_TA_FREE(slot.device_ptr, "slot"),
                                  "cudaFree(resize BF16 device staging slot)")) return false;
        slot.device_ptr = nullptr;
        if (!DEE_CUDA_CHECK_NAMED(DEE_TA_MALLOC(&slot.device_ptr, transfer.source_nbytes, "slot"),
                                  "cudaMalloc(BF16 device staging slot)")) return false;
        slot.device_bytes = transfer.source_nbytes;
    }
    const void* h2d_source = transfer.src;
    if (!transfer.source_pinned) {
        const auto copy_begin = profiler_ && profiler_->enabled()
            ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (transfer.fp4_region_src[0] != nullptr) {
            // Gather the six non-contiguous mmap regions into the pinned slot
            // at their destination offsets in one pass (no heap intermediate).
            auto* dst = static_cast<uint8_t*>(slot.ptr);
            for (int p = 0; p < 3; ++p) {
                std::memcpy(dst + transfer.fp4_packed_offsets[p],
                            transfer.fp4_region_src[p], transfer.fp4_region_nbytes[p]);
                std::memcpy(dst + transfer.fp4_scale_offsets[p],
                            transfer.fp4_region_src[3 + p], transfer.fp4_region_nbytes[3 + p]);
            }
        } else {
            std::memcpy(slot.ptr, transfer.src, transfer.source_nbytes);
        }
        h2d_source = slot.ptr;
        if (profiler_ && profiler_->enabled()) {
            profiler_->add_cpu(CpuStage::MmapToPinned, copy_begin);
            profiler_->note_mmap_copy(transfer.source_nbytes);
        }
        stats_.mmap_to_pinned_bytes += transfer.source_nbytes;
    }
    slot.busy = true;
    next_staging_slot_ = (chosen + 1) % staging_slots_.size();
    transfer.staging_slot = chosen;

    const auto submission_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    cudaEvent_t event = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_CREATE_FLAGS(&event, cudaEventDisableTiming, "event"),
                              "cudaEventCreateWithFlags(prefetch completion)")) return false;
    transfer.event = static_cast<void*>(event);
    // Failed-submit cleanup, shared by every return below (Phase-2 audit L4
    // parity with cuda_submit_host): a failed call may have launched work
    // before the error surfaced, and the caller's release_transfer recycles
    // the staging slot + cache block, so drain the stream first — an
    // unprovable drain fail-stops rather than recycling memory in flight —
    // then destroy the per-transfer event instead of leaving it on the
    // abandoned transfer until reset() (a bounded event-object leak per
    // failed submit). dma_complete is managed_source/host_lease state and is
    // never consulted for legacy transfers, so it is not touched here.
    auto fail_submit = [&]() {
        if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)) != cudaSuccess)
            std::terminate();
        DEE_CUDA_CHECK_NAMED(DEE_TA_EVENT_DESTROY(event, "event"),
                             "cudaEventDestroy(failed prefetch submit)");
        transfer.event = nullptr;
        return false;
    };
    const size_t queue_depth = active_transfers_ + 1;
    if (profiler_ && profiler_->enabled()) {
        profiler_->set_cuda_context(transfer.token, transfer.logical_layer,
                                    transfer.key.expert, transfer.source_nbytes,
                                    static_cast<uint64_t>(transfer.id), queue_depth, chosen);
    }
    const size_t h2d_ticket = profiler_ && profiler_->enabled()
        ? profiler_->cuda_begin(GpuStage::H2D, stream_) : static_cast<size_t>(-1);
    void* copy_destination = (transfer.expand_bf16 || transfer.dequantize_int8 ||
                              transfer.dequantize_int4 || transfer.dequantize_fp4)
        ? slot.device_ptr : transfer.dst;
    if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(copy_destination, h2d_source, transfer.source_nbytes,
                                              cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)),
                              "cudaMemcpyAsync(pinned staging to expert cache)")) return fail_submit();
    if (profiler_ && profiler_->enabled() && !profiler_->cuda_end(h2d_ticket, stream_)) return fail_submit();
    if (transfer.expand_bf16) {
        const size_t elements = transfer.source_nbytes / sizeof(uint16_t);
        const bool converted = transfer.cache_fp16
            ? bf16_to_f16_cuda(static_cast<const uint16_t*>(slot.device_ptr), transfer.dst,
                               elements, static_cast<cudaStream_t>(stream_), profiler_)
            : bf16_to_f32_cuda(static_cast<const uint16_t*>(slot.device_ptr),
                               static_cast<float*>(transfer.dst), elements,
                               static_cast<cudaStream_t>(stream_), profiler_);
        if (!converted) return fail_submit();
    }
    if (transfer.dequantize_int8 &&
        !int8_to_f16_cuda(static_cast<const int8_t*>(slot.device_ptr), transfer.dst,
                          transfer.source_nbytes, transfer.projection_elements,
                          transfer.quant_scales, static_cast<cudaStream_t>(stream_),
                          profiler_)) return fail_submit();
    if (transfer.dequantize_int4 &&
        !int4_to_f16_cuda(static_cast<const uint8_t*>(slot.device_ptr), transfer.dst,
                          transfer.nbytes / sizeof(uint16_t), transfer.projection_elements,
                          transfer.quant_scales, static_cast<cudaStream_t>(stream_),
                          profiler_)) return fail_submit();
    if (transfer.dequantize_fp4) {
        uint16_t* dst16 = static_cast<uint16_t*>(transfer.dst);
        size_t decoded_elems_before = 0;
        for (int p = 0; p < 3; ++p) {
            const uint8_t* packed = static_cast<const uint8_t*>(slot.device_ptr) +
                                    transfer.fp4_packed_offsets[p];
            const uint8_t* scale  = static_cast<const uint8_t*>(slot.device_ptr) +
                                    transfer.fp4_scale_offsets[p];
            if (!fp4_e2m1_to_f16_cuda(packed, scale, dst16 + decoded_elems_before,
                                      transfer.fp4_out[p], transfer.fp4_in[p],
                                      static_cast<cudaStream_t>(stream_), profiler_)) return fail_submit();
            decoded_elems_before += transfer.fp4_out[p] * transfer.fp4_in[p];
        }
    }
    if (!DEE_CUDA_CHECK_NAMED(cudaEventRecord(event, static_cast<cudaStream_t>(stream_)),
                              "cudaEventRecord(prefetch completion)")) return fail_submit();
    if (profiler_ && profiler_->enabled()) {
        profiler_->add_cpu(CpuStage::TransferSubmission, submission_begin);
        profiler_->note_h2d_copy(transfer.source_nbytes);
        profiler_->note_cpu_timeline(CpuTimelineKind::TransferSubmit, submission_begin,
                                     transfer.token, transfer.logical_layer,
                                     transfer.key.expert, transfer.source_nbytes,
                                     static_cast<uint64_t>(transfer.id), queue_depth, chosen);
    }
    stats_.h2d_bytes += transfer.source_nbytes;
    ++stats_.h2d_copies;
    transfer.active_counted = true;
    ++active_transfers_;
    return true;
#else
    (void)index;
    return false;
#endif
}

bool AsyncPrefetcher::cuda_wait(long index, HostWaitReason reason) {
#ifdef DEE_CUDA
    if (index < 0 || index >= static_cast<long>(inflight_.size())) return false;
    Transfer& transfer = inflight_[index];
    const auto tier_begin = transfer.managed_source ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
    const auto wait_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (transfer.event && !DEE_CUDA_CHECK_NAMED(cudaEventSynchronize(static_cast<cudaEvent_t>(transfer.event)),
                                                 "cudaEventSynchronize(prefetch completion)")) return false;
    if (profiler_ && profiler_->enabled()) {
        const auto wait_end = StageProfiler::now();
        const double wait_ms = std::chrono::duration<double, std::milli>(
            wait_end - wait_begin).count();
        profiler_->add_cpu_ms(CpuStage::HostWaiting, wait_ms);
        profiler_->note_host_wait(reason, wait_begin, wait_end, transfer.token,
                                  transfer.logical_layer, transfer.key.expert,
                                  static_cast<uint64_t>(transfer.id), active_transfers_,
                                  transfer.staging_slot);
        profiler_->note_host_synchronization();
    }
    transfer.done = true;
    transfer.dma_complete = true;
    if (transfer.managed_source) experimental_readiness_wait_ms_ +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tier_begin).count();
    if (reason == HostWaitReason::StagingSlot) {
        // The DMA/conversion no longer owns its staging slot, but the cache
        // block must remain pinned until wait(layer, expert) hands it to the
        // compute consumer. Dropping both lifetimes here permits eviction
        // when ring depth is smaller than the active expert batch.
        release_staging(transfer);
    } else {
        release_transfer(transfer);
    }
    return cache_.is_resident(transfer.key.layer, transfer.key.expert);
#else
    (void)index;
    (void)reason;
    return false;
#endif
}

}  // namespace dee
