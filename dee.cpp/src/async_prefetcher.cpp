#include "dee/async_prefetcher.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#ifdef DEE_CUDA
#include "dee/cuda_check.h"
#include <cuda_runtime.h>
#endif

namespace dee {
namespace {

size_t key_id(int layer, int expert) {
    return (static_cast<size_t>(static_cast<uint32_t>(layer)) << 32) |
           static_cast<uint32_t>(expert);
}

}  // namespace

AsyncPrefetcher::AsyncPrefetcher(VramCacheManager& cache, size_t ring_size)
    : cache_(cache), ring_size_(ring_size ? ring_size : 1) {}

AsyncPrefetcher::~AsyncPrefetcher() {
    reset();
#ifdef DEE_CUDA
    for (auto& slot : staging_slots_) {
        if (slot.ptr) DEE_CUDA_CHECK_NAMED(cudaFreeHost(slot.ptr), "cudaFreeHost(pinned staging slot)");
    }
    if (stream_) DEE_CUDA_CHECK_NAMED(cudaStreamDestroy(static_cast<cudaStream_t>(stream_)),
                                       "cudaStreamDestroy(prefetch)");
#endif
}

bool AsyncPrefetcher::init(bool use_cuda) {
    if (!inflight_.empty()) reset();
    use_cuda_ = use_cuda;
    return !use_cuda_ || cuda_init();
}

long AsyncPrefetcher::find_inflight(int layer, int expert) const {
    const auto it = key_to_idx_.find(static_cast<long>(key_id(layer, expert)));
    return it == key_to_idx_.end() ? -1 : it->second;
}

bool AsyncPrefetcher::release_transfer(Transfer& transfer) {
    if (transfer.cache_pin_held) {
        cache_.unpin(transfer.key.layer, transfer.key.expert);
        transfer.cache_pin_held = false;
    }
#ifdef DEE_CUDA
    if (transfer.staging_slot < staging_slots_.size()) staging_slots_[transfer.staging_slot].busy = false;
#endif
    return true;
}

void AsyncPrefetcher::record_request(RequestKind kind, int token, int logical_layer,
                                     int resolved_layer, int expert, int priority,
                                     int evicted_layer, int evicted_expert) {
    ++stats_.requests;
    ++stats_.issued;
    switch (kind) {
        case RequestKind::ResidentHit: ++stats_.resident_hits; break;
        case RequestKind::InflightHit: ++stats_.inflight_hits; break;
        case RequestKind::ColdLoad: ++stats_.cold_loads; break;
    }
    if (profiler_) {
        profiler_->note_request(token, logical_layer, resolved_layer, expert, kind,
                                cache_.used_bytes(), evicted_layer, evicted_expert, priority);
    }
}

long AsyncPrefetcher::prefetch(int layer, int expert, const void* src, size_t nbytes,
                               int priority, int token, int logical_layer) {
    if (!src || nbytes == 0) {
        std::fprintf(stderr, "AsyncPrefetcher: invalid source for expert (%d,%d)\n", layer, expert);
        return -1;
    }
    const long request_key = static_cast<long>(key_id(layer, expert));
    if (std::find(batch_keys_.begin(), batch_keys_.end(), request_key) != batch_keys_.end()) {
        ++stats_.duplicate_requests;
        if (profiler_) profiler_->note_duplicate_request();
    } else {
        batch_keys_.push_back(request_key);
    }
    const long existing = find_inflight(layer, expert);
    if (existing >= 0 && existing < static_cast<long>(inflight_.size())) {
        Transfer& prior = inflight_[existing];
        if (!prior.done && !prior.abandoned) {
            record_request(RequestKind::InflightHit, token, logical_layer, layer, expert, priority);
            return prior.id;  // never duplicate an in-flight DMA
        }
        if (prior.done && cache_.is_resident(layer, expert)) {
            // A resident hit must remain protected while the caller stages
            // other experts in the same batch.  Otherwise a later cold load
            // can evict this hit before wait() consumes it.
            // Route the hit through ensure() so cache-hit accounting, recency,
            // and Oracle priority stay correct on the fast path.
            if (!cache_.ensure(layer, expert, nbytes, priority)) return -1;
            const auto pin_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
            if (!prior.cache_pin_held && !cache_.pin(layer, expert)) return -1;
            if (profiler_ && profiler_->enabled()) profiler_->add_cpu(CpuStage::CacheHitPinning, pin_begin);
            prior.cache_pin_held = true;
            record_request(RequestKind::ResidentHit, token, logical_layer, layer, expert, priority);
            return prior.id;
        }
        key_to_idx_.erase(static_cast<long>(key_id(layer, expert)));
    }
    if (!cache_.ensure(layer, expert, nbytes, priority)) return -1;
    const VramCacheManager::EnsureInfo ensure_info = cache_.last_ensure_info();
    void* dst = cache_.data(layer, expert);
    if (!dst || !cache_.pin(layer, expert)) return -1;

    Transfer transfer;
    transfer.key = ExpertKey{layer, expert};
    transfer.dst = dst;
    transfer.src = src;
    transfer.nbytes = nbytes;
    transfer.id = next_id_++;
    transfer.cache_pin_held = true;
    inflight_.push_back(transfer);
    const long index = static_cast<long>(inflight_.size() - 1);
    key_to_idx_[static_cast<long>(key_id(layer, expert))] = static_cast<int>(index);

    if (use_cuda_ && !cuda_submit(index)) {
        release_transfer(inflight_[index]);
        inflight_[index].abandoned = true;
        key_to_idx_.erase(static_cast<long>(key_id(layer, expert)));
        return -1;
    }
    record_request(RequestKind::ColdLoad, token, logical_layer, layer, expert, priority,
                   ensure_info.evicted ? ensure_info.evicted_key.layer : -1,
                   ensure_info.evicted ? ensure_info.evicted_key.expert : -1);
    return transfer.id;
}

void AsyncPrefetcher::drain_until(int index) {
    for (int i = 0; i <= index && i < static_cast<int>(inflight_.size()); ++i) {
        Transfer& t = inflight_[i];
        if (!t.done && !t.abandoned) {
            std::memcpy(t.dst, t.src, t.nbytes);
            t.done = true;
            release_transfer(t);
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
    if (transfer.done) {
        release_transfer(transfer);
        return cache_.is_resident(layer, expert);
    }
    if (use_cuda_) return cuda_wait(index);
    const auto wait_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    drain_until(static_cast<int>(index));
    if (profiler_ && profiler_->enabled()) profiler_->add_cpu(CpuStage::HostWaiting, wait_begin);
    return transfer.done && cache_.is_resident(layer, expert);
}

void AsyncPrefetcher::synchronize_all() {
    if (use_cuda_) {
#ifdef DEE_CUDA
        const auto wait_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
        if (stream_ && !DEE_CUDA_CHECK_NAMED(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)),
                                              "cudaStreamSynchronize(prefetch)")) return;
        if (stream_ && profiler_ && profiler_->enabled()) {
            profiler_->add_cpu(CpuStage::Synchronization, wait_begin);
            profiler_->note_host_synchronization();
            profiler_->cuda_collect_ready();
        }
#endif
        for (auto& transfer : inflight_) {
            if (!transfer.done && !transfer.abandoned) {
                transfer.done = true;
                release_transfer(transfer);
            }
        }
    } else {
        drain_until(static_cast<int>(inflight_.size()) - 1);
    }
}

void AsyncPrefetcher::reset() {
    synchronize_all();
#ifdef DEE_CUDA
    for (auto& transfer : inflight_) {
        if (transfer.event) {
            DEE_CUDA_CHECK_NAMED(cudaEventDestroy(static_cast<cudaEvent_t>(transfer.event)),
                                 "cudaEventDestroy(prefetch completion)");
            transfer.event = nullptr;
        }
        release_transfer(transfer);
    }
#endif
    inflight_.clear();
    key_to_idx_.clear();
}

bool AsyncPrefetcher::cuda_init() {
#ifdef DEE_CUDA
    cudaStream_t stream = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                              "cudaStreamCreateWithFlags(prefetch)")) return false;
    stream_ = static_cast<void*>(stream);
    return true;
#else
    std::fprintf(stderr, "AsyncPrefetcher: --cuda requested but DEE_CUDA=OFF\n");
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
        if (!staging_slots_[slot].busy && staging_slots_[slot].bytes >= transfer.nbytes) {
            chosen = slot;
            break;
        }
    }
    if (chosen == static_cast<size_t>(-1) && staging_slots_.size() < ring_size_) {
        PinnedStagingSlot slot;
        slot.bytes = transfer.nbytes;
        if (!DEE_CUDA_CHECK_NAMED(cudaMallocHost(&slot.ptr, slot.bytes), "cudaMallocHost(pinned staging slot)")) return false;
        staging_slots_.push_back(slot);
        chosen = staging_slots_.size() - 1;
    }
    if (chosen == static_cast<size_t>(-1)) {
        // A bounded ring is full. Complete the oldest outstanding transfer
        // before reusing its host memory; no CUDA copy can retain that slot.
        for (long pending_index = 0; pending_index < static_cast<long>(inflight_.size()); ++pending_index) {
            Transfer& pending = inflight_[pending_index];
            if (!pending.done && pending.staging_slot < staging_slots_.size()) {
                if (!cuda_wait(pending_index)) return false;
                break;
            }
        }
        for (size_t attempt = 0; attempt < staging_slots_.size(); ++attempt) {
            const size_t slot = (next_staging_slot_ + attempt) % staging_slots_.size();
            if (!staging_slots_[slot].busy && staging_slots_[slot].bytes >= transfer.nbytes) { chosen = slot; break; }
        }
    }
    if (chosen == static_cast<size_t>(-1)) {
        std::fprintf(stderr, "AsyncPrefetcher: pinned staging ring has no slot for %zu bytes\n", transfer.nbytes);
        return false;
    }

    PinnedStagingSlot& slot = staging_slots_[chosen];
    const auto copy_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    std::memcpy(slot.ptr, transfer.src, transfer.nbytes);  // mmap/pageable -> pinned is CPU work
    if (profiler_ && profiler_->enabled()) {
        profiler_->add_cpu(CpuStage::MmapToPinned, copy_begin);
        profiler_->note_mmap_copy(transfer.nbytes);
    }
    stats_.mmap_to_pinned_bytes += transfer.nbytes;
    slot.busy = true;
    next_staging_slot_ = (chosen + 1) % staging_slots_.size();
    transfer.staging_slot = chosen;

    const auto submission_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    cudaEvent_t event = nullptr;
    if (!DEE_CUDA_CHECK_NAMED(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                              "cudaEventCreateWithFlags(prefetch completion)")) return false;
    transfer.event = static_cast<void*>(event);
    const size_t h2d_ticket = profiler_ && profiler_->enabled()
        ? profiler_->cuda_begin(GpuStage::H2D, stream_) : static_cast<size_t>(-1);
    if (!DEE_CUDA_CHECK_NAMED(cudaMemcpyAsync(transfer.dst, slot.ptr, transfer.nbytes,
                                              cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)),
                              "cudaMemcpyAsync(pinned staging to expert cache)")) return false;
    if (profiler_ && profiler_->enabled() && !profiler_->cuda_end(h2d_ticket, stream_)) return false;
    if (!DEE_CUDA_CHECK_NAMED(cudaEventRecord(event, static_cast<cudaStream_t>(stream_)),
                              "cudaEventRecord(prefetch completion)")) return false;
    if (profiler_ && profiler_->enabled()) {
        profiler_->add_cpu(CpuStage::TransferSubmission, submission_begin);
        profiler_->note_h2d_copy(transfer.nbytes);
    }
    stats_.h2d_bytes += transfer.nbytes;
    ++stats_.h2d_copies;
    return true;
#else
    (void)index;
    return false;
#endif
}

bool AsyncPrefetcher::cuda_wait(long index) {
#ifdef DEE_CUDA
    if (index < 0 || index >= static_cast<long>(inflight_.size())) return false;
    Transfer& transfer = inflight_[index];
    const auto wait_begin = profiler_ && profiler_->enabled() ? StageProfiler::now() : StageProfiler::TimePoint{};
    if (transfer.event && !DEE_CUDA_CHECK_NAMED(cudaEventSynchronize(static_cast<cudaEvent_t>(transfer.event)),
                                                 "cudaEventSynchronize(prefetch completion)")) return false;
    if (profiler_ && profiler_->enabled()) {
        profiler_->add_cpu(CpuStage::HostWaiting, wait_begin);
        profiler_->note_host_synchronization();
    }
    transfer.done = true;
    release_transfer(transfer);
    return cache_.is_resident(transfer.key.layer, transfer.key.expert);
#else
    (void)index;
    return false;
#endif
}

}  // namespace dee
