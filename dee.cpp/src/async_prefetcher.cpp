// dee/async_prefetcher.cpp
#include "dee/async_prefetcher.h"

#include <cstdio>
#include <cstring>

namespace dee {

AsyncPrefetcher::AsyncPrefetcher(VramCacheManager& cache, size_t ring_size)
    : cache_(cache), ring_size_(ring_size) {}

AsyncPrefetcher::~AsyncPrefetcher() {
    for (auto& t : inflight_) {
#ifdef DEE_CUDA
        if (t.event) { cudaEventDestroy((cudaEvent_t)t.event); t.event = nullptr; }
#else
        (void)t;
#endif
    }
    if (use_cuda_) {
#ifdef DEE_CUDA
        // real teardown
        if (stream_) { cudaStreamDestroy(*(cudaStream_t*)stream_); delete (cudaStream_t*)stream_; }
        if (host_buf_) { cudaFreeHost(host_buf_); }
#else
        fprintf(stderr, "AsyncPrefetcher: CUDA backend requested but not built (DEE_CUDA=OFF)\n");
#endif
    }
    inflight_.clear();
    key_to_idx_.clear();
}

void AsyncPrefetcher::reset() {
#ifdef DEE_CUDA
    for (auto& t : inflight_) {
        if (t.event) { cudaEventDestroy((cudaEvent_t)t.event); t.event = nullptr; }
    }
#endif
    inflight_.clear();
    key_to_idx_.clear();
}

bool AsyncPrefetcher::init(bool use_cuda) {
    use_cuda_ = use_cuda;
    if (use_cuda_) return cuda_init();
    // mock: nothing to allocate; transfers run lazily on wait()
    return true;
}

bool AsyncPrefetcher::cuda_init() {
#ifdef DEE_CUDA
    stream_ = new cudaStream_t;
    cudaError_t e = cudaStreamCreate((cudaStream_t*)stream_);
    if (e != cudaSuccess) { fprintf(stderr, "cudaStreamCreate failed\n"); return false; }
    // 64 MB pinned host staging buffer (llama.cpp's "works well for NVMe" size)
    host_buf_bytes_ = 64 * 1024 * 1024;
    e = cudaMallocHost(&host_buf_, host_buf_bytes_);
    if (e != cudaSuccess) { fprintf(stderr, "cudaMallocHost failed\n"); host_buf_ = nullptr; }
    return true;
#else
    (void)host_buf_bytes_;
    fprintf(stderr, "AsyncPrefetcher: DEE_CUDA=OFF, cannot use real CUDA stream\n");
    return false;
#endif
}

long AsyncPrefetcher::find_inflight(int layer, int expert) const {
    ExpertKey k{layer, expert};
    auto it = key_to_idx_.find(((size_t)(uint32_t)k.layer << 32) ^ (size_t)(uint32_t)k.expert);
    if (it == key_to_idx_.end()) return -1;
    return it->second;
}

long AsyncPrefetcher::prefetch(int layer, int expert, const void* src, size_t nbytes,
                               int priority) {
    // Reserve VRAM slot (evicting if needed) — this is synchronous metadata work;
    // the COPY itself is what runs on the async stream.
    if (!cache_.ensure(layer, expert, nbytes, priority)) return -1;
    void* dst = cache_.data(layer, expert);
    if (!dst) return -1;

    Transfer t;
    t.key.layer = layer;
    t.key.expert = expert;
    t.dst  = dst;
    t.src  = src;
    t.nbytes = nbytes;
    t.done = false;

    long id = next_id_++;

    // store in ring (drop oldest if over ring_size to bound memory)
    inflight_.push_back(t);
    size_t idx = inflight_.size() - 1;
    key_to_idx_[((size_t)(uint32_t)layer << 32) ^ (size_t)(uint32_t)expert] = (int)idx;
    // keep key_to_idx valid if we ever trim; for the mock we keep all.

    ++stats_.issued;
    if (use_cuda_) cuda_submit((long)idx);
    return id;
}
void AsyncPrefetcher::drain_until(int idx) {
    // mock stream: run copies in submission order up to idx inclusive.
    for (int i = 0; i <= idx && i < (int)inflight_.size(); ++i) {
        Transfer& t = inflight_[i];
        if (!t.done && !t.abandoned && t.src && t.dst) {
            std::memcpy(t.dst, t.src, t.nbytes);
            t.done = true;
        }
    }
}

bool AsyncPrefetcher::wait(int layer, int expert) {
    ++stats_.waited;
    long idx = find_inflight(layer, expert);
    if (idx < 0) {
        // not even issued — treat as fallback (compute reached an un-staged expert)
        ++stats_.fallbacks;
        return false;
    }
    if (use_cuda_) {
        cuda_wait(idx);
        return true;
    }
    // mock: drain the stream up to and including this transfer. This is the
    // normal streaming model (the copy completes exactly when compute needs
    // it) and is NOT a fallback. A genuine fallback is only when compute
    // reaches an expert whose prefetch was never issued (handled above).
    drain_until((int)idx);
    return inflight_[idx].done;
}

void AsyncPrefetcher::synchronize_all() {
    if (use_cuda_) {
#ifdef DEE_CUDA
        if (stream_) cudaStreamSynchronize(*(cudaStream_t*)stream_);
#endif
    } else {
        drain_until((int)inflight_.size() - 1);
    }
}

void AsyncPrefetcher::cuda_submit(long idx) {
#ifdef DEE_CUDA
    if (idx < 0 || idx >= (long)inflight_.size()) return;
    Transfer& t = inflight_[idx];
    if (!stream_) return;
    // Real path: copy straight from the mmap host pointer into the VRAM arena
    // on the secondary stream (HostToDevice), then record a per-expert event so
    // the compute stream can wait on THIS transfer alone.
    cudaError_t e = cudaMemcpyAsync(t.dst, t.src, t.nbytes,
                                    cudaMemcpyHostToDevice, *(cudaStream_t*)stream_);
    if (e != cudaSuccess) fprintf(stderr, "cudaMemcpyAsync failed\n");
    cudaEvent_t ev;
    if (cudaEventCreate(&ev) == cudaSuccess) {
        cudaEventRecord(ev, *(cudaStream_t*)stream_);
        t.event = (void*)ev;
    }
#else
    (void)idx;
#endif
}

void AsyncPrefetcher::cuda_wait(long idx) {
#ifdef DEE_CUDA
    if (idx >= 0 && idx < (long)inflight_.size() && inflight_[idx].event) {
        // Block ONLY this expert's transfer (sync fallback): the compute stream
        // waits on the specific event, so independent transfers never stall it.
        cudaEventSynchronize((cudaEvent_t)inflight_[idx].event);
    }
    if (idx >= 0 && idx < (long)inflight_.size()) inflight_[idx].done = true;
#else
    (void)idx;
#endif
}

} // namespace dee
