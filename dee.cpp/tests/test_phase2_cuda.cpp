// Optional real CUDA synthetic test. CTest reports SKIP (77) without a GPU.
// No model files, inference benchmark, tolerances, or route prediction.
#include "dee/expert_tiers.h"
#include <cuda_runtime.h>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {
using namespace dee;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Policy : HostPlacementPolicy {
    HostResidency residency(const TierExpertKey&) const override { return HostResidency::Dynamic; }
    std::optional<size_t> victim(const TierExpertKey&, const std::vector<HostVictim>& candidates) const override {
        return candidates.empty() ? std::nullopt : std::optional<size_t>(candidates[0].slot);
    }
};
struct Store : ColdExpertStore {
    ColdReadResult read(const TierExpertKey& key, uint8_t* dst, size_t n) override {
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<uint8_t>(i + key.expert * 37);
        return {true, n};
    }
};
struct Gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
    static void CUDART_CB block(void* context) {
        auto& self = *static_cast<Gate*>(context);
        std::unique_lock<std::mutex> lock(self.mutex);
        self.entered = true; self.cv.notify_all();
        self.cv.wait(lock, [&] { return self.released; });
    }
    void release() {
        { std::lock_guard<std::mutex> lock(mutex); released = true; } cv.notify_all();
    }
};
void run(bool pin) {
    Gate gate;
    IdentityCodec codec; Store store;
    auto backend = host_memory_backend(true, 0);
    if (!pin) backend.pin = [](void*, size_t) { return false; };
    HostExpertTier host({4096, 4096, 0, 1, 4096, true}, backend, std::make_shared<Policy>());
    require(!pin || host.stats().pinned_bytes == 4096, "real CUDA host registration failed");
    VramCacheManager cache;
    require(cache.init(8192, {
        [](size_t n) -> void* { void* ptr = nullptr; return cudaMalloc(&ptr, n) == cudaSuccess ? ptr : nullptr; },
        [](void* ptr) { cudaFree(ptr); }, "cuda"}), "CUDA arena init failed");
    AsyncPrefetcher prefetcher(cache);
    require(prefetcher.init(true), "CUDA stream init failed");
    StorageRecord first{{"cuda-synthetic-v1", 0, 1, "packed-bytes-v1"}, 4096, 4096, "identity-v1"};
    DeviceExpertTier device(cache, prefetcher, first.key);
    auto second = first; second.key.expert = 2;
    cudaStream_t compute = nullptr;
    require(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking) == cudaSuccess, "compute stream init");
    try {
        if (pin) {
            require(cudaLaunchHostFunc(static_cast<cudaStream_t>(prefetcher.cuda_stream()), Gate::block, &gate) == cudaSuccess,
                    "enqueue deterministic DMA gate");
            std::unique_lock<std::mutex> lock(gate.mutex);
            gate.cv.wait(lock, [&] { return gate.entered; });
        }
        require(device.stage(first, store, host, codec, 0), "packed H2D stage");
        if (pin) {
            require(prefetcher.wait_on_stream(0, 1, compute), "arm compute dependency");
            require(cache.pin(0, 1), "compute consumer pin"); // Engine's existing ownership step
            require(host.stats().leased_slots == 1, "stream wait must retain source until actual DMA completion");
            require(host.acquire(second, store, codec).status == HostAcquireStatus::Capacity,
                    "incomplete DMA cannot evict host source");
            require(prefetcher.pinned_staging_bytes() == 0, "direct H2D must not allocate a gather ring");
            gate.release();
        }
        require(device.wait(first.key), "synchronous wait confirms DMA even after wait_on_stream");
        uint8_t copied[4096];
        require(cudaMemcpy(copied, cache.data(0, 1), sizeof(copied), cudaMemcpyDeviceToHost) == cudaSuccess, "copy back synthetic bytes");
        for (size_t i = 0; i < sizeof(copied); ++i)
            require(copied[i] == static_cast<uint8_t>(i + 37), "packed byte identity");
        require(host.stats().leased_slots == 0, "completed DMA releases host lease");
        require(prefetcher.stats().mmap_to_pinned_bytes == 0 && prefetcher.stats().h2d_bytes == 4096,
                "direct packed transfer metrics");
        if (pin) require(cache.unpin(0, 1), "release compute consumer pin");
        require(device.stage(second, store, host, codec, 0) && device.wait(second.key), "reuse pinned/pageable host slot");
        require(cudaStreamSynchronize(compute) == cudaSuccess, "test compute stream complete");
        cudaStreamDestroy(compute);
    } catch (...) {
        gate.release();
        prefetcher.synchronize_all();
        cudaStreamSynchronize(compute); cudaStreamDestroy(compute);
        throw;
    }
}
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: real CUDA GPU unavailable\n"; return 77;
    }
    try { run(true); run(false); }
    catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
    std::cout << "PASS: real CUDA pinned direct H2D, event lifetime, pageable fallback, exact bytes\n";
    return 0;
}
