// dee.cpp - Step 5: CPU mock CUDA backend (no toolkit required).
//
// Models real CUDA async semantics on the CPU so the cache + prefetch logic can
// be compiled and tested anywhere:
//   - device memory is a heap buffer.
//   - memcpy_h2d_async enqueues a copy (with a monotonic seq) on a stream.
//   - event_record marks a watermark; event_synchronize drains all copies with
//     seq < watermark (FIFO), so per-expert sync only completes that expert's
//     copy and leaves later copies pending.

#include "dee/cuda_device.h"

#include <cstring>
#include <new>
#include <unordered_map>
#include <vector>

namespace dee {
namespace {

struct MockCopy {
    void* dst;
    const void* src;
    size_t bytes;
    uint64_t seq;
};

struct MockStream {
    std::vector<MockCopy> pending;  // FIFO; front == oldest not-yet-run
    uint64_t next_seq = 0;
};

struct MockEvent {
    MockStream* stream = nullptr;
    uint64_t watermark = 0;  // copies with seq < watermark must be complete
};

class MockCudaDevice : public CudaDevice {
public:
    MockCudaDevice() = default;
    ~MockCudaDevice() override {
        for (auto& kv : allocs_) delete[] static_cast<std::byte*>(kv.first);
        allocs_.clear();
    }

    DevicePtr device_malloc(size_t bytes) override {
        auto* p = new (std::nothrow) std::byte[bytes ? bytes : 1];
        if (!p) return nullptr;
        allocs_[p] = bytes;
        return p;
    }
    void device_free(DevicePtr ptr) override {
        auto it = allocs_.find(ptr);
        if (it != allocs_.end()) {
            delete[] static_cast<std::byte*>(it->first);
            allocs_.erase(it);
        }
    }

    void* stream_create() override { return new MockStream(); }
    void destroy_stream(void* s) override { delete static_cast<MockStream*>(s); }

    void* event_create() override { return new MockEvent(); }
    void destroy_event(void* e) override { delete static_cast<MockEvent*>(e); }
    void event_record(void* e, void* s) override {
        auto* ev = static_cast<MockEvent*>(e);
        auto* st = static_cast<MockStream*>(s);
        ev->stream = st;
        ev->watermark = st->next_seq;
    }
    void event_synchronize(void* e) override {
        auto* ev = static_cast<MockEvent*>(e);
        if (!ev->stream) return;
        auto& q = ev->stream->pending;
        while (!q.empty() && q.front().seq < ev->watermark) {
            const MockCopy& c = q.front();
            std::memcpy(c.dst, c.src, c.bytes);
            q.erase(q.begin());
        }
    }

    void memcpy_h2d_async(DevicePtr dst, const void* src, size_t bytes, void* s) override {
        auto* st = static_cast<MockStream*>(s);
        st->pending.push_back({dst, src, bytes, st->next_seq++});
    }
    void memcpy_d2h_sync(void* dst, DevicePtr src, size_t bytes) override {
        std::memcpy(dst, src, bytes);
    }
    void stream_synchronize(void* s) override {
        auto* st = static_cast<MockStream*>(s);
        for (const MockCopy& c : st->pending) std::memcpy(c.dst, c.src, c.bytes);
        st->pending.clear();
    }
    const char* backend_name() const override { return "mock"; }

private:
    std::unordered_map<void*, size_t> allocs_;  // device ptr -> size
};

}  // namespace

#ifndef DEE_CUDA
std::unique_ptr<CudaDevice> make_cuda_device() { return std::make_unique<MockCudaDevice>(); }
#endif

}  // namespace dee
