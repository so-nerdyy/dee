#pragma once
// dee.cpp - Step 5: GPU memory & async prefetcher.
//
// CudaDevice is the abstraction over GPU memory + streams + events. The default
// build uses a CPU-backed MockCudaDevice (no CUDA toolkit required, so the LRU
// cache and prefetch-queue logic can be exercised on any box). When the project
// is built with -DDEE_CUDA (and a CUDA toolkit is available), the real backend
// in cuda_backend_cuda.cpp is linked instead and the same interfaces map onto
// cudaMalloc / cudaMemcpyAsync / cudaStream_t / cudaEvent_t.

#include <cstddef>
#include <memory>
#include <string>

namespace dee {

// Opaque device (GPU) memory pointer. In the mock backend this is a plain host
// pointer into a heap buffer; in the CUDA backend it is a cudaMalloc address.
using DevicePtr = void*;

// Abstract device backend. All async operations model real CUDA semantics:
//   - memcpy_h2d_async enqueues a copy on a stream (does NOT block).
//   - event_record / event_synchronize let a single prefetch be waited on
//     without blocking the entire stream (the correct idiom for per-expert sync).
//   - stream_synchronize blocks until ALL work on a stream is complete.
class CudaDevice {
public:
    virtual ~CudaDevice() = default;

    // Allocate `bytes` of device memory (one big arena in production). Returns
    // nullptr on failure.
    virtual DevicePtr device_malloc(size_t bytes) = 0;
    virtual void device_free(DevicePtr ptr) = 0;

    // Streams.
    virtual void* stream_create() = 0;
    virtual void destroy_stream(void* stream) = 0;

    // Events (used to synchronize a single prefetch without stalling the stream).
    virtual void* event_create() = 0;
    virtual void destroy_event(void* event) = 0;
    virtual void event_record(void* event, void* stream) = 0;
    virtual void event_synchronize(void* event) = 0;

    // Asynchronous host -> device copy on `stream`.
    virtual void memcpy_h2d_async(DevicePtr dst, const void* src, size_t bytes, void* stream) = 0;

    // Synchronous device -> host copy. Intended for tests/debug verification of
    // transfers; production readback of weights is not on the hot path.
    virtual void memcpy_d2h_sync(void* dst, DevicePtr src, size_t bytes) = 0;

    // Block until all work enqueued on `stream` has completed.
    virtual void stream_synchronize(void* stream) = 0;

    virtual const char* backend_name() const = 0;
};

// Factory: returns the real CUDA device when compiled with DEE_CUDA, otherwise
// the CPU mock. Exactly one definition is emitted (see the per-backend TUs).
std::unique_ptr<CudaDevice> make_cuda_device();

}  // namespace dee
