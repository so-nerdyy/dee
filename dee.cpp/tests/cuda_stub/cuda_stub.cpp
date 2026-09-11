// tests/cuda_stub/cuda_stub.cpp
//
// Host-side mock of the CUDA runtime + the dee::*_cuda conversion kernels
// (whose real definitions live in cuda_convert.cu and require nvcc). Built
// only into the stub-CUDA test target; never linked into dee_core.
//
// Model: one implicit device, streams/events as tiny heap records so the
// real trace_alloc sentinel sees honest create/destroy pairs. Successful
// enqueues bump g.pending; cudaStreamSynchronize drains it. Injected
// failures behave like the defensive-drain contract's worst case: the call
// reports an error but any work it already launched stays in flight.
#include "cuda_stub_control.h"

#include <cublas_v2.h>

#include "dee/cuda_convert.h"

#include <cstdlib>
#include <cstring>

namespace dee_stub {
Control g;

bool should_fail(const char* name) {
    g.calls[name] += 1;
    g.log.push_back(name);
    const auto it = g.fail_at.find(name);
    return it != g.fail_at.end() && it->second == g.calls[name];
}
}  // namespace dee_stub

namespace {
struct StubEvent { bool recorded = false; };
struct StubStream { int dummy = 0; };

StubEvent* as_event(cudaEvent_t e) { return static_cast<StubEvent*>(e); }
}  // namespace

extern "C" {

cudaError_t cudaMalloc(void** ptr, size_t size) {
    if (dee_stub::should_fail("cudaMalloc")) return cudaErrorMemoryAllocation;
    *ptr = std::malloc(size ? size : 1);
    ++dee_stub::g.live_device_allocs;
    return cudaSuccess;
}
cudaError_t cudaMallocHost(void** ptr, size_t size) {
    if (dee_stub::should_fail("cudaMallocHost")) return cudaErrorMemoryAllocation;
    *ptr = std::malloc(size ? size : 1);
    ++dee_stub::g.live_host_allocs;
    return cudaSuccess;
}
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int) {
    if (dee_stub::should_fail("cudaHostAlloc")) return cudaErrorMemoryAllocation;
    *ptr = std::malloc(size ? size : 1);
    ++dee_stub::g.live_host_allocs;
    return cudaSuccess;
}
cudaError_t cudaFree(void* ptr) {
    if (dee_stub::should_fail("cudaFree")) return cudaErrorUnknown;
    std::free(ptr);
    --dee_stub::g.live_device_allocs;
    return cudaSuccess;
}
cudaError_t cudaFreeHost(void* ptr) {
    if (dee_stub::should_fail("cudaFreeHost")) return cudaErrorUnknown;
    std::free(ptr);
    --dee_stub::g.live_host_allocs;
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return cudaEventCreateWithFlags(event, 0);
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int) {
    if (dee_stub::should_fail("cudaEventCreateWithFlags")) {
        *event = nullptr;
        return cudaErrorUnknown;
    }
    *event = new StubEvent{};
    ++dee_stub::g.live_events;
    return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (dee_stub::should_fail("cudaEventDestroy")) return cudaErrorUnknown;
    delete as_event(event);
    --dee_stub::g.live_events;
    return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t) {
    // An injected failure here is the interesting case for the submit paths:
    // preceding work (if any) is still enqueued on the stream.
    if (dee_stub::should_fail("cudaEventRecord")) return cudaErrorUnknown;
    as_event(event)->recorded = true;
    return cudaSuccess;
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (dee_stub::should_fail("cudaEventSynchronize")) return cudaErrorUnknown;
    (void)as_event(event);
    return cudaSuccess;
}
cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (dee_stub::should_fail("cudaEventQuery")) return cudaErrorUnknown;
    return as_event(event)->recorded ? cudaSuccess : cudaErrorNotReady;
}
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    if (dee_stub::should_fail("cudaEventElapsedTime")) return cudaErrorUnknown;
    *ms = 0.0f;
    return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    return cudaStreamCreateWithFlags(stream, 0);
}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int) {
    if (dee_stub::should_fail("cudaStreamCreateWithFlags")) {
        *stream = nullptr;
        return cudaErrorUnknown;
    }
    *stream = new StubStream{};
    ++dee_stub::g.live_streams;
    return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (dee_stub::should_fail("cudaStreamDestroy")) return cudaErrorUnknown;
    delete static_cast<StubStream*>(stream);
    --dee_stub::g.live_streams;
    return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    if (dee_stub::should_fail("cudaStreamSynchronize")) return cudaErrorUnknown;
    ++dee_stub::g.stream_syncs;
    if (dee_stub::g.pending > 0) ++dee_stub::g.syncs_with_pending;
    dee_stub::g.pending = 0;
    return cudaSuccess;
}
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int) {
    if (dee_stub::should_fail("cudaStreamWaitEvent")) return cudaErrorUnknown;
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            enum cudaMemcpyKind, cudaStream_t) {
    const bool fail = dee_stub::should_fail("cudaMemcpyAsync");
    // Enqueue first so an injected failure still leaves work in flight.
    ++dee_stub::g.pending;
    if (fail) return cudaErrorUnknown;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                       enum cudaMemcpyKind) {
    if (dee_stub::should_fail("cudaMemcpy")) return cudaErrorUnknown;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) {
    if (dee_stub::should_fail("cudaGetDevice")) return cudaErrorUnknown;
    *device = 0;
    return cudaSuccess;
}
cudaError_t cudaSetDevice(int) {
    if (dee_stub::should_fail("cudaSetDevice")) return cudaErrorUnknown;
    return cudaSuccess;
}
cudaError_t cudaHostRegister(void*, size_t, unsigned int) {
    if (dee_stub::should_fail("cudaHostRegister")) return cudaErrorUnknown;
    ++dee_stub::g.host_registered;
    return cudaSuccess;
}
cudaError_t cudaHostUnregister(void*) {
    if (dee_stub::should_fail("cudaHostUnregister")) return cudaErrorUnknown;
    --dee_stub::g.host_registered;
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "cudaSuccess";
        case cudaErrorNotReady: return "cudaErrorNotReady";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        default: return "cudaErrorUnknown";
    }
}
const char* cudaGetErrorName(cudaError_t error) { return cudaGetErrorString(error); }
cudaError_t cudaGetLastError() { return cudaSuccess; }

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    *handle = reinterpret_cast<cublasHandle_t>(new int(0));
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    delete static_cast<int*>(handle);
    return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"

// ---- dee:: conversion kernels (real versions live in cuda_convert.cu) ------
// Each models one enqueued stream operation and honors fail_at injection.

namespace dee {
namespace {
bool kernel_call(const char* name) {
    ++dee_stub::g.pending;
    return !dee_stub::should_fail(name);
}
}  // namespace

bool bf16_to_f32_cuda(const uint16_t*, float*, size_t, cudaStream_t, StageProfiler*) {
    return kernel_call("bf16_to_f32_cuda");
}
bool bf16_to_f16_cuda(const uint16_t*, void*, size_t, cudaStream_t, StageProfiler*) {
    return kernel_call("bf16_to_f16_cuda");
}
bool f32_to_f16_cuda(const float*, void*, size_t, cudaStream_t, StageProfiler*) {
    return kernel_call("f32_to_f16_cuda");
}
bool int8_to_f16_cuda(const int8_t*, void*, size_t, size_t, const float[3],
                      cudaStream_t, StageProfiler*) {
    return kernel_call("int8_to_f16_cuda");
}
bool int4_to_f16_cuda(const uint8_t*, void*, size_t, size_t, const float[3],
                      cudaStream_t, StageProfiler*) {
    return kernel_call("int4_to_f16_cuda");
}
bool fp4_e2m1_to_f16_cuda(const uint8_t*, const uint8_t*, void*, size_t, size_t,
                          cudaStream_t, StageProfiler*) {
    return kernel_call("fp4_e2m1_to_f16_cuda");
}
void oracle_relu_cuda(float*, int, cudaStream_t) {
    ++dee_stub::g.pending;
}

}  // namespace dee
