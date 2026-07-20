// dee.cpp - Step 5: real CUDA backend (only compiled with -DDEE_CUDA).
//
// Maps the CudaDevice interface onto the CUDA runtime API. cudaStream_t and
// cudaEvent_t are pointer types, so they round-trip through void* safely.

#include "dee/cuda_device.h"

#ifdef DEE_CUDA

#include <cuda_runtime.h>
#include <cstring>

namespace dee {

class CudaCudaDevice : public CudaDevice {
public:
    DevicePtr device_malloc(size_t bytes) override {
        void* p = nullptr;
        return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
    }
    void device_free(DevicePtr ptr) override {
        if (ptr) cudaFree(ptr);
    }

    void* stream_create() override {
        cudaStream_t s = nullptr;
        cudaStreamCreate(&s);
        return s;
    }
    void destroy_stream(void* s) override { cudaStreamDestroy(static_cast<cudaStream_t>(s)); }

    void* event_create() override {
        cudaEvent_t e = nullptr;
        cudaEventCreate(&e);
        return e;
    }
    void destroy_event(void* e) override { cudaEventDestroy(static_cast<cudaEvent_t>(e)); }
    void event_record(void* e, void* s) override {
        cudaEventRecord(static_cast<cudaEvent_t>(e), static_cast<cudaStream_t>(s));
    }
    void event_synchronize(void* e) override {
        cudaEventSynchronize(static_cast<cudaEvent_t>(e));
    }

    void memcpy_h2d_async(DevicePtr dst, const void* src, size_t bytes, void* s) override {
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(s));
    }
    void memcpy_d2h_sync(void* dst, DevicePtr src, size_t bytes) override {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    }
    void stream_synchronize(void* s) override {
        cudaStreamSynchronize(static_cast<cudaStream_t>(s));
    }

    const char* backend_name() const override { return "cuda"; }
};

std::unique_ptr<CudaDevice> make_cuda_device() { return std::make_unique<CudaCudaDevice>(); }

}  // namespace dee

#endif  // DEE_CUDA
