// tests/cuda_stub/cuda_runtime.h
//
// Minimal cuda_runtime.h replacement for compiling the real DEE_CUDA code
// paths on machines without a CUDA toolkit. Paired with cuda_stub.cpp, which
// implements every entry point as a host-side mock with per-call fault
// injection (see cuda_stub_control.h). The real headers and toolchain remain
// authoritative for shipped builds; this exists so the CPU ctest suite can
// execute AsyncPrefetcher's CUDA failure paths (event lifecycle, stream
// drains, error propagation) instead of asserting them only by contract.
#pragma once
#include <cstddef>

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorNotReady 34
#define cudaErrorUnknown 999
#define cudaErrorMemoryAllocation 2
#define cudaErrorInvalidValue 11

typedef void* cudaStream_t;
typedef void* cudaEvent_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

#define cudaStreamNonBlocking 0x1
#define cudaEventDisableTiming 0x2
#define cudaHostAllocDefault 0x0
#define cudaHostRegisterPortable 0x1

extern "C" {
cudaError_t cudaMalloc(void** ptr, size_t size);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                       enum cudaMemcpyKind kind);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);
cudaError_t cudaGetLastError();
}
