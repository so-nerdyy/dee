// Minimal stand-in for <cuda_runtime.h> used ONLY to compile + run the real
// DEE_CUDA backend path on a box without a CUDA toolkit. It implements the
// symbols the backend calls with trivial (synchronous) semantics so the code
// path is exercised end-to-end. This is NOT a CUDA implementation.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>

typedef enum { cudaSuccess = 0, cudaErrorUnknown = 1 } cudaError_t;
typedef enum { cudaMemcpyHostToDevice = 0, cudaMemcpyDeviceToHost = 1,
               cudaMemcpyDeviceToDevice = 2 } cudaMemcpyKind;

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

inline cudaError_t cudaMalloc(void** ptr, size_t bytes) {
    *ptr = std::malloc(bytes ? bytes : 1);
    return *ptr ? cudaSuccess : cudaErrorUnknown;
}
inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}
inline cudaError_t cudaStreamCreate(cudaStream_t* s) {
    *s = reinterpret_cast<cudaStream_t>(new int(1));
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    delete reinterpret_cast<int*>(s);
    return cudaSuccess;
}
inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    *e = reinterpret_cast<cudaEvent_t>(new int(1));
    return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
    delete reinterpret_cast<int*>(e);
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes,
                                   cudaMemcpyKind, cudaStream_t) {
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind) {
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
