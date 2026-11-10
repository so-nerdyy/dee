#pragma once

#ifdef DEE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>

namespace dee {

inline bool cuda_check(cudaError_t status, const char* operation, const char* file, int line) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[cuda] %s failed at %s:%d: %s (%s)\n", operation, file, line,
                 cudaGetErrorName(status), cudaGetErrorString(status));
    return false;
}

inline bool cuda_check_launch(const char* operation, const char* file, int line) {
    return cuda_check(cudaGetLastError(), operation, file, line);
}

inline const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline bool cublas_check(cublasStatus_t status, const char* operation, const char* file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "[cublas] %s failed at %s:%d: %s (%d)\n", operation, file, line,
                 cublas_status_name(status), static_cast<int>(status));
    return false;
}

}  // namespace dee

#define DEE_CUDA_CHECK(operation) ::dee::cuda_check((operation), #operation, __FILE__, __LINE__)
#define DEE_CUDA_CHECK_NAMED(operation, name) ::dee::cuda_check((operation), (name), __FILE__, __LINE__)
#define DEE_CUDA_CHECK_LAUNCH(name) ::dee::cuda_check_launch((name), __FILE__, __LINE__)
#define DEE_CUBLAS_CHECK(operation) ::dee::cublas_check((operation), #operation, __FILE__, __LINE__)
#define DEE_CUBLAS_CHECK_NAMED(operation, name) ::dee::cublas_check((operation), (name), __FILE__, __LINE__)
#endif
