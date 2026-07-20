#pragma once

#ifdef DEE_CUDA
#include <cuda_runtime.h>

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

}  // namespace dee

#define DEE_CUDA_CHECK(operation) ::dee::cuda_check((operation), #operation, __FILE__, __LINE__)
#define DEE_CUDA_CHECK_NAMED(operation, name) ::dee::cuda_check((operation), (name), __FILE__, __LINE__)
#define DEE_CUDA_CHECK_LAUNCH(name) ::dee::cuda_check_launch((name), __FILE__, __LINE__)
#endif
