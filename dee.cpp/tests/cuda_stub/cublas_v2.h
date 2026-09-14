// tests/cuda_stub/cublas_v2.h — minimal cublas_v2.h replacement for the
// mock-CUDA test target. NOT for linking or running real GPU work.
#pragma once

typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_NOT_INITIALIZED 1
#define CUBLAS_STATUS_ALLOC_FAILED 3
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_STATUS_ARCH_MISMATCH 8
#define CUBLAS_STATUS_MAPPING_ERROR 9
#define CUBLAS_STATUS_EXECUTION_FAILED 13
#define CUBLAS_STATUS_INTERNAL_ERROR 14
#define CUBLAS_STATUS_NOT_SUPPORTED 15

typedef void* cublasHandle_t;

extern "C" {
cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
}
