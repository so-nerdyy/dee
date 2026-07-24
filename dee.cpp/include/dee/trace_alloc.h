// dee/trace_alloc.h
//
// Milestone 3 v5 teardown-forensics instrumentation.
//
// Purpose: capture EVERY CUDA alloc/free + every insertion into the three
// ownership maps (pinned_staging_bf16_ / staging_int8_::pinned /
// registered_mmap_views_bf16_) so the next Kaggle run can report precisely
// which pointer causes the "double free or corruption (!prev)" abort.
//
// Hard rules (do NOT change):
//   1. Lifetime, ownership, and destruction order are unchanged.
//   2. Each macro emits the SAME error code as the original cuda/cuBLAS call.
//   3. In DEE_CUDA=OFF builds, the macros expand to nothing (the wrapped sites
//      only exist inside #ifdef DEE_CUDA blocks anyway).
//
// Output: every alloc registers AllocRec (id, kind, ptr, size, owner,
// file, line, thread, allocator) into a global unordered_map. Every free
// looks up the entry, transitions alive->dead, prints the id. On mismatch
// (pointer never allocated / already freed / wrong allocator) the process is
// aborted via std::abort so the post-mortem stderr captures the FIRST site.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dee::trace_alloc {

enum class Kind : uint8_t {
    Device       = 1,  // cudaMalloc
    Host         = 2,  // cudaMallocHost OR cudaHostAlloc
    Event        = 3,  // cudaEventCreate / cudaEventCreateWithFlags
    Stream       = 4,  // cudaStreamCreate / cudaStreamCreateWithFlags
    CublasHandle = 5   // cublasCreate
};

struct AllocRec {
    uint64_t    id      = 0;        // monotonic, used as the canonical reference
    Kind        kind    = Kind::Device;
    void*       ptr     = nullptr;
    size_t      size    = 0;        // bytes (Device/Host); descriptor size for Event/Stream/CublasHandle
    std::string owner;              // human label (e.g., "d_h_in_", "staging_slot[i].ptr")
    std::string allocator;          // "cudaMalloc" / "cudaMallocHost" / "cudaEventCreate" / ...
    std::string file;               // __FILE__ at the call site
    int         line    = 0;        // __LINE__ at the call site
    long        tid     = 0;        // std::hash<std::thread::id>() at registration
    bool        alive   = true;     // false after the matching free is recorded
};

struct InsertRec {
    std::string map_name;           // "pinned_staging_bf16_" / "staging_int8_" / "registered_mmap_views_bf16_"
    uint64_t    key      = 0;
    void*       ptr      = nullptr;
    std::string origin;             // "cudaHostAlloc" / "cudaMallocHost" / "std::vector::data" / ...
    std::string file;
    int         line     = 0;
};

// Global accessors. Implemented in trace_alloc.cpp.
void enable();
bool is_enabled();

// Alloc-time registration. Always succeeds; on success the entry is added to
// the live table. Returns the assigned id (>=1).
uint64_t record_alloc(Kind k, void* ptr, size_t sz, const char* owner,
                       const char* file, int line, const char* allocator);

// Free-time validation: returns true iff the pointer was live and recorded by
// the SAME allocator tag (or by ANY host-allocator). Returns false (which
// triggers std::abort in the macro) if:
//   - pointer was never allocated (UNALLOC)
//   - pointer already freed (DOUBLE-FREE)
//   - allocator mismatch (MISMATCH)
//
// The "owner" argument is logged only; it is not part of the validation.
bool record_free(void* p, const char* owner, const char* file, int line);

// Map insertion log. Append-only.
void note_insertion(const char* map_name, uint64_t key, void* ptr,
                     const char* origin, const char* file, int line);

// Snapshot helpers for the post-mortem analyzer.
void dump_to_stderr_locked();
size_t live_count();
size_t dead_count();
size_t unalloc_abort_count();
size_t double_free_abort_count();
size_t mismatch_abort_count();

}  // namespace dee::trace_alloc

// ---------------------------------------------------------------------------
// Macros: each one wraps a single CUDA / cuBLAS call. Return type matches the
// original API exactly so existing call sites need no further changes. The
// only place a std::abort fires is when the validation FAILS — that's the
// post-mortem signal.
// ---------------------------------------------------------------------------

#ifdef DEE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

// -------- alloc wrappers ----------------------------------------------------

#define DEE_TA_MALLOC(ptr_out, sz_expr, owner_str)                              \
    [&]() -> cudaError_t {                                                       \
        cudaError_t _ta_e = cudaMalloc((ptr_out), (sz_expr));                    \
        if (_ta_e == cudaSuccess) {                                              \
            dee::trace_alloc::record_alloc(                                      \
                dee::trace_alloc::Kind::Device, *(ptr_out),                      \
                (size_t)(sz_expr), (owner_str), __FILE__, __LINE__,              \
                "cudaMalloc");                                                    \
        } else {                                                                  \
            std::fprintf(stderr,                                                  \
                "[ta_alloc_fail=cudaMalloc owner=%s file=%s line=%d "           \
                "sz=%zu err=%s]\n",                                              \
                (owner_str), __FILE__, __LINE__, (size_t)(sz_expr),              \
                cudaGetErrorString(_ta_e));                                       \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_MALLOC_HOST(ptr_out, sz_expr, owner_str)                          \
    [&]() -> cudaError_t {                                                       \
        cudaError_t _ta_e = cudaMallocHost((ptr_out), (sz_expr));                 \
        if (_ta_e == cudaSuccess) {                                              \
            dee::trace_alloc::record_alloc(                                      \
                dee::trace_alloc::Kind::Host, *(ptr_out),                        \
                (size_t)(sz_expr), (owner_str), __FILE__, __LINE__,              \
                "cudaMallocHost");                                               \
        } else {                                                                  \
            std::fprintf(stderr,                                                  \
                "[ta_alloc_fail=cudaMallocHost owner=%s file=%s line=%d "       \
                "sz=%zu err=%s]\n",                                              \
                (owner_str), __FILE__, __LINE__, (size_t)(sz_expr),              \
                cudaGetErrorString(_ta_e));                                       \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_EVENT_CREATE(event_out, owner_str)                                \
    [&]() -> cudaError_t {                                                       \
        cudaError_t _ta_e = cudaEventCreate((event_out));                        \
        if (_ta_e == cudaSuccess) {                                              \
            dee::trace_alloc::record_alloc(                                      \
                dee::trace_alloc::Kind::Event, *(event_out),                     \
                0, (owner_str), __FILE__, __LINE__, "cudaEventCreate");         \
        } else {                                                                  \
            std::fprintf(stderr,                                                  \
                "[ta_alloc_fail=cudaEventCreate owner=%s file=%s line=%d "      \
                "err=%s]\n",                                                     \
                (owner_str), __FILE__, __LINE__, cudaGetErrorString(_ta_e));      \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_EVENT_CREATE_FLAGS(event_out, flags_expr, owner_str)              \
    [&]() -> cudaError_t {                                                       \
        cudaError_t _ta_e = cudaEventCreateWithFlags((event_out),               \
                                                       (flags_expr));            \
        if (_ta_e == cudaSuccess) {                                              \
            dee::trace_alloc::record_alloc(                                      \
                dee::trace_alloc::Kind::Event, *(event_out), 0,                  \
                (owner_str), __FILE__, __LINE__,                                 \
                "cudaEventCreateWithFlags");                                     \
        } else {                                                                  \
            std::fprintf(stderr,                                                  \
                "[ta_alloc_fail=cudaEventCreateWithFlags owner=%s file=%s "      \
                "line=%d err=%s]\n",                                             \
                (owner_str), __FILE__, __LINE__, cudaGetErrorString(_ta_e));      \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_STREAM_CREATE(stream_out, owner_str)                              \
    [&]() -> cudaError_t {                                                       \
        cudaError_t _ta_e = cudaStreamCreate((stream_out));                      \
        if (_ta_e == cudaSuccess) {                                              \
            dee::trace_alloc::record_alloc(                                      \
                dee::trace_alloc::Kind::Stream, *(stream_out), 0,                \
                (owner_str), __FILE__, __LINE__, "cudaStreamCreate");           \
        } else {                                                                  \
            std::fprintf(stderr,                                                  \
                "[ta_alloc_fail=cudaStreamCreate owner=%s file=%s line=%d "      \
                "err=%s]\n",                                                     \
                (owner_str), __FILE__, __LINE__, cudaGetErrorString(_ta_e));      \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_STREAM_CREATE_FLAGS(stream_out, flags_expr, owner_str)            \
    [&]() -> cudaError_t {                                                       \
        cudaError_t _ta_e = cudaStreamCreateWithFlags((stream_out),             \
                                                        (flags_expr));           \
        if (_ta_e == cudaSuccess) {                                              \
            dee::trace_alloc::record_alloc(                                      \
                dee::trace_alloc::Kind::Stream, *(stream_out), 0,                \
                (owner_str), __FILE__, __LINE__,                                 \
                "cudaStreamCreateWithFlags");                                    \
        } else {                                                                  \
            std::fprintf(stderr,                                                  \
                "[ta_alloc_fail=cudaStreamCreateWithFlags owner=%s file=%s "     \
                "line=%d err=%s]\n",                                             \
                (owner_str), __FILE__, __LINE__, cudaGetErrorString(_ta_e));      \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_CUBLAS_CREATE(handle_out, owner_str)                              \
    [&]() -> cublasStatus_t {                                                    \
        cublasStatus_t _ta_e = cublasCreate((handle_out));                       \
        if (_ta_e == CUBLAS_STATUS_SUCCESS) {                                    \
            dee::trace_alloc::record_alloc(                                      \
                dee::trace_alloc::Kind::CublasHandle,                            \
                *(handle_out), 0, (owner_str), __FILE__, __LINE__,              \
                "cublasCreate");                                                 \
        } else {                                                                  \
            std::fprintf(stderr,                                                  \
                "[ta_alloc_fail=cublasCreate owner=%s file=%s line=%d "          \
                "status=%d]\n",                                                  \
                (owner_str), __FILE__, __LINE__, (int)_ta_e);                    \
        }                                                                         \
        return _ta_e;                                                             \
    }()

// -------- free wrappers ------------------------------------------------------

#define DEE_TA_FREE(p_expr, owner_str)                                           \
    [&]() -> cudaError_t {                                                       \
        if (!dee::trace_alloc::record_free((p_expr), (owner_str),                \
                                             __FILE__, __LINE__)) {              \
            std::abort();                                                         \
        }                                                                         \
        cudaError_t _ta_e = cudaFree((p_expr));                                   \
        if (_ta_e != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                  \
                "[ta_free_fail=cudaFree owner=%s file=%s line=%d "               \
                "err=%s]\n",                                                     \
                (owner_str), __FILE__, __LINE__, cudaGetErrorString(_ta_e));      \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_FREE_HOST(p_expr, owner_str)                                      \
    [&]() -> cudaError_t {                                                       \
        if (!dee::trace_alloc::record_free((p_expr), (owner_str),                \
                                             __FILE__, __LINE__)) {              \
            std::abort();                                                         \
        }                                                                         \
        cudaError_t _ta_e = cudaFreeHost((p_expr));                               \
        if (_ta_e != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                  \
                "[ta_free_fail=cudaFreeHost owner=%s file=%s line=%d "          \
                "err=%s]\n",                                                     \
                (owner_str), __FILE__, __LINE__, cudaGetErrorString(_ta_e));      \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_EVENT_DESTROY(event_expr, owner_str)                              \
    [&]() -> cudaError_t {                                                       \
        if (!dee::trace_alloc::record_free((void*)(event_expr),                  \
                                             (owner_str), __FILE__, __LINE__)) { \
            std::abort();                                                         \
        }                                                                         \
        cudaError_t _ta_e = cudaEventDestroy((event_expr));                      \
        if (_ta_e != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                  \
                "[ta_free_fail=cudaEventDestroy owner=%s file=%s line=%d "      \
                "err=%s]\n",                                                     \
                (owner_str), __FILE__, __LINE__, cudaGetErrorString(_ta_e));      \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_STREAM_DESTROY(stream_expr, owner_str)                            \
    [&]() -> cudaError_t {                                                       \
        if (!dee::trace_alloc::record_free((void*)(stream_expr),                \
                                             (owner_str), __FILE__, __LINE__)) { \
            std::abort();                                                         \
        }                                                                         \
        cudaError_t _ta_e = cudaStreamDestroy((stream_expr));                    \
        if (_ta_e != cudaSuccess) {                                              \
            std::fprintf(stderr,                                                  \
                "[ta_free_fail=cudaStreamDestroy owner=%s file=%s line=%d "     \
                "err=%s]\n",                                                     \
                (owner_str), __FILE__, __LINE__, cudaGetErrorString(_ta_e));      \
        }                                                                         \
        return _ta_e;                                                             \
    }()

#define DEE_TA_CUBLAS_DESTROY(handle_expr, owner_str)                            \
    [&]() -> cublasStatus_t {                                                    \
        if (!dee::trace_alloc::record_free((void*)(handle_expr),                \
                                             (owner_str), __FILE__, __LINE__)) { \
            std::abort();                                                         \
        }                                                                         \
        cublasStatus_t _ta_e = cublasDestroy((handle_expr));                    \
        if (_ta_e != CUBLAS_STATUS_SUCCESS) {                                    \
            std::fprintf(stderr,                                                  \
                "[ta_free_fail=cublasDestroy owner=%s file=%s line=%d "          \
                "status=%d]\n",                                                  \
                (owner_str), __FILE__, __LINE__, (int)_ta_e);                    \
        }                                                                         \
        return _ta_e;                                                             \
    }()

// -------- map insertion logger ---------------------------------------------

// Use BEFORE every `map[key] = ptr;` (or `.pinned = ptr;`) so the post-mortem
// run can correlate which insert line produced which pointer. The `origin`
// tag is required to be honest; the analyst will compare against the
// AllocRec log and reject any insert whose origin disagrees.
#define DEE_TA_INSERT(map_name_str, key_expr, ptr_expr, origin_str)              \
    dee::trace_alloc::note_insertion(                                            \
        (map_name_str), (uint64_t)(key_expr), (void*)(ptr_expr),                \
        (origin_str), __FILE__, __LINE__)

#else  // DEE_CUDA OFF — macros expand to no-ops; lifecycle preserved.

#define DEE_TA_MALLOC(ptr_out, sz_expr, owner_str)               ((void)0)
#define DEE_TA_MALLOC_HOST(ptr_out, sz_expr, owner_str)           ((void)0)
#define DEE_TA_EVENT_CREATE(event_out, owner_str)                 ((void)0)
#define DEE_TA_EVENT_CREATE_FLAGS(event_out, flags_expr, owner_str) ((void)0)
#define DEE_TA_STREAM_CREATE(stream_out, owner_str)               ((void)0)
#define DEE_TA_STREAM_CREATE_FLAGS(stream_out, flags_expr, owner_str) ((void)0)
#define DEE_TA_CUBLAS_CREATE(handle_out, owner_str)               ((void)0)
#define DEE_TA_FREE(p_expr, owner_str)                            ((void)0)
#define DEE_TA_FREE_HOST(p_expr, owner_str)                       ((void)0)
#define DEE_TA_EVENT_DESTROY(event_expr, owner_str)               ((void)0)
#define DEE_TA_STREAM_DESTROY(stream_expr, owner_str)             ((void)0)
#define DEE_TA_CUBLAS_DESTROY(handle_expr, owner_str)             ((void)0)
#define DEE_TA_INSERT(map_name_str, key_expr, ptr_expr, origin_str) ((void)0)

#endif  // DEE_CUDA
