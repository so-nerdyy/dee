// dee/trace_alloc.cpp
//
// Milestone 3 v5 teardown-forensics: thread-safe global sentinel table for
// every CUDA alloc/free + map-insertion log + abort-on-mismatch enforcement.
//
// Header dee/trace_alloc.h owns the public API + macros. This file owns the
// state. Compiles in BOTH DEE_CUDA=ON (Kaggle/CI) and DEE_CUDA=OFF (host
// tests).
//
// Lifecycle rules (do NOT change):
//   - Alloc-time: AllocRec keyed by pointer, allocator tag stored.
//   - Free-time: look up by pointer. Re-free + unalloc + allocator mismatch
//     each emit a stderr line and call std::abort, dumping the entire live
//     table to stderr first.
//   - Insert-time: append-only log via note_insertion.

#include "dee/trace_alloc.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dee::trace_alloc {

namespace {

std::mutex           g_mutex;
bool                 g_enabled = true;
std::atomic<uint64_t> g_next_id{1};
std::atomic<size_t>   g_non_selftest_allocs{0};

std::unordered_map<void*, AllocRec> g_live;
std::vector<InsertRec>              g_insertions;

size_t g_unalloc_aborts       = 0;
size_t g_double_free_aborts   = 0;
size_t g_mismatch_aborts      = 0;
size_t g_uaf_aborts           = 0;

long current_tid() {
    return static_cast<long>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

// v12: the per-allocation stderr lines are forensics, not contract.  With the
// trace always-on, every CUDA alloc/free in the whole process (torch + both
// engine threads) took a global mutex + fprintf into Kaggle's log pipe while
// a 2.5 h decode ran, which is a large share of the measured
// "everything_else" host wall.  Gate ALL informational lines behind
// DEE_TRACE_ALLOC=1 (checked once); the abort/enforcement paths stay
// unconditional.
bool verbose_logging() {
    static const bool enabled = (std::getenv("DEE_TRACE_ALLOC") != nullptr);
    return enabled;
}

const char* kind_name(Kind k) {
    switch (k) {
        case Kind::Device:       return "device";
        case Kind::Host:         return "host";
        case Kind::Event:        return "event";
        case Kind::Stream:       return "stream";
        case Kind::CublasHandle: return "cublas";
    }
    return "unknown";
}

// Internal dump; does NOT take the lock (caller must hold it).
// Noinline so it shows up as a recognizable frame in the post-mortem.
__attribute__((noinline))
void dump_internal_no_lock_() {
    std::fprintf(stderr, "=== TRACE_ALLOC POST-MORTEM DUMP ===\n");
    std::fprintf(stderr,
        "  live_count=%zu insertion_count=%zu unalloc_aborts=%zu "
        "double_free_aborts=%zu mismatch_aborts=%zu uaf_aborts=%zu\n",
        g_live.size(), g_insertions.size(),
        g_unalloc_aborts, g_double_free_aborts, g_mismatch_aborts, g_uaf_aborts);
    std::fprintf(stderr, "  -- live ptrs --\n");
    for (const auto& kv : g_live) {
        const AllocRec& r = kv.second;
        std::fprintf(stderr,
            "    id=%llu ptr=%p kind=%s size=%zu owner=%s alloc=%s "
            "alive=%s site=%s:%d\n",
            (unsigned long long)r.id, r.ptr, kind_name(r.kind),
            r.size, r.owner.c_str(), r.allocator.c_str(),
            r.alive ? "true" : "false",
            r.file.c_str(), r.line);
    }
    std::fprintf(stderr, "  -- insertions (last 64) --\n");
    const size_t first = g_insertions.size() > 64 ? g_insertions.size() - 64 : 0;
    for (size_t i = first; i < g_insertions.size(); ++i) {
        const InsertRec& r = g_insertions[i];
        std::fprintf(stderr,
            "    [%zu] map=%s key=%llu ptr=%p origin=%s site=%s:%d\n",
            i, r.map_name.c_str(), (unsigned long long)r.key, r.ptr,
            r.origin.c_str(), r.file.c_str(), r.line);
    }
    std::fprintf(stderr, "=== END POST-MORTEM ===\n");
    std::fflush(stderr);
}

}  // namespace

void enable()  { g_enabled = true; }
bool is_enabled() { return g_enabled; }

uint64_t record_alloc(Kind k, void* ptr, size_t sz, const char* owner,
                       const char* file, int line, const char* allocator) {
    if (!g_enabled) return 0;
    if (ptr == nullptr) {
        if (verbose_logging()) {
            std::fprintf(stderr,
                "[ta_alloc_skip] kind=%s owner=%s file=%s line=%d "
                "(nullptr returned)\n",
                kind_name(k), owner ? owner : "?", file ? file : "?", line);
        }
        return 0;
    }
    const uint64_t id = g_next_id.fetch_add(1, std::memory_order_relaxed);
    AllocRec rec;
    rec.id        = id;
    rec.kind      = k;
    rec.ptr       = ptr;
    rec.size      = sz;
    rec.owner     = owner ? owner : "?";
    rec.allocator = allocator ? allocator : "?";
    rec.file      = file ? file : "?";
    rec.line      = line;
    rec.tid       = current_tid();
    rec.alive     = true;
    if (rec.owner != "__dee_ta_selftest__") {
        g_non_selftest_allocs.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> g(g_mutex);
    auto it = g_live.find(ptr);
    if (it != g_live.end()) {
        if (it->second.alive) {
            ++g_mismatch_aborts;
            std::fprintf(stderr,
                "[ta_alloc_LIVE_OVERLAP_ABORT] new_id=%llu prev_id=%llu "
                "ptr=%p kind=%s prev_site=%s:%d new_site=%s:%d\n",
                (unsigned long long)id,
                (unsigned long long)it->second.id,
                ptr, kind_name(it->second.kind),
                it->second.file.c_str(), it->second.line,
                rec.file.c_str(), rec.line);
            dump_internal_no_lock_();
            std::abort();
        }
        if (verbose_logging()) {
            std::fprintf(stderr,
                "[ta_alloc_reuse] new_id=%llu prev_id=%llu ptr=%p kind=%s "
                "prev_site=%s:%d new_site=%s:%d\n",
                (unsigned long long)id,
                (unsigned long long)it->second.id,
                ptr, kind_name(it->second.kind),
                it->second.file.c_str(), it->second.line,
                rec.file.c_str(), rec.line);
        }
        it->second = rec;
        return id;
    }
    g_live.emplace(ptr, rec);
    if (verbose_logging()) {
        std::fprintf(stderr,
            "[ta_alloc] id=%llu ptr=%p kind=%s size=%zu owner=%s alloc=%s "
            "file=%s:%d tid=%ld\n",
            (unsigned long long)id, ptr, kind_name(k), sz,
            rec.owner.c_str(), rec.allocator.c_str(),
            rec.file.c_str(), rec.line, rec.tid);
    }
    return id;
}

bool record_free(void* p, Kind expected_kind, const char* owner,
                 const char* file, int line) {
    if (!g_enabled) return true;
    if (p == nullptr) {
        if (verbose_logging()) {
            std::fprintf(stderr,
                "[ta_free_skip] owner=%s file=%s:%d (nullptr)\n",
                owner ? owner : "?", file ? file : "?", line);
        }
        return true;
    }
    std::lock_guard<std::mutex> g(g_mutex);
    auto it = g_live.find(p);
    if (it == g_live.end()) {
        ++g_unalloc_aborts;
        std::fprintf(stderr,
            "[ta_free_UNALLOC_ABORT] owner=%s ptr=%p file=%s:%d "
            "(ptr was never registered as a DEE_TA_* alloc)\n",
            owner ? owner : "?", p, file ? file : "?", line);
        dump_internal_no_lock_();
        std::abort();
    }
    if (!it->second.alive) {
        ++g_double_free_aborts;
        const AllocRec& dead = it->second;
        std::fprintf(stderr,
            "[ta_free_DOUBLE_FREE_ABORT] owner=%s ptr=%p file=%s:%d "
            "alloc=%s original_id=%llu original_site=%s:%d lifetime=%s:%d\n",
            owner ? owner : "?", p, file ? file : "?", line,
            dead.allocator.c_str(),
            (unsigned long long)dead.id,
            dead.file.c_str(), dead.line,
            dead.file.c_str(), dead.line);
        dump_internal_no_lock_();
        std::abort();
    }
    if (it->second.kind != expected_kind) {
        ++g_mismatch_aborts;
        const AllocRec& rec = it->second;
        std::fprintf(stderr,
            "[ta_free_MISMATCH_ABORT] owner=%s ptr=%p file=%s:%d "
            "alloc_kind=%s expected_free_kind=%s alloc=%s original_id=%llu "
            "original_site=%s:%d\n",
            owner ? owner : "?", p, file ? file : "?", line,
            kind_name(rec.kind), kind_name(expected_kind),
            rec.allocator.c_str(), (unsigned long long)rec.id,
            rec.file.c_str(), rec.line);
        dump_internal_no_lock_();
        std::abort();
    }
    return true;
}

void commit_free(void* p, const char* owner, const char* file, int line) {
    if (!g_enabled || p == nullptr) return;
    std::lock_guard<std::mutex> g(g_mutex);
    auto it = g_live.find(p);
    if (it == g_live.end() || !it->second.alive) {
        std::fprintf(stderr,
            "[ta_commit_free_INTERNAL_ERROR] owner=%s ptr=%p file=%s:%d\n",
            owner ? owner : "?", p, file ? file : "?", line);
        dump_internal_no_lock_();
        std::abort();
    }
    it->second.alive = false;
    if (verbose_logging()) {
        std::fprintf(stderr,
            "[ta_free] id=%llu ptr=%p kind=%s alloc=%s owner=%s file=%s:%d\n",
            (unsigned long long)it->second.id, p,
            kind_name(it->second.kind), it->second.allocator.c_str(),
            owner ? owner : "?", file ? file : "?", line);
    }
}

// Cross-check helper: does `claimed` (the caller's reported origin for an
// inserted pointer) agree with `actual` (the AllocRec.allocator tag for the
// same pointer, when one is registered)? Returns true iff they agree or one
// of them is a host-flavor tag (cudaHostAlloc <-> cudaMallocHost are
// semantically identical host registrations). Mismatches are diagnostic, not
// bugs: the apply walker stamps "unbalanced_origin" when it cannot infer the
// true allocator from the source expression; that tag is treated as a
// passive observer and never triggers the mismatch abort. The compound tag
// "cudaMallocHost_or_cudaHostAlloc" (per-map heuristic from the walker)
// means the caller is OK with either host flavor.
bool origin_matches(const char* claimed, const char* actual) {
    if (claimed == nullptr || actual == nullptr) return true;
    if (std::strcmp(claimed, "unbalanced_origin") == 0) return true;  // passive walker
    if (std::strcmp(claimed, "unknown_origin")      == 0) return true;  // passive walker
    if (std::strcmp(claimed, actual)               == 0) return true;
    const bool a_host =
        (std::strcmp(actual,   "cudaHostAlloc")   == 0 ||
         std::strcmp(actual,   "cudaMallocHost")  == 0);
    if (std::strcmp(claimed, "cudaMallocHost_or_cudaHostAlloc") == 0) {
        return a_host;
    }
    const bool c_host =
        (std::strcmp(claimed,  "cudaHostAlloc")   == 0 ||
         std::strcmp(claimed,  "cudaMallocHost")  == 0);
    return a_host && c_host;
}

void note_insertion(const char* map_name, uint64_t key, void* ptr,
                     const char* origin, const char* file, int line) {
    if (!g_enabled) return;
    InsertRec r;
    r.map_name = map_name ? map_name : "?";
    r.key      = key;
    r.ptr      = ptr;
    r.origin   = origin ? origin : "?";
    r.file     = file ? file : "?";
    r.line     = line;
    std::lock_guard<std::mutex> g(g_mutex);

    // Breach A fix: cross-reference origin against the AllocRec (if any) for
    // this exact pointer. If the inserted pointer was previously registered
    // via a different allocator than the caller claims, abort with the
    // precise mismatch diagnostic so the Kaggle stderr can name the first
    // inconsistent insert site instead of conflating it with double-free.
    auto recv_it = g_live.find(ptr);
    if (recv_it != g_live.end()) {
        const AllocRec& rec = recv_it->second;

        // Use-after-free branch: the pointer was allocated then freed, then
        // re-inserted into a map. A cached-but-dead host/device pointer
        // outliving its registration is the most likely v4 root cause.
        // Even if the origin tag is consistent, the liveness is not.
        if (!rec.alive) {
            ++g_uaf_aborts;
            std::fprintf(stderr,
                "[ta_insert_USE_AFTER_FREE_ABORT] map=%s key=%llu ptr=%p "
                "alloc_rec.id=%llu alloc_rec.kind=%s alloc_rec.allocator=%s "
                "alloc_rec.site=%s:%d alloc_rec.alive=false "
                "insert_site=%s:%d\n",
                r.map_name.c_str(), (unsigned long long)key, ptr,
                (unsigned long long)rec.id,
                kind_name(rec.kind), rec.allocator.c_str(),
                rec.file.c_str(), rec.line,
                r.file.c_str(), r.line);
            dump_internal_no_lock_();
            std::abort();
        }

        const char* actual  = rec.allocator.c_str();
        const char* claimed = r.origin.c_str();
        if (!origin_matches(claimed, actual)) {
            ++g_mismatch_aborts;
            std::fprintf(stderr,
                "[ta_insert_MISMATCH_ABORT] map=%s key=%llu ptr=%p "
                "alloc_rec.kind=%s alloc_rec.allocator=%s "
                "alloc_rec.id=%llu alloc_rec.site=%s:%d "
                "claimed_origin=%s insert_site=%s:%d\n",
                r.map_name.c_str(), (unsigned long long)key, ptr,
                kind_name(rec.kind), actual,
                (unsigned long long)rec.id,
                rec.file.c_str(), rec.line,
                claimed, r.file.c_str(), r.line);
            dump_internal_no_lock_();
            std::abort();
        }
    }

    g_insertions.push_back(r);
    std::fprintf(stderr,
        "[ta_insert] map=%s key=%llu ptr=%p origin=%s file=%s:%d%s\n",
        r.map_name.c_str(), (unsigned long long)key, ptr,
        r.origin.c_str(), r.file.c_str(), r.line,
        recv_it != g_live.end() ? " (alloc_rec=match)" : "");
}

void dump_to_stderr_locked() {
    std::lock_guard<std::mutex> g(g_mutex);
    dump_internal_no_lock_();
}

size_t live_count() {
    std::lock_guard<std::mutex> g(g_mutex);
    size_t n = 0;
    for (const auto& kv : g_live) if (kv.second.alive) ++n;
    return n;
}

size_t dead_count() {
    std::lock_guard<std::mutex> g(g_mutex);
    size_t n = 0;
    for (const auto& kv : g_live) if (!kv.second.alive) ++n;
    return n;
}

size_t unalloc_abort_count() {
    std::lock_guard<std::mutex> g(g_mutex);
    return g_unalloc_aborts;
}
size_t double_free_abort_count() {
    std::lock_guard<std::mutex> g(g_mutex);
    return g_double_free_aborts;
}
size_t mismatch_abort_count() {
    std::lock_guard<std::mutex> g(g_mutex);
    return g_mismatch_aborts;
}
size_t uaf_abort_count() {
    std::lock_guard<std::mutex> g(g_mutex);
    return g_uaf_aborts;
}

uint64_t allocation_id(void* ptr, bool* alive) {
    std::lock_guard<std::mutex> g(g_mutex);
    const auto it = g_live.find(ptr);
    if (it == g_live.end()) {
        if (alive) *alive = false;
        return 0;
    }
    if (alive) *alive = it->second.alive;
    return it->second.id;
}

size_t non_selftest_alloc_count() {
    return g_non_selftest_allocs.load(std::memory_order_relaxed);
}

bool startup_self_test() {
    std::fprintf(stderr, "[DEE_TA_SELFTEST_BEGIN]\n");
    std::fflush(stderr);
#ifdef DEE_CUDA
    void* ptr = nullptr;
    const cudaError_t alloc_status =
        DEE_TA_HOST_ALLOC(
            &ptr, 64, cudaHostAllocDefault, "__dee_ta_selftest__");
    if (alloc_status != cudaSuccess || ptr == nullptr) {
        std::fprintf(stderr, "[DEE_TA_SELFTEST_FAIL] stage=alloc status=%d\n",
                     static_cast<int>(alloc_status));
        std::fflush(stderr);
        return false;
    }
    bool alive = false;
    const uint64_t alloc_id = allocation_id(ptr, &alive);
    if (alloc_id == 0 || !alive) {
        std::fprintf(stderr, "[DEE_TA_SELFTEST_FAIL] stage=lookup_alloc\n");
        std::fflush(stderr);
        cudaFreeHost(ptr);
        return false;
    }
    note_insertion("__dee_ta_selftest__", alloc_id, ptr, "cudaHostAlloc",
                   __FILE__, __LINE__);
    std::fprintf(stderr, "[DEE_TA_ALLOC] id=%llu ptr=%p\n",
                 static_cast<unsigned long long>(alloc_id), ptr);
    std::fflush(stderr);

    const cudaError_t free_status =
        DEE_TA_FREE_HOST(ptr, "__dee_ta_selftest__");
    bool alive_after = true;
    const uint64_t free_id = allocation_id(ptr, &alive_after);
    if (free_status != cudaSuccess || free_id != alloc_id || alive_after) {
        std::fprintf(
            stderr,
            "[DEE_TA_SELFTEST_FAIL] stage=free status=%d alloc_id=%llu free_id=%llu alive=%s\n",
            static_cast<int>(free_status),
            static_cast<unsigned long long>(alloc_id),
            static_cast<unsigned long long>(free_id),
            alive_after ? "true" : "false");
        std::fflush(stderr);
        return false;
    }
    std::fprintf(stderr, "[DEE_TA_FREE] id=%llu ptr=%p\n",
                 static_cast<unsigned long long>(free_id), ptr);
    std::fprintf(stderr, "[DEE_TA_SELFTEST_PASS] id=%llu\n",
                 static_cast<unsigned long long>(free_id));
    std::fflush(stderr);
    return true;
#else
    std::fprintf(stderr, "[DEE_TA_SELFTEST_FAIL] stage=DEE_CUDA_OFF\n");
    std::fflush(stderr);
    return false;
#endif
}

}  // namespace dee::trace_alloc
