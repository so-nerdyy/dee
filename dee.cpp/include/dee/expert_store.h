// dee/expert_store.h
//
// Model-neutral routed-expert storage seam.  Residency, routing, and
// arithmetic consume ExpertView records without knowing whether the bytes
// came from canonical safetensors or an expert-major DEE4 bank.

#pragma once

#include "dee/weight_mmap.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dee {

enum class ExpertCodec {
    Unknown,
    DeepSeekFp4E2m1E8m0,
};

// Projection order is gate/w1, up/w3, down/w2.  A DEE4 record is laid out as
// all three packed weights followed by all three scale tensors, exactly as the
// existing FP4 staging/cache path expects.
struct ExpertView {
    std::array<TensorView, 3> weights{};
    std::array<TensorView, 3> scales{};
    const uint8_t* contiguous_data = nullptr;
    size_t contiguous_nbytes = 0;
    ExpertCodec codec = ExpertCodec::Unknown;
    uint64_t record_index = 0;
    const char* integrity_identity = nullptr;

    bool ok() const {
        for (size_t i = 0; i < 3; ++i) {
            if (!weights[i].ok() || !scales[i].ok()) return false;
        }
        return codec != ExpertCodec::Unknown;
    }
};

// Per-fill accounting returned by materialize_ex().  `bytes_read` is the
// number of bytes actually read/copied into dst (== nbytes on success,
// partial count on a mid-read failure, 0 on validation failure).
// `pread_calls` counts the real pread() syscalls issued for this fill (0 on
// memcpy-backed paths).  `resident_bytes` is the page-cache residency of the
// record's source range measured BEFORE the read via Linux mincore; 0
// everywhere else (including all Windows builds and the safetensors gather
// on non-Linux POSIX).
struct MaterializeResult {
    bool success = false;
    std::size_t bytes_read = 0;
    std::size_t resident_bytes = 0;
    std::size_t pread_calls = 0;
};

struct ExpertStoreStats {
    std::string backend;
    std::string integrity_identity;
    uint64_t lookups = 0;
    uint64_t lookup_failures = 0;
    uint64_t source_reads = 0;
    uint64_t contiguous_source_reads = 0;
    uint64_t source_regions = 0;
    uint64_t bytes_requested = 0;
    // Failed fills reported by the consumer via record_source_read_failure()
    // (e.g. a materialize that returned false after the view resolved).
    // Deliberately NOT folded into source_reads or the latency percentiles.
    uint64_t source_read_failures = 0;
    double read_milliseconds = 0.0;
    double average_request_bytes = 0.0;
    double average_read_ms = 0.0;
    double p50_read_ms = 0.0;
    double p95_read_ms = 0.0;
    double max_read_ms = 0.0;
    double read_bandwidth_mib_s = 0.0;
    std::string materialization_mode;
    uint64_t source_read_batches = 0;
    uint64_t concurrent_source_read_batches = 0;
    uint64_t max_source_read_queue_depth = 0;
    uint64_t max_source_read_lanes = 0;
    double source_read_batch_wall_ms = 0.0;
    double source_read_overlap_ms = 0.0;
    double source_read_overlap_percent = 0.0;
    // Fill-path read-service decomposition (dee4 pread path; profiling-only,
    // accumulated via lock-free atomics from worker threads, zero when unused).
    double pread_service_ms = 0.0;      // time inside pread() syscalls
    uint64_t pread_calls = 0;           // pread() syscall invocations
    // pread_short_reads is an EVENT count: the number of pread() calls that
    // returned fewer bytes than requested (not a byte count).  Kept under
    // its historical name; pread_short_read_events is the same counter with
    // the explicit name and is the preferred field going forward.
    uint64_t pread_short_reads = 0;
    uint64_t pread_short_read_events = 0;
    uint64_t pread_bytes = 0;           // bytes delivered by pread
    uint64_t mincore_probed_bytes = 0;  // page-cache residency probed
    uint64_t mincore_resident_bytes = 0;// probed bytes already resident
    // Per-fill entry/exit accounting through materialize_ex() (which
    // materialize() also forwards through).  materialize_calls counts every
    // attempt including validation failures; materialize_failures counts the
    // calls that returned false.  A failed fill still contributes its
    // partial pread telemetry to the fields above.
    uint64_t materialize_calls = 0;
    uint64_t materialize_failures = 0;
};

class ExpertStore {
public:
    virtual ~ExpertStore() = default;
    virtual bool get(int layer, int expert, ExpertView* out) = 0;
    // Return one layout-compatible record for engine shape discovery. This is
    // not a routed lookup: sparse stores may not contain expert zero.
    virtual bool get_layout_reference(int preferred_layer, ExpertView* out) {
        return get(preferred_layer, 0, out);
    }
    virtual const char* backend_name() const = 0;
    virtual const std::string& integrity_identity() const = 0;
    // Copy one exact expert record into caller-owned bounded host storage.
    // The default consumes the already-resolved contiguous mmap view. DEE4
    // overrides this with positional reads on Linux so several independent
    // records can be materialized concurrently without changing lookup,
    // routing, or transfer order.
    //
    // materialize() is now a thin forwarder: it calls materialize_ex() with
    // a null result pointer.  Stores override materialize_ex(); a legacy
    // override of materialize() still works for direct materialize() calls
    // but is bypassed by materialize_ex() and escapes the per-fill counters.
    virtual bool materialize(const ExpertView& view, uint8_t* dst,
                             size_t nbytes) const;
    // Extended fill with per-fill accounting.  Same byte contract as
    // materialize(); `out` may be nullptr.  The base implementation performs
    // the contiguous-view memcpy (identical to the historic materialize()
    // body) and reports best-effort fields: bytes_read = nbytes on success,
    // pread_calls = 0, resident_bytes = 0.  Every implementation — success
    // or failure — counts the call via complete_materialize(), so a failed
    // materialize still lands in stats().materialize_failures (and keeps
    // whatever partial pread telemetry it accumulated).
    virtual bool materialize_ex(const ExpertView& view, void* dst,
                                size_t nbytes, MaterializeResult* out) const;
    virtual const char* materialization_mode() const { return "mmap_memcpy"; }
    // True when materialize() can serve views that lack a single contiguous
    // record buffer (e.g. per-tensor safetensors regions) via positional
    // reads.  The bounded multi-lane path may then accept non-contiguous
    // views; the produced record bytes are identical either way.
    virtual bool can_gather_materialize() const { return false; }

    // Best-effort release of the file-backed pages carrying one resolved
    // record: POSIX_FADV_DONTNEED on the source range plus MADV_DONTNEED on
    // the mapping, so a copied-out record stops double-booking page cache.
    // Never affects correctness — the record stays readable through the
    // store after release.  Returns false (and sets *bytes_released = 0)
    // where the release cannot be performed safely: Windows builds,
    // non-page-aligned record geometry, unresolvable or forged views.
    // Partial releases may leave *bytes_released < record bytes on failure.
    //
    // This is the store-side analogue of WeightMmap::discard_source_pages:
    // the engine's DEE_RELEASE_MMAP_PAGES path only ranges over safetensors
    // shard mmaps and is a silent no-op for dee4 segment files; call this
    // instead to cover both backends.
    virtual bool release_source_pages(const ExpertView& view,
                                      size_t* bytes_released) const;

    // Called by the consumer around the actual source-to-host-L2 copy.  This
    // deliberately measures page-fault/storage wait rather than the cheap
    // arithmetic lookup that merely returns pointers into an mmap.
    void record_source_read(size_t bytes, double milliseconds,
                            size_t regions, bool contiguous);
    // Record a fill that failed after the view resolved (materialize()
    // returned false, a gather region was unresolvable, ...).  Counted only
    // in stats().source_read_failures: it does not enter source_reads,
    // bytes_requested, or the latency percentiles, so failure noise cannot
    // skew the measured read distribution.  Same caller-thread contract as
    // record_source_read().
    void record_source_read_failure();
    void record_source_read_batch(size_t requests, size_t lanes,
                                  double wall_milliseconds,
                                  double summed_read_milliseconds);
    ExpertStoreStats stats() const;
    // Zero every counter reported by stats(), clear the retained
    // read-latency samples, and reset the lock-free worker telemetry.
    // Needed for honest per-prompt accounting (the counters are otherwise
    // process-cumulative).  Caller-thread contract is the same as stats():
    // call only when no lookup/fill is in flight on this store — the plain
    // counters are not synchronized against concurrent workers, and the
    // atomic accumulators can still interleave with an in-flight fill.
    virtual void reset_stats();

protected:
    void record_lookup(bool success);
    // Funnel for materialize_ex() exits: copies `result` to `out` (when
    // non-null), counts the call — and the failure — in the lock-free
    // materialize stats, and returns result.success so implementations can
    // `return complete_materialize(out, result);` from every exit path.
    bool complete_materialize(MaterializeResult* out,
                              const MaterializeResult& result) const;
    // Lock-free read-service accounting for worker threads (profiling-only).
    // Safe to call from materialize() on any thread; never affects reads.
    // This legacy form counts the fill as ONE pread() call.
    void note_pread_service(uint64_t service_ns, uint64_t bytes,
                            uint64_t short_reads, uint64_t probed_bytes,
                            uint64_t resident_bytes) const;
    // Preferred form: `calls` is the actual number of pread() syscalls the
    // fill issued (a short read loops, so calls can exceed 1 per fill).
    void note_pread_service_ex(uint64_t service_ns, uint64_t calls,
                               uint64_t bytes, uint64_t short_reads,
                               uint64_t probed_bytes,
                               uint64_t resident_bytes) const;
    struct ReadTelemetry {
        uint64_t service_ns = 0;
        uint64_t calls = 0;
        uint64_t short_reads = 0;
        uint64_t bytes = 0;
        uint64_t probed_bytes = 0;
        uint64_t resident_bytes = 0;
    };
    // Snapshot (and optionally reset) the worker-thread telemetry.  The six
    // counters are loaded independently with relaxed ordering, so a snapshot
    // taken while fills are in flight can tear (e.g. bytes tallied without
    // the matching call count).  Profiling-only by contract; a seqlock is
    // deliberately not used.
    ReadTelemetry read_telemetry(bool reset = false) const;

private:
    uint64_t lookups_ = 0;
    uint64_t lookup_failures_ = 0;
    uint64_t source_reads_ = 0;
    uint64_t contiguous_source_reads_ = 0;
    uint64_t source_regions_ = 0;
    uint64_t bytes_requested_ = 0;
    uint64_t source_read_failures_ = 0;
    double read_milliseconds_ = 0.0;
    std::vector<double> read_latencies_ms_;
    uint64_t source_read_batches_ = 0;
    uint64_t concurrent_source_read_batches_ = 0;
    uint64_t max_source_read_queue_depth_ = 0;
    uint64_t max_source_read_lanes_ = 0;
    double source_read_batch_wall_ms_ = 0.0;
    double source_read_overlap_ms_ = 0.0;
    // Worker-thread read-service telemetry (mutable atomics; profiling-only).
    mutable std::atomic<uint64_t> pread_service_ns_{0};
    mutable std::atomic<uint64_t> pread_calls_{0};
    mutable std::atomic<uint64_t> pread_short_reads_{0};
    mutable std::atomic<uint64_t> pread_bytes_{0};
    mutable std::atomic<uint64_t> mincore_probed_bytes_{0};
    mutable std::atomic<uint64_t> mincore_resident_bytes_{0};
    // Per-fill entry/exit counters (materialize_ex is const and runs on
    // fill worker threads, so these are atomics like the block above).
    mutable std::atomic<uint64_t> materialize_calls_{0};
    mutable std::atomic<uint64_t> materialize_failures_{0};
};

class SafetensorsExpertStore final : public ExpertStore {
public:
    explicit SafetensorsExpertStore(TensorResolver* resolver);
    bool get(int layer, int expert, ExpertView* out) override;
    const char* backend_name() const override { return "safetensors"; }
    const std::string& integrity_identity() const override { return identity_; }
    // Gathers the six tensor regions (gate/up/down packed weights then
    // scales) into one record buffer via pread() on the owning shard's fd
    // (POSIX) or a per-region memcpy elsewhere.  Byte-identical to the
    // view-by-view fill either way.  MaterializeResult.pread_calls counts
    // the real gather syscalls; resident_bytes is probed per region via
    // mincore on Linux (0 elsewhere) — each region is contiguous inside a
    // single shard mapping, so probing is exact when it runs at all.
    bool materialize_ex(const ExpertView& view, void* dst,
                        size_t nbytes, MaterializeResult* out) const override;
    // Releases each region's source range via the owning shard's
    // WeightMmap::discard_source_pages; needs every region to resolve to a
    // registered shard.  Returns false on Windows and whenever any region
    // fails, with *bytes_released carrying the partial count.
    bool release_source_pages(const ExpertView& view,
                              size_t* bytes_released) const override;
    const char* materialization_mode() const override;
    bool can_gather_materialize() const override;

private:
    TensorResolver* resolver_ = nullptr;
    std::string identity_ = "canonical-safetensors-resolver";
};

// Options for Dee4ExpertStore::open.  Only meaningful for dee4-v4-segmented
// stores today; dee4-v2 and dee4-v3-trace ignore every field.
struct Dee4OpenOptions {
    // Re-hash every segment's mapped bytes against the sha256 seals in the
    // metadata segment table during open() and fail closed on any mismatch.
    // Default true: the per-segment sha256 IS the segmented format's publish
    // integrity seal, so an unsealed segment must never be served.  The cost
    // is a one-time O(total segment bytes) hash pass inside open(); set false
    // only for trusted local mirrors (missing/mis-sized/malformed-segment
    // checks are structural and always apply).
    bool verify_segment_hashes = true;
    // When verifying seals, hash each segment through chunked pread() with
    // POSIX_FADV_DONTNEED trailing the read cursor instead of walking the
    // mmap.  The digest is identical either way; the pread form stops a
    // ~146 GiB seal pass from flooding the page cache.  POSIX only — on
    // Windows (or when a segment fd is unavailable) the reader falls back
    // to the mmap walk.  Default false: page-cache warming during the seal
    // is harmless on hosts that re-read through the mapping anyway.
    bool seal_reads_through_page_cache = false;
};

// DEE4 maps one fixed-stride, expert-major data file. V2 uses dense arithmetic
// lookup. V3-trace stores only the exact routed (layer, expert) union and uses
// an immutable sorted index; missing records fail closed.  V4-segmented keeps
// the v2 dense record index but splits the byte space into one file per
// segment (segments/experts-bucket-NN.dee4 plus a segment table in
// metadata.json); each segment is mmap'd and lookups resolve
// record_index -> (segment, in-segment offset).  Segment files that are
// missing, mis-sized, or (by default) mis-hashed fail closed at open().
class Dee4ExpertStore final : public ExpertStore {
public:
    Dee4ExpertStore();
    ~Dee4ExpertStore() override;

    bool open(const std::string& directory_or_metadata);
    bool open(const std::string& directory_or_metadata,
              const Dee4OpenOptions& options);
    void close();
    bool get(int layer, int expert, ExpertView* out) override;
    bool get_layout_reference(int preferred_layer, ExpertView* out) override;
    const char* backend_name() const override { return backend_.c_str(); }
    const std::string& integrity_identity() const override { return identity_; }
    // Positional pread() fill on POSIX (per-record syscalls counted, mincore
    // residency probed before the read on Linux, partial telemetry recorded
    // even when the read fails); plain memcpy of the mapped record on
    // Windows (pread_calls/resident_bytes reported as 0 there).
    bool materialize_ex(const ExpertView& view, void* dst,
                        size_t nbytes, MaterializeResult* out) const override;
    // POSIX: posix_fadvise(DONTNEED) on the record's file range plus
    // madvise(MADV_DONTNEED) on the mapped range.  The record extent must be
    // page-exact — segment bases are page-aligned by mmap and the real
    // record_bytes (13,369,344 = 4096 * 3264) is a page multiple; both are
    // verified at runtime and the call no-ops with false otherwise.
    // Windows: always false, *bytes_released = 0.
    bool release_source_pages(const ExpertView& view,
                              size_t* bytes_released) const override;
#ifdef _WIN32
    const char* materialization_mode() const override { return "mmap_memcpy"; }
#else
    const char* materialization_mode() const override { return "pread"; }
#endif

    int start_layer() const { return start_layer_; }
    int num_layers() const { return num_layers_; }
    int experts_per_layer() const { return experts_per_layer_; }
    size_t record_bytes() const { return record_bytes_; }
    bool trace_indexed() const { return trace_indexed_; }
    bool segmented() const { return segmented_; }
    size_t segment_count() const { return segments_.size(); }
    size_t stored_records() const { return stored_records_; }
    const std::string& last_error() const { return last_error_; }

private:
    // One mmap'd segment file of a dee4-v4-segmented store.  Records
    // first_record .. first_record+record_count-1 live at
    // base + (record_index - first_record) * record_bytes_.
    struct Segment {
        void* mapping_handle = nullptr;  // Windows only
        int fd = -1;                     // POSIX only; 0 on Windows success
        uint8_t* base = nullptr;
        size_t size = 0;
        size_t first_record = 0;
        size_t record_count = 0;
        std::string file;      // declared store-relative path (error context)
        std::string sha256;    // declared content seal (lowercase hex)
    };
    void* mapping_handle_ = nullptr;
    int fd_ = -1;
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
    int start_layer_ = 0;
    int num_layers_ = 0;
    int experts_per_layer_ = 0;
    size_t record_bytes_ = 0;
    std::array<size_t, 3> weight_offsets_{};
    std::array<size_t, 3> weight_nbytes_{};
    std::array<size_t, 3> weight_out_{};
    std::array<size_t, 3> weight_stored_in_{};
    std::array<size_t, 3> scale_offsets_{};
    std::array<size_t, 3> scale_nbytes_{};
    std::array<size_t, 3> scale_out_{};
    std::array<size_t, 3> scale_in_{};
    struct TraceRecord {
        int layer = 0;
        int expert = 0;
        size_t record_index = 0;
    };
    std::vector<TraceRecord> trace_records_;
    bool trace_indexed_ = false;
    // dee4-v4-segmented: ordered segment table (sorted by first_record,
    // contiguous coverage of [0, stored_records_)) with one mapping each.
    std::vector<Segment> segments_;
    bool segmented_ = false;
    size_t stored_records_ = 0;
    std::string backend_ = "dee4";
    std::string identity_;
    std::string last_error_;

    bool map_file(const std::string& path);
    bool map_segment_file(const std::string& path, Segment* segment);
    void unmap_segment(Segment* segment);
    const Segment* find_segment(size_t record_index) const;
};

}  // namespace dee
