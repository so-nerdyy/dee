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

struct ExpertStoreStats {
    std::string backend;
    std::string integrity_identity;
    uint64_t lookups = 0;
    uint64_t lookup_failures = 0;
    uint64_t source_reads = 0;
    uint64_t contiguous_source_reads = 0;
    uint64_t source_regions = 0;
    uint64_t bytes_requested = 0;
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
    uint64_t pread_calls = 0;           // pread() invocations
    uint64_t pread_short_reads = 0;     // calls returning < requested
    uint64_t pread_bytes = 0;           // bytes delivered by pread
    uint64_t mincore_probed_bytes = 0;  // page-cache residency probed
    uint64_t mincore_resident_bytes = 0;// probed bytes already resident
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
    virtual bool materialize(const ExpertView& view, uint8_t* dst,
                             size_t nbytes) const;
    virtual const char* materialization_mode() const { return "mmap_memcpy"; }
    // True when materialize() can serve views that lack a single contiguous
    // record buffer (e.g. per-tensor safetensors regions) via positional
    // reads.  The bounded multi-lane path may then accept non-contiguous
    // views; the produced record bytes are identical either way.
    virtual bool can_gather_materialize() const { return false; }

    // Called by the consumer around the actual source-to-host-L2 copy.  This
    // deliberately measures page-fault/storage wait rather than the cheap
    // arithmetic lookup that merely returns pointers into an mmap.
    void record_source_read(size_t bytes, double milliseconds,
                            size_t regions, bool contiguous);
    void record_source_read_batch(size_t requests, size_t lanes,
                                  double wall_milliseconds,
                                  double summed_read_milliseconds);
    ExpertStoreStats stats() const;

protected:
    void record_lookup(bool success);
    // Lock-free read-service accounting for worker threads (profiling-only).
    // Safe to call from materialize() on any thread; never affects reads.
    void note_pread_service(uint64_t service_ns, uint64_t bytes,
                            uint64_t short_reads, uint64_t probed_bytes,
                            uint64_t resident_bytes) const;
    struct ReadTelemetry {
        uint64_t service_ns = 0;
        uint64_t calls = 0;
        uint64_t short_reads = 0;
        uint64_t bytes = 0;
        uint64_t probed_bytes = 0;
        uint64_t resident_bytes = 0;
    };
    // Snapshot (and optionally reset) the worker-thread telemetry.
    ReadTelemetry read_telemetry(bool reset = false) const;

private:
    uint64_t lookups_ = 0;
    uint64_t lookup_failures_ = 0;
    uint64_t source_reads_ = 0;
    uint64_t contiguous_source_reads_ = 0;
    uint64_t source_regions_ = 0;
    uint64_t bytes_requested_ = 0;
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
    // view-by-view fill either way.
    bool materialize(const ExpertView& view, uint8_t* dst,
                     size_t nbytes) const override;
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
    bool materialize(const ExpertView& view, uint8_t* dst,
                     size_t nbytes) const override;
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
