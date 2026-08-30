// dee/expert_store.h
//
// Model-neutral routed-expert storage seam.  Residency, routing, and
// arithmetic consume ExpertView records without knowing whether the bytes
// came from canonical safetensors or an expert-major DEE4 bank.

#pragma once

#include "dee/weight_mmap.h"

#include <array>
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

    // Called by the consumer around the actual source-to-host-L2 copy.  This
    // deliberately measures page-fault/storage wait rather than the cheap
    // arithmetic lookup that merely returns pointers into an mmap.
    void record_source_read(size_t bytes, double milliseconds,
                            size_t regions, bool contiguous);
    ExpertStoreStats stats() const;

protected:
    void record_lookup(bool success);

private:
    uint64_t lookups_ = 0;
    uint64_t lookup_failures_ = 0;
    uint64_t source_reads_ = 0;
    uint64_t contiguous_source_reads_ = 0;
    uint64_t source_regions_ = 0;
    uint64_t bytes_requested_ = 0;
    double read_milliseconds_ = 0.0;
    std::vector<double> read_latencies_ms_;
};

class SafetensorsExpertStore final : public ExpertStore {
public:
    explicit SafetensorsExpertStore(TensorResolver* resolver);
    bool get(int layer, int expert, ExpertView* out) override;
    const char* backend_name() const override { return "safetensors"; }
    const std::string& integrity_identity() const override { return identity_; }

private:
    TensorResolver* resolver_ = nullptr;
    std::string identity_ = "canonical-safetensors-resolver";
};

// DEE4 maps one fixed-stride, expert-major data file. V2 uses dense arithmetic
// lookup. V3-trace stores only the exact routed (layer, expert) union and uses
// an immutable sorted index; missing records fail closed.
class Dee4ExpertStore final : public ExpertStore {
public:
    Dee4ExpertStore();
    ~Dee4ExpertStore() override;

    bool open(const std::string& directory_or_metadata);
    void close();
    bool get(int layer, int expert, ExpertView* out) override;
    bool get_layout_reference(int preferred_layer, ExpertView* out) override;
    const char* backend_name() const override { return backend_.c_str(); }
    const std::string& integrity_identity() const override { return identity_; }

    int start_layer() const { return start_layer_; }
    int num_layers() const { return num_layers_; }
    int experts_per_layer() const { return experts_per_layer_; }
    size_t record_bytes() const { return record_bytes_; }
    bool trace_indexed() const { return trace_indexed_; }
    size_t stored_records() const { return stored_records_; }
    const std::string& last_error() const { return last_error_; }

private:
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
    size_t stored_records_ = 0;
    std::string backend_ = "dee4";
    std::string identity_;
    std::string last_error_;

    bool map_file(const std::string& path);
};

}  // namespace dee
