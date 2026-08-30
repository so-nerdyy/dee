#include "dee/expert_store.h"
#include "dee/json_min.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dee {

namespace {

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
        values.size() - 1,
        static_cast<size_t>(std::ceil(q * static_cast<double>(values.size()))) - 1);
    return values[index];
}

bool json_string(const json::Value* root, const char* key, std::string* out) {
    const json::Value* value = root ? root->find(key) : nullptr;
    if (!value || !value->is_string()) return false;
    *out = value->s;
    return true;
}

bool json_nonnegative_int(const json::Value* root, const char* key,
                          size_t* out) {
    const json::Value* value = root ? root->find(key) : nullptr;
    if (!value || !value->is_int() || value->i < 0) return false;
    if (static_cast<unsigned long long>(value->i) >
        static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *out = static_cast<size_t>(value->i);
    return true;
}

template <size_t N>
bool json_size_array(const json::Value* root, const char* key,
                     std::array<size_t, N>* out) {
    const json::Value* value = root ? root->find(key) : nullptr;
    if (!value || !value->is_array() || value->arr.size() != N) return false;
    for (size_t i = 0; i < N; ++i) {
        const auto& item = value->arr[i];
        if (!item || !item->is_int() || item->i < 0) return false;
        if (static_cast<unsigned long long>(item->i) >
            static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        (*out)[i] = static_cast<size_t>(item->i);
    }
    return true;
}

bool checked_mul(size_t lhs, size_t rhs, size_t* out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) return false;
    *out = lhs * rhs;
    return true;
}

bool is_sha256_hex(const std::string& value) {
    if (value.size() != 64) return false;
    for (const unsigned char c : value) {
        if (!std::isxdigit(c)) return false;
    }
    return true;
}

}  // namespace

void ExpertStore::record_lookup(bool success) {
    ++lookups_;
    if (!success) ++lookup_failures_;
}

void ExpertStore::record_source_read(size_t bytes, double milliseconds,
                                     size_t regions, bool contiguous) {
    ++source_reads_;
    if (contiguous) ++contiguous_source_reads_;
    source_regions_ += regions;
    bytes_requested_ += bytes;
    read_milliseconds_ += std::max(0.0, milliseconds);
    // A full generation has only a few thousand cold loads.  Reserving lazily
    // keeps construction cheap while retaining exact p50/p95 measurements.
    if (read_latencies_ms_.capacity() == 0) read_latencies_ms_.reserve(16384);
    read_latencies_ms_.push_back(std::max(0.0, milliseconds));
}

ExpertStoreStats ExpertStore::stats() const {
    ExpertStoreStats result;
    result.backend = backend_name();
    result.integrity_identity = integrity_identity();
    result.lookups = lookups_;
    result.lookup_failures = lookup_failures_;
    result.source_reads = source_reads_;
    result.contiguous_source_reads = contiguous_source_reads_;
    result.source_regions = source_regions_;
    result.bytes_requested = bytes_requested_;
    result.read_milliseconds = read_milliseconds_;
    if (source_reads_ != 0) {
        result.average_request_bytes =
            static_cast<double>(bytes_requested_) / static_cast<double>(source_reads_);
        result.average_read_ms = read_milliseconds_ / static_cast<double>(source_reads_);
    }
    result.p50_read_ms = percentile(read_latencies_ms_, 0.50);
    result.p95_read_ms = percentile(read_latencies_ms_, 0.95);
    result.max_read_ms = read_latencies_ms_.empty()
        ? 0.0 : *std::max_element(read_latencies_ms_.begin(), read_latencies_ms_.end());
    if (read_milliseconds_ > 0.0) {
        result.read_bandwidth_mib_s =
            (static_cast<double>(bytes_requested_) / (1024.0 * 1024.0)) /
            (read_milliseconds_ / 1000.0);
    }
    return result;
}

SafetensorsExpertStore::SafetensorsExpertStore(TensorResolver* resolver)
    : resolver_(resolver) {}

bool SafetensorsExpertStore::get(int layer, int expert, ExpertView* out) {
    if (!out || !resolver_) {
        record_lookup(false);
        return false;
    }
    ExpertView view;
    view.codec = ExpertCodec::DeepSeekFp4E2m1E8m0;
    view.record_index = (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
                        static_cast<uint32_t>(expert);
    view.integrity_identity = identity_.c_str();
    const TensorResolver::Kind kinds[3] = {
        TensorResolver::GATE_PROJ,
        TensorResolver::UP_PROJ,
        TensorResolver::DOWN_PROJ,
    };
    for (size_t i = 0; i < 3; ++i) {
        view.weights[i] = resolver_->resolve_expert(layer, expert, kinds[i]);
        view.scales[i] = resolver_->resolve_expert_scale(layer, expert, kinds[i]);
    }
    const bool success = view.ok();
    record_lookup(success);
    if (!success) return false;
    *out = std::move(view);
    return true;
}

Dee4ExpertStore::Dee4ExpertStore() = default;

Dee4ExpertStore::~Dee4ExpertStore() { close(); }

void Dee4ExpertStore::close() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (mapping_handle_) CloseHandle(static_cast<HANDLE>(mapping_handle_));
    mapping_handle_ = nullptr;
#else
    if (base_ && base_ != MAP_FAILED) munmap(base_, size_);
    if (fd_ >= 0) ::close(fd_);
#endif
    fd_ = -1;
    base_ = nullptr;
    size_ = 0;
    start_layer_ = 0;
    num_layers_ = 0;
    experts_per_layer_ = 0;
    record_bytes_ = 0;
    trace_records_.clear();
    trace_indexed_ = false;
    stored_records_ = 0;
    backend_ = "dee4";
    identity_.clear();
}

bool Dee4ExpertStore::map_file(const std::string& path) {
#ifdef _WIN32
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        last_error_ = "CreateFile failed for DEE4 data";
        return false;
    }
    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart <= 0 ||
        static_cast<unsigned long long>(file_size.QuadPart) >
            static_cast<unsigned long long>(SIZE_MAX)) {
        CloseHandle(file);
        last_error_ = "invalid DEE4 data file size";
        return false;
    }
    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(file);
    if (!mapping) {
        last_error_ = "CreateFileMapping failed for DEE4 data";
        return false;
    }
    void* mapped = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!mapped) {
        CloseHandle(mapping);
        last_error_ = "MapViewOfFile failed for DEE4 data";
        return false;
    }
    mapping_handle_ = mapping;
    base_ = static_cast<uint8_t*>(mapped);
    size_ = static_cast<size_t>(file_size.QuadPart);
    fd_ = 0;
    return true;
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        last_error_ = "open failed for DEE4 data";
        return false;
    }
    struct stat st{};
    if (fstat(fd_, &st) != 0 || st.st_size <= 0 ||
        static_cast<unsigned long long>(st.st_size) >
            static_cast<unsigned long long>(SIZE_MAX)) {
        ::close(fd_);
        fd_ = -1;
        last_error_ = "invalid DEE4 data file size";
        return false;
    }
    size_ = static_cast<size_t>(st.st_size);
    base_ = static_cast<uint8_t*>(
        mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0));
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        last_error_ = "mmap failed for DEE4 data";
        return false;
    }
    // DEE4 turns six scattered tensors into one sequential record per miss.
    posix_madvise(base_, size_, POSIX_MADV_RANDOM);
    return true;
#endif
}

bool Dee4ExpertStore::open(const std::string& directory_or_metadata) {
    close();
    last_error_.clear();
    namespace fs = std::filesystem;
    fs::path input(directory_or_metadata);
    fs::path metadata_path = fs::is_directory(input)
        ? input / "metadata.json" : input;
    if (!fs::is_regular_file(metadata_path)) {
        last_error_ = "DEE4 metadata.json is missing";
        return false;
    }
    std::ifstream stream(metadata_path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(stream)),
                     std::istreambuf_iterator<char>());
    bool parsed = false;
    auto root = json::parse(text, &parsed);
    if (!parsed || !root || !root->is_object()) {
        last_error_ = "DEE4 metadata JSON parse failed";
        return false;
    }
    std::string format;
    std::string codec;
    std::string data_file;
    size_t start_layer = 0;
    size_t num_layers = 0;
    size_t experts_per_layer = 0;
    if (!json_string(root.get(), "format", &format) ||
        (format != "dee4-v2" && format != "dee4-v3-trace") ||
        !json_string(root.get(), "codec", &codec) ||
        codec != "deepseek-fp4-e2m1-e8m0" ||
        !json_string(root.get(), "data_file", &data_file) || data_file.empty() ||
        !json_string(root.get(), "data_sha256", &identity_) ||
        !is_sha256_hex(identity_) ||
        !json_nonnegative_int(root.get(), "start_layer", &start_layer) ||
        !json_nonnegative_int(root.get(), "num_layers", &num_layers) ||
        !json_nonnegative_int(root.get(), "experts_per_layer", &experts_per_layer) ||
        !json_nonnegative_int(root.get(), "record_bytes", &record_bytes_) ||
        !json_size_array(root.get(), "weight_offsets", &weight_offsets_) ||
        !json_size_array(root.get(), "weight_nbytes", &weight_nbytes_) ||
        !json_size_array(root.get(), "weight_out", &weight_out_) ||
        !json_size_array(root.get(), "weight_stored_in", &weight_stored_in_) ||
        !json_size_array(root.get(), "scale_offsets", &scale_offsets_) ||
        !json_size_array(root.get(), "scale_nbytes", &scale_nbytes_) ||
        !json_size_array(root.get(), "scale_out", &scale_out_) ||
        !json_size_array(root.get(), "scale_in", &scale_in_)) {
        last_error_ = "DEE4 metadata schema/codec/layout is invalid";
        close();
        return false;
    }
    if (num_layers == 0 || experts_per_layer == 0 || record_bytes_ == 0 ||
        start_layer > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        num_layers > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        num_layers > static_cast<size_t>(std::numeric_limits<int>::max()) - start_layer ||
        experts_per_layer > static_cast<size_t>(std::numeric_limits<int>::max())) {
        last_error_ = "DEE4 geometry is empty or out of range";
        close();
        return false;
    }
    for (size_t p = 0; p < 3; ++p) {
        if (weight_nbytes_[p] == 0 || scale_nbytes_[p] == 0 ||
            weight_offsets_[p] > record_bytes_ ||
            weight_nbytes_[p] > record_bytes_ - weight_offsets_[p] ||
            scale_offsets_[p] > record_bytes_ ||
            scale_nbytes_[p] > record_bytes_ - scale_offsets_[p]) {
            last_error_ = "DEE4 component range escapes its record";
            close();
            return false;
        }
        size_t expected_weight = 0;
        size_t expected_scale = 0;
        if (!checked_mul(weight_out_[p], weight_stored_in_[p], &expected_weight) ||
            !checked_mul(scale_out_[p], scale_in_[p], &expected_scale) ||
            expected_weight != weight_nbytes_[p] ||
            expected_scale != scale_nbytes_[p]) {
            last_error_ = "DEE4 component shape/byte length mismatch";
            close();
            return false;
        }
    }
    size_t total_experts = 0;
    size_t expected_size = 0;
    if (format == "dee4-v2") {
        if (!checked_mul(num_layers, experts_per_layer, &total_experts)) {
            last_error_ = "DEE4 total size overflow";
            close();
            return false;
        }
    } else {
        size_t declared_total = 0;
        std::string journal_sha;
        std::string final_chain_sha;
        std::string selection_sha;
        const json::Value* records = root->find("records");
        if (!json_nonnegative_int(root.get(), "total_experts", &declared_total) ||
            declared_total == 0 || !records || !records->is_array() ||
            records->arr.size() != declared_total ||
            !json_string(root.get(), "trace_journal_sha256", &journal_sha) ||
            !is_sha256_hex(journal_sha) ||
            !json_string(root.get(), "trace_final_chain_sha256", &final_chain_sha) ||
            !is_sha256_hex(final_chain_sha) ||
            !json_string(root.get(), "selection_sha256", &selection_sha) ||
            !is_sha256_hex(selection_sha)) {
            last_error_ = "DEE4 trace index metadata is invalid";
            close();
            return false;
        }
        trace_records_.reserve(declared_total);
        for (size_t index = 0; index < records->arr.size(); ++index) {
            const json::Value* item = records->arr[index].get();
            size_t layer = 0;
            size_t expert = 0;
            size_t record_index = 0;
            if (!item || !item->is_object() ||
                !json_nonnegative_int(item, "layer", &layer) ||
                !json_nonnegative_int(item, "expert", &expert) ||
                !json_nonnegative_int(item, "record_index", &record_index) ||
                layer < start_layer || layer >= start_layer + num_layers ||
                expert >= experts_per_layer || record_index != index ||
                layer > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                expert > static_cast<size_t>(std::numeric_limits<int>::max())) {
                last_error_ = "DEE4 trace record index is invalid";
                close();
                return false;
            }
            TraceRecord record{
                static_cast<int>(layer), static_cast<int>(expert), record_index};
            if (!trace_records_.empty()) {
                const TraceRecord& previous = trace_records_.back();
                if (record.layer < previous.layer ||
                    (record.layer == previous.layer && record.expert <= previous.expert)) {
                    last_error_ = "DEE4 trace records are not strictly sorted/unique";
                    close();
                    return false;
                }
            }
            trace_records_.push_back(record);
        }
        total_experts = declared_total;
        trace_indexed_ = true;
        backend_ = "dee4_trace";
    }
    if (
        !checked_mul(total_experts, record_bytes_, &expected_size)) {
        last_error_ = "DEE4 total size overflow";
        close();
        return false;
    }
    fs::path data_path = metadata_path.parent_path() / data_file;
    if (!map_file(data_path.string())) {
        close();
        return false;
    }
    if (size_ != expected_size) {
        std::ostringstream message;
        message << "DEE4 data size " << size_ << " does not match expected "
                << expected_size;
        last_error_ = message.str();
        close();
        return false;
    }
    start_layer_ = static_cast<int>(start_layer);
    num_layers_ = static_cast<int>(num_layers);
    experts_per_layer_ = static_cast<int>(experts_per_layer);
    stored_records_ = total_experts;
    return true;
}

bool Dee4ExpertStore::get(int layer, int expert, ExpertView* out) {
    const bool in_range = out && base_ && layer >= start_layer_ &&
        layer < start_layer_ + num_layers_ && expert >= 0 &&
        expert < experts_per_layer_;
    if (!in_range) {
        record_lookup(false);
        return false;
    }
    size_t record_index = 0;
    if (trace_indexed_) {
        const auto found = std::lower_bound(
            trace_records_.begin(), trace_records_.end(),
            std::pair<int, int>{layer, expert},
            [](const TraceRecord& record, const std::pair<int, int>& key) {
                return record.layer < key.first ||
                    (record.layer == key.first && record.expert < key.second);
            });
        if (found == trace_records_.end() || found->layer != layer ||
            found->expert != expert) {
            record_lookup(false);
            return false;
        }
        record_index = found->record_index;
    } else {
        record_index =
            static_cast<size_t>(layer - start_layer_) *
                static_cast<size_t>(experts_per_layer_) +
            static_cast<size_t>(expert);
    }
    const size_t record_offset = record_index * record_bytes_;
    if (record_offset > size_ || record_bytes_ > size_ - record_offset) {
        record_lookup(false);
        return false;
    }
    const uint8_t* record = base_ + record_offset;
    ExpertView view;
    view.codec = ExpertCodec::DeepSeekFp4E2m1E8m0;
    view.contiguous_data = record;
    view.contiguous_nbytes = record_bytes_;
    view.record_index = record_index;
    view.integrity_identity = identity_.c_str();
    for (size_t p = 0; p < 3; ++p) {
        view.weights[p].data = record + weight_offsets_[p];
        view.weights[p].nbytes = weight_nbytes_[p];
        view.weights[p].dtype = DType::I8;
        view.weights[p].shape = {
            static_cast<int64_t>(weight_out_[p]),
            static_cast<int64_t>(weight_stored_in_[p]),
        };
        view.scales[p].data = record + scale_offsets_[p];
        view.scales[p].nbytes = scale_nbytes_[p];
        view.scales[p].dtype = DType::F8;
        view.scales[p].shape = {
            static_cast<int64_t>(scale_out_[p]),
            static_cast<int64_t>(scale_in_[p]),
        };
    }
    record_lookup(true);
    *out = std::move(view);
    return true;
}

bool Dee4ExpertStore::get_layout_reference(int preferred_layer,
                                           ExpertView* out) {
    if (!trace_indexed_) return get(preferred_layer, 0, out);
    if (!out || trace_records_.empty()) {
        record_lookup(false);
        return false;
    }
    const auto found = std::lower_bound(
        trace_records_.begin(), trace_records_.end(), preferred_layer,
        [](const TraceRecord& record, int layer) { return record.layer < layer; });
    const TraceRecord& record = found != trace_records_.end()
        ? *found : trace_records_.front();
    return get(record.layer, record.expert, out);
}

}  // namespace dee
