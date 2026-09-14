#include "dee/expert_store.h"
#include "dee/json_min.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

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

// Minimal SHA-256 (FIPS 180-4) for dee4-v4-segmented seal verification and
// the derived segment-table identity.  Self-contained: dee.cpp links no
// crypto library and the digest length here is fixed.
class Sha256 {
public:
    Sha256() { reset(); }

    void reset() {
        state_[0] = 0x6a09e667u; state_[1] = 0xbb67ae85u;
        state_[2] = 0x3c6ef372u; state_[3] = 0xa54ff53au;
        state_[4] = 0x510e527fu; state_[5] = 0x9b05688cu;
        state_[6] = 0x1f83d9abu; state_[7] = 0x5be0cd19u;
        total_len_ = 0;
        buf_len_ = 0;
    }

    void update(const uint8_t* data, size_t size) {
        total_len_ += size;
        while (size > 0) {
            const size_t take = std::min(size, sizeof(buf_) - buf_len_);
            std::memcpy(buf_ + buf_len_, data, take);
            buf_len_ += take;
            data += take;
            size -= take;
            if (buf_len_ == sizeof(buf_)) {
                compress(buf_);
                buf_len_ = 0;
            }
        }
    }

    void update(const char* data, size_t size) {
        update(reinterpret_cast<const uint8_t*>(data), size);
    }

    // Finalizes; the object must be reset() before reuse.
    void finish(uint8_t out[32]) {
        const uint64_t bit_len = total_len_ * 8;
        const uint8_t one = 0x80;
        update(&one, 1);
        const uint8_t zero = 0;
        while (buf_len_ != 56) update(&zero, 1);
        uint8_t len_block[8];
        for (int i = 0; i < 8; ++i) {
            len_block[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
        }
        update(len_block, sizeof(len_block));
        for (int i = 0; i < 8; ++i) {
            out[4 * i + 0] = static_cast<uint8_t>(state_[i] >> 24);
            out[4 * i + 1] = static_cast<uint8_t>(state_[i] >> 16);
            out[4 * i + 2] = static_cast<uint8_t>(state_[i] >> 8);
            out[4 * i + 3] = static_cast<uint8_t>(state_[i]);
        }
    }

    std::string hexdigest() {
        uint8_t digest[32];
        finish(digest);
        static const char kHex[] = "0123456789abcdef";
        std::string out(64, '0');
        for (int i = 0; i < 32; ++i) {
            out[2 * i] = kHex[digest[i] >> 4];
            out[2 * i + 1] = kHex[digest[i] & 0xf];
        }
        return out;
    }

private:
    static uint32_t rotr(uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    }

    void compress(const uint8_t* block) {
        static const uint32_t kK[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
                   (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
                   static_cast<uint32_t>(block[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
                                (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
                                (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + kK[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    uint32_t state_[8];
    uint64_t total_len_;
    uint8_t buf_[64];
    size_t buf_len_;
};

std::string sha256_hex(const uint8_t* data, size_t size) {
    Sha256 hash;
    hash.update(data, size);
    return hash.hexdigest();
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

bool ExpertStore::materialize(const ExpertView& view, uint8_t* dst,
                              size_t nbytes) const {
    if (!dst || !view.contiguous_data || view.contiguous_nbytes != nbytes ||
        nbytes == 0) {
        return false;
    }
    std::memcpy(dst, view.contiguous_data, nbytes);
    return true;
}

void ExpertStore::record_source_read_batch(
        size_t requests, size_t lanes, double wall_milliseconds,
        double summed_read_milliseconds) {
    if (requests == 0) return;
    ++source_read_batches_;
    if (requests > 1 && lanes > 1) ++concurrent_source_read_batches_;
    max_source_read_queue_depth_ = std::max<uint64_t>(
        max_source_read_queue_depth_, static_cast<uint64_t>(requests));
    max_source_read_lanes_ = std::max<uint64_t>(
        max_source_read_lanes_,
        static_cast<uint64_t>(std::min(requests, std::max<size_t>(1, lanes))));
    const double wall_ms = std::max(0.0, wall_milliseconds);
    const double summed_ms = std::max(0.0, summed_read_milliseconds);
    source_read_batch_wall_ms_ += wall_ms;
    source_read_overlap_ms_ += std::max(0.0, summed_ms - wall_ms);
}

void ExpertStore::note_pread_service(uint64_t service_ns, uint64_t bytes,
                                      uint64_t short_reads,
                                      uint64_t probed_bytes,
                                      uint64_t resident_bytes) const {
    pread_service_ns_.fetch_add(service_ns, std::memory_order_relaxed);
    pread_calls_.fetch_add(1, std::memory_order_relaxed);
    pread_short_reads_.fetch_add(short_reads, std::memory_order_relaxed);
    pread_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    mincore_probed_bytes_.fetch_add(probed_bytes, std::memory_order_relaxed);
    mincore_resident_bytes_.fetch_add(resident_bytes, std::memory_order_relaxed);
}

ExpertStore::ReadTelemetry ExpertStore::read_telemetry(bool reset) const {
    ReadTelemetry out;
    out.service_ns = pread_service_ns_.load(std::memory_order_relaxed);
    out.calls = pread_calls_.load(std::memory_order_relaxed);
    out.short_reads = pread_short_reads_.load(std::memory_order_relaxed);
    out.bytes = pread_bytes_.load(std::memory_order_relaxed);
    out.probed_bytes = mincore_probed_bytes_.load(std::memory_order_relaxed);
    out.resident_bytes = mincore_resident_bytes_.load(std::memory_order_relaxed);
    if (reset) {
        pread_service_ns_.store(0, std::memory_order_relaxed);
        pread_calls_.store(0, std::memory_order_relaxed);
        pread_short_reads_.store(0, std::memory_order_relaxed);
        pread_bytes_.store(0, std::memory_order_relaxed);
        mincore_probed_bytes_.store(0, std::memory_order_relaxed);
        mincore_resident_bytes_.store(0, std::memory_order_relaxed);
    }
    return out;
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
    result.materialization_mode = materialization_mode();
    result.source_read_batches = source_read_batches_;
    result.concurrent_source_read_batches = concurrent_source_read_batches_;
    result.max_source_read_queue_depth = max_source_read_queue_depth_;
    result.max_source_read_lanes = max_source_read_lanes_;
    result.source_read_batch_wall_ms = source_read_batch_wall_ms_;
    result.source_read_overlap_ms = source_read_overlap_ms_;
    result.pread_service_ms =
        static_cast<double>(pread_service_ns_.load(std::memory_order_relaxed)) / 1e6;
    result.pread_calls = pread_calls_.load(std::memory_order_relaxed);
    result.pread_short_reads = pread_short_reads_.load(std::memory_order_relaxed);
    result.pread_bytes = pread_bytes_.load(std::memory_order_relaxed);
    result.mincore_probed_bytes = mincore_probed_bytes_.load(std::memory_order_relaxed);
    result.mincore_resident_bytes =
        mincore_resident_bytes_.load(std::memory_order_relaxed);
    if (read_milliseconds_ > 0.0) {
        result.source_read_overlap_percent =
            std::min(100.0, 100.0 * source_read_overlap_ms_ /
                                read_milliseconds_);
    }
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

void Dee4ExpertStore::unmap_segment(Segment* segment) {
#ifdef _WIN32
    if (segment->base) UnmapViewOfFile(segment->base);
    if (segment->mapping_handle) {
        CloseHandle(static_cast<HANDLE>(segment->mapping_handle));
    }
#else
    if (segment->base && segment->base != MAP_FAILED) {
        munmap(segment->base, segment->size);
    }
    if (segment->fd >= 0) ::close(segment->fd);
#endif
    segment->mapping_handle = nullptr;
    segment->fd = -1;
    segment->base = nullptr;
    segment->size = 0;
}

void Dee4ExpertStore::close() {
    for (Segment& segment : segments_) unmap_segment(&segment);
    segments_.clear();
    segmented_ = false;
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

bool Dee4ExpertStore::map_segment_file(const std::string& path,
                                       Segment* segment) {
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
    segment->mapping_handle = mapping;
    segment->base = static_cast<uint8_t*>(mapped);
    segment->size = static_cast<size_t>(file_size.QuadPart);
    segment->fd = 0;
    return true;
#else
    segment->fd = ::open(path.c_str(), O_RDONLY);
    if (segment->fd < 0) {
        last_error_ = "open failed for DEE4 data";
        return false;
    }
    struct stat st{};
    if (fstat(segment->fd, &st) != 0 || st.st_size <= 0 ||
        static_cast<unsigned long long>(st.st_size) >
            static_cast<unsigned long long>(SIZE_MAX)) {
        ::close(segment->fd);
        segment->fd = -1;
        last_error_ = "invalid DEE4 data file size";
        return false;
    }
    segment->size = static_cast<size_t>(st.st_size);
    segment->base = static_cast<uint8_t*>(
        mmap(nullptr, segment->size, PROT_READ, MAP_SHARED, segment->fd, 0));
    if (segment->base == MAP_FAILED) {
        segment->base = nullptr;
        ::close(segment->fd);
        segment->fd = -1;
        last_error_ = "mmap failed for DEE4 data";
        return false;
    }
    // DEE4 turns six scattered tensors into one sequential record per miss.
    posix_madvise(segment->base, segment->size, POSIX_MADV_RANDOM);
    return true;
#endif
}

bool Dee4ExpertStore::map_file(const std::string& path) {
    Segment segment;
    if (!map_segment_file(path, &segment)) return false;
    mapping_handle_ = segment.mapping_handle;
    fd_ = segment.fd;
    base_ = segment.base;
    size_ = segment.size;
    return true;
}

const Dee4ExpertStore::Segment* Dee4ExpertStore::find_segment(
        size_t record_index) const {
    // The segment table is sorted by first_record with contiguous coverage
    // (validated at open); find the last segment with
    // first_record <= record_index, then check its declared count.
    size_t lo = 0;
    size_t hi = segments_.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (record_index < segments_[mid].first_record) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if (lo == 0) return nullptr;
    const Segment* segment = &segments_[lo - 1];
    if (record_index - segment->first_record >= segment->record_count) {
        return nullptr;
    }
    return segment;
}

bool Dee4ExpertStore::open(const std::string& directory_or_metadata) {
    return open(directory_or_metadata, Dee4OpenOptions{});
}

bool Dee4ExpertStore::open(const std::string& directory_or_metadata,
                           const Dee4OpenOptions& options) {
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
    const bool format_known =
        json_string(root.get(), "format", &format) &&
        (format == "dee4-v2" || format == "dee4-v3-trace" ||
         format == "dee4-v4-segmented");
    if (!format_known ||
        !json_string(root.get(), "codec", &codec) ||
        codec != "deepseek-fp4-e2m1-e8m0" ||
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
    // Monolithic formats carry a single sealed data file; the segmented
    // format carries a segment table instead (data_file is null/absent).
    if (format != "dee4-v4-segmented" &&
        (!json_string(root.get(), "data_file", &data_file) ||
         data_file.empty() ||
         !json_string(root.get(), "data_sha256", &identity_) ||
         !is_sha256_hex(identity_))) {
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
    if (format == "dee4-v2") {
        if (!checked_mul(num_layers, experts_per_layer, &total_experts)) {
            last_error_ = "DEE4 total size overflow";
            close();
            return false;
        }
    } else if (format == "dee4-v4-segmented") {
        // The segment table is the byte-space map: the writer
        // (p3_builder.build_segmented) emits one segment per bucket whose
        // [first_record, first_record+record_count) ranges tile
        // [0, total_experts) contiguously and in order.  A segmented store
        // must NOT name a monolithic data file (the writer emits
        // "data_file": null, which json_min now parses to a Null-typed
        // value; an absent key or an explicit empty string is also
        // accepted for hand-edited metadata).
        const json::Value* data_file_value = root->find("data_file");
        if (data_file_value &&
            !(data_file_value->is_null() ||
              (data_file_value->is_string() && data_file_value->s.empty()))) {
            last_error_ =
                "DEE4 segmented metadata must not name a data_file";
            close();
            return false;
        }
        size_t declared_total = 0;
        const json::Value* segments_value = root->find("segments");
        if (!json_nonnegative_int(root.get(), "total_experts",
                                  &declared_total) ||
            declared_total == 0 ||
            !checked_mul(num_layers, experts_per_layer, &total_experts) ||
            total_experts != declared_total ||
            !segments_value || !segments_value->is_array() ||
            segments_value->arr.size() != num_layers) {
            last_error_ = "DEE4 segmented store metadata is invalid";
            close();
            return false;
        }
        // Optional universe bindings: when present they must be well-formed
        // (the table is what the reader actually consumes; these bind the
        // store to the Phase-3 manifest universe).
        for (const char* key : {"universe_sha256", "manifest_sha256"}) {
            const json::Value* binding = root->find(key);
            if (binding &&
                (!binding->is_string() || !is_sha256_hex(binding->s))) {
                last_error_ = "DEE4 segmented universe binding is invalid";
                close();
                return false;
            }
        }
        segments_.reserve(num_layers);
        size_t next_first = 0;
        for (size_t i = 0; i < segments_value->arr.size(); ++i) {
            const json::Value* item = segments_value->arr[i].get();
            std::string file;
            std::string sha;
            size_t bucket = 0;
            size_t first_record = 0;
            size_t record_count = 0;
            size_t declared_bytes = 0;
            if (!item || !item->is_object() ||
                !json_string(item, "file", &file) || file.empty() ||
                !json_nonnegative_int(item, "bucket", &bucket) ||
                bucket != i ||
                !json_nonnegative_int(item, "first_record", &first_record) ||
                first_record != next_first ||
                !json_nonnegative_int(item, "record_count", &record_count) ||
                record_count != experts_per_layer ||
                !json_nonnegative_int(item, "bytes", &declared_bytes) ||
                !json_string(item, "sha256", &sha) ||
                !is_sha256_hex(sha)) {
                last_error_ = "DEE4 segment table is invalid";
                close();
                return false;
            }
            // `domain` ("main"/"mtp") is informational for tooling only;
            // if present it must at least be a string.
            const json::Value* domain = item->find("domain");
            if (domain && !domain->is_string()) {
                last_error_ = "DEE4 segment table is invalid";
                close();
                return false;
            }
            size_t expected_bytes = 0;
            if (!checked_mul(record_count, record_bytes_, &expected_bytes) ||
                expected_bytes != declared_bytes) {
                last_error_ = "DEE4 segment byte count is inconsistent";
                close();
                return false;
            }
            // `file` is a store-relative path.  Refuse absolute paths and
            // parent escapes so the mmap set can never leave the store dir.
            const fs::path relative(file);
            if (relative.is_absolute() ||
                file.find("..") != std::string::npos) {
                last_error_ = "DEE4 segment path escapes the store";
                close();
                return false;
            }
            for (char& c : sha) {
                c = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
            }
            Segment segment;
            segment.first_record = first_record;
            segment.record_count = record_count;
            segment.file = file;
            segment.sha256 = sha;
            segments_.push_back(std::move(segment));
            next_first += record_count;
        }
        if (next_first != declared_total) {
            last_error_ = "DEE4 segments do not cover the declared universe";
            close();
            return false;
        }
        // The segmented format has no whole-file seal: derive the store
        // identity as sha256 over the ordered concatenation of the segment
        // table's sha256 hex strings (the declared content seals).
        {
            Sha256 table_hash;
            for (const Segment& segment : segments_) {
                table_hash.update(segment.sha256.data(),
                                  segment.sha256.size());
            }
            identity_ = table_hash.hexdigest();
        }
        segmented_ = true;
        backend_ = "dee4_segmented";
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
    if (segmented_) {
        // Map every segment and enforce the exact declared size.  Missing,
        // unreadable, or mis-sized segment files fail closed here.
        for (Segment& segment : segments_) {
            const fs::path segment_path =
                metadata_path.parent_path() / segment.file;
            if (!map_segment_file(segment_path.string(), &segment)) {
                close();
                return false;
            }
            size_t expected_segment_bytes = 0;
            if (!checked_mul(segment.record_count, record_bytes_,
                             &expected_segment_bytes) ||
                segment.size != expected_segment_bytes) {
                std::ostringstream message;
                message << "DEE4 segment " << segment.file << " size "
                        << segment.size << " does not match expected "
                        << expected_segment_bytes;
                last_error_ = message.str();
                close();
                return false;
            }
        }
        if (options.verify_segment_hashes) {
            // The per-segment sha256 is the segmented format's content seal:
            // a segment whose bytes do not match its declared seal must
            // never be served.  One-time O(total bytes) pass over the
            // mapped views.  Progress lines make the (potentially long)
            // seal visible to session drivers instead of a silent block.
            const auto seal_begin = std::chrono::steady_clock::now();
            size_t seal_index = 0;
            for (const Segment& segment : segments_) {
                const auto seg_begin = std::chrono::steady_clock::now();
                if (sha256_hex(segment.base, segment.size) !=
                    segment.sha256) {
                    last_error_ = "DEE4 segment " + segment.file +
                                  " sha256 mismatch";
                    close();
                    return false;
                }
                const double seg_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - seg_begin).count();
                const double total_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - seal_begin).count();
                ++seal_index;
                std::fprintf(stderr,
                    "[dee4] seal %zu/%zu %s %.2f GiB %.0f MiB/s "
                    "seg=%.1fs total=%.1fs\n",
                    seal_index, segments_.size(), segment.file.c_str(),
                    static_cast<double>(segment.size) / (1 << 30),
                    segment.size / (1024.0 * 1024.0) /
                        (seg_s > 0.0 ? seg_s : 1e-9),
                    seg_s, total_s);
            }
        }
    } else {
        size_t expected_size = 0;
        if (!checked_mul(total_experts, record_bytes_, &expected_size)) {
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
    }
    start_layer_ = static_cast<int>(start_layer);
    num_layers_ = static_cast<int>(num_layers);
    experts_per_layer_ = static_cast<int>(experts_per_layer);
    stored_records_ = total_experts;
    return true;
}

bool Dee4ExpertStore::get(int layer, int expert, ExpertView* out) {
    const bool mapped =
        segmented_ ? !segments_.empty() : base_ != nullptr;
    const bool in_range = out && mapped && layer >= start_layer_ &&
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
    const uint8_t* record = nullptr;
    if (segmented_) {
        // record_index -> segment via the declared table, then a fixed
        // in-segment stride.  The table is validated contiguous at open, so
        // a miss here can only mean corruption after open — fail closed.
        const Segment* segment = find_segment(record_index);
        if (segment) {
            const size_t record_offset =
                (record_index - segment->first_record) * record_bytes_;
            if (record_offset <= segment->size &&
                record_bytes_ <= segment->size - record_offset) {
                record = segment->base + record_offset;
            }
        }
        if (!record) {
            record_lookup(false);
            return false;
        }
    } else {
        const size_t record_offset = record_index * record_bytes_;
        if (record_offset > size_ || record_bytes_ > size_ - record_offset) {
            record_lookup(false);
            return false;
        }
        record = base_ + record_offset;
    }
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

bool Dee4ExpertStore::materialize(const ExpertView& view, uint8_t* dst,
                                  size_t nbytes) const {
    if (!dst || nbytes == 0 || nbytes != record_bytes_ ||
        view.contiguous_nbytes != record_bytes_ ||
        view.record_index >= stored_records_) {
        return false;
    }
    // Resolve the backing store the same way get() did: one file for the
    // monolithic formats, the owning segment for dee4-v4-segmented.
    [[maybe_unused]] int fd = fd_;
    const uint8_t* mapped = base_;
    size_t mapped_size = size_;
    size_t offset = static_cast<size_t>(view.record_index) * record_bytes_;
    if (segmented_) {
        const Segment* segment = find_segment(view.record_index);
        if (!segment) return false;
        fd = segment->fd;
        mapped = segment->base;
        mapped_size = segment->size;
        offset = (view.record_index - segment->first_record) * record_bytes_;
    }
    if (!mapped || offset > mapped_size || nbytes > mapped_size - offset ||
        view.contiguous_data != mapped + offset) {
        return false;
    }
#ifdef _WIN32
    std::memcpy(dst, view.contiguous_data, nbytes);
    return true;
#else
    if (fd < 0 || offset > static_cast<size_t>(
            std::numeric_limits<off_t>::max())) {
        return false;
    }
    // Profiling-only page-cache residency probe (mincore): two cheap syscalls
    // per 12.75 MiB record (~microseconds vs ~90 ms reads). Never affects IO.
    uint64_t probed_bytes = 0;
    uint64_t resident_bytes = 0;
#ifdef __linux__
    {
        const size_t kPage = 4096;
        const size_t pages = (nbytes + kPage - 1) / kPage;
        std::vector<unsigned char> vec(pages, 0);
        if (::mincore(const_cast<uint8_t*>(mapped) + offset, nbytes,
                      vec.data()) == 0) {
            probed_bytes = nbytes;
            size_t resident_pages = 0;
            for (size_t i = 0; i < pages; ++i) resident_pages += (vec[i] & 1u);
            resident_bytes = resident_pages * kPage;
            if (resident_bytes > nbytes) resident_bytes = nbytes;
        }
    }
#endif
    const auto read_begin = std::chrono::steady_clock::now();
    size_t copied = 0;
    uint64_t short_reads = 0;
    while (copied < nbytes) {
        const size_t remaining = nbytes - copied;
        if (offset + copied > static_cast<size_t>(
                std::numeric_limits<off_t>::max())) {
            return false;
        }
        const ssize_t count = ::pread(
            fd, dst + copied, remaining,
            static_cast<off_t>(offset + copied));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        if (static_cast<size_t>(count) < remaining) ++short_reads;
        copied += static_cast<size_t>(count);
    }
    const uint64_t service_ns = static_cast<uint64_t>(
        std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - read_begin).count());
    note_pread_service(service_ns, copied, short_reads, probed_bytes,
                       resident_bytes);
    return true;
#endif
}

}  // namespace dee
