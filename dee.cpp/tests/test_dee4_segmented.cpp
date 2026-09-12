// tests/test_dee4_segmented.cpp
//
// Host-only validation for the dee4-v4-segmented reader path added to
// Dee4ExpertStore: a store directory of segments/experts-bucket-NN.dee4
// files plus a segment table in metadata.json (exactly the shape
// tools/phase3/p3_builder.py:build_segmented emits).
//
// Two fixtures, both synthetic:
//
//   1. a small dense store (4 buckets x 4 experts x 70-byte records) with
//      real content and real per-segment sha256 seals — arbitrary
//      (bucket, expert) resolution including pairs straddling segment
//      boundaries, byte-exact round-trip vs. the synthetic writer, the
//      materialize() path, and every fail-closed case: missing segment,
//      mis-sized segment, mis-hashed segment, malformed segment table,
//      out-of-range (bucket, expert);
//
//   2. the true-geometry universe (46 buckets x 256 experts x
//      13,369,344-byte records = 46 sparse 3,422,552,064-byte segment
//      files, ~146.6 GiB logical, only seeded marker extents occupy disk),
//      mirroring tools/phase3/test_p3_dee4_full_geometry.cpp — arbitrary
//      resolution over buckets 0..45 (incl. the mtp draft buckets 43..45),
//      fail-closed at 46/-1/256/-1.  The seal pass is disabled for this
//      fixture (hashing 146 GiB at open is impractical; the segment table
//      still must be structurally exact).
//
// No CUDA or real checkpoint is required.

#include "dee/expert_store.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

int g_failures = 0;
int g_skips = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++g_failures;
    }
}

void skip(const std::string& message) {
    std::fprintf(stderr, "SKIP: %s\n", message.c_str());
    ++g_skips;
}

// ---------------------------------------------------------------------------
// SHA-256 (fixture-side copy used to seal synthetic segments; the store's
// open() verifies these digests with its own implementation, so agreement is
// itself tested.  Known-answer vectors are checked first.)
// ---------------------------------------------------------------------------

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
    static std::string hex(const uint8_t* data, size_t size) {
        Sha256 h;
        h.update(data, size);
        return h.hexdigest();
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

void test_sha256_known_answers() {
    check(Sha256::hex(reinterpret_cast<const uint8_t*>(""), 0) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "sha256 KAT: empty string");
    const std::string abc = "abc";
    check(Sha256::hex(reinterpret_cast<const uint8_t*>(abc.data()), 3) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "sha256 KAT: \"abc\"");
    // 1,000,000 x 'a' multi-block vector (crosses the 64-byte buffer path).
    std::vector<uint8_t> big(1000000, 'a');
    check(Sha256::hex(big.data(), big.size()) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "sha256 KAT: million 'a'");
}

// ---------------------------------------------------------------------------
// Small dense segmented store (4 x 4 x 70 B), writer-faithful layout.
// ---------------------------------------------------------------------------

constexpr int kSmallBuckets = 4;
constexpr int kSmallExperts = 4;
constexpr size_t kSmallRecordBytes = 70;
// Record component geometry (must satisfy out*in == nbytes, like the real
// tables): w1|w3|w2 weights then w1|w3|w2 scales.
constexpr size_t kSmallWeightOffsets[3] = {0, 16, 32};
constexpr size_t kSmallWeightNbytes[3] = {16, 16, 24};
constexpr int kSmallWeightOut[3] = {4, 4, 4};
constexpr int kSmallWeightIn[3] = {4, 4, 6};
constexpr size_t kSmallScaleOffsets[3] = {56, 60, 64};
constexpr size_t kSmallScaleNbytes[3] = {4, 4, 6};
constexpr int kSmallScaleOut[3] = {2, 2, 2};
constexpr int kSmallScaleIn[3] = {2, 2, 3};

// Deterministic record content, identical contract for writer and checker.
uint8_t small_byte(size_t record_index, size_t j) {
    return static_cast<uint8_t>((record_index * 17 + j) & 0xff);
}

std::vector<uint8_t> small_record(size_t record_index) {
    std::vector<uint8_t> record(kSmallRecordBytes);
    for (size_t j = 0; j < record.size(); ++j) {
        record[j] = small_byte(record_index, j);
    }
    return record;
}

std::vector<uint8_t> small_segment(int bucket) {
    std::vector<uint8_t> segment;
    segment.reserve(kSmallExperts * kSmallRecordBytes);
    for (int expert = 0; expert < kSmallExperts; ++expert) {
        const auto record = small_record(
            static_cast<size_t>(bucket) * kSmallExperts + expert);
        segment.insert(segment.end(), record.begin(), record.end());
    }
    return segment;
}

std::filesystem::path make_test_dir(const char* tag) {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        ("dee4-segmented-" + std::string(tag) + "-" +
         std::to_string(suffix));
    std::filesystem::create_directories(path / "segments");
    return path;
}

std::string segment_name(int bucket) {
    char name[64];
    std::snprintf(name, sizeof(name), "experts-bucket-%02d.dee4", bucket);
    return name;
}

// metadata.json in exactly the shape p3_builder.build_segmented emits
// (including "data_file": null and the segment table).  `sha[b]` is the
// declared seal of segment b.
std::string small_metadata_json(const std::array<std::string, kSmallBuckets>& sha) {
    std::ostringstream meta;
    meta << "{\n"
         << "  \"format\": \"dee4-v4-segmented\",\n"
         << "  \"codec\": \"deepseek-fp4-e2m1-e8m0\",\n"
         << "  \"source_repository\": \"synthetic\",\n"
         << "  \"source_revision\": \"test\",\n"
         << "  \"start_layer\": 0,\n"
         << "  \"num_layers\": " << kSmallBuckets << ",\n"
         << "  \"experts_per_layer\": " << kSmallExperts << ",\n"
         << "  \"total_experts\": " << kSmallBuckets * kSmallExperts << ",\n"
         << "  \"record_bytes\": " << kSmallRecordBytes << ",\n"
         << "  \"group_size\": 32,\n"
         << "  \"weight_offsets\": [0, 16, 32],\n"
         << "  \"weight_nbytes\": [16, 16, 24],\n"
         << "  \"weight_out\": [4, 4, 4],\n"
         << "  \"weight_stored_in\": [4, 4, 6],\n"
         << "  \"scale_offsets\": [56, 60, 64],\n"
         << "  \"scale_nbytes\": [4, 4, 6],\n"
         << "  \"scale_out\": [2, 2, 2],\n"
         << "  \"scale_in\": [2, 2, 3],\n"
         << "  \"universe_sha256\": \"" << std::string(64, 'a') << "\",\n"
         << "  \"manifest_sha256\": \"" << std::string(64, 'b') << "\",\n"
         << "  \"mtp_bucket_offset\": 43,\n"
         << "  \"segments\": [\n";
    for (int b = 0; b < kSmallBuckets; ++b) {
        meta << "    {\"file\": \"segments/" << segment_name(b) << "\",\n"
             << "     \"bucket\": " << b << ",\n"
             << "     \"domain\": \"main\",\n"
             << "     \"first_record\": " << b * kSmallExperts << ",\n"
             << "     \"record_count\": " << kSmallExperts << ",\n"
             << "     \"bytes\": " << kSmallExperts * kSmallRecordBytes << ",\n"
             << "     \"sha256\": \"" << sha[b] << "\"}"
             << (b + 1 < kSmallBuckets ? ",\n" : "\n");
    }
    meta << "  ],\n"
         // The writer literally emits data_file: null — the reader must
         // tolerate a bare JSON null here.
         << "  \"data_file\": null,\n"
         << "  \"integrity_file\": \"integrity.jsonl\"\n"
         << "}\n";
    return meta.str();
}

// Write a conforming small segmented store; returns the segment seals.
std::array<std::string, kSmallBuckets> write_small_store(
        const std::filesystem::path& dir) {
    std::array<std::string, kSmallBuckets> sha{};
    std::ostringstream integrity;
    for (int b = 0; b < kSmallBuckets; ++b) {
        const auto segment = small_segment(b);
        sha[b] = Sha256::hex(segment.data(), segment.size());
        std::ofstream stream(dir / "segments" / segment_name(b),
                             std::ios::binary);
        stream.write(reinterpret_cast<const char*>(segment.data()),
                     static_cast<std::streamsize>(segment.size()));
        for (int e = 0; e < kSmallExperts; ++e) {
            const size_t index =
                static_cast<size_t>(b) * kSmallExperts + e;
            const auto record = small_record(index);
            integrity << "{\"bucket\":" << b << ",\"layer\":" << b
                      << ",\"domain\":\"main\",\"expert\":" << e
                      << ",\"record_index\":" << index
                      << ",\"record_offset\":" << index * kSmallRecordBytes
                      << ",\"record_bytes\":" << kSmallRecordBytes
                      << ",\"record_sha256\":\""
                      << Sha256::hex(record.data(), record.size())
                      << "\",\"component_sha256\":{},"
                      << "\"source_shards\":[\"synthetic\"]}\n";
        }
    }
    std::ofstream(dir / "metadata.json", std::ios::binary)
        << small_metadata_json(sha);
    std::ofstream(dir / "integrity.jsonl", std::ios::binary)
        << integrity.str();
    return sha;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(stream)),
                       std::istreambuf_iterator<char>());
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
}

// Expected derived store identity: sha256 over the concatenated segment
// sha256 hex strings, in table order.
std::string expected_identity(
        const std::array<std::string, kSmallBuckets>& sha) {
    Sha256 hash;
    for (const auto& s : sha) hash.update(s.data(), s.size());
    return hash.hexdigest();
}

void check_view(const dee::ExpertView& view, size_t record_index) {
    check(view.ok(), "segmented view is complete");
    check(view.record_index == record_index,
          "segmented record index is dense arithmetic");
    check(view.contiguous_data != nullptr &&
              view.contiguous_nbytes == kSmallRecordBytes,
          "segmented view exposes one contiguous record");
    check(view.codec == dee::ExpertCodec::DeepSeekFp4E2m1E8m0,
          "segmented codec identity");
    // Round-trip: the bytes through the view equal what the synthetic
    // writer produced for this record, component by component.
    for (size_t j = 0; j < kSmallRecordBytes; ++j) {
        if (view.contiguous_data[j] != small_byte(record_index, j)) {
            check(false, "record bytes round-trip the writer's content");
            return;
        }
    }
    for (size_t p = 0; p < 3; ++p) {
        check(view.weights[p].data ==
                      view.contiguous_data + kSmallWeightOffsets[p] &&
                  view.weights[p].nbytes == kSmallWeightNbytes[p] &&
                  view.weights[p].dtype == dee::DType::I8 &&
                  view.weights[p].shape[0] == kSmallWeightOut[p] &&
                  view.weights[p].shape[1] == kSmallWeightIn[p],
              "segmented weight component view resolves");
        check(view.scales[p].data ==
                      view.contiguous_data + kSmallScaleOffsets[p] &&
                  view.scales[p].nbytes == kSmallScaleNbytes[p] &&
                  view.scales[p].dtype == dee::DType::F8 &&
                  view.scales[p].shape[0] == kSmallScaleOut[p] &&
                  view.scales[p].shape[1] == kSmallScaleIn[p],
              "segmented scale component view resolves");
    }
}

void test_small_segmented_store() {
    const auto dir = make_test_dir("small");
    const auto sha = write_small_store(dir);

    dee::Dee4ExpertStore store;
    // Default open: segment seals are verified.
    check(store.open(dir.string()), "segmented store opens with seal verify");
    check(store.segmented() && !store.trace_indexed(),
          "segmented store reports segmented layout");
    check(store.segment_count() == kSmallBuckets, "four segments mapped");
    check(store.start_layer() == 0 && store.num_layers() == kSmallBuckets &&
              store.experts_per_layer() == kSmallExperts &&
              store.record_bytes() == kSmallRecordBytes,
          "segmented geometry loaded");
    check(store.stored_records() == kSmallBuckets * kSmallExperts,
          "segmented stored_records covers the universe");
    check(store.integrity_identity() == expected_identity(sha),
          "segmented identity is the segment-table digest");
    check(std::string(store.backend_name()) == "dee4_segmented",
          "segmented backend has distinct telemetry identity");

    // Every (bucket, expert) resolves — including the last/first records of
    // adjacent segments, i.e. across the segment boundary — with byte-exact
    // content.
    for (int layer = 0; layer < kSmallBuckets; ++layer) {
        for (int expert = 0; expert < kSmallExperts; ++expert) {
            dee::ExpertView view;
            check(store.get(layer, expert, &view),
                  "segmented pair resolves");
            if (view.ok()) {
                check_view(view, static_cast<size_t>(layer) * kSmallExperts +
                                     expert);
            }
        }
    }

    // Opening through the metadata.json path directly also works.
    dee::Dee4ExpertStore by_file;
    check(by_file.open((dir / "metadata.json").string()),
          "segmented store opens via metadata.json path");
    dee::ExpertView view;
    check(by_file.get(2, 3, &view) &&
              view.contiguous_data[0] == small_byte(11, 0),
          "metadata-path open resolves records");
    by_file.close();

    // Fail closed outside the declared geometry.
    check(!store.get(kSmallBuckets, 0, &view) && !store.get(-1, 0, &view) &&
              !store.get(0, kSmallExperts, &view) &&
              !store.get(0, -1, &view),
          "out-of-universe segmented lookups fail closed");

    // materialize() copies the exact record and refuses forged views.
    std::vector<uint8_t> dst(kSmallRecordBytes, 0xee);
    check(store.get(3, 2, &view), "boundary record resolves");
    check(store.materialize(view, dst.data(), dst.size()),
          "segmented record materializes");
    const auto expected = small_record(14);
    check(dst == expected, "materialized bytes equal the writer's record");
    dee::ExpertView forged = view;
    forged.record_index = 3;  // same size, different record -> pointer check
    check(!store.materialize(forged, dst.data(), dst.size()),
          "forged record_index fails materialize identity check");

    check(store.get_layout_reference(1, &view) && view.record_index == 4,
          "layout reference resolves through the segment table");

    const dee::ExpertStoreStats stats = store.stats();
    check(stats.backend == "dee4_segmented",
          "segmented backend telemetry identity");
    check(stats.lookup_failures == 4, "segmented failures recorded");

    store.close();
    std::filesystem::remove_all(dir);
}

void test_missing_segment_fails_closed() {
    const auto dir = make_test_dir("missing");
    write_small_store(dir);
    std::filesystem::remove(dir / "segments" / segment_name(2));
    dee::Dee4ExpertStore store;
    check(!store.open(dir.string()),
          "missing segment file fails open");
    dee::Dee4OpenOptions no_verify;
    no_verify.verify_segment_hashes = false;
    check(!store.open(dir.string(), no_verify),
          "missing segment fails even without hash verify");
    std::filesystem::remove_all(dir);
}

void test_mis_sized_segment_fails_closed() {
    const auto dir = make_test_dir("size");
    write_small_store(dir);
    // Truncate the middle segment by one byte.
    const auto path = dir / "segments" / segment_name(1);
    std::ofstream(path, std::ios::binary | std::ios::trunc)
        .write("x", 1);
    dee::Dee4ExpertStore store;
    check(!store.open(dir.string()),
          "mis-sized segment fails open");
    check(store.last_error().find("does not match expected") !=
              std::string::npos,
          "segment size failure is attributable");
    dee::Dee4OpenOptions no_verify;
    no_verify.verify_segment_hashes = false;
    check(!store.open(dir.string(), no_verify),
          "mis-sized segment fails even without hash verify");
    std::filesystem::remove_all(dir);
}

void test_mis_hashed_segment_fails_closed() {
    const auto dir = make_test_dir("hash");
    write_small_store(dir);
    // Flip one byte inside a segment without changing its size: structure
    // stays valid, only the content seal detects it.
    {
        std::fstream stream(dir / "segments" / segment_name(2),
                            std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(5);
        stream.put('\xff');
    }
    dee::Dee4ExpertStore store;
    check(!store.open(dir.string()),
          "mis-hashed segment fails open under seal verification");
    check(store.last_error().find("sha256 mismatch") != std::string::npos,
          "seal failure is attributable");
    // Documented escape hatch: trusted mirrors may skip the hash pass; the
    // reader then serves the (here corrupted) bytes exactly as the v2 reader
    // serves unsealed content.
    dee::Dee4OpenOptions no_verify;
    no_verify.verify_segment_hashes = false;
    check(store.open(dir.string(), no_verify),
          "verify_segment_hashes=false opens an unsealed segment");
    store.close();
    std::filesystem::remove_all(dir);
}

void test_segment_table_validation() {
    const auto dir = make_test_dir("table");
    const auto sha = write_small_store(dir);
    const std::string good = read_text(dir / "metadata.json");

    dee::Dee4ExpertStore store;
    dee::Dee4OpenOptions no_verify;
    no_verify.verify_segment_hashes = false;

    auto expect_reject = [&](const std::string& meta, const char* what) {
        write_text(dir / "metadata.json", meta);
        check(!store.open(dir.string(), no_verify), what);
    };

    // Segment table must tile [0, total_experts) contiguously: a gap in
    // first_record is rejected.
    std::string gap = good;
    gap.replace(gap.find("\"first_record\": 8"), 17, "\"first_record\": 9");
    expect_reject(gap, "non-contiguous first_record rejected");
    // Table shorter than num_layers is rejected.
    const auto tail = good.rfind("    {\"file\": \"segments/" +
                                 std::string("experts-bucket-03.dee4"));
    std::string short_table = good.substr(0, tail) + "  ],\n" +
        "  \"data_file\": null,\n  \"integrity_file\": \"integrity.jsonl\"\n}\n";
    expect_reject(short_table, "segment table shorter than num_layers");
    // record_count must equal experts_per_layer (writer contract).
    std::string bad_count = good;
    const auto rc = bad_count.find("\"record_count\": 4");
    bad_count.replace(rc, 17, "\"record_count\": 3");
    expect_reject(bad_count, "record_count != experts_per_layer rejected");
    // A malformed seal is rejected even when content hashing is skipped.
    std::string bad_sha = good;
    const auto sh = bad_sha.find("\"sha256\": \"");
    bad_sha.replace(sh + 11, 64, std::string(64, 'z'));
    expect_reject(bad_sha, "non-hex segment sha256 rejected");
    // A segmented store must not name a monolithic data file.
    std::string with_data = good;
    with_data.replace(with_data.find("\"data_file\": null"), 17,
                      "\"data_file\": \"experts.dee4\"");
    expect_reject(with_data, "segmented metadata naming a data_file");
    // Regression gate (json_min native null, JSON_MIN_NULL.md): the writer's
    // literal "data_file": null parses to a Null-typed Value, not "" — the
    // segmented gate must accept is_null() or every conforming store is
    // rejected.  `good` carries the literal null; it must open.
    write_text(dir / "metadata.json", good);
    check(store.open(dir.string(), no_verify),
          "explicit \"data_file\": null opens under native null parsing");
    store.close();
    // bucket ids must match their table position.
    std::string bad_bucket = good;
    bad_bucket.replace(bad_bucket.find("\"bucket\": 2"), 11, "\"bucket\": 9");
    expect_reject(bad_bucket, "out-of-order segment bucket rejected");
    // total_experts must equal num_layers * experts_per_layer.
    std::string bad_total = good;
    bad_total.replace(bad_total.find("\"total_experts\": 16"), 18,
                      "\"total_experts\": 15");
    expect_reject(bad_total, "total_experts inconsistent with geometry");

    // Restore the conforming metadata: the store opens again.
    write_text(dir / "metadata.json", good);
    check(store.open(dir.string(), no_verify),
          "conforming segmented store reopens");
    store.close();
    std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Full-geometry sparse fixture: 46 segments x 3,422,552,064 B (sparse).
// ---------------------------------------------------------------------------

constexpr uint64_t kRealRecordBytes = 13369344;
constexpr int kRealBuckets = 46;
constexpr int kRealExperts = 256;
constexpr uint64_t kRealSegmentBytes =
    kRealRecordBytes * kRealExperts;  // 3,422,552,064

// Marker convention identical to tools/phase3/p3_sparse_store.py.
uint8_t marker_byte(size_t record_index, size_t j) {
    return static_cast<uint8_t>(((record_index & 0xff) + j) & 0xff);
}

bool make_sparse_file(const std::filesystem::path& path, uint64_t size) {
#ifdef _WIN32
    const HANDLE file = CreateFileW(
        path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    DWORD ignored = 0;
    const bool ok =
        DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0,
                        &ignored, nullptr) != 0;
    LARGE_INTEGER at;
    at.QuadPart = static_cast<LONGLONG>(size);
    const bool sized = ok && SetFilePointerEx(file, at, nullptr, FILE_BEGIN) &&
        SetEndOfFile(file);
    CloseHandle(file);
    if (!sized) {
        std::filesystem::remove(path);
        return false;
    }
    return std::filesystem::file_size(path) == size;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    const bool ok = ::ftruncate(fd, static_cast<off_t>(size)) == 0;
    ::close(fd);
    return ok && std::filesystem::file_size(path) == size;
#endif
}

void write_at(const std::filesystem::path& path, uint64_t offset,
              const uint8_t* data, size_t size) {
    std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
    stream.seekp(static_cast<std::streamoff>(offset));
    stream.write(reinterpret_cast<const char*>(data),
                 static_cast<std::streamsize>(size));
}

std::string real_metadata_json() {
    std::ostringstream meta;
    meta << "{\n"
         << "  \"format\": \"dee4-v4-segmented\",\n"
         << "  \"codec\": \"deepseek-fp4-e2m1-e8m0\",\n"
         << "  \"source_repository\": \"deepseek-ai/DeepSeek-V4-Flash-0731\",\n"
         << "  \"source_revision\": "
            "\"9e165c30e2704aec5d9d593cce3eebd58bbef1cb\",\n"
         << "  \"start_layer\": 0,\n"
         << "  \"num_layers\": 46,\n"
         << "  \"experts_per_layer\": 256,\n"
         << "  \"total_experts\": 11776,\n"
         << "  \"record_bytes\": 13369344,\n"
         << "  \"group_size\": 32,\n"
         << "  \"weight_offsets\": [0, 4194304, 8388608],\n"
         << "  \"weight_nbytes\": [4194304, 4194304, 4194304],\n"
         << "  \"weight_out\": [2048, 2048, 4096],\n"
         << "  \"weight_stored_in\": [2048, 2048, 1024],\n"
         << "  \"scale_offsets\": [12582912, 12845056, 13107200],\n"
         << "  \"scale_nbytes\": [262144, 262144, 262144],\n"
         << "  \"scale_out\": [2048, 2048, 4096],\n"
         << "  \"scale_in\": [128, 128, 64],\n"
         << "  \"universe_sha256\": \"" << std::string(64, 'a') << "\",\n"
         << "  \"manifest_sha256\": \"" << std::string(64, 'b') << "\",\n"
         << "  \"mtp_bucket_offset\": 43,\n"
         << "  \"segments\": [\n";
    for (int b = 0; b < kRealBuckets; ++b) {
        meta << "    {\"file\": \"segments/" << segment_name(b) << "\",\n"
             << "     \"bucket\": " << b << ",\n"
             << "     \"domain\": \"" << (b < 43 ? "main" : "mtp") << "\",\n"
             << "     \"first_record\": " << b * kRealExperts << ",\n"
             << "     \"record_count\": " << kRealExperts << ",\n"
             << "     \"bytes\": " << kRealSegmentBytes << ",\n"
             // Seal pass is disabled for this fixture; the field still must
             // be well-formed hex.
             << "     \"sha256\": \"" << std::string(64, '0') << "\"}"
             << (b + 1 < kRealBuckets ? ",\n" : "\n");
    }
    meta << "  ],\n"
         << "  \"data_file\": null,\n"
         << "  \"integrity_file\": \"integrity.jsonl\"\n"
         << "}\n";
    return meta.str();
}

void test_full_geometry_sparse_segments() {
    const auto dir = make_test_dir("full");
    bool sparse_ok = true;
    for (int b = 0; b < kRealBuckets && sparse_ok; ++b) {
        sparse_ok = make_sparse_file(dir / "segments" / segment_name(b),
                                     kRealSegmentBytes);
    }
    if (!sparse_ok) {
        skip("sparse segment files unavailable on this filesystem; "
             "full-geometry fixture skipped");
        std::filesystem::remove_all(dir);
        return;
    }
    // Seed the same sampled records the Python sparse fixture seeds.
    const int seeded[][2] = {{0, 0}, {20, 128}, {42, 255}, {43, 0},
                             {45, 255}};
    for (const auto& pair : seeded) {
        const size_t index =
            static_cast<size_t>(pair[0]) * kRealExperts + pair[1];
        uint8_t marker[16];
        for (size_t j = 0; j < 16; ++j) marker[j] = marker_byte(index, j);
        const auto seg = dir / "segments" / segment_name(pair[0]);
        write_at(seg, static_cast<uint64_t>(pair[1]) * kRealRecordBytes,
                 marker, 16);
        write_at(seg,
                 static_cast<uint64_t>(pair[1] + 1) * kRealRecordBytes - 16,
                 marker, 16);
    }
    write_text(dir / "metadata.json", real_metadata_json());

    dee::Dee4OpenOptions no_verify;
    no_verify.verify_segment_hashes = false;
    dee::Dee4ExpertStore store;
    check(store.open(dir.string(), no_verify),
          "full-geometry segmented store opens");
    if (g_failures) {
        std::filesystem::remove_all(dir);
        return;
    }
    check(store.segmented() && store.segment_count() == 46,
          "46 segments mapped");
    check(store.num_layers() == 46 && store.experts_per_layer() == 256 &&
              store.record_bytes() == kRealRecordBytes,
          "real universe geometry");
    check(store.stored_records() == 11776, "11,776 segmented records");

    const int layers[] = {0, 1, 2, 3, 17, 20, 21, 40, 41, 42, 43, 44, 45};
    const int experts[] = {0, 1, 5, 127, 128, 200, 254, 255};
    size_t resolved = 0;
    for (int layer : layers) {
        for (int expert : experts) {
            dee::ExpertView view;
            check(store.get(layer, expert, &view),
                  "full-geometry segmented pair resolves");
            if (!view.ok()) continue;
            check(view.record_index ==
                      static_cast<size_t>(layer) * 256 + expert,
                  "dense record index across segments");
            check(view.contiguous_nbytes == kRealRecordBytes,
                  "full record exposed");
            check(view.weights[0].nbytes == 4194304 &&
                      view.weights[0].shape[0] == 2048 &&
                      view.weights[2].shape[0] == 4096 &&
                      view.weights[2].shape[1] == 1024,
                  "real component geometry");
            check(view.weights[0].data == view.contiguous_data &&
                      view.weights[1].data ==
                          view.contiguous_data + 4194304 &&
                      view.weights[2].data ==
                          view.contiguous_data + 8388608 &&
                      view.scales[0].data ==
                          view.contiguous_data + 12582912 &&
                      view.scales[2].data ==
                          view.contiguous_data + 13107200,
                  "canonical record layout inside the segment");
            ++resolved;
        }
    }
    check(resolved == sizeof(layers) / sizeof(int) *
                          (sizeof(experts) / sizeof(int)),
          "all sampled segmented pairs resolved");

    dee::ExpertView view;
    check(!store.get(46, 0, &view) && !store.get(-1, 0, &view) &&
              !store.get(0, 256, &view) && !store.get(0, -1, &view),
          "out-of-universe segmented lookups fail closed");

    for (const auto& pair : seeded) {
        dee::ExpertView v;
        check(store.get(pair[0], pair[1], &v), "seeded segment record");
        const size_t index =
            static_cast<size_t>(pair[0]) * kRealExperts + pair[1];
        check(v.contiguous_data[0] == marker_byte(index, 0) &&
                  v.contiguous_data[15] == marker_byte(index, 15) &&
                  v.contiguous_data[kRealRecordBytes - 1] ==
                      marker_byte(index, 15),
                  "seeded marker bytes readable through segment mmap");
    }

    std::vector<uint8_t> dst(kRealRecordBytes, 0xee);
    check(store.get(45, 255, &view), "edge segment record resolves");
    check(store.materialize(view, dst.data(), dst.size()),
          "edge segment record materializes");
    const size_t edge = 45 * 256 + 255;
    check(dst[0] == marker_byte(edge, 0) &&
              dst[kRealRecordBytes - 1] == marker_byte(edge, 15),
          "materialized segment bytes carry seeded content");

    store.close();
    std::filesystem::remove_all(dir);
}

}  // namespace

int main() {
    test_sha256_known_answers();
    test_small_segmented_store();
    test_missing_segment_fails_closed();
    test_mis_sized_segment_fails_closed();
    test_mis_hashed_segment_fails_closed();
    test_segment_table_validation();
    test_full_geometry_sparse_segments();
    if (g_failures == 0) {
        std::printf("ALL PASS (%d skip%s)\n", g_skips,
                    g_skips == 1 ? "" : "s");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
