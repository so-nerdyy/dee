// tests/test_expert_store.cpp
//
// Host-only validation for the DEE4 v2 arithmetic lookup and fail-closed
// metadata/file-size checks.  No CUDA or real checkpoint is required.

#include "dee/expert_store.h"
#include "dee/weight_mmap.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++g_failures;
    }
}

std::filesystem::path make_test_dir() {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        ("dee4-expert-store-" + std::to_string(suffix));
    std::filesystem::create_directories(path);
    return path;
}

void write_fixture(const std::filesystem::path& directory,
                   size_t data_bytes = 160) {
    std::vector<uint8_t> data(data_bytes);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i & 0xff);
    }
    std::ofstream data_stream(directory / "experts.dee4", std::ios::binary);
    data_stream.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size()));
    data_stream.close();

    const std::string metadata = R"JSON({
  "format": "dee4-v2",
  "codec": "deepseek-fp4-e2m1-e8m0",
  "data_file": "experts.dee4",
  "data_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "start_layer": 6,
  "num_layers": 2,
  "experts_per_layer": 2,
  "record_bytes": 40,
  "weight_offsets": [0, 8, 16],
  "weight_nbytes": [8, 8, 8],
  "weight_out": [2, 2, 4],
  "weight_stored_in": [4, 4, 2],
  "scale_offsets": [24, 28, 32],
  "scale_nbytes": [4, 4, 8],
  "scale_out": [2, 2, 4],
  "scale_in": [2, 2, 2]
})JSON";
    std::ofstream metadata_stream(directory / "metadata.json", std::ios::binary);
    metadata_stream << metadata;
}

void write_trace_fixture(const std::filesystem::path& directory) {
    std::vector<uint8_t> data(120);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i + 17) & 0xff);
    }
    std::ofstream data_stream(directory / "experts.dee4", std::ios::binary);
    data_stream.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size()));
    data_stream.close();

    // Deliberately omit (6, 0): trace-backed stores must never silently
    // substitute a dense/arithmetic record for an unseen route.
    const std::string metadata = R"JSON({
  "format": "dee4-v3-trace",
  "codec": "deepseek-fp4-e2m1-e8m0",
  "data_file": "experts.dee4",
  "data_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "start_layer": 6,
  "num_layers": 2,
  "experts_per_layer": 2,
  "record_bytes": 40,
  "weight_offsets": [0, 8, 16],
  "weight_nbytes": [8, 8, 8],
  "weight_out": [2, 2, 4],
  "weight_stored_in": [4, 4, 2],
  "scale_offsets": [24, 28, 32],
  "scale_nbytes": [4, 4, 8],
  "scale_out": [2, 2, 4],
  "scale_in": [2, 2, 2],
  "total_experts": 3,
  "trace_journal_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "trace_final_chain_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "selection_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "records": [
    {"layer": 6, "expert": 1, "record_index": 0},
    {"layer": 7, "expert": 0, "record_index": 1},
    {"layer": 7, "expert": 1, "record_index": 2}
  ]
})JSON";
    std::ofstream metadata_stream(directory / "metadata.json", std::ios::binary);
    metadata_stream << metadata;
}

void test_arithmetic_lookup_and_stats() {
    const auto directory = make_test_dir();
    write_fixture(directory);

    dee::Dee4ExpertStore store;
    check(store.open(directory.string()), "valid DEE4 fixture opens");
    check(store.start_layer() == 6 && store.num_layers() == 2,
          "layer geometry loaded");
    check(store.experts_per_layer() == 2 && store.record_bytes() == 40,
          "expert/record geometry loaded");

    dee::ExpertView view;
    check(store.get(7, 1, &view), "last expert resolves");
    check(view.ok(), "resolved expert view is complete");
    check(view.record_index == 3, "record index is arithmetic");
    check(view.contiguous_data != nullptr && view.contiguous_nbytes == 40,
          "one contiguous record exposed");
    check(view.contiguous_data[0] == 120,
          "record pointer starts at expected byte offset");
    check(view.weights[0].data[0] == 120 &&
          view.weights[1].data[0] == 128 &&
          view.weights[2].data[0] == 136,
          "packed projection offsets resolve");
    check(view.scales[0].data[0] == 144 &&
          view.scales[1].data[0] == 148 &&
          view.scales[2].data[0] == 152,
          "scale offsets resolve");
    check(view.weights[0].dtype == dee::DType::I8 &&
          view.scales[0].dtype == dee::DType::F8,
          "codec dtypes preserved");
    check(!store.get(5, 0, &view) && !store.get(8, 0, &view) &&
          !store.get(6, 2, &view),
          "out-of-range requests fail closed");

    store.record_source_read(40, 2.0, 1, true);
    store.record_source_read(40, 6.0, 1, true);
    const dee::ExpertStoreStats stats = store.stats();
    check(stats.backend == "dee4", "backend telemetry identifies DEE4");
    check(stats.lookups == 4 && stats.lookup_failures == 3,
          "lookup telemetry includes failures");
    check(stats.source_reads == 2 && stats.contiguous_source_reads == 2 &&
          stats.source_regions == 2 && stats.bytes_requested == 80,
          "source-read telemetry is exact");
    check(stats.average_read_ms == 4.0 && stats.p50_read_ms == 2.0 &&
          stats.p95_read_ms == 6.0,
          "read-latency telemetry is exact");

    store.close();
    std::filesystem::remove_all(directory);
}

void test_data_size_mismatch_fails_closed() {
    const auto directory = make_test_dir();
    write_fixture(directory, 159);
    dee::Dee4ExpertStore store;
    check(!store.open(directory.string()),
          "truncated DEE4 data file is rejected");
    check(store.last_error().find("does not match expected") != std::string::npos,
          "size failure is attributable");
    std::filesystem::remove_all(directory);
}

void test_trace_index_lookup_and_fail_closed() {
    const auto directory = make_test_dir();
    write_trace_fixture(directory);
    dee::Dee4ExpertStore store;
    check(store.open(directory.string()), "valid trace DEE4 fixture opens");
    check(store.trace_indexed() && store.stored_records() == 3,
          "trace DEE4 exposes sparse record geometry");
    check(std::string(store.backend_name()) == "dee4_trace",
          "trace backend has distinct telemetry identity");

    dee::ExpertView view;
    check(store.get(7, 1, &view), "selected trace expert resolves");
    check(view.record_index == 2 && view.contiguous_data[0] == 97,
          "trace lookup uses explicit record index");
    std::vector<uint8_t> materialized(40, 0xee);
    check(store.materialize(view, materialized.data(), materialized.size()),
          "exact trace record materializes");
    check(materialized.front() == 97 && materialized.back() == 136,
          "materialized bytes preserve routed expert identity");
    dee::ExpertView forged = view;
    forged.record_index = 1;
    std::fill(materialized.begin(), materialized.end(),
              static_cast<uint8_t>(0xee));
    check(!store.materialize(forged, materialized.data(), materialized.size()),
          "mismatched record index and pointer fail closed");
    check(materialized.front() == 0xee && materialized.back() == 0xee,
          "failed identity check does not expose another record");
    check(!store.get(6, 0, &view),
          "unselected trace expert fails closed instead of using dense offset");
    check(store.get_layout_reference(6, &view),
          "layout reference does not require absent expert zero");
    check(view.record_index == 0 && view.contiguous_data[0] == 17,
          "layout reference selects first routed record at preferred layer");
    store.close();
    std::filesystem::remove_all(directory);
}

// Minimal safetensors shard: one expert (layer 7, expert 3) with six
// regions of distinct byte patterns.  Verifies that gather materialization
// produces [w1|w3|w2|s1|s3|s2] byte-identical to the view memcpy.
void write_safetensors_fixture(const std::filesystem::path& file,
                               std::vector<uint8_t>* expected_record) {
    // w1 gate [4,8] I8=32B, w3 up [4,8] I8=32B, w2 down [4,8] I8=32B,
    // scales [4,2] F8=8B each -> record = 32*3 + 8*3 = 120 B.
    const size_t region_bytes[6] = {32, 32, 32, 8, 8, 8};
    const char* names[6] = {
        "layers.7.ffn.experts.3.w1.weight",
        "layers.7.ffn.experts.3.w3.weight",
        "layers.7.ffn.experts.3.w2.weight",
        "layers.7.ffn.experts.3.w1.scale",
        "layers.7.ffn.experts.3.w3.scale",
        "layers.7.ffn.experts.3.w2.scale"};
    const char* dtypes[6] = {"I8", "I8", "I8", "F8_E8M0", "F8_E8M0", "F8_E8M0"};
    const char* shapes[6] = {"[4,8]", "[4,8]", "[4,8]",
                             "[4,2]", "[4,2]", "[4,2]"};

    std::vector<uint8_t> payload;
    size_t offsets[6];
    for (int i = 0; i < 6; ++i) {
        offsets[i] = payload.size();
        for (size_t b = 0; b < region_bytes[i]; ++b) {
            payload.push_back(static_cast<uint8_t>((i * 37 + b) & 0xff));
        }
    }
    if (expected_record) *expected_record = payload;

    std::string header = "{";
    for (int i = 0; i < 6; ++i) {
        if (i) header += ",";
        header += "\"" + std::string(names[i]) + "\":{\"dtype\":\""
            + dtypes[i] + "\",\"shape\":" + shapes[i]
            + ",\"data_offsets\":[" + std::to_string(offsets[i]) + ","
            + std::to_string(offsets[i] + region_bytes[i]) + "]}";
    }
    header += "}";

    std::ofstream out(file, std::ios::binary);
    const uint64_t hlen = header.size();
    out.write(reinterpret_cast<const char*>(&hlen), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(payload.data()),
              static_cast<std::streamsize>(payload.size()));
}

void test_safetensors_gather_materialize() {
    const auto directory = make_test_dir();
    const auto shard_path = directory / "model-test.safetensors";
    std::vector<uint8_t> expected;
    write_safetensors_fixture(shard_path, &expected);

    dee::WeightMmap mmap;
    check(mmap.open(shard_path.string()), "safetensors fixture opens");
    dee::TensorResolver resolver;
    resolver.set_model(dee::TensorResolver::Model::DEEPSEEK_V4);
    resolver.register_shard(&mmap);

    dee::SafetensorsExpertStore store(&resolver);
    check(store.can_gather_materialize(),
          "safetensors store advertises gather materialization");

    dee::ExpertView view;
    check(store.get(7, 3, &view), "v4 expert resolves");
    check(view.ok(), "resolved view complete");
    check(view.contiguous_data == nullptr,
          "safetensors view is non-contiguous");

    std::vector<uint8_t> gathered(expected.size(), 0xee);
    check(store.materialize(view, gathered.data(), gathered.size()),
          "gather materialize succeeds");
    check(gathered == expected,
          "gathered record is byte-identical to w1|w3|w2|s1|s3|s2");

    std::vector<uint8_t> manual(expected.size(), 0x00);
    {
        size_t off = 0;
        const dee::TensorView* regions[6] = {
            &view.weights[0], &view.weights[1], &view.weights[2],
            &view.scales[0], &view.scales[1], &view.scales[2]};
        for (const dee::TensorView* r : regions) {
            std::memcpy(manual.data() + off, r->data, r->nbytes);
            off += r->nbytes;
        }
    }
    check(gathered == manual, "gather output equals per-region memcpy");

    // fail-closed: wrong nbytes, broken view, null resolver
    check(!store.materialize(view, gathered.data(), gathered.size() - 1),
          "short buffer rejected");
    dee::ExpertView broken = view;
    broken.weights[1].data = nullptr;
    check(!store.materialize(broken, gathered.data(), gathered.size()),
          "incomplete view rejected");
    dee::SafetensorsExpertStore null_store(nullptr);
    check(!null_store.can_gather_materialize(),
          "null resolver cannot gather");
    check(!null_store.materialize(view, gathered.data(), gathered.size()),
          "null resolver materialize fails closed");

    store.stats();  // telemetry must not crash
    mmap.close();
    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    test_arithmetic_lookup_and_stats();
    test_data_size_mismatch_fails_closed();
    test_trace_index_lookup_and_fail_closed();
    test_safetensors_gather_materialize();
    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
