// tests/test_expert_store.cpp
//
// Host-only validation for the DEE4 v2 arithmetic lookup and fail-closed
// metadata/file-size checks.  No CUDA or real checkpoint is required.

#include "dee/expert_store.h"

#include <chrono>
#include <cstdio>
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

}  // namespace

int main() {
    test_arithmetic_lookup_and_stats();
    test_data_size_mismatch_fails_closed();
    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
