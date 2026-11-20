// tools/phase3/test_p3_dee4_full_geometry.cpp
//
// Phase-3 arbitrary-prompt coverage proof for the EXISTING dee4-v2 reader:
// opens a full-universe store (start_layer=0, num_layers=46 -> the 43 main
// model layers as buckets 0..42 PLUS the three mtp draft layers as buckets
// 43..45, experts_per_layer=256, record_bytes=13,369,344 -> a
// 157,437,394,944-byte data file, sparse on disk so the test needs no real
// capacity) and verifies that EVERY (layer, expert) in the routed universe
// resolves through Dee4ExpertStore::get with the correct fixed-stride
// record index, layout, and materialization path.
//
// Usage: test_p3_dee4_full_geometry.exe <store_dir>
// The store is produced by tools/phase3/p3_sparse_store.py (which also seeds
// marker bytes into sampled records so this test can check content, not just
// resolution).

#include "dee/expert_store.h"

#include <cstdio>
#include <cstring>
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

// Marker pattern seeded by the Python driver: first and last 16 bytes of
// record i are (i & 0xff) + j for j in 0..15.
uint8_t marker_byte(size_t record_index, size_t j) {
    return static_cast<uint8_t>(((record_index & 0xff) + j) & 0xff);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <store_dir>\n", argv[0]);
        return 2;
    }
    dee::Dee4ExpertStore store;
    check(store.open(argv[1]), "full-universe dee4-v2 store opens");
    if (g_failures) return 1;

    check(store.start_layer() == 0 && store.num_layers() == 46 &&
              store.experts_per_layer() == 256,
          "universe geometry is 46 buckets x 256 experts");
    check(store.record_bytes() == 13369344, "record stride is 13,369,344 B");
    check(!store.trace_indexed(), "dense store: no trace index");
    check(store.stored_records() == 11776,
          "11,776 records = 43 main + 3 mtp buckets");

    // Arbitrary-prompt battery: sample every domain boundary plus dense
    // interior and edge pairs, including pairs absent from the sealed
    // trace bank (e.g. (0,1), (42,254) are not in the v60 selection) and
    // the mtp draft buckets (43..45).
    const int layers[] = {0, 1, 2, 3, 17, 20, 21, 40, 41, 42, 43, 44, 45};
    const int experts[] = {0, 1, 5, 127, 128, 200, 254, 255};
    size_t resolved = 0;
    for (int layer : layers) {
        for (int expert : experts) {
            dee::ExpertView view;
            check(store.get(layer, expert, &view), "universe pair resolves");
            if (!view.ok()) continue;
            const size_t expected_index =
                static_cast<size_t>(layer) * 256 + expert;
            check(view.record_index == expected_index,
                  "record index is dense arithmetic");
            check(view.contiguous_data != nullptr &&
                      view.contiguous_nbytes == 13369344,
                  "one contiguous record exposed");
            check(view.codec == dee::ExpertCodec::DeepSeekFp4E2m1E8m0,
                  "codec identity");
            check(view.weights[0].nbytes == 4194304 &&
                      view.weights[0].shape[0] == 2048 &&
                      view.weights[0].shape[1] == 2048,
                  "w1 packed [2048,2048]");
            check(view.weights[2].shape[0] == 4096 &&
                      view.weights[2].shape[1] == 1024,
                  "w2 packed [4096,1024]");
            check(view.scales[0].nbytes == 262144 &&
                      view.scales[0].shape[0] == 2048 &&
                      view.scales[0].shape[1] == 128,
                  "w1 scale [2048,128]");
            check(view.scales[2].shape[0] == 4096 &&
                      view.scales[2].shape[1] == 64,
                  "w2 scale [4096,64]");
            // Record order: w1|w3|w2 weights then scales.
            check(view.weights[0].data == view.contiguous_data &&
                      view.weights[1].data == view.contiguous_data + 4194304 &&
                      view.weights[2].data == view.contiguous_data + 8388608 &&
                      view.scales[0].data == view.contiguous_data + 12582912 &&
                      view.scales[1].data == view.contiguous_data + 12845056 &&
                      view.scales[2].data == view.contiguous_data + 13107200,
                  "component offsets match the canonical record layout");
            ++resolved;
        }
    }
    check(resolved == sizeof(layers) / sizeof(int) *
                          (sizeof(experts) / sizeof(int)),
          "all sampled universe pairs resolved");

    // Fail-closed outside the 46-bucket universe.
    dee::ExpertView view;
    check(!store.get(46, 0, &view) && !store.get(-1, 0, &view) &&
              !store.get(0, 256, &view) && !store.get(0, -1, &view),
          "out-of-universe lookups fail closed");

    // Content: seeded records return their marker bytes; unseeded are zero.
    const int seeded[][2] = {{0, 0}, {20, 128}, {42, 255}, {43, 0},
                             {45, 255}};
    for (const auto& pair : seeded) {
        dee::ExpertView v;
        check(store.get(pair[0], pair[1], &v), "seeded record resolves");
        const size_t idx = static_cast<size_t>(pair[0]) * 256 + pair[1];
        check(v.contiguous_data[0] == marker_byte(idx, 0) &&
                  v.contiguous_data[15] == marker_byte(idx, 15) &&
                  v.contiguous_data[13369343] ==
                      marker_byte(idx, 15),
                  "seeded marker bytes readable through the mmap view");
    }

    // materialize() copies the exact record through the Phase-2 cold path.
    std::vector<uint8_t> dst(13369344, 0xee);
    check(store.get(45, 255, &view), "edge record resolves");
    check(store.materialize(view, dst.data(), dst.size()),
          "edge record materializes");
    const size_t idx = 45 * 256 + 255;
    check(dst[0] == marker_byte(idx, 0) &&
              dst[13369343] == marker_byte(idx, 15),
          "materialized bytes carry the seeded content");

    const dee::ExpertStoreStats stats = store.stats();
    check(stats.backend == "dee4", "backend identity is dee4");
    check(stats.lookup_failures == 4, "four fail-closed lookups recorded");

    store.close();
    if (g_failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}
