// ParallelCopier must be byte-identical to serial memcpy for every lane
// count, piece layout and size (including below the parallel threshold).
#include "dee/parallel_copy.h"

#ifndef PC_REPS
#define PC_REPS 200
#endif

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

static int g_fail = 0;
static void check(const char* what, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

static bool one_case(dee::ParallelCopier& copier, std::mt19937& rng,
                     const std::vector<size_t>& sizes) {
    size_t total = 0;
    for (size_t n : sizes) total += n;
    std::vector<uint8_t> src(total), got(total + 64, 0xAB), want(total + 64, 0xAB);
    for (auto& b : src) b = static_cast<uint8_t>(rng());
    // Scatter pieces to shuffled destination offsets (like the FP4 gather).
    std::vector<size_t> order(sizes.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<size_t> src_off(sizes.size()), dst_off(sizes.size());
    for (size_t i = 0, o = 0; i < sizes.size(); ++i) { src_off[i] = o; o += sizes[i]; }
    for (size_t k = 0, o = 0; k < order.size(); ++k) { dst_off[order[k]] = o; o += sizes[order[k]]; }
    std::vector<dee::CopyPiece> pieces;
    for (size_t i = 0; i < sizes.size(); ++i) {
        pieces.push_back({got.data() + dst_off[i], src.data() + src_off[i], sizes[i]});
        if (sizes[i]) std::memcpy(want.data() + dst_off[i], src.data() + src_off[i], sizes[i]);
    }
    copier.run(pieces.data(), pieces.size());
    return got == want;
}

int main() {
    std::mt19937 rng(7);
    for (size_t lanes : {1, 2, 3, 4, 7, 16}) {
        dee::ParallelCopier copier(lanes);
        bool ok = true;
        // DSv4 FP4 pack shape: 3 packed weights + 3 scale tensors, 13,369,344 B.
        ok &= one_case(copier, rng, {4194304, 4194304, 4194304, 262144, 262144, 262144});
        ok &= one_case(copier, rng, {13369344});
        ok &= one_case(copier, rng, {3, 0, 1, 5});                  // below threshold
        ok &= one_case(copier, rng, {(1u << 20) + 1, 0, 7, 12345});  // odd split
        for (int r = 0; r < PC_REPS && ok; ++r) {                     // reuse the pool
            std::vector<size_t> sizes(1 + rng() % 6);
            for (auto& s : sizes) s = rng() % (3u << 20);
            ok &= one_case(copier, rng, sizes);
        }
        char what[64];
        std::snprintf(what, sizeof what, "byte-identical at %zu lane(s)", lanes);
        check(what, ok);
    }
    check("lanes clamp to [1,16]",
          dee::ParallelCopier(0).lanes() == 1 && dee::ParallelCopier(99).lanes() == 16);
    std::printf("%s\n", g_fail ? "FAILED" : "ALL PASSED");
    return g_fail ? 1 : 0;
}
