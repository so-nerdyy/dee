// tests/test_deepseek_v4_fp4_decode.cpp
//
// DSv4 FP4 decode: verify the portable host reference in weight_mmap.cpp
// against the official DeepSeek-V4-Flash-0731 FP4 semantics (mirrored from
// scripts/deepseek_v4_expert_reference.py, which itself is unit-tested
// against the pinned official inference code):
//
//   - 16-entry e2m1fn table (indices 0..7 positive, 8..15 negative);
//   - packed I8 [out, in//2], low nibble -> col 2i, high nibble -> col 2i+1;
//   - scale F8_E8M0 [out, in//32], value = 2^(bits - 127);
//   - w[o, i] = fp4_table[nibble] * scale[o, i//32].
//
// No GPU and no shard file required.  The CUDA kernel (cuda_convert.cu) is a
// thin port of the same table/scale math; this test pins the host semantics
// so the kernel can be checked against it on a T4.
//
// Build (standalone, no cmake needed):
//   g++ -std=c++17 -I../include test_deepseek_v4_fp4_decode.cpp \
//       ../src/weight_mmap.cpp ../src/json_min.cpp -o test_deepseek_v4_fp4_decode

#include "dee/weight_mmap.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;

static void check(const char* what, bool cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

static void check_near(const char* what, float got, float want, float eps = 1e-6f) {
    const bool ok = std::fabs(got - want) <= eps;
    printf("  [%s] %s: got=%.8g want=%.8g\n", ok ? "PASS" : "FAIL", what, got, want);
    if (!ok) ++g_failures;
}

int main() {
    printf("=== dee.cpp DSv4 FP4 (e2m1fn) decode test ===\n");

    // 1. The official 16-entry table.
    const float* table = dee::fp4_e2m1_table();
    const float want_table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    bool table_ok = true;
    for (int i = 0; i < 16; ++i) {
        if (table[i] != want_table[i]) table_ok = false;
    }
    check("e2m1fn table matches official 16 entries", table_ok);

    // 2. Nibble decode: each nibble -> table value, and nibble>=16 masks.
    check_near("nibble 0x0 -> 0.0", dee::fp4_nibble_to_f32(0x0), 0.0f);
    check_near("nibble 0x1 -> 0.5", dee::fp4_nibble_to_f32(0x1), 0.5f);
    check_near("nibble 0x7 -> 6.0", dee::fp4_nibble_to_f32(0x7), 6.0f);
    check_near("nibble 0x8 -> 0.0 (neg zero)", dee::fp4_nibble_to_f32(0x8), 0.0f);
    check_near("nibble 0xF -> -6.0", dee::fp4_nibble_to_f32(0xF), -6.0f);
    check_near("nibble 0x1F masks to 0xF -> -6.0", dee::fp4_nibble_to_f32(0x1F), -6.0f);

    // 3. e8m0 scale decode (verified against torch.float8_e8m0fnu round trips).
    check_near("e8m0 0x7f -> 1.0", dee::e8m0_to_f32(0x7F), 1.0f);
    check_near("e8m0 0x80 -> 2.0", dee::e8m0_to_f32(0x80), 2.0f);
    check_near("e8m0 0x86 -> 128.0", dee::e8m0_to_f32(0x86), 128.0f);
    check_near("e8m0 0x76 -> 2^-9", dee::e8m0_to_f32(0x76), 1.0f / 512.0f, 1e-9f);

    // 4. Full dequantize on a small matrix.  in=64 => packed_in=32, scale_in=2.
    //    Build a 2x64 weight: row 0 uses scale block [1.0, 2.0]; row 1 uses
    //    [4.0, 8.0].  Pack two nibbles per byte so col 2i = low, 2i+1 = high.
    const size_t out = 2, in = 64;
    std::vector<uint8_t> packed(out * (in / 2), 0);
    std::vector<uint8_t> scale(out * (in / 32), 0);
    // row 0: value nibble 0x2 (=1.0) for cols 0..31, nibble 0x4 (=2.0) for 32..63
    // row 1: value nibble 0x3 (=1.5) for cols 0..31, nibble 0x5 (=3.0) for 32..63
    for (size_t i = 0; i < in / 2; ++i) {
        const uint8_t lo0 = 0x2, hi0 = 0x2;   // row 0 = 1.0
        const uint8_t lo1 = 0x3, hi1 = 0x3;   // row 1 = 1.5
        packed[0 * (in / 2) + i] = lo0 | (hi0 << 4);
        packed[1 * (in / 2) + i] = lo1 | (hi1 << 4);
    }
    // scale bytes: row0 blocks [1.0, 2.0]; row1 blocks [4.0, 8.0].
    scale[0 * 2 + 0] = 0x7F; scale[0 * 2 + 1] = 0x80;
    scale[1 * 2 + 0] = 0x81; scale[1 * 2 + 1] = 0x82;

    std::vector<float> dst(out * in, 0.0f);
    dee::fp4_e2m1_dequantize(packed.data(), scale.data(), out, in, dst.data());

    // row 0, col 0 -> 1.0 * 1.0; col 32 -> 1.0 * 2.0
    check_near("w[0,0] = 1.0", dst[0 * in + 0], 1.0f);
    check_near("w[0,32] = 2.0", dst[0 * in + 32], 2.0f);
    // row 1, col 0 -> 1.5 * 4.0; col 32 -> 1.5 * 8.0
    check_near("w[1,0] = 6.0", dst[1 * in + 0], 6.0f);
    check_near("w[1,32] = 12.0", dst[1 * in + 32], 12.0f);

    // 5. Packed nibble order: a byte 0x0F holds low=0xF (-6.0) high=0x0 (0.0),
    //    so col 0 (low) = -6.0 and col 1 (high) = 0.0 under scale 1.0.
    std::vector<uint8_t> one_packed(in / 2, 0);
    std::vector<uint8_t> one_scale(out * (in / 32), 0x7F);
    one_packed[0] = 0x0F;
    std::vector<float> one_dst(out * in, 0.0f);
    dee::fp4_e2m1_dequantize(one_packed.data(), one_scale.data(), out, in, one_dst.data());
    check_near("low nibble -> col 0 = -6.0", one_dst[0], -6.0f);
    check_near("high nibble -> col 1 = 0.0", one_dst[1], 0.0f);

    if (g_failures) {
        printf("\n%d check(s) FAILED\n", g_failures);
        return 1;
    }
    printf("\nall checks passed\n");
    return 0;
}
