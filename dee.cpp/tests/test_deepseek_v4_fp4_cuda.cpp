// tests/test_deepseek_v4_fp4_cuda.cpp
//
// DSv4 FP4 CUDA parity: run fp4_e2m1_to_f16_cuda on-device and compare the
// decoded FP16 weights against the host reference fp4_e2m1_dequantize
// (weight_mmap.cpp), which is pinned to the official convert.py semantics by
// test_deepseek_v4_fp4_decode.cpp.  This test REQUIRES a CUDA device (T4,
// sm_75); it is only built when DEE_CUDA=ON.

#include "dee/cuda_convert.h"
#include "dee/weight_mmap.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s (%s)\n", operation, cudaGetErrorName(status),
                 cudaGetErrorString(status));
    return false;
}

float half_to_float(__half h) {
    return __half2float(h);
}

}  // namespace

int main() {
    std::printf("=== dee.cpp DSv4 FP4 CUDA parity test ===\n");

    // A small but representative shape: in=128 (packed_in=64, scale_in=4).
    const size_t out = 2;
    const size_t in = 128;
    const size_t packed_in = in / 2;
    const size_t scale_in = in / 32;

    // Deterministic packed weights: row 0 nibble = (col/2) % 16 cycling the
    // whole table; row 1 nibble = (col/2 + 5) % 16.  Byte packs low nibble at
    // col 2i and high nibble at col 2i+1.
    std::vector<uint8_t> packed(out * packed_in, 0);
    for (size_t o = 0; o < out; ++o) {
        for (size_t p = 0; p < packed_in; ++p) {
            const uint8_t lo = static_cast<uint8_t>((p + (o ? 5 : 0)) & 0x0F);
            const uint8_t hi = static_cast<uint8_t>((p + 1 + (o ? 5 : 0)) & 0x0F);
            packed[o * packed_in + p] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }
    // Realistic e8m0 scale bytes (0x7d..0x82 -> 2^-2 .. 2^3), cycling per block.
    std::vector<uint8_t> scale(out * scale_in, 0x7F);
    const uint8_t scale_bytes[6] = {0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82};
    for (size_t o = 0; o < out; ++o) {
        for (size_t b = 0; b < scale_in; ++b) {
            scale[o * scale_in + b] = scale_bytes[(b + o) % 6];
        }
    }

    // Host reference (authoritative semantics).
    std::vector<float> ref(out * in, 0.0f);
    dee::fp4_e2m1_dequantize(packed.data(), scale.data(), out, in, ref.data());

    // Device decode.
    uint8_t* d_packed = nullptr;
    uint8_t* d_scale = nullptr;
    void* d_out = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = check(cudaStreamCreate(&stream), "cudaStreamCreate") &&
        check(cudaMalloc(reinterpret_cast<void**>(&d_packed), packed.size()),
              "cudaMalloc(packed)") &&
        check(cudaMalloc(reinterpret_cast<void**>(&d_scale), scale.size()),
              "cudaMalloc(scale)") &&
        check(cudaMalloc(&d_out, out * in * sizeof(__half)),
              "cudaMalloc(output)") &&
        check(cudaMemcpyAsync(d_packed, packed.data(), packed.size(),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(packed)") &&
        check(cudaMemcpyAsync(d_scale, scale.data(), scale.size(),
                              cudaMemcpyHostToDevice, stream),
              "cudaMemcpyAsync(scale)");

    if (ok) {
        ok = dee::fp4_e2m1_to_f16_cuda(d_packed, d_scale, d_out, out, in, stream);
    }

    std::vector<__half> got(out * in);
    if (ok) {
        ok = check(cudaMemcpyAsync(got.data(), d_out, out * in * sizeof(__half),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(output)") &&
            check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    // Compare against the FP32 reference with FP16 rounding tolerance.
    // The reference values are exact products of a small integer/0.5 value and
    // a power of two, so FP16 represents them exactly; still allow one ulp
    // slop for safety.
    int mismatches = 0;
    float max_abs_err = 0.0f;
    if (ok) {
        for (size_t i = 0; i < out * in; ++i) {
            const float g = half_to_float(got[i]);
            const float r = ref[i];
            const float err = std::fabs(g - r);
            if (err > max_abs_err) max_abs_err = err;
            const float tol = 1e-3f * std::fabs(r) + 1e-6f;
            if (err > tol) {
                if (mismatches < 8) {
                    std::printf("  mismatch[%zu] got=%g want=%g\n", i, g, r);
                }
                ++mismatches;
            }
        }
    }

    if (d_packed) cudaFree(d_packed);
    if (d_scale) cudaFree(d_scale);
    if (d_out) cudaFree(d_out);
    if (stream) cudaStreamDestroy(stream);

    std::printf("FP4 e2m1 CUDA parity: %s (max_abs_err=%.6g, mismatches=%d)\n",
                (ok && mismatches == 0) ? "PASS" : "FAIL", max_abs_err,
                mismatches);
    return (ok && mismatches == 0) ? 0 : 1;
}
