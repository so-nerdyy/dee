// tests/test_deepseek_v4_fp4_expert.cpp
//
// End-to-end native FP4 expert path for DeepSeek-V4-Flash-0731.
//
// Generates a mini DEEPSEEK_V4 safetensors shard holding expert 0's packed
// FP4 (e2m1fn) weights + per-block e8m0 scales under the official tensor
// naming, then runs `Engine::moe_forward_experts(layer=0, ...)` with
// `WeightTransferDType::Fp4E2m1` (native transfer-stream dequant + cuBLAS
// SwiGLU) and compares the raw expert output against a host FP32 reference
// computed from the SAME decoded weights.
//
// This pins the full native chain (mmap -> resolve -> stage fp4 -> prefetch
// -> fp4_e2m1_to_f16_cuda dequant -> swiglu_expert_fp16_cuda) rather than the
// dequant kernel in isolation.  CUDA-only (Fp4E2m1 requires the CUDA path).
//
// Build (standalone):
//   nvcc -arch=sm_75 -DDEE_CUDA=1 -I../include \
//        test_deepseek_v4_fp4_expert.cpp \
//        ../src/engine.cpp ../src/async_prefetcher.cpp ../src/vram_cache.cpp \
//        ../src/weight_mmap.cpp ../src/oracle.cpp ../src/pt_loader.cpp \
//        ../src/profiling.cpp ../src/trace_alloc.cpp ../src/json_min.cpp \
//        ../src/dee.cpp ../src/cuda_convert.cu ../src/swiglu_cuda.cu \
//        ../src/rmsnorm_cuda.cu -lcudart -lcublas -o test_deepseek_v4_fp4_expert

#include "dee/engine.h"
#include "dee/weight_mmap.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) std::printf("  [PASS] %s\n", msg); \
    else { std::printf("  [FAIL] %s\n", msg); ++failures; } \
} while (0)

// Both must be multiples of 64: the fp4_e2m1_to_f16_cuda kernel requires
// `in % 64 == 0` for whole-byte packed/block alignment (real model: 2048/4096).
constexpr int kHidden = 128;  // logical hidden width
constexpr int kInter  = 64;   // MoE intermediate width

// Deterministic nibble + scale patterns. Values stay small so the official
// swiglu_limit=10 clamp is a no-op (the engine's SwiGLU does not clamp; that
// is a documented, separately-tracked semantic gap for |activation|>10).
uint8_t nibble_for(int o, int c) { return (uint8_t)((o * 7 + c * 13 + o * c * 3) % 16); }
uint8_t scale_for(int o, int c) {
    static const uint8_t opts[3] = {0x7E /*0.5*/, 0x7F /*1.0*/, 0x80 /*2.0*/};
    return opts[(o + c) % 3];
}

// One projection: (name, dtype, shape, packed bytes, scale bytes).
struct Proj {
    std::string name;
    int out, in;                 // logical [out, in]
    std::vector<uint8_t> packed; // I8 [out, in//2]
    std::vector<uint8_t> scale;  // F8_E8M0 [out, in//32]
};

Proj make_proj(const std::string& name, int out, int in) {
    Proj p;
    p.name = name;
    p.out = out;
    p.in = in;
    p.packed.resize((size_t)out * (in / 2));
    for (int o = 0; o < out; ++o) {
        for (int c = 0; c < in / 2; ++c) {
            const uint8_t lo = nibble_for(o, 2 * c);
            const uint8_t hi = nibble_for(o, 2 * c + 1);
            p.packed[(size_t)o * (in / 2) + c] = (uint8_t)(lo | (hi << 4));
        }
    }
    p.scale.resize((size_t)out * (in / 32));
    for (int o = 0; o < out; ++o) {
        for (int c = 0; c < in / 32; ++c) {
            p.scale[(size_t)o * (in / 32) + c] = scale_for(o, c);
        }
    }
    return p;
}

// Write a mini safetensors shard for layer 0 expert 0.
bool write_v4_shard(const std::string& path) {
    Proj gate = make_proj("layers.0.ffn.experts.0.w1.weight", kInter, kHidden);
    Proj up   = make_proj("layers.0.ffn.experts.0.w3.weight", kInter, kHidden);
    Proj down = make_proj("layers.0.ffn.experts.0.w2.weight", kHidden, kInter);

    struct Blob {
        std::string name;
        std::string dtype;
        std::vector<int> shape;
        const std::vector<uint8_t>* data;
    };
    std::vector<Blob> blobs;
    // DeepSeek-V4 official naming: w1=gate, w3=up, w2=down (+ .scale each).
    blobs.push_back({"layers.0.ffn.experts.0.w1.weight", "I8", {kInter, kHidden / 2}, &gate.packed});
    blobs.push_back({"layers.0.ffn.experts.0.w1.scale", "F8_E8M0", {kInter, kHidden / 32}, &gate.scale});
    blobs.push_back({"layers.0.ffn.experts.0.w3.weight", "I8", {kInter, kHidden / 2}, &up.packed});
    blobs.push_back({"layers.0.ffn.experts.0.w3.scale", "F8_E8M0", {kInter, kHidden / 32}, &up.scale});
    blobs.push_back({"layers.0.ffn.experts.0.w2.weight", "I8", {kHidden, kInter / 2}, &down.packed});
    blobs.push_back({"layers.0.ffn.experts.0.w2.scale", "F8_E8M0", {kHidden, kInter / 32}, &down.scale});

    // Lay out the data section and build the header.
    std::vector<uint8_t> data;
    std::string header = "{";
    size_t offset = 0;
    for (size_t i = 0; i < blobs.size(); ++i) {
        const Blob& b = blobs[i];
        const size_t n = b.data->size();
        data.insert(data.end(), b.data->begin(), b.data->end());
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}",
                 b.name.c_str(), b.dtype.c_str(), b.shape[0], b.shape[1],
                 offset, offset + n);
        header += buf;
        if (i + 1 < blobs.size()) header += ",";
        offset += n;
    }
    header += "}";

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const uint64_t hlen = header.size();
    out.write(reinterpret_cast<const char*>(&hlen), 8);
    out.write(header.data(), (std::streamsize)header.size());
    out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)data.size());
    out.close();
    return true;
}

// FP32 reference SwiGLU (no clamp; activations kept small). y[o] = sum_i
// Wd[o,i] * silu(sum_j Wg[i,j] x[j]) * (sum_j Wu[i,j] x[j]).
void reference_swiglu(const std::vector<float>& Wg, const std::vector<float>& Wu,
                      const std::vector<float>& Wd, const std::vector<float>& x,
                      float swiglu_limit, std::vector<float>& y) {
    std::vector<float> h(kInter, 0.0f);
    for (int i = 0; i < kInter; ++i) {
        float g = 0.0f, u = 0.0f;
        for (int j = 0; j < kHidden; ++j) {
            g += Wg[(size_t)i * kHidden + j] * x[j];
            u += Wu[(size_t)i * kHidden + j] * x[j];
        }
        if (swiglu_limit > 0.0f) {
            g = std::min(g, swiglu_limit);
            u = std::min(std::max(u, -swiglu_limit), swiglu_limit);
        }
        h[i] = (g / (1.0f + std::exp(-g))) * u;
    }
    y.assign(kHidden, 0.0f);
    for (int o = 0; o < kHidden; ++o) {
        float acc = 0.0f;
        for (int i = 0; i < kInter; ++i) acc += Wd[(size_t)o * kInter + i] * h[i];
        y[o] = acc;
    }
}

void decode_proj(const Proj& p, std::vector<float>& dst) {
    dst.resize((size_t)p.out * p.in);
    dee::fp4_e2m1_dequantize(p.packed.data(), p.scale.data(),
                             (size_t)p.out, (size_t)p.in, dst.data());
}

}  // namespace

int main() {
    std::printf("=== dee.cpp DeepSeek-V4 FP4 expert end-to-end test ===\n");

    // Deterministic input in [-1, 1].
    std::vector<float> x(kHidden);
    for (int j = 0; j < kHidden; ++j) {
        x[j] = std::sin((float)(j + 1) * 0.7f) * 0.8f;
    }

    // Reference: decode the same packed tensors the engine will stream.
    Proj gate = make_proj("w1", kInter, kHidden);
    Proj up   = make_proj("w3", kInter, kHidden);
    Proj down = make_proj("w2", kHidden, kInter);
    std::vector<float> Wg, Wu, Wd;
    decode_proj(gate, Wg);
    decode_proj(up, Wu);
    decode_proj(down, Wd);
    std::vector<float> reference, clamped_reference;
    reference_swiglu(Wg, Wu, Wd, x, 0.0f, reference);
    reference_swiglu(Wg, Wu, Wd, x, 1.0f, clamped_reference);

    const std::string shard_path = "tests/data/dsv4_fp4_expert.safetensors";
    CHECK(write_v4_shard(shard_path), "mini DEEPSEEK_V4 FP4 shard written");

    auto run_and_check = [&](float swiglu_limit, const std::vector<float>& ref,
                             const char* label) {
        dee::EngineConfig cfg;
        cfg.shard_path = shard_path;
        cfg.oracle_path = "";       // real-model integration mode (caller owns router)
        cfg.num_tokens = 1;
        cfg.num_layers = 1;
        cfg.base_layer = 0;
        cfg.hidden = kHidden;
        cfg.inter = kInter;
        cfg.num_experts = 1;
        cfg.topk = 1;
        cfg.use_cuda = true;
        cfg.cache_dtype = dee::DeviceCacheDType::Fp16;
        cfg.transfer_dtype = dee::WeightTransferDType::Fp4E2m1;
        cfg.swiglu_limit = swiglu_limit;
        cfg.budget_bytes = 2 * 3ULL * kInter * kHidden * sizeof(uint16_t);

        dee::Engine engine;
        CHECK(engine.init(cfg), label);
        if (failures) return;
        std::vector<float> out(kHidden, 0.0f);
        CHECK(engine.moe_forward_experts(0, x.data(), out.data(), std::vector<int>{0}),
              "native FP4 routed-expert SwiGLU executes");

        bool finite = true;
        double max_abs = 0.0, ss_err = 0.0, ss_ref = 0.0, dot = 0.0, nr = 0.0, ng = 0.0;
        for (int o = 0; o < kHidden; ++o) {
            if (!std::isfinite(out[o])) finite = false;
            const double err = (double)out[o] - ref[o];
            max_abs = std::max(max_abs, std::fabs(err));
            ss_err += err * err;
            ss_ref += (double)ref[o] * ref[o];
            dot += (double)out[o] * ref[o];
            nr += (double)out[o] * out[o];
            ng += (double)ref[o] * ref[o];
        }
        const double rel_rmse = std::sqrt(ss_err / std::max(ss_ref, 1e-12));
        const double cosine = dot / std::max(std::sqrt(nr * ng), 1e-12);
        std::printf("  [%s] max_abs_err=%.6g rel_rmse=%.6g cosine=%.9f\n",
                    label, max_abs, rel_rmse, cosine);
        CHECK(finite, "native output is finite");
        CHECK(rel_rmse < 0.02, "native FP4 expert matches FP32 reference (rel RMSE < 2%)");
        CHECK(cosine > 0.999, "native FP4 expert cosine similarity > 0.999");
    };

    run_and_check(0.0f, reference,
                  "engine init + unclamped SwiGLU (DEEPSEEK_V4 FP4, CUDA, FP16)");
    run_and_check(1.0f, clamped_reference,
                  "engine init + clamped SwiGLU (swiglu_limit=1.0)");

    if (failures) {
        std::printf("### %d FAILED ###\n", failures);
        return 1;
    }
    std::printf("ALL PASS\n");
    return 0;
}
