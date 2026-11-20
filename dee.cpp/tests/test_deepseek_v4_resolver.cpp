// tests/test_deepseek_v4_resolver.cpp
//
// DS6: verify the DeepSeek-V4-Flash-0731 naming dialect and dtype mapping
// added to TensorResolver. No shard file needed -- these are pure string /
// dtype helpers (the layout is asserted against the pinned official index in
// the Python ledger tests, e.g. layers.6.ffn.experts.0.w1.weight [2048,2048]).
//
// Build (standalone, no cmake needed):
//   g++ -std=c++17 -I../include test_deepseek_v4_resolver.cpp \
//       ../src/weight_mmap.cpp ../src/json_min.cpp -o test_deepseek_v4_resolver
//
// Run:
//   ./test_deepseek_v4_resolver

#include "dee/weight_mmap.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(const char* what, bool cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

static void check_eq(const char* what, const std::string& got, const std::string& want) {
    bool ok = got == want;
    printf("  [%s] %s: got=%s\n", ok ? "PASS" : "FAIL", what, got.c_str());
    if (!ok) printf("         want=%s\n", want.c_str());
    if (!ok) ++g_failures;
}

namespace {

// Write a tiny synthetic safetensors shard whose header uses the official
// DeepSeek-V4-Flash-0731 naming, so the routing path (not just the string
// helpers) can be exercised. Returns the file path, or "" on failure.
std::string write_mini_v4_shard(const std::string& path) {
    // (name, dtype, rows, cols, nbytes); data_offsets are cumulative.
    struct T { const char* name; const char* dtype; int rows; int cols; int nbytes; };
    const T tensors[] = {
        {"layers.6.ffn.experts.0.w1.weight", "I8",       4,  4,  16},
        {"layers.6.ffn.experts.0.w1.scale",  "F8_E8M0",  4,  1,   4},
        {"layers.6.ffn.experts.0.w2.weight", "I8",       8,  2,  16},
        {"layers.6.ffn.experts.0.w2.scale",  "F8_E8M0",  8,  1,   8},
        {"layers.6.ffn.shared_experts.w1.weight", "F8_E4M3", 4, 4, 16},
        {"layers.6.hc_attn_fn",              "F32",      8,  0,  32},
        {"mtp.0.dspark_markov_w1.weight",    "F32",      2,  2,  16},
    };
    constexpr int N = 7;
    int offsets[N];
    int total = 0;
    for (int i = 0; i < N; ++i) { offsets[i] = total; total += tensors[i].nbytes; }

    std::string header = "{";
    for (int i = 0; i < N; ++i) {
        char buf[512];
        const T& t = tensors[i];
        if (t.cols == 0) {
            snprintf(buf, sizeof(buf),
                     "\"%s\":{\"dtype\":\"%s\",\"shape\":[%d],"
                     "\"data_offsets\":[%d,%d]}",
                     t.name, t.dtype, t.rows, offsets[i], offsets[i] + t.nbytes);
        } else {
            snprintf(buf, sizeof(buf),
                     "\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],"
                     "\"data_offsets\":[%d,%d]}",
                     t.name, t.dtype, t.rows, t.cols, offsets[i], offsets[i] + t.nbytes);
        }
        header += buf;
        if (i + 1 < N) header += ",";
    }
    header += "}";

    std::vector<uint8_t> bytes(8 + header.size() + (size_t)total, 0);
    const uint64_t hlen = (uint64_t)header.size();
    std::memcpy(bytes.data(), &hlen, 8);
    std::memcpy(bytes.data() + 8, header.data(), header.size());
    // data section: all zeros is fine; the routing test only checks resolution.
    std::ofstream out(path, std::ios::binary);
    if (!out) return "";
    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
    out.close();
    return path;
}

}  // namespace

int main() {
    using dee::TensorResolver;
    printf("=== dee.cpp DeepSeek-V4 resolver test ===\n");

    // ---- dtype mapping ------------------------------------------------
    check("F32", dee::dtype_from_string("F32") == dee::DType::F32);
    check("F16", dee::dtype_from_string("F16") == dee::DType::F16);
    check("BF16", dee::dtype_from_string("BF16") == dee::DType::BF16);
    check("F8_E4M3", dee::dtype_from_string("F8_E4M3") == dee::DType::F8);
    check("F8_E5M2", dee::dtype_from_string("F8_E5M2") == dee::DType::F8);
    check("F8_E8M0", dee::dtype_from_string("F8_E8M0") == dee::DType::F8);
    check("I8 (packed FP4)", dee::dtype_from_string("I8") == dee::DType::I8);
    check("I64 (hash tid2eid)", dee::dtype_from_string("I64") == dee::DType::I64);
    check("UNKNOWN stays unknown", dee::dtype_from_string("Q4") == dee::DType::UNKNOWN);
    check("dtype_to_string(I8)", std::string(dee::dtype_to_string(dee::DType::I8)) == "I8");
    check("dtype_to_string(F8)", std::string(dee::dtype_to_string(dee::DType::F8)) == "F8");

    // ---- DeepSeek-V4 expert naming ------------------------------------
    check_eq("expert w1 (GATE_PROJ)",
             TensorResolver::v4_expert_tensor_name(6, 0, TensorResolver::GATE_PROJ),
             "layers.6.ffn.experts.0.w1.weight");
    check_eq("expert w3 (UP_PROJ)",
             TensorResolver::v4_expert_tensor_name(6, 0, TensorResolver::UP_PROJ),
             "layers.6.ffn.experts.0.w3.weight");
    check_eq("expert w2 (DOWN_PROJ)",
             TensorResolver::v4_expert_tensor_name(6, 0, TensorResolver::DOWN_PROJ),
             "layers.6.ffn.experts.0.w2.weight");
    check_eq("expert 255 layer 42",
             TensorResolver::v4_expert_tensor_name(42, 255, TensorResolver::UP_PROJ),
             "layers.42.ffn.experts.255.w3.weight");

    // ---- scale pairing ------------------------------------------------
    check_eq("w1 scale", TensorResolver::v4_expert_scale_name(6, 0, TensorResolver::GATE_PROJ),
             "layers.6.ffn.experts.0.w1.scale");
    check_eq("w3 scale", TensorResolver::v4_expert_scale_name(6, 0, TensorResolver::UP_PROJ),
             "layers.6.ffn.experts.0.w3.scale");

    // ---- shared expert naming -----------------------------------------
    check_eq("shared w1", TensorResolver::v4_shared_expert_tensor_name(6, TensorResolver::GATE_PROJ),
             "layers.6.ffn.shared_experts.w1.weight");
    check_eq("shared w2", TensorResolver::v4_shared_expert_tensor_name(6, TensorResolver::DOWN_PROJ),
             "layers.6.ffn.shared_experts.w2.weight");

    // ---- Ornith naming is unchanged (default dialect) -----------------
    check_eq("ornith gate unchanged",
             TensorResolver::expert_tensor_name(0, 0, TensorResolver::GATE_PROJ),
             "model.language_model.layers.0.mlp.experts.0.gate_proj.weight");

    // ---- dialect switch on an instance --------------------------------
    dee::TensorResolver resolver;
    check("default model is ORNITH", resolver.model() == TensorResolver::Model::ORNITH);
    resolver.set_model(TensorResolver::Model::DEEPSEEK_V4);
    check("set_model(DEEPSEEK_V4) takes effect",
          resolver.model() == TensorResolver::Model::DEEPSEEK_V4);

    // resolve_expert_scale on the ORNITH default must be !ok() even with no
    // shards registered (i.e. no scale lookup is attempted).
    dee::TensorResolver ornith_only;
    check("scale on ORNITH resolver is !ok()",
          !ornith_only.resolve_expert_scale(0, 0, TensorResolver::GATE_PROJ).ok());

    // ---- routing through the DEEPSEEK_V4 dialect (real shard) ----------
    const std::string shard_path = "tests/data/dsv4_mini_shard.safetensors";
    if (write_mini_v4_shard(shard_path).empty()) {
        printf("  [FAIL] could not write mini v4 shard\n");
        ++g_failures;
    } else {
        dee::WeightMmap mmap;
        check("open mini v4 shard", mmap.open(shard_path));
        if (mmap.is_open()) {
            printf("  tensors parsed = %zu\n", mmap.tensors().size());
            check("parsed all 7 mini v4 tensors", mmap.tensors().size() == 7);

            dee::TensorResolver v4;
            v4.register_shard(&mmap);
            v4.set_model(TensorResolver::Model::DEEPSEEK_V4);

            dee::TensorView w1 = v4.resolve_expert(6, 0, TensorResolver::GATE_PROJ);
            check("v4 resolve_expert(GATE_PROJ) ok", w1.ok());
            check("v4 w1 dtype is I8", w1.dtype == dee::DType::I8);
            check("v4 w1 nbytes == 16", w1.nbytes == 16);
            check("v4 w1 shape [4,4]", w1.shape.size() == 2 && w1.shape[0] == 4 && w1.shape[1] == 4);

            dee::TensorView w2 = v4.resolve_expert(6, 0, TensorResolver::DOWN_PROJ);
            check("v4 resolve_expert(DOWN_PROJ) ok", w2.ok());

            dee::TensorView s1 = v4.resolve_expert_scale(6, 0, TensorResolver::GATE_PROJ);
            check("v4 resolve_expert_scale(GATE_PROJ) ok", s1.ok());
            check("v4 w1.scale dtype is F8", s1.dtype == dee::DType::F8);
            check("v4 w1.scale shape [4,1]", s1.shape.size() == 2 && s1.shape[0] == 4 && s1.shape[1] == 1);

            check("v4 shared expert resolves",
                  v4.resolve_tensor("layers.6.ffn.shared_experts.w1.weight").ok());
            check("v4 hash tensor resolves",
                  v4.resolve_tensor("layers.6.hc_attn_fn").ok());
            check("v4 dspark tensor resolves",
                  v4.resolve_tensor("mtp.0.dspark_markov_w1.weight").ok());

            // Negative: ORNITH dialect must not find the v4 name via resolve_expert.
            dee::TensorResolver ornith_vs_v4;
            ornith_vs_v4.register_shard(&mmap);
            check("ORNITH dialect does not find v4 expert name",
                  !ornith_vs_v4.resolve_expert(6, 0, TensorResolver::GATE_PROJ).ok());

            mmap.close();
        }
    }

    printf("=== %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}
