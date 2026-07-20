// tests/test_weight_mmap.cpp
//
// Step 4 smoke test: open a (synthetic) safetensors shard, resolve
// Layer 0 / Expert 0 weights, print + verify the first 5 float values.
//
// Build (standalone, no cmake needed):
//   g++ -std=c++17 -I../include test_weight_mmap.cpp ../src/weight_mmap.cpp \
//       ../src/json_min.cpp -o test_weight_mmap
//
// Run:
//   ./test_weight_mmap [path-to-shard.safetensors]

#include "dee/weight_mmap.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(const char* what, bool cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_failures;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1)
        ? argv[1]
        : "tests/data/layer0_shard.safetensors";

    printf("=== dee.cpp Step 4 smoke test ===\n");
    printf("shard: %s\n", path.c_str());

    dee::WeightMmap mmap;
    check("open + mmap", mmap.open(path));
    if (!mmap.is_open()) {
        printf("abort: could not open shard\n");
        return 1;
    }
    printf("  file_size = %zu bytes, tensors parsed = %zu\n",
           mmap.file_size(), mmap.tensors().size());

    dee::TensorResolver resolver;
    resolver.register_shard(&mmap);

    // Expected values (from tests/gen_synthetic_shard.py)
    const float exp_gate[5] = {0.5f, -1.25f, 2.0f, -3.75f, 4.25f};
    const float exp_up[5]   = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    const float exp_down[5] = {0.125f, -0.25f, 0.375f, -0.5f, 0.625f};

    auto dump_and_check = [&](const char* label, const dee::TensorView& v,
                              const float* expected) {
        check((std::string("resolve ") + label).c_str(), v.ok());
        if (!v.ok()) return;
        printf("  %s: dtype=%s shape=[", label, dee::dtype_to_string(v.dtype));
        for (size_t i = 0; i < v.shape.size(); ++i)
            printf("%lld%s", (long long)v.shape[i], i+1<v.shape.size()?",":"");
        printf("] nbytes=%zu\n", v.nbytes);

        // BF16 -> float32, print first 5
        const uint16_t* p = reinterpret_cast<const uint16_t*>(v.data);
        printf("    first 5 values: ");
        bool good = true;
        for (int i = 0; i < 5; ++i) {
            float f = dee::bf16_to_f32(p[i]);
            printf("%.4f ", f);
            if (std::fabs(f - expected[i]) > 1e-2f) good = false;
        }
        printf("\n");
        check((std::string("values match (") + label + ")").c_str(), good);
    };

    dump_and_check("layer0/expert0/gate_proj",
                   resolver.resolve_expert(0, 0, dee::TensorResolver::GATE_PROJ),
                   exp_gate);
    dump_and_check("layer0/expert0/up_proj",
                   resolver.resolve_expert(0, 0, dee::TensorResolver::UP_PROJ),
                   exp_up);
    dump_and_check("layer0/expert0/down_proj",
                   resolver.resolve_expert(0, 0, dee::TensorResolver::DOWN_PROJ),
                   exp_down);

    // Negative control: a non-existent expert must report !ok()
    dee::TensorView missing =
        resolver.resolve_expert(0, 99, dee::TensorResolver::GATE_PROJ);
    check("missing expert reports !ok()", !missing.ok());

    printf("=== %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}
