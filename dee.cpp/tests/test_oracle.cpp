// tests/test_oracle.cpp
//
// Step 7 — Oracle loader + OracleScheduler test.
//
// Validates: (1) PtLoader parses the REAL oracle.pt (ZIP+deflate+pickle) and
// recovers 40 layers x 6 tensors with correct names/shapes/buffer sizes;
// (2) a known tensor's contents are finite and match a torch-free Python
// reference (cross-checked via dump_manifest + python harness);
// (3) OracleScheduler runs the 3-layer MLP and returns a valid top-K.
//
// Build (no cmake):
//   g++ -std=c++17 -O2 -I../include test_oracle.cpp \
//       ../src/pt_loader.cpp ../src/oracle.cpp -lz -o test_oracle

#include "dee/pt_loader.h"
#include "dee/oracle.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(const char* what, bool cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_fail;
}

static bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

int main(int argc, char** argv) {
    printf("=== dee.cpp Step 7 Oracle test ===\n");
    std::string oracle_path = (argc > 1) ? argv[1]
        : "/mnt/c/Users/carth/Downloads/dynamic_expert_eviction/oracle.pt";

    // ---- PtLoader ----
    dee::PtLoader loader;
    check("open oracle.pt", loader.open(oracle_path));
    if (g_fail) return 1;
    const auto& T = loader.tensors();
    printf("  tensors found: %zu\n", T.size());
    check("240 tensors (40 layers x 6)", T.size() == 240);

    // name + shape + buffer-size consistency for every tensor
    bool names_ok = true, shapes_ok = true, sizes_ok = true;
    std::set<int> layers_seen;
    for (auto& kv : T) {
        const dee::PtTensor& t = kv.second;
        // expected name pattern layers.L.net.{0,2,4}.{weight,bias}
        std::string p = t.name;
        bool ok = p.rfind("layers.", 0) == 0 &&
                  (p.find(".net.0.weight") != std::string::npos ||
                   p.find(".net.0.bias")   != std::string::npos ||
                   p.find(".net.2.weight") != std::string::npos ||
                   p.find(".net.2.bias")   != std::string::npos ||
                   p.find(".net.4.weight") != std::string::npos ||
                   p.find(".net.4.bias")   != std::string::npos);
        if (!ok) names_ok = false;
        int L = std::atoi(p.substr(7, p.find('.', 7) - 7).c_str());
        layers_seen.insert(L);
        // shape check
        if (p.find("net.0.") != std::string::npos) {
            if (p.find("bias") != std::string::npos) {
                if (t.shape != std::vector<int64_t>{256}) shapes_ok = false;
            } else {
                if (t.shape != std::vector<int64_t>{256, 2048}) shapes_ok = false;
            }
        } else { // net.2 / net.4
            if (p.find("bias") != std::string::npos) {
                if (t.shape != std::vector<int64_t>{256}) shapes_ok = false;
            } else {
                if (t.shape != std::vector<int64_t>{256, 256}) shapes_ok = false;
            }
        }
    }
    check("all tensor names match layers.L.net.{0,2,4}.{weight,bias}", names_ok);
    check("exactly 40 layer indices", layers_seen.size() == 40);
    check("all shapes correct (256x2048 / 256x256 / 256)", shapes_ok);

    // dump manifest for the python cross-check
    std::string manifest = "/tmp/oracle_manifest.json";
    check("dump manifest", loader.dump_manifest(manifest));

    // ---- read a known tensor + buffer-size sanity ----
    std::vector<float> w00;
    check("read layers.0.net.0.weight", loader.read_tensor("layers.0.net.0.weight", w00));
    check("net.0.weight has 256*2048 floats", w00.size() == 256UL * 2048UL);
    check("net.0.weight values finite", all_finite(w00));
    // every storage buffer size must equal nbytes (offset 0 case)
    bool bufsz_ok = true;
    for (auto& kv : T) {
        const dee::PtTensor& t = kv.second;
        // buffers_ is private; instead verify via read_tensor length == nbytes/4
        std::vector<float> tmp;
        if (!loader.read_tensor(t.name, tmp) || tmp.size() != t.nbytes()/4) { bufsz_ok = false; break; }
    }
    check("every tensor reads exactly nbytes", bufsz_ok);

    // ---- OracleScheduler ----
    dee::OracleScheduler oracle;
    check("oracle.load", oracle.load(oracle_path));
    if (g_fail) return 1;
    check("oracle.num_layers == 40", oracle.num_layers() == 40);
    check("oracle.num_experts == 256", oracle.num_experts() == 256);

    // deterministic synthetic hidden (not random — reproducible)
    std::vector<float> hidden(2048);
    for (int i = 0; i < 2048; ++i) hidden[i] = std::sin(0.01f * i) * 0.5f;

    std::vector<int> topk;
    oracle.predict(0, hidden.data(), 8, topk);
    check("predict returns 8 experts", topk.size() == 8);
    bool in_range = true;
    std::set<int> uniq;
    for (int e : topk) { if (e < 0 || e >= 256) in_range = false; uniq.insert(e); }
    check("all predicted experts in [0,256)", in_range);
    check("predicted experts distinct", uniq.size() == 8);

    // forward produces 256 finite logits
    std::vector<float> logits;
    oracle.forward(0, hidden.data(), logits);
    check("forward -> 256 logits", logits.size() == 256);
    check("logits finite", all_finite(logits));

    // write the predicted top-K (layer 0) + hidden + logits for the torch-free
    // Python cross-check (rebuilds weights from raw oracle/data/N buffers).
    {
        std::ofstream f("/tmp/oracle_cpp_topk.txt");
        for (int e : topk) f << e << " ";
        f << "\n";
        // also dump full logits
        std::ofstream g("/tmp/oracle_cpp_logits.txt");
        for (float v : logits) g << v << "\n";
        // dump hidden so the Python reference uses the EXACT same input
        std::ofstream h("/tmp/oracle_cpp_hidden.txt");
        for (float v : hidden) h << v << "\n";
        // combined JSON for the harness
        std::ofstream j("/tmp/oracle_cpp_check.json");
        j << "{\"layer\":0,\"topk\":" << (int)topk.size() << ",\"predicted\":[";
        for (size_t k = 0; k < topk.size(); ++k) j << (k?",":"") << topk[k];
        j << "],\"hidden\":[";
        for (size_t k = 0; k < hidden.size(); ++k) j << (k?",":"") << hidden[k];
        j << "]}";
    }

    // sanity: logits should not all be identical (MLP is non-degenerate)
    float mn = logits[0], mx = logits[0];
    for (float v : logits) { mn = std::min(mn, v); mx = std::max(mx, v); }
    check("logits have spread (max-min > 1e-3)", (mx - mn) > 1e-3f);

    printf("=== %s ===\n", g_fail == 0 ? "ALL PASS" : "FAILURES");
    return g_fail == 0 ? 0 : 1;
}
