// Standalone test for dee universality prototypes (not part of CMake).
// Build: g++ -std=c++17 -Wall -Wextra -I. test_universality.cpp -o test_universality
// Run: ./test_universality  (exit 0 = all checks pass)
#include <cstdio>
#include <string>

#include "expert_codec.hpp"
#include "expert_descriptor.hpp"
#include "model_adapter.hpp"

namespace du = dee_universality;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond)                                                 \
  do {                                                              \
    if (cond) {                                                     \
      ++g_pass;                                                     \
    } else {                                                        \
      ++g_fail;                                                     \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                               \
  } while (0)

// The generic core may only accept this view: proves no projection/model
// leakage if a descriptor change still compiles against this signature.
static std::uint64_t schedule_bytes(const du::CacheKey& k) {
  return k.record_bytes;
}

// Minimal test-double codecs (interface conformance only, no real decode).
struct FakeMxfp4 : du::ExpertCodec {
  std::string id() const override { return "mxfp4-e2m1-e8m0"; }
  bool decode_tile(const du::TileReq& r) const override {
    return r.packed && r.scales && r.dst && r.out_elems && r.in_elems &&
           r.scale_block == 32 && (r.in_elems % 64 == 0);
  }
  std::uint64_t storage_bytes(std::uint64_t o,
                              std::uint64_t i) const override {
    return o * i / 2 + o * i / 32;
  }
  bool validate(const du::ProjectionDesc& p, std::string& e) const override {
    if (p.scale_block != 32) {
      e = "mxfp4 needs block 32";
      return false;
    }
    return true;
  }
  std::string metadata_layout() const override {
    return "[gate][up][down][gate_s][up_s][down_s]";
  }
  bool supported_backend(const std::string& b) const override {
    return b == "cuda-fp16" || b == "cpu-f32";  // never native-fp4 exec
  }
};

struct FakeBf16 : du::ExpertCodec {
  std::string id() const override { return "bf16"; }
  bool decode_tile(const du::TileReq& r) const override {
    return r.packed && r.dst && r.out_elems && r.in_elems;
  }
  std::uint64_t storage_bytes(std::uint64_t o,
                              std::uint64_t i) const override {
    return o * i * 2;
  }
  bool validate(const du::ProjectionDesc&, std::string&) const override {
    return true;
  }
  std::string metadata_layout() const override { return "[up][gate][down]"; }
  bool supported_backend(const std::string& b) const override {
    return b == "cuda-fp16" || b == "cuda-fp32" || b == "cpu-f32";
  }
};

int main() {
  std::string err;
  du::Dsv4Adapter dsv4;
  du::SecondModelAdapter second;

  // 1. Adapters identify themselves and parse geometry.
  CHECK(dsv4.name() == "dsv4-flash-0731");
  CHECK(second.name() == "second-model-sketch");
  du::ModelGeometry g;
  CHECK(dsv4.parse_config("{}", g, err) && g.hidden == 4096 &&
        g.num_experts == 256 && g.top_k == 6);
  CHECK(second.parse_config("{}", g, err) && g.hidden == 2048 &&
        g.num_experts == 64 && g.top_k == 4);

  // 2. Tensor names differ per adapter (w1/w3/w2 vs *_proj, scales vs none).
  CHECK(dsv4.expert_weight_name(6, 0, 0) == "layers.6.ffn.experts.0.w1.weight");
  CHECK(dsv4.expert_weight_name(6, 0, 1) == "layers.6.ffn.experts.0.w3.weight");
  CHECK(!dsv4.expert_scale_name(6, 0, 0).empty());
  CHECK(second.expert_weight_name(6, 0, 0) ==
        "model.layers.6.mlp.experts.0.up_proj.weight");
  CHECK(second.expert_scale_name(6, 0, 0).empty());

  // 3. Both adapters emit valid descriptors; DSV4 record is 12.75 MiB.
  auto dd = dsv4.describe_layer(6)[0];
  auto sd = second.describe_layer(6)[0];
  CHECK(dd.ok(err) && sd.ok(err));
  CHECK(dd.record_bytes == 13369344u);
  CHECK(dd.projections.size() == 3 && sd.projections.size() == 3);
  CHECK(dd.projections[0].role == "gate");
  CHECK(sd.projections[0].role == "up");  // reordered roles expressible
  CHECK(sd.projections[0].scale_nbytes == 0);  // scaleless codec expressible

  // 4. Core sees only CacheKey: same scheduler path serves both models.
  CHECK(schedule_bytes(dd.cache_key()) == 13369344u);
  CHECK(schedule_bytes(sd.cache_key()) == sd.record_bytes);
  CHECK(dd.cache_key().codec == "mxfp4-e2m1-e8m0");
  CHECK(sd.cache_key().codec == "bf16");

  // 5. Invalid descriptors fail closed.
  du::ExpertDescriptor bad;
  CHECK(!bad.ok(err));
  du::ExpertDescriptor over = dd;
  over.record_bytes = 8;
  CHECK(!over.ok(err));

  // 6. Codec registry: two plugins coexist; unknown id -> nullptr.
  du::CodecRegistry reg;
  reg.add(std::make_shared<FakeMxfp4>());
  reg.add(std::make_shared<FakeBf16>());
  CHECK(reg.ids().size() == 2);
  auto mx = reg.get("mxfp4-e2m1-e8m0");
  auto bf = reg.get("bf16");
  CHECK(mx && bf);
  CHECK(reg.get("iq2") == nullptr);  // slot reserved, nothing claimed
  CHECK(mx->storage_bytes(2048, 4096) == 2048u * 4096u / 2 + 2048u * 4096u / 32);
  CHECK(bf->storage_bytes(1024, 2048) == 1024u * 2048u * 2);
  CHECK(!mx->supported_backend("cuda-fp4-native"));  // no native exec claim
  float dst[4] = {0, 0, 0, 0};
  std::uint8_t pk[2] = {0, 0}, sc[1] = {0};
  du::TileReq good{pk, sc, 2, 64, 32, dst};
  du::TileReq unaligned{pk, sc, 2, 48, 32, dst};
  CHECK(mx->decode_tile(good));
  CHECK(!mx->decode_tile(unaligned));  // fail-closed on shape
  CHECK(mx->metadata_layout() != bf->metadata_layout());

  // 7. Router/combine differ per model without core forks.
  CHECK(dsv4.router().scoring == "sqrtsoftplus" && dsv4.router().scale == 1.5 &&
        dsv4.router().swiglu_limit == 10.0);
  CHECK(second.router().scoring == "softmax" && second.router().scale == 1.0 &&
        second.router().swiglu_limit == 0.0);
  CHECK(dsv4.combine().shared_expert_add &&
        dsv4.combine().norm_formula == "rms-plain");
  CHECK(!second.combine().shared_expert_add &&
        second.combine().norm_formula == "rms-plus1");

  std::printf("universality prototypes: %d passed, %d failed\n", g_pass,
              g_fail);
  return g_fail == 0 ? 0 : 1;
}
