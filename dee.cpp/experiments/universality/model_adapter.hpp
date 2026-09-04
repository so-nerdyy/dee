// dee universality prototype: model adapter (interface only).
//
// Replaces the TensorResolver dialect switch and the
// transfer_dtype==Fp4 -> DEEPSEEK_V4 conflation (engine.cpp:3452) with an
// explicit per-model implementation. Dsv4Adapter preserves reference
// behavior; SecondModelAdapter is a STRUCTURAL sketch with illustrative
// geometry (not a real checkpoint — see SECOND_MODEL_STRESS_TEST.md).
// Not wired into production. Dependency-free C++17.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "expert_descriptor.hpp"

namespace dee_universality {

struct ModelGeometry {
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  int num_layers = 0;
  int num_experts = 0;
  int top_k = 0;
  int num_shared = 0;
};

struct RouterDesc {
  std::string scoring = "softmax";  // softmax | sqrtsoftplus | sigmoid
  int top_k = 1;
  bool renorm = true;
  double scale = 1.0;  // 1.5 for DSV4, 1.0 elsewhere
  bool bias_select_only = false;
  bool hash_routing = false;
  double swiglu_limit = 0.0;  // 10.0 DSV4, 0.0 Ornith-style
};

class ModelAdapter {
 public:
  virtual ~ModelAdapter() = default;
  virtual std::string name() const = 0;
  virtual bool parse_config(const std::string& json, ModelGeometry& out,
                            std::string& err) const = 0;
  virtual std::string expert_weight_name(int layer, int expert,
                                         int proj) const = 0;
  virtual std::string expert_scale_name(int layer, int expert,
                                        int proj) const = 0;  // "" if none
  virtual std::string router_weight_name(int layer) const = 0;
  virtual RouterDesc router() const = 0;
  virtual std::vector<ExpertDescriptor> describe_layer(
      int layer) const = 0;
  virtual CombineDesc combine() const = 0;
};

// Reference behavior for DeepSeek-V4-Flash-0731 (structural sketch: names,
// counts, and record size mirror the audited sources; numerics stay in the
// Python reference, not here).
class Dsv4Adapter : public ModelAdapter {
 public:
  std::string name() const override { return "dsv4-flash-0731"; }
  bool parse_config(const std::string&, ModelGeometry& out,
                    std::string&) const override {
    out.hidden = 4096;
    out.intermediate = 2048;
    out.num_layers = 43;
    out.num_experts = 256;
    out.top_k = 6;
    out.num_shared = 1;
    return true;
  }
  std::string expert_weight_name(int l, int e, int p) const override {
    static const char* k[3] = {"w1", "w3", "w2"};
    return "layers." + std::to_string(l) + ".ffn.experts." +
           std::to_string(e) + "." + k[p % 3] + ".weight";
  }
  std::string expert_scale_name(int l, int e, int p) const override {
    static const char* k[3] = {"w1", "w3", "w2"};
    return "layers." + std::to_string(l) + ".ffn.experts." +
           std::to_string(e) + "." + k[p % 3] + ".scale";
  }
  std::string router_weight_name(int) const override {
    return "<external-hf-router>";  // DSV4 routes outside the MoE server
  }
  RouterDesc router() const override {
    RouterDesc r;
    r.scoring = "sqrtsoftplus";
    r.top_k = 6;
    r.renorm = true;
    r.scale = 1.5;
    r.swiglu_limit = 10.0;
    return r;
  }
  std::vector<ExpertDescriptor> describe_layer(int layer) const override {
    ExpertDescriptor d;
    d.id = {layer, 0};
    d.codec = "mxfp4-e2m1-e8m0";
    d.record_bytes = 13369344;  // 12.75 MiB packed + e8m0, per P2.3 doc
    d.activation = "silu-clamp10";
    const std::uint64_t w = 2048ull * 4096ull / 2;  // packed bytes per proj
    const std::uint64_t s = 2048ull * 4096ull / 32;  // e8m0 scales per proj
    const char* roles[3] = {"gate", "up", "down"};
    std::uint64_t off = 0, soff = 3 * w;
    for (int p = 0; p < 3; ++p) {
      ProjectionDesc pr;
      pr.role = roles[p];
      pr.out_elems = (p == 2) ? 4096 : 2048;
      pr.in_elems = (p == 2) ? 2048 : 4096;
      pr.weight_offset = off;
      pr.weight_nbytes = w;
      pr.scale_offset = soff;
      pr.scale_nbytes = s;
      pr.scale_block = 32;
      d.projections.push_back(pr);
      off += w;
      soff += s;
    }
    CombineDesc c;
    c.weighted_sum = true;
    c.shared_expert_add = true;
    c.norm_formula = "rms-plain";
    d.combine = c;
    return {d};
  }
  CombineDesc combine() const override {
    CombineDesc c;
    c.weighted_sum = true;
    c.shared_expert_add = true;
    c.norm_formula = "rms-plain";
    return c;
  }
};

// Structural sketch of a DIFFERENT MoE (illustrative geometry only — proves
// the same interfaces express reordered projections, missing scales, and a
// different norm/combine without touching core APIs).
class SecondModelAdapter : public ModelAdapter {
 public:
  std::string name() const override { return "second-model-sketch"; }
  bool parse_config(const std::string&, ModelGeometry& out,
                    std::string&) const override {
    out.hidden = 2048;
    out.intermediate = 1024;
    out.num_layers = 24;
    out.num_experts = 64;
    out.top_k = 4;
    out.num_shared = 0;
    return true;
  }
  std::string expert_weight_name(int l, int e, int p) const override {
    static const char* k[3] = {"up_proj", "gate_proj", "down_proj"};
    return "model.layers." + std::to_string(l) + ".mlp.experts." +
           std::to_string(e) + "." + k[p % 3] + ".weight";
  }
  std::string expert_scale_name(int, int, int) const override {
    return "";  // native precision: no per-projection scales
  }
  std::string router_weight_name(int l) const override {
    return "model.layers." + std::to_string(l) + ".mlp.gate.weight";
  }
  RouterDesc router() const override {
    RouterDesc r;  // plain softmax router, no scale, no clamp
    r.scoring = "softmax";
    r.top_k = 4;
    return r;
  }
  std::vector<ExpertDescriptor> describe_layer(int layer) const override {
    ExpertDescriptor d;
    d.id = {layer, 0};
    d.codec = "bf16";
    d.activation = "silu";
    const std::uint64_t elems = 1024ull * 2048ull * 2;
    const char* roles[3] = {"up", "gate", "down"};  // deliberately reordered
    std::uint64_t off = 0;
    for (int p = 0; p < 3; ++p) {
      ProjectionDesc pr;
      pr.role = roles[p];
      pr.out_elems = (p == 2) ? 2048 : 1024;
      pr.in_elems = (p == 2) ? 1024 : 2048;
      pr.weight_offset = off;
      pr.weight_nbytes = elems;
      pr.scale_block = 0;
      d.projections.push_back(pr);
      off += elems;
    }
    d.record_bytes = off;
    CombineDesc c;
    c.weighted_sum = true;
    c.norm_formula = "rms-plus1";
    d.combine = c;
    return {d};
  }
  CombineDesc combine() const override {
    CombineDesc c;
    c.weighted_sum = true;
    c.norm_formula = "rms-plus1";
    return c;
  }
};

}  // namespace dee_universality
