// dee universality prototype: expert descriptor (interface only).
//
// Not wired into production. Proves a generic ExpertDescriptor can express
// the current DSV4 expert and a structurally different second-model expert
// while cache/scheduler APIs see only CacheKey (id + bytes + codec).
// Dependency-free C++17; see test_universality.cpp.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dee_universality {

struct ExpertId {
  int layer = -1;
  int expert = -1;
  bool operator==(const ExpertId& o) const {
    return layer == o.layer && expert == o.expert;
  }
  bool operator<(const ExpertId& o) const {
    return layer < o.layer || (layer == o.layer && expert < o.expert);
  }
};

// Everything the byte caches / scheduler / prefetcher may know.
struct CacheKey {
  ExpertId id;
  std::uint64_t record_bytes = 0;
  std::string codec;  // CodecId, e.g. "mxfp4-e2m1-e8m0"
  double exec_cost_hint = 0.0;
};

struct ProjectionDesc {
  std::string role;  // informational only; never branched on by core
  std::uint64_t out_elems = 0;
  std::uint64_t in_elems = 0;
  std::uint64_t weight_offset = 0;
  std::uint64_t weight_nbytes = 0;
  std::uint64_t scale_offset = 0;
  std::uint64_t scale_nbytes = 0;  // 0/0 when the codec carries no scales
  std::uint32_t scale_block = 0;
};

struct CombineDesc {
  bool weighted_sum = true;
  bool residual_add = false;  // owned by dense frontend unless true
  bool shared_expert_add = false;
  std::string norm_formula;  // "rms-plus1" | "rms-plain" | "none"
};

struct ExpertDescriptor {
  ExpertId id;
  std::string codec;
  std::vector<ProjectionDesc> projections;  // N entries, any order
  std::uint64_t record_bytes = 0;
  std::string activation;  // e.g. "silu-clamp10", "silu", "gelu"
  CombineDesc combine;
  std::map<std::string, std::string> model_meta;  // never read by core

  bool ok(std::string& err) const {
    if (projections.empty()) {
      err = "descriptor needs >=1 projection";
      return false;
    }
    std::uint64_t span = 0;
    for (const auto& p : projections) {
      if (p.out_elems == 0 || p.in_elems == 0) {
        err = "projection has zero extent: " + p.role;
        return false;
      }
      const std::uint64_t end =
          p.weight_offset + p.weight_nbytes > p.scale_offset + p.scale_nbytes
              ? p.weight_offset + p.weight_nbytes
              : p.scale_offset + p.scale_nbytes;
      if (end > span) span = end;
    }
    if (span > record_bytes) {
      err = "projections overrun record_bytes";
      return false;
    }
    if (record_bytes == 0 || codec.empty()) {
      err = "record_bytes and codec are required";
      return false;
    }
    return true;
  }

  // The ONLY view the generic core is allowed to take.
  CacheKey cache_key() const {
    CacheKey k;
    k.id = id;
    k.record_bytes = record_bytes;
    k.codec = codec;
    return k;
  }
};

}  // namespace dee_universality
