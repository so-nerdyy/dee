// dee universality prototype: codec boundary (interface only).
//
// Mirrors ABSTRACTION_DESIGN.md section 6: one codec = one plugin with
// decode_tile / storage_bytes / validate / metadata_layout /
// supported_backend. STQ/IQ2 are NOT implemented here; the registry only
// reserves their ids for research/expert-precision. Not wired into
// production. Dependency-free C++17.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "expert_descriptor.hpp"

namespace dee_universality {

struct TileReq {
  const std::uint8_t* packed = nullptr;
  const std::uint8_t* scales = nullptr;  // nullptr when codec has none
  std::uint64_t out_elems = 0;
  std::uint64_t in_elems = 0;
  std::uint32_t scale_block = 0;
  float* dst = nullptr;  // out_elems * in_elems floats
};

class ExpertCodec {
 public:
  virtual ~ExpertCodec() = default;
  virtual std::string id() const = 0;
  // Fail-closed: false on unsupported shape, never silent mis-decode.
  virtual bool decode_tile(const TileReq& req) const = 0;
  virtual std::uint64_t storage_bytes(std::uint64_t out,
                                      std::uint64_t in) const = 0;
  virtual bool validate(const ProjectionDesc& p, std::string& err) const = 0;
  virtual std::string metadata_layout() const = 0;
  // 4-bit weights NEVER imply native FP4 execution; each codec declares
  // backends it can actually run ("cuda-fp16", "cuda-fp32", "cpu-f32").
  virtual bool supported_backend(const std::string& backend) const = 0;
};

class CodecRegistry {
 public:
  void add(std::shared_ptr<ExpertCodec> codec) {
    codecs_[codec->id()] = std::move(codec);
  }
  std::shared_ptr<ExpertCodec> get(const std::string& id) const {
    const auto it = codecs_.find(id);
    return it == codecs_.end() ? nullptr : it->second;
  }
  std::vector<std::string> ids() const {
    std::vector<std::string> out;
    for (const auto& kv : codecs_) out.push_back(kv.first);
    return out;
  }

 private:
  std::map<std::string, std::shared_ptr<ExpertCodec>> codecs_;
};

}  // namespace dee_universality
