// dee/weight_mmap.cpp
#include "dee/weight_mmap.h"
#include "dee/json_min.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace dee {

// ---- dtype helpers --------------------------------------------------------
DType dtype_from_string(const std::string& s) {
    if (s == "F32" || s == "F32E4M3" || s == "F32E5M2") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    return DType::UNKNOWN;
}
const char* dtype_to_string(DType d) {
    switch (d) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::BF16: return "BF16";
        default:          return "UNKNOWN";
    }
}

// BF16 (top 16 bits of float32) -> float32
float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// IEEE half (F16) -> float32
float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) u = sign << 31;
        else {
            int e = 0;
            while ((mant & 0x400) == 0) { mant <<= 1; --e; }
            mant &= 0x3ff;
            u = (sign << 31) | ((127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        u = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        u = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---- WeightMmap -----------------------------------------------------------
WeightMmap::WeightMmap() = default;

WeightMmap::~WeightMmap() { close(); }

void WeightMmap::close() {
    if (base_ && base_ != MAP_FAILED) munmap(base_, size_);
    base_ = nullptr; size_ = 0;
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    tensors_.clear();
    header_json_.clear();
}

bool WeightMmap::map_file(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { fprintf(stderr, "WeightMmap: open failed: %s\n", path.c_str()); return false; }
    struct stat st;
    if (fstat(fd_, &st) != 0) { fprintf(stderr, "WeightMmap: fstat failed\n"); return false; }
    size_ = (size_t)st.st_size;
    if (size_ < 8) { fprintf(stderr, "WeightMmap: file too small\n"); return false; }
    base_ = (uint8_t*)mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (base_ == MAP_FAILED) { fprintf(stderr, "WeightMmap: mmap failed\n"); base_ = nullptr; return false; }
    // Random-access pattern (we slice experts out of order) — tell the kernel,
    // exactly as llama.cpp does with POSIX_MADV_RANDOM.
    posix_madvise(base_, size_, POSIX_MADV_RANDOM);
    return true;
}

bool WeightMmap::open(const std::string& path) {
    close();
    if (!map_file(path)) return false;
    // safetensors: first 8 bytes = uint64 LE header length.
    uint64_t hlen = 0;
    std::memcpy(&hlen, base_, 8);
    if (hlen == 0 || 8 + hlen > size_) { fprintf(stderr, "WeightMmap: bad header len\n"); return false; }
    header_json_.assign((const char*)(base_ + 8), (size_t)hlen);
    return parse_header_json(header_json_);
}

bool WeightMmap::open(const std::string& path, const std::string& json_header) {
    close();
    if (!map_file(path)) return false;
    header_json_ = json_header;
    return parse_header_json(json_header);
}

bool WeightMmap::parse_header() { return parse_header_json(header_json_); }

bool WeightMmap::parse_header_json(const std::string& json) {
    bool ok = false;
    auto root = json::parse(json, &ok);
    if (!ok || !root || !root->is_object()) {
        fprintf(stderr, "WeightMmap: header JSON parse failed\n");
        return false;
    }
    for (const auto& kv : root->obj) {
        const std::string& name = kv.first;
        const json::Value* v = kv.second.get();
        if (name == "__metadata__") continue;
        if (!v->is_object()) continue;
        const json::Value* dtype_v = v->find("dtype");
        const json::Value* shape_v = v->find("shape");
        const json::Value* off_v   = v->find("data_offsets");
        if (!dtype_v || !dtype_v->is_string() || !off_v || !off_v->is_array() ||
            off_v->arr.size() < 2) {
            continue;
        }
        TensorMeta m;
        m.dtype = dtype_from_string(dtype_v->s);
        if (shape_v && shape_v->is_array()) {
            for (const auto& d : shape_v->arr) if (d->is_int()) m.shape.push_back((int64_t)d->i);
        }
        long long start = off_v->arr[0]->i;
        long long end   = off_v->arr[1]->i;
        m.data_offset = (size_t)start;
        m.nbytes      = (size_t)(end - start);
        tensors_[name] = m;
    }
    return true;
}

TensorView WeightMmap::lookup(const std::string& tensor_name) const {
    TensorView view;
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) return view;
    const TensorMeta& m = it->second;
    // data section begins at file offset (8 + header_len); but we stored
    // data_offset relative to the data section, so absolute = base + 8 + hlen + off.
    // We recompute hlen from the live header.
    if (header_json_.empty()) return view;
    uint64_t hlen = 0;
    std::memcpy(&hlen, base_, 8);
    size_t abs = 8 + (size_t)hlen + m.data_offset;
    view.data   = base_ + abs;
    view.nbytes = m.nbytes;
    view.dtype  = m.dtype;
    view.shape  = m.shape;
    return view;
}

// ---- TensorResolver --------------------------------------------------------
std::string TensorResolver::expert_tensor_name(int layer, int expert, TensorResolver::Kind kind) {
    const char* k = nullptr;
    switch (kind) {
        case GATE_PROJ: k = "gate_proj"; break;
        case UP_PROJ:   k = "up_proj";   break;
        case DOWN_PROJ: k = "down_proj"; break;
    }
    char buf[256];
    snprintf(buf, sizeof(buf),
             "model.language_model.layers.%d.mlp.experts.%d.%s.weight",
             layer, expert, k);
    return std::string(buf);
}

void TensorResolver::register_shard(WeightMmap* mmap) {
    if (mmap) shards_.push_back(mmap);
}

TensorView TensorResolver::resolve_expert(int layer, int expert, Kind kind) const {
    std::string name = expert_tensor_name(layer, expert, kind);
    for (auto* sh : shards_) {
        TensorView v = sh->lookup(name);
        if (v.ok()) return v;
    }
    return TensorView{}; // !ok()
}

} // namespace dee
