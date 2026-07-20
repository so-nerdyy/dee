// dee/pt_loader.h
//
// Step 7 — minimal PyTorch .pt reader (dependency-free).
//
// oracle.pt is a PyTorch ZIP archive: a pickle (oracle/data.pkl) describing
// tensor metadata, plus raw storage buffers (oracle/data/N). This reader
// parses the ZIP (handling DEFLATE via zlib) and a focused subset of the
// pickle opcodes to recover, per tensor, its (name, storage buffer index,
// byte offset, shape, dtype). It does NOT depend on libtorch.
//
// The Oracle architecture is fixed and known (Linear(2048,256) -> ReLU ->
// Linear(256,256) -> ReLU -> Linear(256,256)), so tensor shapes are derived
// from the tensor name + this architecture; the pickle only needs to supply
// the name -> storage-buffer mapping and any non-zero in-buffer offset.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace dee {

struct PtTensor {
    std::string name;     // e.g. "layers.0.net.0.weight"
    int         storage   = -1;   // index N for oracle/data/N
    size_t      offset    = 0;    // byte offset within the storage buffer
    std::vector<int64_t> shape;
    int         ndim      = 0;
    // dtype: 0 = F32 (FloatStorage). Oracle is all float32.
    int         dtype     = 0;    // 0=F32
    size_t      nbytes() const { size_t n = 1; for (auto d : shape) n *= (size_t)d; return n * 4; }
};

// Maps the layer local name ("net.0.weight") to its shape given the fixed arch.
// D=2048 (hidden), H=256 (mlp), E=256 (experts).
std::vector<int64_t> oracle_tensor_shape(const std::string& local_name,
                                          int D = 2048, int H = 256, int E = 256);

class PtLoader {
public:
    ~PtLoader() { close(); }

    // Open + parse a PyTorch .pt (zip) archive. Returns false on failure.
    bool open(const std::string& path);

    void close();

    bool is_open() const { return !path_.empty() && !tensors_.empty(); }

    // All tensors found (keyed by full name).
    const std::unordered_map<std::string, PtTensor>& tensors() const { return tensors_; }

    // Read a tensor's float32 data into `out` (caller allocates nbytes()/4 floats).
    // Returns false if the tensor is missing or the buffer is unreadable.
    bool read_tensor(const std::string& name, float* out) const;

    // Convenience: read into a std::vector<float>.
    bool read_tensor(const std::string& name, std::vector<float>& out) const;

    // Dump a manifest (name -> storage, offset, shape) as JSON to `path`
    // (used by the torch-free Python test harness for cross-validation).
    bool dump_manifest(const std::string& path) const;

    const std::string& error() const { return err_; }

private:
    std::string path_;
    std::string err_;

    // inflated storage buffers: index N -> bytes of oracle/data/N
    std::unordered_map<int, std::vector<uint8_t>> buffers_;
    std::unordered_map<std::string, PtTensor>     tensors_;
    std::vector<uint8_t>                          pkl_bytes_; // deferred pickle

    bool parse_zip(const std::vector<uint8_t>& raw);
    bool inflate_entry(const uint8_t* deflated, size_t deflated_len,
                       std::vector<uint8_t>& out);
    bool parse_pickle(const std::vector<uint8_t>& pkl);
};

} // namespace dee
