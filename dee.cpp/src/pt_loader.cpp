// dee/pt_loader.cpp
#include "dee/pt_loader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#  include <io.h>
#else
#  include <unistd.h>
#endif

// zlib (system library; present on Linux/macOS; on Windows use zlibwapi)
#include <zlib.h>

namespace dee {

// ---- architecture shape table ------------------------------------------------
std::vector<int64_t> oracle_tensor_shape(const std::string& local, int D, int H, int E) {
    // local names: net.0.weight / net.0.bias / net.2.weight / net.2.bias / net.4.weight / net.4.bias
    bool is_bias = local.find("bias") != std::string::npos;
    bool is_first = local.find("net.0.") != std::string::npos;
    bool is_last  = local.find("net.4.") != std::string::npos;
    if (is_bias) return { (int64_t)(is_first ? H : H) }; // bias is (out,)
    if (is_first)  return { (int64_t)H, (int64_t)D };     // (H, D)
    // hidden or output linear: (H, H)
    return { (int64_t)H, (int64_t)H };
}

// ---- file read helper --------------------------------------------------------
static bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz <= 0) return false;
    out.resize((size_t)sz);
    f.read((char*)out.data(), sz);
    return (bool)f || f.eof();
}

// ---- ZIP parsing -------------------------------------------------------------
// Minimal ZIP reader: find End-Of-Central-Directory, walk central directory
// entries, for each named entry find its local header + (optionally) inflate.
bool PtLoader::parse_zip(const std::vector<uint8_t>& raw) {
    const uint8_t* d = raw.data();
    size_t n = raw.size();
    // EOCD signature: 0x06054b50
    size_t eocd = std::string::npos;
    for (size_t i = n >= 22 ? n - 22 : 0; i + 22 <= n; --i) {
        if (d[i] == 0x50 && d[i+1] == 0x4b && d[i+2] == 0x05 && d[i+3] == 0x06) {
            eocd = i; break;
        }
    }
    if (eocd == std::string::npos) { err_ = "not a ZIP (no EOCD)"; return false; }
    uint32_t cd_offset = *(const uint32_t*)(d + eocd + 16);
    uint16_t cd_count  = *(const uint16_t*)(d + eocd + 10);

    size_t p = cd_offset;
    for (uint16_t e = 0; e < cd_count; ++e) {
        if (p + 46 > n) break;
        uint16_t method   = *(const uint16_t*)(d + p + 10);
        uint32_t comp_size = *(const uint32_t*)(d + p + 20);
        uint32_t uncomp    = *(const uint32_t*)(d + p + 24);
        uint16_t nlen      = *(const uint16_t*)(d + p + 28);
        uint16_t elen      = *(const uint16_t*)(d + p + 30);
        uint16_t clen      = *(const uint16_t*)(d + p + 32);
        std::string name((const char*)(d + p + 46), nlen);
        size_t lho = *(const uint32_t*)(d + p + 42); // local header offset
        p += 46 + nlen + elen + clen;

        // local header: find actual compressed data start
        if (lho + 30 > n) continue;
        uint16_t lnlen = *(const uint16_t*)(d + lho + 26);
        uint16_t lelen = *(const uint16_t*)(d + lho + 28);
        size_t data_off = lho + 30 + lnlen + lelen;
        if (data_off + comp_size > n) continue;

        if (name == "oracle/data.pkl") {
            std::vector<uint8_t> inflated;
            if (method == 0) {
                inflated.assign(d + data_off, d + data_off + comp_size);
            } else if (method == 8) {
                if (!inflate_entry(d + data_off, comp_size, inflated)) {
                    err_ = "inflate failed for data.pkl"; return false;
                }
            } else { continue; }
            // Defer pickle parsing until ALL oracle/data/N buffers are loaded,
            // because tensors are derived from storage-key indices (0..239).
            pkl_bytes_ = std::move(inflated);
        } else if (name.rfind("oracle/data/", 0) == 0) {
            std::string idx_s = name.substr(std::string("oracle/data/").size());
            int idx = std::atoi(idx_s.c_str());
            std::vector<uint8_t> inflated;
            if (method == 0) {
                inflated.assign(d + data_off, d + data_off + comp_size);
            } else if (method == 8) {
                if (!inflate_entry(d + data_off, comp_size, inflated)) {
                    err_ = "inflate failed for " + name; return false;
                }
            } else { continue; }
            buffers_[idx] = std::move(inflated);
            (void)uncomp;
        }
    }
    if (!pkl_bytes_.empty() && !parse_pickle(pkl_bytes_)) return false;
    return true;
}

bool PtLoader::inflate_entry(const uint8_t* src, size_t slen, std::vector<uint8_t>& out) {
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) return false; // raw deflate (no zlib header)
    zs.next_in = (Bytef*)src;
    zs.avail_in = (uInt)slen;
    out.clear();
    out.reserve(slen * 4 + 4096);
    uint8_t chunk[65536];
    int ret;
    do {
        zs.next_out = chunk;
        zs.avail_out = sizeof(chunk);
        ret = inflate(&zs, Z_NO_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END && ret != Z_BUF_ERROR) {
            inflateEnd(&zs); return false;
        }
        size_t got = sizeof(chunk) - zs.avail_out;
        out.insert(out.end(), chunk, chunk + got);
    } while (ret != Z_STREAM_END);
    inflateEnd(&zs);
    return true;
}


bool PtLoader::parse_pickle(const std::vector<uint8_t>& pkl) {
    // Tolerant scanner for the PyTorch .pt pickle produced by our Oracle
    // exporter. The format is highly regular: for each tensor the pickle emits
    // a BINUNICODE tensor-name ("net.0.weight", ...) and, a short distance
    // later, a decimal-digit storage key (the integer N in oracle/data/N).
    // Each storage buffer holds exactly one tensor (offset 0), so we only need
    // (name -> storage index).
    //
    // ROBUSTNESS: PyTorch may re-use a tensor-name string via memo references,
    // so a naive name scan can miss a few occurrences. We therefore derive the
    // canonical tensor list from the storage keys themselves: storage G in
    // [0, num_storage) maps to layer = G/6, local = G%6, with the local-name
    // table below (the emission order is fixed by the exporter). We still run
    // the name scan and VALIDATE it against the table; any mismatch is a real
    // format change and is reported as an error.
    const uint8_t* d = pkl.data();
    size_t n = pkl.size();

    auto read_binunicode = [&](size_t pos, std::string& out) -> bool {
        if (pos >= n) return false;
        uint8_t op = d[pos];
        if (op == 'X') { // BINUNICODE: 4-byte LE length
            if (pos + 5 > n) return false;
            uint32_t len = *(const uint32_t*)(d + pos + 1);
            if (pos + 5 + len > n) return false;
            out.assign((const char*)(d + pos + 5), len);
            return true;
        } else if (op == 0x8c) { // SHORT_BINUNICODE: 1-byte length
            if (pos + 2 > n) return false;
            uint8_t len = d[pos + 1];
            if (pos + 2 + len > n) return false;
            out.assign((const char*)(d + pos + 2), len);
            return true;
        }
        return false;
    };

    // canonical local names in emission order (per layer: 6 tensors)
    static const char* kLocalNames[6] = {
        "net.0.weight", "net.0.bias", "net.2.weight",
        "net.2.bias",   "net.4.weight", "net.4.bias"
    };
    auto local_shape = [](int local) -> std::vector<int64_t> {
        switch (local) {
            case 0: return {256, 2048}; // net.0.weight  (Linear 2048->256)
            case 1: return {256};       // net.0.bias
            case 2: return {256, 256};  // net.2.weight  (Linear 256->256)
            case 3: return {256};       // net.2.bias
            case 4: return {256, 256};  // net.4.weight  (Linear 256->256)
            case 5: return {256};       // net.4.bias
        }
        return {};
    };

    // 1) name scan -> (name, storage) for VALIDATION only
    struct Found { std::string name; int storage; };
    std::vector<Found> found;
    size_t i = 0;
    if (n > 1 && d[0] == 0x80) i = 2; // skip PROTO
    while (i < n) {
        if (d[i] == 'X' || d[i] == 0x8c) {
            std::string s;
            if (read_binunicode(i, s)) {
                bool is_tensor = (s.find("net.") == 0) &&
                    (s.find(".weight") != std::string::npos || s.find(".bias") != std::string::npos);
                if (is_tensor) {
                    int storage = -1;
                    size_t j = i + 1;
                    size_t limit = std::min(n, i + 200);
                    while (j < limit) {
                        std::string key;
                        if ((d[j] == 'X' || d[j] == 0x8c) && read_binunicode(j, key) && !key.empty()) {
                            bool alldigits = true;
                            for (char c : key) if (c < '0' || c > '9') { alldigits = false; break; }
                            if (alldigits) { storage = std::atoi(key.c_str()); break; }
                        }
                        ++j;
                    }
                    found.push_back({s, storage});
                }
                // advance past this string
                if (d[i] == 'X') { uint32_t len = *(const uint32_t*)(d + i + 1); i += 5 + len; }
                else { uint8_t len = d[i + 1]; i += 2 + len; }
                continue;
            }
        }
        ++i;
    }

    // 2) build the canonical tensor list from storage keys 0..(num_storage-1)
    int num_storage = (int)buffers_.size();
    for (int G = 0; G < num_storage; ++G) {
        int layer = G / 6;
        int local = G % 6;
        const char* lname = kLocalNames[local];
        std::string full = "layers." + std::to_string(layer) + "." + std::string(lname);
        PtTensor t;
        t.name = full;
        t.storage = G;
        t.offset = 0;
        t.shape = local_shape(local);
        t.ndim = (int)t.shape.size();
        tensors_[full] = t;
    }

    // 3) VALIDATE: every scanned (name, storage) must agree with the table.
    for (auto& f : found) {
        if (f.storage < 0 || f.storage >= num_storage) continue; // unscanned storage
        int local = f.storage % 6;
        if (std::string(kLocalNames[local]) != f.name) {
            err_ = "oracle tensor order mismatch at storage " + std::to_string(f.storage) +
                   ": expected " + kLocalNames[local] + " but pickle says " + f.name;
            return false;
        }
    }
    return true;
}
// ---- open / read -------------------------------------------------------------
bool PtLoader::open(const std::string& path) {
    close();
    path_ = path;
    std::vector<uint8_t> raw;
    if (!read_file(path, raw)) { err_ = "cannot read file"; return false; }
    if (raw.size() < 4 || raw[0] != 'P' || raw[1] != 'K') { err_ = "not a ZIP (bad magic)"; return false; }
    if (!parse_zip(raw)) return false;
    return !tensors_.empty();
}

void PtLoader::close() {
    buffers_.clear();
    tensors_.clear();
    path_.clear();
}

bool PtLoader::read_tensor(const std::string& name, float* out) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;
    const PtTensor& t = it->second;
    auto buf = buffers_.find(t.storage);
    if (buf == buffers_.end()) return false;
    const std::vector<uint8_t>& b = buf->second;
    size_t need = t.offset + t.nbytes();
    if (need > b.size()) {
        // tolerate offset misinterpretation: if offset makes it overflow but
        // offset==0 fits, use whole buffer.
        if (t.offset != 0 && t.nbytes() <= b.size()) {
            std::memcpy(out, b.data(), t.nbytes());
            return true;
        }
        return false;
    }
    std::memcpy(out, b.data() + t.offset, t.nbytes());
    return true;
}

bool PtLoader::read_tensor(const std::string& name, std::vector<float>& out) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;
    out.resize(it->second.nbytes() / 4);
    return read_tensor(name, out.data());
}

bool PtLoader::dump_manifest(const std::string& path) const {
    std::stringstream ss;
    ss << "{\n";
    bool first = true;
    for (auto& kv : tensors_) {
        const PtTensor& t = kv.second;
        if (!first) ss << ",\n";
        first = false;
        ss << "  \"" << t.name << "\": {\"storage\":" << t.storage
           << ", \"offset\":" << t.offset << ", \"shape\":[";
        for (size_t k = 0; k < t.shape.size(); ++k) ss << (k? ",":"") << t.shape[k];
        ss << "], \"nbytes\":" << t.nbytes() << "}";
    }
    ss << "\n}\n";
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << ss.str();
    return true;
}


} // namespace dee
