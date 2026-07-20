#include "dee/engine.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace {

void usage(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n"
        "  --help             Show this help text\n"
        "  --shard PATH       safetensors shard (default: tests/data/ornith_moe256.safetensors)\n"
        "  --oracle PATH      Oracle model (default: oracle.pt)\n"
        "  --tokens N         Measured autoregressive tokens (default: 32)\n"
        "  --warmup N         Warmup tokens before measurement (default: 2)\n"
        "  --topk N           Experts per layer (default: 8)\n"
        "  --layers N         Decoder layers (default: 40, clamped to oracle)\n"
        "  --budget BYTES     Expert-cache budget (0 = four expert blobs)\n"
        "  --cuda             Use CUDA; requires a DEE_CUDA build and a GPU\n"
        "  --verbose          Print additional configuration details\n",
        argv0);
}

bool parse_positive(const char* text, const char* option, int& out, bool allow_zero = false) {
    if (!text || !*text) {
        std::fprintf(stderr, "[cli] %s requires an integer value\n", option);
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno || !end || *end || value < (allow_zero ? 0 : 1) || value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "[cli] invalid %s value: %s\n", option, text);
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool parse_size(const char* text, size_t& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || !end || *end || value > std::numeric_limits<size_t>::max()) {
        std::fprintf(stderr, "[cli] invalid --budget value: %s\n", text);
        return false;
    }
    out = static_cast<size_t>(value);
    return true;
}

const char* require_value(int& index, int argc, char** argv, const char* option) {
    if (index + 1 >= argc) {
        std::fprintf(stderr, "[cli] %s requires a value\n", option);
        return nullptr;
    }
    return argv[++index];
}

void print_result(const dee::EngineConfig& cfg, const dee::EngineStats& stats, int warmup) {
    std::printf("\n=== dee.cpp benchmark ===\n");
    if (cfg.use_cuda) {
        std::printf("CUDA device name       : %s\n", stats.cuda_device_name.c_str());
        std::printf("compute capability     : %d.%d\n", stats.cuda_compute_major, stats.cuda_compute_minor);
        std::printf("CUDA runtime version   : %d\n", stats.cuda_runtime_version);
    } else {
        std::printf("backend                : CPU/mock\n");
    }
    std::printf("token count            : %d\n", stats.tokens);
    std::printf("warmup count           : %d\n", warmup);
    std::printf("elapsed measured time  : %.6f s\n", stats.elapsed_sec);
    std::printf("tokens per second      : %.3f\n", stats.tok_per_sec);
    std::printf("configured cache budget: %zu bytes\n", cfg.budget_bytes);
    std::printf("peak expert-cache VRAM : %zu bytes\n", stats.peak_vram);
    std::printf("cache hits             : %llu\n", static_cast<unsigned long long>(stats.cache_hits));
    std::printf("cache loads            : %llu\n", static_cast<unsigned long long>(stats.cache_loads));
    std::printf("evictions              : %llu\n", static_cast<unsigned long long>(stats.evictions));
    std::printf("prefetch requests      : %llu\n", static_cast<unsigned long long>(stats.prefetch_issued));
    std::printf("prefetch fallbacks     : %llu\n", static_cast<unsigned long long>(stats.prefetch_fallbacks));
    std::printf("output finite status   : %s\n", stats.hidden_finite ? "yes" : "NO");
}

}  // namespace

int main(int argc, char** argv) {
    dee::EngineConfig cfg;
    cfg.shard_path = "tests/data/ornith_moe256.safetensors";
    cfg.oracle_path = "oracle.pt";
    int warmup_tokens = 2;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        if (arg == "--cuda") { cfg.use_cuda = true; continue; }
        if (arg == "--verbose") { cfg.verbose = true; continue; }
        const char* value = nullptr;
        if (arg == "--shard") {
            if (!(value = require_value(i, argc, argv, "--shard"))) return 2;
            cfg.shard_path = value;
        } else if (arg == "--oracle") {
            if (!(value = require_value(i, argc, argv, "--oracle"))) return 2;
            cfg.oracle_path = value;
        } else if (arg == "--tokens") {
            if (!(value = require_value(i, argc, argv, "--tokens")) || !parse_positive(value, "--tokens", cfg.num_tokens)) return 2;
        } else if (arg == "--warmup") {
            if (!(value = require_value(i, argc, argv, "--warmup")) || !parse_positive(value, "--warmup", warmup_tokens, true)) return 2;
        } else if (arg == "--topk") {
            if (!(value = require_value(i, argc, argv, "--topk")) || !parse_positive(value, "--topk", cfg.topk)) return 2;
        } else if (arg == "--layers") {
            if (!(value = require_value(i, argc, argv, "--layers")) || !parse_positive(value, "--layers", cfg.num_layers)) return 2;
        } else if (arg == "--budget") {
            if (!(value = require_value(i, argc, argv, "--budget")) || !parse_size(value, cfg.budget_bytes)) return 2;
        } else {
            std::fprintf(stderr, "[cli] unknown option: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

#ifndef DEE_CUDA
    if (cfg.use_cuda) {
        std::fprintf(stderr, "[cli] --cuda was requested, but this binary was built with DEE_CUDA=OFF. Reconfigure with -DDEE_CUDA=ON.\n");
        return 2;
    }
#endif

    if (warmup_tokens > 0) {
        dee::EngineConfig warmup = cfg;
        warmup.num_tokens = warmup_tokens;
        dee::Engine warmup_engine;
        if (!warmup_engine.init(warmup) || !warmup_engine.generate()) return 1;
    }

    dee::Engine engine;
    if (!engine.init(cfg)) return 1;
    if (!engine.generate()) {
        std::fprintf(stderr, "[cli] benchmark failed\n");
        return 1;
    }
    print_result(engine.config(), engine.stats(), warmup_tokens);
    return engine.stats().hidden_finite ? 0 : 1;
}
