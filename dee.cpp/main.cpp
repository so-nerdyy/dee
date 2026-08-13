#include "dee/engine.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <sstream>
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
        "  --prefetch-depth N Bounded staging/transfer ring depth (default: 64)\n"
        "  --cache-dtype D    Device expert cache: fp16 with --cuda, or fp32 fallback\n"
        "  --transfer-dtype D Expert transfer: int8 CUDA default, bf16, or experimental int4\n"
        "  --hidden N         Model hidden dim (default: 2048, must match shard)\n"
        "  --inter N          Expert intermediate dim (default: 256, must match shard)\n"
        "  --cuda             Use CUDA; requires a DEE_CUDA build and a GPU\n"
        "  --profile-stages   Enable detailed CPU/CUDA stage timing\n"
        "  --profile-scenario M Controlled profile: end-to-end, full-resident,\n"
        "                       resident-bypass, transfer-only, compute-only,\n"
        "                       oracle-only, or cache-metadata-only\n"
        "  --profile-json P   Write benchmark and stage summary JSON to P\n"
        "  --profile-timeline P Write common-origin CUDA Chrome trace JSON to P\n"
        "  --trace-requests P Write detailed expert-request trace JSON to P\n"
        "  --dynamic-quantization Quantize experts on first use instead of startup prepack\n"
        "  --oracle-margin F  GPU Oracle boundary-aware strict routing margin threshold\n"
        "                      (sigmoid logit units; >0 enables CPU fallback for tight margins,\n"
        "                      0 = raw GPU mode which may diverge from CPU routing)\n"
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

bool parse_size(const char* text, const char* option, size_t& out) {
    if (!text || !*text) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno || !end || *end || value > std::numeric_limits<size_t>::max()) {
        std::fprintf(stderr, "[cli] invalid %s value: %s\n", option, text);
        return false;
    }
    out = static_cast<size_t>(value);
    return true;
}

bool parse_scenario(const char* text, dee::BenchmarkScenario& out) {
    if (!text) return false;
    const std::string value(text);
    if (value == "end-to-end") out = dee::BenchmarkScenario::EndToEnd;
    else if (value == "full-resident") out = dee::BenchmarkScenario::FullResident;
    else if (value == "resident-bypass") out = dee::BenchmarkScenario::ResidentBypass;
    else if (value == "transfer-only") out = dee::BenchmarkScenario::TransferOnly;
    else if (value == "compute-only") out = dee::BenchmarkScenario::ComputeOnly;
    else if (value == "oracle-only") out = dee::BenchmarkScenario::OracleOnly;
    else if (value == "cache-metadata-only") out = dee::BenchmarkScenario::CacheMetadataOnly;
    else {
        std::fprintf(stderr, "[cli] invalid --profile-scenario value: %s\n", text);
        return false;
    }
    return true;
}

bool parse_cache_dtype(const char* text, dee::DeviceCacheDType& out) {
    if (!text) return false;
    const std::string value(text);
    if (value == "fp32") out = dee::DeviceCacheDType::Fp32;
    else if (value == "fp16") out = dee::DeviceCacheDType::Fp16;
    else {
        std::fprintf(stderr, "[cli] invalid --cache-dtype value: %s (expected fp32 or fp16)\n", text);
        return false;
    }
    return true;
}

bool parse_transfer_dtype(const char* text, dee::WeightTransferDType& out) {
    if (!text) return false;
    const std::string value(text);
    if (value == "bf16") out = dee::WeightTransferDType::Bf16;
    else if (value == "int8") out = dee::WeightTransferDType::Int8;
    else if (value == "int4") out = dee::WeightTransferDType::Int4;
    else if (value == "fp4") out = dee::WeightTransferDType::Fp4E2m1;
    else {
        std::fprintf(stderr, "[cli] invalid --transfer-dtype value: %s (expected bf16, int8, int4, or fp4)\n", text);
        return false;
    }
    return true;
}

bool parse_float_nonneg(const char* text, const char* option, float& out) {
    if (!text || !*text) {
        std::fprintf(stderr, "[cli] %s requires a float value\n", option);
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(text, &end);
    if (errno || !end || *end || value < 0.0f || !std::isfinite(value)) {
        std::fprintf(stderr, "[cli] invalid %s value: %s (expected non-negative finite float)\n",
                     option, text);
        return false;
    }
    out = value;
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
    std::printf("profile scenario       : %s\n", dee::benchmark_scenario_name(cfg.scenario));
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
    std::printf("device cache dtype      : %s\n", dee::device_cache_dtype_name(cfg.cache_dtype));
    std::printf("weight transfer dtype   : %s\n", dee::weight_transfer_dtype_name(cfg.transfer_dtype));
    std::printf("prefetch ring depth     : %zu\n", cfg.prefetch_depth);
    if (cfg.oracle_strict_margin > 0.0f) {
        std::printf("oracle strict margin    : %.6g (CPU fallback for ambiguous calls)\n",
                    cfg.oracle_strict_margin);
        std::printf("oracle gpu/cpu/fallback : %llu / %llu (calls)\n",
                    static_cast<unsigned long long>(stats.oracle_boundary.gpu_calls),
                    static_cast<unsigned long long>(stats.oracle_boundary.cpu_fallback_calls));
    } else if (cfg.use_cuda) {
        std::printf("oracle mode            : raw GPU (legacy; may diverge from CPU routing)\n");
    }
    std::printf("quantized source prepack: %s\n",
                stats.quantized_prepack_complete ? "complete" : "dynamic/disabled");
    std::printf("prepack startup ms      : %.3f\n", stats.quantized_prepack_ms);
    std::printf("prepack experts/bytes   : %llu / %zu\n",
                static_cast<unsigned long long>(stats.quantized_prepack_experts),
                stats.quantized_prepack_bytes);
    std::printf("peak expert-cache VRAM : %zu bytes\n", stats.peak_vram);
    std::printf("cache hits             : %llu\n", static_cast<unsigned long long>(stats.cache_hits));
    std::printf("resident hits          : %llu\n", static_cast<unsigned long long>(stats.resident_hits));
    std::printf("in-flight hits         : %llu\n", static_cast<unsigned long long>(stats.inflight_hits));
    std::printf("cold loads             : %llu\n", static_cast<unsigned long long>(stats.cold_loads));
    std::printf("duplicate requests     : %llu\n", static_cast<unsigned long long>(stats.duplicate_requests));
    std::printf("cache loads            : %llu\n", static_cast<unsigned long long>(stats.cache_loads));
    std::printf("evictions              : %llu\n", static_cast<unsigned long long>(stats.evictions));
    std::printf("prefetch requests      : %llu\n", static_cast<unsigned long long>(stats.prefetch_issued));
    std::printf("prefetch fallbacks     : %llu\n", static_cast<unsigned long long>(stats.prefetch_fallbacks));
    std::printf("output finite status   : %s\n", stats.hidden_finite ? "yes" : "NO");

    if (!stats.profile.enabled) return;
    const dee::StageProfile& profile = stats.profile;
    const double wall_ms = stats.elapsed_sec * 1000.0;
    std::printf("\n=== stage profile (warmup excluded) ===\n");
    for (size_t i = 0; i < static_cast<size_t>(dee::CpuStage::Count); ++i) {
        const double percent = wall_ms > 0.0 ? profile.cpu_ms[i] * 100.0 / wall_ms : 0.0;
        std::printf("CPU %-22s %10.3f ms  %6.2f%% wall\n",
                    dee::cpu_stage_name(static_cast<dee::CpuStage>(i)), profile.cpu_ms[i], percent);
    }
    for (size_t i = 0; i < static_cast<size_t>(dee::GpuStage::Count); ++i) {
        const double percent = wall_ms > 0.0 ? profile.gpu_ms[i] * 100.0 / wall_ms : 0.0;
        std::printf("GPU %-22s %10.3f ms  %6.2f%% wall  samples=%llu\n",
                    dee::gpu_stage_name(static_cast<dee::GpuStage>(i)), profile.gpu_ms[i], percent,
                    static_cast<unsigned long long>(profile.gpu_samples[i]));
    }
    std::printf("\nOracle internal timing (%llu calls):\n",
                static_cast<unsigned long long>(profile.oracle_calls));
    for (size_t i = 0; i < static_cast<size_t>(dee::OracleStage::Count); ++i) {
        const double per_call_us = profile.oracle_calls
            ? profile.oracle_ms[i] * 1000.0 / profile.oracle_calls : 0.0;
        std::printf("  %-22s %10.3f ms  %8.3f us/call\n",
                    dee::oracle_stage_name(static_cast<dee::OracleStage>(i)),
                    profile.oracle_ms[i], per_call_us);
    }
    std::printf("  allocations/bytes      %10llu / %llu\n",
                static_cast<unsigned long long>(profile.oracle_allocations),
                static_cast<unsigned long long>(profile.oracle_allocation_bytes));
    std::printf("GPU compute total      : %.3f ms (%.2f%% wall)\n", profile.total_gpu_compute_ms,
                wall_ms > 0.0 ? profile.total_gpu_compute_ms * 100.0 / wall_ms : 0.0);
    std::printf("GPU transfer total     : %.3f ms (%.2f%% wall)\n", profile.total_gpu_transfer_ms,
                wall_ms > 0.0 ? profile.total_gpu_transfer_ms * 100.0 / wall_ms : 0.0);
    std::printf("GPU weight conversion  : %.3f ms (%.2f%% wall)\n",
                profile.total_gpu_weight_conversion_ms,
                wall_ms > 0.0 ? profile.total_gpu_weight_conversion_ms * 100.0 / wall_ms : 0.0);
    std::printf("layer wall total/avg   : %.3f / %.3f ms\n", profile.layer_wall_ms,
                profile.layer_count ? profile.layer_wall_ms / profile.layer_count : 0.0);
    std::printf("token latency avg      : %.3f ms\n", profile.token_latency_avg_ms);
    std::printf("token latency median   : %.3f ms\n", profile.token_latency_median_ms);
    std::printf("token latency p95      : %.3f ms\n", profile.token_latency_p95_ms);
    std::printf("token latency max      : %.3f ms\n", profile.token_latency_max_ms);
    std::printf("avg expert request     : %.3f us\n", profile.average_expert_request_us);
    std::printf("avg cold load          : %.3f us\n", profile.average_cold_load_us);
    std::printf("mmap->pinned bytes     : %llu\n", static_cast<unsigned long long>(profile.mmap_to_pinned_bytes));
    std::printf("H2D bytes/copies/avg   : %llu / %llu / %.1f\n",
                static_cast<unsigned long long>(profile.h2d_bytes),
                static_cast<unsigned long long>(profile.h2d_copies), profile.average_h2d_copy_bytes);
    std::printf("cuBLAS calls           : %llu\n", static_cast<unsigned long long>(profile.cublas_calls));
    std::printf("CUDA kernel launches   : %llu\n", static_cast<unsigned long long>(profile.kernel_launches));
    std::printf("stream waits           : %llu\n", static_cast<unsigned long long>(profile.stream_waits));
    std::printf("host synchronizations  : %llu\n", static_cast<unsigned long long>(profile.host_synchronizations));
    std::printf("host wait attribution  :\n");
    for (size_t i = 0; i < static_cast<size_t>(dee::HostWaitReason::Count); ++i) {
        std::printf("  %-20s %10.3f ms  count=%llu\n",
                    dee::host_wait_reason_name(static_cast<dee::HostWaitReason>(i)),
                    profile.host_wait_ms[i],
                    static_cast<unsigned long long>(profile.host_wait_count[i]));
    }
    if (profile.gpu_timeline_span_ms > 0.0) {
        std::printf("GPU timeline span      : %.3f ms\n", profile.gpu_timeline_span_ms);
        std::printf("copy engine active/util: %.3f ms / %.2f%%\n", profile.copy_engine_active_ms,
                    profile.copy_engine_utilization * 100.0);
        std::printf("compute active/util    : %.3f ms / %.2f%%\n", profile.compute_engine_active_ms,
                    profile.compute_engine_utilization * 100.0);
        std::printf("copy/compute overlap   : %.3f ms / %.2f%%\n", profile.copy_compute_overlap_ms,
                    profile.copy_compute_overlap_fraction * 100.0);
        std::printf("neither engine active  : %.3f ms / %.2f%%\n", profile.gpu_neither_active_ms,
                    profile.gpu_neither_active_ms * 100.0 / profile.gpu_timeline_span_ms);
        std::printf("transfer queue avg/max : %.3f / %llu\n",
                    profile.average_transfer_queue_depth,
                    static_cast<unsigned long long>(profile.max_transfer_queue_depth));
        std::printf("idle attributed        : %.3f ms / %.2f%%\n",
                    profile.idle_attributed_ms,
                    profile.idle_attributed_fraction * 100.0);
        std::printf("GPU idle attribution   :\n");
        for (size_t i = 0; i < static_cast<size_t>(dee::IdleGapCategory::Count); ++i) {
            std::printf("  %-35s %9.3f ms  count=%llu avg=%7.3f p95=%7.3f max=%7.3f\n",
                        dee::idle_gap_category_name(static_cast<dee::IdleGapCategory>(i)),
                        profile.idle_gap_ms[i],
                        static_cast<unsigned long long>(profile.idle_gap_count[i]),
                        profile.idle_gap_avg_ms[i], profile.idle_gap_p95_ms[i],
                        profile.idle_gap_max_ms[i]);
        }
        std::printf("cache-readiness breakdown:\n");
        for (size_t i = 0;
             i < static_cast<size_t>(dee::ReadinessWaitCategory::Count); ++i) {
            std::printf("  %-35s %9.3f ms  count=%llu\n",
                        dee::readiness_wait_category_name(
                            static_cast<dee::ReadinessWaitCategory>(i)),
                        profile.readiness_wait_ms[i],
                        static_cast<unsigned long long>(profile.readiness_wait_count[i]));
        }
        std::printf("top GPU idle gaps      :\n");
        for (const dee::IdleGapRecord& gap : profile.top_idle_gaps) {
            std::printf("  %9.3f ms  %-35s t=%d l=%d e=%d transfer=%llu\n",
                        gap.end_ms - gap.start_ms,
                        dee::idle_gap_category_name(gap.category), gap.token,
                        gap.logical_layer, gap.expert,
                        static_cast<unsigned long long>(gap.transfer_id));
        }
    }
    std::printf("timing events allocated: %llu (bounded pool)\n",
                static_cast<unsigned long long>(profile.timing_events_allocated));
    std::printf("unique requested/loaded: %llu / %llu\n",
                static_cast<unsigned long long>(profile.unique_experts_requested),
                static_cast<unsigned long long>(profile.unique_experts_loaded));
    std::printf("working set avg/max    : %.2f / %llu experts per token\n",
                profile.average_working_set_per_token,
                static_cast<unsigned long long>(profile.max_working_set_per_token));
    std::printf("reused before eviction : %.4f\n", profile.reused_before_eviction_fraction);
    std::printf("Oracle adjacent overlap: %.4f (random expectation %.4f)\n",
                profile.oracle_adjacent_topk_overlap, profile.oracle_random_overlap_expectation);
    std::printf("pinned blocks skipped  : %llu\n",
                static_cast<unsigned long long>(profile.pinned_blocks_skipped));
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

std::string benchmark_json(const dee::EngineConfig& cfg, const dee::EngineStats& stats,
                           int warmup, bool include_trace) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\"benchmark\":{\"backend\":\"" << (cfg.use_cuda ? "cuda" : "cpu") << '\"'
        << ",\"scenario\":\"" << dee::benchmark_scenario_name(cfg.scenario) << '\"'
        << ",\"cuda_device\":\"" << json_escape(stats.cuda_device_name) << '\"'
        << ",\"compute_capability\":\"" << stats.cuda_compute_major << '.' << stats.cuda_compute_minor << '\"'
        << ",\"cuda_runtime_version\":" << stats.cuda_runtime_version
        << ",\"tokens\":" << stats.tokens
        << ",\"warmup\":" << warmup
        << ",\"elapsed_seconds\":" << stats.elapsed_sec
        << ",\"tokens_per_second\":" << stats.tok_per_sec
        << ",\"cache_budget_bytes\":" << cfg.budget_bytes
        << ",\"cache_dtype\":\"" << dee::device_cache_dtype_name(cfg.cache_dtype) << '\"'
        << ",\"transfer_dtype\":\"" << dee::weight_transfer_dtype_name(cfg.transfer_dtype) << '\"'
        << ",\"quantized_prepack_complete\":"
        << (stats.quantized_prepack_complete ? "true" : "false")
        << ",\"quantized_prepack_startup_ms\":" << stats.quantized_prepack_ms
        << ",\"quantized_prepack_experts\":" << stats.quantized_prepack_experts
        << ",\"quantized_prepack_bytes\":" << stats.quantized_prepack_bytes
        << ",\"peak_cache_bytes\":" << stats.peak_vram
        << ",\"requests\":" << stats.prefetch_issued
        << ",\"resident_hits\":" << stats.resident_hits
        << ",\"inflight_hits\":" << stats.inflight_hits
        << ",\"cold_loads\":" << stats.cold_loads
        << ",\"duplicates\":" << stats.duplicate_requests
        << ",\"evictions\":" << stats.evictions
        << ",\"prefetch_fallbacks\":" << stats.prefetch_fallbacks
        << ",\"output_finite\":" << (stats.hidden_finite ? "true" : "false") << '}';
    if (include_trace) {
        out << ",\"final_hidden\":[";
        for (size_t i = 0; i < stats.final_hidden.size(); ++i) {
            if (i) out << ',';
            out << stats.final_hidden[i];
        }
        out << ']';
    }
    out << ",\"profile\":" << dee::stage_profile_json(stats.profile, include_trace) << '}';
    return out.str();
}

bool write_text_file(const std::string& path, const std::string& contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::fprintf(stderr, "[cli] cannot open profile output: %s\n", path.c_str());
        return false;
    }
    output << contents << '\n';
    if (!output) {
        std::fprintf(stderr, "[cli] cannot write profile output: %s\n", path.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    dee::EngineConfig cfg;
    cfg.shard_path = "tests/data/ornith_moe256.safetensors";
    cfg.oracle_path = "oracle.pt";
    int warmup_tokens = 2;
    std::string profile_json_path;
    std::string trace_json_path;
    std::string timeline_json_path;
    bool cache_dtype_explicit = false;
    bool transfer_dtype_explicit = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        if (arg == "--cuda") { cfg.use_cuda = true; continue; }
        if (arg == "--profile-stages") { cfg.profile_stages = true; continue; }
        if (arg == "--dynamic-quantization") {
            cfg.prepack_quantized_source = false;
            continue;
        }
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
            if (!(value = require_value(i, argc, argv, "--budget")) ||
                !parse_size(value, "--budget", cfg.budget_bytes)) return 2;
        } else if (arg == "--prefetch-depth") {
            if (!(value = require_value(i, argc, argv, "--prefetch-depth")) ||
                !parse_size(value, "--prefetch-depth", cfg.prefetch_depth) ||
                cfg.prefetch_depth == 0) return 2;
        } else if (arg == "--cache-dtype") {
            if (!(value = require_value(i, argc, argv, "--cache-dtype")) ||
                !parse_cache_dtype(value, cfg.cache_dtype)) return 2;
            cache_dtype_explicit = true;
        } else if (arg == "--transfer-dtype") {
            if (!(value = require_value(i, argc, argv, "--transfer-dtype")) ||
                !parse_transfer_dtype(value, cfg.transfer_dtype)) return 2;
            transfer_dtype_explicit = true;
        } else if (arg == "--hidden") {
            if (!(value = require_value(i, argc, argv, "--hidden")) ||
                !parse_positive(value, "--hidden", cfg.hidden)) return 2;
        } else if (arg == "--inter") {
            if (!(value = require_value(i, argc, argv, "--inter")) ||
                !parse_positive(value, "--inter", cfg.inter)) return 2;
        } else if (arg == "--oracle-margin") {
            if (!(value = require_value(i, argc, argv, "--oracle-margin")) ||
                !parse_float_nonneg(value, "--oracle-margin", cfg.oracle_strict_margin)) return 2;
        } else if (arg == "--profile-json") {
            if (!(value = require_value(i, argc, argv, "--profile-json"))) return 2;
            profile_json_path = value;
            cfg.profile_stages = true;
        } else if (arg == "--profile-timeline") {
            if (!(value = require_value(i, argc, argv, "--profile-timeline"))) return 2;
            timeline_json_path = value;
            cfg.profile_stages = true;
            cfg.profile_timeline = true;
        } else if (arg == "--profile-scenario") {
            if (!(value = require_value(i, argc, argv, "--profile-scenario")) ||
                !parse_scenario(value, cfg.scenario)) return 2;
            cfg.profile_stages = true;
        } else if (arg == "--trace-requests") {
            if (!(value = require_value(i, argc, argv, "--trace-requests"))) return 2;
            trace_json_path = value;
            cfg.profile_stages = true;
            cfg.trace_requests = true;
        } else {
            std::fprintf(stderr, "[cli] unknown option: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (cfg.use_cuda && !cache_dtype_explicit) cfg.cache_dtype = dee::DeviceCacheDType::Fp16;
    if (cfg.use_cuda && !transfer_dtype_explicit) cfg.transfer_dtype = dee::WeightTransferDType::Int8;

#ifndef DEE_CUDA
    if (cfg.use_cuda) {
        std::fprintf(stderr, "[cli] --cuda was requested, but this binary was built with DEE_CUDA=OFF. Reconfigure with -DDEE_CUDA=ON.\n");
        return 2;
    }
#endif

    if (cfg.cache_dtype == dee::DeviceCacheDType::Fp16 && !cfg.use_cuda) {
        std::fprintf(stderr, "[cli] --cache-dtype fp16 requires --cuda\n");
        return 2;
    }
    if (cfg.transfer_dtype != dee::WeightTransferDType::Bf16 &&
        (!cfg.use_cuda || cfg.cache_dtype != dee::DeviceCacheDType::Fp16)) {
        std::fprintf(stderr, "[cli] quantized transfer requires --cuda with --cache-dtype fp16\n");
        return 2;
    }

    if (cfg.profile_timeline && !cfg.use_cuda) {
        std::fprintf(stderr, "[cli] --profile-timeline requires --cuda\n");
        return 2;
    }

    if (cfg.scenario != dee::BenchmarkScenario::EndToEnd && !cfg.use_cuda) {
        std::fprintf(stderr, "[cli] --profile-scenario %s requires --cuda\n",
                     dee::benchmark_scenario_name(cfg.scenario));
        return 2;
    }

    if (warmup_tokens > 0) {
        dee::EngineConfig warmup = cfg;
        warmup.num_tokens = warmup_tokens;
        warmup.profile_stages = false;
        warmup.trace_requests = false;
        warmup.profile_timeline = false;
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
    if (!profile_json_path.empty() &&
        !write_text_file(profile_json_path, benchmark_json(engine.config(), engine.stats(), warmup_tokens, false))) return 1;
    if (!trace_json_path.empty() &&
        !write_text_file(trace_json_path, benchmark_json(engine.config(), engine.stats(), warmup_tokens, true))) return 1;
    if (!timeline_json_path.empty() &&
        !write_text_file(timeline_json_path, dee::cuda_timeline_json(engine.stats().profile))) return 1;
    return engine.stats().hidden_finite ? 0 : 1;
}
