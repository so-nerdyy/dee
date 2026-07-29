// dee.cpp/pydee/pydee.cpp
//
// pybind11 Python module for the dee.cpp MoE expert engine. Exposes the
// minimum surface needed by the Python HF adapter for real-model integration:
//
//   import pydee
//   cfg = pydee.EngineConfig()
//   cfg.shard_path = "/.../model-00001-of-00016.safetensors"
//   cfg.num_experts = 256
//   cfg.num_layers = 40
//   cfg.hidden = 2048
//   cfg.inter = 512
//   cfg.oracle_path = ""              # caller owns routing (HF model)
//   cfg.transfer_dtype = pydee.WeightTransferDType.Bf16
//   cfg.use_cuda = False              # CPU parity first; switch on T4 later
//   engine = pydee.Engine()
//   engine.init(cfg)
//   expert_outs = np.empty((top_k, hidden), dtype=np.float32)
//   ok = engine.moe_forward_experts(layer_idx, hidden_np, expert_outs, [e0,...])
//
// All numeric buffers are passed via numpy's buffer protocol so torch tensors
// (after `.cpu().numpy()`) and np.ndarray are interchangeable.
//
// Build (from this directory):
//   python3 -m pip install pybind11 --break-system-packages --user
//   python3 setup.py build_ext --inplace

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>

#include "dee/engine.h"
#include "dee/trace_alloc.h"

namespace py = pybind11;

PYBIND11_MODULE(pydee_core, m) {
    if (sizeof(dee::Engine) != dee::engine_abi_size()) {
        throw py::import_error(
            "pydee/dee_core Engine ABI mismatch: rebuild both with the same DEE_CUDA setting");
    }
    m.doc() = "pydee: Python binding for dee.cpp MoE expert execution "
              "(real-model integration mode; caller owns routing + combine).";
    m.def("_trace_alloc_selftest", &dee::trace_alloc::startup_self_test,
          "Run the harmless traced host-allocation connectivity proof.");
    m.def("_trace_alloc_stats", []() {
        py::dict result;
        result["live"] = dee::trace_alloc::live_count();
        result["dead"] = dee::trace_alloc::dead_count();
        result["non_selftest_allocs"] =
            dee::trace_alloc::non_selftest_alloc_count();
        result["unalloc_aborts"] = dee::trace_alloc::unalloc_abort_count();
        result["double_free_aborts"] =
            dee::trace_alloc::double_free_abort_count();
        result["mismatch_aborts"] =
            dee::trace_alloc::mismatch_abort_count();
        result["uaf_aborts"] = dee::trace_alloc::uaf_abort_count();
        return result;
    }, "Return bounded trace-allocation connectivity counters.");

    py::enum_<dee::DeviceCacheDType>(m, "DeviceCacheDType")
        .value("Fp32", dee::DeviceCacheDType::Fp32)
        .value("Fp16", dee::DeviceCacheDType::Fp16);
    py::enum_<dee::WeightTransferDType>(m, "WeightTransferDType")
        .value("Bf16", dee::WeightTransferDType::Bf16)
        .value("Int8", dee::WeightTransferDType::Int8)
        .value("Int4", dee::WeightTransferDType::Int4);

    py::class_<dee::EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("shard_path", &dee::EngineConfig::shard_path)
        .def_readwrite("shard_paths", &dee::EngineConfig::shard_paths)
        .def_readwrite("oracle_path", &dee::EngineConfig::oracle_path)
        .def_readwrite("num_tokens", &dee::EngineConfig::num_tokens)
        .def_readwrite("topk", &dee::EngineConfig::topk)
        .def_readwrite("num_layers", &dee::EngineConfig::num_layers)
        .def_readwrite("num_experts", &dee::EngineConfig::num_experts)
        .def_readwrite("base_layer", &dee::EngineConfig::base_layer)
        .def_readwrite("device_id", &dee::EngineConfig::device_id)
        .def_readwrite("budget_bytes", &dee::EngineConfig::budget_bytes)
        .def_readwrite("cache_dtype", &dee::EngineConfig::cache_dtype)
        .def_readwrite("transfer_dtype", &dee::EngineConfig::transfer_dtype)
        .def_readwrite("use_cuda", &dee::EngineConfig::use_cuda)
        .def_readwrite("hidden", &dee::EngineConfig::hidden)
        .def_readwrite("inter", &dee::EngineConfig::inter)
        .def_readwrite("verbose", &dee::EngineConfig::verbose)
        .def_readwrite("prefetch_depth", &dee::EngineConfig::prefetch_depth)
        .def_readwrite("profile_stages", &dee::EngineConfig::profile_stages)
        .def_readwrite("trace_requests", &dee::EngineConfig::trace_requests)
        .def_readwrite("profile_timeline", &dee::EngineConfig::profile_timeline)
        .def_readwrite("debug_validate_cache", &dee::EngineConfig::debug_validate_cache);

    py::class_<dee::Engine>(m, "Engine")
        .def(py::init<>())
        .def("init", &dee::Engine::init, py::arg("cfg"))
        .def("hidden_dim", &dee::Engine::hidden_dim)
        .def("inter_dim", &dee::Engine::inter_dim)
        .def("reset_runtime_cache", &dee::Engine::reset_runtime_cache,
             "Evict all streamed experts and reset live cache/transfer counters.")
        .def("validate_cache_invariants", [](const dee::Engine& self) {
            std::string error;
            const bool valid = self.validate_cache_invariants(&error);
            return py::make_tuple(valid, error);
        }, "Return (valid, error) for cache pointer/range/generation/pin invariants.")
        .def("reset_external_profile", &dee::Engine::reset_external_profile,
             "Reset measurement counters without evicting resident experts.")
        .def("set_external_token", &dee::Engine::set_external_token,
             py::arg("token"),
             "Attach an external prefill/decode-step index to trace records.")
        .def("external_profile_json", &dee::Engine::external_profile_json,
             py::arg("total_wall_ms"),
             "Return the measurement-only stage/cache/request profile.")
        .def("external_timeline_json", &dee::Engine::external_timeline_json,
             py::arg("total_wall_ms"),
             "Return the bounded CUDA/host timeline as Chrome trace JSON.")
        .def("route_topk", [](
                dee::Engine& self,
                int layer,
                py::array_t<float, py::array::c_style | py::array::forcecast> h_in) {
            auto in_buf = h_in.request();
            const size_t H = static_cast<size_t>(self.hidden_dim());
            const dee::EngineConfig& cfg = self.config();
            if (in_buf.size != H) {
                throw std::runtime_error("h_in size does not match hidden_dim");
            }
            py::array_t<float> logits(static_cast<size_t>(cfg.num_experts));
            py::array_t<float> weights(static_cast<size_t>(cfg.topk));
            py::array_t<int> experts(static_cast<size_t>(cfg.topk));
            bool ok = false;
            {
                py::gil_scoped_release release;
                ok = self.route_topk(
                    layer, static_cast<float*>(in_buf.ptr), logits.mutable_data(),
                    weights.mutable_data(), experts.mutable_data());
            }
            if (!ok) throw std::runtime_error("dee.cpp route_topk failed (see stderr)");
            return py::make_tuple(logits, weights, experts);
        }, py::arg("layer"), py::arg("h_in"),
           "Run the real checkpoint router and return (logits, topk_weights, expert_ids).")
        .def("route_topk_batch", [](
                dee::Engine& self,
                int layer,
                py::array_t<float, py::array::c_style | py::array::forcecast> h_in) {
            auto in_buf = h_in.request();
            if (in_buf.ndim != 2 || in_buf.shape[1] != self.hidden_dim()) {
                throw std::runtime_error("h_in must have shape [tokens, hidden_dim]");
            }
            const py::ssize_t tokens = in_buf.shape[0];
            const dee::EngineConfig& cfg = self.config();
            py::array_t<float> logits({tokens, static_cast<py::ssize_t>(cfg.num_experts)});
            py::array_t<float> weights({tokens, static_cast<py::ssize_t>(cfg.topk)});
            py::array_t<int> experts({tokens, static_cast<py::ssize_t>(cfg.topk)});
            bool ok = false;
            {
                py::gil_scoped_release release;
                ok = self.route_topk_batch(
                    layer, static_cast<float*>(in_buf.ptr), static_cast<int>(tokens),
                    logits.mutable_data(), weights.mutable_data(), experts.mutable_data());
            }
            if (!ok) throw std::runtime_error("dee.cpp route_topk_batch failed (see stderr)");
            return py::make_tuple(logits, weights, experts);
        }, py::arg("layer"), py::arg("h_in"),
           "Run the real checkpoint router for a [tokens, hidden] batch.")
        .def("moe_forward_experts", [](
                dee::Engine& self,
                int layer,
                py::array_t<float, py::array::c_style | py::array::forcecast> h_in,
                py::array_t<float, py::array::c_style | py::array::forcecast> experts_out,
                std::vector<int> experts) -> bool {
            auto in_buf = h_in.request();
            auto out_buf = experts_out.request();
            const size_t H = (size_t)self.hidden_dim();
            if (in_buf.size != H) {
                throw std::runtime_error(
                    "h_in.size (" + std::to_string(in_buf.size) +
                    ") != hidden_dim (" + std::to_string(H) + ")");
            }
            if (out_buf.size != experts.size() * H) {
                throw std::runtime_error(
                    "experts_out.size (" + std::to_string(out_buf.size) +
                    ") != " + std::to_string(experts.size()) + " * hidden_dim (" +
                    std::to_string(H) + ")");
            }
            bool ok = false;
            {
                py::gil_scoped_release release;
                ok = self.moe_forward_experts(
                    layer,
                    static_cast<float*>(in_buf.ptr),
                    static_cast<float*>(out_buf.ptr),
                    experts);
            }
            return ok;
        }, py::arg("layer"), py::arg("h_in"), py::arg("experts_out"),
           py::arg("experts"),
           R"pbdoc(
                Run SwiGLU forward for each requested expert and return per-expert
                FP32 outputs to `experts_out` (layout: [K, hidden_dim], contiguous).
                Caller handles the gate-weighted sum combine (matching HF reference).
            )pbdoc")
        .def("moe_forward_batch", [](
                dee::Engine& self,
                int layer,
                py::array_t<float, py::array::c_style | py::array::forcecast> h_in,
                py::array_t<int, py::array::c_style | py::array::forcecast> expert_ids) {
            auto in_buf = h_in.request();
            auto ids_buf = expert_ids.request();
            if (in_buf.ndim != 2 || in_buf.shape[1] != self.hidden_dim()) {
                throw std::runtime_error("h_in must have shape [tokens, hidden_dim]");
            }
            if (ids_buf.ndim != 2 || ids_buf.shape[0] != in_buf.shape[0]) {
                throw std::runtime_error("expert_ids must have shape [tokens, topk]");
            }
            const py::ssize_t tokens = in_buf.shape[0];
            const py::ssize_t topk = ids_buf.shape[1];
            if (topk != self.config().topk) {
                throw std::runtime_error("expert_ids topk does not match EngineConfig.topk");
            }
            py::array_t<float> output({
                tokens, topk, static_cast<py::ssize_t>(self.hidden_dim())
            });
            bool ok = false;
            {
                py::gil_scoped_release release;
                ok = self.moe_forward_batch(
                    layer, static_cast<float*>(in_buf.ptr), static_cast<int>(tokens),
                    static_cast<int*>(ids_buf.ptr), static_cast<int>(topk),
                    output.mutable_data());
            }
            if (!ok) {
                std::string msg = "dee.cpp batched MoE failed (moe_forward_batch("
                    "layer=" + std::to_string(layer)
                    + " tokens=" + std::to_string(tokens)
                    + " topk=" + std::to_string(topk)
                    + " device=" + std::to_string(self.config().device_id) + "))";
                const std::string& native = self.last_error_message();
                if (!native.empty()) msg += " | native: " + native;
                else msg += " | native: <no detailed diagnostic recorded>";
                throw std::runtime_error(msg);
            }
            return output;
        }, py::arg("layer"), py::arg("h_in"), py::arg("expert_ids"),
           "Run token batches grouped by expert with eager-compatible CUDA GEMM shapes.")
        .def("moe_forward_batch_device", [](
                dee::Engine& self,
                int layer,
                uintptr_t d_h_in_ptr,
                int tokens,
                py::array_t<int, py::array::c_style | py::array::forcecast> expert_ids,
                int topk,
                uintptr_t d_experts_out_ptr) -> bool {
            auto ids_buf = expert_ids.request();
            if (ids_buf.ndim != 2 || ids_buf.shape[0] != tokens ||
                ids_buf.shape[1] != topk) {
                throw std::runtime_error("expert_ids must have shape [tokens, topk]");
            }
            if (topk != self.config().topk) {
                throw std::runtime_error("expert_ids topk does not match EngineConfig.topk");
            }
            bool ok = false;
            {
                py::gil_scoped_release release;
                ok = self.moe_forward_batch_device(
                    layer,
                    reinterpret_cast<const void*>(d_h_in_ptr),
                    tokens,
                    static_cast<int*>(ids_buf.ptr),
                    topk,
                    reinterpret_cast<void*>(d_experts_out_ptr));
            }
            return ok;
        }, py::arg("layer"), py::arg("d_h_in_ptr"), py::arg("tokens"),
           py::arg("expert_ids"), py::arg("topk"), py::arg("d_experts_out_ptr"),
           R"pbdoc(
                Run MoE forward with device-resident hidden and outputs.
                d_h_in_ptr: device pointer to FP16 hidden [tokens, hidden].
                expert_ids: numpy int32 array [tokens, topk] (host-side for grouping).
                d_experts_out_ptr: device pointer to FP32 output [tokens, topk, hidden].
                Returns True on success; caller must sync the compute stream before
                reading d_experts_out.
            )pbdoc")
        .def("moe_forward_combined_device", [](
                dee::Engine& self,
                int layer,
                uintptr_t d_h_in_ptr,
                int tokens,
                uintptr_t d_expert_ids_ptr,
                int topk,
                uintptr_t d_weights_ptr,
                uintptr_t d_output_ptr,
                uintptr_t d_raw_trace_ptr,
                uintptr_t external_stream_ptr) -> bool {
            if (topk != self.config().topk) {
                throw std::runtime_error(
                    "expert_ids topk does not match EngineConfig.topk");
            }
            bool ok = false;
            {
                py::gil_scoped_release release;
                ok = self.moe_forward_combined_device(
                    layer,
                    reinterpret_cast<const void*>(d_h_in_ptr),
                    tokens,
                    reinterpret_cast<const int64_t*>(d_expert_ids_ptr),
                    topk,
                    reinterpret_cast<const float*>(d_weights_ptr),
                    reinterpret_cast<void*>(d_output_ptr),
                    reinterpret_cast<void*>(d_raw_trace_ptr),
                    reinterpret_cast<void*>(external_stream_ptr));
            }
            return ok;
        }, py::arg("layer"), py::arg("d_h_in_ptr"), py::arg("tokens"),
           py::arg("d_expert_ids_ptr"), py::arg("topk"),
           py::arg("d_weights_ptr"), py::arg("d_output_ptr"),
           py::arg("d_raw_trace_ptr"),
           py::arg("external_stream_ptr"),
           R"pbdoc(
                Run exact combined MoE on device tensors.
                Hidden/output are FP16, weights are FP32, expert IDs are int64.
                Completion is handed from the engine compute stream to the
                supplied PyTorch CUDA stream. Optional raw trace output is
                FP32 [tokens, topk, hidden].
            )pbdoc")
        .def("compute_stream_handle", &dee::Engine::compute_stream_handle,
             "Return the native compute-stream handle for allocator lifetime tracking.")
        .def("last_error_message", [](const dee::Engine& self) -> std::string {
            return self.last_error_message();
        }, "Return the most recent native diagnostic captured by the failure "
           "paths of moe_forward_experts/moe_forward_batch/moe_forward_batch_device. "
           "Empty string if the last call succeeded or no detail was captured.")
        .def("last_stats_json", [](const dee::Engine& self) -> std::string {
            std::ostringstream ss;
            const dee::EngineStats s = self.runtime_stats();
            ss << "{"
               << "\"tokens\":" << s.tokens
               << ",\"elapsed_sec\":" << s.elapsed_sec
               << ",\"tok_per_sec\":" << s.tok_per_sec
               << ",\"cache_hits\":" << s.cache_hits
               << ",\"cache_loads\":" << s.cache_loads
               << ",\"cold_loads\":" << s.cold_loads
               << ",\"resident_hits\":" << s.resident_hits
               << ",\"inflight_hits\":" << s.inflight_hits
               << ",\"evictions\":" << s.evictions
               << ",\"fallbacks\":" << s.fallbacks
               << ",\"prefetch_issued\":" << s.prefetch_issued
               << ",\"prefetch_fallbacks\":" << s.prefetch_fallbacks
               << ",\"duplicate_requests\":" << s.duplicate_requests
               << ",\"h2d_bytes\":" << s.h2d_bytes
               << ",\"h2d_copies\":" << s.h2d_copies
               << ",\"hidden_finite\":" << (s.hidden_finite ? "true" : "false")
               << ",\"peak_vram\":" << s.peak_vram
               << ",\"current_vram\":" << s.current_vram
               << ",\"resident_experts\":" << s.resident_experts
               << ",\"host_pinned_expert_staging_bytes\":"
               << s.host_pinned_expert_staging_bytes
               << ",\"host_pageable_expert_staging_bytes\":"
               << s.host_pageable_expert_staging_bytes
               << ",\"host_router_weight_bytes\":" << s.host_router_weight_bytes
               << ",\"host_hidden_buffer_bytes\":" << s.host_hidden_buffer_bytes
               << ",\"host_moe_dispatch_bytes\":" << s.host_moe_dispatch_bytes
               << ",\"host_prefetch_ring_bytes\":" << s.host_prefetch_ring_bytes
               << ",\"host_prefetch_ring_slots\":" << s.host_prefetch_ring_slots
               << ",\"peak_transient_host_bytes\":" << s.peak_transient_host_bytes
               << ",\"device_expert_cache_reserved_bytes\":"
               << s.device_expert_cache_reserved_bytes
               << ",\"device_prefetch_staging_bytes\":"
               << s.device_prefetch_staging_bytes
               << ",\"device_fixed_work_buffer_bytes\":"
               << s.device_fixed_work_buffer_bytes
               << ",\"device_router_weight_bytes\":" << s.device_router_weight_bytes
               << ",\"device_router_dynamic_bytes\":" << s.device_router_dynamic_bytes
               << ",\"device_moe_batch_buffer_bytes\":"
               << s.device_moe_batch_buffer_bytes
               << ",\"device_moe_raw_workspace_bytes\":"
               << s.device_moe_raw_workspace_bytes
               << ",\"device_oracle_scratch_bytes\":"
               << s.device_oracle_scratch_bytes
               << ",\"cuda_total\":" << s.cuda_total
               << ",\"cuda_free\":" << s.cuda_free
               << "}";
            return ss.str();
        });
}
