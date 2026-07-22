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

#include "dee/engine.h"

namespace py = pybind11;

PYBIND11_MODULE(pydee, m) {
    m.doc() = "pydee: Python binding for dee.cpp MoE expert execution "
              "(real-model integration mode; caller owns routing + combine).";

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
        .def_readwrite("oracle_path", &dee::EngineConfig::oracle_path)
        .def_readwrite("topk", &dee::EngineConfig::topk)
        .def_readwrite("num_layers", &dee::EngineConfig::num_layers)
        .def_readwrite("num_experts", &dee::EngineConfig::num_experts)
        .def_readwrite("budget_bytes", &dee::EngineConfig::budget_bytes)
        .def_readwrite("cache_dtype", &dee::EngineConfig::cache_dtype)
        .def_readwrite("transfer_dtype", &dee::EngineConfig::transfer_dtype)
        .def_readwrite("use_cuda", &dee::EngineConfig::use_cuda)
        .def_readwrite("hidden", &dee::EngineConfig::hidden)
        .def_readwrite("inter", &dee::EngineConfig::inter)
        .def_readwrite("verbose", &dee::EngineConfig::verbose)
        .def_readwrite("prefetch_depth", &dee::EngineConfig::prefetch_depth)
        .def_readwrite("profile_stages", &dee::EngineConfig::profile_stages);

    py::class_<dee::Engine>(m, "Engine")
        .def(py::init<>())
        .def("init", &dee::Engine::init, py::arg("cfg"))
        .def("hidden_dim", &dee::Engine::hidden_dim)
        .def("inter_dim", &dee::Engine::inter_dim)
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
            return self.moe_forward_experts(
                layer,
                static_cast<float*>(in_buf.ptr),
                static_cast<float*>(out_buf.ptr),
                experts);
        }, py::arg("layer"), py::arg("h_in"), py::arg("experts_out"),
           py::arg("experts"),
           R"pbdoc(
                Run SwiGLU forward for each requested expert and return per-expert
                FP32 outputs to `experts_out` (layout: [K, hidden_dim], contiguous).
                Caller handles the gate-weighted sum combine (matching HF reference).
            )pbdoc")
        .def("last_stats_json", [](const dee::Engine& self) -> std::string {
            std::ostringstream ss;
            const dee::EngineStats& s = self.stats();
            ss << "{"
               << "\"tokens\":" << s.tokens
               << ",\"elapsed_sec\":" << s.elapsed_sec
               << ",\"tok_per_sec\":" << s.tok_per_sec
               << ",\"cache_hits\":" << s.cache_hits
               << ",\"cold_loads\":" << s.cold_loads
               << ",\"evictions\":" << s.evictions
               << ",\"hidden_finite\":" << (s.hidden_finite ? "true" : "false")
               << ",\"peak_vram\":" << s.peak_vram
               << "}";
            return ss.str();
        });
}
