// Offline correctness probe only. Caller owns one canonical expert record.
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "kt_bridge/reference_cpu_executor.hpp"

template <class T> std::vector<T> read_exact(const char* path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    std::vector<T> v(n);
    if (!f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T)) ||
        f.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("missing or wrong-sized input");
    return v;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            using namespace dee::ktbridge;
            std::vector<uint8_t> p(512, 0x11), s(32, 127);
            std::vector<float> x(32, .1f), y(32);
            PackedExpertView v{{p.data(), s.data(), 32, 32, 512, 32},
                               {p.data(), s.data(), 32, 32, 512, 32},
                               {p.data(), s.data(), 32, 32, 512, 32}};
            ExecuteConfig config;
            config.swiglu_limit = std::numeric_limits<float>::quiet_NaN();
            if (validate_execute_args(v, x.data(), 32, 1, config, y.data(), 32) != ExecuteError::kConfig)
                throw std::runtime_error("NaN clamp accepted");
            config.swiglu_limit = 10;
            const size_t huge = std::numeric_limits<size_t>::max() / 32 + 1;
            v.gate.out = v.up.out = v.down.in = huge;
            for (auto* q : {&v.gate, &v.up, &v.down}) q->packed_nbytes = q->scale_nbytes = 0;
            if (validate_execute_args(v, x.data(), 32, 1, config, y.data(), 32) != ExecuteError::kShape)
                throw std::runtime_error("overflow shape accepted");
            std::cout << "reference guards OK\n";
            return 0;
        }
        if (argc != 5) throw std::runtime_error("usage: probe record x output routing_weight");
        constexpr size_t H = 4096, I = 2048, P = H * I / 2, S = H * I / 32;
        auto record = read_exact<uint8_t>(argv[1], 3 * (P + S));
        auto x = read_exact<float>(argv[2], H);
        std::vector<float> y(H);
        const auto* b = record.data();
        dee::ktbridge::PackedExpertView view{
            {b, b + 3 * P, I, H, P, S},
            {b + P, b + 3 * P + S, I, H, P, S},
            {b + 2 * P, b + 3 * P + 2 * S, H, I, P, S}};
        dee::ktbridge::ReferenceCpuExecutor executor;
        const auto err = executor.execute(0, 0, view, x.data(), H,
            std::stof(argv[4]), {}, y.data(), H);
        if (err != dee::ktbridge::ExecuteError::kOk)
            throw std::runtime_error(dee::ktbridge::execute_error_name(err));
        std::ofstream out(argv[3], std::ios::binary);
        if (!out.write(reinterpret_cast<const char*>(y.data()), H * sizeof(float)))
            throw std::runtime_error("cannot write output");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
