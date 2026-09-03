// kt_bridge smoke: Reference + KT executors on a tiny synthetic expert.
// Built by CMake (MSVC-friendly); also mirrored inline in the pytest.
#include <cstdio>
#include <vector>

#include "kt_bridge/cpu_executor.hpp"
#include "kt_bridge/kt_cpu_executor.hpp"
#include "kt_bridge/reference_cpu_executor.hpp"

int main() {
    const size_t H = 32, I = 32;
    std::vector<uint8_t> p1(I * H / 2, 0x11), s1(I * H / 32, 0x7F);
    std::vector<uint8_t> p3(I * H / 2, 0x23), s3(I * H / 32, 0x80);
    std::vector<uint8_t> p2(H * I / 2, 0x45), s2(H * I / 32, 0x7F);
    dee::ktbridge::PackedExpertView v{
        {p1.data(), s1.data(), I, H, p1.size(), s1.size()},
        {p3.data(), s3.data(), I, H, p3.size(), s3.size()},
        {p2.data(), s2.data(), H, I, p2.size(), s2.size()}};
    std::vector<float> x(H, 0.1f), y1(H, 0), y2(H, 0);
    dee::ktbridge::ExecuteConfig cfg;
    dee::ktbridge::ReferenceCpuExecutor r;
    dee::ktbridge::KTransformersCpuExecutor k;
    auto e1 = r.execute(0, 0, v, x.data(), H, 1.0f, cfg, y1.data(), H);
    auto e2 = k.execute(0, 0, v, x.data(), H, 1.0f, cfg, y2.data(), H);
    if (e1 != dee::ktbridge::ExecuteError::kOk) { printf("ref err %d\n", (int)e1); return 1; }
    if (e2 != dee::ktbridge::ExecuteError::kOk) { printf("kt err %d\n", (int)e2); return 2; }
    std::vector<float> y1b(H, 0);
    r.execute(0, 0, v, x.data(), H, 1.0f, cfg, y1b.data(), H);
    for (size_t i = 0; i < H; ++i)
        if (y1[i] != y1b[i]) { printf("nondet %zu\n", i); return 3; }
    for (float f : y1)
        if (!(f == f) || f > 1e30f || f < -1e30f) { printf("nonfinite\n"); return 4; }
    s1[0] = 0xFF;
    auto e3 = r.execute(0, 0, v, x.data(), H, 1.0f, cfg, y1.data(), H);
    if (e3 != dee::ktbridge::ExecuteError::kScale) { printf("ff not rejected\n"); return 5; }
    printf("smoke OK ref=%.6f kt=%.6f\n", y1[0], y2[0]);
    return 0;
}
