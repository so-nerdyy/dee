// Isolated C1-PRE tile-local decode/consume prototype (NOT production).
//
// Premise under test: "a packed MXFP4 tile can be decoded immediately before
// use without materializing the full FP16 projection." One output row is the
// tile: the candidate kernel decodes packed nibbles + e8m0 scales on the fly
// and accumulates the dot product in the SAME order as the reference (which
// dot-products a pre-decoded FP16 row). Identical FP op order + identical
// per-element rounding (__float2half_rn BEFORE the multiply, exactly as
// production decode does) makes the comparison a BITWISE gate, not a
// tolerance gate. No full-GEMM speedup is claimed from this tile test.
//
// Build (Kaggle T4): nvcc -O3 -arch=sm_75 -o c1_tile_consume
//     c1_tile_consume.cu
// Run: ./c1_tile_consume --rows 512 --in 4096 [--iters N] [--verify-only]
//      [--packed-file p.bin --scale-file s.bin --x-file x.f16]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(call, what)                                                 \
  do {                                                                 \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
      std::printf("{\"status\":\"cuda-error\",\"where\":\"%s\","        \
                  "\"cuda_error\":\"%s\"}\n",                          \
                  what, cudaGetErrorString(_e));                       \
      return 2;                                                        \
    }                                                                  \
  } while (0)

__constant__ float kFp4E2m1Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float fp4_scale_value(uint8_t bits) {
  const int exponent = static_cast<int>(bits) - 127;
  if (exponent >= 127) return __int_as_float(0x7F7FFFFFu);
  if (exponent <= -127) return 0.0f;
  return __uint_as_float(static_cast<unsigned int>(exponent + 127) << 23);
}

constexpr int kTileThreads = 32;

// Production scalar decode port (reference materialization path).
__global__ void fp4_decode_scalar(const uint8_t* packed, const uint8_t* scale,
                                  __half* dst, size_t rows, size_t in) {
  __shared__ float s_table[16];
  if (threadIdx.x < 16) s_table[threadIdx.x] = kFp4E2m1Table[threadIdx.x];
  __syncthreads();
  const size_t o = blockIdx.y;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (o >= rows || i >= in) return;
  const uint8_t byte = packed[o * (in >> 1) + (i >> 1)];
  const uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);
  const float s = fp4_scale_value(scale[o * (in >> 5) + (i >> 5)]);
  dst[o * in + i] = __float2half_rn(s_table[nibble] * s);
}

// Reference consumer: dot over PRE-DECODED fp16 row. 32 threads stride over
// the row with sequential fmaf chains; thread 0 reduces in thread order.
__global__ void dot_predecoded(const __half* W, const __half* x, float* y,
                               size_t rows, size_t in) {
  __shared__ float part[kTileThreads];
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const __half* w = W + row * in;
  const int t = threadIdx.x;
  float acc = 0.0f;
  for (size_t i = t; i < in; i += kTileThreads)
    acc = fmaf(__half2float(w[i]), __half2float(x[i]), acc);
  part[t] = acc;
  __syncthreads();
  if (t == 0) {
    float s = 0.0f;
    for (int k = 0; k < kTileThreads; ++k) s += part[k];
    y[row] = s;
  }
}

// Candidate: SAME accumulation skeleton, but each weight is decoded on the
// fly (table lookup + scale + __float2half_rn, exactly the production
// per-element semantics) instead of loaded from a materialized row.
__global__ void dot_tile_consume(const uint8_t* packed, const uint8_t* scale,
                                 const __half* x, float* y, size_t rows,
                                 size_t in) {
  __shared__ float s_table[16];
  __shared__ float part[kTileThreads];
  if (threadIdx.x < 16) s_table[threadIdx.x] = kFp4E2m1Table[threadIdx.x];
  __syncthreads();
  const size_t row = blockIdx.x;
  if (row >= rows) return;
  const uint8_t* prow = packed + row * (in >> 1);
  const uint8_t* srow = scale + row * (in >> 5);
  const int t = threadIdx.x;
  float acc = 0.0f;
  for (size_t i = t; i < in; i += kTileThreads) {
    const uint8_t byte = prow[i >> 1];
    const uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    const float s = fp4_scale_value(srow[i >> 5]);
    const float wv = __half2float(__float2half_rn(s_table[nibble] * s));
    acc = fmaf(wv, __half2float(x[i]), acc);
  }
  part[t] = acc;
  __syncthreads();
  if (t == 0) {
    float s = 0.0f;
    for (int k = 0; k < kTileThreads; ++k) s += part[k];
    y[row] = s;
  }
}

static uint64_t lcg_state = 0xC1U;
static uint8_t lcg_byte() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<uint8_t>(lcg_state >> 24);
}

static bool read_file(const char* path, std::vector<uint8_t>& out,
                      size_t expect) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  out.resize(expect);
  const size_t n = std::fread(out.data(), 1, expect, f);
  std::fclose(f);
  return n == expect;
}

int main(int argc, char** argv) {
  size_t rows = 512, in = 4096;
  int iters = 200, warmup = 20;
  const char *pf = nullptr, *sf = nullptr, *xf = nullptr;
  bool verify_only = false;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--rows") && a + 1 < argc) rows = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--in") && a + 1 < argc) in = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--iters") && a + 1 < argc) iters = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--packed-file") && a + 1 < argc) pf = argv[++a];
    else if (!std::strcmp(argv[a], "--scale-file") && a + 1 < argc) sf = argv[++a];
    else if (!std::strcmp(argv[a], "--x-file") && a + 1 < argc) xf = argv[++a];
    else if (!std::strcmp(argv[a], "--verify-only")) verify_only = true;
  }
  if (in == 0 || rows == 0 || (in % 64) != 0) {
    std::printf("{\"status\":\"bad-args\"}\n");
    return 2;
  }
  int device = -1;
  CK(cudaGetDevice(&device), "get-device");
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, device), "get-props");

  const size_t packed_n = rows * (in >> 1);
  const size_t scale_n = rows * (in >> 5);
  std::vector<uint8_t> h_p(packed_n), h_s(scale_n), h_x(in * 2);
  const bool real = pf && sf;
  if (real) {
    if (!read_file(pf, h_p, packed_n) || !read_file(sf, h_s, scale_n)) {
      std::printf("{\"status\":\"fixture-error\"}\n");
      return 2;
    }
  } else {
    for (size_t k = 0; k < packed_n; ++k) h_p[k] = lcg_byte();
    for (size_t k = 0; k < scale_n; ++k)
      h_s[k] = static_cast<uint8_t>(0x7d + (lcg_byte() % 6));
  }
  if (xf) {
    if (!read_file(xf, h_x, in * 2)) {
      std::printf("{\"status\":\"fixture-error\"}\n");
      return 2;
    }
  } else {
    for (size_t k = 0; k < in * 2; ++k) h_x[k] = lcg_byte();
  }

  uint8_t *d_p = nullptr, *d_s = nullptr;
  __half *d_w = nullptr, *d_x = nullptr;
  float *d_yref = nullptr, *d_ycand = nullptr;
  CK(cudaMalloc(&d_p, packed_n), "m-p");
  CK(cudaMalloc(&d_s, scale_n), "m-s");
  CK(cudaMalloc(&d_w, rows * in * 2), "m-w");
  CK(cudaMalloc(&d_x, in * 2), "m-x");
  CK(cudaMalloc(&d_yref, rows * 4), "m-yr");
  CK(cudaMalloc(&d_ycand, rows * 4), "m-yc");
  CK(cudaMemcpy(d_p, h_p.data(), packed_n, cudaMemcpyHostToDevice), "h-p");
  CK(cudaMemcpy(d_s, h_s.data(), scale_n, cudaMemcpyHostToDevice), "h-s");
  CK(cudaMemcpy(d_x, h_x.data(), in * 2, cudaMemcpyHostToDevice), "h-x");

  const dim3 dec_grid(static_cast<unsigned int>((in + 255) / 256),
                      static_cast<unsigned int>(rows));
  const unsigned int row_blocks = static_cast<unsigned int>(rows);
  for (int w = 0; w < warmup; ++w) {
    fp4_decode_scalar<<<dec_grid, 256>>>(d_p, d_s, d_w, rows, in);
    dot_predecoded<<<row_blocks, kTileThreads>>>(d_w, d_x, d_yref, rows, in);
    dot_tile_consume<<<row_blocks, kTileThreads>>>(d_p, d_s, d_x, d_ycand,
                                                   rows, in);
  }
  CK(cudaDeviceSynchronize(), "warm-sync");

  fp4_decode_scalar<<<dec_grid, 256>>>(d_p, d_s, d_w, rows, in);
  dot_predecoded<<<row_blocks, kTileThreads>>>(d_w, d_x, d_yref, rows, in);
  dot_tile_consume<<<row_blocks, kTileThreads>>>(d_p, d_s, d_x, d_ycand, rows,
                                                 in);
  CK(cudaDeviceSynchronize(), "verify-sync");
  std::vector<uint8_t> h_yr(rows * 4), h_yc(rows * 4);
  CK(cudaMemcpy(h_yr.data(), d_yref, rows * 4, cudaMemcpyDeviceToHost), "d-yr");
  CK(cudaMemcpy(h_yc.data(), d_ycand, rows * 4, cudaMemcpyDeviceToHost), "d-yc");
  const bool bitwise = !std::memcmp(h_yr.data(), h_yc.data(), rows * 4);

  float ref_ms = -1.0f, cand_ms = -1.0f;
  if (!verify_only && bitwise) {
    cudaEvent_t s0, s1;
    CK(cudaEventCreate(&s0), "ev0");
    CK(cudaEventCreate(&s1), "ev1");
    CK(cudaEventRecord(s0), "r0");
    for (int k = 0; k < iters; ++k) {
      fp4_decode_scalar<<<dec_grid, 256>>>(d_p, d_s, d_w, rows, in);
      dot_predecoded<<<row_blocks, kTileThreads>>>(d_w, d_x, d_yref, rows, in);
    }
    CK(cudaEventRecord(s1), "r1");
    CK(cudaEventSynchronize(s1), "s-r");
    CK(cudaEventElapsedTime(&ref_ms, s0, s1), "e-r");
    ref_ms /= iters;
    CK(cudaEventRecord(s0), "r2");
    for (int k = 0; k < iters; ++k)
      dot_tile_consume<<<row_blocks, kTileThreads>>>(d_p, d_s, d_x, d_ycand,
                                                     rows, in);
    CK(cudaEventRecord(s1), "r3");
    CK(cudaEventSynchronize(s1), "s-c");
    CK(cudaEventElapsedTime(&cand_ms, s0, s1), "e-c");
    cand_ms /= iters;
    cudaEventDestroy(s0);
    cudaEventDestroy(s1);
  }
  const size_t decoded = rows * in * 2;
  std::printf("{"
              "\"status\":\"ok\",\"test\":\"c1-tile-consume\","
              "\"device\":\"%s\",\"sm\":\"%d%d\","
              "\"fixture\":\"%s\",\"rows\":%zu,\"in\":%zu,"
              "\"packed_bytes\":%zu,"
              "\"decoded_bytes_avoided\":%zu,"
              "\"tile_temp_bytes_per_block\":%d,"
              "\"launches_ref\":2,\"launches_cand\":1,"
              "\"bitwise_identical\":%s,"
              "\"admission\":\"%s\","
              "\"ref_ms\":%.6f,\"cand_ms\":%.6f,\"iters\":%d}\n",
              prop.name, prop.major, prop.minor, real ? "real" : "synthetic",
              rows, in, packed_n + scale_n, 2 * decoded,
              static_cast<int>(sizeof(float) * (16 + kTileThreads)),
              bitwise ? "true" : "false",
              bitwise ? "BITWISE_IDENTICAL" : "NUMERICALLY_DIFFERENT", ref_ms,
              cand_ms, verify_only ? 0 : iters);
  cudaFree(d_p);
  cudaFree(d_s);
  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_yref);
  cudaFree(d_ycand);
  return bitwise ? 0 : 1;
}
