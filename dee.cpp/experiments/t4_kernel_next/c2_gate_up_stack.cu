// Isolated C2 gate/up consolidation prototype (NOT production).
//
// Question: can the two FP16 GEMMs (W1 gate, W3 up, each m=2048) be replaced
// by ONE m=4096 GEMM over a stacked [W1;W3] buffer with bitwise-identical
// output? Concatenation is NOT assumed exact: cuBLAS may choose a different
// algorithm per (m,n,k), so the binary gates BITWISE_IDENTICAL vs
// NUMERICALLY_DIFFERENT and reports timings either way (timings are data,
// not an exactness claim).
//
// Faithfulness: starts from official packed W1/W3 bytes + e8m0 scales,
// decodes with the production scalar algorithm into (a) separate buffers and
// (b) one stacked buffer, verifies (a)==(b) halves bitwise, then compares
// 2x(m=2048) vs 1x(m=4096) cublasGemmEx with the production epilogue
// (FP16 in, FP16 out, FP32 compute, TENSOR_OP).
//
// Build (Kaggle T4): nvcc -O3 -arch=sm_75 -o c2_gate_up_stack
//     c2_gate_up_stack.cu -lcublas
// Run: ./c2_gate_up_stack --out 2048 --in 4096 [--iters N] [--verify-only]
//      [--packed1 w1.bin --scale1 w1s.bin --packed3 w3.bin --scale3 w3s.bin]
//      [--gate-file g.f16 --up-file u.f16 --x-file x.f16] (raw bytes, skip decode)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cublas_v2.h>
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

#define CB(call, what)                                                 \
  do {                                                                 \
    cublasStatus_t _s = (call);                                        \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                 \
      std::printf("{\"status\":\"cublas-error\",\"where\":\"%s\","      \
                  "\"code\":%d}\n",                                    \
                  what, static_cast<int>(_s));                         \
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

// Production scalar decode port (one thread per element).
__global__ void fp4_decode_scalar(const uint8_t* packed, const uint8_t* scale,
                                  __half* dst, size_t out, size_t in) {
  __shared__ float s_table[16];
  if (threadIdx.x < 16) s_table[threadIdx.x] = kFp4E2m1Table[threadIdx.x];
  __syncthreads();
  const size_t o = blockIdx.y;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= in) return;
  const uint8_t byte = packed[o * (in >> 1) + (i >> 1)];
  const uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);
  const float s = fp4_scale_value(scale[o * (in >> 5) + (i >> 5)]);
  dst[o * in + i] = __float2half_rn(s_table[nibble] * s);
}

static uint64_t lcg_state = 0xC2U;
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

// Production-epilogue GEMM: row-major FP16 W[rows,cols] x FP16 in[cols] ->
// FP16 out[rows], FP32 accumulate, tensor-op. Mirrors
// dee::gemm_fp16_row_major_to_fp16.
static cublasStatus_t prod_gemm(cublasHandle_t h, const __half* w,
                                const __half* x, __half* y, int rows,
                                int cols, cudaStream_t s) {
  const float alpha = 1.0f, beta = 0.0f;
  return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols, &alpha, w,
                      CUDA_R_16F, cols, x, CUDA_R_16F, cols, &beta, y,
                      CUDA_R_16F, rows, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

int main(int argc, char** argv) {
  size_t out = 2048, in = 4096;
  int iters = 200, warmup = 20;
  const char *p1f = nullptr, *s1f = nullptr, *p3f = nullptr, *s3f = nullptr;
  bool verify_only = false;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--out") && a + 1 < argc) out = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--in") && a + 1 < argc) in = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--iters") && a + 1 < argc) iters = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--packed1") && a + 1 < argc) p1f = argv[++a];
    else if (!std::strcmp(argv[a], "--scale1") && a + 1 < argc) s1f = argv[++a];
    else if (!std::strcmp(argv[a], "--packed3") && a + 1 < argc) p3f = argv[++a];
    else if (!std::strcmp(argv[a], "--scale3") && a + 1 < argc) s3f = argv[++a];
    else if (!std::strcmp(argv[a], "--verify-only")) verify_only = true;
  }
  if (in == 0 || out == 0 || (in % 64) != 0) {
    std::printf("{\"status\":\"bad-args\"}\n");
    return 2;
  }
  int device = -1;
  CK(cudaGetDevice(&device), "get-device");
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, device), "get-props");

  const size_t proj_packed = out * (in >> 1);
  const size_t proj_scale = out * (in >> 5);
  const size_t proj_elems = out * in;
  std::vector<uint8_t> h_p1(proj_packed), h_s1(proj_scale),
      h_p3(proj_packed), h_s3(proj_scale);
  const bool real = p1f && s1f && p3f && s3f;
  if (real) {
    if (!read_file(p1f, h_p1, proj_packed) || !read_file(s1f, h_s1, proj_scale) ||
        !read_file(p3f, h_p3, proj_packed) || !read_file(s3f, h_s3, proj_scale)) {
      std::printf("{\"status\":\"fixture-error\"}\n");
      return 2;
    }
  } else {
    for (size_t k = 0; k < proj_packed; ++k) {
      h_p1[k] = lcg_byte();
      h_p3[k] = lcg_byte();
    }
    for (size_t k = 0; k < proj_scale; ++k) {
      h_s1[k] = static_cast<uint8_t>(0x7d + (lcg_byte() % 6));
      h_s3[k] = static_cast<uint8_t>(0x7d + (lcg_byte() % 6));
    }
  }
  std::vector<uint8_t> h_x(in * 2);
  for (size_t k = 0; k < in * 2; ++k) h_x[k] = lcg_byte();

  uint8_t *d_p1 = nullptr, *d_s1 = nullptr, *d_p3 = nullptr, *d_s3 = nullptr;
  __half *d_w1 = nullptr, *d_w3 = nullptr, *d_stack = nullptr, *d_x = nullptr;
  __half *d_gate = nullptr, *d_up = nullptr, *d_both = nullptr;
  CK(cudaMalloc(&d_p1, proj_packed), "m-p1");
  CK(cudaMalloc(&d_s1, proj_scale), "m-s1");
  CK(cudaMalloc(&d_p3, proj_packed), "m-p3");
  CK(cudaMalloc(&d_s3, proj_scale), "m-s3");
  CK(cudaMalloc(&d_w1, proj_elems * 2), "m-w1");
  CK(cudaMalloc(&d_w3, proj_elems * 2), "m-w3");
  CK(cudaMalloc(&d_stack, 2 * proj_elems * 2), "m-stack");
  CK(cudaMalloc(&d_x, in * 2), "m-x");
  CK(cudaMalloc(&d_gate, out * 2), "m-gate");
  CK(cudaMalloc(&d_up, out * 2), "m-up");
  CK(cudaMalloc(&d_both, 2 * out * 2), "m-both");
  CK(cudaMemcpy(d_p1, h_p1.data(), proj_packed, cudaMemcpyHostToDevice), "h-p1");
  CK(cudaMemcpy(d_s1, h_s1.data(), proj_scale, cudaMemcpyHostToDevice), "h-s1");
  CK(cudaMemcpy(d_p3, h_p3.data(), proj_packed, cudaMemcpyHostToDevice), "h-p3");
  CK(cudaMemcpy(d_s3, h_s3.data(), proj_scale, cudaMemcpyHostToDevice), "h-s3");
  CK(cudaMemcpy(d_x, h_x.data(), in * 2, cudaMemcpyHostToDevice), "h-x");

  cublasHandle_t cublas{};
  CB(cublasCreate(&cublas), "cublas-create");
  cudaStream_t stream = nullptr;
  CK(cudaStreamCreate(&stream), "stream");
  CB(cublasSetStream(cublas, stream), "cublas-stream");

  const dim3 grid(static_cast<unsigned int>((in + 255) / 256),
                  static_cast<unsigned int>(out));
  // Decode: separate buffers AND stacked halves (same algorithm).
  fp4_decode_scalar<<<grid, 256, 0, stream>>>(d_p1, d_s1, d_w1, out, in);
  fp4_decode_scalar<<<grid, 256, 0, stream>>>(d_p3, d_s3, d_w3, out, in);
  fp4_decode_scalar<<<grid, 256, 0, stream>>>(d_p1, d_s1, d_stack, out, in);
  fp4_decode_scalar<<<grid, 256, 0, stream>>>(
      d_p3, d_s3, d_stack + proj_elems, out, in);
  CK(cudaStreamSynchronize(stream), "decode-sync");

  // Stacked-halves == separate buffers (decode-layout equivalence)?
  std::vector<uint8_t> h_w1(proj_elems * 2), h_s1h(proj_elems * 2),
      h_w3(proj_elems * 2), h_s3h(proj_elems * 2);
  CK(cudaMemcpy(h_w1.data(), d_w1, proj_elems * 2, cudaMemcpyDeviceToHost), "d-w1");
  CK(cudaMemcpy(h_w3.data(), d_w3, proj_elems * 2, cudaMemcpyDeviceToHost), "d-w3");
  CK(cudaMemcpy(h_s1h.data(), d_stack, proj_elems * 2, cudaMemcpyDeviceToHost), "d-s1");
  CK(cudaMemcpy(h_s3h.data(), d_stack + proj_elems, proj_elems * 2,
                cudaMemcpyDeviceToHost), "d-s3");
  const bool stacked_ok =
      !std::memcmp(h_w1.data(), h_s1h.data(), proj_elems * 2) &&
      !std::memcmp(h_w3.data(), h_s3h.data(), proj_elems * 2);

  // Path A: 2x m=2048 GEMMs. Path B: 1x m=4096 GEMM over stacked.
  const int o = static_cast<int>(out), n = static_cast<int>(in);
  for (int w = 0; w < warmup; ++w) {
    CB(prod_gemm(cublas, d_w1, d_x, d_gate, o, n, stream), "w-a1");
    CB(prod_gemm(cublas, d_w3, d_x, d_up, o, n, stream), "w-a2");
    CB(prod_gemm(cublas, d_stack, d_x, d_both, 2 * o, n, stream), "w-b");
  }
  CK(cudaStreamSynchronize(stream), "warm-sync");
  CB(prod_gemm(cublas, d_w1, d_x, d_gate, o, n, stream), "a1");
  CB(prod_gemm(cublas, d_w3, d_x, d_up, o, n, stream), "a2");
  CB(prod_gemm(cublas, d_stack, d_x, d_both, 2 * o, n, stream), "b");
  CK(cudaStreamSynchronize(stream), "verify-sync");

  std::vector<uint8_t> h_gate(out * 2), h_up(out * 2), h_both(2 * out * 2);
  CK(cudaMemcpy(h_gate.data(), d_gate, out * 2, cudaMemcpyDeviceToHost), "d-g");
  CK(cudaMemcpy(h_up.data(), d_up, out * 2, cudaMemcpyDeviceToHost), "d-u");
  CK(cudaMemcpy(h_both.data(), d_both, 2 * out * 2, cudaMemcpyDeviceToHost), "d-b");
  bool gemm_bitwise = !std::memcmp(h_gate.data(), h_both.data(), out * 2) &&
                      !std::memcmp(h_up.data(), h_both.data() + out * 2, out * 2);
  const char* admission = "NUMERICALLY_DIFFERENT";
  if (stacked_ok && gemm_bitwise) admission = "BITWISE_IDENTICAL";

  float a_ms = -1.0f, b_ms = -1.0f;
  if (!verify_only) {
    cudaEvent_t s0, s1;
    CK(cudaEventCreate(&s0), "ev0");
    CK(cudaEventCreate(&s1), "ev1");
    CK(cudaEventRecord(s0, stream), "r0");
    for (int k = 0; k < iters; ++k) {
      CB(prod_gemm(cublas, d_w1, d_x, d_gate, o, n, stream), "t-a1");
      CB(prod_gemm(cublas, d_w3, d_x, d_up, o, n, stream), "t-a2");
    }
    CK(cudaEventRecord(s1, stream), "r1");
    CK(cudaEventSynchronize(s1), "s-a");
    CK(cudaEventElapsedTime(&a_ms, s0, s1), "e-a");
    a_ms /= iters;
    CK(cudaEventRecord(s0, stream), "r2");
    for (int k = 0; k < iters; ++k)
      CB(prod_gemm(cublas, d_stack, d_x, d_both, 2 * o, n, stream), "t-b");
    CK(cudaEventRecord(s1, stream), "r3");
    CK(cudaEventSynchronize(s1), "s-b");
    CK(cudaEventElapsedTime(&b_ms, s0, s1), "e-b");
    b_ms /= iters;
    cudaEventDestroy(s0);
    cudaEventDestroy(s1);
  }
  const size_t wbytes = proj_elems * 2;
  std::printf("{"
              "\"status\":\"ok\",\"test\":\"c2-gate-up-stack\","
              "\"device\":\"%s\",\"sm\":\"%d%d\","
              "\"fixture\":\"%s\",\"out\":%zu,\"in\":%zu,"
              "\"packed_bytes\":%zu,\"weight_bytes_per_path\":%zu,"
              "\"stacked_temp_bytes\":%zu,"
              "\"launches_path_a\":2,\"launches_path_b\":1,"
              "\"stacked_decode_identical\":%s,\"gemm_bitwise\":%s,"
              "\"admission\":\"%s\","
              "\"path_a_ms\":%.6f,\"path_b_ms\":%.6f,\"iters\":%d}\n",
              prop.name, prop.major, prop.minor, real ? "real" : "synthetic",
              out, in, 2 * (proj_packed + proj_scale), 2 * wbytes, 2 * wbytes,
              stacked_ok ? "true" : "false", gemm_bitwise ? "true" : "false",
              admission, a_ms, b_ms, verify_only ? 0 : iters);
  cublasDestroy(cublas);
  cudaStreamDestroy(stream);
  cudaFree(d_p1);
  cudaFree(d_s1);
  cudaFree(d_p3);
  cudaFree(d_s3);
  cudaFree(d_w1);
  cudaFree(d_w3);
  cudaFree(d_stack);
  cudaFree(d_x);
  cudaFree(d_gate);
  cudaFree(d_up);
  cudaFree(d_both);
  return (stacked_ok && gemm_bitwise) ? 0 : 1;
}
