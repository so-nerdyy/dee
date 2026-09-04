// Isolated T4 FP4-decode microbench prototype (NOT production).
//
// Compares the production decode algorithm (scalar port of
// dee::fp4_e2m1_to_f16_kernel, one thread per element) against a vectorized
// candidate (one thread per 32-element scale block, 4x uint32 packed loads).
// Both compute value = table[nibble] * scale via __float2half_rn, so the
// candidate is bitwise-identical by construction; the binary verifies with
// memcmp and reports timings as JSON.
//
// Build on a CUDA host (Kaggle 2xT4):
//   nvcc -O3 -arch=sm_75 -o fp4_decode_vec fp4_decode_vec.cu
// Run:
//   ./fp4_decode_vec --out 2048 --in 4096 [--iters N] [--verify-only]
//   ./fp4_decode_vec --packed-file w.bin --scale-file s.bin \
//       --out 2048 --in 4096 [--iters N]
// The .bin files are raw little-endian bytes, no header.
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

// Official 16-entry e2m1fn table (convert.py), bit-identical to production.
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

// Reference: verbatim port of production fp4_e2m1_to_f16_kernel.
__global__ void fp4_decode_ref(const uint8_t* packed, const uint8_t* scale,
                               __half* dst, size_t out, size_t in) {
  __shared__ float s_table[16];
  if (threadIdx.x < 16) s_table[threadIdx.x] = kFp4E2m1Table[threadIdx.x];
  __syncthreads();
  const size_t o = blockIdx.y;
  const size_t i =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= in) return;
  const size_t half_in = in >> 1;
  const size_t scale_stride = in >> 5;
  const uint8_t byte = packed[o * half_in + (i >> 1)];
  const uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0F);
  const float s = fp4_scale_value(scale[o * scale_stride + (i >> 5)]);
  dst[o * in + i] = __float2half_rn(s_table[nibble] * s);
}

// Candidate: one thread decodes a full 32-element scale block using four
// vectorized uint32 packed loads and a single scale fetch. Same table, same
// scale float, same __float2half_rn per element -> bitwise identical output.
__global__ void fp4_decode_vec(const uint8_t* packed, const uint8_t* scale,
                               __half* dst, size_t out, size_t in) {
  __shared__ float s_table[16];
  if (threadIdx.x < 16) s_table[threadIdx.x] = kFp4E2m1Table[threadIdx.x];
  __syncthreads();
  const size_t chunks_per_row = in >> 5;
  const size_t line =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (line >= out * chunks_per_row) return;
  const size_t o = line / chunks_per_row;
  const size_t chunk = line % chunks_per_row;
  const size_t i0 = chunk << 5;
  const uint32_t* v = reinterpret_cast<const uint32_t*>(
      packed + o * (in >> 1) + (i0 >> 1));
  const uint32_t w0 = v[0], w1 = v[1], w2 = v[2], w3 = v[3];
  const float s = fp4_scale_value(scale[o * chunks_per_row + chunk]);
  __half* drow = dst + o * in + i0;
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    const uint32_t w = k < 8 ? w0 : (k < 16 ? w1 : (k < 24 ? w2 : w3));
    const uint8_t nib = static_cast<uint8_t>((w >> ((k & 7) * 4)) & 0xFu);
    drow[k] = __float2half_rn(s_table[nib] * s);
  }
}

static uint64_t lcg_state = 0x12345678u;
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
  size_t out = 2048, in = 4096;
  int iters = 200, warmup = 20;
  const char* packed_file = nullptr;
  const char* scale_file = nullptr;
  bool verify_only = false;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--out") && a + 1 < argc) out = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--in") && a + 1 < argc) in = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--iters") && a + 1 < argc) iters = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--packed-file") && a + 1 < argc) packed_file = argv[++a];
    else if (!std::strcmp(argv[a], "--scale-file") && a + 1 < argc) scale_file = argv[++a];
    else if (!std::strcmp(argv[a], "--verify-only")) verify_only = true;
  }
  if (in == 0 || out == 0 || (in % 64) != 0 || (in % 32) != 0) {
    std::printf("{\"status\":\"bad-args\",\"hint\":\"in must be a multiple of 64\"}\n");
    return 2;
  }
  int device = -1;
  CK(cudaGetDevice(&device), "get-device");
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, device), "get-props");

  const size_t packed_n = out * (in >> 1);
  const size_t scale_n = out * (in >> 5);
  const size_t elems = out * in;
  std::vector<uint8_t> h_packed(packed_n), h_scale(scale_n);
  if (packed_file && scale_file) {
    if (!read_file(packed_file, h_packed, packed_n) ||
        !read_file(scale_file, h_scale, scale_n)) {
      std::printf("{\"status\":\"fixture-error\"}\n");
      return 2;
    }
  } else {
    for (size_t k = 0; k < packed_n; ++k) h_packed[k] = lcg_byte();
    for (size_t k = 0; k < scale_n; ++k)
      h_scale[k] = static_cast<uint8_t>(0x7d + (lcg_byte() % 6));  // exponents -2..3
  }

  uint8_t *d_packed = nullptr, *d_scale = nullptr;
  __half *d_ref = nullptr, *d_cand = nullptr;
  CK(cudaMalloc(&d_packed, packed_n), "malloc-packed");
  CK(cudaMalloc(&d_scale, scale_n), "malloc-scale");
  CK(cudaMalloc(&d_ref, elems * sizeof(__half)), "malloc-ref");
  CK(cudaMalloc(&d_cand, elems * sizeof(__half)), "malloc-cand");
  CK(cudaMemcpy(d_packed, h_packed.data(), packed_n, cudaMemcpyHostToDevice),
     "h2d-packed");
  CK(cudaMemcpy(d_scale, h_scale.data(), scale_n, cudaMemcpyHostToDevice),
     "h2d-scale");

  const dim3 ref_grid(static_cast<unsigned int>((in + 255) / 256),
                      static_cast<unsigned int>(out));
  const size_t chunks = out * (in >> 5);
  const unsigned int vec_blocks =
      static_cast<unsigned int>((chunks + 255) / 256);
  for (int w = 0; w < warmup; ++w) {
    fp4_decode_ref<<<ref_grid, 256>>>(d_packed, d_scale, d_ref, out, in);
    fp4_decode_vec<<<vec_blocks, 256>>>(d_packed, d_scale, d_cand, out, in);
  }
  CK(cudaDeviceSynchronize(), "warmup-sync");

  fp4_decode_ref<<<ref_grid, 256>>>(d_packed, d_scale, d_ref, out, in);
  fp4_decode_vec<<<vec_blocks, 256>>>(d_packed, d_scale, d_cand, out, in);
  CK(cudaDeviceSynchronize(), "verify-sync");
  std::vector<uint8_t> h_ref(elems * 2), h_cand(elems * 2);
  CK(cudaMemcpy(h_ref.data(), d_ref, elems * 2, cudaMemcpyDeviceToHost),
     "d2h-ref");
  CK(cudaMemcpy(h_cand.data(), d_cand, elems * 2, cudaMemcpyDeviceToHost),
     "d2h-cand");
  size_t first_diff = elems;
  for (size_t k = 0; k < elems * 2; ++k) {
    if (h_ref[k] != h_cand[k]) {
      first_diff = k / 2;
      break;
    }
  }
  const bool bitwise = (first_diff == elems);

  float ref_ms = -1.0f, vec_ms = -1.0f;
  if (!verify_only && bitwise) {
    cudaEvent_t s0, s1;
    CK(cudaEventCreate(&s0), "ev0");
    CK(cudaEventCreate(&s1), "ev1");
    CK(cudaEventRecord(s0), "rec0");
    for (int k = 0; k < iters; ++k)
      fp4_decode_ref<<<ref_grid, 256>>>(d_packed, d_scale, d_ref, out, in);
    CK(cudaEventRecord(s1), "rec1");
    CK(cudaEventSynchronize(s1), "sync-ref");
    CK(cudaEventElapsedTime(&ref_ms, s0, s1), "elapsed-ref");
    ref_ms /= iters;
    CK(cudaEventRecord(s0), "rec2");
    for (int k = 0; k < iters; ++k)
      fp4_decode_vec<<<vec_blocks, 256>>>(d_packed, d_scale, d_cand, out, in);
    CK(cudaEventRecord(s1), "rec3");
    CK(cudaEventSynchronize(s1), "sync-vec");
    CK(cudaEventElapsedTime(&vec_ms, s0, s1), "elapsed-vec");
    vec_ms /= iters;
    cudaEventDestroy(s0);
    cudaEventDestroy(s1);
  }
  std::printf("{"
              "\"status\":\"ok\","
              "\"device\":\"%s\",\"sm\":\"%d%d\","
              "\"out\":%zu,\"in\":%zu,"
              "\"packed_bytes\":%zu,\"decoded_bytes\":%zu,"
              "\"bitwise_identical\":%s,\"first_diff_elem\":%s,"
              "\"ref_ms\":%.6f,\"vec_ms\":%.6f,\"iters\":%d}\n",
              prop.name, prop.major, prop.minor, out, in, packed_n + scale_n,
              elems * 2, bitwise ? "true" : "false",
              bitwise ? "null" : std::to_string(first_diff).c_str(), ref_ms,
              vec_ms, verify_only ? 0 : iters);
  cudaFree(d_packed);
  cudaFree(d_scale);
  cudaFree(d_ref);
  cudaFree(d_cand);
  return bitwise ? 0 : 1;
}
