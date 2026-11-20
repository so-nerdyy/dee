#include "dee/profiling.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace {

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(
        stderr, "%s: %s (%s)\n", operation, cudaGetErrorName(status),
        cudaGetErrorString(status));
    return false;
}

}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    void* source = nullptr;
    void* destination = nullptr;
    bool ok =
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags") &&
        check(cudaMalloc(&source, 4096), "cudaMalloc(source)") &&
        check(cudaMalloc(&destination, 4096), "cudaMalloc(destination)");

    dee::StageProfiler profiler;
    profiler.configure(true, false, 4096, 256, true);
    if (ok) {
        ok = profiler.begin_cuda_timeline(stream, stream);
    }
    if (ok) {
        profiler.set_cuda_context(0, 0, 7, 4096);
        const size_t ticket = profiler.cuda_begin(
            dee::GpuStage::D2DGather, static_cast<void*>(stream));
        ok =
            check(
                cudaMemcpyAsync(
                    destination, source, 4096, cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync(D2D)") &&
            profiler.cuda_end(ticket, static_cast<void*>(stream)) &&
            check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(epoch 1)") &&
            profiler.cuda_collect_ready() &&
            profiler.pending_cuda_samples() == 0;
        profiler.note_d2d_gather_copy(4096);
    }
    const dee::StageProfile first = profiler.finish(1.0, 0, 0, 0, 0, 0, 0);
    ok = ok &&
        first.gpu_samples[
            static_cast<size_t>(dee::GpuStage::D2DGather)] == 1 &&
        first.d2d_gather_copies == 1 &&
        first.d2d_gather_bytes == 4096;

    // Mirror Engine::reset_external_profile: streams are synchronized and all
    // ready samples are collected against the old origin before configure()
    // clears the epoch and begin_cuda_timeline() replaces that origin.
    if (ok) {
        const size_t stale_ticket = profiler.cuda_begin(
            dee::GpuStage::D2DScatter, static_cast<void*>(stream));
        ok =
            check(
                cudaMemcpyAsync(
                    source, destination, 4096, cudaMemcpyDeviceToDevice, stream),
                "cudaMemcpyAsync(stale D2D)") &&
            profiler.cuda_end(stale_ticket, static_cast<void*>(stream)) &&
            check(
                cudaStreamSynchronize(stream),
                "cudaStreamSynchronize(stale epoch)") &&
            profiler.cuda_collect_ready() &&
            profiler.pending_cuda_samples() == 0;
    }
    if (ok) {
        profiler.configure(true, false, 4096, 256, true);
        ok = profiler.begin_cuda_timeline(stream, stream);
    }
    const dee::StageProfile second = profiler.finish(1.0, 0, 0, 0, 0, 0, 0);
    ok = ok &&
        second.gpu_samples[
            static_cast<size_t>(dee::GpuStage::D2DGather)] == 0 &&
        second.timeline.empty() &&
        profiler.pending_cuda_samples() == 0;

    if (source) cudaFree(source);
    if (destination) cudaFree(destination);
    if (stream) cudaStreamDestroy(stream);
    std::printf("CUDA profiling epoch reset: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
