// tests/cuda_stub/cuda_stub_control.h
//
// Test-facing control surface for the mock CUDA runtime (cuda_stub.cpp).
// The mock models ONE stream and ONE device: every successful enqueue
// (cudaMemcpyAsync, dee::*_cuda conversion kernels) increments `pending`,
// and cudaStreamSynchronize drains it — letting a test observe both that a
// failed submit drained the stream and that the per-transfer event was
// destroyed instead of leaked.
//
// Fault model: fail_at["entry_point"] = N fails the Nth call (1-based) to
// that entry point with cudaErrorUnknown (dee:: conversion kernels return
// false). A failed call still leaves its enqueued work behind, matching the
// defensive-drain contract ("a failed call may have launched work before the
// error surfaced").
#pragma once

#include <cuda_runtime.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace dee_stub {

struct Control {
    std::unordered_map<std::string, int> calls;     // calls seen per entry point
    std::unordered_map<std::string, int> fail_at;   // call index to fail (0 = never)
    std::vector<std::string> log;                   // ordered entry-point names

    int pending = 0;            // enqueued work units on the mock stream
    int stream_syncs = 0;       // cudaStreamSynchronize calls
    int syncs_with_pending = 0; // drains that actually cleared in-flight work
    int live_events = 0;        // created minus destroyed
    int live_streams = 0;
    int live_device_allocs = 0;
    int live_host_allocs = 0;
    int host_registered = 0;

    void reset() { *this = Control{}; }
    // Fail the next call to `name` (or the Nth when call_index is given).
    void fail_next(const char* name, int call_index = 0) {
        fail_at[name] = call_index > 0 ? call_index : calls[name] + 1;
    }
};

extern Control g;

// Entry-point bookkeeping shared by every mock function: logs the call,
// returns true when this call index is the injected failure.
bool should_fail(const char* name);

}  // namespace dee_stub
