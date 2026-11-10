// tools/fill_replay.cpp — fill-path replay + rider (host-only, no CUDA).
//
// Replays exact production expert-record access through the REAL
// Dee4ExpertStore::materialize (positional pread path) and the REAL
// HostPackCache::get_batch (lanes, queueing, LRU accounting), driven by a
// route journal or synthetic patterns. Measures what the seals cannot:
// per-request service distributions, QD scaling, cold-vs-warm (pass 1 vs
// pass 2), mincore residency, and short-read counts.
//
// Build (no CUDA needed):
//   cmake -S dee.cpp -B dee.cpp/build-replay -DDEE_CUDA=OFF -DDEE_BUILD_FILL_REPLAY=ON
//   cmake --build dee.cpp/build-replay --target fill_replay
//
// Modes:
//   replay --bank DIR --journal J [--lanes K] [--qdepth B]
//       J lines: "<layer> <expert>" in production call order (token-major).
//       Batches mimic prepare_fp4_experts (B records per get_batch).
//   rider --bank DIR [--lanes 1..8] [--pattern seq|rand]
//       Raw path capability: sequential vs random record reads, QD sweep,
//       pass 1 (cold-ish) vs pass 2 (warm). No cache, direct materialize.
//
// Output: fill_timeline.json (StageProfiler emitter schema) + summary JSON
// to stdout. Timings are host-wall measurements, never estimates.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "dee/expert_store.h"
#include "dee/host_pack_cache.h"
#include "dee/profiling.h"

namespace {

struct Args {
    std::string mode = "replay";
    std::string bank;
    std::string journal;
    std::string pattern = "journal";
    size_t lanes = 3;
    size_t qdepth = 6;
    size_t max_records = 0;  // 0 = all
    size_t budget_bytes = 0;  // 0 = 8.5 GiB production-like
    int passes = 2;
    std::string out = "fill_timeline.json";
};

bool parse_args(int argc, char** argv, Args* out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name, std::string* dst) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "fill_replay: %s needs a value\n", name);
                return false;
            }
            *dst = argv[++i];
            return true;
        };
        auto need_size = [&](const char* name, size_t* dst) {
            std::string text;
            if (!need(name, &text)) return false;
            *dst = static_cast<size_t>(std::stoull(text));
            return true;
        };
        if (arg == "--mode") {
            if (!need("--mode", &out->mode)) return false;
        } else if (arg == "--bank") {
            if (!need("--bank", &out->bank)) return false;
        } else if (arg == "--journal") {
            if (!need("--journal", &out->journal)) return false;
        } else if (arg == "--pattern") {
            if (!need("--pattern", &out->pattern)) return false;
        } else if (arg == "--lanes") {
            if (!need_size("--lanes", &out->lanes)) return false;
        } else if (arg == "--qdepth") {
            if (!need_size("--qdepth", &out->qdepth)) return false;
        } else if (arg == "--max-records") {
            if (!need_size("--max-records", &out->max_records)) return false;
        } else if (arg == "--budget-bytes") {
            if (!need_size("--budget-bytes", &out->budget_bytes)) return false;
        } else if (arg == "--passes") {
            size_t passes = 0;
            if (!need_size("--passes", &passes)) return false;
            out->passes = static_cast<int>(passes);
        } else if (arg == "--out") {
            if (!need("--out", &out->out)) return false;
        } else {
            std::fprintf(stderr, "fill_replay: unknown arg %s\n", arg.c_str());
            return false;
        }
    }
    if (out->bank.empty()) {
        std::fprintf(stderr, "fill_replay: --bank is required\n");
        return false;
    }
    if (out->mode != "replay" && out->mode != "rider") {
        std::fprintf(stderr, "fill_replay: --mode must be replay|rider\n");
        return false;
    }
    if (out->lanes == 0 || out->lanes > dee::HostPackCache::kMaxFillLanes) {
        std::fprintf(stderr, "fill_replay: --lanes out of range\n");
        return false;
    }
    return true;
}

struct FillThunk {
    dee::ExpertStore* store = nullptr;
    dee::ExpertView view{};
};

bool fill_thunk(void* context, uint8_t* dst, size_t nbytes) {
    auto* thunk = static_cast<FillThunk*>(context);
    return thunk && thunk->store &&
        thunk->store->materialize(thunk->view, dst, nbytes);
}

// Journal lines: "<layer> <expert> [token]" in production call order
// (token-major). The token labels attribution only, never timing.
bool load_journal(const std::string& path,
                  std::vector<std::pair<int, int>>* calls,
                  std::vector<int>* tokens) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::fprintf(stderr, "fill_replay: cannot open journal %s\n",
                     path.c_str());
        return false;
    }
    std::string line;
    size_t index = 0;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        int layer = -1, expert = -1;
        long token = -1;
        if (!(fields >> layer >> expert)) {
            std::fprintf(stderr, "fill_replay: malformed journal line\n");
            return false;
        }
        if (!(fields >> token)) token = static_cast<long>(index / 43);
        calls->emplace_back(layer, expert);
        tokens->push_back(static_cast<int>(token));
        ++index;
    }
    return !calls->empty();
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int run_replay(const Args& args, dee::Dee4ExpertStore* store,
               dee::HostPackCache* cache, dee::StageProfiler* profiler,
               const std::vector<std::pair<int, int>>& calls,
               const std::vector<int>& tokens) {
    if (!cache->set_fill_lanes(args.lanes)) {
        std::fprintf(stderr, "fill_replay: set_fill_lanes failed\n");
        return 2;
    }
    cache->set_fill_profiler(profiler);
    const size_t record_bytes = store->record_bytes();
    std::vector<dee::ExpertView> views(calls.size());
    std::vector<FillThunk> thunks(calls.size());
    for (size_t i = 0; i < calls.size(); ++i) {
        if (!store->get(calls[i].first, calls[i].second, &views[i])) {
            std::fprintf(stderr, "fill_replay: get(%d,%d) failed\n",
                         calls[i].first, calls[i].second);
            return 2;
        }
        thunks[i].store = store;
        thunks[i].view = views[i];
    }
    std::vector<dee::HostPackCache::BatchRequest> requests(args.qdepth);
    std::vector<dee::HostPackCache::BatchResult> results(args.qdepth);
    for (size_t first = 0; first < calls.size();) {
        const size_t batch_count =
            std::min(args.qdepth, calls.size() - first);
        int layer = calls[first].first;
        cache->set_fill_context(tokens[first], layer, 0);
        for (size_t i = 0; i < batch_count; ++i) {
            const size_t at = first + i;
            requests[i].key =
                (static_cast<uint64_t>(static_cast<uint32_t>(calls[at].first))
                 << 32) |
                static_cast<uint32_t>(calls[at].second);
            requests[i].nbytes = record_bytes;
            requests[i].fill = &fill_thunk;
            requests[i].context = &thunks[at];
            requests[i].source_order =
                static_cast<uint64_t>(views[at].record_index);
        }
        if (!cache->get_batch(requests.data(), batch_count, results.data())) {
            std::fprintf(stderr, "fill_replay: get_batch failed at %zu\n",
                         first);
            return 2;
        }
        for (size_t i = 0; i < batch_count; ++i) {
            if (!results[i].success) {
                std::fprintf(stderr, "fill_replay: fill failed at %zu\n",
                             first + i);
                return 2;
            }
        }
        first += batch_count;
    }
    return 0;
}

int run_rider(const Args& args, dee::Dee4ExpertStore* store) {
    const size_t record_bytes = store->record_bytes();
    const size_t stored = store->stored_records();
    if (stored == 0) {
        std::fprintf(stderr, "fill_replay: empty bank\n");
        return 2;
    }
    // Rider portability note: Dee4ExpertStore::get() requires (layer,expert)
    // within bank geometry. The rider therefore walks layers/experts in
    // bank order (sequential) or shuffled (random), resolving via get().
    std::vector<std::pair<int, int>> order;
    for (int layer = store->start_layer();
         layer < store->start_layer() + store->num_layers(); ++layer) {
        for (int expert = 0; expert < store->experts_per_layer(); ++expert) {
            order.emplace_back(layer, expert);
            if (args.max_records != 0 && order.size() >= args.max_records) break;
        }
        if (args.max_records != 0 && order.size() >= args.max_records) break;
    }
    if (args.pattern == "rand") {
        std::mt19937_64 rng(0x51ab1e);
        std::shuffle(order.begin(), order.end(), rng);
    } else if (args.pattern != "seq") {
        std::fprintf(stderr, "fill_replay: --pattern must be seq|rand\n");
        return 2;
    }
    std::cout << "{\"rider\":{\"pattern\":\"" << args.pattern
              << "\",\"records\":" << order.size() << ",\"lanes\":[";
    for (size_t lanes = 1; lanes <= args.lanes; ++lanes) {
        if (lanes > 1) std::cout << ',';
        std::vector<std::thread> workers;
        std::vector<double> per_lane_ms(lanes, 0.0);
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        auto start = std::chrono::steady_clock::now();
        for (size_t lane = 0; lane < lanes; ++lane) {
            workers.emplace_back([&, lane] {
                dee::ExpertView view{};
                std::vector<uint8_t> local(record_bytes);
                for (;;) {
                    const size_t index =
                        next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= order.size() || failed.load()) return;
                    if (!store->get(order[index].first, order[index].second,
                                    &view)) {
                        failed.store(true);
                        return;
                    }
                    const auto begin = std::chrono::steady_clock::now();
                    if (!store->materialize(view, local.data(), record_bytes)) {
                        failed.store(true);
                        return;
                    }
                    per_lane_ms[lane] +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
                }
            });
        }
        for (auto& worker : workers) worker.join();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count();
        if (failed.load()) {
            std::fprintf(stderr, "fill_replay: rider read failed\n");
            return 2;
        }
        const double gib =
            static_cast<double>(order.size() * record_bytes) /
            static_cast<double>(1ULL << 30);
        std::cout << "{\"lanes\":" << lanes << ",\"wall_ms\":" << wall_ms
                  << ",\"gibps\":" << gib / (wall_ms / 1000.0) << '}';
    }
    std::cout << "]}}" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) return 2;
    dee::Dee4ExpertStore store;
    if (!store.open(args.bank)) {
        std::fprintf(stderr, "fill_replay: open bank failed: %s\n",
                     store.last_error().c_str());
        return 2;
    }
    if (store.record_bytes() == 0) {
        std::fprintf(stderr, "fill_replay: bad record geometry\n");
        return 2;
    }
    if (args.mode == "rider") return run_rider(args, &store);
    std::vector<std::pair<int, int>> calls;
    std::vector<int> tokens;
    if (args.pattern == "journal") {
        if (args.journal.empty() || !load_journal(args.journal, &calls, &tokens)) return 2;
    } else {
        std::fprintf(stderr, "fill_replay: replay needs --journal\n");
        return 2;
    }
    if (args.max_records != 0 && calls.size() > args.max_records) {
        calls.resize(args.max_records);
        tokens.resize(args.max_records);
    }
    dee::StageProfiler profiler;
    profiler.configure(true, false, store.record_bytes(), 256, false);
    dee::HostPackCache cache;
    cache.set_budget(args.budget_bytes != 0
                         ? args.budget_bytes
                         : (size_t(8500) << 20));  // production-like 8.5 GiB
    int code = 0;
    for (int pass = 0; pass < args.passes && code == 0; ++pass) {
        cache.clear();
        code = run_replay(args, &store, &cache, &profiler, calls, tokens);
        std::fprintf(stderr, "fill_replay: pass %d done code=%d\n", pass, code);
    }
    if (code != 0) return code;
    std::ofstream out(args.out, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "fill_replay: cannot write %s\n", args.out.c_str());
        return 2;
    }
    out << profiler.fill_timeline_json();
    const dee::ExpertStoreStats stats = store.stats();
    std::cout << "{\"replay\":{\"calls\":" << calls.size()
              << ",\"record_bytes\":" << store.record_bytes()
              << ",\"lanes\":" << args.lanes << ",\"qdepth\":" << args.qdepth
              << ",\"pread_service_ms\":" << stats.pread_service_ms
              << ",\"pread_calls\":" << stats.pread_calls
              << ",\"pread_short_reads\":" << stats.pread_short_reads
              << ",\"pread_bytes\":" << stats.pread_bytes
              << ",\"mincore_probed_bytes\":" << stats.mincore_probed_bytes
              << ",\"mincore_resident_bytes\":" << stats.mincore_resident_bytes
              << "}}" << std::endl;
    return 0;
}
