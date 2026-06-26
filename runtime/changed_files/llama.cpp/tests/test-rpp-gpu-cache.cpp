#include "llama-rpp-gpu-cache.h"

#include "ggml-backend.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr size_t EXPERT_BYTES = 4096;
constexpr int N_EXPERTS = 16;

void write_model(const std::string & path) {
    std::vector<unsigned char> data(N_EXPERTS * EXPERT_BYTES);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<unsigned char>((i / EXPERT_BYTES) + 1);
    }
    std::ofstream output(path, std::ios::binary);
    assert(output.is_open());
    output.write(reinterpret_cast<const char *>(data.data()), data.size());
}

void write_map(const std::string & path) {
    std::ofstream output(path);
    assert(output.is_open());
    output << "layer,expert,byte_start,byte_end\n";
    for (int expert = 0; expert < N_EXPERTS; ++expert) {
        output << "0," << expert << ","
               << expert * EXPERT_BYTES << ","
               << (expert + 1) * EXPERT_BYTES << "\n";
    }
}

} // namespace

int main() {
    const std::string model_path = "test-rpp-gpu-cache-model.bin";
    const std::string map_path = "test-rpp-gpu-cache-map.csv";
    write_model(model_path);
    write_map(map_path);

    llama_rpp_host_prefetcher source;
    std::string error;
    assert(source.configure(model_path, map_path, 1, &error));
    assert(source.max_expert_bytes() == EXPERT_BYTES);
    assert(source.expert_bytes(0, 3) == EXPERT_BYTES);

    llama_rpp_gpu_cache cache;
    const bool configured = cache.configure(
            &source,
            nullptr,
            8 * EXPERT_BYTES,
            EXPERT_BYTES,
            1,
            llama_rpp_gpu_queue_policy::fifo,
            &error);

    if (configured) {
        assert(configured);

        cache.prefetch({1, 10, LLAMA_RPP_PHASE_DECODE, 0}, 0, {1, 2});
        assert(cache.wait_idle(5000));

        const llama_rpp_prefetch_key event0 = {
            1, 10, LLAMA_RPP_PHASE_DECODE, 0,
        };
        const auto first = cache.ensure_resident(event0, 0, {1, 2, 3, 4});
        assert(first.success);
        assert(first.requested == 4);
        assert(first.ready_hits == 2);
        assert(first.loaded_on_demand == 2);
        assert(first.failed == 0);
        assert(first.correction_bytes == 2 * EXPERT_BYTES);
        assert(first.locations.size() == 4);
        assert(first.selected_for_prefetch == 2);
        assert(first.prefetch_ready_at_use == 2);
        assert(first.expert_timings.size() == 4);

        const llama_rpp_prefetch_key event1 = {
            1, 11, LLAMA_RPP_PHASE_DECODE, 0,
        };
        const auto second = cache.ensure_resident(event1, 0, {1, 2, 3, 4});
        assert(second.success);
        assert(second.ready_hits == 4);
        assert(second.loaded_on_demand == 0);
        assert(second.correction_bytes == 0);

        const llama_rpp_prefetch_key event2 = {
            1, 12, LLAMA_RPP_PHASE_DECODE, 0,
        };
        const auto third = cache.ensure_resident(
                event2, 0, {8, 9, 10, 11, 12, 13, 14, 15});
        assert(third.success);
        assert(third.loaded_on_demand == 8);
        assert(third.evictions >= 4);
        assert(third.resident_entries == 8);

        const auto saved = cache.snapshot(event0);
        assert(saved.success);
        assert(saved.ready_hits == 2);
    } else {
        assert(!error.empty());
    }

    std::remove(model_path.c_str());
    std::remove(map_path.c_str());
    return 0;
}
