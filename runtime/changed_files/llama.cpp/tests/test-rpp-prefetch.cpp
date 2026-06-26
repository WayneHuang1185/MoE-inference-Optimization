#include "llama-rpp-prefetch.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void write_model(const std::string & path) {
    std::vector<unsigned char> data(4 * 4096);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<unsigned char>((i / 4096) + 1);
    }
    std::ofstream output(path, std::ios::binary);
    assert(output.is_open());
    output.write(reinterpret_cast<const char *>(data.data()), data.size());
}

void write_map(const std::string & path) {
    std::ofstream output(path);
    assert(output.is_open());
    output
        << "layer,expert,component,kind,tensor_name,byte_start,byte_end,n_bytes,"
           "page_start,page_end,page_count\n"
        << "0,1,ffn_down_exps,weight,blk.0.ffn_down_exps.weight,"
           "4096,8192,4096,1,2,1\n"
        << "0,1,ffn_gate_up_exps,weight,blk.0.ffn_gate_up_exps.weight,"
           "12288,16384,4096,3,4,1\n"
        << "1,2,ffn_down_exps,weight,blk.1.ffn_down_exps.weight,"
           "8192,12288,4096,2,3,1\n";
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 3) {
        llama_rpp_host_prefetcher prefetcher;
        std::string error;
        if (!prefetcher.configure(argv[1], argv[2], 1, &error)) {
            std::cerr << error << '\n';
            return 1;
        }
        const llama_rpp_prefetch_key key = {
            0,
            0,
            LLAMA_RPP_PHASE_DECODE,
            0,
        };
        if (!prefetcher.enqueue(key, {24, 54, 92, 102, 1, 69, 99, 114}) ||
                !prefetcher.wait_idle(30000)) {
            std::cerr << "real page-map pretouch did not complete\n";
            return 1;
        }
        const auto snapshot = prefetcher.snapshot(key);
        std::cout
            << "state=" << llama_rpp_prefetch_state_name(snapshot.state)
            << " pages=" << snapshot.pages
            << " bytes=" << snapshot.bytes
            << " duration_us=" << snapshot.duration_us
            << " major_faults=" << snapshot.major_faults
            << " minor_faults=" << snapshot.minor_faults
            << '\n';
        return snapshot.state == llama_rpp_prefetch_state::done ? 0 : 1;
    }

    const std::string model_path = "test-rpp-prefetch-model.bin";
    const std::string map_path = "test-rpp-prefetch-map.csv";
    write_model(model_path);
    write_map(map_path);

    {
        llama_rpp_host_prefetcher prefetcher;
        std::string error;
        assert(prefetcher.configure(model_path, map_path, 2, &error));
        assert(error.empty());
        assert(prefetcher.enabled());

        const llama_rpp_prefetch_key key = {
            7,
            11,
            LLAMA_RPP_PHASE_DECODE,
            0,
        };
        assert(prefetcher.enqueue(key, {1}));
        assert(!prefetcher.enqueue(key, {1}));
        assert(prefetcher.wait_idle(2000));

        const auto snapshot = prefetcher.snapshot(key);
        assert(snapshot.state == llama_rpp_prefetch_state::done);
        assert(snapshot.pages == 2);
        assert(snapshot.bytes == 8192);
        assert(snapshot.enqueue_us > 0);
        assert(snapshot.start_us >= snapshot.enqueue_us);
        assert(snapshot.complete_us >= snapshot.start_us);
        assert(snapshot.duration_us >= 0);
        assert(snapshot.minor_faults >= 0);
        assert(snapshot.major_faults >= 0);

        std::vector<unsigned char> packed(8192);
        llama_rpp_expert_pack_result pack;
        assert(prefetcher.pack_experts(
                0, {1}, packed.data(), packed.size(), &pack, &error));
        assert(pack.bytes == 8192);
        assert(pack.ranges == 2);
        assert(pack.duration_us >= 0);
        assert(packed.front() == 2);
        assert(packed[4096] == 4);

        const llama_rpp_prefetch_key missing = {
            7,
            11,
            LLAMA_RPP_PHASE_DECODE,
            9,
        };
        assert(!prefetcher.enqueue(missing, {99}));
        assert(prefetcher.snapshot(missing).state == llama_rpp_prefetch_state::missing);
    }

    std::remove(model_path.c_str());
    std::remove(map_path.c_str());
    return 0;
}
