#include "llama-rpp-gpu-transfer.h"

#include "ggml-backend.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <fstream>
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
        << "layer,expert,byte_start,byte_end\n"
        << "0,1,4096,8192\n"
        << "0,2,8192,12288\n";
}

} // namespace

int main() {
    const std::string model_path = "test-rpp-gpu-transfer-model.bin";
    const std::string map_path = "test-rpp-gpu-transfer-map.csv";
    write_model(model_path);
    write_map(map_path);

    llama_rpp_host_prefetcher source;
    std::string error;
    assert(source.configure(model_path, map_path, 1, &error));

    llama_rpp_gpu_transfer transfer;
    const bool configured = transfer.configure(
            &source,
            4 * 1024 * 1024,
            1 * 1024 * 1024,
            "",
            &error);

    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
        assert(!configured);
        assert(!error.empty());
    } else if (configured) {
        const llama_rpp_prefetch_key key = {
            1,
            2,
            LLAMA_RPP_PHASE_DECODE,
            0,
        };
        assert(transfer.enqueue(key, {1, 2}));
        assert(transfer.wait_idle(5000));
        const auto snapshot = transfer.snapshot(key);
        assert(snapshot.state == llama_rpp_gpu_transfer_state::done);
        assert(snapshot.bytes == 8192);
        assert(snapshot.cache_capacity == 4 * 1024 * 1024);
        assert(snapshot.pack_us >= 0);
        assert(snapshot.transfer_us >= 0);
    }

    std::remove(model_path.c_str());
    std::remove(map_path.c_str());
    return 0;
}
