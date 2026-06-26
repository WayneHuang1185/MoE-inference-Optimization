#include "llama-rpp-runtime.h"
#include "llama-batch.h"

#include <nlohmann/json.hpp>

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

void write_file(const std::string & path, const std::string & content) {
    std::ofstream output(path);
    assert(output.is_open());
    output << content;
}

void write_binary_file(const std::string & path, size_t size) {
    std::vector<unsigned char> data(size, 1);
    std::ofstream output(path, std::ios::binary);
    assert(output.is_open());
    output.write(reinterpret_cast<const char *>(data.data()), data.size());
}

} // namespace

int main() {
    const std::string predictions_path = "test-rpp-runtime-predictions.jsonl";
    const std::string trace_path = "test-rpp-runtime-trace.jsonl";
    const std::string model_path = "test-rpp-runtime-model.bin";
    const std::string page_map_path = "test-rpp-runtime-page-map.csv";

    write_file(predictions_path,
        R"({"seq_id":3,"position":10,"phase":"prefill","layer":0,"predicted_experts":[1,2,7],"confidence":0.8})" "\n"
        R"({"seq_id":4,"position":20,"phase":"decode","layer":0,"predicted_experts":[3,4,8],"confidence":0.7})" "\n");
    write_binary_file(model_path, 10 * 4096);
    write_file(page_map_path,
        "layer,expert,byte_start,byte_end\n"
        "0,1,4096,8192\n"
        "0,2,8192,12288\n"
        "0,3,12288,16384\n"
        "0,4,16384,20480\n"
        "0,7,28672,32768\n"
        "0,8,32768,36864\n");

    {
        llama_rpp_runtime runtime(nullptr, nullptr, nullptr);
        const llama_rpp_config config = {
            "replay",
            predictions_path.c_str(),
            trace_path.c_str(),
            model_path.c_str(),
            page_map_path.c_str(),
            "pretouch",
            "off",
            "off",
            "off",
            2,
            2,
            "topk",
            1,
            0,
            0.0f,
            1,
            256,
            64,
            1,
            "fifo",
            "lru",
            true,
            true,
        };
        std::string error;
        assert(runtime.configure(config, &error));
        assert(error.empty());

        const llama_rpp_token_metadata metadata[] = {
            {3, 30, 10, 101, LLAMA_RPP_PHASE_PREFILL},
            {4, 40, 20, 202, LLAMA_RPP_PHASE_DECODE},
        };
        assert(runtime.set_batch_metadata(metadata, 2, &error));

        llama_token tokens[] = {101, 202};
        llama_pos positions[] = {10, 20};
        int32_t n_seq_id[] = {1, 1};
        llama_seq_id seq0[] = {30};
        llama_seq_id seq1[] = {40};
        llama_seq_id * seq_ids[] = {seq0, seq1};
        int8_t outputs[] = {0, 1};

        llama_ubatch ubatch = {};
        ubatch.n_tokens = 2;
        ubatch.token = tokens;
        ubatch.pos = positions;
        ubatch.n_seq_id = n_seq_id;
        ubatch.seq_id = seq_ids;
        ubatch.output = outputs;

        runtime.begin_ubatch(ubatch);

        int32_t topk[] = {
            1, 2, 9,
            3, 5, 8,
        };
        ggml_tensor tensor = {};
        tensor.type = GGML_TYPE_I32;
        tensor.ne[0] = 3;
        tensor.ne[1] = 2;
        tensor.ne[2] = 1;
        tensor.ne[3] = 1;
        tensor.nb[0] = sizeof(int32_t);
        tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
        tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
        tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
        tensor.data = topk;
        std::strncpy(tensor.name, "ffn_moe_topk-0", sizeof(tensor.name) - 1);

        assert(llama_rpp_runtime::eval_callback(&tensor, true, &runtime));
        assert(llama_rpp_runtime::eval_callback(&tensor, false, &runtime));
        runtime.end_ubatch();
    }

    std::ifstream trace(trace_path);
    assert(trace.is_open());
    std::string line0;
    std::string line1;
    assert(std::getline(trace, line0));
    assert(std::getline(trace, line1));

    const json event0 = json::parse(line0);
    const json event1 = json::parse(line1);

    assert(event0["request_id"] == 3);
    assert(event0["seq_id"] == 30);
    assert(event0["position"] == 10);
    assert(event0["phase"] == "prefill");
    assert(event0["true_experts"] == json::array({1, 2, 9}));
    assert(event0["hit_experts"] == json::array({1, 2}));
    assert(event0["missing_experts"] == json::array({9}));
    assert(event0["wasted_experts"] == json::array({7}));
    assert(event0["host_prefetch_state"] != "missing");
    assert(event0["prefetch_top_k"] == 2);
    assert(event0["prefetched_experts"] == json::array({1, 2}));
    assert(event0["host_prefetch_pages"] == 2);
    assert(event0["host_prefetch_bytes"] == 2 * 4096);
    assert(event0["gpu_transfer_state"] == "missing");

    assert(event1["request_id"] == 4);
    assert(event1["seq_id"] == 40);
    assert(event1["position"] == 20);
    assert(event1["phase"] == "decode");
    assert(event1["true_experts"] == json::array({3, 5, 8}));
    assert(event1["hit_experts"] == json::array({3, 8}));
    assert(event1["missing_experts"] == json::array({5}));
    assert(event1["wasted_experts"] == json::array({4}));
    assert(event1["host_prefetch_state"] != "missing");
    assert(event1["prefetched_experts"] == json::array({3, 4}));
    assert(event1["host_prefetch_pages"] == 2);
    assert(event1["host_prefetch_bytes"] == 2 * 4096);
    assert(event1["gpu_transfer_state"] == "missing");

    std::remove(predictions_path.c_str());
    std::remove(trace_path.c_str());
    std::remove(model_path.c_str());
    std::remove(page_map_path.c_str());
    return 0;
}
