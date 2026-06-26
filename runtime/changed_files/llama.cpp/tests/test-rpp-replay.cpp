#include "llama-rpp-predictor.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

std::string make_temp_path(const char * suffix) {
    return std::string("test-rpp-replay-") + suffix + ".jsonl";
}

void write_file(const std::string & path, const std::string & content) {
    std::ofstream output(path);
    assert(output.is_open());
    output << content;
}

void test_load_and_lookup() {
    const std::string path = make_temp_path("valid");
    write_file(path,
        R"({"prompt_id":7,"token_id":12,"phase":"decode","layer":1,"predicted_experts":[9,3],"confidence":0.75,"expert_confidences":[0.8,0.7]})" "\n"
        R"({"prompt_id":7,"token_id":12,"phase":"decode","layer":0,"predicted_experts":[4,2],"expert_confidences":[0.9,0.5]})" "\n"
        R"({"seq_id":8,"position":2,"phase":"prefill","layer":0,"predicted_experts":[1,5]})" "\n"
        R"({"request_id":-1,"position":3,"phase":"prefill","layer":0,"predicted_experts":[6,7]})" "\n");

    llama_rpp_replay_predictor predictor;
    std::string error;
    assert(predictor.load_jsonl(path, &error));
    assert(error.empty());
    assert(predictor.stats().lines == 4);
    assert(predictor.stats().routes == 3);
    assert(predictor.stats().layer_predictions == 4);

    const llama_rpp_trace_key decode_key = {7, 12, LLAMA_RPP_PHASE_DECODE};
    const llama_rpp_route * decode = predictor.find(decode_key);
    assert(decode != nullptr);
    assert(decode->layers.size() == 2);
    assert(decode->layers[0].layer == 0);
    assert(decode->layers[1].layer == 1);
    assert(decode->layers[0].experts[0] == 4);
    assert(decode->layers[0].confidence > 0.69f);
    assert(decode->layers[0].confidence < 0.71f);

    const llama_rpp_trace_key prefill_key = {8, 2, LLAMA_RPP_PHASE_PREFILL};
    assert(predictor.find(prefill_key) != nullptr);

    const llama_rpp_trace_key missing_key = {7, 13, LLAMA_RPP_PHASE_DECODE};
    assert(predictor.find(missing_key) == nullptr);

    const llama_rpp_trace_key wildcard_key = {12345, 3, LLAMA_RPP_PHASE_PREFILL};
    const llama_rpp_route * wildcard = predictor.find(wildcard_key);
    assert(wildcard != nullptr);
    assert(wildcard->layers[0].experts[0] == 6);

    std::remove(path.c_str());
}

void test_duplicate_layer_is_rejected() {
    const std::string path = make_temp_path("duplicate");
    write_file(path,
        R"({"prompt_id":1,"token_id":2,"phase":"dataset","layer":0,"predicted_experts":[1,2]})" "\n"
        R"({"prompt_id":1,"token_id":2,"phase":"dataset","layer":0,"predicted_experts":[3,4]})" "\n");

    llama_rpp_replay_predictor predictor;
    std::string error;
    assert(!predictor.load_jsonl(path, &error));
    assert(error.find("duplicate prediction") != std::string::npos);
    assert(predictor.stats().routes == 0);

    std::remove(path.c_str());
}

void test_invalid_confidence_count_is_rejected() {
    const std::string path = make_temp_path("confidence");
    write_file(path,
        R"({"prompt_id":1,"token_id":2,"phase":"dataset","layer":0,"predicted_experts":[1,2],"expert_confidences":[0.5]})" "\n");

    llama_rpp_replay_predictor predictor;
    std::string error;
    assert(!predictor.load_jsonl(path, &error));
    assert(error.find("size must match") != std::string::npos);

    std::remove(path.c_str());
}

void test_online_upsert() {
    llama_rpp_replay_predictor predictor;
    llama_rpp_route route;
    route.layers = {
        {1, {8, 9}, {0.8f, 0.7f}, 0.75f},
        {0, {3, 4}, {0.9f, 0.6f}, 0.75f},
    };
    const llama_rpp_trace_key key = {42, 7, LLAMA_RPP_PHASE_DECODE};
    predictor.upsert(key, std::move(route));

    const auto * found = predictor.find(key);
    assert(found != nullptr);
    assert(found->layers.size() == 2);
    assert(found->layers[0].layer == 0);
    assert(found->layers[1].layer == 1);
    assert(predictor.stats().routes == 1);
    assert(predictor.stats().layer_predictions == 2);
}

} // namespace

int main(int argc, char ** argv) {
    test_load_and_lookup();
    test_duplicate_layer_is_rejected();
    test_invalid_confidence_count_is_rejected();
    test_online_upsert();

    if (argc == 2 || argc == 3) {
        const size_t max_lines = argc == 3
                ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10))
                : 0;
        llama_rpp_replay_predictor predictor;
        std::string error;
        assert(predictor.load_jsonl(argv[1], &error, max_lines));
        assert(error.empty());
        std::printf(
                "loaded lines=%zu routes=%zu layer_predictions=%zu\n",
                predictor.stats().lines,
                predictor.stats().routes,
                predictor.stats().layer_predictions);
    }

    return 0;
}
