#include "arg.h"
#include "common.h"

#undef NDEBUG
#include <cassert>
#include <string>
#include <vector>

namespace {

std::vector<char *> as_argv(std::vector<std::string> & args) {
    std::vector<char *> result;
    for (auto & arg : args) {
        result.push_back(arg.data());
    }
    return result;
}

bool parse(std::vector<std::string> args, common_params & params) {
    auto argv = as_argv(args);
    return common_params_parse(
            static_cast<int>(argv.size()), argv.data(), params, LLAMA_EXAMPLE_SERVER);
}

} // namespace

int main() {
    {
        common_params params;
        assert(parse({
            "llama-server",
            "--rpp-mode", "replay",
            "--rpp-predictions", "prediction_trace.jsonl",
            "--rpp-prefetch-depth", "2",
            "--rpp-prefetch-top-k", "4",
            "--rpp-prefetch-admission", "feo",
            "--rpp-feo-min-count", "2",
            "--rpp-feo-max-experts", "12",
            "--rpp-feo-density-threshold", "0.25",
            "--rpp-page-map", "expert_page_map.csv",
            "--rpp-host-prefetch", "pretouch",
            "--rpp-prefetch-threads", "3",
            "--rpp-gpu-transfer", "off",
            "--rpp-gpu-correction", "off",
            "--rpp-gpu-cache-mib", "128",
            "--rpp-gpu-staging-mib", "32",
            "--rpp-gpu-copy-workers", "2",
            "--rpp-gpu-queue-policy", "deadline",
            "--rpp-gpu-reclaim-policy", "feo",
            "--rpp-prefill",
            "--no-rpp-decode",
            "--rpp-trace", "rpp_runtime_trace.jsonl",
        }, params));

        assert(params.rpp.mode == "replay");
        assert(params.rpp.predictions == "prediction_trace.jsonl");
        assert(params.rpp.prefetch_depth == 2);
        assert(params.rpp.prefetch_top_k == 4);
        assert(params.rpp.prefetch_admission == "feo");
        assert(params.rpp.feo_min_count == 2);
        assert(params.rpp.feo_max_experts == 12);
        assert(params.rpp.feo_density_threshold == 0.25f);
        assert(params.rpp.page_map == "expert_page_map.csv");
        assert(params.rpp.host_prefetch == "pretouch");
        assert(params.rpp.prefetch_threads == 3);
        assert(params.rpp.gpu_transfer == "off");
        assert(params.rpp.gpu_correction == "off");
        assert(params.rpp.gpu_cache_mib == 128);
        assert(params.rpp.gpu_staging_mib == 32);
        assert(params.rpp.gpu_copy_workers == 2);
        assert(params.rpp.gpu_queue_policy == "deadline");
        assert(params.rpp.gpu_reclaim_policy == "feo");
        assert(params.rpp.enable_prefill);
        assert(!params.rpp.enable_decode);
        assert(params.rpp.trace == "rpp_runtime_trace.jsonl");
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-mode", "replay"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-prefetch-depth", "-1"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-prefetch-top-k", "0"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-prefetch-admission", "invalid"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-feo-min-count", "0"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-feo-max-experts", "-1"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-feo-density-threshold", "1.5"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-gpu-reclaim-policy", "invalid"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-mode", "invalid"}, params));
    }

    {
        common_params params;
        assert(parse({
            "llama-server",
            "--rpp-mode", "online",
            "--rpp-sidecar-url", "http://127.0.0.1:19001",
            "--rpp-sidecar-timeout-ms", "5000",
            "--no-rpp-prefill",
            "--rpp-decode",
        }, params));
        assert(params.rpp.mode == "online");
        assert(params.rpp.sidecar_url == "http://127.0.0.1:19001");
        assert(params.rpp.sidecar_timeout_ms == 5000);
        assert(!params.rpp.enable_prefill);
        assert(params.rpp.enable_decode);
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-sidecar-timeout-ms", "0"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-host-prefetch", "pretouch"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-host-prefetch", "invalid"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-prefetch-threads", "0"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-gpu-transfer", "on"}, params));
    }

    {
        common_params params;
        assert(parse({
            "llama-server",
            "--rpp-mode", "replay",
            "--rpp-predictions", "prediction_trace.jsonl",
            "--rpp-page-map", "expert_page_map.csv",
            "--rpp-gpu-transfer", "on",
        }, params));
        assert(params.rpp.gpu_transfer == "on");
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-gpu-transfer", "invalid"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-gpu-correction", "on"}, params));
    }

    {
        common_params params;
        assert(parse({
            "llama-server",
            "--rpp-mode", "replay",
            "--rpp-predictions", "prediction_trace.jsonl",
            "--rpp-page-map", "expert_page_map.csv",
            "--rpp-gpu-correction", "on",
        }, params));
        assert(params.rpp.gpu_correction == "on");
    }

    {
        common_params params;
        assert(!parse({
            "llama-server",
            "--rpp-mode", "replay",
            "--rpp-predictions", "prediction_trace.jsonl",
            "--rpp-page-map", "expert_page_map.csv",
            "--rpp-gpu-transfer", "on",
            "--rpp-gpu-correction", "on",
        }, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-gpu-correction", "invalid"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-gpu-cache-mib", "0"}, params));
    }

    {
        common_params params;
        assert(!parse({"llama-server", "--rpp-gpu-staging-mib", "0"}, params));
    }

    return 0;
}
