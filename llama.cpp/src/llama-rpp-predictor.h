#pragma once

#include "llama-rpp-types.h"

#include <cstddef>
#include <map>
#include <string>

struct llama_rpp_replay_stats {
    size_t lines = 0;
    size_t routes = 0;
    size_t layer_predictions = 0;
};

class llama_rpp_replay_predictor {
public:
    bool load_jsonl(
            const std::string & path,
            std::string * error = nullptr,
            size_t max_lines = 0);

    const llama_rpp_route * find(const llama_rpp_trace_key & key) const;
    void upsert(const llama_rpp_trace_key & key, llama_rpp_route route);

    const llama_rpp_replay_stats & stats() const;

    void clear();

private:
    std::map<llama_rpp_trace_key, llama_rpp_route> routes_;
    llama_rpp_replay_stats stats_;
};
