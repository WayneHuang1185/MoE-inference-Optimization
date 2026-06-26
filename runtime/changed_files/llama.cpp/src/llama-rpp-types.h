#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

inline const char * llama_rpp_phase_name(enum llama_rpp_phase phase) {
    switch (phase) {
        case LLAMA_RPP_PHASE_DATASET: return "dataset";
        case LLAMA_RPP_PHASE_PREFILL: return "prefill";
        case LLAMA_RPP_PHASE_DECODE:  return "decode";
        case LLAMA_RPP_PHASE_UNKNOWN: return "unknown";
    }
    return "unknown";
}

inline enum llama_rpp_phase llama_rpp_phase_from_string(const std::string & value) {
    if (value == "dataset") {
        return LLAMA_RPP_PHASE_DATASET;
    }
    if (value == "prefill") {
        return LLAMA_RPP_PHASE_PREFILL;
    }
    if (value == "decode") {
        return LLAMA_RPP_PHASE_DECODE;
    }
    return LLAMA_RPP_PHASE_UNKNOWN;
}

// Runtime identity for one token submitted to llama_decode().
struct llama_rpp_token_key {
    llama_seq_id   seq_id = -1;
    llama_pos      pos = -1;
    llama_token    token = LLAMA_TOKEN_NULL;
    enum llama_rpp_phase phase = LLAMA_RPP_PHASE_UNKNOWN;
};

// Offline traces use a request/sample id and token position. The existing
// prediction_trace.jsonl calls token_position "token_id", so keep this
// separate from llama_rpp_token_key::token to avoid confusing it with a
// vocabulary token id.
struct llama_rpp_trace_key {
    int64_t         request_id = -1;
    int64_t         token_position = -1;
    enum llama_rpp_phase phase = LLAMA_RPP_PHASE_UNKNOWN;

    bool operator<(const llama_rpp_trace_key & other) const {
        if (request_id != other.request_id) {
            return request_id < other.request_id;
        }
        if (token_position != other.token_position) {
            return token_position < other.token_position;
        }
        return static_cast<int>(phase) < static_cast<int>(other.phase);
    }
};

struct llama_rpp_layer_prediction {
    int32_t layer = -1;

    std::vector<int32_t> experts;
    std::vector<float>   expert_confidences;

    float confidence = 0.0f;
};

struct llama_rpp_route {
    std::vector<llama_rpp_layer_prediction> layers;

    const llama_rpp_layer_prediction * find_layer(int32_t layer) const {
        for (const auto & item : layers) {
            if (item.layer == layer) {
                return &item;
            }
        }
        return nullptr;
    }
};
