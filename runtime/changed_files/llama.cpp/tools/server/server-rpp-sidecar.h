#pragma once

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

struct server_rpp_layer_prediction {
    int32_t layer = -1;
    std::vector<int32_t> experts;
    std::vector<float> confidences;
    float confidence = 0.0f;
};

struct server_rpp_prediction {
    std::vector<server_rpp_layer_prediction> layers;
    double inference_ms = 0.0;
    bool cropped = false;
    int32_t model_token_count = 0;
};

struct server_rpp_batch_request {
    int32_t id = -1;
    llama_tokens token_ids;
};

struct server_rpp_batch_prediction {
    int32_t id = -1;
    server_rpp_prediction prediction;
};

class server_rpp_sidecar {
public:
    server_rpp_sidecar(std::string url, int32_t timeout_ms);

    bool health(std::string * error) const;

    bool predict(
            const llama_tokens & token_ids,
            server_rpp_prediction * prediction,
            std::string * error) const;

    bool predict_batch(
            const std::vector<server_rpp_batch_request> & requests,
            std::vector<server_rpp_batch_prediction> * predictions,
            double * batch_inference_ms,
            std::string * error) const;

private:
    std::string url_;
    int32_t timeout_ms_ = 30000;
};
