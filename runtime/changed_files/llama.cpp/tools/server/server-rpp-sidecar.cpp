#include "server-rpp-sidecar.h"

#include "http.h"

#include <nlohmann/json.hpp>

#include <chrono>

namespace {

using json = nlohmann::json;

bool check_response(
        const httplib::Result & response,
        json * body,
        std::string * error) {
    if (!response) {
        if (error) {
            *error = "RPP sidecar request failed: " + httplib::to_string(response.error());
        }
        return false;
    }
    try {
        *body = json::parse(response->body);
    } catch (const std::exception & e) {
        if (error) {
            *error = std::string("invalid RPP sidecar JSON: ") + e.what();
        }
        return false;
    }
    if (response->status < 200 || response->status >= 300) {
        if (error) {
            *error = body->value("error", "RPP sidecar returned HTTP " +
                    std::to_string(response->status));
        }
        return false;
    }
    return true;
}

} // namespace

server_rpp_sidecar::server_rpp_sidecar(std::string url, int32_t timeout_ms) :
    url_(std::move(url)),
    timeout_ms_(timeout_ms) {
}

bool server_rpp_sidecar::health(std::string * error) const {
    auto [client, parts] = common_http_client(url_);
    const auto timeout = std::chrono::milliseconds(timeout_ms_);
    client.set_connection_timeout(timeout);
    client.set_read_timeout(timeout);
    client.set_write_timeout(timeout);
    json body;
    return check_response(client.Get("/health"), &body, error);
}

bool server_rpp_sidecar::predict(
        const llama_tokens & token_ids,
        server_rpp_prediction * prediction,
        std::string * error) const {
    if (!prediction || token_ids.empty()) {
        if (error) {
            *error = "RPP sidecar requires a non-empty token sequence";
        }
        return false;
    }

    auto [client, parts] = common_http_client(url_);
    const auto timeout = std::chrono::milliseconds(timeout_ms_);
    client.set_connection_timeout(timeout);
    client.set_read_timeout(timeout);
    client.set_write_timeout(timeout);

    const json request = {{"token_ids", token_ids}};
    json body;
    if (!check_response(
                client.Post("/predict", request.dump(), "application/json"),
                &body,
                error)) {
        return false;
    }

    try {
        prediction->layers.clear();
        prediction->inference_ms = body.value("inference_ms", 0.0);
        prediction->cropped = body.value("cropped", false);
        prediction->model_token_count = body.value("model_token_count", 0);
        const auto & layers = body.at("layers");
        if (!layers.is_array() || layers.empty()) {
            throw std::runtime_error("layers must be a non-empty array");
        }
        for (const auto & item : layers) {
            server_rpp_layer_prediction layer;
            layer.layer = item.at("layer").get<int32_t>();
            layer.experts = item.at("predicted_experts").get<std::vector<int32_t>>();
            layer.confidences =
                item.value("expert_confidences", std::vector<float>{});
            layer.confidence = item.value("confidence", 0.0f);
            if (layer.experts.empty() ||
                    (!layer.confidences.empty() &&
                     layer.confidences.size() != layer.experts.size())) {
                throw std::runtime_error("invalid expert arrays");
            }
            prediction->layers.push_back(std::move(layer));
        }
    } catch (const std::exception & e) {
        if (error) {
            *error = std::string("invalid RPP sidecar response: ") + e.what();
        }
        prediction->layers.clear();
        return false;
    }
    return true;
}

bool server_rpp_sidecar::predict_batch(
        const std::vector<server_rpp_batch_request> & requests,
        std::vector<server_rpp_batch_prediction> * predictions,
        double * batch_inference_ms,
        std::string * error) const {
    if (!predictions || requests.empty()) {
        if (error) {
            *error = "RPP sidecar batch requires non-empty requests";
        }
        return false;
    }

    json items = json::array();
    for (const auto & request : requests) {
        if (request.token_ids.empty()) {
            if (error) {
                *error = "RPP sidecar batch item has empty token sequence";
            }
            return false;
        }
        items.push_back({
            {"id", request.id},
            {"token_ids", request.token_ids},
        });
    }

    auto [client, parts] = common_http_client(url_);
    const auto timeout = std::chrono::milliseconds(timeout_ms_);
    client.set_connection_timeout(timeout);
    client.set_read_timeout(timeout);
    client.set_write_timeout(timeout);

    const json request = {{"requests", items}};
    json body;
    if (!check_response(
                client.Post("/predict_batch", request.dump(), "application/json"),
                &body,
                error)) {
        return false;
    }

    try {
        predictions->clear();
        if (batch_inference_ms) {
            *batch_inference_ms = body.value("inference_ms", 0.0);
        }
        const auto & results = body.at("results");
        if (!results.is_array() || results.size() != requests.size()) {
            throw std::runtime_error("results size does not match request size");
        }
        for (const auto & result : results) {
            server_rpp_batch_prediction output;
            output.id = result.at("id").get<int32_t>();
            output.prediction.inference_ms = body.value("inference_ms", 0.0);
            output.prediction.cropped = result.value("cropped", false);
            output.prediction.model_token_count =
                result.value("model_token_count", 0);

            const auto & layers = result.at("layers");
            if (!layers.is_array() || layers.empty()) {
                throw std::runtime_error("layers must be a non-empty array");
            }
            for (const auto & item : layers) {
                server_rpp_layer_prediction layer;
                layer.layer = item.at("layer").get<int32_t>();
                layer.experts = item.at("predicted_experts").get<std::vector<int32_t>>();
                layer.confidences =
                    item.value("expert_confidences", std::vector<float>{});
                layer.confidence = item.value("confidence", 0.0f);
                if (layer.experts.empty() ||
                        (!layer.confidences.empty() &&
                         layer.confidences.size() != layer.experts.size())) {
                    throw std::runtime_error("invalid expert arrays");
                }
                output.prediction.layers.push_back(std::move(layer));
            }
            predictions->push_back(std::move(output));
        }
    } catch (const std::exception & e) {
        if (error) {
            *error = std::string("invalid RPP sidecar batch response: ") + e.what();
        }
        predictions->clear();
        return false;
    }
    return true;
}
