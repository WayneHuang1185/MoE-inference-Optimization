#include "llama-rpp-predictor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

using json = nlohmann::json;

int64_t read_integer_alias(
        const json & obj,
        std::initializer_list<const char *> names,
        const char * description) {
    for (const char * name : names) {
        auto it = obj.find(name);
        if (it != obj.end()) {
            if (!it->is_number_integer()) {
                throw std::runtime_error(std::string(name) + " must be an integer");
            }
            return it->get<int64_t>();
        }
    }
    throw std::runtime_error(std::string("missing ") + description);
}

std::string format_line_error(size_t line_number, const std::string & message) {
    std::ostringstream out;
    out << "prediction trace line " << line_number << ": " << message;
    return out.str();
}

} // namespace

bool llama_rpp_replay_predictor::load_jsonl(
        const std::string & path,
        std::string * error,
        size_t max_lines) {
    clear();

    std::ifstream input(path);
    if (!input.is_open()) {
        if (error) {
            *error = "failed to open prediction trace: " + path;
        }
        return false;
    }

    try {
        std::string line;
        size_t line_number = 0;

        while (std::getline(input, line)) {
            ++line_number;
            if (max_lines != 0 && stats_.lines >= max_lines) {
                break;
            }
            if (line.empty()) {
                continue;
            }

            try {
                const json obj = json::parse(line);

                llama_rpp_trace_key key;
                key.request_id = read_integer_alias(
                        obj, {"request_id", "seq_id", "prompt_id"}, "request/sequence/prompt id");
                key.token_position = read_integer_alias(
                        obj, {"token_position", "position", "token_id"}, "token position");

                const std::string phase_text = obj.value("phase", "unknown");
                key.phase = llama_rpp_phase_from_string(phase_text);
                if (key.phase == LLAMA_RPP_PHASE_UNKNOWN && phase_text != "unknown" && !phase_text.empty()) {
                    throw std::runtime_error("unsupported phase: " + phase_text);
                }

                const int64_t layer64 = read_integer_alias(obj, {"layer"}, "layer");
                if (layer64 < 0 || layer64 > INT32_MAX) {
                    throw std::runtime_error("layer is outside int32 range");
                }

                auto experts_it = obj.find("predicted_experts");
                if (experts_it == obj.end() || !experts_it->is_array() || experts_it->empty()) {
                    throw std::runtime_error("predicted_experts must be a non-empty array");
                }

                llama_rpp_layer_prediction prediction;
                prediction.layer = static_cast<int32_t>(layer64);
                for (const auto & value : *experts_it) {
                    if (!value.is_number_integer()) {
                        throw std::runtime_error("predicted_experts entries must be integers");
                    }
                    const int64_t expert = value.get<int64_t>();
                    if (expert < 0 || expert > INT32_MAX) {
                        throw std::runtime_error("expert id is outside int32 range");
                    }
                    prediction.experts.push_back(static_cast<int32_t>(expert));
                }

                auto confidences_it = obj.find("expert_confidences");
                if (confidences_it != obj.end()) {
                    if (!confidences_it->is_array()) {
                        throw std::runtime_error("expert_confidences must be an array");
                    }
                    for (const auto & value : *confidences_it) {
                        if (!value.is_number()) {
                            throw std::runtime_error("expert_confidences entries must be numbers");
                        }
                        prediction.expert_confidences.push_back(value.get<float>());
                    }
                    if (prediction.expert_confidences.size() != prediction.experts.size()) {
                        throw std::runtime_error(
                                "expert_confidences size must match predicted_experts size");
                    }
                }

                if (auto confidence_it = obj.find("confidence"); confidence_it != obj.end()) {
                    if (!confidence_it->is_number()) {
                        throw std::runtime_error("confidence must be a number");
                    }
                    prediction.confidence = confidence_it->get<float>();
                } else if (!prediction.expert_confidences.empty()) {
                    float total = 0.0f;
                    for (float value : prediction.expert_confidences) {
                        total += value;
                    }
                    prediction.confidence = total / prediction.expert_confidences.size();
                }

                auto & route = routes_[key];
                if (route.find_layer(prediction.layer) != nullptr) {
                    throw std::runtime_error(
                            "duplicate prediction for request/token/phase/layer");
                }
                route.layers.push_back(std::move(prediction));

                ++stats_.lines;
                ++stats_.layer_predictions;
            } catch (const std::exception & e) {
                throw std::runtime_error(format_line_error(line_number, e.what()));
            }
        }

        for (auto & item : routes_) {
            auto & layers = item.second.layers;
            std::sort(layers.begin(), layers.end(), [](const auto & lhs, const auto & rhs) {
                return lhs.layer < rhs.layer;
            });
        }
        stats_.routes = routes_.size();
    } catch (const std::exception & e) {
        clear();
        if (error) {
            *error = e.what();
        }
        return false;
    }

    return true;
}

const llama_rpp_route * llama_rpp_replay_predictor::find(const llama_rpp_trace_key & key) const {
    auto it = routes_.find(key);
    if (it != routes_.end()) {
        return &it->second;
    }

    // request_id=-1 is a single-request wildcard used by the prompt predictor
    // runner. Exact request IDs always take precedence. The wildcard should
    // only be used with one active request because position alone cannot
    // distinguish concurrent sequences.
    llama_rpp_trace_key wildcard = key;
    wildcard.request_id = -1;
    it = routes_.find(wildcard);
    return it == routes_.end() ? nullptr : &it->second;
}

void llama_rpp_replay_predictor::upsert(
        const llama_rpp_trace_key & key,
        llama_rpp_route route) {
    std::sort(route.layers.begin(), route.layers.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.layer < rhs.layer;
    });
    routes_[key] = std::move(route);
    stats_.routes = routes_.size();
    stats_.layer_predictions = 0;
    for (const auto & item : routes_) {
        stats_.layer_predictions += item.second.layers.size();
    }
}

const llama_rpp_replay_stats & llama_rpp_replay_predictor::stats() const {
    return stats_;
}

void llama_rpp_replay_predictor::clear() {
    routes_.clear();
    stats_ = {};
}
