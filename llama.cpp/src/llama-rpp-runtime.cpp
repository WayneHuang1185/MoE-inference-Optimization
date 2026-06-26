#include "llama-rpp-runtime.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

namespace {

using json = nlohmann::json;

bool starts_with(const char * value, const char * prefix) {
    return value != nullptr && prefix != nullptr &&
        std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

bool is_layer_tensor(const char * name, const char * base) {
    if (!name || !base) {
        return false;
    }
    const size_t base_size = std::strlen(base);
    if (std::strncmp(name, base, base_size) != 0 || name[base_size] != '-') {
        return false;
    }
    char * end = nullptr;
    std::strtol(name + base_size + 1, &end, 10);
    return end != name + base_size + 1 && *end == '\0';
}

int32_t parse_layer(const char * name) {
    if (!name) {
        return -1;
    }
    const char * separator = std::strrchr(name, '-');
    if (!separator || separator[1] == '\0') {
        return -1;
    }
    char * end = nullptr;
    const long layer = std::strtol(separator + 1, &end, 10);
    return end != separator + 1 && *end == '\0'
        ? static_cast<int32_t>(layer)
        : -1;
}

std::vector<int32_t> set_difference(
        const std::vector<int32_t> & lhs,
        const std::vector<int32_t> & rhs) {
    const std::set<int32_t> rhs_set(rhs.begin(), rhs.end());
    std::vector<int32_t> result;
    for (int32_t value : lhs) {
        if (rhs_set.count(value) == 0) {
            result.push_back(value);
        }
    }
    return result;
}

std::vector<int32_t> set_intersection(
        const std::vector<int32_t> & lhs,
        const std::vector<int32_t> & rhs) {
    const std::set<int32_t> rhs_set(rhs.begin(), rhs.end());
    std::vector<int32_t> result;
    for (int32_t value : lhs) {
        if (rhs_set.count(value) != 0) {
            result.push_back(value);
        }
    }
    return result;
}

} // namespace

llama_rpp_runtime::llama_rpp_runtime(
        const llama_model * model,
        ggml_backend_sched_eval_callback downstream_callback,
        void * downstream_user_data) :
    downstream_callback_(downstream_callback),
    downstream_user_data_(downstream_user_data),
    model_(model) {
}

bool llama_rpp_runtime::configure(const llama_rpp_config & config, std::string * error) {
    enabled_ = false;
    host_prefetch_enabled_ = false;
    gpu_compute_enabled_ = false;
    router_callback_seen_ = false;
    unsupported_router_tensor_logged_ = false;
    replay_.clear();
    gpu_cache_.stop();
    gpu_transfer_.stop();
    host_prefetcher_.stop();
    trace_.close();

    const std::string mode = config.mode ? config.mode : "off";
    if (mode == "off") {
        return true;
    }
    if (mode != "replay" && mode != "online") {
        if (error) {
            *error = "unsupported RPP mode: " + mode;
        }
        return false;
    }
    if (mode == "replay" &&
            (!config.predictions_path || config.predictions_path[0] == '\0')) {
        if (error) {
            *error = "replay mode requires a prediction trace";
        }
        return false;
    }
    if (config.prefetch_depth < 0) {
        if (error) {
            *error = "prefetch depth must be non-negative";
        }
        return false;
    }
    if (config.prefetch_top_k <= 0) {
        if (error) {
            *error = "prefetch top-k must be positive";
        }
        return false;
    }
    if (config.prefetch_threads <= 0) {
        if (error) {
            *error = "prefetch thread count must be positive";
        }
        return false;
    }
    if (mode == "replay" && !replay_.load_jsonl(config.predictions_path, error)) {
        return false;
    }

    const std::string host_prefetch = config.host_prefetch ? config.host_prefetch : "off";
    if (host_prefetch != "off" && host_prefetch != "pretouch") {
        if (error) {
            *error = "unsupported host prefetch mode: " + host_prefetch;
        }
        replay_.clear();
        return false;
    }
    const std::string gpu_transfer = config.gpu_transfer ? config.gpu_transfer : "off";
    if (gpu_transfer != "off" && gpu_transfer != "on") {
        if (error) {
            *error = "unsupported GPU transfer mode: " + gpu_transfer;
        }
        replay_.clear();
        return false;
    }
    const std::string gpu_correction =
        config.gpu_correction ? config.gpu_correction : "off";
    if (gpu_correction != "off" && gpu_correction != "on") {
        if (error) {
            *error = "unsupported GPU correction mode: " + gpu_correction;
        }
        replay_.clear();
        return false;
    }
    const std::string gpu_compute = config.gpu_compute ? config.gpu_compute : "off";
    if (gpu_compute != "off" && gpu_compute != "on") {
        if (error) {
            *error = "unsupported GPU compute mode: " + gpu_compute;
        }
        replay_.clear();
        return false;
    }
    if (gpu_compute == "on" && gpu_correction != "on") {
        if (error) {
            *error = "GPU compute mode requires GPU correction mode";
        }
        replay_.clear();
        return false;
    }
    if (gpu_transfer == "on" && gpu_correction == "on") {
        if (error) {
            *error = "GPU transfer-only and GPU correction modes are mutually exclusive";
        }
        replay_.clear();
        return false;
    }
    if (config.gpu_cache_mib <= 0 || config.gpu_staging_mib <= 0 ||
            config.gpu_copy_workers <= 0) {
        if (error) {
            *error = "GPU cache, staging size, and copy worker count must be positive";
        }
        replay_.clear();
        return false;
    }
    const std::string gpu_queue_policy =
        config.gpu_queue_policy ? config.gpu_queue_policy : "fifo";
    if (gpu_queue_policy != "fifo" && gpu_queue_policy != "deadline") {
        if (error) {
            *error = "unsupported GPU queue policy: " + gpu_queue_policy;
        }
        replay_.clear();
        return false;
    }
    if (host_prefetch == "pretouch" || gpu_transfer == "on" || gpu_correction == "on") {
        if (!config.model_path || config.model_path[0] == '\0' ||
                !config.page_map_path || config.page_map_path[0] == '\0') {
            if (error) {
                *error = "RPP data movement requires model and page map paths";
            }
            replay_.clear();
            return false;
        }
        if (!host_prefetcher_.configure(
                    config.model_path,
                    config.page_map_path,
                    config.prefetch_threads,
                    error)) {
            replay_.clear();
            return false;
        }
    }
    if (gpu_transfer == "on") {
        constexpr uint64_t mib = 1024ULL * 1024ULL;
        const std::string completion_trace =
            config.trace_path && config.trace_path[0] != '\0'
                ? std::string(config.trace_path) + ".gpu_transfer.jsonl"
                : std::string();
        if (!gpu_transfer_.configure(
                    &host_prefetcher_,
                    static_cast<uint64_t>(config.gpu_cache_mib) * mib,
                    static_cast<uint64_t>(config.gpu_staging_mib) * mib,
                    completion_trace,
                    error)) {
            host_prefetcher_.stop();
            replay_.clear();
            return false;
        }
    }
    if (gpu_correction == "on") {
        constexpr uint64_t mib = 1024ULL * 1024ULL;
        if (!gpu_cache_.configure(
                    &host_prefetcher_,
                    model_,
                    static_cast<uint64_t>(config.gpu_cache_mib) * mib,
                    static_cast<uint64_t>(config.gpu_staging_mib) * mib,
                    config.gpu_copy_workers,
                    gpu_queue_policy == "deadline"
                        ? llama_rpp_gpu_queue_policy::deadline
                        : llama_rpp_gpu_queue_policy::fifo,
                    error)) {
            gpu_transfer_.stop();
            host_prefetcher_.stop();
            replay_.clear();
            return false;
        }
    }
    host_prefetch_enabled_ = host_prefetch == "pretouch";
    gpu_compute_enabled_ = gpu_compute == "on";

    if (config.trace_path && config.trace_path[0] != '\0') {
        trace_.open(config.trace_path, std::ios::out | std::ios::trunc);
        if (!trace_.is_open()) {
            if (error) {
                *error = std::string("failed to open RPP trace output: ") + config.trace_path;
            }
            gpu_cache_.stop();
            gpu_transfer_.stop();
            host_prefetcher_.stop();
            replay_.clear();
            return false;
        }
    }

    enable_prefill_ = config.enable_prefill;
    enable_decode_ = config.enable_decode;
    prefetch_depth_ = config.prefetch_depth;
    prefetch_top_k_ = config.prefetch_top_k;
    enabled_ = true;
    LLAMA_LOG_INFO(
            "%s: loaded %zu RPP routes (%zu layer predictions), prefill=%d, "
            "decode=%d, depth=%d, top_k=%d\n",
            __func__,
            replay_.stats().routes,
            replay_.stats().layer_predictions,
            enable_prefill_,
            enable_decode_,
            prefetch_depth_,
            prefetch_top_k_);
    return true;
}

bool llama_rpp_runtime::enabled() const {
    return enabled_;
}

llama_rpp_gpu_cache * llama_rpp_runtime::compute_cache_for_active_ubatch() {
    if (!gpu_compute_enabled_ || active_ubatch_.empty()) {
        return nullptr;
    }
    for (const auto & token : active_ubatch_) {
        if (token.phase != LLAMA_RPP_PHASE_DECODE || !phase_enabled(token.phase)) {
            return nullptr;
        }
    }
    return &gpu_cache_;
}

bool llama_rpp_runtime::set_batch_metadata(
        const llama_rpp_token_metadata * tokens,
        size_t n_tokens,
        std::string * error) {
    batch_metadata_.clear();
    if (!enabled_) {
        return true;
    }
    if (n_tokens > 0 && tokens == nullptr) {
        if (error) {
            *error = "RPP batch metadata pointer is null";
        }
        return false;
    }

    for (size_t i = 0; i < n_tokens; ++i) {
        const auto & token = tokens[i];
        if (token.seq_id < 0 || token.pos < 0) {
            if (error) {
                *error = "RPP batch metadata has invalid seq_id or position";
            }
            batch_metadata_.clear();
            return false;
        }
        const seq_pos_key key = {token.seq_id, token.pos};
        if (!batch_metadata_.emplace(key, token).second) {
            if (error) {
                *error = "duplicate RPP batch metadata for seq_id/position";
            }
            batch_metadata_.clear();
            return false;
        }
    }
    return true;
}

bool llama_rpp_runtime::submit_route(
        int64_t request_id,
        llama_pos position,
        enum llama_rpp_phase phase,
        const llama_rpp_layer_prediction_input * layers,
        size_t n_layers,
        std::string * error) {
    if (!enabled_) {
        if (error) {
            *error = "RPP runtime is disabled";
        }
        return false;
    }
    if (request_id < 0 || position < 0 || phase == LLAMA_RPP_PHASE_UNKNOWN ||
            layers == nullptr || n_layers == 0) {
        if (error) {
            *error = "invalid online RPP route";
        }
        return false;
    }

    llama_rpp_route route;
    route.layers.reserve(n_layers);
    std::set<int32_t> seen_layers;
    for (size_t i = 0; i < n_layers; ++i) {
        const auto & input = layers[i];
        if (input.layer < 0 || input.experts == nullptr || input.n_experts == 0 ||
                !seen_layers.insert(input.layer).second) {
            if (error) {
                *error = "invalid or duplicate online RPP layer";
            }
            return false;
        }
        llama_rpp_layer_prediction prediction;
        prediction.layer = input.layer;
        prediction.confidence = input.confidence;
        prediction.experts.assign(input.experts, input.experts + input.n_experts);
        if (input.expert_confidences != nullptr) {
            prediction.expert_confidences.assign(
                    input.expert_confidences,
                    input.expert_confidences + input.n_experts);
        }
        route.layers.push_back(std::move(prediction));
    }
    replay_.upsert({request_id, position, phase}, std::move(route));
    return true;
}

void llama_rpp_runtime::begin_ubatch(const llama_ubatch & ubatch) {
    active_ubatch_.clear();
    pending_true_routes_.clear();
    if (!enabled_) {
        return;
    }

    active_ubatch_.reserve(ubatch.n_tokens);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        llama_rpp_token_metadata token = {
            -1,
            ubatch.n_seq_id[i] > 0 ? ubatch.seq_id[i][0] : -1,
            ubatch.pos[i],
            ubatch.token ? ubatch.token[i] : LLAMA_TOKEN_NULL,
            LLAMA_RPP_PHASE_UNKNOWN,
        };

        const auto it = batch_metadata_.find({token.seq_id, token.pos});
        if (it != batch_metadata_.end()) {
            token = it->second;
        }
        active_ubatch_.push_back(token);

        for (int32_t layer = 0; layer < prefetch_depth_; ++layer) {
            schedule_prefetch(token, layer);
        }
    }
}

void llama_rpp_runtime::end_ubatch() {
    if (gpu_compute_enabled_ && model_) {
        for (int32_t layer = 0;
                layer < static_cast<int32_t>(model_->layers.size());
                ++layer) {
            gpu_cache_.release_layer(layer);
        }
    }
    pending_true_routes_.clear();
    active_ubatch_.clear();
    downstream_requested_.clear();
}

bool llama_rpp_runtime::eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * runtime = static_cast<llama_rpp_runtime *>(user_data);
    if (!runtime) {
        return true;
    }

    const bool is_topk = starts_with(tensor ? tensor->name : nullptr, "ffn_moe_topk-");
    const bool is_cache_ids =
        starts_with(tensor ? tensor->name : nullptr, "ffn_moe_cache_ids-");
    const bool is_moe_out =
        is_layer_tensor(tensor ? tensor->name : nullptr, "ffn_moe");
    const bool rpp_requested = runtime->enabled_ &&
        runtime->should_observe_router() &&
        (is_topk || (runtime->gpu_compute_enabled_ && (is_cache_ids || is_moe_out)));

    if (ask) {
        if (rpp_requested && !runtime->router_callback_seen_) {
            runtime->router_callback_seen_ = true;
            LLAMA_LOG_INFO(
                    "%s: observing true router tensor %s, type=%s, contiguous=%d, "
                    "shape=[%lld,%lld], strides=[%zu,%zu]\n",
                    __func__,
                    tensor->name,
                    ggml_type_name(tensor->type),
                    ggml_is_contiguous(tensor),
                    static_cast<long long>(tensor->ne[0]),
                    static_cast<long long>(tensor->ne[1]),
                    tensor->nb[0],
                    tensor->nb[1]);
        }
        const bool downstream_requested = runtime->call_downstream(tensor, true);
        if (downstream_requested) {
            runtime->downstream_requested_.insert(tensor);
        }
        return rpp_requested || downstream_requested;
    }

    bool result = true;
    if (rpp_requested && is_topk) {
        result = runtime->observe_true_route(tensor);
    } else if (rpp_requested && is_cache_ids) {
        result = runtime->remap_cache_ids(tensor);
    }
    if (runtime->downstream_requested_.erase(tensor) != 0) {
        result = runtime->call_downstream(tensor, false) && result;
    }
    return result;
}

bool llama_rpp_runtime::should_observe_router() const {
    for (const auto & token : active_ubatch_) {
        if (phase_enabled(token.phase)) {
            return true;
        }
    }
    return false;
}

bool llama_rpp_runtime::observe_true_route(ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_I32) {
        if (tensor && !unsupported_router_tensor_logged_) {
            unsupported_router_tensor_logged_ = true;
            LLAMA_LOG_ERROR(
                    "%s: cannot observe %s: type=%s\n",
                    __func__,
                    tensor->name,
                    ggml_type_name(tensor->type));
        }
        return true;
    }

    const int32_t layer = parse_layer(tensor->name);
    if (layer < 0) {
        return true;
    }

    const int64_t n_experts = tensor->ne[0];
    const int64_t n_tokens = tensor->ne[1];
    if (n_experts <= 0 || n_tokens <= 0 ||
            static_cast<size_t>(n_tokens) != active_ubatch_.size()) {
        return true;
    }

    std::vector<std::vector<int32_t>> layer_routes;
    layer_routes.reserve(n_tokens);
    for (int64_t token_index = 0; token_index < n_tokens; ++token_index) {
        std::vector<int32_t> true_experts;
        true_experts.reserve(n_experts);
        for (int64_t expert_index = 0; expert_index < n_experts; ++expert_index) {
            int32_t expert = -1;
            const size_t offset =
                static_cast<size_t>(token_index) * tensor->nb[1] +
                static_cast<size_t>(expert_index) * tensor->nb[0];
            if (tensor->buffer != nullptr) {
                ggml_backend_tensor_get(tensor, &expert, offset, sizeof(expert));
            } else if (tensor->data != nullptr) {
                std::memcpy(
                        &expert,
                        static_cast<const char *>(tensor->data) + offset,
                        sizeof(expert));
            } else {
                return true;
            }
            true_experts.push_back(expert);
        }
        layer_routes.push_back(true_experts);
        schedule_prefetch(active_ubatch_[token_index], layer + prefetch_depth_);
        if (!gpu_compute_enabled_ &&
                gpu_cache_.enabled() &&
                phase_enabled(active_ubatch_[token_index].phase)) {
            const llama_rpp_prefetch_key correction_key = {
                active_ubatch_[token_index].request_id,
                active_ubatch_[token_index].pos,
                active_ubatch_[token_index].phase,
                layer,
            };
            gpu_cache_.ensure_resident(correction_key, layer, true_experts);
        }
        if (!gpu_compute_enabled_) {
            write_trace_event(active_ubatch_[token_index], layer, true_experts);
        }
    }
    if (gpu_compute_enabled_) {
        pending_true_routes_[layer] = std::move(layer_routes);
    }
    return true;
}

bool llama_rpp_runtime::remap_cache_ids(ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_I32) {
        return false;
    }
    const int32_t layer = parse_layer(tensor->name);
    const auto pending = pending_true_routes_.find(layer);
    if (layer < 0 || pending == pending_true_routes_.end()) {
        return false;
    }
    const int64_t n_experts = tensor->ne[0];
    const int64_t n_tokens = tensor->ne[1];
    if (n_tokens != static_cast<int64_t>(pending->second.size()) ||
            n_tokens != static_cast<int64_t>(active_ubatch_.size())) {
        return false;
    }

    for (int64_t token_index = 0; token_index < n_tokens; ++token_index) {
        const auto & true_experts = pending->second[token_index];
        const llama_rpp_prefetch_key correction_key = {
            active_ubatch_[token_index].request_id,
            active_ubatch_[token_index].pos,
            active_ubatch_[token_index].phase,
            layer,
        };
        const auto correction = gpu_cache_.ensure_resident(
                correction_key, layer, true_experts, true);
        if (!correction.success) {
            return false;
        }
        for (int64_t expert_index = 0; expert_index < n_experts; ++expert_index) {
            const int32_t slot = gpu_cache_.slot_for(
                    layer, true_experts[expert_index]);
            if (slot < 0) {
                return false;
            }
            const size_t offset =
                static_cast<size_t>(token_index) * tensor->nb[1] +
                static_cast<size_t>(expert_index) * tensor->nb[0];
            ggml_backend_tensor_set(tensor, &slot, offset, sizeof(slot));
        }
        write_trace_event(active_ubatch_[token_index], layer, true_experts);
    }
    pending_true_routes_.erase(pending);
    return true;
}

bool llama_rpp_runtime::call_downstream(ggml_tensor * tensor, bool ask) {
    return downstream_callback_
        ? downstream_callback_(tensor, ask, downstream_user_data_)
        : false;
}

bool llama_rpp_runtime::phase_enabled(enum llama_rpp_phase phase) const {
    if (phase == LLAMA_RPP_PHASE_PREFILL) {
        return enable_prefill_;
    }
    if (phase == LLAMA_RPP_PHASE_DECODE) {
        return enable_decode_;
    }
    return false;
}

void llama_rpp_runtime::schedule_prefetch(
        const llama_rpp_token_metadata & token,
        int32_t layer) {
    if (!host_prefetcher_.enabled() || prefetch_depth_ <= 0 ||
            layer < 0 || !phase_enabled(token.phase)) {
        return;
    }

    const llama_rpp_trace_key route_key = {
        token.request_id,
        token.pos,
        token.phase,
    };
    const llama_rpp_route * route = replay_.find(route_key);
    const llama_rpp_layer_prediction * prediction =
        route ? route->find_layer(layer) : nullptr;
    if (!prediction) {
        return;
    }

    const llama_rpp_prefetch_key prefetch_key = {
        token.request_id,
        token.pos,
        token.phase,
        layer,
    };
    const size_t selected_count = std::min(
            prediction->experts.size(),
            static_cast<size_t>(prefetch_top_k_));
    const std::vector<int32_t> selected_experts(
            prediction->experts.begin(),
            prediction->experts.begin() + selected_count);
    if (host_prefetch_enabled_) {
        host_prefetcher_.enqueue(prefetch_key, selected_experts);
    }
    gpu_cache_.prefetch(prefetch_key, layer, selected_experts);
    gpu_transfer_.enqueue(prefetch_key, selected_experts);
}

void llama_rpp_runtime::write_trace_event(
        const llama_rpp_token_metadata & token,
        int32_t layer,
        const std::vector<int32_t> & true_experts) {
    if (!trace_.is_open()) {
        return;
    }
    if (!phase_enabled(token.phase)) {
        return;
    }

    const llama_rpp_trace_key key = {
        token.request_id,
        token.pos,
        token.phase,
    };
    const llama_rpp_route * route = replay_.find(key);
    const llama_rpp_layer_prediction * predicted = route ? route->find_layer(layer) : nullptr;

    const std::vector<int32_t> predicted_experts =
        predicted ? predicted->experts : std::vector<int32_t>{};
    const size_t prefetched_count = std::min(
            predicted_experts.size(),
            static_cast<size_t>(prefetch_top_k_));
    const std::vector<int32_t> prefetched_experts(
            predicted_experts.begin(),
            predicted_experts.begin() + prefetched_count);
    const auto hits = set_intersection(true_experts, predicted_experts);
    const auto misses = set_difference(true_experts, predicted_experts);
    const auto wasted = set_difference(predicted_experts, true_experts);
    const llama_rpp_prefetch_key prefetch_key = {
        token.request_id,
        token.pos,
        token.phase,
        layer,
    };
    const auto prefetch = host_prefetcher_.snapshot(prefetch_key);
    const auto correction = gpu_cache_.snapshot(prefetch_key);
    const auto gpu_transfer = gpu_transfer_.snapshot(prefetch_key);
    const int64_t router_observed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

    json event = {
        {"request_id", token.request_id},
        {"seq_id", token.seq_id},
        {"position", token.pos},
        {"token_id", token.token},
        {"phase", llama_rpp_phase_name(token.phase)},
        {"layer", layer},
        {"predicted_experts", predicted_experts},
        {"prefetched_experts", prefetched_experts},
        {"true_experts", true_experts},
        {"hit_experts", hits},
        {"missing_experts", misses},
        {"wasted_experts", wasted},
        {"prediction_found", predicted != nullptr},
        {"prefetch_depth", prefetch_depth_},
        {"prefetch_top_k", prefetch_top_k_},
        {"host_prefetch_state", llama_rpp_prefetch_state_name(prefetch.state)},
        {"host_prefetch_pages", prefetch.pages},
        {"host_prefetch_bytes", prefetch.bytes},
        {"host_prefetch_us", prefetch.duration_us},
        {"host_prefetch_minor_faults", prefetch.minor_faults},
        {"host_prefetch_major_faults", prefetch.major_faults},
        {"host_prefetch_ready_before_router",
            prefetch.state == llama_rpp_prefetch_state::done &&
            prefetch.complete_us <= router_observed_us},
        {"gpu_transfer_state", llama_rpp_gpu_transfer_state_name(gpu_transfer.state)},
        {"gpu_transfer_bytes", gpu_transfer.bytes},
        {"gpu_transfer_pack_us", gpu_transfer.pack_us},
        {"gpu_transfer_h2d_us", gpu_transfer.transfer_us},
        {"gpu_cache_offset", gpu_transfer.cache_offset},
        {"gpu_cache_capacity", gpu_transfer.cache_capacity},
        {"gpu_cache_used", gpu_transfer.cache_used},
        {"gpu_transfer_device", gpu_transfer.device},
        {"gpu_transfer_ready_before_router",
            gpu_transfer.state == llama_rpp_gpu_transfer_state::done &&
            gpu_transfer.complete_us <= router_observed_us},
        {"gpu_correction_attempted", correction.attempted},
        {"gpu_correction_success", correction.success},
        {"gpu_correction_requested", correction.requested},
        {"gpu_correction_ready_hits", correction.ready_hits},
        {"gpu_correction_waited_prefetch", correction.waited_prefetch},
        {"gpu_correction_loaded_on_demand", correction.loaded_on_demand},
        {"gpu_correction_failed", correction.failed},
        {"gpu_correction_evictions", correction.evictions},
        {"gpu_prefetch_selected_true", correction.selected_for_prefetch},
        {"gpu_prefetch_ready_at_use", correction.prefetch_ready_at_use},
        {"gpu_prefetch_queued_at_use", correction.prefetch_queued_at_use},
        {"gpu_prefetch_loading_at_use", correction.prefetch_loading_at_use},
        {"gpu_prefetch_absent_at_use", correction.prefetch_absent_at_use},
        {"gpu_prefetch_evicted_before_use", correction.prefetch_evicted_before_use},
        {"gpu_correction_bytes", correction.correction_bytes},
        {"gpu_correction_us", correction.correction_us},
        {"gpu_correction_resident_entries", correction.resident_entries},
        {"gpu_correction_cache_slots", correction.cache_slots},
        {"gpu_correction_slot_stride", correction.slot_stride},
        {"gpu_compute_enabled", gpu_compute_enabled_},
    };
    if (!correction.locations.empty()) {
        event["gpu_correction_locations"] = json::array();
        for (const auto & location : correction.locations) {
            event["gpu_correction_locations"].push_back({
                {"expert", location.expert},
                {"slot", location.slot},
                {"offset", location.offset},
                {"bytes", location.bytes},
            });
        }
    }
    if (!correction.expert_timings.empty()) {
        event["gpu_expert_timings"] = json::array();
        for (const auto & timing : correction.expert_timings) {
            json timing_event = {
                {"expert", timing.expert},
                {"selected_for_prefetch", timing.selected_for_prefetch},
                {"evicted_before_use", timing.evicted_before_use},
                {"state_at_use", timing.state_at_use},
                {"enqueue_us", timing.enqueue_us},
                {"copy_start_us", timing.start_us},
                {"copy_complete_us", timing.complete_us},
                {"use_barrier_us", timing.use_us},
            };
            if (timing.enqueue_us > 0) {
                timing_event["available_overlap_us"] =
                    timing.use_us - timing.enqueue_us;
            }
            if (timing.start_us > 0 && timing.enqueue_us > 0) {
                timing_event["copy_queue_us"] =
                    timing.start_us - timing.enqueue_us;
            }
            if (timing.complete_us > 0 && timing.start_us > 0) {
                timing_event["copy_service_us"] =
                    timing.complete_us - timing.start_us;
            }
            if (timing.complete_us > 0) {
                timing_event["ready_slack_us"] =
                    timing.use_us - timing.complete_us;
            }
            event["gpu_expert_timings"].push_back(std::move(timing_event));
        }
    }
    if (!correction.error.empty()) {
        event["gpu_correction_error"] = correction.error;
    }
    if (prefetch.enqueue_us > 0) {
        event["host_prefetch_enqueue_to_router_us"] =
            router_observed_us - prefetch.enqueue_us;
    }
    if (prefetch.start_us > 0) {
        event["host_prefetch_queue_us"] =
            prefetch.start_us - prefetch.enqueue_us;
    }
    if (prefetch.complete_us > 0) {
        event["host_prefetch_complete_to_router_us"] =
            router_observed_us - prefetch.complete_us;
    }
    if (!prefetch.error.empty()) {
        event["host_prefetch_error"] = prefetch.error;
    }
    if (gpu_transfer.enqueue_us > 0) {
        event["gpu_transfer_enqueue_to_router_us"] =
            router_observed_us - gpu_transfer.enqueue_us;
    }
    if (gpu_transfer.start_us > 0) {
        event["gpu_transfer_queue_us"] =
            gpu_transfer.start_us - gpu_transfer.enqueue_us;
    }
    if (gpu_transfer.complete_us > 0) {
        event["gpu_transfer_complete_to_router_us"] =
            router_observed_us - gpu_transfer.complete_us;
    }
    if (!gpu_transfer.error.empty()) {
        event["gpu_transfer_error"] = gpu_transfer.error;
    }
    if (predicted) {
        event["confidence"] = predicted->confidence;
        event["expert_confidences"] = predicted->expert_confidences;
    }

    trace_ << event.dump() << '\n';
    trace_.flush();
}
