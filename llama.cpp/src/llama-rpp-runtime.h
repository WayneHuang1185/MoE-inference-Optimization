#pragma once

#include "llama-rpp-gpu-cache.h"
#include "llama-rpp-gpu-transfer.h"
#include "llama-rpp-prefetch.h"
#include "llama-rpp-predictor.h"

#include "ggml-backend.h"

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct llama_ubatch;
struct llama_model;

class llama_rpp_runtime {
public:
    llama_rpp_runtime(
            const llama_model * model,
            ggml_backend_sched_eval_callback downstream_callback,
            void * downstream_user_data);

    bool configure(const llama_rpp_config & config, std::string * error);

    bool enabled() const;
    llama_rpp_gpu_cache * compute_cache_for_active_ubatch();

    bool set_batch_metadata(
            const llama_rpp_token_metadata * tokens,
            size_t n_tokens,
            std::string * error);

    bool submit_route(
            int64_t request_id,
            llama_pos position,
            enum llama_rpp_phase phase,
            const llama_rpp_layer_prediction_input * layers,
            size_t n_layers,
            std::string * error);

    void begin_ubatch(const llama_ubatch & ubatch);
    void end_ubatch();

    static bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data);

private:
    struct seq_pos_key {
        llama_seq_id seq_id;
        llama_pos pos;

        bool operator<(const seq_pos_key & other) const {
            return seq_id != other.seq_id ? seq_id < other.seq_id : pos < other.pos;
        }
    };

    bool observe_true_route(ggml_tensor * tensor);
    bool remap_cache_ids(ggml_tensor * tensor);
    bool should_observe_router() const;
    bool call_downstream(ggml_tensor * tensor, bool ask);
    bool phase_enabled(enum llama_rpp_phase phase) const;
    void schedule_prefetch(
            const llama_rpp_token_metadata & token,
            int32_t layer);
    void write_trace_event(
            const llama_rpp_token_metadata & token,
            int32_t layer,
            const std::vector<int32_t> & true_experts);

    bool enabled_ = false;
    bool enable_prefill_ = false;
    bool enable_decode_ = true;
    bool host_prefetch_enabled_ = false;
    bool gpu_compute_enabled_ = false;
    bool router_callback_seen_ = false;
    bool unsupported_router_tensor_logged_ = false;
    int32_t prefetch_depth_ = 0;
    int32_t prefetch_top_k_ = 8;

    llama_rpp_replay_predictor replay_;
    llama_rpp_host_prefetcher host_prefetcher_;
    llama_rpp_gpu_cache gpu_cache_;
    llama_rpp_gpu_transfer gpu_transfer_;

    std::map<seq_pos_key, llama_rpp_token_metadata> batch_metadata_;
    std::vector<llama_rpp_token_metadata> active_ubatch_;
    std::map<int32_t, std::vector<std::vector<int32_t>>> pending_true_routes_;

    std::ofstream trace_;

    ggml_backend_sched_eval_callback downstream_callback_ = nullptr;
    void * downstream_user_data_ = nullptr;
    const llama_model * model_ = nullptr;
    std::set<ggml_tensor *> downstream_requested_;
};
