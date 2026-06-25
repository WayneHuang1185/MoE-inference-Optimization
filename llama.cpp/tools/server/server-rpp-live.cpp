#include "server-rpp-live.h"

#include "ggml.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

#if defined(LLAMA_RPP_LIVE)
#include <ATen/Parallel.h>
#include <torch/script.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

namespace {

struct expert_key {
    int layer = -1;
    int expert = -1;

    bool operator<(const expert_key & other) const {
        return std::tie(layer, expert) < std::tie(other.layer, other.expert);
    }

    bool operator==(const expert_key & other) const {
        return layer == other.layer && expert == other.expert;
    }
};

struct oracle_token {
    std::string sample_id;
    std::string phase = "decode";
    int32_t token_index = 0;
    int32_t decode_index = 0;
    int32_t token_id = -1;
    std::vector<expert_key> experts;
};

struct queued_token {
    std::string sample_id;
    int32_t sample_order = 0;
    int32_t slot_id = -1;
    std::string phase = "decode";
    int32_t token_index = 0;
    int32_t decode_index = 0;
    int32_t token_id = -1;
    std::vector<expert_key> experts;
};

struct predicted_token {
    std::string sample_id;
    int32_t slot_id = -1;
    int32_t task_id = -1;
    std::string phase = "decode";
    int32_t token_index = 0;
    int32_t decode_index = 0;
    int32_t token_id = -1;
    int32_t batch_index = -1;
    int32_t ubatch_id = -1;
    int32_t ubatch_offset = -1;
    std::vector<expert_key> experts;
};

struct expert_range {
    int layer = -1;
    int expert = -1;
    std::string role;
    int64_t file_start = 0;
    int64_t file_end = 0;
};

struct sample_state {
    int32_t order = 0;
    size_t cursor = 0;
};

struct planned_ubatch {
    int64_t source_decode_call_id = -1;
    int32_t source_ubatch_id = -1;
    int64_t target_decode_call_id = -1;
    int32_t target_ubatch_id = -1;
    int32_t target_first_batch_order = -1;
    int32_t target_token_count = 0;
    int32_t prefill_token_count = 0;
    int32_t decode_token_count = 0;
    std::vector<predicted_token> tokens;
    std::map<expert_key, int> counters;
};

struct async_prefetch_job {
    expert_key key;
    int64_t source_decode_call_id = -1;
    int32_t source_ubatch_id = -1;
    int64_t target_decode_call_id = -1;
    int32_t target_ubatch_id = -1;
    int32_t target_first_batch_order = -1;
    int32_t target_token_count = 0;
    int32_t counter = 0;
    double density = 0.0;
    int32_t required_count = 0;
    int32_t target_ubatch_distance = 0;
    int64_t enqueue_ts_us = 0;
};

struct expert_runtime_state {
    int64_t last_used_decode_call_id = -1;
    int32_t last_used_ubatch_id = -1;
    int64_t last_prefetched_decode_call_id = -1;
    int32_t last_prefetched_ubatch_id = -1;
    int64_t protected_until_decode_call_id = -1;
    int32_t protected_until_ubatch_id = -1;
    bool inflight = false;
};

struct reclaim_candidate {
    expert_key key;
    double future_density_sum = 0.0;
    int32_t future_counter_sum = 0;
    int32_t next_use_distance = std::numeric_limits<int32_t>::max();
    int64_t last_used_decode_call_id = -1;
    int64_t resident_bytes = 0;
    std::string decision = "candidate";
};

struct live_config {
    bool enabled = false;
    bool cont_batching = true;
    bool prefetch_prefill_only = false;
    std::string scheduler;
    std::string model_path;
    std::string tensor_ranges_path;
    std::string oracle_trace_path;
    std::string rpp_live_model_path;
    std::string prefetch_io_mode = "readahead";
    std::string prefetch_queue_policy = "priority";
    std::string prefetch_fifo_candidate_order = "density";
    std::string reclaim_mode = "trace-priority";
    std::filesystem::path out_dir;
    int32_t window_size = 10;
    int32_t prefetch_threshold = 5;
    int32_t ubatch_prefetch_min_count = 5;
    int32_t predict_topk = 8;
    int32_t rpp_max_seq_len = 512;
    int32_t predictor_threads = 1;
    int32_t async_prefetch_queue_cap = 512;
    int32_t ubatch_prefetch_max_experts = 0;
    int32_t prefetch_max_staleness_ubatches = 0;
    int32_t reclaim_queue_cap = 512;
    int32_t reclaim_lookahead_ubatches = 4;
    int32_t reclaim_protect_ubatches = 2;
    int32_t cross_ubatch_layer_prefetch = 0;
    int32_t cross_ubatch_layer_lookahead = 1;
    int32_t layer_frontier_prefetch = 0;
    int32_t layer_frontier_lookahead_layers = 8;
    int32_t layer_frontier_jobs_per_tick = 64;
    int32_t layer_frontier_max_distance = 60;
    double ubatch_prefetch_density_threshold = 0.5;
    bool async_prefetch = false;
};

struct physical_ubatch_runtime {
    planned_ubatch plan;
    std::map<int32_t, std::vector<async_prefetch_job>> jobs_by_layer;
    int32_t released_until_layer = -1;
    std::set<int32_t> frontier_released_layers;
};

std::mutex g_mutex;
std::atomic<int64_t> g_decode_call_id { 0 };
live_config g_config;
std::unordered_map<std::string, std::vector<oracle_token>> g_oracle;
std::map<std::pair<int, int>, std::vector<expert_range>> g_ranges;
std::map<std::string, sample_state> g_samples;
std::vector<std::string> g_sample_order;
std::vector<queued_token> g_queue;
std::vector<predicted_token> g_pending_predicted_order;
std::map<expert_key, int> g_counters;
std::set<expert_key> g_last_planned;
bool g_loaded = false;
bool g_header_written = false;
bool g_warned_unloaded = false;
bool g_prefetch_closed = false;
int64_t g_prefetch_stop_decode_call = -1;
int64_t g_queue_rows = 0;
int64_t g_counter_rows = 0;
int64_t g_prefetch_rows = 0;
int64_t g_ubatch_prefetch_rows = 0;
int64_t g_ubatch_counter_rows = 0;
int64_t g_ubatch_lead_rows = 0;
int64_t g_reclaim_rows = 0;
int64_t g_mismatch_rows = 0;
int64_t g_token_slot_mismatch_rows = 0;
int64_t g_prefill_token_slot_mismatch_rows = 0;
int64_t g_decode_token_slot_mismatch_rows = 0;
int64_t g_model_order_rows = 0;
int64_t g_predicted_order_rows = 0;
int64_t g_predicted_model_order_rows = 0;
int64_t g_order_prediction_mismatch_rows = 0;
int64_t g_ubatch_prediction_mismatch_rows = 0;
int64_t g_slot_state_rows = 0;
int64_t g_fadvise_calls = 0;
int64_t g_fadvise_errors = 0;
int64_t g_advised_bytes = 0;
int64_t g_decode_rpp_prediction_rows = 0;
int64_t g_decode_rpp_missing_prediction_rows = 0;
int64_t g_rpp_timing_rows = 0;
int64_t g_async_prefetch_enqueued_rows = 0;
int64_t g_async_prefetch_completed_rows = 0;
int64_t g_async_prefetch_dropped_rows = 0;
int64_t g_ubatch_plan_rows = 0;
int64_t g_expert_use_plan_rows = 0;
int64_t g_ubatch_decode_end_rows = 0;
int64_t g_physical_ubatch_rows = 0;
int64_t g_layer_progress_rows = 0;
int64_t g_cross_ubatch_released_rows = 0;
int64_t g_layer_frontier_released_rows = 0;
int64_t g_prefetch_priority_planned_rows = 0;
int64_t g_prefetch_priority_enqueued_rows = 0;
int64_t g_prefetch_priority_completed_rows = 0;
int64_t g_prefetch_priority_dropped_low_priority_rows = 0;
int64_t g_prefetch_priority_replaced_rows = 0;
int64_t g_prefetch_priority_stale_rows = 0;
int64_t g_reclaim_priority_candidates = 0;
int64_t g_reclaim_priority_trace_rows = 0;
int64_t g_reclaim_fadvise_calls = 0;
int64_t g_reclaim_fadvise_errors = 0;
int64_t g_reclaim_advised_bytes = 0;
int64_t g_latest_source_decode_call_id = -1;
int32_t g_latest_source_ubatch_id = -1;
std::map<int32_t, std::string> g_slot_sample_ids;
std::map<int32_t, std::vector<int32_t>> g_slot_decode_histories;
std::map<expert_key, expert_runtime_state> g_expert_states;
std::vector<physical_ubatch_runtime> g_active_physical_ubatches;
int64_t g_eval_decode_call_id = -1;
int32_t g_eval_current_ubatch = -1;
int32_t g_eval_last_layer = -1;

std::mutex g_async_mutex;
std::condition_variable g_async_cv;
std::vector<async_prefetch_job> g_async_queue;
std::thread g_async_worker;
bool g_async_worker_running = false;
bool g_async_stop = false;
bool g_atexit_registered = false;

#if defined(LLAMA_RPP_LIVE)
std::unique_ptr<torch::jit::script::Module> g_rpp_module;
#endif

#if defined(__unix__) || defined(__APPLE__)
int g_model_fd = -1;
#endif

bool queue_is_all_prefill();
std::vector<std::pair<expert_key, int>> sorted_prefetch_candidates(const std::map<expert_key, int> & counters);
void start_async_prefetch_worker_if_needed();

std::vector<std::string> split_csv_line(const std::string & line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

void ensure_out_dir() {
    if (g_config.out_dir.empty()) {
        const char * env = std::getenv("RPP_LIVE_OUT_DIR");
        g_config.out_dir = env && env[0] ? std::filesystem::path(env) : std::filesystem::path("rpp_live");
    }
    std::filesystem::create_directories(g_config.out_dir);
}

void append_line(const std::string & name, const std::string & line) {
    std::ofstream out(g_config.out_dir / name, std::ios::app);
    if (out.good()) {
        out << line << '\n';
    }
}

std::string csv_escape(const std::string & value) {
    bool quote = false;
    for (char c : value) {
        quote = quote || c == ',' || c == '"' || c == '\n' || c == '\r';
    }
    if (!quote) {
        return value;
    }
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

int64_t monotonic_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void append_rpp_timing(
        int64_t decode_call_id,
        const std::string & stage,
        int64_t start_ts_us,
        int64_t end_ts_us,
        int64_t rows,
        int64_t tokens,
        const std::string & detail = "") {
    append_line(
        "rpp_timing_trace.csv",
        std::to_string(decode_call_id) + "," +
        csv_escape(stage) + "," +
        std::to_string(start_ts_us) + "," +
        std::to_string(end_ts_us) + "," +
        std::to_string(end_ts_us - start_ts_us) + "," +
        std::to_string(rows) + "," +
        std::to_string(tokens) + "," +
        csv_escape(detail));
    ++g_rpp_timing_rows;
}

const char * slot_state_name(int32_t state) {
    switch (state) {
        case 0: return "IDLE";
        case 1: return "WAIT_OTHER";
        case 2: return "STARTED";
        case 3: return "PROCESSING_PROMPT";
        case 4: return "DONE_PROMPT";
        case 5: return "GENERATING";
        default: return "UNKNOWN";
    }
}

bool should_break_for_prompt_checkpoint(int32_t task_n_tokens, int32_t prompt_n_tokens, int32_t n_batch, int32_t n_ubatch) {
    const int32_t checkpoint_offsets[] = {4 + n_ubatch, 4};
    for (const int32_t offset : checkpoint_offsets) {
        const int32_t n_last = std::min(n_batch, offset);
        if (task_n_tokens == prompt_n_tokens + n_last) {
            return true;
        }
    }
    return false;
}

void write_headers_once() {
    if (g_header_written) {
        return;
    }
    ensure_out_dir();
    append_line("queue_trace.csv", "decode_call_id,queue_index,sample_id,sample_order,slot_id,phase,token_index,decode_index,token_id,expert_count");
    append_line("expert_counter_trace.csv", "decode_call_id,layer,expert,counter");
    append_line("prefetch_events.csv", "decode_call_id,layer,expert,counter,fadvise_calls,fadvise_errors,advised_bytes");
    append_line("ubatch_counter_trace.csv", "source_decode_call_id,current_ubatch_id,target_decode_call_id,target_ubatch_id,target_first_batch_order,target_token_count,layer,expert,counter,density");
    append_line("ubatch_prefetch_events.csv", "source_decode_call_id,current_ubatch_id,target_decode_call_id,target_ubatch_id,target_first_batch_order,target_token_count,layer,expert,counter,density,required_count,min_count,density_threshold,prefetch_io_mode,fadvise_ts_us,fadvise_calls,fadvise_errors,advised_bytes");
    append_line("ubatch_prefetch_async_trace.csv", "source_decode_call_id,source_ubatch_id,target_decode_call_id,target_ubatch_id,target_first_batch_order,target_token_count,layer,expert,counter,density,required_count,enqueue_ts_us,start_ts_us,end_ts_us,queue_wait_us,run_us,fadvise_calls,fadvise_errors,advised_bytes,status");
    append_line("ubatch_plan_trace.csv", "source_decode_call_id,source_ubatch_id,target_decode_call_id,target_ubatch_id,target_first_batch_order,target_token_count,prefill_token_count,decode_token_count,candidate_experts,planned_ts_us");
    append_line("prefetch_priority_trace.csv", "status,source_decode_call_id,source_ubatch_id,target_decode_call_id,target_ubatch_id,target_first_batch_order,target_token_count,target_ubatch_distance,layer,expert,counter,density,required_count,queue_size,enqueue_ts_us,start_ts_us,end_ts_us,fadvise_calls,fadvise_errors,advised_bytes");
    append_line("reclaim_priority_trace.csv", "decode_call_id,ubatch_id,layer,expert,future_density_sum,future_counter_sum,next_use_distance,last_used_decode_call_id,resident_bytes,queue_rank,mode,decision,fadvise_calls,fadvise_errors,advised_bytes");
    append_line("expert_state_trace.csv", "decode_call_id,ubatch_id,layer,expert,last_used_decode_call_id,last_used_ubatch_id,last_prefetched_decode_call_id,last_prefetched_ubatch_id,protected_until_decode_call_id,protected_until_ubatch_id,inflight");
    append_line("ubatch_decode_trace.csv", "decode_call_id,current_ubatch_id,logical_batch_offset,logical_batch_n_tokens,runtime_n_ubatch,decode_ts_us");
    append_line("ubatch_decode_end_trace.csv", "decode_call_id,current_ubatch_id,logical_batch_offset,logical_batch_n_tokens,runtime_n_ubatch,decode_end_ts_us,decode_ret");
    append_line("expert_use_plan_trace.csv", "decode_call_id,ubatch_id,layer,expert,counter,density,target_token_count,prefill_token_count,decode_token_count,decode_start_ts_us");
    append_line("physical_ubatch_trace.csv", "decode_call_id,physical_ubatch_id,n_tokens,n_seqs,n_seq_tokens,first_batch_index,last_batch_index,batch_indices");
    append_line("layer_progress_trace.csv", "decode_call_id,physical_ubatch_id,layer,node_name,ts_us");
    append_line("cross_ubatch_release_trace.csv", "decode_call_id,source_ubatch_id,source_layer,target_ubatch_id,released_from_layer,released_to_layer,released_jobs,ts_us");
    append_line("layer_frontier_release_trace.csv", "decode_call_id,source_ubatch_id,source_layer,target_ubatch_id,target_layer,distance,released_jobs,ts_us");
    append_line("ubatch_prefetch_lead_trace.csv", "source_decode_call_id,target_decode_call_id,source_ubatch_id,target_ubatch_id,target_first_batch_order,target_token_count,fadvise_ts_us,pre_decode_ready_ts_us,lead_us,fadvise_calls,fadvise_errors,advised_bytes");
    append_line("reclaim_candidates.csv", "decode_call_id,layer,expert,previously_planned,present_counter,threshold,mode");
    append_line("oracle_mismatch.csv", "decode_call_id,sample_id,decode_index,expected_token_id,actual_token_id,slot_id");
    append_line("token_slot_mismatch.csv", "decode_call_id,slot_id,sample_id,phase,token_index,expected_token_id,actual_token_id");
    append_line("actual_model_order_trace.csv", "decode_call_id,batch_order,logical_batch_index,ubatch_id,ubatch_offset,runtime_n_ubatch,seq_id,slot_id,task_id,sample_id,slot_state,n_decoded,prompt_n_tokens,task_n_tokens,is_generating,pos,token_id,output");
    append_line("slot_state_trace.csv", "decode_call_id,slot_order,slot_id,task_id,sample_id,slot_state,n_decoded,prompt_n_tokens,task_n_tokens,i_batch,is_generating");
    append_line("predicted_model_order_trace.csv", "decode_call_id,batch_order,slot_id,task_id,sample_id,phase,token_index,decode_index,token_id,ubatch_id,ubatch_offset");
    append_line("predicted_global_order_trace.csv", "decode_call_id,queue_index,slot_id,task_id,sample_id,phase,token_index,decode_index,token_id,ubatch_id,ubatch_offset");
    append_line("global_order_prediction_mismatch.csv", "decode_call_id,actual_index,expected_slot_id,actual_slot_id,expected_sample_id,actual_sample_id,expected_phase,actual_phase,expected_token_index,actual_token_index,expected_decode_index,actual_decode_index,expected_ubatch_id,actual_ubatch_id,expected_ubatch_offset,actual_ubatch_offset,reason");
    append_line("decode_rpp_prediction_trace.csv", "decode_call_id,slot_id,task_id,sample_id,decode_index,token_id,history_len,used_len,expert_count");
    append_line("decode_rpp_missing_prediction.csv", "decode_call_id,slot_id,task_id,sample_id,decode_index,token_id,reason");
    append_line("rpp_timing_trace.csv", "decode_call_id,stage,start_ts_us,end_ts_us,duration_us,rows,tokens,detail");
    g_header_written = true;
}

void write_summary() {
    ensure_out_dir();
    json summary = {
        {"scheduler", g_config.scheduler},
        {"oracle_trace", g_config.oracle_trace_path},
        {"rpp_live_model", g_config.rpp_live_model_path},
        {"tensor_ranges", g_config.tensor_ranges_path},
        {"window_size", g_config.window_size},
        {"prefetch_count_threshold", g_config.prefetch_threshold},
        {"prefetch_io_mode", g_config.prefetch_io_mode},
        {"prefetch_queue_policy", g_config.prefetch_queue_policy},
        {"prefetch_fifo_candidate_order", g_config.prefetch_fifo_candidate_order},
        {"prefetch_async", g_config.async_prefetch},
        {"prefetch_async_queue_cap", g_config.async_prefetch_queue_cap},
        {"prefetch_max_staleness_ubatches", g_config.prefetch_max_staleness_ubatches},
        {"ubatch_prefetch_max_experts", g_config.ubatch_prefetch_max_experts},
        {"ubatch_prefetch_min_count", g_config.ubatch_prefetch_min_count},
        {"ubatch_prefetch_density_threshold", g_config.ubatch_prefetch_density_threshold},
        {"prefetch_prefill_only", g_config.prefetch_prefill_only},
        {"predict_topk", g_config.predict_topk},
        {"rpp_max_seq_len", g_config.rpp_max_seq_len},
        {"prefetch_closed", g_prefetch_closed},
        {"prefetch_stop_decode_call", g_prefetch_stop_decode_call},
        {"reclaim_mode", g_config.reclaim_mode},
        {"reclaim_queue_cap", g_config.reclaim_queue_cap},
        {"reclaim_lookahead_ubatches", g_config.reclaim_lookahead_ubatches},
        {"reclaim_protect_ubatches", g_config.reclaim_protect_ubatches},
        {"cross_ubatch_layer_prefetch", g_config.cross_ubatch_layer_prefetch},
        {"cross_ubatch_layer_lookahead", g_config.cross_ubatch_layer_lookahead},
        {"layer_frontier_prefetch", g_config.layer_frontier_prefetch},
        {"layer_frontier_lookahead_layers", g_config.layer_frontier_lookahead_layers},
        {"layer_frontier_jobs_per_tick", g_config.layer_frontier_jobs_per_tick},
        {"layer_frontier_max_distance", g_config.layer_frontier_max_distance},
        {"oracle_samples", g_oracle.size()},
        {"decode_calls", g_decode_call_id.load()},
        {"ubatch_plan_rows", g_ubatch_plan_rows},
        {"expert_use_plan_rows", g_expert_use_plan_rows},
        {"ubatch_decode_end_rows", g_ubatch_decode_end_rows},
        {"physical_ubatch_rows", g_physical_ubatch_rows},
        {"layer_progress_rows", g_layer_progress_rows},
        {"cross_ubatch_released_rows", g_cross_ubatch_released_rows},
        {"layer_frontier_released_rows", g_layer_frontier_released_rows},
        {"queue_rows", g_queue_rows},
        {"expert_counter_rows", g_counter_rows},
        {"prefetch_events", g_prefetch_rows},
        {"ubatch_prefetch_events", g_ubatch_prefetch_rows},
        {"ubatch_counter_rows", g_ubatch_counter_rows},
        {"ubatch_prefetch_lead_rows", g_ubatch_lead_rows},
        {"reclaim_candidates", g_reclaim_rows},
        {"oracle_mismatches", g_mismatch_rows},
        {"token_slot_mismatches", g_token_slot_mismatch_rows},
        {"prefill_token_slot_mismatches", g_prefill_token_slot_mismatch_rows},
        {"decode_token_slot_mismatches", g_decode_token_slot_mismatch_rows},
        {"model_order_rows", g_model_order_rows},
        {"predicted_model_order_rows", g_predicted_model_order_rows},
        {"predicted_order_rows", g_predicted_order_rows},
        {"global_order_prediction_mismatches", g_order_prediction_mismatch_rows},
        {"ubatch_prediction_mismatches", g_ubatch_prediction_mismatch_rows},
        {"slot_state_rows", g_slot_state_rows},
        {"fadvise_calls", g_fadvise_calls},
        {"fadvise_errors", g_fadvise_errors},
        {"advised_bytes", g_advised_bytes},
        {"decode_rpp_prediction_rows", g_decode_rpp_prediction_rows},
        {"decode_rpp_missing_prediction_rows", g_decode_rpp_missing_prediction_rows},
        {"rpp_timing_rows", g_rpp_timing_rows},
        {"async_prefetch_enqueued_rows", g_async_prefetch_enqueued_rows},
        {"async_prefetch_completed_rows", g_async_prefetch_completed_rows},
        {"async_prefetch_dropped_rows", g_async_prefetch_dropped_rows},
        {"prefetch_priority_planned_rows", g_prefetch_priority_planned_rows},
        {"prefetch_priority_enqueued_rows", g_prefetch_priority_enqueued_rows},
        {"prefetch_priority_completed_rows", g_prefetch_priority_completed_rows},
        {"prefetch_priority_dropped_low_priority_rows", g_prefetch_priority_dropped_low_priority_rows},
        {"prefetch_priority_replaced_rows", g_prefetch_priority_replaced_rows},
        {"prefetch_priority_stale_rows", g_prefetch_priority_stale_rows},
        {"reclaim_priority_candidates", g_reclaim_priority_candidates},
        {"reclaim_priority_trace_rows", g_reclaim_priority_trace_rows},
        {"reclaim_fadvise_calls", g_reclaim_fadvise_calls},
        {"reclaim_fadvise_errors", g_reclaim_fadvise_errors},
        {"reclaim_advised_bytes", g_reclaim_advised_bytes},
    };
    std::ofstream out(g_config.out_dir / "summary.json");
    if (out.good()) {
        out << summary.dump(2) << '\n';
    }

    std::ofstream report(g_config.out_dir / "REPORT.md");
    if (report.good()) {
        report << "# Live Oracle Window RPP\n\n";
        report << "- scheduler: `" << g_config.scheduler << "`\n";
        report << "- oracle samples: `" << g_oracle.size() << "`\n";
        report << "- RPP live model: `" << g_config.rpp_live_model_path << "`\n";
        report << "- window size: `" << g_config.window_size << "`\n";
        report << "- prefetch count threshold: `" << g_config.prefetch_threshold << "`\n";
        report << "- prefetch io mode: `" << g_config.prefetch_io_mode << "`\n";
        report << "- prefetch queue policy: `" << g_config.prefetch_queue_policy << "`\n";
        report << "- prefetch FIFO candidate order: `" << g_config.prefetch_fifo_candidate_order << "`\n";
        report << "- prefetch async: `" << g_config.async_prefetch << "`\n";
        report << "- prefetch async queue cap: `" << g_config.async_prefetch_queue_cap << "`\n";
        report << "- prefetch max staleness ubatches: `" << g_config.prefetch_max_staleness_ubatches << "`\n";
        report << "- ubatch prefetch max experts: `" << g_config.ubatch_prefetch_max_experts << "`\n";
        report << "- ubatch prefetch min count: `" << g_config.ubatch_prefetch_min_count << "`\n";
        report << "- ubatch prefetch density threshold: `" << g_config.ubatch_prefetch_density_threshold << "`\n";
        report << "- prefetch prefill only: `" << g_config.prefetch_prefill_only << "`\n";
        report << "- predict topk: `" << g_config.predict_topk << "`\n";
        report << "- RPP max seq len: `" << g_config.rpp_max_seq_len << "`\n";
        report << "- reclaim mode: `" << g_config.reclaim_mode << "`\n";
        report << "- reclaim lookahead ubatches: `" << g_config.reclaim_lookahead_ubatches << "`\n";
        report << "- cross ubatch layer prefetch: `" << g_config.cross_ubatch_layer_prefetch << "`\n";
        report << "- cross ubatch layer lookahead: `" << g_config.cross_ubatch_layer_lookahead << "`\n";
        report << "- layer frontier prefetch: `" << g_config.layer_frontier_prefetch << "`\n";
        report << "- layer frontier lookahead layers: `" << g_config.layer_frontier_lookahead_layers << "`\n";
        report << "- layer frontier jobs per tick: `" << g_config.layer_frontier_jobs_per_tick << "`\n";
        report << "- layer frontier max distance: `" << g_config.layer_frontier_max_distance << "`\n";
        report << "- prefetch stop decode call: `" << g_prefetch_stop_decode_call << "`\n";
        report << "- ubatch plan rows: `" << g_ubatch_plan_rows << "`\n";
        report << "- physical ubatch rows: `" << g_physical_ubatch_rows << "`\n";
        report << "- layer progress rows: `" << g_layer_progress_rows << "`\n";
        report << "- cross ubatch released rows: `" << g_cross_ubatch_released_rows << "`\n";
        report << "- layer frontier released rows: `" << g_layer_frontier_released_rows << "`\n";
        report << "- queue rows: `" << g_queue_rows << "`\n";
        report << "- expert counter rows: `" << g_counter_rows << "`\n";
        report << "- prefetch events: `" << g_prefetch_rows << "`\n";
        report << "- ubatch prefetch events: `" << g_ubatch_prefetch_rows << "`\n";
        report << "- ubatch counter rows: `" << g_ubatch_counter_rows << "`\n";
        report << "- ubatch prefetch lead rows: `" << g_ubatch_lead_rows << "`\n";
        report << "- reclaim candidates: `" << g_reclaim_rows << "`\n";
        report << "- oracle mismatches: `" << g_mismatch_rows << "`\n";
        report << "- token slot mismatches: `" << g_token_slot_mismatch_rows << "`\n";
        report << "- prefill token slot mismatches: `" << g_prefill_token_slot_mismatch_rows << "`\n";
        report << "- decode token slot mismatches: `" << g_decode_token_slot_mismatch_rows << "`\n";
        report << "- global order prediction mismatches: `" << g_order_prediction_mismatch_rows << "`\n";
        report << "- ubatch prediction mismatches: `" << g_ubatch_prediction_mismatch_rows << "`\n";
        report << "- fadvise calls: `" << g_fadvise_calls << "`\n";
        report << "- fadvise errors: `" << g_fadvise_errors << "`\n";
        report << "- advised bytes: `" << g_advised_bytes << "`\n";
        report << "- decode RPP prediction rows: `" << g_decode_rpp_prediction_rows << "`\n";
        report << "- decode RPP missing prediction rows: `" << g_decode_rpp_missing_prediction_rows << "`\n";
        report << "- RPP timing rows: `" << g_rpp_timing_rows << "`\n";
        report << "- async prefetch enqueued rows: `" << g_async_prefetch_enqueued_rows << "`\n";
        report << "- async prefetch completed rows: `" << g_async_prefetch_completed_rows << "`\n";
        report << "- async prefetch dropped rows: `" << g_async_prefetch_dropped_rows << "`\n";
        report << "- prefetch priority enqueued rows: `" << g_prefetch_priority_enqueued_rows << "`\n";
        report << "- prefetch priority completed rows: `" << g_prefetch_priority_completed_rows << "`\n";
        report << "- prefetch priority dropped low priority rows: `" << g_prefetch_priority_dropped_low_priority_rows << "`\n";
        report << "- prefetch priority replaced rows: `" << g_prefetch_priority_replaced_rows << "`\n";
        report << "- prefetch priority stale rows: `" << g_prefetch_priority_stale_rows << "`\n";
        report << "- reclaim priority candidates: `" << g_reclaim_priority_candidates << "`\n";
        report << "- reclaim priority trace rows: `" << g_reclaim_priority_trace_rows << "`\n";
        report << "- reclaim fadvise calls: `" << g_reclaim_fadvise_calls << "`\n";
        report << "- reclaim fadvise errors: `" << g_reclaim_fadvise_errors << "`\n";
        report << "- reclaim advised bytes: `" << g_reclaim_advised_bytes << "`\n";
        report << "\nArtifacts: `queue_trace.csv`, `expert_counter_trace.csv`, `prefetch_events.csv`, "
               << "`ubatch_counter_trace.csv`, `ubatch_prefetch_events.csv`, `ubatch_plan_trace.csv`, "
               << "`expert_use_plan_trace.csv`, `ubatch_decode_end_trace.csv`, "
               << "`physical_ubatch_trace.csv`, `layer_progress_trace.csv`, `cross_ubatch_release_trace.csv`, "
               << "`layer_frontier_release_trace.csv`, "
               << "`prefetch_priority_trace.csv`, `reclaim_priority_trace.csv`, `expert_state_trace.csv`, "
               << "`ubatch_decode_trace.csv`, `ubatch_prefetch_lead_trace.csv`, "
               << "`reclaim_candidates.csv`, `oracle_mismatch.csv`, `token_slot_mismatch.csv`, `actual_model_order_trace.csv`, "
               << "`predicted_global_order_trace.csv`, `global_order_prediction_mismatch.csv`, "
               << "`rpp_timing_trace.csv`, `summary.json`.\n";
    }
}

void load_oracle_trace() {
    if (g_config.oracle_trace_path.empty()) {
        return;
    }
    std::ifstream in(g_config.oracle_trace_path);
    if (!in.good()) {
        throw std::runtime_error("failed to open RPP oracle trace: " + g_config.oracle_trace_path);
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        json row = json::parse(line);
        oracle_token token;
        token.sample_id = row.at("sample_id").get<std::string>();
        token.phase = row.value("phase", std::string("decode"));
        token.token_index = row.value("token_index", token.phase == "prefill" ? row.value("prefill_index", 0) : row.value("decode_index", row.value("decode_pos", 0)));
        token.decode_index = token.phase == "decode" ? token.token_index : -1;
        if (row.contains("token_id") && !row["token_id"].is_null()) {
            token.token_id = row["token_id"].get<int32_t>();
        }
        std::set<expert_key> unique;
        for (const auto & item : row.at("experts")) {
            expert_key key { item.at("layer").get<int>(), item.at("expert").get<int>() };
            if (unique.insert(key).second) {
                token.experts.push_back(key);
            }
        }
        g_oracle[token.sample_id].push_back(std::move(token));
    }
    for (auto & it : g_oracle) {
        std::sort(it.second.begin(), it.second.end(), [](const oracle_token & a, const oracle_token & b) {
            if (a.phase != b.phase) {
                return a.phase < b.phase;
            }
            return a.token_index < b.token_index;
        });
    }
}

void load_rpp_live_model() {
    if (g_config.rpp_live_model_path.empty()) {
        return;
    }
#if defined(LLAMA_RPP_LIVE)
    if (g_config.predictor_threads > 0) {
        at::set_num_threads(g_config.predictor_threads);
        at::set_num_interop_threads(1);
    }
    g_rpp_module = std::make_unique<torch::jit::script::Module>(torch::jit::load(g_config.rpp_live_model_path, torch::kCPU));
    g_rpp_module->eval();
    LOG_INF("RPP live loaded TorchScript model: %s\n", g_config.rpp_live_model_path.c_str());
#else
    throw std::runtime_error("RPP live model requested but llama-server was built without LLAMA_RPP_LIVE");
#endif
}

void load_tensor_ranges() {
    if (g_config.tensor_ranges_path.empty()) {
        return;
    }
    std::ifstream in(g_config.tensor_ranges_path);
    if (!in.good()) {
        throw std::runtime_error("failed to open RPP tensor ranges: " + g_config.tensor_ranges_path);
    }
    std::string header_line;
    if (!std::getline(in, header_line)) {
        return;
    }
    const std::vector<std::string> header = split_csv_line(header_line);
    std::map<std::string, size_t> col;
    for (size_t i = 0; i < header.size(); ++i) {
        col[header[i]] = i;
    }
    auto get = [&](const std::vector<std::string> & row, const std::string & name) -> std::string {
        auto it = col.find(name);
        if (it == col.end() || it->second >= row.size()) {
            return "";
        }
        return row[it->second];
    };
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> row = split_csv_line(line);
        const std::string name = get(row, "name");
        const std::string layer_s = get(row, "layer");
        const std::string n_bytes_s = get(row, "n_bytes");
        if (layer_s.empty() || n_bytes_s.empty()) {
            continue;
        }
        std::string role;
        if (name.size() >= std::strlen(".ffn_down_exps.weight") &&
                name.compare(name.size() - std::strlen(".ffn_down_exps.weight"), std::strlen(".ffn_down_exps.weight"), ".ffn_down_exps.weight") == 0) {
            role = "down";
        } else if (name.size() >= std::strlen(".ffn_gate_up_exps.weight") &&
                name.compare(name.size() - std::strlen(".ffn_gate_up_exps.weight"), std::strlen(".ffn_gate_up_exps.weight"), ".ffn_gate_up_exps.weight") == 0) {
            role = "gate_up";
        } else {
            continue;
        }
        const int layer = std::stoi(layer_s);
        const int64_t n_bytes = std::stoll(n_bytes_s);
        const int64_t file_start = std::stoll(get(row, "file_start"));
        const int64_t expert_bytes = n_bytes / 128;
        for (int expert = 0; expert < 128; ++expert) {
            g_ranges[{ layer, expert }].push_back({
                layer,
                expert,
                role,
                file_start + expert * expert_bytes,
                file_start + (expert + 1) * expert_bytes,
            });
        }
    }
}

void load_once() {
    if (g_loaded) {
        return;
    }
    write_headers_once();
    load_oracle_trace();
    load_rpp_live_model();
    load_tensor_ranges();
#if defined(__unix__) || defined(__APPLE__)
    if (!g_config.model_path.empty()) {
        g_model_fd = open(g_config.model_path.c_str(), O_RDONLY);
        if (g_model_fd < 0) {
            LOG_WRN("RPP live failed to open model for fadvise: %s\n", std::strerror(errno));
        }
    }
#endif
    start_async_prefetch_worker_if_needed();
    g_loaded = true;
    write_summary();
}

bool ensure_loaded_or_warned() {
    if (!g_config.enabled) {
        return false;
    }
    try {
        load_once();
        return true;
    } catch (const std::exception & e) {
        if (!g_warned_unloaded) {
            LOG_WRN("RPP live oracle-window disabled after load failure: %s\n", e.what());
            g_warned_unloaded = true;
        }
        return false;
    }
}

void register_active_samples(const std::vector<rpp_live_slot_info> & slots) {
    for (const auto & slot : slots) {
        if (slot.oracle_sample_id.empty()) {
            continue;
        }
        if (g_oracle.find(slot.oracle_sample_id) == g_oracle.end()) {
            continue;
        }
        if (g_samples.find(slot.oracle_sample_id) == g_samples.end()) {
            const int32_t order = (int32_t) g_sample_order.size();
            g_samples[slot.oracle_sample_id] = { order, 0 };
            g_sample_order.push_back(slot.oracle_sample_id);
        }
    }
}

const oracle_token * find_oracle_token(
        const std::string & sample_id,
        const std::string & phase,
        int32_t token_index) {
    auto oracle_it = g_oracle.find(sample_id);
    if (oracle_it == g_oracle.end()) {
        return nullptr;
    }
    for (const oracle_token & token : oracle_it->second) {
        if (token.phase == phase && token.token_index == token_index) {
            return &token;
        }
    }
    return nullptr;
}

std::unordered_map<int32_t, rpp_live_slot_info> slot_by_seq(const std::vector<rpp_live_slot_info> & slots) {
    std::unordered_map<int32_t, rpp_live_slot_info> out;
    for (const auto & slot : slots) {
        out[slot.seq_id] = slot;
    }
    return out;
}

std::unordered_map<int32_t, rpp_live_slot_info> slot_by_slot_id(const std::vector<rpp_live_slot_info> & slots) {
    std::unordered_map<int32_t, rpp_live_slot_info> out;
    for (const auto & slot : slots) {
        out[slot.slot_id] = slot;
    }
    return out;
}

void sync_slot_decode_histories(const std::vector<rpp_live_slot_info> & slots) {
    std::set<int32_t> active_slots;
    for (const rpp_live_slot_info & slot : slots) {
        active_slots.insert(slot.slot_id);
        if (slot.oracle_sample_id.empty() || slot.slot_state == 0) {
            g_slot_sample_ids.erase(slot.slot_id);
            g_slot_decode_histories.erase(slot.slot_id);
            continue;
        }
        auto sample_it = g_slot_sample_ids.find(slot.slot_id);
        if (sample_it == g_slot_sample_ids.end() || sample_it->second != slot.oracle_sample_id) {
            g_slot_sample_ids[slot.slot_id] = slot.oracle_sample_id;
            g_slot_decode_histories[slot.slot_id].clear();
        }
    }

    for (auto it = g_slot_sample_ids.begin(); it != g_slot_sample_ids.end();) {
        if (active_slots.count(it->first) == 0) {
            g_slot_decode_histories.erase(it->first);
            it = g_slot_sample_ids.erase(it);
        } else {
            ++it;
        }
    }
}

void write_slot_state_trace(
        const std::vector<rpp_live_slot_info> & slots,
        int64_t decode_call_id) {
    for (size_t i = 0; i < slots.size(); ++i) {
        const rpp_live_slot_info & slot = slots[i];
        append_line(
            "slot_state_trace.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(i) + "," +
            std::to_string(slot.slot_id) + "," +
            std::to_string(slot.task_id) + "," +
            csv_escape(slot.oracle_sample_id) + "," +
            csv_escape(slot_state_name(slot.slot_state)) + "," +
            std::to_string(slot.n_decoded) + "," +
            std::to_string(slot.prompt_n_tokens) + "," +
            std::to_string(slot.task_n_tokens) + "," +
            std::to_string(slot.i_batch) + "," +
            std::to_string(slot.is_generating ? 1 : 0));
        ++g_slot_state_rows;
    }
}

void predict_decode_rpp_experts(
        std::vector<predicted_token> & actual_tokens,
        const std::unordered_map<int32_t, rpp_live_slot_info> & slots_by_id,
        int64_t decode_call_id) {
    const int64_t total_start_ts_us = monotonic_us();
    std::vector<size_t> decode_indices;
    decode_indices.reserve(actual_tokens.size());
    std::vector<std::vector<int32_t>> histories;
    histories.reserve(actual_tokens.size());
    std::vector<int32_t> history_lengths;
    history_lengths.reserve(actual_tokens.size());
    int32_t max_used_len = 0;

    for (size_t i = 0; i < actual_tokens.size(); ++i) {
        predicted_token & token = actual_tokens[i];
        if (token.phase != "decode") {
            continue;
        }
        auto slot_it = slots_by_id.find(token.slot_id);
        if (slot_it == slots_by_id.end()) {
            append_line(
                "decode_rpp_missing_prediction.csv",
                std::to_string(decode_call_id) + "," +
                std::to_string(token.slot_id) + "," +
                std::to_string(token.task_id) + "," +
                csv_escape(token.sample_id) + "," +
                std::to_string(token.decode_index) + "," +
                std::to_string(token.token_id) + ",missing_slot");
            ++g_decode_rpp_missing_prediction_rows;
            continue;
        }
        std::vector<int32_t> history = slot_it->second.prompt_token_ids;
        auto hist_it = g_slot_decode_histories.find(token.slot_id);
        if (hist_it != g_slot_decode_histories.end()) {
            history.insert(history.end(), hist_it->second.begin(), hist_it->second.end());
        }
        if (token.token_id >= 0) {
            history.push_back(token.token_id);
        }
        const int32_t history_len = (int32_t) history.size();
        if (history.empty()) {
            append_line(
                "decode_rpp_missing_prediction.csv",
                std::to_string(decode_call_id) + "," +
                std::to_string(token.slot_id) + "," +
                std::to_string(token.task_id) + "," +
                csv_escape(token.sample_id) + "," +
                std::to_string(token.decode_index) + "," +
                std::to_string(token.token_id) + ",empty_history");
            ++g_decode_rpp_missing_prediction_rows;
            continue;
        }
        const int32_t max_seq_len = std::max<int32_t>(1, g_config.rpp_max_seq_len);
        if ((int32_t) history.size() > max_seq_len) {
            history.erase(history.begin(), history.end() - max_seq_len);
        }
        max_used_len = std::max<int32_t>(max_used_len, (int32_t) history.size());
        decode_indices.push_back(i);
        history_lengths.push_back(history_len);
        histories.push_back(std::move(history));
    }
    const int64_t prepare_end_ts_us = monotonic_us();
    append_rpp_timing(
        decode_call_id,
        "decode_rpp_prepare",
        total_start_ts_us,
        prepare_end_ts_us,
        (int64_t) decode_indices.size(),
        (int64_t) actual_tokens.size(),
        "max_used_len=" + std::to_string(max_used_len));

    if (decode_indices.empty()) {
        append_rpp_timing(
            decode_call_id,
            "decode_rpp_total",
            total_start_ts_us,
            monotonic_us(),
            0,
            (int64_t) actual_tokens.size(),
            "empty_decode_indices");
        return;
    }

#if defined(LLAMA_RPP_LIVE)
    if (!g_rpp_module) {
        for (const size_t idx : decode_indices) {
            const predicted_token & token = actual_tokens[idx];
            append_line(
                "decode_rpp_missing_prediction.csv",
                std::to_string(decode_call_id) + "," +
                std::to_string(token.slot_id) + "," +
                std::to_string(token.task_id) + "," +
                csv_escape(token.sample_id) + "," +
                std::to_string(token.decode_index) + "," +
                std::to_string(token.token_id) + ",model_not_loaded");
            ++g_decode_rpp_missing_prediction_rows;
        }
        return;
    }

    try {
        torch::NoGradGuard no_grad;
        const int64_t tensor_start_ts_us = monotonic_us();
        const int64_t bsz = (int64_t) histories.size();
        const int64_t seq_len = std::max<int32_t>(1, max_used_len);
        auto ids = torch::zeros({bsz, seq_len}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto mask = torch::zeros({bsz, seq_len}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
        auto lengths = torch::zeros({bsz}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto ids_a = ids.accessor<int64_t, 2>();
        auto mask_a = mask.accessor<bool, 2>();
        auto lengths_a = lengths.accessor<int64_t, 1>();
        for (int64_t b = 0; b < bsz; ++b) {
            lengths_a[b] = (int64_t) histories[(size_t) b].size();
            for (int64_t s = 0; s < lengths_a[b]; ++s) {
                ids_a[b][s] = (int64_t) histories[(size_t) b][(size_t) s];
                mask_a[b][s] = true;
            }
        }
        std::vector<torch::jit::IValue> inputs;
        inputs.reserve(3);
        inputs.emplace_back(ids);
        inputs.emplace_back(mask);
        inputs.emplace_back(lengths);
        const int64_t tensor_end_ts_us = monotonic_us();
        append_rpp_timing(
            decode_call_id,
            "decode_rpp_tensor_fill",
            tensor_start_ts_us,
            tensor_end_ts_us,
            bsz,
            seq_len,
            "bsz=" + std::to_string(bsz));
        const int64_t forward_start_ts_us = monotonic_us();
        torch::Tensor logits = g_rpp_module->forward(inputs).toTensor().to(torch::kCPU);
        const int64_t forward_end_ts_us = monotonic_us();
        append_rpp_timing(
            decode_call_id,
            "decode_rpp_forward",
            forward_start_ts_us,
            forward_end_ts_us,
            bsz,
            seq_len,
            "");
        if (logits.dim() != 3 || logits.size(0) != bsz || logits.size(1) < 1 || logits.size(2) < 1) {
            throw std::runtime_error("unexpected RPP TorchScript output shape");
        }
        const int64_t n_layers = logits.size(1);
        const int64_t n_experts = logits.size(2);
        const int64_t topk = std::min<int64_t>(std::max<int32_t>(1, g_config.predict_topk), n_experts);
        const int64_t topk_start_ts_us = monotonic_us();
        for (int64_t b = 0; b < bsz; ++b) {
            predicted_token & token = actual_tokens[decode_indices[(size_t) b]];
            std::set<expert_key> unique;
            for (int64_t layer = 0; layer < n_layers; ++layer) {
                torch::Tensor layer_logits = logits[b][layer];
                torch::Tensor indices = std::get<1>(torch::topk(layer_logits, topk, -1, true, true)).to(torch::kCPU);
                auto idx_a = indices.accessor<int64_t, 1>();
                for (int64_t k = 0; k < topk; ++k) {
                    unique.insert({(int) layer, (int) idx_a[k]});
                }
            }
            token.experts.assign(unique.begin(), unique.end());
            append_line(
                "decode_rpp_prediction_trace.csv",
                std::to_string(decode_call_id) + "," +
                std::to_string(token.slot_id) + "," +
                std::to_string(token.task_id) + "," +
                csv_escape(token.sample_id) + "," +
                std::to_string(token.decode_index) + "," +
                std::to_string(token.token_id) + "," +
                std::to_string(history_lengths[(size_t) b]) + "," +
                std::to_string(histories[(size_t) b].size()) + "," +
                std::to_string(token.experts.size()));
            ++g_decode_rpp_prediction_rows;
        }
        const int64_t topk_end_ts_us = monotonic_us();
        append_rpp_timing(
            decode_call_id,
            "decode_rpp_topk",
            topk_start_ts_us,
            topk_end_ts_us,
            bsz,
            n_layers * topk,
            "n_layers=" + std::to_string(n_layers) + ";topk=" + std::to_string(topk));
    } catch (const std::exception & e) {
        for (const size_t idx : decode_indices) {
            const predicted_token & token = actual_tokens[idx];
            append_line(
                "decode_rpp_missing_prediction.csv",
                std::to_string(decode_call_id) + "," +
                std::to_string(token.slot_id) + "," +
                std::to_string(token.task_id) + "," +
                csv_escape(token.sample_id) + "," +
                std::to_string(token.decode_index) + "," +
                std::to_string(token.token_id) + "," +
                csv_escape(std::string("forward_error:") + e.what()));
            ++g_decode_rpp_missing_prediction_rows;
        }
        append_rpp_timing(
            decode_call_id,
            "decode_rpp_error",
            total_start_ts_us,
            monotonic_us(),
            (int64_t) decode_indices.size(),
            (int64_t) actual_tokens.size(),
            e.what());
    }
    append_rpp_timing(
        decode_call_id,
        "decode_rpp_total",
        total_start_ts_us,
        monotonic_us(),
        (int64_t) decode_indices.size(),
        (int64_t) actual_tokens.size(),
        "");
#else
    for (const size_t idx : decode_indices) {
        const predicted_token & token = actual_tokens[idx];
        append_line(
            "decode_rpp_missing_prediction.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(token.slot_id) + "," +
            std::to_string(token.task_id) + "," +
            csv_escape(token.sample_id) + "," +
            std::to_string(token.decode_index) + "," +
        std::to_string(token.token_id) + ",not_built_with_libtorch");
        ++g_decode_rpp_missing_prediction_rows;
    }
    append_rpp_timing(
        decode_call_id,
        "decode_rpp_total",
        total_start_ts_us,
        monotonic_us(),
        (int64_t) decode_indices.size(),
        (int64_t) actual_tokens.size(),
        "not_built_with_libtorch");
#endif
}

void commit_decode_histories(const std::vector<predicted_token> & actual_tokens) {
    for (const predicted_token & token : actual_tokens) {
        if (token.phase != "decode" || token.token_id < 0) {
            continue;
        }
        g_slot_decode_histories[token.slot_id].push_back(token.token_id);
    }
}

void write_model_order_trace(
        const llama_batch & batch,
        const std::unordered_map<int32_t, rpp_live_slot_info> & by_seq,
        int32_t logical_batch_offset,
        int32_t runtime_n_ubatch,
        int64_t decode_call_id) {
    const int32_t ubatch = std::max<int32_t>(1, runtime_n_ubatch);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int32_t logical_index = logical_batch_offset + i;
        const int32_t n_seq = batch.n_seq_id ? batch.n_seq_id[i] : 1;
        for (int32_t s = 0; s < n_seq; ++s) {
            const int32_t seq_id = batch.seq_id && batch.seq_id[i] ? batch.seq_id[i][s] : 0;
            auto slot_it = by_seq.find(seq_id);
            const rpp_live_slot_info * slot = slot_it == by_seq.end() ? nullptr : &slot_it->second;
            append_line(
                "actual_model_order_trace.csv",
                std::to_string(decode_call_id) + "," +
                std::to_string(i) + "," +
                std::to_string(logical_index) + "," +
                std::to_string(logical_index / ubatch) + "," +
                std::to_string(logical_index % ubatch) + "," +
                std::to_string(ubatch) + "," +
                std::to_string(seq_id) + "," +
                std::to_string(slot ? slot->slot_id : -1) + "," +
                std::to_string(slot ? slot->task_id : -1) + "," +
                csv_escape(slot ? slot->oracle_sample_id : "") + "," +
                csv_escape(slot ? slot_state_name(slot->slot_state) : "UNKNOWN") + "," +
                std::to_string(slot ? slot->n_decoded : -1) + "," +
                std::to_string(slot ? slot->prompt_n_tokens : -1) + "," +
                std::to_string(slot ? slot->task_n_tokens : -1) + "," +
                std::to_string(slot ? (slot->is_generating ? 1 : 0) : 0) + "," +
                std::to_string(batch.pos ? batch.pos[i] : -1) + "," +
                std::to_string(batch.token ? batch.token[i] : -1) + "," +
                std::to_string(batch.logits ? (batch.logits[i] != 0 ? 1 : 0) : 0));
            ++g_model_order_rows;
        }
    }
}

std::vector<predicted_token> advance_current_tokens(
        const llama_batch & batch,
        const std::unordered_map<int32_t, rpp_live_slot_info> & by_seq,
        int32_t logical_batch_offset,
        int32_t runtime_n_ubatch,
        int64_t decode_call_id) {
    std::vector<predicted_token> actual_inputs;
    const int32_t ubatch = std::max<int32_t>(1, runtime_n_ubatch);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int32_t logical_index = logical_batch_offset + i;
        const int32_t n_seq = batch.n_seq_id ? batch.n_seq_id[i] : 1;
        for (int32_t s = 0; s < n_seq; ++s) {
            const int32_t seq_id = batch.seq_id && batch.seq_id[i] ? batch.seq_id[i][s] : 0;
            auto slot_it = by_seq.find(seq_id);
            if (slot_it == by_seq.end()) {
                continue;
            }
            const rpp_live_slot_info & slot = slot_it->second;
            if (slot.oracle_sample_id.empty()) {
                continue;
            }
            predicted_token actual;
            actual.sample_id = slot.oracle_sample_id;
            actual.slot_id = slot.slot_id;
            actual.task_id = slot.task_id;
            actual.phase = slot.is_generating ? "decode" : "prefill";
            actual.token_index = slot.is_generating ? slot.n_decoded - 1 : (batch.pos ? batch.pos[i] : -1);
            actual.decode_index = actual.phase == "decode" ? actual.token_index : -1;
            actual.token_id = batch.token ? batch.token[i] : -1;
            actual.batch_index = i;
            actual.ubatch_id = logical_index / ubatch;
            actual.ubatch_offset = logical_index % ubatch;
            actual_inputs.push_back(actual);

            const oracle_token * oracle = find_oracle_token(actual.sample_id, actual.phase, actual.token_index);
            if (oracle != nullptr && oracle->token_id >= 0 && actual.token_id != oracle->token_id) {
                append_line(
                    "token_slot_mismatch.csv",
                    std::to_string(decode_call_id) + "," +
                    std::to_string(actual.slot_id) + "," +
                    csv_escape(actual.sample_id) + "," +
                    csv_escape(actual.phase) + "," +
                    std::to_string(actual.token_index) + "," +
                    std::to_string(oracle->token_id) + "," +
                    std::to_string(actual.token_id));
                ++g_token_slot_mismatch_rows;
                if (actual.phase == "prefill") {
                    ++g_prefill_token_slot_mismatch_rows;
                } else if (actual.phase == "decode") {
                    ++g_decode_token_slot_mismatch_rows;
                }
            }

            if (!slot.is_generating || slot.n_decoded <= 0) {
                continue;
            }
            auto sample_it = g_samples.find(slot.oracle_sample_id);
            auto oracle_it = g_oracle.find(slot.oracle_sample_id);
            if (sample_it == g_samples.end() || oracle_it == g_oracle.end()) {
                continue;
            }
            sample_state & state = sample_it->second;
            const size_t expected_idx = (size_t) (slot.n_decoded - 1);
            const oracle_token * expected = find_oracle_token(slot.oracle_sample_id, "decode", (int32_t) expected_idx);
            if (expected == nullptr) {
                continue;
            }
            const int32_t actual_token = batch.token ? batch.token[i] : -1;
            if (expected->token_id >= 0 && actual_token != expected->token_id) {
                append_line(
                    "oracle_mismatch.csv",
                    std::to_string(decode_call_id) + "," +
                    csv_escape(slot.oracle_sample_id) + "," +
                    std::to_string(expected->decode_index) + "," +
                    std::to_string(expected->token_id) + "," +
                    std::to_string(actual_token) + "," +
                    std::to_string(slot.slot_id));
                ++g_mismatch_rows;
            }
            state.cursor = std::max(state.cursor, expected_idx + 1);
            break;
        }
    }
    return actual_inputs;
}

void compare_predicted_to_actual(
        const std::vector<predicted_token> & actual,
        int64_t decode_call_id) {
    if (g_pending_predicted_order.empty()) {
        return;
    }
    const size_t n_compare = actual.size();
    for (size_t i = 0; i < n_compare; ++i) {
        const predicted_token * expected = i < g_pending_predicted_order.size() ? &g_pending_predicted_order[i] : nullptr;
        std::string reason;
        if (expected == nullptr) {
            reason = "missing_prediction";
        } else if (expected->slot_id != actual[i].slot_id) {
            reason = "slot_id";
        } else if (expected->sample_id != actual[i].sample_id) {
            reason = "sample_id";
        } else if (expected->phase != actual[i].phase) {
            reason = "phase";
        } else if (expected->token_index != actual[i].token_index) {
            reason = "token_index";
        } else if (expected->decode_index != actual[i].decode_index) {
            reason = "decode_index";
        } else if (expected->ubatch_id != actual[i].ubatch_id) {
            reason = "ubatch_id";
            ++g_ubatch_prediction_mismatch_rows;
        } else if (expected->ubatch_offset != actual[i].ubatch_offset) {
            reason = "ubatch_offset";
            ++g_ubatch_prediction_mismatch_rows;
        }
        if (reason.empty()) {
            continue;
        }
        append_line(
            "global_order_prediction_mismatch.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(i) + "," +
            std::to_string(expected ? expected->slot_id : -1) + "," +
            std::to_string(actual[i].slot_id) + "," +
            csv_escape(expected ? expected->sample_id : "") + "," +
            csv_escape(actual[i].sample_id) + "," +
            csv_escape(expected ? expected->phase : "") + "," +
            csv_escape(actual[i].phase) + "," +
            std::to_string(expected ? expected->token_index : -1) + "," +
            std::to_string(actual[i].token_index) + "," +
            std::to_string(expected ? expected->decode_index : -1) + "," +
            std::to_string(actual[i].decode_index) + "," +
            std::to_string(expected ? expected->ubatch_id : -1) + "," +
            std::to_string(actual[i].ubatch_id) + "," +
            std::to_string(expected ? expected->ubatch_offset : -1) + "," +
            std::to_string(actual[i].ubatch_offset) + "," +
            csv_escape(reason));
        ++g_order_prediction_mismatch_rows;
    }
}

std::vector<predicted_token> predict_global_decode_order(
        const llama_batch & batch,
        const std::vector<rpp_live_slot_info> & slots,
        int32_t runtime_n_batch,
        int32_t runtime_n_ubatch,
        int32_t max_tokens,
        int32_t max_steps) {
    struct sim_slot {
        int32_t slot_id = -1;
        int32_t task_id = -1;
        int32_t state = 0;
        int32_t prompt_n_tokens = 0;
        int32_t task_n_tokens = 0;
        std::string sample_id;
        std::vector<int32_t> prompt_token_ids;
        size_t cursor = 0;
    };

    std::set<int32_t> output_slots;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits || batch.logits[i] == 0) {
            continue;
        }
        const int32_t n_seq = batch.n_seq_id ? batch.n_seq_id[i] : 1;
        for (int32_t s = 0; s < n_seq; ++s) {
            const int32_t seq_id = batch.seq_id && batch.seq_id[i] ? batch.seq_id[i][s] : -1;
            output_slots.insert(seq_id);
        }
    }

    std::vector<sim_slot> sim;
    sim.reserve(slots.size());
    for (const rpp_live_slot_info & slot : slots) {
        sim_slot cur;
        cur.slot_id = slot.slot_id;
        cur.task_id = slot.task_id;
        cur.state = slot.slot_state;
        cur.prompt_n_tokens = slot.prompt_n_tokens;
        cur.task_n_tokens = slot.task_n_tokens;
        cur.sample_id = slot.oracle_sample_id;
        cur.prompt_token_ids = slot.prompt_token_ids;

        auto sample_it = g_samples.find(cur.sample_id);
        if (sample_it != g_samples.end()) {
            cur.cursor = sample_it->second.cursor;
        }

        // This hook runs before llama_decode(), but the next prediction window is
        // for tokens that enter the model after the current batch view completes.
        if (output_slots.count(slot.seq_id) > 0 && (cur.state == 4 || cur.state == 5)) {
            cur.state = 5;
        }
        sim.push_back(std::move(cur));
    }

    std::vector<predicted_token> predicted;
    const int32_t n_batch = std::max<int32_t>(1, runtime_n_batch);
    const int32_t n_ubatch = std::max<int32_t>(1, runtime_n_ubatch);
    const int32_t token_limit = std::max<int32_t>(1, max_tokens);
    const int32_t step_limit = std::max<int32_t>(1, max_steps);
    for (int guard = 0; guard < step_limit && (int) predicted.size() < token_limit; ++guard) {
        int32_t batch_tokens = 0;
        std::vector<size_t> done_prompt_slots;

        for (sim_slot & slot : sim) {
            if (slot.state != 5 || slot.sample_id.empty()) {
                continue;
            }
            const oracle_token * oracle = find_oracle_token(slot.sample_id, "decode", (int32_t) slot.cursor);
            if (oracle == nullptr) {
                continue;
            }
            const int32_t logical_index = batch_tokens;
            predicted.push_back({
                slot.sample_id,
                slot.slot_id,
                slot.task_id,
                "decode",
                oracle->token_index,
                oracle->decode_index,
                oracle->token_id,
                logical_index,
                logical_index / n_ubatch,
                logical_index % n_ubatch,
                {},
            });
            ++slot.cursor;
            ++batch_tokens;
            if ((int) predicted.size() >= token_limit) {
                break;
            }
        }
        if ((int) predicted.size() >= token_limit) {
            break;
        }

        if (g_config.cont_batching || batch_tokens == 0) {
            for (size_t i = 0; i < sim.size() && batch_tokens < n_batch; ++i) {
                sim_slot & slot = sim[i];
                if (slot.state != 2 && slot.state != 3) {
                    continue;
                }
                if (slot.task_n_tokens <= 0 || slot.prompt_n_tokens >= slot.task_n_tokens) {
                    continue;
                }
                slot.state = 3;
                while (slot.prompt_n_tokens < slot.task_n_tokens && batch_tokens < n_batch) {
                    const int32_t token_index = slot.prompt_n_tokens;
                    const int32_t token_id =
                        token_index >= 0 && (size_t) token_index < slot.prompt_token_ids.size() ?
                        slot.prompt_token_ids[(size_t) token_index] : -1;
                    const int32_t logical_index = batch_tokens;
                    predicted.push_back({
                        slot.sample_id,
                        slot.slot_id,
                        slot.task_id,
                        "prefill",
                        token_index,
                        -1,
                        token_id,
                        logical_index,
                        logical_index / n_ubatch,
                        logical_index % n_ubatch,
                        {},
                    });
                    ++slot.prompt_n_tokens;
                    ++batch_tokens;
                    if ((int) predicted.size() >= token_limit) {
                        break;
                    }
                    if (should_break_for_prompt_checkpoint(slot.task_n_tokens, slot.prompt_n_tokens, n_batch, n_ubatch)) {
                        break;
                    }
                }
                if (slot.prompt_n_tokens == slot.task_n_tokens) {
                    done_prompt_slots.push_back(i);
                }
                if ((int) predicted.size() >= token_limit) {
                    break;
                }
            }
        }

        for (const size_t idx : done_prompt_slots) {
            sim[idx].state = 5;
        }

        if (batch_tokens == 0) {
            break;
        }
    }
    return predicted;
}

void write_predicted_order_trace(
        const std::vector<predicted_token> & predicted,
        int64_t decode_call_id) {
    for (size_t i = 0; i < predicted.size(); ++i) {
        const predicted_token & row = predicted[i];
        append_line(
            "predicted_global_order_trace.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(i) + "," +
            std::to_string(row.slot_id) + "," +
            std::to_string(row.task_id) + "," +
            csv_escape(row.sample_id) + "," +
            csv_escape(row.phase) + "," +
            std::to_string(row.token_index) + "," +
            std::to_string(row.decode_index) + "," +
            std::to_string(row.token_id) + "," +
            std::to_string(row.ubatch_id) + "," +
            std::to_string(row.ubatch_offset));
        ++g_predicted_order_rows;
    }
}

void write_predicted_model_order_trace(
        const std::vector<predicted_token> & predicted,
        int64_t decode_call_id) {
    for (size_t i = 0; i < predicted.size(); ++i) {
        const predicted_token & row = predicted[i];
        append_line(
            "predicted_model_order_trace.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(i) + "," +
            std::to_string(row.slot_id) + "," +
            std::to_string(row.task_id) + "," +
            csv_escape(row.sample_id) + "," +
            csv_escape(row.phase) + "," +
            std::to_string(row.token_index) + "," +
            std::to_string(row.decode_index) + "," +
            std::to_string(row.token_id) + "," +
            std::to_string(row.ubatch_id) + "," +
            std::to_string(row.ubatch_offset));
        ++g_predicted_model_order_rows;
    }
}

void rebuild_queue_from_prediction(const std::vector<predicted_token> & predicted) {
    g_queue.clear();
    for (const predicted_token & pred : predicted) {
        const oracle_token * token = find_oracle_token(pred.sample_id, pred.phase, pred.token_index);
        if (token == nullptr) {
            continue;
        }
        int32_t sample_order = -1;
        auto sample_it = g_samples.find(pred.sample_id);
        if (sample_it != g_samples.end()) {
            sample_order = sample_it->second.order;
        }
        g_queue.push_back({
            pred.sample_id,
            sample_order,
            pred.slot_id,
            pred.phase,
            pred.token_index,
            token->decode_index,
            token->token_id,
            token->experts,
        });
        if ((int) g_queue.size() >= g_config.window_size) {
            break;
        }
    }
}

std::vector<std::pair<expert_key, int>> recompute_counters() {
    g_counters.clear();
    for (const queued_token & token : g_queue) {
        for (const expert_key & key : token.experts) {
            g_counters[key] += 1;
        }
    }
    return sorted_prefetch_candidates(g_counters);
}

void write_queue_and_counter_trace(int64_t decode_call_id) {
    for (size_t i = 0; i < g_queue.size(); ++i) {
        const queued_token & row = g_queue[i];
        append_line(
            "queue_trace.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(i) + "," +
            csv_escape(row.sample_id) + "," +
            std::to_string(row.sample_order) + "," +
            std::to_string(row.slot_id) + "," +
            csv_escape(row.phase) + "," +
            std::to_string(row.token_index) + "," +
            std::to_string(row.decode_index) + "," +
            std::to_string(row.token_id) + "," +
            std::to_string(row.experts.size()));
        ++g_queue_rows;
    }
    for (const auto & it : g_counters) {
        append_line(
            "expert_counter_trace.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(it.first.layer) + "," +
            std::to_string(it.first.expert) + "," +
            std::to_string(it.second));
        ++g_counter_rows;
    }
}

std::tuple<int64_t, int64_t, int64_t> prefetch_one(const expert_key & key) {
    int64_t calls = 0;
    int64_t errors = 0;
    int64_t bytes = 0;
#if defined(__unix__) || defined(__APPLE__)
    if (g_model_fd < 0) {
        return { calls, errors, bytes };
    }
    auto it = g_ranges.find({ key.layer, key.expert });
    if (it == g_ranges.end()) {
        return { calls, errors, bytes };
    }
    for (const expert_range & range : it->second) {
        const int64_t len = range.file_end - range.file_start;
        int rc = 0;
        if (len <= 0) {
            continue;
        }
        if (g_config.prefetch_io_mode == "mmap") {
            const long page_size = sysconf(_SC_PAGESIZE);
            const int64_t page = page_size > 0 ? (int64_t) page_size : 4096;
            const int64_t map_start = (range.file_start / page) * page;
            const int64_t delta = range.file_start - map_start;
            const size_t map_len = (size_t) (delta + len);
            void * mapped = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, g_model_fd, (off_t) map_start);
            if (mapped == MAP_FAILED) {
                rc = -1;
            } else {
                volatile unsigned char sink = 0;
                const unsigned char * base = static_cast<const unsigned char *>(mapped);
                const int64_t touch_begin = (delta / page) * page;
                const int64_t touch_end = delta + len;
                for (int64_t off = touch_begin; off < touch_end; off += page) {
                    sink ^= base[off];
                }
                (void) sink;
                if (munmap(mapped, map_len) != 0) {
                    rc = -1;
                }
            }
        } else if (g_config.prefetch_io_mode == "readahead") {
#if defined(__linux__)
            rc = readahead(g_model_fd, (off64_t) range.file_start, (size_t) len);
#else
            rc = -1;
            errno = ENOSYS;
#endif
        } else {
            rc = posix_fadvise(g_model_fd, range.file_start, len, POSIX_FADV_WILLNEED);
        }
        ++calls;
        bytes += len;
        if (rc != 0) {
            ++errors;
        }
    }
#endif
    return { calls, errors, bytes };
}

std::tuple<int64_t, int64_t, int64_t> reclaim_one(const expert_key & key) {
    int64_t calls = 0;
    int64_t errors = 0;
    int64_t bytes = 0;
#if defined(__unix__) || defined(__APPLE__)
    if (g_model_fd < 0) {
        return { calls, errors, bytes };
    }
    auto it = g_ranges.find({ key.layer, key.expert });
    if (it == g_ranges.end()) {
        return { calls, errors, bytes };
    }
    for (const expert_range & range : it->second) {
        const int64_t len = range.file_end - range.file_start;
        if (len <= 0) {
            continue;
        }
        const int rc = posix_fadvise(g_model_fd, range.file_start, len, POSIX_FADV_DONTNEED);
        ++calls;
        bytes += len;
        if (rc != 0) {
            ++errors;
        }
    }
#endif
    return { calls, errors, bytes };
}

int64_t expert_resident_bytes(const expert_key & key) {
    int64_t bytes = 0;
    auto it = g_ranges.find({ key.layer, key.expert });
    if (it == g_ranges.end()) {
        return bytes;
    }
    for (const expert_range & range : it->second) {
        bytes += std::max<int64_t>(0, range.file_end - range.file_start);
    }
    return bytes;
}

bool ubatch_point_leq(int64_t a_decode, int32_t a_ubatch, int64_t b_decode, int32_t b_ubatch) {
    return std::tie(a_decode, a_ubatch) <= std::tie(b_decode, b_ubatch);
}

bool prefetch_priority_higher(const async_prefetch_job & a, const async_prefetch_job & b) {
    if (a.counter != b.counter) {
        return a.counter > b.counter;
    }
    if (a.density != b.density) {
        return a.density > b.density;
    }
    if (a.target_token_count != b.target_token_count) {
        return a.target_token_count > b.target_token_count;
    }
    if (a.target_ubatch_distance != b.target_ubatch_distance) {
        return a.target_ubatch_distance < b.target_ubatch_distance;
    }
    if (a.source_decode_call_id != b.source_decode_call_id) {
        return a.source_decode_call_id > b.source_decode_call_id;
    }
    if (a.key.expert != b.key.expert) {
        return a.key.expert < b.key.expert;
    }
    if (a.key.layer != b.key.layer) {
        return a.key.layer < b.key.layer;
    }
    return false;
}

bool prefetch_queue_is_priority() {
    return g_config.prefetch_queue_policy == "priority";
}

size_t highest_prefetch_priority_index(const std::vector<async_prefetch_job> & queue) {
    size_t best = 0;
    for (size_t i = 1; i < queue.size(); ++i) {
        if (prefetch_priority_higher(queue[i], queue[best])) {
            best = i;
        }
    }
    return best;
}

size_t lowest_prefetch_priority_index(const std::vector<async_prefetch_job> & queue) {
    size_t worst = 0;
    for (size_t i = 1; i < queue.size(); ++i) {
        if (prefetch_priority_higher(queue[worst], queue[i])) {
            worst = i;
        }
    }
    return worst;
}

void append_prefetch_priority_trace(
        const async_prefetch_job & job,
        const std::string & status,
        int64_t start_ts_us,
        int64_t end_ts_us,
        int64_t calls,
        int64_t errors,
        int64_t bytes,
        int32_t queue_size) {
    append_line(
        "prefetch_priority_trace.csv",
        csv_escape(status) + "," +
        std::to_string(job.source_decode_call_id) + "," +
        std::to_string(job.source_ubatch_id) + "," +
        std::to_string(job.target_decode_call_id) + "," +
        std::to_string(job.target_ubatch_id) + "," +
        std::to_string(job.target_first_batch_order) + "," +
        std::to_string(job.target_token_count) + "," +
        std::to_string(job.target_ubatch_distance) + "," +
        std::to_string(job.key.layer) + "," +
        std::to_string(job.key.expert) + "," +
        std::to_string(job.counter) + "," +
        std::to_string(job.density) + "," +
        std::to_string(job.required_count) + "," +
        std::to_string(queue_size) + "," +
        std::to_string(job.enqueue_ts_us) + "," +
        std::to_string(start_ts_us) + "," +
        std::to_string(end_ts_us) + "," +
        std::to_string(calls) + "," +
        std::to_string(errors) + "," +
        std::to_string(bytes));
}

void append_async_prefetch_trace(
        const async_prefetch_job & job,
        int64_t start_ts_us,
        int64_t end_ts_us,
        int64_t calls,
        int64_t errors,
        int64_t bytes,
        const std::string & status) {
    append_line(
        "ubatch_prefetch_async_trace.csv",
        std::to_string(job.source_decode_call_id) + "," +
        std::to_string(job.source_ubatch_id) + "," +
        std::to_string(job.target_decode_call_id) + "," +
        std::to_string(job.target_ubatch_id) + "," +
        std::to_string(job.target_first_batch_order) + "," +
        std::to_string(job.target_token_count) + "," +
        std::to_string(job.key.layer) + "," +
        std::to_string(job.key.expert) + "," +
        std::to_string(job.counter) + "," +
        std::to_string(job.density) + "," +
        std::to_string(job.required_count) + "," +
        std::to_string(job.enqueue_ts_us) + "," +
        std::to_string(start_ts_us) + "," +
        std::to_string(end_ts_us) + "," +
        std::to_string(start_ts_us - job.enqueue_ts_us) + "," +
        std::to_string(end_ts_us - start_ts_us) + "," +
        std::to_string(calls) + "," +
        std::to_string(errors) + "," +
        std::to_string(bytes) + "," +
        csv_escape(status));
}

void async_prefetch_worker_main() {
    for (;;) {
        async_prefetch_job job;
        {
            std::unique_lock<std::mutex> lock(g_async_mutex);
            g_async_cv.wait(lock, [] {
                return g_async_stop || !g_async_queue.empty();
            });
            if (g_async_stop && g_async_queue.empty()) {
                return;
            }
            const size_t next = prefetch_queue_is_priority() ? highest_prefetch_priority_index(g_async_queue) : 0;
            job = g_async_queue[next];
            g_async_queue.erase(g_async_queue.begin() + (ptrdiff_t) next);
        }

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            expert_runtime_state & state = g_expert_states[job.key];
            state.inflight = true;
            const bool stale =
                g_latest_source_decode_call_id > job.target_decode_call_id + g_config.prefetch_max_staleness_ubatches;
            const bool layer_stale =
                g_config.layer_frontier_prefetch != 0 &&
                job.target_decode_call_id == g_eval_decode_call_id &&
                (job.target_ubatch_id < g_eval_current_ubatch ||
                 (job.target_ubatch_id == g_eval_current_ubatch && job.key.layer <= g_eval_last_layer));
            if (stale || layer_stale) {
                state.inflight = false;
                ++g_prefetch_priority_stale_rows;
                const char * status = layer_stale ? "dropped_layer_stale" : "dropped_stale";
                append_prefetch_priority_trace(job, status, 0, 0, 0, 0, 0, -1);
                append_async_prefetch_trace(job, 0, 0, 0, 0, 0, status);
                continue;
            }
        }

        const int64_t start_ts_us = monotonic_us();
        int64_t calls = 0;
        int64_t errors = 0;
        int64_t bytes = 0;
        std::tie(calls, errors, bytes) = prefetch_one(job.key);
        const int64_t end_ts_us = monotonic_us();

        std::lock_guard<std::mutex> lock(g_mutex);
        g_fadvise_calls += calls;
        g_fadvise_errors += errors;
        g_advised_bytes += bytes;
        ++g_async_prefetch_completed_rows;
        ++g_prefetch_priority_completed_rows;
        expert_runtime_state & state = g_expert_states[job.key];
        state.inflight = false;
        state.last_prefetched_decode_call_id = job.target_decode_call_id;
        state.last_prefetched_ubatch_id = job.target_ubatch_id;
        state.protected_until_decode_call_id = job.target_decode_call_id;
        state.protected_until_ubatch_id = job.target_ubatch_id + g_config.reclaim_protect_ubatches;
        append_async_prefetch_trace(job, start_ts_us, end_ts_us, calls, errors, bytes, "completed");
        append_prefetch_priority_trace(job, "completed", start_ts_us, end_ts_us, calls, errors, bytes, -1);
        append_line(
            "ubatch_prefetch_lead_trace.csv",
            std::to_string(job.source_decode_call_id) + "," +
            std::to_string(job.target_decode_call_id) + "," +
            std::to_string(job.source_ubatch_id) + "," +
            std::to_string(job.target_ubatch_id) + "," +
            std::to_string(job.target_first_batch_order) + "," +
            std::to_string(job.target_token_count) + "," +
            std::to_string(job.enqueue_ts_us) + "," +
            std::to_string(end_ts_us) + "," +
            std::to_string(end_ts_us - job.enqueue_ts_us) + "," +
            std::to_string(calls) + "," +
            std::to_string(errors) + "," +
            std::to_string(bytes));
        ++g_ubatch_lead_rows;
        if (g_async_prefetch_completed_rows % 128 == 0) {
            write_summary();
        }
    }
}

void stop_async_prefetch_worker() {
    {
        std::lock_guard<std::mutex> lock(g_async_mutex);
        g_async_stop = true;
    }
    g_async_cv.notify_all();
    if (g_async_worker.joinable()) {
        g_async_worker.join();
    }
    g_async_worker_running = false;
}

void rpp_live_shutdown_atexit() {
    stop_async_prefetch_worker();
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_loaded) {
        write_summary();
    }
}

void start_async_prefetch_worker_if_needed() {
    if (!g_config.async_prefetch || g_async_worker_running) {
        return;
    }
    g_async_stop = false;
    g_async_worker = std::thread(async_prefetch_worker_main);
    g_async_worker_running = true;
    if (!g_atexit_registered) {
        std::atexit(rpp_live_shutdown_atexit);
        g_atexit_registered = true;
    }
}

bool enqueue_async_prefetch_job(const async_prefetch_job & job) {
    if (!g_config.async_prefetch) {
        return false;
    }
    ++g_prefetch_priority_planned_rows;
    append_prefetch_priority_trace(job, "planned", 0, 0, 0, 0, 0, -1);
    int32_t queue_size_after = -1;
    {
        std::lock_guard<std::mutex> lock(g_async_mutex);
        const int32_t cap = std::max<int32_t>(1, g_config.async_prefetch_queue_cap);
        if ((int32_t) g_async_queue.size() >= cap) {
            if (!prefetch_queue_is_priority()) {
                ++g_async_prefetch_dropped_rows;
                ++g_prefetch_priority_dropped_low_priority_rows;
                append_prefetch_priority_trace(job, "dropped_queue_full", 0, 0, 0, 0, 0, (int32_t) g_async_queue.size());
                append_async_prefetch_trace(job, 0, 0, 0, 0, 0, "dropped_queue_full");
                return true;
            }
            const size_t worst = lowest_prefetch_priority_index(g_async_queue);
            if (prefetch_priority_higher(job, g_async_queue[worst])) {
                async_prefetch_job replaced = g_async_queue[worst];
                g_async_queue[worst] = job;
                ++g_prefetch_priority_replaced_rows;
                append_prefetch_priority_trace(replaced, "replaced_by_higher_priority", 0, 0, 0, 0, 0, (int32_t) g_async_queue.size());
                append_async_prefetch_trace(replaced, 0, 0, 0, 0, 0, "replaced_by_higher_priority");
            } else {
                ++g_async_prefetch_dropped_rows;
                ++g_prefetch_priority_dropped_low_priority_rows;
                append_prefetch_priority_trace(job, "dropped_low_priority", 0, 0, 0, 0, 0, (int32_t) g_async_queue.size());
                append_async_prefetch_trace(job, 0, 0, 0, 0, 0, "dropped_low_priority");
                return true;
            }
        } else {
            g_async_queue.push_back(job);
        }
        queue_size_after = (int32_t) g_async_queue.size();
    }
    ++g_async_prefetch_enqueued_rows;
    ++g_prefetch_priority_enqueued_rows;
    append_prefetch_priority_trace(job, "enqueued", 0, 0, 0, 0, 0, queue_size_after);
    g_async_cv.notify_one();
    return true;
}

int64_t write_ubatch_decode_trace(
        int64_t decode_call_id,
        int32_t logical_batch_offset,
        int32_t logical_batch_n_tokens,
        int32_t runtime_n_ubatch) {
    const int32_t n_ubatch = std::max<int32_t>(1, runtime_n_ubatch);
    const int32_t current_ubatch_id = logical_batch_offset / n_ubatch;
    const int64_t decode_ts_us = monotonic_us();
    append_line(
        "ubatch_decode_trace.csv",
        std::to_string(decode_call_id) + "," +
        std::to_string(current_ubatch_id) + "," +
        std::to_string(logical_batch_offset) + "," +
        std::to_string(logical_batch_n_tokens) + "," +
        std::to_string(n_ubatch) + "," +
        std::to_string(decode_ts_us));
    return decode_ts_us;
}

std::vector<std::pair<expert_key, int>> sorted_prefetch_candidates(const std::map<expert_key, int> & counters) {
    std::vector<std::pair<expert_key, int>> candidates;
    for (const auto & it : counters) {
        if (it.second >= g_config.prefetch_threshold) {
            candidates.push_back(it);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    return candidates;
}

int32_t ubatch_required_prefetch_count(int32_t token_count) {
    const int32_t min_count = std::max<int32_t>(1, g_config.ubatch_prefetch_min_count);
    const double density_threshold = std::max(0.0, g_config.ubatch_prefetch_density_threshold);
    const int32_t density_count = std::max<int32_t>(
        1,
        (int32_t) std::ceil((double) std::max<int32_t>(1, token_count) * density_threshold));
    return std::max<int32_t>(min_count, density_count);
}

std::vector<std::pair<expert_key, int>> sorted_ubatch_prefetch_candidates(
        const std::map<expert_key, int> & counters,
        int32_t token_count) {
    std::vector<std::pair<expert_key, int>> candidates;
    const double denom = std::max<int32_t>(1, token_count);
    const int32_t required_count = ubatch_required_prefetch_count(token_count);
    for (const auto & it : counters) {
        if (it.second >= required_count) {
            candidates.push_back(it);
        }
    }
    if (g_config.prefetch_queue_policy == "fifo" &&
            (g_config.prefetch_fifo_candidate_order == "layer-counter" ||
             g_config.prefetch_fifo_candidate_order == "early-layer-counter")) {
        std::sort(candidates.begin(), candidates.end(), [denom](const auto & a, const auto & b) {
            if (a.first.layer != b.first.layer) {
                return a.first.layer < b.first.layer;
            }
            if (a.second != b.second) {
                return a.second > b.second;
            }
            const double da = (double) a.second / denom;
            const double db = (double) b.second / denom;
            if (da != db) {
                return da > db;
            }
            return a.first.expert < b.first.expert;
        });
        return candidates;
    }

    std::sort(candidates.begin(), candidates.end(), [denom](const auto & a, const auto & b) {
        const double da = (double) a.second / denom;
        const double db = (double) b.second / denom;
        if (da != db) {
            return da > db;
        }
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    return candidates;
}

std::set<int32_t> batch_seq_set(const llama_batch & batch, int32_t index) {
    std::set<int32_t> out;
    const int32_t n_seq = batch.n_seq_id ? batch.n_seq_id[index] : 1;
    for (int32_t s = 0; s < n_seq; ++s) {
        out.insert(batch.seq_id && batch.seq_id[index] ? batch.seq_id[index][s] : 0);
    }
    return out;
}

bool seq_sets_overlap(const std::set<int32_t> & a, const std::set<int32_t> & b) {
    for (int32_t value : a) {
        if (b.count(value) > 0) {
            return true;
        }
    }
    return false;
}

std::string join_i32s(const std::vector<int32_t> & values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ' ';
        }
        oss << values[i];
    }
    return oss.str();
}

std::vector<std::vector<int32_t>> split_equal_batch_indices(
        const llama_batch & batch,
        int32_t runtime_n_ubatch,
        bool sequential) {
    const int32_t n_tokens = std::max<int32_t>(0, batch.n_tokens);
    const int32_t n_ubatch = std::max<int32_t>(1, runtime_n_ubatch);
    std::vector<std::set<int32_t>> seq_sets;
    seq_sets.reserve((size_t) n_tokens);
    std::map<std::set<int32_t>, std::vector<int32_t>> seq_set_map;
    for (int32_t i = 0; i < n_tokens; ++i) {
        std::set<int32_t> cur = batch_seq_set(batch, i);
        seq_sets.push_back(cur);
        seq_set_map[cur].push_back(i);
    }

    std::vector<char> used((size_t) n_tokens, 0);
    std::vector<std::vector<int32_t>> ubatches;
    for (;;) {
        std::vector<std::set<int32_t>> cur_seq_sets;
        int32_t last_seq_id = -1;
        for (int32_t i = 0; i < n_tokens; ++i) {
            if (used[(size_t) i]) {
                continue;
            }
            bool add = true;
            for (const auto & existing : cur_seq_sets) {
                if (seq_sets_overlap(existing, seq_sets[(size_t) i])) {
                    add = false;
                    break;
                }
            }
            const int32_t seq_id = batch.seq_id && batch.seq_id[i] ? batch.seq_id[i][0] : 0;
            if (sequential) {
                add = add && (cur_seq_sets.empty() || seq_id == last_seq_id + 1);
            }
            if (add) {
                cur_seq_sets.push_back(seq_sets[(size_t) i]);
                last_seq_id = seq_id;
                if ((int32_t) cur_seq_sets.size() > n_ubatch) {
                    break;
                }
            }
        }

        const int32_t n_seqs = (int32_t) cur_seq_sets.size();
        if (n_seqs == 0) {
            break;
        }

        std::vector<int32_t> cur_idx((size_t) n_seqs, 0);
        for (int32_t s = 0; s < n_seqs; ++s) {
            const auto & indexes = seq_set_map[cur_seq_sets[(size_t) s]];
            while (cur_idx[(size_t) s] < (int32_t) indexes.size() && used[(size_t) indexes[(size_t) cur_idx[(size_t) s]]]) {
                ++cur_idx[(size_t) s];
            }
        }

        std::vector<std::vector<int32_t>> idxs_per_seq((size_t) n_seqs);
        for (;;) {
            bool can_expand = true;
            for (int32_t s = 0; s < n_seqs; ++s) {
                const auto & indexes = seq_set_map[cur_seq_sets[(size_t) s]];
                if (cur_idx[(size_t) s] >= (int32_t) indexes.size()) {
                    can_expand = false;
                    break;
                }
            }
            if (!can_expand) {
                break;
            }
            for (int32_t s = 0; s < n_seqs; ++s) {
                const auto & indexes = seq_set_map[cur_seq_sets[(size_t) s]];
                const int32_t idx = indexes[(size_t) cur_idx[(size_t) s]];
                idxs_per_seq[(size_t) s].push_back(idx);
                used[(size_t) idx] = 1;
                ++cur_idx[(size_t) s];
            }
            if (((int32_t) idxs_per_seq[0].size() + 1) * n_seqs > n_ubatch) {
                break;
            }
        }

        std::vector<int32_t> idxs;
        for (const auto & per_seq : idxs_per_seq) {
            idxs.insert(idxs.end(), per_seq.begin(), per_seq.end());
        }
        if (idxs.empty()) {
            break;
        }
        ubatches.push_back(std::move(idxs));
    }
    return ubatches;
}

std::vector<std::vector<int32_t>> split_physical_ubatch_indices(
        const llama_batch & batch,
        int32_t runtime_n_ubatch) {
    std::vector<std::vector<int32_t>> ubatches = split_equal_batch_indices(batch, runtime_n_ubatch, true);
    size_t used = 0;
    for (const auto & ubatch : ubatches) {
        used += ubatch.size();
    }
    if (used == (size_t) std::max<int32_t>(0, batch.n_tokens)) {
        return ubatches;
    }
    return split_equal_batch_indices(batch, runtime_n_ubatch, false);
}

std::vector<planned_ubatch> build_physical_ubatch_plans(
        int64_t decode_call_id,
        int32_t source_ubatch_id,
        const llama_batch & batch,
        int32_t runtime_n_ubatch,
        const std::vector<predicted_token> & actual_tokens) {
    std::map<int32_t, std::vector<predicted_token>> tokens_by_batch_index;
    for (const predicted_token & pred : actual_tokens) {
        tokens_by_batch_index[pred.batch_index].push_back(pred);
    }

    const auto physical_indices = split_physical_ubatch_indices(batch, runtime_n_ubatch);
    std::vector<planned_ubatch> plans;
    plans.reserve(physical_indices.size());
    for (size_t ubatch_index = 0; ubatch_index < physical_indices.size(); ++ubatch_index) {
        const auto & indices = physical_indices[ubatch_index];
        if (indices.empty()) {
            continue;
        }
        planned_ubatch plan;
        plan.source_decode_call_id = decode_call_id;
        plan.source_ubatch_id = source_ubatch_id;
        plan.target_decode_call_id = decode_call_id;
        plan.target_ubatch_id = (int32_t) ubatch_index;
        plan.target_first_batch_order = indices.front();

        for (size_t offset = 0; offset < indices.size(); ++offset) {
            const int32_t batch_index = indices[offset];
            auto token_it = tokens_by_batch_index.find(batch_index);
            if (token_it == tokens_by_batch_index.end()) {
                continue;
            }
            for (predicted_token pred : token_it->second) {
                pred.ubatch_id = (int32_t) ubatch_index;
                pred.ubatch_offset = (int32_t) offset;
                plan.tokens.push_back(pred);
                ++plan.target_token_count;
                if (pred.phase == "prefill") {
                    ++plan.prefill_token_count;
                    const oracle_token * token = find_oracle_token(pred.sample_id, "prefill", pred.token_index);
                    if (token == nullptr) {
                        continue;
                    }
                    for (const expert_key & key : token->experts) {
                        plan.counters[key] += 1;
                    }
                } else if (pred.phase == "decode") {
                    ++plan.decode_token_count;
                    if (pred.experts.empty()) {
                        append_line(
                            "decode_rpp_missing_prediction.csv",
                            std::to_string(decode_call_id) + "," +
                            std::to_string(pred.slot_id) + "," +
                            std::to_string(pred.task_id) + "," +
                            csv_escape(pred.sample_id) + "," +
                            std::to_string(pred.decode_index) + "," +
                            std::to_string(pred.token_id) + ",missing_decode_prefetch_experts");
                        ++g_decode_rpp_missing_prediction_rows;
                        continue;
                    }
                    for (const expert_key & key : pred.experts) {
                        plan.counters[key] += 1;
                    }
                }
            }
        }

        std::set<std::set<int32_t>> physical_seq_sets;
        for (int32_t batch_index : indices) {
            physical_seq_sets.insert(batch_seq_set(batch, batch_index));
        }
        const int32_t physical_n_seqs = (int32_t) physical_seq_sets.size();
        const int32_t physical_n_seq_tokens =
            physical_n_seqs > 0 ? (int32_t) indices.size() / physical_n_seqs : 0;
        append_line(
            "physical_ubatch_trace.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(ubatch_index) + "," +
            std::to_string(indices.size()) + "," +
            std::to_string(physical_n_seqs) + "," +
            std::to_string(physical_n_seq_tokens) + "," +
            std::to_string(indices.front()) + "," +
            std::to_string(indices.back()) + "," +
            csv_escape(join_i32s(indices)));
        ++g_physical_ubatch_rows;
        plans.push_back(std::move(plan));
    }
    return plans;
}

void write_ubatch_plan_trace(const planned_ubatch & plan) {
    append_line(
        "ubatch_plan_trace.csv",
        std::to_string(plan.source_decode_call_id) + "," +
        std::to_string(plan.source_ubatch_id) + "," +
        std::to_string(plan.target_decode_call_id) + "," +
        std::to_string(plan.target_ubatch_id) + "," +
        std::to_string(plan.target_first_batch_order) + "," +
        std::to_string(plan.target_token_count) + "," +
        std::to_string(plan.prefill_token_count) + "," +
        std::to_string(plan.decode_token_count) + "," +
        std::to_string(plan.counters.size()) + "," +
        std::to_string(monotonic_us()));
    ++g_ubatch_plan_rows;
}

void write_expert_use_plan_trace(const planned_ubatch & plan, int64_t decode_start_ts_us) {
    for (const auto & it : plan.counters) {
        const double density = (double) it.second / std::max<int32_t>(1, plan.target_token_count);
        append_line(
            "expert_use_plan_trace.csv",
            std::to_string(plan.target_decode_call_id) + "," +
            std::to_string(plan.target_ubatch_id) + "," +
            std::to_string(it.first.layer) + "," +
            std::to_string(it.first.expert) + "," +
            std::to_string(it.second) + "," +
            std::to_string(density) + "," +
            std::to_string(plan.target_token_count) + "," +
            std::to_string(plan.prefill_token_count) + "," +
            std::to_string(plan.decode_token_count) + "," +
            std::to_string(decode_start_ts_us));
        ++g_expert_use_plan_rows;
    }
}

void update_expert_use_state(const planned_ubatch & plan) {
    for (const auto & it : plan.counters) {
        expert_runtime_state & state = g_expert_states[it.first];
        state.last_used_decode_call_id = plan.target_decode_call_id;
        state.last_used_ubatch_id = plan.target_ubatch_id;
    }
}

void append_expert_state_trace(int64_t decode_call_id, int32_t ubatch_id) {
    for (const auto & it : g_expert_states) {
        const expert_key & key = it.first;
        const expert_runtime_state & state = it.second;
        append_line(
            "expert_state_trace.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(ubatch_id) + "," +
            std::to_string(key.layer) + "," +
            std::to_string(key.expert) + "," +
            std::to_string(state.last_used_decode_call_id) + "," +
            std::to_string(state.last_used_ubatch_id) + "," +
            std::to_string(state.last_prefetched_decode_call_id) + "," +
            std::to_string(state.last_prefetched_ubatch_id) + "," +
            std::to_string(state.protected_until_decode_call_id) + "," +
            std::to_string(state.protected_until_ubatch_id) + "," +
            std::to_string(state.inflight ? 1 : 0));
    }
}

bool submit_current_ubatch_prefetch(const planned_ubatch & plan, bool allow_prefetch_for_event) {
    const int32_t ubatch_distance = plan.target_ubatch_id - plan.source_ubatch_id;
    if (ubatch_distance != 0) {
        return false;
    }

    for (const auto & it : plan.counters) {
        const double density = (double) it.second / std::max<int32_t>(1, plan.target_token_count);
        append_line(
            "ubatch_counter_trace.csv",
            std::to_string(plan.source_decode_call_id) + "," +
            std::to_string(plan.source_ubatch_id) + "," +
            std::to_string(plan.target_decode_call_id) + "," +
            std::to_string(plan.target_ubatch_id) + "," +
            std::to_string(plan.target_first_batch_order) + "," +
            std::to_string(plan.target_token_count) + "," +
            std::to_string(it.first.layer) + "," +
            std::to_string(it.first.expert) + "," +
            std::to_string(it.second) + "," +
            std::to_string(density));
        ++g_ubatch_counter_rows;
    }

    if (plan.target_token_count == 0) {
        return false;
    }
    const bool target_all_prefill = plan.decode_token_count == 0 && plan.prefill_token_count > 0;
    if (g_config.prefetch_prefill_only && !target_all_prefill) {
        g_prefetch_closed = true;
        g_prefetch_stop_decode_call = plan.source_decode_call_id;
        return false;
    }
    const bool allow_prefetch =
        allow_prefetch_for_event &&
        (!g_config.prefetch_prefill_only || (!g_prefetch_closed && target_all_prefill));
    if (!allow_prefetch) {
        return false;
    }

    const int64_t enqueue_ts_us = monotonic_us();
    int64_t sync_fadvise_calls = 0;
    int64_t sync_fadvise_errors = 0;
    int64_t sync_advised_bytes = 0;

    const int32_t required_count = ubatch_required_prefetch_count(plan.target_token_count);
    const auto candidates = sorted_ubatch_prefetch_candidates(plan.counters, plan.target_token_count);
    int32_t planned_experts = 0;
    bool submitted_any = false;
    for (const auto & item : candidates) {
        if (g_config.ubatch_prefetch_max_experts > 0 && planned_experts >= g_config.ubatch_prefetch_max_experts) {
            break;
        }
        const expert_key & key = item.first;
        const int counter = item.second;
        const double density = (double) counter / std::max<int32_t>(1, plan.target_token_count);
        int64_t calls = 0;
        int64_t errors = 0;
        int64_t bytes = 0;
        async_prefetch_job async_job;
        async_job.key = key;
        async_job.source_decode_call_id = plan.source_decode_call_id;
        async_job.source_ubatch_id = plan.source_ubatch_id;
        async_job.target_decode_call_id = plan.target_decode_call_id;
        async_job.target_ubatch_id = plan.target_ubatch_id;
        async_job.target_first_batch_order = plan.target_first_batch_order;
        async_job.target_token_count = plan.target_token_count;
        async_job.counter = counter;
        async_job.density = density;
        async_job.required_count = required_count;
        async_job.target_ubatch_distance = ubatch_distance;
        async_job.enqueue_ts_us = enqueue_ts_us;
        const bool queued_async = enqueue_async_prefetch_job(async_job);
        if (!queued_async) {
            std::tie(calls, errors, bytes) = prefetch_one(key);
            sync_fadvise_calls += calls;
            sync_fadvise_errors += errors;
            sync_advised_bytes += bytes;
            g_fadvise_calls += calls;
            g_fadvise_errors += errors;
            g_advised_bytes += bytes;
            expert_runtime_state & state = g_expert_states[key];
            state.last_prefetched_decode_call_id = plan.target_decode_call_id;
            state.last_prefetched_ubatch_id = plan.target_ubatch_id;
            state.protected_until_decode_call_id = plan.target_decode_call_id;
            state.protected_until_ubatch_id = plan.target_ubatch_id + g_config.reclaim_protect_ubatches;
        }
        append_line(
            "ubatch_prefetch_events.csv",
            std::to_string(plan.source_decode_call_id) + "," +
            std::to_string(plan.source_ubatch_id) + "," +
            std::to_string(plan.target_decode_call_id) + "," +
            std::to_string(plan.target_ubatch_id) + "," +
            std::to_string(plan.target_first_batch_order) + "," +
            std::to_string(plan.target_token_count) + "," +
            std::to_string(key.layer) + "," +
            std::to_string(key.expert) + "," +
            std::to_string(counter) + "," +
            std::to_string(density) + "," +
            std::to_string(required_count) + "," +
            std::to_string(g_config.ubatch_prefetch_min_count) + "," +
            std::to_string(g_config.ubatch_prefetch_density_threshold) + "," +
            csv_escape(g_config.prefetch_io_mode) + "," +
            std::to_string(enqueue_ts_us) + "," +
            std::to_string(calls) + "," +
            std::to_string(errors) + "," +
            std::to_string(bytes));
        ++g_ubatch_prefetch_rows;
        ++planned_experts;
        submitted_any = true;
    }

    if (sync_fadvise_calls > 0 || sync_fadvise_errors > 0 || sync_advised_bytes > 0) {
        const int64_t pre_decode_ready_ts_us = monotonic_us();
        append_line(
            "ubatch_prefetch_lead_trace.csv",
            std::to_string(plan.source_decode_call_id) + "," +
            std::to_string(plan.target_decode_call_id) + "," +
            std::to_string(plan.source_ubatch_id) + "," +
            std::to_string(plan.target_ubatch_id) + "," +
            std::to_string(plan.target_first_batch_order) + "," +
            std::to_string(plan.target_token_count) + "," +
            std::to_string(enqueue_ts_us) + "," +
            std::to_string(pre_decode_ready_ts_us) + "," +
            std::to_string(pre_decode_ready_ts_us - enqueue_ts_us) + "," +
            std::to_string(sync_fadvise_calls) + "," +
            std::to_string(sync_fadvise_errors) + "," +
            std::to_string(sync_advised_bytes));
        ++g_ubatch_lead_rows;
    }
    return submitted_any;
}

physical_ubatch_runtime make_physical_runtime(const planned_ubatch & plan) {
    physical_ubatch_runtime runtime;
    runtime.plan = plan;
    const int32_t required_count = ubatch_required_prefetch_count(plan.target_token_count);
    const auto candidates = sorted_ubatch_prefetch_candidates(plan.counters, plan.target_token_count);
    int32_t planned_experts = 0;
    for (const auto & item : candidates) {
        if (g_config.ubatch_prefetch_max_experts > 0 && planned_experts >= g_config.ubatch_prefetch_max_experts) {
            break;
        }
        const expert_key & key = item.first;
        const int counter = item.second;
        const double density = (double) counter / std::max<int32_t>(1, plan.target_token_count);
        async_prefetch_job async_job;
        async_job.key = key;
        async_job.source_decode_call_id = plan.source_decode_call_id;
        async_job.source_ubatch_id = plan.source_ubatch_id;
        async_job.target_decode_call_id = plan.target_decode_call_id;
        async_job.target_ubatch_id = plan.target_ubatch_id;
        async_job.target_first_batch_order = plan.target_first_batch_order;
        async_job.target_token_count = plan.target_token_count;
        async_job.counter = counter;
        async_job.density = density;
        async_job.required_count = required_count;
        async_job.target_ubatch_distance = plan.target_ubatch_id - plan.source_ubatch_id;
        runtime.jobs_by_layer[key.layer].push_back(async_job);
        ++planned_experts;
    }
    return runtime;
}

int32_t active_layer_count_locked() {
    int32_t max_layer = -1;
    for (const physical_ubatch_runtime & runtime : g_active_physical_ubatches) {
        if (!runtime.jobs_by_layer.empty()) {
            max_layer = std::max<int32_t>(max_layer, runtime.jobs_by_layer.rbegin()->first);
        }
    }
    return std::max<int32_t>(1, max_layer + 1);
}

int32_t release_cross_ubatch_prefetch_locked(int32_t source_ubatch_id, int32_t source_layer) {
    if (!g_config.async_prefetch || g_config.cross_ubatch_layer_prefetch == 0) {
        return 0;
    }
    if (source_layer <= 0) {
        return 0;
    }
    const int32_t lookahead = std::max<int32_t>(1, g_config.cross_ubatch_layer_lookahead);
    int32_t released_jobs = 0;
    for (int32_t delta = 1; delta <= lookahead; ++delta) {
        const int32_t target_ubatch_id = source_ubatch_id + delta;
        if (target_ubatch_id < 0 || target_ubatch_id >= (int32_t) g_active_physical_ubatches.size()) {
            continue;
        }
        physical_ubatch_runtime & target = g_active_physical_ubatches[(size_t) target_ubatch_id];
        const int32_t release_to_layer = source_layer - 1;
        const int32_t release_from_layer = target.released_until_layer + 1;
        if (release_from_layer > release_to_layer) {
            continue;
        }
        for (int32_t layer = release_from_layer; layer <= release_to_layer; ++layer) {
            auto layer_it = target.jobs_by_layer.find(layer);
            if (layer_it == target.jobs_by_layer.end()) {
                continue;
            }
            for (async_prefetch_job job : layer_it->second) {
                job.enqueue_ts_us = monotonic_us();
                job.source_ubatch_id = source_ubatch_id;
                job.target_ubatch_distance = target_ubatch_id - source_ubatch_id;
                enqueue_async_prefetch_job(job);
                ++released_jobs;
            }
        }
        target.released_until_layer = std::max(target.released_until_layer, release_to_layer);
        append_line(
            "cross_ubatch_release_trace.csv",
            std::to_string(g_eval_decode_call_id) + "," +
            std::to_string(source_ubatch_id) + "," +
            std::to_string(source_layer) + "," +
            std::to_string(target_ubatch_id) + "," +
            std::to_string(release_from_layer) + "," +
            std::to_string(release_to_layer) + "," +
            std::to_string(released_jobs) + "," +
            std::to_string(monotonic_us()));
        ++g_cross_ubatch_released_rows;
    }
    return released_jobs;
}

int32_t release_layer_frontier_prefetch_locked(int32_t source_ubatch_id, int32_t source_layer) {
    if (!g_config.async_prefetch || g_config.layer_frontier_prefetch == 0) {
        return 0;
    }
    if (source_ubatch_id < 0 || source_layer < 0 || g_active_physical_ubatches.empty()) {
        return 0;
    }
    const int32_t max_distance = std::max<int32_t>(
        1,
        std::min(g_config.layer_frontier_max_distance, g_config.layer_frontier_lookahead_layers));
    struct release_candidate {
        int32_t distance = 0;
        int32_t target_ubatch_id = -1;
        int32_t target_layer = -1;
    };
    const int32_t n_layers = active_layer_count_locked();
    std::vector<release_candidate> candidates;
    for (int32_t target_ubatch_id = source_ubatch_id;
            target_ubatch_id < (int32_t) g_active_physical_ubatches.size();
            ++target_ubatch_id) {
        physical_ubatch_runtime & target = g_active_physical_ubatches[(size_t) target_ubatch_id];
        for (const auto & layer_it : target.jobs_by_layer) {
            const int32_t target_layer = layer_it.first;
            if (target.frontier_released_layers.count(target_layer) > 0) {
                continue;
            }
            const int32_t distance =
                (target_ubatch_id - source_ubatch_id) * n_layers + (target_layer - source_layer);
            if (distance <= 0 || distance > max_distance) {
                continue;
            }
            candidates.push_back({ distance, target_ubatch_id, target_layer });
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const release_candidate & a, const release_candidate & b) {
        if (a.distance != b.distance) {
            return a.distance < b.distance;
        }
        if (a.target_ubatch_id != b.target_ubatch_id) {
            return a.target_ubatch_id < b.target_ubatch_id;
        }
        return a.target_layer < b.target_layer;
    });

    int32_t released_jobs_total = 0;
    const int32_t jobs_per_tick = std::max<int32_t>(1, g_config.layer_frontier_jobs_per_tick);
    for (const release_candidate & cand : candidates) {
        physical_ubatch_runtime & target = g_active_physical_ubatches[(size_t) cand.target_ubatch_id];
        if (target.frontier_released_layers.count(cand.target_layer) > 0) {
            continue;
        }
        auto layer_it = target.jobs_by_layer.find(cand.target_layer);
        if (layer_it == target.jobs_by_layer.end()) {
            target.frontier_released_layers.insert(cand.target_layer);
            continue;
        }
        int32_t released_jobs = 0;
        for (async_prefetch_job job : layer_it->second) {
            job.enqueue_ts_us = monotonic_us();
            job.source_ubatch_id = source_ubatch_id;
            job.target_ubatch_distance = cand.target_ubatch_id - source_ubatch_id;
            enqueue_async_prefetch_job(job);
            ++released_jobs;
            ++released_jobs_total;
        }
        target.frontier_released_layers.insert(cand.target_layer);
        append_line(
            "layer_frontier_release_trace.csv",
            std::to_string(g_eval_decode_call_id) + "," +
            std::to_string(source_ubatch_id) + "," +
            std::to_string(source_layer) + "," +
            std::to_string(cand.target_ubatch_id) + "," +
            std::to_string(cand.target_layer) + "," +
            std::to_string(cand.distance) + "," +
            std::to_string(released_jobs) + "," +
            std::to_string(monotonic_us()));
        ++g_layer_frontier_released_rows;
        if (released_jobs_total >= jobs_per_tick) {
            break;
        }
    }
    return released_jobs_total;
}

bool reclaim_priority_higher(const reclaim_candidate & a, const reclaim_candidate & b) {
    if (a.future_density_sum != b.future_density_sum) {
        return a.future_density_sum < b.future_density_sum;
    }
    if (a.future_counter_sum != b.future_counter_sum) {
        return a.future_counter_sum < b.future_counter_sum;
    }
    if (a.next_use_distance != b.next_use_distance) {
        return a.next_use_distance > b.next_use_distance;
    }
    if (a.last_used_decode_call_id != b.last_used_decode_call_id) {
        return a.last_used_decode_call_id < b.last_used_decode_call_id;
    }
    if (a.resident_bytes != b.resident_bytes) {
        return a.resident_bytes > b.resident_bytes;
    }
    if (a.key.layer != b.key.layer) {
        return a.key.layer < b.key.layer;
    }
    return a.key.expert < b.key.expert;
}

void plan_reclaim_priority_trace(const std::vector<planned_ubatch> & plans, int64_t decode_call_id, int32_t source_ubatch_id) {
    if (g_config.reclaim_mode == "off") {
        return;
    }
    std::map<expert_key, double> future_density;
    std::map<expert_key, int32_t> future_counter;
    std::map<expert_key, int32_t> next_use;
    std::set<expert_key> future_high_priority;
    const int32_t lookahead = std::max<int32_t>(0, g_config.reclaim_lookahead_ubatches);
    for (const planned_ubatch & plan : plans) {
        const int32_t distance = plan.target_ubatch_id - source_ubatch_id;
        if (distance < 0 || distance > lookahead) {
            continue;
        }
        const int32_t required_count = ubatch_required_prefetch_count(plan.target_token_count);
        for (const auto & it : plan.counters) {
            const double density = (double) it.second / std::max<int32_t>(1, plan.target_token_count);
            future_density[it.first] += density;
            future_counter[it.first] += it.second;
            auto next_it = next_use.find(it.first);
            if (next_it == next_use.end() || distance < next_it->second) {
                next_use[it.first] = distance;
            }
            if (it.second >= required_count) {
                future_high_priority.insert(it.first);
            }
        }
    }

    std::vector<reclaim_candidate> candidates;
    candidates.reserve(g_expert_states.size());
    for (const auto & it : g_expert_states) {
        reclaim_candidate cand;
        cand.key = it.first;
        auto dens_it = future_density.find(cand.key);
        auto count_it = future_counter.find(cand.key);
        auto next_it = next_use.find(cand.key);
        cand.future_density_sum = dens_it == future_density.end() ? 0.0 : dens_it->second;
        cand.future_counter_sum = count_it == future_counter.end() ? 0 : count_it->second;
        cand.next_use_distance = next_it == next_use.end() ? std::numeric_limits<int32_t>::max() : next_it->second;
        cand.last_used_decode_call_id = it.second.last_used_decode_call_id;
        cand.resident_bytes = expert_resident_bytes(cand.key);
        if (it.second.inflight) {
            cand.decision = "skipped_inflight";
        } else if (ubatch_point_leq(decode_call_id, source_ubatch_id, it.second.protected_until_decode_call_id, it.second.protected_until_ubatch_id)) {
            cand.decision = "skipped_protected";
        } else if (future_high_priority.count(cand.key) > 0) {
            cand.decision = "skipped_future_high_priority";
        } else if (g_config.reclaim_mode == "trace-priority" || g_config.reclaim_mode == "log-only") {
            cand.decision = "trace_only_candidate";
        } else {
            cand.decision = "not_implemented";
        }
        candidates.push_back(std::move(cand));
    }
    std::sort(candidates.begin(), candidates.end(), reclaim_priority_higher);
    const int32_t cap = std::max<int32_t>(1, g_config.reclaim_queue_cap);
    const int32_t n = std::min<int32_t>(cap, (int32_t) candidates.size());
    for (int32_t i = 0; i < n; ++i) {
        reclaim_candidate & cand = candidates[(size_t) i];
        int64_t calls = 0;
        int64_t errors = 0;
        int64_t bytes = 0;
        if (cand.decision == "not_implemented" && g_config.reclaim_mode == "fadvise-dontneed") {
            std::tie(calls, errors, bytes) = reclaim_one(cand.key);
            g_reclaim_fadvise_calls += calls;
            g_reclaim_fadvise_errors += errors;
            g_reclaim_advised_bytes += bytes;
            cand.decision = errors == 0 ? "evicted_fadvise_dontneed" : "evict_fadvise_error";
        }
        append_line(
            "reclaim_priority_trace.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(source_ubatch_id) + "," +
            std::to_string(cand.key.layer) + "," +
            std::to_string(cand.key.expert) + "," +
            std::to_string(cand.future_density_sum) + "," +
            std::to_string(cand.future_counter_sum) + "," +
            std::to_string(cand.next_use_distance) + "," +
            std::to_string(cand.last_used_decode_call_id) + "," +
            std::to_string(cand.resident_bytes) + "," +
            std::to_string(i) + "," +
            csv_escape(g_config.reclaim_mode) + "," +
            csv_escape(cand.decision) + "," +
            std::to_string(calls) + "," +
            std::to_string(errors) + "," +
            std::to_string(bytes));
        ++g_reclaim_priority_trace_rows;
        if (cand.decision == "trace_only_candidate" || cand.decision == "evicted_fadvise_dontneed") {
            ++g_reclaim_priority_candidates;
        }
    }
}

void handle_prefetch_and_reclaim(
        const std::vector<std::pair<expert_key, int>> & candidates,
        int64_t decode_call_id,
        bool allow_prefetch) {
    std::set<expert_key> planned_now;
    for (const auto & item : candidates) {
        const expert_key & key = item.first;
        const int counter = item.second;
        planned_now.insert(key);
        int64_t calls = 0;
        int64_t errors = 0;
        int64_t bytes = 0;
        if (allow_prefetch) {
            std::tie(calls, errors, bytes) = prefetch_one(key);
        }
        g_fadvise_calls += calls;
        g_fadvise_errors += errors;
        g_advised_bytes += bytes;
        append_line(
            "prefetch_events.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(key.layer) + "," +
            std::to_string(key.expert) + "," +
            std::to_string(counter) + "," +
            std::to_string(calls) + "," +
            std::to_string(errors) + "," +
            std::to_string(bytes));
        ++g_prefetch_rows;
    }

    for (const expert_key & key : g_last_planned) {
        if (planned_now.count(key) > 0) {
            continue;
        }
        const int counter = g_counters.count(key) ? g_counters[key] : 0;
        append_line(
            "reclaim_candidates.csv",
            std::to_string(decode_call_id) + "," +
            std::to_string(key.layer) + "," +
            std::to_string(key.expert) + ",1," +
            std::to_string(counter) + "," +
            std::to_string(g_config.prefetch_threshold) + "," +
            csv_escape(g_config.reclaim_mode));
        ++g_reclaim_rows;
    }
    g_last_planned = std::move(planned_now);
}

void update_prediction_queue_and_prefetch(
        const std::vector<predicted_token> & predicted_order,
        int64_t event_id) {
    write_predicted_order_trace(predicted_order, event_id);
    rebuild_queue_from_prediction(predicted_order);
    const auto candidates = recompute_counters();
    write_queue_and_counter_trace(event_id);
    if (g_config.prefetch_prefill_only && !g_prefetch_closed && !queue_is_all_prefill()) {
        g_prefetch_closed = true;
        g_prefetch_stop_decode_call = event_id;
    }
    handle_prefetch_and_reclaim(candidates, event_id, false);
}

void run_prediction_cycle(
        const llama_batch & batch,
        const std::vector<rpp_live_slot_info> & slots,
        int32_t runtime_n_batch,
        int32_t runtime_n_ubatch,
        int64_t event_id,
        bool allow_prefetch_for_event) {
    register_active_samples(slots);
    const int32_t n_batch = std::max<int32_t>(1, runtime_n_batch);
    const int32_t window_size = std::max<int32_t>(1, g_config.window_size);
    const auto predicted_current_batch =
        predict_global_decode_order(batch, slots, runtime_n_batch, runtime_n_ubatch, n_batch, 1);
    const auto predicted_window =
        predict_global_decode_order(batch, slots, runtime_n_batch, runtime_n_ubatch, window_size, 10000);

    write_predicted_model_order_trace(predicted_current_batch, event_id);
    g_pending_predicted_order = predicted_current_batch;
    (void) allow_prefetch_for_event;
    update_prediction_queue_and_prefetch(predicted_window, event_id);
}

void write_legacy_live_order_event(
        const llama_batch & batch,
        const std::vector<rpp_live_slot_info> & slots,
        int32_t logical_batch_offset,
        int32_t logical_batch_n_tokens,
        int64_t decode_call_id) {
    const char * path = std::getenv("LLAMA_LIVE_ORDER_HOOK");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    json event;
    event["event"]                  = "pre_decode_batch";
    event["decode_call_id"]         = decode_call_id;
    event["logical_batch_offset"]   = logical_batch_offset;
    event["logical_batch_n_tokens"] = logical_batch_n_tokens;
    event["n_tokens"]               = batch.n_tokens;
    event["slots"]                  = json::array();
    event["tokens"]                 = json::array();

    for (const rpp_live_slot_info & slot : slots) {
        event["slots"].push_back({
            { "slot_id", slot.slot_id },
            { "seq_id",  slot.seq_id },
            { "task_id", slot.task_id },
            { "slot_state", slot_state_name(slot.slot_state) },
            { "n_decoded", slot.n_decoded },
            { "prompt_n_tokens", slot.prompt_n_tokens },
            { "task_n_tokens", slot.task_n_tokens },
            { "i_batch", slot.i_batch },
            { "is_generating", slot.is_generating },
            { "rpp_oracle_sample_id", slot.oracle_sample_id },
        });
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int32_t n_seq_id = batch.n_seq_id ? batch.n_seq_id[i] : 1;
        json token;
        token["logical_batch_index"] = logical_batch_offset + i;
        token["token_id"]            = batch.token ? batch.token[i] : 0;
        token["pos"]                 = batch.pos ? batch.pos[i] : 0;
        token["n_seq_id"]            = n_seq_id;
        token["output"]              = batch.logits ? batch.logits[i] != 0 : false;
        token["seq_id"]              = json::array();
        for (int32_t s = 0; s < n_seq_id; ++s) {
            token["seq_id"].push_back(batch.seq_id && batch.seq_id[i] ? batch.seq_id[i][s] : 0);
        }
        event["tokens"].push_back(std::move(token));
    }

    std::ofstream out(path, std::ios::app);
    if (out.good()) {
        out << event.dump() << '\n';
    }
}

bool env_truthy(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

double env_double(const char * name, double fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed)) {
        return fallback;
    }
    return parsed;
}

int32_t env_i32(const char * name, int32_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return (int32_t) parsed;
}

std::string env_string(const char * name, const std::string & fallback) {
    const char * value = std::getenv(name);
    return value && value[0] ? std::string(value) : fallback;
}

std::string env_prefetch_io_mode() {
    const char * value = std::getenv("RPP_PREFETCH_IO_MODE");
    std::string mode = value && value[0] ? value : "readahead";
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    if (mode == "fadvise" || mode == "readahead" || mode == "mmap") {
        return mode;
    }
    return "readahead";
}

bool queue_is_all_prefill() {
    if (g_queue.empty()) {
        return false;
    }
    for (const queued_token & token : g_queue) {
        if (token.phase != "prefill") {
            return false;
        }
    }
    return true;
}

bool parse_moe_layer_node(const char * name, int32_t & layer) {
    if (name == nullptr) {
        return false;
    }
    std::string_view view(name);
    constexpr std::string_view prefix = "ffn_moe_gate_up-";
    if (view.rfind(prefix, 0) != 0) {
        return false;
    }
    const std::string_view suffix = view.substr(prefix.size());
    if (suffix.empty()) {
        return false;
    }
    int32_t parsed = 0;
    for (char ch : suffix) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        parsed = parsed * 10 + (ch - '0');
    }
    layer = parsed;
    return true;
}

bool rpp_live_eval_callback_impl(ggml_tensor * t, bool ask, void *) {
    int32_t layer = -1;
    const bool observe = t != nullptr && parse_moe_layer_node(t->name, layer);
    if (ask) {
        return observe;
    }
    if (!observe) {
        return true;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || g_eval_decode_call_id < 0 ||
            (g_config.cross_ubatch_layer_prefetch == 0 && g_config.layer_frontier_prefetch == 0)) {
        return true;
    }
    if (layer == 0) {
        if (g_eval_current_ubatch < 0 || g_eval_last_layer > 0) {
            ++g_eval_current_ubatch;
            g_eval_last_layer = -1;
        }
    }
    if (g_eval_current_ubatch < 0) {
        g_eval_current_ubatch = 0;
    }
    if (g_eval_current_ubatch >= (int32_t) g_active_physical_ubatches.size()) {
        return true;
    }

    append_line(
        "layer_progress_trace.csv",
        std::to_string(g_eval_decode_call_id) + "," +
        std::to_string(g_eval_current_ubatch) + "," +
        std::to_string(layer) + "," +
        csv_escape(t->name) + "," +
        std::to_string(monotonic_us()));
    ++g_layer_progress_rows;
    g_eval_last_layer = layer;
    release_layer_frontier_prefetch_locked(g_eval_current_ubatch, layer);
    if (g_config.layer_frontier_prefetch == 0) {
        release_cross_ubatch_prefetch_locked(g_eval_current_ubatch, layer);
    }
    return true;
}

} // namespace

void rpp_live_attach_eval_callback(common_params & params) {
    params.cb_eval = rpp_live_eval_callback_impl;
    params.cb_eval_user_data = nullptr;
}

void rpp_live_configure(const common_params & params) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_config = {};
    g_config.scheduler = params.rpp_scheduler;
    g_config.cont_batching = params.cont_batching;
    g_config.prefetch_prefill_only = env_truthy("RPP_PREFETCH_PREFILL_ONLY");
    g_config.model_path = params.model.path;
    g_config.tensor_ranges_path = params.rpp_tensor_ranges;
    g_config.oracle_trace_path = params.rpp_oracle_trace;
    g_config.rpp_live_model_path = params.rpp_live_model;
    g_config.reclaim_mode = env_string("RPP_RECLAIM_MODE", params.rpp_reclaim_mode.empty() ? "trace-priority" : params.rpp_reclaim_mode);
    g_config.prefetch_queue_policy = env_string("RPP_PREFETCH_QUEUE_POLICY", "priority");
    g_config.prefetch_fifo_candidate_order = env_string("RPP_PREFETCH_FIFO_CANDIDATE_ORDER", "density");
    g_config.prefetch_io_mode = env_prefetch_io_mode();
    g_config.window_size = std::max(1, params.rpp_window_size);
    g_config.prefetch_threshold = std::max(1, params.rpp_prefetch_count_threshold);
    g_config.ubatch_prefetch_min_count = std::max<int32_t>(1, env_i32("RPP_UBATCH_PREFETCH_MIN_COUNT", g_config.prefetch_threshold));
    g_config.ubatch_prefetch_density_threshold = std::max(0.0, env_double("RPP_UBATCH_PREFETCH_DENSITY_THRESHOLD", 0.5));
    g_config.predict_topk = std::max<int32_t>(1, params.rpp_predict_topk);
    g_config.rpp_max_seq_len = std::max<int32_t>(1, env_i32("RPP_MAX_SEQ_LEN", 512));
    g_config.predictor_threads = std::max<int32_t>(1, params.rpp_predictor_threads);
    g_config.async_prefetch = env_truthy("RPP_PREFETCH_ASYNC");
    g_config.async_prefetch_queue_cap = std::max<int32_t>(1, env_i32("RPP_PREFETCH_QUEUE_CAP", 512));
    g_config.ubatch_prefetch_max_experts = std::max<int32_t>(0, env_i32("RPP_UBATCH_PREFETCH_MAX_EXPERTS", 0));
    g_config.prefetch_max_staleness_ubatches = std::max<int32_t>(0, env_i32("RPP_PREFETCH_MAX_STALENESS_UBATCHES", 0));
    g_config.reclaim_queue_cap = std::max<int32_t>(1, env_i32("RPP_RECLAIM_QUEUE_CAP", 512));
    g_config.reclaim_lookahead_ubatches = std::max<int32_t>(0, env_i32("RPP_RECLAIM_LOOKAHEAD_UBATCHES", 4));
    g_config.reclaim_protect_ubatches = std::max<int32_t>(0, env_i32("RPP_RECLAIM_PROTECT_UBATCHES", 2));
    g_config.cross_ubatch_layer_prefetch = env_truthy("RPP_CROSS_UBATCH_LAYER_PREFETCH") ? 1 : 0;
    g_config.cross_ubatch_layer_lookahead = std::max<int32_t>(1, env_i32("RPP_CROSS_UBATCH_LAYER_LOOKAHEAD", 1));
    g_config.layer_frontier_prefetch = env_truthy("RPP_LAYER_FRONTIER_PREFETCH") ? 1 : 0;
    g_config.layer_frontier_lookahead_layers = std::max<int32_t>(1, env_i32("RPP_LAYER_FRONTIER_LOOKAHEAD_LAYERS", 8));
    g_config.layer_frontier_jobs_per_tick = std::max<int32_t>(1, env_i32("RPP_LAYER_FRONTIER_JOBS_PER_TICK", 64));
    g_config.layer_frontier_max_distance = std::max<int32_t>(1, env_i32("RPP_LAYER_FRONTIER_MAX_DISTANCE", 60));
    g_prefetch_closed = false;
    g_prefetch_stop_decode_call = -1;
    const char * out_dir = std::getenv("RPP_LIVE_OUT_DIR");
    if (out_dir && out_dir[0]) {
        g_config.out_dir = out_dir;
    }
    g_config.enabled =
        (params.rpp_scheduler == "oracle-window" && !params.rpp_oracle_trace.empty()) ||
        (!params.rpp_live_model.empty() && !params.rpp_tensor_ranges.empty());
}

int64_t rpp_live_pre_decode(
        const llama_batch & batch,
        const std::vector<rpp_live_slot_info> & slots,
        int32_t logical_batch_offset,
        int32_t logical_batch_n_tokens,
        int32_t runtime_n_batch,
        int32_t runtime_n_ubatch) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const int64_t decode_call_id = g_decode_call_id.fetch_add(1);
    write_legacy_live_order_event(batch, slots, logical_batch_offset, logical_batch_n_tokens, decode_call_id);

    if (!ensure_loaded_or_warned()) {
        return decode_call_id;
    }

    const int64_t decode_start_ts_us =
        write_ubatch_decode_trace(decode_call_id, logical_batch_offset, logical_batch_n_tokens, runtime_n_ubatch);
    register_active_samples(slots);
    sync_slot_decode_histories(slots);
    const auto by_seq = slot_by_seq(slots);
    const auto by_slot = slot_by_slot_id(slots);
    write_slot_state_trace(slots, decode_call_id);
    write_model_order_trace(batch, by_seq, logical_batch_offset, runtime_n_ubatch, decode_call_id);
    const int64_t advance_start_ts_us = monotonic_us();
    auto actual_generated_inputs = advance_current_tokens(batch, by_seq, logical_batch_offset, runtime_n_ubatch, decode_call_id);
    append_rpp_timing(
        decode_call_id,
        "pre_decode_advance_current_tokens",
        advance_start_ts_us,
        monotonic_us(),
        (int64_t) actual_generated_inputs.size(),
        logical_batch_n_tokens,
        "");
    const int64_t predict_start_ts_us = monotonic_us();
    predict_decode_rpp_experts(actual_generated_inputs, by_slot, decode_call_id);
    append_rpp_timing(
        decode_call_id,
        "pre_decode_predict_decode_rpp_experts",
        predict_start_ts_us,
        monotonic_us(),
        (int64_t) actual_generated_inputs.size(),
        logical_batch_n_tokens,
        "");
    const int64_t compare_start_ts_us = monotonic_us();
    compare_predicted_to_actual(actual_generated_inputs, decode_call_id);
    commit_decode_histories(actual_generated_inputs);
    append_rpp_timing(
        decode_call_id,
        "pre_decode_compare_commit",
        compare_start_ts_us,
        monotonic_us(),
        (int64_t) actual_generated_inputs.size(),
        logical_batch_n_tokens,
        "");

    const int32_t source_ubatch_id = 0;
    g_latest_source_decode_call_id = decode_call_id;
    g_latest_source_ubatch_id = source_ubatch_id;
    const int64_t plan_start_ts_us = monotonic_us();
    auto plans = build_physical_ubatch_plans(decode_call_id, source_ubatch_id, batch, runtime_n_ubatch, actual_generated_inputs);
    g_active_physical_ubatches.clear();
    g_active_physical_ubatches.reserve(plans.size());
    g_eval_decode_call_id = decode_call_id;
    g_eval_current_ubatch = -1;
    g_eval_last_layer = -1;
    for (const planned_ubatch & plan : plans) {
        write_ubatch_plan_trace(plan);
        write_expert_use_plan_trace(plan, decode_start_ts_us);
        update_expert_use_state(plan);
        g_active_physical_ubatches.push_back(make_physical_runtime(plan));
        if (g_config.cross_ubatch_layer_prefetch == 0 &&
                g_config.layer_frontier_prefetch == 0 &&
                plan.target_ubatch_id == source_ubatch_id) {
            submit_current_ubatch_prefetch(plan, true);
        }
    }
    append_rpp_timing(
        decode_call_id,
        "pre_decode_physical_plan",
        plan_start_ts_us,
        monotonic_us(),
        (int64_t) plans.size(),
        logical_batch_n_tokens,
        "");
    const int64_t reclaim_start_ts_us = monotonic_us();
    plan_reclaim_priority_trace(plans, decode_call_id, source_ubatch_id);
    append_rpp_timing(
        decode_call_id,
        "pre_decode_reclaim_plan",
        reclaim_start_ts_us,
        monotonic_us(),
        (int64_t) plans.size(),
        logical_batch_n_tokens,
        "");
    if (decode_call_id % 16 == 0) {
        append_expert_state_trace(decode_call_id, source_ubatch_id);
    }

    const bool all_oracle_samples_registered =
        !g_oracle.empty() && g_samples.size() >= g_oracle.size();
    if (!all_oracle_samples_registered) {
        g_pending_predicted_order.clear();
        g_queue.clear();
        g_counters.clear();
        write_summary();
        return decode_call_id;
    }

    const int64_t prediction_cycle_start_ts_us = monotonic_us();
    run_prediction_cycle(batch, slots, runtime_n_batch, runtime_n_ubatch, decode_call_id, false);
    append_rpp_timing(
        decode_call_id,
        "pre_decode_prediction_cycle",
        prediction_cycle_start_ts_us,
        monotonic_us(),
        0,
        logical_batch_n_tokens,
        "");
    write_summary();
    return decode_call_id;
}

void rpp_live_post_decode(
        int64_t decode_call_id,
        int32_t logical_batch_offset,
        int32_t logical_batch_n_tokens,
        int32_t runtime_n_ubatch,
        int32_t decode_ret) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) {
        return;
    }
    const int32_t n_ubatch = std::max<int32_t>(1, runtime_n_ubatch);
    const int32_t current_ubatch_id = logical_batch_offset / n_ubatch;
    append_line(
        "ubatch_decode_end_trace.csv",
        std::to_string(decode_call_id) + "," +
        std::to_string(current_ubatch_id) + "," +
        std::to_string(logical_batch_offset) + "," +
        std::to_string(logical_batch_n_tokens) + "," +
        std::to_string(n_ubatch) + "," +
        std::to_string(monotonic_us()) + "," +
        std::to_string(decode_ret));
    ++g_ubatch_decode_end_rows;
    if (g_eval_decode_call_id == decode_call_id) {
        g_eval_decode_call_id = -1;
        g_eval_current_ubatch = -1;
        g_eval_last_layer = -1;
    }
}

void rpp_live_pre_batch(
        const std::vector<rpp_live_slot_info> & slots,
        int32_t runtime_n_batch,
        int32_t runtime_n_ubatch) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!ensure_loaded_or_warned()) {
        return;
    }
    const int64_t event_id = g_decode_call_id.load();
    write_slot_state_trace(slots, event_id);
    llama_batch empty_batch {};
    run_prediction_cycle(empty_batch, slots, runtime_n_batch, runtime_n_ubatch, event_id, false);
    write_summary();
}
