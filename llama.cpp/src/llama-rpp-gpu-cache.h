#pragma once

#include "llama-rpp-prefetch.h"

#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

struct llama_model;

struct llama_rpp_expert_key {
    int32_t layer = -1;
    int32_t expert = -1;

    bool operator<(const llama_rpp_expert_key & other) const {
        return layer != other.layer ? layer < other.layer : expert < other.expert;
    }
};

enum class llama_rpp_gpu_cache_entry_state {
    missing,
    queued,
    loading,
    ready,
    failed,
};

struct llama_rpp_gpu_cache_location {
    int32_t expert = -1;
    int32_t slot = -1;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct llama_rpp_gpu_expert_timing {
    int32_t expert = -1;
    bool selected_for_prefetch = false;
    bool evicted_before_use = false;
    std::string state_at_use;
    int64_t enqueue_us = 0;
    int64_t start_us = 0;
    int64_t complete_us = 0;
    int64_t use_us = 0;
};

struct llama_rpp_gpu_correction_snapshot {
    bool attempted = false;
    bool success = false;
    int32_t requested = 0;
    int32_t ready_hits = 0;
    int32_t waited_prefetch = 0;
    int32_t loaded_on_demand = 0;
    int32_t failed = 0;
    int32_t evictions = 0;
    int32_t selected_for_prefetch = 0;
    int32_t prefetch_ready_at_use = 0;
    int32_t prefetch_queued_at_use = 0;
    int32_t prefetch_loading_at_use = 0;
    int32_t prefetch_absent_at_use = 0;
    int32_t prefetch_evicted_before_use = 0;
    uint64_t correction_bytes = 0;
    int64_t correction_us = 0;
    int32_t resident_entries = 0;
    int32_t cache_slots = 0;
    uint64_t slot_stride = 0;
    std::vector<llama_rpp_gpu_cache_location> locations;
    std::vector<llama_rpp_gpu_expert_timing> expert_timings;
    std::string error;
};

enum class llama_rpp_gpu_queue_policy {
    fifo,
    deadline,
};

class llama_rpp_gpu_cache {
public:
    llama_rpp_gpu_cache();
    ~llama_rpp_gpu_cache();

    bool configure(
            llama_rpp_host_prefetcher * source,
            const llama_model * model,
            uint64_t cache_bytes,
            uint64_t staging_bytes,
            int32_t copy_workers,
            llama_rpp_gpu_queue_policy queue_policy,
            std::string * error);

    void stop();
    bool enabled() const;

    void prefetch(
            const llama_rpp_prefetch_key & event_key,
            int32_t layer,
            const std::vector<int32_t> & experts);

    llama_rpp_gpu_correction_snapshot ensure_resident(
            const llama_rpp_prefetch_key & event_key,
            int32_t layer,
            const std::vector<int32_t> & experts,
            bool keep_protected = false);

    void release_layer(int32_t layer);
    int32_t slot_for(int32_t layer, int32_t expert) const;

    ggml_tensor * gate_up_tensor() const;
    ggml_tensor * gate_tensor(int32_t layer) const;
    ggml_tensor * up_tensor(int32_t layer) const;
    ggml_tensor * down_tensor(int32_t layer) const;
    ggml_tensor * down_scale_tensor() const;
    int32_t slot_count() const;

    llama_rpp_gpu_correction_snapshot snapshot(
            const llama_rpp_prefetch_key & event_key) const;

    bool wait_idle(int64_t timeout_ms);

private:
    struct entry {
        llama_rpp_gpu_cache_entry_state state =
            llama_rpp_gpu_cache_entry_state::missing;
        int32_t slot = -1;
        uint64_t bytes = 0;
        uint64_t last_use = 0;
        uint64_t generation = 0;
        int64_t enqueue_us = 0;
        int64_t start_us = 0;
        int64_t complete_us = 0;
        std::string error;
    };

    struct prefetch_observation {
        uint64_t generation = 0;
        int64_t enqueue_us = 0;
    };

    struct job {
        llama_rpp_expert_key key;
        bool high_priority = false;
        uint64_t order = 0;
    };

    struct worker_resources {
        ggml_backend_t backend = nullptr;
        ggml_backend_event_t event = nullptr;
        ggml_backend_buffer_t staging_buffer = nullptr;
        uint8_t * staging_data = nullptr;
        std::thread thread;
    };

    enum class expert_cache_layout {
        none,
        merged_gate_up,
        separate_gate_up,
    };

    bool enqueue_locked(const llama_rpp_expert_key & key, bool high_priority);
    void promote_locked(const llama_rpp_expert_key & key);
    job pop_job_locked();
    int32_t reserve_slot_locked(const llama_rpp_expert_key & incoming);
    bool all_ready_or_failed_locked(
            const std::vector<llama_rpp_expert_key> & keys) const;
    int32_t resident_count_locked() const;
    void worker_loop(int32_t worker_index);

    llama_rpp_host_prefetcher * source_ = nullptr;

    ggml_backend_dev_t device_ = nullptr;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    ggml_context * tensor_context_ = nullptr;
    ggml_tensor * gate_up_tensor_ = nullptr;
    std::map<ggml_type, ggml_tensor *> gate_tensors_;
    std::map<ggml_type, ggml_tensor *> up_tensors_;
    std::map<ggml_type, ggml_tensor *> down_tensors_;
    ggml_tensor * down_scale_tensor_ = nullptr;
    std::vector<ggml_type> layer_gate_types_;
    std::vector<ggml_type> layer_up_types_;
    std::vector<ggml_type> layer_down_types_;
    expert_cache_layout layout_ = expert_cache_layout::none;

    uint64_t cache_capacity_ = 0;
    uint64_t staging_capacity_ = 0;
    uint64_t slot_stride_ = 0;
    int32_t slot_count_ = 0;
    uint64_t clock_ = 0;
    uint64_t generation_ = 0;
    uint64_t enqueue_order_ = 0;
    int32_t eviction_count_ = 0;
    int32_t active_workers_ = 0;
    llama_rpp_gpu_queue_policy queue_policy_ = llama_rpp_gpu_queue_policy::fifo;

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable state_cv_;
    std::condition_variable idle_cv_;
    std::deque<job> queue_;
    std::map<llama_rpp_expert_key, entry> entries_;
    std::vector<llama_rpp_expert_key> slot_keys_;
    std::vector<bool> slot_occupied_;
    std::set<llama_rpp_expert_key> protected_;
    std::map<
        llama_rpp_prefetch_key,
        std::map<llama_rpp_expert_key, prefetch_observation>> prefetch_observations_;
    std::map<llama_rpp_prefetch_key, llama_rpp_gpu_correction_snapshot> corrections_;
    std::vector<worker_resources> workers_;
    bool active_ = false;
    bool stopping_ = false;
};
