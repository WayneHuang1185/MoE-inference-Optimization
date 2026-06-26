#pragma once

#include "llama-rpp-types.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct llama_rpp_prefetch_key {
    int64_t request_id = -1;
    int64_t token_position = -1;
    enum llama_rpp_phase phase = LLAMA_RPP_PHASE_UNKNOWN;
    int32_t layer = -1;

    bool operator<(const llama_rpp_prefetch_key & other) const {
        if (request_id != other.request_id) {
            return request_id < other.request_id;
        }
        if (token_position != other.token_position) {
            return token_position < other.token_position;
        }
        if (phase != other.phase) {
            return static_cast<int>(phase) < static_cast<int>(other.phase);
        }
        return layer < other.layer;
    }
};

enum class llama_rpp_prefetch_state {
    missing,
    queued,
    running,
    done,
    failed,
};

struct llama_rpp_prefetch_snapshot {
    llama_rpp_prefetch_state state = llama_rpp_prefetch_state::missing;
    uint64_t pages = 0;
    uint64_t bytes = 0;
    int64_t enqueue_us = 0;
    int64_t start_us = 0;
    int64_t complete_us = 0;
    int64_t duration_us = 0;
    int64_t minor_faults = 0;
    int64_t major_faults = 0;
    std::string error;
};

struct llama_rpp_expert_pack_result {
    uint64_t bytes = 0;
    uint64_t ranges = 0;
    int64_t duration_us = 0;
};

enum class llama_rpp_expert_component {
    gate_up_weight,
    down_weight,
    down_scale,
};

const char * llama_rpp_prefetch_state_name(llama_rpp_prefetch_state state);

class llama_rpp_host_prefetcher {
public:
    llama_rpp_host_prefetcher();
    ~llama_rpp_host_prefetcher();

    bool configure(
            const std::string & model_path,
            const std::string & page_map_path,
            int32_t n_threads,
            std::string * error);

    void stop();

    bool enabled() const;

    bool enqueue(
            const llama_rpp_prefetch_key & key,
            const std::vector<int32_t> & experts);

    llama_rpp_prefetch_snapshot snapshot(const llama_rpp_prefetch_key & key) const;

    bool wait_idle(int64_t timeout_ms);

    bool pack_experts(
            int32_t layer,
            const std::vector<int32_t> & experts,
            void * destination,
            size_t capacity,
            llama_rpp_expert_pack_result * result,
            std::string * error) const;

    bool pack_expert_component(
            int32_t layer,
            int32_t expert,
            llama_rpp_expert_component component,
            void * destination,
            size_t capacity,
            uint64_t * bytes,
            std::string * error) const;

    uint64_t component_bytes(
            int32_t layer,
            int32_t expert,
            llama_rpp_expert_component component) const;
    uint64_t expert_bytes(int32_t layer, int32_t expert) const;
    uint64_t max_expert_bytes() const;

private:
    struct byte_range {
        uint64_t begin = 0;
        uint64_t end = 0;
    };

    struct component_key {
        int32_t layer = -1;
        int32_t expert = -1;
        llama_rpp_expert_component component =
            llama_rpp_expert_component::gate_up_weight;

        bool operator<(const component_key & other) const {
            if (layer != other.layer) {
                return layer < other.layer;
            }
            if (expert != other.expert) {
                return expert < other.expert;
            }
            return static_cast<int>(component) < static_cast<int>(other.component);
        }
    };

    struct job {
        llama_rpp_prefetch_key key;
        std::vector<byte_range> ranges;
    };

    struct job_record {
        llama_rpp_prefetch_snapshot snapshot;
    };

    bool load_page_map(const std::string & path, std::string * error);
    std::vector<byte_range> ranges_for(
            int32_t layer,
            const std::vector<int32_t> & experts) const;
    void worker_loop();

    int fd_ = -1;
    const uint8_t * mapping_ = nullptr;
    uint64_t mapping_size_ = 0;
    uint64_t page_size_ = 4096;
    uint64_t max_expert_bytes_ = 0;

    std::map<std::pair<int32_t, int32_t>, std::vector<byte_range>> expert_ranges_;
    std::map<component_key, byte_range> component_ranges_;

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable idle_cv_;
    std::deque<job> queue_;
    std::map<llama_rpp_prefetch_key, job_record> records_;
    std::vector<std::thread> workers_;
    size_t active_workers_ = 0;
    bool stopping_ = false;
};
