#pragma once

#include "llama-rpp-prefetch.h"

#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

enum class llama_rpp_gpu_transfer_state {
    missing,
    queued,
    running,
    done,
    failed,
};

struct llama_rpp_gpu_transfer_snapshot {
    llama_rpp_gpu_transfer_state state = llama_rpp_gpu_transfer_state::missing;
    uint64_t bytes = 0;
    uint64_t cache_offset = 0;
    uint64_t cache_capacity = 0;
    uint64_t cache_used = 0;
    int64_t enqueue_us = 0;
    int64_t start_us = 0;
    int64_t complete_us = 0;
    int64_t pack_us = 0;
    int64_t transfer_us = 0;
    std::string device;
    std::string error;
};

const char * llama_rpp_gpu_transfer_state_name(llama_rpp_gpu_transfer_state state);

class llama_rpp_gpu_transfer {
public:
    llama_rpp_gpu_transfer();
    ~llama_rpp_gpu_transfer();

    bool configure(
            llama_rpp_host_prefetcher * source,
            uint64_t cache_bytes,
            uint64_t staging_bytes,
            const std::string & completion_trace_path,
            std::string * error);

    void stop();
    bool enabled() const;

    bool enqueue(
            const llama_rpp_prefetch_key & key,
            const std::vector<int32_t> & experts);

    llama_rpp_gpu_transfer_snapshot snapshot(const llama_rpp_prefetch_key & key) const;
    bool wait_idle(int64_t timeout_ms);

private:
    struct job {
        llama_rpp_prefetch_key key;
        std::vector<int32_t> experts;
    };

    void worker_loop();
    uint64_t reserve_cache_range(uint64_t bytes);

    llama_rpp_host_prefetcher * source_ = nullptr;

    ggml_backend_dev_t device_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_event_t event_ = nullptr;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    ggml_backend_buffer_t staging_buffer_ = nullptr;
    ggml_context * tensor_context_ = nullptr;
    ggml_tensor * cache_tensor_ = nullptr;
    uint8_t * staging_data_ = nullptr;

    uint64_t cache_capacity_ = 0;
    uint64_t cache_cursor_ = 0;
    uint64_t cache_used_ = 0;
    uint64_t staging_capacity_ = 0;
    uint64_t cache_alignment_ = 1;
    std::string device_name_;

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable idle_cv_;
    std::deque<job> queue_;
    std::map<llama_rpp_prefetch_key, llama_rpp_gpu_transfer_snapshot> records_;
    std::thread worker_;
    std::ofstream completion_trace_;
    bool active_ = false;
    bool stopping_ = false;
};
