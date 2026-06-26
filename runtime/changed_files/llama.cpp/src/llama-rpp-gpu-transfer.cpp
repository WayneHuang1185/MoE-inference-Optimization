#include "llama-rpp-gpu-transfer.h"

#include "ggml.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>

namespace {

using json = nlohmann::json;

constexpr size_t RPP_MAX_GPU_PENDING_JOBS = 4096;
constexpr size_t RPP_MAX_GPU_JOB_RECORDS = 65536;

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace

const char * llama_rpp_gpu_transfer_state_name(llama_rpp_gpu_transfer_state state) {
    switch (state) {
        case llama_rpp_gpu_transfer_state::missing: return "missing";
        case llama_rpp_gpu_transfer_state::queued:  return "queued";
        case llama_rpp_gpu_transfer_state::running: return "running";
        case llama_rpp_gpu_transfer_state::done:    return "done";
        case llama_rpp_gpu_transfer_state::failed:  return "failed";
    }
    return "unknown";
}

llama_rpp_gpu_transfer::llama_rpp_gpu_transfer() = default;

llama_rpp_gpu_transfer::~llama_rpp_gpu_transfer() {
    stop();
}

bool llama_rpp_gpu_transfer::configure(
        llama_rpp_host_prefetcher * source,
        uint64_t cache_bytes,
        uint64_t staging_bytes,
        const std::string & completion_trace_path,
        std::string * error) {
    stop();
    if (!source || !source->enabled()) {
        if (error) {
            *error = "GPU transfer requires a configured GGUF expert data source";
        }
        return false;
    }
    if (cache_bytes == 0 || staging_bytes == 0) {
        if (error) {
            *error = "GPU cache and staging sizes must be positive";
        }
        return false;
    }

    device_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device_) {
        if (error) {
            *error = "no GGML GPU device is available";
        }
        return false;
    }

    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device_, &props);
    device_name_ = props.description ? props.description : ggml_backend_dev_name(device_);
    if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
        if (error) {
            *error = "GPU device lacks async, pinned-host-buffer, or event support";
        }
        return false;
    }

    backend_ = ggml_backend_dev_init(device_, nullptr);
    if (!backend_) {
        if (error) {
            *error = "failed to initialize GGML GPU transfer backend";
        }
        stop();
        return false;
    }

    const ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(device_);
    const ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device_);
    if (!gpu_buft || !host_buft) {
        if (error) {
            *error = "GPU device does not expose required buffer types";
        }
        stop();
        return false;
    }

    const ggml_init_params params = {
        ggml_tensor_overhead() + 1024,
        nullptr,
        true,
    };
    tensor_context_ = ggml_init(params);
    if (!tensor_context_) {
        if (error) {
            *error = "failed to create GPU cache tensor context";
        }
        stop();
        return false;
    }
    cache_tensor_ = ggml_new_tensor_1d(
            tensor_context_, GGML_TYPE_I8, static_cast<int64_t>(cache_bytes));
    cache_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(tensor_context_, gpu_buft);
    staging_buffer_ = ggml_backend_buft_alloc_buffer(host_buft, staging_bytes);
    event_ = ggml_backend_event_new(device_);
    if (!cache_buffer_ || !staging_buffer_ || !event_) {
        if (error) {
            *error = "failed to allocate GPU cache, pinned staging buffer, or event";
        }
        stop();
        return false;
    }

    staging_data_ = static_cast<uint8_t *>(ggml_backend_buffer_get_base(staging_buffer_));
    cache_capacity_ = cache_bytes;
    staging_capacity_ = staging_bytes;
    cache_alignment_ = std::max<uint64_t>(1, ggml_backend_get_alignment(backend_));
    source_ = source;
    if (!completion_trace_path.empty()) {
        completion_trace_.open(completion_trace_path, std::ios::out | std::ios::trunc);
        if (!completion_trace_.is_open()) {
            if (error) {
                *error = "failed to open GPU transfer completion trace";
            }
            stop();
            return false;
        }
    }
    stopping_ = false;
    worker_ = std::thread(&llama_rpp_gpu_transfer::worker_loop, this);
    return true;
}

void llama_rpp_gpu_transfer::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }

    if (event_) {
        ggml_backend_event_synchronize(event_);
        ggml_backend_event_free(event_);
    }
    if (staging_buffer_) {
        ggml_backend_buffer_free(staging_buffer_);
    }
    if (cache_buffer_) {
        ggml_backend_buffer_free(cache_buffer_);
    }
    if (tensor_context_) {
        ggml_free(tensor_context_);
    }
    if (backend_) {
        ggml_backend_free(backend_);
    }

    source_ = nullptr;
    completion_trace_.close();
    device_ = nullptr;
    backend_ = nullptr;
    event_ = nullptr;
    cache_buffer_ = nullptr;
    staging_buffer_ = nullptr;
    tensor_context_ = nullptr;
    cache_tensor_ = nullptr;
    staging_data_ = nullptr;
    cache_capacity_ = 0;
    cache_cursor_ = 0;
    cache_used_ = 0;
    staging_capacity_ = 0;
    cache_alignment_ = 1;
    device_name_.clear();

    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
    records_.clear();
    active_ = false;
    stopping_ = false;
}

bool llama_rpp_gpu_transfer::enabled() const {
    return backend_ != nullptr;
}

bool llama_rpp_gpu_transfer::enqueue(
        const llama_rpp_prefetch_key & key,
        const std::vector<int32_t> & experts) {
    if (!enabled() || experts.empty()) {
        return false;
    }

    llama_rpp_gpu_transfer_snapshot record;
    record.state = llama_rpp_gpu_transfer_state::queued;
    record.enqueue_us = now_us();
    record.cache_capacity = cache_capacity_;
    record.device = device_name_;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || records_.count(key) != 0) {
            return false;
        }
        if (records_.size() >= RPP_MAX_GPU_JOB_RECORDS) {
            for (auto it = records_.begin();
                    it != records_.end() && records_.size() >= RPP_MAX_GPU_JOB_RECORDS;) {
                if (it->second.state == llama_rpp_gpu_transfer_state::done ||
                        it->second.state == llama_rpp_gpu_transfer_state::failed) {
                    it = records_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (queue_.size() >= RPP_MAX_GPU_PENDING_JOBS) {
            record.state = llama_rpp_gpu_transfer_state::failed;
            record.complete_us = record.enqueue_us;
            record.error = "GPU transfer queue capacity exceeded";
            records_.emplace(key, std::move(record));
            return false;
        }
        records_.emplace(key, std::move(record));
        queue_.push_back({key, experts});
    }
    work_cv_.notify_one();
    return true;
}

llama_rpp_gpu_transfer_snapshot llama_rpp_gpu_transfer::snapshot(
        const llama_rpp_prefetch_key & key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(key);
    return it == records_.end() ? llama_rpp_gpu_transfer_snapshot{} : it->second;
}

bool llama_rpp_gpu_transfer::wait_idle(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
        return queue_.empty() && !active_;
    });
}

uint64_t llama_rpp_gpu_transfer::reserve_cache_range(uint64_t bytes) {
    const uint64_t aligned = align_up(cache_cursor_, cache_alignment_);
    if (aligned + bytes > cache_capacity_) {
        cache_cursor_ = bytes;
        return 0;
    }
    cache_cursor_ = aligned + bytes;
    return aligned;
}

void llama_rpp_gpu_transfer::worker_loop() {
    while (true) {
        job current;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this] {
                return stopping_ || !queue_.empty();
            });
            if (stopping_ && queue_.empty()) {
                return;
            }
            current = std::move(queue_.front());
            queue_.pop_front();
            active_ = true;
            auto & record = records_.at(current.key);
            record.state = llama_rpp_gpu_transfer_state::running;
            record.start_us = now_us();
        }

        llama_rpp_expert_pack_result pack;
        std::string error;
        if (!source_->pack_experts(
                    current.key.layer,
                    current.experts,
                    staging_data_,
                    staging_capacity_,
                    &pack,
                    &error)) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto & record = records_.at(current.key);
            record.state = llama_rpp_gpu_transfer_state::failed;
            record.complete_us = now_us();
            record.error = error;
            active_ = false;
            if (queue_.empty()) {
                idle_cv_.notify_all();
            }
            if (completion_trace_.is_open()) {
                completion_trace_ << json({
                    {"request_id", current.key.request_id},
                    {"position", current.key.token_position},
                    {"phase", llama_rpp_phase_name(current.key.phase)},
                    {"layer", current.key.layer},
                    {"predicted_experts", current.experts},
                    {"state", llama_rpp_gpu_transfer_state_name(record.state)},
                    {"bytes", 0},
                    {"pack_us", 0},
                    {"h2d_us", 0},
                    {"queue_us", record.start_us - record.enqueue_us},
                    {"total_us", record.complete_us - record.enqueue_us},
                    {"cache_offset", 0},
                    {"cache_capacity", record.cache_capacity},
                    {"cache_used", record.cache_used},
                    {"device", record.device},
                    {"error", record.error},
                }).dump() << '\n';
            }
            continue;
        }

        uint64_t cache_offset = 0;
        if (pack.bytes > cache_capacity_) {
            error = "packed experts exceed GPU dummy cache capacity";
        } else {
            cache_offset = reserve_cache_range(pack.bytes);
            const int64_t transfer_start_us = now_us();
            ggml_backend_tensor_set_async(
                    backend_,
                    cache_tensor_,
                    staging_data_,
                    cache_offset,
                    pack.bytes);
            ggml_backend_event_record(event_, backend_);
            ggml_backend_event_synchronize(event_);

            std::lock_guard<std::mutex> lock(mutex_);
            auto & record = records_.at(current.key);
            record.bytes = pack.bytes;
            record.cache_offset = cache_offset;
            record.cache_used = std::min<uint64_t>(
                    cache_capacity_, cache_used_ + pack.bytes);
            cache_used_ = record.cache_used;
            record.pack_us = pack.duration_us;
            record.transfer_us = now_us() - transfer_start_us;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto & record = records_.at(current.key);
            record.complete_us = now_us();
            record.error = error;
            record.state = error.empty()
                ? llama_rpp_gpu_transfer_state::done
                : llama_rpp_gpu_transfer_state::failed;
            active_ = false;
            if (queue_.empty()) {
                idle_cv_.notify_all();
            }
            if (completion_trace_.is_open()) {
                completion_trace_ << json({
                    {"request_id", current.key.request_id},
                    {"position", current.key.token_position},
                    {"phase", llama_rpp_phase_name(current.key.phase)},
                    {"layer", current.key.layer},
                    {"predicted_experts", current.experts},
                    {"state", llama_rpp_gpu_transfer_state_name(record.state)},
                    {"bytes", record.bytes},
                    {"pack_us", record.pack_us},
                    {"h2d_us", record.transfer_us},
                    {"queue_us", record.start_us - record.enqueue_us},
                    {"total_us", record.complete_us - record.enqueue_us},
                    {"cache_offset", record.cache_offset},
                    {"cache_capacity", record.cache_capacity},
                    {"cache_used", record.cache_used},
                    {"device", record.device},
                    {"error", record.error},
                }).dump() << '\n';
            }
        }
    }
}
