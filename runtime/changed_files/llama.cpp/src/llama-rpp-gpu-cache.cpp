#include "llama-rpp-gpu-cache.h"

#include "ggml.h"
#include "llama-model.h"

#include <algorithm>
#include <chrono>
#include <limits>

namespace {

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

const char * entry_state_name(llama_rpp_gpu_cache_entry_state state) {
    switch (state) {
        case llama_rpp_gpu_cache_entry_state::missing: return "missing";
        case llama_rpp_gpu_cache_entry_state::queued:  return "queued";
        case llama_rpp_gpu_cache_entry_state::loading: return "loading";
        case llama_rpp_gpu_cache_entry_state::ready:   return "ready";
        case llama_rpp_gpu_cache_entry_state::failed:  return "failed";
    }
    return "unknown";
}

struct backend_guard {
    ggml_backend_t backend = nullptr;

    explicit backend_guard(ggml_backend_t backend) : backend(backend) {}
    ~backend_guard() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    ggml_backend_t release() {
        ggml_backend_t result = backend;
        backend = nullptr;
        return result;
    }
};

} // namespace

llama_rpp_gpu_cache::llama_rpp_gpu_cache() = default;

llama_rpp_gpu_cache::~llama_rpp_gpu_cache() {
    stop();
}

bool llama_rpp_gpu_cache::configure(
        llama_rpp_host_prefetcher * source,
        const llama_model * model,
        uint64_t cache_bytes,
        uint64_t staging_bytes,
        int32_t copy_workers,
        llama_rpp_gpu_queue_policy queue_policy,
        std::string * error) {
    stop();
    if (!source || !source->enabled() || !model) {
        if (error) {
            *error = "GPU correction cache requires a configured GGUF expert data source";
        }
        return false;
    }
    if (copy_workers <= 0) {
        if (error) {
            *error = "GPU copy worker count must be positive";
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
    if (model->layers.empty()) {
        if (error) {
            *error = "GPU correction cache requires a model with MoE layers";
        }
        stop();
        return false;
    }
    const auto & layer0 = model->layers.front();
    const bool has_merged_gate_up =
        layer0.ffn_gate_up_exps && layer0.ffn_down_exps &&
        layer0.ffn_down_exps_s;
    const bool has_separate_gate_up =
        layer0.ffn_gate_exps && layer0.ffn_up_exps && layer0.ffn_down_exps;
    if (has_merged_gate_up) {
        layout_ = expert_cache_layout::merged_gate_up;
    } else if (has_separate_gate_up) {
        layout_ = expert_cache_layout::separate_gate_up;
    } else {
        if (error) {
            *error = "unsupported MoE expert tensor layout for RPP GPU cache";
        }
        stop();
        return false;
    }

    uint64_t gate_up_bytes = 0;
    uint64_t scale_bytes = 0;
    std::map<ggml_type, const ggml_tensor *> gate_prototypes;
    std::map<ggml_type, const ggml_tensor *> up_prototypes;
    std::map<ggml_type, const ggml_tensor *> down_prototypes;
    layer_gate_types_.reserve(model->layers.size());
    layer_up_types_.reserve(model->layers.size());
    layer_down_types_.reserve(model->layers.size());
    for (size_t layer = 0; layer < model->layers.size(); ++layer) {
        const auto & model_layer = model->layers[layer];
        if (layout_ == expert_cache_layout::merged_gate_up) {
            if (!model_layer.ffn_gate_up_exps || !model_layer.ffn_down_exps ||
                    !model_layer.ffn_down_exps_s) {
                if (error) {
                    *error = "merged gate_up MoE layer has incomplete expert tensors";
                }
                stop();
                return false;
            }
            if (model_layer.ffn_gate_up_exps->type != layer0.ffn_gate_up_exps->type ||
                    model_layer.ffn_gate_up_exps->ne[0] != layer0.ffn_gate_up_exps->ne[0] ||
                    model_layer.ffn_gate_up_exps->ne[1] != layer0.ffn_gate_up_exps->ne[1] ||
                    model_layer.ffn_down_exps_s->type != layer0.ffn_down_exps_s->type) {
                if (error) {
                    *error = "merged gate_up/scale cache layout varies by layer";
                }
                stop();
                return false;
            }
            const uint64_t mapped_gate_up = source->component_bytes(
                    static_cast<int32_t>(layer), 0,
                    llama_rpp_expert_component::gate_up_weight);
            const uint64_t mapped_down = source->component_bytes(
                    static_cast<int32_t>(layer), 0,
                    llama_rpp_expert_component::down_weight);
            const uint64_t mapped_scale = source->component_bytes(
                    static_cast<int32_t>(layer), 0,
                    llama_rpp_expert_component::down_scale);
            if (mapped_gate_up != model_layer.ffn_gate_up_exps->nb[2] ||
                    mapped_down != model_layer.ffn_down_exps->nb[2] ||
                    mapped_scale != model_layer.ffn_down_exps_s->nb[0]) {
                if (error) {
                    *error = "page-map component sizes do not match merged gate_up tensor layout";
                }
                stop();
                return false;
            }
            if (layer == 0) {
                gate_up_bytes = mapped_gate_up;
                scale_bytes = mapped_scale;
            }
        } else {
            if (!model_layer.ffn_gate_exps || !model_layer.ffn_up_exps ||
                    !model_layer.ffn_down_exps) {
                if (error) {
                    *error = "separate gate/up MoE layer has incomplete expert tensors";
                }
                stop();
                return false;
            }
            if (model_layer.ffn_gate_exps_s || model_layer.ffn_up_exps_s ||
                    model_layer.ffn_down_exps_s) {
                if (error) {
                    *error = "separate MoE expert scale tensors are not supported by RPP GPU cache yet";
                }
                stop();
                return false;
            }
            const uint64_t mapped_gate = source->component_bytes(
                    static_cast<int32_t>(layer), 0,
                    llama_rpp_expert_component::gate_weight);
            const uint64_t mapped_up = source->component_bytes(
                    static_cast<int32_t>(layer), 0,
                    llama_rpp_expert_component::up_weight);
            const uint64_t mapped_down = source->component_bytes(
                    static_cast<int32_t>(layer), 0,
                    llama_rpp_expert_component::down_weight);
            if (mapped_gate != model_layer.ffn_gate_exps->nb[2] ||
                    mapped_up != model_layer.ffn_up_exps->nb[2] ||
                    mapped_down != model_layer.ffn_down_exps->nb[2]) {
                if (error) {
                    *error = "page-map component sizes do not match separate gate/up tensor layout";
                }
                stop();
                return false;
            }
            layer_gate_types_.push_back(model_layer.ffn_gate_exps->type);
            layer_up_types_.push_back(model_layer.ffn_up_exps->type);
            gate_prototypes.emplace(
                    model_layer.ffn_gate_exps->type,
                    model_layer.ffn_gate_exps);
            up_prototypes.emplace(
                    model_layer.ffn_up_exps->type,
                    model_layer.ffn_up_exps);
        }
        layer_down_types_.push_back(model_layer.ffn_down_exps->type);
        down_prototypes.emplace(
                model_layer.ffn_down_exps->type,
                model_layer.ffn_down_exps);
    }
    backend_guard setup_backend(ggml_backend_dev_init(device_, nullptr));
    if (!setup_backend.backend) {
        if (error) {
            *error = "failed to initialize GPU correction cache backend";
        }
        stop();
        return false;
    }
    const uint64_t alignment = std::max<uint64_t>(
            1, ggml_backend_get_alignment(setup_backend.backend));
    uint64_t storage_per_slot = 0;
    if (layout_ == expert_cache_layout::merged_gate_up) {
        storage_per_slot += gate_up_bytes + scale_bytes;
    } else {
        for (const auto & item : gate_prototypes) {
            storage_per_slot += item.second->nb[2];
        }
        for (const auto & item : up_prototypes) {
            storage_per_slot += item.second->nb[2];
        }
    }
    for (const auto & item : down_prototypes) {
        storage_per_slot += item.second->nb[2];
    }
    slot_stride_ = align_up(storage_per_slot, alignment);
    if (slot_stride_ == 0 || staging_bytes < slot_stride_) {
        if (error) {
            *error = "GPU staging buffer is smaller than one expert bundle";
        }
        stop();
        return false;
    }
    slot_count_ = static_cast<int32_t>(cache_bytes / slot_stride_);
    if (slot_count_ < 8) {
        if (error) {
            *error = "GPU correction cache must hold at least 8 expert bundles";
        }
        stop();
        return false;
    }
    cache_capacity_ = static_cast<uint64_t>(slot_count_) * slot_stride_;
    staging_capacity_ = staging_bytes;

    const auto gpu_buft = ggml_backend_dev_buffer_type(device_);
    const auto host_buft = ggml_backend_dev_host_buffer_type(device_);
    if (!gpu_buft || !host_buft) {
        if (error) {
            *error = "GPU device does not expose cache or pinned host buffer types";
        }
        stop();
        return false;
    }

    const size_t cache_tensor_count =
        layout_ == expert_cache_layout::merged_gate_up
            ? 2 + down_prototypes.size()
            : gate_prototypes.size() + up_prototypes.size() + down_prototypes.size();
    const ggml_init_params params = {
        cache_tensor_count * ggml_tensor_overhead() + 1024,
        nullptr,
        true,
    };
    tensor_context_ = ggml_init(params);
    if (!tensor_context_) {
        if (error) {
            *error = "failed to create GPU correction cache tensor context";
        }
        stop();
        return false;
    }
    if (layout_ == expert_cache_layout::merged_gate_up) {
        gate_up_tensor_ = ggml_new_tensor_3d(
                tensor_context_,
                layer0.ffn_gate_up_exps->type,
                layer0.ffn_gate_up_exps->ne[0],
                layer0.ffn_gate_up_exps->ne[1],
                slot_count_);
        ggml_set_name(gate_up_tensor_, "rpp.merged.gate_up_cache");
    } else {
        for (const auto & item : gate_prototypes) {
            const auto * prototype = item.second;
            auto * tensor = ggml_new_tensor_3d(
                    tensor_context_,
                    prototype->type,
                    prototype->ne[0],
                    prototype->ne[1],
                    slot_count_);
            ggml_format_name(tensor, "rpp.separate.gate_cache.%s",
                    ggml_type_name(prototype->type));
            gate_tensors_[item.first] = tensor;
        }
        for (const auto & item : up_prototypes) {
            const auto * prototype = item.second;
            auto * tensor = ggml_new_tensor_3d(
                    tensor_context_,
                    prototype->type,
                    prototype->ne[0],
                    prototype->ne[1],
                    slot_count_);
            ggml_format_name(tensor, "rpp.separate.up_cache.%s",
                    ggml_type_name(prototype->type));
            up_tensors_[item.first] = tensor;
        }
    }
    for (const auto & item : down_prototypes) {
        const auto * prototype = item.second;
        auto * tensor = ggml_new_tensor_3d(
                tensor_context_,
                prototype->type,
                prototype->ne[0],
                prototype->ne[1],
                slot_count_);
        ggml_format_name(tensor, "rpp.%s.down_cache.%s",
                layout_ == expert_cache_layout::merged_gate_up ? "merged" : "separate",
                ggml_type_name(prototype->type));
        down_tensors_[item.first] = tensor;
    }
    if (layout_ == expert_cache_layout::merged_gate_up) {
        down_scale_tensor_ = ggml_new_tensor_1d(
                tensor_context_,
                layer0.ffn_down_exps_s->type,
                slot_count_);
        if (gate_up_tensor_->nb[2] != gate_up_bytes ||
                down_scale_tensor_->nb[0] != scale_bytes) {
            if (error) {
                *error = "page-map component sizes do not match merged gate_up tensor layout";
            }
            stop();
            return false;
        }
        ggml_set_name(down_scale_tensor_, "rpp.merged.down_scale_cache");
    }
    cache_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(tensor_context_, gpu_buft);
    if (!cache_buffer_) {
        if (error) {
            *error = "failed to allocate GPU correction cache resources";
        }
        stop();
        return false;
    }
    workers_.resize(copy_workers);
    for (int32_t i = 0; i < copy_workers; ++i) {
        auto & worker = workers_[i];
        worker.backend = i == 0 ? setup_backend.release() : ggml_backend_dev_init(device_, nullptr);
        worker.event = ggml_backend_event_new(device_);
        worker.staging_buffer = ggml_backend_buft_alloc_buffer(host_buft, staging_capacity_);
        if (!worker.backend || !worker.event || !worker.staging_buffer) {
            if (error) {
                *error = "failed to allocate GPU copy worker resources";
            }
            stop();
            return false;
        }
        worker.staging_data =
            static_cast<uint8_t *>(ggml_backend_buffer_get_base(worker.staging_buffer));
    }
    source_ = source;
    queue_policy_ = queue_policy;
    slot_keys_.resize(slot_count_);
    slot_occupied_.assign(slot_count_, false);
    stopping_ = false;
    for (int32_t i = 0; i < static_cast<int32_t>(workers_.size()); ++i) {
        workers_[i].thread = std::thread(&llama_rpp_gpu_cache::worker_loop, this, i);
    }
    return true;
}

void llama_rpp_gpu_cache::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();
    state_cv_.notify_all();
    for (auto & worker : workers_) {
        if (worker.thread.joinable()) {
            worker.thread.join();
        }
    }

    for (auto & worker : workers_) {
        if (worker.event) {
            ggml_backend_event_synchronize(worker.event);
            ggml_backend_event_free(worker.event);
        }
        if (worker.staging_buffer) {
            ggml_backend_buffer_free(worker.staging_buffer);
        }
        if (worker.backend) {
            ggml_backend_free(worker.backend);
        }
    }
    workers_.clear();
    if (cache_buffer_) {
        ggml_backend_buffer_free(cache_buffer_);
    }
    if (tensor_context_) {
        ggml_free(tensor_context_);
    }

    source_ = nullptr;
    device_ = nullptr;
    cache_buffer_ = nullptr;
    tensor_context_ = nullptr;
    gate_up_tensor_ = nullptr;
    gate_tensors_.clear();
    up_tensors_.clear();
    down_tensors_.clear();
    down_scale_tensor_ = nullptr;
    layer_gate_types_.clear();
    layer_up_types_.clear();
    layer_down_types_.clear();
    layout_ = expert_cache_layout::none;
    cache_capacity_ = 0;
    staging_capacity_ = 0;
    slot_stride_ = 0;
    slot_count_ = 0;
    clock_ = 0;
    generation_ = 0;
    enqueue_order_ = 0;
    eviction_count_ = 0;
    active_workers_ = 0;
    queue_policy_ = llama_rpp_gpu_queue_policy::fifo;

    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
    entries_.clear();
    slot_keys_.clear();
    slot_occupied_.clear();
    protected_.clear();
    prefetch_observations_.clear();
    corrections_.clear();
    active_ = false;
    stopping_ = false;
}

bool llama_rpp_gpu_cache::enabled() const {
    return cache_buffer_ != nullptr && !workers_.empty();
}

void llama_rpp_gpu_cache::prefetch(
        const llama_rpp_prefetch_key & event_key,
        int32_t layer,
        const std::vector<int32_t> & experts) {
    if (!enabled()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int32_t expert : experts) {
            const llama_rpp_expert_key key = {layer, expert};
            enqueue_locked(key, false);
            const auto it = entries_.find(key);
            if (it != entries_.end()) {
                prefetch_observations_[event_key][key] = {
                    it->second.generation,
                    it->second.enqueue_us,
                };
            }
        }
    }
    work_cv_.notify_one();
}

llama_rpp_gpu_correction_snapshot llama_rpp_gpu_cache::ensure_resident(
        const llama_rpp_prefetch_key & event_key,
        int32_t layer,
        const std::vector<int32_t> & experts,
        bool keep_protected) {
    llama_rpp_gpu_correction_snapshot result;
    result.attempted = true;
    result.requested = static_cast<int32_t>(experts.size());
    result.cache_slots = slot_count_;
    result.slot_stride = slot_stride_;
    const int64_t started = now_us();
    const int64_t use_us = started;

    std::vector<llama_rpp_expert_key> keys;
    keys.reserve(experts.size());
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const int32_t evictions_before = eviction_count_;
        for (int32_t expert : experts) {
            const llama_rpp_expert_key key = {layer, expert};
            keys.push_back(key);
            protected_.insert(key);
            auto it = entries_.find(key);
            const auto event_it = prefetch_observations_.find(event_key);
            const auto observation_it = event_it == prefetch_observations_.end()
                ? std::map<llama_rpp_expert_key, prefetch_observation>::const_iterator{}
                : event_it->second.find(key);
            const bool selected_for_prefetch =
                event_it != prefetch_observations_.end() &&
                observation_it != event_it->second.end();
            llama_rpp_gpu_expert_timing timing;
            timing.expert = expert;
            timing.selected_for_prefetch = selected_for_prefetch;
            timing.use_us = use_us;
            if (selected_for_prefetch) {
                ++result.selected_for_prefetch;
                timing.enqueue_us = observation_it->second.enqueue_us;
                timing.evicted_before_use =
                    it == entries_.end() ||
                    it->second.generation != observation_it->second.generation;
                if (timing.evicted_before_use) {
                    ++result.prefetch_evicted_before_use;
                }
            }
            if (it != entries_.end()) {
                timing.state_at_use = entry_state_name(it->second.state);
                timing.start_us = it->second.start_us;
                timing.complete_us = it->second.complete_us;
            } else {
                timing.state_at_use = "missing";
            }
            if (selected_for_prefetch) {
                if (timing.state_at_use == "ready") {
                    ++result.prefetch_ready_at_use;
                } else if (timing.state_at_use == "queued") {
                    ++result.prefetch_queued_at_use;
                } else if (timing.state_at_use == "loading") {
                    ++result.prefetch_loading_at_use;
                } else {
                    ++result.prefetch_absent_at_use;
                }
            }
            result.expert_timings.push_back(std::move(timing));

            if (it != entries_.end() &&
                    it->second.state == llama_rpp_gpu_cache_entry_state::ready) {
                ++result.ready_hits;
                it->second.last_use = ++clock_;
            } else if (it != entries_.end() &&
                    (it->second.state == llama_rpp_gpu_cache_entry_state::queued ||
                     it->second.state == llama_rpp_gpu_cache_entry_state::loading)) {
                ++result.waited_prefetch;
                if (it->second.state == llama_rpp_gpu_cache_entry_state::queued) {
                    promote_locked(key);
                }
            } else {
                if (enqueue_locked(key, true)) {
                    ++result.loaded_on_demand;
                    result.correction_bytes += source_->expert_bytes(layer, expert);
                }
            }
        }
        result.evictions = eviction_count_ - evictions_before;
        work_cv_.notify_one();
        state_cv_.wait(lock, [this, &keys] {
            return stopping_ || all_ready_or_failed_locked(keys);
        });

        for (const auto & key : keys) {
            const auto it = entries_.find(key);
            if (it == entries_.end() ||
                    it->second.state != llama_rpp_gpu_cache_entry_state::ready) {
                ++result.failed;
                if (result.error.empty()) {
                    result.error = it == entries_.end()
                        ? "expert disappeared from GPU cache"
                        : it->second.error;
                }
                continue;
            }
            it->second.last_use = ++clock_;
            result.locations.push_back({
                key.expert,
                it->second.slot,
                static_cast<uint64_t>(it->second.slot) * slot_stride_,
                it->second.bytes,
            });
            auto timing = std::find_if(
                    result.expert_timings.begin(),
                    result.expert_timings.end(),
                    [&key](const llama_rpp_gpu_expert_timing & item) {
                        return item.expert == key.expert;
                    });
            if (timing != result.expert_timings.end()) {
                timing->start_us = it->second.start_us;
                timing->complete_us = it->second.complete_us;
            }
        }
        if (!keep_protected) {
            for (const auto & key : keys) {
                protected_.erase(key);
            }
        }
        result.resident_entries = resident_count_locked();
        result.success = result.failed == 0 && !stopping_;
        result.correction_us = now_us() - started;
        corrections_[event_key] = result;
        prefetch_observations_.erase(event_key);
    }
    return result;
}

void llama_rpp_gpu_cache::release_layer(int32_t layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = protected_.begin(); it != protected_.end();) {
        if (it->layer == layer) {
            it = protected_.erase(it);
        } else {
            ++it;
        }
    }
    state_cv_.notify_all();
}

int32_t llama_rpp_gpu_cache::slot_for(int32_t layer, int32_t expert) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find({layer, expert});
    return it != entries_.end() &&
            it->second.state == llama_rpp_gpu_cache_entry_state::ready
        ? it->second.slot
        : -1;
}

ggml_tensor * llama_rpp_gpu_cache::gate_up_tensor() const {
    return gate_up_tensor_;
}

ggml_tensor * llama_rpp_gpu_cache::gate_tensor(int32_t layer) const {
    if (layer < 0 || layer >= static_cast<int32_t>(layer_gate_types_.size())) {
        return nullptr;
    }
    const auto it = gate_tensors_.find(layer_gate_types_[layer]);
    return it == gate_tensors_.end() ? nullptr : it->second;
}

ggml_tensor * llama_rpp_gpu_cache::up_tensor(int32_t layer) const {
    if (layer < 0 || layer >= static_cast<int32_t>(layer_up_types_.size())) {
        return nullptr;
    }
    const auto it = up_tensors_.find(layer_up_types_[layer]);
    return it == up_tensors_.end() ? nullptr : it->second;
}

ggml_tensor * llama_rpp_gpu_cache::down_tensor(int32_t layer) const {
    if (layer < 0 || layer >= static_cast<int32_t>(layer_down_types_.size())) {
        return nullptr;
    }
    const auto it = down_tensors_.find(layer_down_types_[layer]);
    return it == down_tensors_.end() ? nullptr : it->second;
}

ggml_tensor * llama_rpp_gpu_cache::down_scale_tensor() const {
    return down_scale_tensor_;
}

int32_t llama_rpp_gpu_cache::slot_count() const {
    return slot_count_;
}

llama_rpp_gpu_correction_snapshot llama_rpp_gpu_cache::snapshot(
        const llama_rpp_prefetch_key & event_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = corrections_.find(event_key);
    return it == corrections_.end()
        ? llama_rpp_gpu_correction_snapshot{}
        : it->second;
}

bool llama_rpp_gpu_cache::wait_idle(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
        return queue_.empty() && active_workers_ == 0;
    });
}

bool llama_rpp_gpu_cache::enqueue_locked(
        const llama_rpp_expert_key & key,
        bool high_priority) {
    auto existing = entries_.find(key);
    if (existing != entries_.end()) {
        if (existing->second.state == llama_rpp_gpu_cache_entry_state::ready) {
            existing->second.last_use = ++clock_;
            return false;
        }
        if (existing->second.state == llama_rpp_gpu_cache_entry_state::queued ||
                existing->second.state == llama_rpp_gpu_cache_entry_state::loading) {
            return false;
        }
    }

    const int32_t slot = reserve_slot_locked(key);
    if (slot < 0) {
        entry failed;
        failed.state = llama_rpp_gpu_cache_entry_state::failed;
        failed.error = "no evictable GPU expert cache slot";
        entries_[key] = std::move(failed);
        return false;
    }
    entry value;
    value.state = llama_rpp_gpu_cache_entry_state::queued;
    value.slot = slot;
    value.bytes = source_->expert_bytes(key.layer, key.expert);
    value.last_use = ++clock_;
    value.generation = ++generation_;
    value.enqueue_us = now_us();
    entries_[key] = std::move(value);
    queue_.push_back({key, high_priority, ++enqueue_order_});
    return true;
}

void llama_rpp_gpu_cache::promote_locked(const llama_rpp_expert_key & key) {
    auto queued = std::find_if(
            queue_.begin(), queue_.end(), [&key](const job & item) {
                return item.key.layer == key.layer && item.key.expert == key.expert;
            });
    if (queued == queue_.end()) {
        return;
    }
    queued->high_priority = true;
    queued->order = 0;
}

llama_rpp_gpu_cache::job llama_rpp_gpu_cache::pop_job_locked() {
    auto best = queue_.begin();
    if (queue_policy_ == llama_rpp_gpu_queue_policy::fifo) {
        const auto urgent = std::find_if(
                queue_.begin(), queue_.end(), [](const job & item) {
                    return item.high_priority;
                });
        if (urgent != queue_.end()) {
            best = urgent;
        }
    } else {
        for (auto it = std::next(queue_.begin()); it != queue_.end(); ++it) {
            if (it->high_priority != best->high_priority) {
                if (it->high_priority) {
                    best = it;
                }
                continue;
            }
            if (it->key.layer != best->key.layer) {
                if (it->key.layer < best->key.layer) {
                    best = it;
                }
                continue;
            }
            if (it->order < best->order) {
                best = it;
            }
        }
    }
    const job value = *best;
    queue_.erase(best);
    return value;
}

int32_t llama_rpp_gpu_cache::reserve_slot_locked(
        const llama_rpp_expert_key & incoming) {
    for (int32_t slot = 0; slot < slot_count_; ++slot) {
        if (!slot_occupied_[slot]) {
            slot_occupied_[slot] = true;
            slot_keys_[slot] = incoming;
            return slot;
        }
    }

    int32_t victim_slot = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (int32_t slot = 0; slot < slot_count_; ++slot) {
        const auto key = slot_keys_[slot];
        if (protected_.count(key) != 0) {
            continue;
        }
        const auto it = entries_.find(key);
        if (it == entries_.end()) {
            victim_slot = slot;
            break;
        }
        if ((it->second.state == llama_rpp_gpu_cache_entry_state::queued ||
                it->second.state == llama_rpp_gpu_cache_entry_state::ready ||
                it->second.state == llama_rpp_gpu_cache_entry_state::failed) &&
                it->second.last_use < oldest) {
            oldest = it->second.last_use;
            victim_slot = slot;
        }
    }
    if (victim_slot < 0) {
        return -1;
    }
    entries_.erase(slot_keys_[victim_slot]);
    slot_keys_[victim_slot] = incoming;
    ++eviction_count_;
    return victim_slot;
}

bool llama_rpp_gpu_cache::all_ready_or_failed_locked(
        const std::vector<llama_rpp_expert_key> & keys) const {
    for (const auto & key : keys) {
        const auto it = entries_.find(key);
        if (it == entries_.end()) {
            return false;
        }
        if (it->second.state != llama_rpp_gpu_cache_entry_state::ready &&
                it->second.state != llama_rpp_gpu_cache_entry_state::failed) {
            return false;
        }
    }
    return true;
}

int32_t llama_rpp_gpu_cache::resident_count_locked() const {
    int32_t count = 0;
    for (const auto & item : entries_) {
        count += item.second.state == llama_rpp_gpu_cache_entry_state::ready ? 1 : 0;
    }
    return count;
}

void llama_rpp_gpu_cache::worker_loop(int32_t worker_index) {
    auto & worker = workers_[worker_index];
    while (true) {
        job current;
        int32_t slot = -1;
        uint64_t expected_bytes = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this] {
                return stopping_ || !queue_.empty();
            });
            if (stopping_ && queue_.empty()) {
                return;
            }
            current = pop_job_locked();
            auto it = entries_.find(current.key);
            if (it == entries_.end() ||
                    it->second.state != llama_rpp_gpu_cache_entry_state::queued) {
                continue;
            }
            it->second.state = llama_rpp_gpu_cache_entry_state::loading;
            it->second.start_us = now_us();
            slot = it->second.slot;
            expected_bytes = it->second.bytes;
            ++active_workers_;
            active_ = true;
        }

        uint64_t gate_up_bytes = 0;
        uint64_t gate_bytes = 0;
        uint64_t up_bytes = 0;
        uint64_t down_bytes = 0;
        uint64_t scale_bytes = 0;
        std::string error;
        if (expected_bytes > 0 && layout_ == expert_cache_layout::merged_gate_up) {
            const bool packed = source_->pack_expert_component(
                        current.key.layer,
                        current.key.expert,
                        llama_rpp_expert_component::gate_up_weight,
                        worker.staging_data,
                        staging_capacity_,
                        &gate_up_bytes,
                        &error);
            if (packed) {
                ggml_backend_tensor_set_async(
                        worker.backend,
                        gate_up_tensor_,
                        worker.staging_data,
                        static_cast<uint64_t>(slot) * gate_up_tensor_->nb[2],
                        gate_up_bytes);
                ggml_backend_event_record(worker.event, worker.backend);
                ggml_backend_event_synchronize(worker.event);
            }
        } else if (expected_bytes > 0 &&
                layout_ == expert_cache_layout::separate_gate_up) {
            ggml_tensor * gate_tensor = this->gate_tensor(current.key.layer);
            if (gate_tensor == nullptr) {
                error = "missing RPP gate cache tensor for layer";
            } else if (source_->pack_expert_component(
                        current.key.layer,
                        current.key.expert,
                        llama_rpp_expert_component::gate_weight,
                        worker.staging_data,
                        staging_capacity_,
                        &gate_bytes,
                        &error)) {
                ggml_backend_tensor_set_async(
                        worker.backend,
                        gate_tensor,
                        worker.staging_data,
                        static_cast<uint64_t>(slot) * gate_tensor->nb[2],
                        gate_bytes);
                ggml_backend_event_record(worker.event, worker.backend);
                ggml_backend_event_synchronize(worker.event);
            }
            ggml_tensor * up_tensor = this->up_tensor(current.key.layer);
            if (error.empty() && up_tensor == nullptr) {
                error = "missing RPP up cache tensor for layer";
            } else if (error.empty() && source_->pack_expert_component(
                        current.key.layer,
                        current.key.expert,
                        llama_rpp_expert_component::up_weight,
                        worker.staging_data,
                        staging_capacity_,
                        &up_bytes,
                        &error)) {
                ggml_backend_tensor_set_async(
                        worker.backend,
                        up_tensor,
                        worker.staging_data,
                        static_cast<uint64_t>(slot) * up_tensor->nb[2],
                        up_bytes);
                ggml_backend_event_record(worker.event, worker.backend);
                ggml_backend_event_synchronize(worker.event);
            }
        }
        ggml_tensor * down_tensor = this->down_tensor(current.key.layer);
        if (error.empty() && down_tensor == nullptr) {
            error = "missing RPP down cache tensor for layer";
        } else if (error.empty() &&
                source_->pack_expert_component(
                    current.key.layer,
                    current.key.expert,
                    llama_rpp_expert_component::down_weight,
                    worker.staging_data,
                    staging_capacity_,
                    &down_bytes,
                    &error)) {
            ggml_backend_tensor_set_async(
                    worker.backend,
                    down_tensor,
                    worker.staging_data,
                    static_cast<uint64_t>(slot) * down_tensor->nb[2],
                    down_bytes);
            ggml_backend_event_record(worker.event, worker.backend);
            ggml_backend_event_synchronize(worker.event);
        }
        if (error.empty() && layout_ == expert_cache_layout::merged_gate_up &&
                source_->pack_expert_component(
                    current.key.layer,
                    current.key.expert,
                    llama_rpp_expert_component::down_scale,
                    worker.staging_data,
                    staging_capacity_,
                    &scale_bytes,
                    &error)) {
            ggml_backend_tensor_set_async(
                    worker.backend,
                    down_scale_tensor_,
                    worker.staging_data,
                    static_cast<uint64_t>(slot) * down_scale_tensor_->nb[0],
                    scale_bytes);
            ggml_backend_event_record(worker.event, worker.backend);
            ggml_backend_event_synchronize(worker.event);
        }
        const uint64_t copied_bytes =
            gate_up_bytes + gate_bytes + up_bytes + down_bytes + scale_bytes;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = entries_.find(current.key);
            if (it != entries_.end() && it->second.slot == slot) {
                it->second.error = error;
                it->second.bytes = copied_bytes;
                it->second.last_use = ++clock_;
                it->second.complete_us = now_us();
                it->second.state = error.empty() && copied_bytes == expected_bytes
                    ? llama_rpp_gpu_cache_entry_state::ready
                    : llama_rpp_gpu_cache_entry_state::failed;
            }
            --active_workers_;
            active_ = active_workers_ > 0;
            state_cv_.notify_all();
            if (queue_.empty() && active_workers_ == 0) {
                idle_cv_.notify_all();
            }
        }
    }
}
