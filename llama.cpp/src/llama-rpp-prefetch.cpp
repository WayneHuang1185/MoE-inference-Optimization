#include "llama-rpp-prefetch.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr size_t RPP_MAX_PENDING_JOBS = 4096;
constexpr size_t RPP_MAX_JOB_RECORDS = 65536;

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::string> parse_csv_row(const std::string & line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (ch == ',' && !quoted) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    if (quoted) {
        throw std::runtime_error("unterminated quoted CSV field");
    }
    fields.push_back(current);
    return fields;
}

uint64_t parse_u64(const std::string & value, const char * field) {
    size_t consumed = 0;
    const uint64_t result = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error(std::string("invalid ") + field);
    }
    return result;
}

int32_t parse_i32(const std::string & value, const char * field) {
    size_t consumed = 0;
    const long result = std::stol(value, &consumed);
    if (consumed != value.size() ||
            result < std::numeric_limits<int32_t>::min() ||
            result > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(std::string("invalid ") + field);
    }
    return static_cast<int32_t>(result);
}

} // namespace

const char * llama_rpp_prefetch_state_name(llama_rpp_prefetch_state state) {
    switch (state) {
        case llama_rpp_prefetch_state::missing: return "missing";
        case llama_rpp_prefetch_state::queued:  return "queued";
        case llama_rpp_prefetch_state::running: return "running";
        case llama_rpp_prefetch_state::done:    return "done";
        case llama_rpp_prefetch_state::failed:  return "failed";
    }
    return "unknown";
}

llama_rpp_host_prefetcher::llama_rpp_host_prefetcher() = default;

llama_rpp_host_prefetcher::~llama_rpp_host_prefetcher() {
    stop();
}

bool llama_rpp_host_prefetcher::configure(
        const std::string & model_path,
        const std::string & page_map_path,
        int32_t n_threads,
        std::string * error) {
    stop();

#if !defined(__linux__) && !defined(__APPLE__)
    if (error) {
        *error = "RPP host pretouch requires POSIX mmap";
    }
    return false;
#else
    if (n_threads <= 0) {
        if (error) {
            *error = "RPP prefetch thread count must be positive";
        }
        return false;
    }

    if (!load_page_map(page_map_path, error)) {
        return false;
    }

    fd_ = open(model_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        if (error) {
            *error = "failed to open RPP model file: " + std::string(std::strerror(errno));
        }
        expert_ranges_.clear();
        component_ranges_.clear();
        return false;
    }

    struct stat file_stat = {};
    if (fstat(fd_, &file_stat) != 0 || file_stat.st_size <= 0) {
        if (error) {
            *error = "failed to stat RPP model file";
        }
        stop();
        return false;
    }
    mapping_size_ = static_cast<uint64_t>(file_stat.st_size);
    page_size_ = static_cast<uint64_t>(std::max<long>(1, sysconf(_SC_PAGESIZE)));

    void * mapping = mmap(nullptr, mapping_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapping == MAP_FAILED) {
        mapping_ = nullptr;
        if (error) {
            *error = "failed to mmap RPP model file: " + std::string(std::strerror(errno));
        }
        stop();
        return false;
    }
    mapping_ = static_cast<const uint8_t *>(mapping);

    for (const auto & item : expert_ranges_) {
        for (const auto & range : item.second) {
            if (range.end > mapping_size_) {
                if (error) {
                    *error = "RPP page map range exceeds model file size";
                }
                stop();
                return false;
            }
        }
    }

    stopping_ = false;
    for (int32_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back(&llama_rpp_host_prefetcher::worker_loop, this);
    }
    return true;
#endif
}

void llama_rpp_host_prefetcher::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    work_cv_.notify_all();
    for (auto & worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

#if defined(__linux__) || defined(__APPLE__)
    if (mapping_) {
        munmap(const_cast<uint8_t *>(mapping_), mapping_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
#endif

    mapping_ = nullptr;
    mapping_size_ = 0;
    max_expert_bytes_ = 0;
    fd_ = -1;
    expert_ranges_.clear();
    component_ranges_.clear();

    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
    records_.clear();
    active_workers_ = 0;
    stopping_ = false;
}

bool llama_rpp_host_prefetcher::enabled() const {
    return mapping_ != nullptr;
}

bool llama_rpp_host_prefetcher::enqueue(
        const llama_rpp_prefetch_key & key,
        const std::vector<int32_t> & experts) {
    if (!enabled()) {
        return false;
    }

    auto ranges = ranges_for(key.layer, experts);
    if (ranges.empty()) {
        return false;
    }

    llama_rpp_prefetch_snapshot snapshot;
    snapshot.state = llama_rpp_prefetch_state::queued;
    snapshot.enqueue_us = now_us();
    for (const auto & range : ranges) {
        snapshot.bytes += range.end - range.begin;
        snapshot.pages += (range.end - range.begin + page_size_ - 1) / page_size_;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || records_.count(key) != 0) {
            return false;
        }
        if (records_.size() >= RPP_MAX_JOB_RECORDS) {
            for (auto it = records_.begin();
                    it != records_.end() && records_.size() >= RPP_MAX_JOB_RECORDS;) {
                if (it->second.snapshot.state == llama_rpp_prefetch_state::done ||
                        it->second.snapshot.state == llama_rpp_prefetch_state::failed) {
                    it = records_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (queue_.size() >= RPP_MAX_PENDING_JOBS) {
            snapshot.state = llama_rpp_prefetch_state::failed;
            snapshot.complete_us = snapshot.enqueue_us;
            snapshot.error = "prefetch queue capacity exceeded";
            records_.emplace(key, job_record{snapshot});
            return false;
        }
        records_.emplace(key, job_record{snapshot});
        queue_.push_back(job{key, std::move(ranges)});
    }
    work_cv_.notify_one();
    return true;
}

llama_rpp_prefetch_snapshot llama_rpp_host_prefetcher::snapshot(
        const llama_rpp_prefetch_key & key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = records_.find(key);
    return it == records_.end() ? llama_rpp_prefetch_snapshot{} : it->second.snapshot;
}

bool llama_rpp_host_prefetcher::wait_idle(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    return idle_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
        return queue_.empty() && active_workers_ == 0;
    });
}

bool llama_rpp_host_prefetcher::pack_experts(
        int32_t layer,
        const std::vector<int32_t> & experts,
        void * destination,
        size_t capacity,
        llama_rpp_expert_pack_result * result,
        std::string * error) const {
    if (!mapping_ || !destination) {
        if (error) {
            *error = "RPP expert data source is not configured";
        }
        return false;
    }

    std::vector<byte_range> ranges;
    for (int32_t expert : experts) {
        const auto it = expert_ranges_.find({layer, expert});
        if (it == expert_ranges_.end()) {
            if (error) {
                *error = "missing page-map entry for layer/expert";
            }
            return false;
        }
        ranges.insert(ranges.end(), it->second.begin(), it->second.end());
    }

    uint64_t total = 0;
    for (const auto & range : ranges) {
        total += range.end - range.begin;
    }
    if (total > capacity) {
        if (error) {
            *error = "expert slices exceed staging buffer capacity";
        }
        return false;
    }

    const int64_t start_us = now_us();
    auto * output = static_cast<uint8_t *>(destination);
    size_t offset = 0;
    for (const auto & range : ranges) {
        const size_t size = static_cast<size_t>(range.end - range.begin);
        std::memcpy(output + offset, mapping_ + range.begin, size);
        offset += size;
    }

    if (result) {
        result->bytes = total;
        result->ranges = ranges.size();
        result->duration_us = now_us() - start_us;
    }
    return true;
}

bool llama_rpp_host_prefetcher::pack_expert_component(
        int32_t layer,
        int32_t expert,
        llama_rpp_expert_component component,
        void * destination,
        size_t capacity,
        uint64_t * bytes,
        std::string * error) const {
    if (!mapping_ || !destination) {
        if (error) {
            *error = "RPP expert data source is not configured";
        }
        return false;
    }
    const auto it = component_ranges_.find({layer, expert, component});
    if (it == component_ranges_.end()) {
        if (error) {
            *error = "missing page-map component for layer/expert";
        }
        return false;
    }
    const uint64_t size = it->second.end - it->second.begin;
    if (size > capacity) {
        if (error) {
            *error = "expert component exceeds staging buffer capacity";
        }
        return false;
    }
    std::memcpy(destination, mapping_ + it->second.begin, size);
    if (bytes) {
        *bytes = size;
    }
    return true;
}

uint64_t llama_rpp_host_prefetcher::component_bytes(
        int32_t layer,
        int32_t expert,
        llama_rpp_expert_component component) const {
    const auto it = component_ranges_.find({layer, expert, component});
    return it == component_ranges_.end() ? 0 : it->second.end - it->second.begin;
}

uint64_t llama_rpp_host_prefetcher::expert_bytes(int32_t layer, int32_t expert) const {
    const auto it = expert_ranges_.find({layer, expert});
    if (it == expert_ranges_.end()) {
        return 0;
    }
    uint64_t total = 0;
    for (const auto & range : it->second) {
        total += range.end - range.begin;
    }
    return total;
}

uint64_t llama_rpp_host_prefetcher::max_expert_bytes() const {
    return max_expert_bytes_;
}

bool llama_rpp_host_prefetcher::load_page_map(
        const std::string & path,
        std::string * error) {
    expert_ranges_.clear();
    component_ranges_.clear();
    max_expert_bytes_ = 0;
    std::ifstream input(path);
    if (!input.is_open()) {
        if (error) {
            *error = "failed to open RPP page map: " + path;
        }
        return false;
    }

    try {
        std::string line;
        if (!std::getline(input, line)) {
            throw std::runtime_error("empty page map");
        }
        const auto header = parse_csv_row(line);
        std::map<std::string, size_t> columns;
        for (size_t i = 0; i < header.size(); ++i) {
            columns[header[i]] = i;
        }
        for (const char * required : {
                "layer", "expert", "byte_start", "byte_end"}) {
            if (columns.count(required) == 0) {
                throw std::runtime_error(std::string("missing page map column: ") + required);
            }
        }

        size_t line_number = 1;
        while (std::getline(input, line)) {
            ++line_number;
            if (line.empty()) {
                continue;
            }
            try {
                const auto row = parse_csv_row(line);
                if (row.size() != header.size()) {
                    throw std::runtime_error("column count mismatch");
                }
                const int32_t layer = parse_i32(row[columns["layer"]], "layer");
                const int32_t expert = parse_i32(row[columns["expert"]], "expert");
                const uint64_t begin = parse_u64(row[columns["byte_start"]], "byte_start");
                const uint64_t end = parse_u64(row[columns["byte_end"]], "byte_end");
                if (layer < 0 || expert < 0 || begin >= end) {
                    throw std::runtime_error("invalid page map range");
                }
                expert_ranges_[{layer, expert}].push_back({begin, end});
                if (columns.count("component") != 0 && columns.count("kind") != 0) {
                    const std::string & component_name = row[columns["component"]];
                    const std::string & kind = row[columns["kind"]];
                    llama_rpp_expert_component component;
                    if (component_name == "ffn_gate_up_exps" && kind == "weight") {
                        component = llama_rpp_expert_component::gate_up_weight;
                    } else if (component_name == "ffn_down_exps" && kind == "weight") {
                        component = llama_rpp_expert_component::down_weight;
                    } else if (component_name == "ffn_down_exps" && kind == "scale") {
                        component = llama_rpp_expert_component::down_scale;
                    } else {
                        throw std::runtime_error(
                                "unsupported expert component: " + component_name + "." + kind);
                    }
                    if (!component_ranges_.emplace(
                                component_key{layer, expert, component},
                                byte_range{begin, end}).second) {
                        throw std::runtime_error("duplicate expert component");
                    }
                }
            } catch (const std::exception & e) {
                throw std::runtime_error(
                        "page map line " + std::to_string(line_number) + ": " + e.what());
            }
        }
        if (expert_ranges_.empty()) {
            throw std::runtime_error("page map contains no expert ranges");
        }
        for (const auto & item : expert_ranges_) {
            uint64_t total = 0;
            for (const auto & range : item.second) {
                total += range.end - range.begin;
            }
            max_expert_bytes_ = std::max(max_expert_bytes_, total);
        }
        if (max_expert_bytes_ == 0) {
            throw std::runtime_error("page map expert ranges contain no bytes");
        }
    } catch (const std::exception & e) {
        expert_ranges_.clear();
        component_ranges_.clear();
        max_expert_bytes_ = 0;
        if (error) {
            *error = e.what();
        }
        return false;
    }
    return true;
}

std::vector<llama_rpp_host_prefetcher::byte_range>
llama_rpp_host_prefetcher::ranges_for(
        int32_t layer,
        const std::vector<int32_t> & experts) const {
    std::vector<byte_range> ranges;
    for (int32_t expert : experts) {
        const auto it = expert_ranges_.find({layer, expert});
        if (it != expert_ranges_.end()) {
            ranges.insert(ranges.end(), it->second.begin(), it->second.end());
        }
    }
    if (ranges.empty()) {
        return ranges;
    }

    for (auto & range : ranges) {
        range.begin -= range.begin % page_size_;
        range.end = std::min<uint64_t>(
                mapping_size_,
                ((range.end + page_size_ - 1) / page_size_) * page_size_);
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.begin != rhs.begin ? lhs.begin < rhs.begin : lhs.end < rhs.end;
    });

    std::vector<byte_range> merged;
    for (const auto & range : ranges) {
        if (merged.empty() || range.begin > merged.back().end) {
            merged.push_back(range);
        } else {
            merged.back().end = std::max(merged.back().end, range.end);
        }
    }
    return merged;
}

void llama_rpp_host_prefetcher::worker_loop() {
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
            ++active_workers_;
            auto & record = records_.at(current.key).snapshot;
            record.state = llama_rpp_prefetch_state::running;
            record.start_us = now_us();
        }

        std::string failure;
        volatile uint8_t checksum = 0;
#if defined(RUSAGE_THREAD)
        struct rusage usage_before = {};
        struct rusage usage_after = {};
        getrusage(RUSAGE_THREAD, &usage_before);
#endif
        try {
            for (const auto & range : current.ranges) {
                for (uint64_t offset = range.begin; offset < range.end; offset += page_size_) {
                    checksum ^= mapping_[offset];
                }
            }
        } catch (const std::exception & e) {
            failure = e.what();
        } catch (...) {
            failure = "unknown pretouch failure";
        }
        (void) checksum;
#if defined(RUSAGE_THREAD)
        getrusage(RUSAGE_THREAD, &usage_after);
#endif

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto & record = records_.at(current.key).snapshot;
            record.complete_us = now_us();
            record.duration_us = record.complete_us - record.start_us;
#if defined(RUSAGE_THREAD)
            record.minor_faults = usage_after.ru_minflt - usage_before.ru_minflt;
            record.major_faults = usage_after.ru_majflt - usage_before.ru_majflt;
#endif
            record.error = failure;
            record.state = failure.empty()
                ? llama_rpp_prefetch_state::done
                : llama_rpp_prefetch_state::failed;
            --active_workers_;
            if (queue_.empty() && active_workers_ == 0) {
                idle_cv_.notify_all();
            }
        }
    }
}
