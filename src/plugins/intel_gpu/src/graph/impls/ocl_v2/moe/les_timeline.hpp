#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ov::intel_gpu::ocl::moe::timeline {

inline bool enabled() {
    static const bool value = [] {
        const auto* setting = std::getenv("OV_GPU_LES_TIMELINE");
        return setting && std::string(setting) == "1";
    }();
    return value;
}

inline uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline bool completion_enabled() {
    static const bool value = [] {
        const auto* setting = std::getenv("OV_GPU_LES_TIMELINE_COMPLETION");
        return setting && std::string(setting) == "1";
    }();
    return enabled() && value;
}

inline std::string escape(const std::string& value) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string result;
    for (unsigned char character : value) {
        if (character <= 32 || character >= 127 || character == '%' || character == '=') {
            result += '%';
            result += hex[character >> 4];
            result += hex[character & 15];
        } else {
            result += static_cast<char>(character);
        }
    }
    return result.empty() ? "-" : result;
}

inline std::string ids(const std::vector<uint32_t>& values) {
    std::string result;
    for (auto value : values) {
        if (!result.empty())
            result += ',';
        result += std::to_string(value);
    }
    return result.empty() ? "-" : result;
}

inline void write(const std::string& fields, uint64_t observed_ns) {
    static std::mutex mutex;
    static uint64_t sequence = 0;
    std::lock_guard<std::mutex> lock(mutex);
    const auto line = "LES_TIMELINE\tv=1\tseq=" + std::to_string(++sequence) +
                      "\tobs_ns=" + std::to_string(observed_ns) + '\t' + fields + '\n';
    std::fwrite(line.data(), 1, line.size(), stderr);
}

struct Key {
    uint64_t network = 0;
    uint64_t request = 0;
    uint64_t token = 0;
    uint64_t epoch = 0;
    uint32_t layer = 0;
    std::string name;
    std::string request_phase = "unlabeled";
    std::string iteration = "NA";

    std::string fields() const {
        return "network=" + std::to_string(network) + "\trequest=" + std::to_string(request) +
               "\ttoken=" + std::to_string(token) + "\tepoch=" + std::to_string(epoch) +
               "\tlayer=" + std::to_string(layer) + "\tlayer_id=" + escape(name) +
               "\trequest_phase=" + escape(request_phase) + "\titer=" + escape(iteration);
    }
};

struct Selection {
    std::string phase;
    std::string iteration;
    std::vector<uint64_t> tokens;

    explicit Selection(const char* setting) {
        if (!setting)
            return;
        const std::string value(setting);
        const auto first = value.find(':');
        const auto second = first == std::string::npos ? first : value.find(':', first + 1);
        const auto invalid = [] {
            throw std::invalid_argument("OV_GPU_LES_TIMELINE_SAMPLE must be phase:iteration:token[,token...] (canonical unsigned integers, tokens > 0)");
        };
        const auto number = [&](const std::string& text) {
            if (text.empty() || (text.size() > 1 && text.front() == '0'))
                invalid();
            uint64_t result = 0;
            for (const auto digit : text) {
                if (digit < '0' || digit > '9' || result > (std::numeric_limits<uint64_t>::max() - (digit - '0')) / 10)
                    invalid();
                result = result * 10 + (digit - '0');
            }
            return result;
        };
        if (first == std::string::npos || second == std::string::npos)
            invalid();
        phase = value.substr(0, first);
        if (phase != "measured" && phase != "warmup")
            invalid();
        iteration = value.substr(first + 1, second - first - 1);
        number(iteration);
        size_t begin = second + 1;
        do {
            const auto end = value.find(',', begin);
            const auto token = number(value.substr(begin, end - begin));
            if (!token || std::find(tokens.begin(), tokens.end(), token) != tokens.end())
                invalid();
            tokens.push_back(token);
            if (end == std::string::npos)
                break;
            begin = end + 1;
        } while (true);
    }

    bool matches(const Key& key) const {
        return tokens.empty() || (key.request_phase == phase && key.iteration == iteration &&
               std::find(tokens.begin(), tokens.end(), key.token) != tokens.end());
    }
};

inline const Selection& selection() {
    static const Selection value(std::getenv("OV_GPU_LES_TIMELINE_SAMPLE"));
    return value;
}

struct Demand {
    std::string context;
    uint32_t expert = 0;
    uint64_t mark_ns = 0;
    std::atomic<bool> reported{false};

    ~Demand() {
        if (!reported.load())
            finish("unavailable", now_ns(), "\tcache=NA\tslot=NA\tload_gen=NA");
    }

    void finish(const char* status, uint64_t ready_ns, const std::string& load) {
        if (reported.exchange(true))
            return;
        const bool success = std::string(status) == "ready";
        const auto end = std::max(mark_ns, ready_ns);
        write(context + "\tstage=demand_result\texpert=" + std::to_string(expert) + load +
              "\tstatus=" + status + "\tdemand_mark_ns=" + std::to_string(mark_ns) +
              "\tweights_ready_ns=" + (success ? std::to_string(end) : "NA") +
              "\tlatency_ns=" + (success ? std::to_string(end - mark_ns) : "NA"), end);
    }
};

struct Load {
    std::mutex mutex;
    std::string identity;
    std::string origin;
    std::string status;
    uint64_t terminal_ns = 0;
    bool flushed = false;
    bool visible = true;
    bool start_flushed = false;
    uint64_t start_ns = 0;
    std::vector<std::shared_ptr<Demand>> demands;

    void set_origin(const Key& key, uint32_t expert, const char* kind,
                    const std::optional<Key>& prefetch_source = {}) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (origin.empty())
                origin = key.fields() + "\texpert=" + std::to_string(expert) + "\torigin=" + kind +
                         "\tprefetch_source_layer=" + (prefetch_source ? std::to_string(prefetch_source->layer) : "NA") +
                         "\tprefetch_source_layer_id=" + (prefetch_source ? escape(prefetch_source->name) : "NA");
        }
        flush();
    }

    void start(uint64_t stamp) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!start_ns)
                start_ns = stamp;
        }
        flush();
    }

    void expose() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            visible = true;
        }
        flush();
    }

    void watch(const std::shared_ptr<Demand>& demand) {
        expose();
        std::string result;
        uint64_t stamp = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            result = status;
            stamp = terminal_ns;
            if (result.empty()) {
                if (std::find(demands.begin(), demands.end(), demand) == demands.end())
                    demands.push_back(demand);
                return;
            }
        }
        demand->finish(result.c_str(), stamp, identity);
    }

    void terminal(const char* result, uint64_t stamp) {
        std::lock_guard<std::mutex> lock(mutex);
        if (status.empty()) {
            status = result;
            terminal_ns = stamp;
        }
    }

    void flush() {
        std::vector<std::shared_ptr<Demand>> waiting;
        std::string result;
        uint64_t stamp = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!visible || origin.empty())
                return;
            if (start_ns && !start_flushed) {
                start_flushed = true;
                write(origin + "\tstage=load_start" + identity, start_ns);
            }
            if (flushed || status.empty())
                return;
            flushed = true;
            waiting.swap(demands);
            result = status;
            stamp = terminal_ns;
            write(origin + "\tstage=load_result" + identity + "\tstatus=" + result, stamp);
        }
        for (const auto& demand : waiting)
            demand->finish(result.c_str(), stamp, identity);
    }
};

struct Entry {
    uint32_t expert;
    size_t slot;
    uint64_t generation;
    bool loading;
    bool ready;
    std::shared_ptr<Load> load;
};

struct Snapshot {
    uint64_t stamp_ns = 0;
    std::vector<Entry> entries;
    uintptr_t cache = 0;

    std::string fields() const {
        std::vector<uint32_t> resident, ready, loading;
        std::string slots;
        for (const auto& entry : entries) {
            if (entry.load)
                entry.load->expose();
            resident.push_back(entry.expert);
            if (entry.ready)
                ready.push_back(entry.expert);
            if (entry.loading)
                loading.push_back(entry.expert);
            if (!slots.empty())
                slots += ',';
            slots += std::to_string(entry.expert) + ':' + std::to_string(entry.slot) + ':' +
                     std::to_string(entry.generation);
        }
        return "\tcache=" + std::to_string(cache) + "\tresident=" + ids(resident) + "\tready=" + ids(ready) + "\tloading=" + ids(loading) +
               "\tslots=" + (slots.empty() ? "-" : slots);
    }
};

struct Step {
    const Key key;
    const std::string context;
    bool baseline_observed = false;
    std::vector<std::shared_ptr<Demand>> demands;
    std::mutex prediction_mutex;
    std::vector<uint32_t> predicted_ids;
    std::vector<uint32_t> actual_ids;
    bool prediction_available = false;
    bool actual_available = false;
    bool recall_reported = false;

    explicit Step(Key value) : key(std::move(value)), context(key.fields()) {}

    void event(const char* stage, const std::string& fields = {}, uint64_t stamp = now_ns()) const {
        write(context + "\tstage=" + stage + fields, stamp);
    }

    void prediction(const std::vector<uint32_t>& values, bool actual) {
        std::string recall;
        {
            std::lock_guard<std::mutex> lock(prediction_mutex);
            (actual ? actual_ids : predicted_ids) = values;
            (actual ? actual_available : prediction_available) = true;
            if (actual_available && prediction_available && !recall_reported) {
                recall_reported = true;
                auto unique = actual_ids;
                std::sort(unique.begin(), unique.end());
                unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
                size_t matches = 0;
                for (auto expert : unique)
                    matches += std::find(predicted_ids.begin(), predicted_ids.end(), expert) != predicted_ids.end();
                recall = "\tpredicted=" + ids(predicted_ids) + "\tactual=" + ids(unique) +
                         "\tmatched=" + std::to_string(matches) + "\tactual_count=" + std::to_string(unique.size());
            }
        }
        if (!recall.empty())
            event("recall", recall);
    }

    void observe(const char* stage, const Snapshot& snapshot, const std::vector<uint32_t>& actual,
                 size_t wave = 0, size_t pass = 0, bool first = false, const std::string& extra = {}) {
        std::vector<uint32_t> ready, pending, missing, remaining;
        for (auto expert : actual) {
            const auto found = std::find_if(snapshot.entries.begin(), snapshot.entries.end(),
                [expert](const Entry& entry) { return entry.expert == expert; });
            const bool hit = found != snapshot.entries.end() && found->ready;
            const bool inflight = found != snapshot.entries.end() && found->loading;
            (hit ? ready : inflight ? pending : missing).push_back(expert);
            if (!hit)
                remaining.push_back(expert);
            if (first && std::none_of(demands.begin(), demands.end(),
                                      [expert](const auto& demand) { return demand->expert == expert; })) {
                auto demand = std::make_shared<Demand>();
                demand->context = context;
                demand->expert = expert;
                demand->mark_ns = snapshot.stamp_ns;
                demands.push_back(demand);
                if (hit)
                    demand->finish("ready", snapshot.stamp_ns,
                        found->load ? found->load->identity : "\tcache=" + std::to_string(snapshot.cache) + "\tslot=" +
                        std::to_string(found->slot) + "\tload_gen=" + std::to_string(found->generation));
                else if (inflight && found->load)
                    found->load->watch(demand);
            }
            if (!first && std::string(stage) != "after_pass_complete" && found != snapshot.entries.end() && found->load)
                link(expert, found->load);
        }
        event(stage, "\twave=" + std::to_string(wave) + "\tpass=" + std::to_string(pass) +
              "\tactual=" + ids(actual) + "\tdemand_ready=" + ids(ready) + "\tdemand_pending=" + ids(pending) +
              "\tdemand_missing=" + ids(missing) + "\tremaining=" + ids(remaining) + snapshot.fields() + extra,
              snapshot.stamp_ns);
    }

    void link(uint32_t expert, const std::shared_ptr<Load>& load) const {
        if (!load)
            return;
        load->expose();
        for (const auto& demand : demands) {
            if (demand->expert == expert)
                load->watch(demand);
        }
    }
};

inline std::shared_ptr<Step> make_step(Key key, const Selection& filter = selection()) {
    return filter.matches(key) ? std::make_shared<Step>(std::move(key)) : nullptr;
}

inline std::shared_ptr<Load> make_load(const void* cache, size_t slot, uint64_t generation) {
    if (!enabled())
        return {};
    auto load = std::make_shared<Load>();
    load->visible = selection().tokens.empty();
    load->identity = "\tcache=" + std::to_string(reinterpret_cast<uintptr_t>(cache)) +
                     "\tslot=" + std::to_string(slot) + "\tload_gen=" + std::to_string(generation);
    return load;
}

}