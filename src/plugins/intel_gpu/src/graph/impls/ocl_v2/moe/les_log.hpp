#pragma once

#include <algorithm>
#include <chrono>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ov::intel_gpu::ocl::moe {

struct TimestampCacheEvent {
    uint64_t sequence;
    int64_t timestamp;
    std::string state;
    size_t expert;
};

inline const char* les_event_name(char kind) {
    switch (kind) {
    case 'R': return u8"\u9884\u8ba2";
    case 'J': return u8"\u5165\u961f";
    case 'B': return u8"\u52a0\u8f7d";
    case 'D': return u8"\u642c\u5b8c";
    case 'I': return u8"\u642c\u5b8c";
    case 'E': return u8"\u9a71\u9010";
    case 'X': return u8"\u53d6\u6d88\u642c\u8fd0";
    case 'F': return u8"\u5931\u8d25";
    case 'P': return u8"\u5360\u7528";
    case 'U': return u8"\u89e3\u9664\u5360\u7528";
    case 'T': return u8"\u79fb\u81f3\u961f\u5c3e";
    default: throw std::invalid_argument("Unknown LES cache event");
    }
}

struct TimestampLayerRecord {
    uint32_t layer_index = 0;
    std::vector<size_t> expert_ids;
    std::vector<size_t> topk_expert_ids;
    std::vector<size_t> ready_expert_ids;
    std::vector<size_t> inflight_expert_ids;
    std::vector<size_t> missing_expert_ids;
    std::vector<size_t> demand_expert_ids;
    std::vector<size_t> predicted_expert_ids;
    std::vector<size_t> prefetching_expert_ids;
    std::vector<TimestampCacheEvent> cache_events;
    std::vector<size_t> cache_events_expert_hit_ids;
};

struct TimestampLogRecord {
    size_t token_index = 0;
    int64_t timestamp = 0;
    std::string source_code;
    std::string tag;
    uint32_t layer_index = 0;
    uint32_t target_index = 0;
    std::vector<size_t> target_expert_ids;
    std::vector<size_t> ready_expert_ids;
    std::vector<size_t> inflight_expert_ids;
    std::vector<size_t> partition_missing_expert_ids;
    std::vector<size_t> target_expert_hit_ids;
    std::vector<size_t> missing_expert_ids;
    std::vector<size_t> prefetching_expert_ids;
    std::vector<size_t> target_topk_expert_ids;
    std::vector<size_t> predicted_expert_ids;
    std::vector<TimestampCacheEvent> cache_events;
    std::vector<size_t> cache_events_expert_hit_ids;
    TimestampLayerRecord current;
    TimestampLayerRecord target;

    TimestampLayerRecord layer_snapshot(uint32_t index) const {
        return {index, target_expert_ids, target_topk_expert_ids, ready_expert_ids, inflight_expert_ids,
                partition_missing_expert_ids, missing_expert_ids, predicted_expert_ids, prefetching_expert_ids,
                cache_events, cache_events_expert_hit_ids};
    }
};

class LESLog {
public:
    static LESLog& instance() {
        static LESLog log;
        return log;
    }

    bool enabled() const { return m_enabled; }

    void write_timestamp(TimestampLogRecord record, bool csv = true) {
        if (!m_enabled)
            return;
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!csv) {
            record.current = record.layer_snapshot(record.layer_index);
            record.target = record.current;
        }
        const auto current_events = incremental_events(record.current, record.token_index, csv);
        const auto target_events = record.current.layer_index == record.target.layer_index
            ? current_events : incremental_events(record.target, record.token_index, csv);
        if (record.current.layer_index == record.target.layer_index)
            record.target = record.current;
        const std::array<std::string, 28> fields = {
            std::to_string(record.token_index), std::to_string(record.timestamp), std::to_string(++m_sequence),
            std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())), record.source_code,
            record.tag, std::to_string(record.current.layer_index), ids(record.current.expert_ids),
            ids(record.current.topk_expert_ids), ids(record.current.ready_expert_ids), ids(record.current.inflight_expert_ids),
            ids(record.current.missing_expert_ids), ids(record.current.demand_expert_ids), ids(record.current.predicted_expert_ids),
            ids(record.current.prefetching_expert_ids), current_events, ids(record.current.cache_events_expert_hit_ids),
            std::to_string(record.target.layer_index), ids(record.target.expert_ids), ids(record.target.topk_expert_ids),
            ids(record.target.ready_expert_ids), ids(record.target.inflight_expert_ids), ids(record.target.missing_expert_ids),
            ids(record.target.demand_expert_ids), ids(record.target.predicted_expert_ids), ids(record.target.prefetching_expert_ids),
            target_events, ids(record.target.cache_events_expert_hit_ids)};
        if (m_stream.is_open()) {
            auto tag = record.tag;
            const auto separator = tag.find('-');
            if (separator != std::string::npos)
                tag[separator] = '_';
            m_stream << "tok" << fields[0] << '-' << fields[1] << '-' << fields[2] << '-' << fields[3]
                     << ' ' << record.source_code << '-' << tag << " [" << fields[6] << '-' << fields[17] << ']';
            for (size_t index = 7; index < fields.size(); ++index) {
                if (index != 17)
                    m_stream << ' ' << columns[index] << '=' << fields[index];
            }
        }
        for (size_t index = 0; index < fields.size(); ++index) {
            if (csv && m_csv.is_open())
                m_csv << (index ? "," : "") << csv_cell(fields[index]);
        }
        if (m_stream.is_open()) {
            m_stream << '\n';
            m_stream.flush();
            if (!m_stream)
                throw std::runtime_error("Cannot write OV_GPU_MOE_LOG_PATH");
        }
        if (csv && m_csv.is_open()) {
            m_csv << '\n';
            m_csv.flush();
            if (!m_csv)
                throw std::runtime_error("Cannot write LOG_CSV_PATH");
        }
    }

    bool text_enabled() const { return m_stream.is_open(); }

private:
    inline static constexpr std::array<const char*, 28> columns = {
        "token_index", "timestamp", "seq", "thread", "source_code", "tag", "current_layer_index",
        "current_expert_ids", "current_topk_expert_ids", "current_ready_expert_ids", "current_inflight_expert_ids",
        "current_missing_expert_ids", "current_demand_expert_ids", "current_predicted_expert_ids",
        "current_prefetching_expert_ids", "current_cache_events", "current_cache_events_expert_hit_ids",
        "target_layer_index", "target_expert_ids", "target_topk_expert_ids", "target_ready_expert_ids",
        "target_inflight_expert_ids", "target_missing_expert_ids", "target_demand_expert_ids",
        "target_predicted_expert_ids", "target_prefetching_expert_ids", "target_cache_events", "target_cache_events_expert_hit_ids"};

    std::string incremental_events(TimestampLayerRecord& layer, size_t token, bool csv) {
        auto& cursor = m_event_cursors[{token, layer.layer_index}];
        std::ostringstream events;
        events << '[';
        bool first = true;
        layer.cache_events_expert_hit_ids.clear();
        for (const auto& event : layer.cache_events) {
            if (csv ? event.sequence <= cursor : event.sequence != layer.cache_events.back().sequence)
                continue;
            events << (first ? "" : ",") << '[' << event.sequence << ',' << event.timestamp
                   << ",\"" << event.state << "\"," << event.expert << ']';
            first = false;
            if (std::find(layer.topk_expert_ids.begin(), layer.topk_expert_ids.end(), event.expert) != layer.topk_expert_ids.end() &&
                std::find(layer.cache_events_expert_hit_ids.begin(), layer.cache_events_expert_hit_ids.end(), event.expert) ==
                    layer.cache_events_expert_hit_ids.end())
                layer.cache_events_expert_hit_ids.push_back(event.expert);
        }
        if (csv && !layer.cache_events.empty())
            cursor = layer.cache_events.back().sequence;
        events << ']';
        return events.str();
    }

    template <typename Range>
    static std::string ids(const Range& values) {
        std::ostringstream text;
        text << '[';
        for (size_t index = 0; index < values.size(); ++index)
            text << (index ? "," : "") << values[index];
        text << ']';
        return text.str();
    }

    static std::string csv_cell(const std::string& value) {
        std::string encoded = "\"";
        for (const char character : value) {
            if (character == '"')
                encoded += '"';
            encoded += character;
        }
        return encoded + '"';
    }

    LESLog() {
        const char* path = std::getenv("OV_GPU_MOE_LOG_PATH");
        const char* csv_path = std::getenv("LOG_CSV_PATH");
        m_enabled = (path && *path) || (csv_path && *csv_path);
        if (path && *path) {
            m_stream.open(path);
            if (!m_stream)
                throw std::runtime_error("Cannot open OV_GPU_MOE_LOG_PATH");
        }
        if (csv_path && *csv_path) {
            if (path && std::string(path) == csv_path)
                throw std::runtime_error("Log and CSV paths must be different");
            m_csv.open(csv_path);
            if (!m_csv)
                throw std::runtime_error("Cannot open LOG_CSV_PATH");
            m_csv << "\xEF\xBB\xBF";
            for (size_t index = 0; index < columns.size(); ++index)
                m_csv << (index ? "," : "") << columns[index];
            m_csv << '\n';
        }
    }

    bool m_enabled = false;
    uint64_t m_sequence = 0;
    std::map<std::pair<size_t, uint32_t>, uint64_t> m_event_cursors;
    std::mutex m_mutex;
    std::ofstream m_stream;
    std::ofstream m_csv;
};

}  // namespace ov::intel_gpu::ocl::moe
