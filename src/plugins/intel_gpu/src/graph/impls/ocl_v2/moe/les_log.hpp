#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace ov::intel_gpu::ocl::moe {

inline constexpr char les_evicted_prefix[] = u8"[\u9a71\u9010]";
inline constexpr char les_ready_prefix[] = u8"[\u642c\u5b8c]";
inline constexpr char les_copying_prefix[] = u8"[\u5728\u642c]";
inline constexpr char les_queued_prefix[] = u8"[\u5f85\u642c]";
inline constexpr char les_changed_prefix[] = u8"[\u7f13\u5b58]";

class LESLog {
public:
    static LESLog& instance() {
        static LESLog log;
        return log;
    }

    bool enabled() const { return m_enabled; }

    void write(const std::string& layer, const char* file, int line, const char* function,
               const std::string& message, const char* prefix = "[LES]", const std::string& experts = "") {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        m_stream << prefix << (std::string_view(prefix) == "[LES]" ? " " : ":") << short_layer(layer);
        if (!experts.empty())
            m_stream << ' ' << experts;
        m_stream << " seq=" << ++m_sequence << " ts_us=" << time
             << " thread=" << std::this_thread::get_id() << " source=" << short_file(file) << ':' << line
                 << " function=" << function << ' ' << message << '\n';
        m_stream.flush();
        if (!m_stream)
            throw std::runtime_error("Cannot write OV_GPU_MOE_LOG_PATH");
    }

private:
    static std::string short_layer(const std::string& layer) {
        constexpr std::string_view prefix = "moecompressed:";
        constexpr std::string_view marker = ".layers.";
        const std::string_view name(layer);
        const auto position = name.find(marker);
        if (name.substr(0, prefix.size()) != prefix || position == std::string_view::npos)
            return layer;
        const auto begin = position + marker.size();
        const auto end = name.find_first_not_of("0123456789", begin);
        if (end == begin || end == std::string_view::npos || name[end] != '.')
            return layer;
        return "moecompressed." + std::string(name.substr(begin, end - begin));
    }

    static const char* short_file(const char* file) {
        const char* name = file;
        for (const char* cursor = file; *cursor; ++cursor) {
            if (*cursor == '/' || *cursor == '\\')
                name = cursor + 1;
        }
        return name;
    }

    LESLog() {
        const char* path = std::getenv("OV_GPU_MOE_LOG_PATH");
        m_enabled = path && *path;
        if (m_enabled) {
            m_stream.open(path);
            if (!m_stream)
                throw std::runtime_error("Cannot open OV_GPU_MOE_LOG_PATH");
        }
    }

    bool m_enabled = false;
    uint64_t m_sequence = 0;
    std::mutex m_mutex;
    std::ofstream m_stream;
};

template <typename Range>
std::string les_ids(const Range& values) {
    std::ostringstream text;
    text << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first)
            text << ',';
        text << value;
        first = false;
    }
    text << ']';
    return text.str();
}

}  // namespace ov::intel_gpu::ocl::moe

#define MOE_LES_LOG_TAG(prefix, layer, message) do { \
    if (::ov::intel_gpu::ocl::moe::LESLog::instance().enabled()) { \
        std::ostringstream les_message; \
        les_message << message; \
        ::ov::intel_gpu::ocl::moe::LESLog::instance().write(layer, __FILE__, __LINE__, __func__, les_message.str(), prefix); \
    } \
} while (false)

#define MOE_LES_LOG(layer, message) MOE_LES_LOG_TAG("[LES]", layer, message)
