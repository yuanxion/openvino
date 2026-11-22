#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ov::intel_gpu::ocl::moe {

class LESTimeline {
public:
    static bool enabled() {
        static const bool active = [] {
            const char* path = std::getenv("OV_GPU_LES_TRACE");
            return path && *path;
        }();
        return active;
    }

    static int64_t now() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static uint64_t thread_id() {
#ifdef __linux__
        return static_cast<uint64_t>(syscall(SYS_gettid));
#else
        return 0;
#endif
    }

    static std::string quote(const std::string& value) {
        std::ostringstream text;
        text << '"';
        for (unsigned char character : value) {
            if (character == '"' || character == '\\')
                text << '\\' << character;
            else if (character < 0x20)
                text << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(character);
            else
                text << character;
        }
        text << '"';
        return text.str();
    }

    template <typename Value>
    static std::string array(const std::vector<Value>& values) {
        std::ostringstream text;
        text << '[';
        for (size_t index = 0; index < values.size(); ++index)
            text << (index ? "," : "") << values[index];
        text << ']';
        return text.str();
    }

    static void instant(const std::string& name, const std::string& args) {
        if (enabled())
            instance().write(name, args, now(), 0, 'i', thread_id());
    }

    static void span(const std::string& name, const std::string& args, int64_t start, int64_t end, uint64_t thread) {
        if (enabled())
            instance().write(name, args, start, end - start, 'X', thread);
    }

private:
    LESTimeline() : output(std::getenv("OV_GPU_LES_TRACE")) {
        output << "[\n{\"ph\":\"M\",\"name\":\"les_clock\",\"pid\":" << process_id()
               << ",\"tid\":0,\"args\":{\"clock\":\"steady_clock\",\"unit\":\"us\",\"absolute\":true}}";
    }

    ~LESTimeline() {
        output << "\n]\n";
    }

    static LESTimeline& instance() {
        static LESTimeline timeline;
        return timeline;
    }

    static uint64_t process_id() {
#ifdef __linux__
        return static_cast<uint64_t>(getpid());
#else
        return 0;
#endif
    }

    void write(const std::string& name, const std::string& args, int64_t start, int64_t duration,
               char phase, uint64_t thread) {
        std::lock_guard<std::mutex> lock(mutex);
        output << ",\n{\"ph\":\"" << phase << "\",\"cat\":\"LES\",\"name\":" << quote(name)
               << ",\"pid\":" << process_id() << ",\"tid\":" << thread << ",\"ts\":" << start;
        if (phase == 'X')
            output << ",\"dur\":" << duration;
        else
            output << ",\"s\":\"t\"";
        output << ",\"args\":" << args << '}';
    }

    std::ofstream output;
    std::mutex mutex;
};

struct LESLoadContext {
    std::string request;
    size_t token_index = 0;
    uint32_t layer_index = 0;
    uint32_t target_layer_index = 0;
    std::vector<uint32_t> topk_expert_ids;
    std::vector<uint32_t> prediction_expert_ids;

    std::string arguments(size_t expert, size_t slot, uint64_t generation,
                          bool speculative, bool demanded) const {
        const auto loading = LESTimeline::array(std::vector<size_t>{expert});
        return "{\"request\":" + LESTimeline::quote(request) +
               ",\"token_index\":" + std::to_string(token_index) +
               ",\"layer_index\":" + std::to_string(layer_index) +
               ",\"target_layer_index\":" + std::to_string(target_layer_index) +
               ",\"expert_id\":" + std::to_string(expert) + ",\"slot\":" + std::to_string(slot) +
               ",\"generation\":" + std::to_string(generation) +
               ",\"origin\":" + LESTimeline::quote(speculative ? "prefetch" : "demand") +
               ",\"demanded\":" + (demanded ? "true" : "false") +
               ",\"demand\":" + (speculative ? "[]" : loading) +
               ",\"prefetch\":" + (speculative ? loading : "[]") +
               ",\"topk_expert_ids\":" + LESTimeline::array(topk_expert_ids) +
               ",\"prediction_expert_ids\":" + LESTimeline::array(prediction_expert_ids) + '}';
    }
};

}  // namespace ov::intel_gpu::ocl::moe