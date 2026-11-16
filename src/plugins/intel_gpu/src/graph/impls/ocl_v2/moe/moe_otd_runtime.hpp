// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "lru_cache.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

cldnn::stream::ptr create_copy_stream(cldnn::engine& engine, const cldnn::ExecutionConfig& config);

// Lightweight perf counters for OTD profiling.
// Enabled by setting MOE_OTD_PERF_LOG=1 environment variable.
// Counters are printed to stderr on process exit.
struct OtdPerfCounters {
    struct DemandCounts {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t pending = 0;
        void add(uint64_t ready, uint64_t absent, uint64_t loading) {
            hits += ready;
            misses += absent + loading;
            pending += loading;
        }
    };
    enum class PredictionStatus { MATCHED, MISSING, STALE, LATE };
    struct RecallCounts {
        uint64_t matched = 0;
        uint64_t actual = 0;
        uint64_t compared = 0;
        uint64_t missing = 0;
        uint64_t stale = 0;
        uint64_t late = 0;
        static PredictionStatus classify(bool present, uint64_t predicted_generation,
                         uint64_t demand_generation, bool ready) {
            return !present ? PredictionStatus::MISSING :
               predicted_generation != demand_generation ? PredictionStatus::STALE :
               !ready ? PredictionStatus::LATE : PredictionStatus::MATCHED;
        }
        void add(const std::vector<uint32_t>& predicted, const std::vector<uint32_t>& demand,
                 PredictionStatus status) {
            if (status != PredictionStatus::MATCHED) {
                missing += status == PredictionStatus::MISSING;
                stale += status == PredictionStatus::STALE;
                late += status == PredictionStatus::LATE;
                return;
            }
            const std::set<uint32_t> unique_prediction(predicted.begin(), predicted.end());
            const std::set<uint32_t> unique_actual(demand.begin(), demand.end());
            ++compared;
            actual += unique_actual.size();
            for (auto expert : unique_actual) {
                matched += unique_prediction.count(expert);
            }
        }
    };
    struct LayerCounts {
        std::array<DemandCounts, 3> demand;
        RecallCounts recall;
    };
    mutable std::mutex layer_mutex;
    std::map<std::string, LayerCounts> layers;

    void record_demand(const std::string& layer, size_t token_num, uint64_t ready,
                       uint64_t absent, uint64_t loading) {
        std::lock_guard<std::mutex> lock(layer_mutex);
        layers[layer].demand[token_num == 1 ? 1 : token_num > 1 ? 0 : 2].add(ready, absent, loading);
    }
    void record_recall(const std::string& layer, const std::vector<uint32_t>& predicted,
                       const std::vector<uint32_t>& actual, PredictionStatus status) {
        std::lock_guard<std::mutex> lock(layer_mutex);
        layers[layer].recall.add(predicted, actual, status);
    }
    std::atomic<uint64_t> gpu_hits{0};
    std::atomic<uint64_t> gpu_misses{0};
    std::atomic<uint64_t> disk_io_ns{0};
    std::atomic<uint64_t> transpose_ns{0};
    std::atomic<uint64_t> gpu_copy_ns{0};
    std::atomic<uint64_t> tensor_load_count{0};
    std::atomic<uint64_t> batched_fallbacks{0};
    std::atomic<uint64_t> grouped_fallbacks{0};
    std::atomic<uint64_t> created_onednn_kernels{0};
    std::atomic<uint64_t> les_worker_tasks{0};
    std::atomic<uint64_t> les_upload_chunks{0};
    std::atomic<uint64_t> les_upload_bytes{0};
    std::atomic<uint64_t> les_waves{0};
    std::atomic<uint64_t> les_ready_pairs{0};
    std::atomic<uint64_t> les_missing_pairs{0};
    std::atomic<uint64_t> les_shared_pairs{0};
    std::atomic<uint64_t> les_prediction_matches{0};
    std::atomic<uint64_t> les_prediction_pairs{0};
    std::atomic<uint64_t> les_proxy_weight_initializations{0};
    std::atomic<uint64_t> les_proxy_fc_ns{0};
    std::atomic<uint64_t> les_proxy_topk_ns{0};
    std::atomic<uint64_t> les_proxy_submit_ns{0};
    std::atomic<uint64_t> les_prefetch_experts{0};
    std::atomic<uint64_t> les_prefetch_failures{0};
    std::atomic<uint64_t> les_prefetch_useful_bytes{0};
    std::atomic<uint64_t> les_prefetch_useless_bytes{0};
    std::atomic<uint64_t> les_prefetch_late_bytes{0};
    std::atomic<uint64_t> les_prefetch_extra_misses{0};
    std::atomic<uint64_t> les_prefetch_deduplicated{0};
    std::atomic<uint64_t> les_prefetch_promoted{0};
    std::atomic<uint64_t> les_prefetch_skipped{0};
    std::atomic<uint64_t> les_early_predictions{0};
    std::atomic<uint64_t> les_candidate_raw_ids{0};
    std::atomic<uint64_t> les_candidate_duplicates{0};
    std::array<std::atomic<uint64_t>, 6> les_candidate_outcomes{};
    std::atomic<uint64_t> les_loads_completed{0};
    std::atomic<uint64_t> les_loads_canceled{0};
    std::atomic<uint64_t> les_loads_failed{0};
    std::atomic<uint64_t> les_loads_superseded{0};
    std::atomic<uint64_t> les_tasks_expired_unread{0};
    std::atomic<uint64_t> les_tasks_canceled_unread{0};
    std::atomic<uint64_t> les_tasks_unbound{0};
    std::atomic<uint64_t> les_legacy_blocked_candidates{0};
    std::atomic<uint64_t> les_demand_retries{0};

    void dump() const;
};

template <typename Attempt>
void visit_speculative_candidates(const std::vector<uint32_t>& predictions, OtdPerfCounters* perf, Attempt&& attempt) {
    std::vector<uint32_t> unique;
    for (auto expert : predictions) {
        if (std::find(unique.begin(), unique.end(), expert) == unique.end()) {
            unique.push_back(expert);
        }
    }
    if (perf) {
        perf->les_candidate_raw_ids.fetch_add(predictions.size(), std::memory_order_relaxed);
        perf->les_candidate_duplicates.fetch_add(predictions.size() - unique.size(), std::memory_order_relaxed);
    }
    for (auto expert : unique) {
        const auto outcome = attempt(expert, unique);
        if (perf) {
            perf->les_candidate_outcomes[static_cast<size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
        }
    }
}

OtdPerfCounters* get_perf_counters();
bool les_trace_enabled();
bool les_prediction_compare_enabled();
bool layer_perf_enabled();
uint64_t les_device_time(cldnn::stream& stream);
void les_trace_interval(const char* kind, uint64_t start, uint64_t end);

struct PredictionReadback {
    std::mutex mutex;
    bool ready = false;
    std::vector<uint32_t> experts;

    void publish(const std::vector<uint32_t>& values) {
        std::lock_guard<std::mutex> lock(mutex);
        experts = values;
        ready = true;
    }
    bool snapshot(std::vector<uint32_t>& values) {
        std::lock_guard<std::mutex> lock(mutex);
        values = experts;
        return ready;
    }
};

class OtdLayerPerfScope {
public:
    OtdLayerPerfScope(const std::string& layer, size_t token_num, OtdPerfCounters* counters)
        : _previous(_current), _layer(layer), _token_num(token_num), _perf(counters) { _current = this; }
    OtdLayerPerfScope(const std::string& layer, size_t token_num)
        : OtdLayerPerfScope(layer, token_num, layer_perf_enabled() ? get_perf_counters() : nullptr) {}
    ~OtdLayerPerfScope() { _current = _previous; }
    OtdLayerPerfScope(const OtdLayerPerfScope&) = delete;
    OtdLayerPerfScope& operator=(const OtdLayerPerfScope&) = delete;

    static void record(uint32_t expert, bool hit, bool pending = false) {
        auto* scope = _current;
        if (scope && scope->_perf && scope->_seen.insert(expert).second) {
            scope->_perf->record_demand(scope->_layer, scope->_token_num, hit ? 1 : 0,
                                       !hit && !pending ? 1 : 0, pending ? 1 : 0);
        }
    }

private:
    inline static thread_local OtdLayerPerfScope* _current = nullptr;
    OtdLayerPerfScope* _previous;
    const std::string _layer;
    size_t _token_num;
    OtdPerfCounters* _perf;
    std::set<uint32_t> _seen;
};

class ParallelWeightReader {
public:
    explicit ParallelWeightReader(const std::filesystem::path& weights_path) : _weights_path(weights_path) {}

    ~ParallelWeightReader() {
        if (_opened) {
            ov::util::close_file_handle(_shared_handle);
        }
    }

    ParallelWeightReader(const ParallelWeightReader&) = delete;
    ParallelWeightReader& operator=(const ParallelWeightReader&) = delete;

    const std::filesystem::path& path() const {
        return _weights_path;
    }

    size_t file_size() {
        ensure_open();
        return _file_size;
    }

    void read(char* dst, size_t size, size_t file_offset) {
        ensure_open();
        if (!ov::util::positional_read(_shared_handle, dst, size, file_offset)) {
            OPENVINO_THROW("Failed to read enough bytes from OTD weight file");
        }
    }

private:
    void ensure_open() {
        if (!_opened) {
            std::streamoff file_size = 0;
            ov::util::get_file_handle_and_size(_weights_path, 0, _shared_handle, file_size);
            _file_size = static_cast<size_t>(file_size);
            _opened = true;
        }
    }

    std::filesystem::path _weights_path;
    ov::FileHandle _shared_handle{};
    size_t _file_size = 0;
    bool _opened = false;
};

void maybe_transpose_scale_zp(const cldnn::MOECompressed::Config& config,
                              size_t offset_pos,
                              const cldnn::layout& layout,
                              std::vector<uint8_t>& payload,
                              size_t per_expert_size);

void fill_weights_memory(cldnn::stream& exec_stream,
                         const cldnn::MOECompressed::Config& config,
                         const std::vector<size_t>& weight_bin_offsets,
                         ParallelWeightReader& weight_reader,
                         cldnn::moe_weights& wei_mem,
                         const std::vector<uint32_t>& experts_list,
                         const std::vector<size_t>& lru_experts);

struct LookaheadStagingPool {
    struct Chunk {
        cldnn::memory::ptr memory;
        cldnn::event::ptr upload;
    };
    static constexpr size_t chunk_bytes = 8 * 1024 * 1024;
    std::array<Chunk, 2> chunks;
    size_t cursor = 0;
};

std::vector<cldnn::event::ptr> fill_weights_memory_async(cldnn::stream& transfer_stream,
                                                      const cldnn::MOECompressed::Config& config,
                                                      const std::vector<size_t>& weight_bin_offsets,
                                                      ParallelWeightReader& weight_reader,
                                                      cldnn::moe_weights& weights,
                                                      const std::vector<uint32_t>& experts,
                                                      const std::vector<size_t>& slots,
                                                      uint64_t* uploaded_bytes = nullptr,
                                                      LookaheadStagingPool* staging_pool = nullptr);

}  // namespace ov::intel_gpu::ocl::moe_otd