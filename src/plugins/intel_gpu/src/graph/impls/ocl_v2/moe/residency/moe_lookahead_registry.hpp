// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>
#include <atomic>
#include <optional>
#include <unordered_map>

#include "lookahead_router.hpp"
#include "moe_lookahead_prefetcher.hpp"
#include "../expert_weight_providers.hpp"

namespace ov::intel_gpu::ocl::moe {

struct LookaheadLayerDesc {
    const cldnn::primitive_inst* moe_inst = nullptr;
    const cldnn::primitive_inst* router_fc_inst = nullptr;
    cldnn::memory::ptr weight_mem;
    uint32_t num_experts = 0;
    uint32_t hidden_size = 0;
    bool is_supported = false;
};

class LookaheadContext {
public:
    struct CacheSnapshot {
        size_t completed_token = 0;
        int64_t time_us = 0;
        std::vector<size_t> ready;
    };

    struct Prediction {
        cldnn::memory::ptr hidden;
        cldnn::event::ptr hidden_ready;
        cldnn::memory::ptr predicted_indices;
        cldnn::memory::ptr predicted_weights;
        cldnn::event::ptr prediction_ready;
        std::shared_ptr<std::atomic<bool>> consumed;
        bool submitted = false;
        std::shared_ptr<std::vector<uint32_t>> cpu_indices;
    };

    explicit LookaheadContext(cldnn::network& network);
    ~LookaheadContext();
    void build_once(cldnn::network& network);
    const LookaheadLayerDesc* get_next_layer(const cldnn::primitive_inst* current_moe) const;
    ExpertPrefetcher& prefetcher() { return *m_prefetcher; }
    uint64_t epoch() const { return m_epoch; }
    const std::vector<cldnn::program::les_router_link>& links() const { return m_links; }

    void enter(const cldnn::primitive_id& id);
    void demand_partitioned(const cldnn::primitive_id& id);
    void pause_for_prefill();
    void capture_cache_baseline();
    void begin_handoff(const cldnn::primitive_id& source, size_t token, bool selected);
    void checkpoint_handoff(const cldnn::primitive_id& source);
    void finish_handoff(const cldnn::primitive_id& target, size_t token, const std::vector<uint32_t>& actual);
    CacheSnapshot cache_baseline(const cldnn::primitive_id& id,
                                 const std::string& request,
                                 size_t decode_token,
                                 bool prefill = false) const;
    std::shared_ptr<IExpertWeightProvider> bind(const cldnn::primitive_id& id,
                                               std::shared_ptr<IExpertWeightProvider> provider);

    std::optional<LRUCache::Trace> cache_snapshot(const cldnn::primitive_id& id) const {
        const auto found = m_providers.find(id);
        if (found == m_providers.end() || !found->second)
            return std::nullopt;
        return found->second->cache().snapshot();
    }

    void set_cache_log_window(const cldnn::primitive_id& id, bool enabled) {
        const auto found = m_providers.find(id);
        if (found != m_providers.end() && found->second)
            found->second->cache().set_log_window(enabled);
    }

    Prediction prediction(const cldnn::primitive_id& id) const {
        auto found = m_predictions.find(id);
        return found == m_predictions.end() ? Prediction{} : found->second;
    }

    void predict(const LookaheadLayerDesc& next_layer,
                 const cldnn::memory::ptr& hidden,
                 const cldnn::event::ptr& ready);
    void prefetch(const cldnn::primitive_inst* current_moe);

private:
    cldnn::network& m_network;
    cldnn::stream::ptr m_compute_stream;
    std::vector<cldnn::program::les_router_link> m_links;
    std::once_flag m_init_flag;
    std::unordered_map<cldnn::primitive_id, uint32_t> m_layer_indices;
    std::unordered_map<cldnn::primitive_id, size_t> m_source_links;
    std::vector<LookaheadLayerDesc> m_layers;
    std::unordered_map<const cldnn::primitive_inst*, size_t> m_moe_to_next_index;
    uint64_t m_epoch = 0;
    std::optional<uint32_t> m_cursor;
    std::unordered_map<cldnn::primitive_id, std::shared_ptr<OffloadExpertWeightProvider>> m_providers;
    std::unordered_map<cldnn::primitive_id, std::unique_ptr<LookaheadRouter>> m_routers;
    std::unordered_map<cldnn::primitive_id, Prediction> m_predictions;
    std::unique_ptr<ExpertPrefetcher> m_prefetcher;
    std::string m_baseline_request;
    size_t m_baseline_completed = 0;
    std::unordered_map<cldnn::primitive_id, CacheSnapshot> m_cache_baselines;
    std::unordered_map<cldnn::primitive_id, CacheSnapshot> m_prefill_baselines;
    struct Handoff {
        cldnn::primitive_id source;
        size_t token = 0;
        size_t checkpoint_events = 0;
        int64_t checkpoint_us = 0, prefetch_us = 0;
        cldnn::memory::ptr predicted_indices;
    };
    std::unordered_map<cldnn::primitive_id, Handoff> m_handoffs;
};

class LookaheadRegistry {
public:
    static std::shared_ptr<LookaheadContext> acquire(cldnn::network& network);
};

}