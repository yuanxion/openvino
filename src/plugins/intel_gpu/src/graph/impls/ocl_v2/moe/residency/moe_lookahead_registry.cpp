#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_lookahead_registry.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace ov::intel_gpu::ocl::moe {

LookaheadContext::LookaheadContext(cldnn::network& network)
    : m_network(network), m_compute_stream(network.get_stream_ptr()) {
    build_once(network);
    m_prefetcher = std::make_unique<ExpertPrefetcher>(network.get_engine(), network.get_config());
}

LookaheadContext::~LookaheadContext() {
    m_prefetcher.reset();
    try {
        m_compute_stream->finish();
    } catch (const std::exception& exception) {
        GPU_DEBUG_INFO << "Lookahead compute teardown failed: " << exception.what() << std::endl;
    }
}

void LookaheadContext::pause_for_prefill() {
    m_compute_stream->flush();
    m_prefetcher->advance(++m_epoch, std::numeric_limits<uint32_t>::max());
    m_prefetcher->drain();
    m_cursor.reset();
}

void LookaheadContext::capture_cache_baseline() {
    const char* path = std::getenv("LOG_BASELINE_CSV_PATH");
    if (!path || !*path || m_providers.empty())
        return;
    const char* context = std::getenv("LOG_REQUEST_CONTEXT");
    const char* selection = std::getenv("LOG_REQUEST");
    OPENVINO_ASSERT(context && *context, "LES baseline requires request labels");
    if (selection && *selection && std::string(selection) != context)
        return;
    const char* run = std::getenv("LOG_RUN_ID");
    OPENVINO_ASSERT(run && *run && std::getenv("LOG_TOKEN_ID"), "LES baseline requires LOG_RUN_ID and LOG_TOKEN_ID");
    OPENVINO_ASSERT(m_providers.size() == m_layers.size(), "LES baseline requires all layer providers");
    if (m_baseline_request != context) {
        m_baseline_request = context;
        m_baseline_completed = 0;
        m_cache_baselines.clear();
        m_prefill_baselines.clear();
    } else {
        ++m_baseline_completed;
    }
    for (const auto& layer : m_layers) {
        const auto& id = layer.moe_inst->id();
        auto ready = m_providers.at(id)->cache().resident_experts();
        const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        m_cache_baselines[id] = {m_baseline_completed, now, std::move(ready)};
    }
    if (m_baseline_completed == 0)
        m_prefill_baselines = m_cache_baselines;
    static std::mutex csv_mutex;
    std::lock_guard<std::mutex> lock(csv_mutex);
    static std::ofstream csv(path);
    OPENVINO_ASSERT(csv.is_open(), "Cannot open LOG_BASELINE_CSV_PATH");
    if (csv.tellp() == std::streampos(0)) {
        csv << "\xEF\xBB\xBF"
            << "运行标识,请求标识,完成decode编号,阶段,层ID,快照时间戳(微秒),ready专家ids,ready专家个数\n";
    }
    const auto text = [&](const std::string& value) {
        csv << '"';
        for (const char character : value) {
            if (character == '"')
                csv << '"';
            csv << character;
        }
        csv << '"';
    };
    for (const auto& layer : m_layers) {
        const auto& id = layer.moe_inst->id();
        const auto& snapshot = m_cache_baselines.at(id);
        text(run); csv << ',';
        text(m_baseline_request); csv << ',' << snapshot.completed_token << ',';
        csv << (snapshot.completed_token == 0 ? "prefill" : "decode") << ',';
        text(id); csv << ',' << snapshot.time_us << ",\"[";
        for (size_t index = 0; index < snapshot.ready.size(); ++index) {
            if (index)
                csv << ',';
            csv << snapshot.ready[index];
        }
        csv << "]\"," << snapshot.ready.size() << '\n';
    }
    csv.flush();
    OPENVINO_ASSERT(csv.good(), "Cannot write LOG_BASELINE_CSV_PATH");
}

LookaheadContext::CacheSnapshot LookaheadContext::cache_baseline(const cldnn::primitive_id& id,
                                                               const std::string& request,
                                                               size_t decode_token,
                                                               bool prefill) const {
    OPENVINO_ASSERT(request == m_baseline_request && m_baseline_completed + 1 == decode_token,
                    "LES baseline request/token mismatch");
    const auto& snapshots = prefill ? m_prefill_baselines : m_cache_baselines;
    OPENVINO_ASSERT(snapshots.count(id), "Missing LES cache baseline for ", id);
    return snapshots.at(id);
}

void LookaheadContext::begin_handoff(const cldnn::primitive_id& source, size_t token, bool selected) {
    const char* path = std::getenv("LOG_HANDOFF_CSV_PATH");
    if (!path || !*path || !selected)
        return;
    OPENVINO_ASSERT(token > 0 && std::getenv("LOG_RUN_ID"), "LES handoff requires token selection and LOG_RUN_ID");
    const auto index = m_layer_indices.at(source);
    if (index == 0)
        OPENVINO_ASSERT(m_handoffs.empty(), "Unfinished LES handoff from previous token");
    if (index + 1 == m_layers.size())
        return;
    const auto& target = m_layers[index + 1].moe_inst->id();
    OPENVINO_ASSERT(!m_handoffs.count(target), "Duplicate LES handoff");
    m_providers.at(target)->cache().begin_trace();
    m_handoffs.emplace(target, Handoff{source, token});
}

void LookaheadContext::checkpoint_handoff(const cldnn::primitive_id& source) {
    const auto index = m_layer_indices.at(source);
    if (index + 1 == m_layers.size())
        return;
    const auto& target = m_layers[index + 1].moe_inst->id();
    auto found = m_handoffs.find(target);
    if (found != m_handoffs.end()) {
        const auto checkpoint = m_providers.at(target)->cache().trace_checkpoint();
        found->second.checkpoint_events = checkpoint.first;
        found->second.checkpoint_us = checkpoint.second;
    }
}

void LookaheadContext::finish_handoff(const cldnn::primitive_id& target,
                                    size_t token,
                                    const std::vector<uint32_t>& actual) {
    const char* path = std::getenv("LOG_HANDOFF_CSV_PATH");
    if (!path || !*path)
        return;
    const auto index = m_layer_indices.at(target);
    LRUCache::Trace trace;
    Handoff handoff;
    if (index == 0) {
        trace = m_providers.at(target)->cache().snapshot();
    } else {
        auto found = m_handoffs.find(target);
        OPENVINO_ASSERT(found != m_handoffs.end() && found->second.token == token, "LES handoff token mismatch");
        handoff = found->second;
        OPENVINO_ASSERT(handoff.checkpoint_us != 0, "LES handoff source checkpoint missing");
        trace = m_providers.at(target)->cache().end_trace();
        m_handoffs.erase(found);
        std::set<size_t> reconstructed(trace.original.begin(), trace.original.end());
        for (const auto& event : trace.events) {
            if (event.kind == 'P' || event.kind == 'I' || event.kind == 'D')
                reconstructed.insert(event.expert);
            else if (event.kind == 'E' || event.kind == 'X')
                reconstructed.erase(event.expert);
        }
        OPENVINO_ASSERT(reconstructed == std::set<size_t>(trace.current.begin(), trace.current.end()),
                        "LES handoff event replay differs from actual ready cache");
    }
    const auto collect = [&](char kind, size_t limit) {
        std::vector<size_t> result;
        for (size_t event_index = 0; event_index < limit; ++event_index) {
            const auto& event = trace.events[event_index];
            if (event.kind == kind && std::find(result.begin(), result.end(), event.expert) == result.end())
                result.push_back(event.expert);
        }
        return result;
    };
    OPENVINO_ASSERT(handoff.checkpoint_events <= trace.events.size(), "Invalid handoff checkpoint");
    std::vector<size_t> hits;
    for (const auto expert : actual) {
        if (std::find(trace.current.begin(), trace.current.end(), expert) != trace.current.end())
            hits.push_back(expert);
    }
    std::vector<size_t> predicted;
    size_t recall = 0;
    const bool prediction_observed = handoff.predicted_indices != nullptr;
    if (prediction_observed) {
        OPENVINO_ASSERT(handoff.predicted_indices->get_allocation_type() == cldnn::allocation_type::usm_host &&
                            handoff.predicted_indices->count() == actual.size(),
                        "Invalid handoff prediction buffer");
        cldnn::mem_lock<uint32_t, cldnn::mem_lock_type::read> indices(handoff.predicted_indices, *m_compute_stream);
        predicted.assign(indices.data(), indices.data() + actual.size());
        for (const auto expert : actual) {
            if (std::find(predicted.begin(), predicted.end(), expert) != predicted.end())
                ++recall;
        }
    }
    static std::mutex csv_mutex;
    std::lock_guard<std::mutex> lock(csv_mutex);
    static std::ofstream csv(path);
    OPENVINO_ASSERT(csv.is_open(), "Cannot open LOG_HANDOFF_CSV_PATH");
    if (csv.tellp() == std::streampos(0)) {
        csv << "\xEF\xBB\xBF"
            << "运行标识,请求标识,Token编号,源层ID,目标层ID,边界,跟踪开始时间戳(微秒),"
            << "预取提交时间戳(微秒),源层CPU提交结束时间戳(微秒),目标层readback时间戳(微秒),"
            << "1-原始ready专家ids,1-原始ready专家个数,"
            << "2-源层结束前prefetched专家ids,2-源层结束前prefetched专家个数,"
            << "3-源层结束前驱逐的专家ids,3-源层结束前驱逐的专家个数,"
            << "4-到层已完成prefetched专家ids,4-到层已完成prefetched专家个数,"
            << "5-到层累计驱逐的专家ids,5-到层累计驱逐的专家个数,"
            << "6-到层直接查询ready专家ids,6-到层直接查询ready专家个数,"
            << "7-到层hit专家ids,7-到层hit专家个数,"
            << "继承在途完成的专家ids,继承在途完成的专家个数,"
            << "需求加载完成的专家ids,需求加载完成的专家个数,"
            << "取消的专家ids,取消的专家个数,真实topk专家ids,有序缓存事件,事件重放一致,"
            << "源层预测专家ids,源层预测专家个数,预测recall个数,"
            << "起点pending专家ids,起点pending专家个数,起点搬运中专家ids,起点搬运中专家个数,"
            << "到层pending专家ids,到层pending专家个数,到层搬运中专家ids,到层搬运中专家个数\n";
    }
    const auto text = [&](const std::string& value) {
        csv << '"';
        for (const char character : value) {
            if (character == '"')
                csv << '"';
            csv << character;
        }
        csv << '"';
    };
    const auto ids = [&](const std::vector<size_t>& experts, bool observed = true) {
        if (!observed) {
            csv << "NA,NA";
            return;
        }
        csv << "\"[";
        for (size_t expert_index = 0; expert_index < experts.size(); ++expert_index) {
            if (expert_index)
                csv << ',';
            csv << experts[expert_index];
        }
        csv << "]\"," << experts.size();
    };
    const char* request = std::getenv("LOG_REQUEST_CONTEXT");
    const char* run = std::getenv("LOG_RUN_ID");
    OPENVINO_ASSERT(request && run, "LES handoff requires request/run labels");
    text(run); csv << ','; text(request); csv << ',' << token << ',';
    text(index == 0 ? "NA" : handoff.source); csv << ','; text(target); csv << ',';
    csv << (index == 0 ? "first_no_source" : (index + 1 == m_layers.size() ? "last_no_next" : "normal")) << ',';
    if (index == 0)
        csv << "NA,NA,NA,";
    else {
        csv << trace.start_us << ',';
        if (handoff.prefetch_us)
            csv << handoff.prefetch_us;
        else
            csv << "NA";
        csv << ',' << handoff.checkpoint_us << ',';
    }
    csv << trace.end_us << ',';
    ids(trace.original, index != 0); csv << ',';
    ids(collect('P', handoff.checkpoint_events), index != 0); csv << ',';
    ids(collect('E', handoff.checkpoint_events), index != 0); csv << ',';
    ids(collect('P', trace.events.size()), index != 0); csv << ',';
    ids(collect('E', trace.events.size()), index != 0); csv << ',';
    ids(trace.current); csv << ','; ids(hits); csv << ',';
    ids(collect('I', trace.events.size()), index != 0); csv << ',';
    ids(collect('D', trace.events.size()), index != 0); csv << ',';
    ids(collect('X', trace.events.size()), index != 0); csv << ',';
    std::ostringstream actual_text;
    actual_text << '[';
    for (size_t expert_index = 0; expert_index < actual.size(); ++expert_index) {
        if (expert_index)
            actual_text << ',';
        actual_text << actual[expert_index];
    }
    actual_text << ']'; text(actual_text.str()); csv << ',';
    std::ostringstream events;
    events << '[';
    for (size_t event_index = 0; event_index < trace.events.size(); ++event_index) {
        if (event_index)
            events << ',';
        const auto& event = trace.events[event_index];
        events << '[' << event_index + 1 << ',' << event.time_us << ",\"" << event.kind << "\"," << event.expert << ']';
    }
    events << ']'; text(events.str()); csv << (index == 0 ? ",NA," : ",true,");
    ids(predicted, prediction_observed); csv << ',';
    if (prediction_observed)
        csv << recall;
    else
        csv << "NA";
    csv << ','; ids(trace.original_pending, index != 0);
    csv << ','; ids(trace.original_copying, index != 0);
    csv << ','; ids(trace.current_pending);
    csv << ','; ids(trace.current_copying);
    csv << '\n';
    csv.flush();
    OPENVINO_ASSERT(csv.good(), "Cannot write LOG_HANDOFF_CSV_PATH");
}

void LookaheadContext::build_once(cldnn::network& network) {
    std::call_once(m_init_flag, [&] {
        m_links = network.get_program()->get_les_router_links();
        for (const auto& id : network.get_executed_primitive_ids()) {
            auto instance = network.get_primitive(id);
            if (instance->type() == cldnn::moe_3gemm_fused_compressed::type_id()) {
                m_layer_indices.emplace(id, static_cast<uint32_t>(m_layer_indices.size()));
                LookaheadLayerDesc descriptor;
                descriptor.moe_inst = instance.get();
                m_layers.push_back(std::move(descriptor));
            }
        }
        for (size_t index = 0; index < m_links.size(); ++index) {
            const auto& link = m_links[index];
            if (link.status != cldnn::program::les_router_status::supported ||
                !m_layer_indices.count(link.chain.source_moe) || !m_layer_indices.count(link.chain.target_moe)) {
                continue;
            }
            m_source_links.emplace(link.chain.source_moe, index);
            const auto target_index = m_layer_indices.at(link.chain.target_moe);
            auto& descriptor = m_layers[target_index];
            descriptor.router_fc_inst = network.get_primitive(link.projection).get();
            descriptor.weight_mem = network.get_primitive(link.weights)->output_memory_ptr();
            descriptor.num_experts = static_cast<uint32_t>(link.experts);
            const auto shape = network.get_program()->get_node(link.chain.target_hidden).get_output_layout().get_partial_shape();
            if (shape.rank().is_static() && shape.size() > 0 && shape[shape.size() - 1].is_static()) {
                descriptor.hidden_size = static_cast<uint32_t>(shape[shape.size() - 1].get_length());
            }
            descriptor.is_supported = true;
            m_moe_to_next_index.emplace(network.get_primitive(link.chain.source_moe).get(), target_index);
            m_routers.emplace(link.chain.target_moe,
                              std::make_unique<LookaheadRouter>(network, link.chain.target_hidden, link.router, link.projection));
        }
    });
}

const LookaheadLayerDesc* LookaheadContext::get_next_layer(const cldnn::primitive_inst* current_moe) const {
    const auto found = m_moe_to_next_index.find(current_moe);
    if (found == m_moe_to_next_index.end()) {
        return nullptr;
    }
    const auto& descriptor = m_layers[found->second];
    return descriptor.is_supported ? &descriptor : nullptr;
}

std::optional<TimestampLogRecord> LookaheadContext::timestamp_snapshot(const cldnn::primitive_id& id,
                                                                     const std::vector<size_t>& topk_experts) const {
    const auto provider = m_providers.find(id);
    if (provider == m_providers.end())
        return std::nullopt;
    return provider->second->cache().timestamp_snapshot(topk_experts);
}

std::pair<TimestampLogRecord, TimestampLogRecord> LookaheadContext::timestamp_pair(
    const cldnn::primitive_id& current, const cldnn::primitive_id& target) const {
    return LRUCache::timestamp_pair(m_providers.at(current)->cache(), m_providers.at(target)->cache());
}

void LookaheadContext::set_timestamp_arguments(const cldnn::primitive_id& id, const std::vector<size_t>& topk,
                                             const std::vector<size_t>& demand) {
    m_providers.at(id)->cache().set_timestamp_arguments(topk, demand);
}

void LookaheadContext::set_timestamp_prediction(const cldnn::primitive_id& id, const std::vector<size_t>& predicted,
                                              const std::vector<size_t>& submitted) {
    m_providers.at(id)->cache().set_timestamp_prediction(predicted, submitted);
}

void LookaheadContext::prepare_timestamp_window(const cldnn::primitive_id& id, size_t token, bool emit) {
    if (!LESLog::instance().enabled() && !LESTimeline::enabled())
        return;
    const auto found = m_layer_indices.find(id);
    if (found == m_layer_indices.end() || (m_cursor && found->second > *m_cursor) || m_timestamp_epoch == m_epoch + 1)
        return;
    for (const auto& provider : m_providers) {
        provider.second->cache().set_log_window(true, true);
        provider.second->cache().set_log_context(token, layer_index(provider.first), emit);
    }
    m_timestamp_epoch = m_epoch + 1;
}

void LookaheadContext::enter(const cldnn::primitive_id& id) {
    m_prefetcher->rethrow_if_failed();
    auto found = m_layer_indices.find(id);
    if (found == m_layer_indices.end()) {
        return;
    }
    if (!m_cursor || found->second <= *m_cursor) {
        prepare_timestamp_window(id);
        ++m_epoch;
        m_prefetcher->advance(m_epoch, 0);
    }
    m_cursor = found->second;
}

void LookaheadContext::demand_partitioned(const cldnn::primitive_id& id) {
    auto found = m_layer_indices.find(id);
    if (found != m_layer_indices.end()) {
        m_prefetcher->advance(m_epoch, found->second);
    }
}

std::shared_ptr<IExpertWeightProvider> LookaheadContext::bind(const cldnn::primitive_id& id,
                                                            std::shared_ptr<IExpertWeightProvider> provider) {
    const auto existing = m_providers.find(id);
    if (existing != m_providers.end()) {
        return existing->second;
    }
    auto offload = std::dynamic_pointer_cast<OffloadExpertWeightProvider>(provider);
    if (offload) {
        offload->cache().set_log_layer(id);
        m_prefetcher->bind(offload);
        m_providers[id] = std::move(offload);
    }
    return provider;
}

void LookaheadContext::predict(const LookaheadLayerDesc& next_layer,
                               const cldnn::memory::ptr& hidden,
                               const cldnn::event::ptr& ready) {
    if (!next_layer.is_supported || !next_layer.moe_inst) {
        return;
    }
    const auto& target = next_layer.moe_inst->id();
    auto& state = m_predictions[target];
    if (state.consumed && !state.consumed->load(std::memory_order_acquire)) {
        return;
    }
    auto result = m_routers.at(target)->enqueue(hidden, ready);
    state = {hidden, ready, result.indices, result.weights, result.ready,
             std::make_shared<std::atomic<bool>>(true), false, result.cpu_indices};
    auto handoff = m_handoffs.find(target);
    if (handoff != m_handoffs.end())
        handoff->second.predicted_indices = result.indices;
}

bool LookaheadContext::prefetch(const cldnn::primitive_inst* current_moe) {
    const auto* next_layer = get_next_layer(current_moe);
    if (!next_layer) {
        return false;
    }
    const auto& target = next_layer->moe_inst->id();
    auto& state = m_predictions[target];
    auto provider = m_providers.find(target);
    if (state.submitted || !state.prediction_ready || provider == m_providers.end()) {
        return false;
    }
    auto weights = provider->second;
    const auto& link = m_links.at(m_source_links.at(current_moe->id()));
    if (!weights || weights->resident_capacity() <= link.top_k) {
        return false;
    }
    state.submitted = true;
    state.consumed->store(false, std::memory_order_release);
    m_network.get_stream().flush();
    auto handoff = m_handoffs.find(target);
    if (handoff != m_handoffs.end()) {
        handoff->second.prefetch_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    auto trace_context = m_providers.at(current_moe->id())->cache().load_trace_context();
    if (trace_context) {
        auto context = std::make_shared<LESLoadContext>(*trace_context);
        context->target_layer_index = m_layer_indices.at(target);
        trace_context = std::move(context);
    }
    m_prefetcher->submit_prefetch({m_layer_indices.at(target), m_epoch,
                                  state.predicted_indices, state.prediction_ready, weights, state.consumed,
                                  state.cpu_indices, std::move(trace_context)});
    return true;
}

std::shared_ptr<LookaheadContext> LookaheadRegistry::acquire(cldnn::network& network) {
    const auto& config = network.get_config();
    if (!config.get_moe_lookahead_prefetch() || config.get_offload_ratio() == 0 || network.get_id() == 0) {
        return {};
    }
    OPENVINO_ASSERT(config.get_offload_ratio() < 100 && config.get_num_streams() == 1 && !config.get_record_replay(),
                    "LES requires offload ratio in [1,99], one stream and no record/replay");
    static std::mutex mutex;
    static std::unordered_map<uint32_t, std::weak_ptr<LookaheadContext>> registry;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto entry = registry.begin(); entry != registry.end();) {
        entry = entry->second.expired() ? registry.erase(entry) : std::next(entry);
    }
    auto& entry = registry[network.get_id()];
    auto context = entry.lock();
    if (!context) {
        context = std::make_shared<LookaheadContext>(network);
        entry = context;
    }
    return context;
}

}
