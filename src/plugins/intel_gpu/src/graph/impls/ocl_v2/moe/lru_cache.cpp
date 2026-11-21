// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lru_cache.hpp"
#include <algorithm>
#include <chrono>
#include <unordered_set>
#include "openvino/core/except.hpp"

namespace ov::intel_gpu::ocl::moe {

void LRUCache::log_cache_state_unlocked(char kind, size_t expert, int64_t event_us,
                                       const std::string& details, bool snapshot) const {
    const char* prefix = les_changed_prefix;
    if (kind == 'R' || kind == 'F' || kind == 'J')
        prefix = les_queued_prefix;
    else if (kind == 'B')
        prefix = les_copying_prefix;
    else if (kind == 'D')
        prefix = les_ready_prefix;
    else if (kind == 'E' || kind == 'X')
        prefix = les_evicted_prefix;
    std::vector<size_t> experts;
    if (kind == 'E' || kind == 'X') {
        experts.push_back(expert);
    } else {
        for (const auto& node : m_list) {
            const bool queued = node.state.loading && !node.state.copying;
            const bool selected = (kind == 'R' || kind == 'F' || kind == 'J') ? queued :
                                  kind == 'B' ? node.state.copying : m_filled_list[node.lru_expert_no];
            if (selected)
                experts.push_back(node.expert);
        }
    }
    std::ostringstream text;
    text << "event=" << (snapshot ? "cache_state" : "cache_change") << " cache=" << this
         << " window=" << m_log_window_id << " state=" << kind << " event_us=" << event_us;
    if (!snapshot) {
        text << " expert=" << expert;
        const auto found = m_map.find(Key{expert});
        if (found != m_map.end())
            text << ' ' << log_node_unlocked(*found->second);
    }
    text << details << u8" \u6709\u5e8f\u7f13\u5b58\u4e8b\u4ef6=[";
    for (size_t index = 0; index < m_ordered_cache_events.size(); ++index) {
        if (index)
            text << ',';
        const auto& event = m_ordered_cache_events[index];
        text << '[' << index + 1 << ',' << event.time_us << ",\"" << event.kind << "\"," << event.expert << ']';
    }
    text << ']';
    LESLog::instance().write(m_log_layer, __FILE__, __LINE__, __func__, text.str(), prefix, les_ids(experts));
}

void LRUCache::record_cache_event_unlocked(char kind, size_t expert, const std::string& details) {
    MOE_LES_LOG(m_log_layer, "event=record_cache_event_unlocked kind=" << kind << " expert=" << expert);
    if (!LESLog::instance().enabled() || !m_log_window)
        return;
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    m_ordered_cache_events.push_back({now, kind, expert});
    log_cache_state_unlocked(kind, expert, now, details);
}

std::vector<LRUCache::TraceEvent> LRUCache::ordered_cache_events() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_ordered_cache_events;
}

void LRUCache::set_log_window(bool enabled) {
    if (!LESLog::instance().enabled())
        return;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_log_window == enabled)
        return;
    if (enabled) {
        m_ordered_cache_events.clear();
        m_log_window = true;
        ++m_log_window_id;
        // MOE_LES_LOG(m_log_layer, "event=cache_window phase=begin cache=" << this << " window=" << m_log_window_id);

    }
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    for (const char kind : {'D', 'B', 'R'})
        log_cache_state_unlocked(kind, 0, now, enabled ? " boundary=begin" : " boundary=end", true);
    if (!enabled) {
        MOE_LES_LOG(m_log_layer, "event=cache_window phase=end cache=" << this << " window=" << m_log_window_id);
        m_log_window = false;
    }
}

void LRUCache::log_enqueue(const Load& load, uint64_t epoch, size_t queue_depth) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto found = m_map.find(Key{load.expert_id});
    if (found == m_map.end() || found->second->state.generation != load.generation)
        return;
    std::ostringstream details;
    details << " origin=speculative epoch=" << epoch << " queue_depth=" << queue_depth;
    record_cache_event_unlocked('J', load.expert_id, details.str());
}

std::string LRUCache::log_node_unlocked(const Node& node) const {
    std::ostringstream text;
    text << "cache=" << this << " expert=" << node.expert << " slot=" << node.lru_expert_no
         << " generation=" << node.state.generation << " status=" << static_cast<int>(node.state.status)
         << " filled=" << m_filled_list[node.lru_expert_no] << " loading=" << node.state.loading
         << " copying=" << node.state.copying << " demanded=" << node.state.demanded
         << " speculative=" << node.state.speculative << " ready_event=" << node.state.ready_event.get()
         << " last_use_event=" << node.state.last_use_event.get();
    return text.str();
}

void LRUCache::trace_event_unlocked(char kind, size_t expert) {
    if (m_trace_active) {
        const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        m_trace.events.push_back({now, kind, expert});
    }
}

void LRUCache::begin_trace() {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(!m_trace_active, "LRU trace already active");
    MOE_LES_LOG(m_log_layer, "event=trace_begin cache=" << this);
    m_trace = {};
    m_trace_generation = m_generation;
    for (const auto& node : m_list) {
        if (m_filled_list[node.lru_expert_no])
            m_trace.original.push_back(node.expert);
        if (node.state.loading)
            m_trace.original_pending.push_back(node.expert);
        if (node.state.copying)
            m_trace.original_copying.push_back(node.expert);
    }
    m_trace.start_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    m_trace_active = true;
}

std::pair<size_t, int64_t> LRUCache::trace_checkpoint() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(m_trace_active, "LRU trace was not started");
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return {m_trace.events.size(), now};
}

void LRUCache::capture_snapshot_unlocked(Trace& trace) const {
    for (const auto& node : m_list) {
        if (m_filled_list[node.lru_expert_no])
            trace.current.push_back(node.expert);
        if (node.state.loading)
            trace.current_pending.push_back(node.expert);
        if (node.state.copying)
            trace.current_copying.push_back(node.expert);
        if (node.state.loading && !node.state.copying)
            trace.current_queued.push_back(node.expert);
    }
    trace.end_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

LRUCache::Trace LRUCache::snapshot() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    Trace trace;
    capture_snapshot_unlocked(trace);
    return trace;
}

LRUCache::Trace LRUCache::end_trace() {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(m_trace_active, "LRU trace was not started");
    MOE_LES_LOG(m_log_layer, "event=trace_end cache=" << this);
    capture_snapshot_unlocked(m_trace);
    m_trace_active = false;
    return std::move(m_trace);
}

ExpertSlotReservations::ExpertSlotReservations(size_t capacity) : _slots(capacity) {
    if (capacity == 0) {
        OPENVINO_THROW("LES requires at least one resident slot");
    }
}

std::optional<ExpertSlotReservations::Ticket> ExpertSlotReservations::reserve(const std::vector<uint32_t>& experts, uint64_t epoch, bool speculative) {
    std::lock_guard<std::mutex> lock(_mutex);
    std::unordered_set<uint32_t> unique(experts.begin(), experts.end());
    if (unique.size() > _slots.size() || epoch < _epoch || (epoch != _epoch && !_active.empty())) {
        return std::nullopt;
    }
    auto planned = _slots;
    std::unordered_map<uint32_t, size_t> mapping;
    Record record{epoch, {}, {}};
    Ticket result{_next_id + 1, epoch, {}, {}, {}, {}, {}};
    auto clock = _clock;
    for (auto expert : unique) {
        for (size_t index = 0; index < planned.size(); ++index) {
            auto& slot = planned[index];
            if (slot.expert == expert) {
                if (!slot.ready) {
                    return std::nullopt;
                }
                ++slot.pins;
                if (!speculative) {
                    slot.age = ++clock;
                }
                mapping.emplace(expert, index);
                record.generations.emplace_back(index, slot.generation);
                break;
            }
        }
    }
    for (auto expert : experts) {
        if (mapping.count(expert)) {
            continue;
        }
        size_t candidate = planned.size();
        for (size_t index = 0; index < planned.size(); ++index) {
            const auto& slot = planned[index];
            if (slot.pins == 0 &&
                (candidate == planned.size() || !slot.expert || slot.age < planned[candidate].age)) {
                candidate = index;
                if (!slot.expert) {
                    break;
                }
            }
        }
        if (candidate == planned.size()) {
            return std::nullopt;
        }
        auto& slot = planned[candidate];
        if (speculative) {
            record.replaced.emplace_back(candidate, slot);
        }
        slot.expert = expert;
        ++slot.generation;
        slot.age = ++clock;
        slot.pins = 1;
        slot.ready = false;
        mapping.emplace(expert, candidate);
        record.generations.emplace_back(candidate, slot.generation);
        result.missing_experts.push_back(expert);
        result.missing_slots.push_back(candidate);
    }
    for (size_t pair = 0; pair < experts.size(); ++pair) {
        auto slot = mapping.at(experts[pair]);
        result.slots.push_back(slot);
        (planned[slot].ready ? result.ready_pairs : result.missing_pairs).push_back(pair);
    }
    _active.emplace(result.id, std::move(record));
    _slots.swap(planned);
    _epoch = epoch;
    _clock = clock;
    ++_next_id;
    return result;
}

void ExpertSlotReservations::discard_unused(const Ticket& ticket, const std::vector<uint32_t>& used) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto record = _active.find(ticket.id);
    if (record == _active.end() || record->second.epoch != ticket.epoch) {
        return;
    }
    for (size_t index = 0; index < ticket.missing_slots.size(); ++index) {
        auto slot_index = ticket.missing_slots[index];
        auto generation = std::find_if(record->second.generations.begin(), record->second.generations.end(),
                                       [&](const auto& entry) { return entry.first == slot_index; });
        auto& slot = _slots[slot_index];
        if (generation != record->second.generations.end() && slot.generation == generation->second && slot.pins == 1 &&
            std::find(used.begin(), used.end(), ticket.missing_experts[index]) == used.end()) {
            slot.ready = false;
        }
    }
}

bool ExpertSlotReservations::rollback(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto record = _active.find(ticket.id);
    if (record == _active.end() || record->second.epoch != ticket.epoch) {
        return false;
    }
    for (const auto& entry : record->second.generations) {
        if (_slots[entry.first].generation != entry.second || _slots[entry.first].pins != 1) {
            return false;
        }
    }
    for (const auto& entry : record->second.generations) {
        --_slots[entry.first].pins;
    }
    for (const auto& entry : record->second.replaced) {
        _slots[entry.first] = entry.second;
    }
    _active.erase(record);
    return true;
}

bool ExpertSlotReservations::complete(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto record = _active.find(ticket.id);
    if (record == _active.end() || record->second.epoch != ticket.epoch) {
        return false;
    }
    for (const auto& entry : record->second.generations) {
        if (_slots[entry.first].generation != entry.second) {
            return false;
        }
    }
    for (const auto& entry : record->second.generations) {
        _slots[entry.first].ready = true;
    }
    return true;
}

bool ExpertSlotReservations::release(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto record = _active.find(ticket.id);
    if (record == _active.end() || record->second.epoch != ticket.epoch) {
        return false;
    }
    for (const auto& entry : record->second.generations) {
        auto& slot = _slots[entry.first];
        --slot.pins;
        if (!slot.ready && slot.pins == 0) {
            slot.expert.reset();
        }
    }
    _active.erase(record);
    return true;
}

LRUCache::LRUCache(size_t max_total_experts) : m_max_total_experts(max_total_experts), m_total_experts(0), m_to_filled_lru_expert_no(0) {
    m_filled_list.resize(max_total_experts, false);
    m_slot_last_use.resize(max_total_experts);
}

void LRUCache::move_to_end(std::list<Node>::iterator it) {
    if (std::next(it) == m_list.end()) {
        return;
    }
    m_list.splice(m_list.end(), m_list, it);
    record_cache_event_unlocked('T', it->expert);
}

std::optional<LRUCache::Partition> LRUCache::partition_3way(const std::vector<uint32_t>& topk_ids,
                                                        const EventFactory& make_event,
                                                        bool allow_pending_reuse) {
    MOE_LES_LOG(m_log_layer, "event=partition_3way topk_ids=" << les_ids(topk_ids));
    std::lock_guard<std::mutex> lock(m_mutex);
    const std::unordered_set<uint32_t> requested(topk_ids.begin(), topk_ids.end());
    if (requested.size() > m_max_total_experts) {
        return std::nullopt;
    }
    for (auto expert : requested) {
        auto found = m_map.find(Key{expert});
        OPENVINO_ASSERT(found == m_map.end() || found->second->state.loading ||
                            m_filled_list[found->second->lru_expert_no],
                        "Lookahead cannot consume an unfilled OTD slot");
    }
    Partition result;
    std::unordered_set<uint32_t> missing;
    std::vector<bool> occupied(m_max_total_experts, false);
    for (const auto& node : m_list) {
        occupied[node.lru_expert_no] = true;
    }
    std::vector<std::list<Node>::iterator> victims;
    for (auto expert : topk_ids) {
        if (m_map.count(Key{expert}) || !missing.insert(expert).second) {
            continue;
        }
        auto free_slot = std::find(occupied.begin(), occupied.end(), false);
        size_t slot = static_cast<size_t>(free_slot - occupied.begin());
        cldnn::event::ptr last_use;
        if (free_slot == occupied.end()) {
            auto victim = std::find_if(m_list.begin(), m_list.end(), [&](const Node& node) {
                return !requested.count(static_cast<uint32_t>(node.expert)) &&
                                             ((node.state.status == SlotStatus::READY && !node.state.loading) ||
                                                (allow_pending_reuse && node.state.status == SlotStatus::PENDING_LOAD &&
                                                 node.state.demanded && node.state.last_use_event)) &&
                       std::none_of(victims.begin(), victims.end(), [&](const auto& chosen) { return &*chosen == &node; });
            });
            if (victim == m_list.end()) {
                return std::nullopt;
            }
            victims.push_back(victim);
            slot = victim->lru_expert_no;
            last_use = victim->state.last_use_event;
        } else {
            occupied[slot] = true;
            last_use = m_slot_last_use[slot];
        }
        result.missing.push_back({expert, slot, 0, {}, last_use});
    }
    for (auto& load : result.missing) {
        load.ready_event = make_event();
        OPENVINO_ASSERT(load.ready_event, "Lookahead load requires a completion event");
    }
    for (auto victim : victims) {
        const auto expert = victim->expert;
        trace_event_unlocked('E', victim->expert);
        m_map.erase(Key{victim->expert});
        m_list.erase(victim);
        --m_total_experts;
        record_cache_event_unlocked('E', expert);
    }
    for (auto& load : result.missing) {
        load.generation = ++m_generation;
        m_list.push_back(Node{load.expert_id, load.slot,
                             {SlotStatus::PENDING_LOAD, load.ready_event, load.last_use_event, load.generation, true}});
        m_map.emplace(Key{load.expert_id}, std::prev(m_list.end()));
        m_filled_list[load.slot] = false;
        ++m_total_experts;
        m_list.back().state.reserved_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        record_cache_event_unlocked('R', load.expert_id, " origin=demand");
        trace_event_unlocked('Q', load.expert_id);
    }
    std::unordered_set<uint32_t> waiting;
    for (size_t pair = 0; pair < topk_ids.size(); ++pair) {
        const auto expert = topk_ids[pair];
        auto node = m_map.at(Key{expert});
        result.slots.push_back(node->lru_expert_no);
        if (missing.count(expert)) {
            result.missing_pairs.push_back(pair);
        } else if (node->state.loading) {
            result.inflight_pairs.push_back(pair);
        } else {
            OPENVINO_ASSERT(m_filled_list[node->lru_expert_no], "Lookahead cannot consume an unfilled OTD slot");
            result.ready_pairs.push_back(pair);
            if (node->state.ready_event && waiting.insert(expert).second) {
                result.ready_events.push_back(node->state.ready_event);
            }
        }
        if (node->state.loading && waiting.insert(expert).second) {
            result.wait_events.push_back(node->state.ready_event);
        }
        node->state.status = SlotStatus::PINNED;
        node->state.demanded = true;
        record_cache_event_unlocked('P', expert);
        move_to_end(node);
    }
    return result;
}

std::optional<LRUCache::Load> LRUCache::reserve_speculative(uint32_t expert_id,
                                                         size_t top_k,
                                                         const EventFactory& make_event,
                                                         const std::vector<uint32_t>& predicted_experts) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (top_k >= m_max_total_experts || m_map.count(Key{expert_id})) {
        return std::nullopt;
    }
    const auto quota = std::min({top_k, m_max_total_experts / 4, m_max_total_experts - top_k});
    const auto pending = static_cast<size_t>(std::count_if(m_list.begin(), m_list.end(), [](const Node& node) {
        return node.state.loading;
    }));
    if (pending >= quota) {
        return std::nullopt;
    }
    std::vector<bool> occupied(m_max_total_experts, false);
    for (const auto& node : m_list) {
        occupied[node.lru_expert_no] = true;
    }
    auto free_slot = std::find(occupied.begin(), occupied.end(), false);
    size_t slot = static_cast<size_t>(free_slot - occupied.begin());
    auto victim = m_list.end();
    cldnn::event::ptr last_use;
    if (free_slot == occupied.end()) {
        victim = std::find_if(m_list.begin(), m_list.end(), [&](const Node& node) {
            return node.state.status == SlotStatus::READY && !node.state.loading &&
                   std::find(predicted_experts.begin(), predicted_experts.end(), node.expert) == predicted_experts.end();
        });
        if (victim == m_list.end()) {
            return std::nullopt;
        }
        slot = victim->lru_expert_no;
        last_use = victim->state.last_use_event;
        if (LESLog::instance().enabled()) {
            const auto oldest = std::find_if(m_list.begin(), m_list.end(), [](const Node& node) {
                return node.state.status == SlotStatus::READY && !node.state.loading;
            });
            MOE_LES_LOG(m_log_layer, "event=speculative_victim incoming=" << expert_id << " victim=" << victim->expert
                        << " slot=" << slot << " predicted=" << les_ids(predicted_experts)
                        << " oldest_ready=" << oldest->expert);
        }
    }
    auto ready = make_event();
    if (victim == m_list.end()) {
        last_use = m_slot_last_use[slot];
    }
    OPENVINO_ASSERT(ready, "Lookahead load requires a completion event");
    if (victim != m_list.end()) {
        const auto expert = victim->expert;
        trace_event_unlocked('E', victim->expert);
        m_map.erase(Key{victim->expert});
        m_list.erase(victim);
        --m_total_experts;
        record_cache_event_unlocked('E', expert);
    }
    Load load{expert_id, slot, ++m_generation, ready, last_use};
    m_list.push_back(Node{expert_id, slot, {SlotStatus::PENDING_LOAD, ready, last_use, load.generation, true}});
    m_list.back().state.speculative = true;
    m_map.emplace(Key{expert_id}, std::prev(m_list.end()));
    m_filled_list[slot] = false;
    ++m_total_experts;
    m_list.back().state.reserved_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    record_cache_event_unlocked('R', expert_id, " origin=speculative");
    trace_event_unlocked('R', expert_id);
    return load;
}

std::optional<LRUCache::Load> LRUCache::get_pending_load(uint32_t expert_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_map.find(Key{expert_id});
    if (found == m_map.end() || !found->second->state.loading) {
        return std::nullopt;
    }
    const auto& node = *found->second;
    return Load{expert_id, node.lru_expert_no, node.state.generation, node.state.ready_event, node.state.last_use_event};
}

bool LRUCache::start_load(const Load& load) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_map.find(Key{load.expert_id});
    if (found == m_map.end() || found->second->state.generation != load.generation ||
        !found->second->state.loading || found->second->state.copying)
        return false;
    found->second->state.copying = true;
    if (found->second->state.copy_started_us == 0) {
        found->second->state.copy_started_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    record_cache_event_unlocked('B', load.expert_id);
    trace_event_unlocked('B', load.expert_id);
    return true;
}

bool LRUCache::fail_load(const Load& load) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_map.find(Key{load.expert_id});
    if (found == m_map.end() || found->second->state.generation != load.generation || !found->second->state.loading)
        return false;
    found->second->state.copying = false;
    record_cache_event_unlocked('F', load.expert_id);
    trace_event_unlocked('F', load.expert_id);
    return true;
}

bool LRUCache::complete_load(const Load& load) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_map.find(Key{load.expert_id});
    if (found == m_map.end() || found->second->state.generation != load.generation) {
        return false;
    }
    auto& node = *found->second;
    const bool was_loading = node.state.loading;
    if (node.state.loading) {
        const char kind = node.state.generation <= m_trace_generation ? 'I' : (node.state.speculative ? 'P' : 'D');
        trace_event_unlocked(kind, node.expert);
    }
    node.state.loading = false;
    node.state.copying = false;
    if (node.state.status != SlotStatus::PINNED) {
        node.state.status = SlotStatus::READY;
    }
    m_filled_list[node.lru_expert_no] = true;
    if (was_loading) {
        node.state.completed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    if (was_loading && LESLog::instance().enabled()) {
        std::ostringstream details;
        details << " reserved_us=" << node.state.reserved_us << " copy_started_us=" << node.state.copy_started_us
                << " completed_us=" << node.state.completed_us
                << " reserve_to_complete_us=" << (node.state.reserved_us ? node.state.completed_us - node.state.reserved_us : -1)
                << " queue_us=" << (node.state.copy_started_us ? node.state.copy_started_us - node.state.reserved_us : -1)
                << " copy_us=" << (node.state.copy_started_us ? node.state.completed_us - node.state.copy_started_us : -1);
        record_cache_event_unlocked('D', load.expert_id, details.str());
    }
    return true;
}

bool LRUCache::cancel_load(const Load& load) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_map.find(Key{load.expert_id});
    if (found == m_map.end() || found->second->state.generation != load.generation ||
        found->second->state.demanded || !found->second->state.loading) {
        return false;
    }
    m_filled_list[found->second->lru_expert_no] = false;
    trace_event_unlocked('X', found->second->expert);
    m_list.erase(found->second);
    m_map.erase(found);
    --m_total_experts;
    record_cache_event_unlocked('X', load.expert_id);
    return true;
}

void LRUCache::unpin_all(const cldnn::event::ptr& complete_event) {
    OPENVINO_ASSERT(complete_event, "Lookahead unpin requires a GPU consumer event");
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& node : m_list) {
        if (node.state.status == SlotStatus::PINNED) {
            node.state.last_use_event = complete_event;
            m_slot_last_use[node.lru_expert_no] = complete_event;
            node.state.status = node.state.loading ? SlotStatus::PENDING_LOAD : SlotStatus::READY;
            record_cache_event_unlocked('U', node.expert);
        }
    }
}

void LRUCache::evict_one_unlocked() {
    if (m_list.empty()) {
        return;
    }

    auto candidate = std::find_if(m_list.begin(), m_list.end(), [](const Node& node) {
        return node.state.status == SlotStatus::READY && !node.state.loading;
    });
    OPENVINO_ASSERT(candidate != m_list.end(), "Cannot evict a pinned or pending lookahead slot");
    auto& oldest = *candidate;

    const auto expert = oldest.expert;
    m_filled_list[oldest.lru_expert_no] = false;
    m_to_filled_lru_expert_no = oldest.lru_expert_no;
    Key key{oldest.expert};
    m_map.erase(key);
    m_list.erase(candidate);
    --m_total_experts;
    record_cache_event_unlocked('E', expert);
}

void LRUCache::evict_one() {
    std::lock_guard<std::mutex> lock(m_mutex);
    evict_one_unlocked();
}

std::pair<size_t, bool> LRUCache::get_lru_item(size_t expert) {
    std::lock_guard<std::mutex> lock(m_mutex);

    Key key{expert};
    auto it = m_map.find(key);
    if (it == m_map.end()) {
        size_t to_filled_no = 0;
        if (m_total_experts >= m_max_total_experts) {
            evict_one_unlocked();
            to_filled_no = m_to_filled_lru_expert_no;
        } else {
            std::vector<bool> occupied(m_max_total_experts, false);
            for (const auto& node : m_list) {
                occupied[node.lru_expert_no] = true;
            }
            to_filled_no = static_cast<size_t>(std::find(occupied.begin(), occupied.end(), false) - occupied.begin());
        }
        m_list.push_back(Node{expert, to_filled_no});
        auto new_it = std::prev(m_list.end());
        m_map[key] = new_it;
        ++m_total_experts;
        record_cache_event_unlocked('R', expert, " origin=synchronous");
        return {to_filled_no, false};
    }
    OPENVINO_ASSERT(!it->second->state.loading && it->second->state.status != SlotStatus::PINNED,
                    "Synchronous OTD cannot acquire an active lookahead slot");
    move_to_end(it->second);
    const bool is_hit = m_filled_list[it->second->lru_expert_no];
    return {it->second->lru_expert_no, is_hit};
}

}  // namespace ov::intel_gpu::ocl::moe
