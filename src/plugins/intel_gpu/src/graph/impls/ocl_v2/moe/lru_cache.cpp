// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lru_cache.hpp"
#include <algorithm>
#include <unordered_set>

namespace ov::intel_gpu::ocl::moe {

ExpertSlotReservations::ExpertSlotReservations(size_t capacity) : _slots(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("LES requires at least one resident slot");
    }
}

std::optional<ExpertSlotReservations::Ticket> ExpertSlotReservations::reserve(const std::vector<uint32_t>& experts, uint64_t epoch) {
    std::lock_guard<std::mutex> lock(_mutex);
    std::unordered_set<uint32_t> unique(experts.begin(), experts.end());
    if (unique.size() > _slots.size() || epoch < _epoch || (epoch != _epoch && !_active.empty())) {
        return std::nullopt;
    }
    auto planned = _slots;
    std::unordered_map<uint32_t, size_t> mapping;
    Record record{epoch, {}};
    Ticket result{_next_id + 1, epoch, {}, {}, {}};
    auto clock = _clock;
    for (auto expert : unique) {
        for (size_t index = 0; index < planned.size(); ++index) {
            auto& slot = planned[index];
            if (slot.expert == expert) {
                if (!slot.ready) {
                    return std::nullopt;
                }
                ++slot.pins;
                slot.age = ++clock;
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
            if (slot.pins == 0 && (candidate == planned.size() || !slot.expert || slot.age < planned[candidate].age)) {
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
    for (auto expert : experts) {
        result.slots.push_back(mapping.at(expert));
    }
    _active.emplace(result.id, std::move(record));
    _slots.swap(planned);
    _epoch = epoch;
    _clock = clock;
    ++_next_id;
    return result;
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
}

void LRUCache::move_to_end(std::list<Node>::iterator it) {
    if (std::next(it) == m_list.end()) {
        return;
    }
    m_list.splice(m_list.end(), m_list, it);
}

void LRUCache::evict_one_unlocked() {
    if (m_list.empty()) {
        return;
    }

    auto& oldest = m_list.front();

    m_filled_list[oldest.lru_expert_no] = false;
    m_to_filled_lru_expert_no = oldest.lru_expert_no;
    Key key{oldest.expert};
    m_map.erase(key);
    m_list.pop_front();
    --m_total_experts;
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
            to_filled_no = m_total_experts;
        }
        m_list.push_back(Node{expert, to_filled_no});
        auto new_it = std::prev(m_list.end());
        m_map[key] = new_it;
        ++m_total_experts;
        return {to_filled_no, false};
    }
    move_to_end(it->second);
    const bool is_hit = m_filled_list[it->second->lru_expert_no];
    return {it->second->lru_expert_no, is_hit};
}

}  // namespace ov::intel_gpu::ocl::moe
