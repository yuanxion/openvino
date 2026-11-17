// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "ocl_v2/moe/lru_cache.hpp"
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "ocl_v2/moe/moe_otd_runtime.hpp"

#include <thread>
#include <vector>
#include <set>
#include <atomic>
#include <cstdlib>
#include <cstring>

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::LRUCache;
using ov::intel_gpu::ocl::moe::ExpertSlotReservations;
using ov::intel_gpu::ocl::moe_otd::OtdPerfCounters;
using ov::intel_gpu::ocl::moe_otd::OtdLayerPerfScope;

namespace {
class LookaheadTestEvent : public cldnn::event {
    void wait_impl() override { ADD_FAILURE() << "LRU operations must not wait for GPU events"; }
    void set_impl() override {}
    bool is_set_impl() override { return false; }
};

cldnn::event::ptr make_lookahead_test_event() {
    return std::make_shared<LookaheadTestEvent>();
}
}

TEST(moe_les_timeline, snapshots_cover_resident_ready_loading_and_generations) {
    using namespace ov::intel_gpu::ocl::moe::timeline;
    const Selection sample("measured:2:1,128,255");
    Key key{1, 2, 1, 4, 0, "layer", "measured", "2"};
    testing::internal::CaptureStderr();
    for (const auto token : {1U, 128U, 255U}) {
        key.token = token;
        for (uint32_t layer = 0; layer < 40; ++layer) {
            key.layer = layer;
            auto selected = make_step(key, sample);
            ASSERT_TRUE(selected);
            Snapshot ready;
            ready.stamp_ns = 100;
            for (uint32_t expert = 0; expert < 8; ++expert)
                ready.entries.push_back({expert, expert, 0, false, true, {}});
            selected->observe("first_demand", ready, {0, 1, 2, 3, 4, 5, 6, 7}, 0, 0, true);
            EXPECT_EQ(selected->demands.size(), 8U);
        }
    }
    const auto grid = testing::internal::GetCapturedStderr();
    std::istringstream grid_lines(grid);
    std::string grid_line;
    size_t steps = 0, demand_results = 0;
    uint64_t previous_sequence = 0;
    while (std::getline(grid_lines, grid_line)) {
        steps += grid_line.find("\tstage=first_demand\t") != std::string::npos;
        demand_results += grid_line.find("\tstage=demand_result\t") != std::string::npos;
        const auto sequence = std::stoull(grid_line.substr(grid_line.find("\tseq=") + 5));
        EXPECT_GT(sequence, previous_sequence);
        previous_sequence = sequence;
    }
    EXPECT_EQ(steps, 120U);
    EXPECT_EQ(demand_results, 960U);
    testing::internal::CaptureStderr();
    key.token = 2;
    EXPECT_FALSE(make_step(key, sample));
    key.token = 1;
    key.iteration = "1";
    EXPECT_FALSE(make_step(key, sample));
    key.iteration = "2";
    key.request_phase = "warmup";
    EXPECT_FALSE(make_step(key, sample));
    EXPECT_FALSE(make_step(Key{}, sample));
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
    EXPECT_TRUE(make_step(Key{}, Selection(nullptr)));
    for (const auto* bad : {"", "measured", "measured:2", "other:2:1", "measured:NA:1",
                            "measured:-1:1", "measured:02:1", "measured:2:0", "measured:2:01",
                            "measured:2:1,", "measured:2:,1", "measured:2:1,1", "measured:2:1:2",
                            "measured:2: 1", "measured:2:18446744073709551616"})
        EXPECT_THROW(Selection{bad}, std::invalid_argument) << bad;
    LRUCache cache(3);
    cache.set_filled(cache.get_lru_item(1).first);
    const auto first = cache.snapshot();
    auto pending = cache.reserve_speculative(2, 1, make_lookahead_test_event);
    ASSERT_TRUE(pending);
    EXPECT_EQ(static_cast<bool>(pending->trace), ov::intel_gpu::ocl::moe::timeline::enabled());
    LRUCache::SpeculativeObservation observation;
    LRUCache::SpeculativeOutcome outcome;
    size_t extra_reservations = 0;
    const auto make_extra_event = [&] {
        ++extra_reservations;
        return make_lookahead_test_event();
    };
    EXPECT_FALSE(cache.reserve_speculative(2, 1, make_extra_event, &outcome, {}, &observation));
    EXPECT_EQ(outcome, LRUCache::SpeculativeOutcome::LOADING);
    ASSERT_TRUE(observation.entry);
    EXPECT_EQ(observation.cache, reinterpret_cast<uintptr_t>(&cache));
    EXPECT_EQ(observation.entry->expert, 2U);
    EXPECT_EQ(observation.entry->slot, pending->slot);
    EXPECT_EQ(observation.entry->generation, pending->generation);
    EXPECT_TRUE(observation.entry->loading);
    EXPECT_LE(first.stamp_ns, observation.stamp_ns);
    const auto loading_observation = observation;
    const auto second = cache.snapshot();
    EXPECT_LE(observation.stamp_ns, second.stamp_ns);
    ASSERT_EQ(first.entries.size(), 1U);
    ASSERT_EQ(second.entries.size(), 2U);
    EXPECT_LE(first.stamp_ns, second.stamp_ns);
    EXPECT_EQ(second.cache, reinterpret_cast<uintptr_t>(&cache));
    EXPECT_EQ(second.fields(), "\tcache=" + std::to_string(second.cache) +
        "\tresident=1,2\tready=1\tloading=2\tslots=1:0:0,2:1:" + std::to_string(pending->generation));
    EXPECT_TRUE(cache.cancel_load(*pending));
    if (pending->trace) {
        EXPECT_EQ(pending->trace->status, "canceled");
        EXPECT_GE(pending->trace->terminal_ns, second.stamp_ns);
    }
    auto replacement = cache.reserve_speculative(2, 1, make_lookahead_test_event);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->slot, pending->slot);
    EXPECT_GT(replacement->generation, pending->generation);
    EXPECT_FALSE(cache.complete_load(*pending));
    EXPECT_TRUE(cache.snapshot().entries.back().loading);
    EXPECT_TRUE(cache.complete_load(*replacement));
    EXPECT_TRUE(cache.snapshot().entries.back().ready);
    EXPECT_FALSE(cache.reserve_speculative(2, 1, make_extra_event, &outcome, {}, &observation));
    EXPECT_EQ(outcome, LRUCache::SpeculativeOutcome::RESIDENT);
    ASSERT_TRUE(observation.entry);
    EXPECT_EQ(observation.entry->slot, replacement->slot);
    EXPECT_EQ(observation.entry->generation, replacement->generation);
    EXPECT_TRUE(observation.entry->ready);
    EXPECT_FALSE(observation.entry->loading);
    EXPECT_EQ(loading_observation.entry->generation, pending->generation);
    EXPECT_EQ(extra_reservations, 0U);
    EXPECT_FALSE(cache.reserve_speculative(1, 1, make_extra_event, &outcome, {}, &observation));
    ASSERT_TRUE(observation.entry);
    EXPECT_EQ(observation.entry->generation, 0U);
    EXPECT_TRUE(observation.entry->ready);
    if (replacement->trace) {
        EXPECT_EQ(replacement->trace->status, "ready");
        EXPECT_GE(replacement->trace->terminal_ns, second.stamp_ns);
    }
}

TEST(moe_les_timeline, ready_zero_pending_new_load_promotion_and_terminal_once) {
    using namespace ov::intel_gpu::ocl::moe::timeline;
    auto load = std::make_shared<Load>();
    load->identity = "\tcache=1\tslot=1\tload_gen=1";
    load->origin = Key{1, 2, 3, 4, 5, "layer\t%=\n"}.fields();
    Step step({1, 2, 3, 4, 5, "layer\t%=\n"});
    EXPECT_NE(step.context.find("layer_id=layer%09%25%3D%0A"), std::string::npos);
    testing::internal::CaptureStderr();
    step.observe("first_demand", {100, {{1, 0, 0, false, true, {}}, {2, 1, 1, true, false, load}}, 1},
                 {1, 2, 3}, 0, 0, true);
    ASSERT_EQ(step.demands.size(), 3U);
    EXPECT_TRUE(step.demands[0]->reported);
    EXPECT_FALSE(step.demands[1]->reported);
    EXPECT_FALSE(step.demands[2]->reported);
    step.link(2, load);
    step.link(2, load);
    EXPECT_EQ(load->demands.size(), 1U);
    auto new_load = std::make_shared<Load>();
    new_load->identity = "\tcache=1\tslot=2\tload_gen=2";
    new_load->origin = step.context;
    step.link(3, new_load);
    load->terminal("ready", 120);
    load->flush();
    load->terminal("failed", 130);
    load->flush();
    new_load->terminal("failed", 140);
    new_load->flush();
    step.prediction({2, 3}, true);
    step.prediction({2, 4}, false);
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("demand_mark_ns=100\tweights_ready_ns=100\tlatency_ns=0"), std::string::npos);
    EXPECT_NE(output.find("demand_mark_ns=100\tweights_ready_ns=120\tlatency_ns=20"), std::string::npos);
    EXPECT_NE(output.find("status=failed\tdemand_mark_ns=100\tweights_ready_ns=NA\tlatency_ns=NA"), std::string::npos);
    EXPECT_NE(output.find("matched=1\tactual_count=2"), std::string::npos);
    EXPECT_EQ(load->status, "ready");
    EXPECT_TRUE(load->demands.empty());
    EXPECT_TRUE(step.demands[1]->reported);
    EXPECT_TRUE(step.demands[2]->reported);

    auto foreign = std::make_shared<Load>();
    foreign->identity = "\tcache=9\tslot=2\tload_gen=7";
    foreign->visible = false;
    const Key origin{1, 1, 256, 9, 5, "foreign", "measured", "1"};
    testing::internal::CaptureStderr();
    foreign->set_origin(origin, 8, "speculative");
    foreign->start(40);
    foreign->terminal("ready", 60);
    foreign->flush();
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
    Step selected({1, 2, 1, 10, 5, "selected", "measured", "2"});
    testing::internal::CaptureStderr();
    selected.observe("first_demand", {100, {{8, 2, 7, false, true, foreign}}, 9}, {8}, 0, 0, true);
    foreign->set_origin(selected.key, 8, "demand");
    auto late = std::make_shared<Demand>();
    late->context = selected.context;
    late->expert = 8;
    late->mark_ns = 50;
    foreign->watch(late);
    std::vector<std::thread> exposing;
    for (size_t worker = 0; worker < 4; ++worker)
        exposing.emplace_back([foreign] { foreign->expose(); });
    for (auto& worker : exposing)
        worker.join();
    const auto dependency = testing::internal::GetCapturedStderr();
    const auto start = dependency.find("\tstage=load_start");
    const auto result = dependency.find("\tstage=load_result");
    ASSERT_NE(start, std::string::npos);
    ASSERT_NE(result, std::string::npos);
    EXPECT_EQ(dependency.find("\tstage=load_start", start + 1), std::string::npos);
    EXPECT_EQ(dependency.find("\tstage=load_result", result + 1), std::string::npos);
    EXPECT_NE(dependency.find("\tobs_ns=40\t" + origin.fields()), std::string::npos);
    EXPECT_NE(dependency.find("\tobs_ns=60\t" + origin.fields()), std::string::npos);
    EXPECT_NE(dependency.find("demand_mark_ns=50\tweights_ready_ns=60\tlatency_ns=10"), std::string::npos);
    EXPECT_NE(dependency.find("demand_mark_ns=100\tweights_ready_ns=100\tlatency_ns=0"), std::string::npos);
    EXPECT_EQ(foreign->origin, origin.fields() + "\texpert=8\torigin=speculative");

    auto baseline = std::make_shared<Load>();
    baseline->identity = "\tcache=9\tslot=3\tload_gen=8";
    baseline->visible = false;
    baseline->set_origin(origin, 9, "speculative");
    baseline->start(70);
    baseline->terminal("ready", 80);
    auto candidate = std::make_shared<Load>();
    candidate->identity = "\tcache=9\tslot=4\tload_gen=9";
    candidate->visible = false;
    candidate->set_origin(origin, 10, "speculative");
    candidate->start(90);
    testing::internal::CaptureStderr();
    Snapshot snapshot{100, {{9, 3, 8, false, true, baseline}}, 9};
    selected.event("pre_prefetch", snapshot.fields(), snapshot.stamp_ns);
    selected.link(10, candidate);
    candidate->terminal("ready", 110);
    candidate->flush();
    const auto additional = testing::internal::GetCapturedStderr();
    EXPECT_NE(additional.find("\tstage=load_result" + baseline->identity), std::string::npos);
    EXPECT_NE(additional.find("\tstage=load_result" + candidate->identity), std::string::npos);
    EXPECT_TRUE(candidate->demands.empty());
    EXPECT_EQ(selected.demands.size(), 1U);
}

TEST(moe_les_timeline, cancellation_and_stale_results_never_become_ready) {
    using namespace ov::intel_gpu::ocl::moe::timeline;
    auto load = std::make_shared<Load>();
    load->origin = Key{}.fields();
    auto demand = std::make_shared<Demand>();
    demand->mark_ns = 100;
    load->watch(demand);
    testing::internal::CaptureStderr();
    load->terminal("canceled", 110);
    load->terminal("ready", 120);
    load->flush();
    auto stale = std::make_shared<Load>();
    stale->origin = Key{}.fields();
    stale->terminal("superseded", 130);
    auto late = std::make_shared<Demand>();
    late->mark_ns = 100;
    stale->watch(late);
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_EQ(load->status, "canceled");
    EXPECT_NE(output.find("status=canceled\tdemand_mark_ns=100\tweights_ready_ns=NA"), std::string::npos);
    EXPECT_NE(output.find("status=superseded\tdemand_mark_ns=100\tweights_ready_ns=NA"), std::string::npos);
    EXPECT_EQ(output.find("status=ready"), std::string::npos);
    EXPECT_TRUE(late->reported);
}

TEST(moe_les_timeline, concurrent_lines_have_unique_sequence_and_monotonic_worker_observations) {
    using namespace ov::intel_gpu::ocl::moe::timeline;
    testing::internal::CaptureStderr();
    std::vector<std::thread> workers;
    for (size_t worker = 0; worker < 4; ++worker) {
        workers.emplace_back([worker] {
            Step step({1, 1, 1, 1, static_cast<uint32_t>(worker), "layer\n\t%="});
            for (size_t observation = 0; observation < 16; ++observation)
                step.event("test", "\tworker=" + std::to_string(worker));
        });
    }
    for (auto& worker : workers)
        worker.join();
    const auto output = testing::internal::GetCapturedStderr();
    std::istringstream lines(output);
    std::string line;
    uint64_t previous_sequence = 0;
    std::array<uint64_t, 4> previous_stamp{};
    size_t count = 0;
    while (std::getline(lines, line)) {
        EXPECT_EQ(line.find("LES_TIMELINE\tv=1\tseq="), 0U);
        EXPECT_NE(line.find("\tlayer_id=layer%0A%09%25%3D\t"), std::string::npos);
        EXPECT_NE(line.find("\trequest_phase=unlabeled\titer=NA\t"), std::string::npos);
        EXPECT_NE(line.find("\tstage=test\t"), std::string::npos);
        const auto sequence = std::stoull(line.substr(line.find("\tseq=") + 5));
        const auto stamp = std::stoull(line.substr(line.find("\tobs_ns=") + 8));
        const auto worker = std::stoull(line.substr(line.find("\tworker=") + 8));
        EXPECT_GT(sequence, previous_sequence);
        ASSERT_LT(worker, previous_stamp.size());
        EXPECT_GE(stamp, previous_stamp[worker]);
        previous_sequence = sequence;
        previous_stamp[worker] = stamp;
        ++count;
    }
    EXPECT_EQ(count, 64U);
}

TEST(moe_lookahead_lru, partitions_and_pins_ready_inflight_missing_atomically) {
    LRUCache cache(8);
    const auto resident = cache.get_lru_item(1);
    cache.set_filled(resident.first);
    const auto speculative = cache.reserve_speculative(2, 3, make_lookahead_test_event);
    ASSERT_TRUE(speculative);
    EXPECT_FALSE(cache.reserve_speculative(2, 3, make_lookahead_test_event));
    LRUCache::Partition observation;
    ov::intel_gpu::ocl::moe::timeline::Snapshot snapshot;
    const std::vector<uint32_t> actual{1, 2, 3, 2};
    const auto demand = cache.partition_3way(actual, make_lookahead_test_event, false, &observation, &snapshot);
    ASSERT_TRUE(demand);
    EXPECT_EQ(observation.ready_pairs, demand->ready_pairs);
    EXPECT_EQ(observation.inflight_pairs, demand->inflight_pairs);
    EXPECT_EQ(observation.missing_pairs, demand->missing_pairs);
    ASSERT_EQ(snapshot.entries.size(), 2U);
    EXPECT_TRUE(snapshot.entries[0].ready);
    EXPECT_TRUE(snapshot.entries[1].loading);
    EXPECT_EQ(cache.snapshot().entries.size(), 3U);
    ov::intel_gpu::ocl::moe::timeline::Step step({1, 2, 3, 4, 5, "partition"});
    testing::internal::CaptureStderr();
    step.observe("first_demand", snapshot, actual, 0, 0, true);
    step.observe("before_wave_partition", snapshot, actual);
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("\tstage=first_demand\twave=0\tpass=0\tactual=1,2,3,2\tdemand_ready=1\tdemand_pending=2,2\tdemand_missing=3\t"),
              std::string::npos);
    EXPECT_NE(output.find("\tstage=before_wave_partition\twave=0\tpass=0\tactual=1,2,3,2\tdemand_ready=1\tdemand_pending=2,2\tdemand_missing=3\t"),
              std::string::npos);
    const auto captured_observation = "\tobs_ns=" + std::to_string(snapshot.stamp_ns) + '\t';
    const auto first_observation = output.find(captured_observation);
    ASSERT_NE(first_observation, std::string::npos);
    EXPECT_NE(output.find(captured_observation, first_observation + captured_observation.size()), std::string::npos);
    ASSERT_EQ(step.demands.size(), 3U);
    for (const auto& marked : step.demands)
        EXPECT_EQ(marked->mark_ns, snapshot.stamp_ns);
    EXPECT_EQ(demand->ready_pairs, (std::vector<size_t>{0}));
    EXPECT_EQ(demand->inflight_pairs, (std::vector<size_t>{1, 3}));
    EXPECT_EQ(demand->missing_pairs, (std::vector<size_t>{2}));
    ASSERT_EQ(demand->missing.size(), 1U);
    EXPECT_EQ(demand->missing.front().expert_id, 3U);
    EXPECT_EQ(demand->slots[1], speculative->slot);
    EXPECT_EQ(demand->slots[1], demand->slots[3]);
    ASSERT_EQ(demand->wait_events.size(), 2U);
    EXPECT_EQ(demand->wait_events.front(), speculative->ready_event);
    EXPECT_FALSE(cache.cancel_load(*speculative));
    cache.unpin_all(make_lookahead_test_event());
    EXPECT_FALSE(cache.cancel_load(*speculative));
    EXPECT_TRUE(cache.complete_load(*speculative));
    EXPECT_TRUE(cache.complete_load(demand->missing.front()));
    const auto hits = cache.partition_3way({1, 2, 3}, make_lookahead_test_event);
    ASSERT_TRUE(hits);
    EXPECT_EQ(hits->ready_pairs.size(), 3U);
    EXPECT_TRUE(hits->wait_events.empty());
    ASSERT_EQ(hits->ready_events.size(), 2U);
    EXPECT_EQ(hits->ready_events.front(), speculative->ready_event);
}

TEST(moe_lookahead_lru, preserves_war_event_on_slot_replacement) {
    LRUCache cache(1);
    auto first = cache.partition_3way({1}, make_lookahead_test_event);
    ASSERT_TRUE(first);
    ASSERT_TRUE(cache.complete_load(first->missing.front()));
    EXPECT_FALSE(cache.partition_3way({2}, make_lookahead_test_event));
    auto consumer = make_lookahead_test_event();
    cache.unpin_all(consumer);
    auto next = cache.partition_3way({2}, make_lookahead_test_event);
    ASSERT_TRUE(next);
    ASSERT_EQ(next->missing.size(), 1U);
    EXPECT_EQ(next->missing.front().last_use_event, consumer);
    EXPECT_EQ(next->missing.front().slot, first->missing.front().slot);
    EXPECT_FALSE(cache.complete_load(first->missing.front()));
}

TEST(moe_lookahead_lru, pending_waves_reuse_slots_only_after_consumer_is_recorded) {
    LRUCache cache(1);
    auto first = cache.partition_3way({1}, make_lookahead_test_event);
    ASSERT_TRUE(first);
    EXPECT_FALSE(cache.partition_3way({2}, make_lookahead_test_event, true));
    auto consumer = make_lookahead_test_event();
    cache.unpin_all(consumer);
    EXPECT_FALSE(cache.partition_3way({2}, make_lookahead_test_event));
    auto next = cache.partition_3way({2}, make_lookahead_test_event, true);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->slots, first->slots);
    EXPECT_EQ(next->missing.front().last_use_event, consumer);
    EXPECT_FALSE(cache.complete_load(first->missing.front()));
    auto pending = cache.get_pending_load(2);
    ASSERT_TRUE(pending);
    EXPECT_EQ(pending->generation, next->missing.front().generation);
    EXPECT_TRUE(cache.complete_load(next->missing.front()));
}

TEST(moe_lookahead_lru, full_top8_has_no_speculative_quota) {
    LRUCache cache(8);
    for (uint32_t expert = 0; expert < 8; ++expert) {
        LRUCache::SpeculativeOutcome outcome;
        ASSERT_TRUE(cache.reserve_speculative(expert, 8, make_lookahead_test_event, &outcome));
        EXPECT_EQ(outcome, LRUCache::SpeculativeOutcome::RESERVED);
    }
    LRUCache::SpeculativeOutcome outcome;
    EXPECT_FALSE(cache.reserve_speculative(7, 8, make_lookahead_test_event, &outcome));
    EXPECT_EQ(outcome, LRUCache::SpeculativeOutcome::LOADING);
    EXPECT_FALSE(cache.reserve_speculative(8, 8, make_lookahead_test_event, &outcome));
    EXPECT_EQ(outcome, LRUCache::SpeculativeOutcome::BLOCKED);
    auto demand = cache.partition_3way({0, 1, 2, 3, 4, 5, 6, 7}, make_lookahead_test_event);
    ASSERT_TRUE(demand);
    EXPECT_TRUE(demand->missing.empty());
    EXPECT_EQ(demand->inflight_pairs.size(), 8U);
}

TEST(moe_lookahead_lru, full_top8_counts_every_unique_candidate_with_small_capacity) {
    using namespace ov::intel_gpu::ocl::moe_otd;
    using Outcome = LRUCache::SpeculativeOutcome;
    LRUCache cache(4);
    OtdPerfCounters perf;
    cache.set_filled(cache.get_lru_item(6).first);
    cache.set_filled(cache.get_lru_item(7).first);
    ASSERT_TRUE(cache.reserve_speculative(0, 8, make_lookahead_test_event));
    size_t attempts = 0;
    size_t events_created = 0;
    visit_speculative_candidates({0, 1, 2, 3, 4, 5, 6, 7, 0, 7}, &perf,
        [&](uint32_t expert, const std::vector<uint32_t>& protected_experts) {
            ++attempts;
            Outcome outcome;
            cache.reserve_speculative(expert, 8, [&] {
                ++events_created;
                return make_lookahead_test_event();
            }, &outcome, protected_experts);
            return outcome;
        });
    EXPECT_EQ(attempts, 8U);
    EXPECT_EQ(events_created, 1U);
    EXPECT_EQ(perf.les_candidate_raw_ids.load(), 10U);
    EXPECT_EQ(perf.les_candidate_duplicates.load(), 2U);
    EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(Outcome::RESERVED)].load(), 1U);
    EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(Outcome::RESIDENT)].load(), 2U);
    EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(Outcome::LOADING)].load(), 1U);
    EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(Outcome::BLOCKED)].load(), 4U);
    EXPECT_TRUE(cache.get_lru_item(6).second);
    EXPECT_TRUE(cache.get_lru_item(7).second);
    EXPECT_TRUE(perf.layers.empty());
    EXPECT_EQ(perf.gpu_hits.load() + perf.gpu_misses.load(), 0U);
}

TEST(moe_lookahead_lru, completed_predictions_do_not_cascade_evict_each_other) {
    using namespace ov::intel_gpu::ocl::moe_otd;
    using Outcome = LRUCache::SpeculativeOutcome;
    LRUCache cache(3);
    OtdPerfCounters perf;
    visit_speculative_candidates({0, 1, 2, 3, 4, 5, 6, 7}, &perf,
        [&](uint32_t expert, const std::vector<uint32_t>& protected_experts) {
            Outcome outcome;
            if (auto load = cache.reserve_speculative(expert, 8, make_lookahead_test_event, &outcome, protected_experts)) {
                EXPECT_TRUE(cache.complete_load(*load));
            }
            return outcome;
        });
    EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(Outcome::RESERVED)].load(), 3U);
    EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(Outcome::BLOCKED)].load(), 5U);
    for (uint32_t expert = 0; expert < 3; ++expert) {
        EXPECT_TRUE(cache.get_lru_item(expert).second);
    }
}

TEST(moe_lookahead_lru, mismatched_pending_predictions_allow_retry_without_inflating_hits) {
    LRUCache cache(4);
    std::vector<LRUCache::Load> predicted;
    for (uint32_t expert = 0; expert < 4; ++expert) {
        auto load = cache.reserve_speculative(expert, 8, make_lookahead_test_event);
        ASSERT_TRUE(load);
        predicted.push_back(*load);
    }
    const std::vector<uint32_t> actual{0, 4, 5, 6};
    LRUCache::Partition observation;
    ov::intel_gpu::ocl::moe::timeline::Snapshot snapshot;
    EXPECT_FALSE(cache.partition_3way(actual, make_lookahead_test_event, false, &observation, &snapshot));
    ASSERT_EQ(snapshot.entries.size(), 4U);
    EXPECT_EQ(snapshot.cache, reinterpret_cast<uintptr_t>(&cache));
    const auto captured_stamp = snapshot.stamp_ns;
    EXPECT_GT(captured_stamp, 0U);
    for (const auto& entry : snapshot.entries) {
        EXPECT_TRUE(entry.loading);
        EXPECT_FALSE(entry.ready);
    }
    EXPECT_TRUE(observation.ready_pairs.empty());
    EXPECT_EQ(observation.inflight_pairs, (std::vector<size_t>{0}));
    EXPECT_EQ(observation.missing_pairs, (std::vector<size_t>{1, 2, 3}));
    for (const auto& load : predicted) {
        ASSERT_TRUE(cache.complete_load(load));
    }
    const auto retry = cache.partition_3way(actual, make_lookahead_test_event);
    ASSERT_TRUE(retry);
    EXPECT_EQ(snapshot.stamp_ns, captured_stamp);
    EXPECT_TRUE(snapshot.entries.front().loading);
    EXPECT_FALSE(snapshot.entries.front().ready);
    EXPECT_EQ(retry->ready_pairs, (std::vector<size_t>{0}));
    EXPECT_EQ(retry->missing_pairs.size(), 3U);
    EXPECT_EQ(retry->slots[0], predicted.front().slot);
    OtdPerfCounters perf;
    {
        OtdLayerPerfScope scope("layer0", 1, &perf);
        for (auto pair : observation.ready_pairs) {
            OtdLayerPerfScope::record(actual[pair], true);
        }
        for (auto pair : observation.inflight_pairs) {
            OtdLayerPerfScope::record(actual[pair], false, true);
        }
        for (auto pair : observation.missing_pairs) {
            OtdLayerPerfScope::record(actual[pair], false);
        }
    }
    const auto& counts = perf.layers.at("layer0").demand[1];
    EXPECT_EQ(counts.hits, 0U);
    EXPECT_EQ(counts.misses, 4U);
    EXPECT_EQ(counts.pending, 1U);
}

TEST(moe_lookahead_lru, full_top8_reports_pinned_slots_as_blocked_without_overwrite) {
    using namespace ov::intel_gpu::ocl::moe_otd;
    using Outcome = LRUCache::SpeculativeOutcome;
    LRUCache cache(2);
    OtdPerfCounters perf;
    auto demand = cache.partition_3way({90, 91}, make_lookahead_test_event);
    ASSERT_TRUE(demand);
    for (const auto& load : demand->missing) {
        ASSERT_TRUE(cache.complete_load(load));
    }
    visit_speculative_candidates({0, 1, 2, 3, 4, 5, 6, 7}, &perf,
        [&](uint32_t expert, const std::vector<uint32_t>& protected_experts) {
            Outcome outcome;
            EXPECT_FALSE(cache.reserve_speculative(expert, 8, make_lookahead_test_event, &outcome, protected_experts));
            return outcome;
        });
    EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(Outcome::BLOCKED)].load(), 8U);
    const auto still_pinned = cache.partition_3way({90, 91}, make_lookahead_test_event);
    ASSERT_TRUE(still_pinned);
    EXPECT_EQ(still_pinned->ready_pairs.size(), 2U);
    EXPECT_EQ(still_pinned->slots, demand->slots);
}

TEST(moe_lookahead_lru, expired_and_canceled_predictions_account_for_remaining_unique_candidates) {
    using namespace ov::intel_gpu::ocl::moe_otd;
    using Outcome = LRUCache::SpeculativeOutcome;
    for (auto terminal : {Outcome::EXPIRED, Outcome::CANCELED}) {
        LRUCache cache(8);
        OtdPerfCounters perf;
        size_t attempts = 0;
        visit_speculative_candidates({0, 1, 2, 3, 4, 5, 6, 7, 1}, &perf,
            [&](uint32_t expert, const std::vector<uint32_t>& protected_experts) {
                if (++attempts > 2) {
                    return terminal;
                }
                Outcome outcome;
                cache.reserve_speculative(expert, 8, make_lookahead_test_event, &outcome, protected_experts);
                return outcome;
            });
        EXPECT_EQ(attempts, 8U);
        EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(Outcome::RESERVED)].load(), 2U);
        EXPECT_EQ(perf.les_candidate_outcomes[static_cast<size_t>(terminal)].load(), 6U);
        EXPECT_EQ(cache.size(), 2U);
    }
}

TEST(moe_les_statistics, prefetch_summary_distinguishes_candidates_from_loads_and_unread_tasks) {
    OtdPerfCounters perf;
    using Outcome = LRUCache::SpeculativeOutcome;
    ov::intel_gpu::ocl::moe_otd::visit_speculative_candidates({0, 1, 2, 3, 4, 5, 5}, &perf,
        [](uint32_t expert, const std::vector<uint32_t>&) { return static_cast<Outcome>(expert); });
    perf.les_loads_completed = 1;
    perf.les_loads_canceled = 2;
    perf.les_loads_failed = 3;
    perf.les_loads_superseded = 4;
    perf.les_tasks_expired_unread = 5;
    perf.les_tasks_canceled_unread = 6;
    perf.les_tasks_unbound = 7;
    perf.les_legacy_blocked_candidates = 8;
    testing::internal::CaptureStderr();
    perf.dump();
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("[LES_PREFETCH] layer=ALL phase=decode path=worker policy=protect_prediction_set"
                          " raw_ids=7 duplicates=1 candidates=6 reserved=1 resident=1 loading=1 blocked=1 expired=1 canceled=1"
                          " loads_completed=1 loads_canceled=2 loads_failed=3 loads_superseded=4"
                          " tasks_expired_unread=5 tasks_canceled_unread=6 tasks_unbound=7"), std::string::npos);
    EXPECT_NE(output.find("[LES_PREFETCH] layer=ALL path=legacy policy=atomic_full_set blocked_candidates=8"), std::string::npos);
}

TEST(moe_les_statistics, interleaved_phases_keep_local_denominators_and_deduplicate) {
    OtdPerfCounters perf;
    {
        OtdLayerPerfScope prefill("layer0", 1024, &perf);
        OtdLayerPerfScope::record(1, true);
        {
            OtdLayerPerfScope decode("layer0", 1, &perf);
            OtdLayerPerfScope::record(1, false, true);
            OtdLayerPerfScope::record(1, true);
            OtdLayerPerfScope::record(2, false);
            OtdLayerPerfScope::record(3, true);
        }
        OtdLayerPerfScope::record(2, true);
        OtdLayerPerfScope::record(3, false);
    }
    {
        OtdLayerPerfScope decode("layer0", 1, &perf);
        OtdLayerPerfScope::record(1, true);
    }
    const auto& phases = perf.layers.at("layer0").demand;
    EXPECT_EQ(phases[0].hits, 2U);
    EXPECT_EQ(phases[0].misses, 1U);
    EXPECT_EQ(phases[0].pending, 0U);
    EXPECT_DOUBLE_EQ(100.0 * phases[0].hits / (phases[0].hits + phases[0].misses), 200.0 / 3);
    EXPECT_EQ(phases[1].hits, 2U);
    EXPECT_EQ(phases[1].misses, 2U);
    EXPECT_EQ(phases[1].pending, 1U);
    EXPECT_DOUBLE_EQ(100.0 * phases[1].hits / (phases[1].hits + phases[1].misses), 50.0);
    EXPECT_EQ(phases[2].hits + phases[2].misses, 0U);
}

TEST(moe_les_statistics, environment_gates_require_exact_one) {
    using namespace ov::intel_gpu::ocl::moe_otd;
    const auto enabled = [](const char* name) {
        const auto* value = std::getenv(name);
        return value && std::strcmp(value, "1") == 0;
    };
    EXPECT_EQ(layer_perf_enabled(), enabled("MOE_OTD_LAYER_PERF_LOG"));
    EXPECT_EQ(les_prediction_compare_enabled(), enabled("OV_GPU_LES_PREDICT_COMPARE"));
    EXPECT_EQ(get_perf_counters() != nullptr,
              enabled("MOE_OTD_LAYER_PERF_LOG") || enabled("OV_GPU_LES_PREDICT_COMPARE") || enabled("MOE_OTD_PERF_LOG"));
}

TEST(moe_les_statistics, speculative_probes_and_worker_threads_are_not_demand) {
    OtdPerfCounters perf;
    LRUCache cache(8);
    {
        OtdLayerPerfScope scope("layer0", 1, &perf);
        ASSERT_TRUE(cache.reserve_speculative(0, 8, make_lookahead_test_event));
        std::thread worker([] { OtdLayerPerfScope::record(1, true); });
        worker.join();
        OtdLayerPerfScope disabled("disabled", 1, nullptr);
        OtdLayerPerfScope::record(2, false);
    }
    EXPECT_TRUE(perf.layers.empty());
    {
        OtdLayerPerfScope unknown("layer0", 0, &perf);
        OtdLayerPerfScope::record(1, false);
    }
    EXPECT_EQ(perf.layers.at("layer0").demand[2].misses, 1U);
}

TEST(moe_les_statistics, ready_pending_and_missing_use_first_demand_observation) {
    OtdPerfCounters perf;
    LRUCache cache(8);
    cache.set_filled(cache.get_lru_item(1).first);
    ASSERT_TRUE(cache.reserve_speculative(2, 8, make_lookahead_test_event));
    const std::vector<uint32_t> experts{1, 2, 3, 2};
    const auto partition = cache.partition_3way(experts, make_lookahead_test_event);
    ASSERT_TRUE(partition);
    OtdLayerPerfScope scope("layer0", 1, &perf);
    for (auto pair : partition->ready_pairs) {
        OtdLayerPerfScope::record(experts[pair], true);
    }
    for (auto pair : partition->inflight_pairs) {
        OtdLayerPerfScope::record(experts[pair], false, true);
    }
    for (auto pair : partition->missing_pairs) {
        OtdLayerPerfScope::record(experts[pair], false);
    }
    const auto& counts = perf.layers.at("layer0").demand[1];
    EXPECT_EQ(counts.hits, 1U);
    EXPECT_EQ(counts.misses, 2U);
    EXPECT_EQ(counts.pending, 1U);
}

TEST(moe_les_statistics, recall_is_set_intersection_not_rank_or_duplicate_credit) {
    using Status = OtdPerfCounters::PredictionStatus;
    OtdPerfCounters::RecallCounts complete, partial, zero;
    complete.add({3, 2, 1, 1}, {1, 2, 3, 3}, Status::MATCHED);
    EXPECT_EQ(complete.matched, 3U);
    EXPECT_EQ(complete.actual, 3U);
    partial.add({1, 1, 8, 9}, {1, 2, 3, 3}, Status::MATCHED);
    EXPECT_EQ(partial.matched, 1U);
    EXPECT_EQ(partial.actual, 3U);
    zero.add({7, 8}, {1, 2, 3}, Status::MATCHED);
    EXPECT_EQ(zero.matched, 0U);
    EXPECT_EQ(zero.actual, 3U);
    EXPECT_EQ(zero.compared, 1U);
}

TEST(moe_les_statistics, recall_rejects_missing_stale_and_late_generations) {
    using Status = OtdPerfCounters::PredictionStatus;
    using Recall = OtdPerfCounters::RecallCounts;
    ov::intel_gpu::ocl::moe_otd::PredictionReadback readback;
    std::vector<uint32_t> prediction;
    EXPECT_FALSE(readback.snapshot(prediction));
    EXPECT_EQ(Recall::classify(false, 0, 2, false), Status::MISSING);
    EXPECT_EQ(Recall::classify(true, 1, 2, false), Status::STALE);
    EXPECT_EQ(Recall::classify(true, 2, 2, false), Status::LATE);
    readback.publish({0, 1, 2, 3, 4, 5, 6, 7});
    ASSERT_TRUE(readback.snapshot(prediction));
    EXPECT_EQ(prediction.size(), 8U);
    EXPECT_EQ(Recall::classify(true, 2, 2, true), Status::MATCHED);
    EXPECT_EQ(Recall::classify(true, 2, 3, true), Status::STALE);
    Recall counts;
    counts.add(prediction, {0, 1}, Status::MISSING);
    counts.add(prediction, {0, 1}, Status::STALE);
    counts.add(prediction, {0, 1}, Status::LATE);
    EXPECT_EQ(counts.actual, 0U);
    EXPECT_EQ(counts.matched, 0U);
    EXPECT_EQ(counts.compared, 0U);
    EXPECT_EQ(counts.missing, 1U);
    EXPECT_EQ(counts.stale, 1U);
    EXPECT_EQ(counts.late, 1U);
}

TEST(moe_les_statistics, phase_and_recall_logs_have_independent_all_summaries) {
    using namespace ov::intel_gpu::ocl::moe_otd;
    if (!layer_perf_enabled() || !les_prediction_compare_enabled()) {
        GTEST_SKIP() << "Requires MOE_OTD_LAYER_PERF_LOG=1 OV_GPU_LES_PREDICT_COMPARE=1";
    }
    OtdPerfCounters perf;
    perf.record_demand("layer0", 1024, 3, 1, 0);
    perf.record_demand("layer0", 1, 1, 1, 2);
    perf.record_demand("layer1", 1, 2, 0, 0);
    perf.record_demand("layer1", 1024, 1, 1, 0);
    perf.record_recall("layer0", {1, 2}, {2, 3}, OtdPerfCounters::PredictionStatus::MATCHED);
    perf.record_recall("layer1", {}, {2, 3}, OtdPerfCounters::PredictionStatus::LATE);
    testing::internal::CaptureStderr();
    perf.dump();
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("[OTD_CACHE] layer=layer0 phase=prefill hits=3 misses=1 pending=0 hit_pct=75 miss_pct=25"),
              std::string::npos);
    EXPECT_NE(output.find("[OTD_CACHE] layer=layer0 phase=decode hits=1 misses=3 pending=2 hit_pct=25 miss_pct=75"),
              std::string::npos);
    EXPECT_NE(output.find("[OTD_CACHE] layer=ALL phase=prefill hits=4 misses=2 pending=0 hit_pct=66.6667 miss_pct=33.3333"),
              std::string::npos);
    EXPECT_NE(output.find("[OTD_CACHE] layer=ALL phase=decode hits=3 misses=3 pending=2 hit_pct=50 miss_pct=50"),
              std::string::npos);
    EXPECT_NE(output.find("[LES_RECALL] layer=ALL phase=decode matched=1 actual=2 recall_pct=50 compared=1 missing=0 stale=0 late=1"),
              std::string::npos);
    EXPECT_EQ(output.find("phase=all"), std::string::npos);
}

TEST(moe_lookahead_lru, cancelled_replacement_retains_physical_slot_war) {
    LRUCache cache(4);
    auto initial = cache.partition_3way({1, 2, 3, 4}, make_lookahead_test_event);
    ASSERT_TRUE(initial);
    for (const auto& load : initial->missing) {
        ASSERT_TRUE(cache.complete_load(load));
    }
    auto consumer = make_lookahead_test_event();
    cache.unpin_all(consumer);
    auto speculative = cache.reserve_speculative(5, 1, make_lookahead_test_event);
    ASSERT_TRUE(speculative);
    ASSERT_TRUE(cache.cancel_load(*speculative));
    auto replacement = cache.partition_3way({6}, make_lookahead_test_event);
    ASSERT_TRUE(replacement);
    ASSERT_EQ(replacement->missing.size(), 1U);
    EXPECT_EQ(replacement->missing.front().slot, speculative->slot);
    EXPECT_EQ(replacement->missing.front().last_use_event, consumer);
}

TEST(moe_lookahead_lru, cancellation_and_failed_reservation_do_not_alias_slots) {
    LRUCache cache(8);
    auto cancelled = cache.reserve_speculative(1, 2, make_lookahead_test_event);
    ASSERT_TRUE(cancelled);
    ASSERT_TRUE(cache.reserve_speculative(2, 2, make_lookahead_test_event));
    ASSERT_TRUE(cache.cancel_load(*cancelled));
    EXPECT_FALSE(cache.complete_load(*cancelled));
    EXPECT_FALSE(cache.partition_3way({0, 1, 2, 3, 4, 5, 6, 7, 8}, make_lookahead_test_event));
    EXPECT_EQ(cache.size(), 1U);
    auto demand = cache.partition_3way({3, 2}, make_lookahead_test_event);
    ASSERT_TRUE(demand);
    EXPECT_NE(demand->slots[0], demand->slots[1]);
}

TEST(moe_les_reservations, deduplicates_and_rejects_capacity_without_mutation) {
    ExpertSlotReservations cache(2);
    EXPECT_FALSE(cache.reserve({1, 2, 3}, 1));
    auto lease = cache.reserve({1, 1, 2}, 1);
    ASSERT_TRUE(lease);
    EXPECT_EQ(lease->slots[0], lease->slots[1]);
    EXPECT_EQ(lease->missing_experts.size(), 2U);
    EXPECT_TRUE(cache.complete(*lease));
    EXPECT_TRUE(cache.release(*lease));
    auto hits = cache.reserve({1, 2}, 2);
    ASSERT_TRUE(hits);
    EXPECT_TRUE(hits->missing_experts.empty());
}

TEST(moe_les_reservations, full_top8_uses_safe_atomic_reservation_without_legacy_budget) {
    ExpertSlotReservations reservations(8);
    const std::vector<uint32_t> prediction{0, 1, 2, 3, 4, 5, 6, 7, 7};
    const auto full = reservations.reserve(prediction, 1, true);
    ASSERT_TRUE(full);
    EXPECT_EQ(full->missing_experts.size(), 8U);
    EXPECT_EQ(full->slots[7], full->slots[8]);
    EXPECT_FALSE(reservations.reserve({8}, 1, true));
    ASSERT_TRUE(reservations.rollback(*full));
    ExpertSlotReservations small(4);
    EXPECT_FALSE(small.reserve(prediction, 1, true));
    const auto demand = small.reserve({90, 91, 92, 93}, 1);
    ASSERT_TRUE(demand);
    EXPECT_EQ(demand->missing_experts.size(), 4U);
}

TEST(moe_les_reservations, protects_hits_before_assigning_misses) {
    ExpertSlotReservations cache(2);
    auto initial = cache.reserve({1, 2}, 1).value();
    cache.complete(initial);
    cache.release(initial);
    auto lease = cache.reserve({3, 1}, 1).value();
    EXPECT_EQ(lease.slots[1], initial.slots[0]);
    EXPECT_EQ(lease.missing_experts, std::vector<uint32_t>{3});
    EXPECT_NE(lease.slots[0], lease.slots[1]);
}

TEST(moe_les_reservations, pinned_consumers_prevent_eviction) {
    ExpertSlotReservations cache(1);
    auto first = cache.reserve({1}, 1).value();
    cache.complete(first);
    auto second = cache.reserve({1}, 1).value();
    EXPECT_FALSE(cache.reserve({2}, 1));
    cache.release(first);
    EXPECT_FALSE(cache.reserve({2}, 1));
    cache.release(second);
    EXPECT_TRUE(cache.reserve({2}, 1));
}

TEST(moe_les_reservations, pending_is_not_ready_and_abort_permits_retry) {
    ExpertSlotReservations cache(1);
    auto first = cache.reserve({1}, 1).value();
    EXPECT_FALSE(cache.reserve({1}, 1));
    EXPECT_FALSE(cache.reserve({2}, 2));
    cache.release(first);
    auto next = cache.reserve({2}, 2).value();
    EXPECT_FALSE(cache.complete(first));
    EXPECT_FALSE(cache.release(first));
    EXPECT_FALSE(cache.reserve({2}, 1));
    EXPECT_TRUE(cache.complete(next));
    EXPECT_TRUE(cache.release(next));
}

TEST(moe_les_reservations, rejects_zero_capacity) {
    EXPECT_ANY_THROW(ExpertSlotReservations(0));
}

TEST(moe_les_reservations, partitions_all_top8_masks_in_canonical_pair_order) {
    const std::vector<uint32_t> experts{7, 0, 5, 2, 6, 1, 4, 3, 3, 4, 1, 6, 2, 5, 0, 7};
    for (uint32_t mask = 0; mask < 256; ++mask) {
        SCOPED_TRACE(mask);
        ExpertSlotReservations cache(8);
        std::vector<uint32_t> warm;
        for (uint32_t expert = 0; expert < 8; ++expert) {
            if (mask & (1U << expert)) {
                warm.push_back(expert);
            }
        }
        auto initial = cache.reserve(warm, 1);
        ASSERT_TRUE(initial);
        ASSERT_TRUE(cache.complete(*initial));
        ASSERT_TRUE(cache.release(*initial));
        auto demand = cache.reserve(experts, 2);
        ASSERT_TRUE(demand);
        std::vector<size_t> ready;
        std::vector<size_t> missing;
        for (size_t pair = 0; pair < experts.size(); ++pair) {
            (mask & (1U << experts[pair]) ? ready : missing).push_back(pair);
            EXPECT_EQ(demand->slots[pair], demand->slots[experts.size() - 1 - pair]);
        }
        EXPECT_EQ(demand->ready_pairs, ready);
        EXPECT_EQ(demand->missing_pairs, missing);
        ASSERT_TRUE(cache.complete(*demand));
        EXPECT_EQ(demand->ready_pairs, ready);
        EXPECT_EQ(demand->missing_pairs, missing);
        ASSERT_TRUE(cache.release(*demand));
        auto hits = cache.reserve(experts, 3);
        ASSERT_TRUE(hits);
        EXPECT_EQ(hits->ready_pairs.size(), experts.size());
        EXPECT_TRUE(hits->missing_pairs.empty());
        EXPECT_TRUE(hits->missing_experts.empty());
        ASSERT_TRUE(cache.release(*hits));
    }
}

// --------------------------------------------------
// Basic construction and initial state
// --------------------------------------------------

TEST(moe_lru_cache, initial_state) {
    LRUCache cache(4);

    ASSERT_EQ(cache.size(), 0U);
    ASSERT_FALSE(cache.is_initialized());
}

// --------------------------------------------------
// get_lru_item: insert (miss) and hit
// --------------------------------------------------

TEST(moe_lru_cache, single_insert_is_miss) {
    LRUCache cache(4);

    auto [slot, hit] = cache.get_lru_item(/*expert=*/0);
    ASSERT_FALSE(hit);        // first access is a miss
    ASSERT_EQ(slot, 0U);     // first slot assigned
    ASSERT_EQ(cache.size(), 1U);
}

TEST(moe_lru_cache, second_access_without_fill_is_miss) {
    // get_lru_item returns is_hit based on m_filled_list, not just presence
    LRUCache cache(4);

    cache.get_lru_item(0);  // insert, slot 0
    auto [slot, hit] = cache.get_lru_item(0);  // access again

    ASSERT_EQ(slot, 0U);
    ASSERT_FALSE(hit);  // filled_list[0] is still false
}

TEST(moe_lru_cache, access_after_set_filled_is_hit) {
    LRUCache cache(4);

    auto [slot, miss] = cache.get_lru_item(0);
    cache.set_filled(slot);

    auto [slot2, hit] = cache.get_lru_item(0);
    ASSERT_EQ(slot2, slot);
    ASSERT_TRUE(hit);
}

// --------------------------------------------------
// Sequential slot allocation
// --------------------------------------------------

TEST(moe_lru_cache, slots_assigned_sequentially) {
    const size_t cap = 4;
    LRUCache cache(cap);

    for (size_t i = 0; i < cap; ++i) {
        auto [slot, hit] = cache.get_lru_item(/*expert=*/i);
        ASSERT_EQ(slot, i);
        ASSERT_FALSE(hit);
    }
    ASSERT_EQ(cache.size(), cap);
}

// --------------------------------------------------
// Eviction: oldest entry evicted when cache is full
// --------------------------------------------------

TEST(moe_lru_cache, eviction_when_full) {
    LRUCache cache(3);

    // Fill cache: experts 0, 1, 2
    cache.get_lru_item(0);  // slot 0
    cache.get_lru_item(1);  // slot 1
    cache.get_lru_item(2);  // slot 2
    ASSERT_EQ(cache.size(), 3U);

    // Insert expert 3 -> evicts expert 0 (oldest), reuses slot 0
    auto [slot, hit] = cache.get_lru_item(3);
    ASSERT_FALSE(hit);
    ASSERT_EQ(slot, 0U);   // expert 0's slot is recycled
    ASSERT_EQ(cache.size(), 3U);  // size stays at capacity
}

TEST(moe_lru_cache, evicted_entry_becomes_miss) {
    LRUCache cache(2);

    // Fill: expert 0 (slot 0), expert 1 (slot 1)
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // Insert expert 2 -> evicts expert 0
    cache.get_lru_item(2);

    // Access expert 0 again -> should be a miss (it was evicted)
    auto [slot, hit] = cache.get_lru_item(0);
    ASSERT_FALSE(hit);
}

// --------------------------------------------------
// LRU ordering: recently accessed items survive eviction
// --------------------------------------------------

TEST(moe_lru_cache, lru_order_refresh_on_access) {
    LRUCache cache(3);

    // Insert experts 0, 1, 2 in order
    cache.get_lru_item(0);  // slot 0  (LRU order: 0)
    cache.get_lru_item(1);  // slot 1  (LRU order: 0, 1)
    cache.get_lru_item(2);  // slot 2  (LRU order: 0, 1, 2)

    // Access expert 0 -> moves to most recent
    // (LRU order: 1, 2, 0)
    cache.get_lru_item(0);

    // Insert expert 3 -> should evict expert 1 (now the oldest)
    auto [slot, hit] = cache.get_lru_item(3);
    ASSERT_FALSE(hit);
    ASSERT_EQ(slot, 1U);  // reuses expert 1's slot
}

TEST(moe_lru_cache, double_refresh_changes_eviction_order) {
    LRUCache cache(3);

    // Insert 0, 1, 2
    cache.get_lru_item(0);  // slot 0
    cache.get_lru_item(1);  // slot 1
    cache.get_lru_item(2);  // slot 2

    // Refresh 0, then refresh 1  -> LRU order: 2, 0, 1
    cache.get_lru_item(0);
    cache.get_lru_item(1);

    // Insert 3 -> evicts expert 2 (oldest)
    auto [slot3, hit3] = cache.get_lru_item(3);
    ASSERT_FALSE(hit3);
    ASSERT_EQ(slot3, 2U);  // reuses expert 2's slot

    // Insert 4 -> evicts expert 0 (now oldest)
    auto [slot4, hit4] = cache.get_lru_item(4);
    ASSERT_FALSE(hit4);
    ASSERT_EQ(slot4, 0U);  // reuses expert 0's slot
}

// --------------------------------------------------
// Multi-layer: (layer, expert) pairs are independent
// --------------------------------------------------

// Since LRU cache is per-layer (each MoE layer has its own cache instance),
// the layer dimension has been removed from the cache key. This test verifies
// that distinct expert ids get independent slots.
TEST(moe_lru_cache, distinct_experts_get_independent_slots) {
    LRUCache cache(4);

    auto [s0, h0] = cache.get_lru_item(/*expert=*/0);
    auto [s1, h1] = cache.get_lru_item(/*expert=*/1);

    ASSERT_FALSE(h0);
    ASSERT_FALSE(h1);
    ASSERT_NE(s0, s1);  // different slots for different experts
    ASSERT_EQ(cache.size(), 2U);
}

TEST(moe_lru_cache, same_expert_gets_same_slot) {
    LRUCache cache(4);

    cache.get_lru_item(5);
    cache.set_filled(0);

    auto [s0, h0] = cache.get_lru_item(5);

    ASSERT_TRUE(h0);
    ASSERT_EQ(s0, 0U);
}

// --------------------------------------------------
// set_filled / filled tracking
// --------------------------------------------------

TEST(moe_lru_cache, filled_cleared_on_eviction) {
    LRUCache cache(2);

    // Insert expert 0, mark filled
    auto [s0, miss0] = cache.get_lru_item(0);
    cache.set_filled(s0);

    // Insert expert 1
    cache.get_lru_item(1);

    // Insert expert 2 -> evicts expert 0, reuses slot s0
    auto [s2, h2] = cache.get_lru_item(2);
    ASSERT_EQ(s2, s0);         // recycled slot
    ASSERT_FALSE(h2);          // filled was cleared during eviction

    // Even after re-inserting to the same slot, accessing it returns not-filled
    auto [s2b, h2b] = cache.get_lru_item(2);
    ASSERT_FALSE(h2b);  // not filled until set_filled is called
}

TEST(moe_lru_cache, set_filled_out_of_range_is_safe) {
    LRUCache cache(2);
    // Should not crash
    cache.set_filled(100);
    cache.set_filled(std::numeric_limits<size_t>::max());
}

// --------------------------------------------------
// evict_one: explicit eviction
// --------------------------------------------------

TEST(moe_lru_cache, explicit_evict_reduces_size) {
    LRUCache cache(4);

    cache.get_lru_item(0);
    cache.get_lru_item(1);
    ASSERT_EQ(cache.size(), 2U);

    cache.evict_one();
    ASSERT_EQ(cache.size(), 1U);

    cache.evict_one();
    ASSERT_EQ(cache.size(), 0U);
}

TEST(moe_lru_cache, evict_on_empty_is_safe) {
    LRUCache cache(4);
    // Should not crash
    cache.evict_one();
    ASSERT_EQ(cache.size(), 0U);
}

// --------------------------------------------------
// Capacity = 1 edge case
// --------------------------------------------------

TEST(moe_lru_cache, capacity_one) {
    LRUCache cache(1);

    auto [s0, h0] = cache.get_lru_item(0);
    ASSERT_EQ(s0, 0U);
    ASSERT_FALSE(h0);

    cache.set_filled(s0);

    // Access same -> hit
    auto [s0b, h0b] = cache.get_lru_item(0);
    ASSERT_EQ(s0b, 0U);
    ASSERT_TRUE(h0b);

    // Insert new -> evicts the only entry, reuses slot 0
    auto [s1, h1] = cache.get_lru_item(1);
    ASSERT_EQ(s1, 0U);
    ASSERT_FALSE(h1);
    ASSERT_EQ(cache.size(), 1U);
}

// --------------------------------------------------
// Stress: many inserts and evictions
// --------------------------------------------------

TEST(moe_lru_cache, stress_many_experts) {
    const size_t cap = 8;
    LRUCache cache(cap);

    // Insert 100 unique (layer=0, expert=i) entries into a cache with capacity 8
    for (size_t i = 0; i < 100; ++i) {
        auto [slot, hit] = cache.get_lru_item(i);
        ASSERT_LT(slot, cap);
        if (i < cap) {
            ASSERT_EQ(slot, i);
        }
    }
    ASSERT_EQ(cache.size(), cap);

    // The last `cap` experts should still be in the cache
    for (size_t i = 100 - cap; i < 100; ++i) {
        auto [slot, hit] = cache.get_lru_item(i);
        // They are in cache (though not filled)
        ASSERT_LT(slot, cap);
    }
}

// --------------------------------------------------
// Thread safety: concurrent get_lru_item calls
// --------------------------------------------------

TEST(moe_lru_cache, concurrent_access) {
    const size_t cap = 16;
    LRUCache cache(cap);
    const int num_threads = 4;
    const int ops_per_thread = 200;

    std::vector<std::thread> threads;
    std::atomic<bool> start{false};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load()) {}  // spin until all threads are ready
            for (int i = 0; i < ops_per_thread; ++i) {
                size_t expert = t * 32 + i % 32;
                auto [slot, hit] = cache.get_lru_item(expert);
                ASSERT_LT(slot, cap);
                if (!hit) {
                    cache.set_filled(slot);
                }
            }
        });
    }

    start.store(true);
    for (auto& th : threads) {
        th.join();
    }

    ASSERT_LE(cache.size(), cap);
}

