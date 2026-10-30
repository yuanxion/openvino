// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"
#include <array>
#include <chrono>
#include <fstream>

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
using ov::intel_gpu::ocl::moe::ResidentExpertWeightProvider;

namespace {
namespace otd = ov::intel_gpu::ocl::moe_otd;

class moe_les_upload : public ::testing::Test {
protected:
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ov_les_upload_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");

    void TearDown() override { std::filesystem::remove(path); }

    void write(const std::vector<uint8_t>& bytes) {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ASSERT_TRUE(output.good());
    }

    stream::ptr lane(bool profiling = false) {
        auto& engine = get_test_engine();
        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order), ov::enable_profiling(profiling));
        return engine.create_stream(config);
    }

    memory::ptr buffer(size_t count, data_types type = data_types::u8, size_t experts = 1) {
        return get_test_engine().allocate_memory(layout{type, format::bfyx,
            {static_cast<int>(experts), 1, 1, static_cast<int>(count)}});
    }
};

TEST_F(moe_les_upload, all_nine_components_are_ready_and_repacked) {
    MOECompressed::Config config{};
    config.num_expert = 2;
    config.hidden_size = 4;
    config.inter_size = 4;
    config.group_size = 2;
    for (bool profiling : {false, true}) {
        auto transfer = lane(profiling);
        moe_weights weights;
        std::array<memory::ptr*, 9> tensors{&weights.gate_w, &weights.up_w, &weights.down_w,
            &weights.gate_s, &weights.up_s, &weights.down_s, &weights.gate_z, &weights.up_z, &weights.down_z};
        std::vector<size_t> offsets;
        std::vector<uint8_t> file;
        for (size_t component = 0; component < tensors.size(); ++component) {
            *tensors[component] = component < 3 ? buffer(16, data_types::u8, 2) :
                buffer(8, component < 6 ? data_types::f16 : data_types::u4, 2);
            offsets.push_back(file.size());
            auto bytes = (*tensors[component])->get_layout().bytes_count();
            for (size_t index = 0; index < bytes; ++index) {
                file.push_back(static_cast<uint8_t>((index + component * 17) % 251));
            }
        }
        write(file);
        otd::ParallelWeightReader reader(path);
        auto events = otd::fill_weights_memory_async(*transfer, config, offsets, reader, weights, {1}, {1});
        ASSERT_EQ(events.size(), 9U);
        for (const auto& event : events) {
            EXPECT_TRUE(event->is_set());
        }
        for (size_t component = 0; component < tensors.size(); ++component) {
            auto tensor = *tensors[component];
            auto bytes = tensor->get_layout().bytes_count() / 2;
            auto begin = file.begin() + offsets[component] + bytes;
            std::vector<uint8_t> expected(begin, begin + bytes);
            otd::maybe_transpose_scale_zp(config, component, tensor->get_layout(), expected, bytes);
            std::vector<uint8_t> actual(bytes);
            tensor->copy_to(*transfer, actual.data(), bytes, 0, bytes, true);
            EXPECT_EQ(actual, expected);
        }
    }
}

TEST_F(moe_les_upload, staging_reuse_preserves_large_tensor_source) {
    std::vector<uint8_t> source(16 * 1024 * 1024 + 37);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint8_t>(index % 251);
    }
    write(source);
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(source.size());
    auto transfer = lane();
    otd::ParallelWeightReader reader(path);
    auto events = otd::fill_weights_memory_async(*transfer, config, std::vector<size_t>(9), reader, weights, {0}, {0});
    EXPECT_EQ(events.size(), 3U);
    std::vector<uint8_t> actual(source.size());
    weights.gate_w->copy_to(*transfer, actual.data(), true);
    EXPECT_EQ(actual, source);
}

TEST_F(moe_les_upload, validates_entire_plan_before_first_upload) {
    write(std::vector<uint8_t>(32, 7));
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(16);
    weights.up_w = buffer(16);
    auto transfer = lane();
    std::vector<uint8_t> sentinel(16, 77);
    weights.gate_w->copy_from(*transfer, sentinel.data(), true);
    std::vector<size_t> offsets(9);
    offsets[1] = 100;
    otd::ParallelWeightReader reader(path);
    EXPECT_ANY_THROW(otd::fill_weights_memory_async(*transfer, config, offsets, reader, weights, {0}, {0}));
    std::vector<uint8_t> actual(16);
    weights.gate_w->copy_to(*transfer, actual.data(), true);
    EXPECT_EQ(actual, sentinel);
}

TEST_F(moe_les_upload, truncated_file_is_an_error_not_gpu_ready) {
    write(std::vector<uint8_t>(32, 7));
    otd::ParallelWeightReader reader(path);
    ASSERT_EQ(reader.file_size(), 32U);
    std::filesystem::resize_file(path, 8);
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(32);
    auto transfer = lane();
    EXPECT_ANY_THROW(otd::fill_weights_memory_async(*transfer, config, std::vector<size_t>(9), reader, weights, {0}, {0}));
    EXPECT_NO_THROW(transfer->finish());
}

TEST_F(moe_les_upload, missing_file_reports_open_failure) {
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(16);
    auto transfer = lane();
    otd::ParallelWeightReader reader(path);
    EXPECT_ANY_THROW(otd::fill_weights_memory_async(*transfer, config, std::vector<size_t>(9), reader, weights, {0}, {0}));
}

TEST_F(moe_les_upload, later_short_read_drains_already_submitted_dma) {
    write(std::vector<uint8_t>(64, 7));
    otd::ParallelWeightReader reader(path);
    ASSERT_EQ(reader.file_size(), 64U);
    std::filesystem::resize_file(path, 32);
    MOECompressed::Config config{};
    config.num_expert = 1;
    moe_weights weights;
    weights.gate_w = buffer(32);
    weights.up_w = buffer(32);
    auto transfer = lane();
    std::vector<size_t> offsets(9);
    offsets[1] = 32;
    EXPECT_ANY_THROW(otd::fill_weights_memory_async(*transfer, config, offsets, reader, weights, {0}, {0}));
    std::vector<uint8_t> actual(32);
    weights.gate_w->copy_to(*transfer, actual.data(), true);
    EXPECT_EQ(actual, std::vector<uint8_t>(32, 7));
}
}  // namespace

TEST(moe_expert_weight_provider, resident_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_FALSE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 0U);

    // Fully resident: try_acquire_simultaneous returns the expert ids unchanged, including duplicates
    // (no remap, no dedup) - the expert id is already the addressable slot.
    std::vector<uint32_t> experts = {3, 1, 4, 1, 5, 9, 2, 6};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 1, 4, 1, 5, 9, 2, 6}));
}

TEST(moe_expert_weight_provider, offload_construction_state) {
    MOECompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};
    OffloadExpertWeightProvider provider(/*capacity=*/16, config, offsets, "/path/to/weights.bin");

    ASSERT_TRUE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 16U);
    // Resident buffers are bound lazily at first execute, not at construction.
    ASSERT_FALSE(provider.is_bound());
    ASSERT_FALSE(provider.cache().is_initialized());
}

TEST(moe_expert_weight_provider, offload_bind_resident_buffers) {
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/8, config, {}, "/path/to/weights.bin");
    ASSERT_FALSE(provider.is_bound());

    cldnn::moe_weights resident;  // null device pointers are fine; only the binding flag matters here
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
    ASSERT_TRUE(provider.cache().is_initialized());

    // Idempotent: a second bind keeps the provider bound.
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
}

TEST(moe_expert_weight_provider, resident_try_acquire_simultaneous_is_identity) {
    ResidentExpertWeightProvider provider;
    std::vector<uint32_t> experts = {3, 7, 3, 1, 7};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 7, 3, 1, 7}));
}

TEST(moe_expert_weight_provider, resident_acquire_one_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_EQ(provider.acquire_one(42, get_test_stream()), 42U);
    ASSERT_EQ(provider.acquire_one(0, get_test_stream()), static_cast<size_t>(0));
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_capacity_overflow) {
    // Capacity=2 means at most 2 unique experts can be acquired simultaneously.
    // Requesting 3 unique experts must return nullopt (capacity overflow).
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate the LRU cache so experts 0,1 are hits (no disk I/O needed)
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 3 unique experts (0, 1, 2) exceeds capacity=2 -> nullopt
    std::vector<uint32_t> experts = {0, 1, 2, 0, 1};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_FALSE(lease.has_value());
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_within_capacity) {
    // Same setup but with only 2 unique experts -> should succeed
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 2 unique experts (0, 1) with duplicates -> within capacity=2
    std::vector<uint32_t> experts = {0, 1, 0, 1, 0};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    // Slots should be valid (cache returned slots for experts 0 and 1)
    ASSERT_EQ(lease->size(), 5U);
    // Duplicates must map to the same slot
    ASSERT_EQ((*lease)[0], (*lease)[2]);
    ASSERT_EQ((*lease)[0], (*lease)[4]);
    ASSERT_EQ((*lease)[1], (*lease)[3]);
}

TEST(moe_expert_weight_provider, offload_acquire_one_basic) {
    // acquire_one always succeeds (evicts LRU if needed)
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    auto [slot0, hit0] = cache.get_lru_item(5);
    cache.set_filled(slot0);

    // acquire_one should return a valid slot
    auto slot = provider.acquire_one(5, get_test_stream());
    ASSERT_EQ(slot, slot0);
}
