// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/runtime/les_context.hpp"
#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "runtime/ocl/ocl_event.hpp"

#include <chrono>
#include <future>

namespace ov::test {

using ::tests::get_test_default_config;
using ::tests::get_test_engine;

TEST(moe_offload_property_test, execution_config_roundtrip) {
    auto config = get_test_default_config(get_test_engine());

    ASSERT_EQ(config.get_offload_ratio(), 0U);

    config.set_property(ov::intel_gpu::offload_ratio(37));

    ASSERT_EQ(config.get_offload_ratio(), 37U);
}

TEST(moe_offload_property_test, default_value_is_zero) {
    auto config = get_test_default_config(get_test_engine());
    ASSERT_EQ(config.get_offload_ratio(), 0U);
}

TEST(moe_offload_property_test, set_and_get_various_values) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::offload_ratio(1));
    ASSERT_EQ(config.get_offload_ratio(), 1U);

    config.set_property(ov::intel_gpu::offload_ratio(50));
    ASSERT_EQ(config.get_offload_ratio(), 50U);

    config.set_property(ov::intel_gpu::offload_ratio(100));
    ASSERT_EQ(config.get_offload_ratio(), 100U);
}

TEST(moe_offload_property_test, set_back_to_zero_disables) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::offload_ratio(37));
    ASSERT_EQ(config.get_offload_ratio(), 37U);

    config.set_property(ov::intel_gpu::offload_ratio(0));
    ASSERT_EQ(config.get_offload_ratio(), 0U);
}

TEST(moe_les_config, disabled_by_default) {
    cldnn::ExecutionConfig config;
    EXPECT_FALSE(config.get_enable_les());
    EXPECT_EQ(config.get_les_queue_capacity(), 32U);
}

TEST(moe_les_config, roundtrip_and_clone) {
    cldnn::ExecutionConfig config;
    config.set_property({ov::intel_gpu::enable_les(true), ov::intel_gpu::les_queue_capacity(7)});
    auto cloned = config.clone();
    EXPECT_TRUE(cloned.get_enable_les());
    EXPECT_EQ(cloned.get_les_queue_capacity(), 7U);
    config.set_property(ov::intel_gpu::enable_les(false));
    EXPECT_FALSE(config.get_enable_les());
    EXPECT_TRUE(cloned.get_enable_les());
}

TEST(moe_les_config, rejects_zero_capacity) {
    cldnn::ExecutionConfig config;
    EXPECT_ANY_THROW(config.set_property(ov::intel_gpu::les_queue_capacity(0)));
}

TEST(moe_les_config, rejects_excessive_capacity) {
    cldnn::ExecutionConfig config;
    EXPECT_ANY_THROW(config.set_property(ov::intel_gpu::les_queue_capacity(1025)));
}

TEST(moe_les_config, accepts_capacity_boundaries) {
    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::les_queue_capacity(1));
    EXPECT_EQ(config.get_les_queue_capacity(), 1U);
    config.set_property(ov::intel_gpu::les_queue_capacity(1024));
    EXPECT_EQ(config.get_les_queue_capacity(), 1024U);
}

TEST(moe_les_network, non_moe_graph_creates_no_context_and_preserves_output) {
    auto& engine = get_test_engine();
    cldnn::layout input_layout{cldnn::data_types::f32, cldnn::format::bfyx, {1, 1, 1, 2}};
    auto input = engine.allocate_memory(input_layout);
    tests::set_values(input, std::vector<float>{-2.0f, 3.0f});
    cldnn::topology topology;
    topology.add(cldnn::input_layout("input", input_layout));
    topology.add(cldnn::activation("output", cldnn::input_info("input"), cldnn::activation_func::relu));
    for (size_t ratio : {0U, 50U}) {
        auto config = get_test_default_config(engine);
        config.set_property({ov::intel_gpu::enable_les(true), ov::intel_gpu::offload_ratio(ratio)});
        cldnn::network network(engine, topology, config);
        EXPECT_EQ(network.get_les_context(), nullptr);
        network.set_input_data("input", input);
        auto output = network.execute().at("output").get_memory();
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> values(output, network.get_stream());
        EXPECT_FLOAT_EQ(values[0], 0.0f);
        EXPECT_FLOAT_EQ(values[1], 3.0f);
        EXPECT_EQ(network.get_les_epoch(), 0U);
    }
}

TEST(moe_les_network, transfer_lane_is_retained_until_active_upload_finishes) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::queue_type(cldnn::QueueTypes::in_order));
    auto transfer = engine.create_stream(config);
    std::weak_ptr<cldnn::stream> lifetime = transfer;
    auto context = std::make_unique<cldnn::les_context>(2, std::move(transfer));
    auto memory = engine.allocate_memory(cldnn::layout{cldnn::data_types::f32, cldnn::format::bfyx, {1, 1, 1, 2}});
    cl::UserEvent gate(dynamic_cast<cldnn::ocl::ocl_engine&>(engine).get_cl_context());
    std::promise<void> submitted;
    auto ready = submitted.get_future();
    auto ticket = context->submit(context->begin_epoch(), [&, lane = context->get_transfer_stream(), memory] {
        std::vector<float> source{2.0f, 3.0f};
        dynamic_cast<cldnn::ocl::ocl_stream&>(*lane).enqueue_external_barrier({std::make_shared<cldnn::ocl::ocl_event>(gate)});
        auto uploaded = memory->copy_from(*lane, source.data(), false);
        lane->flush();
        submitted.set_value();
        uploaded->wait();
    });
    auto status = ready.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        gate.setStatus(CL_COMPLETE);
        ASSERT_EQ(status, std::future_status::ready);
    }
    auto destroyed = std::async(std::launch::async, [owned = std::move(context)]() mutable { owned.reset(); });
    auto blocked = destroyed.wait_for(std::chrono::milliseconds(20));
    gate.setStatus(CL_COMPLETE);
    EXPECT_EQ(blocked, std::future_status::timeout);
    ASSERT_EQ(destroyed.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    destroyed.get();
    EXPECT_THROW(ticket.completion.get(), cldnn::les_task_cancelled);
    EXPECT_TRUE(lifetime.expired());
    cldnn::mem_lock<float, cldnn::mem_lock_type::read> values(memory, tests::get_test_stream());
    EXPECT_FLOAT_EQ(values[0], 2.0f);
    EXPECT_FLOAT_EQ(values[1], 3.0f);
}

}  // namespace ov::test