// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/internal_properties.hpp"

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

}  // namespace ov::test