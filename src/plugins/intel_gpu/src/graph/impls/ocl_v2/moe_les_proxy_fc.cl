// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if LES_PROXY_FC_ENABLE
__attribute__((intel_reqd_sub_group_size(LES_PROXY_SUBGROUP_SIZE)))
KERNEL(moe_les_proxy_fc)(const __global MOE_DTYPE* hidden,
                         const __global MOE_DTYPE* weights,
                         const __global MOE_DTYPE* bias,
                         const uint hidden_size,
                         const uint has_bias,
                         __global MOE_DTYPE* logits) {
    const uint expert = (uint)get_group_id(0);
    if (expert >= LES_PROXY_EXPERTS) {
        return;
    }
    const uint lane = get_sub_group_local_id();
    float sum = 0.0f;
    for (uint index = lane; index < hidden_size; index += LES_PROXY_SUBGROUP_SIZE) {
        sum = fma((float)weights[expert * hidden_size + index], (float)hidden[index], sum);
    }
    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
        logits[expert] = (MOE_DTYPE)(sum + (has_bias ? (float)bias[expert] : 0.0f));
    }
}
#endif