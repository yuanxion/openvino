// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if LES_PROXY_FC_ENABLE
KERNEL(moe_les_proxy_fc)(const __global MOE_DTYPE* hidden,
                         const __global MOE_DTYPE* weights,
                         const __global MOE_DTYPE* bias,
                         const uint hidden_size,
                         const uint has_bias,
                         __global MOE_DTYPE* logits) {
    const uint expert = (uint)get_global_id(0);
    float sum = has_bias ? (float)bias[expert] : 0.0f;
    for (uint index = 0; index < hidden_size; ++index) {
        sum = fma((float)weights[expert * hidden_size + index], (float)hidden[index], sum);
    }
    logits[expert] = (MOE_DTYPE)sum;
}
#endif