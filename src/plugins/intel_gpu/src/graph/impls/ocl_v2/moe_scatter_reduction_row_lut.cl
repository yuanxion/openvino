// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define VLOAD  CAT(vload,  VEC_BLK_SIZE)
#define VSTORE CAT(vstore, VEC_BLK_SIZE)
#define INPUT_VEC_TYPE  MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_BLK_SIZE)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_BLK_SIZE)

// Weighted reduction of the expert-major GEMM output back into token order.
//
// Input rows are grouped by expert, so the row holding the (token, k)-th expert output cannot be
// derived from the token id alone. `row_lut` is the host-built inverse of the gather order:
// row_lut[token * ACTIVE_EXPERTS + k] is that row, or negative when the pair is unused. It
// replaces the per-token linear search over the expert token lists.
KERNEL(moe_scatter_reduction_row_lut)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* expert_weights,
    const __global INPUT2_TYPE* row_lut,
    __global OUTPUT_TYPE* output
)
{
    const uint token = (uint)get_group_id(0);
    const uint lane_base = (uint)get_local_id(0) * VEC_BLK_SIZE * BATCHES_PER_THREAD;

    OUTPUT_VEC_TYPE acc[BATCHES_PER_THREAD];
    for (uint j = 0; j < BATCHES_PER_THREAD; j++) {
        acc[j] = TO_OUTPUT_TYPE(0);
    }

    for (uint k = 0; k < ACTIVE_EXPERTS; k++) {
        const int row = row_lut[token * ACTIVE_EXPERTS + k];
        if (row < 0)
            continue;

        const INPUT1_TYPE expert_weight = expert_weights[token * ACTIVE_EXPERTS + k];
        const uint in_base = (uint)row * HIDDEN_SIZE + lane_base;

        for (uint j = 0; j < BATCHES_PER_THREAD; j++) {
            INPUT_VEC_TYPE input_data = VLOAD(0, &input[in_base + j * VEC_BLK_SIZE]);
#ifdef LES_CANONICAL_SCATTER
            const uint pair_base = (token * ACTIVE_EXPERTS + k) * HIDDEN_SIZE + lane_base;
            VSTORE(input_data, 0, &output[pair_base + j * VEC_BLK_SIZE]);
#else
            input_data *= expert_weight;
            acc[j] += input_data;
#endif
        }
    }

#ifndef LES_CANONICAL_SCATTER
    const uint out_base = token * HIDDEN_SIZE + lane_base;
    for (uint j = 0; j < BATCHES_PER_THREAD; j++) {
        VSTORE(acc[j], 0, &output[out_base + j * VEC_BLK_SIZE]);
    }
#endif
}
