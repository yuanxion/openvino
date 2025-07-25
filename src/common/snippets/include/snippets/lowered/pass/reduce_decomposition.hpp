// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface ReduceDecomposition
 * @brief Decomposes snippets::Reduce operations to a range of low-level operations on linear IR
 * @attention Only Reduce by last dimension is supported
 * @ingroup snippets
 */
class ReduceDecomposition : public RangedPass {
public:
    OPENVINO_RTTI("ReduceDecomposition", "", RangedPass);
    explicit ReduceDecomposition(size_t vector_size);
    bool run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) override;

private:
    size_t m_vector_size = 0;
};

}  // namespace ov::snippets::lowered::pass
