// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_list.hpp"

#include <vector>

#if defined(OV_CPU_WITH_ACL) || defined(OV_CPU_WITH_SHL)
#    include <memory>

#    include "nodes/executors/executor.hpp"
#endif

#if defined(OV_CPU_WITH_SHL)
#    include "nodes/executors/shl/shl_eltwise.hpp"
#endif
#include "utils/arch_macros.h"

namespace ov::intel_cpu {

const std::vector<EltwiseExecutorDesc>& getEltwiseExecutorsList() {
    static std::vector<EltwiseExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclEltwiseExecutorBuilder>())
            OV_CPU_INSTANCE_SHL(ExecutorType::Shl, std::make_shared<ShlEltwiseExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu
