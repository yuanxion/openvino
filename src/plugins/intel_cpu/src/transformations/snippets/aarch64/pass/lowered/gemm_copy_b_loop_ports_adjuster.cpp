// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_copy_b_loop_ports_adjuster.hpp"

#include <memory>

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"
#include "snippets/runtime_configurator.hpp"
#include "transformations/snippets/aarch64/pass/lowered/adjust_gemm_copy_b_loop_ports.hpp"

namespace ov::intel_cpu::pass::aarch64 {

GemmCopyBLoopPortsAdjuster::GemmCopyBLoopPortsAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                       const CPURuntimeConfigurator* configurator)
    : ov::snippets::lowered::pass::RuntimeOptimizer(configurator) {
    if (!linear_ir->is_dynamic()) {
        return;
    }

    const auto& pass = std::make_shared<intel_cpu::pass::aarch64::AdjustGemmCopyBLoopPorts>();
    pass->run(*linear_ir);
    const auto& affected_uni_loops = pass->get_affected_loops();
    const snippets::lowered::LoopManagerPtr& loop_manager = linear_ir->get_loop_manager();
    const auto& loop_map = loop_manager->get_map();
    for (const auto& p : loop_map) {
        if (const auto& exp_loop = ov::as_type_ptr<snippets::lowered::ExpandedLoopInfo>(p.second)) {
            const auto& uni_loop = exp_loop->get_unified_loop_info();
            if (affected_uni_loops.count(uni_loop)) {
                m_affected_uni2exp_map[uni_loop].push_back(exp_loop);
            }
        }
    }
}

bool GemmCopyBLoopPortsAdjuster::run([[maybe_unused]] const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::GemmCopyBLoopPortsAdjuster")
    for (const auto& p : m_affected_uni2exp_map) {
        const auto& uni_loop = p.first;
        const auto& exp_loops = p.second;
        snippets::RuntimeConfigurator::LoopInfoRuntimeParamsMap initialized_info;
        if (intel_cpu::pass::aarch64::AdjustGemmCopyBLoopPorts::update_loop_info(uni_loop)) {
            initialized_info[uni_loop] = snippets::RuntimeConfigurator::get_loop_runtime_params(uni_loop);
            for (const auto& exp_loop : exp_loops) {
                snippets::RuntimeConfigurator::update_expanded_loop_info(exp_loop, initialized_info);
            }
        }
    }
    return true;
}

}  // namespace ov::intel_cpu::pass::aarch64
