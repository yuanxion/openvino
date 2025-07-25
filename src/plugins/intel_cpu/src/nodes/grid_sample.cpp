// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample.hpp"

#include <memory>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/grid_sample.hpp"

using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <algorithm>
#    include <cpu/x64/cpu_isa_traits.hpp>
#    include <cstddef>
#    include <cstdint>
#    include <functional>
#    include <numeric>
#    include <oneapi/dnnl/dnnl_common.hpp>

#    include "cpu_types.h"
#    include "graph_context.h"
#    include "memory_desc/cpu_memory_desc.h"
#    include "node.h"
#    include "nodes/kernels/x64/grid_sample.hpp"
#    include "onednn/iml_type_mapper.h"
#    include "openvino/core/except.hpp"
#    include "openvino/core/parallel.hpp"
#    include "openvino/core/type/element_type.hpp"
#    include "shape_inference/shape_inference_cpu.hpp"

using namespace dnnl::impl::cpu;
#endif  // OPENVINO_ARCH_X86_64

bool GridSample::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v9::GridSample>(op)) {
            errorMessage = "Not supported GridSample operation version. CPU plug-in supports only 9th version.";
            return false;
        }
#if defined(OPENVINO_ARCH_X86_64)
        if (!x64::mayiuse(x64::sse41)) {
            errorMessage = "Not supported CPU instructions set.";
            return false;
        }
#else
        return false;
#endif  // OPENVINO_ARCH_X86_64
    } catch (...) {
        return false;
    }

    return true;
}

#if defined(OPENVINO_ARCH_X86_64)

GridSample::GridSample(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    CPU_NODE_ASSERT(op->get_input_size() == 2 && op->get_output_size() == 1,
                    "has incorrect number of input/output ports.");

    const auto& dataShape = getInputShapeAtPort(IN_DATA);
    CPU_NODE_ASSERT(dataShape.getRank() == 4, "has incorrect rank of the Data input.");

    const auto& gridShape = getInputShapeAtPort(IN_GRID);
    CPU_NODE_ASSERT(gridShape.getRank() == 4, "has incorrect rank of the Grid input.");
    CPU_NODE_ASSERT(!gridShape.isStatic() || gridShape.getDims()[3] == 2,
                    "has incorrect shape of the Grid input. The 4th dimension should be equal to 2.");

    const auto& attributes = ov::as_type_ptr<ov::op::v9::GridSample>(op)->get_attributes();
    alignCorners = attributes.align_corners;
    switch (attributes.mode) {
    case op::v9::GridSample::InterpolationMode::BILINEAR:
        interpolationMode = GridSampleInterpolationMode::BILINEAR;
        break;
    case op::v9::GridSample::InterpolationMode::BICUBIC:
        interpolationMode = GridSampleInterpolationMode::BICUBIC;
        break;
    case op::v9::GridSample::InterpolationMode::NEAREST:
        interpolationMode = GridSampleInterpolationMode::NEAREST;
        break;
    default:
        CPU_NODE_THROW("supports only BILINEAR, BICUBIC, NEAREST interpolation modes.");
    }
    switch (attributes.padding_mode) {
    case op::v9::GridSample::PaddingMode::ZEROS:
        paddingMode = GridSamplePaddingMode::ZEROS;
        break;
    case op::v9::GridSample::PaddingMode::BORDER:
        paddingMode = GridSamplePaddingMode::BORDER;
        break;
    case op::v9::GridSample::PaddingMode::REFLECTION:
        paddingMode = GridSamplePaddingMode::REFLECTION;
        break;
    default:
        CPU_NODE_THROW("supports only BORDER, REFLECTION, ZEROS paddings modes.");
    }
}

void GridSample::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    dataPrecision = getOriginalInputPrecisionAtPort(IN_DATA);
    if (dataPrecision != ov::element::i32) {
        dataPrecision = ov::element::f32;
    }
    dataTypeSize = dataPrecision.size();
    gridTypeSize = gridPrecision.size();

    impl_desc_type implType = jit_sse42;
    if (x64::mayiuse(x64::avx512_core)) {
        implType = jit_avx512;
    } else if (x64::mayiuse(x64::avx2)) {
        implType = jit_avx2;
    }

    // 95905 - to add nspc layout support.
    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision}, {LayoutType::ncsp, gridPrecision}},
                         {{LayoutType::ncsp, dataPrecision}},
                         implType);
}

void GridSample::createPrimitive() {
    kernel::GridSampleKernelConfParams jcp;

    jcp.inDataPrc = dataPrecision;
    jcp.gridPrc = gridPrecision;
    jcp.dynamicShapes = isDynamicNode();
    jcp.alignCorners = alignCorners;
    jcp.interpolationMode = interpolationMode;
    jcp.paddingMode = paddingMode;

    const auto& srcDataDims = getInputShapeAtPort(IN_DATA).getDims();
    if (!jcp.dynamicShapes) {
        jcp.batchNum = srcDataDims[0];
        jcp.cannelNum = srcDataDims[1];
        jcp.dynamicBatch = false;
        jcp.dynamicChannel = false;
        jcp.srcBatchStepB =
            std::accumulate(srcDataDims.begin() + 1, srcDataDims.end(), dataTypeSize, std::multiplies<>());
    } else {
        jcp.dynamicBatch = srcDataDims[0] == Shape::UNDEFINED_DIM;
        jcp.batchNum = jcp.dynamicBatch ? 1LU : srcDataDims[0];
        jcp.dynamicChannel = srcDataDims[1] == Shape::UNDEFINED_DIM;
        jcp.cannelNum = jcp.dynamicChannel ? 1LU : srcDataDims[1];
    }

    if (x64::mayiuse(x64::avx512_core)) {
        jitKernel = std::make_shared<kernel::GridSampleKernel<x64::avx512_core>>(jcp);
    } else if (x64::mayiuse(x64::avx2)) {
        jitKernel = std::make_shared<kernel::GridSampleKernel<x64::avx2>>(jcp);
    } else if (x64::mayiuse(x64::sse41)) {
        jitKernel = std::make_shared<kernel::GridSampleKernel<x64::sse41>>(jcp);
    }
    CPU_NODE_ASSERT(jitKernel, "could not create JIT kernel.");
    jitKernel->create_ker();

    m_threads_num = parallel_get_max_threads();
    execParamsPerThread.resize(m_threads_num);
    if (!x64::mayiuse(x64::avx512_core)) {
        const auto dataElPerVec = jitKernel->getDataElPerVec();
        parallel_nt(m_threads_num, [&](const int ithr, [[maybe_unused]] const int nthr) {
            auto& p = execParamsPerThread[ithr];

            p.srcHeightF.resize(dataElPerVec);
            p.srcWidthF.resize(dataElPerVec);
            p.srcWidthB.resize(dataElPerVec);
            p.dataTypeSize.resize(dataElPerVec);
            p.srcHeightSub1F.resize(dataElPerVec);
            p.srcWidthSub1F.resize(dataElPerVec);
            p.srcHeightMul2F.resize(dataElPerVec);
            p.srcWidthMul2F.resize(dataElPerVec);
            p.srcHeightMul2Sub1F.resize(dataElPerVec);
            p.srcWidthMul2Sub1F.resize(dataElPerVec);
            if (alignCorners) {
                p.wDenormCoefF.resize(dataElPerVec);
                p.hDenormCoefF.resize(dataElPerVec);
            }
            if (interpolationMode == GridSampleInterpolationMode::BICUBIC) {
                const size_t vecNum = paddingMode == GridSamplePaddingMode::ZEROS ? 32 : 16;
                p.buffer.resize(dataElPerVec * dataTypeSize * vecNum);
            }
        });
    }

    Node::createPrimitive();
}

void GridSample::prepareParams() {
    auto dataMemPtr = getSrcMemoryAtPort(IN_DATA);
    CPU_NODE_ASSERT(dataMemPtr && dataMemPtr->isDefined(), "has undefined input data memory.");
    auto gridMemPtr = getSrcMemoryAtPort(IN_GRID);
    CPU_NODE_ASSERT(gridMemPtr && gridMemPtr->isDefined(), "has undefined input grid memory.");
    auto dstMemPtr = getDstMemoryAtPort(0);
    CPU_NODE_ASSERT(dstMemPtr && dstMemPtr->isDefined(), "has undefined output memory.");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor() != nullptr, "has unidentified preferable primitive descriptor.");

    const uint64_t dataElPerVec = jitKernel->getDataElPerVec();
    const auto& srcDataShape = dataMemPtr->getStaticDims();
    const auto& dstShape = dstMemPtr->getStaticDims();
    const uint64_t totalWork = dstShape[2] * dstShape[3];
    const uint64_t wpt = ((totalWork / dataElPerVec) / m_threads_num + 1) * dataElPerVec;

    parallel_nt(m_threads_num, [&](const int ithr, [[maybe_unused]] const int nthr) {
        const uint64_t dstStart = std::min(wpt * ithr, totalWork);
        const uint64_t dstEnd = std::min(wpt * (ithr + 1), totalWork);

        auto& p = execParamsPerThread[ithr];

        p.workAmount = dstEnd - dstStart;
        if (p.workAmount == 0LU) {
            return;
        }

        p.batchNum = srcDataShape[0];
        p.channelsNum = srcDataShape[1];
        p.srcHeightF[0] = srcDataShape[2];
        p.srcWidthF[0] = srcDataShape[3];

        p.gridStartB = dstStart * 2 * gridTypeSize;
        p.dstStartB = dstStart * dataTypeSize;

        p.srcBatchStepB =
            std::accumulate(srcDataShape.begin() + 1, srcDataShape.end(), dataTypeSize, std::multiplies<>());
        p.gridBatchStepB = (dstShape[2] * dstShape[3] - p.workAmount) * 2 * gridTypeSize;
        p.dstBatchStepB = (dstShape[1] * dstShape[2] * dstShape[3] - p.workAmount) * dataTypeSize;

        p.srcChannelStepB = srcDataShape[2] * srcDataShape[3] * dataTypeSize;
        p.dstChannelStepB = dstShape[2] * dstShape[3] * dataTypeSize;
        p.dataTypeSize[0] = dataTypeSize;

        p.srcHeightSub1F[0] = p.srcHeightF[0] - 1.F;
        p.srcWidthSub1F[0] = p.srcWidthF[0] - 1.F;
        p.srcHeightMul2F[0] = p.srcHeightF[0] * 2.F;
        p.srcWidthMul2F[0] = p.srcWidthF[0] * 2.F;
        if (interpolationMode == GridSampleInterpolationMode::BICUBIC && srcDataShape[3] >= 4) {
            p.srcWidthB[0] = (srcDataShape[3] - 3) * dataTypeSize;
        } else {
            p.srcWidthB[0] = srcDataShape[3] * dataTypeSize;
        }
        if (alignCorners) {
            p.srcHeightMul2Sub1F[0] = p.srcHeightF[0] == 1.F ? 1.F : p.srcHeightSub1F[0] * 2.F;
            p.srcWidthMul2Sub1F[0] = p.srcWidthF[0] == 1.F ? 1.F : p.srcWidthSub1F[0] * 2.F;
            p.wDenormCoefF[0] = (p.srcWidthF[0] - 1.F) / 2.F;
            p.hDenormCoefF[0] = (p.srcHeightF[0] - 1.F) / 2.F;
        } else {
            p.srcHeightMul2Sub1F[0] = p.srcHeightMul2F[0] - 1.F;
            p.srcWidthMul2Sub1F[0] = p.srcWidthMul2F[0] - 1.F;
        }
        if (!x64::mayiuse(x64::avx512_core)) {
            std::fill(p.srcHeightF.begin(), p.srcHeightF.end(), p.srcHeightF[0]);
            std::fill(p.srcWidthF.begin(), p.srcWidthF.end(), p.srcWidthF[0]);
            std::fill(p.dataTypeSize.begin(), p.dataTypeSize.end(), p.dataTypeSize[0]);
            std::fill(p.srcHeightSub1F.begin(), p.srcHeightSub1F.end(), p.srcHeightSub1F[0]);
            std::fill(p.srcWidthSub1F.begin(), p.srcWidthSub1F.end(), p.srcWidthSub1F[0]);
            std::fill(p.srcHeightMul2F.begin(), p.srcHeightMul2F.end(), p.srcHeightMul2F[0]);
            std::fill(p.srcWidthMul2F.begin(), p.srcWidthMul2F.end(), p.srcWidthMul2F[0]);
            std::fill(p.srcWidthB.begin(), p.srcWidthB.end(), p.srcWidthB[0]);
            std::fill(p.srcHeightMul2Sub1F.begin(), p.srcHeightMul2Sub1F.end(), p.srcHeightMul2Sub1F[0]);
            std::fill(p.srcWidthMul2Sub1F.begin(), p.srcWidthMul2Sub1F.end(), p.srcWidthMul2Sub1F[0]);
            if (alignCorners) {
                std::fill(p.wDenormCoefF.begin(), p.wDenormCoefF.end(), p.wDenormCoefF[0]);
                std::fill(p.hDenormCoefF.begin(), p.hDenormCoefF.end(), p.hDenormCoefF[0]);
            }
        }
    });
}

void GridSample::execute([[maybe_unused]] const dnnl::stream& strm) {
    const void* srcData = getSrcDataAtPort(IN_DATA);
    const uint8_t* gridData = getSrcDataAtPortAs<uint8_t>(IN_GRID);
    auto* dstData = getDstDataAtPortAs<uint8_t>(0);

    auto threadBody = [&](const int ithr, [[maybe_unused]] const int nthr) {
        const auto& p = execParamsPerThread[ithr];
        auto arg = kernel::GridSamplesKernelExecArgs();
        if (p.workAmount == 0LU) {
            return;
        }

        arg.src = srcData;
        arg.grid = gridData + p.gridStartB;
        arg.dst = dstData + p.dstStartB;
        arg.batchNum = p.batchNum;
        arg.channelsNum = p.channelsNum;
        arg.srcHeightF = p.srcHeightF.data();
        arg.srcWidthF = p.srcWidthF.data();
        arg.srcWidthB = p.srcWidthB.data();
        arg.srcChannelStepB = p.srcChannelStepB;
        arg.dstChannelStepB = p.dstChannelStepB;
        arg.srcBatchStepB = p.srcBatchStepB;
        arg.gridBatchStepB = p.gridBatchStepB;
        arg.dstBatchStepB = p.dstBatchStepB;
        arg.srcHeightSub1F = p.srcHeightSub1F.data();
        arg.srcWidthSub1F = p.srcWidthSub1F.data();
        arg.srcWidthMul2F = p.srcWidthMul2F.data();
        arg.srcHeightMul2F = p.srcHeightMul2F.data();
        arg.srcHeightMul2Sub1F = p.srcHeightMul2Sub1F.data();
        arg.srcWidthMul2Sub1F = p.srcWidthMul2Sub1F.data();
        arg.wDenormCoefF = p.wDenormCoefF.data();
        arg.hDenormCoefF = p.hDenormCoefF.data();
        arg.dataTypeSize = p.dataTypeSize.data();
        arg.buffer = p.buffer.data();
        arg.workAmount = p.workAmount;

        (*jitKernel)(&arg);
    };

    parallel_nt(m_threads_num, threadBody);
}

void GridSample::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool GridSample::created() const {
    return getType() == Type::GridSample;
}

#endif  // OPENVINO_ARCH_X86_64
