// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define CL_VERSION_3_0 1
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "impls/registry/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/add.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "register.hpp"
#include "runtime/ocl/ocl_event.hpp"
#include "runtime/ocl/ocl_memory.hpp"
#include "runtime/ocl/ocl_stream.hpp"
#include "sync_tensor_inst.h"

#include "openvino/core/type/float16.hpp"

namespace cldnn {
namespace cpu {

#define CL_MEM_ALLOCATION_HANDLE_INTEL 0x10050
static std::map<int, std::string> oclErrorCode = {
    {0, "CL_SUCCESS"},
    {-1, "CL_DEVICE_NOT_FOUND"},
    {-2, "CL_DEVICE_NOT_AVAILABLE"},
    {-3, "CL_COMPILER_NOT_AVAILABLE"},
    {-4, "CL_MEM_OBJECT_ALLOCATION_FAILURE"},
    {-5, "CL_OUT_OF_RESOURCES"},
    {-6, "CL_OUT_OF_HOST_MEMORY"},
    {-7, "CL_PROFILING_INFO_NOT_AVAILABLE"},
    {-8, "CL_MEM_COPY_OVERLAP"},
    {-9, "CL_IMAGE_FORMAT_MISMATCH"},
    {-10, "CL_IMAGE_FORMAT_NOT_SUPPORTED"},
    {-11, "CL_BUILD_PROGRAM_FAILURE"},
    {-12, "CL_MAP_FAILURE"},
    {-13, "CL_MISALIGNED_SUB_BUFFER_OFFSET"},
    {-14, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"},
    {-15, "CL_COMPILE_PROGRAM_FAILURE"},
    {-16, "CL_LINKER_NOT_AVAILABLE"},
    {-17, "CL_LINK_PROGRAM_FAILURE"},
    {-18, "CL_DEVICE_PARTITION_FAILED"},
    {-19, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"},
    {-30, "CL_INVALID_VALUE"},
    {-31, "CL_INVALID_DEVICE_TYPE"},
    {-32, "CL_INVALID_PLATFORM"},
    {-33, "CL_INVALID_DEVICE"},
    {-34, "CL_INVALID_CONTEXT"},
    {-35, "CL_INVALID_QUEUE_PROPERTIES"},
    {-36, "CL_INVALID_COMMAND_QUEUE"},
    {-37, "CL_INVALID_HOST_PTR"},
    {-38, "CL_INVALID_MEM_OBJECT"},
    {-39, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"},
    {-40, "CL_INVALID_IMAGE_SIZE"},
    {-41, "CL_INVALID_SAMPLER"},
    {-42, "CL_INVALID_BINARY"},
    {-43, "CL_INVALID_BUILD_OPTIONS"},
    {-44, "CL_INVALID_PROGRAM"},
    {-45, "CL_INVALID_PROGRAM_EXECUTABLE"},
    {-46, "CL_INVALID_KERNEL_NAME"},
    {-47, "CL_INVALID_KERNEL_DEFINITION"},
    {-48, "CL_INVALID_KERNEL"},
    {-49, "CL_INVALID_ARG_INDEX"},
    {-50, "CL_INVALID_ARG_VALUE"},
    {-51, "CL_INVALID_ARG_SIZE"},
    {-52, "CL_INVALID_KERNEL_ARGS"},
    {-53, "CL_INVALID_WORK_DIMENSION"},
    {-54, "CL_INVALID_WORK_GROUP_SIZE"},
    {-55, "CL_INVALID_WORK_ITEM_SIZE"},
    {-56, "CL_INVALID_GLOBAL_OFFSET"},
    {-57, "CL_INVALID_EVENT_WAIT_LIST"},
    {-58, "CL_INVALID_EVENT"},
    {-59, "CL_INVALID_OPERATION"},
    {-60, "CL_INVALID_GL_OBJECT"},
    {-61, "CL_INVALID_BUFFER_SIZE"},
    {-62, "CL_INVALID_MIP_LEVEL"},
    {-63, "CL_INVALID_GLOBAL_WORK_SIZE"},
    {-64, "CL_INVALID_PROPERTY"},
    {-65, "CL_INVALID_IMAGE_DESCRIPTOR"},
    {-66, "CL_INVALID_COMPILER_OPTIONS"},
    {-67, "CL_INVALID_LINKER_OPTIONS"},
    {-68, "CL_INVALID_DEVICE_PARTITION_COUNT"},
    {-69, "CL_INVALID_PIPE_SIZE"},
    {-70, "CL_INVALID_DEVICE_QUEUE"},
    {-71, "CL_INVALID_SPEC_ID"},
    {-72, "CL_MAX_SIZE_RESTRICTION_EXCEEDED"},
};
#define CHECK_OCL_ERROR(err, msg)                                                                            \
    if (err < 0) {                                                                                           \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n",                                      \
               __FUNCTION__,                                                                                 \
               __LINE__,                                                                                     \
               msg,                                                                                          \
               err,                                                                                          \
               errstr.c_str());                                                                              \
    }

#define CHECK_OCL_ERROR_RETURN(err, msg, ret)                                                                \
    if (err < 0) {                                                                                           \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n",                                      \
               __FUNCTION__,                                                                                 \
               __LINE__,                                                                                     \
               msg,                                                                                          \
               err,                                                                                          \
               errstr.c_str());                                                                              \
        return ret;                                                                                          \
    }

#define CHECK_OCL_ERROR_EXIT(err, msg)                                                                       \
    if (err < 0) {                                                                                           \
        std::string errstr = (oclErrorCode.find(err) != oclErrorCode.end()) ? oclErrorCode[err] : "Unknown"; \
        printf("ERROR: oclContext::%s, line = %d, %s! err = %d (%s)\n",                                      \
               __FUNCTION__,                                                                                 \
               __LINE__,                                                                                     \
               msg,                                                                                          \
               err,                                                                                          \
               errstr.c_str());                                                                              \
        exit(1);                                                                                             \
    }
static std::mutex debug_mutex;
static const std::chrono::_V2::system_clock::time_point perf_dump_start() {
    return std::chrono::high_resolution_clock::now();
}

static void perf_dump_done(const std::chrono::_V2::system_clock::time_point& start,
                           std::string str,
                           bool enable = false) {
    if (enable) {
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed_1 = end - start;
        {
            std::lock_guard<std::mutex> lock(debug_mutex);
            std::cout << str << " cost: " << elapsed_1.count() << " ms" << std::endl << std::endl;
        }
    }
}

struct sync_tensor_impl : public typed_primitive_impl<sync_tensor> {
    using parent = typed_primitive_impl<sync_tensor>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::sync_tensor_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<sync_tensor_impl>(*this);
    }

    sync_tensor_impl() : parent() {}

    explicit sync_tensor_impl(const sync_tensor_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<sync_tensor>());
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    void check_typed_values(const ov::element::Type_t& data_type, size_t element_count,
        std::vector<uint8_t>& srcBuf, size_t w_rank) {
        printf("[check_typed_values: %ld] size: %ld \n", w_rank, srcBuf.size());
        switch (data_type) {
        case ov::element::f16: {
            uint16_t* ptr = reinterpret_cast<uint16_t*> (srcBuf.data());
            for (size_t i = 0; i < element_count; ++i) {
                ov::float16 val = ov::float16::from_bits(ptr[i]);
                std::cout << " [" << i << "] " << val << std::endl;
            }
            break;
        }
        case ov::element::i8: {
            int8_t* ptr = reinterpret_cast<int8_t*> (srcBuf.data());
            for (size_t i = 0; i < element_count; ++i) {
                std::cout << " [" << i << "] " << ptr[i] << std::endl;
            }
            break;
        }
        case ov::element::f32: {
            float* ptr = reinterpret_cast<float*> (srcBuf.data());
            for (size_t i = 0; i < element_count; ++i) {
                std::cout << " [" << i << "] " << ptr[i] << std::endl;
            }
            break;
        }
        default:
            break;
        }
    }

    void read_cl_buf(cl_command_queue& queue, const ov::element::Type_t& data_type, size_t element_count,
        cl_mem& src, std::vector<uint8_t>& srcBuf, size_t w_rank) {
        cl_int err;
        err = clEnqueueReadBuffer(queue, src, CL_TRUE, 0, srcBuf.size(), srcBuf.data(), 0, NULL, NULL);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");

        check_typed_values(data_type, element_count, srcBuf, w_rank);
    }

    void do_self_rank_all_reduce(cl_command_queue& queue, const ov::element::Type_t& data_type, size_t element_count,
        cl_mem& dst, std::vector<uint8_t>& dstBuf, size_t w_rank) {
        cl_int err;
        err = clEnqueueWriteBuffer(queue, dst, CL_TRUE, 0, dstBuf.size(), dstBuf.data(), 0, NULL, NULL);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueWriteBuffer failed");

        // std::vector<uint8_t> tempBuf(dstBuf.size(), 0);
        // err = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, tempBuf.size(), tempBuf.data(), 0, NULL, NULL);
        // CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");
        // check_typed_values(data_type, element_count, dstBuf, w_rank);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, sync_tensor_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "sync_tensor::execute_impl");
        auto& stream = instance.get_network().get_stream();
        const bool pass_through_events = false;
        //    (stream.get_queue_type() == QueueTypes::out_of_order) && instance.get_node().is_in_shape_of_subgraph();

        auto w_rank = instance.get_network().get_program()->get_config().subStreamExecConfig.get_rank()[0];
        auto w_size = instance.get_network().get_program()->get_config().get_context_for_tp().size();
        auto start = perf_dump_start();
        if (!pass_through_events) {
            for (auto e : events) {
                e->wait();
            }
        }
        perf_dump_done(start, std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait events"));

        auto sub_mem_mgr = instance.get_network().get_sub_mem_mgr();
        auto id = sub_mem_mgr->get_memory_id(w_rank);
        sub_mem_mgr->set_memory_used(id, w_rank);
        std::vector<uint8_t>& sharedBuf = sub_mem_mgr->sharedHostBuf;

        auto start_1 = perf_dump_start();
        while (true) {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            if (sub_mem_mgr->_use_count[id] == w_size) {
                sub_mem_mgr->_use_count[id] = 0;
                for (size_t i = 0; i < w_size; i++) {
                    sub_mem_mgr->_memorys_table[id][i].flag = false;
                }
            }
            if (sub_mem_mgr->_use_count[id] == 0) {
                break;
            }
        }
        perf_dump_done(start_1,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor wait data ready"));

        auto& ocl_stream = downcast<ocl::ocl_stream>(stream);
        auto queue = ocl_stream.get_cl_queue().get();

        int dst_rank = (w_rank + 1) % w_size;
        auto src_cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(w_rank))->get_buffer().get();
        auto dst_cl_buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(instance.output_memory_ptr(dst_rank))->get_buffer().get();

        size_t dst_size = instance.output_memory(dst_rank).size();
        std::vector<uint8_t> srcBuf(dst_size, 0);

        size_t element_count = dst_size;
        const ov::element::Type_t& data_type = instance.output_memory(dst_rank).get_layout().data_type;
        auto bitwidth = ov::element::Type(data_type).bitwidth();

        switch (data_type) {
        case ov::element::f16:
            std::cout << "f16 bits: " << bitwidth << std::endl;
            element_count /= 2;
            break;
        case ov::element::i8:
            std::cout << "i8 bits: " << bitwidth << std::endl;
            break;
        case ov::element::f32:
            std::cout << "f32 bits: " << bitwidth << std::endl;
            element_count /= 4;
            break;
        default:
            std::cout << "Unknow data type, exiting..." << std::endl;
            exit(-1);
        }

        std::cout << "[sync_tensor: " << w_rank << "] dst_size: " << dst_size << ", element_count: " << element_count << std::endl;
        read_cl_buf(queue, data_type, element_count, src_cl_buf, srcBuf, w_rank);

        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            if (sharedBuf.size() != dst_size) {
                sharedBuf.resize(dst_size);
            }
            // reduce
            std::cout << "[sync_tensor: " << w_rank << "] self reduce "<< std::endl;
            switch (data_type) {
            case ov::element::f16: {
                uint16_t* dst_ptr = reinterpret_cast<uint16_t*>(sharedBuf.data());
                uint16_t* src_ptr = reinterpret_cast<uint16_t*>(srcBuf.data());
                for (size_t i = 0; i < element_count; ++i) {
                    ov::float16 f = ov::float16::from_bits(dst_ptr[i]);
                    std::cout << " [" << i << "] f = " << f << std::endl;
                    f += ov::float16::from_bits(src_ptr[i]);
                    std::cout << " [" << i << "] f += " << f << std::endl;
                    dst_ptr[i] = f.to_bits();
                }
                break;
            }
            case ov::element::i8: {
                int8_t* dst_ptr = reinterpret_cast<int8_t*>(sharedBuf.data());
                int8_t* src_ptr = reinterpret_cast<int8_t*>(srcBuf.data());
                for (size_t i = 0; i < element_count; ++i) {
                    dst_ptr[i] += src_ptr[i];
                    std::cout << " [" << i << "] src: " << src_ptr[i] << ", dst_ptr: " << dst_ptr[i] << std::endl;
                }
                break;
            }
            case ov::element::f32: {
                float* dst_ptr = reinterpret_cast<float*>(sharedBuf.data());
                float* src_ptr = reinterpret_cast<float*>(srcBuf.data());
                for (size_t i = 0; i < element_count; ++i) {
                    dst_ptr[i] += src_ptr[i];
                    std::cout << " [" << i << "] src: " << src_ptr[i] << ", dst_ptr: " << dst_ptr[i] << std::endl;
                }
                break;
            }
            default:
                break;
            }

            sub_mem_mgr->_memorys_table[id][w_rank].flag = true;
        }

        std::vector<int> wait_list(w_size, 1);
        auto start_2 = perf_dump_start();
        wait_list[w_rank] = 0;  // no need to wait for itself
        size_t data_size = 0;
        event::ptr sync_event = nullptr;
        std::cout << "[sync_tensor: " << w_rank << "] wait flag... "<< std::endl;

        while (true) {
            int wait_size = 0;
            for (int idx = 0; idx < static_cast<int>(w_size); idx++) {
                if (idx != w_rank && wait_list[idx] > 0 && sub_mem_mgr->_memorys_table[id][idx].flag) {
                    wait_list[idx] = 0;
                }
                wait_size += wait_list[idx];
            }
            if (wait_size == 0) {
                // data on all ranks are ready
                if (instance.get_impl_params()->need_add) {
                    // all_reduce
                    do_self_rank_all_reduce(queue, data_type, element_count, dst_cl_buf, sharedBuf, w_rank);
                } else {
                    // all_gather
                }

                break;
            }
        }
        perf_dump_done(start_2,
                       std::string("rank[") + std::to_string(w_rank) + std::string("] sync_tensor p2p write ") +
                           std::to_string(data_size) + " bytes",
                       false);

        std::cout << "[sync_tensor: " << w_rank << "] need_add: " << instance.get_impl_params()->need_add << std::endl;

        {
            std::lock_guard<std::mutex> lock(sub_mem_mgr->_flagMutex);
            sub_mem_mgr->_use_count[id]++;
        }
        if (pass_through_events) {
            if (events.size() > 1) {
                return stream.group_events(events);
            } else if (events.size() == 1) {
                return events[0];
            }
        }
        return stream.create_user_event(true);
        // return sync_events.size() > 0 ? stream.group_events(sync_events) : stream.create_user_event(true);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const sync_tensor_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<sync_tensor_impl>();
    }
};

namespace detail {

attach_sync_tensor_impl::attach_sync_tensor_impl() {
    implementation_map<sync_tensor>::add(impl_types::cpu, shape_types::dynamic_shape, sync_tensor_impl::create, {});
    implementation_map<sync_tensor>::add(impl_types::cpu, shape_types::static_shape, sync_tensor_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::sync_tensor_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sync_tensor)
