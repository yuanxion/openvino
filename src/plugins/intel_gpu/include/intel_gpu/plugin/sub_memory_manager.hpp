// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <assert.h>

namespace ov {
namespace intel_gpu {
class SubMemoryManager {
public:
    using ptr = std::shared_ptr<SubMemoryManager>;
    using cptr = const std::shared_ptr<SubMemoryManager>;
    struct MemoryInfo {
        void* send_buf;
        std::shared_ptr<void> buf;
        bool flag;
        bool last_used;
        std::shared_ptr<cldnn::stream> stream_ptr;
        std::vector<cldnn::memory::ptr> recv_bufs;  // todo: avoid mem copy
        // std::vector<uint64_t> remote_fd; // remote handle for map back
        std::vector<cldnn::event::ptr> events;
    };

    SubMemoryManager(int num_sub_streams) {
        assert(num_sub_streams);
         _num_sub_streams = num_sub_streams;
        MemoryInfo memory_info;
        memory_info.flag = false;
        memory_info.last_used = false;
        // memory_info.remote_fd.assign(2, 0);
        memory_info.events.assign(2, nullptr);
        std::vector<MemoryInfo> memorys;
        memorys.assign(_num_sub_streams, memory_info);
        _memorys_table.assign(2, memorys);
        _use_count.assign(2, 0);
    }

    int get_memory_id(int sub_stream_id) {
        for (int i = 0; i < 2; i++) {
            if (!_memorys_table[i][sub_stream_id].last_used) {
                return i;
            }
        }
        return -1;
    }

    void set_memory_used(int memory_id, int sub_stream_id) {
        _memorys_table[memory_id][sub_stream_id].last_used = true;
        _memorys_table[(memory_id + 1) % 2][sub_stream_id].last_used = false;
    }

    int _num_sub_streams;
    std::vector<std::vector<MemoryInfo>> _memorys_table;
    std::vector<size_t> _use_count;
    std::mutex _flagMutex;
    std::vector<uint8_t> sharedHostBuf;
};
}  // namespace intel_gpu
}  // namespace ov
