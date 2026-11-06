// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include <iostream>
#include <fstream>
#include <mutex>

namespace ov::intel_gpu {
std::mutex& get_verbose_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::ostream& get_verbose_stream() {
#ifdef GPU_DEBUG_CONFIG
    if (!ExecutionConfig::get_log_to_file().empty()) {
        static std::ofstream fout;
        if (!fout.is_open()) {
            fout.open(ExecutionConfig::get_log_to_file());
        }
        return fout;
    }
    return std::cout;
#else
    return std::cout;
#endif
}
}  // namespace ov::intel_gpu
