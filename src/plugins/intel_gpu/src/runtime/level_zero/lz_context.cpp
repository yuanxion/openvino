// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lz_context.h"

lzContext::lzContext() {
    printf("INFO: Enter %s \n", __FUNCTION__);
}

lzContext::~lzContext() {
    printf("INFO: Enter %s \n", __FUNCTION__);
    ze_result_t result;

    if (kernelTsEvent) {
        result = zeEventDestroy(kernelTsEvent);
        CHECK_ZE_STATUS(result, "zeEventDestroy kernelTsEvent");
        kernelTsEvent = nullptr;
    }

    if (sharedBuf) {
        result = zeMemFree(context, sharedBuf);
        CHECK_ZE_STATUS(result, "zeMemFree sharedBuf");
        sharedBuf = nullptr;
    }

    if (timestampBuffer) {
        result = zeMemFree(context, timestampBuffer);
        CHECK_ZE_STATUS(result, "zeMemFree timestampBuffer");
        timestampBuffer = nullptr;
    }

    if (command_list) {
        result = zeCommandListDestroy(command_list);
        CHECK_ZE_STATUS(result, "zeCommandListDestroy");
        command_list = nullptr;
    }

    if (command_queue) {
        result = zeCommandQueueDestroy(command_queue);
        CHECK_ZE_STATUS(result, "zeCommandQueueDestroy");
        command_queue = nullptr;
    }

    if (context) {
        result = zeContextDestroy(context);
        CHECK_ZE_STATUS(result, "zeContextDestroy");
        context = nullptr;
    }

    pDevice = nullptr;
    pDriver = nullptr;
}


const char* lzContext::kernelSpvFile = nullptr;
const char* lzContext::kernelFuncName = nullptr;
std::vector<char> lzContext::kernelSpvBin;

ze_device_handle_t lzContext::findDevice(ze_driver_handle_t pDriver, ze_device_type_t type, uint32_t devIdx) {
    // get all devices
    uint32_t deviceCount = 0;
    zeDeviceGet(pDriver, &deviceCount, nullptr);

    std::vector<ze_device_handle_t> devices(deviceCount);
    zeDeviceGet(pDriver, &deviceCount, devices.data());

    ze_device_handle_t found = nullptr;

    // for each device, find the first one matching the type
    for (uint32_t device = 0; device < deviceCount; ++device) {
        auto phDevice = devices[device];

        ze_device_properties_t device_properties = {};
        device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        zeDeviceGetProperties(phDevice, &device_properties);
        printf("#### device count = [%d/%d], devcie_name = %s\n", device, deviceCount, device_properties.name);

        if (type == device_properties.type && device == devIdx) {
            found = phDevice;

            ze_driver_properties_t driver_properties = {};
            driver_properties.stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
            zeDriverGetProperties(pDriver, &driver_properties);

            std::cout << "Found " << std::to_string(type) << " device..." << "\n";
            std::cout << "Driver version: " << driver_properties.driverVersion << "\n";

            ze_api_version_t version = {};
            zeDriverGetApiVersion(pDriver, &version);
            std::cout << "API version: " << std::to_string(version) << "\n";
            // std::cout << std::to_string(device_properties) << "\n";

            ze_device_compute_properties_t compute_properties = {};
            compute_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
            zeDeviceGetComputeProperties(phDevice, &compute_properties);
            // std::cout << to_string(compute_properties) << "\n";

            uint32_t memoryCount = 0;
            zeDeviceGetMemoryProperties(phDevice, &memoryCount, nullptr);
            auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
            for (uint32_t mem = 0; mem < memoryCount; ++mem) {
                pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
                pMemoryProperties[mem].pNext = nullptr;
            }
            zeDeviceGetMemoryProperties(phDevice, &memoryCount, pMemoryProperties);
            for (uint32_t mem = 0; mem < memoryCount; ++mem) {
                // std::cout << to_string( pMemoryProperties[ mem ] ) << "\n";
            }
            delete[] pMemoryProperties;

            ze_device_memory_access_properties_t memory_access_properties = {};
            memory_access_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
            zeDeviceGetMemoryAccessProperties(phDevice, &memory_access_properties);
            // std::cout << to_string( memory_access_properties ) << "\n";

            uint32_t cacheCount = 0;
            zeDeviceGetCacheProperties(phDevice, &cacheCount, nullptr);
            auto pCacheProperties = new ze_device_cache_properties_t[cacheCount];
            for (uint32_t cache = 0; cache < cacheCount; ++cache) {
                pCacheProperties[cache].stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
                pCacheProperties[cache].pNext = nullptr;
            }
            zeDeviceGetCacheProperties(phDevice, &cacheCount, pCacheProperties);
            for (uint32_t cache = 0; cache < cacheCount; ++cache) {
                // std::cout << to_string( pCacheProperties[ cache ] ) << "\n";
            }
            delete[] pCacheProperties;

            ze_device_image_properties_t image_properties = {};
            image_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES;
            zeDeviceGetImageProperties(phDevice, &image_properties);
            // std::cout << to_string( image_properties ) << "\n";

            break;
        }
    }

    return found;
}

void lzContext::initTimeStamp() {
    ze_result_t result;
    ze_event_pool_desc_t eventPoolDesc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
    eventPoolDesc.count = 1;
    eventPoolDesc.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    result = zeEventPoolCreate(context, &eventPoolDesc, 1, &pDevice, &eventPool);
    CHECK_ZE_STATUS(result, "zeEventPoolCreate");

    ze_event_desc_t eventDesc = {ZE_STRUCTURE_TYPE_EVENT_DESC};
    eventDesc.index = 0;
    eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    result = zeEventCreate(eventPool, &eventDesc, &kernelTsEvent);
    CHECK_ZE_STATUS(result, "zeEventCreate kernelTsEvent");

    ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    zeMemAllocHost(context, &hostDesc, sizeof(ze_kernel_timestamp_result_t), 1, &timestampBuffer);
    CHECK_ZE_STATUS(result, "zeMemAllocHost");
    memset(timestampBuffer, 0, sizeof(ze_kernel_timestamp_result_t));
}

int lzContext::initZe(int devIdx) {
    printf("[lzContext] initZe enter \n");
    ze_result_t result;
    // size_t size = 0;
    // size_t alignment = 0;
    result = zeInit(0);
    CHECK_ZE_STATUS(result, "zeInit");

    uint32_t driverCount = 0;
    result = zeDriverGet(&driverCount, nullptr);
    CHECK_ZE_STATUS(result, "zeDriverGet");
    printf("INFO: driver count = %d\n", driverCount);

    std::vector<ze_driver_handle_t> drivers(driverCount);
    result = zeDriverGet(&driverCount, drivers.data());
    CHECK_ZE_STATUS(result, "zeDriverGet");

    for (uint32_t driver = 0; driver < driverCount; ++driver) {
        pDriver = drivers[driver];
        pDevice = findDevice(pDriver, type, devIdx);
        if (pDevice) {
            printf("INFO: find device handle = 0x%08lx\n", (uint64_t)pDevice);
            break;
        }
    }

    if (!pDevice) {
        printf("ERROR: cannot find a proper device\n");
        return -1;
    }

    // Create the context
    ze_context_desc_t context_desc = {};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    result = zeContextCreate(pDriver, &context_desc, &context);
    CHECK_ZE_STATUS(result, "zeContextCreate");

    // Create command list
    auto start_cmd = std::chrono::high_resolution_clock::now();

    // ze_command_list_desc_t descriptor_cmdlist = {};
    // descriptor_cmdlist.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    // descriptor_cmdlist.pNext = nullptr;
    // descriptor_cmdlist.flags = 0;
    // descriptor_cmdlist.commandQueueGroupOrdinal = 0;
    // result = zeCommandListCreate(context, pDevice, &descriptor_cmdlist, &command_list);
    // CHECK_ZE_STATUS(result, "zeCommandListCreate");

    // Create command queue
    ze_command_queue_desc_t descriptor_cmdqueue = {};
    descriptor_cmdqueue.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    descriptor_cmdqueue.pNext = nullptr;
    descriptor_cmdqueue.flags = 0;
    descriptor_cmdqueue.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    descriptor_cmdqueue.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    descriptor_cmdqueue.ordinal = 0;
    descriptor_cmdqueue.index = 0;

    // create Immediate command_list instead
    result = zeCommandListCreateImmediate(context, pDevice, &descriptor_cmdqueue, &command_list);
    CHECK_ZE_STATUS(result, "zeCommandListCreate");
    auto end_cmd_list = std::chrono::high_resolution_clock::now();

    result = zeCommandQueueCreate(context, pDevice, &descriptor_cmdqueue, &command_queue);
    CHECK_ZE_STATUS(result, "zeCommandQueueCreate");
    auto end_cmd_queue = std::chrono::high_resolution_clock::now();

    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    result = zeDeviceGetProperties(pDevice, &properties);
    CHECK_ZE_STATUS(result, "zeDeviceGetProperties");
    deviceProperties = properties;

    int64_t ts_cmd_list = std::chrono::duration_cast<std::chrono::microseconds>(end_cmd_list - start_cmd).count();
    int64_t ts_cmd_queue = std::chrono::duration_cast<std::chrono::microseconds>(end_cmd_queue - end_cmd_list).count();
    printf("[runKernel] ts_cmd_list: %ld us\n", ts_cmd_list);
    printf("[runKernel] ts_cmd_queue: %ld us\n", ts_cmd_queue);

    initTimeStamp();

    printf("[lzContext] initZe done \n");
    return 0;
}

void* lzContext::createBuffer(size_t elemCount, int offset) {
    ze_result_t result;
    void *devBuf = nullptr;

    std::vector<uint32_t> hostBuf(elemCount, 0);
    for (size_t i = 0; i < elemCount; i++)
        hostBuf[i] = offset + (i % 1024);

    ze_device_mem_alloc_desc_t device_desc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        nullptr,
        0,
        0};
    result = zeMemAllocDevice(context, &device_desc, elemCount * sizeof(uint32_t), 1, pDevice, &devBuf);
    CHECK_ZE_STATUS(result, "zeMemAllocDevice");

    result = zeCommandListAppendMemoryCopy(command_list, devBuf, hostBuf.data(), elemCount * sizeof(uint32_t), nullptr, 0, nullptr);
    CHECK_ZE_STATUS(result, "zeCommandListAppendMemoryCopy");

    result = zeCommandListAppendBarrier(command_list, nullptr, 0, nullptr);
    CHECK_ZE_STATUS(result, "zeCommandListAppendBarrier");

    result = zeCommandListClose(command_list);
    CHECK_ZE_STATUS(result, "zeCommandListClose");

    result = zeCommandQueueExecuteCommandLists(command_queue, 1, &command_list, nullptr);
    CHECK_ZE_STATUS(result, "zeCommandQueueExecuteCommandLists");

    result = zeCommandQueueSynchronize(command_queue, UINT64_MAX);
    CHECK_ZE_STATUS(result, "zeCommandQueueSynchronize");

    result = zeCommandListReset(command_list);
    CHECK_ZE_STATUS(result, "zeCommandListReset");

    return devBuf;
}

void lzContext::readBuffer(std::vector<uint32_t> &hostDst, void *devSrc, size_t size) {
    ze_result_t result;
    result = zeCommandListAppendMemoryCopy(command_list, hostDst.data(), devSrc, size, nullptr, 0, nullptr);
    CHECK_ZE_STATUS(result, "zeCommandListAppendMemoryCopy");
    result = zeCommandListClose(command_list);
    CHECK_ZE_STATUS(result, "zeCommandListClose");
    result = zeCommandQueueExecuteCommandLists(command_queue, 1, &command_list, nullptr);
    CHECK_ZE_STATUS(result, "zeCommandQueueExecuteCommandLists");
    result = zeCommandQueueSynchronize(command_queue, UINT64_MAX);
    CHECK_ZE_STATUS(result, "zeCommandQueueSynchronize");
    result = zeCommandListReset(command_list);
    CHECK_ZE_STATUS(result, "zeCommandListReset");
}

void lzContext::writeBuffer(std::vector<uint32_t> hostSrc, void *devDst, size_t size) {
    ze_result_t result;

    result = zeCommandListAppendMemoryCopy(command_list, devDst, hostSrc.data(), size, nullptr, 0, nullptr);
    CHECK_ZE_STATUS(result, "zeCommandListAppendMemoryCopy");

    result = zeCommandListClose(command_list);
    CHECK_ZE_STATUS(result, "zeCommandListClose");

    result = zeCommandQueueExecuteCommandLists(command_queue, 1, &command_list, nullptr);
    CHECK_ZE_STATUS(result, "zeCommandQueueExecuteCommandLists");

    result = zeCommandQueueSynchronize(command_queue, UINT64_MAX);
    CHECK_ZE_STATUS(result, "zeCommandQueueSynchronize");

    result = zeCommandListReset(command_list);
    CHECK_ZE_STATUS(result, "zeCommandListReset");
}

void queryP2P(ze_device_handle_t dev0, ze_device_handle_t dev1) {
    ze_result_t result;
    ze_device_p2p_properties_t p2pProperties = {};
    p2pProperties.stype = ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES;
    p2pProperties.pNext = nullptr;
    p2pProperties.flags = 0;
    result = zeDeviceGetP2PProperties(dev0, dev1, &p2pProperties);

    printf("%s, dev0 = %p, dev1 = %p, flags = %d, result = %d\n", __FUNCTION__, dev0, dev1, p2pProperties.flags, result);
}

void lzContext::readKernel(const char *spvFile, const char *funcName) {
    printf("[lzContext] %s enter \n", __FUNCTION__);

    kernelSpvFile = const_cast<char*>(spvFile);
    kernelFuncName = const_cast<char*>(funcName);
    // static std::vector<char> kernelSpvBin;
    static bool initialized = false;
    printf("[lzContext] kernelFuncName: %s\n", kernelFuncName);

    if (!initialized) {
        FILE *fp = nullptr;
        size_t nsize = 0;
        fp = fopen(kernelSpvFile, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            nsize = (size_t)ftell(fp);
            fseek(fp, 0, SEEK_SET);

            kernelSpvBin.resize(nsize + 1);
            memset(kernelSpvBin.data(), 0, kernelSpvBin.size());
            fread(kernelSpvBin.data(), sizeof(unsigned char), nsize, fp);

            fclose(fp);
        } else {
            printf("ERROR: cannot open kernel spv file %s\n", kernelSpvFile);
            exit(1);
        }

        initialized = true;
    }

    printf("[lzContext] %s done \n", __FUNCTION__);
}

int lzContext::initKernel() {
    auto start_init = std::chrono::high_resolution_clock::now();

    ze_result_t result;

    if (module == nullptr) {
        // Create module
        ze_module_desc_t module_desc = {};
        module_desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
        module_desc.pNext = nullptr;
        module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        module_desc.inputSize = static_cast<uint32_t>(kernelSpvBin.size());
        module_desc.pInputModule = reinterpret_cast<const uint8_t *>(kernelSpvBin.data());
        module_desc.pBuildFlags = nullptr;
        result = zeModuleCreate(context, pDevice, &module_desc, &module, nullptr);
        CHECK_ZE_STATUS(result, "zeModuleCreate");
    }
    auto end_create_module = std::chrono::high_resolution_clock::now();

    // Create kernel
    ze_kernel_desc_t function_desc = {};
    function_desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    function_desc.pNext = nullptr;
    function_desc.flags = 0;
    function_desc.pKernelName = kernelFuncName;
    result = zeKernelCreate(module, &function_desc, &function);
    CHECK_ZE_STATUS(result, "zeKernelCreate");

    auto end_create_kernel = std::chrono::high_resolution_clock::now();

    int64_t ts_create_module = std::chrono::duration_cast<std::chrono::microseconds>(end_create_module - start_init).count();
    int64_t ts_create_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_create_kernel - end_create_module).count();
    printf("[runKernel] ts_create_module: %ld us\n", ts_create_module);
    printf("[runKernel] ts_create_kernel: %ld us\n", ts_create_kernel);

    return 0;
}

void lzContext::runKernel(const char *spvFile, const char *funcName, void *remoteBuf, void *devBuf, const size_t elemCount,
    const int srcOffsetX, const int srcOffsetY, const int strideX, const int strideY, const int width) {
    auto start = std::chrono::high_resolution_clock::now();
    ze_result_t result;
    const int size = elemCount / 4; // use int instead of uint8_t
    printf("[lz_kernel] elemCount: %ld, size: %d \n", elemCount, size);
    if (size < 1) return;

    // kernelSpvFile = spvFile;
    // kernelFuncName = funcName;

    // initKernel();
    auto end_pre = std::chrono::high_resolution_clock::now();

    // set kernel arguments
    result = zeKernelSetArgumentValue(function, 0, sizeof(devBuf), &devBuf);
    CHECK_ZE_STATUS(result, "zeKernelSetArgumentValue");
    auto end_set_arg0 = std::chrono::high_resolution_clock::now();

    result = zeKernelSetArgumentValue(function, 1, sizeof(remoteBuf), &remoteBuf);
    CHECK_ZE_STATUS(result, "zeKernelSetArgumentValue");
    auto end_set_arg1 = std::chrono::high_resolution_clock::now();

    result = zeKernelSetArgumentValue(function, 2, sizeof(int), &srcOffsetX);
    CHECK_ZE_STATUS(result, "zeKernelSetArgumentValue");
    auto end_set_arg2 = std::chrono::high_resolution_clock::now();

    result = zeKernelSetArgumentValue(function, 3, sizeof(int), &srcOffsetY);
    CHECK_ZE_STATUS(result, "zeKernelSetArgumentValue");
    auto end_set_arg3 = std::chrono::high_resolution_clock::now();

    result = zeKernelSetArgumentValue(function, 4, sizeof(int), &strideX);
    CHECK_ZE_STATUS(result, "zeKernelSetArgumentValue");
    auto end_set_arg4 = std::chrono::high_resolution_clock::now();

    result = zeKernelSetArgumentValue(function, 5, sizeof(int), &strideY);
    CHECK_ZE_STATUS(result, "zeKernelSetArgumentValue");
    auto end_set_arg5 = std::chrono::high_resolution_clock::now();

    result = zeKernelSetArgumentValue(function, 6, sizeof(int), &width);
    CHECK_ZE_STATUS(result, "zeKernelSetArgumentValue");
    auto end_set_arg6 = std::chrono::high_resolution_clock::now();

    result = zeKernelSetArgumentValue(function, 7, sizeof(int), &size);
    CHECK_ZE_STATUS(result, "zeKernelSetArgumentValue");
    auto end_set_arg7 = std::chrono::high_resolution_clock::now();

    uint32_t groupSize = 256;
    zeKernelSetGroupSize(function, groupSize, 1, 1);
    auto end_set_grp_size = std::chrono::high_resolution_clock::now();

    // ze_group_count_t groupCount = {static_cast<uint32_t>(elemCount), 1, 1};
    // ze_group_count_t groupCount = {static_cast<uint32_t>(elemCount + groupSize - 1) / groupSize, 1, 1};
    ze_group_count_t groupCount = {static_cast<uint32_t>(size + groupSize - 1) / groupSize, 1, 1};
    result = zeCommandListAppendLaunchKernel(command_list, function, &groupCount, kernelTsEvent, 0, nullptr);
    CHECK_ZE_STATUS(result, "zeCommandListAppendLaunchKernel");
    auto end_append_kernel = std::chrono::high_resolution_clock::now();

    // result = zeCommandListAppendBarrier(command_list, nullptr, 0, nullptr);
    // CHECK_ZE_STATUS(result, "zeCommandListAppendBarrier");
    auto end_append_barrier = std::chrono::high_resolution_clock::now();

    // result = zeCommandListAppendQueryKernelTimestamps(command_list, 1u, &kernelTsEvent, timestampBuffer, nullptr, nullptr, 0u, nullptr);
    // CHECK_ZE_STATUS(result, "zeCommandListAppendQueryKernelTimestamps");
    auto end_append_ts = std::chrono::high_resolution_clock::now();

    // result = zeCommandListClose(command_list);
    // CHECK_ZE_STATUS(result, "zeCommandListClose");
    auto end_list_close = std::chrono::high_resolution_clock::now();

    // result = zeCommandQueueExecuteCommandLists(command_queue, 1, &command_list, nullptr);
    // CHECK_ZE_STATUS(result, "zeCommandQueueExecuteCommandLists");
    auto end_queue_exec = std::chrono::high_resolution_clock::now();

    zeEventHostSynchronize(kernelTsEvent, UINT64_MAX);
    zeEventHostReset(kernelTsEvent);

    // result = zeCommandQueueSynchronize(command_queue, UINT64_MAX);
    // CHECK_ZE_STATUS(result, "zeCommandQueueSynchronize");
    auto end_queue_sync = std::chrono::high_resolution_clock::now();

    // result = zeCommandListReset(command_list);
    // CHECK_ZE_STATUS(result, "zeCommandListReset");
    auto end_list_reset = std::chrono::high_resolution_clock::now();

    ze_kernel_timestamp_result_t *kernelTsResults = reinterpret_cast<ze_kernel_timestamp_result_t *>(timestampBuffer);
    uint64_t timerResolution = deviceProperties.timerResolution;
    uint64_t kernelDuration = kernelTsResults->context.kernelEnd - kernelTsResults->context.kernelStart;

    std::cout << "Kernel timestamp statistics (prior to V1.2): \n"
              << std::fixed
              << "\tGlobal start : " << std::dec << kernelTsResults->global.kernelStart << " cycles\n"
              << "\tKernel start: " << std::dec << kernelTsResults->context.kernelStart << " cycles\n"
              << "\tKernel end: " << std::dec << kernelTsResults->context.kernelEnd << " cycles\n"
              << "\tGlobal end: " << std::dec << kernelTsResults->global.kernelEnd << " cycles\n"
              << "\ttimerResolution: " << std::dec << timerResolution << " ns\n"
              << "\tKernel duration : " << std::dec << kernelDuration << " cycles\n"
              << "\tKernel Time: " << kernelDuration * timerResolution / 1000.0 << " us\n";

    double gpuKernelTime = kernelDuration * timerResolution / 1000.0;
    // double bandWidth = elemCount * sizeof(uint32_t) / (gpuKernelTime / 1e6) / 1e9;
    double bandWidth = size * sizeof(uint32_t) / (gpuKernelTime / 1e6) / 1e9;
    printf("#### gpuKernelTime = %f, elemCount = %ld, Bandwidth = %f GB/s\n", gpuKernelTime, elemCount, bandWidth);

    auto end_statistics = std::chrono::high_resolution_clock::now();

    int64_t ts_pre = std::chrono::duration_cast<std::chrono::microseconds>(end_pre - start).count();
    int64_t ts_arg0 = std::chrono::duration_cast<std::chrono::microseconds>(end_set_arg0 - end_pre).count();
    int64_t ts_arg1 = std::chrono::duration_cast<std::chrono::microseconds>(end_set_arg1 - end_set_arg0).count();
    int64_t ts_arg2 = std::chrono::duration_cast<std::chrono::microseconds>(end_set_arg2 - end_set_arg1).count();
    int64_t ts_arg3 = std::chrono::duration_cast<std::chrono::microseconds>(end_set_arg3 - end_set_arg2).count();
    int64_t ts_arg4 = std::chrono::duration_cast<std::chrono::microseconds>(end_set_arg4 - end_set_arg3).count();
    int64_t ts_arg5 = std::chrono::duration_cast<std::chrono::microseconds>(end_set_arg5 - end_set_arg4).count();
    int64_t ts_arg6 = std::chrono::duration_cast<std::chrono::microseconds>(end_set_arg6 - end_set_arg5).count();
    int64_t ts_arg7 = std::chrono::duration_cast<std::chrono::microseconds>(end_set_arg7 - end_set_arg6).count();
    int64_t ts_grp_size = std::chrono::duration_cast<std::chrono::microseconds>(end_set_grp_size - end_set_arg7).count();
    int64_t ts_append_kernel = std::chrono::duration_cast<std::chrono::microseconds>(end_append_kernel - end_set_grp_size).count();
    int64_t ts_append_barrier = std::chrono::duration_cast<std::chrono::microseconds>(end_append_barrier - end_append_kernel).count();
    int64_t ts_append_ts = std::chrono::duration_cast<std::chrono::microseconds>(end_append_ts - end_append_barrier).count();
    int64_t ts_list_close = std::chrono::duration_cast<std::chrono::microseconds>(end_list_close - end_append_ts).count();
    int64_t ts_queue_exec = std::chrono::duration_cast<std::chrono::microseconds>(end_queue_exec - end_list_close).count();
    int64_t ts_queue_sync = std::chrono::duration_cast<std::chrono::microseconds>(end_queue_sync - end_queue_exec).count();
    int64_t ts_list_reset = std::chrono::duration_cast<std::chrono::microseconds>(end_list_reset - end_queue_sync).count();
    int64_t ts_statistics = std::chrono::duration_cast<std::chrono::microseconds>(end_statistics - end_list_reset).count();

    printf("[runKernel] ts_pre: %ld us\n", ts_pre);
    printf("[runKernel] ts_arg0: %ld us\n", ts_arg0);
    printf("[runKernel] ts_arg1: %ld us\n", ts_arg1);
    printf("[runKernel] ts_arg2: %ld us\n", ts_arg2);
    printf("[runKernel] ts_arg3: %ld us\n", ts_arg3);
    printf("[runKernel] ts_arg4: %ld us\n", ts_arg4);
    printf("[runKernel] ts_arg5: %ld us\n", ts_arg5);
    printf("[runKernel] ts_arg6: %ld us\n", ts_arg6);
    printf("[runKernel] ts_arg7: %ld us\n", ts_arg7);
    printf("[runKernel] ts_grp_size: %ld us\n", ts_grp_size);
    printf("[runKernel] ts_append_kernel: %ld us\n", ts_append_kernel);
    printf("[runKernel] ts_append_barrier: %ld us\n", ts_append_barrier);
    printf("[runKernel] ts_append_ts: %ld us\n", ts_append_ts);
    printf("[runKernel] ts_list_close: %ld us\n", ts_list_close);
    printf("[runKernel] ts_queue_exec: %ld us\n", ts_queue_exec);
    printf("[runKernel] ts_queue_sync: %ld us\n", ts_queue_sync);
    printf("[runKernel] ts_list_reset: %ld us\n", ts_list_reset);
    printf("[runKernel] ts_statistics: %ld us\n", ts_statistics);
}

void *lzContext::createFromHandle(uint64_t handle, size_t bufSize) {
    ze_result_t result;
    printf("handle = %ld, bufSize = %ld\n", handle, bufSize);

    ze_external_memory_import_fd_t import_fd = {};
    import_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    import_fd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    import_fd.fd = handle;
    ze_device_mem_alloc_desc_t alloc_desc = {};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.pNext = &import_fd;

    // void *sharedBuf = nullptr;
    if (sharedBuf) {
        result = zeMemFree(context, sharedBuf);
        CHECK_ZE_STATUS(result, "zeMemFree sharedBuf");
        sharedBuf = nullptr;
    }

    result = zeMemAllocDevice(context, &alloc_desc, bufSize, 1, pDevice, &sharedBuf);
    CHECK_ZE_STATUS(result, "zeMemAllocDevice");

    ze_memory_allocation_properties_t props = {};
    result = zeMemGetAllocProperties(context, sharedBuf, &props, nullptr);
    CHECK_ZE_STATUS(result, "zeMemGetAllocProperties");
    printf("MemAllocINFO: memory = %p, stype = %d, pNext = 0x%08lx, type = %d, id = 0x%08lx, pagesize = %ld\n",
           sharedBuf, props.stype, (uint64_t)props.pNext, props.type, props.id, props.pageSize);

    return sharedBuf;
}

void lzContext::printBuffer(void *ptr, size_t count) {
    std::vector<uint32_t> outBuf(count, 0);
    readBuffer(outBuf, ptr, count*sizeof(uint32_t));
    printf("The first %ld elements in level-zero ptr = %p are: \n", count, ptr);
    for (size_t i = 0; i < count; i++) {
        printf("%d, ", outBuf[i]);
        if (i && i % 16 == 0)
            printf("\n");
    }
    printf("\n");
}
