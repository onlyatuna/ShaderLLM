#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <level_zero/ze_api.h>

#define ZE_CHECK(res) if(res != ZE_RESULT_SUCCESS) { std::cerr << "L0 Error: 0x" << std::hex << (int)res << std::dec << " at line " << __LINE__ << std::endl; exit(1); }

int main() {
    std::cout << "========== CHIMERA-V M3: Intel Level Zero Extreme Benchmark ==========" << std::endl;

    // 1. 初始化
    ZE_CHECK(zeInit(0));
    ze_driver_handle_t hDriver = nullptr;
    uint32_t driverCount = 1;
    ZE_CHECK(zeDriverGet(&driverCount, &hDriver));

    ze_device_handle_t hDevice = nullptr;
    uint32_t deviceCount = 1;
    ZE_CHECK(zeDeviceGet(hDriver, &deviceCount, &hDevice));

    ze_context_desc_t contextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ze_context_handle_t hContext;
    ZE_CHECK(zeContextCreate(hDriver, &contextDesc, &hContext));

    // 2. 準備記憶體 (使用 FP32 確保相容性)
    const uint32_t W = 512, H = 64, C = 2560;
    size_t stateSize = (size_t)W * H * C * sizeof(float);
    size_t weightSize = (size_t)C * C * sizeof(float);

    ze_device_mem_alloc_desc_t deviceDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    
    void *d_input, *d_output, *d_weights;
    ZE_CHECK(zeMemAllocShared(hContext, &deviceDesc, &hostDesc, stateSize, 1, hDevice, &d_input));
    ZE_CHECK(zeMemAllocShared(hContext, &deviceDesc, &hostDesc, stateSize, 1, hDevice, &d_output));
    ZE_CHECK(zeMemAllocShared(hContext, &deviceDesc, &hostDesc, weightSize, 1, hDevice, &d_weights));

    // 初始化數據 (FP32 1.0f 與 0.5f)
    std::fill((float*)d_input, (float*)d_input + (stateSize/4), 1.0f);
    std::fill((float*)d_weights, (float*)d_weights + (weightSize/4), 0.5f);
    std::memset(d_output, 0, stateSize);

    // 3. 載入專屬 L0 SPIR-V Module
    std::ifstream f("src/shaders/nca_l0.spv", std::ios::binary);
    std::vector<char> spirv((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = spirv.size();
    moduleDesc.pInputModule = (const uint8_t*)spirv.data();
    
    ze_module_handle_t hModule;
    ZE_CHECK(zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, nullptr));

    ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
    kernelDesc.pKernelName = "main";
    ze_kernel_handle_t hKernel;
    ZE_CHECK(zeKernelCreate(hModule, &kernelDesc, &hKernel));

    // 4. 設定參數
    struct Config { uint32_t w, h, c; float dt; uint32_t row; } cfg = {W, H, C, 0.01f, 0};
    ZE_CHECK(zeKernelSetArgumentValue(hKernel, 0, sizeof(void*), &d_input));
    ZE_CHECK(zeKernelSetArgumentValue(hKernel, 1, sizeof(void*), &d_output));
    ZE_CHECK(zeKernelSetArgumentValue(hKernel, 2, sizeof(void*), &d_weights));
    // Push Constant 在 L0 也是作為 Argument 傳遞 (通常是最後一個或特定 Binding)
    // 這裡我們針對之前的 Shader 佈局進行映射
    ZE_CHECK(zeKernelSetArgumentValue(hKernel, 3, sizeof(Config), &cfg));

    // 5. 建立命令隊列 (立即執行模式)
    ze_command_queue_desc_t queueDesc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
    ze_command_list_handle_t hCmdList;
    ZE_CHECK(zeCommandListCreateImmediate(hContext, hDevice, &queueDesc, &hCmdList));

    // 6. 執行跑分
    std::cout << "[L0 Benchmark] Starting 1 step execution (USM Mode)..." << std::endl;
    ze_group_count_t launchArgs = {(W * H + 31) / 32, (C + 31) / 32, 1};
    
    auto start = std::chrono::high_resolution_clock::now();
    ZE_CHECK(zeCommandListAppendLaunchKernel(hCmdList, hKernel, &launchArgs, nullptr, 0, nullptr));
    
    // L0 的 Immediate List 不需要 Submit，但需要同步確認完成
    ZE_CHECK(zeCommandListHostSynchronize(hCmdList, UINT64_MAX));
    auto end = std::chrono::high_resolution_clock::now();

    // 7. 數值驗證
    uint16_t* res = (uint16_t*)d_output;
    std::cout << "\n--- L0 Numerical Verification ---" << std::endl;
    std::cout << "[Check] Output[0]: 0x" << std::hex << res[0] << std::dec << std::endl;

    std::chrono::duration<double> diff = end - start;
    double t = diff.count();
    std::cout << "Avg Time: " << t * 1000.0 << " ms | L0 Perf: " << (2.0*W*H*C*C/1e12)/t << " TFLOPS" << std::endl;

    // 清理
    zeMemFree(hContext, d_input);
    zeMemFree(hContext, d_output);
    zeMemFree(hContext, d_weights);
    zeKernelDestroy(hKernel);
    zeModuleDestroy(hModule);
    zeContextDestroy(hContext);

    return 0;
}
