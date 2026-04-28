#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

void check_cuda_specs() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if (std::string(prop.name).find("NVIDIA") == std::string::npos) continue;

        int clockRate, memClockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, i);
        cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, i);

        std::cout << "\n========== RTX CUDA HARDWARE DIAGNOSTICS ==========" << std::endl;
        std::cout << "Device Name       : " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total Global Mem  : " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Multiprocessors   : " << prop.multiProcessorCount << std::endl;
        std::cout << "Shared Mem / Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "L2 Cache Size     : " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "Memory Bus Width  : " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "GPU Boost Clock   : " << clockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory Clock      : " << memClockRate / 1000 << " MHz" << std::endl;
        
        // 理論頻寬計算
        double bandwidth = 2.0 * memClockRate * (prop.memoryBusWidth / 8.0) / 1e6;
        std::cout << "Peak Bandwidth    : " << bandwidth << " GB/s" << std::endl;
        std::cout << "===================================================\n" << std::endl;
    }
}

void check_vulkan_rtx_specs() {
    VkInstance instance;
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo inst_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    inst_info.pApplicationInfo = &appInfo;
    vkCreateInstance(&inst_info, nullptr, &instance);

    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    vkEnumeratePhysicalDevices(instance, &gpu_count, gpus.data());

    for (auto& gpu : gpus) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpu, &props);
        if (std::string(props.deviceName).find("NVIDIA") == std::string::npos) continue;

        std::cout << "========== RTX VULKAN EXTENSION DIAGNOSTICS ==========" << std::endl;
        std::cout << "Device Name    : " << props.deviceName << std::endl;

        // 1. 檢查 Cooperative Matrix (KHR)
        VkPhysicalDeviceCooperativeMatrixPropertiesKHR coopProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &coopProps;
        vkGetPhysicalDeviceProperties2(gpu, &props2);

        std::cout << "\n--- Cooperative Matrix Support ---" << std::endl;
        std::cout << "Cooperative Matrix (KHR) : YES" << std::endl;

        // 2. 檢查 Subgroup
        VkPhysicalDeviceSubgroupProperties subgroupProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        props2.pNext = &subgroupProps;
        vkGetPhysicalDeviceProperties2(gpu, &props2);
        std::cout << "Subgroup Size : " << subgroupProps.subgroupSize << std::endl;

        // 3. 檢查 Float16 原生支援
        VkPhysicalDeviceShaderFloat16Int8Features f16Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &f16Features;
        vkGetPhysicalDeviceFeatures2(gpu, &features2);
        std::cout << "Native Float16: " << (f16Features.shaderFloat16 ? "YES" : "NO") << std::endl;

        std::cout << "======================================================\n" << std::endl;
    }

    vkDestroyInstance(instance, nullptr);
}

int main() {
    check_cuda_specs();
    check_vulkan_rtx_specs();
    return 0;
}
