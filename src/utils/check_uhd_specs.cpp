#include <iostream>
#include <vector>
#include <string>
#include <vulkan/vulkan.h>

#define VK_CHECK(res) if(res != VK_SUCCESS) { std::cerr << "Vulkan Error: " << res << std::endl; return; }

void check_uhd_specs() {
    VkInstance instance;
    VkInstanceCreateInfo inst_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    vkCreateInstance(&inst_info, nullptr, &instance);

    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    vkEnumeratePhysicalDevices(instance, &gpu_count, gpus.data());

    for (auto& gpu : gpus) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpu, &props);

        // 只檢查 Intel GPU
        if (std::string(props.deviceName).find("Intel") == std::string::npos) continue;

        std::cout << "\n========== UHD 770 HARDWARE DIAGNOSTICS ==========" << std::endl;
        std::cout << "Device Name    : " << props.deviceName << std::endl;
        std::cout << "Driver Version : " << props.driverVersion << std::endl;
        std::cout << "API Version    : " << (props.apiVersion >> 22) << "." << ((props.apiVersion >> 12) & 0x3ff) << std::endl;

        // 1. 檢查 Subgroup 特性 (這對 Intel 性能至關重要)
        VkPhysicalDeviceSubgroupProperties subgroupProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &subgroupProps;
        vkGetPhysicalDeviceProperties2(gpu, &props2);

        std::cout << "\n--- Subgroup Properties ---" << std::endl;
        std::cout << "Subgroup Size : " << subgroupProps.subgroupSize << std::endl;
        std::cout << "Supported Stages : " << ((subgroupProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ? "Compute " : "") << std::endl;
        std::cout << "Supported Ops    : " << ((subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? "Arithmetic " : "") 
                  << ((subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ? "Shuffle " : "") << std::endl;

        // 2. 檢查記憶體限制
        std::cout << "\n--- Memory Limits ---" << std::endl;
        std::cout << "Max Shared Memory Size : " << props.limits.maxComputeSharedMemorySize / 1024 << " KB" << std::endl;
        std::cout << "Max Workgroup Invocations : " << props.limits.maxComputeWorkGroupInvocations << std::endl;
        std::cout << "Storage Buffer Alignment  : " << props.limits.minStorageBufferOffsetAlignment << " bytes" << std::endl;

        // 3. 檢查 FP16 特性支援
        VkPhysicalDeviceShaderFloat16Int8Features f16Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &f16Features;
        vkGetPhysicalDeviceFeatures2(gpu, &features2);

        std::cout << "\n--- Feature Support ---" << std::endl;
        std::cout << "Native Float16 (Arithmetic) : " << (f16Features.shaderFloat16 ? "YES" : "NO") << std::endl;

        // 4. 檢查記憶體類型 (確認是否為 UMA 架構)
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(gpu, &memProps);
        std::cout << "\n--- Memory Architecture ---" << std::endl;
        bool is_uma = true;
        for(uint32_t i=0; i<memProps.memoryHeapCount; i++) {
            if(!(memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) is_uma = false;
            std::cout << "Heap " << i << " Size: " << memProps.memoryHeaps[i].size / (1024*1024) << " MB" << std::endl;
        }
        std::cout << "UMA Detected : " << (is_uma ? "YES" : "NO") << std::endl;
        std::cout << "==================================================\n" << std::endl;
    }

    vkDestroyInstance(instance, nullptr);
}

int main() {
    check_uhd_specs();
    return 0;
}
