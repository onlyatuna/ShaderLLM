#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <iomanip>

/**
 * ShaderLLM: Vulkan 硬體極限偵測工具
 * 
 * 目的：確認 RTX 5060 Ti 是否能單次綁定 5.6GB 的 Storage Buffer。
 */

int main() {
    VkInstance instance;
    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "❌ 無法建立 Vulkan Instance" << std::endl;
        return 1;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        
        std::cout << "\n--- 裝置資訊: " << props.deviceName << " ---" << std::endl;
        
        // 核心指標：maxStorageBufferRange (單一 Buffer 的最大長度)
        VkDeviceSize maxRange = props.limits.maxStorageBufferRange;
        
        std::cout << "🚀 Max Storage Buffer Range: " << std::fixed << std::setprecision(2)
                  << (double)maxRange / (1024.0 * 1024.0 * 1024.0) << " GB (" 
                  << maxRange << " bytes)" << std::endl;

        // 檢查是否能容納 5.6GB
        const VkDeviceSize pleSize = (VkDeviceSize)(5.6 * 1024.0 * 1024.0 * 1024.0);
        if (maxRange >= pleSize) {
            std::cout << "✅ 完美！硬體支援單次綁定 5.6GB PLE Table。" << std::endl;
        } else {
            std::cout << "⚠️ 警告：硬體限制單次綁定為 " << (double)maxRange / (1024.0 * 1024.0 * 1024.0) 
                      << " GB。我們需要切分 PLE Table。" << std::endl;
        }
        
        // 額外檢查：是否支援 FP16
        std::cout << "💎 Max Bound Descriptor Sets: " << props.limits.maxBoundDescriptorSets << std::endl;
        std::cout << "💎 Max Per Stage Storage Buffers: " << props.limits.maxPerStageDescriptorStorageBuffers << std::endl;
    }

    vkDestroyInstance(instance, nullptr);
    return 0;
}
