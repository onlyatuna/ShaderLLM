#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>

// 定義函數指針型別
typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkCooperativeMatrixPropertiesKHR* pProperties);

int main() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    // 雖然只是查詢，但最好在 Instance 層級宣告可能用到的擴充 (某些驅動要求)
    VkInstance instance;
    vkCreateInstance(&createInfo, nullptr, &instance);

    // 手動獲取擴充函數位址
    auto vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = 
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        std::cout << "\n========== " << props.deviceName << " ==========" << std::endl;

        if (!vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
            std::cout << "Function pointer vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR NOT found in loader." << std::endl;
            continue;
        }

        uint32_t coopTypeCount = 0;
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &coopTypeCount, nullptr);
        
        if (coopTypeCount == 0) {
            std::cout << "Cooperative Matrix KHR NOT supported by hardware/driver." << std::endl;
            continue;
        }

        std::vector<VkCooperativeMatrixPropertiesKHR> coopTypes(coopTypeCount);
        for(auto& t : coopTypes) {
            t.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        }
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &coopTypeCount, coopTypes.data());

        std::cout << "Supported Cooperative Matrix Types: " << coopTypeCount << std::endl;
        for (const auto& t : coopTypes) {
            std::cout << " - Matrix [M:" << t.MSize << ", N:" << t.NSize << ", K:" << t.KSize << "] "
                      << "A:" << t.AType << " B:" << t.BType << " C:" << t.CType << " Res:" << t.ResultType 
                      << " (Scope:" << t.scope << ")" << std::endl;
        }
    }

    vkDestroyInstance(instance, nullptr);
    return 0;
}
