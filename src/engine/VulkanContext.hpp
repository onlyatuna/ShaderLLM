#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <optional>
#include <string>

class VulkanContext {
public:
    VulkanContext() {
        createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
    }

    ~VulkanContext() {
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    VkInstance getInstance() const { return instance; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VkDevice getDevice() const { return device; }
    VkQueue getComputeQueue() const { return computeQueue; }
    uint32_t getQueueFamilyIndex() const { return queueFamilyIndex; }

    struct DeviceInfo {
        std::string name;
        uint32_t subgroupSize;
        VkSubgroupFeatureFlags supportedStages;
    };

    DeviceInfo getDeviceInfo() const {
        VkPhysicalDeviceProperties2 props2{};
        VkPhysicalDeviceSubgroupProperties subgroupProps{};
        subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroupProps;
        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
        return { props2.properties.deviceName, subgroupProps.subgroupSize, subgroupProps.supportedStages };
    }

private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t queueFamilyIndex = (uint32_t)-1;

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        vkCreateInstance(&createInfo, nullptr, &instance);
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        const char* pref = std::getenv("CHIMERA_PREFER_INTEL");
        bool preferIntel = (pref && std::string(pref) == "1");
        
        for (const auto& d : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            std::string name = props.deviceName;
            
            if (preferIntel && name.find("Intel") != std::string::npos) {
                physicalDevice = d;
                std::cout << "[VulkanContext] Explicitly picked Intel GPU: " << name << std::endl;
                return;
            }

            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && !preferIntel) {
                physicalDevice = d;
                break;
            }
        }
        if (physicalDevice == VK_NULL_HANDLE) physicalDevice = devices[0];
    }

    void createLogicalDevice() {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::string name = props.deviceName;
        bool isIntel = (name.find("Intel") != std::string::npos);

        uint32_t queueCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queueFamilies.data());

        // 🚀 修正三：純粹非同步計算佇列 (Async Compute Escape)
        // 尋找專屬的計算通道，避開 Windows 桌面視窗管理員 (DWM) 的圖形搶佔
        int fallbackQueueIndex = -1;
        for (uint32_t i = 0; i < queueCount; i++) {
            bool hasCompute = (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT);
            bool hasGraphics = (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT);
            
            if (hasCompute) {
                fallbackQueueIndex = i; // 記住任何支援 Compute 的佇列作為退路
                if (!hasGraphics) {
                    // 找到純粹的 ACE (Async Compute Engine)！這是一條不受圖形干擾的專屬賽道
                    queueFamilyIndex = i;
                    std::cout << "[VulkanContext] M10.9.4: Engaged Dedicated Async Compute Engine (Queue " << i << ")" << std::endl;
                    break;
                }
            }
        }
        
        // 若找不到純計算佇列，則退回通用佇列
        if (queueFamilyIndex == (uint32_t)-1 && fallbackQueueIndex != -1) {
            queueFamilyIndex = fallbackQueueIndex;
            std::cout << "[VulkanContext] Warning: No dedicated ACE found. Using fallback Queue " << queueFamilyIndex << std::endl;
        }

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkPhysicalDeviceShaderFloat16Int8Features f16Features{};
        f16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        f16Features.shaderFloat16 = VK_TRUE;

        // 【新增】16-bit 記憶體存取特性
        VkPhysicalDevice16BitStorageFeatures storage16Features{};
        storage16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        storage16Features.storageBuffer16BitAccess = VK_TRUE;
        f16Features.pNext = &storage16Features;

        std::vector<const char*> extensions = { 
            VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
            VK_KHR_16BIT_STORAGE_EXTENSION_NAME 
        };

        if (!isIntel) {
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures{};
            coopFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
            coopFeatures.cooperativeMatrix = VK_TRUE;
            storage16Features.pNext = &coopFeatures; // 串接到鍊條末端
            extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        }

        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = extensions.data();
        deviceCreateInfo.pNext = &f16Features;

        if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device!");
        }

        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
    }
};
