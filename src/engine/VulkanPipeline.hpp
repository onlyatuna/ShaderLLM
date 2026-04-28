#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

/**
 * @brief VulkanPipeline 負責管理 NCA 演化的推論管線。
 * 封裝了 Shader 加載、Descriptor Set 配置與 Pipeline 建立。
 */
class VulkanPipeline {
public:
    struct Config {
        uint32_t width;
        uint32_t height;
        uint32_t channels;
        float dt;
        uint32_t row_offset; // 修正：補上此欄位以匹配 Engine 與 Shader
    };

    VulkanPipeline(VkDevice device, const std::string& shaderPath, uint32_t bindingCount = 3) 
        : device(device), bindingCount(bindingCount) {
        createDescriptorSetLayout();
        createPipeline(shaderPath);
        std::cout << "[VulkanPipeline] Inference pipeline created successfully." << std::endl;
    }

    ~VulkanPipeline() {
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }
    VkPipeline getPipeline() const { return pipeline; }
    VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }

private:
    VkDevice device;
    uint32_t bindingCount;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    void createDescriptorSetLayout() {
        std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
        for (uint32_t i = 0; i < bindingCount; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindingCount;
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }

    void createPipeline(const std::string& shaderPath) {
        auto shaderCode = readFile(shaderPath);
        VkShaderModule shaderModule = createShaderModule(shaderCode);

        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageInfo.module = shaderModule;
        shaderStageInfo.pName = "main";

        // 配置 Push Constants
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(Config);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.stage = shaderStageInfo;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute pipeline!");
        }

        vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        
        if (!file.is_open()) {
            // 🚀 核心修正：路徑感知搜索 (解決 CWD 與 Binary 目錄不一致問題)
            std::vector<std::string> searchPaths = {
                filename,                             // 原路徑
                "build/bin/" + filename,             // CMake 預設輸出
                "bin/" + filename,                   // 統一輸出目錄
                "../" + filename                     // 相對於執行檔 (若在 bin 內執行)
            };

            for (const auto& path : searchPaths) {
                file.open(path, std::ios::ate | std::ios::binary);
                if (file.is_open()) {
                    std::cout << "[VulkanPipeline] Resource located via secondary search: " << path << std::endl;
                    break;
                }
            }
        }

        if (!file.is_open()) {
            throw std::runtime_error("!!! FATAL: Failed to open shader file: " + filename + 
                                   "\n(Search attempted in CWD and build output directories)");
        }
        
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }
        return shaderModule;
    }
};
