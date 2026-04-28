#pragma once
#include <vulkan/vulkan.h>
#include "RetinaState.hpp"
#include "VulkanPipeline.hpp"
#include <vector>
#include <memory>
#include <cstring>
#include <fstream>

class NcaEngine {
public:
    NcaEngine(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamilyIndex)
        : device(device), physicalDevice(physicalDevice), queue(queue), queueFamilyIndex(queueFamilyIndex) {
        
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

        pipeline = std::make_unique<VulkanPipeline>(device, "nca_evolve.spv");
        createDescriptorPool();
    }

    ~NcaEngine() {
        cleanupBuffers();
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    template <uint32_t W, uint32_t H, uint32_t C>
    void prepareResources(const RetinaState<W, H, C>& state) {
        size_t size = state.size_bytes();
        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gpuBufferIn, gpuBufferInMemory);
        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gpuBufferOut, gpuBufferOutMemory);
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        size_t weight_size = (size_t)C * C * sizeof(uint16_t);
        createBuffer(weight_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, weightBuffer, weightMemory);

        updateDescriptorSets(size, weight_size);
    }

    void upload(const void* data, size_t size) {
        void* mappedData;
        vkMapMemory(device, stagingBufferMemory, 0, size, 0, &mappedData);
        std::memcpy(mappedData, data, size);
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer(stagingBuffer, gpuBufferIn, size);
    }

    void loadWeights(const std::string& path, size_t size) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open weights file: " + path);

        void* mappedData;
        vkMapMemory(device, stagingBufferMemory, 0, size, 0, &mappedData);
        file.read(static_cast<char*>(mappedData), size);
        vkUnmapMemory(device, stagingBufferMemory);

        copyBuffer(stagingBuffer, weightBuffer, size);
        std::cout << "[NcaEngine] Weights loaded: " << path << std::endl;
    }

    void evolve(uint32_t w, uint32_t h, uint32_t c) {
        VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &allocInfo, &cmd);

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
        vkBeginCommandBuffer(cmd, &beginInfo);
        
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipelineLayout(), 0, 1, &descriptorSet, 0, nullptr);

        VulkanPipeline::Config pushConfig{ w, h, c, 0.1f };
        vkCmdPushConstants(cmd, pipeline->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConfig), &pushConfig);

        vkCmdDispatch(cmd, (w + 15) / 16, (h + 15) / 16, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr };
        vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
        std::swap(gpuBufferIn, gpuBufferOut);
    }

private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkQueue queue;
    uint32_t queueFamilyIndex;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    std::unique_ptr<VulkanPipeline> pipeline;
    VkBuffer gpuBufferIn, gpuBufferOut, stagingBuffer, weightBuffer;
    VkDeviceMemory gpuBufferInMemory, gpuBufferOutMemory, stagingBufferMemory, weightMemory;

    void createDescriptorPool() {
        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &poolSize };
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
    }

    void updateDescriptorSets(VkDeviceSize dataSize, VkDeviceSize weightSize) {
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 1, nullptr };
        VkDescriptorSetLayout layout = pipeline->getDescriptorSetLayout();
        allocInfo.pSetLayouts = &layout;
        vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

        VkDescriptorBufferInfo bInfos[3] = { {gpuBufferIn, 0, dataSize}, {gpuBufferOut, 0, dataSize}, {weightBuffer, 0, weightSize} };
        VkWriteDescriptorSet writes[3] = {};
        for(int i=0; i<3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptorSet;
            writes[i].dstBinding = i;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].descriptorCount = 1;
            writes[i].pBufferInfo = &bInfos[i];
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
        VkBufferCreateInfo bInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr };
        vkCreateBuffer(device, &bInfo, nullptr, &buf);
        VkMemoryRequirements mReqs;
        vkGetBufferMemoryRequirements(device, buf, &mReqs);
        VkMemoryAllocateInfo aInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mReqs.size, findMemoryType(mReqs.memoryTypeBits, props) };
        vkAllocateMemory(device, &aInfo, nullptr, &mem);
        vkBindBufferMemory(device, buf, mem, 0);
    }

    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBufferAllocateInfo aInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &aInfo, &cmd);
        VkCommandBufferBeginInfo bInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr };
        vkBeginCommandBuffer(cmd, &bInfo);
        VkBufferCopy region{ 0, 0, size };
        vkCmdCopyBuffer(cmd, src, dst, 1, &region);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr };
        vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mProps);
        for (uint32_t i=0; i<mProps.memoryTypeCount; i++) {
            if ((filter & (1 << i)) && (mProps.memoryTypes[i].propertyFlags & props) == props) return i;
        }
        return 0;
    }

    void cleanupBuffers() {
        vkDestroyBuffer(device, gpuBufferIn, nullptr); vkFreeMemory(device, gpuBufferInMemory, nullptr);
        vkDestroyBuffer(device, gpuBufferOut, nullptr); vkFreeMemory(device, gpuBufferOutMemory, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr); vkFreeMemory(device, stagingBufferMemory, nullptr);
        vkDestroyBuffer(device, weightBuffer, nullptr); vkFreeMemory(device, weightMemory, nullptr);
    }
};
