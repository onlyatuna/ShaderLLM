#pragma once
#include "VulkanContext.hpp"
#include "VulkanPipeline.hpp"
#include "RetinaState.hpp"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

class VulkanEngine {
public:
    VulkanEngine() {
        context = std::make_unique<VulkanContext>();
        auto info = context->getDeviceInfo();
        std::cout << "[VulkanEngine] Device: " << info.name 
                  << " | Subgroup Size: " << info.subgroupSize << std::endl;
        
        pipeline = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_evolve.spv");
        createDescriptorPool();
        
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(context->getDevice(), &fenceInfo, nullptr, &renderFence);
    }

    ~VulkanEngine() {
        cleanupBuffers();
        vkDestroyFence(context->getDevice(), renderFence, nullptr);
        vkDestroyDescriptorPool(context->getDevice(), descriptorPool, nullptr);
    }

    template <uint32_t W, uint32_t H, uint32_t C>
    void prepareResources(const RetinaState<W, H, C>& state) {
        size_t size = state.size_bytes();
        size_t weight_size = (size_t)C * C * sizeof(uint16_t);
        VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, d_input, d_input_memory);
        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, d_output, d_output_memory);
        createBuffer(weight_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, d_weights, d_weights_memory);

        allocateDescriptorSets();
        updateDescriptorSets();
    }

    void upload(const void* data, size_t size) {
        uploadToBuffer(d_input, d_input_memory, data, size);
    }

    void loadWeights(const std::string& path, size_t size) {
        std::vector<uint16_t> hostWeights(size / sizeof(uint16_t));
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Weights file not found: " + path);
        file.read(reinterpret_cast<char*>(hostWeights.data()), size);
        uploadToBuffer(d_weights, d_weights_memory, hostWeights.data(), size);
    }

    void evolve(uint32_t w, uint32_t h, uint32_t c) {
        const uint32_t slice_size = 1280; 
        uint32_t num_pixels = w * h;

        for (uint32_t row_start = 0; row_start < c; row_start += slice_size) {
            uint32_t current_slice = std::min(slice_size, c - row_start);

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = getCommandPool();
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer;
            vkAllocateCommandBuffers(context->getDevice(), &allocInfo, &commandBuffer);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);

            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = (pingPongIndex == 0 ? d_input : d_output);
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipeline());
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
                                    pipeline->getPipelineLayout(), 0, 1, 
                                    &descriptorSets[pingPongIndex], 0, nullptr);

            struct SliceConfig {
                uint32_t width;
                uint32_t height;
                uint32_t channels;
                float dt;
                uint32_t row_offset; 
            } slice_cfg = {w, h, c, 0.01f, row_start};

            vkCmdPushConstants(commandBuffer, pipeline->getPipelineLayout(), 
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SliceConfig), &slice_cfg);

            // M6.0 Dispatch: 每個 Workgroup (128 threads) 負責計算 64x64 的資料塊
            vkCmdDispatch(commandBuffer, (num_pixels + 63) / 64, (current_slice + 63) / 64, 1);

            vkEndCommandBuffer(commandBuffer);

            vkResetFences(context->getDevice(), 1, &renderFence);
            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            if (vkQueueSubmit(context->getComputeQueue(), 1, &submitInfo, renderFence) != VK_SUCCESS) {
                throw std::runtime_error("Failed to submit compute slice!");
            }

            VkResult res = vkWaitForFences(context->getDevice(), 1, &renderFence, VK_TRUE, 5000000000ULL);
            if (res == VK_TIMEOUT) throw std::runtime_error("Slice TIMEOUT! TDR limit hit.");

            vkFreeCommandBuffers(context->getDevice(), getCommandPool(), 1, &commandBuffer);
        }
        
        pingPongIndex = 1 - pingPongIndex;
    }

private:
    std::unique_ptr<VulkanContext> context;
    std::unique_ptr<VulkanPipeline> pipeline;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSets[2];
    VkFence renderFence;
    uint32_t pingPongIndex = 0;
    VkBuffer d_input, d_output, d_weights;
    VkDeviceMemory d_input_memory, d_output_memory, d_weights_memory;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    void createDescriptorPool() {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 6;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 2;
        vkCreateDescriptorPool(context->getDevice(), &poolInfo, nullptr, &descriptorPool);
    }

    void allocateDescriptorSets() {
        VkDescriptorSetLayout layout = pipeline->getDescriptorSetLayout();
        VkDescriptorSetLayout layouts[2] = {layout, layout};
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 2;
        allocInfo.pSetLayouts = layouts;
        vkAllocateDescriptorSets(context->getDevice(), &allocInfo, descriptorSets);
    }

    void updateDescriptorSets() {
        for (int i = 0; i < 2; ++i) {
            VkDescriptorBufferInfo b0 = { (i == 0 ? d_input : d_output), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo b1 = { (i == 0 ? d_output : d_input), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo b2 = { d_weights, 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet writes[3] = {};
            for (int j = 0; j < 3; ++j) {
                writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[j].dstSet = descriptorSets[i];
                writes[j].dstBinding = j;
                writes[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[j].descriptorCount = 1;
            }
            writes[0].pBufferInfo = &b0;
            writes[1].pBufferInfo = &b1;
            writes[2].pBufferInfo = &b2;
            vkUpdateDescriptorSets(context->getDevice(), 3, writes, 0, nullptr);
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(context->getDevice(), &bufferInfo, nullptr, &buffer);

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(context->getDevice(), buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        vkAllocateMemory(context->getDevice(), &allocInfo, nullptr, &bufferMemory);
        vkBindBufferMemory(context->getDevice(), buffer, bufferMemory, 0);
    }

    void uploadToBuffer(VkBuffer buffer, VkDeviceMemory memory, const void* data, size_t size) {
        void* mapped;
        vkMapMemory(context->getDevice(), memory, 0, size, 0, &mapped);
        memcpy(mapped, data, size);
        vkUnmapMemory(context->getDevice(), memory);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(context->getPhysicalDevice(), &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
        }
        throw std::runtime_error("Failed to find memory type!");
    }

    VkCommandPool getCommandPool() {
        if (commandPool == VK_NULL_HANDLE) {
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = context->getQueueFamilyIndex();
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            vkCreateCommandPool(context->getDevice(), &poolInfo, nullptr, &commandPool);
        }
        return commandPool;
    }

    void cleanupBuffers() {
        vkDestroyBuffer(context->getDevice(), d_input, nullptr);
        vkFreeMemory(context->getDevice(), d_input_memory, nullptr);
        vkDestroyBuffer(context->getDevice(), d_output, nullptr);
        vkFreeMemory(context->getDevice(), d_output_memory, nullptr);
        vkDestroyBuffer(context->getDevice(), d_weights, nullptr);
        vkFreeMemory(context->getDevice(), d_weights_memory, nullptr);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(context->getDevice(), commandPool, nullptr);
    }
};
