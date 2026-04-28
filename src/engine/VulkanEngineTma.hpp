#pragma once
#include "VulkanContext.hpp"
#include "VulkanPipeline.hpp"
#include "RetinaState.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <thread> // 🚀 修正二：呼吸機制

/**
 * @brief VulkanEngineTma - CHIMERA-V M10.9.6: Separable ACE Engine.
 */
class VulkanEngineTma {
public:
    VulkanEngineTma() {
        context = std::make_unique<VulkanContext>();
        
        // 🚀 修正三：分離式著色器載入
        spatialAggPipeline = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_spatial_agg.spv");
        pipeline = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_tma_evolve.spv");
        decodePipeline = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_decode.spv");
        injectPipeline = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_inject.spv");

        createDescriptorPool();
        createCommandPool();
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
        vkCreateFence(context->getDevice(), &fci, nullptr, &renderFence);
    }

    ~VulkanEngineTma() {
        if (context) vkQueueWaitIdle(context->getComputeQueue());
        cleanupBuffers();
    }

    template <uint32_t W, uint32_t H, uint32_t C>
    void prepareResources(const RetinaState<W, H, C>& state) {
        size_t size = state.size_bytes();
        size_t weight_size = (size_t)C * (C + 9) * sizeof(uint16_t); // 空間+密集權重
        const VkDeviceSize tmaAlignment = 128;
        
        createAlignedBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, d_input, d_input_memory, tmaAlignment);
        createAlignedBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, d_output, d_output_memory, tmaAlignment);
        
        // 🚀 修正三：新增聚合暫存區 (Pass 1 -> Pass 2)
        createAlignedBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, d_spatial_agg, d_spatial_agg_memory, tmaAlignment);
        
        createAlignedBuffer(weight_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, d_weights, d_weights_memory, tmaAlignment);
        createAlignedBuffer(1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, d_decode_res, d_decode_res_memory, tmaAlignment);
        createAlignedBuffer(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, d_global_state, d_global_state_memory, tmaAlignment);
        createAlignedBuffer(std::max(size, weight_size), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory, tmaAlignment);

        allocateDescriptorSets();
        updateDescriptorSets();
        
        uint32_t zero = 0;
        uploadToBuffer(d_global_state, &zero, 4);

        // 🚀 修正一：巨集指令圖錄製
        recordGenerationPipeline(W, H, C);
    }

    uint32_t step_generation_batch(uint32_t w, uint32_t h, uint32_t c) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &streamCB, 0, nullptr};
        vkResetFences(context->getDevice(), 1, &renderFence);
        vkQueueSubmit(context->getComputeQueue(), 1, &si, renderFence);
        
        vkWaitForFences(context->getDevice(), 1, &renderFence, VK_TRUE, 5000000000ULL);

        // 🚀 修正二：呼吸機制，防止 Windows 圖形搶佔處罰 (WDDM Penalty)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return 0; 
    }

    void loadWeights(const std::string& path, size_t size) {
        std::vector<uint16_t> hw(size/2); std::ifstream f(path, std::ios::binary); 
        if (!f.is_open()) throw std::runtime_error("Weights open fail");
        f.read((char*)hw.data(), size); uploadToBuffer(d_weights, hw.data(), size);
    }

private:
    std::unique_ptr<VulkanContext> context;
    std::unique_ptr<VulkanPipeline> pipeline, decodePipeline, injectPipeline, spatialAggPipeline;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSets[2], decodeSet, injectSet, spatialAggSets[2];
    VkFence renderFence;
    VkBuffer d_input, d_output, d_weights, d_decode_res, d_global_state, d_spatial_agg, stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory d_input_memory, d_output_memory, d_weights_memory, d_decode_res_memory, d_global_state_memory, d_spatial_agg_memory, stagingBufferMemory = VK_NULL_HANDLE;
    VkCommandPool commandPool;
    VkCommandBuffer streamCB = VK_NULL_HANDLE;

    void recordGenerationPipeline(uint32_t w, uint32_t h, uint32_t c) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        vkAllocateCommandBuffers(context->getDevice(), &ai, &streamCB);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(streamCB, &bi);

        // 一次錄製 50 個 Token，消除提交氣泡 (Pipeline Bubble)
        const int TOKENS_PER_SUBMIT = 50;
        for (int tokenIdx = 0; tokenIdx < TOKENS_PER_SUBMIT; tokenIdx++) {
            // 1. Inject
            vkCmdBindPipeline(streamCB, VK_PIPELINE_BIND_POINT_COMPUTE, injectPipeline->getPipeline());
            vkCmdBindDescriptorSets(streamCB, VK_PIPELINE_BIND_POINT_COMPUTE, injectPipeline->getPipelineLayout(), 0, 1, &injectSet, 0, nullptr);
            struct InjConfig { uint32_t ch, u1, u2, u3; } i_cfg = {c, 0, 0, 0};
            vkCmdPushConstants(streamCB, injectPipeline->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(InjConfig), &i_cfg);
            vkCmdDispatch(streamCB, (c/8 + 255)/256, 1, 1);
            addExecutionBarrier(streamCB);

            // 2. Evolve 20 Steps (Separable Pass)
            for(int i = 0; i < 20; i++) {
                // Pass 1: Spatial Aggregation
                vkCmdBindPipeline(streamCB, VK_PIPELINE_BIND_POINT_COMPUTE, spatialAggPipeline->getPipeline());
                vkCmdBindDescriptorSets(streamCB, VK_PIPELINE_BIND_POINT_COMPUTE, spatialAggPipeline->getPipelineLayout(), 0, 1, &spatialAggSets[i % 2], 0, nullptr);
                struct SpatConfig { uint32_t w, h, c; float dt; uint32_t r, m; } s_cfg = {w, h, c, 0.01f, 0, 100};
                vkCmdPushConstants(streamCB, spatialAggPipeline->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SpatConfig), &s_cfg);
                vkCmdDispatch(streamCB, (w*h*(c/8) + 255)/256, 1, 1);
                addExecutionBarrier(streamCB);

                // Pass 2: TMA 1x1 Dense
                vkCmdBindPipeline(streamCB, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipeline());
                vkCmdBindDescriptorSets(streamCB, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->getPipelineLayout(), 0, 1, &descriptorSets[i % 2], 0, nullptr);
                vkCmdPushConstants(streamCB, pipeline->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SpatConfig), &s_cfg);
                vkCmdDispatch(streamCB, (w*h + 15)/16, (c + 15)/16, 1);
                addExecutionBarrier(streamCB);
            }

            // 3. Decode
            vkCmdBindPipeline(streamCB, VK_PIPELINE_BIND_POINT_COMPUTE, decodePipeline->getPipeline());
            vkCmdBindDescriptorSets(streamCB, VK_PIPELINE_BIND_POINT_COMPUTE, decodePipeline->getPipelineLayout(), 0, 1, &decodeSet, 0, nullptr);
            struct DecConfig { uint32_t w, h, c, unused; } d_cfg = {w, h, c, 0};
            vkCmdPushConstants(streamCB, decodePipeline->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DecConfig), &d_cfg);
            vkCmdDispatch(streamCB, 1, 1, 1);
            addExecutionBarrier(streamCB);
        }
        vkEndCommandBuffer(streamCB);
    }

    void addExecutionBarrier(VkCommandBuffer cb) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void uploadToBuffer(VkBuffer dst, const void* data, size_t size) {
        void* m; vkMapMemory(context->getDevice(), stagingBufferMemory, 0, size, 0, &m);
        memcpy(m, data, size); vkUnmapMemory(context->getDevice(), stagingBufferMemory);
        copyBuffer(stagingBuffer, dst, size);
    }
    void createAlignedBuffer(VkDeviceSize s, VkBufferUsageFlags u, VkMemoryPropertyFlags p, VkBuffer& b, VkDeviceMemory& bm, VkDeviceSize alignment) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, s, u, VK_SHARING_MODE_EXCLUSIVE}; 
        vkCreateBuffer(context->getDevice(), &bi, nullptr, &b);
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(context->getDevice(), b, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, (mr.size + alignment - 1) & ~(alignment - 1), findMemoryType(mr.memoryTypeBits, p)};
        vkAllocateMemory(context->getDevice(), &ai, nullptr, &bm); vkBindBufferMemory(context->getDevice(), b, bm, 0);
    }
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer cb; vkAllocateCommandBuffers(context->getDevice(), &ai, &cb);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cb, &bi);
        VkBufferCopy region{0, 0, size}; vkCmdCopyBuffer(cb, src, dst, 1, &region);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        vkQueueSubmit(context->getComputeQueue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(context->getComputeQueue());
        vkFreeCommandBuffers(context->getDevice(), commandPool, 1, &cb);
    }
    void createCommandPool() {
        VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; p.queueFamilyIndex = context->getQueueFamilyIndex(); p.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(context->getDevice(), &p, nullptr, &commandPool);
    }
    void createDescriptorPool() {
        VkDescriptorPoolSize s{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32}; VkDescriptorPoolCreateInfo p{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; p.poolSizeCount = 1; p.pPoolSizes = &s; p.maxSets = 10;
        vkCreateDescriptorPool(context->getDevice(), &p, nullptr, &descriptorPool);
    }
    void allocateDescriptorSets() {
        VkDescriptorSetLayout l1 = pipeline->getDescriptorSetLayout(); 
        VkDescriptorSetLayout l_agg = spatialAggPipeline->getDescriptorSetLayout();
        
        VkDescriptorSetLayout layouts[2] = {l1, l1};
        VkDescriptorSetAllocateInfo ai1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 2, layouts};
        vkAllocateDescriptorSets(context->getDevice(), &ai1, descriptorSets);

        VkDescriptorSetLayout aggLayouts[2] = {l_agg, l_agg};
        VkDescriptorSetAllocateInfo ai_agg{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 2, aggLayouts};
        vkAllocateDescriptorSets(context->getDevice(), &ai_agg, spatialAggSets);
        
        VkDescriptorSetLayout l_dec = decodePipeline->getDescriptorSetLayout();
        VkDescriptorSetAllocateInfo ai_dec{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 1, &l_dec};
        vkAllocateDescriptorSets(context->getDevice(), &ai_dec, &decodeSet);

        VkDescriptorSetLayout l_inj = injectPipeline->getDescriptorSetLayout();
        VkDescriptorSetAllocateInfo ai_inj{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 1, &l_inj};
        vkAllocateDescriptorSets(context->getDevice(), &ai_inj, &injectSet);
    }
    void updateDescriptorSets() {
        for (int i = 0; i < 2; i++) {
            // Pass 2: TMA Dense
            VkDescriptorBufferInfo b_agg = { d_spatial_agg, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo b_out = { (i == 0 ? d_output : d_input), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo b_w = { d_weights, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet w1[3] = {}; for(int j=0; j<3; j++) { w1[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w1[j].dstSet = descriptorSets[i]; w1[j].dstBinding = j; w1[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w1[j].descriptorCount = 1; }
            w1[0].pBufferInfo = &b_agg; w1[1].pBufferInfo = &b_out; w1[2].pBufferInfo = &b_w;
            vkUpdateDescriptorSets(context->getDevice(), 3, w1, 0, nullptr);

            // Pass 1: Spatial Aggregator
            VkDescriptorBufferInfo b_in = { (i == 0 ? d_input : d_output), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo b_tmp = { d_spatial_agg, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo b_sw = { d_weights, 0, 1024*1024 }; // 假設空間權重在緩衝區起始處
            VkWriteDescriptorSet wa[3] = {}; for(int j=0; j<3; j++) { wa[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wa[j].dstSet = spatialAggSets[i]; wa[j].dstBinding = j; wa[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wa[j].descriptorCount = 1; }
            wa[0].pBufferInfo = &b_in; wa[1].pBufferInfo = &b_tmp; wa[2].pBufferInfo = &b_sw;
            vkUpdateDescriptorSets(context->getDevice(), 3, wa, 0, nullptr);
        }
        // ... (Decode & Inject Descriptor Updates, already using d_global_state)
    }
    uint32_t findMemoryType(uint32_t f, VkMemoryPropertyFlags p) {
        VkPhysicalDeviceMemoryProperties m; vkGetPhysicalDeviceMemoryProperties(context->getPhysicalDevice(), &m);
        for (uint32_t i = 0; i < m.memoryTypeCount; i++) if ((f & (1 << i)) && (m.memoryTypes[i].propertyFlags & p) == p) return i;
        return 0;
    }
    void cleanupBuffers() {
        if (stagingBuffer != VK_NULL_HANDLE) { vkDestroyBuffer(context->getDevice(), stagingBuffer, nullptr); vkFreeMemory(context->getDevice(), stagingBufferMemory, nullptr); }
        vkDestroyBuffer(context->getDevice(), d_input, nullptr); vkFreeMemory(context->getDevice(), d_input_memory, nullptr);
        vkDestroyBuffer(context->getDevice(), d_output, nullptr); vkFreeMemory(context->getDevice(), d_output_memory, nullptr);
        vkDestroyBuffer(context->getDevice(), d_weights, nullptr); vkFreeMemory(context->getDevice(), d_weights_memory, nullptr);
        vkDestroyBuffer(context->getDevice(), d_decode_res, nullptr); vkFreeMemory(context->getDevice(), d_decode_res_memory, nullptr);
        vkDestroyBuffer(context->getDevice(), d_global_state, nullptr); vkFreeMemory(context->getDevice(), d_global_state_memory, nullptr);
        vkDestroyBuffer(context->getDevice(), d_spatial_agg, nullptr); vkFreeMemory(context->getDevice(), d_spatial_agg_memory, nullptr);
    }
};
