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

class VulkanEngineOpt {
public:
    VulkanEngineOpt() {
        context = std::make_unique<VulkanContext>();
        auto info = context->getDeviceInfo();
        bool isIntel = (info.name.find("Intel") != std::string::npos);
        
        pipelineAgg = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_spatial_agg.spv", 3);
        pipelineEvolve = std::make_unique<VulkanPipeline>(context->getDevice(), isIntel ? "src/shaders/nca_uhd.spv" : "src/shaders/nca_evolve.spv", 4);
        pipelineRMSNorm = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_rmsnorm.spv", 3);
        pipelineDecode = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_decode.spv", 5);
        pipelineInject = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_inject.spv", 5);
        pipelinePleGate = std::make_unique<VulkanPipeline>(context->getDevice(), "src/shaders/nca_ple_gate.spv", 4);

        createDescriptorPool();
        createCommandPool();
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
        vkCreateFence(context->getDevice(), &fci, nullptr, &renderFence);
    }

    ~VulkanEngineOpt() {
        if (context) vkQueueWaitIdle(context->getComputeQueue());
        cleanupBuffers();
        if (context && context->getDevice()) {
            vkDestroyFence(context->getDevice(), renderFence, nullptr);
            vkDestroyDescriptorPool(context->getDevice(), descriptorPool, nullptr);
            vkDestroyCommandPool(context->getDevice(), commandPool, nullptr);
        }
    }

    // --- 🚀 穩固的基礎 API ---
    void upload(const void* data, size_t size) {
        if (stagingBuffer == VK_NULL_HANDLE) throw std::runtime_error("Staging buffer not initialized!");
        void* m; vkMapMemory(context->getDevice(), stagingBufferMemory, 0, size, 0, &m); 
        memcpy(m, data, size); 
        vkUnmapMemory(context->getDevice(), stagingBufferMemory);
        copyBuffer(stagingBuffer, d_input, size); 
    }

    void upload_persistent_field(const void* data, size_t size) {
        if (stagingBuffer == VK_NULL_HANDLE) throw std::runtime_error("Staging buffer not initialized!");
        void* m; vkMapMemory(context->getDevice(), stagingBufferMemory, 0, size, 0, &m); 
        memcpy(m, data, size); 
        vkUnmapMemory(context->getDevice(), stagingBufferMemory);
        copyBuffer(stagingBuffer, d_persistent_field, size); 
    }

    void download(void* data, size_t size) {
        if (stagingBuffer == VK_NULL_HANDLE) throw std::runtime_error("Staging buffer not initialized!");
        copyBuffer((pingPongIndex == 1 ? d_output : d_input), stagingBuffer, size); 
        void* m; vkMapMemory(context->getDevice(), stagingBufferMemory, 0, size, 0, &m); 
        memcpy(data, m, size); 
        vkUnmapMemory(context->getDevice(), stagingBufferMemory);
    }

    void downloadTokens(uint32_t* results, uint32_t count) {
        VkDeviceSize size = count * sizeof(uint32_t);
        copyBuffer(d_token_history, stagingBuffer, size);
        void* m; vkMapMemory(context->getDevice(), stagingBufferMemory, 0, size, 0, &m);
        memcpy(results, m, size);
        vkUnmapMemory(context->getDevice(), stagingBufferMemory);
    }

    void loadWeights(const std::string& path, size_t size) {
        if (stagingBuffer == VK_NULL_HANDLE) throw std::runtime_error("Staging buffer not initialized!");
        std::vector<uint16_t> hw(size/2); 
        std::ifstream f(path, std::ios::binary); 
        if (!f.is_open()) throw std::runtime_error("Failed to open weight file: " + path);
        f.read((char*)hw.data(), size);
        void* m; vkMapMemory(context->getDevice(), stagingBufferMemory, 0, size, 0, &m); 
        memcpy(m, hw.data(), size); 
        vkUnmapMemory(context->getDevice(), stagingBufferMemory);
        copyBuffer(stagingBuffer, d_weights, size); 
    }

    // --- 🚀 演化核心 API ---
    void evolve_batch(uint32_t iterations, uint32_t w, uint32_t h, uint32_t c) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer cb; vkAllocateCommandBuffers(context->getDevice(), &ai, &cb);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cb, &bi);
        for(uint32_t i=0; i<iterations; i++) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineEvolve->getPipeline());
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineEvolve->getPipelineLayout(), 0, 1, &descriptorSetsEvolve[pingPongIndex], 0, nullptr);
            struct { uint32_t w, h, c; float dt; uint32_t row_off; } cfg = {w, h, c, 0.01f, 0};
            vkCmdPushConstants(cb, pipelineEvolve->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &cfg);
            vkCmdDispatch(cb, (w*h+63)/64, (c+63)/64, 1);
            pingPongIndex = 1 - pingPongIndex;
            insertBarrier(cb, (pingPongIndex == 0 ? d_input : d_output));
        }
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        vkQueueSubmit(context->getComputeQueue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(context->getComputeQueue());
        vkFreeCommandBuffers(context->getDevice(), commandPool, 1, &cb);
    }

    void generate_fused(uint32_t prompt_len, uint32_t tokens, uint32_t w, uint32_t h, uint32_t c) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer cb; vkAllocateCommandBuffers(context->getDevice(), &ai, &cb);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cb, &bi);
        vkCmdFillBuffer(cb, d_global_state, 0, 4, prompt_len);
        for (uint32_t t = 0; t < tokens; t++) {
            for (int s = 0; s < 42; s++) {
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineAgg->getPipeline());
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineAgg->getPipelineLayout(), 0, 1, &descriptorSetsAgg[pingPongIndex], 0, nullptr);
                struct { uint32_t w, h, c; float dt; uint32_t row_off; uint32_t mass; } cfgAgg = {w, h, c, 0.01f, 0, t+1};
                vkCmdPushConstants(cb, pipelineAgg->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, &cfgAgg);
                vkCmdDispatch(cb, (w * h * c / 8 + 255) / 256, 1, 1);
                insertBarrier(cb, d_spatial_temp);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineEvolve->getPipeline());
                for (uint32_t rs = 0; rs < c; rs += 1280) {
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineEvolve->getPipelineLayout(), 0, 1, &descriptorSetsEvolve[pingPongIndex], 0, nullptr);
                    struct { uint32_t w, h, c; float dt; uint32_t row_off; } cfgEv = {w, h, c, 0.01f, rs};
                    vkCmdPushConstants(cb, pipelineEvolve->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, &cfgEv);
                    vkCmdDispatch(cb, (w*h+63)/64, (std::min(1280u, c-rs)+63)/64, 1);
                }
                
                // 🚀 執行獨立的 RMSNorm pass (全通道 FP32)
                insertBarrier(cb, (pingPongIndex == 0 ? d_output : d_input));
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineRMSNorm->getPipeline());
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineRMSNorm->getPipelineLayout(), 0, 1, &descriptorSetsRMSNorm[pingPongIndex], 0, nullptr);
                struct { uint32_t w, h, c; float dt; } cfgRms = {w, h, c, 0.01f};
                vkCmdPushConstants(cb, pipelineRMSNorm->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &cfgRms);
                vkCmdDispatch(cb, (w*h+63)/64, 1, 1);

                pingPongIndex = 1 - pingPongIndex;
                insertBarrier(cb, (pingPongIndex == 0 ? d_input : d_output));
            }
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineDecode->getPipeline());
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineDecode->getPipelineLayout(), 0, 1, &descriptorSetsDecode[pingPongIndex], 0, nullptr);
            struct { uint32_t w, h, c; float temperature; float top_p; float penalty; } dcfg = {w, h, c, 0.1f, 0.9f, 1.15f};
            vkCmdPushConstants(cb, pipelineDecode->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, &dcfg);
            vkCmdDispatch(cb, 1, 1, 1);
            insertBarrier(cb, d_decode_result);
            VkBufferCopy copyRegion{ sizeof(uint32_t), t * sizeof(uint32_t), sizeof(uint32_t) };
            vkCmdCopyBuffer(cb, d_decode_result, d_token_history, 1, &copyRegion);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineInject->getPipeline());
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineInject->getPipelineLayout(), 0, 1, &descriptorSetsInject[pingPongIndex], 0, nullptr);
            struct { uint32_t c, u1, u2, u3; } icfg = {c, 0, 0, 0};
            vkCmdPushConstants(cb, pipelineInject->getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &icfg);
            vkCmdDispatch(cb, (c / 8 + 255) / 256, 1, 1);
            insertBarrier(cb, (pingPongIndex == 0 ? d_input : d_output));
            insertBarrier(cb, d_persistent_field);
        }
        vkEndCommandBuffer(cb);
        vkResetFences(context->getDevice(), 1, &renderFence);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        vkQueueSubmit(context->getComputeQueue(), 1, &si, renderFence);
        vkWaitForFences(context->getDevice(), 1, &renderFence, VK_TRUE, 5000000000ULL);
        vkFreeCommandBuffers(context->getDevice(), commandPool, 1, &cb);
    }

    template <uint32_t W, uint32_t H, uint32_t C>
    void prepareResources(const RetinaState<W, H, C>& state) {
        size_t size = state.size_bytes();
        size_t weight_size = 9ULL * C * C * sizeof(uint16_t);
        size_t embed_size = 262144ULL * C * sizeof(uint16_t); // 🚀 更新為 Gemma-4 詞表大小
        VkMemoryPropertyFlags vramFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vramFlags, d_input, d_input_memory);
        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vramFlags, d_output, d_output_memory);
        createBuffer(size * 9, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, vramFlags, d_spatial_temp, d_spatial_temp_memory);
        createBuffer(weight_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vramFlags, d_weights, d_weights_memory);
        createBuffer(embed_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vramFlags, d_embed_table, d_embed_memory);
        createBuffer(8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vramFlags, d_decode_result, d_decode_memory);
        createBuffer(4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vramFlags, d_global_state, d_global_memory);
        createBuffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vramFlags, d_token_history, d_token_history_memory);
        createBuffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, vramFlags, d_persistent_field, d_persistent_field_memory);
        createBuffer(std::max(size, embed_size), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        allocateDescriptorSets();
        updateDescriptorSets();
    }

private:
    std::unique_ptr<VulkanContext> context;
    std::unique_ptr<VulkanPipeline> pipelineAgg, pipelineEvolve, pipelineRMSNorm, pipelineDecode, pipelineInject;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSetsAgg[2], descriptorSetsEvolve[2], descriptorSetsRMSNorm[2], descriptorSetsDecode[2], descriptorSetsInject[2];
    VkFence renderFence;
    uint32_t pingPongIndex = 0;
    VkBuffer d_input, d_output, d_spatial_temp, d_weights, d_embed_table, d_decode_result, d_global_state, d_token_history, d_persistent_field, stagingBuffer;
    VkDeviceMemory d_input_memory, d_output_memory, d_spatial_temp_memory, d_weights_memory, d_embed_memory, d_decode_memory, d_global_memory, d_token_history_memory, d_persistent_field_memory, stagingBufferMemory;
    VkCommandPool commandPool;

    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer cb; vkAllocateCommandBuffers(context->getDevice(), &ai, &cb);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cb, &bi);
        VkBufferCopy region{0, 0, size};
        vkCmdCopyBuffer(cb, src, dst, 1, &region);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr};
        vkQueueSubmit(context->getComputeQueue(), 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(context->getComputeQueue());
        vkFreeCommandBuffers(context->getDevice(), commandPool, 1, &cb);
    }
    void insertBarrier(VkCommandBuffer cb, VkBuffer buf) {
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, buf, 0, VK_WHOLE_SIZE};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
    }
    void createCommandPool() {
        VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; p.queueFamilyIndex = context->getQueueFamilyIndex(); p.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(context->getDevice(), &p, nullptr, &commandPool);
    }
    void createDescriptorPool() {
        VkDescriptorPoolSize s{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
        VkDescriptorPoolCreateInfo p{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; p.poolSizeCount = 1; p.pPoolSizes = &s; p.maxSets = 16;
        vkCreateDescriptorPool(context->getDevice(), &p, nullptr, &descriptorPool);
    }
    void allocateDescriptorSets() {
        auto d = context->getDevice();
        VkDescriptorSetLayout lAgg = pipelineAgg->getDescriptorSetLayout();
        VkDescriptorSetLayout lEvo = pipelineEvolve->getDescriptorSetLayout();
        VkDescriptorSetLayout lRms = pipelineRMSNorm->getDescriptorSetLayout();
        VkDescriptorSetLayout lDec = pipelineDecode->getDescriptorSetLayout();
        VkDescriptorSetLayout lInj = pipelineInject->getDescriptorSetLayout();
        VkDescriptorSetLayout layouts[2] = {lAgg, lAgg};
        VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 2, layouts};
        vkAllocateDescriptorSets(d, &a, descriptorSetsAgg);
        VkDescriptorSetLayout layoutsEvo[2] = {lEvo, lEvo}; a.pSetLayouts = layoutsEvo; vkAllocateDescriptorSets(d, &a, descriptorSetsEvolve);
        VkDescriptorSetLayout layoutsRms[2] = {lRms, lRms}; a.pSetLayouts = layoutsRms; vkAllocateDescriptorSets(d, &a, descriptorSetsRMSNorm);
        VkDescriptorSetLayout layoutsDec[2] = {lDec, lDec}; a.pSetLayouts = layoutsDec; vkAllocateDescriptorSets(d, &a, descriptorSetsDecode);
        VkDescriptorSetLayout layoutsInj[2] = {lInj, lInj}; a.pSetLayouts = layoutsInj; vkAllocateDescriptorSets(d, &a, descriptorSetsInject);
    }
    void updateDescriptorSets() {
        for (int i = 0; i < 2; i++) {
            VkBuffer b_in = (i == 0 ? d_input : d_output);
            VkBuffer b_out = (i == 0 ? d_output : d_input);
            writeDS(descriptorSetsAgg[i], {b_in, d_spatial_temp, d_weights});
            writeDS(descriptorSetsEvolve[i], {d_spatial_temp, b_out, d_weights, d_persistent_field});
            writeDS(descriptorSetsRMSNorm[i], {b_in, b_out, d_persistent_field, b_out}); // 🚀 修正 2：寫回 b_out 確保 Ping-Pong 連續性
            writeDS(descriptorSetsDecode[i], {b_in, d_decode_result, d_global_state, d_token_history, d_embed_table}); // 🚀 修正 3：讀取最新的 b_in
            writeDS(descriptorSetsInject[i], {d_embed_table, b_in, d_decode_result, d_global_state, d_persistent_field});
        }
    }
    void writeDS(VkDescriptorSet ds, std::vector<VkBuffer> bufs) {
        std::vector<VkDescriptorBufferInfo> info(bufs.size());
        std::vector<VkWriteDescriptorSet> w(bufs.size());
        for(size_t i=0; i<bufs.size(); i++) {
            info[i] = {bufs[i], 0, VK_WHOLE_SIZE};
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info[i], nullptr};
        }
        vkUpdateDescriptorSets(context->getDevice(), (uint32_t)w.size(), w.data(), 0, nullptr);
    }
    void createBuffer(VkDeviceSize s, VkBufferUsageFlags u, VkMemoryPropertyFlags p, VkBuffer& b, VkDeviceMemory& bm) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, s, u, VK_SHARING_MODE_EXCLUSIVE}; 
        vkCreateBuffer(context->getDevice(), &bi, nullptr, &b);
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(context->getDevice(), b, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, findMemoryType(mr.memoryTypeBits, p)};
        vkAllocateMemory(context->getDevice(), &ai, nullptr, &bm); vkBindBufferMemory(context->getDevice(), b, bm, 0);
    }
    uint32_t findMemoryType(uint32_t f, VkMemoryPropertyFlags p) {
        VkPhysicalDeviceMemoryProperties m; vkGetPhysicalDeviceMemoryProperties(context->getPhysicalDevice(), &m);
        for (uint32_t i = 0; i < m.memoryTypeCount; i++) if ((f & (1 << i)) && (m.memoryTypes[i].propertyFlags & p) == p) return i;
        return 0;
    }
    void cleanupBuffers() {
        auto d = context->getDevice();
        vkDestroyBuffer(d, d_input, nullptr); vkFreeMemory(d, d_input_memory, nullptr);
        vkDestroyBuffer(d, d_output, nullptr); vkFreeMemory(d, d_output_memory, nullptr);
        vkDestroyBuffer(d, d_spatial_temp, nullptr); vkFreeMemory(d, d_spatial_temp_memory, nullptr);
        vkDestroyBuffer(d, d_weights, nullptr); vkFreeMemory(d, d_weights_memory, nullptr);
        vkDestroyBuffer(d, d_embed_table, nullptr); vkFreeMemory(d, d_embed_memory, nullptr);
        vkDestroyBuffer(d, d_decode_result, nullptr); vkFreeMemory(d, d_decode_memory, nullptr);
        vkDestroyBuffer(d, d_global_state, nullptr); vkFreeMemory(d, d_global_memory, nullptr);
        vkDestroyBuffer(d, d_token_history, nullptr); vkFreeMemory(d, d_token_history_memory, nullptr);
        vkDestroyBuffer(d, d_persistent_field, nullptr); vkFreeMemory(d, d_persistent_field_memory, nullptr);
        vkDestroyBuffer(d, stagingBuffer, nullptr); vkFreeMemory(d, stagingBufferMemory, nullptr);
    }
};
eMemory(d, d_weights_memory, nullptr);
        vkDestroyBuffer(d, d_embed_table, nullptr); vkFreeMemory(d, d_embed_memory, nullptr);
        vkDestroyBuffer(d, d_decode_result, nullptr); vkFreeMemory(d, d_decode_memory, nullptr);
        vkDestroyBuffer(d, d_global_state, nullptr); vkFreeMemory(d, d_global_memory, nullptr);
        vkDestroyBuffer(d, d_token_history, nullptr); vkFreeMemory(d, d_token_history_memory, nullptr);
        vkDestroyBuffer(d, d_persistent_field, nullptr); vkFreeMemory(d, d_persistent_field_memory, nullptr);
        vkDestroyBuffer(d, stagingBuffer, nullptr); vkFreeMemory(d, stagingBufferMemory, nullptr);
    }
};
