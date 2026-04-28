#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "engine/VulkanEngineTma.hpp"
#include "engine/RetinaState.hpp"

int main() {
    std::cout << "========== CHIMERA-V M10: True 3x3 Spatial Diffusion Test ==========" << std::endl;

    try {
        VulkanEngineTma engine;
        // 🚀 修正：對齊 Production 權重維度 (64x64 視網膜, 2560 通道)
        const uint32_t W = 64, H = 64, C = 2560; 
        RetinaState<W, H, C> hostState;

        std::cout << "[Step 1] Preparing Aligned VRAM for 3x3 Kernels..." << std::endl;
        engine.prepareResources(hostState);

        // 載入 112.5 MB 的完整 3x3 空間權重
        std::cout << "[Step 1.2] Loading Production 3x3 Weights (2560-ch * 9 directions)..." << std::endl;
        // 🚀 修正：正確計算 112.5 MB 大小
        engine.loadWeights("weights/gemma_nca_weights_3x3.bin", (size_t)C * (C * 9) * sizeof(uint16_t));

        // --- 🚀 M10.9.4: 自動化巨集空間擴散生成 ---
        std::cout << "[Step 2] Starting Autonomous Spatial Diffusion (50 tokens batch)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // 🚨 修正：一次呼叫，GPU 內部全速流轉 50 個 Token
        engine.step_generation_batch(W, H, C);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << "\n--- M10 Physical Verification ---" << std::endl;
        std::cout << "Total time for 50 spatial tokens: " << (diff.count() * 1000.0) << " ms" << std::endl;
        std::cout << "Avg time per spatial token generation: " << (diff.count() * 1000.0 / 50.0) << " ms" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "!!! FATAL: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
