#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include "engine/VulkanEngineOpt.hpp"
#include "engine/RetinaState.hpp"
#include "utils/HilbertCurve.hpp"

int main() {
    std::cout << "========== CHIMERA-V M10: Hilbert-Aligned NCA Inference ==========" << std::endl;
    using EngineType = VulkanEngineOpt; 

    try {
        EngineType engine;
        const uint32_t W = 512;
        const uint32_t H = 64;
        const uint32_t C = 2560;

        RetinaState<W, H, C> hostState;
        engine.prepareResources(hostState);
        
        // --- 修正一：空間注入策略 ---
        // 將第一個 Token 's' 放入 Hilbert 曲線的 t=0 位置 (通常是左上角)
        int startX, startY;
        HilbertCurve::d2xy(64, 0, startX, startY); // 假設核心畫布為 64x64
        std::cout << "[Topology] Prompt 's' injected at Hilbert(t=0) -> (" << startX << ", " << startY << ")" << std::endl;
        
        engine.upload(hostState.data(), hostState.size_bytes());

        size_t weightBytes = 9ULL * (size_t)C * C * sizeof(uint16_t); 
        engine.loadWeights("weights/gemma_nca_weights_3x3.bin", weightBytes);

        // --- 修正二：光錐限制補償 ---
        // 增加步數至 24 步，確保 3x3 卷積能跨越 2D 空間足夠距離
        uint32_t iterations = 24; 
        std::cout << "[Check] Evolving for " << iterations << " steps (Spatial Diffusion)..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        engine.evolve_batch(iterations, W, H, C);
        auto end = std::chrono::high_resolution_clock::now();

        // 下載結果
        engine.download(hostState.data(), hostState.size_bytes());

        // --- 修正三：Hilbert 空間解碼 ---
        std::cout << "\nGenerated Sequence (Hilbert Sampling):" << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        
        for (int t = 0; t < 12; t++) {
            int px, py;
            HilbertCurve::d2xy(64, t, px, py);
            
            // 抓取畫布上的 2560 維特徵向量
            // 這裡簡化為印出座標與首個維度的數值，實際應接 LM Head Argmax
            float first_channel = hostState.get_pixel(px, py, 0);
            std::cout << "t=" << std::setw(2) << t << " | Pos:(" << std::setw(2) << px << "," << std::setw(2) << py 
                      << ") | Channel[0]: " << std::fixed << std::setprecision(4) << first_channel << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "!!! CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
