#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <fstream>
#include "engine/CudaEngine.hpp"
#include "engine/RetinaState.hpp"

// FP32 轉 FP16 (用於測試數據生成)
// 注意：目前為測試版，捨棄了次正規化數。正式載入 Gemma 權重時建議改用 F16C 指令集。
uint16_t float_to_half_cuda(float f) {
    uint32_t i = *((uint32_t*)&f);
    uint32_t s = (i >> 16) & 0x00008000;
    uint32_t e = ((i >> 23) & 0x000000ff) - (127 - 15);
    uint32_t m = i & 0x007fffff;
    if (e <= 0) return (uint16_t)s; 
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

int main() {
    std::cout << "========== CHIMERA-V M2: Independent CUDA Benchmark ==========" << std::endl;
    
    try {
        CudaEngine engine;
        CHIMERA_V_Standard hostState; 
        const uint32_t W = 512, H = 64, C = 2560;

        // 1. 初始化資料：正規化至 [-0.5, 0.5]
        std::vector<uint16_t> initialData(hostState.size_bytes() / 2);
        std::default_random_engine generator(42);
        std::uniform_real_distribution<float> state_dist(-0.5f, 0.5f);
        for(auto& val : initialData) val = float_to_half_cuda(state_dist(generator));

        engine.prepareResources(hostState);
        engine.upload(initialData.data(), hostState.size_bytes());

        // 2. 生成並載入 Xavier 權重
        size_t weight_elements = (size_t)C * C;
        std::vector<uint16_t> xavierWeights(weight_elements);
        std::normal_distribution<float> weight_dist(0.0f, 0.01976f);
        for(size_t i = 0; i < weight_elements; ++i) xavierWeights[i] = float_to_half_cuda(weight_dist(generator));
        
        std::ofstream wfile("weights/gemma_cuda_weights.bin", std::ios::binary);
        wfile.write(reinterpret_cast<const char*>(xavierWeights.data()), weight_elements * 2);
        wfile.close();

        engine.loadWeights("weights/gemma_cuda_weights.bin", weight_elements * 2);
        std::cout << "[Check] Environment Ready. Starting 100 iterations for stable benchmark..." << std::endl;

        // 3. 執行壓測 (使用非同步派發)
        const int iterations = 100;
        // 預熱 (Warmup)
        for(int i=0; i<20; i++) engine.evolve(W, H, C);
        engine.sync();

        std::cout << "[Benchmark] Dispatching 100 steps to CUDA Stream..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            engine.evolve(W, H, C); 
        }
        engine.sync(); // 確保所有 GPU 命令執行完畢後才停止計時
        auto end = std::chrono::high_resolution_clock::now();

        // 4. 報告
        std::chrono::duration<double> diff = end - start;
        double avg_time_ms = (diff.count() * 1000.0) / iterations;
        
        double actual_ops = 2.0 * (W * H) * C * C;
        double tflops = (actual_ops / 1e12) / (avg_time_ms / 1000.0);

        std::cout << "\n--- CUDA Performance Report ---" << std::endl;
        std::cout << "Engine Type             : CUDA (cuBLAS + Tanh Kernel)" << std::endl;
        std::cout << "Matrix Dimensions       : " << C << " x " << C << std::endl;
        std::cout << "Average Time per Step   : " << avg_time_ms << " ms" << std::endl;
        std::cout << "Peak Performance        : " << tflops << " TFLOPS" << std::endl;
        std::cout << "-------------------------------" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "!!! CUDA ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
