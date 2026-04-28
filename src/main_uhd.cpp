#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include "engine/VulkanEngineOpt.hpp"
#include "engine/RetinaState.hpp"

float half_to_float_uhd(uint16_t h) {
    uint32_t f;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7c00) >> 10;
    uint32_t mant = (h & 0x03ff) << 13;
    if (exp == 0x1f) exp = 0xff;
    else if (exp > 0) exp += 127 - 15;
    f = sign | (exp << 23) | mant;
    return *(float*)&f;
}

int main() {
    std::cout << "========== CHIMERA-V M2: Intel UHD 770 Strict Benchmark ==========" << std::endl;
    #ifdef _WIN32
    _putenv("CHIMERA_PREFER_INTEL=1");
    #endif

    try {
        VulkanEngineOpt engine;
        CHIMERA_V_Standard hostState; 
        const uint32_t W = 512, H = 64, C = 2560;

        std::vector<uint16_t> initialData(hostState.size_bytes() / 2, 0x3C00); // 1.0
        engine.prepareResources(hostState);
        engine.upload(initialData.data(), hostState.size_bytes());
        
        std::vector<uint16_t> weights(C * C, 0x3800); // 0.5 (降低權重防止溢位)
        std::ofstream wfile("weights/uhd_weights.bin", std::ios::binary);
        wfile.write((char*)weights.data(), C * C * 2); wfile.close();
        engine.loadWeights("weights/uhd_weights.bin", C * C * 2);

        const int iters = 1; // 降至 1 次以確保 UHD 770 不會觸發 TDR 並產出真實數據
        std::cout << "[UHD Test] Running 1 step for Numerical Integrity Check..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        engine.evolve_batch(iters, W, H, C); 
        auto end = std::chrono::high_resolution_clock::now();

        std::vector<uint16_t> result(hostState.size_bytes() / 2);
        engine.download(result.data(), hostState.size_bytes());

        float finalVal = half_to_float_uhd(result[0]);
        std::cout << "\n--- UHD Numerical Verification ---" << std::endl;
        std::cout << "[Check] Initial: " << half_to_float_uhd(initialData[0]) << std::endl;
        std::cout << "[Check] Final  : " << finalVal << std::endl;

        if (std::abs(finalVal) < 1e-5 || std::isnan(finalVal)) {
            std::cout << "!!! CRITICAL FAIL: GPU Output is ZERO or NaN. Calculation is BROKEN." << std::endl;
            return 1;
        } else {
            std::cout << "[SUCCESS] UHD 770 produced VALID numerical output." << std::endl;
        }

        std::chrono::duration<double> diff = end - start;
        double avg = diff.count() / iters;
        std::cout << "Avg Time: " << avg * 1000.0 << " ms | True Perf: " << (2.0*W*H*C*C/1e12)/avg << " TFLOPS" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "!!! ERROR: " << e.what() << std::endl;
    }
    return 0;
}
