#include <iostream>
#include <vector>
#include "engine/VulkanEngineOpt.hpp"
#include "engine/RetinaState.hpp"

float half_to_float_test(uint16_t h) {
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
    std::cout << "========== CHIMERA-V: Data Flow Integrity Test ==========" << std::endl;

    try {
        // 1. 強制載入驗證用 Shader
        // 注意：為了測試，我們手動指定 shader 路徑
        #ifdef _WIN32
        _putenv("CHIMERA_VERIFY_SHADER=src/shaders/verify_flow.spv");
        #endif

        VulkanEngineOpt engine;
        CHIMERA_V_Standard state;
        engine.prepareResources(state);

        // 2. 設定初始值為 1.0
        std::vector<uint16_t> initial(state.size_bytes() / 2, 0x3C00); 
        engine.upload(initial.data(), state.size_bytes());

        // 3. 執行 2 輪演化
        // 第 1 步：1.0 + 1.0 = 2.0
        // 第 2 步：2.0 + 1.0 = 3.0
        std::cout << "[Test] Running 2 steps of accumulation..." << std::endl;
        engine.evolve_batch(2, 512, 64, 2560);

        // 4. 下載結果
        std::vector<uint16_t> result(state.size_bytes() / 2);
        engine.download(result.data(), state.size_bytes());

        float val = half_to_float_test(result[0]);
        std::cout << "\n--- Integrity Results ---" << std::endl;
        std::cout << "Initial Value : 1.0" << std::endl;
        std::cout << "Final Value   : " << val << std::endl;

        if (std::abs(val - 3.0f) < 0.1f) {
            std::cout << "[SUCCESS] Ping-Pong Data Flow is VERIFIED. Loopback is perfect." << std::endl;
        } else if (std::abs(val - 2.0f) < 0.1f) {
            std::cout << "[FAIL] Data Flow Stuck! Result is 2.0, meaning no Ping-Pong occurred." << std::endl;
            return 1;
        } else {
            std::cout << "[FAIL] Data Corrupted! Value: " << val << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Test Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
