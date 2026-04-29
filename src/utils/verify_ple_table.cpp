#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iomanip>

/**
 * ShaderLLM PLE 權重驗證工具
 * 
 * 目的：驗證 5.6GB gemma_ple_table.bin 的二進位解析與 FP16 精度。
 * 佈局：[VOCAB_SIZE (262144), LAYERS (42) * PLE_DIM (256)]
 * 總維度：262144 x 10752
 */

// 16-bit float (half) 轉 32-bit float (simple placeholder)
// 注意：這是一個基本的轉換，主要用於驗證數值是否在合理區間
float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x01;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;

    if (exp == 0) {
        return (sign ? -1.0f : 1.0f) * std::pow(2, -14) * (mant / 1024.0f);
    } else if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }

    return (sign ? -1.0f : 1.0f) * std::pow(2, (int)exp - 15) * (1.0f + mant / 1024.0f);
}

int main() {
    const char* filename = "weights/gemma_ple_table.bin";
    const size_t VOCAB_SIZE = 262144;
    const size_t PLE_DIM_TOTAL = 10752; // 42 layers * 256 dims
    const size_t LAYERS = 42;
    const size_t DIM_PER_LAYER = 256;

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "❌ 無法開啟檔案: " << filename << std::endl;
        std::cerr << "請先執行 python training/export_ple_weights.py 產生檔案。" << std::endl;
        return 1;
    }

    std::streamsize fileSize = file.tellg();
    std::cout << "📂 檔案開啟成功: " << filename << std::endl;
    std::cout << "📊 檔案大小: " << std::fixed << std::setprecision(2) 
              << (double)fileSize / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

    // 驗證大小是否符合預期 (VOCAB_SIZE * PLE_DIM_TOTAL * sizeof(half))
    const size_t expectedSize = VOCAB_SIZE * PLE_DIM_TOTAL * 2;
    if (fileSize != expectedSize) {
        std::cerr << "⚠️ 警告：檔案大小不符合預期！" << std::endl;
        std::cerr << "預期: " << expectedSize << " bytes" << std::endl;
        std::cerr << "實際: " << fileSize << " bytes" << std::endl;
    }

    // 抽查目標：Token ID 100000, Layer 20, Dim 0
    size_t targetToken = 100000;
    size_t targetLayer = 20;
    size_t targetDim = 0;

    // 計算 Offset: (token * PLE_DIM_TOTAL + layer * DIM_PER_LAYER + dim) * 2 bytes
    size_t offset = (targetToken * PLE_DIM_TOTAL + targetLayer * DIM_PER_LAYER + targetDim) * 2;

    file.seekg(offset, std::ios::beg);
    uint16_t h_val;
    file.read(reinterpret_cast<char*>(&h_val), sizeof(uint16_t));

    if (file.gcount() != sizeof(uint16_t)) {
        std::cerr << "❌ 讀取失敗。" << std::endl;
        return 1;
    }

    float f_val = half_to_float(h_val);

    std::cout << "\n--- 抽查驗證結果 ---" << std::endl;
    std::cout << "📍 位置: Token[" << targetToken << "], Layer[" << targetLayer << "], Dim[" << targetDim << "]" << std::endl;
    std::cout << "🔢 原始二進位 (Hex): 0x" << std::hex << h_val << std::dec << std::endl;
    std::cout << "🧪 轉換為 Float32: " << f_val << std::endl;
    std::cout << "--------------------\n" << std::endl;

    std::cout << "💡 請開啟 Python 環境並執行以下代碼進行比對：" << std::endl;
    std::cout << "import numpy as np" << std::endl;
    std::cout << "data = np.fromfile('" << filename << "', dtype=np.float16).reshape(262144, 10752)" << std::endl;
    std::cout << "print(f'Python 數值: {data[" << targetToken << ", " << targetLayer * DIM_PER_LAYER + targetDim << "]}')" << std::endl;

    return 0;
}
