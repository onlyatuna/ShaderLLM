#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <limits>

// FP32 轉 FP16 (從 main_opt.cpp 提取)
uint16_t float_to_half(float f) {
    uint32_t i = *((uint32_t*)&f);
    uint32_t s = (i >> 16) & 0x00008000;
    int32_t e = ((i >> 23) & 0x000000ff) - (127 - 15);
    uint32_t m = i & 0x007fffff;
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m = (m | 0x00800000) >> (1 - e);
        return (uint16_t)(s | (m >> 13));
    }
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | (e << 10) | (m >> 13));
}

// FP16 轉 FP32 (從 main_opt.cpp 提取)
float half_to_float(uint16_t h) {
    uint32_t f;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7c00) >> 10;
    uint32_t mant = (h & 0x03ff) << 13;
    if (exp == 0x1f) exp = 0xff;
    else if (exp > 0) exp += 127 - 15;
    f = sign | (exp << 23) | mant;
    return *(float*)&f;
}

void test_value(float f) {
    uint16_t h = float_to_half(f);
    float f2 = half_to_float(h);
    float err = std::abs(f - f2);
    std::cout << std::fixed << std::setprecision(8)
              << "Original: " << std::setw(12) << f 
              << " | FP16 Hex: 0x" << std::hex << std::setw(4) << std::setfill('0') << h << std::dec
              << " | Restored: " << std::setw(12) << f2 
              << " | Abs Error: " << err << std::endl;
}

int main() {
    std::cout << "========== 底層精度與轉換誤差驗證 (FP32 <-> FP16) ==========\n";
    
    std::cout << "\n[1] 邊界條件與特殊數值測試 (Edge Cases)\n";
    test_value(0.0f);
    test_value(-0.0f);
    test_value(1.0f);
    test_value(-1.0f);
    test_value(0.5f);
    test_value(-0.5f);
    test_value(65504.0f);     // FP16 最大正規數
    test_value(65536.0f);     // 超出 FP16 範圍 (應轉為 Infinity 0x7c00)
    test_value(5.96046e-8f);  // 極小數
    test_value(1e-10f);       // Underflow (應歸零)
    
    std::cout << "\n[2] 權重分佈區間誤差測試 (-1.0 to 1.0, 100 萬次採樣)\n";
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    double max_err = 0.0;
    double sum_err = 0.0;
    int test_count = 1000000;
    
    for (int i = 0; i < test_count; ++i) {
        float f = dist(gen);
        uint16_t h = float_to_half(f);
        float f2 = half_to_float(h);
        double err = std::abs((double)f - (double)f2);
        if (err > max_err) max_err = err;
        sum_err += err;
    }
    
    std::cout << "測試樣本數 : " << test_count << "\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "最大絕對誤差 : " << max_err << " (理論上應小於 0.00097656)\n";
    std::cout << "平均絕對誤差 : " << (sum_err / test_count) << "\n";
    
    std::cout << "\n[3] NCA 演化區間誤差測試 (0.0 to 10.0, 代表 Tanh 累積結果)\n";
    std::uniform_real_distribution<float> dist2(0.0f, 10.0f);
    max_err = 0.0;
    sum_err = 0.0;
    
    for (int i = 0; i < test_count; ++i) {
        float f = dist2(gen);
        uint16_t h = float_to_half(f);
        float f2 = half_to_float(h);
        double err = std::abs((double)f - (double)f2);
        if (err > max_err) max_err = err;
        sum_err += err;
    }
    
    std::cout << "測試樣本數 : " << test_count << "\n";
    std::cout << "最大絕對誤差 : " << max_err << "\n";
    std::cout << "平均絕對誤差 : " << (sum_err / test_count) << "\n";
    
    return 0;
}