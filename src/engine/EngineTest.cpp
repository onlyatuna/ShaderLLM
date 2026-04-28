#include <iostream>
#include <vector>
#include <iomanip>
#include "../utils/HilbertCurve.hpp"
#include "RetinaState.hpp"

int main() {
    std::cout << "--- CHIMERA-V Extreme Engine Test ---" << std::endl;

    try {
        // 1. 實例化 Extreme 模式 (512x512x2560)
        // 這將分配約 1.25 GB 的對齊記憶體
        CHIMERA_V_Extreme retina;
        std::cout << "[Step 1] RetinaState Extreme initialized successfully." << std::endl;

        // 2. 模擬一段 Token 序列 (例如 10 個 Token)
        std::vector<uint32_t> mock_tokens = {101, 102, 103, 104, 105, 106, 107, 108, 109, 110};
        std::cout << "[Step 2] Mapping " << mock_tokens.size() << " tokens via Hilbert Curve..." << std::endl;

        // 3. 執行空間映射並填充數據
        for (size_t i = 0; i < mock_tokens.size(); ++i) {
            // 使用 512 階的希爾伯特曲線
            HilbertCurve::Point p = HilbertCurve::indexToPoint(512, static_cast<uint32_t>(i));
            
            // 獲取該座標的 2560 維 Channel 起始位址
            uint16_t* channels = retina.get_pixel_channels(p.x, p.y);
            
            if (channels) {
                // 模擬填充：將第一個 Channel 設為 Token ID (僅作驗證用)
                channels[0] = static_cast<uint16_t>(mock_tokens[i]);
                
                std::cout << "  Token[" << i << "] ID:" << mock_tokens[i] 
                          << " -> Position: (" << std::setw(3) << p.x << ", " 
                          << std::setw(3) << p.y << ")" << std::endl;
            }
        }

        // 4. 驗證相鄰性 (Spatial Locality Check)
        // 在 Hilbert Curve 中，Index 0 與 Index 1 的曼哈頓距離應為 1
        HilbertCurve::Point p0 = HilbertCurve::indexToPoint(512, 0);
        HilbertCurve::Point p1 = HilbertCurve::indexToPoint(512, 1);
        int distance = std::abs((int)p0.x - (int)p1.x) + std::abs((int)p0.y - (int)p1.y);
        
        std::cout << "[Step 3] Spatial Locality Verification:" << std::endl;
        std::cout << "  Distance between Token 0 and 1: " << distance << " unit(s)" << std::endl;
        
        if (distance == 1) {
            std::cout << "  Result: SUCCESS (Locality preserved)" << std::endl;
        } else {
            std::cout << "  Result: WARNING (Distance > 1)" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[Error] Test failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "--- End of Test ---" << std::endl;
    return 0;
}
