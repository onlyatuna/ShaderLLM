#pragma once
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <iostream>

/**
 * @brief RetinaState 模板化管理 NCA 的 2D 視網膜狀態。
 * 支援從邊緣設備到極致桌面端硬體的維度切換。
 */
template <uint32_t W, uint32_t H, uint32_t C>
class RetinaState {
public:
    static constexpr uint32_t WIDTH = W;
    static constexpr uint32_t HEIGHT = H;
    static constexpr uint32_t CHANNELS = C;
    static constexpr size_t ALIGNMENT = 64; 

    RetinaState() {
        size_t total_elements = static_cast<size_t>(WIDTH) * HEIGHT * CHANNELS;
        size_t total_bytes = total_elements * sizeof(uint16_t);

        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(total_bytes, ALIGNMENT);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&ptr, ALIGNMENT, total_bytes) != 0) {
            throw std::bad_alloc();
        }
#endif

        data_ptr = static_cast<uint16_t*>(ptr);
        std::memset(data_ptr, 0, total_bytes); 
        
        std::cout << "[RetinaState] Allocated " << memory_usage_mb() 
                  << " MB (Aligned " << ALIGNMENT << " bytes)" << std::endl;
    }

    ~RetinaState() {
        if (data_ptr) {
#ifdef _WIN32
            _aligned_free(data_ptr);
#else
            std::free(data_ptr);
#endif
        }
    }

    RetinaState(const RetinaState&) = delete;
    RetinaState& operator=(const RetinaState&) = delete;

    RetinaState(RetinaState&& other) noexcept : data_ptr(other.data_ptr) {
        other.data_ptr = nullptr;
    }

    inline uint16_t* get_pixel_channels(uint32_t x, uint32_t y) {
        if (x >= WIDTH || y >= HEIGHT) return nullptr;
        return &data_ptr[(static_cast<size_t>(y) * WIDTH + x) * CHANNELS];
    }

    // --- 🚀 新增：方便讀取的 API ---
    inline float get_pixel(uint32_t x, uint32_t y, uint32_t c) {
        if (x >= WIDTH || y >= HEIGHT || c >= CHANNELS) return 0.0f;
        uint16_t h = data_ptr[(static_cast<size_t>(y) * WIDTH + x) * CHANNELS + c];
        // 簡易的 half 轉 float (僅用於 DEBUG 打印)
        uint32_t t1 = h & 0x7fff; uint32_t t2 = h & 0x8000; uint32_t t3 = h & 0x7c00;
        t1 <<= 13; t2 <<= 16; t1 += 0x38000000; t1 = (t3 == 0 ? 0 : t1);
        t1 |= t2;
        float f; std::memcpy(&f, &t1, 4);
        return f;
    }

    [[nodiscard]] const uint16_t* data() const { return data_ptr; }
    [[nodiscard]] uint16_t* data() { return data_ptr; } // 🚀 非 const 版本用於下載
    [[nodiscard]] size_t size_bytes() const { 
        return static_cast<size_t>(WIDTH) * HEIGHT * CHANNELS * sizeof(uint16_t); 
    }

    [[nodiscard]] float memory_usage_mb() const {
        return static_cast<float>(size_bytes()) / (1024.0f * 1024.0f);
    }

private:
    uint16_t* data_ptr = nullptr; // 修正：確保 data_ptr 被正確宣告
};

// --- CHIMERA-V 預定義預設配置 ---
using CHIMERA_V_Extreme = RetinaState<512, 512, 2560>;
using CHIMERA_V_Standard = RetinaState<512, 64, 2560>;
using CHIMERA_V_Edge = RetinaState<256, 32, 896>;
