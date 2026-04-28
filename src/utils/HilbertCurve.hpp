#pragma once
#include <cstdint>
#include <algorithm>

/**
 * @brief Hilbert Curve 座標轉換工具
 * 用於保持 1D Token 序列在 2D 空間中的局部關聯性。
 */
class HilbertCurve {
public:
    struct Point {
        uint32_t x;
        uint32_t y;
    };

    // 將 1D 索引轉為 2D 座標 (適用於 2^n 的正方形空間)
    static Point indexToPoint(uint32_t n, uint32_t d) {
        Point p{0, 0};
        uint32_t t = d;
        for (uint32_t s = 1; s < n; s <<= 1) {
            uint32_t rx = 1 & (t / 2);
            uint32_t ry = 1 & (t ^ rx);
            rotate(s, &p.x, &p.y, rx, ry);
            p.x += s * rx;
            p.y += s * ry;
            t /= 4;
        }
        return p;
    }

private:
    static void rotate(uint32_t n, uint32_t* x, uint32_t* y, uint32_t rx, uint32_t ry) {
        if (ry == 0) {
            if (rx == 1) {
                *x = n - 1 - *x;
                *y = n - 1 - *y;
            }
            std::swap(*x, *y);
        }
    }
};
