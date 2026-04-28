// nca_inject.slang - CHIMERA-V M10.9.9: Grid-Stride Loop Injection
import slang;

struct InjectConfig {
    uint channels;
    uint unused1;
    uint unused2;
    uint unused3;
};

[[vk::push_constant]]
InjectConfig config;

[[vk::binding(0)]] StructuredBuffer<uint4> embeddingTable;
[[vk::binding(1)]] RWStructuredBuffer<uint4> retinaState;
[[vk::binding(2)]] StructuredBuffer<uint2> decodeResult; // .y = tokenId
[[vk::binding(3)]] StructuredBuffer<uint> globalState;   
[[vk::binding(4)]] RWStructuredBuffer<uint4> persistentField;

// 🚀 GPU 內建 Hilbert 轉換
uint2 d2xy(uint n, uint d) {
    uint t = d;
    uint x = 0, y = 0;
    for (uint s = 1; s < n; s <<= 1) {
        uint rx = 1 & (t / 2);
        uint ry = 1 & (t ^ rx);
        if (ry == 0) {
            if (rx == 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            uint temp = x; x = y; y = temp;
        }
        x += s * rx;
        y += s * ry;
        t /= 4;
    }
    return uint2(x, y);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint totalUint4PerToken = config.channels / 8;
    
    // 1. 獲取剛剛解碼出的 Token
    uint tokenId = decodeResult[0].y;
    
    // 2. 獲取當前時間座標
    uint t = globalState[0] - 1;
    uint2 pos = d2xy(64, t);
    uint pixelIdx = pos.y * 64 + pos.x;

    // 🚀 核心修正二：Grid-Stride Loop 保證 2560 通道完全寫入
    for (uint chIdx = threadId.x; chIdx < totalUint4PerToken; chIdx += 256) {
        uint srcIdx = (tokenId) * totalUint4PerToken + chIdx;
        uint dstIdx = pixelIdx * totalUint4PerToken + chIdx;

        // 🚀 核心優化：Additive Injection
        uint4 srcData = embeddingTable[srcIdx];
        uint4 dstData = retinaState[dstIdx];
        
        uint4 result;
        for(int i=0; i<4; i++) {
            uint s_low = srcData[i] & 0xFFFF;
            uint s_high = srcData[i] >> 16;
            uint d_low = dstData[i] & 0xFFFF;
            uint d_high = dstData[i] >> 16;
            
            // 訊號強度已經在 Embedding 匯出時經過 sqrt(hidden_size) 放大，恢復純粹加法
            float res_low = f16tof32(s_low) + f16tof32(d_low);
            float res_high = f16tof32(s_high) + f16tof32(d_high);
            
            result[i] = (f32tof16(res_high) << 16) | f32tof16(res_low);
        }

        retinaState[dstIdx] = result;
        persistentField[dstIdx] = result;
    }
}
