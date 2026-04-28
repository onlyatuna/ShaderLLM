// nca_decode.slang - CHIMERA-V M10.9.9: Weight Tying (Dot Product) Decode + Repetition Penalty
import slang;

struct DecodeConfig {
    uint width, height, channels;
    float temperature; 
    float top_p; 
    float penalty; 
};

[[vk::push_constant]]
DecodeConfig config;

[[vk::binding(0)]] RWStructuredBuffer<half> currentState;
[[vk::binding(1)]] RWStructuredBuffer<uint2> bestResult; 
[[vk::binding(2)]] RWStructuredBuffer<uint> globalState; 
[[vk::binding(3)]] RWStructuredBuffer<uint> tokenHistory; 
[[vk::binding(4)]] StructuredBuffer<uint4> embeddingTable; // 🚀 新增: Weight Tying 解碼矩陣

// 🚀 高效 PCG 哈希
uint hash(uint x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

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

groupshared float s_max_vals[256];
groupshared uint s_max_tokens[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_GroupThreadID)
{
    uint currentT = globalState[0];
    uint2 pos = d2xy(64, currentT);
    uint pixelIdx = pos.y * 64 + pos.x;
    uint pixelBase = pixelIdx * config.channels;

    float localMax = -1e20f;
    uint localToken = 256;
    
    uint seed = hash(pixelIdx ^ hash(currentT) ^ threadId.x);
    float u = float(seed % 10000) / 10000.0f + 1e-6f;
    float gumbelNoise = -log(-log(u)) * config.temperature;

    for (uint i = threadId.x; i < 262144; i += 256) {
        if (i < 512) continue; // 🚀 提高門檻，避開所有特殊區塊

        // 🚀 核心修正一：矩陣內積 (Dot Product) 避免特徵空間碰撞
        float val = 0.0f;
        uint emb_base = i * (config.channels / 8);
        uint state_base = pixelIdx * config.channels;

        for(uint c = 0; c < (config.channels / 8); c++) {
            uint4 emb_val = embeddingTable[emb_base + c];
            uint c_half = c * 8;
            
            float s0 = float(currentState[state_base + c_half + 0]);
            float s1 = float(currentState[state_base + c_half + 1]);
            float s2 = float(currentState[state_base + c_half + 2]);
            float s3 = float(currentState[state_base + c_half + 3]);
            float s4 = float(currentState[state_base + c_half + 4]);
            float s5 = float(currentState[state_base + c_half + 5]);
            float s6 = float(currentState[state_base + c_half + 6]);
            float s7 = float(currentState[state_base + c_half + 7]);

            float e0 = f16tof32(emb_val.x & 0xFFFF);
            float e1 = f16tof32(emb_val.x >> 16);
            float e2 = f16tof32(emb_val.y & 0xFFFF);
            float e3 = f16tof32(emb_val.y >> 16);
            float e4 = f16tof32(emb_val.z & 0xFFFF);
            float e5 = f16tof32(emb_val.z >> 16);
            float e6 = f16tof32(emb_val.w & 0xFFFF);
            float e7 = f16tof32(emb_val.w >> 16);

            val += s0 * e0 + s1 * e1 + s2 * e2 + s3 * e3 + s4 * e4 + s5 * e5 + s6 * e6 + s7 * e7;
        }
        
        float p_coeff = 1.0f;
        uint lookback = min(currentT, 128u);
        for (uint h = 0; h < lookback; h++) {
            if (tokenHistory[h] == i) {
                p_coeff = config.penalty; 
                break;
            }
        }
        
        if (val > 0.0f) val /= p_coeff;
        else val *= p_coeff;

        val += gumbelNoise;

        if (val > localMax) {
            localMax = val;
            localToken = i;
        }
    }

    // 🚀 安全的 Reduction 流程
    s_max_vals[threadId.x] = localMax;
    s_max_tokens[threadId.x] = localToken;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (threadId.x < stride) {
            if (s_max_vals[threadId.x + stride] > s_max_vals[threadId.x]) {
                s_max_vals[threadId.x] = s_max_vals[threadId.x + stride];
                s_max_tokens[threadId.x] = s_max_tokens[threadId.x + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (threadId.x == 0) {
        bestResult[0] = uint2(asuint(s_max_vals[0]), s_max_tokens[0]);
        globalState[0] = currentT + 1;
    }
}
