// src/shaders/nca_ple_gate.slang
// CHIMERA-V M10.9.10: Absolute Isomorphism - Layer-Specific PLE Injection
import slang;

struct PLEConfig {
    uint width, height, channels;
    uint ple_dim;      // 256
    uint layer_idx;    // 0 到 41
};
[[vk::push_constant]] PLEConfig config;

// 綁定
[[vk::binding(0)]] RWStructuredBuffer<half> retinaState;   // 當前 2560 維狀態
[[vk::binding(1)]] StructuredBuffer<half> pleTable;        // 5.6GB (10752維)
[[vk::binding(2)]] StructuredBuffer<half> pleWeights;      // 包含所有層的 Gate/Proj/Norm
[[vk::binding(3)]] StructuredBuffer<uint> tokenHistory;    // 讀取該位置對應的 Token ID

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint pixelIdx = threadId.x;
    if (pixelIdx >= config.width * config.height) return;

    // 取得該像素對應的 Token ID
    uint tokenId = tokenHistory[pixelIdx]; 
    if (tokenId == 0) return; // 無效區域不注入

    // 定位當前層的權重 Offset
    // Gate(256x2560=655360) + Proj(2560x256=655360) + Norm(2560) = 1313280 halfs per layer
    uint w_gate_offset = config.layer_idx * 1313280;
    uint w_proj_offset = w_gate_offset + 655360;
    uint w_norm_offset = w_proj_offset + 655360;
    
    uint ple_offset = tokenId * 10752 + config.layer_idx * config.ple_dim;
    uint state_offset = pixelIdx * config.channels;

    // 1. Gate = retinaState * W_gate (壓縮至 256 維)
    float gate[256];
    for (uint i = 0; i < 256; i++) {
        float sum = 0.0;
        for (uint j = 0; j < 2560; j++) {
            sum += float(retinaState[state_offset + j]) * float(pleWeights[w_gate_offset + i * 2560 + j]);
        }
        gate[i] = sum;
    }
    
    // GELUTanh 激活
    for (uint i = 0; i < 256; i++) {
        float x = gate[i];
        gate[i] = 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));
    }
    
    // 2. 獲取 PLE 並與 Gate 相乘 (Hadamard Product)
    for (uint i = 0; i < 256; i++) {
        gate[i] *= float(pleTable[ple_offset + i]);
    }
    
    // 3. 維度升華: E = M * W_proj (放大回 2560 維)
    float proj[2560];
    for (uint i = 0; i < 2560; i++) {
        float sum = 0.0;
        for (uint j = 0; j < 256; j++) {
            sum += gate[j] * float(pleWeights[w_proj_offset + i * 256 + j]);
        }
        proj[i] = sum;
    }
    
    // 4. 正規化: E_norm = RMSNorm(E)
    float sum_sq = 0.0;
    for (uint i = 0; i < 2560; i++) {
        sum_sq += proj[i] * proj[i];
    }
    float rms = sqrt(sum_sq / 2560.0 + 1e-6);
    
    // 5. 殘差融合
    for (uint i = 0; i < 2560; i++) {
        float norm_val = (proj[i] / rms) * float(pleWeights[w_norm_offset + i]);
        float old_val = float(retinaState[state_offset + i]);
        retinaState[state_offset + i] = half(old_val + norm_val);
    }
}
