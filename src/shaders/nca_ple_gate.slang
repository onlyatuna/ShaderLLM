import slang;

/**
 * ShaderLLM: nca_ple_gate.slang (Dual-Channel Routing Version)
 * 
 * 任務：繞過 Vulkan 4GB maxStorageBufferRange 限制
 * 策略：將 5.6GB PLE Table 切分為 0-20 層與 21-41 層兩個通道
 */

[[vk::binding(0, 0)]] RWStructuredBuffer<half> retinaState;      // (HEIGHT * WIDTH, 2560)
[[vk::binding(1, 0)]] StructuredBuffer<uint> tokenIds;           // (HEIGHT * WIDTH)
// --- 雙通道路由 ---
[[vk::binding(2, 0)]] StructuredBuffer<uint4> pleTable0;         // Layers 0-20 (2.8GB)
[[vk::binding(3, 0)]] StructuredBuffer<uint4> pleTable1;         // Layers 21-41 (2.8GB)
[[vk::binding(4, 0)]] StructuredBuffer<half> layerWeights;       // 閘門權重 (105MB)

static const uint RETINA_DIM = 2560;
static const uint GATE_DIM = 256;
static const uint SUBGROUP_SIZE = 32;
static const uint LAYERS_PER_TABLE = 21; // 42 / 2 = 21

groupshared half s_retina[RETINA_DIM];

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)] 
void main(
    uint3 id : SV_DispatchThreadID,
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID
) {
    uint pixelIdx = groupId.x;
    uint threadIdx = groupThreadId.x;
    uint tokenId = tokenIds[pixelIdx];
    
    // TODO: 透過 PushConstant 傳入真實 layer_idx
    uint layer_idx = 0; 

    // --- 步驟 1: 協同讀取 Retina State ---
    for (uint i = 0; i < 80; ++i) {
        uint dimIdx = i * SUBGROUP_SIZE + threadIdx;
        s_retina[dimIdx] = retinaState[pixelIdx * RETINA_DIM + dimIdx];
    }
    GroupMemoryBarrierWithGroupSync();

    // --- 步驟 2: 計算 Gate 壓縮 ---
    half gate_outputs[8];
    uint gate_offset_base = layer_idx * (256 * 2560 + 2560 * 256 + 2560);
    
    for (uint g = 0; g < 8; ++g) {
        uint outDim = g * SUBGROUP_SIZE + threadIdx;
        float sum = 0.0f;
        uint weight_start = gate_offset_base + outDim * RETINA_DIM;
        for (uint d = 0; d < RETINA_DIM; ++d) {
            sum += (float)s_retina[d] * (float)layerWeights[weight_start + d];
        }
        
        // --- 步驟 3: 雙通道路由注入 PLE 訊號 ---
        uint4 raw_ple;
        uint local_layer_idx;
        
        // 核心路由邏輯
        if (layer_idx < 21) {
            local_layer_idx = layer_idx;
            uint ple_idx_total = (tokenId * LAYERS_PER_TABLE * GATE_DIM) + (local_layer_idx * GATE_DIM) + outDim;
            raw_ple = pleTable0[ple_idx_total / 8];
        } else {
            local_layer_idx = layer_idx - 21;
            uint ple_idx_total = (tokenId * LAYERS_PER_TABLE * GATE_DIM) + (local_layer_idx * GATE_DIM) + outDim;
            raw_ple = pleTable1[ple_idx_total / 8];
        }

        uint ple_idx_in_uint4 = outDim % 8; // 此處 outDim 在各 thread 內是局部偏移
        uint raw_val = raw_ple[ple_idx_in_uint4 / 2];
        uint shift = (ple_idx_in_uint4 % 2) * 16;
        half ple_val = half(f16tof32((raw_val >> shift) & 0xFFFF));
            
        gate_outputs[g] = half(sum) * ple_val;
    }

    // --- 步驟 4: 計算 Projection 升華 ---
    uint proj_offset_base = gate_offset_base + (256 * 2560);
    for (uint i = 0; i < 80; ++i) {
        uint outDim = i * SUBGROUP_SIZE + threadIdx;
        float sum = 0.0f;
        for (uint g = 0; g < 256; ++g) {
            uint srcThread = g % SUBGROUP_SIZE;
            uint srcIdx = g / SUBGROUP_SIZE;
            half g_val = WaveReadLaneAt(gate_outputs[srcIdx], srcThread);
            uint w_idx = proj_offset_base + outDim * 256 + g;
            sum += (float)g_val * (float)layerWeights[w_idx];
        }
        
        // --- 步驟 5: 殘差注入 (加上 Tanh 避震器) ---
        uint targetIdx = pixelIdx * RETINA_DIM + outDim;
        float safe_sum = tanh(sum) * 0.5f; // 抑制異常峰值並限制影響力
        retinaState[targetIdx] += half(safe_sum); 
    }
}
