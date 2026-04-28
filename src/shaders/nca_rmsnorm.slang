// nca_rmsnorm.slang
import slang;

struct Config {
    uint width, height, channels;
    float dt;
};

[[vk::push_constant]]
Config config;

[[vk::binding(0)]] RWStructuredBuffer<uint4> bufX;         // b_in (x_old)
[[vk::binding(1)]] RWStructuredBuffer<uint4> bufDelta;     // b_out (delta)
[[vk::binding(2)]] RWStructuredBuffer<uint4> bufPrompt;    // d_persistent_field (prompt)
[[vk::binding(3)]] RWStructuredBuffer<uint4> bufOutput;    // b_in (overwrite x_old with x_new)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint pixel_idx = threadId.x;
    if (pixel_idx >= config.width * config.height) return;

    // 🚀 修正 1：強制使用 32-bit float 作為累加器，避免 FP16 超新星爆發 (RMSNorm Overflow)
    float sum_sq = 0.0;
    
    // Pass 1: Compute x_new = x_old + tanh(delta + prompt * 0.5) and accumulate sum_sq
    uint num_uint4 = config.channels / 8;
    uint base_idx = pixel_idx * num_uint4;
    
    for(uint i = 0; i < num_uint4; i++) {
        uint4 x_old_val = bufX[base_idx + i];
        uint4 delta_val = bufDelta[base_idx + i];
        uint4 prompt_val = bufPrompt[base_idx + i];
        
        // Unpack and compute for 8 halfs (4 uints containing 2 halfs each)
        float v[8];
        v[0] = f16tof32(x_old_val.x & 0xFFFF) + tanh(f16tof32(delta_val.x & 0xFFFF) + f16tof32(prompt_val.x & 0xFFFF) * 0.5);
        v[1] = f16tof32(x_old_val.x >> 16) + tanh(f16tof32(delta_val.x >> 16) + f16tof32(prompt_val.x >> 16) * 0.5);
        v[2] = f16tof32(x_old_val.y & 0xFFFF) + tanh(f16tof32(delta_val.y & 0xFFFF) + f16tof32(prompt_val.y & 0xFFFF) * 0.5);
        v[3] = f16tof32(x_old_val.y >> 16) + tanh(f16tof32(delta_val.y >> 16) + f16tof32(prompt_val.y >> 16) * 0.5);
        v[4] = f16tof32(x_old_val.z & 0xFFFF) + tanh(f16tof32(delta_val.z & 0xFFFF) + f16tof32(prompt_val.z & 0xFFFF) * 0.5);
        v[5] = f16tof32(x_old_val.z >> 16) + tanh(f16tof32(delta_val.z >> 16) + f16tof32(prompt_val.z >> 16) * 0.5);
        v[6] = f16tof32(x_old_val.w & 0xFFFF) + tanh(f16tof32(delta_val.w & 0xFFFF) + f16tof32(prompt_val.w & 0xFFFF) * 0.5);
        v[7] = f16tof32(x_old_val.w >> 16) + tanh(f16tof32(delta_val.w >> 16) + f16tof32(prompt_val.w >> 16) * 0.5);
        
        for(int j=0; j<8; j++) sum_sq += v[j] * v[j];
        
        // Store temporarily in bufDelta
        bufDelta[base_idx + i] = uint4(
            (f32tof16(v[1]) << 16) | f32tof16(v[0]),
            (f32tof16(v[3]) << 16) | f32tof16(v[2]),
            (f32tof16(v[5]) << 16) | f32tof16(v[4]),
            (f32tof16(v[7]) << 16) | f32tof16(v[6])
        );
    }
    
    float rms = sqrt(sum_sq / float(config.channels) + 1e-8);
    
    // Pass 2: Normalize and write output
    for(uint i = 0; i < num_uint4; i++) {
        uint4 temp_val = bufDelta[base_idx + i];
        float v[8];
        v[0] = f16tof32(temp_val.x & 0xFFFF) / rms;
        v[1] = f16tof32(temp_val.x >> 16) / rms;
        v[2] = f16tof32(temp_val.y & 0xFFFF) / rms;
        v[3] = f16tof32(temp_val.y >> 16) / rms;
        v[4] = f16tof32(temp_val.z & 0xFFFF) / rms;
        v[5] = f16tof32(temp_val.z >> 16) / rms;
        v[6] = f16tof32(temp_val.w & 0xFFFF) / rms;
        v[7] = f16tof32(temp_val.w >> 16) / rms;
        
        bufOutput[base_idx + i] = uint4(
            (f32tof16(v[1]) << 16) | f32tof16(v[0]),
            (f32tof16(v[3]) << 16) | f32tof16(v[2]),
            (f32tof16(v[5]) << 16) | f32tof16(v[4]),
            (f32tof16(v[7]) << 16) | f32tof16(v[6])
        );
    }
}
