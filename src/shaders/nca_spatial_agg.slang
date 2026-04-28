// nca_spatial_agg.slang - CHIMERA-V M10.9.8: Pass 1 (True Im2Col Spatial Aggregator)
import slang;

struct Config {
    uint width, height, channels;
    float dt;
    uint unused;
    uint active_mass;
};

[[vk::push_constant]]
Config config;

[[vk::binding(0)]] RWStructuredBuffer<uint4> bufIn;
[[vk::binding(1)]] RWStructuredBuffer<uint4> bufSpatialAgg; 

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint pixelIdx = threadId.x / (config.channels / 8);
    uint chUint4Idx = threadId.x % (config.channels / 8);
    
    if (pixelIdx >= config.width * config.height) return;

    int px = int(pixelIdx % config.width);
    int py = int(pixelIdx / config.width);
    
    uint n_idx = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = (px + dx + int(config.width)) % int(config.width);
            int ny = (py + dy + int(config.height)) % int(config.height);
            uint neighborIdx = uint(ny) * config.width + uint(nx);
            
            uint4 data = bufIn[neighborIdx * (config.channels / 8) + chUint4Idx];
            
            // 寫入對應的 9 倍展開空間 (Im2Col)
            uint outIdx = pixelIdx * (9 * config.channels / 8) + n_idx * (config.channels / 8) + chUint4Idx;
            bufSpatialAgg[outIdx] = data;
            n_idx++;
        }
    }
}
