#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

/**
 * @brief 高效點對點 tanh 激活核函式 (使用 half2 向量化優化)
 */
__global__ void apply_tanh_kernel_vec(__half2* data, size_t num_half2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_half2) {
        float2 val = __half22float2(data[idx]);
        val.x = tanhf(val.x);
        val.y = tanhf(val.y);
        data[idx] = __float22half2_rn(val);
    }
}

// 保留標量版作為邊界處理
__global__ void apply_tanh_kernel_scalar(__half* data, size_t elements, size_t start_idx) {
    size_t idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(tanhf(val));
    }
}

extern "C" void launch_apply_tanh(__half* d_data, size_t elements, cudaStream_t stream) {
    size_t num_half2 = elements / 2;
    if (num_half2 > 0) {
        uint32_t threads = 256;
        uint32_t blocks = (uint32_t)((num_half2 + threads - 1) / threads);
        apply_tanh_kernel_vec<<<blocks, threads, 0, stream>>>((__half2*)d_data, num_half2);
    }
    
    // 處理奇數剩餘元素
    if (elements % 2 != 0) {
        apply_tanh_kernel_scalar<<<1, 1, 0, stream>>>(d_data, elements, elements - 1);
    }
}

// 供舊版或測試使用的 NCA Kernel (可視需求保留)
extern "C" void launch_nca_evolve(
    const uint16_t* d_input, uint16_t* d_output, const uint16_t* d_weights,
    uint32_t w, uint32_t h, uint32_t c, cudaStream_t stream) 
{
    // 在 cuBLAS 模式下，此函數不被 CudaEngine 直接呼叫
}
