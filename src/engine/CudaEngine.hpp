#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include "RetinaState.hpp"

// CUDA 錯誤檢查宏
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// cuBLAS 錯誤檢查宏
#define CUBLAS_CHECK(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", status, file, line);
        exit(status);
    }
}

extern "C" void launch_apply_tanh(__half* d_data, size_t elements, cudaStream_t stream);

class CudaEngine {
public:
    CudaEngine() {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        // 啟用 Tensor Core 加速
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        std::cout << "[CudaEngine] Reverted to cuBLAS + Async Stream." << std::endl;
    }

    ~CudaEngine() {
        cleanup();
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
    }

    template <uint32_t W, uint32_t H, uint32_t C>
    void prepareResources(const RetinaState<W, H, C>& state) {
        size_bytes = state.size_bytes();
        num_pixels = W * H;
        channels = C;
        size_t weight_bytes = (size_t)C * C * sizeof(uint16_t);

        CUDA_CHECK(cudaMalloc(&d_input, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, size_bytes));
        CUDA_CHECK(cudaMalloc(&d_weights, weight_bytes));
        
        CUDA_CHECK(cudaMemset(d_input, 0, size_bytes));
        CUDA_CHECK(cudaMemset(d_output, 0, size_bytes));
        CUDA_CHECK(cudaMemset(d_weights, 0, weight_bytes));
    }

    void upload(const void* data, size_t size) {
        CUDA_CHECK(cudaMemcpyAsync(d_input, data, size, cudaMemcpyHostToDevice, stream));
    }

    void loadWeights(const std::string& path, size_t size) {
        std::vector<uint16_t> hostWeights(size / sizeof(uint16_t));
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Weights file not found: " + path);
        file.read(reinterpret_cast<char*>(hostWeights.data()), size);
        CUDA_CHECK(cudaMemcpyAsync(d_weights, hostWeights.data(), size, cudaMemcpyHostToDevice, stream));
    }

    void evolve(uint32_t w, uint32_t h, uint32_t c) {
        const __half alpha = __float2half(1.0f);
        const __half beta = __float2half(0.0f);

        // 使用 cuBLAS 進行高效矩陣乘法
        CUBLAS_CHECK(cublasHgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            c, num_pixels, c, 
            &alpha, 
            (const __half*)d_weights, c, 
            (const __half*)d_input, c, 
            &beta, 
            (__half*)d_output, c));

        // 應用 tanh 激活函數 (非同步)
        launch_apply_tanh((__half*)d_output, num_pixels * c, stream);

        // 乒乓交換 (非同步安全)
        std::swap(d_input, d_output);
    }

    void sync() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

private:
    cublasHandle_t handle;
    cudaStream_t stream;
    uint16_t *d_input = nullptr, *d_output = nullptr, *d_weights = nullptr;
    size_t size_bytes, num_pixels, channels;

    void cleanup() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_weights) cudaFree(d_weights);
    }
};
