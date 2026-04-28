#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;

int main() {
    std::cout << "========== CHIMERA-V M3: Intel oneMKL Extreme Benchmark ==========" << std::endl;

    device dev = device(gpu_selector_v);
    queue q(dev, property::queue::in_order());
    std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

    const uint32_t W = 512, H = 64, C = 2560;
    const uint32_t num_pixels = W * H;

    // 分配 USM 共享記憶體
    float* d_input = malloc_shared<float>(num_pixels * C, q);
    float* d_weights = malloc_shared<float>(C * C, q);
    float* d_output = malloc_shared<float>(num_pixels * C, q);

    // 初始化
    q.fill(d_input, 1.0f, num_pixels * C);
    q.fill(d_weights, 0.01f, C * C); // 使用較小權重防止 tanh 飽和過快
    q.wait();

    std::cout << "[oneMKL] Executing Optimized GEMM (32768 x 2560 x 2560)..." << std::endl;

    // --- oneMKL 核心呼叫 ---
    // 計算 C = alpha * A * B + beta * C
    // A [C, C], B [C, num_pixels], Output [C, num_pixels]
    float alpha = 1.0f;
    float beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();

    oneapi::mkl::blas::column_major::gemm(q, 
        oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        C, num_pixels, C,
        alpha, d_weights, C,
        d_input, C,
        beta, d_output, C).wait();

    auto end = std::chrono::high_resolution_clock::now();

    // 數值驗證
    std::cout << "\n--- oneMKL Numerical Verification ---" << std::endl;
    std::cout << "[Check] Output[0]: " << d_output[0] << std::endl;

    std::chrono::duration<double> diff = end - start;
    double t = diff.count();
    std::cout << "Avg Time: " << t * 1000.0 << " ms | MKL Perf: " << (2.0*W*H*C*C/1e12)/t << " TFLOPS" << std::endl;

    free(d_input, q);
    free(d_weights, q);
    free(d_output, q);

    return 0;
}
