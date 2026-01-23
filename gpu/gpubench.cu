#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// Proste makro do sprawdzania błędów CUDA
static inline bool check_cuda(const char* msg, cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(err));
        return false;
    }
    return true;
}

extern "C" {

// Zwraca liczbę urządzeń CUDA w systemie
int gpu_get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] cudaGetDeviceCount failed: %s\n",
                     cudaGetErrorString(err));
        return 0;
    }
    return count;
}

// Zwraca nazwę urządzenia CUDA o zadanym ID.
// Zwraca 0 przy sukcesie, wartość != 0 przy błędzie.
int gpu_get_device_name(int device_id, char* name, int max_len) {
    if (!name || max_len <= 0) return -1;

    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] cudaGetDeviceProperties failed: %s\n",
                     cudaGetErrorString(err));
        return -1;
    }

    std::strncpy(name, prop.name, max_len - 1);
    name[max_len - 1] = '\0';
    return 0;
}

// ===========================================================
//  Kernel: mem_copy_kernel
//  - prosty kernel kopiujący dane z src do dst
//  - iteruje 'iters' razy po tym samym buforze
// ===========================================================
__global__ void mem_copy_kernel(float* dst, const float* src, size_t n, int iters) {
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (int it = 0; it < iters; ++it) {
        for (size_t i = idx; i < n; i += stride) {
            dst[i] = src[i];
        }
    }
}

// Zwraca czas (w sekundach) wykonania 'iters' iteracji na buforze 'bytes'.
// bytes – rozmiar bufora w bajtach (float32).
// device_id – ID urządzenia CUDA.
// iters – liczba iteracji w kernelu.
double gpu_mem_copy_elapsed(size_t bytes, int device_id, int iters) {
    if (bytes == 0 || iters <= 0) return 0.0;

    cudaError_t err = cudaSetDevice(device_id);
    if (!check_cuda("cudaSetDevice", err)) return -1.0;

    size_t n = bytes / sizeof(float);
    if (n == 0) return -1.0;

    float* d_src = nullptr;
    float* d_dst = nullptr;

    err = cudaMalloc(&d_src, n * sizeof(float));
    if (!check_cuda("cudaMalloc d_src", err)) return -1.0;

    err = cudaMalloc(&d_dst, n * sizeof(float));
    if (!check_cuda("cudaMalloc d_dst", err)) {
        cudaFree(d_src);
        return -1.0;
    }

    // Proste zerowanie (nie wpływa istotnie na pomiar)
    cudaMemset(d_src, 0, n * sizeof(float));
    cudaMemset(d_dst, 0, n * sizeof(float));

    int blockSize = 256;
    int gridSize  = static_cast<int>((n + blockSize - 1) / blockSize);
    if (gridSize <= 0) gridSize = 1;
    if (gridSize > 65535) gridSize = 65535;

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    mem_copy_kernel<<<gridSize, blockSize>>>(d_dst, d_src, n, iters);

    err = cudaGetLastError();
    if (!check_cuda("mem_copy_kernel launch", err)) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_src);
        cudaFree(d_dst);
        return -1.0;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);

    double seconds = static_cast<double>(ms) / 1000.0;
    return seconds;
}

// ===========================================================
//  Kernel: fma_kernel
//  a[i] = fmaf(a[i], b[i], c[i]) wielokrotnie
// ===========================================================
__global__ void fma_kernel(float* a, const float* b, const float* c,
                           size_t n, int iters) {
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (int it = 0; it < iters; ++it) {
        for (size_t i = idx; i < n; i += stride) {
            a[i] = fmaf(a[i], b[i], c[i]);  // 2 FLOP na operację
        }
    }
}

// gpu_fma_elapsed:
//   n        – liczba elementów (float32)
//   device_id – ID urządzenia CUDA
//   iters    – liczba iteracji w kernelu
// Zwraca czas w sekundach.
double gpu_fma_elapsed(size_t n, int device_id, int iters) {
    if (n == 0 || iters <= 0) return 0.0;

    cudaError_t err = cudaSetDevice(device_id);
    if (!check_cuda("cudaSetDevice", err)) return -1.0;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    err = cudaMalloc(&d_a, n * sizeof(float));
    if (!check_cuda("cudaMalloc d_a", err)) return -1.0;

    err = cudaMalloc(&d_b, n * sizeof(float));
    if (!check_cuda("cudaMalloc d_b", err)) {
        cudaFree(d_a);
        return -1.0;
    }

    err = cudaMalloc(&d_c, n * sizeof(float));
    if (!check_cuda("cudaMalloc d_c", err)) {
        cudaFree(d_a);
        cudaFree(d_b);
        return -1.0;
    }

    cudaMemset(d_a, 0, n * sizeof(float));
    cudaMemset(d_b, 0, n * sizeof(float));
    cudaMemset(d_c, 0, n * sizeof(float));

    int blockSize = 256;
    int gridSize  = static_cast<int>((n + blockSize - 1) / blockSize);
    if (gridSize <= 0) gridSize = 1;
    if (gridSize > 65535) gridSize = 65535;

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    fma_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n, iters);

    err = cudaGetLastError();
    if (!check_cuda("fma_kernel launch", err)) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -1.0;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    double seconds = static_cast<double>(ms) / 1000.0;
    return seconds;
}

} // extern "C"
