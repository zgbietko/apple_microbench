// gpu/lib/gpubench.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

extern "C" {

static void print_cuda_error(const char* msg, cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpubench] %s: %s\n", msg, cudaGetErrorString(err));
    }
}

int gpu_get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        print_cuda_error("cudaGetDeviceCount failed", err);
        return 0;
    }
    return count;
}

int gpu_get_device_name(int device_id, char* buffer, int buffer_len) {
    if (!buffer || buffer_len <= 0) return -1;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        print_cuda_error("cudaGetDeviceCount failed", err);
        return -1;
    }
    if (device_id < 0 || device_id >= count) {
        fprintf(stderr, "[gpubench] invalid device_id %d (count=%d)\n", device_id, count);
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        print_cuda_error("cudaGetDeviceProperties failed", err);
        return -1;
    }

    std::strncpy(buffer, prop.name, buffer_len - 1);
    buffer[buffer_len - 1] = '\0';
    return 0;
}

static cudaError_t set_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        print_cuda_error("cudaSetDevice failed", err);
    }
    return err;
}

// =======================
//  MEMCOPY BANDWIDTH
// =======================

__global__ void mem_copy_kernel(const float* __restrict__ src,
                                float* __restrict__ dst,
                                size_t n,
                                int iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = src[idx];
    // prosta pętla, żeby trochę pracy zrobić
    for (int it = 0; it < iters; ++it) {
        dst[idx] = val;
        val += 1e-7f;  // drobna zmiana, żeby kompilator nie wywalił
    }
}

/**
 * Uruchamia kernel mem_copy_kernel na GPU i zwraca czas w sekundach.
 * bytes_per_iter – rozmiar bufora (w bajtach) kopiowanego w każdej iteracji.
 * iters          – liczba powtórzeń w kernelu (pętla wewnątrz GPU).
 * device_id      – indeks urządzenia CUDA.
 */
double gpu_mem_copy_elapsed(size_t bytes_per_iter, int iters, int device_id) {
    if (set_device(device_id) != cudaSuccess) {
        return 0.0;
    }

    size_t n = bytes_per_iter / sizeof(float);
    if (n == 0) {
        fprintf(stderr, "[gpubench] gpu_mem_copy_elapsed: n == 0\n");
        return 0.0;
    }

    float* d_src = nullptr;
    float* d_dst = nullptr;

    cudaError_t err = cudaMalloc(&d_src, bytes_per_iter);
    if (err != cudaSuccess) {
        print_cuda_error("cudaMalloc d_src failed", err);
        return 0.0;
    }
    err = cudaMalloc(&d_dst, bytes_per_iter);
    if (err != cudaSuccess) {
        print_cuda_error("cudaMalloc d_dst failed", err);
        cudaFree(d_src);
        return 0.0;
    }

    // jakieś dane początkowe
    err = cudaMemset(d_src, 0, bytes_per_iter);
    print_cuda_error("cudaMemset d_src failed", err);

    dim3 block(256);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaEventRecord(start);
    print_cuda_error("cudaEventRecord(start) failed", err);

    mem_copy_kernel<<<grid, block>>>(d_src, d_dst, n, iters);

    err = cudaEventRecord(stop);
    print_cuda_error("cudaEventRecord(stop) failed", err);

    err = cudaEventSynchronize(stop);
    print_cuda_error("cudaEventSynchronize(stop) failed", err);

    float ms = 0.0f;
    err = cudaEventElapsedTime(&ms, start, stop);
    print_cuda_error("cudaEventElapsedTime failed", err);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_src);
    cudaFree(d_dst);

    double seconds = ms * 1e-3;
    return seconds;
}

// =======================
//  FMA THROUGHPUT
// =======================

__global__ void fma_kernel(float* __restrict__ a,
                           const float* __restrict__ b,
                           const float* __restrict__ c,
                           size_t n,
                           int iters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float va = a[idx];
    float vb = b[idx];
    float vc = c[idx];

    for (int it = 0; it < iters; ++it) {
        // 1 mul + 1 add => 2 FLOP
        va = va * vb + vc;
    }

    a[idx] = va;
}

/**
 * Uruchamia kernel fma_kernel na GPU i zwraca czas w sekundach.
 * n         – długość wektora (liczba floatów).
 * iters     – liczba powtórzeń pętli FMA w kernelu.
 * device_id – indeks urządzenia CUDA.
 */
double gpu_fma_elapsed(size_t n, int iters, int device_id) {
    if (set_device(device_id) != cudaSuccess) {
        return 0.0;
    }

    if (n == 0) {
        fprintf(stderr, "[gpubench] gpu_fma_elapsed: n == 0\n");
        return 0.0;
    }

    size_t bytes = n * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaError_t err = cudaMalloc(&d_a, bytes);
    if (err != cudaSuccess) {
        print_cuda_error("cudaMalloc d_a failed", err);
        return 0.0;
    }
    err = cudaMalloc(&d_b, bytes);
    if (err != cudaSuccess) {
        print_cuda_error("cudaMalloc d_b failed", err);
        cudaFree(d_a);
        return 0.0;
    }
    err = cudaMalloc(&d_c, bytes);
    if (err != cudaSuccess) {
        print_cuda_error("cudaMalloc d_c failed", err);
        cudaFree(d_a);
        cudaFree(d_b);
        return 0.0;
    }

    // Inicjalizacja jakimiś danymi
    err = cudaMemset(d_a, 0, bytes);
    print_cuda_error("cudaMemset d_a failed", err);
    err = cudaMemset(d_b, 0, bytes);
    print_cuda_error("cudaMemset d_b failed", err);
    err = cudaMemset(d_c, 0, bytes);
    print_cuda_error("cudaMemset d_c failed", err);

    dim3 block(256);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    err = cudaEventRecord(start);
    print_cuda_error("cudaEventRecord(start) failed", err);

    fma_kernel<<<grid, block>>>(d_a, d_b, d_c, n, iters);

    err = cudaEventRecord(stop);
    print_cuda_error("cudaEventRecord(stop) failed", err);

    err = cudaEventSynchronize(stop);
    print_cuda_error("cudaEventSynchronize(stop) failed", err);

    float ms = 0.0f;
    err = cudaEventElapsedTime(&ms, start, stop);
    print_cuda_error("cudaEventElapsedTime failed", err);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    double seconds = ms * 1e-3;
    return seconds;
}

} // extern "C"
