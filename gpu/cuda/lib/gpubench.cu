#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>

extern "C" {

/**
 * Returns the number of CUDA devices visible to the process.
 * On success returns device count (>=0).
 * On failure returns negative cudaError_t code.
 */
int gpu_cuda_get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return -static_cast<int>(err);
    }
    return count;
}

/**
 * Writes device name to name_buf (null-terminated).
 * buf_len is the size of the buffer in bytes.
 * Returns 0 on success, or negative cudaError_t code on failure.
 */
int gpu_cuda_get_device_name(int device_index, char* name_buf, int buf_len) {
    if (!name_buf || buf_len <= 0) {
        return -1;
    }
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_index);
    if (err != cudaSuccess) {
        return -static_cast<int>(err);
    }
    std::snprintf(name_buf, static_cast<std::size_t>(buf_len), "%s", prop.name);
    return 0;
}

/**
 * Returns basic device properties:
 *  - compute capability (major / minor)
 *  - global memory size in bytes
 *
 * Any of the output pointers may be nullptr if the caller is not
 * interested in that field.
 *
 * Returns 0 on success, or negative cudaError_t code on failure.
 */
int gpu_cuda_get_device_props(
    int device_index,
    int* cc_major,
    int* cc_minor,
    std::size_t* global_mem_bytes
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_index);
    if (err != cudaSuccess) {
        return -static_cast<int>(err);
    }
    if (cc_major) {
        *cc_major = prop.major;
    }
    if (cc_minor) {
        *cc_minor = prop.minor;
    }
    if (global_mem_bytes) {
        *global_mem_bytes = prop.totalGlobalMem;
    }
    return 0;
}

__global__ void mem_copy_kernel(float* dst, const float* src, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/**
 * Measures global-memory bandwidth using a simple device-to-device
 * copy kernel. The function:
 *  - sets the active device,
 *  - allocates two device buffers of size `bytes`,
 *  - runs mem_copy_kernel `iters` times,
 *  - measures elapsed time using CUDA events,
 *  - frees resources.
 *
 * On success returns 0 and writes elapsed time [ms] to *elapsed_ms_out.
 * On failure returns negative cudaError_t code.
 */
int gpu_cuda_memcpy_bandwidth(
    int device_index,
    std::size_t bytes,
    int iters,
    double* elapsed_ms_out
) {
    if (!elapsed_ms_out) {
        return -1;
    }
    if (iters <= 0) {
        return -2;
    }

    cudaError_t err = cudaSetDevice(device_index);
    if (err != cudaSuccess) {
        return -static_cast<int>(err);
    }

    std::size_t n = bytes / sizeof(float);
    if (n == 0) {
        n = 1;
    }

    float* d_src = nullptr;
    float* d_dst = nullptr;
    err = cudaMalloc(&d_src, n * sizeof(float));
    if (err != cudaSuccess) {
        return -static_cast<int>(err);
    }
    err = cudaMalloc(&d_dst, n * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_src);
        return -static_cast<int>(err);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x));
    if (grid.x == 0) {
        grid.x = 1;
    }
    if (grid.x > 65535) {
        grid.x = 65535;
    }

    // Warm-up
    mem_copy_kernel<<<grid, block>>>(d_dst, d_src, n);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        return -static_cast<int>(err);
    }

    // Timed loop
    cudaEventRecord(start, nullptr);
    for (int i = 0; i < iters; ++i) {
        mem_copy_kernel<<<grid, block>>>(d_dst, d_src, n);
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    *elapsed_ms_out = static_cast<double>(ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}

__global__ void fma_kernel(
    const float* a,
    const float* b,
    float* c,
    std::size_t n,
    int iters_inner
) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float av = a[idx];
    float bv = b[idx];
    float cv = c[idx];

    // Each iteration performs one fused multiply-add.
    for (int k = 0; k < iters_inner; ++k) {
        cv = fmaf(av, bv, cv);
    }

    c[idx] = cv;
}

/**
 * Measures FMA throughput:
 *  - sets device,
 *  - allocates three device buffers (a, b, c) of length n,
 *  - launches fma_kernel once with the requested iters_inner,
 *  - measures elapsed time using CUDA events,
 *  - frees resources.
 *
 * On success returns 0 and writes elapsed time [ms] to *elapsed_ms_out.
 * On failure returns negative cudaError_t code.
 */
int gpu_cuda_fma_throughput(
    int device_index,
    std::size_t n,
    int iters_inner,
    double* elapsed_ms_out
) {
    if (!elapsed_ms_out) {
        return -1;
    }
    if (iters_inner <= 0) {
        return -2;
    }

    cudaError_t err = cudaSetDevice(device_index);
    if (err != cudaSuccess) {
        return -static_cast<int>(err);
    }

    std::size_t bytes = n * sizeof(float);
    if (bytes == 0) {
        bytes = sizeof(float);
        n = 1;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    err = cudaMalloc(&d_a, bytes);
    if (err != cudaSuccess) {
        return -static_cast<int>(err);
    }
    err = cudaMalloc(&d_b, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        return -static_cast<int>(err);
    }
    err = cudaMalloc(&d_c, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return -static_cast<int>(err);
    }

    dim3 block(256);
    dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x));
    if (grid.x == 0) {
        grid.x = 1;
    }
    if (grid.x > 65535) {
        grid.x = 65535;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    fma_kernel<<<grid, block>>>(d_a, d_b, d_c, n, iters_inner);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return -static_cast<int>(err);
    }

    // Timed run
    cudaEventRecord(start, nullptr);
    fma_kernel<<<grid, block>>>(d_a, d_b, d_c, n, iters_inner);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    *elapsed_ms_out = static_cast<double>(ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

} // extern "C"
