#include <metal_stdlib>
using namespace metal;

// Prosty copy kernel do pomiaru przepustowości pamięci
kernel void mem_copy_kernel(device const float *src [[ buffer(0) ]],
                            device float       *dst [[ buffer(1) ]],
                            uint index              [[ thread_position_in_grid ]]) {
    dst[index] = src[index];
}

// Kernel FMA do pomiaru przepustowości obliczeń.
// Każdy wątek wykonuje inner_iters operacji fma.
kernel void fma_kernel(device const float *a [[ buffer(0) ]],
                       device const float *b [[ buffer(1) ]],
                       device float       *c [[ buffer(2) ]],
                       constant uint      &inner_iters [[ buffer(3) ]],
                       uint index              [[ thread_position_in_grid ]]) {
    float x   = a[index];
    float y   = b[index];
    float acc = 0.0f;

    for (uint i = 0; i < inner_iters; ++i) {
        acc = fma(x, y, acc);
    }

    c[index] = acc;
}
