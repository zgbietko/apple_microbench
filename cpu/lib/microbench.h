// cpu/lib/microbench.h
#ifndef MICROBENCH_H
#define MICROBENCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Przepustowość: dst[i] = src[i]
void mem_copy_kernel(float *dst, const float *src, size_t n);

// Pointer-chasing: buf to tablica indeksów [0, n-1], iters = liczba kroków
void pointer_chase_kernel(uint32_t *buf, size_t n, size_t iters);

// Wielowątkowy COPY (pthreads)
void mem_copy_kernel_mt(float *dst, const float *src, size_t n, int num_threads);

// FMA compute: a[i] = a[i] * b[i] + c[i], powtarzane 'iters' razy (1 wątek)
void fma_kernel(float *a, const float *b, const float *c, size_t n, size_t iters);

// Peak FMA: maksymalne obciążenie CPU (wielowątkowo, lokalne dane)
void fma_peak_mt(size_t iters, int num_threads);

#ifdef __cplusplus
}
#endif

#endif // MICROBENCH_H
