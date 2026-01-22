// cpu/lib/microbench.c
#include "microbench.h"
#include <pthread.h>

/* --- Jednowątkowy COPY -------------------------------------------------- */

void mem_copy_kernel(float *dst, const float *src, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

/* --- Pointer-chasing ----------------------------------------------------- */

volatile uint32_t pc_sink = 0;

void pointer_chase_kernel(uint32_t *buf, size_t n, size_t iters)
{
    size_t idx = 0;

    for (size_t i = 0; i < iters; i++) {
        idx = buf[idx];
    }

    pc_sink = (uint32_t)idx;
}

/* --- Wielowątkowy COPY na pthreads -------------------------------------- */

typedef struct {
    float       *dst;
    const float *src;
    size_t       start;
    size_t       end;
} copy_task_t;

static void *copy_worker(void *arg)
{
    copy_task_t *t = (copy_task_t *)arg;
    for (size_t i = t->start; i < t->end; i++) {
        t->dst[i] = t->src[i];
    }
    return NULL;
}

void mem_copy_kernel_mt(float *dst, const float *src, size_t n, int num_threads)
{
    if (num_threads <= 1 || n == 0) {
        mem_copy_kernel(dst, src, n);
        return;
    }

    if (num_threads > 64) {
        num_threads = 64; // prosty limit bezpieczeństwa
    }

    pthread_t   threads[64];
    copy_task_t tasks[64];

    size_t base = n / (size_t)num_threads;
    size_t rem  = n % (size_t)num_threads;
    size_t offs = 0;

    for (int t = 0; t < num_threads; ++t) {
        size_t len   = base + (t < (int)rem ? 1 : 0);
        size_t start = offs;
        size_t end   = start + len;

        tasks[t].dst   = dst;
        tasks[t].src   = src;
        tasks[t].start = start;
        tasks[t].end   = end;

        pthread_create(&threads[t], NULL, copy_worker, &tasks[t]);
        offs = end;
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
}

/* --- FMA compute throughput (1 wątek, wektor zewnętrzny) ---------------- */

volatile float fma_sink = 0.0f;

/*
 * Kernel obliczeniowy na zewnętrznych buforach:
 * FLOP count: 2 * n * iters (1 MUL + 1 ADD na element i iterację).
 */
void fma_kernel(float *a, const float *b, const float *c, size_t n, size_t iters)
{
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    size_t unroll = 4;
    size_t limit  = (n / unroll) * unroll;

    for (size_t it = 0; it < iters; ++it) {
        // główna pętla – rozbita na 4 strumienie
        for (size_t i = 0; i < limit; i += unroll) {
            float a0 = a[i + 0];
            float a1 = a[i + 1];
            float a2 = a[i + 2];
            float a3 = a[i + 3];

            float b0 = b[i + 0];
            float b1 = b[i + 1];
            float b2 = b[i + 2];
            float b3 = b[i + 3];

            float c0 = c[i + 0];
            float c1 = c[i + 1];
            float c2 = c[i + 2];
            float c3 = c[i + 3];

            a0 = a0 * b0 + c0;
            a1 = a1 * b1 + c1;
            a2 = a2 * b2 + c2;
            a3 = a3 * b3 + c3;

            a[i + 0] = a0;
            a[i + 1] = a1;
            a[i + 2] = a2;
            a[i + 3] = a3;

            s0 += a0;
            s1 += a1;
            s2 += a2;
            s3 += a3;
        }

        // ogon – jeśli n nie jest wielokrotnością 4
        for (size_t i = limit; i < n; ++i) {
            float v = a[i] * b[i] + c[i];
            a[i] = v;
            s0 += v;
        }
    }

    fma_sink = s0 + s1 + s2 + s3;
}

/* --- Peak FMA: maksymalne obciążenie CPU (wielowątkowo, lokalne dane) --- */

#define FMA_PEAK_N 256  // liczba elementów na wątek (musi zgadzać się z Pythonem)

volatile float fma_peak_sink = 0.0f;

typedef struct {
    size_t iters;
} fma_peak_task_t;

static void *fma_peak_worker(void *arg)
{
    fma_peak_task_t *task = (fma_peak_task_t *)arg;
    size_t iters = task->iters;

    // Mały wektor na stosie – w całości w L1
    float a[FMA_PEAK_N];
    float b[FMA_PEAK_N];
    float c[FMA_PEAK_N];

    for (int i = 0; i < FMA_PEAK_N; ++i) {
        a[i] = 1.0f;
        b[i] = 1.0f + 0.001f * (float)i;
        c[i] = 0.5f;
    }

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float s5 = 0.0f;
    float s6 = 0.0f;
    float s7 = 0.0f;

    const size_t unroll = 8;
    const size_t limit  = (FMA_PEAK_N / unroll) * unroll;

    for (size_t it = 0; it < iters; ++it) {
        for (size_t i = 0; i < limit; i += unroll) {
            float a0 = a[i + 0];
            float a1 = a[i + 1];
            float a2 = a[i + 2];
            float a3 = a[i + 3];
            float a4 = a[i + 4];
            float a5 = a[i + 5];
            float a6 = a[i + 6];
            float a7 = a[i + 7];

            float b0 = b[i + 0];
            float b1 = b[i + 1];
            float b2 = b[i + 2];
            float b3 = b[i + 3];
            float b4 = b[i + 4];
            float b5 = b[i + 5];
            float b6 = b[i + 6];
            float b7 = b[i + 7];

            float c0 = c[i + 0];
            float c1 = c[i + 1];
            float c2 = c[i + 2];
            float c3 = c[i + 3];
            float c4 = c[i + 4];
            float c5 = c[i + 5];
            float c6 = c[i + 6];
            float c7 = c[i + 7];

            a0 = a0 * b0 + c0;
            a1 = a1 * b1 + c1;
            a2 = a2 * b2 + c2;
            a3 = a3 * b3 + c3;
            a4 = a4 * b4 + c4;
            a5 = a5 * b5 + c5;
            a6 = a6 * b6 + c6;
            a7 = a7 * b7 + c7;

            a[i + 0] = a0;
            a[i + 1] = a1;
            a[i + 2] = a2;
            a[i + 3] = a3;
            a[i + 4] = a4;
            a[i + 5] = a5;
            a[i + 6] = a6;
            a[i + 7] = a7;

            s0 += a0;
            s1 += a1;
            s2 += a2;
            s3 += a3;
            s4 += a4;
            s5 += a5;
            s6 += a6;
            s7 += a7;
        }

        // ogon (gdyby FMA_PEAK_N nie było wielokrotnością 8)
        for (size_t i = limit; i < FMA_PEAK_N; ++i) {
            float v = a[i] * b[i] + c[i];
            a[i] = v;
            s0 += v;
        }
    }

    float sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    // zapis do volatile, żeby kompilator nie wyrzucił pętli
    fma_peak_sink += sum;

    return NULL;
}

void fma_peak_mt(size_t iters, int num_threads)
{
    if (num_threads < 1) {
        num_threads = 1;
    }
    if (num_threads > 64) {
        num_threads = 64;
    }

    pthread_t        threads[64];
    fma_peak_task_t tasks[64];

    for (int t = 0; t < num_threads; ++t) {
        tasks[t].iters = iters;
        pthread_create(&threads[t], NULL, fma_peak_worker, &tasks[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
}
