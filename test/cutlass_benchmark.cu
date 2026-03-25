#define _BENCHMARK
#ifdef _BENCHMARK


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#include <cuda.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include "utils/packer.h"

int compare(const void* a, const void* b) { return (*(float*)a > *(float*)b) ? 1 : 0; }

int main(void) {

    srand(time(NULL));

    constexpr int m = 32768, n = m, k = 2*m;
    constexpr int packed_k = (k+8-1) / 8;

    // Initialize tensors (host-side) 
    int *x_unpacked_h = (int *)malloc(m * sizeof(int) * k);
    int *w_unpacked_h = (int *)malloc(n * sizeof(int) * k);
    for (int i = 0; i < m*k; ++i) x_unpacked_h[i] = (rand() % 15) - 7;
    for (int i = 0; i < n*k; ++i) w_unpacked_h[i] = (rand() % 15) - 7;

    // Pack tensors
    int *x_packed_h = (int *)malloc(m * sizeof(int) * packed_k);
    int *w_packed_h = (int *)malloc(n * sizeof(int) * packed_k);
    pack(x_unpacked_h, x_packed_h, m, k);
    pack(w_unpacked_h, w_packed_h, n, k);

    // Initialize device-side tensors
    void *x_packed_d, *w_packed_d, *y_d;
    cudaMalloc(&x_packed_d, m*sizeof(int)*packed_k);
    cudaMalloc(&w_packed_d, n*sizeof(int)*packed_k);
    cudaMalloc(&y_d, m*n*sizeof(int));
    cudaMemcpy(x_packed_d, (void*)x_packed_h, m*sizeof(int)*packed_k, cudaMemcpyHostToDevice);
    cudaMemcpy(w_packed_d, (void*)w_packed_h, n*sizeof(int)*packed_k, cudaMemcpyHostToDevice);

    // Init cutlass kernel
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t,
        cutlass::layout::RowMajor,
        cutlass::int4b_t,
        cutlass::layout::ColumnMajor,
        int32_t,
        cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80
    >;
    Gemm gemm_op;

    typename Gemm::Arguments args{
        {m, n, k}, // Gemm Problem Size
        {(cutlass::int4b_t *)x_packed_d, k}, // Tensor A
        {(cutlass::int4b_t *)w_packed_d, k}, // Tensor B
        {(int32_t *)y_d, n}, // Tensor C
        {(int32_t *)y_d, n}, // Tensor D
        {1.0, 0.0} // alpha, beta
    };

    // Warm-up phase
    for (int i = 0; i < 5; ++i) {
        gemm_op(args);
    }

    // Cache config (to clean-up L2 cache)
    int device = 0;
    int l2_size = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
    size_t size_fill = l2_size * 2;
    void *dummy;
    cudaMalloc(&dummy, size_fill);

    // Main loop
    int n_iters = 50;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float *runtimes = (float*)malloc(sizeof(float)*n_iters);

    for (int i = 0; i < n_iters; ++i) {
        // Make L2 cache pure
        cudaMemsetAsync(dummy, 0, size_fill);
        cudaDeviceSynchronize();

        // Get latency
        cudaEventRecord(start, 0);
        gemm_op(args);
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        float cur_ms = 0.0;
        cudaEventElapsedTime(&cur_ms, start, end);

        runtimes[i] = cur_ms;
    }

    qsort((void *)runtimes, n_iters, sizeof(float), &compare);

    double median_ms = (double)runtimes[n_iters/2];
    double mac = 2 * double(m) * double(n) * double(k);
    double tops = mac / (median_ms * 1e9);

    printf("[PERF] %lf TOPS\n", tops);

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(x_packed_d);
    cudaFree(w_packed_d);
    cudaFree(y_d);
    free(x_packed_h);   free(w_packed_h);
    free(x_unpacked_h); free(w_unpacked_h);

    return 0;
}

#endif