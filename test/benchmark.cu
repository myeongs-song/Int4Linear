//#define _BENCHMARK
#ifdef _BENCHMARK


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#include <cuda.h>

#include "kernels/linear_v1.5.h"
#include "utils/packer.h"


int compare(const void* a, const void* b) { return (*(float*)a > *(float*)b) ? 1 : 0; }


int main(void) {

    srand(time(NULL));

    int m = 8192, n = 8192, k = 8192*2;
    int packed_k = (k+8-1) / 8;

    // Initialize tensors (host-side) 
    int *x_unpacked_h = (int *)malloc(m * sizeof(int) * k);
    int *w_unpacked_h = (int *)malloc(n * sizeof(int) * k);
    int *y_ref = (int *)malloc(m * n * sizeof(int));
    int *y_h = (int *)malloc(m * n * sizeof(int));
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

    using config = LinearConfig<6, 6, 7, 5, 4>;

    constexpr int n_threads = config::kWarpsPerThreadblockM * config::kWarpsPerThreadblockN * (1 << LOG2_WARP_SIZE);
    constexpr int out_tile_size_m = config::kThreadblockShapeM;
    constexpr int out_tile_size_n = config::kThreadblockShapeN;
    int n_blocks_m = (m+out_tile_size_m-1)/out_tile_size_m;
    int n_blocks_n = (n+out_tile_size_n-1)/out_tile_size_n;
    dim3 blockDim(n_threads, 1);
    dim3 gridDim(n_blocks_n, n_blocks_m);

    LinearArgs args = {
        .X_ptr = x_packed_d,
        .W_ptr = w_packed_d,
        .Y_ptr = y_d,
        .m = m, .n = n, .k = k
    };

    // Warp-up phase
    for (int i = 0; i < 5; ++i) {
        linear_v1_5_kernel<config><<<gridDim, blockDim>>>(args);
        cudaDeviceSynchronize();
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
        linear_v1_5_kernel<config><<<gridDim, blockDim>>>(args);
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

    return 0;
}

#endif