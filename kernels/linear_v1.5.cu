#ifndef _CUSTOM_INT4_LINEAR_CU
#define _CUSTOM_INT4_LINEAR_CU

#include <cuda.h>
#include <stdio.h>

#include "linear_v1.5.h"

void linear_v1_5_launch(
    void *x_packed_d, void *w_packed_d, void *y_d, int m, int n, int k
) {
    using config = LinearConfig<7, 6, 7, 5, 4>;

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

    linear_v1_5_kernel<config><<<gridDim, blockDim>>>(args);
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA FAILED: %s\n", cudaGetErrorString(err));
    }
    else printf("CUDA SUCCESS!\n");

}


#endif