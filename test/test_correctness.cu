#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>

#include "kernels/linear_v3.h"
#include "utils/packer.h"

//#define _DEBUG_LINEAR

void linear_ref(int *x, int *w, int *y, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            y[i*n+j] = 0;
            for (int l = 0; l < k; ++l) y[i*n+j] += x[i*k+l]*w[j*k+l];
        }
    }
}


bool is_same(int *a, int *b, int count) {
    for (int i = 0; i < count; ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

#ifdef _DEBUG_LINEAR

int main(void) {

    srand(time(NULL));

    int m = 512, n = 1024, k = 2048;
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

    linear_v3_launch(x_packed_d, w_packed_d, y_d, m, n, k);

    // Copy the result to the host
    cudaMemcpy(y_h, y_d, m*n*sizeof(int), cudaMemcpyDeviceToHost);

    // Get the reference answer
    linear_ref(x_unpacked_h, w_unpacked_h, y_ref, m, n, k);

    printf("[Ref] ");
    for (int i = 0; i < 32; ++i) printf("%d ", y_ref[i]);
    printf("\n");
    printf("[Ans] ");
    for (int i = 0; i < 32; ++i) printf("%d ", y_h[i]);
    printf("\n");

    if (is_same(y_ref, y_h, m*n)) printf("TEST PASSED!\n");
    else printf("TEST FAILED!\n");

    return 0;
}

#endif