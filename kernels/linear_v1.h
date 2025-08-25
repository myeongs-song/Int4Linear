#ifndef _CUSTOM_INT4_LINEAR_H
#define _CUSTOM_INT4_LINEAR_H

#include <cuda.h>

#define LOG2_WARP_SIZE 5

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    int kWarpTileShapeH_,
    int kWarpTileShapeW_,
    int kNInnerIters_
>
struct LinearConfig {
    static constexpr int kWarpTileShapeH = kWarpTileShapeH_;
    static constexpr int kWarpTileShapeW = kWarpTileShapeW_;
    static constexpr int kNInnerIters = kNInnerIters_;
    static constexpr int kBlockTileShapeM = kWarpTileShapeH_ * 16;
    static constexpr int kBlockTileShapeN = kWarpTileShapeW_ * 8;
    static constexpr int kNThreadsPerBlocks = kWarpTileShapeH_ * kWarpTileShapeW_ * 32;
    static constexpr int kBlockTileShapeK = kNInnerIters_ * 64;
    static constexpr int kPackedBlockTileShapeK = kBlockTileShapeK / 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct LinearArgs {
    void * __restrict__ X_ptr; // M x K
    void * __restrict__ W_ptr; // N x K
    void * Y_ptr; 
    int m; 
    int n; 
    int k; 
};

////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct LinearArgs LinearArgs;

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__
void mma_sync_aligned_m16n8k64_rowcol_s4s4s32(int32_t *C, int32_t *A, int32_t *B) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]),
          "r"(C[2]), "r"(C[3]));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename LinearConfig>
__global__ void linear_v1_kernel(LinearArgs args)
{
    constexpr int kWarpTileShapeH = LinearConfig::kWarpTileShapeH;
    constexpr int kWarpTileShapeW = LinearConfig::kWarpTileShapeW;
    constexpr int kBlockTileShapeM = LinearConfig::kBlockTileShapeM;
    constexpr int kBlockTileShapeN = LinearConfig::kBlockTileShapeN;
    constexpr int kBlockTileShapeK = LinearConfig::kBlockTileShapeK;
    constexpr int kPackedBlockTileShapeK = LinearConfig::kPackedBlockTileShapeK;
    constexpr int kNInnerIters = LinearConfig::kNInnerIters;

    __shared__ int smem_x[kBlockTileShapeM][kPackedBlockTileShapeK];
    __shared__ int smem_w[kBlockTileShapeN][kPackedBlockTileShapeK];

    int32_t X_frag[4];
    int32_t W_frag[2];
    int32_t Y_frag[4] = {0, 0, 0, 0};

    int32_t* X_ptr = reinterpret_cast<int32_t*>(args.X_ptr);
    int32_t* W_ptr = reinterpret_cast<int32_t*>(args.W_ptr);
    int32_t* Y_ptr = reinterpret_cast<int32_t*>(args.Y_ptr);

    int warp_id = threadIdx.x >> LOG2_WARP_SIZE;
    int lane_id = threadIdx.x & ~(-1 << LOG2_WARP_SIZE);
    
    int x_mma_row_offset0 = lane_id >> 2;
    int x_mma_row_offset1 = x_mma_row_offset0 + 8;
    int x_mma_col_offset0 = lane_id & 0x3;
    int x_mma_col_offset1 = x_mma_col_offset0 + 4;

    int w_mma_row_offset0 = x_mma_row_offset0;
    int w_mma_col_offset0 = x_mma_col_offset0;
    int w_mma_col_offset1 = x_mma_col_offset1;

    int x_warp_row_offset = (warp_id / kWarpTileShapeW) << 4;
    int w_warp_row_offset = (warp_id % kWarpTileShapeW) << 3;
    
    int n_outer_iters = (args.k + kBlockTileShapeK - 1) / (kBlockTileShapeK);

    constexpr int kNXEntriesPerThread
        = kBlockTileShapeM * kPackedBlockTileShapeK / 
        ((kWarpTileShapeH * kWarpTileShapeW) << LOG2_WARP_SIZE);
    constexpr int kNWEntriesPerThread
        = kBlockTileShapeN * kPackedBlockTileShapeK / 
        ((kWarpTileShapeH * kWarpTileShapeW) << LOG2_WARP_SIZE);

    for (int outer_iter = 0; outer_iter < n_outer_iters; ++outer_iter) {
        // mem -> shmem
        int x_offset = blockIdx.y * kBlockTileShapeM * (args.k>>3) + outer_iter * kPackedBlockTileShapeK;
        int w_offset = blockIdx.x * kBlockTileShapeN * (args.k>>3) + outer_iter * kPackedBlockTileShapeK;
        
        #pragma unroll
        for (int i = 0; i < kNXEntriesPerThread; ++i) {
            int flattened_thread_id = threadIdx.x + i * blockDim.x;
            // TODO: put outbound checking
            int smem_x_row_id = flattened_thread_id / kPackedBlockTileShapeK;
            int smem_x_col_id = flattened_thread_id % kPackedBlockTileShapeK;
            smem_x[smem_x_row_id][smem_x_col_id] = X_ptr[x_offset + (smem_x_row_id*(args.k>>3)) + smem_x_col_id];
        }
        #pragma unroll
        for (int i = 0; i < kNWEntriesPerThread; ++i) {
            int flattened_thread_id = threadIdx.x + i * blockDim.x;
            // TODO: put outbound checking
            int smem_w_row_id = flattened_thread_id / kPackedBlockTileShapeK;
            int smem_w_col_id = flattened_thread_id % kPackedBlockTileShapeK;
            smem_w[smem_w_row_id][smem_w_col_id] = W_ptr[w_offset + (smem_w_row_id*(args.k>>3)) + smem_w_col_id];
        }
        __syncthreads();

        #pragma unroll
        for (int inner_iter = 0; inner_iter < kNInnerIters; ++inner_iter) {
            // shmem -> reg
            X_frag[0] = smem_x[x_warp_row_offset+x_mma_row_offset0][(inner_iter<<3) + x_mma_col_offset0];
            X_frag[1] = smem_x[x_warp_row_offset+x_mma_row_offset1][(inner_iter<<3) + x_mma_col_offset0];
            X_frag[2] = smem_x[x_warp_row_offset+x_mma_row_offset0][(inner_iter<<3) + x_mma_col_offset1];
            X_frag[3] = smem_x[x_warp_row_offset+x_mma_row_offset1][(inner_iter<<3) + x_mma_col_offset1];

            W_frag[0] = smem_w[w_warp_row_offset+w_mma_row_offset0][(inner_iter<<3) + w_mma_col_offset0];
            W_frag[1] = smem_w[w_warp_row_offset+w_mma_row_offset0][(inner_iter<<3) + w_mma_col_offset1];
            
            // compute
            mma_sync_aligned_m16n8k64_rowcol_s4s4s32(Y_frag, X_frag, W_frag);
        }

        __syncthreads();
    }

    // store
    int y_block_row_offset = blockIdx.y * kBlockTileShapeM;
    int y_block_col_offset = blockIdx.x * kBlockTileShapeN;
    int y_warp_row_offset = x_warp_row_offset;
    int y_warp_col_offset = w_warp_row_offset;
    int y_row_idx0 = y_block_row_offset + y_warp_row_offset + (lane_id>>2);
    int y_row_idx1 = y_row_idx0 + 8;
    int y_col_idx0 = y_block_col_offset + y_warp_col_offset + ((lane_id&0x3) << 1);
    int y_col_idx1 = y_col_idx0 + 1;

    __syncthreads();
    
    Y_ptr[y_row_idx0*args.n+y_col_idx0] = Y_frag[0];
    Y_ptr[y_row_idx0*args.n+y_col_idx1] = Y_frag[1];
    Y_ptr[y_row_idx1*args.n+y_col_idx0] = Y_frag[2];
    Y_ptr[y_row_idx1*args.n+y_col_idx1] = Y_frag[3];

    
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void linear_v1_h2w4k4_launch(void *x_packed_d, void *w_packed_d, void *y_d, int m, int n, int k);

////////////////////////////////////////////////////////////////////////////////////////////////////


#endif