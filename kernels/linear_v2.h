#ifndef _CUSTOM_INT4_LINEAR_H
#define _CUSTOM_INT4_LINEAR_H

#include <cuda.h>
#include <cuda_pipeline.h>

#define LOG2_WARP_SIZE 5
#define cdiv(a, b) ((a) + (b) - 1) / (b)

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    int kLog2ThreadblockShapeM_, int kLog2ThreadblockShapeN_, int kLog2ThreadblockShapeK_,
    int kLog2WarpShapeM_, int kLog2WarpShapeN_>
struct LinearConfig {
    static constexpr int kThreadblockShapeM = (1 << kLog2ThreadblockShapeM_);
    static constexpr int kThreadblockShapeN = (1 << kLog2ThreadblockShapeN_);
    static constexpr int kThreadblockShapeK = (1 << kLog2ThreadblockShapeK_);
    static constexpr int kThreadblockShapeKPacked = (1 << (kLog2ThreadblockShapeK_ - 3));
 
    static constexpr int kWarpShapeM = (1 << kLog2WarpShapeM_);
    static constexpr int kWarpShapeN = (1 << kLog2WarpShapeN_);

    static constexpr int kWarpsPerThreadblockM = (1 << (kLog2ThreadblockShapeM_ - kLog2WarpShapeM_));
    static constexpr int kWarpsPerThreadblockN = (1 << (kLog2ThreadblockShapeN_ - kLog2WarpShapeN_));

    static constexpr int kInstsPerWarpM = (1 << (kLog2WarpShapeM_ - 4));
    static constexpr int kInstsPerWarpN = (1 << (kLog2WarpShapeN_ - 3));
    static constexpr int kInstsPerWarpK = (1 << (kLog2ThreadblockShapeK_ - 6)); // kNInnerIters
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

template <unsigned CopySize>
__device__ __forceinline__ void cp_async_cg_shared_global(void* smem_dst, const void* gmem_src) {
    static_assert(CopySize == 4 || CopySize == 8 || CopySize == 16, "Unsupported copy size.");
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    uint64_t gmem_addr = __cvta_generic_to_global(gmem_src);
    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n" : : "r"(smem_addr), "l"(gmem_addr), "n"(CopySize));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile ("cp.async.commit_group;\n");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n" : : "n"(N));
}

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

__device__ __forceinline__
void ldmatrix_sync_aligned_m8n8_x2_b16(int32_t *reg, uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,  %1}, [%2];\n"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(addr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__
void ldmatrix_sync_aligned_m8n8_x4_b16(int32_t *reg, uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,  %1,  %2,  %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename LinearConfig>
__global__ void linear_v2_kernel(LinearArgs args)
{
    constexpr int kThreadblockShapeM = LinearConfig::kThreadblockShapeM;
    constexpr int kThreadblockShapeN = LinearConfig::kThreadblockShapeN;
    constexpr int kThreadblockShapeK = LinearConfig::kThreadblockShapeK;
    constexpr int kThreadblockShapeKPacked = LinearConfig::kThreadblockShapeKPacked;
 
    constexpr int kWarpShapeM = LinearConfig::kWarpShapeM;
    constexpr int kWarpShapeN = LinearConfig::kWarpShapeN;

    constexpr int kWarpsPerThreadblockM = LinearConfig::kWarpsPerThreadblockM;
    constexpr int kWarpsPerThreadblockN = LinearConfig::kWarpsPerThreadblockN;

    constexpr int kInstsPerWarpM = LinearConfig::kInstsPerWarpM;
    constexpr int kInstsPerWarpN = LinearConfig::kInstsPerWarpN;
    constexpr int kInstsPerWarpK = LinearConfig::kInstsPerWarpK;

    __shared__ int smem_x[2][kThreadblockShapeM][kThreadblockShapeKPacked+4];
    __shared__ int smem_w[2][kThreadblockShapeN][kThreadblockShapeKPacked+4];

    int32_t X_frag[4];
    int32_t W_frag[2];
    int32_t Y_frag[kInstsPerWarpM][kInstsPerWarpN][4];
    memset(Y_frag, 0, sizeof(Y_frag));

    int32_t* __restrict__ X_ptr = reinterpret_cast<int32_t*>(args.X_ptr);
    int32_t* __restrict__ W_ptr = reinterpret_cast<int32_t*>(args.W_ptr);
    int32_t* Y_ptr = reinterpret_cast<int32_t*>(args.Y_ptr);

    int warp_id = threadIdx.x >> LOG2_WARP_SIZE;
    int lane_id = threadIdx.x & ~(-1 << LOG2_WARP_SIZE);
    
    int x_warp_row_offset = (warp_id / kWarpsPerThreadblockN) * kWarpShapeM; 
    int w_warp_row_offset = (warp_id % kWarpsPerThreadblockN) * kWarpShapeN;
    
    int x_mma_row_offset = x_warp_row_offset + (lane_id & 0xf);
    int x_mma_col_offset = (lane_id >> 4) << 2;
    int w_mma_row_offset = w_warp_row_offset + (lane_id & 0x7);
    int w_mma_col_offset = (lane_id >> 3) << 2;
    
    int n_outer_iters = (args.k + kThreadblockShapeK - 1) / (kThreadblockShapeK);

    constexpr int kNThreads = (kWarpsPerThreadblockM * kWarpsPerThreadblockN) << LOG2_WARP_SIZE;
    constexpr int kNXCopyIters = cdiv(kThreadblockShapeM * kThreadblockShapeKPacked, kNThreads*4);
    constexpr int kNWCopyIters = cdiv(kThreadblockShapeN * kThreadblockShapeKPacked, kNThreads*4);
    
    int x_offset = blockIdx.y * kThreadblockShapeM * (args.k>>3);
    int w_offset = blockIdx.x * kThreadblockShapeN * (args.k>>3);
            
    #pragma unroll
    for (int i = 0; i < kNXCopyIters; ++i) {
        int flattened_thread_id = threadIdx.x + i * blockDim.x;
        int smem_x_row_id = flattened_thread_id / (kThreadblockShapeKPacked/4);
        int smem_x_col_id = (flattened_thread_id % (kThreadblockShapeKPacked/4))<<2;
        if (smem_x_row_id < kThreadblockShapeM) { // Out-of-bounds check
            cp_async_cg_shared_global<16>(
                &smem_x[0][smem_x_row_id][smem_x_col_id], 
                &X_ptr[x_offset + (smem_x_row_id*(args.k>>3)) + smem_x_col_id]);
        }
    }
    #pragma unroll
    for (int i = 0; i < kNWCopyIters; ++i) {
        int flattened_thread_id = threadIdx.x + i * blockDim.x;
        int smem_w_row_id = flattened_thread_id / (kThreadblockShapeKPacked/4);
        int smem_w_col_id = (flattened_thread_id % (kThreadblockShapeKPacked/4))<<2;
        if (smem_w_row_id < kThreadblockShapeN) { // Out-of-bounds check
            cp_async_cg_shared_global<16>(
                &smem_w[0][smem_w_row_id][smem_w_col_id], 
                &W_ptr[w_offset + (smem_w_row_id*(args.k>>3)) + smem_w_col_id]);
        }
    }
    cp_async_commit_group();

    for (int outer_iter = 1; outer_iter < n_outer_iters; ++outer_iter) {
        int load_buf_idx = outer_iter & 0x1;
        int compute_buf_idx = (outer_iter - 1) & 0x1;

        x_offset = blockIdx.y * kThreadblockShapeM * (args.k>>3) + outer_iter * kThreadblockShapeKPacked;
        w_offset = blockIdx.x * kThreadblockShapeN * (args.k>>3) + outer_iter * kThreadblockShapeKPacked;
        
        cp_async_wait_group<0>();
        __syncthreads();
        #pragma unroll
        for (int mid = 0; mid < kInstsPerWarpM; ++mid) {
            int x_inst_row_offset = x_mma_row_offset + (mid<<4);
            #pragma unroll
            for (int nid = 0; nid < kInstsPerWarpN; ++nid) {
                int w_inst_row_offset = w_mma_row_offset + (nid<<3);
                #pragma unroll
                for (int kid = 0; kid < kInstsPerWarpK; ++kid) {
                    // shmem -> reg
                    uint32_t x_addr = __cvta_generic_to_shared(&(smem_x[compute_buf_idx][x_inst_row_offset][x_mma_col_offset + (kid<<3)]));
                    uint32_t w_addr = __cvta_generic_to_shared(&(smem_w[compute_buf_idx][w_inst_row_offset][w_mma_col_offset + (kid<<3)]));
                    ldmatrix_sync_aligned_m8n8_x4_b16(X_frag, x_addr);
                    ldmatrix_sync_aligned_m8n8_x2_b16(W_frag, w_addr);
                    // compute
                    mma_sync_aligned_m16n8k64_rowcol_s4s4s32(Y_frag[mid][nid], X_frag, W_frag);
                }
            }
        }
        
        #pragma unroll
        for (int i = 0; i < kNXCopyIters; ++i) {
            int flattened_thread_id = threadIdx.x + i * blockDim.x;
            int smem_x_row_id = flattened_thread_id / (kThreadblockShapeKPacked/4);
            int smem_x_col_id = (flattened_thread_id % (kThreadblockShapeKPacked/4))<<2;
            if (smem_x_row_id < kThreadblockShapeM) { // Out-of-bounds check
                cp_async_cg_shared_global<16>(
                    &smem_x[load_buf_idx][smem_x_row_id][smem_x_col_id], 
                    &X_ptr[x_offset + (smem_x_row_id*(args.k>>3)) + smem_x_col_id]);
            }
        }
        #pragma unroll
        for (int i = 0; i < kNWCopyIters; ++i) {
            int flattened_thread_id = threadIdx.x + i * blockDim.x;
            int smem_w_row_id = flattened_thread_id / (kThreadblockShapeKPacked/4);
            int smem_w_col_id = (flattened_thread_id % (kThreadblockShapeKPacked/4))<<2;
            if (smem_w_row_id < kThreadblockShapeN) { // Out-of-bounds check
                cp_async_cg_shared_global<16>(
                    &smem_w[load_buf_idx][smem_w_row_id][smem_w_col_id], 
                    &W_ptr[w_offset + (smem_w_row_id*(args.k>>3)) + smem_w_col_id]);
            }
        }
        cp_async_commit_group();
    }

    int last_buf_idx = (n_outer_iters - 1) % 2;
    cp_async_wait_group<0>();
    __syncthreads();
    #pragma unroll
    for (int mid = 0; mid < kInstsPerWarpM; ++mid) {
        int x_inst_row_offset = x_mma_row_offset + (mid<<4);
        #pragma unroll
        for (int nid = 0; nid < kInstsPerWarpN; ++nid) {
            int w_inst_row_offset = w_mma_row_offset + (nid<<3);
            #pragma unroll
            for (int kid = 0; kid < kInstsPerWarpK; ++kid) {
                // shmem -> reg
                uint32_t x_addr = __cvta_generic_to_shared(&(smem_x[last_buf_idx][x_inst_row_offset][x_mma_col_offset + (kid<<3)]));
                uint32_t w_addr = __cvta_generic_to_shared(&(smem_w[last_buf_idx][w_inst_row_offset][w_mma_col_offset + (kid<<3)]));
                ldmatrix_sync_aligned_m8n8_x4_b16(X_frag, x_addr);
                ldmatrix_sync_aligned_m8n8_x2_b16(W_frag, w_addr);
                // compute
                mma_sync_aligned_m16n8k64_rowcol_s4s4s32(Y_frag[mid][nid], X_frag, W_frag);
            }
        }
    }
    
    int y_block_row_offset = blockIdx.y * kThreadblockShapeM;
    int y_block_col_offset = blockIdx.x * kThreadblockShapeN;
    int y_warp_row_offset = y_block_row_offset + x_warp_row_offset;
    int y_warp_col_offset = y_block_col_offset + w_warp_row_offset;

    #pragma unroll
    for (int mid = 0; mid < kInstsPerWarpM; ++mid) {
        int y_inst_row_offset = y_warp_row_offset + mid * 16;
        for (int nid = 0; nid < kInstsPerWarpN; ++nid) {
            int y_inst_col_offset = y_warp_col_offset + nid * 8;

            int y_row_idx0 = y_inst_row_offset + (lane_id>>2);
            int y_row_idx1 = y_row_idx0 + 8;
            int y_col_idx0 = y_inst_col_offset + ((lane_id&0x3) << 1);
            int y_col_idx1 = y_col_idx0 + 1;

            if (y_row_idx0 < args.m && y_col_idx0 < args.n) Y_ptr[y_row_idx0*args.n+y_col_idx0] = Y_frag[mid][nid][0];
            if (y_row_idx0 < args.m && y_col_idx1 < args.n) Y_ptr[y_row_idx0*args.n+y_col_idx1] = Y_frag[mid][nid][1];
            if (y_row_idx1 < args.m && y_col_idx0 < args.n) Y_ptr[y_row_idx1*args.n+y_col_idx0] = Y_frag[mid][nid][2];
            if (y_row_idx1 < args.m && y_col_idx1 < args.n) Y_ptr[y_row_idx1*args.n+y_col_idx1] = Y_frag[mid][nid][3];
        }
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

void linear_v2_launch(void *x_packed_d, void *w_packed_d, void *y_d, int m, int n, int k);

////////////////////////////////////////////////////////////////////////////////////////////////////


#endif