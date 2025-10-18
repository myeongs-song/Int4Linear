#ifndef _CUSTOM_INT4_LINEAR_H
#define _CUSTOM_INT4_LINEAR_H

#include <cuda.h>
#include <cuda_runtime.h>

#define LOG2_WARP_SIZE 5
#define cdiv(a, b) (((a) + (b) - 1) / (b))


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    int kLog2ThreadblockShapeM_, int kLog2ThreadblockShapeN_, int kLog2ThreadblockShapeK_,
    int kLog2WarpShapeM_, int kLog2WarpShapeN_, int kStages_ = 3>
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

    static constexpr int kStages = kStages_;

    static constexpr int kSmemSize = kStages * kThreadblockShapeKPacked * (kThreadblockShapeM + kThreadblockShapeN) * sizeof(int32_t);
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

template <unsigned SizeInBytes>
__device__ __forceinline__ void cp_async_cg_shared_global(uint32_t smem_dst, const void* gmem_src) {
    static_assert(SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16, "Invalid copy size.");
    uint64_t gmem_addr = __cvta_generic_to_global(gmem_src);
    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n" : : "r"(smem_dst), "l"(gmem_addr), "n"(SizeInBytes));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile ("cp.async.commit_group;\n");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
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

// CuTe-style swizzling

//  0b   0000 0000 0000 0000
//                  ^^^^^^^^ col idx
//              ^^^^ row idx
//  INT8 -> 16 x 128
//  128-bit load == 16-byte load

template <int BBits, int MBase, int SShift>
struct Swizzle {
    static constexpr int num_bits = BBits;
    static constexpr int num_base = MBase;
    static constexpr int num_shift = SShift;

    static constexpr uint32_t mask = ((1 << num_bits) - 1) << (num_base + num_shift);

    __host__ __device__ int operator()(uint32_t byte_addr) {
        return byte_addr ^ ((byte_addr & mask) >> num_shift);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Device-side structure to run the linear operation
template <typename LinearConfig>
struct Int4LinearDevice {

    // Compile-time constants
    static constexpr int kThreadblockShapeM = LinearConfig::kThreadblockShapeM;
    static constexpr int kThreadblockShapeN = LinearConfig::kThreadblockShapeN;
    static constexpr int kThreadblockShapeK = LinearConfig::kThreadblockShapeK;
    static constexpr int kThreadblockShapeKPacked = LinearConfig::kThreadblockShapeKPacked;
 
    static constexpr int kWarpShapeM = LinearConfig::kWarpShapeM;
    static constexpr int kWarpShapeN = LinearConfig::kWarpShapeN;

    static constexpr int kWarpsPerThreadblockM = LinearConfig::kWarpsPerThreadblockM;
    static constexpr int kWarpsPerThreadblockN = LinearConfig::kWarpsPerThreadblockN;

    static constexpr int kInstsPerWarpM = LinearConfig::kInstsPerWarpM;
    static constexpr int kInstsPerWarpN = LinearConfig::kInstsPerWarpN;
    static constexpr int kInstsPerWarpK = LinearConfig::kInstsPerWarpK;

    static constexpr int kStages = LinearConfig::kStages;

    static constexpr int kNThreads = (kWarpsPerThreadblockM * kWarpsPerThreadblockN) << LOG2_WARP_SIZE;
    static constexpr int kNXCopyIters = cdiv(kThreadblockShapeM*kThreadblockShapeKPacked, kNThreads*4*kInstsPerWarpK);
    static constexpr int kNWCopyIters = cdiv(kThreadblockShapeN*kThreadblockShapeKPacked, kNThreads*4*kInstsPerWarpK);

    static_assert((kInstsPerWarpK & 1) == 0, "kInstsPerWarpK must be even to use pipelined MMA.");      
    static_assert(kStages > 2, "kStages must be larger than 2.");

    // Data members
    // Block, warp, thread id
    int block_id_x_;
    int block_id_y_;
    int thread_id_;
    int warp_id_;
    int lane_id_;

    // Arguments
    LinearArgs args_;
    int x_size_;
    int w_size_;
    int y_size_;

    // Data storage
    int32_t __restrict__ *X_ptr_; 
    int32_t __restrict__ *W_ptr_; 
    int32_t __restrict__ *Y_ptr_; 
    int32_t *smem_x_;
    int32_t *smem_w_;
    int32_t X_frag_[2][kInstsPerWarpM][4];
    int32_t W_frag_[2][kInstsPerWarpN][2];   
    int32_t Y_frag_[kInstsPerWarpM][kInstsPerWarpN][4];
    
    // Address offsets
    int x_gmem_base_offset_;  // block-specific
    int w_gmem_base_offset_;  // block-specific
    int x_smem_base_offset_;  // thread-specific
    int w_smem_base_offset_;  // thread-specific
    int x_gmem_load_offset_;  // block-specific
    int w_gmem_load_offset_;  // block-specific
    int x_smem_store_offset_; // thread-specific
    int w_smem_store_offset_; // thread-specific
    int x_smem_load_offset_;  // thread-specific
    int w_smem_load_offset_;  // thread-specific
    Swizzle<3, 4, 3> swizzle_;      //  TODO: <3, 4, 3> is for kThreadblockShapeK == 128. We need to make this more general.

    // loop control variables
    int n_threadblock_k_iters_;

    // Methods
    __device__ __forceinline__ Int4LinearDevice(
        int block_id_x, int block_id_y, int thread_id, 
        LinearArgs args, 
        int32_t *smem_x, int32_t *smem_w
    ):
        block_id_x_(block_id_x), block_id_y_(block_id_y), thread_id_(thread_id), 
        args_(args), 
        smem_x_(smem_x), smem_w_(smem_w) {

        memset(Y_frag_, 0, sizeof(Y_frag_));
        X_ptr_ = reinterpret_cast<int32_t*>(args.X_ptr);
        W_ptr_ = reinterpret_cast<int32_t*>(args.W_ptr);
        Y_ptr_ = reinterpret_cast<int32_t*>(args.Y_ptr);

        x_size_ = args.m * (args.k >> 3);
        w_size_ = args.n * (args.k >> 3); 
        y_size_ = args.m * args.n;

        warp_id_ = thread_id >> LOG2_WARP_SIZE;
        lane_id_ = thread_id & ((1 << LOG2_WARP_SIZE) - 1);

        x_gmem_base_offset_ = block_id_y_ * kThreadblockShapeM * (args.k >> 3);
        w_gmem_base_offset_ = block_id_x_ * kThreadblockShapeN * (args.k >> 3);
        x_smem_base_offset_ = ((warp_id_ / kWarpsPerThreadblockN) * kWarpShapeM     // warp-specific offset in each threadblock-tile
                            +  (lane_id_ & 0xf)) * kThreadblockShapeKPacked         // row offset in each warp-tile (for .x4 ldmatrix)
                            + ((lane_id_ >> 4) << 2);                               // col offset in each warp-tile (for .x4 ldmatrix)
        w_smem_base_offset_ = ((warp_id_ % kWarpsPerThreadblockN) * kWarpShapeN     // warp-specific offset in each threadblock-tile
                            +  (lane_id_ & 0x7)) * kThreadblockShapeKPacked         // row offset in each warp-tile (for .x2 ldmatrix)
                            + ((lane_id_ >> 3) << 2);                               // col offset in each warp-tile (for .x2 ldmatrix)

        x_gmem_load_offset_ = x_gmem_base_offset_;  // after kInstsPerWarpK fill_smem calls, this will be increased by kThreadblockShapeKPacked
        w_gmem_load_offset_ = w_gmem_base_offset_;  // after kInstsPerWarpK fill_smem calls, this will be increased by kThreadblockShapeKPacked
        x_smem_store_offset_ = 0;
        w_smem_store_offset_ = 0;
        x_smem_load_offset_ = x_smem_base_offset_;
        w_smem_load_offset_ = w_smem_base_offset_;

        n_threadblock_k_iters_ = cdiv(args.k, kThreadblockShapeK);
    }

    // Jang
    __device__ __forceinline__ void fill_smem(const int &buf_id) {

        int x_smem_buf_offset = buf_id * (kThreadblockShapeM * kThreadblockShapeKPacked);
        int w_smem_buf_offset = buf_id * (kThreadblockShapeN * kThreadblockShapeKPacked);

        #pragma unroll
        for (int i = 0; i < kNXCopyIters; ++i) {
            int gmem_row_offset = x_smem_store_offset_ / kThreadblockShapeKPacked;
            int gmem_col_offset = x_smem_store_offset_ % kThreadblockShapeKPacked;
            int gmem_offset = x_gmem_load_offset_ + gmem_row_offset*(args_.k>>3) + gmem_col_offset;
            uint32_t smem_addr = __cvta_generic_to_shared(
                &smem_x_[x_smem_store_offset_ + x_smem_buf_offset]
            );
            if (gmem_offset < x_size_) {
                cp_async_cg_shared_global<16>(
                    swizzle_(smem_addr),
                    &X_ptr_[gmem_offset]
                );
            }
            else {
                *(reinterpret_cast<int4*>(&smem_x_[x_smem_store_offset_ + x_smem_buf_offset])) = make_int4(0, 0, 0, 0);
            }
            x_smem_store_offset_ += (kNThreads * 4);
        }
        #pragma unroll
        for (int i = 0; i < kNWCopyIters; ++i) {
            int gmem_row_offset = w_smem_store_offset_ / kThreadblockShapeKPacked;
            int gmem_col_offset = w_smem_store_offset_ % kThreadblockShapeKPacked;
            int gmem_offset = w_gmem_load_offset_ + gmem_row_offset*(args_.k>>3) + gmem_col_offset;
            uint32_t smem_addr = __cvta_generic_to_shared(
                &smem_w_[w_smem_store_offset_ + w_smem_buf_offset]
            );
            if (gmem_offset < w_size_) {
                cp_async_cg_shared_global<16>(
                    swizzle_(smem_addr),
                    &W_ptr_[gmem_offset]
                );
            }
            else {
                *(reinterpret_cast<int4*>(&smem_w_[w_smem_store_offset_ + w_smem_buf_offset])) = make_int4(0, 0, 0, 0);
            }
            w_smem_store_offset_ += (kNThreads * 4);
        }
    }

    // Song
    __device__ __forceinline__ void fill_regs(const int& warp_k, const int& buf_id, const int& frag_id) {
        uint32_t smem_addr;
        #pragma unroll
        for (int m = 0; m < kInstsPerWarpM; ++m) {
            smem_addr = __cvta_generic_to_shared(
                &smem_x_[x_smem_load_offset_]
            );
            ldmatrix_sync_aligned_m8n8_x4_b16(X_frag_[frag_id][m], swizzle_(smem_addr));
            x_smem_load_offset_ += ((kThreadblockShapeKPacked)<<4);
        }
        #pragma unroll
        for (int n = 0; n < kInstsPerWarpN; ++n) {
            smem_addr = __cvta_generic_to_shared(
                &smem_w_[w_smem_load_offset_]
            );
            ldmatrix_sync_aligned_m8n8_x2_b16(W_frag_[frag_id][n], swizzle_(smem_addr));
            w_smem_load_offset_ += ((kThreadblockShapeKPacked)<<3);
        }
        // move the offset in column direction (advance tile when computing current tile is done)
        if (warp_k + 1 >= kInstsPerWarpK) {
            int next_buf_id = (buf_id >= kStages-1)? 0 : (buf_id+1);
            x_smem_load_offset_ = x_smem_base_offset_ 
                                + (next_buf_id * (kThreadblockShapeM * kThreadblockShapeKPacked))
                                + ((warp_k + 1 - kInstsPerWarpK)<<3);   
            w_smem_load_offset_ = w_smem_base_offset_ 
                                + (next_buf_id * (kThreadblockShapeN * kThreadblockShapeKPacked))
                                + ((warp_k + 1 - kInstsPerWarpK)<<3);   
        }
        else {
            x_smem_load_offset_ = x_smem_base_offset_ 
                                + (buf_id * (kThreadblockShapeM * kThreadblockShapeKPacked)) 
                                + ((warp_k+1)<<3);   
            w_smem_load_offset_ = w_smem_base_offset_ 
                                + (buf_id * (kThreadblockShapeN * kThreadblockShapeKPacked))
                                + ((warp_k+1)<<3);
        }
    }


    // Jang
    __device__ __forceinline__ void prologue() {
        #pragma unroll
        for (int stage = 0; stage < kStages-1; ++stage) {
            x_smem_store_offset_ = thread_id_ * 4;
            w_smem_store_offset_ = thread_id_ * 4;
            for (int k = 0; k < kInstsPerWarpK; ++k) {
                this->fill_smem(stage);
            }
            cp_async_commit_group();
            x_gmem_load_offset_ += kThreadblockShapeKPacked;
            w_gmem_load_offset_ += kThreadblockShapeKPacked;
        }
    }
    
    // Song
    /* for int m in ~:
     *      for int n in ~:
     * for: 
     *    0 1 2 3
     *    7 6 5 4
     *    8 9 10 11
     */
    __device__ __forceinline__ void warp_mn_iter(int frag_id) {
        #pragma unroll
        for (int m = 0; m < kInstsPerWarpM; ++m) {
            #pragma unroll
            for (int n = 0; n < kInstsPerWarpN; ++n) {
                int n_z = ((m & 0x1) ? (kInstsPerWarpN-1-n) : n);
                mma_sync_aligned_m16n8k64_rowcol_s4s4s32(
                    Y_frag_[m][n_z], X_frag_[frag_id][m], W_frag_[frag_id][n_z]
                );
            }
        }
    }
    /* 
     *  smem -> reg (prologue)
     *  for warp_k in ~:
     *     smem -> reg (warp_k + 1) // except last warp_k
     *     warp_mn_iter() (warp_k)
     *     if (not last warp_k) 
     *          cp.async
     *     if (second last warp_k)
     *          cp.async
     *          commit_group
     *          wait_group<kStages-2> // in cutlass, we need to "transform" at the last warp_k 
     *          __syncthreads()
     */
    __device__ __forceinline__ 
    void warp_k_iter(const int &load_buf_id, const int &store_buf_id, const int &load) {

        x_smem_store_offset_ = thread_id_ * 4;
        w_smem_store_offset_ = thread_id_ * 4;
        #pragma unroll
        for (int warp_k = 0; warp_k < kInstsPerWarpK; ++warp_k) {

            this->fill_regs(warp_k+1, load_buf_id, ((warp_k+1)&0x1));
            this->warp_mn_iter(warp_k&0x1);
            
            if (warp_k + 1 < kInstsPerWarpK && load) {
                this->fill_smem(store_buf_id);
            }
            if (warp_k + 2 == kInstsPerWarpK) {
                if (load) {
                    this->fill_smem(store_buf_id);
                }
                cp_async_commit_group();
                cp_async_wait_group<kStages - 2>();
                __syncthreads();
            } 
        }
        x_gmem_load_offset_ += kThreadblockShapeKPacked;
        w_gmem_load_offset_ += kThreadblockShapeKPacked;
    }

    __device__ __forceinline__ void threadblock_k_iter() {
        
        int load_buf_id = 0;
        int store_buf_id = kStages-1;
        this->fill_regs(0, 0, 0);
        // main loop
        for (int block_k = 0; block_k < (n_threadblock_k_iters_-kStages+1); ++block_k) {
            this->warp_k_iter(load_buf_id, store_buf_id, 1);
            load_buf_id = ((load_buf_id >= kStages-1)? 0 : (load_buf_id+1));
            store_buf_id = ((store_buf_id >= kStages-1)? 0 : (store_buf_id+1));
        }
        // epilogue
        for (int block_k = 0; block_k < (kStages-1); ++block_k) {
            this->warp_k_iter(load_buf_id, 0, 0);
            load_buf_id = ((load_buf_id >= kStages-1)? 0 : (load_buf_id+1));
        }
    }

    // Jang
    __device__ __forceinline__ void store_back() {
        // store Y_frag to Y_ptr 
        int y_block_row_offset = block_id_y_ * kThreadblockShapeM;
        int y_block_col_offset = block_id_x_ * kThreadblockShapeN;
        int x_warp_row_offset = (warp_id_ / kWarpsPerThreadblockN) * kWarpShapeM; 
        int w_warp_row_offset = (warp_id_ % kWarpsPerThreadblockN) * kWarpShapeN;
        int y_warp_row_offset = y_block_row_offset + x_warp_row_offset;
        int y_warp_col_offset = y_block_col_offset + w_warp_row_offset;

        #pragma unroll
        for (int mid = 0; mid < kInstsPerWarpM; ++mid) {
            int y_inst_row_offset = y_warp_row_offset + mid * 16;
            for (int nid = 0; nid < kInstsPerWarpN; ++nid) {
                int y_inst_col_offset = y_warp_col_offset + nid * 8;

                int y_row_idx0 = y_inst_row_offset + (lane_id_>>2);
                int y_row_idx1 = y_row_idx0 + 8;
                int y_col_idx0 = y_inst_col_offset + ((lane_id_&0x3) << 1);
                int y_col_idx1 = y_col_idx0 + 1;

                if (y_row_idx0 < args_.m && y_col_idx0 < args_.n) Y_ptr_[y_row_idx0*args_.n+y_col_idx0] = Y_frag_[mid][nid][0];
                if (y_row_idx0 < args_.m && y_col_idx1 < args_.n) Y_ptr_[y_row_idx0*args_.n+y_col_idx1] = Y_frag_[mid][nid][1];
                if (y_row_idx1 < args_.m && y_col_idx0 < args_.n) Y_ptr_[y_row_idx1*args_.n+y_col_idx0] = Y_frag_[mid][nid][2];
                if (y_row_idx1 < args_.m && y_col_idx1 < args_.n) Y_ptr_[y_row_idx1*args_.n+y_col_idx1] = Y_frag_[mid][nid][3];
            }
        }
    }

    // Song
    // prologue -> (barrier) -> threadblock_k_iter -> store_back
    __device__ __forceinline__ void operator()() {
        this->prologue();
        cp_async_wait_group<kStages-2>();
        __syncthreads();
        this->threadblock_k_iter();
        cp_async_wait_group<0>();
        __syncthreads();
        this->store_back();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


// Global kernel
template<typename LinearConfig>
__global__ void linear_v10_kernel(LinearArgs args) {

    // Shared memory allocation
    extern __shared__ int32_t smem[];
    int32_t *smem_x = smem; 
    int32_t *smem_w = smem + (LinearConfig::kThreadblockShapeM * LinearConfig::kThreadblockShapeKPacked * LinearConfig::kStages);

    // Block, thread id
    int block_id_x = blockIdx.x;
    int block_id_y = blockIdx.y;
    int thread_id = threadIdx.x;

    // Device object
    Int4LinearDevice<LinearConfig> linear_device(block_id_x, block_id_y, thread_id, args, smem_x, smem_w);

    // Run
    linear_device();
};


////////////////////////////////////////////////////////////////////////////////////////////////////

void linear_v10_launch(void *x_packed_d, void *w_packed_d, void *y_d, int m, int n, int k);

////////////////////////////////////////////////////////////////////////////////////////////////////


#endif