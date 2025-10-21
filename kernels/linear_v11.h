#ifndef _CUSTOM_INT4_LINEAR_H
#define _CUSTOM_INT4_LINEAR_H

#include <cuda.h>
#include <cuda_runtime.h>

#define LOG2_WARP_SIZE 5
#define cdiv(a, b) (((a) + (b) - 1) / (b))


////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    int kLog2ThreadblockShapeM_, int kLog2ThreadblockShapeN_, int kLog2ThreadblockShapeK_,
    int kLog2WarpShapeM_, int kLog2WarpShapeN_, int kStages_ = 3, int kElementsPerVector_ = 4>
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
    static constexpr int kElementsPerVector = kElementsPerVector_;
    static constexpr int kSmemSize = kStages * kThreadblockShapeKPacked * (kThreadblockShapeM + kThreadblockShapeN) * sizeof(int32_t);

    static_assert((kElementsPerVector == 1) || (kElementsPerVector == 2) || (kElementsPerVector == 4), "kElementsPerVector must be 1, 2, or 4.");
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

///////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned SizeInBytes>
__device__ __forceinline__ void cp_async_predicated(uint32_t smem_dst, const void* gmem_src, bool valid) {
    static_assert(SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16, "Invalid copy size.");
    uint64_t gmem_addr = __cvta_generic_to_global(gmem_src);
    int predicated = (valid)? SizeInBytes : 0;
    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" : : "r"(smem_dst), "l"(gmem_addr), "n"(SizeInBytes), "r"(predicated));
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

// Threadblock-tile iterator (for gmem -> smem)
// Song
template <
    int kThreadblockShape0_, int kThreadblockShape1Packed_, 
    int kElementsPerVector_, int kThreads_, int kStages_>
struct ThreadblockIterator {

    // Compile-time constants
    static constexpr int kThreadblockShape0 = kThreadblockShape0_;              // dim = 0 (m or n)
    static constexpr int kThreadblockShape1Packed = kThreadblockShape1Packed_;  // dim = 1 (k)
    static constexpr int kElementsPerVector = kElementsPerVector_;
    static constexpr int kElementsPerBuffer = kThreadblockShape0 * kThreadblockShape1Packed;
    static constexpr int kThreads = kThreads_;
    static constexpr int kStages = kStages_;
    static constexpr int kStride = kThreads * kElementsPerVector;

    // Data members
    // id
    int thread_id_;

    // base offset
    int gmem_base_offset_;
    int smem_base_offset_;

    // pre-computed addresses, offsets, and masks
    uint32_t smem_base_addr_;
    uint32_t smem_buf_addr_;
    uint32_t smem_addr_;
    
    // matrix size-related
    int csize_packed_;
    int csize_relative_; // for the last iteration in col direction
    int rsize_relative_; // for the last iteration in row direction

    // current states
    int cur_buf_;
    int cur_row_;
    int cur_col_;

    // Methods
    // NOTE: csize is in int32_t units
    __device__ __forceinline__ ThreadblockIterator(
        const int &rsize, const int &csize_packed, 
        const int &relative_block_id, const int &thread_id, 
        int32_t *smem_ptr,
        int gmem_base_offset
    ):
        csize_packed_(csize_packed),
        thread_id_(thread_id),
        gmem_base_offset_(gmem_base_offset) {  

        // init smem_addrs
        smem_base_offset_ = thread_id_ * kElementsPerVector;
        smem_base_addr_ = __cvta_generic_to_shared(smem_ptr + smem_base_offset_);
        smem_buf_addr_ = smem_base_addr_;
        smem_addr_ = smem_buf_addr_;

        // init current states
        cur_buf_ = 0;
        cur_row_ = (thread_id_*kElementsPerVector) / kThreadblockShape1Packed;
        cur_col_ = (thread_id_*kElementsPerVector) % kThreadblockShape1Packed;

        // get sizes for masking
        int csize_relative = csize_packed % kThreadblockShape1Packed; // for the last iteration in col direction
        csize_relative_ = (csize_relative == 0) ? kThreadblockShape1Packed : csize_relative;
        rsize_relative_ = rsize - (relative_block_id * kThreadblockShape0);
    }

    // advance to the next buffer
    __device__ __forceinline__ void advance() {
        cur_buf_ = (cur_buf_ < kStages-1) ? (cur_buf_+1) : 0;
        smem_buf_addr_ = smem_base_addr_ + (cur_buf_ * kElementsPerBuffer * sizeof(int32_t));
        smem_addr_ = smem_buf_addr_;
        cur_row_ = smem_base_offset_ / kThreadblockShape1Packed;
        cur_col_ = smem_base_offset_ % kThreadblockShape1Packed;
        gmem_base_offset_ += kThreadblockShape1Packed;
    }

    // check if the current position is valid in inner loops
    __device__ __forceinline__ bool valid_inner() {
        return cur_row_ < rsize_relative_;
    }

    // check if the current position is valid in the last iteration
    __device__ __forceinline__ bool valid_last() {
        bool row_valid = (cur_row_ < rsize_relative_);
        bool col_valid = (cur_col_ < csize_relative_);
        return row_valid && col_valid;
    }

    // directly return the current smem address
    __device__ __forceinline__ uint32_t get_smem_addr() {
        return smem_addr_;
    }

    // unfortunately, csize_packed_ is not a compile-time constant
    // so we need to compute gmem offset at each call
    __device__ __forceinline__ int get_gmem_offset() {
        return gmem_base_offset_ + cur_row_*csize_packed_ + cur_col_;
    }

    // move to the next smem address and gmem row/col offset
    __device__ __forceinline__ void operator ++() {
        smem_addr_ += (kStride * sizeof(int32_t));
        // NOTE: kStride and kThreadblockShape1Packed are always power of 2 
        // That is, one of them is always divisible by the other one.
        if constexpr (kStride >= kThreadblockShape1Packed) {
            cur_row_ += (kStride / kThreadblockShape1Packed);
        }
        else {
            cur_col_ += kStride;
            if (cur_col_ >= kThreadblockShape1Packed) {
                cur_col_ = 0;
                cur_row_++;
            }         
        }
    }
}; 

///////////////////////////////////////////////////////////////////////////////////////////////////

// Warp iterator (for smem -> reg)
// Song
template <
    int kThreadblockShape0_, int kThreadblockShape1Packed_, int kLog2RowsPerInsts_,
    int kInstsPerWarp0_, int kInstsPerWarp1_, int kStages_>
struct WarpIterator {

    // Compile-time constants
    static constexpr int kThreadblockShape0 = kThreadblockShape0_;              // dim = 0 (m or n)
    static constexpr int kThreadblockShape1Packed = kThreadblockShape1Packed_;  // dim = 1 (k)
    static constexpr int kElementsPerBuffer = kThreadblockShape0 * kThreadblockShape1Packed;
    static constexpr int kLog2RowsPerInsts = kLog2RowsPerInsts_;                // depend on inst shape
    static constexpr int kInstsPerWarp0 = kInstsPerWarp0_;  // dim = 0 (m or n)
    static constexpr int kInstsPerWarp1 = kInstsPerWarp1_;  // dim = 1 (k)
    static constexpr int kStages = kStages_;

    // Data members
    // smem ptr
    int32_t * __restrict__ smem_ptr;

    // base offset
    uint32_t smem_base_addr_;
    uint32_t smem_buf_addr_;
    uint32_t smem_addr_;

    // current states
    int cur_k_; // inst along k-axis
    int cur_buf_;

    __device__ __forceinline__ WarpIterator(
        int32_t *smem_ptr, 
        const int &smem_base_offset
    ): 
        smem_ptr(smem_ptr) {

        smem_base_addr_ = __cvta_generic_to_shared(smem_ptr + smem_base_offset);
        smem_buf_addr_ = smem_base_addr_;
        smem_addr_ = smem_buf_addr_;
        
        cur_k_ = 0;
        cur_buf_ = 0;
    }

    // advance to the next buffer or next position along k-axis
    __device__ __forceinline__ void advance() {
        cur_k_++;
        if (cur_k_ == kInstsPerWarp1) {
            cur_k_ = 0;
            cur_buf_ = (cur_buf_ < kStages-1) ? (cur_buf_+1) : 0;    
            smem_buf_addr_ = smem_base_addr_ + (cur_buf_ * kElementsPerBuffer * sizeof(int32_t));
            smem_addr_ = smem_buf_addr_;
        }
        else {
            smem_addr_ = smem_buf_addr_ + ((cur_k_ << 3) * sizeof(int32_t));
        }
    }

    // directly return the current smem address
    __device__ __forceinline__ uint32_t get_smem_addr() {
        return smem_addr_;
    }

    // move to the next position along row direction
    __device__ __forceinline__ void operator ++() {
        smem_addr_ += ((kThreadblockShape1Packed << kLog2RowsPerInsts) * sizeof(int32_t));
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////


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
    static constexpr int kElementsPerVector = LinearConfig::kElementsPerVector;

    static constexpr int kThreads = (kWarpsPerThreadblockM * kWarpsPerThreadblockN) << LOG2_WARP_SIZE;
    static constexpr int kXPositions = cdiv(kThreadblockShapeM*kThreadblockShapeKPacked, kThreads*kElementsPerVector);
    static constexpr int kWPositions = cdiv(kThreadblockShapeN*kThreadblockShapeKPacked, kThreads*kElementsPerVector);
    static constexpr int kXCopyIters = cdiv(kXPositions, kInstsPerWarpK);
    static constexpr int kWCopyIters = cdiv(kWPositions, kInstsPerWarpK);

    static_assert((kInstsPerWarpK & 1) == 0, "kInstsPerWarpK must be even to use pipelined MMA.");      
    static_assert(kStages > 2, "kStages must be larger than 2.");

    using ThreadblockIteratorX = ThreadblockIterator<
        kThreadblockShapeM, kThreadblockShapeKPacked, kElementsPerVector, kThreads, kStages
    >;
    using ThreadblockIteratorW = ThreadblockIterator<
        kThreadblockShapeN, kThreadblockShapeKPacked, kElementsPerVector, kThreads, kStages
    >;
    using WarpIteratorX = WarpIterator<
        kThreadblockShapeM, kThreadblockShapeKPacked, 4, kInstsPerWarpM, kInstsPerWarpK, kStages
    >;
    using WarpIteratorW = WarpIterator<
        kThreadblockShapeN, kThreadblockShapeKPacked, 3, kInstsPerWarpN, kInstsPerWarpK, kStages
    >;

    // Data members
    // Block, warp, thread id
    int block_id_x_;
    int block_id_y_;
    int thread_id_;
    int warp_id_;
    int lane_id_;

    // Arguments
    LinearArgs args_;

    // Data storage
    int32_t __restrict__ *X_ptr_; 
    int32_t __restrict__ *W_ptr_; 
    int32_t __restrict__ *Y_ptr_; 
    int32_t *smem_x_;
    int32_t *smem_w_;
    int32_t X_frag_[2][kInstsPerWarpM][4];
    int32_t W_frag_[2][kInstsPerWarpN][2];   
    int32_t Y_frag_[kInstsPerWarpM][kInstsPerWarpN][4];
    
    Swizzle<3, 4, 3> swizzle_;      //  TODO: We need to make this more general.

    // iterators
    ThreadblockIteratorX threadblock_iterator_x_;
    ThreadblockIteratorW threadblock_iterator_w_;
    WarpIteratorX warp_iterator_x_;
    WarpIteratorW warp_iterator_w_;

    // loop control variables
    int n_threadblock_k_iters_;

    // Methods
    __device__ __forceinline__ Int4LinearDevice(
        const int &block_id_x, const int &block_id_y, const int &thread_id, 
        LinearArgs &args, 
        int32_t *smem_x, int32_t *smem_w
    ):
        block_id_x_(block_id_x), block_id_y_(block_id_y), thread_id_(thread_id), 
        warp_id_(thread_id >> LOG2_WARP_SIZE), lane_id_(thread_id & ((1 << LOG2_WARP_SIZE) - 1)),
        args_(args), 
        smem_x_(smem_x), smem_w_(smem_w), 
        n_threadblock_k_iters_(cdiv(args.k, kThreadblockShapeK)),
        threadblock_iterator_x_(
            args.m, (args.k >> 3),
            block_id_y_, thread_id_,
            smem_x_,
            block_id_y_ * kThreadblockShapeM * (args.k >> 3)
        ),
        threadblock_iterator_w_(
            args.n, (args.k >> 3),
            block_id_x_, thread_id_,
            smem_w_,
            block_id_x_ * kThreadblockShapeN * (args.k >> 3)
        ),
        warp_iterator_x_(
            smem_x_,
            ((warp_id_ / kWarpsPerThreadblockN) * kWarpShapeM   // warp-specific offset in each threadblock-tile
            +  (lane_id_ & 0xf)) * kThreadblockShapeKPacked     // row offset in each warp-tile (for .x4 ldmatrix)
            + ((lane_id_ >> 4) << 2)
        ),
        warp_iterator_w_(
            smem_w_,
            ((warp_id_ % kWarpsPerThreadblockN) * kWarpShapeN   // warp-specific offset in each threadblock-tile
            +  (lane_id_ & 0x7)) * kThreadblockShapeKPacked     // row offset in each warp-tile (for .x2 ldmatrix)
            + ((lane_id_ >> 3) << 2)
        ) {

        // Initialize accumulators
        memset(Y_frag_, 0, sizeof(Y_frag_));

        X_ptr_ = reinterpret_cast<int32_t*>(args.X_ptr);
        W_ptr_ = reinterpret_cast<int32_t*>(args.W_ptr);
        Y_ptr_ = reinterpret_cast<int32_t*>(args.Y_ptr);
    }

    // Song
    template <int kValidationMode = 0>
    __device__ __forceinline__ void fill_smem() {

        #pragma unroll
        for (int i = 0; i < kXCopyIters; ++i) {
            int gmem_offset = threadblock_iterator_x_.get_gmem_offset();
            uint32_t smem_addr = threadblock_iterator_x_.get_smem_addr();
            bool valid;
            if constexpr (kValidationMode == 0) {
                valid = threadblock_iterator_x_.valid_inner();
            }
            else {
                valid = threadblock_iterator_x_.valid_last();
            }
            if (valid) {
                cp_async_cg_shared_global<kElementsPerVector*4>(smem_addr, &X_ptr_[gmem_offset], valid);
            }
            else {
                if constexpr (kElementsPerVector == 1) {
                    smem_x_[x_smem_store_offset_ + x_smem_buf_offset] = 0;
                }
                else if constexpr (kElementsPerVector == 2) {
                    *(reinterpret_cast<int2*>(&smem_x_[x_smem_store_offset_ + x_smem_buf_offset])) = make_int2(0, 0);
                }
                else {
                    *(reinterpret_cast<int4*>(&smem_x_[x_smem_store_offset_ + x_smem_buf_offset])) = make_int4(0, 0, 0, 0);
                }
            }
            ++threadblock_iterator_x_;
        }
        #pragma unroll
        for (int i = 0; i < kWCopyIters; ++i) {
            int gmem_offset = threadblock_iterator_w_.get_gmem_offset();
            uint32_t smem_addr = threadblock_iterator_w_.get_smem_addr();
            bool valid;
            if constexpr (kValidationMode == 0) {
                valid = threadblock_iterator_w_.valid_inner();
            }
            else {
                valid = threadblock_iterator_w_.valid_last();
            }
            if (valid) {
                cp_async_predicated<kElementsPerVector*4>(smem_addr, &W_ptr_[gmem_offset], valid);
            }
            else {
                if constexpr (kElementsPerVector == 1) {
                    smem_w_[w_smem_store_offset_ + w_smem_buf_offset] = 0;
                }
                else if constexpr (kElementsPerVector == 2) {
                    *(reinterpret_cast<int2*>(&smem_w_[w_smem_store_offset_ + w_smem_buf_offset])) = make_int2(0, 0);
                }
                else {
                    *(reinterpret_cast<int4*>(&smem_w_[w_smem_store_offset_ + w_smem_buf_offset])) = make_int4(0, 0, 0, 0);
                }
            }
            ++threadblock_iterator_w_;
        }
    }

    // Song
    __device__ __forceinline__ void fill_regs(const int& frag_id) {
        #pragma unroll
        for (int m = 0; m < kInstsPerWarpM; ++m) {
            uint32_t smem_addr = warp_iterator_x_.get_smem_addr();
            ldmatrix_sync_aligned_m8n8_x4_b16(X_frag_[frag_id][m], smem_addr);
            ++warp_iterator_x_;
        }
        #pragma unroll
        for (int n = 0; n < kInstsPerWarpN; ++n) {
            uint32_t smem_addr = warp_iterator_w_.get_smem_addr();
            ldmatrix_sync_aligned_m8n8_x2_b16(W_frag_[frag_id][n], smem_addr);
            ++warp_iterator_w_;
        }
        // move the offset in column direction or to the next buffer
        warp_iterator_x_.advance();
        warp_iterator_w_.advance();
    }

    // Jang
    __device__ __forceinline__ void prologue() {
        #pragma unroll
        for (int stage = 0; stage < kStages-1; ++stage) {
            for (int k = 0; k < kInstsPerWarpK; ++k) {
                this->fill_smem<0>();
            }
            cp_async_commit_group();
            threadblock_iterator_x_.advance();
            threadblock_iterator_w_.advance();
        }
    }
    
    // Song
    __device__ __forceinline__ void warp_mn_iter(const int &frag_id) {
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
    
    // Song, cutlass-style pipelined k-iteration
    template <int kValidationMode = 0>
    __device__ __forceinline__ 
    void warp_k_iter(const int &load) {

        #pragma unroll
        for (int warp_k = 0; warp_k < kInstsPerWarpK; ++warp_k) {

            this->fill_regs(((warp_k+1)&0x1));
            this->warp_mn_iter(warp_k&0x1);
            
            if (warp_k + 1 < kInstsPerWarpK && load) {
                this->fill_smem<kValidationMode>();
            }
            if (warp_k + 2 == kInstsPerWarpK) {
                if (load) {
                    this->fill_smem<kValidationMode>();
                }
                cp_async_commit_group();
                cp_async_wait_group<kStages - 2>();
                __syncthreads();
            } 
        }
        threadblock_iterator_x_.advance();
        threadblock_iterator_w_.advance();
    }

    __device__ __forceinline__ void threadblock_k_iter() {
        
        this->fill_regs(0);
        // main loop
        for (int block_k = 0; block_k < (n_threadblock_k_iters_-kStages); ++block_k) {
            this->warp_k_iter<0>(1);
        }
        // load last tile with appropriate validation
        this->warp_k_iter<1>(1);  
        // epilogue
        for (int block_k = 0; block_k < (kStages-1); ++block_k) {
            this->warp_k_iter<0>(0);
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
            int y_row_idx0 = y_warp_row_offset + (mid*16) + (lane_id_>>2);
            int y_row_idx1 = y_row_idx0 + 8;

            #pragma unroll
            for (int nid = 0; nid < kInstsPerWarpN; ++nid) {
                int y_col_idx0 = y_warp_col_offset + (nid*8) + ((lane_id_&0x3)<<1);
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
__global__ void linear_v11_kernel(LinearArgs args) {

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

void linear_v11_launch(void *x_packed_d, void *w_packed_d, void *y_d, int m, int n, int k);

////////////////////////////////////////////////////////////////////////////////////////////////////


#endif