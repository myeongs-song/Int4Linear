/**
 * [Summary]
 *  Custom 4-bit integer linear layer kernel written by Myeongsoo Song
 *  This kernel takse 4-bit quantized activations (X) and weights (W) as input
 *  and produces 32-bit output results (Y = X @ W.T)
 *
 * [Acknowledgment]
 *  The overall structure and execution flow exactly follows the multi-stage GEMM kernel
 *  design in CUTLASS. This implementation was developed with reference to CUTLASS.
 *  https://github.com/NVIDIA/cutlass
 */

#ifndef _CUSTOM_INT4_LINEAR_20H
#define _CUSTOM_INT4_LINEAR_20H

#include <cuda.h>
#include <cuda_runtime.h>

#define LOG2_WARP_SIZE 5
#define cdiv(a, b) (((a) + (b) - 1) / (b))
#define czero(x)   (((x) > 0)? (x) : 0)


////////////////////////////////////////////////////////////////////////////////////////////////////
// Layer configuration
////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    int kLog2ThreadblockShapeM_, 
    int kLog2ThreadblockShapeN_, 
    int kLog2ThreadblockShapeK_,
    int kLog2WarpShapeM_, 
    int kLog2WarpShapeN_, 
    int kStages_ = 3>
struct LinearConfig {
    static constexpr int kThreadblockShapeM = (1 << kLog2ThreadblockShapeM_);
    static constexpr int kThreadblockShapeN = (1 << kLog2ThreadblockShapeN_);
    static constexpr int kThreadblockShapeK = (1 << kLog2ThreadblockShapeK_);
    static constexpr int kThreadblockShapeKPacked = (1 << (kLog2ThreadblockShapeK_ - 3));
    static constexpr int kLog2ThreadblockShapeK = kLog2ThreadblockShapeK_;

    static constexpr int kWarpShapeM = (1 << kLog2WarpShapeM_);
    static constexpr int kWarpShapeN = (1 << kLog2WarpShapeN_);

    static constexpr int kWarpsPerThreadblockM = (1 << (kLog2ThreadblockShapeM_ - kLog2WarpShapeM_));
    static constexpr int kWarpsPerThreadblockN = (1 << (kLog2ThreadblockShapeN_ - kLog2WarpShapeN_));

    static constexpr int kInstsPerWarpM = (1 << (kLog2WarpShapeM_ - 4));
    static constexpr int kInstsPerWarpN = (1 << (kLog2WarpShapeN_ - 3));
    static constexpr int kInstsPerWarpK = (1 << (kLog2ThreadblockShapeK_ - 6)); 

    static constexpr int kStages = kStages_;

    static constexpr int kSmemSize = 
        kStages * kThreadblockShapeKPacked * (kThreadblockShapeM + kThreadblockShapeN) * sizeof(int32_t);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Linear layer arguments
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
// PTX wrappers
////////////////////////////////////////////////////////////////////////////////////////////////////

template <unsigned SizeInBytes>
__device__ __forceinline__ void cp_async_cg_shared_global_pred(uint32_t smem_dst, const void* gmem_src, bool valid) {
    static_assert(SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16, "[Int4Linear] Invalid copy size.");
    uint64_t gmem_addr = __cvta_generic_to_global(gmem_src);
    int src_size = (valid) ? SizeInBytes : 0;
    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" : : "r"(smem_dst), "l"(gmem_addr), "n"(SizeInBytes), "r"(src_size));
}


__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile ("cp.async.commit_group;\n");
}


template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n" : : "n"(N));
}


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


__device__ __forceinline__
void ldmatrix_sync_aligned_m8n8_x2_b16(int32_t *reg, uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,  %1}, [%2];\n"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(addr));
}


__device__ __forceinline__
void ldmatrix_sync_aligned_m8n8_x4_b16(int32_t *reg, uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,  %1,  %2,  %3}, [%4];\n"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(addr));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// CuTe-style swizzling
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BBits, int MBase, int SShift>
struct Swizzle {
    static constexpr int num_bits = BBits;
    static constexpr int num_base = MBase;
    static constexpr int num_shift = SShift;

    static constexpr uint32_t mask = ((1 << num_bits) - 1) << (num_base + num_shift);

    __host__ __device__ uint32_t operator()(uint32_t byte_addr) {
        return byte_addr ^ ((byte_addr & mask) >> num_shift);
    }
};



////////////////////////////////////////////////////////////////////////////////////////////////////
// Threadblock-tile iterator (for gmem -> smem)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int kThreadblockShape0_, 
    int kThreadblockShape1Packed_, 
    int kThreads_, int kStages_>
struct ThreadblockIterator {

    // Compile-time constants
    static constexpr int kThreadblockShape0 = kThreadblockShape0_;
    static constexpr int kStages            = kStages_;
    static constexpr int kVectorSizeInBytes = 16;   // design choice
    static constexpr int kBlockWidthInBytes = kThreadblockShape1Packed_ * sizeof(int32_t);
    static constexpr int kVectorsPerRow     = kBlockWidthInBytes / kVectorSizeInBytes;
    static constexpr int kBytesPerInnerIter = kThreads_ * kVectorSizeInBytes;
    static constexpr int kRowsPerInnerIter  = kBytesPerInnerIter / kBlockWidthInBytes;
    static constexpr int kBlockSizeInBytes  = kThreadblockShape0 * kBlockWidthInBytes;

    // Constraints
    static_assert(
        kBytesPerInnerIter >= kBlockWidthInBytes, 
        "[Int4Linear] Threads must cover at least one full row"
    );
    static_assert(
        kBytesPerInnerIter % kBlockWidthInBytes == 0,
        "[Int4Linear] Threads must cover integer number of rows"
    );
    static_assert(
        kBlockWidthInBytes % kVectorSizeInBytes == 0, 
        "[Int4Linear] Tile width must be 16-byte aligned."
    );
    

    // Data members
    // matrix-specific
    int             num_rows_;
    int             csize_in_bytes_;

    // base offset
    const uint8_t * gmem_base_ptr_;
    uint32_t        smem_base_addr_;
    int             base_row_;

    // pre-computed strides
    int             gmem_stride_in_bytes_;
    uint32_t        smem_stride_in_bytes_;
    
    // gmem ptr & smem addr
    const uint8_t * gmem_ptr_;
    uint32_t        smem_addr_;

    // states
    int             cur_stage_;
    int             cur_k_;
    int             cur_row_;
    int             cur_col_;
    bool            cur_col_valid_;

    // Methods
    __device__ __forceinline__ ThreadblockIterator(
        const int rsize, 
        const int csize_in_bytes, 
        const int block_id, const int thread_id, 
        const void *gmem_ptr, int32_t *smem_ptr
    ) {  
        // per-thread initial location
        int thread_row = thread_id / kVectorsPerRow;
        int thread_col_in_bytes = (thread_id % kVectorsPerRow) * kVectorSizeInBytes;

        // for validation logic
        base_row_ = thread_row;
        cur_row_  = thread_row;
        num_rows_ = czero(rsize - (block_id * kThreadblockShape0));
        csize_in_bytes_ = csize_in_bytes;
        cur_col_        = thread_col_in_bytes;
        cur_col_valid_  = (thread_col_in_bytes < csize_in_bytes);

        // compute offsets
        smem_base_addr_ = __cvta_generic_to_shared(smem_ptr);
        smem_base_addr_ += thread_row * kBlockWidthInBytes + thread_col_in_bytes;
        int gmem_base_offset_in_bytes = (
            block_id * kThreadblockShape0 * csize_in_bytes +
            thread_row * csize_in_bytes + thread_col_in_bytes
        );
        gmem_base_ptr_ = reinterpret_cast<const uint8_t *>(gmem_ptr) + gmem_base_offset_in_bytes;

        // init current states
        smem_addr_  = smem_base_addr_;
        gmem_ptr_   = gmem_base_ptr_;
        cur_k_      = 0;
        cur_stage_  = 0;

        // pre-compute strides
        gmem_stride_in_bytes_ = kRowsPerInnerIter * csize_in_bytes;
        smem_stride_in_bytes_ = kRowsPerInnerIter * kBlockWidthInBytes;
    }

    // advance to the next buffer
    __device__ __forceinline__ void advance() {
        cur_k_++;
        cur_stage_++;
        cur_row_ = base_row_;
        gmem_ptr_ = gmem_base_ptr_ + cur_k_ * kBlockWidthInBytes;
        cur_col_ += kBlockWidthInBytes;
        cur_col_valid_ = (cur_col_ < csize_in_bytes_);
        if (cur_stage_ == kStages) {
            cur_stage_ = 0;
            smem_addr_ = smem_base_addr_;
        }
        else {
            smem_addr_ = smem_base_addr_ + cur_stage_ * kBlockSizeInBytes;
        }
    }

    // directly return the current smem address
    __device__ __forceinline__ uint32_t get_smem_addr() {
        return smem_addr_;
    }

    // directly return the current gmem pointer
    __device__ __forceinline__ const void * get_gmem_ptr() {
        return reinterpret_cast<const void*>(gmem_ptr_);
    }

    // move to the next smem address and gmem row/col offset
    __device__ __forceinline__ void operator ++() {
        gmem_ptr_   += gmem_stride_in_bytes_;
        smem_addr_  += smem_stride_in_bytes_;
        cur_row_    += kRowsPerInnerIter; 
    }

    // check if the current position is valid
    __device__ __forceinline__ bool valid() const {
        return cur_col_valid_ && (cur_row_ < num_rows_);
    }
}; 


///////////////////////////////////////////////////////////////////////////////////////////////////
// Warp iterator (for smem -> reg)
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int kThreadblockShape0_, 
    int kThreadblockShape1Packed_, 
    int kLog2RowsPerLoad_,
    int kLoadsPerWarp0_, 
    int kInstsPerWarp1_, 
    int kStages_>
struct WarpIterator {

    // Compile-time constants
    static constexpr int kStages                = kStages_;
    static constexpr int kDim0StepSizeInBytes   = (kThreadblockShape1Packed_ << kLog2RowsPerLoad_) * sizeof(int32_t);
    static constexpr int kDim1StepSizeInBytes   = 8 * sizeof(int32_t);
    static constexpr int kBlockSizeInBytes      = kThreadblockShape0_ * kThreadblockShape1Packed_ * sizeof(int32_t);
    static constexpr int kDim0RewindStepSizeInBytes = kLoadsPerWarp0_ * kDim0StepSizeInBytes;
    static constexpr int kDim1RewindStepSizeInBytes = kInstsPerWarp1_ * kDim1StepSizeInBytes;

    // Data members
    // addr
    uint32_t smem_base_addr_;
    uint32_t smem_addr_;

    __device__ __forceinline__ WarpIterator(
        int32_t *smem_ptr, 
        int smem_base_offset
    ) {
        smem_base_addr_ = __cvta_generic_to_shared(smem_ptr + smem_base_offset);
        smem_addr_ = smem_base_addr_;
    }

    // advance to the next position along k-axis
    __device__ __forceinline__ void advance_k() {
        smem_addr_ += static_cast<uint32_t>(kDim1StepSizeInBytes - kDim0RewindStepSizeInBytes);
    }

    // advance to the next buffer
    __device__ __forceinline__ void advance_buf(int next_stage) {
        if (next_stage == 0) {
            smem_addr_ = smem_base_addr_;
        }
        else {
            smem_addr_ += static_cast<uint32_t>(kBlockSizeInBytes - kDim1RewindStepSizeInBytes);
        }
    }

    // directly return the current smem address
    __device__ __forceinline__ uint32_t get_smem_addr() {
        return smem_addr_;
    }

    // move to the next position along row direction
    __device__ __forceinline__ void operator ++() {
        smem_addr_ += static_cast<uint32_t>(kDim0StepSizeInBytes);
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// Fragment in registers
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Shape0_, 
    int Shape1_>
struct Fragment {
    int32_t data[Shape0_ * Shape1_];
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Device-side structure to run the linear operation
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename LinearConfig>
struct Int4LinearDevice {

    // Compile-time constants
    // Shape configuration
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

    // Kernel configuration
    static constexpr int kStages = LinearConfig::kStages;
    static constexpr int kElementsPerVector = 4;

    // Launch configuration
    static constexpr int kThreads = (kWarpsPerThreadblockM * kWarpsPerThreadblockN) << LOG2_WARP_SIZE;
    
    // gmem -> smem related constants
    static constexpr int kXPositions = cdiv(kThreadblockShapeM*kThreadblockShapeKPacked, kThreads*kElementsPerVector);
    static constexpr int kWPositions = cdiv(kThreadblockShapeN*kThreadblockShapeKPacked, kThreads*kElementsPerVector);
    static constexpr int kXCopyIters = cdiv(kXPositions, kInstsPerWarpK);
    static constexpr int kWCopyIters = cdiv(kWPositions, kInstsPerWarpK);

    // smem -> regs related constants
    static constexpr int kWLog2RowsPerLoad = (kInstsPerWarpN & 0x1)? 3 : 4;
    static constexpr int kLoadsPerWarpN    = (kWarpShapeN >> kWLog2RowsPerLoad);
    
    // swizzle
    static constexpr int kLog2ThreadblockShapeKInBytes = LinearConfig::kLog2ThreadblockShapeK;
    static constexpr int kSwizzleBase   = 4;  // 16-byte
    static constexpr int kSwizzleShift  = kLog2ThreadblockShapeKInBytes - kSwizzleBase;
    static constexpr int kSwizzleBits   = kSwizzleShift;

    static_assert(
        (kInstsPerWarpK & 1) == 0, 
        "[Int4Linear] Pipeline Error: kInstsPerWarpK must be even to use pipelined MMA."
    );      
    static_assert(
        kStages > 2, 
        "[Int4Linear] Pipeline Error: kStages must be larger than 2."
    );
    static_assert(
        kThreads <= 1024, 
        "[Int4Linear] Invalid Configuration: Threads per block cannot exceed 1024."
    );
    static_assert(
        kThreads % 32 == 0, 
        "[Int4Linear] Threads must be a multiple of warp size (32)."
    );
    static_assert(
        kXPositions % kInstsPerWarpK == 0, 
        "[Int4Linear] Pipeline Error: X load positions must be evenly divisible by kInstsPerWarpK."
    );
    static_assert(
        kWPositions % kInstsPerWarpK == 0, 
        "[Int4Linear] Pipeline Error: W load positions must be evenly divisible by kInstsPerWarpK."
    );

    using ThreadblockIteratorX = ThreadblockIterator<
        kThreadblockShapeM, kThreadblockShapeKPacked, kThreads, kStages
    >;
    using ThreadblockIteratorW = ThreadblockIterator<
        kThreadblockShapeN, kThreadblockShapeKPacked, kThreads, kStages
    >;
    using WarpIteratorX = WarpIterator<
        kThreadblockShapeM, kThreadblockShapeKPacked, 4, kInstsPerWarpM, kInstsPerWarpK, kStages
    >;
    using WarpIteratorW = WarpIterator<
        kThreadblockShapeN, kThreadblockShapeKPacked, kWLog2RowsPerLoad, kLoadsPerWarpN, kInstsPerWarpK, kStages
    >;
    using FragmentX = Fragment<kInstsPerWarpM, 4>;
    using FragmentW = Fragment<kInstsPerWarpN, 2>;
    using FragmentY = Fragment<kInstsPerWarpM * kInstsPerWarpN, 4>;

    using SwizzleOp = Swizzle<kSwizzleBits, kSwizzleBase, kSwizzleShift>;

    // Data members
    // Block, warp, thread id
    int block_id_x_;
    int block_id_y_;
    int warp_id_;
    int lane_id_;

    // Arguments
    LinearArgs args_;

    // Data storage
    FragmentY Y_frag_;

    // Swizzle
    SwizzleOp swizzle_;      

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
        block_id_x_(block_id_x), block_id_y_(block_id_y), 
        warp_id_(thread_id >> LOG2_WARP_SIZE), lane_id_(thread_id & ((1 << LOG2_WARP_SIZE) - 1)),
        args_(args), 
        n_threadblock_k_iters_(cdiv(args.k, kThreadblockShapeK)),
        threadblock_iterator_x_(
            args.m, (args.k >> 1),
            block_id_y, thread_id,
            args.X_ptr, smem_x
        ),
        threadblock_iterator_w_(
            args.n, (args.k >> 1),
            block_id_x, thread_id,
            args.W_ptr, smem_w 
        ),
        warp_iterator_x_(
            smem_x,
            ((warp_id_ / kWarpsPerThreadblockN) * kWarpShapeM       // warp-specific offset in each threadblock-tile
            +  (lane_id_ & 0xf)) * kThreadblockShapeKPacked         // row offset in each warp-tile (for .x4 ldmatrix)
            + ((lane_id_ >> 4) << 2)                                // column (along-k) offset
        ),
        warp_iterator_w_(
            smem_w,
            (kInstsPerWarpN & 0x1) ?                                // Use ldmatrix x2 : x4
            (((warp_id_ % kWarpsPerThreadblockN) * kWarpShapeN      // warp-specific offset in each threadblock-tile
            +  (lane_id_ & 0x7)) * kThreadblockShapeKPacked         // row offset in each warp-tile (for .x2 ldmatrix)
            + ((lane_id_ >> 3) << 2))                               // column (along-k) offset
            :
            (((warp_id_ % kWarpsPerThreadblockN) * kWarpShapeN      // warp-specific offset in each threadblock-tile
            + (lane_id_ & 0x7)
            + ((lane_id_ >> 4) << 3)) * kThreadblockShapeKPacked    // row offset in each warp-tile (for .x2 ldmatrix)
            + (((lane_id_ & 0xf) >> 3) << 2))                       // column (along-k) offset
        ) {

        // Initialize accumulators
        #pragma unroll
        for (int mn = 0; mn < kInstsPerWarpM * kInstsPerWarpN; ++mn) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                Y_frag_.data[mn * 4 + r] = 0;
            }
        }
    }

    // Fill shared memory
    __device__ __forceinline__ void fill_smem() {
        #pragma unroll
        for (int i = 0; i < kXCopyIters; ++i) {
            const void *gmem_ptr = threadblock_iterator_x_.get_gmem_ptr();
            uint32_t smem_addr = swizzle_(threadblock_iterator_x_.get_smem_addr());
            bool valid = threadblock_iterator_x_.valid();
            cp_async_cg_shared_global_pred<16>(smem_addr, gmem_ptr, valid);
            ++threadblock_iterator_x_;
        }
        #pragma unroll
        for (int i = 0; i < kWCopyIters; ++i) {
            const void *gmem_ptr = threadblock_iterator_w_.get_gmem_ptr();
            uint32_t smem_addr = swizzle_(threadblock_iterator_w_.get_smem_addr());
            bool valid = threadblock_iterator_w_.valid();
            cp_async_cg_shared_global_pred<16>(smem_addr, gmem_ptr, valid);
            ++threadblock_iterator_w_;
        }
    }

    // Fill registers from the shared memory
    __device__ __forceinline__ void fill_regs(FragmentX &x_frag, FragmentW &w_frag) {
        #pragma unroll
        for (int m = 0; m < kInstsPerWarpM; ++m) {
            uint32_t smem_addr = swizzle_(warp_iterator_x_.get_smem_addr());
            ldmatrix_sync_aligned_m8n8_x4_b16(&x_frag.data[m*4], smem_addr);
            ++warp_iterator_x_;
        }

        if constexpr (kInstsPerWarpN & 0x1) {
            #pragma unroll
            for (int n = 0; n < kInstsPerWarpN; ++n) {
                uint32_t smem_addr = swizzle_(warp_iterator_w_.get_smem_addr());
                ldmatrix_sync_aligned_m8n8_x2_b16(&w_frag.data[n*2], smem_addr);
                ++warp_iterator_w_;
            }
        }
        // We can safely use x4 ldmatrix inside this block
        else {
            #pragma unroll
            for (int n = 0; n < (kInstsPerWarpN/2); ++n) {
                uint32_t smem_addr = swizzle_(warp_iterator_w_.get_smem_addr());
                ldmatrix_sync_aligned_m8n8_x4_b16(&w_frag.data[n*4], smem_addr);
                ++warp_iterator_w_;
            }
        }
        
        // move the offset in column direction
        warp_iterator_x_.advance_k();
        warp_iterator_w_.advance_k();
    }

    // Fill the pipeline
    __device__ __forceinline__ void prologue() {
        #pragma unroll
        for (int stage = 0; stage < kStages-1; ++stage) {
            for (int k = 0; k < kInstsPerWarpK; ++k) {
                this->fill_smem();
            }
            cp_async_commit_group();
            threadblock_iterator_x_.advance();
            threadblock_iterator_w_.advance();
        }
    }
    
    // mma
    __device__ __forceinline__ void warp_mn_iter(
        FragmentY &y_frag, FragmentX &x_frag, FragmentW &w_frag
    ) {
        #pragma unroll
        for (int m = 0; m < kInstsPerWarpM; ++m) {
            #pragma unroll
            for (int n = 0; n < kInstsPerWarpN; ++n) {
                const int n_z = ((m & 0x1) ? (kInstsPerWarpN-1-n) : n);

                const int y_idx = (m * kInstsPerWarpN + n_z) * 4;
                const int x_idx = m * 4;
                const int w_idx = n_z * 2;

                mma_sync_aligned_m16n8k64_rowcol_s4s4s32(
                    &y_frag.data[y_idx], &x_frag.data[x_idx], &w_frag.data[w_idx]
                );
            }
        }
    }
    
    // cutlass-style pipelined k-iteration
    __device__ __forceinline__ 
    void warp_k_iter(
        const int &load, 
        int next_stage, 
        FragmentX x_frags[2],
        FragmentW w_frags[2]
    ) {

        #pragma unroll
        for (int warp_k = 0; warp_k < kInstsPerWarpK; ++warp_k) {

            int next_frag_idx   = (warp_k+1) & 0x1;
            int cur_frag_idx    = warp_k & 0x1;

            this->fill_regs(x_frags[next_frag_idx], w_frags[next_frag_idx]);
            this->warp_mn_iter(Y_frag_, x_frags[cur_frag_idx], w_frags[cur_frag_idx]);
            
            if (warp_k + 1 < kInstsPerWarpK && load) {
                this->fill_smem();
            }
            if (warp_k + 2 == kInstsPerWarpK) {
                if (load) {
                    this->fill_smem();
                }
                cp_async_commit_group();
                warp_iterator_x_.advance_buf(next_stage);
                warp_iterator_w_.advance_buf(next_stage);
                cp_async_wait_group<kStages - 2>();
                __syncthreads();
            } 
        }
        threadblock_iterator_x_.advance();
        threadblock_iterator_w_.advance();
    }

    __device__ __forceinline__ void threadblock_k_iter() {

        FragmentX x_frags[2];
        FragmentW w_frags[2];
        
        this->fill_regs(x_frags[0], w_frags[0]);
        int next_stage = 1;
        // main loop
        for (int block_k = 0; block_k < (n_threadblock_k_iters_-kStages+1); ++block_k) {
            this->warp_k_iter(1, next_stage, x_frags, w_frags);
            next_stage = (next_stage + 1) % kStages;
        }
        // epilogue
        for (int block_k = 0; block_k < (kStages-1); ++block_k) {
            this->warp_k_iter(0, next_stage, x_frags, w_frags);
            next_stage = (next_stage + 1) % kStages;
        }
    }

    // store Y_frag to Y_ptr 
    __device__ __forceinline__ void store_back() {

        int32_t *Y_ptr = reinterpret_cast<int32_t *>(args_.Y_ptr);

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

                if (y_row_idx0 < args_.m && y_col_idx0 < args_.n) Y_ptr[y_row_idx0*args_.n+y_col_idx0] = Y_frag_.data[(mid*kInstsPerWarpN + nid)*4 + 0];
                if (y_row_idx0 < args_.m && y_col_idx1 < args_.n) Y_ptr[y_row_idx0*args_.n+y_col_idx1] = Y_frag_.data[(mid*kInstsPerWarpN + nid)*4 + 1];
                if (y_row_idx1 < args_.m && y_col_idx0 < args_.n) Y_ptr[y_row_idx1*args_.n+y_col_idx0] = Y_frag_.data[(mid*kInstsPerWarpN + nid)*4 + 2];
                if (y_row_idx1 < args_.m && y_col_idx1 < args_.n) Y_ptr[y_row_idx1*args_.n+y_col_idx1] = Y_frag_.data[(mid*kInstsPerWarpN + nid)*4 + 3];
            }
        }
    }

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
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename LinearConfig>
__global__ void linear_v20_kernel(LinearArgs args) {

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
// Host-side launcher
////////////////////////////////////////////////////////////////////////////////////////////////////

void linear_v20_launch(void *x_packed_d, void *w_packed_d, void *y_d, int m, int n, int k);

#endif 