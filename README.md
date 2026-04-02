# High-Performance 4-bit Integer Linear Layer from Scratch

## Introduction & Motivation
The AI industry is actively pushing the boundaries of quantization down to INT4 for efficient model deployment. This paradigm shift has significantly increased the demand for highly optimized, native 4-bit linear layer implementations.

While state-of-the-art GEMM libraries like NVIDIA CUTLASS deliver exceptional performance, they heavily rely on deep abstraction layers and highly complex template logic. Furthermore, the standard cuBLAS library currently lacks native support for sub-byte (4-bit) GEMM kernels. 

This project aims to build a custom 4-bit integer linear kernel **from the ground up**. By establishing a transparent and readable codebase, this project demonstrates how developers can effortlessly construct and adapt high-performance custom kernels tailored to specific architectural requirements, without getting bogged down by extreme abstractions.

<br>

## Architecture & Background
To maximize data reuse and compute efficiency, this kernel adopts a **Hierarchical Tiling Strategy**. Computations are distributed across the GPU hardware hierarchy:
* **Global Memory → Shared Memory → Register File → CUDA/Tensor Cores**
* The workload is partitioned into Threadblock Tiles, Warp Tiles, and Thread Tiles to ensure continuous data delivery to the compute units.

<img width="1418" height="330" alt="image" src="https://github.com/user-attachments/assets/1d6ea292-79c4-401b-8430-c8f3961312d4" />
*Source: NVIDIA corporation, [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)*

<br>

## The Optimization Journey

To achieve and surpass the performance of CUTLASS, four core optimization strategies were implemented:

### 1. Leveraging Hardware Primitives
* **Direct Tensor Core Utilization:** The kernel directly invokes PTX instructions (`mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`) to achieve maximum throughput for INT4 matrix multiply-accumulate operations.
<img width="1545" height="649" alt="image" src="https://github.com/user-attachments/assets/ef998977-825f-44f8-ba96-10cc0f494b49" />

* **Efficient Memory Loading:** Utilized the `ldmatrix.sync.aligned.m8n8.x4.shared.b16` instruction to load operand tiles directly from Shared Memory to Registers with optimal bandwidth.
<img width="1498" height="352" alt="image" src="https://github.com/user-attachments/assets/79d35575-f87b-4866-9764-06f2b498935b" />


### 2. Software Pipelining
A multi-stage pipeline was implemented to effectively hide Global Memory latency by overlapping memory transactions with math operations:
* **Asynchronous Memory Copy:** Exploited `cp.async.cg.shared.global` along with `commit_group`/`wait_group` barriers to pipeline data transfers directly into Shared Memory (bypassing L1 cache).
<img width="1862" height="382" alt="image" src="https://github.com/user-attachments/assets/64568a49-0cc7-4a06-bb5e-6d7970a2dda9" />

* **Register Double Buffering:** Maintained two sets of registers to interleave `ldmatrix` loads and `mma` operations, keeping the Tensor Cores fully saturated.
* **Enhanced Instruction-Level Parallelism (ILP):** Expanded the register allocation footprint per thread to allow the warp scheduler to interleave a higher volume of independent math and memory instructions.
* **Maximized Dynamic Shared Memory:** Used `cudaFuncSetAttribute` to allocate up to 96KB of dynamic shared memory per block, enabling a deeper pipeline (more stages) and larger threadblock tiles.

### 3. Reducing Computational Overhead
* **Constant Offset-Based Iterators:** Replaced sequential pointer increments with constant offsets. This breaks data dependencies, allows the compiler to resolve address calculations statically, and prevents warp scheduler stall cycles.
* **Bulk, Coalesced Memory Access:** Implemented 128-bit vectorized loads (the maximum native vector width) for input operands and 64-bit coalesced stores for the output matrix, significantly reducing the total number of load/store instructions.

### 4. Micro-Architectural Tweaks
* **Shared Memory Swizzling:** To prevent severe N-way bank conflicts during `ldmatrix` operations, a custom XOR-based swizzling layout (`<3, 4, 3>`) was applied. This creates a bijective mapping that perfectly rearranges the storage, eliminating bank conflicts entirely.
<img width="1499" height="620" alt="image" src="https://github.com/user-attachments/assets/a2d3bc44-faa0-4ba9-a5b6-6acd6f54bdb7" />

* **Zig-Zag Computation:** Reused the weight operand register consumed at the end of one row immediately at the beginning of the subsequent row. This maximizes temporal locality at the register level and alleviates operand collector bottlenecks.

<br>

## Results & Discussion
The custom kernel was benchmarked against the CUTLASS baseline on an **NVIDIA RTX 3090**.

<img width="1476" height="431" alt="image" src="https://github.com/user-attachments/assets/75c05a7c-d468-4787-b53e-fe19f40c5688" />


Across varying matrix dimensions (both $M=N=K$ and $2M=2N=K$), **the custom kernel consistently outperforms CUTLASS.** **Key reasons for outperforming the baseline:**
1. **Elimination of Epilogue Overhead:** The custom kernel explicitly removes generalized epilogue stages (e.g., element-wise additions) that CUTLASS includes by default, streamlining the execution path.
2. **Architecture-Specific Tuning:** While CUTLASS's default Ampere configurations are heavily biased toward the datacenter A100 architecture, this kernel was meticulously tuned specifically for the RTX 3090's cache and SM limits.

<br>

## Future Works
* **Constant Stride-Based Swizzling:** Pre-determining the swizzled layout at compile time to eliminate runtime bitwise instructions (`LOP3`, `SHF`), further reducing register pressure.
* **Staged Store:** Accumulating final output elements in a shared memory buffer before writing to global memory, enabling maximum-width 128-bit global store instructions.
* **Threadblock Swizzling:** Implementing a more sophisticated mapping between Threadblock IDs and matrix tiles to maximize L2 cache hit rates.
* **Support for Bias Addition:** Implement bias vector broadcasting and addition to the output matrix to fully support standard linear layer operations.

<br>

> **Conclusion**
> The theoretical peak performance of hardware can only be fully unlocked when the underlying software implementation is meticulously crafted to support it. This project demonstrates the power of hardware-aware software design, utilizing software pipelining and compiler-driven meta-programming to push a custom kernel to its absolute limits.

## Acknowledgment
The overall structure and execution flow of this custom kernel exactly follow the multi-stage GEMM kernel design in **CUTLASS**. This implementation was developed with deep reference to their exceptional work. 

* **Reference:** [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
