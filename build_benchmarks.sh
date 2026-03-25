#!/bin/bash

# A command's failure will cause the script to exit immediately.
set -e

echo "Compiling the CUTLASS reference..."

# Compiler and executable settings
NVCC=nvcc
CUTLASS_OUTPUT_NAME=benchmark_cutlass

# Source files
CUTLASS_SOURCE="test/cutlass_benchmark.cu"
CPP_SOURCE="utils/packer.cpp"

# Compiler flags
# -std=c++17: Use the C++17 standard.
# -I.: Add the current directory (project root) to the include path.
#      This allows the compiler to find the 'cutlass' and 'utils' directories.
# -gencode arch=compute_86,code=sm_86: Target NVIDIA Ampere architecture (e.g., RTX 30 series, A100).
#                                      If you use a different GPU, you may need to change this.
#                                      (e.g., Turing: sm_75, Volta: sm_70, Hopper: sm_90)
CUTLASS_FLAGS="-std=c++17 -I../cutlass/include -I. -O3 -gencode arch=compute_86,code=sm_86 -lineinfo"

# Compilation command
# This single command compiles both the .cu and .cpp files and links them together.
${NVCC} -o ${CUTLASS_OUTPUT_NAME} ${CUTLASS_FLAGS} ${CUTLASS_SOURCE} ${CPP_SOURCE}

echo "Compilation successful: ${CUTLASS_OUTPUT_NAME}"


echo "Compiling Int4Linear benchmark..."

# Compiler and executable settings
NVCC=nvcc
MY_OUTPUT_NAME=benchmark_int4linear

# Source files
MY_SOURCE="test/benchmark.cu"

MY_FLAGS="-std=c++17 -I./kernels/include -I. -O3 -gencode arch=compute_86,code=sm_86 -lineinfo"
${NVCC} -o ${MY_OUTPUT_NAME} ${MY_FLAGS} ${MY_SOURCE} ${CPP_SOURCE}

echo "Compilation successful: ${MY_OUTPUT_NAME}"
