#!/bin/bash
set -e

mkdir -p resources
echo "Compiling CUDA sources..."
nvcc -ptx -o resources/vanity_kernel.ptx cuda_src/vanity.cu

echo "CUDA PTX compilation completed" 