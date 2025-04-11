.PHONY: all clean build run cuda-ptx

NVCC := nvcc
CUDA_COMPUTE := 50
RUST_FEATURES := cuda

all: cuda-ptx build

# Build the Rust project with the CUDA feature
build:
	cargo build --release --features $(RUST_FEATURES)

# Compile CUDA kernels to PTX
cuda-ptx: resources/vanity_kernel.ptx

resources/vanity_kernel.ptx: src/cuda/kernels/vanity_kernel.cu
	@mkdir -p resources
	$(NVCC) -ptx -m64 -arch=sm_$(CUDA_COMPUTE) -o $@ $<

# Create the CUDA kernel source file if it doesn't exist
src/cuda/kernels/vanity_kernel.cu:
	@mkdir -p src/cuda/kernels
	@echo "Extracting kernel code from Rust file"
	@awk '/\/\/ Will be compiled for CUDA device/{flag=1;next} /^\}$$/{if(flag){flag=0}} flag' src/cuda/vanity_kernel.rs > $@

# Run the vanity address generator
run: build
	./target/release/rust_vanity

# Clean build artifacts
clean:
	cargo clean
	rm -f resources/vanity_kernel.ptx 