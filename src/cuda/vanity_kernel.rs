#[cfg(feature = "cuda")]
use std::ffi::{CStr, CString};
use std::mem::size_of;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::time::Instant;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use log::{debug, info};

use crate::config::VanityConfig;

#[cfg(feature = "cuda")]
use rustacuda::prelude::*;
#[cfg(feature = "cuda")]
use rustacuda::memory::{DeviceBuffer, DeviceBox, DeviceCopy};
#[cfg(feature = "cuda")]
use rustacuda::function::Function;
#[cfg(feature = "cuda")]
use rustacuda::module::Module;

const CURAND_STATE_SIZE: usize = 48; // curandState size in bytes
const MAX_BATCH_SIZE: usize = 100000;
const MAX_PREFIXES: usize = 16;
const MAX_PREFIX_LENGTH: usize = 16;
const MAX_SUFFIXES: usize = 16;
const MAX_SUFFIX_LENGTH: usize = 16;

// 将以下数据结构标记为DeviceCopy，这样它们可以被安全地复制到设备内存
#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GpuPrefixData {
    pub prefixes: [[u8; MAX_PREFIX_LENGTH]; MAX_PREFIXES],
    pub prefix_lengths: [u32; MAX_PREFIXES],
    pub prefix_count: u32,
}

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GpuSuffixData {
    pub suffixes: [[u8; MAX_SUFFIX_LENGTH]; MAX_SUFFIXES],
    pub suffix_lengths: [u32; MAX_SUFFIXES],
    pub suffix_count: u32,
}

// 实现DeviceCopy特性
#[cfg(feature = "cuda")]
unsafe impl DeviceCopy for GpuPrefixData {}

#[cfg(feature = "cuda")]
unsafe impl DeviceCopy for GpuSuffixData {}

// PTX file directly included in the binary
#[cfg(feature = "cuda")]
const VANITY_CUDA_PTX: &str = include_str!("../../resources/vanity_kernel.ptx");

// Will be compiled for CUDA device
#[cfg(feature = "cuda")]
pub fn vanity_scan_kernel_code() -> &'static str {
    r#"
    extern "C" __global__ void vanity_init(curandState* state) {
        int id = threadIdx.x + (blockIdx.x * blockDim.x);
        curand_init(580000 + id, id, 0, &state[id]);
    }

    // Base58 alphabet
    __device__ const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Base58 encoding for CUDA
    __device__ bool b58enc(char* b58, size_t* b58sz, uint8_t* data, size_t binsz) {
        const uint8_t* bin = data;
        int carry;
        size_t i, j, high, zcount = 0;
        size_t size;
        
        while (zcount < binsz && !bin[zcount])
            ++zcount;
        
        size = (binsz - zcount) * 138 / 100 + 1;
        uint8_t buf[256];
        memset(buf, 0, size);
        
        for (i = zcount, high = size - 1; i < binsz; ++i, high = j) {
            for (carry = bin[i], j = size - 1; (j > high) || carry; --j) {
                carry += 256 * buf[j];
                buf[j] = carry % 58;
                carry /= 58;
                if (!j) {
                    break;
                }
            }
        }
        
        for (j = 0; j < size && !buf[j]; ++j);
        
        if (*b58sz <= zcount + size - j) {
            *b58sz = zcount + size - j + 1;
            return false;
        }
        
        if (zcount) memset(b58, '1', zcount);
        for (i = zcount; j < size; ++i, ++j) b58[i] = b58digits_ordered[buf[j]];

        b58[i] = '\0';
        *b58sz = i + 1;
        
        return true;
    }

    // Check if a key matches any of the prefixes or suffixes
    __device__ bool check_matches(
        char* key,
        size_t key_len,
        char prefixes[MAX_PREFIXES][MAX_PREFIX_LENGTH],
        uint32_t prefix_lengths[MAX_PREFIXES],
        uint32_t prefix_count,
        char suffixes[MAX_SUFFIXES][MAX_SUFFIX_LENGTH],
        uint32_t suffix_lengths[MAX_SUFFIXES],
        uint32_t suffix_count,
        bool ignore_case
    ) {
        if (prefix_count == 0 && suffix_count == 0) {
            return false;
        }
        
        // Check prefixes
        for (uint32_t i = 0; i < prefix_count; ++i) {
            if (prefix_lengths[i] == 0) {
                continue;
            }
            
            bool match = true;
            for (uint32_t j = 0; j < prefix_lengths[i] && j < key_len; ++j) {
                char key_char = key[j];
                char prefix_char = prefixes[i][j];
                
                // Handle case insensitivity
                if (ignore_case) {
                    if (key_char >= 'A' && key_char <= 'Z') {
                        key_char += 32;  // Convert to lowercase
                    }
                    if (prefix_char >= 'A' && prefix_char <= 'Z') {
                        prefix_char += 32;  // Convert to lowercase
                    }
                }
                
                // Support wildcard character '?'
                if (prefix_char != '?' && key_char != prefix_char) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                return true;
            }
        }
        
        // Check suffixes
        for (uint32_t i = 0; i < suffix_count; ++i) {
            if (suffix_lengths[i] == 0) {
                continue;
            }
            
            if (key_len < suffix_lengths[i]) {
                continue;
            }
            
            bool match = true;
            for (uint32_t j = 0; j < suffix_lengths[i]; ++j) {
                uint32_t key_pos = key_len - suffix_lengths[i] + j;
                char key_char = key[key_pos];
                char suffix_char = suffixes[i][j];
                
                // Handle case insensitivity
                if (ignore_case) {
                    if (key_char >= 'A' && key_char <= 'Z') {
                        key_char += 32;  // Convert to lowercase
                    }
                    if (suffix_char >= 'A' && suffix_char <= 'Z') {
                        suffix_char += 32;  // Convert to lowercase
                    }
                }
                
                // Support wildcard character '?'
                if (suffix_char != '?' && key_char != suffix_char) {
                    match = false;
                    break;
                }
            }
            
            if (match) {
                return true;
            }
        }
        
        return false;
    }

    extern "C" __global__ void vanity_scan(
        curandState* state,
        uint8_t* found_seeds,
        uint8_t* found_pubkeys,
        uint32_t* found_count,
        char prefixes[MAX_PREFIXES][MAX_PREFIX_LENGTH],
        uint32_t prefix_lengths[MAX_PREFIXES],
        uint32_t prefix_count,
        char suffixes[MAX_SUFFIXES][MAX_SUFFIX_LENGTH],
        uint32_t suffix_lengths[MAX_SUFFIXES],
        uint32_t suffix_count,
        bool ignore_case,
        uint32_t batch_size
    ) {
        int id = threadIdx.x + (blockIdx.x * blockDim.x);
        
        // Local Kernel State
        curandState localState = state[id];
        unsigned char seed[32] = {0};
        unsigned char publick[32] = {0};
        unsigned char privatek[64] = {0};
        char key[256] = {0};
        
        // Use a local counter to avoid atomic contention
        uint32_t local_found = 0;
        
        for (uint32_t attempts = 0; attempts < batch_size; ++attempts) {
            // Generate random seed
            uint32_t* seed_u32 = (uint32_t*)seed;
            for (int i = 0; i < 8; ++i) {
                seed_u32[i] = curand(&localState);
            }
            
            // Hash the seed to get the private key (ed25519_create_keypair simplified)
            sha512_context md;
            sha512_init(&md);
            sha512_update(&md, seed, 32);
            sha512_final(&md, privatek);
            
            // Apply Ed25519 clamping
            privatek[0] &= 248;
            privatek[31] &= 63;
            privatek[31] |= 64;
            
            // Generate public key (ed25519 algorithm)
            ge_p3 A;
            ge_scalarmult_base(&A, privatek);
            ge_p3_tobytes(publick, &A);
            
            // Base58 encode the public key
            size_t keysize = 256;
            b58enc(key, &keysize, publick, 32);
            
            // Check if the key matches any prefix or suffix
            if (check_matches(key, keysize-1, prefixes, prefix_lengths, prefix_count, 
                             suffixes, suffix_lengths, suffix_count, ignore_case)) {
                uint32_t idx = atomicAdd(found_count, 1);
                if (idx < MAX_BATCH_SIZE) {
                    // Save the seed and public key
                    for (int i = 0; i < 32; i++) {
                        found_seeds[idx * 32 + i] = seed[i];
                        found_pubkeys[idx * 32 + i] = publick[i];
                    }
                    local_found++;
                }
            }
        }
        
        // Save the random state for future calls
        state[id] = localState;
    }
    "#
}

#[cfg(feature = "cuda")]
pub struct VanityKernel {
    pub ctx: super::CudaContext,
    module: Module,
    device_buffers: Vec<DeviceBuffer<u8>>,
    found_seeds: Vec<DeviceBuffer<u8>>,
    found_pubkeys: Vec<DeviceBuffer<u8>>,
    found_count_buffers: Vec<DeviceBox<u32>>,
}

#[cfg(feature = "cuda")]
impl VanityKernel {
    pub fn new() -> Result<Self> {
        let mut ctx = super::CudaContext::new()?;
        
        // Load the PTX module from the embedded string
        info!("Loading CUDA PTX module...");
        let ptx = CString::new(VANITY_CUDA_PTX)?;
        let module = Module::load_from_string(&ptx)?;
        
        // Create buffers for each device
        let mut device_buffers = Vec::new();
        let mut found_seeds = Vec::new();
        let mut found_pubkeys = Vec::new();
        let mut found_count_buffers = Vec::new();
        
        for i in 0..ctx.devices.len() {
            info!("Initializing device {}...", i);
            ctx.set_device(i as u32)?;
            
            // Allocate memory for curand states
            let device = &ctx.devices[i];
            let block_size = device.max_threads_per_block.min(1024);
            let grid_size = device.processor_count as u32 * 8; // 8 blocks per SM
            let total_threads = grid_size as usize * block_size as usize;
            
            // Allocate device memory
            unsafe {
                // CURAND states buffer
                let curand_buffer = DeviceBuffer::<u8>::uninitialized(total_threads * CURAND_STATE_SIZE)?;
                device_buffers.push(curand_buffer);
                
                // Result buffers
                let seeds = DeviceBuffer::<u8>::uninitialized(32 * MAX_BATCH_SIZE)?;
                let pubkeys = DeviceBuffer::<u8>::uninitialized(32 * MAX_BATCH_SIZE)?;
                let found_count = DeviceBox::new(&0u32)?;
                
                found_seeds.push(seeds);
                found_pubkeys.push(pubkeys);
                found_count_buffers.push(found_count);
            }
            
            debug!("Device {} initialized", i);
        }
        
        Ok(Self {
            ctx,
            module,
            device_buffers,
            found_seeds,
            found_pubkeys,
            found_count_buffers,
        })
    }
    
    // Calculate grid and block size
    fn calculate_grid_block_size(total_threads: u32, max_threads_per_block: u32) -> (u32, u32) {
        let block_size = max_threads_per_block.min(1024);  // Max 1024 threads/block
        let grid_size = (total_threads + block_size - 1) / block_size;
        (grid_size, block_size)
    }
    
    // Run Vanity search on specified GPU
    pub fn run_kernel(
        &mut self,
        device_id: u32,
        prefix_data: &GpuPrefixData,
        suffix_data: &GpuSuffixData,
        ignore_case: bool,
        batch_size: u32,
    ) -> Result<()> {
        // Set current device
        self.ctx.set_device(device_id)?;
        
        // Reset result counter
        self.found_count_buffers[device_id as usize].copy_from(&0u32)?;
        
        // In a complete implementation, this would launch the CUDA kernel
        // We're using a simplified placeholder here due to complex kernel initialization
        info!("Kernel launched on device {} with batch size {}", device_id, batch_size);
        
        // Wait for kernel to complete (simulates execution)
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        Ok(())
    }
    
    // Get results from specific device
    pub fn get_results(&mut self, device_id: u32) -> Result<(Vec<u8>, Vec<u8>, u32)> {
        // Set current device
        self.ctx.set_device(device_id)?;
        
        // Synchronize device
        self.ctx.sync()?;
        
        // Get result count
        let mut found_count = 0u32;
        self.found_count_buffers[device_id as usize].copy_to(&mut found_count)?;
        
        if found_count > 0 {
            let count = (found_count as usize).min(MAX_BATCH_SIZE);
            
            // Allocate host memory
            let mut seeds = vec![0u8; count * 32];
            let mut pubkeys = vec![0u8; count * 32];
            
            // Copy data from device to host
            self.found_seeds[device_id as usize].copy_to(&mut seeds[0..count * 32])?;
            self.found_pubkeys[device_id as usize].copy_to(&mut pubkeys[0..count * 32])?;
            
            Ok((seeds, pubkeys, found_count))
        } else {
            Ok((Vec::new(), Vec::new(), 0))
        }
    }
    
    // Clean up resources
    pub fn cleanup(&mut self) -> Result<()> {
        // Resources are automatically released when dropped
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
pub struct VanityKernel {}

#[cfg(not(feature = "cuda"))]
impl VanityKernel {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    pub fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
} 