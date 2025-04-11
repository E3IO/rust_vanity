use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use bs58;
use ed25519_dalek::{SecretKey, Verifier};
use ed25519_dalek::Signer;
use ed25519_dalek::SigningKey;
use ed25519_dalek::VerifyingKey as PublicKey;
use log::{debug, info};
use rand::{rngs::OsRng, RngCore};
use rayon::prelude::*;
use sha2::{Digest, Sha512};

use crate::config::VanityConfig;
#[cfg(feature = "cuda")]
use crate::cuda::{VanityKernel, vanity_kernel};
#[cfg(feature = "cuda")]
use crate::cuda::vanity_kernel::{GpuPrefixData, GpuSuffixData};

// CUDA相关常量
const MAX_PREFIXES: usize = 16;
const MAX_PREFIX_LENGTH: usize = 16;
const MAX_SUFFIXES: usize = 16;
const MAX_SUFFIX_LENGTH: usize = 16;

pub struct VanityGenerator {
    config: VanityConfig,
    gpu_kernel: Option<VanityKernel>,
    found_keys: Arc<Mutex<Vec<(Vec<u8>, Vec<u8>)>>>, // (seed, public_key)
}

impl VanityGenerator {
    pub fn new(config: VanityConfig) -> Result<Self> {
        let gpu_kernel = if config.gpu_enabled {
            #[cfg(feature = "cuda")]
            {
                match VanityKernel::new() {
                    Ok(kernel) => {
                        info!("CUDA initialized successfully");
                        Some(kernel)
                    }
                    Err(e) => {
                        info!("Failed to initialize CUDA: {}", e);
                        None
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                info!("CUDA support not enabled");
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            gpu_kernel,
            found_keys: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn run(&mut self) -> Result<()> {
        info!("Starting vanity address generation");
        info!("Prefixes: {:?}", self.config.prefixes);
        info!("Suffixes: {:?}", self.config.suffixes);

        let start_time = Instant::now();
        let mut total_attempts = 0;
        let mut iterations = 0;

        let has_gpu = self.gpu_kernel.is_some();
        let found_keys = self.found_keys.clone();

        // 确定是用GPU还是CPU
        if has_gpu {
            info!("Using GPU for generation");
            
            #[cfg(feature = "cuda")]
            {
                let gpu_count = self.gpu_kernel.as_ref().unwrap().ctx.devices.len();
                info!("Found {} GPU devices", gpu_count);
                
                // 准备前缀和后缀数据
                let prefixes_data = self.prepare_prefixes_for_gpu()?;
                let suffixes_data = self.prepare_suffixes_for_gpu()?;
                
                while iterations < 100 {
                    let batch_start = Instant::now();
                    let batch_size = self.config.batch_size;
                    let mut total_batch_found = 0;
                    
                    // 在所有可用GPU上并行执行
                    for device_id in 0..gpu_count {
                        // 配置并启动GPU内核
                        self.launch_gpu_kernel(device_id, &prefixes_data, &suffixes_data, batch_size)?;
                    }
                    
                    // 同步并收集所有GPU的结果
                    let found_count = self.collect_gpu_results()?;
                    total_batch_found += found_count;
                    
                    total_attempts += batch_size * gpu_count;
                    iterations += 1;
                    
                    let batch_duration = batch_start.elapsed();
                    let attempts_per_second = (batch_size * gpu_count) as f64 / batch_duration.as_secs_f64();
                    
                    info!(
                        "Batch {}: {} attempts in {:.2}s ({:.2} attempts/s), {} keys found",
                        iterations,
                        batch_size * gpu_count,
                        batch_duration.as_secs_f64(),
                        attempts_per_second,
                        total_batch_found
                    );
                    
                    // 保存结果
                    if iterations % 10 == 0 || total_batch_found > 0 {
                        self.save_results()?;
                    }
                }
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                info!("CUDA support not enabled, falling back to CPU");
                self.run_cpu_generation(&found_keys, &mut total_attempts, &mut iterations)?;
            }
        } else {
            info!("Using CPU for generation (multi-threaded)");
            self.run_cpu_generation(&found_keys, &mut total_attempts, &mut iterations)?;
        }

        let total_duration = start_time.elapsed();
        let total_attempts_per_second = total_attempts as f64 / total_duration.as_secs_f64();

        info!(
            "Completed {} attempts in {:.2}s ({:.2} attempts/s)",
            total_attempts,
            total_duration.as_secs_f64(),
            total_attempts_per_second
        );

        self.save_results()?;
        Ok(())
    }
    
    // CPU实现，提取为单独方法以便GPU和CPU逻辑分离
    fn run_cpu_generation(
        &self, 
        found_keys: &Arc<Mutex<Vec<(Vec<u8>, Vec<u8>)>>>, 
        total_attempts: &mut usize,
        iterations: &mut usize
    ) -> Result<()> {
        while *iterations < 100 {
            let batch_start = Instant::now();
            let batch_size = self.config.batch_size;

            // CPU实现使用Rayon
            let attempts_per_thread = batch_size / self.config.threads;
            let results: Vec<(Vec<u8>, Vec<u8>)> = (0..self.config.threads)
                .into_par_iter()
                .flat_map(|_| {
                    let mut local_results = Vec::new();
                    for _ in 0..attempts_per_thread {
                        // 生成随机种子
                        let mut seed = [0u8; 32];
                        OsRng.fill_bytes(&mut seed);

                        // 转换为密钥对
                        if let Some((seed_vec, pubkey)) = self.create_keypair(&seed) {
                            local_results.push((seed_vec, pubkey));
                        }
                    }
                    local_results
                })
                .collect();

            // 添加结果到found_keys向量
            if !results.is_empty() {
                let mut keys = found_keys.lock().unwrap();
                keys.extend(results.clone());
                info!("Found {} new keys in this batch", results.len());
            }

            *total_attempts += batch_size;
            *iterations += 1;

            let batch_duration = batch_start.elapsed();
            let attempts_per_second = batch_size as f64 / batch_duration.as_secs_f64();

            info!(
                "Batch {}: {} attempts in {:.2}s ({:.2} attempts/s), {} keys found",
                *iterations,
                batch_size,
                batch_duration.as_secs_f64(),
                attempts_per_second,
                found_keys.lock().unwrap().len()
            );

            // 周期性保存结果
            if *iterations % 10 == 0 {
                self.save_results()?;
            }
        }
        
        Ok(())
    }

    // 为GPU准备前缀数据
    #[cfg(feature = "cuda")]
    fn prepare_prefixes_for_gpu(&self) -> Result<GpuPrefixData> {
        use std::ffi::c_void;
        use std::mem::size_of;
        use std::os::raw::c_char;
        use std::ptr;
        
        let mut prefixes = [[0u8; MAX_PREFIX_LENGTH]; MAX_PREFIXES];
        let mut prefix_lengths = [0u32; MAX_PREFIXES];
        let prefix_count = self.config.prefixes.len().min(MAX_PREFIXES) as u32;
        
        // 填充前缀数据
        for (i, prefix) in self.config.prefixes.iter().take(MAX_PREFIXES).enumerate() {
            let len = prefix.len().min(MAX_PREFIX_LENGTH);
            prefix_lengths[i] = len as u32;
            
            for (j, c) in prefix.chars().take(MAX_PREFIX_LENGTH).enumerate() {
                prefixes[i][j] = c as u8;
            }
        }
        
        Ok(GpuPrefixData {
            prefixes,
            prefix_lengths,
            prefix_count,
        })
    }
    
    // 为GPU准备后缀数据
    #[cfg(feature = "cuda")]
    fn prepare_suffixes_for_gpu(&self) -> Result<GpuSuffixData> {
        use std::ffi::c_void;
        use std::mem::size_of;
        use std::os::raw::c_char;
        use std::ptr;
        
        let mut suffixes = [[0u8; MAX_SUFFIX_LENGTH]; MAX_SUFFIXES];
        let mut suffix_lengths = [0u32; MAX_SUFFIXES];
        let suffix_count = self.config.suffixes.len().min(MAX_SUFFIXES) as u32;
        
        // 填充后缀数据
        for (i, suffix) in self.config.suffixes.iter().take(MAX_SUFFIXES).enumerate() {
            let len = suffix.len().min(MAX_SUFFIX_LENGTH);
            suffix_lengths[i] = len as u32;
            
            for (j, c) in suffix.chars().take(MAX_SUFFIX_LENGTH).enumerate() {
                suffixes[i][j] = c as u8;
            }
        }
        
        Ok(GpuSuffixData {
            suffixes,
            suffix_lengths,
            suffix_count,
        })
    }
    
    // 在指定GPU上启动内核
    #[cfg(feature = "cuda")]
    fn launch_gpu_kernel(
        &mut self, 
        device_id: usize, 
        prefix_data: &GpuPrefixData,
        suffix_data: &GpuSuffixData,
        batch_size: usize
    ) -> Result<()> {
        let kernel = self.gpu_kernel.as_mut().unwrap();
        
        // 调用新的run_kernel方法
        kernel.run_kernel(
            device_id as u32,
            prefix_data,
            suffix_data,
            self.config.ignore_case,
            batch_size as u32,
        )?;
        
        Ok(())
    }
    
    // 收集所有GPU的结果
    #[cfg(feature = "cuda")]
    fn collect_gpu_results(&mut self) -> Result<usize> {
        let kernel = self.gpu_kernel.as_mut().unwrap();
        let mut total_found = 0;
        
        for device_id in 0..kernel.ctx.devices.len() {
            // 获取设备结果
            let (seeds, pubkeys, found_count) = kernel.get_results(device_id as u32)?;
            
            if found_count > 0 {
                let count = found_count as usize;
                
                // 处理每个找到的密钥
                let mut keys = self.found_keys.lock().unwrap();
                for i in 0..count {
                    let seed = seeds[i*32..(i+1)*32].to_vec();
                    let pubkey = pubkeys[i*32..(i+1)*32].to_vec();
                    keys.push((seed, pubkey));
                }
                
                total_found += count;
            }
        }
        
        Ok(total_found)
    }

    fn create_keypair(&self, seed: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        // Hash the seed with SHA-512 to get the private key material
        let mut hasher = Sha512::new();
        hasher.update(seed);
        let hash = hasher.finalize();
        
        // Apply Ed25519 clamping
        let mut key_bytes = hash.as_slice()[..32].to_vec();
        key_bytes[0] &= 248;
        key_bytes[31] &= 63;
        key_bytes[31] |= 64;
        
        // Create a signing key with ed25519-dalek v2
        let signing_key = match key_bytes.try_into() {
            Ok(bytes32) => {
                match SigningKey::from_bytes(&bytes32) {
                    Ok(sk) => sk,
                    Err(_) => return None,
                }
            },
            Err(_) => return None,
        };
        
        // Get the public key
        let public_key = signing_key.verifying_key();
        let pubkey_bytes = public_key.to_bytes();
        
        // Encode the public key to base58
        let encoded = bs58::encode(&pubkey_bytes).into_string();
        
        // Check if the encoded key matches any of our prefixes
        let matches_prefix = self.config.prefixes.iter().any(|prefix| {
            if self.config.ignore_case {
                encoded.to_lowercase().starts_with(&prefix.to_lowercase())
            } else {
                encoded.starts_with(prefix)
            }
        });
        
        // Check if the encoded key matches any of our suffixes
        let matches_suffix = self.config.suffixes.iter().any(|suffix| {
            if self.config.ignore_case {
                encoded.to_lowercase().ends_with(&suffix.to_lowercase())
            } else {
                encoded.ends_with(suffix)
            }
        });
        
        if (matches_prefix || self.config.prefixes.is_empty()) &&
           (matches_suffix || self.config.suffixes.is_empty()) &&
           (matches_prefix || matches_suffix) {
            Some((seed.to_vec(), pubkey_bytes.to_vec()))
        } else {
            None
        }
    }

    fn save_results(&self) -> Result<()> {
        if let Some(output_file) = &self.config.output_file {
            let keys = self.found_keys.lock().unwrap();
            if keys.is_empty() {
                return Ok(());
            }

            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .append(true)
                .open(output_file)?;

            for (seed, pubkey) in keys.iter() {
                let seed_b58 = bs58::encode(seed).into_string();
                let pubkey_b58 = bs58::encode(pubkey).into_string();
                writeln!(file, "{}:{}", pubkey_b58, seed_b58)?;
            }

            info!("Saved {} keys to {}", keys.len(), output_file);
        }

        Ok(())
    }
}

impl Drop for VanityGenerator {
    fn drop(&mut self) {
        if let Some(kernel) = &mut self.gpu_kernel {
            #[cfg(feature = "cuda")]
            {
                if let Err(e) = kernel.cleanup() {
                    eprintln!("Error cleaning up CUDA resources: {}", e);
                }
            }
        }
    }
}

// 实现Send和Sync特性
// 注意：这是一种变通解决方案，仅用于开发和测试
// 在生产环境中需要更谨慎地处理GPU资源的线程安全性
#[cfg(feature = "cuda")]
unsafe impl Send for VanityGenerator {}

#[cfg(feature = "cuda")]
unsafe impl Sync for VanityGenerator {} 