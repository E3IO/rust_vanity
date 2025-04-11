use std::ffi::CString;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use log::{debug, info};

// 导出vanity_kernel模块
pub mod vanity_kernel;
// 导出VanityKernel
pub use vanity_kernel::VanityKernel;

#[cfg(feature = "cuda")]
use rustacuda::prelude::*;
#[cfg(feature = "cuda")]
use rustacuda::memory::{DeviceBox, DeviceBuffer, DeviceCopy};
#[cfg(feature = "cuda")]
use rustacuda::function::{BlockSize, GridSize};
#[cfg(feature = "cuda")]
use rustacuda::device::DeviceAttribute;

#[cfg(feature = "cuda")]
pub struct CudaDevice {
    pub device_id: u32,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub warp_size: i32,
    pub processor_count: i32,
    pub max_threads_per_block: u32,
    pub max_threads_dim: [u32; 3],
    pub max_grid_size: [u32; 3],
}

#[cfg(feature = "cuda")]
pub struct CudaContext {
    pub context: Context,
    pub devices: Vec<CudaDevice>,
    pub current_device: u32,
    pub stream: Stream,
}

#[cfg(feature = "cuda")]
impl CudaContext {
    pub fn new() -> Result<Self> {
        // 初始化CUDA
        rustacuda::init(CudaFlags::empty())?;
        
        // 获取设备数量
        let device_count = Device::count()?;
        if device_count == 0 {
            return Err(anyhow!("No CUDA devices found"));
        }
        
        // 获取第一个设备并创建上下文
        let device = Device::get_device(0)?;
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, 
            device
        )?;
        
        // 创建流
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        
        // 收集所有设备信息
        let mut devices = Vec::new();
        
        for i in 0..device_count {
            let device = Device::get_device(i)?;
            let name = device.name()?;
            
            // 获取设备属性
            let major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
            let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
            let compute_capability = (major, minor);
            
            let warp_size = device.get_attribute(DeviceAttribute::WarpSize)?;
            let processor_count = device.get_attribute(DeviceAttribute::MultiprocessorCount)?;
            let max_threads_per_block = device.get_attribute(DeviceAttribute::MaxThreadsPerBlock)? as u32;
            
            let max_block_dim_x = device.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
            let max_block_dim_y = device.get_attribute(DeviceAttribute::MaxBlockDimY)? as u32;
            let max_block_dim_z = device.get_attribute(DeviceAttribute::MaxBlockDimZ)? as u32;
            
            let max_grid_dim_x = device.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
            let max_grid_dim_y = device.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
            let max_grid_dim_z = device.get_attribute(DeviceAttribute::MaxGridDimZ)? as u32;
            
            let cuda_device = CudaDevice {
                device_id: i,
                name,
                compute_capability,
                warp_size,
                processor_count,
                max_threads_per_block,
                max_threads_dim: [max_block_dim_x, max_block_dim_y, max_block_dim_z],
                max_grid_size: [max_grid_dim_x, max_grid_dim_y, max_grid_dim_z],
            };
            
            info!("GPU {}: {} (Compute Capability {}.{})", 
                i, cuda_device.name, 
                cuda_device.compute_capability.0, cuda_device.compute_capability.1);
            info!("   - Multiprocessors: {}, Warp Size: {}", 
                cuda_device.processor_count, cuda_device.warp_size);
            info!("   - Max Threads Per Block: {}", cuda_device.max_threads_per_block);
            
            devices.push(cuda_device);
        }
        
        Ok(Self {
            context,
            devices,
            current_device: 0,
            stream,
        })
    }

    pub fn set_device(&mut self, device_id: u32) -> Result<()> {
        if device_id >= self.devices.len() as u32 {
            return Err(anyhow!("Invalid device ID: {}", device_id));
        }
        
        // rustacuda库没有set_device方法，但我们可以使用重新创建context的方式
        // 先获取设备
        let device = Device::get_device(device_id)?;
        
        // 弹出当前上下文
        self.context.pop()?;
        
        // 创建并推送新的上下文
        self.context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        )?;
        
        self.current_device = device_id;
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
pub struct CudaContext {}

#[cfg(not(feature = "cuda"))]
impl CudaContext {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }
} 