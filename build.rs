use std::env;
use std::path::PathBuf;

fn main() {
    // 检查是否启用了CUDA特性
    if cfg!(feature = "cuda") {
        // 设置CUDA链接路径
        let cuda_path = PathBuf::from(env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string()));
        let cuda_lib_path = cuda_path.join("lib64");
        
        println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-lib=dylib=curand");
        
        // 重新编译条件
        println!("cargo:rerun-if-changed=src/cuda");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
    }
} 