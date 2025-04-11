# 用法总结

为了使Rust Vanity在多GPU环境中正常运行，请遵循以下步骤：

1. 编译CUDA源码：
   ```
   ./compile_cuda.sh
   ```

2. 构建Rust项目：
   ```
   cargo build --release
   ```

3. 运行vanity地址生成器：
   ```
   ./target/release/rust_vanity -p 你的前缀 -s 你的后缀
   ```

4. 查看生成的密钥：
   ```
   cat found_keys.txt
   ```

## 故障排除

如果遇到编译或运行问题：
1. 确保CUDA工具链正确安装
2. 检查PTX文件是否已生成
3. 确保rustacuda库正确安装

## 多GPU测试

```
# 查看所有可用GPU
nvidia-smi

# 测试程序
./target/release/rust_vanity -v
``` 