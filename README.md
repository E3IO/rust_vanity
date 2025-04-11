# Rust Vanity

一个高性能的ed25519密钥vanity地址生成器，基于Rust实现，支持GPU加速。

## 功能特点

- 多线程CPU支持，使用Rayon进行并行化
- CUDA GPU支持，实现高性能密钥生成
- **多GPU支持**，自动检测并利用系统中的所有可用GPU
- 支持前缀和后缀匹配
- 支持大小写不敏感匹配
- 批量生成并保存到文件
- 优化的Base58编码实现
- 支持通配符（用"?"表示）

## 编译要求

- Rust 1.53+
- CUDA Toolkit 10.0+ (可选，仅在启用GPU支持时需要)

## 安装

```
git clone https://github.com/yourusername/rust_vanity.git
cd rust_vanity
```

### 编译CUDA源码

如果需要启用CUDA支持，首先需要编译CUDA源码为PTX文件：

```bash
# 确保CUDA工具链已安装
./compile_cuda.sh
```

然后再编译Rust项目：

```bash
cargo build --release
```

如果不需要CUDA支持，可以禁用默认特性：

```
cargo build --release --no-default-features
```

### 解决编译问题

如果遇到以下常见问题，可以按照对应方法解决：

1. **找不到PTX文件**：
   - 运行`./compile_cuda.sh`生成PTX文件
   - 确保PTX文件位于`resources/vanity_kernel.ptx`

2. **rustacuda API版本不兼容**：
   - 如果发生API更改，请查阅rustacuda最新文档
   - 修改代码以匹配最新的API调用方式

3. **多GPU环境中的线程安全问题**：
   - 在`VanityGenerator`结构体中添加标记：
   ```rust
   unsafe impl Send for VanityGenerator {}
   unsafe impl Sync for VanityGenerator {}
   ```
   - 注意：这是一种变通解决方案，实际应用中需谨慎使用

## CUDA实现细节

本项目使用rustacuda库来实现多GPU支持。整体架构如下：

1. **CudaContext**：负责CUDA环境初始化和多设备管理
2. **VanityKernel**：封装CUDA内核实现，支持多GPU并行执行
3. **GpuPrefixData/GpuSuffixData**：在CPU和GPU之间传输数据的结构

### PTX文件生成

CUDA源码需要使用nvcc编译为PTX文件，这是使用rustacuda执行GPU代码所必需的：

```bash
nvcc -ptx -o resources/vanity_kernel.ptx cuda_src/vanity.cu
```

生成的PTX文件包含两个主要函数：
- `vanity_init`：初始化CUDA随机数生成器
- `vanity_scan`：执行vanity地址搜索

### 多GPU支持架构

我们的多GPU实现采用以下架构：

1. 自动检测系统中所有可用的CUDA GPU
2. 为每个GPU创建独立的内存和执行上下文
3. 在所有GPU上并行执行vanity搜索
4. 使用原子操作安全地收集来自各GPU的结果

在多GPU系统上，性能提升几乎是线性的 - 两块相同的GPU将提供接近两倍的性能。

## 使用方法

```
USAGE:
    rust_vanity [OPTIONS]

OPTIONS:
    -p, --prefixes <PREFIXES>       前缀列表，用逗号分隔
    -s, --suffixes <SUFFIXES>       后缀列表，用逗号分隔
    -i, --ignore-case <BOOL>        是否忽略大小写匹配 [default: true]
    -g, --gpu <BOOL>                是否使用GPU加速 [default: true]
    -t, --threads <NUM>             CPU线程数 [default: 8]
    -b, --batch-size <NUM>          每批处理的数量 [default: 100000]
    -o, --output <FILE>             输出文件名 [default: found_keys.txt]
    -v, --verbose                   启用详细日志
    -h, --help                      显示帮助信息
    -V, --version                   显示版本信息
```

## 示例

查找以"hello"开头的地址：

```
./rust_vanity -p hello
```

查找以"world"结尾的地址：

```
./rust_vanity -s world
```

同时查找多个前缀：

```
./rust_vanity -p hello,hi,hey
```

使用通配符查找某个模式（例如，以"a"开头，第三位是"c"）：

```
./rust_vanity -p "a?c"
```

禁用GPU加速，仅使用CPU：

```
./rust_vanity -p hello -g false
```

## 多GPU环境测试

要在多GPU环境下测试性能，请按照以下步骤操作：

1. 运行并观察自动检测到的GPU数量：
   ```bash
   ./rust_vanity -v
   ```

2. 测试单GPU性能作为基准：
   ```bash
   CUDA_VISIBLE_DEVICES=0 ./rust_vanity -p test -b 1000000
   ```

3. 测试多GPU性能：
   ```bash
   ./rust_vanity -p test -b 1000000
   ```

4. 比较单GPU和多GPU的性能差异，应接近线性扩展

## 输出格式

生成的密钥对将保存到指定的输出文件中，每行一个密钥对，格式为：

```
public_key:private_key
```

两个值都以Base58编码表示。

## 性能优化

1. GPU性能提示：
   - 增加批处理大小可以提高GPU利用率
   - 在多GPU系统上自动使用所有可用GPU
   - 对于大型GPU，尝试增加`--batch-size`参数以获得更好的性能
   - 如需更高性能，可以自定义编译选项：`nvcc -arch=sm_75 -O3 ...`（根据GPU架构调整）

2. CPU性能提示：
   - 调整线程数以匹配您的CPU核心数
   - 较小的批处理大小对CPU更有效

## 故障排除

如果遇到以下问题：

1. **"No CUDA devices found"**：
   - 确保CUDA驱动程序已正确安装
   - 确保环境变量`CUDA_PATH`设置正确
   - 尝试运行`nvidia-smi`检查GPU状态

2. **编译错误"cannot find -lcuda"**：
   - 确保CUDA工具包已正确安装
   - 设置`LD_LIBRARY_PATH`指向CUDA库目录

3. **PTX文件加载错误**：
   - 重新运行`./compile_cuda.sh`
   - 确保使用的CUDA工具链版本与运行时兼容

## 许可证

MIT

## 注意事项

1. 此软件仅用于教育和研究目的
2. 不要使用此软件生成的密钥用于存储实际资产
3. 使用此软件产生的所有后果由用户自行承担 