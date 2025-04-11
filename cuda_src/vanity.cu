#include <curand_kernel.h>
#include <stdint.h>

extern "C" {

// 常量定义
#define MAX_BATCH_SIZE 100000
#define MAX_PREFIXES 16
#define MAX_PREFIX_LENGTH 16
#define MAX_SUFFIXES 16
#define MAX_SUFFIX_LENGTH 16

// Base58字符表
__device__ const char b58digits_ordered[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Base58编码函数
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

// 检查是否匹配前缀或后缀
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
    
    // 检查前缀
    for (uint32_t i = 0; i < prefix_count; ++i) {
        if (prefix_lengths[i] == 0) {
            continue;
        }
        
        bool match = true;
        for (uint32_t j = 0; j < prefix_lengths[i] && j < key_len; ++j) {
            char key_char = key[j];
            char prefix_char = prefixes[i][j];
            
            // 处理大小写不敏感
            if (ignore_case) {
                if (key_char >= 'A' && key_char <= 'Z') {
                    key_char += 32;  // 转换为小写
                }
                if (prefix_char >= 'A' && prefix_char <= 'Z') {
                    prefix_char += 32;  // 转换为小写
                }
            }
            
            // 支持通配符'?'
            if (prefix_char != '?' && key_char != prefix_char) {
                match = false;
                break;
            }
        }
        
        if (match) {
            return true;
        }
    }
    
    // 检查后缀
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
            
            // 处理大小写不敏感
            if (ignore_case) {
                if (key_char >= 'A' && key_char <= 'Z') {
                    key_char += 32;  // 转换为小写
                }
                if (suffix_char >= 'A' && suffix_char <= 'Z') {
                    suffix_char += 32;  // 转换为小写
                }
            }
            
            // 支持通配符'?'
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

// 初始化CUDA随机数生成器
__global__ void vanity_init(curandState* state) {
    int id = threadIdx.x + (blockIdx.x * blockDim.x);
    curand_init(580000 + id, id, 0, &state[id]);
}

// 模拟ED25519算法的简化版本（实际应用中需使用完整实现）
__device__ void simple_ed25519_pubkey(uint8_t* seed, uint8_t* pubkey) {
    // 这里简化处理，仅作为示例
    // 实际应用中应当使用完整的ED25519算法
    for (int i = 0; i < 32; i++) {
        pubkey[i] = seed[i] ^ 0xAA; // 简单异或作为示例
    }
}

// 主要的vanity搜索内核
__global__ void vanity_scan(
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
    
    // 本地内核状态
    curandState localState = state[id];
    unsigned char seed[32] = {0};
    unsigned char pubkey[32] = {0};
    char key[256] = {0};
    
    // 使用本地计数器以避免原子操作竞争
    uint32_t local_found = 0;
    
    for (uint32_t attempts = 0; attempts < batch_size; ++attempts) {
        // 生成随机种子
        uint32_t* seed_u32 = (uint32_t*)seed;
        for (int i = 0; i < 8; ++i) {
            seed_u32[i] = curand(&localState);
        }
        
        // 生成公钥（简化版本）
        simple_ed25519_pubkey(seed, pubkey);
        
        // Base58编码公钥
        size_t keysize = 256;
        b58enc(key, &keysize, pubkey, 32);
        
        // 检查是否匹配前缀或后缀
        if (check_matches(key, keysize-1, prefixes, prefix_lengths, prefix_count, 
                        suffixes, suffix_lengths, suffix_count, ignore_case)) {
            uint32_t idx = atomicAdd(found_count, 1);
            if (idx < MAX_BATCH_SIZE) {
                // 保存种子和公钥
                for (int i = 0; i < 32; i++) {
                    found_seeds[idx * 32 + i] = seed[i];
                    found_pubkeys[idx * 32 + i] = pubkey[i];
                }
                local_found++;
            }
        }
    }
    
    // 保存随机状态以便将来调用
    state[id] = localState;
}

} // extern "C" 