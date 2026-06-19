# mamba-windows-cuda vs 官方 mamba-ssm 对比

## 概览

| 特性 | 官方 mamba-ssm | mamba-windows-cuda (本项目) |
|------|----------------|---------------------------|
| **平台** | Linux (主要) | Windows + Linux |
| **GPU 支持** | NVIDIA (CUDA), AMD (ROCm) | NVIDIA (CUDA) |
| **Python** | 3.8+ | 3.9+ |
| **PyTorch** | 1.12+ | 2.0+ |
| **安装方式** | `pip install mamba-ssm` | `pip install -e .` (JIT编译) |
| **维护者** | Tri Dao, Albert Gu | zouyuanqing |

## 功能对比

| 功能 | 官方 mamba-ssm | mamba-windows-cuda |
|------|----------------|-------------------|
| **Selective Scan** | ✅ 完整实现 | ✅ N=16 专用 |
| **前向传播** | ✅ CUDA kernel | ✅ CUDA kernel |
| **反向传播** | ✅ CUDA kernel | ✅ PyTorch + CUDA |
| **FP16** | ✅ | ✅ |
| **FP32** | ✅ | ✅ |
| **BF16** | ✅ | ❌ |
| **Complex64** | ✅ | ❌ |
| **Variable B/C** | ✅ (B,C随时间变化) | ❌ (B,C固定) |
| **h_prev (流式)** | ✅ | ✅ |
| **delta_softplus** | ✅ | ❌ |
| **delta_bias** | ✅ | ❌ |
| **z (门控)** | ✅ | ❌ |
| **Mamba Block** | ✅ 完整模块 | ❌ 仅算子 |
| **Mamba-2** | ✅ | ❌ |
| **Mamba-3** | ✅ | ❌ |
| **并行扫描** | ✅ (Mamba-2) | ✅ (pscan.py) |
| **Kernel Fusion** | ✅ | ✅ |
| **Windows 支持** | ❌ (需WSL) | ✅ 原生 |

## 性能对比

### 官方 mamba-ssm (A100 80GB)

根据论文和社区 benchmark：

| 操作 | d_model=256, L=2048 | d_model=768, L=4096 |
|------|---------------------|---------------------|
| Forward | ~5ms | ~25ms |
| Backward | ~12ms | ~60ms |
| 总计 | ~17ms | ~85ms |

**关键特性**：
- 使用 Triton 编写的优化 kernel
- 针对 A100/H100 深度优化
- 支持 tensor cores (Mamba-2)
- 理论吞吐量：~312 TFLOPS (H100)

### mamba-windows-cuda (RTX 5070 Laptop)

| 操作 | d_model=256, L=2048 | d_model=768, L=4096 |
|------|---------------------|---------------------|
| Forward (原始) | ~15ms | ~45ms |
| Forward (Fused) | ~10ms | ~29ms |
| Backward (原始) | ~40ms | ~120ms |
| Backward (Fused) | ~20ms | ~56ms |
| **总计 (Fused)** | **~30ms** | **~85ms** |

### 性能差异分析

| 因素 | 影响 |
|------|------|
| **GPU 差异** | RTX 5070 vs A100: A100 ~2-3x 更快 |
| **内存带宽** | A100: 2TB/s, RTX 5070: ~500GB/s |
| **Tensor Cores** | 官方 Mamba-2 使用 tensor cores |
| **优化程度** | 官方经过多年优化，本项目较新 |
| **N=16 专用** | 本项目只支持 N=16，更简单 |

**归一化对比**（考虑 GPU 差异）：

| 操作 | 官方 (A100) | 本项目 (5070) | 比率 |
|------|-------------|---------------|------|
| Forward | 5ms | 10ms | ~2x (GPU差异) |
| Backward | 12ms | 20ms | ~1.7x |
| 总计 | 17ms | 30ms | ~1.8x |

**结论**：考虑 GPU 差异后，性能差距约 1.5-2x，主要来自：
1. 官方使用 Triton 优化的 kernel
2. 官方 Mamba-2 使用 tensor cores
3. 本项目是简化实现，专注 Windows 兼容性

## 代码结构对比

### 官方 mamba-ssm
```
state-spaces/mamba/
├── csrc/
│   ├── selective_scan/
│   │   ├── selective_scan.cpp           # C++ 接口
│   │   ├── selective_scan_cuda.cu       # CUDA 主实现
│   │   ├── selective_scan_fwd_kernel.cuh
│   │   └── selective_scan_bwd_kernel.cuh
│   └── causal_conv1d/
├── mamba_ssm/
│   ├── modules/
│   │   ├── mamba_simple.py              # Mamba-1 模块
│   │   ├── mamba2.py                    # Mamba-2 模块
│   │   └── mamba3.py                    # Mamba-3 模块
│   ├── ops/
│   │   └── selective_scan_interface.py  # Python 接口
│   └── models/
└── benchmarks/
```

### mamba-windows-cuda
```
mamba-windows-cuda/
├── mamba_windows_cuda/
│   ├── mamba_cuda.py                    # 主实现 (前向+反向)
│   ├── selective_scan_cuda_ext.py       # CUDA backward + fusion
│   ├── pscan.py                         # Parallel scan
│   └── tests/
├── benchmark.py
└── pyproject.toml
```

## 适用场景

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| **生产环境 (Linux)** | 官方 mamba-ssm | 性能最优，功能完整 |
| **Windows 开发** | mamba-windows-cuda | 原生支持，无需 WSL |
| **快速原型** | mamba-windows-cuda | JIT 编译，易于修改 |
| **N=16 专用** | mamba-windows-cuda | 更简单，针对性优化 |
| **完整 Mamba 模型** | 官方 mamba-ssm | 包含完整 Block 和模型 |
| **Mamba-2/3** | 官方 mamba-ssm | 本项目不支持 |
| **学术研究** | 两者皆可 | 看具体需求 |

## 本项目的优势

1. **Windows 原生支持**
   - 无需 WSL 或虚拟机
   - 自动检测 MSVC 环境
   - JIT 编译，开箱即用

2. **简化实现**
   - 专注 N=16 (Mamba 常用配置)
   - 代码更易理解和修改
   - 适合学习和研究

3. **渐进式优化**
   - 原始实现 → Parallel Scan → CUDA Backward → Kernel Fusion
   - 每步都有对应代码，便于学习

4. **完整测试**
   - 流式一致性测试
   - 梯度正确性测试
   - 数值稳定性测试

## 本项目的局限

1. **功能有限**
   - 只支持 N=16
   - 不支持 variable B/C
   - 不支持 delta_softplus/bias
   - 不支持 z (门控)

2. **性能差距**
   - 比官方慢 ~1.5-2x (归一化后)
   - 没有使用 tensor cores
   - Backward 已升级为 CUDA kernel（v0.5.0，Mamba-1 重计算策略）；`h_prev` 模式仍使用 PyTorch backward

3. **平台限制**
   - 主要针对 Windows
   - Linux 支持未充分测试

## 如何选择

```
你的需求是什么？
│
├── 需要完整 Mamba 模型？
│   └── 是 → 官方 mamba-ssm
│
├── 需要 Windows 原生支持？
│   └── 是 → mamba-windows-cuda
│
├── 需要 Mamba-2/3？
│   └── 是 → 官方 mamba-ssm
│
├── 需要学习 selective scan？
│   └── 是 → mamba-windows-cuda (代码更简单)
│
├── 需要最高性能？
│   └── 是 → 官方 mamba-ssm (Linux + A100/H100)
│
└── 需要快速原型开发？
    └── 是 → mamba-windows-cuda (JIT 编译)
```

## 未来计划

- [ ] 支持 N=32, N=64
- [ ] 支持 variable B/C
- [ ] 实现 Triton kernel (跨平台)
- [ ] 支持 Mamba-2 SSD 算法
- [ ] 添加 tensor cores 支持
- [ ] 优化 Windows 下的编译速度

## 参考链接

- **官方仓库**: https://github.com/state-spaces/mamba
- **Mamba 论文**: https://arxiv.org/abs/2312.00752
- **Mamba-2 论文**: https://arxiv.org/abs/2405.21060
- **本项目**: https://github.com/zouyuanqing/mamba-windows-cuda
