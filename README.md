# mamba-windows-cuda

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">

**[English](#english)** | **[中文](#中文)**

</div>

## English

### mamba-windows-cuda

Windows-compatible CUDA implementation of Mamba selective-scan operation, targeting **N=16** SSM (common Mamba configuration), supporting:

- FP16 / FP32
- `h_prev` as input final state (for streaming/chunking)
- `forward_with_state()` returns `h_last` (for chunked concatenation of ultra-long sequences)

The package performs **JIT compilation** of CUDA extensions locally via `torch.utils.cpp_extension.load_inline()`: initial import/first instantiation triggers compilation, subsequent use reuses cached artifacts.

### Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

### Features

- ⚡ **Fast**: CUDA-accelerated selective scan with kernel fusion (1.4-1.8x speedup)
- 🔮 **Kernel Fusion**: Discretization + scan + output merged into single CUDA kernel
- 🔙 **CUDA Backward**: Native CUDA backward pass with Mamba-1 recomputation strategy
- 🔀 **Parallel Scan**: Blelloch work-efficient parallel scan for long sequences
- 💾 **Checkpoint Strategies**: Configurable memory/speed tradeoff (40%+ memory savings)
- 🏁 **Windows Support**: Full Windows compatibility with automatic MSVC environment setup
- 🔧 **JIT Compilation**: On-the-fly compilation of CUDA kernels
- 🧪 **Thoroughly Tested**: Comprehensive test coverage with numerical accuracy validation
- 📐 **Flexible Shapes**: Support for various batch sizes, sequence lengths, and dimensions

### Requirements

- Python >= 3.9
- PyTorch with CUDA support
- NVIDIA CUDA Toolkit (for compilation)
- Visual Studio / Build Tools (MSVC) for Windows

### Installation

#### From Source

```bash
pip install -e .
```

### Usage

#### Basic Usage

```python
import torch
from mamba_windows_cuda import SelectiveScanCuda

# Initialize the kernel
kernel = SelectiveScanCuda().cuda()

# Prepare input tensors
B, L, D, N = 1, 1024, 256, 16  # batch, length, dim, state

u = torch.randn(B, L, D, device='cuda').half()
delta = torch.ones(B, L, D, device='cuda').half()
A = (-torch.rand(D, N, device='cuda') * 0.1).half()
B_ssm = torch.randn(B, L, N, device='cuda').half()
C_ssm = torch.randn(B, L, N, device='cuda').half()
D_ssm = torch.ones(D, device='cuda').half()

# Basic forward pass
out = kernel(u, delta, A, B_ssm, C_ssm, D_ssm)

# Forward pass with state
out, h_last = kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm)
```

#### Streaming Usage (for long sequences)

```python
import torch
from mamba_windows_cuda import SelectiveScanCuda

kernel = SelectiveScanCuda().cuda()

# Initialize state
h = torch.zeros(1, 2048, 16, device='cuda').half()

# Process sequence in chunks
for chunk in sequence_chunks:
    out, h = kernel.forward_with_state(chunk['u'], chunk['delta'], 
                                      chunk['A'], chunk['B'], 
                                      chunk['C'], chunk['D'], h_prev=h)
```

#### Training with Checkpoint Strategy

```python
import torch
from mamba_windows_cuda import SelectiveScanCuda

kernel = SelectiveScanCuda().cuda()

# forward() uses CUDA backward by default (no h_prev)
u = torch.randn(1, 1024, 256, device='cuda', requires_grad=True)
delta = torch.ones(1, 1024, 256, device='cuda', requires_grad=True)
out = kernel(u, delta, A, B_ssm, C_ssm, D_ssm)  # CUDA backward
loss = out.sum()
loss.backward()  # efficient CUDA backward pass
```

- When `h_prev` is provided, falls back to PyTorch backward with saved intermediates.
- See `selective_scan_cuda_ext.py` for checkpoint level configuration.

### API Reference

#### `SelectiveScanCuda`

The main class implementing the CUDA-accelerated selective scan operation.

##### Backward Behavior

- `forward()` (without `h_prev`): Uses **CUDA backward kernel** with Mamba-1 recomputation (memory efficient).
- `forward_with_state()` or `forward(h_prev=...)`: Falls back to **PyTorch backward** with saved intermediates (handles stateful mode).

##### Methods

- `forward(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: Forward pass with CUDA backward (default) or PyTorch backward (when `h_prev` is provided)
- `forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: Forward pass returning `(output, h_last)` state tuple

#### `parallel_scan_fn` / `parallel_scan_ref`

Blelloch work-efficient parallel scan (O(log n) depth). Exported from `pscan.py`.

```python
from mamba_windows_cuda import parallel_scan_fn, parallel_scan_ref

# parallel_scan_fn: CUDA-accelerated parallel scan
# parallel_scan_ref: Reference sequential scan for validation
```

##### Parameters

All parameters should be PyTorch tensors on CUDA device with `float16` or `float32` dtype.

- `u`: Input tensor of shape `(B, L, D)`
- `delta`: Delta tensor of shape `(B, L, D)`
- `A`: A matrix of shape `(D, 16)` (currently only N=16 is supported)
- `B_ssm`: B matrix of shape `(B, L, 16)`
- `C_ssm`: C matrix of shape `(B, L, 16)`
- `D_ssm`: D vector of shape `(D,)`
- `h_prev`: Optional initial hidden state of shape `(B, D, 16)`

##### Returns

- `forward()`: Output tensor of shape `(B, L, D)`
- `forward_with_state()`: Tuple of `(output_tensor, final_hidden_state)` where final_hidden_state is of shape `(B, D, 16)`

#### Tensor Shapes and Dtype Constraints

- `u`: `(B, L, D)`
- `delta`: `(B, L, D)`
- `A`: `(D, 16)` (currently only supports `N=16`)
- `B_ssm`: `(B, L, 16)`
- `C_ssm`: `(B, L, 16)`
- `D_ssm`: `(D,)`
- `h_prev` (optional): `(B, D, 16)`
- `out`: `(B, L, D)`
- `h_last` (optional return): `(B, D, 16)`

Dtype: `float16` or `float32`, and `u/delta/A/B_ssm/C_ssm/D_ssm/h_prev` must have the same dtype.

### Windows Compilation Dependencies

This package requires the ability to compile PyTorch CUDA extensions on Windows, which typically requires:

- PyTorch with CUDA installed (`torch.cuda.is_available()` returns `True`)
- NVIDIA CUDA Toolkit (for `nvcc` compiler)
- Visual Studio / Build Tools (MSVC)

The code attempts to automatically set up the MSVC environment variables via `vswhere.exe` + `VsDevCmd.bat` (see `mamba_windows_cuda/mamba_cuda.py`).

If you want to explicitly control the compilation target architecture, you can set:

- `TORCH_CUDA_ARCH_LIST` (e.g., `8.6`, `8.9`, etc.)

### Testing

```bash
python -m unittest -v mamba_windows_cuda.tests.test_selective_scan
```

#### Test Coverage

- FP16/FP32 numerical consistency with reference implementation (error threshold)
- Non-zero initial state tests (streaming/chunking scenarios)
- Streaming consistency tests (chunked vs full sequence execution)
- Gradient correctness tests (autograd flow + numerical accuracy)
- Long sequences and extreme sizes: `L=16384/32768`, `D=768/1024/2048`
- Image feature sequence streaming: `L=1280*1280`, `D=2048` (chunked with `h_last`)
- Input validation tests (wrong shapes, wrong types)
- Numerical stability tests (extreme values)

### Performance

See [BENCHMARK_RESULTS.md](https://github.com/zouyuanqing/mamba-windows-cuda/blob/main/BENCHMARK_RESULTS.md) for full data.

#### Kernel Fusion Speedup (RTX 5070 Laptop, FP16)

| Config | Original | Fused | Speedup |
|--------|----------|-------|---------|
| B=1, L=512, D=256 | 2.50ms | 1.80ms | **1.39x** |
| B=1, L=1024, D=256 | 5.10ms | 3.40ms | **1.50x** |
| B=1, L=2048, D=512 | 15.20ms | 9.80ms | **1.55x** |
| B=1, L=4096, D=768 | 45.30ms | 28.50ms | **1.59x** |
| B=1, L=8192, D=1024 | 120.50ms | 72.30ms | **1.67x** |
| B=1, L=16384, D=2048 | 350.20ms | 195.80ms | **1.79x** |

#### Training Speedup (Forward + Backward)

| Config | Original | Fused(ckpt0) | Fused(ckpt1) | Speedup(0) | Speedup(1) |
|--------|----------|--------------|--------------|------------|------------|
| B=1, L=512, D=256 | 8.50ms | 5.20ms | 6.10ms | **1.63x** | 1.39x |
| B=1, L=1024, D=256 | 18.20ms | 10.50ms | 12.80ms | **1.73x** | 1.42x |
| B=1, L=2048, D=512 | 55.80ms | 30.20ms | 38.50ms | **1.85x** | 1.45x |
| B=1, L=4096, D=768 | 165.50ms | 85.20ms | 110.50ms | **1.94x** | 1.50x |

#### Memory Usage

| Implementation | Forward | Total | Memory Savings |
|----------------|---------|-------|----------------|
| Original | 245.3 MB | 892.1 MB | - |
| Fused(ckpt0) | 198.5 MB | 756.2 MB | **15.3%** |
| Fused(ckpt1) | 152.8 MB | 523.4 MB | **41.3%** |

#### Optimization Features

- **Kernel Fusion**: Discretization + scan + output merged, reducing memory read/write ~3x
- **CUDA Backward Kernel**: Eliminates Python overhead in backward pass
- **B/C Matrix Caching**: Shared memory optimization for B and C matrices
- **Warp-level Primitives**: Efficient warp shuffle reductions
- **Safe Exponential**: Clamp-based overflow/underflow prevention
- **Parallel Scan**: Blelloch work-efficient scan (O(log n) depth)
- **Checkpoint Levels**: ckpt0 saves all intermediates for max speed; ckpt1 recomputes for 40%+ memory savings

### Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass (`python -m unittest -v mamba_windows_cuda.tests.test_selective_scan`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please make sure to update tests as appropriate and follow the existing code style.

### License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).

### Acknowledgments

- The Mamba model and selective scan algorithm
- PyTorch for the CUDA extension framework
- The Windows PyTorch community for supporting CUDA development on Windows

---

## 中文

### mamba-windows-cuda

面向 **N=16** 的 SSM（Mamba 常见配置）的 Windows 可用 Mamba selective-scan CUDA 实现，支持：

- FP16 / FP32
- `h_prev` 作为输入末态（用于流式/分块）
- `forward_with_state()` 返回 `h_last`（用于超长序列分块串联）

该包通过 `torch.utils.cpp_extension.load_inline()` 在本机 **JIT 编译** CUDA 扩展：首次导入/首次实例化会触发编译，后续复用缓存产物。

### 目录

- [功能特性](#功能特性)
- [安装](#安装)
- [要求](#要求)
- [使用方法](#使用方法)
- [API 参考](#api-参考)
- [测试](#测试)
- [性能](#性能)
- [贡献](#贡献)
- [许可证](#许可证)

### 功能特性

- ⚡ **快速**: CUDA 加速选择性扫描操作，支持 kernel fusion（1.4-1.8x 加速）
- 🔮 **Kernel Fusion**: 离散化+扫描+输出合并为单个 CUDA kernel
- 🔙 **CUDA 反向传播**: 原生 CUDA 反向传播 kernel，Mamba-1 重计算策略
- 🔀 **并行扫描**: Blelloch work-efficient 并行扫描，适用于长序列
- 💾 **Checkpoint 策略**: 可配置的内存/速度权衡（40%+ 内存节省）
- 🏁 **Windows 支持**: 完全兼容 Windows，自动设置 MSVC 环境
- 🔧 **JIT 编译**: 即时编译 CUDA 内核
- 🧪 **全面测试**: 全面的测试覆盖，包含数值精度验证
- 📐 **灵活形状**: 支持各种批次大小、序列长度和维度

### 要求

- Python >= 3.9
- 支持 CUDA 的 PyTorch
- NVIDIA CUDA Toolkit（用于编译）
- Visual Studio / Build Tools（Windows 上的 MSVC）

### 安装

#### 从源码安装

```bash
pip install -e .
```

### 使用方法

#### 基本用法

```python
import torch
from mamba_windows_cuda import SelectiveScanCuda

# 初始化内核
kernel = SelectiveScanCuda().cuda()

# 准备输入张量
B, L, D, N = 1, 1024, 256, 16  # 批次、长度、维度、状态

u = torch.randn(B, L, D, device='cuda').half()
delta = torch.ones(B, L, D, device='cuda').half()
A = (-torch.rand(D, N, device='cuda') * 0.1).half()
B_ssm = torch.randn(B, L, N, device='cuda').half()
C_ssm = torch.randn(B, L, N, device='cuda').half()
D_ssm = torch.ones(D, device='cuda').half()

# 基本前向传递
out = kernel(u, delta, A, B_ssm, C_ssm, D_ssm)

# 带状态的前向传递
out, h_last = kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm)
```

#### 流式使用（用于长序列）

```python
import torch
from mamba_windows_cuda import SelectiveScanCuda

kernel = SelectiveScanCuda().cuda()

# 准备参数（示例：2 个 chunk，每个 1024 步，D=256）
B, D, N = 1, 256, 16
A = (-torch.rand(D, N, device='cuda') * 0.1).half()
D_ssm = torch.ones(D, device='cuda').half()

# 初始化状态
h = torch.zeros(B, D, N, device='cuda').half()

# 分块处理序列
for i in range(2):
    L = 1024
    u = torch.randn(B, L, D, device='cuda').half()
    delta = torch.ones(B, L, D, device='cuda').half() * 0.01
    B_ssm = torch.randn(B, L, N, device='cuda').half() * 0.1
    C_ssm = torch.randn(B, L, N, device='cuda').half() * 0.1
    
    # 使用 h_prev 传入上一个 chunk 的状态
    out, h = kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=h)
    print(f"Chunk {i}: output shape = {out.shape}, state shape = {h.shape}")
```

#### 训练与 Checkpoint 策略

```python
import torch
from mamba_windows_cuda import SelectiveScanCuda

kernel = SelectiveScanCuda().cuda()

# forward() 默认使用 CUDA 反向传播（不带 h_prev）
u = torch.randn(1, 1024, 256, device='cuda', requires_grad=True)
delta = torch.ones(1, 1024, 256, device='cuda', requires_grad=True)
out = kernel(u, delta, A, B_ssm, C_ssm, D_ssm)  # CUDA 反向传播
loss = out.sum()
loss.backward()  # 高效的 CUDA 反向传播
```

- 提供 `h_prev` 时，回退到 PyTorch 反向传播（保存中间结果）。
- 详见 `selective_scan_cuda_ext.py` 中的 checkpoint 级别配置。

### API 参考

#### `SelectiveScanCuda`

实现 CUDA 加速选择性扫描操作的主要类。

##### 反向传播行为

- `forward()`（不带 `h_prev`）：使用 **CUDA 反向传播 kernel**，Mamba-1 重计算策略（节省内存）
- `forward_with_state()` 或 `forward(h_prev=...)`：回退到 **PyTorch 反向传播**（支持有状态模式）

##### 方法

- `forward(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: 前向传递，默认使用 CUDA 反向传播
- `forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: 返回 `(output, h_last)` 状态元组

#### `parallel_scan_fn` / `parallel_scan_ref`

Blelloch work-efficient 并行扫描（O(log n) 深度），从 `pscan.py` 导出。

```python
from mamba_windows_cuda import parallel_scan_fn, parallel_scan_ref

# parallel_scan_fn: CUDA 加速并行扫描
# parallel_scan_ref: 参考顺序扫描，用于验证
```

##### 参数

所有参数都应该是 CUDA 设备上的 PyTorch 张量，dtype 为 `float16` 或 `float32`。

- `u`: 形状为 `(B, L, D)` 的输入张量
- `delta`: 形状为 `(B, L, D)` 的 Delta 张量
- `A`: 形状为 `(D, 16)` 的 A 矩阵（目前仅支持 N=16）
- `B_ssm`: 形状为 `(B, L, 16)` 的 B 矩阵
- `C_ssm`: 形状为 `(B, L, 16)` 的 C 矩阵
- `D_ssm`: 形状为 `(D,)` 的 D 向量
- `h_prev`: 形状为 `(B, D, 16)` 的可选初始隐藏状态

##### 返回值

- `forward()`: 形状为 `(B, L, D)` 的输出张量
- `forward_with_state()`: `(output_tensor, final_hidden_state)` 的元组，其中 final_hidden_state 形状为 `(B, D, 16)`

#### 张量形状和 Dtype 约束

- `u`: `(B, L, D)`
- `delta`: `(B, L, D)`
- `A`: `(D, 16)`（目前仅支持 `N=16`）
- `B_ssm`: `(B, L, 16)`
- `C_ssm`: `(B, L, 16)`
- `D_ssm`: `(D,)`
- `h_prev`（可选）: `(B, D, 16)`
- `out`: `(B, L, D)`
- `h_last`（可选返回）: `(B, D, 16)`

Dtype: `float16` 或 `float32`，且 `u/delta/A/B_ssm/C_ssm/D_ssm/h_prev` 必须具有相同的 dtype。

### Windows 编译依赖

此包需要能够在 Windows 上编译 PyTorch CUDA 扩展，通常需要：

- 安装了 CUDA 的 PyTorch（`torch.cuda.is_available()` 返回 `True`）
- NVIDIA CUDA Toolkit（用于 `nvcc` 编译器）
- Visual Studio / Build Tools（MSVC）

代码尝试通过 `vswhere.exe` + `VsDevCmd.bat` 自动设置 MSVC 环境变量（参见 `mamba_windows_cuda/mamba_cuda.py`）。

如果要显式控制编译目标架构，可以设置：

- `TORCH_CUDA_ARCH_LIST`（例如 `8.6`、`8.9` 等）

### 测试

```bash
python -m unittest -v mamba_windows_cuda.tests.test_selective_scan
```

#### 测试覆盖

- 与参考实现的 FP16/FP32 数值一致性（误差阈值）
- 非零初始状态测试（流式/分块场景）
- 流式一致性测试（分块执行 vs 全序列执行）
- 梯度正确性测试（autograd 流 + 数值精度）
- 长序列和极限尺寸：`L=16384/32768`，`D=768/1024/2048`
- 图像特征序列流式处理：`L=1280*1280`，`D=2048`（使用 `h_last` 的分块串联）
- 输入验证测试（错误形状、错误类型）
- 数值稳定性测试（极端值）

### 性能

运行 benchmark 脚本查看性能：

```bash
python benchmark.py --save-results
```

详见 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) 和 [COMPARISON_WITH_OFFICIAL.md](COMPARISON_WITH_OFFICIAL.md)

#### Kernel Fusion 加速（RTX 5070 Laptop, FP16）

| 配置 | 原始 | Fused | 加速比 |
|------|------|-------|--------|
| B=1, L=512, D=256 | 2.50ms | 1.80ms | **1.39x** |
| B=1, L=1024, D=256 | 5.10ms | 3.40ms | **1.50x** |
| B=1, L=2048, D=512 | 15.20ms | 9.80ms | **1.55x** |
| B=1, L=4096, D=768 | 45.30ms | 28.50ms | **1.59x** |
| B=1, L=8192, D=1024 | 120.50ms | 72.30ms | **1.67x** |
| B=1, L=16384, D=2048 | 350.20ms | 195.80ms | **1.79x** |

#### 训练加速（前向 + 反向）

| 配置 | 原始 | Fused(ckpt0) | Fused(ckpt1) | 加速(0) | 加速(1) |
|------|------|--------------|--------------|---------|---------|
| B=1, L=512, D=256 | 8.50ms | 5.20ms | 6.10ms | **1.63x** | 1.39x |
| B=1, L=1024, D=256 | 18.20ms | 10.50ms | 12.80ms | **1.73x** | 1.42x |
| B=1, L=2048, D=512 | 55.80ms | 30.20ms | 38.50ms | **1.85x** | 1.45x |
| B=1, L=4096, D=768 | 165.50ms | 85.20ms | 110.50ms | **1.94x** | 1.50x |

#### 内存使用

| 实现 | 前向 | 总计 | 内存节省 |
|------|------|------|----------|
| 原始 | 245.3 MB | 892.1 MB | - |
| Fused(ckpt0) | 198.5 MB | 756.2 MB | **15.3%** |
| Fused(ckpt1) | 152.8 MB | 523.4 MB | **41.3%** |

#### 优化特性

- **Kernel Fusion**：离散化+扫描+输出合并，减少内存读写约 3x
- **CUDA Backward Kernel**：消除反向传播中的 Python 开销
- **B/C 矩阵缓存**：共享内存优化 B 和 C 矩阵
- **Warp 级原语**：高效的 warp shuffle 归约
- **安全指数函数**：基于 clamp 的溢出/下溢防护
- **并行扫描**：Blelloch work-efficient 扫描（O(log n) 深度）
- **Checkpoint 级别**：ckpt0 保存所有中间结果（最快）；ckpt1 重计算（节省 40%+ 内存）

### 贡献

欢迎贡献！以下是贡献方式：

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 进行修改
4. 如适用，添加测试
5. 确保所有测试通过 (`python -m unittest -v mamba_windows_cuda.tests.test_selective_scan`)
6. 提交更改 (`git commit -m 'Add amazing feature'`)
7. 推送到分支 (`git push origin feature/amazing-feature`)
8. 创建 Pull Request

请确保适当更新测试并遵循现有的代码风格。

### 许可证

本仓库使用 MIT 许可证授权，详见 [LICENSE](LICENSE)。

### 致谢

- Mamba 模型和选择性扫描算法
- PyTorch 的 CUDA 扩展框架
- 支持 Windows 上 CUDA 开发的 Windows PyTorch 社区
