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

- ⚡ **Fast**: CUDA-accelerated selective scan operation for Mamba models
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

### API Reference

#### `SelectiveScanCuda`

The main class implementing the CUDA-accelerated selective scan operation.

##### Methods

- `forward(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: Basic forward pass
- `forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: Forward pass returning state

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

- FP16 numerical consistency with reference implementation (error threshold)
- Long sequences and extreme sizes: `L=16384/32768`, `D=768/1024/2048`
- Streaming for image feature sequences: `L=1280*1280`, `D=2048` (chunked concatenation with `h_last`)

### Performance

This implementation is optimized for Windows and provides efficient selective scan operations for Mamba models. It includes:

- Shared memory optimization for B and C matrices
- Warp-level primitives for efficient reductions
- Support for both FP16 and FP32 precision
- Optimized kernel launch parameters for different tensor sizes

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

- ⚡ **快速**: 针对 Mamba 模型的 CUDA 加速选择性扫描操作
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

### API 参考

#### `SelectiveScanCuda`

实现 CUDA 加速选择性扫描操作的主要类。

##### 方法

- `forward(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: 基本前向传递
- `forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: 返回状态的前向传递

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
- 长序列和极限尺寸：`L=16384/32768`，`D=768/1024/2048`
- 图像特征序列流式处理：`L=1280*1280`，`D=2048`（使用 `h_last` 的分块串联）
- 输入验证测试（错误形状、错误类型）
- 数值稳定性测试（极端值）

### 性能

运行 benchmark 脚本查看性能：

```bash
python benchmark.py
```

此实现在 Windows 上进行了优化，为 Mamba 模型提供高效的选择性扫描操作。它包括：

- B 和 C 矩阵的共享内存优化
- 用于高效归约的 Warp 级原语
- 支持 FP16 和 FP32 精度
- 针对不同张量尺寸优化的内核启动参数
- 安全指数函数防止数值溢出/下溢

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
