# mamba-windows-cuda

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Windows 下可用的 Mamba selective-scan CUDA 实现，面向 **N=16** 的 SSM（Mamba 常见配置），支持：

- FP16 / FP32
- `h_prev` 作为输入末态（用于流式/分块）
- `forward_with_state()` 返回 `h_last`（用于超长序列分块串联）

该包通过 `torch.utils.cpp_extension.load_inline()` 在本机 **JIT 编译** CUDA 扩展：首次导入/首次实例化会触发编译，后续复用缓存产物。

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Features

- ⚡ **Fast**: CUDA-accelerated selective scan operation for Mamba models
- 🏁 **Windows Support**: Full Windows compatibility with automatic MSVC environment setup
- 🔧 **JIT Compilation**: On-the-fly compilation of CUDA kernels
- 🧪 **Thoroughly Tested**: Comprehensive test coverage with numerical accuracy validation
- 📐 **Flexible Shapes**: Support for various batch sizes, sequence lengths, and dimensions

## Requirements

- Python >= 3.9
- PyTorch with CUDA support
- NVIDIA CUDA Toolkit (for compilation)
- Visual Studio / Build Tools (MSVC) for Windows

## Installation

### From Source

```bash
pip install -e .
```

### Direct Installation

```bash
pip install mamba-windows-cuda
```

## Usage

### Basic Usage

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

### Streaming Usage (for long sequences)

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

## API Reference

### `SelectiveScanCuda`

The main class implementing the CUDA-accelerated selective scan operation.

#### Methods

- `forward(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: Basic forward pass
- `forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None)`: Forward pass returning state

#### Parameters

All parameters should be PyTorch tensors on CUDA device with `float16` or `float32` dtype.

- `u`: Input tensor of shape `(B, L, D)`
- `delta`: Delta tensor of shape `(B, L, D)`
- `A`: A matrix of shape `(D, 16)` (currently only N=16 is supported)
- `B_ssm`: B matrix of shape `(B, L, 16)`
- `C_ssm`: C matrix of shape `(B, L, 16)`
- `D_ssm`: D vector of shape `(D,)`
- `h_prev`: Optional initial hidden state of shape `(B, D, 16)`

#### Returns

- `forward()`: Output tensor of shape `(B, L, D)`
- `forward_with_state()`: Tuple of `(output_tensor, final_hidden_state)` where final_hidden_state is of shape `(B, D, 16)`

### Tensor Shapes and Dtype Constraints

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

## Windows Compilation Dependencies

This package requires the ability to compile PyTorch CUDA extensions on Windows, which typically requires:

- PyTorch with CUDA installed (`torch.cuda.is_available()` returns `True`)
- NVIDIA CUDA Toolkit (for `nvcc` compiler)
- Visual Studio / Build Tools (MSVC)

The code attempts to automatically set up the MSVC environment variables via `vswhere.exe` + `VsDevCmd.bat` (see `mamba_windows_cuda/mamba_cuda.py`).

If you want to explicitly control the compilation target architecture, you can set:

- `TORCH_CUDA_ARCH_LIST` (e.g., `8.6`, `8.9`, etc.)

## Testing

```bash
python -m unittest -v mamba_windows_cuda.tests.test_selective_scan
```

### Test Coverage

- FP16 numerical consistency with reference implementation (error threshold)
- Long sequences and extreme sizes: `L=16384/32768`, `D=768/1024/2048`
- Streaming for image feature sequences: `L=1280*1280`, `D=2048` (chunked concatenation with `h_last`)

## Performance

This implementation is optimized for Windows and provides efficient selective scan operations for Mamba models. It includes:

- Shared memory optimization for B and C matrices
- Warp-level primitives for efficient reductions
- Support for both FP16 and FP32 precision
- Optimized kernel launch parameters for different tensor sizes

## Contributing

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Mamba model and selective scan algorithm
- PyTorch for the CUDA extension framework
- The Windows PyTorch community for supporting CUDA development on Windows