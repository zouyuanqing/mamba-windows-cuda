# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-08

### Added
- **Autograd Support**: Full backward pass support for training
  - `torch.autograd.Function` with optimized backward
  - Saved intermediates to avoid recomputation
  - Gradient flow through all parameters (u, delta, A, B, C, D, h_prev)
- **Gradient Tests**:
  - `test_gradient_flow`: Verify gradients exist and are finite
  - `test_gradient_numerical_accuracy`: Compare with reference implementation

### Changed
- **Version**: Updated from 0.2.0 to 0.3.0
- **SelectiveScanCuda**: Now supports autograd for training

## [0.2.0] - 2026-06-08

### Added
- **Input Validation**: Comprehensive input checks for device, dtype, contiguity, and shape
  - Device checks: All tensors must be on CUDA
  - Dtype checks: All tensors must have matching dtype (float32 or float16)
  - Contiguity checks: Last dimension stride must be 1
  - Shape checks: All tensors must have correct dimensions and sizes
  - h_prev validation: Shape must be (B, D, N) when provided
- **Safe Exponential Function**: `safe_exp()` to prevent overflow/underflow in CUDA kernel
- **Compilation Options**: Environment variable `MAMBA_CUDA_USE_FAST_MATH` to control precision vs speed
  - `MAMBA_CUDA_USE_FAST_MATH=1` (default): Use fast math for better performance
  - `MAMBA_CUDA_USE_FAST_MATH=0`: Use standard math for higher precision
- **Enhanced Tests**: 
  - Non-zero h_prev tests for streaming/chunking scenarios
  - Streaming consistency tests (chunked vs full sequence)
  - Input validation tests
  - Numerical stability tests with extreme values
- **Benchmark Script**: `benchmark.py` for performance measurement
- **Changelog**: This file

### Changed
- **Version**: Updated from 0.1.0 to 0.2.0
- **Documentation**: Improved README with complete streaming example

### Fixed
- **Numerical Stability**: Use `safe_exp()` instead of `__expf()` to prevent overflow

## [0.1.0] - 2026-06-07

### Added
- Initial release
- FP16/FP32 support
- Forward pass with optional h_prev
- Forward with state (returns h_last)
- JIT compilation via `torch.utils.cpp_extension.load_inline()`
- Automatic MSVC environment setup on Windows
- Basic tests with reference implementation validation
