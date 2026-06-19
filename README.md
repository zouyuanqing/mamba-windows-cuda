# Mamba Windows CUDA

Windows 兼容的 Mamba selective-scan CUDA 实现。

## 核心特性

- **JIT 编译**: torch.utils.cpp_extension.load_inline，无需预编译
- **MSVC 自动检测**: vswhere.exe 自动定位 Visual Studio
- **性能**: 与官方 mamba-ssm 持平（<3% 差异）
- **精度**: 支持 FP16/FP32
- **流式处理**: h_prev/h_last 状态管理
- **反向传播**: torch.autograd.Function 包装，Mamba-1 风格分块重计算策略，支持训练
- **跨平台**: 同一份 CUDA 内联源码可编译到 CUDA/ROCm/CANN

## 技术实现

- 共享内存缓存 B/C 矩阵，Warp shuffle 归约
- 反向传播：Phase 1 前向扫描 + 边界状态存储 → Phase 2 分块重计算 + 梯度传播
- 6 梯度（u, delta, A, B, C, D）与 PyTorch 参考实现 MaxErr < 1e-5
- B/C 梯度使用 float32 atomicAdd 跨 D 通道求和（避免 warp 竞争）

## 快速使用

```python
import torch
from mamba_windows_cuda import SelectiveScanCuda

kernel = SelectiveScanCuda()

# 训练时自动走 torch.autograd.Function 反向传播
u = torch.randn(2, 1024, 96, device='cuda', requires_grad=True)
delta = torch.randn(2, 1024, 96, device='cuda', requires_grad=True)
A = -torch.exp(torch.randn(96, 16, device='cuda'))
B = torch.randn(2, 1024, 16, device='cuda', requires_grad=True)
C = torch.randn(2, 1024, 16, device='cuda', requires_grad=True)
D = torch.ones(96, device='cuda')

out = kernel(u, delta, A, B, C, D)  # forward + 可反向传播
out.sum().backward()  # CUDA backward kernel 正确计算梯度
```

