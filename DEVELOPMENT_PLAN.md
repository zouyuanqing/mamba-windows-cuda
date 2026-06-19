# mamba-windows-cuda 开发计划

基于官方 state-spaces/mamba 仓库的先进实践，制定以下改进计划。

---

## 一、问题诊断（已完成）

| 问题 | 严重程度 | 来源 |
|------|----------|------|
| 缺少 h_prev 形状验证 | 高 | 代码审查 |
| 缺少设备检查 (is_cuda) | 高 | 对标官方 |
| 缺少连续性检查 (stride) | 中 | 对标官方 |
| 测试覆盖不足（无非零 h_prev） | 中 | 代码审查 |
| 快速数学精度问题 | 低 | 代码审查 |
| README 流式示例不完整 | 低 | 代码审查 |
| 缺少 benchmark 脚本 | 低 | 对标官方 |

---

## 二、改进方案（对标官方 mamba）

### 2.1 输入验证增强

**官方做法**（来自 state-spaces/mamba/csrc/selective_scan/selective_scan.cpp）：

```cpp
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
    #x " must have shape (" #__VA_ARGS__ ")")

// 设备检查
TORCH_CHECK(u.is_cuda());
TORCH_CHECK(delta.is_cuda());

// 连续性检查（最后一维 stride 为 1）
TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

// 形状检查
CHECK_SHAPE(u, batch_size, dim, seqlen);
CHECK_SHAPE(delta, batch_size, dim, seqlen);
CHECK_SHAPE(A, dim, dstate);

// 可选参数检查
if (D_.has_value()) {
    auto D = D_.value();
    TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(D.is_cuda());
    TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
    CHECK_SHAPE(D, dim);
}
```

**我们的实现**：

```cpp
// 1. 设备检查
TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
TORCH_CHECK(delta.is_cuda(), "delta must be a CUDA tensor");
TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
TORCH_CHECK(B_ssm.is_cuda(), "B_ssm must be a CUDA tensor");
TORCH_CHECK(C_ssm.is_cuda(), "C_ssm must be a CUDA tensor");
TORCH_CHECK(D_ssm.is_cuda(), "D_ssm must be a CUDA tensor");

// 2. 连续性检查
TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1, "u must be contiguous in the last dimension");
TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1, "delta must be contiguous in the last dimension");

// 3. 形状检查（使用 CHECK_SHAPE 宏）
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
    #x " must have shape (" #__VA_ARGS__ ")")

CHECK_SHAPE(u, B, L, D);
CHECK_SHAPE(delta, B, L, D);
CHECK_SHAPE(A, D, N);
CHECK_SHAPE(B_ssm, B, L, N);
CHECK_SHAPE(C_ssm, B, L, N);
CHECK_SHAPE(D_ssm, D);

// 4. h_prev 形状检查
if (h_prev_opt.has_value()) {
    auto h_prev = h_prev_opt.value();
    TORCH_CHECK(h_prev.is_cuda(), "h_prev must be a CUDA tensor");
    TORCH_CHECK(h_prev.scalar_type() == u.scalar_type(), "h_prev dtype must match u dtype");
    TORCH_CHECK(h_prev.stride(-1) == 1 || h_prev.size(-1) == 1, "h_prev must be contiguous in the last dimension");
    CHECK_SHAPE(h_prev, B, D, N);
}
```

---

### 2.2 测试策略增强

**官方做法**（来自 state-spaces/mamba/tests/ops/test_selective_scan.py）：

1. **参数化测试**：使用 pytest.mark.parametrize 测试各种配置组合
2. **精度阈值**：根据 dtype 设置不同的 rtol 和 atol
   - float32: rtol=6e-4, atol=2e-3
   - float16: rtol=3e-3, atol=5e-3
   - bfloat16: rtol=3e-2, atol=5e-2
3. **梯度测试**：测试反向传播的梯度正确性
4. **详细输出**：打印 max diff 和 mean diff 用于调试

**我们的实现**：

```python
import pytest
import torch

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('B,L,D', [
    (1, 1024, 256),
    (2, 2048, 512),
    (1, 16384, 768),
    (1, 32768, 2048),
])
@pytest.mark.parametrize('has_h_prev', [False, True])
def test_selective_scan(dtype, B, L, D, has_h_prev):
    N = 16
    device = 'cuda'
    
    # 精度阈值
    rtol, atol = (6e-4, 2e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建输入
    u = torch.randn(B, L, D, device=device, dtype=dtype)
    delta = torch.ones(B, L, D, device=device, dtype=dtype) * 0.01
    A = (-torch.rand(D, N, device=device, dtype=dtype) * 0.1)
    B_ssm = torch.randn(B, L, N, device=device, dtype=dtype) * 0.1
    C_ssm = torch.randn(B, L, N, device=device, dtype=dtype) * 0.1
    D_ssm = torch.ones(D, device=device, dtype=dtype) * 0.1
    
    # 测试非零 h_prev
    if has_h_prev:
        h_prev = torch.randn(B, D, N, device=device, dtype=dtype)
    else:
        h_prev = None
    
    # 运行内核
    kernel = SelectiveScanCuda().to(device)
    out, h_last = kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
    
    # 运行参考实现
    ref_out, ref_h_last = selective_scan_reference(u, delta, A, B_ssm, C_ssm, D_ssm, 
                                                   h_prev if h_prev is not None else torch.zeros(B, D, N, device=device, dtype=dtype))
    
    # 验证输出
    print(f'Output max diff: {(out - ref_out).abs().max().item()}')
    print(f'Output mean diff: {(out - ref_out).abs().mean().item()}')
    assert torch.allclose(out, ref_out, rtol=rtol, atol=atol)
    
    # 验证状态
    if has_h_prev:
        print(f'State max diff: {(h_last - ref_h_last).abs().max().item()}')
        assert torch.allclose(h_last, ref_h_last, rtol=rtol, atol=atol)
```

---

### 2.3 数值稳定性改进

**问题**：
- `__expf` 快速数学精度约 2 ULP
- delta 较大时可能溢出
- delta 较小时可能下溢

**解决方案**（来自调研的最佳实践）：

1. **添加 safe_exp 函数**（防止溢出/下溢）：

```cuda
__device__ __forceinline__ float safe_exp(float x) {
    const float max_exp = 88.0f;  // exp(88) ≈ 1.6e38
    const float min_exp = -88.0f; // exp(-88) ≈ 6.3e-39
    x = fminf(fmaxf(x, min_exp), max_exp);
    return __expf(x);
}
```

2. **提供编译选项**：让用户选择精度/速度权衡

```python
# 在 _get_selective_scan_cuda_module() 中
extra_cuda_cflags = ["-O3", "-allow-unsupported-compiler"]

if os.environ.get("MAMBA_CUDA_USE_FAST_MATH", "1") == "1":
    extra_cuda_cflags.append("--use_fast_math")
else:
    extra_cuda_cflags.append("--ftz=true")  # Flush-to-zero for denormals
```

3. **内核中添加 clamp**（可选）：

```cuda
// 在计算 dA_exp 前 clamp dt_A
float dt_A = delta_val * A_val;
dt_A = fminf(fmaxf(dt_A, -88.0f), 88.0f);  // 防止 exp 溢出
float dA_exp = safe_exp(dt_A);
```

---

### 2.4 文档和示例完善

**README 改进**：

```markdown
### 流式使用（用于长序列）

```python
import torch
from mamba_windows_cuda import SelectiveScanCuda

kernel = SelectiveScanCuda().cuda()

# 准备输入（示例：2 个 chunk，每个 1024 步，D=256）
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

---

### 2.5 Benchmark 脚本

```python
# benchmark.py
import torch
import time
from mamba_windows_cuda import SelectiveScanCuda

def benchmark():
    device = 'cuda'
    kernel = SelectiveScanCuda().to(device)
    
    configs = [
        (1, 1024, 256, 16),
        (1, 2048, 512, 16),
        (1, 4096, 768, 16),
        (1, 8192, 1024, 16),
        (1, 16384, 2048, 16),
    ]
    
    for B, L, D, N in configs:
        u = torch.randn(B, L, D, device=device).half()
        delta = torch.ones(B, L, D, device=device).half()
        A = (-torch.rand(D, N, device=device) * 0.1).half()
        B_ssm = torch.randn(B, L, N, device=device).half()
        C_ssm = torch.randn(B, L, N, device=device).half()
        D_ssm = torch.ones(D, device=device).half()
        
        # Warmup
        for _ in range(10):
            _ = kernel(u, delta, A, B_ssm, C_ssm, D_ssm)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            _ = kernel(u, delta, A, B_ssm, C_ssm, D_ssm)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100
        
        print(f"B={B}, L={L}, D={D}, N={N}: {elapsed*1000:.2f} ms")

if __name__ == '__main__':
    benchmark()
```

---

## 三、开发任务清单

### Phase 1: 输入验证增强（优先级：高）

- [ ] 添加 CHECK_SHAPE 宏
- [ ] 添加设备检查 (is_cuda)
- [ ] 添加连续性检查 (stride)
- [ ] 添加 h_prev 形状验证
- [ ] 更新 Python 端的输入检查

### Phase 2: 测试覆盖增强（优先级：高）

- [ ] 添加非零 h_prev 测试用例
- [ ] 添加参数化测试（不同 dtype、不同尺寸）
- [ ] 添加精度阈值配置
- [ ] 添加详细输出（max diff、mean diff）

### Phase 3: 数值稳定性（优先级：中）

- [ ] 添加编译选项开关（MAMBA_CUDA_USE_FAST_MATH）
- [ ] 添加可选的 clamp 操作
- [ ] 更新文档说明精度/速度权衡

### Phase 4: 文档和示例（优先级：中）

- [ ] 完善 README 流式示例
- [ ] 添加 benchmark.py
- [ ] 添加 CHANGELOG.md

### Phase 5: 版本发布（优先级：低）

- [ ] 更新版本号到 0.2.0
- [ ] 添加 GitHub Actions CI
- [ ] 创建 Release

---

## 四、参考资料

1. **官方 mamba 仓库**：
   - 输入验证：`csrc/selective_scan/selective_scan.cpp`
   - 测试策略：`tests/ops/test_selective_scan.py`

2. **PyTorch CUDA 扩展最佳实践**：
   - 输入验证：https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch
   - CUDA Graph：https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/quick-checklist.html

3. **数值稳定性**：
   - IEEE 754 浮点标准
   - CUDA 数学函数精度：https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html

---

## 五、预期收益

| 改进项 | 收益 |
|--------|------|
| 输入验证增强 | 防止越界访问，提供清晰错误信息 |
| 测试覆盖增强 | 提高代码质量，防止回归 |
| 数值稳定性 | 用户可选择精度/速度权衡 |
| 文档完善 | 降低使用门槛，减少 issue |
| Benchmark | 量化性能，便于优化 |

---

*计划制定时间：2026-06-08*
*基于官方 state-spaces/mamba 仓库对标分析*
