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

> 状态更新（2026-06-19）：Phase 1-4 已全部完成并随 v0.2.0 - v0.5.0 发布。Phase 5 部分完成。

### Phase 1: 输入验证增强 ✅ v0.2.0

- [x] 添加 CHECK_SHAPE 宏
- [x] 添加设备检查 (is_cuda)
- [x] 添加连续性检查 (stride)
- [x] 添加 h_prev 形状验证
- [x] 更新 Python 端的输入检查

### Phase 2: 测试覆盖增强 ✅ v0.2.0 - v0.3.0

- [x] 添加非零 h_prev 测试用例
- [x] 添加流式一致性测试（分块 vs 全序列）
- [x] 添加梯度正确性测试（v0.3.0 autograd 支持）
- [x] 添加输入验证测试（错误形状、错误类型）
- [x] 添加数值稳定性测试（极端值）
- [x] 添加精度阈值配置（FP16/FP32 区分）
- [x] 添加详细输出（max diff、mean diff）

### Phase 3: 数值稳定性 ✅ v0.2.0

- [x] 添加 safe_exp 函数（clamp 防止溢出/下溢）
- [x] 添加编译选项开关（MAMBA_CUDA_USE_FAST_MATH）
- [x] 更新文档说明精度/速度权衡

### Phase 4: 文档和示例 ✅ v0.2.0 - v0.5.0

- [x] 完善 README 流式示例
- [x] 添加 benchmark.py（支持 --save-results）
- [x] 添加 CHANGELOG.md
- [x] 添加 BENCHMARK_RESULTS.md（fusion vs 原始对比数据）
- [x] 添加 COMPARISON_WITH_OFFICIAL.md（与官方 mamba-ssm 对标）
- [x] README 同步 v0.5.0 新特性（v0.5.0 docs sync）

### Phase 5: 版本发布

- [x] 更新版本号到 0.5.0
- [ ] 添加 GitHub Actions CI
- [ ] 创建 GitHub Release

---

## 三、计划外的关键进展（v0.3.0 - v0.5.0）

这部分不在最初计划中，但实际推进过程中完成：

### v0.3.0 - Autograd 支持

- [x] `torch.autograd.Function` 包装
- [x] PyTorch backward + 保存中间结果策略
- [x] 梯度通过所有参数（u, delta, A, B, C, D, h_prev）

### v0.4.0 - 并行扫描

- [x] Blelloch work-efficient parallel scan（O(log n) 深度）
- [x] `parallel_scan_fn` / `parallel_scan_ref` 导出
- [x] 基于 mamba.py 的实现思路

### v0.5.0 - CUDA Backward + Kernel Fusion

- [x] 原生 CUDA backward kernel（Mamba-1 重计算策略）
- [x] Kernel Fusion（离散化 + 扫描 + 输出合并到单 kernel）
- [x] `SelectiveScanFn` 使用 CUDA backward（无 h_prev 时默认）
- [x] Checkpoint 级别（ckpt0 vs ckpt1 内存/速度权衡）
- [x] 实测 1.4-1.8x 推理加速 / 1.6-1.9x 训练加速
- [x] 实测 41.3% 训练内存节省（ckpt1）

---

## 四、后续路线图（待规划）

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

## 五、预期收益（已兑现）

| 改进项 | 收益 | 实际效果 |
|--------|------|---------|
| 输入验证增强 | 防止越界访问，提供清晰错误信息 | ✅ 全面 TORCH_CHECK 覆盖 |
| 测试覆盖增强 | 提高代码质量，防止回归 | ✅ 含梯度/流式/数值稳定性测试 |
| 数值稳定性 | 用户可选择精度/速度权衡 | ✅ MAMBA_CUDA_USE_FAST_MATH 开关 |
| 文档完善 | 降低使用门槛，减少 issue | ✅ README + CHANGELOG + COMPARISON |
| Benchmark | 量化性能，便于优化 | ✅ Fused vs Original 全套数据 |
| **CUDA Backward**（计划外）| 训练效率 | ✅ 1.6-1.9x 训练加速 |
| **Kernel Fusion**（计划外）| 推理效率 | ✅ 1.4-1.8x 推理加速 |
| **Checkpoint 策略**（计划外）| 内存优化 | ✅ 41.3% 训练内存节省 |

---

*计划制定时间：2026-06-08*
*最近更新：2026-06-19（v0.5.0 进度同步）*
*基于官方 state-spaces/mamba 仓库对标分析*
