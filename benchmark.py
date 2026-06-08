#!/usr/bin/env python3
"""
Comprehensive benchmark for mamba-windows-cuda.

Compares:
1. Original sequential implementation
2. Parallel scan implementation
3. Fused CUDA kernel (forward)
4. Fused CUDA kernel (forward + backward)
5. Different checkpoint levels

Usage:
    python benchmark.py
    python benchmark.py --save-results
"""

import torch
import time
import argparse
import json
from datetime import datetime
from mamba_windows_cuda import SelectiveScanCuda, parallel_scan_fn
from mamba_windows_cuda.selective_scan_cuda_ext import FusedSelectiveScan


def benchmark_forward_only():
    """Benchmark forward pass only (inference)."""
    device = 'cuda'
    if not torch.cuda.is_available():
        print("CUDA not available")
        return None
    
    kernel_original = SelectiveScanCuda().to(device)
    kernel_fused = FusedSelectiveScan(checkpoint_lvl=0).to(device)
    
    configs = [
        (1, 512, 256, 16),
        (1, 1024, 256, 16),
        (1, 2048, 512, 16),
        (1, 4096, 768, 16),
        (1, 8192, 1024, 16),
        (1, 16384, 2048, 16),
    ]
    
    results = []
    
    print("\n" + "=" * 80)
    print("BENCHMARK: Forward Pass Only (Inference)")
    print("=" * 80)
    print(f"{'B':>4} {'L':>6} {'D':>5} {'N':>3} | {'Original':>12} {'Fused':>12} {'Speedup':>10}")
    print("-" * 80)
    
    for B, L, D, N in configs:
        u = torch.randn(B, L, D, device=device).half()
        delta = torch.ones(B, L, D, device=device).half()
        A = (-torch.rand(D, N, device=device) * 0.1).half()
        B_ssm = torch.randn(B, L, N, device=device).half()
        C_ssm = torch.randn(B, L, N, device=device).half()
        D_ssm = torch.ones(D, device=device).half()
        
        # Warmup
        for _ in range(5):
            _ = kernel_original(u, delta, A, B_ssm, C_ssm, D_ssm)
            _ = kernel_fused(u, delta, A, B_ssm, C_ssm, D_ssm)
        torch.cuda.synchronize()
        
        # Benchmark original
        start = time.time()
        for _ in range(50):
            _ = kernel_original(u, delta, A, B_ssm, C_ssm, D_ssm)
        torch.cuda.synchronize()
        time_original = (time.time() - start) / 50
        
        # Benchmark fused
        start = time.time()
        for _ in range(50):
            _ = kernel_fused(u, delta, A, B_ssm, C_ssm, D_ssm)
        torch.cuda.synchronize()
        time_fused = (time.time() - start) / 50
        
        speedup = time_original / time_fused
        
        print(f"{B:>4} {L:>6} {D:>5} {N:>3} | {time_original*1000:>10.2f}ms {time_fused*1000:>10.2f}ms {speedup:>9.2f}x")
        
        results.append({
            'config': {'B': B, 'L': L, 'D': D, 'N': N},
            'forward_original_ms': time_original * 1000,
            'forward_fused_ms': time_fused * 1000,
            'forward_speedup': speedup,
        })
    
    return results


def benchmark_forward_backward():
    """Benchmark forward + backward pass (training)."""
    device = 'cuda'
    if not torch.cuda.is_available():
        return None
    
    kernel_original = SelectiveScanCuda().to(device)
    kernel_fused_ckpt0 = FusedSelectiveScan(checkpoint_lvl=0).to(device)
    kernel_fused_ckpt1 = FusedSelectiveScan(checkpoint_lvl=1).to(device)
    
    configs = [
        (1, 512, 256, 16),
        (1, 1024, 256, 16),
        (1, 2048, 512, 16),
        (1, 4096, 768, 16),
    ]
    
    results = []
    
    print("\n" + "=" * 100)
    print("BENCHMARK: Forward + Backward Pass (Training)")
    print("=" * 100)
    print(f"{'B':>4} {'L':>6} {'D':>5} {'N':>3} | {'Original':>12} {'Fused(ckpt0)':>14} {'Fused(ckpt1)':>14} {'Speedup(0)':>12} {'Speedup(1)':>12}")
    print("-" * 100)
    
    for B, L, D, N in configs:
        # Original
        u1 = torch.randn(B, L, D, device=device, dtype=torch.float32, requires_grad=True)
        delta1 = torch.ones(B, L, D, device=device, dtype=torch.float32, requires_grad=True) * 0.01
        A1 = (-torch.rand(D, N, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        B1 = (torch.randn(B, L, N, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        C1 = (torch.randn(B, L, N, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        D1 = (torch.ones(D, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        
        # Fused ckpt0
        u2 = u1.detach().clone().requires_grad_(True)
        delta2 = delta1.detach().clone().requires_grad_(True)
        A2 = A1.detach().clone().requires_grad_(True)
        B2 = B1.detach().clone().requires_grad_(True)
        C2 = C1.detach().clone().requires_grad_(True)
        D2 = D1.detach().clone().requires_grad_(True)
        
        # Fused ckpt1
        u3 = u1.detach().clone().requires_grad_(True)
        delta3 = delta1.detach().clone().requires_grad_(True)
        A3 = A1.detach().clone().requires_grad_(True)
        B3 = B1.detach().clone().requires_grad_(True)
        C3 = C1.detach().clone().requires_grad_(True)
        D3 = D1.detach().clone().requires_grad_(True)
        
        # Warmup
        for _ in range(3):
            out1, h1 = kernel_original.forward_with_state(u1, delta1, A1, B1, C1, D1)
            out1.sum().backward(retain_graph=True)
            
            out2, h2 = kernel_fused_ckpt0.forward_with_state(u2, delta2, A2, B2, C2, D2)
            out2.sum().backward(retain_graph=True)
            
            out3, h3 = kernel_fused_ckpt1.forward_with_state(u3, delta3, A3, B3, C3, D3)
            out3.sum().backward(retain_graph=True)
        torch.cuda.synchronize()
        
        # Benchmark original
        start = time.time()
        for _ in range(20):
            u1.grad, delta1.grad, A1.grad, B1.grad, C1.grad, D1.grad = None, None, None, None, None, None
            out1, h1 = kernel_original.forward_with_state(u1, delta1, A1, B1, C1, D1)
            (out1.sum() + h1.sum()).backward()
        torch.cuda.synchronize()
        time_original = (time.time() - start) / 20
        
        # Benchmark fused ckpt0
        start = time.time()
        for _ in range(20):
            u2.grad, delta2.grad, A2.grad, B2.grad, C2.grad, D2.grad = None, None, None, None, None, None
            out2, h2 = kernel_fused_ckpt0.forward_with_state(u2, delta2, A2, B2, C2, D2)
            (out2.sum() + h2.sum()).backward()
        torch.cuda.synchronize()
        time_fused0 = (time.time() - start) / 20
        
        # Benchmark fused ckpt1
        start = time.time()
        for _ in range(20):
            u3.grad, delta3.grad, A3.grad, B3.grad, C3.grad, D3.grad = None, None, None, None, None, None
            out3, h3 = kernel_fused_ckpt1.forward_with_state(u3, delta3, A3, B3, C3, D3)
            (out3.sum() + h3.sum()).backward()
        torch.cuda.synchronize()
        time_fused1 = (time.time() - start) / 20
        
        speedup0 = time_original / time_fused0
        speedup1 = time_original / time_fused1
        
        print(f"{B:>4} {L:>6} {D:>5} {N:>3} | {time_original*1000:>10.2f}ms {time_fused0*1000:>12.2f}ms {time_fused1*1000:>12.2f}ms {speedup0:>11.2f}x {speedup1:>11.2f}x")
        
        results.append({
            'config': {'B': B, 'L': L, 'D': D, 'N': N},
            'train_original_ms': time_original * 1000,
            'train_fused_ckpt0_ms': time_fused0 * 1000,
            'train_fused_ckpt1_ms': time_fused1 * 1000,
            'train_speedup_ckpt0': speedup0,
            'train_speedup_ckpt1': speedup1,
        })
    
    return results


def benchmark_memory():
    """Benchmark memory usage."""
    device = 'cuda'
    if not torch.cuda.is_available():
        return None
    
    kernel_original = SelectiveScanCuda().to(device)
    kernel_fused_ckpt0 = FusedSelectiveScan(checkpoint_lvl=0).to(device)
    kernel_fused_ckpt1 = FusedSelectiveScan(checkpoint_lvl=1).to(device)
    
    B, L, D, N = 1, 4096, 768, 16
    
    results = []
    
    print("\n" + "=" * 80)
    print("BENCHMARK: Memory Usage")
    print("=" * 80)
    print(f"Config: B={B}, L={L}, D={D}, N={N}")
    print("-" * 80)
    
    for name, kernel in [("Original", kernel_original), 
                          ("Fused(ckpt0)", kernel_fused_ckpt0),
                          ("Fused(ckpt1)", kernel_fused_ckpt1)]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        u = torch.randn(B, L, D, device=device, dtype=torch.float32, requires_grad=True)
        delta = torch.ones(B, L, D, device=device, dtype=torch.float32, requires_grad=True) * 0.01
        A = (-torch.rand(D, N, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        B_ssm = (torch.randn(B, L, N, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        C_ssm = (torch.randn(B, L, N, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        D_ssm = (torch.ones(D, device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
        
        # Forward
        out, h = kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm)
        fwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Backward
        (out.sum() + h.sum()).backward()
        bwd_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        print(f"{name:>15} | Forward: {fwd_mem:>8.1f} MB | Total: {bwd_mem:>8.1f} MB")
        
        results.append({
            'name': name,
            'forward_memory_mb': fwd_mem,
            'total_memory_mb': bwd_mem,
        })
        
        del u, delta, A, B_ssm, C_ssm, D_ssm, out, h
        torch.cuda.empty_cache()
    
    return results


def generate_markdown_table(forward_results, train_results, memory_results):
    """Generate markdown benchmark table."""
    md = """# Mamba Windows CUDA Benchmark Results

**Date**: {date}  
**Device**: {device}  
**PyTorch**: {pytorch}  
**CUDA**: {cuda}

## 1. Forward Pass (Inference)

| Config | Original | Fused | Speedup |
|--------|----------|-------|---------|
""".format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        device=torch.cuda.get_device_name(0),
        pytorch=torch.__version__,
        cuda=torch.version.cuda,
    )
    
    if forward_results:
        for r in forward_results:
            cfg = r['config']
            md += f"| B={cfg['B']}, L={cfg['L']}, D={cfg['D']} | {r['forward_original_ms']:.2f}ms | {r['forward_fused_ms']:.2f}ms | {r['forward_speedup']:.2f}x |\n"
    
    md += """
## 2. Forward + Backward (Training)

| Config | Original | Fused(ckpt0) | Fused(ckpt1) | Speedup(0) | Speedup(1) |
|--------|----------|--------------|--------------|------------|------------|
"""
    
    if train_results:
        for r in train_results:
            cfg = r['config']
            md += f"| B={cfg['B']}, L={cfg['L']}, D={cfg['D']} | {r['train_original_ms']:.2f}ms | {r['train_fused_ckpt0_ms']:.2f}ms | {r['train_fused_ckpt1_ms']:.2f}ms | {r['train_speedup_ckpt0']:.2f}x | {r['train_speedup_ckpt1']:.2f}x |\n"
    
    md += """
## 3. Memory Usage

| Implementation | Forward | Total |
|----------------|---------|-------|
"""
    
    if memory_results:
        for r in memory_results:
            md += f"| {r['name']} | {r['forward_memory_mb']:.1f} MB | {r['total_memory_mb']:.1f} MB |\n"
    
    md += """
## Key Findings

1. **Kernel Fusion**: Combines discretization, scan, and output into single kernel
2. **Custom Backward**: CUDA backward kernel avoids Python overhead
3. **Checkpoint Levels**: Trade memory for computation
   - Level 0: Save all intermediates (fastest backward, most memory)
   - Level 1: Recompute intermediates (slower backward, less memory)

## Recommendations

- **Inference**: Use `FusedSelectiveScan` for best performance
- **Training (short sequences)**: Use `FusedSelectiveScan(checkpoint_lvl=0)`
- **Training (long sequences)**: Use `FusedSelectiveScan(checkpoint_lvl=1)` to save memory
"""
    
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("MAMBA WINDOWS CUDA BENCHMARK")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    forward_results = benchmark_forward_only()
    train_results = benchmark_forward_backward()
    memory_results = benchmark_memory()
    
    if args.save_results and forward_results:
        md = generate_markdown_table(forward_results, train_results, memory_results)
        
        with open('BENCHMARK_RESULTS.md', 'w', encoding='utf-8') as f:
            f.write(md)
        print("\nResults saved to BENCHMARK_RESULTS.md")
        
        # Save JSON
        results = {
            'date': datetime.now().isoformat(),
            'device': torch.cuda.get_device_name(0),
            'forward': forward_results,
            'train': train_results,
            'memory': memory_results,
        }
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved to benchmark_results.json")


if __name__ == '__main__':
    main()
