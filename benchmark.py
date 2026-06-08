#!/usr/bin/env python3
"""
Benchmark script for mamba-windows-cuda.

Usage:
    python benchmark.py
"""

import torch
import time
from mamba_windows_cuda import SelectiveScanCuda


def benchmark_forward():
    """Benchmark forward pass."""
    device = 'cuda'
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    kernel = SelectiveScanCuda().to(device)
    
    # Configuration: (B, L, D, N)
    configs = [
        (1, 1024, 256, 16),
        (1, 2048, 512, 16),
        (1, 4096, 768, 16),
        (1, 8192, 1024, 16),
        (1, 16384, 2048, 16),
    ]
    
    print("=" * 70)
    print("Mamba Selective Scan CUDA Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 70)
    
    for B, L, D, N in configs:
        # Create inputs
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
        num_iters = 100
        for _ in range(num_iters):
            _ = kernel(u, delta, A, B_ssm, C_ssm, D_ssm)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_iters
        
        # Calculate throughput
        elements = B * L * D
        throughput = elements / elapsed / 1e9  # Gelements/sec
        
        print(f"B={B}, L={L:5d}, D={D:4d}, N={N:2d}: {elapsed*1000:8.2f} ms, {throughput:6.2f} Gelements/sec")
    
    print("=" * 70)


def benchmark_forward_with_state():
    """Benchmark forward_with_state pass."""
    device = 'cuda'
    if not torch.cuda.is_available():
        return
    
    kernel = SelectiveScanCuda().to(device)
    
    configs = [
        (1, 1024, 256, 16),
        (1, 2048, 512, 16),
        (1, 4096, 768, 16),
    ]
    
    print("\n" + "=" * 70)
    print("Forward with State Benchmark")
    print("=" * 70)
    
    for B, L, D, N in configs:
        u = torch.randn(B, L, D, device=device).half()
        delta = torch.ones(B, L, D, device=device).half()
        A = (-torch.rand(D, N, device=device) * 0.1).half()
        B_ssm = torch.randn(B, L, N, device=device).half()
        C_ssm = torch.randn(B, L, N, device=device).half()
        D_ssm = torch.ones(D, device=device).half()
        h_prev = torch.zeros(B, D, N, device=device).half()
        
        # Warmup
        for _ in range(10):
            _, _ = kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        num_iters = 100
        for _ in range(num_iters):
            _, _ = kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_iters
        
        print(f"B={B}, L={L:5d}, D={D:4d}, N={N:2d}: {elapsed*1000:8.2f} ms")
    
    print("=" * 70)


def benchmark_streaming():
    """Benchmark streaming (chunked) execution."""
    device = 'cuda'
    if not torch.cuda.is_available():
        return
    
    kernel = SelectiveScanCuda().to(device)
    
    print("\n" + "=" * 70)
    print("Streaming (Chunked) Benchmark")
    print("=" * 70)
    
    B, D, N = 1, 512, 16
    L_total = 16384
    chunk_sizes = [2048, 4096, 8192]
    
    A = (-torch.rand(D, N, device=device) * 0.1).half()
    D_ssm = torch.ones(D, device=device).half()
    
    for chunk_L in chunk_sizes:
        # Create full sequence
        u_full = torch.randn(B, L_total, D, device=device).half()
        delta_full = torch.ones(B, L_total, D, device=device).half()
        B_ssm_full = torch.randn(B, L_total, N, device=device).half()
        C_ssm_full = torch.randn(B, L_total, N, device=device).half()
        
        # Warmup
        h = torch.zeros(B, D, N, device=device).half()
        for _ in range(3):
            for i in range(0, L_total, chunk_L):
                end = min(i + chunk_L, L_total)
                _, h = kernel.forward_with_state(
                    u_full[:, i:end, :],
                    delta_full[:, i:end, :],
                    A,
                    B_ssm_full[:, i:end, :],
                    C_ssm_full[:, i:end, :],
                    D_ssm,
                    h_prev=h
                )
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        num_iters = 10
        for _ in range(num_iters):
            h = torch.zeros(B, D, N, device=device).half()
            for i in range(0, L_total, chunk_L):
                end = min(i + chunk_L, L_total)
                _, h = kernel.forward_with_state(
                    u_full[:, i:end, :],
                    delta_full[:, i:end, :],
                    A,
                    B_ssm_full[:, i:end, :],
                    C_ssm_full[:, i:end, :],
                    D_ssm,
                    h_prev=h
                )
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_iters
        
        num_chunks = (L_total + chunk_L - 1) // chunk_L
        print(f"L_total={L_total}, chunk_L={chunk_L:5d}, chunks={num_chunks:3d}: {elapsed*1000:8.2f} ms")
    
    print("=" * 70)


if __name__ == '__main__':
    benchmark_forward()
    benchmark_forward_with_state()
    benchmark_streaming()
