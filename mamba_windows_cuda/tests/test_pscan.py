"""Tests for parallel scan implementation."""

import unittest
import torch
from mamba_windows_cuda.pscan import parallel_scan_fn, parallel_scan_ref


class TestParallelScan(unittest.TestCase):
    """Test parallel scan implementation."""
    
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_parallel_scan_correctness(self):
        """Test that parallel scan produces correct results."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")
        
        B, D, L, N = 2, 32, 128, 16
        torch.manual_seed(42)
        
        # Create random inputs
        A = torch.randn(B, D, L, N, device=self.device).abs() * 0.1  # Ensure |A| < 1
        X = torch.randn(B, D, L, N, device=self.device)
        
        # Parallel scan
        H_parallel = parallel_scan_fn(A, X)
        
        # Reference sequential scan
        H_ref = parallel_scan_ref(A, X)
        
        # Compare
        diff = (H_parallel - H_ref).abs().max().item()
        print(f"Parallel scan max diff: {diff:.6f}")
        self.assertLess(diff, 1e-4, f"Parallel scan diff too large: {diff}")
    
    def test_parallel_scan_gradient(self):
        """Test that gradients flow correctly through parallel scan."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")
        
        B, D, L, N = 1, 16, 64, 8
        torch.manual_seed(123)
        
        # Create inputs with gradients
        A = torch.randn(B, D, L, N, device=self.device, requires_grad=True) * 0.1
        X = torch.randn(B, D, L, N, device=self.device, requires_grad=True)
        
        # Forward
        H = parallel_scan_fn(A, X)
        
        # Backward
        loss = H.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        self.assertIsNotNone(A.grad, "A grad is None")
        self.assertIsNotNone(X.grad, "X grad is None")
        self.assertTrue(torch.isfinite(A.grad).all(), "A grad contains NaN/Inf")
        self.assertTrue(torch.isfinite(X.grad).all(), "X grad contains NaN/Inf")
        
        print(f"Gradient norms: A={A.grad.norm():.4f}, X={X.grad.norm():.4f}")
    
    def test_parallel_scan_long_sequence(self):
        """Test parallel scan with long sequences."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")
        
        B, D, L, N = 1, 64, 1024, 16
        torch.manual_seed(456)
        
        A = torch.randn(B, D, L, N, device=self.device).abs() * 0.05
        X = torch.randn(B, D, L, N, device=self.device)
        
        # This should complete without OOM
        H = parallel_scan_fn(A, X)
        
        self.assertEqual(H.shape, (B, D, L, N))
        self.assertTrue(torch.isfinite(H).all(), "Output contains NaN/Inf")
        
        print(f"Long sequence test passed: shape={H.shape}")


if __name__ == "__main__":
    unittest.main()
