import unittest
import pytest
import torch

from mamba_windows_cuda import SelectiveScanCuda


def selective_scan_reference(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev):
    """Reference implementation for validation."""
    u_p = u.transpose(1, 2)  # (B, D, L)
    delta_p = delta.transpose(1, 2)  # (B, D, L)
    B_p = B_ssm.transpose(1, 2)  # (B, N, L)
    C_p = C_ssm.transpose(1, 2)  # (B, N, L)

    # Compute cumulative sum of delta
    delta_cumsum = torch.cumsum(delta_p, dim=2)  # (B, D, L)
    
    # Compute exp(A * delta_cumsum) for each state
    L_t = delta_cumsum.unsqueeze(2) * A.view(1, -1, A.shape[1], 1)  # (B, D, N, L)
    Ats = torch.exp(L_t)  # (B, D, N, L)
    inv_Ats = torch.exp(-L_t)  # (B, D, N, L)
    
    # Compute delta * u * B for each time step
    dBu = (delta_p * u_p).unsqueeze(2) * B_p.unsqueeze(1)  # (B, D, N, L)
    
    # Compute cumulative sum with exponential weighting
    sum_term = torch.cumsum(inv_Ats * dBu, dim=3)  # (B, D, N, L)
    
    # Compute h_t = Ats * (h_prev + sum_term)
    h_t = Ats * (h_prev.unsqueeze(-1) + sum_term)  # (B, D, N, L)
    
    # Compute output y = sum(C * h_t) + D * u
    y = torch.sum(C_p.unsqueeze(1) * h_t, dim=2)  # (B, D, L)
    y = y + u_p * D_ssm.view(1, -1, 1)  # (B, D, L)
    
    return y.transpose(1, 2), h_t[:, :, :, -1]  # (B, L, D), (B, D, N)


class TestSelectiveScanCUDA(unittest.TestCase):
    """Unit tests for SelectiveScanCuda."""
    
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kernel = SelectiveScanCuda().to(self.device)

    def test_consistency_fp16(self):
        """Test FP16 numerical consistency with reference implementation."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 1024, 256, 16
        torch.manual_seed(0)
        u = (torch.randn(B, L, D, device=self.device) * 0.1).half()
        delta = (torch.ones(B, L, D, device=self.device) * 0.01).half()
        A = (-torch.rand(D, N, device=self.device) * 0.1).half()
        B_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        C_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        D_ssm = (torch.ones(D, device=self.device) * 0.1).half()
        h_prev = torch.zeros(B, D, N, device=self.device).half()

        with torch.no_grad():
            out = self.kernel(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
            ref, _ = selective_scan_reference(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)

        self.assertTrue(torch.isfinite(ref).all(), "Reference output contains NaN/Inf")
        self.assertTrue(torch.isfinite(out).all(), "CUDA output contains NaN/Inf")
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"FP16 max diff: {diff:.6f}")
        self.assertLess(diff, 1e-3, f"FP16 diff too large: {diff}")

    def test_consistency_fp32(self):
        """Test FP32 numerical consistency with reference implementation."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 1024, 256, 16
        torch.manual_seed(42)
        u = (torch.randn(B, L, D, device=self.device) * 0.1).float()
        delta = (torch.ones(B, L, D, device=self.device) * 0.01).float()
        A = (-torch.rand(D, N, device=self.device) * 0.1).float()
        B_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).float()
        C_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).float()
        D_ssm = (torch.ones(D, device=self.device) * 0.1).float()
        h_prev = torch.zeros(B, D, N, device=self.device).float()

        with torch.no_grad():
            out = self.kernel(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
            ref, _ = selective_scan_reference(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)

        self.assertTrue(torch.isfinite(ref).all(), "Reference output contains NaN/Inf")
        self.assertTrue(torch.isfinite(out).all(), "CUDA output contains NaN/Inf")
        diff = (out - ref).abs().max().item()
        print(f"FP32 max diff: {diff:.6f}")
        self.assertLess(diff, 1e-3, f"FP32 diff too large: {diff}")

    def test_nonzero_h_prev(self):
        """Test with non-zero initial state (critical for streaming/chunking)."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 1024, 256, 16
        torch.manual_seed(123)
        u = (torch.randn(B, L, D, device=self.device) * 0.1).half()
        delta = (torch.ones(B, L, D, device=self.device) * 0.01).half()
        A = (-torch.rand(D, N, device=self.device) * 0.1).half()
        B_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        C_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        D_ssm = (torch.ones(D, device=self.device) * 0.1).half()
        
        # Non-zero initial state
        h_prev = torch.randn(B, D, N, device=self.device).half()

        with torch.no_grad():
            out = self.kernel(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
            ref, ref_h_last = selective_scan_reference(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)

        self.assertTrue(torch.isfinite(out).all(), "CUDA output contains NaN/Inf")
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"Non-zero h_prev max diff: {diff:.6f}")
        self.assertLess(diff, 1e-3, f"Non-zero h_prev diff too large: {diff}")

    def test_forward_with_state(self):
        """Test forward_with_state returns correct h_last."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 1024, 256, 16
        torch.manual_seed(456)
        u = (torch.randn(B, L, D, device=self.device) * 0.1).half()
        delta = (torch.ones(B, L, D, device=self.device) * 0.01).half()
        A = (-torch.rand(D, N, device=self.device) * 0.1).half()
        B_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        C_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        D_ssm = (torch.ones(D, device=self.device) * 0.1).half()
        h_prev = torch.zeros(B, D, N, device=self.device).half()

        with torch.no_grad():
            out, h_last = self.kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
            ref_out, ref_h_last = selective_scan_reference(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)

        self.assertTrue(torch.isfinite(h_last).all(), "h_last contains NaN/Inf")
        out_diff = (out.float() - ref_out.float()).abs().max().item()
        state_diff = (h_last.float() - ref_h_last.float()).abs().max().item()
        print(f"forward_with_state output diff: {out_diff:.6f}, state diff: {state_diff:.6f}")
        self.assertLess(out_diff, 1e-3, f"Output diff too large: {out_diff}")
        self.assertLess(state_diff, 1e-3, f"State diff too large: {state_diff}")

    def test_streaming_consistency(self):
        """Test that streaming (chunked) execution matches full sequence execution."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, D, N = 1, 256, 16
        L_total = 2048
        chunk_L = 1024
        torch.manual_seed(789)
        
        A = (-torch.rand(D, N, device=self.device) * 0.1).half()
        D_ssm = (torch.ones(D, device=self.device) * 0.1).half()
        
        # Full sequence
        u_full = (torch.randn(B, L_total, D, device=self.device) * 0.1).half()
        delta_full = (torch.ones(B, L_total, D, device=self.device) * 0.01).half()
        B_ssm_full = (torch.randn(B, L_total, N, device=self.device) * 0.1).half()
        C_ssm_full = (torch.randn(B, L_total, N, device=self.device) * 0.1).half()
        h_prev = torch.zeros(B, D, N, device=self.device).half()
        
        with torch.no_grad():
            out_full, _ = self.kernel.forward_with_state(
                u_full, delta_full, A, B_ssm_full, C_ssm_full, D_ssm, h_prev
            )
        
        # Chunked execution
        h = torch.zeros(B, D, N, device=self.device).half()
        out_chunks = []
        
        with torch.no_grad():
            for i in range(0, L_total, chunk_L):
                end = min(i + chunk_L, L_total)
                u_chunk = u_full[:, i:end, :]
                delta_chunk = delta_full[:, i:end, :]
                B_chunk = B_ssm_full[:, i:end, :]
                C_chunk = C_ssm_full[:, i:end, :]
                
                out_chunk, h = self.kernel.forward_with_state(
                    u_chunk, delta_chunk, A, B_chunk, C_chunk, D_ssm, h_prev=h
                )
                out_chunks.append(out_chunk)
        
        out_streaming = torch.cat(out_chunks, dim=1)
        diff = (out_full.float() - out_streaming.float()).abs().max().item()
        print(f"Streaming consistency max diff: {diff:.6f}")
        self.assertLess(diff, 1e-3, f"Streaming diff too large: {diff}")

    def test_large_shapes_and_streaming(self):
        """Test large shapes and streaming for image feature sequences."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        torch.manual_seed(123)
        cases = [
            (1, 16384, 768, 16),
            (1, 16384, 1024, 16),
            (1, 32768, 2048, 16),
        ]
        for B, L, D, N in cases:
            u = (torch.randn(B, L, D, device=self.device) * 0.1).half()
            delta = (torch.ones(B, L, D, device=self.device) * 0.01).half()
            A = (-torch.rand(D, N, device=self.device) * 0.1).half()
            B_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
            C_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
            D_ssm = (torch.ones(D, device=self.device) * 0.1).half()

            with torch.no_grad():
                out = self.kernel(u, delta, A, B_ssm, C_ssm, D_ssm, None)
            self.assertTrue(torch.isfinite(out).all(), f"NaN/Inf in output for shape ({B},{L},{D})")

        B, D, N = 1, 2048, 16
        L_total = 1280 * 1280
        chunk_L = 16384

        A = (-torch.rand(D, N, device=self.device) * 0.1).half()
        D_ssm = (torch.ones(D, device=self.device) * 0.1).half()
        h = torch.zeros(B, D, N, device=self.device).half()

        with torch.no_grad():
            steps = (L_total + chunk_L - 1) // chunk_L
            for s in range(steps):
                Lc = min(chunk_L, L_total - s * chunk_L)
                u = (torch.randn(B, Lc, D, device=self.device) * 0.1).half()
                delta = (torch.ones(B, Lc, D, device=self.device) * 0.01).half()
                B_ssm = (torch.randn(B, Lc, N, device=self.device) * 0.1).half()
                C_ssm = (torch.randn(B, Lc, N, device=self.device) * 0.1).half()
                out, h = self.kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h)

        self.assertTrue(torch.isfinite(out).all(), "Final output contains NaN/Inf")
        self.assertTrue(torch.isfinite(h).all(), "Final state contains NaN/Inf")

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 1024, 256, 16
        
        # Valid inputs
        u = torch.randn(B, L, D, device=self.device).half()
        delta = torch.ones(B, L, D, device=self.device).half()
        A = (-torch.rand(D, N, device=self.device) * 0.1).half()
        B_ssm = torch.randn(B, L, N, device=self.device).half()
        C_ssm = torch.randn(B, L, N, device=self.device).half()
        D_ssm = torch.ones(D, device=self.device).half()
        
        # Test wrong h_prev shape
        h_wrong = torch.randn(B, D, N + 1, device=self.device).half()  # Wrong N
        with self.assertRaises(RuntimeError):
            self.kernel(u, delta, A, B_ssm, C_ssm, D_ssm, h_wrong)
        
        # Test wrong h_prev batch size
        h_wrong_batch = torch.randn(B + 1, D, N, device=self.device).half()
        with self.assertRaises(RuntimeError):
            self.kernel(u, delta, A, B_ssm, C_ssm, D_ssm, h_wrong_batch)

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme input values."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 512, 64, 16
        torch.manual_seed(999)
        
        # Extreme delta values
        u = (torch.randn(B, L, D, device=self.device) * 0.1).half()
        delta = (torch.ones(B, L, D, device=self.device) * 10.0).half()  # Large delta
        A = (-torch.rand(D, N, device=self.device) * 0.1).half()
        B_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        C_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        D_ssm = (torch.ones(D, device=self.device) * 0.1).half()
        h_prev = torch.zeros(B, D, N, device=self.device).half()

        with torch.no_grad():
            out = self.kernel(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
        
        # Should not contain NaN/Inf even with extreme values
        self.assertTrue(torch.isfinite(out).all(), "Output contains NaN/Inf with extreme delta")

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the operation."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 256, 64, 16
        torch.manual_seed(42)
        
        # Create inputs with requires_grad=True
        u = torch.randn(B, L, D, device=self.device, dtype=torch.float32, requires_grad=True)
        delta = torch.ones(B, L, D, device=self.device, dtype=torch.float32, requires_grad=True) * 0.01
        A = (-torch.rand(D, N, device=self.device, dtype=torch.float32) * 0.1).requires_grad_(True)
        B_ssm = (torch.randn(B, L, N, device=self.device, dtype=torch.float32) * 0.1).requires_grad_(True)
        C_ssm = (torch.randn(B, L, N, device=self.device, dtype=torch.float32) * 0.1).requires_grad_(True)
        D_ssm = (torch.ones(D, device=self.device, dtype=torch.float32) * 0.1).requires_grad_(True)
        h_prev = torch.zeros(B, D, N, device=self.device, dtype=torch.float32, requires_grad=True)

        # Forward pass
        out, h_last = self.kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
        
        # Compute loss
        loss = out.sum() + h_last.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        self.assertIsNotNone(u.grad, "u grad is None")
        self.assertIsNotNone(delta.grad, "delta grad is None")
        self.assertIsNotNone(A.grad, "A grad is None")
        self.assertIsNotNone(B_ssm.grad, "B_ssm grad is None")
        self.assertIsNotNone(C_ssm.grad, "C_ssm grad is None")
        self.assertIsNotNone(D_ssm.grad, "D_ssm grad is None")
        self.assertIsNotNone(h_prev.grad, "h_prev grad is None")
        
        # Check for NaN/Inf in gradients
        self.assertTrue(torch.isfinite(u.grad).all(), "u grad contains NaN/Inf")
        self.assertTrue(torch.isfinite(delta.grad).all(), "delta grad contains NaN/Inf")
        self.assertTrue(torch.isfinite(A.grad).all(), "A grad contains NaN/Inf")
        self.assertTrue(torch.isfinite(B_ssm.grad).all(), "B_ssm grad contains NaN/Inf")
        self.assertTrue(torch.isfinite(C_ssm.grad).all(), "C_ssm grad contains NaN/Inf")
        self.assertTrue(torch.isfinite(D_ssm.grad).all(), "D_ssm grad contains NaN/Inf")
        
        print(f"Gradient norms: u={u.grad.norm():.4f}, delta={delta.grad.norm():.4f}, A={A.grad.norm():.4f}")

    def test_gradient_numerical_accuracy(self):
        """Test gradient accuracy against reference implementation."""
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 128, 32, 16
        torch.manual_seed(123)
        
        # Use float32 for accuracy
        u = torch.randn(B, L, D, device=self.device, dtype=torch.float32, requires_grad=True)
        delta = torch.ones(B, L, D, device=self.device, dtype=torch.float32, requires_grad=True) * 0.01
        A = (-torch.rand(D, N, device=self.device, dtype=torch.float32) * 0.1).requires_grad_(True)
        B_ssm = (torch.randn(B, L, N, device=self.device, dtype=torch.float32) * 0.1).requires_grad_(True)
        C_ssm = (torch.randn(B, L, N, device=self.device, dtype=torch.float32) * 0.1).requires_grad_(True)
        D_ssm = (torch.ones(D, device=self.device, dtype=torch.float32) * 0.1).requires_grad_(True)

        # CUDA implementation
        out_cuda, h_last_cuda = self.kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm)
        loss_cuda = out_cuda.sum() + h_last_cuda.sum()
        loss_cuda.backward()
        
        du_cuda = u.grad.clone()
        dA_cuda = A.grad.clone()
        
        # Reset gradients
        u.grad.zero_()
        A.grad.zero_()
        
        # Reference implementation
        from mamba_windows_cuda.mamba_cuda import selective_scan_ref
        out_ref, h_last_ref = selective_scan_ref(u, delta, A, B_ssm, C_ssm, D_ssm)
        loss_ref = out_ref.sum() + h_last_ref.sum()
        loss_ref.backward()
        
        du_ref = u.grad.clone()
        dA_ref = A.grad.clone()
        
        # Compare gradients
        du_diff = (du_cuda - du_ref).abs().max().item()
        dA_diff = (dA_cuda - dA_ref).abs().max().item()
        
        print(f"Gradient diff: du={du_diff:.6f}, dA={dA_diff:.6f}")
        
        # Allow some tolerance for numerical differences
        self.assertLess(du_diff, 1e-3, f"du gradient diff too large: {du_diff}")
        self.assertLess(dA_diff, 1e-3, f"dA gradient diff too large: {dA_diff}")


if __name__ == "__main__":
    unittest.main()
