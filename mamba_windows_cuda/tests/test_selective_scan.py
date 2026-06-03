import unittest

import torch

from mamba_windows_cuda import SelectiveScanCuda


def selective_scan_reference(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev):
    u_p = u.transpose(1, 2)
    delta_p = delta.transpose(1, 2)
    B_p = B_ssm.transpose(1, 2)
    C_p = C_ssm.transpose(1, 2)

    delta_cumsum = torch.cumsum(delta_p, dim=2)
    L_t = delta_cumsum.unsqueeze(2) * A.view(1, -1, A.shape[1], 1)
    Ats = torch.exp(L_t)
    inv_Ats = torch.exp(-L_t)
    dBu = (delta_p * u_p).unsqueeze(2) * B_p.unsqueeze(1)
    sum_term = torch.cumsum(inv_Ats * dBu, dim=3)
    h_t = Ats * (h_prev.unsqueeze(-1) + sum_term)
    y = torch.sum(C_p.unsqueeze(1) * h_t, dim=2)
    y = y + u_p * D_ssm.view(1, -1, 1)
    return y.transpose(1, 2), h_t[:, :, :, -1]


class TestSelectiveScanCUDA(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kernel = SelectiveScanCuda().to(self.device)

    def test_consistency_fp16(self):
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
        self.assertLess(diff, 1e-3, f"FP16 max error {diff:.6f} exceeds threshold 1e-3")

    def test_consistency_fp32(self):
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
        self.assertLess(diff, 1e-4, f"FP32 max error {diff:.6f} exceeds threshold 1e-4")

    def test_forward_with_state(self):
        if self.device == "cpu":
            self.skipTest("CUDA not available")

        B, L, D, N = 1, 512, 256, 16
        torch.manual_seed(7)
        u = (torch.randn(B, L, D, device=self.device) * 0.1).half()
        delta = (torch.ones(B, L, D, device=self.device) * 0.01).half()
        A = (-torch.rand(D, N, device=self.device) * 0.1).half()
        B_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        C_ssm = (torch.randn(B, L, N, device=self.device) * 0.1).half()
        D_ssm = (torch.ones(D, device=self.device) * 0.1).half()

        with torch.no_grad():
            out, h_last = self.kernel.forward_with_state(u, delta, A, B_ssm, C_ssm, D_ssm, None)
            ref_out, ref_h = selective_scan_reference(u, delta, A, B_ssm, C_ssm, D_ssm,
                                                      torch.zeros(B, D, N, device=self.device).half())

        self.assertTrue(torch.isfinite(out).all(), "CUDA output contains NaN/Inf")
        self.assertTrue(torch.isfinite(h_last).all(), "CUDA h_last contains NaN/Inf")
        out_diff = (out.float() - ref_out.float()).abs().max().item()
        h_diff = (h_last.float() - ref_h.float()).abs().max().item()
        self.assertLess(out_diff, 1e-3, f"forward_with_state output error {out_diff:.6f}")
        self.assertLess(h_diff, 1e-3, f"forward_with_state h_last error {h_diff:.6f}")

    def test_large_shapes_and_streaming(self):
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
            self.assertTrue(torch.isfinite(out).all(), f"NaN/Inf in output for shape (B={B}, L={L}, D={D})")

        # Streaming test: chunked concatenation with h_last
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

        self.assertTrue(torch.isfinite(out).all(), "Streaming final output contains NaN/Inf")
        self.assertTrue(torch.isfinite(h).all(), "Streaming final state contains NaN/Inf")


if __name__ == "__main__":
    unittest.main()
