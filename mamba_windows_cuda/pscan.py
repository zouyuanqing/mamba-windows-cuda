"""
Parallel Scan implementation for Mamba selective scan.

Based on Blelloch's work-efficient parallel scan algorithm.
Provides O(log n) depth instead of O(n) for sequential scan.

Reference:
- mamba.py: https://github.com/alxndrTL/mamba.py
- Parallel prefix sum: https://mlsys.org/media/mlsys-2020/Slides/1407.pdf
"""

import math
import torch


class PScan(torch.autograd.Function):
    """
    Parallel Scan autograd function.
    
    Computes: H[t] = A[t] * H[t-1] + X[t] in parallel.
    
    Uses Blelloch up-sweep/down-sweep algorithm:
    - Up-sweep: reduction phase, O(n) work, O(log n) depth
    - Down-sweep: distribution phase, O(n) work, O(log n) depth
    
    Total: O(2*log2(n)) steps instead of O(n) sequential steps.
    """
    
    @staticmethod
    def pscan(A, X):
        """
        In-place parallel scan.
        
        Args:
            A: (B, D, L, N) - transition coefficients
            X: (B, D, L, N) - input values
        
        Modifies X in-place to contain the scan result.
        """
        B, D, L, N = A.size()
        
        # Ensure L is power of 2 by padding
        num_steps = int(math.log2(L))
        
        # Up-sweep (reduction)
        for k in range(num_steps):
            T = 2 ** (num_steps - k)
            stride = 2 ** k
            
            # View as pairs and combine
            A_view = A[:, :, :T*stride:stride].view(B, D, T//2, 2, N)
            X_view = X[:, :, :T*stride:stride].view(B, D, T//2, 2, N)
            
            # X[left] = A[right] * X[left] + X[right]
            X_view[:, :, :, 1].add_(A_view[:, :, :, 1] * X_view[:, :, :, 0])
            # A[left] = A[left] * A[right]
            A_view[:, :, :, 1].mul_(A_view[:, :, :, 0])
        
        # Down-sweep (distribution)
        for k in range(num_steps - 1, -1, -1):
            T = 2 ** (num_steps - k)
            stride = 2 ** k
            
            A_view = A[:, :, :T*stride:stride].view(B, D, T//2, 2, N)
            X_view = X[:, :, :T*stride:stride].view(B, D, T//2, 2, N)
            
            # X[right] = A[left] * X[right] + X[left]
            if k > 0:  # Skip first step
                X_view[:, :, 1:, 1].add_(
                    A_view[:, :, 1:, 0] * X_view[:, :, :-1, 1]
                )
    
    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Forward pass: compute parallel scan.
        
        Args:
            A_in: (B, D, L, N) - transition coefficients (exp(delta * A))
            X_in: (B, D, L, N) - input values (delta * u * B)
        
        Returns:
            H: (B, D, L, N) - states at each time step
        """
        # Clone to avoid modifying inputs
        A = A_in.clone()
        X = X_in.clone()
        
        # Perform parallel scan in-place
        PScan.pscan(A, X)
        
        # Save for backward
        ctx.save_for_backward(A_in, X)
        
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients using reverse parallel scan.
        
        Args:
            grad_output: (B, D, L, N) - gradient w.r.t. output
        
        Returns:
            grad_A: gradient w.r.t. A
            grad_X: gradient w.r.t. X
        """
        A_in, X = ctx.saved_tensors
        B, D, L, N = A_in.size()
        
        # Clone A and flip for reverse scan
        A = A_in.clone().transpose(2, 1)  # (B, D, L, N) -> (B, D, L, N)
        
        # Flip A and grad_output for reverse scan
        A_flipped = torch.cat([A[:, :, :1], A[:, :, 1:].flip(2)], dim=2)
        grad_flipped = grad_output.transpose(2, 1).flip(2)
        
        # Perform reverse parallel scan
        PScan.pscan(A_flipped, grad_flipped)
        
        # Flip back
        grad_scan = grad_flipped.flip(2)
        
        # Compute gradients
        # grad_A[t] = H[t-1] * grad_scan[t]
        # grad_X[t] = grad_scan[t]
        
        # Shift H to get H[t-1]
        H_shifted = torch.zeros_like(X)
        H_shifted[:, :, 1:] = X[:, :, :-1]
        
        grad_A = H_shifted * grad_scan
        grad_X = grad_scan
        
        return grad_A, grad_X


def parallel_scan_fn(A, X):
    """
    Convenience function for parallel scan.
    
    Args:
        A: (B, D, L, N) - transition coefficients
        X: (B, D, L, N) - input values
    
    Returns:
        H: (B, D, L, N) - states at each time step
    """
    return PScan.apply(A, X)


def parallel_scan_ref(A, X):
    """
    Reference sequential scan for validation.
    
    Args:
        A: (B, D, L, N) - transition coefficients
        X: (B, D, L, N) - input values
    
    Returns:
        H: (B, D, L, N) - states at each time step
    """
    B, D, L, N = A.shape
    H = torch.zeros_like(X)
    H[:, :, 0] = X[:, :, 0]
    
    for t in range(1, L):
        H[:, :, t] = A[:, :, t] * H[:, :, t-1] + X[:, :, t]
    
    return H
