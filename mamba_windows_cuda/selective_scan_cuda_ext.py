"""
Extended CUDA implementation with backward kernel and kernel fusion.

Based on official state-spaces/mamba implementation:
https://github.com/state-spaces/mamba

Features:
1. Custom CUDA backward kernel
2. Kernel fusion (discretization + scan + output)
3. Checkpoint level support
"""

import os
import shutil
import torch
from torch.utils.cpp_extension import load_inline


def _ensure_msvc_env():
    if os.name != "nt":
        return
    if shutil.which("cl"):
        return
    try:
        vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
        if not os.path.exists(vswhere):
            return
        import subprocess
        cmd = [vswhere, "-latest", "-products", "*", "-requires",
               "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
               "-property", "installationPath"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        vs_path = result.stdout.strip()
        if not vs_path:
            return
        batch_file = os.path.join(vs_path, "Common7", "Tools", "VsDevCmd.bat")
        if not os.path.exists(batch_file):
            return
        cmd = f'"{batch_file}" -arch=x64 && set'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key] = value
    except Exception:
        return


_ensure_msvc_env()


# ============================================================================
# CUDA Kernel Source Code
# ============================================================================

cuda_source = r"""
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define N_STATE 16
#define WARP_SIZE 32

// ============================================================================
// Helper functions
// ============================================================================

__device__ __forceinline__ float safe_exp(float x) {
    x = fminf(fmaxf(x, -88.0f), 88.0f);
    return __expf(x);
}

template <typename scalar_t>
__device__ __forceinline__ float load_to_float(const scalar_t* p) {
    return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float load_to_float<at::Half>(const at::Half* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

template <typename scalar_t>
__device__ __forceinline__ void store_from_float(scalar_t* p, float v) {
    *p = static_cast<scalar_t>(v);
}

template <>
__device__ __forceinline__ void store_from_float<at::Half>(at::Half* p, float v) {
    *reinterpret_cast<__half*>(p) = __float2half_rn(v);
}

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
// Fused Forward Kernel: Discretization + Scan + Output
// ============================================================================

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void fused_selective_scan_fwd_kernel(
    const scalar_t* __restrict__ u,          // [B, L, D]
    const scalar_t* __restrict__ delta,      // [B, L, D]
    const scalar_t* __restrict__ A,          // [D, N]
    const scalar_t* __restrict__ B,          // [B, L, N]
    const scalar_t* __restrict__ C,          // [B, L, N]
    const scalar_t* __restrict__ D_ssm,      // [D]
    const scalar_t* __restrict__ h_prev,     // [B, D, N] optional
    scalar_t* __restrict__ h_last,           // [B, D, N] optional
    scalar_t* __restrict__ out,              // [B, L, D]
    // Intermediate storage for backward
    float* __restrict__ h_states,            // [B, D, N, L] optional
    float* __restrict__ dA_out,              // [B, D, N, L] optional
    int B_size, int L, int D, int N,
    bool save_intermediates
) {
    int lane = threadIdx.x;
    int warp_id = threadIdx.y;
    int b_idx = blockIdx.y;
    int d_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (d_idx >= D) return;

    extern __shared__ float smem[];
    float* B_shared = smem;
    float* C_shared = smem + N_STATE;

    // Load A values
    float A_val = 0.0f;
    if (lane < N_STATE) {
        A_val = load_to_float(A + d_idx * N_STATE + lane);
    }
    
    // Load D value
    float D_val = load_to_float(D_ssm + d_idx);

    // Initialize state from h_prev
    float h_val = 0.0f;
    if (h_prev != nullptr && lane < N_STATE) {
        h_val = load_to_float(h_prev + (static_cast<long long>(b_idx) * D + d_idx) * N_STATE + lane);
    }

    // Calculate offsets
    long long batch_offset = (long long)b_idx * L * D;
    long long batch_offset_ssm = (long long)b_idx * L * N_STATE;
    
    const scalar_t* u_ptr = u + batch_offset + d_idx;
    const scalar_t* delta_ptr = delta + batch_offset + d_idx;
    const scalar_t* B_ptr = B + batch_offset_ssm;
    const scalar_t* C_ptr = C + batch_offset_ssm;
    scalar_t* out_ptr = out + batch_offset + d_idx;

    // Forward scan
    unsigned mask = 0xFFFF;
    for (int l = 0; l < L; ++l) {
        // Load B and C for this timestep
        if (warp_id == 0 && lane < N_STATE) {
            B_shared[lane] = load_to_float(B_ptr + l * N_STATE + lane);
            C_shared[lane] = load_to_float(C_ptr + l * N_STATE + lane);
        }
        __syncthreads();

        // Broadcast u and delta from lane 0
        float u_val = 0.0f;
        float delta_val = 0.0f;
        if (lane == 0) {
            u_val = load_to_float(u_ptr + l * D);
            delta_val = load_to_float(delta_ptr + l * D);
        }
        u_val = __shfl_sync(0xFFFFFFFF, u_val, 0);
        delta_val = __shfl_sync(0xFFFFFFFF, delta_val, 0);

        if (lane < N_STATE) {
            // Fused: Discretization + Scan
            // dA = exp(delta * A)
            float dt_A = delta_val * A_val;
            float dA_exp = safe_exp(dt_A);
            
            // dBu = delta * u * B
            float dt_u_B = delta_val * u_val * B_shared[lane];
            
            // State update: h = dA * h + dBu
            h_val = dA_exp * h_val + dt_u_B;

            // Save intermediates for backward if needed
            if (save_intermediates && h_states != nullptr) {
                long long state_offset = ((long long)b_idx * D + d_idx) * N_STATE * L + lane * L + l;
                h_states[state_offset] = h_val;
            }
            if (save_intermediates && dA_out != nullptr) {
                long long state_offset = ((long long)b_idx * D + d_idx) * N_STATE * L + lane * L + l;
                dA_out[state_offset] = dA_exp;
            }

            // Output: y = sum(C * h) + D * u
            float y_contrib = h_val * C_shared[lane];
            y_contrib += __shfl_down_sync(mask, y_contrib, 8);
            y_contrib += __shfl_down_sync(mask, y_contrib, 4);
            y_contrib += __shfl_down_sync(mask, y_contrib, 2);
            y_contrib += __shfl_down_sync(mask, y_contrib, 1);

            if (lane == 0) {
                store_from_float(out_ptr + l * D, y_contrib + u_val * D_val);
            }
        }
        __syncthreads();
    }

    // Save final state
    if (h_last != nullptr && lane < N_STATE) {
        store_from_float(h_last + (static_cast<long long>(b_idx) * D + d_idx) * N_STATE + lane, h_val);
    }
}

// ============================================================================
// Backward Kernel
// ============================================================================

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void selective_scan_bwd_kernel(
    const scalar_t* __restrict__ dout,       // [B, L, D]
    const scalar_t* __restrict__ u,          // [B, L, D]
    const scalar_t* __restrict__ delta,      // [B, L, D]
    const scalar_t* __restrict__ A,          // [D, N]
    const scalar_t* __restrict__ B,          // [B, L, N]
    const scalar_t* __restrict__ C,          // [B, L, N]
    const scalar_t* __restrict__ D_ssm,      // [D]
    const float* __restrict__ h_states,      // [B, D, N, L]
    const float* __restrict__ dA_states,     // [B, D, N, L]
    const scalar_t* __restrict__ dh_last,    // [B, D, N] optional
    scalar_t* __restrict__ du,               // [B, L, D]
    scalar_t* __restrict__ ddelta,           // [B, L, D]
    float* __restrict__ dA,                  // [D, N]
    scalar_t* __restrict__ dB,               // [B, L, N]
    scalar_t* __restrict__ dC,               // [B, L, N]
    float* __restrict__ dD,                  // [D]
    int B_size, int L, int D, int N
) {
    int lane = threadIdx.x;
    int warp_id = threadIdx.y;
    int b_idx = blockIdx.y;
    int d_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (d_idx >= D) return;

    extern __shared__ float smem[];
    float* B_shared = smem;
    float* C_shared = smem + N_STATE;
    float* dh_shared = smem + 2 * N_STATE;

    // Load A values
    float A_val = 0.0f;
    if (lane < N_STATE) {
        A_val = load_to_float(A + d_idx * N_STATE + lane);
    }
    
    // Load D value
    float D_val = load_to_float(D_ssm + d_idx);

    // Initialize gradient for state from dh_last
    float dh_val = 0.0f;
    if (dh_last != nullptr && lane < N_STATE) {
        dh_val = load_to_float(dh_last + (static_cast<long long>(b_idx) * D + d_idx) * N_STATE + lane);
    }

    // Accumulators for parameter gradients
    float dA_acc = 0.0f;
    float dD_acc = 0.0f;

    // Calculate offsets
    long long batch_offset = (long long)b_idx * L * D;
    long long batch_offset_ssm = (long long)b_idx * L * N_STATE;
    
    const scalar_t* dout_ptr = dout + batch_offset + d_idx;
    const scalar_t* u_ptr = u + batch_offset + d_idx;
    const scalar_t* delta_ptr = delta + batch_offset + d_idx;
    const scalar_t* B_ptr = B + batch_offset_ssm;
    const scalar_t* C_ptr = C + batch_offset_ssm;
    scalar_t* du_ptr = du + batch_offset + d_idx;
    scalar_t* ddelta_ptr = ddelta + batch_offset + d_idx;

    // Backward scan (reverse in time)
    unsigned mask = 0xFFFF;
    for (int l = L - 1; l >= 0; --l) {
        // Load B and C for this timestep
        if (warp_id == 0 && lane < N_STATE) {
            B_shared[lane] = load_to_float(B_ptr + l * N_STATE + lane);
            C_shared[lane] = load_to_float(C_ptr + l * N_STATE + lane);
        }
        __syncthreads();

        // Broadcast dout from lane 0
        float dout_val = 0.0f;
        if (lane == 0) {
            dout_val = load_to_float(dout_ptr + l * D);
        }
        dout_val = __shfl_sync(0xFFFFFFFF, dout_val, 0);

        // Load u and delta
        float u_val = 0.0f;
        float delta_val = 0.0f;
        if (lane == 0) {
            u_val = load_to_float(u_ptr + l * D);
            delta_val = load_to_float(delta_ptr + l * D);
        }
        u_val = __shfl_sync(0xFFFFFFFF, u_val, 0);
        delta_val = __shfl_sync(0xFFFFFFFF, delta_val, 0);

        if (lane < N_STATE) {
            // Get saved intermediate values
            long long state_offset = ((long long)b_idx * D + d_idx) * N_STATE * L + lane * L + l;
            float h_prev_val = (l > 0) ? h_states[state_offset - 1] : 0.0f;
            float dA_exp = dA_states[state_offset];

            // Gradient from output: dy/dh = C
            float dh_from_out = dout_val * C_shared[lane];
            
            // Total gradient for h
            dh_val = dh_val + dh_from_out;

            // Gradient for C: dC = dout * h
            float dC_val = dout_val * h_val;
            // Atomic add for dC (accumulated across batch)
            atomicAdd(&dC[b_idx * L * N + l * N + lane], __float2half(dC_val));

            // Gradient for dA: dA += dh * h_prev
            dA_acc += dh_val * h_prev_val * delta_val;

            // Gradient for h_prev (propagate to previous timestep)
            float dh_prev = dh_val * dA_exp;

            // Gradient for delta from dA: ddelta += dh * h_prev * A * dA_exp
            float ddelta_val = dh_val * h_prev_val * A_val * dA_exp;

            // Gradient for u from output: du += D * dout
            float du_val = D_val * dout_val;

            // Gradient for delta from dBu: ddelta += u * B * dh
            ddelta_val += u_val * B_shared[lane] * dh_val;

            // Gradient for u from dBu: du += delta * B * dh
            du_val += delta_val * B_shared[lane] * dh_val;

            // Gradient for B: dB = delta * u * dh
            float dB_val = delta_val * u_val * dh_val;

            // Warp reduce and store
            du_val = warp_reduce_sum(du_val);
            ddelta_val = warp_reduce_sum(ddelta_val);

            if (lane == 0) {
                atomicAdd(&du[b_idx * L * D + l * D + d_idx], __float2half(du_val));
                atomicAdd(&ddelta[b_idx * L * D + l * D + d_idx], __float2half(ddelta_val));
            }

            // Update dh for next iteration
            dh_val = dh_prev;
        }
        __syncthreads();
    }

    // Accumulate dA across batch and time
    if (lane < N_STATE) {
        atomicAdd(&dA[d_idx * N_STATE + lane], dA_acc);
    }

    // Gradient for D: dD = sum(du * dout)
    if (lane == 0) {
        float dD_val = 0.0f;
        for (int l = 0; l < L; ++l) {
            float u_val = load_to_float(u_ptr + l * D);
            float dout_val = load_to_float(dout_ptr + l * D);
            dD_val += u_val * dout_val;
        }
        atomicAdd(&dD[d_idx], dD_val);
    }
}

// ============================================================================
// C++ Interface Functions
// ============================================================================

std::vector<torch::Tensor> fused_selective_scan_forward(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B_ssm,
    torch::Tensor C_ssm,
    torch::Tensor D_ssm,
    torch::optional<torch::Tensor> h_prev_opt,
    bool save_intermediates
) {
    // Validation
    TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
    TORCH_CHECK(u.scalar_type() == torch::kFloat || u.scalar_type() == torch::kHalf, "u must be float32/float16");

    int B = u.size(0);
    int L = u.size(1);
    int D = u.size(2);
    int N = A.size(1);
    TORCH_CHECK(N == 16, "Only N=16 is supported");

    auto out = torch::empty_like(u);
    auto h_last = torch::empty({B, D, N}, u.options());
    
    // Optional intermediate storage
    torch::Tensor h_states, dA_out;
    if (save_intermediates) {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(u.device());
        h_states = torch::empty({B, D, N, L}, opts);
        dA_out = torch::empty({B, D, N, L}, opts);
    }

    const void* h_prev_ptr = nullptr;
    if (h_prev_opt.has_value()) {
        h_prev_ptr = h_prev_opt.value().data_ptr();
    }

    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((D + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
    size_t shmem = 2 * N_STATE * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "fused_selective_scan_forward", [&] {
        fused_selective_scan_fwd_kernel<scalar_t, WARPS_PER_BLOCK><<<grid, block, shmem>>>(
            (const scalar_t*)u.data_ptr(),
            (const scalar_t*)delta.data_ptr(),
            (const scalar_t*)A.data_ptr(),
            (const scalar_t*)B_ssm.data_ptr(),
            (const scalar_t*)C_ssm.data_ptr(),
            (const scalar_t*)D_ssm.data_ptr(),
            h_prev_opt.has_value() ? (const scalar_t*)h_prev_ptr : nullptr,
            (scalar_t*)h_last.data_ptr(),
            (scalar_t*)out.data_ptr(),
            save_intermediates ? (float*)h_states.data_ptr() : nullptr,
            save_intermediates ? (float*)dA_out.data_ptr() : nullptr,
            B, L, D, N,
            save_intermediates
        );
    });

    if (save_intermediates) {
        return {out, h_last, h_states, dA_out};
    }
    return {out, h_last};
}

std::vector<torch::Tensor> selective_scan_backward(
    torch::Tensor dout,
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B_ssm,
    torch::Tensor C_ssm,
    torch::Tensor D_ssm,
    torch::Tensor h_states,
    torch::Tensor dA_states,
    torch::optional<torch::Tensor> dh_last_opt
) {
    // Validation
    TORCH_CHECK(dout.is_cuda(), "dout must be a CUDA tensor");

    int B = u.size(0);
    int L = u.size(1);
    int D = u.size(2);
    int N = A.size(1);

    auto du = torch::zeros_like(u);
    auto ddelta = torch::zeros_like(delta);
    auto dA = torch::zeros({D, N}, torch::TensorOptions().dtype(torch::kFloat32).device(u.device()));
    auto dB = torch::zeros_like(B_ssm);
    auto dC = torch::zeros_like(C_ssm);
    auto dD = torch::zeros({D}, torch::TensorOptions().dtype(torch::kFloat32).device(u.device()));

    const void* dh_last_ptr = nullptr;
    if (dh_last_opt.has_value()) {
        dh_last_ptr = dh_last_opt.value().data_ptr();
    }

    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((D + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
    size_t shmem = 3 * N_STATE * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "selective_scan_backward", [&] {
        selective_scan_bwd_kernel<scalar_t, WARPS_PER_BLOCK><<<grid, block, shmem>>>(
            (const scalar_t*)dout.data_ptr(),
            (const scalar_t*)u.data_ptr(),
            (const scalar_t*)delta.data_ptr(),
            (const scalar_t*)A.data_ptr(),
            (const scalar_t*)B_ssm.data_ptr(),
            (const scalar_t*)C_ssm.data_ptr(),
            (const scalar_t*)D_ssm.data_ptr(),
            (const float*)h_states.data_ptr(),
            (const float*)dA_states.data_ptr(),
            dh_last_opt.has_value() ? (const scalar_t*)dh_last_ptr : nullptr,
            (scalar_t*)du.data_ptr(),
            (scalar_t*)ddelta.data_ptr(),
            (float*)dA.data_ptr(),
            (scalar_t*)dB.data_ptr(),
            (scalar_t*)dC.data_ptr(),
            (float*)dD.data_ptr(),
            B, L, D, N
        );
    });

    return {du, ddelta, dA, dB, dC, dD};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_selective_scan_forward(
    torch::Tensor u, torch::Tensor delta, torch::Tensor A,
    torch::Tensor B_ssm, torch::Tensor C_ssm, torch::Tensor D_ssm,
    torch::optional<torch::Tensor> h_prev_opt, bool save_intermediates);

std::vector<torch::Tensor> selective_scan_backward(
    torch::Tensor dout, torch::Tensor u, torch::Tensor delta,
    torch::Tensor A, torch::Tensor B_ssm, torch::Tensor C_ssm,
    torch::Tensor D_ssm, torch::Tensor h_states, torch::Tensor dA_states,
    torch::optional<torch::Tensor> dh_last_opt);
"""

_fused_module = None

def _get_fused_module():
    global _fused_module
    if _fused_module is not None:
        return _fused_module
    
    arch_flags = []
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        maj, minr = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{minr}"
        arch_flags = [
            f"-gencode=arch=compute_{maj}{minr},code=sm_{maj}{minr}",
            f"-gencode=arch=compute_{maj}{minr},code=compute_{maj}{minr}",
        ]
    
    extra_cuda_cflags = ["-O3", "-allow-unsupported-compiler"]
    use_fast_math = os.environ.get("MAMBA_CUDA_USE_FAST_MATH", "1") == "1"
    if use_fast_math:
        extra_cuda_cflags.append("--use_fast_math")
    
    _fused_module = load_inline(
        name="mamba_fused_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["fused_selective_scan_forward", "selective_scan_backward"],
        verbose=False,
        extra_cuda_cflags=extra_cuda_cflags + arch_flags,
    )
    return _fused_module


class FusedSelectiveScanFunction(torch.autograd.Function):
    """
    Fused selective scan with custom CUDA backward kernel.
    
    Features:
    1. Fused forward kernel (discretization + scan + output)
    2. Custom CUDA backward kernel
    3. Checkpoint level support
    """
    
    @staticmethod
    def forward(ctx, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None, checkpoint_lvl=0):
        """
        Forward pass with optional intermediate saving.
        
        checkpoint_lvl:
            0: Save all intermediates (fastest backward, most memory)
            1: Save minimal intermediates (slower backward, less memory)
        """
        module = _get_fused_module()
        
        save_intermediates = (checkpoint_lvl == 0)
        
        results = module.fused_selective_scan_forward(
            u.contiguous(), delta.contiguous(), A.contiguous(),
            B_ssm.contiguous(), C_ssm.contiguous(), D_ssm.contiguous(),
            h_prev, save_intermediates
        )
        
        if save_intermediates:
            out, h_last, h_states, dA_states = results
            ctx.save_for_backward(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev,
                                  h_states, dA_states)
        else:
            out, h_last = results
            ctx.save_for_backward(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
            ctx.h_states = None
            ctx.dA_states = None
        
        ctx.checkpoint_lvl = checkpoint_lvl
        
        return out, h_last
    
    @staticmethod
    def backward(ctx, dout, dh_last):
        module = _get_fused_module()
        
        if ctx.checkpoint_lvl == 0:
            # Use saved intermediates
            u, delta, A, B_ssm, C_ssm, D_ssm, h_prev, h_states, dA_states = ctx.saved_tensors
        else:
            # Recompute intermediates
            u, delta, A, B_ssm, C_ssm, D_ssm, h_prev = ctx.saved_tensors
            # Recompute forward to get intermediates
            results = module.fused_selective_scan_forward(
                u.contiguous(), delta.contiguous(), A.contiguous(),
                B_ssm.contiguous(), C_ssm.contiguous(), D_ssm.contiguous(),
                h_prev, true  # save intermediates
            )
            _, _, h_states, dA_states = results
        
        # Call CUDA backward kernel
        results = module.selective_scan_backward(
            dout.contiguous(), u.contiguous(), delta.contiguous(),
            A.contiguous(), B_ssm.contiguous(), C_ssm.contiguous(),
            D_ssm.contiguous(), h_states, dA_states,
            dh_last if dh_last is not None else None
        )
        
        du, ddelta, dA, dB, dC, dD = results
        
        return du, ddelta, dA, dB, dC, dD, None, None


class FusedSelectiveScan(torch.nn.Module):
    """
    Fused Selective Scan module with custom CUDA backward kernel.
    
    Args:
        checkpoint_lvl: Checkpoint level (0 or 1)
            0: Save all intermediates (default, fastest backward)
            1: Recompute intermediates in backward (less memory)
    """
    
    def __init__(self, checkpoint_lvl=0):
        super().__init__()
        self.checkpoint_lvl = checkpoint_lvl
    
    def forward(self, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        """
        Forward pass.
        
        Args:
            u: (B, L, D)
            delta: (B, L, D)
            A: (D, N)
            B_ssm: (B, L, N)
            C_ssm: (B, L, N)
            D_ssm: (D,)
            h_prev: (B, D, N) optional
        
        Returns:
            out: (B, L, D)
        """
        out, _ = FusedSelectiveScanFunction.apply(
            u, delta, A, B_ssm, C_ssm, D_ssm, h_prev, self.checkpoint_lvl
        )
        return out
    
    def forward_with_state(self, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        """
        Forward pass that also returns final state.
        """
        return FusedSelectiveScanFunction.apply(
            u, delta, A, B_ssm, C_ssm, D_ssm, h_prev, self.checkpoint_lvl
        )
