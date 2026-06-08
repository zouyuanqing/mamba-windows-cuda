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

        cmd = [
            vswhere,
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ]
        import subprocess

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


cuda_source = r"""
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define N_STATE 16

// Safe exponential function to prevent overflow/underflow
__device__ __forceinline__ float safe_exp(float x) {
    const float max_exp = 88.0f;  // exp(88) ≈ 1.6e38
    const float min_exp = -88.0f; // exp(-88) ≈ 6.3e-39
    x = fminf(fmaxf(x, min_exp), max_exp);
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

// Forward kernel (same as before)
template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void selective_scan_fwd_kernel_v4(
    const scalar_t* __restrict__ u,      // [B, L, D]
    const scalar_t* __restrict__ delta,  // [B, L, D]
    const scalar_t* __restrict__ A,      // [D, N]
    const scalar_t* __restrict__ B_ssm,  // [B, L, N]
    const scalar_t* __restrict__ C_ssm,  // [B, L, N]
    const scalar_t* __restrict__ D_ssm,  // [D]
    const scalar_t* __restrict__ h_prev, // [B, D, N] (Optional, can be null)
    scalar_t* __restrict__ h_last,       // [B, D, N] (Optional, can be null)
    scalar_t* __restrict__ out,          // [B, L, D]
    int B, int L, int D, int N
) {
    int lane = threadIdx.x;
    int warp_id = threadIdx.y;
    int b_idx = blockIdx.y;
    int d_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (d_idx >= D) return;

    extern __shared__ float smem[];
    float* B_shared = smem;
    float* C_shared = smem + N_STATE;

    float A_val = 0.0f;
    if (lane < N_STATE) {
        A_val = load_to_float(A + d_idx * N_STATE + lane);
    }
    float D_val = load_to_float(D_ssm + d_idx);

    float h_val = 0.0f;
    if (h_prev != nullptr && lane < N_STATE) {
        h_val = load_to_float(h_prev + (static_cast<long long>(b_idx) * D + d_idx) * N_STATE + lane);
    }

    long long batch_offset = (long long)b_idx * L * D;
    const scalar_t* u_ptr = u + batch_offset + d_idx;
    const scalar_t* delta_ptr = delta + batch_offset + d_idx;
    long long batch_offset_ssm = (long long)b_idx * L * N_STATE;
    const scalar_t* B_ptr = B_ssm + batch_offset_ssm;
    const scalar_t* C_ptr = C_ssm + batch_offset_ssm;
    scalar_t* out_ptr = out + batch_offset + d_idx;

    unsigned mask = 0xFFFF;
    for (int l = 0; l < L; ++l) {
        if (warp_id == 0 && lane < N_STATE) {
            B_shared[lane] = load_to_float(B_ptr + l * N_STATE + lane);
            C_shared[lane] = load_to_float(C_ptr + l * N_STATE + lane);
        }
        __syncthreads();

        float u_val = 0.0f;
        float delta_val = 0.0f;
        if (lane == 0) {
            u_val = load_to_float(u_ptr + l * D);
            delta_val = load_to_float(delta_ptr + l * D);
        }
        u_val = __shfl_sync(0xFFFFFFFF, u_val, 0);
        delta_val = __shfl_sync(0xFFFFFFFF, delta_val, 0);

        if (lane < N_STATE) {
            float dt_A = delta_val * A_val;
            float dA_exp = safe_exp(dt_A);
            float dt_u_B = delta_val * u_val * B_shared[lane];
            h_val = dA_exp * h_val + dt_u_B;

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

    if (h_last != nullptr && lane < N_STATE) {
        store_from_float(h_last + (static_cast<long long>(b_idx) * D + d_idx) * N_STATE + lane, h_val);
    }
}

torch::Tensor selective_scan_cuda_forward(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B_ssm,
    torch::Tensor C_ssm,
    torch::Tensor D_ssm,
    torch::optional<torch::Tensor> h_prev_opt
) {
    // Device checks
    TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
    TORCH_CHECK(delta.is_cuda(), "delta must be a CUDA tensor");
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B_ssm.is_cuda(), "B_ssm must be a CUDA tensor");
    TORCH_CHECK(C_ssm.is_cuda(), "C_ssm must be a CUDA tensor");
    TORCH_CHECK(D_ssm.is_cuda(), "D_ssm must be a CUDA tensor");
    
    // Dtype checks
    TORCH_CHECK(u.scalar_type() == torch::kFloat || u.scalar_type() == torch::kHalf, "u must be float32/float16");
    TORCH_CHECK(delta.scalar_type() == u.scalar_type(), "delta dtype must match u dtype");
    TORCH_CHECK(A.scalar_type() == u.scalar_type(), "A dtype must match u dtype");
    TORCH_CHECK(B_ssm.scalar_type() == u.scalar_type(), "B_ssm dtype must match u dtype");
    TORCH_CHECK(C_ssm.scalar_type() == u.scalar_type(), "C_ssm dtype must match u dtype");
    TORCH_CHECK(D_ssm.scalar_type() == u.scalar_type(), "D_ssm dtype must match u dtype");
    
    // Contiguity checks
    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1, "u must be contiguous in the last dimension");
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1, "delta must be contiguous in the last dimension");
    TORCH_CHECK(A.stride(-1) == 1 || A.size(-1) == 1, "A must be contiguous in the last dimension");
    TORCH_CHECK(B_ssm.stride(-1) == 1 || B_ssm.size(-1) == 1, "B_ssm must be contiguous in the last dimension");
    TORCH_CHECK(C_ssm.stride(-1) == 1 || C_ssm.size(-1) == 1, "C_ssm must be contiguous in the last dimension");
    TORCH_CHECK(D_ssm.stride(-1) == 1 || D_ssm.size(-1) == 1, "D_ssm must be contiguous in the last dimension");

    int B = u.size(0);
    int L = u.size(1);
    int D = u.size(2);
    int N = A.size(1);
    
    // Shape checks
    TORCH_CHECK(N == 16, "Only N=16 is supported");
    TORCH_CHECK(u.dim() == 3, "u must be 3D (batch, seqlen, dim)");
    TORCH_CHECK(delta.dim() == 3, "delta must be 3D (batch, seqlen, dim)");
    TORCH_CHECK(A.dim() == 2, "A must be 2D (dim, state)");
    TORCH_CHECK(B_ssm.dim() == 3, "B_ssm must be 3D (batch, seqlen, state)");
    TORCH_CHECK(C_ssm.dim() == 3, "C_ssm must be 3D (batch, seqlen, state)");
    TORCH_CHECK(D_ssm.dim() == 1, "D_ssm must be 1D (dim)");
    
    TORCH_CHECK(u.size(0) == B && u.size(1) == L && u.size(2) == D, "u shape mismatch");
    TORCH_CHECK(delta.size(0) == B && delta.size(1) == L && delta.size(2) == D, "delta shape mismatch");
    TORCH_CHECK(A.size(0) == D && A.size(1) == N, "A shape mismatch");
    TORCH_CHECK(B_ssm.size(0) == B && B_ssm.size(1) == L && B_ssm.size(2) == N, "B_ssm shape mismatch");
    TORCH_CHECK(C_ssm.size(0) == B && C_ssm.size(1) == L && C_ssm.size(2) == N, "C_ssm shape mismatch");
    TORCH_CHECK(D_ssm.size(0) == D, "D_ssm shape mismatch");
    
    // h_prev validation
    if (h_prev_opt.has_value()) {
        auto h_prev = h_prev_opt.value();
        TORCH_CHECK(h_prev.is_cuda(), "h_prev must be a CUDA tensor");
        TORCH_CHECK(h_prev.scalar_type() == u.scalar_type(), "h_prev dtype must match u dtype");
        TORCH_CHECK(h_prev.stride(-1) == 1 || h_prev.size(-1) == 1, "h_prev must be contiguous in the last dimension");
        TORCH_CHECK(h_prev.dim() == 3, "h_prev must be 3D (batch, dim, state)");
        TORCH_CHECK(h_prev.size(0) == B && h_prev.size(1) == D && h_prev.size(2) == N, "h_prev shape must be (B, D, N)");
    }

    auto out = torch::empty_like(u);
    const void* h_prev_ptr = nullptr;
    if (h_prev_opt.has_value()) {
        h_prev_ptr = h_prev_opt.value().data_ptr();
    }

    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((D + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
    size_t shmem = 2 * N_STATE * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "selective_scan_cuda_forward", [&] {
        const scalar_t* u_ptr = (const scalar_t*)u.data_ptr();
        const scalar_t* delta_ptr = (const scalar_t*)delta.data_ptr();
        const scalar_t* A_ptr = (const scalar_t*)A.data_ptr();
        const scalar_t* B_ptr = (const scalar_t*)B_ssm.data_ptr();
        const scalar_t* C_ptr = (const scalar_t*)C_ssm.data_ptr();
        const scalar_t* D_ptr = (const scalar_t*)D_ssm.data_ptr();
        const scalar_t* h_ptr = (const scalar_t*)h_prev_ptr;
        scalar_t* out_ptr = (scalar_t*)out.data_ptr();

        selective_scan_fwd_kernel_v4<scalar_t, WARPS_PER_BLOCK><<<grid, block, shmem>>>(
            u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
            h_prev_opt.has_value() ? h_ptr : nullptr,
            nullptr, out_ptr, B, L, D, N
        );
    });

    return out;
}

std::vector<torch::Tensor> selective_scan_cuda_forward_with_state(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B_ssm,
    torch::Tensor C_ssm,
    torch::Tensor D_ssm,
    torch::optional<torch::Tensor> h_prev_opt
) {
    // Same validation as forward
    TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
    TORCH_CHECK(delta.is_cuda(), "delta must be a CUDA tensor");
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B_ssm.is_cuda(), "B_ssm must be a CUDA tensor");
    TORCH_CHECK(C_ssm.is_cuda(), "C_ssm must be a CUDA tensor");
    TORCH_CHECK(D_ssm.is_cuda(), "D_ssm must be a CUDA tensor");
    TORCH_CHECK(u.scalar_type() == torch::kFloat || u.scalar_type() == torch::kHalf, "u must be float32/float16");
    TORCH_CHECK(delta.scalar_type() == u.scalar_type(), "delta dtype must match u dtype");
    TORCH_CHECK(A.scalar_type() == u.scalar_type(), "A dtype must match u dtype");
    TORCH_CHECK(B_ssm.scalar_type() == u.scalar_type(), "B_ssm dtype must match u dtype");
    TORCH_CHECK(C_ssm.scalar_type() == u.scalar_type(), "C_ssm dtype must match u dtype");
    TORCH_CHECK(D_ssm.scalar_type() == u.scalar_type(), "D_ssm dtype must match u dtype");
    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1, "u must be contiguous in the last dimension");
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1, "delta must be contiguous in the last dimension");
    TORCH_CHECK(A.stride(-1) == 1 || A.size(-1) == 1, "A must be contiguous in the last dimension");
    TORCH_CHECK(B_ssm.stride(-1) == 1 || B_ssm.size(-1) == 1, "B_ssm must be contiguous in the last dimension");
    TORCH_CHECK(C_ssm.stride(-1) == 1 || C_ssm.size(-1) == 1, "C_ssm must be contiguous in the last dimension");
    TORCH_CHECK(D_ssm.stride(-1) == 1 || D_ssm.size(-1) == 1, "D_ssm must be contiguous in the last dimension");

    int B = u.size(0);
    int L = u.size(1);
    int D = u.size(2);
    int N = A.size(1);
    
    TORCH_CHECK(N == 16, "Only N=16 is supported");
    TORCH_CHECK(u.dim() == 3, "u must be 3D (batch, seqlen, dim)");
    TORCH_CHECK(delta.dim() == 3, "delta must be 3D (batch, seqlen, dim)");
    TORCH_CHECK(A.dim() == 2, "A must be 2D (dim, state)");
    TORCH_CHECK(B_ssm.dim() == 3, "B_ssm must be 3D (batch, seqlen, state)");
    TORCH_CHECK(C_ssm.dim() == 3, "C_ssm must be 3D (batch, seqlen, state)");
    TORCH_CHECK(D_ssm.dim() == 1, "D_ssm must be 1D (dim)");
    TORCH_CHECK(u.size(0) == B && u.size(1) == L && u.size(2) == D, "u shape mismatch");
    TORCH_CHECK(delta.size(0) == B && delta.size(1) == L && delta.size(2) == D, "delta shape mismatch");
    TORCH_CHECK(A.size(0) == D && A.size(1) == N, "A shape mismatch");
    TORCH_CHECK(B_ssm.size(0) == B && B_ssm.size(1) == L && B_ssm.size(2) == N, "B_ssm shape mismatch");
    TORCH_CHECK(C_ssm.size(0) == B && C_ssm.size(1) == L && C_ssm.size(2) == N, "C_ssm shape mismatch");
    TORCH_CHECK(D_ssm.size(0) == D, "D_ssm shape mismatch");
    
    if (h_prev_opt.has_value()) {
        auto h_prev = h_prev_opt.value();
        TORCH_CHECK(h_prev.is_cuda(), "h_prev must be a CUDA tensor");
        TORCH_CHECK(h_prev.scalar_type() == u.scalar_type(), "h_prev dtype must match u dtype");
        TORCH_CHECK(h_prev.stride(-1) == 1 || h_prev.size(-1) == 1, "h_prev must be contiguous in the last dimension");
        TORCH_CHECK(h_prev.dim() == 3, "h_prev must be 3D (batch, dim, state)");
        TORCH_CHECK(h_prev.size(0) == B && h_prev.size(1) == D && h_prev.size(2) == N, "h_prev shape must be (B, D, N)");
    }

    auto out = torch::empty_like(u);
    auto h_last = torch::empty({B, D, N}, u.options());
    const void* h_prev_ptr = nullptr;
    if (h_prev_opt.has_value()) {
        h_prev_ptr = h_prev_opt.value().data_ptr();
    }

    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((D + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
    size_t shmem = 2 * N_STATE * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "selective_scan_cuda_forward_with_state", [&] {
        const scalar_t* u_ptr = (const scalar_t*)u.data_ptr();
        const scalar_t* delta_ptr = (const scalar_t*)delta.data_ptr();
        const scalar_t* A_ptr = (const scalar_t*)A.data_ptr();
        const scalar_t* B_ptr = (const scalar_t*)B_ssm.data_ptr();
        const scalar_t* C_ptr = (const scalar_t*)C_ssm.data_ptr();
        const scalar_t* D_ptr = (const scalar_t*)D_ssm.data_ptr();
        const scalar_t* h_ptr = (const scalar_t*)h_prev_ptr;
        scalar_t* out_ptr = (scalar_t*)out.data_ptr();
        scalar_t* h_last_ptr = (scalar_t*)h_last.data_ptr();

        selective_scan_fwd_kernel_v4<scalar_t, WARPS_PER_BLOCK><<<grid, block, shmem>>>(
            u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
            h_prev_opt.has_value() ? h_ptr : nullptr,
            h_last_ptr, out_ptr, B, L, D, N
        );
    });

    return {out, h_last};
}
"""

cpp_source = """
torch::Tensor selective_scan_cuda_forward(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B_ssm,
    torch::Tensor C_ssm,
    torch::Tensor D_ssm,
    torch::optional<torch::Tensor> h_prev_opt
);

std::vector<torch::Tensor> selective_scan_cuda_forward_with_state(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B_ssm,
    torch::Tensor C_ssm,
    torch::Tensor D_ssm,
    torch::optional<torch::Tensor> h_prev_opt
);
"""

_selective_scan_cuda_module = None


def _get_selective_scan_cuda_module():
    global _selective_scan_cuda_module
    if _selective_scan_cuda_module is not None:
        return _selective_scan_cuda_module

    arch_flags = []
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        maj, minr = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{minr}"
        arch_flags = [
            f"-gencode=arch=compute_{maj}{minr},code=sm_{maj}{minr}",
            f"-gencode=arch=compute_{maj}{minr},code=compute_{maj}{minr}",
        ]

    # Compile options
    extra_cuda_cflags = ["-O3", "-allow-unsupported-compiler"]
    
    use_fast_math = os.environ.get("MAMBA_CUDA_USE_FAST_MATH", "1") == "1"
    if use_fast_math:
        extra_cuda_cflags.append("--use_fast_math")
    else:
        extra_cuda_cflags.append("--ftz=true")
    
    _selective_scan_cuda_module = load_inline(
        name="mamba_windows_selective_scan_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["selective_scan_cuda_forward", "selective_scan_cuda_forward_with_state"],
        verbose=False,
        extra_cuda_cflags=extra_cuda_cflags + arch_flags,
    )
    return _selective_scan_cuda_module


def selective_scan_ref(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None, return_intermediates=False):
    """
    Reference implementation of selective scan in pure PyTorch.
    Supports autograd for training.
    
    Args:
        u: (B, L, D)
        delta: (B, L, D)
        A: (D, N)
        B_ssm: (B, L, N)
        C_ssm: (B, L, N)
        D_ssm: (D,)
        h_prev: (B, D, N) or None
        return_intermediates: if True, return intermediate states for backward
    
    Returns:
        out: (B, L, D)
        h_last: (B, D, N)
        intermediates (optional): dict with 'h_states', 'dA', 'dBu'
    """
    batch, seqlen, dim = u.shape
    N = A.shape[1]
    
    # Transpose for computation: (B, D, L) and (B, N, L)
    u_t = u.transpose(1, 2)           # (B, D, L)
    delta_t = delta.transpose(1, 2)   # (B, D, L)
    B_t = B_ssm.transpose(1, 2)       # (B, N, L)
    C_t = C_ssm.transpose(1, 2)       # (B, N, L)
    
    # Initialize state
    if h_prev is None:
        h_prev = torch.zeros(batch, dim, N, device=u.device, dtype=u.dtype)
    
    # Compute dA = exp(delta * A) for each time step
    # delta: (B, D, L), A: (D, N) -> dA: (B, D, N, L)
    delta_A = torch.einsum('bdl,dn->bdln', delta_t, A)  # (B, D, N, L)
    dA = torch.exp(delta_A)  # (B, D, N, L)
    
    # Compute dBu = delta * u * B for each time step
    # delta: (B, D, L), u: (B, D, L), B: (B, N, L) -> dBu: (B, D, N, L)
    dBu = delta_t.unsqueeze(2) * u_t.unsqueeze(2) * B_t.unsqueeze(1)  # (B, D, N, L)
    
    # Compute state h_t recursively: h_t = dA_t * h_{t-1} + dBu_t
    h = h_prev.unsqueeze(-1)  # (B, D, N, 1)
    h_states = []
    for t in range(seqlen):
        h = dA[:, :, :, t:t+1] * h + dBu[:, :, :, t:t+1]  # (B, D, N, 1)
        h_states.append(h)
    
    h_states = torch.cat(h_states, dim=-1)  # (B, D, N, L)
    h_last = h.squeeze(-1)  # (B, D, N)
    
    # Compute output: y = sum(C * h) + D * u
    y = torch.sum(C_t.unsqueeze(1) * h_states, dim=2)  # (B, D, L)
    y = y + u_t * D_ssm.view(1, -1, 1)  # (B, D, L)
    
    out = y.transpose(1, 2)  # (B, L, D)
    
    if return_intermediates:
        intermediates = {
            'h_states': h_states,  # (B, D, N, L)
            'dA': dA,              # (B, D, N, L)
            'dBu': dBu,            # (B, D, N, L)
            'u_t': u_t,            # (B, D, L)
            'delta_t': delta_t,    # (B, D, L)
            'B_t': B_t,            # (B, N, L)
            'C_t': C_t,            # (B, N, L)
            'h_prev': h_prev,      # (B, D, N)
        }
        return out, h_last, intermediates
    
    return out, h_last


def selective_scan_backward(dout, dh_last, intermediates, D_ssm):
    """
    Compute gradients for selective scan.
    
    Args:
        dout: (B, L, D) gradient w.r.t. output
        dh_last: (B, D, N) gradient w.r.t. h_last
        intermediates: dict with forward intermediates
        D_ssm: (D,)
    
    Returns:
        du, ddelta, dA, dB_ssm, dC_ssm, dD_ssm, dh_prev
    """
    h_states = intermediates['h_states']  # (B, D, N, L)
    dA = intermediates['dA']              # (B, D, N, L)
    dBu = intermediates['dBu']            # (B, D, N, L)
    u_t = intermediates['u_t']            # (B, D, L)
    delta_t = intermediates['delta_t']    # (B, D, L)
    B_t = intermediates['B_t']            # (B, N, L)
    C_t = intermediates['C_t']            # (B, N, L)
    h_prev = intermediates['h_prev']      # (B, D, N)
    
    B, D, N, L = h_states.shape
    seqlen = L
    
    # Transpose dout: (B, L, D) -> (B, D, L)
    dout_t = dout.transpose(1, 2)
    
    # Initialize gradients
    du_t = torch.zeros_like(u_t)
    ddelta_t = torch.zeros_like(delta_t)
    dA_grad = torch.zeros_like(dA)
    dB_t_grad = torch.zeros_like(B_t)
    dC_t_grad = torch.zeros_like(C_t)
    dD_ssm = torch.zeros(D, device=dout.device, dtype=dout.dtype)
    dh = dh_last.unsqueeze(-1)  # (B, D, N, 1)
    
    # Backward pass (reverse in time)
    for t in range(seqlen - 1, -1, -1):
        # Gradient from output: y_t = sum(C_t * h_t) + D * u_t
        # dy_t/dh_t = C_t, dy_t/du_t = D
        dh_from_out = dout_t[:, :, t:t+1].unsqueeze(2) * C_t[:, :, t:t+1].unsqueeze(1)  # (B, D, N, 1)
        
        # Total gradient for h_t
        dh = dh + dh_from_out  # (B, D, N, 1)
        
        # Gradient for C_t: dC_t = dout_t * h_t
        dC_t_grad[:, :, t] = (dout_t[:, :, t:t+1].unsqueeze(2) * h_states[:, :, :, t:t+1]).sum(dim=1).squeeze(-1)
        
        # Gradient for u_t from D: du_t += D * dout_t
        du_t[:, :, t] += D_ssm * dout_t[:, :, t]
        
        # Gradient for D: dD += u_t * dout_t
        dD_ssm += (u_t[:, :, t] * dout_t[:, :, t]).sum(dim=0)
        
        # h_t = dA_t * h_{t-1} + dBu_t
        # dh_t/ddA_t = h_{t-1}, dh_t/dh_{t-1} = dA_t
        h_prev_t = h_states[:, :, :, t-1:t] if t > 0 else h_prev.unsqueeze(-1)
        
        # Gradient for dA_t
        dA_grad[:, :, :, t] = (dh * h_prev_t).squeeze(-1)
        
        # Gradient for h_{t-1}
        dh = dh * dA[:, :, :, t:t+1]
        
        # dBu_t = delta_t * u_t * B_t
        # dh/d(dBu_t) = 1 (since h_t = dA_t * h_{t-1} + dBu_t)
        dh_dBu = dh  # (B, D, N, 1)
        
        # Gradient for delta_t from dBu: ddelta_t += u_t * B_t * dh_dBu
        ddelta_t[:, :, t] += (u_t[:, :, t:t+1].unsqueeze(2) * B_t[:, :, t:t+1].unsqueeze(1) * dh_dBu).sum(dim=2).squeeze(-1)
        
        # Gradient for u_t from dBu: du_t += delta_t * B_t * dh_dBu
        du_t[:, :, t] += (delta_t[:, :, t:t+1].unsqueeze(2) * B_t[:, :, t:t+1].unsqueeze(1) * dh_dBu).sum(dim=2).squeeze(-1)
        
        # Gradient for B_t: dB_t += delta_t * u_t * dh_dBu
        dB_t_grad[:, :, t] += (delta_t[:, :, t:t+1].unsqueeze(1) * u_t[:, :, t:t+1].unsqueeze(1) * dh_dBu).sum(dim=1).squeeze(-1)
    
    # Gradient for h_prev
    dh_prev = dh.squeeze(-1)  # (B, D, N)
    
    # Gradient for A (needs to sum over time and batch)
    # dA comes from dA_grad which is (B, D, N, L)
    # A is (D, N), so we need to sum over B and L
    dA_final = dA_grad.sum(dim=(0, 3))  # (D, N)
    
    # Transpose gradients back
    du = du_t.transpose(1, 2)           # (B, L, D)
    ddelta = ddelta_t.transpose(1, 2)   # (B, L, D)
    dB_ssm = dB_t_grad.transpose(1, 2) # (B, L, N)
    dC_ssm = dC_t_grad.transpose(1, 2) # (B, L, N)
    
    return du, ddelta, dA_final, dB_ssm, dC_ssm, dD_ssm, dh_prev


class SelectiveScanCudaFunction(torch.autograd.Function):
    """
    Custom autograd function for selective scan.
    Uses CUDA kernel for forward, optimized PyTorch backward with saved intermediates.
    """
    
    @staticmethod
    def forward(ctx, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        # Use CUDA kernel for forward
        module = _get_selective_scan_cuda_module()
        out, h_last = module.selective_scan_cuda_forward_with_state(
            u.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            B_ssm.contiguous(),
            C_ssm.contiguous(),
            D_ssm.contiguous(),
            h_prev,
        )
        
        # Compute and save intermediates for backward (avoids recomputation)
        with torch.no_grad():
            _, _, intermediates = selective_scan_ref(
                u, delta, A, B_ssm, C_ssm, D_ssm, h_prev, return_intermediates=True
            )
        
        ctx.save_for_backward(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
        ctx.intermediates = intermediates
        
        return out, h_last
    
    @staticmethod
    def backward(ctx, dout, dh_last):
        u, delta, A, B_ssm, C_ssm, D_ssm, h_prev = ctx.saved_tensors
        intermediates = ctx.intermediates
        
        # Use optimized backward with saved intermediates
        du, ddelta, dA, dB_ssm, dC_ssm, dD_ssm, dh_prev = selective_scan_backward(
            dout.contiguous(),
            dh_last.contiguous() if dh_last is not None else torch.zeros_like(intermediates['h_prev']),
            intermediates,
            D_ssm,
        )
        
        return (
            du,
            ddelta,
            dA,
            dB_ssm,
            dC_ssm,
            dD_ssm,
            dh_prev if h_prev is not None else None,
        )


class SelectiveScanCuda(torch.nn.Module):
    """
    Mamba selective scan with CUDA acceleration and autograd support.
    
    Forward uses CUDA kernel for speed.
    Backward uses PyTorch reference for correctness (can be optimized with custom backward kernel).
    """
    
    def __init__(self):
        super().__init__()
        self._module = None
    
    def _get_module(self):
        if self._module is None:
            self._module = _get_selective_scan_cuda_module()
        return self._module

    def forward(self, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        """
        Forward pass for selective scan.
        
        Args:
            u: (B, L, D) input tensor
            delta: (B, L, D) delta tensor
            A: (D, N) state transition matrix
            B_ssm: (B, L, N) input matrix
            C_ssm: (B, L, N) output matrix
            D_ssm: (D,) skip connection
            h_prev: (B, D, N) initial state (optional)
        
        Returns:
            out: (B, L, D) output tensor
        """
        out, _ = SelectiveScanCudaFunction.apply(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
        return out

    def forward_with_state(self, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        """
        Forward pass that also returns the final state.
        
        Args:
            u: (B, L, D) input tensor
            delta: (B, L, D) delta tensor
            A: (D, N) state transition matrix
            B_ssm: (B, L, N) input matrix
            C_ssm: (B, L, N) output matrix
            D_ssm: (D,) skip connection
            h_prev: (B, D, N) initial state (optional)
        
        Returns:
            out: (B, L, D) output tensor
            h_last: (B, D, N) final state
        """
        return SelectiveScanCudaFunction.apply(u, delta, A, B_ssm, C_ssm, D_ssm, h_prev)
