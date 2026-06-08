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
            float dA_exp = safe_exp(dt_A);  // Use safe_exp to prevent overflow/underflow
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
    
    // Contiguity checks (last dimension stride should be 1)
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
            u_ptr,
            delta_ptr,
            A_ptr,
            B_ptr,
            C_ptr,
            D_ptr,
            h_prev_opt.has_value() ? h_ptr : nullptr,
            nullptr,
            out_ptr,
            B, L, D, N
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
    
    // Contiguity checks (last dimension stride should be 1)
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
            u_ptr,
            delta_ptr,
            A_ptr,
            B_ptr,
            C_ptr,
            D_ptr,
            h_prev_opt.has_value() ? h_ptr : nullptr,
            h_last_ptr,
            out_ptr,
            B, L, D, N
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

    # Compile options: allow users to choose precision vs speed tradeoff
    extra_cuda_cflags = ["-O3", "-allow-unsupported-compiler"]
    
    # Default: use fast math for better performance
    # Set MAMBA_CUDA_USE_FAST_MATH=0 for higher precision
    use_fast_math = os.environ.get("MAMBA_CUDA_USE_FAST_MATH", "1") == "1"
    if use_fast_math:
        extra_cuda_cflags.append("--use_fast_math")
    else:
        extra_cuda_cflags.append("--ftz=true")  # Flush-to-zero for denormals
    
    _selective_scan_cuda_module = load_inline(
        name="mamba_windows_selective_scan_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["selective_scan_cuda_forward", "selective_scan_cuda_forward_with_state"],
        verbose=False,
        extra_cuda_cflags=extra_cuda_cflags + arch_flags,
    )
    return _selective_scan_cuda_module


class SelectiveScanCuda(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = _get_selective_scan_cuda_module()

    def forward(self, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        if self.module is None:
            raise RuntimeError("CUDA kernel not compiled")

        return self.module.selective_scan_cuda_forward(
            u.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            B_ssm.contiguous(),
            C_ssm.contiguous(),
            D_ssm.contiguous(),
            h_prev,
        )

    def forward_with_state(self, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        if self.module is None:
            raise RuntimeError("CUDA kernel not compiled")

        out, h_last = self.module.selective_scan_cuda_forward_with_state(
            u.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            B_ssm.contiguous(),
            C_ssm.contiguous(),
            D_ssm.contiguous(),
            h_prev,
        )
        return out, h_last
