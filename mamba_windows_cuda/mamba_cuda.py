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
            float dA_exp = __expf(dt_A);
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
    TORCH_CHECK(u.scalar_type() == torch::kFloat || u.scalar_type() == torch::kHalf, "u must be float32/float16");
    TORCH_CHECK(delta.scalar_type() == u.scalar_type(), "delta dtype must match u dtype");
    TORCH_CHECK(A.scalar_type() == u.scalar_type(), "A dtype must match u dtype");
    TORCH_CHECK(B_ssm.scalar_type() == u.scalar_type(), "B_ssm dtype must match u dtype");
    TORCH_CHECK(C_ssm.scalar_type() == u.scalar_type(), "C_ssm dtype must match u dtype");
    TORCH_CHECK(D_ssm.scalar_type() == u.scalar_type(), "D_ssm dtype must match u dtype");
    if (h_prev_opt.has_value()) {
        TORCH_CHECK(h_prev_opt.value().scalar_type() == u.scalar_type(), "h_prev dtype must match u dtype");
    }

    int B = u.size(0);
    int L = u.size(1);
    int D = u.size(2);
    int N = A.size(1);
    TORCH_CHECK(N == 16, "Only N=16 is supported");

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
    TORCH_CHECK(u.scalar_type() == torch::kFloat || u.scalar_type() == torch::kHalf, "u must be float32/float16");
    TORCH_CHECK(delta.scalar_type() == u.scalar_type(), "delta dtype must match u dtype");
    TORCH_CHECK(A.scalar_type() == u.scalar_type(), "A dtype must match u dtype");
    TORCH_CHECK(B_ssm.scalar_type() == u.scalar_type(), "B_ssm dtype must match u dtype");
    TORCH_CHECK(C_ssm.scalar_type() == u.scalar_type(), "C_ssm dtype must match u dtype");
    TORCH_CHECK(D_ssm.scalar_type() == u.scalar_type(), "D_ssm dtype must match u dtype");
    if (h_prev_opt.has_value()) {
        TORCH_CHECK(h_prev_opt.value().scalar_type() == u.scalar_type(), "h_prev dtype must match u dtype");
    }

    int B = u.size(0);
    int L = u.size(1);
    int D = u.size(2);
    int N = A.size(1);
    TORCH_CHECK(N == 16, "Only N=16 is supported");

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

// ====================================================================
// Backward pass kernel — Mamba-1 recomputation strategy:
// Phase 1: forward scan + store boundary states (every CHUNK_SIZE steps)
// Phase 2: backward scan — recompute forward per-chunk from boundaries,
//          then propagate gradients through the recurrence
// ====================================================================
#define CHUNK_SIZE 64

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void selective_scan_bwd_kernel_v1(
    const scalar_t* __restrict__ u,           // [B, L, D]
    const scalar_t* __restrict__ delta,       // [B, L, D]
    const scalar_t* __restrict__ A,           // [D, N]
    const scalar_t* __restrict__ B_ssm,       // [B, L, N]
    const scalar_t* __restrict__ C_ssm,       // [B, L, N]
    const scalar_t* __restrict__ D_ssm,       // [D]
    const scalar_t* __restrict__ grad_out,    // [B, L, D]
    scalar_t* __restrict__ grad_u,            // [B, L, D]
    scalar_t* __restrict__ grad_delta,        // [B, L, D]
    float* __restrict__ grad_A_accum,         // [D, N] float32 — atomic add
    float* __restrict__ grad_B_accum,         // [B, L, N] float32 — atomic add (shared across D)
    float* __restrict__ grad_C_accum,         // [B, L, N] float32 — atomic add (shared across D)
    float* __restrict__ grad_D_accum,         // [D] float32 — atomic add
    scalar_t* __restrict__ h_bank,            // [B, D, ceil(L/CHUNK), N] — boundary states
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
    float* h_smem = smem + 2 * N_STATE;

    float A_vals[N_STATE] = {0};
    float D_val = 0.0f;
    if (lane < N_STATE) {
        A_vals[lane] = load_to_float(A + d_idx * N + lane);
    }
    if (lane == 0) {
        D_val = load_to_float(D_ssm + d_idx);
    }
    D_val = __shfl_sync(0xFFFFFFFF, D_val, 0);
    for (int i = 0; i < N_STATE; i++) {
        A_vals[i] = __shfl_sync(0xFFFFFFFF, A_vals[i], i % 32);
    }

    long long batch_offset_u = (long long)b_idx * L * D;
    const scalar_t* u_ptr = u + batch_offset_u;
    const scalar_t* delta_ptr = delta + batch_offset_u;
    const scalar_t* go_ptr = grad_out + batch_offset_u;
    scalar_t* gu_ptr = grad_u + batch_offset_u;
    scalar_t* gd_ptr = grad_delta + batch_offset_u;

    long long batch_offset_ssm = (long long)b_idx * L * N;
    const scalar_t* B_ptr = B_ssm + batch_offset_ssm;
    const scalar_t* C_ptr = C_ssm + batch_offset_ssm;
    float* gB_ptr = grad_B_accum + batch_offset_ssm;
    float* gC_ptr = grad_C_accum + batch_offset_ssm;

    constexpr int CHUNK = CHUNK_SIZE;
    int n_chunks = (L + CHUNK - 1) / CHUNK;

    unsigned mask = 0xFFFF;

    float grad_A_local[N_STATE] = {0};
    float grad_D_local = 0.0f;

    // === Phase 1: Full forward scan with boundary state storage ===
    float h_state[N_STATE] = {0};
    int chunk_idx = 0;

    for (int t = 0; t < L; t++) {
        if (warp_id == 0 && lane < N_STATE) {
            B_shared[lane] = load_to_float(B_ptr + t * N + lane);
            C_shared[lane] = load_to_float(C_ptr + t * N + lane);
        }
        __syncthreads();

        float u_val = 0.0f, delta_val = 0.0f;
        if (lane == 0) {
            u_val = load_to_float(u_ptr + t * D + d_idx);
            delta_val = load_to_float(delta_ptr + t * D + d_idx);
        }
        u_val = __shfl_sync(mask, u_val, 0);
        delta_val = __shfl_sync(mask, delta_val, 0);

        if (lane < N_STATE) {
            float dA_exp = __expf(delta_val * A_vals[lane]);
            h_state[lane] = dA_exp * h_state[lane] + delta_val * u_val * B_shared[lane];
        }
        __syncthreads();

        if (t == L - 1 || (t + 1) % CHUNK == 0) {
            if (lane < N_STATE) {
                long long hb_idx = ((long long)b_idx * D + d_idx) * n_chunks * N + chunk_idx * N + lane;
                store_from_float(h_bank + hb_idx, h_state[lane]);
            }
            chunk_idx++;
        }
    }

    __syncthreads();

    // === Phase 2: Backward scan ===
    float dh_next[N_STATE] = {0};

    for (int chunk = n_chunks - 1; chunk >= 0; chunk--) {
        int chunk_start = chunk * CHUNK;
        int chunk_end = min(chunk_start + CHUNK, L);
        int chunk_len = chunk_end - chunk_start;

        float h_cur[N_STATE];
        if (chunk == 0) {
            for (int n = 0; n < N_STATE; n++) h_cur[n] = 0.0f;
        } else {
            long long hb_idx = ((long long)b_idx * D + d_idx) * n_chunks * N + (chunk - 1) * N;
            for (int n = 0; n < N_STATE; n++) {
                h_cur[n] = load_to_float(h_bank + hb_idx + n);
            }
        }

        float* my_h = h_smem + warp_id * CHUNK * N_STATE;
        for (int t_off = 0; t_off < chunk_len; t_off++) {
            int t = chunk_start + t_off;

            if (warp_id == 0 && lane < N_STATE) {
                B_shared[lane] = load_to_float(B_ptr + t * N + lane);
                C_shared[lane] = load_to_float(C_ptr + t * N + lane);
            }
            __syncthreads();

            float u_val = 0.0f, delta_val = 0.0f;
            if (lane == 0) {
                u_val = load_to_float(u_ptr + t * D + d_idx);
                delta_val = load_to_float(delta_ptr + t * D + d_idx);
            }
            u_val = __shfl_sync(mask, u_val, 0);
            delta_val = __shfl_sync(mask, delta_val, 0);

            if (lane < N_STATE) {
                float dA_exp = __expf(delta_val * A_vals[lane]);
                h_cur[lane] = dA_exp * h_cur[lane] + delta_val * u_val * B_shared[lane];
                my_h[t_off * N_STATE + lane] = h_cur[lane];
            }
            __syncthreads();
        }

        for (int t_off = chunk_len - 1; t_off >= 0; t_off--) {
            int t = chunk_start + t_off;

            float u_val = 0.0f, delta_val = 0.0f, dy = 0.0f;
            if (lane == 0) {
                u_val = load_to_float(u_ptr + t * D + d_idx);
                delta_val = load_to_float(delta_ptr + t * D + d_idx);
                dy = load_to_float(go_ptr + t * D + d_idx);
            }
            u_val = __shfl_sync(mask, u_val, 0);
            delta_val = __shfl_sync(mask, delta_val, 0);
            dy = __shfl_sync(mask, dy, 0);

            float delta_next = 0.0f;
            if (t < L - 1) {
                if (lane == 0) {
                    delta_next = load_to_float(delta_ptr + (t + 1) * D + d_idx);
                }
                delta_next = __shfl_sync(mask, delta_next, 0);
            }

            if (warp_id == 0 && lane < N_STATE) {
                B_shared[lane] = load_to_float(B_ptr + t * N + lane);
                C_shared[lane] = load_to_float(C_ptr + t * N + lane);
            }
            __syncthreads();

            if (lane < N_STATE) {
                float h_t_n = my_h[t_off * N_STATE + lane];

                float h_tm1_n;
                if (t_off > 0) {
                    h_tm1_n = my_h[(t_off - 1) * N_STATE + lane];
                } else if (chunk > 0) {
                    long long hb_idx = ((long long)b_idx * D + d_idx) * n_chunks * N + (chunk - 1) * N + lane;
                    h_tm1_n = load_to_float(h_bank + hb_idx);
                } else {
                    h_tm1_n = 0.0f;
                }

                float B_t_n = B_shared[lane];
                float C_t_n = C_shared[lane];
                float A_n = A_vals[lane];
                float dA_exp_t = __expf(delta_val * A_n);

                float dh_t_n = dy * C_t_n;
                if (t < L - 1) {
                    dh_t_n += dh_next[lane] * __expf(delta_next * A_n);
                }
                dh_next[lane] = dh_t_n;

                atomicAdd(gC_ptr + (long long)t * N + lane, dy * h_t_n);
                atomicAdd(gB_ptr + (long long)t * N + lane, dh_t_n * delta_val * u_val);
                grad_A_local[lane] += dh_t_n * delta_val * dA_exp_t * h_tm1_n;

                float ddelta_contrib = dh_t_n * A_n * dA_exp_t * h_tm1_n + dh_t_n * B_t_n * u_val;
                ddelta_contrib += __shfl_down_sync(mask, ddelta_contrib, 8);
                ddelta_contrib += __shfl_down_sync(mask, ddelta_contrib, 4);
                ddelta_contrib += __shfl_down_sync(mask, ddelta_contrib, 2);
                ddelta_contrib += __shfl_down_sync(mask, ddelta_contrib, 1);
                if (lane == 0) {
                    store_from_float(gd_ptr + (long long)t * D + d_idx, ddelta_contrib);
                }

                float du_contrib = dh_t_n * B_t_n * delta_val;
                du_contrib += __shfl_down_sync(mask, du_contrib, 8);
                du_contrib += __shfl_down_sync(mask, du_contrib, 4);
                du_contrib += __shfl_down_sync(mask, du_contrib, 2);
                du_contrib += __shfl_down_sync(mask, du_contrib, 1);
                if (lane == 0) {
                    store_from_float(gu_ptr + (long long)t * D + d_idx, du_contrib + dy * D_val);
                }

                grad_D_local += dy * u_val;
            }
            __syncthreads();
        }
    }

    if (lane < N_STATE) {
        atomicAdd(grad_A_accum + (long long)d_idx * N + lane, grad_A_local[lane]);
    }
    if (lane == 0) {
        atomicAdd(grad_D_accum + d_idx, grad_D_local);
    }
}

std::vector<torch::Tensor> selective_scan_cuda_backward(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B_ssm,
    torch::Tensor C_ssm,
    torch::Tensor D_ssm,
    torch::Tensor grad_out
) {
    TORCH_CHECK(u.scalar_type() == torch::kFloat || u.scalar_type() == torch::kHalf, "u must be float32/float16");
    TORCH_CHECK(delta.scalar_type() == u.scalar_type(), "delta dtype must match u dtype");
    TORCH_CHECK(A.scalar_type() == u.scalar_type(), "A dtype must match u dtype");
    TORCH_CHECK(B_ssm.scalar_type() == u.scalar_type(), "B_ssm dtype must match u dtype");
    TORCH_CHECK(C_ssm.scalar_type() == u.scalar_type(), "C_ssm dtype must match u dtype");
    TORCH_CHECK(D_ssm.scalar_type() == u.scalar_type(), "D_ssm dtype must match u dtype");
    TORCH_CHECK(grad_out.scalar_type() == u.scalar_type(), "grad_out dtype must match u dtype");

    int B = u.size(0);
    int L = u.size(1);
    int D = u.size(2);
    int N = A.size(1);
    TORCH_CHECK(N == 16, "Only N=16 is supported");

    constexpr int CHUNK = 64;
    int n_chunks = (L + CHUNK - 1) / CHUNK;

    auto grad_u = torch::zeros_like(u);
    auto grad_delta = torch::zeros_like(delta);
    // Always float32 for atomic accumulators (safe on all architectures,
    // and B/C need atomic summation across D channels)
    auto grad_A_accum = torch::zeros({D, N}, torch::TensorOptions().dtype(torch::kFloat32).device(u.device()));
    auto grad_B_accum = torch::zeros({B, L, N}, torch::TensorOptions().dtype(torch::kFloat32).device(u.device()));
    auto grad_C_accum = torch::zeros({B, L, N}, torch::TensorOptions().dtype(torch::kFloat32).device(u.device()));
    auto grad_D_accum = torch::zeros({D}, torch::TensorOptions().dtype(torch::kFloat32).device(u.device()));
    auto h_bank = torch::zeros({B, D, n_chunks, N}, u.options());

    constexpr int WARPS_PER_BLOCK = 8;
    dim3 block(32, WARPS_PER_BLOCK);
    dim3 grid((D + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, B);
    size_t shmem = (2 * N_STATE + WARPS_PER_BLOCK * CHUNK * N_STATE) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "selective_scan_cuda_backward", [&] {
        const scalar_t* u_ptr = (const scalar_t*)u.data_ptr();
        const scalar_t* delta_ptr = (const scalar_t*)delta.data_ptr();
        const scalar_t* A_ptr = (const scalar_t*)A.data_ptr();
        const scalar_t* B_ptr = (const scalar_t*)B_ssm.data_ptr();
        const scalar_t* C_ptr = (const scalar_t*)C_ssm.data_ptr();
        const scalar_t* D_ptr = (const scalar_t*)D_ssm.data_ptr();
        const scalar_t* go_ptr = (const scalar_t*)grad_out.data_ptr();
        scalar_t* gu_ptr = (scalar_t*)grad_u.data_ptr();
        scalar_t* gd_ptr = (scalar_t*)grad_delta.data_ptr();
        float* gA_ptr = grad_A_accum.data_ptr<float>();
        float* gB_ptr = grad_B_accum.data_ptr<float>();
        float* gC_ptr = grad_C_accum.data_ptr<float>();
        float* gD_ptr = grad_D_accum.data_ptr<float>();
        scalar_t* hb_ptr = (scalar_t*)h_bank.data_ptr();

        selective_scan_bwd_kernel_v1<scalar_t, WARPS_PER_BLOCK><<<grid, block, shmem>>>(
            u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, go_ptr,
            gu_ptr, gd_ptr, gA_ptr, gB_ptr, gC_ptr, gD_ptr, hb_ptr,
            B, L, D, N
        );
    });

    // Cast accumulated float32 gradients to input dtype
    auto grad_A = grad_A_accum.to(u.scalar_type());
    auto grad_B_out = grad_B_accum.to(u.scalar_type());
    auto grad_C_out = grad_C_accum.to(u.scalar_type());
    auto grad_D_out = grad_D_accum.to(u.scalar_type());

    return {grad_u, grad_delta, grad_A, grad_B_out, grad_C_out, grad_D_out};
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

std::vector<torch::Tensor> selective_scan_cuda_backward(
    torch::Tensor u,
    torch::Tensor delta,
    torch::Tensor A,
    torch::Tensor B_ssm,
    torch::Tensor C_ssm,
    torch::Tensor D_ssm,
    torch::Tensor grad_out
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

    _selective_scan_cuda_module = load_inline(
        name="mamba_windows_selective_scan_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["selective_scan_cuda_forward", "selective_scan_cuda_forward_with_state", "selective_scan_cuda_backward"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler",
                           "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING",
                           "-Xcompiler", "/Zc:preprocessor",
                           "-Xcompiler", "/permissive-",
                           "-Xcompiler", "/std:c++17"] + arch_flags,
        extra_cflags=["/permissive-", "/std:c++17"],
    )
    return _selective_scan_cuda_module


class SelectiveScanFn(torch.autograd.Function):
    """Autograd Function wrapping the CUDA selective scan with proper backward pass.

    Uses Mamba-1 recomputation strategy: the backward kernel internally
    recomputes the forward scan from saved inputs with chunk boundary states,
    then propagates gradients through the recurrence.
    """
    _module = None

    @staticmethod
    def _get_module():
        if SelectiveScanFn._module is None:
            SelectiveScanFn._module = _get_selective_scan_cuda_module()
        return SelectiveScanFn._module

    @staticmethod
    def forward(ctx, u, delta, A, B_ssm, C_ssm, D_ssm):
        mod = SelectiveScanFn._get_module()
        ctx.save_for_backward(u, delta, A, B_ssm, C_ssm, D_ssm)
        return mod.selective_scan_cuda_forward(
            u.contiguous(), delta.contiguous(), A.contiguous(),
            B_ssm.contiguous(), C_ssm.contiguous(), D_ssm.contiguous(),
            None  # h_prev not used in basic forward
        )

    @staticmethod
    def backward(ctx, grad_out):
        u, delta, A, B_ssm, C_ssm, D_ssm = ctx.saved_tensors
        mod = SelectiveScanFn._get_module()
        grad_u, grad_delta, grad_A, grad_B_ssm, grad_C_ssm, grad_D_ssm = \
            mod.selective_scan_cuda_backward(
                u.contiguous(), delta.contiguous(), A.contiguous(),
                B_ssm.contiguous(), C_ssm.contiguous(), D_ssm.contiguous(),
                grad_out.contiguous()
            )
        return grad_u, grad_delta, grad_A, grad_B_ssm, grad_C_ssm, grad_D_ssm


class SelectiveScanCuda(torch.nn.Module):
    """CUDA-accelerated selective scan with autograd backward pass.

    Delegates to SelectiveScanFn.apply() for proper autograd tracing.
    The forward_with_state method uses the raw CUDA forward for inference
    (returns final hidden state but doesn't support backprop through it).
    """
    def __init__(self):
        super().__init__()
        self.module = _get_selective_scan_cuda_module()

    def forward(self, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        if self.module is None:
            raise RuntimeError("CUDA kernel not compiled")

        if h_prev is not None:
            # With initial state: use raw CUDA (no autograd for h_prev branch)
            return self.module.selective_scan_cuda_forward(
                u.contiguous(), delta.contiguous(), A.contiguous(),
                B_ssm.contiguous(), C_ssm.contiguous(), D_ssm.contiguous(),
                h_prev,
            )

        # Use autograd Function for proper backward pass
        return SelectiveScanFn.apply(
            u.contiguous(), delta.contiguous(), A.contiguous(),
            B_ssm.contiguous(), C_ssm.contiguous(), D_ssm.contiguous(),
        )

    def forward_with_state(self, u, delta, A, B_ssm, C_ssm, D_ssm, h_prev=None):
        """Forward pass that also returns the final hidden state.

        NOTE: This path does NOT support backward through the returned state.
        Use for inference or streaming inference only.
        """
        if self.module is None:
            raise RuntimeError("CUDA kernel not compiled")

        out, h_last = self.module.selective_scan_cuda_forward_with_state(
            u.contiguous(), delta.contiguous(), A.contiguous(),
            B_ssm.contiguous(), C_ssm.contiguous(), D_ssm.contiguous(),
            h_prev,
        )
        return out, h_last
