import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}, num_warps=nw, num_stages=ns)
        for bm in [32, 64, 128]
        for bn in [32, 64, 128]
        for nw in [4, 8]
        for ns in [2, 3, 4]
    ],
    key=['PATCH_SIZE', 'num_q_patches', 'num_k_patches'],
)
@triton.jit
def dba_pattern_gemm_kernel(
    Q, K, Out,
    stride_qb, stride_qt, stride_qd,  # Q strides (Batch*Head, T, Dim)
    stride_kb, stride_kt, stride_kd,  # K strides
    stride_ob, stride_om, stride_on,  # Out strides (Batch*Head, M, N)
    q_start_idx,
    num_q_patches,                    # NEW: total Q patches in this chunk
    num_k_patches,                    # NEW: total K patches
    PATCH_SIZE: tl.constexpr,
    HALF_P: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    # --- Causality Check (Early Exit) ---
    # Convert Q start token index to patch index offset
    q_start_patch = q_start_idx // PATCH_SIZE
    if IS_CAUSAL:
        if (pid_n * BLOCK_N) > (q_start_patch + (pid_m + 1) * BLOCK_M - 1):
            return

    # Offsets for current block
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Valid patch masks
    valid_m = offs_m < num_q_patches
    valid_n = offs_n < num_k_patches

    # Safe indices: out-of-range rows/cols map to 0 to keep pointers in-bounds
    safe_offs_m = tl.where(valid_m, offs_m, 0)
    safe_offs_n = tl.where(valid_n, offs_n, 0)

    # Base pointers (using 64-bit arithmetic for safety)
    Q_ptr_base = Q + pid_bh.to(tl.int64) * stride_qb
    K_ptr_base = K + pid_bh.to(tl.int64) * stride_kb
    Out_base = Out + pid_bh.to(tl.int64) * stride_ob


    # --- Pointers Setup ---
    # # Q: Starts at i=0.
    # # Row 1: offs_m * P + 0
    # # Row 2: offs_m * P + 0 + P/2
    q_idx_base = q_start_idx + safe_offs_m * PATCH_SIZE
    q_ptr_1 = Q_ptr_base + q_idx_base[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q_ptr_2 = Q_ptr_base + (q_idx_base + HALF_P)[:, None] * stride_qt + offs_d[None, :] * stride_qd

    # K: Starts at i=0 -> Group (P/2 - 1).
    start_group = HALF_P - 1
    k_col_base = safe_offs_n * PATCH_SIZE + start_group * 2
    k_ptr_1 = K_ptr_base + k_col_base[:, None] * stride_kt + offs_d[None, :] * stride_kd
    k_ptr_2 = K_ptr_base + (k_col_base + 1)[:, None] * stride_kt + offs_d[None, :] * stride_kd

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over the pattern's vertical structure (0 to P/2 - 1)
    q_step = stride_qt
    k_step = 2 * stride_kt

    for i in range(HALF_P):
        # # Sum randomly accessible rows from global memory

        q_swapped = (
            tl.load(q_ptr_1, mask=valid_m[:, None], other=0.0) +
            tl.load(q_ptr_2, mask=valid_m[:, None], other=0.0)
        )
        k_swapped = (
            tl.load(k_ptr_1, mask=valid_n[:, None], other=0.0) +
            tl.load(k_ptr_2, mask=valid_n[:, None], other=0.0)
        )

        # --- Dot Product ---
        acc += tl.dot(q_swapped, tl.trans(k_swapped))

        # Advance pointers
        q_ptr_1 += q_step
        q_ptr_2 += q_step
        k_ptr_1 -= k_step
        k_ptr_2 -= k_step

    # Store result
    out_ptr = Out_base + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    # tl.store(out_ptr, acc.to(Out.type.element_ty))
    tl.store(out_ptr, acc.to(Out.type.element_ty), mask=valid_m[:, None] & valid_n[None, :])

def dba_pattern_gemm(q, k, patch_size, start_token, num_q_patches, k_pad_len, is_causal=True):
    B, H, Q_Len, D = q.shape
    _, _, K_Len, _ = k.shape

    # Flatten Batch and Head for grid launch
    q_in = q.reshape(B * H, Q_Len, D)
    k_in = k.reshape(B * H, K_Len, D)

    num_k_patches = k_pad_len // patch_size

    output = torch.empty((B * H, num_q_patches, num_k_patches), device=q.device, dtype=q.dtype)

    grid = lambda META: (
        triton.cdiv(num_q_patches, META['BLOCK_M']),
        triton.cdiv(num_k_patches, META['BLOCK_N']),
        B * H
    )

    dba_pattern_gemm_kernel[grid](
        q_in, k_in, output,
        q_in.stride(0), q_in.stride(1), q_in.stride(2),
        k_in.stride(0), k_in.stride(1), k_in.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        start_token,
        num_q_patches,
        num_k_patches,
        patch_size,
        patch_size // 2,
        D,
        IS_CAUSAL=is_causal
    )

    return output.view(B, H, num_q_patches, num_k_patches)

def tune_kernel():
    B, H, D = 1, 32, 128

    q_lens = [4096, 8192, 8192 * 2, 8192 * 4, 8192 * 8, 8192 * 16, 8192 * 32]
    patch_sizes = [8, 16, 32]

    for patch_size in patch_sizes:
        for Q_Len in q_lens:
            K_Len = Q_Len
            print(f"\n=========================================================================")
            print(f"Tuning: Q_Len={Q_Len}, K_Len={K_Len}, patch_size={patch_size}")
            print(f"=========================================================================")

            q = torch.randn((B, H, Q_Len, D), device='cuda', dtype=torch.float16)
            k = torch.randn((B, H, K_Len, D), device='cuda', dtype=torch.float16)

            chunk_size = int(
                max(
                    min(
                        max(2048, 1 << (K_Len - 1).bit_length()),
                        128 * 1024 * 2048 // (1 << (K_Len - 1).bit_length()),
                    ),
                    2048,
                )
            )

            start_token = 0
            num_q_patches = chunk_size // patch_size
            k_pad_len = ((K_Len + 128 - 1) // 128) * 128 # block_size = 128

            print("Starting Triton Autotune")
            out = dba_pattern_gemm(q, k, patch_size, start_token, num_q_patches, k_pad_len, is_causal=True)

            num_k_patches = k_pad_len // patch_size
            cache_key = (patch_size, num_q_patches, num_k_patches)

            if hasattr(dba_pattern_gemm_kernel, 'cache') and cache_key in dba_pattern_gemm_kernel.cache:
                best_config = dba_pattern_gemm_kernel.cache[cache_key]
            else:
                best_config = dba_pattern_gemm_kernel.best_config

            print("\n-------------------------------------------------")
            print(f"Best configuration found on the current GPU for (patch_size={patch_size}, Q_Len={Q_Len}):")
            print("BLOCK_M =", best_config.kwargs.get("BLOCK_M"))
            print("BLOCK_N =", best_config.kwargs.get("BLOCK_N"))
            print("num_warps =", best_config.num_warps)
            print("num_stages =", best_config.num_stages)
            print("-------------------------------------------------\n")

            print("Running performance benchmark...")
            ms = triton.testing.do_bench(
                lambda: dba_pattern_gemm(q, k, patch_size, start_token, num_q_patches, k_pad_len, is_causal=True)
            )
            print(f"Kernel execution time with the best configuration: {ms:.3f} ms\n")

            del q, k, out
            torch.cuda.empty_cache()

if __name__ == '__main__':
    tune_kernel()
