import torch
import math
import torch.nn.functional as F
import triton
import triton.language as tl
from block_sparse_flash_attn import block_sparse_flash_attn_func # My new kernel

@triton.jit
def softmax_fuse_block_sum_kernel_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len, # we assume k_len is divisible by chunk size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size
    num_iters_before_causal = (chunk_start + (block_id + 1) * block_size - 1) // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = In + batch_id * input_stride_0 + head_id * input_stride_1 + block_id * block_size * input_stride_2
    input_ptr = input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2

    output_ptr = Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    l_i_inv = 1.0 / l_i

    sum_mask = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

    for iter in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + iter * segment_size)
        X = tl.where(mask, X, -1.0e6)
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

    for iter in range(num_iters_before_causal + 1, num_iters):
        X = tl.zeros([segment_size // block_size], dtype=tl.float32)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

@triton.jit
def softmax_fuse_block_sum_kernel_non_causal(
    In,
    Out,
    scale,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    real_q_len,
    k_len, # we assume k_len is divisible by chunk size
    chunk_start,
    chunk_end,
    segment_size: tl.constexpr,
    block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = In + batch_id * input_stride_0 + head_id * input_stride_1 + block_id * block_size * input_stride_2
    input_ptr = input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2

    output_ptr = Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)

        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local

        m_i = m_new

    l_i_inv = 1.0 / l_i

    sum_mask = offs_q[:, None] < real_q_len

    for iter in range(0, num_iters):
        X = tl.load(input_ptr + iter * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + iter * segment_size // block_size, X.to(Out.type.element_ty))

def softmax_fuse_block_sum(attn_weights_slice, reshaped_block_size, segment_size, chunk_start, chunk_end, real_q_len, scale, is_causal=True):
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape
    assert q_len % reshaped_block_size == 0
    try:
        assert k_len % segment_size == 0
    except:
        breakpoint()
    assert segment_size % reshaped_block_size == 0
    assert attn_weights_slice.stride(-1) == 1

    output = torch.empty((batch_size, num_heads, q_len // reshaped_block_size, k_len // reshaped_block_size), dtype=attn_weights_slice.dtype, device=attn_weights_slice.device)

    grid = (q_len // reshaped_block_size, num_heads, batch_size)

    if is_causal:
        softmax_fuse_block_sum_kernel_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )
    else:
        softmax_fuse_block_sum_kernel_non_causal[grid](
            attn_weights_slice,
            output,
            scale,
            attn_weights_slice.stride(0),
            attn_weights_slice.stride(1),
            attn_weights_slice.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            real_q_len,
            k_len,
            chunk_start,
            chunk_end,
            segment_size,
            reshaped_block_size,
        )

    return output

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

    # A100 config:
    if "A100" in torch.cuda.get_device_properties(torch.cuda.current_device()).name:
        if patch_size == 8:
            BLOCK_M = 128
            BLOCK_N = 128
            num_warps = 8
            num_stages = 2
        elif patch_size == 16:
            if Q_Len <= 4096:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
            elif Q_Len <= 8192:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 2
            elif Q_Len <= 16384:
                BLOCK_M = 128
                BLOCK_N = 128
                num_warps = 8
                num_stages = 2
            elif Q_Len <= 32768:
                BLOCK_M = 128
                BLOCK_N = 128
                num_warps = 8
                num_stages = 2
            elif Q_Len <= 65536:
                BLOCK_M = 128
                BLOCK_N = 128
                num_warps = 8
                num_stages = 2
            elif Q_Len <= 131072:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 2
            else:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 2
        elif patch_size == 32:
            BLOCK_M = 64
            BLOCK_N = 64
            num_warps = 8
            num_stages = 3
    # H20 config:
    elif "H20" in torch.cuda.get_device_properties(torch.cuda.current_device()).name:
        if patch_size == 8:
            if Q_Len <= 4096:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
            elif Q_Len <= 8192:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 2
            elif Q_Len <= 16384:
                BLOCK_M = 64
                BLOCK_N = 128
                num_warps = 4
                num_stages = 2
            elif Q_Len <= 32768:
                BLOCK_M = 64
                BLOCK_N = 128
                num_warps = 4
                num_stages = 4
            elif Q_Len <= 65536:
                BLOCK_M = 64
                BLOCK_N = 128
                num_warps = 4
                num_stages = 2
            elif Q_Len <= 131072:
                BLOCK_M = 64
                BLOCK_N = 128
                num_warps = 4
                num_stages = 4
            else:
                BLOCK_M = 128
                BLOCK_N = 128
                num_warps = 4
                num_stages = 4
        elif patch_size == 16:
            if Q_Len <= 4096:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
            elif Q_Len <= 8192:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
            elif Q_Len <= 16384:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
            elif Q_Len <= 32768:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
            elif Q_Len <= 65536:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 4
            elif Q_Len <= 131072:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 2
            else:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
        elif patch_size == 32:
            if Q_Len <= 4096:
                BLOCK_M = 32
                BLOCK_N = 32
                num_warps = 4
                num_stages = 3
            elif Q_Len <= 8192:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 4
            elif Q_Len <= 16384:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 2
            elif Q_Len <= 32768:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
            elif Q_Len <= 65536:
                BLOCK_M = 64
                BLOCK_N = 64
                num_warps = 8
                num_stages = 3
            elif Q_Len <= 131072:
                BLOCK_M = 32
                BLOCK_N = 32
                num_warps = 4
                num_stages = 3
            else:
                BLOCK_M = 32
                BLOCK_N = 32
                num_warps = 4
                num_stages = 3
    # Default fallback
    else:
        BLOCK_M = 64
        BLOCK_N = 64
        num_warps = 4
        num_stages = 2

    grid = (
        triton.cdiv(num_q_patches, BLOCK_M),
        triton.cdiv(num_k_patches, BLOCK_N),
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
        IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=1
    )

    return output.view(B, H, num_q_patches, num_k_patches)

def create_block_mask_pytorch(
    q, k,
    block_size=128,
    patch_size=8, # Can be 8, 16, 32...
    causal=True,
    cumulative_p_threshold=0.9,
    chunk_size=None,
    keep_sink=True,
    keep_local=True,
):
    """
    Create block sparse mask using double-band anti-diagonal pattern.

    Pattern for patch_size=8 (K=2, each row selects 2 consecutive K positions):
       0 1 2 3 4 5 6 7
      +----------------
    0 | . . . . . . 1 1
    1 | . . . . 1 1 . .
    2 | . . 1 1 . . . .
    3 | 1 1 . . . . . .
    4 | . . . . . . 1 1
    5 | . . . . 1 1 . .
    6 | . . 1 1 . . . .
    7 | 1 1 . . . . . .

    Row i selects columns: (patch_size - 2 - 2*i) % patch_size and (patch_size - 1 - 2*i) % patch_size
    """
    device = q.device
    B, H, Q_Len, D = q.shape
    _, _, K_Len, _ = k.shape

    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (K_Len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (K_Len - 1).bit_length()),
                ),
                2048,
            )
        )

    # --- GENERALIZED CONSTANTS ---
    # The pattern always selects 2 columns (e.g., "1 1")
    K_BAND = 2

    # The pattern always repeats twice vertically (Top Half / Bottom Half)
    # e.g., for P=16, rows 0-7 are unique, rows 8-15 repeat them.
    Q_REPEATS = 2

    # Derived dimensions
    # For P=16: 16 // 2 = 8 groups.
    num_groups = patch_size // K_BAND

    # 1. Padding
    q_pad_len = ((Q_Len + block_size - 1) // block_size) * block_size
    if q_pad_len > Q_Len: q = F.pad(q, (0, 0, 0, q_pad_len - Q_Len))
    k_pad_len = ((K_Len + block_size - 1) // block_size) * block_size
    if k_pad_len > K_Len: k = F.pad(k, (0, 0, 0, k_pad_len - K_Len))

    num_chunks = (q_pad_len + chunk_size - 1) // chunk_size
    num_k_patches = k_pad_len // patch_size
    k_block_num = k_pad_len // block_size

    # =================================================================================
    # PHASE 1: K Projection (Sum & Accumulate)
    # =================================================================================
    # Goal: Sum consecutive K_BAND columns.
    # Shape: (B, H, K_patches, num_groups, K_BAND, D)
    k_patched = k.view(B, H, num_k_patches, num_groups, K_BAND, D)

    # Sum the band (0+1, 2+3, ...)
    # Shape: (B, H, K_patches, num_groups, D)
    k_summed = k_patched.sum(dim=-2)

    # ALIGNMENT:
    # Pattern Logic: Row 0 (Index 0) targets the LAST K-Group (Index N).
    # We FLIP K here so that Index 0 contains the Last Group's data.
    # This aligns the indices: Q[0] will multiply with K_flipped[0] (which is G_last).
    k_aligned = k_summed.flip(dims=[-2])

    # Flatten for MatMul
    # Shape: (B, H, num_groups * D, K_patches) (Transposed)
    k_in = k_aligned.reshape(B, H, num_k_patches, -1).transpose(-1, -2)

    block_mask_list = []
    attn_probs_list = []

    for i in range(num_chunks):
        start_token = i * chunk_size
        end_token = min(start_token + chunk_size, q_pad_len)
        chunk_len = end_token - start_token
        num_q_patches = chunk_len // patch_size

        # =============================================================================
        # PHASE 2: Q Projection (Sum & Accumulate)
        # =============================================================================
        q_chunk = q[:, :, start_token:end_token, :]

        # Goal: Sum the repeating halves.
        # Structure: The patch is split into Q_REPEATS (2) equal vertical parts.
        # Height of one part = patch_size // Q_REPEATS
        rows_per_part = patch_size // Q_REPEATS

        # View: (B, H, Q_patches, Q_REPEATS, rows_per_part, D)
        # Dim -3 (Q_REPEATS) separates the Top Half from Bottom Half
        q_view = q_chunk.view(B, H, num_q_patches, Q_REPEATS, rows_per_part, D)

        # Sum the halves (Row i + Row i+P/2)
        # Shape: (B, H, Q_patches, rows_per_part, D)
        # Note: rows_per_part is exactly equal to num_groups (P/2 == P/2)
        q_summed = q_view.sum(dim=-3)

        # Flatten for MatMul
        # Shape: (B, H, Q_patches, num_groups * D)
        q_in = q_summed.reshape(B, H, num_q_patches, -1)

        # =============================================================================
        # PHASE 3: Efficient Scoring
        # =============================================================================
        # MatMul calculates: (Sum of Q Halves) . (Sum of K Bands)
        scores = torch.matmul(q_in, k_in)

        scores.div_(math.sqrt(D) * patch_size * 2)

        if causal:
            q_idx = torch.arange(num_q_patches, device=device).view(1, -1, 1) + (start_token // patch_size)
            k_idx = torch.arange(num_k_patches, device=device).view(1, 1, -1)
            mask = q_idx >= k_idx
            scores.masked_fill_(~mask.unsqueeze(1), float("-inf"))

        # =============================================================================
        # PHASE 4: Block Aggregation
        # =============================================================================
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs.nan_to_num_(0.0)

        patches_per_block = block_size // patch_size
        num_q_chunk_blocks = chunk_len // block_size

        # Reshape to Blocks
        attn_probs = attn_probs.view(B, H, num_q_chunk_blocks, patches_per_block, k_block_num, patches_per_block)

        # Aggregation: Sum K, Sum Q (Total Mass logic)
        block_attn_probs = attn_probs.sum(dim=-1).sum(dim=-2)
        attn_probs_list.append(block_attn_probs)

        # =============================================================================
        # PHASE 5: Selection Logic (Standard)
        # =============================================================================
        chunk_start_block = start_token // block_size
        q_blk_idx = torch.arange(num_q_chunk_blocks, device=device).view(-1, 1) + chunk_start_block
        k_blk_idx = torch.arange(k_block_num, device=device).view(1, -1)

        mandatory_mask = torch.zeros_like(block_attn_probs, dtype=torch.bool)
        if keep_sink: mandatory_mask[..., 0] = True
        if keep_local: mandatory_mask |= (q_blk_idx == k_blk_idx).unsqueeze(0).unsqueeze(0)

        sort_scores = block_attn_probs.clone()
        sort_scores.masked_fill_(mandatory_mask, 1000.0)

        _, sorted_indices = torch.sort(sort_scores, dim=-1, descending=True)
        sorted_probs = torch.gather(block_attn_probs, -1, sorted_indices)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        total_mass = block_attn_probs.sum(dim=-1, keepdim=True)
        if isinstance(cumulative_p_threshold, torch.Tensor):
            if cumulative_p_threshold.dim() == 1 and cumulative_p_threshold.size(0) == H:
                cumulative_p_threshold = cumulative_p_threshold.view(1, H, 1, 1)
        required_mass = total_mass * cumulative_p_threshold
        cumsum_without_self = torch.cat([torch.zeros_like(cumsum_probs[..., :1]), cumsum_probs[..., :-1]], dim=-1)
        keep_mask_sorted = cumsum_without_self < required_mass

        chunk_mask = torch.zeros_like(block_attn_probs, dtype=torch.bool)
        chunk_mask.scatter_(dim=-1, index=sorted_indices, src=keep_mask_sorted)
        chunk_mask |= mandatory_mask

        if causal:
            chunk_mask &= (q_blk_idx >= k_blk_idx).unsqueeze(0).unsqueeze(0)

        block_mask_list.append(chunk_mask)

    return torch.cat(block_mask_list, dim=-2), torch.cat(attn_probs_list, dim=-2)

def create_block_mask_triton_only_softmax(
    q, k,
    block_size=128,
    patch_size=8, # Can be 8, 16, 32...
    causal=True,
    cumulative_p_threshold=0.9,
    chunk_size=None,
    keep_sink=True,
    keep_local=True,
):
    """
    Create block sparse mask using double-band anti-diagonal pattern.

    Pattern for patch_size=8 (K=2, each row selects 2 consecutive K positions):
       0 1 2 3 4 5 6 7
      +----------------
    0 | . . . . . . 1 1
    1 | . . . . 1 1 . .
    2 | . . 1 1 . . . .
    3 | 1 1 . . . . . .
    4 | . . . . . . 1 1
    5 | . . . . 1 1 . .
    6 | . . 1 1 . . . .
    7 | 1 1 . . . . . .

    Row i selects columns: (patch_size - 2 - 2*i) % patch_size and (patch_size - 1 - 2*i) % patch_size
    """
    device = q.device
    B, H, Q_Len, D = q.shape
    _, _, K_Len, _ = k.shape


    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (K_Len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (K_Len - 1).bit_length()),
                ),
                2048,
            )
        )

    # --- GENERALIZED CONSTANTS ---
    # The pattern always selects 2 columns (e.g., "1 1")
    K_BAND = 2

    # The pattern always repeats twice vertically (Top Half / Bottom Half)
    # e.g., for P=16, rows 0-7 are unique, rows 8-15 repeat them.
    Q_REPEATS = 2

    # Derived dimensions
    # For P=16: 16 // 2 = 8 groups.
    num_groups = patch_size // K_BAND

    # 1. Padding
    q_pad_len = ((Q_Len + block_size - 1) // block_size) * block_size
    if q_pad_len > Q_Len: q = F.pad(q, (0, 0, 0, q_pad_len - Q_Len))
    k_pad_len = ((K_Len + block_size - 1) // block_size) * block_size
    if k_pad_len > K_Len: k = F.pad(k, (0, 0, 0, k_pad_len - K_Len))

    num_chunks = (q_pad_len + chunk_size - 1) // chunk_size
    num_k_patches = k_pad_len // patch_size
    k_block_num = k_pad_len // block_size

    # =================================================================================
    # PHASE 1: K Projection (Sum & Accumulate)
    # =================================================================================
    # Goal: Sum consecutive K_BAND columns.
    # Shape: (B, H, K_patches, num_groups, K_BAND, D)
    k_patched = k.view(B, H, num_k_patches, num_groups, K_BAND, D)

    # Sum the band (0+1, 2+3, ...)
    # Shape: (B, H, K_patches, num_groups, D)
    k_summed = k_patched.sum(dim=-2)

    # ALIGNMENT:
    # Pattern Logic: Row 0 (Index 0) targets the LAST K-Group (Index N).
    # We FLIP K here so that Index 0 contains the Last Group's data.
    # This aligns the indices: Q[0] will multiply with K_flipped[0] (which is G_last).
    k_aligned = k_summed.flip(dims=[-2])

    # Flatten for MatMul
    # Shape: (B, H, num_groups * D, K_patches) (Transposed)
    k_in = k_aligned.reshape(B, H, num_k_patches, -1).transpose(-1, -2)

    block_mask_list = []
    attn_probs_list = []

    for i in range(num_chunks):
        start_token = i * chunk_size
        end_token = min(start_token + chunk_size, q_pad_len)
        chunk_len = end_token - start_token
        num_q_patches = chunk_len // patch_size

        # =============================================================================
        # PHASE 2: Q Projection (Sum & Accumulate)
        # =============================================================================
        q_chunk = q[:, :, start_token:end_token, :]

        # Goal: Sum the repeating halves.
        # Structure: The patch is split into Q_REPEATS (2) equal vertical parts.
        # Height of one part = patch_size // Q_REPEATS
        rows_per_part = patch_size // Q_REPEATS

        # View: (B, H, Q_patches, Q_REPEATS, rows_per_part, D)
        # Dim -3 (Q_REPEATS) separates the Top Half from Bottom Half
        q_view = q_chunk.view(B, H, num_q_patches, Q_REPEATS, rows_per_part, D)

        # Sum the halves (Row i + Row i+P/2)
        # Shape: (B, H, Q_patches, rows_per_part, D)
        # Note: rows_per_part is exactly equal to num_groups (P/2 == P/2)
        q_summed = q_view.sum(dim=-3)

        # Flatten for MatMul
        # Shape: (B, H, Q_patches, num_groups * D)
        q_in = q_summed.reshape(B, H, num_q_patches, -1)

        # =============================================================================
        # PHASE 3: Efficient Scoring
        # =============================================================================
        # MatMul calculates: (Sum of Q Halves) . (Sum of K Bands)
        scores = torch.matmul(q_in, k_in)

        # =============================================================================
        # PHASE 4: Block Aggregation
        # =============================================================================
        patches_per_block = block_size // patch_size
        chunk_start_patch = start_token // patch_size
        chunk_end_patch = end_token // patch_size
        real_q_patches = (Q_Len + patch_size - 1) // patch_size

        # segment_size should match patches_per_block to ensure we aggregate exactly one block's worth of K patches
        # into one output value.
        block_attn_probs = softmax_fuse_block_sum(
            scores,
            patches_per_block,  # reshaped_block_size (Aggregation factor for Q)
            patches_per_block,  # segment_size (Aggregation factor for K)
            chunk_start_patch,
            chunk_end_patch,
            real_q_patches,
            math.log2(math.e) / (math.sqrt(D) * patch_size * 2),
            is_causal=causal,
        )

        attn_probs_list.append(block_attn_probs)
        num_q_chunk_blocks = chunk_len // block_size
        # =============================================================================
        # PHASE 5: Selection Logic (Standard)
        # =============================================================================
        chunk_start_block = start_token // block_size
        q_blk_idx = torch.arange(num_q_chunk_blocks, device=device).view(-1, 1) + chunk_start_block
        k_blk_idx = torch.arange(k_block_num, device=device).view(1, -1)

        mandatory_mask = torch.zeros_like(block_attn_probs, dtype=torch.bool)
        if keep_sink: mandatory_mask[..., 0] = True
        if keep_local: mandatory_mask |= (q_blk_idx == k_blk_idx).unsqueeze(0).unsqueeze(0)

        sort_scores = block_attn_probs.clone()
        sort_scores.masked_fill_(mandatory_mask, 1000.0)

        _, sorted_indices = torch.sort(sort_scores, dim=-1, descending=True)
        sorted_probs = torch.gather(block_attn_probs, -1, sorted_indices)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        total_mass = block_attn_probs.sum(dim=-1, keepdim=True)
        if isinstance(cumulative_p_threshold, torch.Tensor):
            if cumulative_p_threshold.dim() == 1 and cumulative_p_threshold.size(0) == H:
                cumulative_p_threshold = cumulative_p_threshold.view(1, H, 1, 1)
        required_mass = total_mass * cumulative_p_threshold
        cumsum_without_self = torch.cat([torch.zeros_like(cumsum_probs[..., :1]), cumsum_probs[..., :-1]], dim=-1)
        keep_mask_sorted = cumsum_without_self < required_mass

        chunk_mask = torch.zeros_like(block_attn_probs, dtype=torch.bool)
        chunk_mask.scatter_(dim=-1, index=sorted_indices, src=keep_mask_sorted)
        chunk_mask |= mandatory_mask

        if causal:
            chunk_mask &= (q_blk_idx >= k_blk_idx).unsqueeze(0).unsqueeze(0)

        block_mask_list.append(chunk_mask)

    return torch.cat(block_mask_list, dim=-2), torch.cat(attn_probs_list, dim=-2)

def create_block_mask_softmax_gemm(
    q, k,
    block_size=128,
    patch_size=8, # Can be 8, 16, 32...
    causal=True,
    cumulative_p_threshold=0.9,
    chunk_size=None,
    keep_sink=True,
    keep_local=True,
):
    """
    Create block sparse mask using double-band anti-diagonal pattern.

    Pattern for patch_size=8 (K=2, each row selects 2 consecutive K positions):
       0 1 2 3 4 5 6 7
      +----------------
    0 | . . . . . . 1 1
    1 | . . . . 1 1 . .
    2 | . . 1 1 . . . .
    3 | 1 1 . . . . . .
    4 | . . . . . . 1 1
    5 | . . . . 1 1 . .
    6 | . . 1 1 . . . .
    7 | 1 1 . . . . . .

    Row i selects columns: (patch_size - 2 - 2*i) % patch_size and (patch_size - 1 - 2*i) % patch_size
    """

    device = q.device
    B, H, Q_Len, D = q.shape
    _, _, K_Len, _ = k.shape


    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (K_Len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (K_Len - 1).bit_length()),
                ),
                2048,
            )
        )

    # Padding
    q_pad_len = ((Q_Len + block_size - 1) // block_size) * block_size
    if q_pad_len > Q_Len: q = F.pad(q, (0, 0, 0, q_pad_len - Q_Len))
    k_pad_len = ((K_Len + block_size - 1) // block_size) * block_size
    if k_pad_len > K_Len: k = F.pad(k, (0, 0, 0, k_pad_len - K_Len))

    num_chunks = (q_pad_len + chunk_size - 1) // chunk_size
    num_k_patches = k_pad_len // patch_size
    k_block_num = k_pad_len // block_size

    block_mask_list = []
    attn_probs_list = []

    for i in range(num_chunks):
        start_token = i * chunk_size
        end_token = min(start_token + chunk_size, q_pad_len)
        chunk_len = end_token - start_token
        num_q_patches = chunk_len // patch_size

        # =============================================================================
        # PHASE 1 & 2 & 3: Efficient Scoring
        # =============================================================================
        # Fused Kernel: Reshapes Q (sum halves) and K (sum bands + flip) and performs MatMul
        scores = dba_pattern_gemm(q, k, patch_size, start_token, num_q_patches, k_pad_len, is_causal=causal)

        # =============================================================================
        # PHASE 4: Block Aggregation
        # =============================================================================
        patches_per_block = block_size // patch_size
        chunk_start_patch = start_token // patch_size
        chunk_end_patch = end_token // patch_size
        real_q_patches = (Q_Len + patch_size - 1) // patch_size

        # segment_size should match patches_per_block to ensure we aggregate exactly one block's worth of K patches
        # into one output value.
        block_attn_probs = softmax_fuse_block_sum(
            scores,
            patches_per_block,  # reshaped_block_size (Aggregation factor for Q)
            patches_per_block,  # segment_size (Aggregation factor for K)
            chunk_start_patch,
            chunk_end_patch,
            real_q_patches,
            math.log2(math.e) / (math.sqrt(D) * patch_size * 2),
            is_causal=causal,
        )

        attn_probs_list.append(block_attn_probs)
        num_q_chunk_blocks = chunk_len // block_size
        # =============================================================================
        # PHASE 5: Selection Logic (Standard)
        # =============================================================================
        chunk_start_block = start_token // block_size
        q_blk_idx = torch.arange(num_q_chunk_blocks, device=device).view(-1, 1) + chunk_start_block
        k_blk_idx = torch.arange(k_block_num, device=device).view(1, -1)

        mandatory_mask = torch.zeros_like(block_attn_probs, dtype=torch.bool)
        if keep_sink: mandatory_mask[..., 0] = True
        if keep_local: mandatory_mask |= (q_blk_idx == k_blk_idx).unsqueeze(0).unsqueeze(0)

        sort_scores = block_attn_probs.clone()
        sort_scores.masked_fill_(mandatory_mask, 1000.0)

        _, sorted_indices = torch.sort(sort_scores, dim=-1, descending=True)
        sorted_probs = torch.gather(block_attn_probs, -1, sorted_indices)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        total_mass = block_attn_probs.sum(dim=-1, keepdim=True)
        if isinstance(cumulative_p_threshold, torch.Tensor):
            if cumulative_p_threshold.dim() == 1 and cumulative_p_threshold.size(0) == H:
                cumulative_p_threshold = cumulative_p_threshold.view(1, H, 1, 1)
        required_mass = total_mass * cumulative_p_threshold
        cumsum_without_self = torch.cat([torch.zeros_like(cumsum_probs[..., :1]), cumsum_probs[..., :-1]], dim=-1)
        keep_mask_sorted = cumsum_without_self < required_mass

        chunk_mask = torch.zeros_like(block_attn_probs, dtype=torch.bool)
        chunk_mask.scatter_(dim=-1, index=sorted_indices, src=keep_mask_sorted)
        chunk_mask |= mandatory_mask

        if causal:
            chunk_mask &= (q_blk_idx >= k_blk_idx).unsqueeze(0).unsqueeze(0)

        block_mask_list.append(chunk_mask)

    return torch.cat(block_mask_list, dim=-2), torch.cat(attn_probs_list, dim=-2)

def create_block_mask(
    q, k,
    block_size=128,
    patch_size=8, # Can be 8, 16, 32...
    causal=True,
    cumulative_p_threshold=0.9,
    chunk_size=None,
    keep_sink=True,
    keep_local=True,
):
    B, H, Q_Len, D = q.shape

    # A100:
    if "A100" in torch.cuda.get_device_properties(torch.cuda.current_device()).name:
        if causal:
            if patch_size == 8:
                if Q_Len <= 16384:
                    return create_block_mask_softmax_gemm(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
                else:
                    return create_block_mask_triton_only_softmax(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
            elif patch_size == 16:
                if Q_Len <= 32768:
                    return create_block_mask_softmax_gemm(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
                else:
                    return create_block_mask_triton_only_softmax(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
            else: # patch_size == 32
                if Q_Len <= 65536:
                    return create_block_mask_softmax_gemm(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
                else:
                    return create_block_mask_pytorch(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
        else: # causal == False
            if patch_size == 8:
                if Q_Len <= 4096:
                    return create_block_mask_softmax_gemm(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
                else:
                    return create_block_mask_triton_only_softmax(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
            elif patch_size == 16:
                if Q_Len <= 4096:
                    return create_block_mask_softmax_gemm(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
                else:
                    return create_block_mask_pytorch(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
            else: # patch_size == 32
                if Q_Len <= 8192:
                    return create_block_mask_softmax_gemm(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
                else:
                    return create_block_mask_pytorch(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
    # H20:
    elif "H20" in torch.cuda.get_device_properties(torch.cuda.current_device()).name:
        if causal:
            if patch_size == 32 and Q_Len >= 131072:
                return create_block_mask_pytorch(
                    q, k,
                    block_size=block_size,
                    patch_size=patch_size,
                    causal=causal,
                    cumulative_p_threshold=cumulative_p_threshold,
                    chunk_size=chunk_size,
                    keep_sink=keep_sink,
                    keep_local=keep_local,
                )
            else:
                return create_block_mask_softmax_gemm(
                    q, k,
                    block_size=block_size,
                    patch_size=patch_size,
                    causal=causal,
                    cumulative_p_threshold=cumulative_p_threshold,
                    chunk_size=chunk_size,
                    keep_sink=keep_sink,
                    keep_local=keep_local,
                )
        else: # causal == False
            if patch_size == 8:
                if Q_Len < 65536:
                    return create_block_mask_softmax_gemm(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
                else:
                    return create_block_mask_triton_only_softmax(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
            elif patch_size in [16, 32]:
                if Q_Len < 16384:
                    return create_block_mask_softmax_gemm(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
                else:
                    return create_block_mask_pytorch(
                        q, k,
                        block_size=block_size,
                        patch_size=patch_size,
                        causal=causal,
                        cumulative_p_threshold=cumulative_p_threshold,
                        chunk_size=chunk_size,
                        keep_sink=keep_sink,
                        keep_local=keep_local,
                    )
    else:
        return create_block_mask_softmax_gemm(
            q, k,
            block_size=block_size,
            patch_size=patch_size,
            causal=causal,
            cumulative_p_threshold=cumulative_p_threshold,
            chunk_size=chunk_size,
            keep_sink=keep_sink,
            keep_local=keep_local,
        )

#####################################################
def DBA_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_block_size=128,
    patch_size=8,
    causal=True,
    cumulative_p_threshold=0.9,
    chunk_size=None,
    keep_sink=True,
    keep_local=True,
):

    batch_size, num_heads, q_len, head_dim = q.shape
    _, _, k_len, _ = k.shape

    q_block_num = (q_len + sparse_block_size - 1) // sparse_block_size
    k_block_num = (k_len + sparse_block_size - 1) // sparse_block_size

    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )

    # ================================================
    # Create Block Sparse Mask
    block_sparse_mask, _ = create_block_mask(
        q, k,
        block_size=sparse_block_size,
        patch_size=patch_size,
        causal=causal,
        cumulative_p_threshold=cumulative_p_threshold,
        chunk_size=chunk_size,
        keep_sink=keep_sink,
        keep_local=keep_local,
    )

    # Ensure all tensors are on the same device
    k, v, block_sparse_mask = k.to(q.device), v.to(q.device), block_sparse_mask.to(q.device)

    # ================================================
    # Block Sparse Flash Attention
    assert sparse_block_size == 128
    assert batch_size == 1
    q = q.transpose(1, 2).view(q_len, num_heads, head_dim)
    k = k.transpose(1, 2).view(k_len, num_heads, head_dim)
    v = v.transpose(1, 2).view(k_len, num_heads, head_dim)

    # cumulative sequence lengths
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.tensor([0, k_len], dtype=torch.int32, device=k.device)

    out = block_sparse_flash_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        q_len,
        k_len,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=True,
        return_attn_probs=False,
        block_sparse_mask=block_sparse_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        sparse_mask_row_dim=sparse_block_size,
        sparse_mask_col_dim=sparse_block_size,
    )

    out = out.view(batch_size, q_len, num_heads, head_dim).transpose(1, 2)

    del q, block_sparse_mask
    return out
