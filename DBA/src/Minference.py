import torch
import math

from minference import vertical_slash_sparse_attention

def sum_all_diagonal_matrix(mat: torch.Tensor):
    """Helper function to calculate the sum of attention scores along diagonals."""
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device) # Zero matrix used for padding
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1) # pads the matrix on left and right
    mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)) # Change the strides
    sum_diags = torch.sum(mat_strided, 2) # Sums the resulting matrix's columns
    return sum_diags[:, :, 1:]

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Helper function to expand key/value states for Grouped Query Attention (GQA)."""
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)

def Minference_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    vertical_size: int = 512,
    slash_size: int = 2048,
):
    """
    Standalone MInference prefill specifically for Llama 3.1 8B Instruct.
    Calculates sparse attention strictly using the "vertical_and_slash" pattern.

    Args:
        query_states: [bsz, num_heads, q_len, head_dim]
        key_states: [bsz, num_kv_heads, q_len, head_dim]
        value_states: [bsz, num_kv_heads, q_len, head_dim]
        vertical_size: default 512 (from Llama 3.1 8B Instruct 128k config)
        slash_size: default 2048 (from Llama 3.1 8B Instruct 128k config)
    """

    bsz, num_heads, q_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]

    # 1. Handle GQA by repeating KV heads to match Q heads
    if num_heads != num_kv_heads:
        num_key_value_groups = num_heads // num_kv_heads
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

    output = torch.empty_like(query_states)
    # 2. Setup the causal mask for the last_q tokens used during pattern estimation
    last_q = min(64, q_len)
    arange = torch.arange(last_q, device=query_states.device)
    LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]

    # Bound the sizes to prevent going out of bounds for short prompts
    vertical_size = min(q_len, max(vertical_size, 30))
    slash_size = min(q_len, max(slash_size, 50))

    # 3. Iterate over each attention head to calculate sparse topologies
    for head in range(num_heads):
        # Extract individual head [bsz, 1, q_len, head_dim]
        q = query_states[:, head:head+1, :, :]
        k = key_states[:, head:head+1, :, :]
        v = value_states[:, head:head+1, :, :]

        # Calculate attention logits for only the last `last_q` queries
        qk = torch.einsum('bhmk, bhnk -> bhmn', q[:, :, -last_q:, :], k) / math.sqrt(head_dim)

        # Apply causal masking to the tail tokens
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK, qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)

        # --- Extract Vertical Patterns ---
        # Sum attention over queries to find globally important keys
        vertical = qk.sum(-2, keepdim=True)
        vertical[..., :30] = torch.inf # Force keeping the first 30 tokens (System prompt/sink tokens)
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        # --- Extract Slash (Local) Patterns ---
        # Sum diagonals to find important local context bounds
        slash = sum_all_diagonal_matrix(qk)[..., :-last_q + 1]
        slash[..., -100:] = torch.inf # Force keeping the most recent 100 tokens locally
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        # 4. Dispatch to the underlying custom Triton Kernel
        attn_output = vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
        output[:, head:head+1, :, :] = attn_output

    return output