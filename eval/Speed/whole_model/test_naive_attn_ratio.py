import os
import time
import math
import gc
import torch
import types
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, LlamaForCausalLM

from DBA.src.dual_band_attention import DBA_forward
from DBA.src.Fullprefill import Full_prefill
from DBA.threshold.llama_threshold import llama_fuse_16, llama_fuse_8

import warnings
import logging
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", message=".*TORCH_CUDA_ARCH_LIST.*")
logging.getLogger("flashinfer.jit").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

global_attn_start_events = []
global_attn_end_events = []

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def naive_attention_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, q_chunk_size: int = 64) -> torch.Tensor:
    """Pure PyTorch dense attention (no flash/sparse kernels), computed in Q chunks to reduce peak memory."""
    _, _, q_len, head_dim = q.shape
    k_len = k.shape[-2]
    scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    k_t = k.transpose(-2, -1).contiguous()
    k_idx = torch.arange(k_len, device=q.device).view(1, 1, 1, k_len) if causal else None

    for start in range(0, q_len, q_chunk_size):
        end = min(start + q_chunk_size, q_len)
        q_chunk = q[:, :, start:end, :]
        scores = torch.matmul(q_chunk, k_t) * scale

        if causal:
            q_idx = torch.arange(start, end, device=q.device).view(1, 1, -1, 1)
            scores.masked_fill_(k_idx > q_idx, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        out[:, :, start:end, :] = torch.matmul(probs, v)
        del q_chunk, scores, probs

    return out

def naive_attention_prefill_sdpa_math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, q_chunk_size: int = 64) -> torch.Tensor:
    """PyTorch native math-SDPA implementation with Q-only chunking (no K chunking)."""
    _, _, q_len, _ = q.shape
    k_len = k.shape[-2]

    out = torch.empty_like(q)
    k_idx = torch.arange(k_len, device=q.device).view(1, 1, 1, k_len) if causal else None

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        for start in range(0, q_len, q_chunk_size):
            end = min(start + q_chunk_size, q_len)
            q_chunk = q[:, :, start:end, :]

            attn_mask = None
            if causal:
                q_idx = torch.arange(start, end, device=q.device).view(1, 1, -1, 1)
                attn_mask = q_idx >= k_idx

            out[:, :, start:end, :] = F.scaled_dot_product_attention(
                q_chunk,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            del q_chunk, attn_mask

    return out

def generate_prompt(tokenizer, target_len):
    context = "A quick brown fox jumps over the lazy dog. What the dog doing?\n"
    needle = "This is a special needle that we need to find."
    if os.path.exists("needle.txt"):
        with open("needle.txt", "r") as f:
           needle = f.read()

    num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
    num_tokens_needle = len(tokenizer.encode(needle, add_special_tokens=False))

    prefix = "This is a very long story book with knowledge of Flash Attention 4, which you need to remember for later question: <book> "
    suffix = "</book>\n Based on the content of the book, please briefly tell me about Flash Attention 4.\nAnswer:"
    static_tokens = len(tokenizer.encode(prefix, add_special_tokens=True)) + len(tokenizer.encode(suffix, add_special_tokens=False))

    available_context_tokens = target_len - static_tokens - num_tokens_needle
    num_repetitions = 0 if available_context_tokens < 0 else available_context_tokens // num_tokens_context

    text = (
        prefix
        + context * (num_repetitions // 2)
        + needle
        + context * (num_repetitions - num_repetitions // 2)
        + suffix
    )

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    if input_ids.shape[1] > target_len:
        input_ids = input_ids[:, :target_len]
    else:
        pad_len = target_len - input_ids.shape[1]
        input_ids = torch.cat([input_ids, torch.zeros((1, pad_len), dtype=torch.long, device="cuda")], dim=1)
    return input_ids

def make_custom_forward(method_name, threshold_val, target_len, naive_q_chunk_size, naive_impl):
    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        global global_attn_start_events, global_attn_end_events
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        del hidden_states

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=key_states.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, kv_seq_len)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Avoid extra full-tensor copies from .contiguous() to reduce peak memory.
        q = query_states
        k = key_states
        v = value_states

        del query_states, key_states, value_states, cos, sin

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        if method_name == 'Naive':
            if naive_impl == 'sdpa_math':
                attn_output = naive_attention_prefill_sdpa_math(q, k, v, causal=True, q_chunk_size=naive_q_chunk_size)
            elif naive_impl == 'matmul':
                attn_output = naive_attention_prefill(q, k, v, causal=True, q_chunk_size=naive_q_chunk_size)
            else:
                raise ValueError(f"Unknown naive_impl {naive_impl}")
            del q, k, v
        elif method_name == 'Full':
            attn_output = Full_prefill(q, k, v, causal=True)
            del q, k, v
        elif method_name == 'DBA_16':
            attn_output = DBA_forward(q, k, v, sparse_block_size=128, patch_size=16, causal=True, cumulative_p_threshold=threshold_val, chunk_size=min(32768, target_len), keep_sink=True, keep_local=True)
            del q, k, v
        else:
            raise ValueError(f"Unknown method {method_name}")

        end_event.record()
        global_attn_start_events.append(start_event)
        global_attn_end_events.append(end_event)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, None
    return custom_forward

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "../../../../Models/Llama_3_1_8B_Instruct/"))

    # Benchmark settings (fixed in code, no environment variables required)
    # seq_lens = [32, 64, 128]
    seq_lens = [128]
    naive_impl = 'sdpa_math'  # options: 'sdpa_math', 'matmul'
    naive_q_chunk_size = 3072
    run_baselines = False

    print("Loading tokenizer and model for naive attention ratio testing...")
    print(f"Sequence lengths (K): {seq_lens}")
    print(f"Naive implementation: {naive_impl}")
    print(f"Naive Q chunk size: {naive_q_chunk_size}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Detected {num_layers} attention layers in the model.")

    methods_cfg = [
        ('Naive Attention', 'Naive', None),
    ]
    if run_baselines:
        threshold_8_all = torch.tensor(llama_fuse_8, device="cuda")
        methods_cfg.extend([
            ('Full Attention', 'Full', None),
            ('DBA 16', 'DBA_16', threshold_8_all),
        ])

    all_results = []

    # Store original forward methods
    original_forwards = [model.model.layers[i].self_attn.forward for i in range(num_layers)]

    for seq_len in seq_lens:
        target_len = seq_len * 1024
        input_ids = generate_prompt(tokenizer, target_len)
        seq_len_str = f"{seq_len}K"
        print(f"\n--- Running benchmark array for sequence length: {seq_len_str} ---")

        layer_results = {"Length": seq_len_str}

        for display_name, method_name, thresholds in methods_cfg:
            print(f"Benchmarking {display_name} ...")

            # Monkey-patch layers
            for i in range(num_layers):
                t_val = thresholds[i % len(thresholds)] if thresholds is not None else None
                new_fwd = make_custom_forward(method_name, t_val, target_len, naive_q_chunk_size, naive_impl)
                model.model.layers[i].self_attn.forward = types.MethodType(new_fwd, model.model.layers[i].self_attn)

            torch.cuda.empty_cache()

            # For memory-focused reproduction, run exactly one measured forward pass.
            num_warmups = 0
            num_iterations = 1

            method_time = float('inf')
            avg_attn_time_ms = float('inf')
            global global_attn_start_events, global_attn_end_events
            with torch.inference_mode():
                try:
                    for _ in range(num_warmups):
                        global_attn_start_events.clear()
                        global_attn_end_events.clear()
                        out = model(input_ids, use_cache=False)
                        del out
                    torch.cuda.synchronize()
                    t_start = time.time()
                    total_attn_time_ms = 0.0
                    for _ in range(num_iterations):
                        global_attn_start_events.clear()
                        global_attn_end_events.clear()
                        out = model(input_ids, use_cache=False)
                        del out
                        torch.cuda.synchronize()
                        attn_time_ms = sum([s.elapsed_time(e) for s, e in zip(global_attn_start_events, global_attn_end_events)])
                        total_attn_time_ms += attn_time_ms
                    method_time = (time.time() - t_start) / num_iterations
                    avg_attn_time_ms = total_attn_time_ms / num_iterations
                except Exception as e:
                    print(f"Exception during {display_name}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            # Restore original forwards just in case
            for i in range(num_layers):
                model.model.layers[i].self_attn.forward = original_forwards[i]

            layer_results[f"{display_name} TTFT (ms)"] = method_time * 1000 if method_time != float('inf') else float('nan')
            layer_results[f"{display_name} Attn Total (ms)"] = avg_attn_time_ms if avg_attn_time_ms != float('inf') else float('nan')
            layer_results[f"{display_name} Attn/TTFT (%)"] = (avg_attn_time_ms / (method_time * 1000) * 100) if method_time != float('inf') and method_time > 0 else float('nan')
            if method_time != float('inf'):
                print(f"  TTFT: {method_time * 1000:.2f} ms")
                print(f"  Attn Total: {avg_attn_time_ms:.2f} ms")
                print(f"  Attn Ratio: {layer_results[f'{display_name} Attn/TTFT (%)']:.2f}%")

        # Calculate speedups when a baseline is present
        if "Full Attention TTFT (ms)" in layer_results:
            base_time = layer_results["Full Attention TTFT (ms)"]
            for display_name, _, _ in methods_cfg:
                if display_name == 'Full Attention':
                    continue
                t_ms = layer_results[f"{display_name} TTFT (ms)"]
                layer_results[f"{display_name} SpdUp"] = base_time / t_ms if t_ms == t_ms and t_ms > 0 else 0.0

        all_results.append(layer_results)

    df = pd.DataFrame(all_results)
    save_dir = "benchmark_naive_attn_ratio_result"
    os.makedirs(save_dir, exist_ok=True)

    df.to_excel(os.path.join(save_dir, "attn_ratio.xlsx"), index=False)

    # Plot
    num_lengths = len(df['Length'])
    num_methods = len(methods_cfg)
    fig, axes = plt.subplots(nrows=num_lengths, ncols=num_methods, figsize=(6 * num_methods, 5 * num_lengths))

    # Ensure axes is 2D array for easy indexing
    if num_lengths == 1 and num_methods == 1:
        axes = np.array([[axes]])
    elif num_lengths == 1:
        axes = np.array([axes])
    elif num_methods == 1:
        axes = np.array([[ax] for ax in axes])

    for i, row in df.iterrows():
        length_str = row['Length']
        for j, (display_name, _, _) in enumerate(methods_cfg):
            ax = axes[i, j]

            t_total = row.get(f"{display_name} TTFT (ms)")
            t_attn = row.get(f"{display_name} Attn Total (ms)")

            if pd.isna(t_total) or pd.isna(t_attn) or t_total <= 0:
                ax.axis('off')
                continue

            t_other = max(0, t_total - t_attn)

            ax.pie([t_attn, t_other], labels=['Attention (Prefill)', 'Other Ops'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
            ax.set_title(f"Length {length_str} - {display_name}\nTotal Forward Time: {t_total:.1f} ms")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "attn_ratio_pie_chart.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved attention ratio results to '{save_dir}'.")

if __name__ == "__main__":
    main()
