import os
import time
import torch
import types

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, LlamaForCausalLM

from DBA.src.dual_band_attention import DBA_forward, create_block_mask
from DBA.src.Fullprefill import Full_prefill
from DBA.src.Xattention import Xattention_prefill, xattn_estimate
from DBA.threshold.llama_threshold import llama_fuse_16, llama_fuse_8

import warnings
import logging
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.getLogger("flashinfer.jit").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

SPARSE_METHODS = {"XAttn_8", "XAttn_16", "DBA_8", "DBA_16"}


def update_mask_stats(mask_stats, method_name, layer_index, block_mask, q_block_num, k_block_num):
    mask = block_mask[:, :, :q_block_num, :k_block_num].to(torch.bool)

    q_idx = torch.arange(q_block_num, device=mask.device).unsqueeze(1)
    k_idx = torch.arange(k_block_num, device=mask.device).unsqueeze(0)
    causal_offset = max(k_block_num - q_block_num, 0)
    causal_valid = (q_idx + causal_offset) >= k_idx

    valid = causal_valid.unsqueeze(0).unsqueeze(0)
    kept = (mask & valid).sum().item()
    total = (valid.sum() * mask.shape[0] * mask.shape[1]).item()

    stats = mask_stats.setdefault(method_name, {"kept": 0.0, "total": 0.0, "layers": {}})
    stats["kept"] += kept
    stats["total"] += total

    layer_key = -1 if layer_index is None else int(layer_index)
    layer_stats = stats["layers"].setdefault(layer_key, {"kept": 0.0, "total": 0.0})
    layer_stats["kept"] += kept
    layer_stats["total"] += total

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

def make_custom_forward(method_name, threshold_val, target_len, collect_mask=False, mask_stats=None, layer_index=None):
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
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, kv_seq_len)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        q = query_states.contiguous()
        k = key_states.contiguous()
        v = value_states.contiguous()

        del query_states, key_states, value_states, cos, sin

        chunk_size = min(32768, target_len)

        if method_name == 'Full':
            attn_output = Full_prefill(q, k, v, causal=True)
            del q, k, v
        elif method_name == 'XAttn_8':
            if collect_mask and mask_stats is not None:
                q_block_num = (q.shape[-2] + 127) // 128
                k_block_num = (k.shape[-2] + 127) // 128
                _, block_mask = xattn_estimate(
                    q,
                    k,
                    block_size=128,
                    stride=8,
                    norm=1,
                    threshold=threshold_val,
                    use_triton=True,
                    causal=True,
                    chunk_size=chunk_size,
                    keep_sink=True,
                    keep_recent=True,
                )
                update_mask_stats(mask_stats, method_name, layer_index, block_mask, q_block_num, k_block_num)
                del block_mask
            attn_output = Xattention_prefill(q, k, v, stride=8, threshold=threshold_val, use_triton=True, chunk_size=chunk_size, keep_sink=True, keep_recent=True)
            del q, k, v
        elif method_name == 'XAttn_16':
            if collect_mask and mask_stats is not None:
                q_block_num = (q.shape[-2] + 127) // 128
                k_block_num = (k.shape[-2] + 127) // 128
                _, block_mask = xattn_estimate(
                    q,
                    k,
                    block_size=128,
                    stride=16,
                    norm=1,
                    threshold=threshold_val,
                    use_triton=True,
                    causal=True,
                    chunk_size=chunk_size,
                    keep_sink=True,
                    keep_recent=True,
                )
                update_mask_stats(mask_stats, method_name, layer_index, block_mask, q_block_num, k_block_num)
                del block_mask
            attn_output = Xattention_prefill(q, k, v, stride=16, threshold=threshold_val, use_triton=True, chunk_size=chunk_size, keep_sink=True, keep_recent=True)
            del q, k, v
        elif method_name == 'DBA_8':
            if collect_mask and mask_stats is not None:
                q_block_num = (q.shape[-2] + 127) // 128
                k_block_num = (k.shape[-2] + 127) // 128
                block_mask, _ = create_block_mask(
                    q,
                    k,
                    block_size=128,
                    patch_size=8,
                    causal=True,
                    cumulative_p_threshold=threshold_val,
                    chunk_size=chunk_size,
                    keep_sink=True,
                    keep_local=True,
                )
                update_mask_stats(mask_stats, method_name, layer_index, block_mask, q_block_num, k_block_num)
                del block_mask
            attn_output = DBA_forward(q, k, v, sparse_block_size=128, patch_size=8, causal=True, cumulative_p_threshold=threshold_val, chunk_size=chunk_size, keep_sink=True, keep_local=True)
            del q, k, v
        elif method_name == 'DBA_16':
            if collect_mask and mask_stats is not None:
                q_block_num = (q.shape[-2] + 127) // 128
                k_block_num = (k.shape[-2] + 127) // 128
                block_mask, _ = create_block_mask(
                    q,
                    k,
                    block_size=128,
                    patch_size=16,
                    causal=True,
                    cumulative_p_threshold=threshold_val,
                    chunk_size=chunk_size,
                    keep_sink=True,
                    keep_local=True,
                )
                update_mask_stats(mask_stats, method_name, layer_index, block_mask, q_block_num, k_block_num)
                del block_mask
            attn_output = DBA_forward(q, k, v, sparse_block_size=128, patch_size=16, causal=True, cumulative_p_threshold=threshold_val, chunk_size=chunk_size, keep_sink=True, keep_local=True)
            del q, k, v
        else:
            raise ValueError(f"Unknown method {method_name}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, None
    return custom_forward

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "../../../../Models/Llama_3_1_8B_Instruct/"))
    seq_lens = [4, 8, 16, 32, 64, 128]

    print("Loading tokenizer and model for whole model TTFT benchmarking...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Detected {num_layers} attention layers in the model.")

    threshold_8_all = torch.tensor(llama_fuse_8, device="cuda")
    threshold_16_all = torch.tensor(llama_fuse_16, device="cuda")

    print("Using RULER/LongBench threshold lists shared by XAttention and DBA (8->llama_fuse_8, 16->llama_fuse_16).")

    methods_cfg = [
        ('Full Attention', 'Full', None),
        ('XAttn 8', 'XAttn_8', threshold_8_all),
        ('XAttn 16', 'XAttn_16', threshold_16_all),
        ('DBA 8', 'DBA_8', threshold_8_all),
        ('DBA 16', 'DBA_16', threshold_16_all)
    ]

    all_results = []
    layer_detail_rows = []

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
                new_fwd = make_custom_forward(method_name, t_val, target_len, collect_mask=False, mask_stats=None, layer_index=i)
                model.model.layers[i].self_attn.forward = types.MethodType(new_fwd, model.model.layers[i].self_attn)

            torch.cuda.empty_cache()

            num_warmups = 2
            num_iterations = 2

            method_time = float('inf')
            with torch.no_grad():
                try:
                    for _ in range(num_warmups):
                        out = model(input_ids, use_cache=False)
                        del out
                    torch.cuda.synchronize()
                    t_start = time.time()
                    for _ in range(num_iterations):
                        out = model(input_ids, use_cache=False)
                        del out
                    torch.cuda.synchronize()
                    method_time = (time.time() - t_start) / num_iterations
                except Exception as e:
                    print(f"Exception during {display_name}: {e}")

            # Restore original forwards just in case
            for i in range(num_layers):
                model.model.layers[i].self_attn.forward = original_forwards[i]

            layer_results[f"{display_name} TTFT (ms)"] = method_time * 1000 if method_time != float('inf') else float('nan')

            mask_keep = float('nan')
            mask_sparsity = float('nan')
            layer_mean_sparsity = float('nan')
            if method_name in SPARSE_METHODS:
                method_mask_stats = {}

                for i in range(num_layers):
                    t_val = thresholds[i % len(thresholds)] if thresholds is not None else None
                    new_fwd = make_custom_forward(method_name, t_val, target_len, collect_mask=True, mask_stats=method_mask_stats, layer_index=i)
                    model.model.layers[i].self_attn.forward = types.MethodType(new_fwd, model.model.layers[i].self_attn)

                with torch.no_grad():
                    try:
                        out = model(input_ids, use_cache=False)
                        del out
                        torch.cuda.synchronize()
                    except Exception as e:
                        print(f"Exception during {display_name} mask collection: {e}")

                for i in range(num_layers):
                    model.model.layers[i].self_attn.forward = original_forwards[i]

                stats = method_mask_stats.get(method_name)
                if stats is not None:
                    if stats["total"] > 0:
                        mask_keep = stats["kept"] / stats["total"]
                        mask_sparsity = 1.0 - mask_keep
                        print(f"  Mask sparsity (causal-valid blocks): {mask_sparsity * 100:.2f}%")

                    layer_sparsity_vals = []
                    for layer_idx in sorted(stats["layers"].keys()):
                        layer_stats = stats["layers"][layer_idx]
                        if layer_stats["total"] <= 0:
                            continue

                        layer_keep = layer_stats["kept"] / layer_stats["total"]
                        layer_sparsity = 1.0 - layer_keep
                        layer_sparsity_vals.append(layer_sparsity)

                        layer_detail_rows.append({
                            "Length": seq_len_str,
                            "Method": display_name,
                            "Method Key": method_name,
                            "Layer": int(layer_idx),
                            "Mask Keep (%)": layer_keep * 100,
                            "Mask Sparsity (%)": layer_sparsity * 100,
                        })

                    if len(layer_sparsity_vals) > 0:
                        layer_mean_sparsity = float(np.mean(layer_sparsity_vals))
                        print(f"  Layer-mean mask sparsity: {layer_mean_sparsity * 100:.2f}%")

            layer_results[f"{display_name} Mask Keep (%)"] = mask_keep * 100 if mask_keep == mask_keep else float('nan')
            layer_results[f"{display_name} Mask Sparsity (%)"] = mask_sparsity * 100 if mask_sparsity == mask_sparsity else float('nan')
            layer_results[f"{display_name} Layer Mean Sparsity (%)"] = layer_mean_sparsity * 100 if layer_mean_sparsity == layer_mean_sparsity else float('nan')

        # Calculate speedups
        base_time = layer_results["Full Attention TTFT (ms)"]
        for display_name, _, _ in methods_cfg:
            if display_name == 'Full Attention': continue
            t_ms = layer_results[f"{display_name} TTFT (ms)"]
            layer_results[f"{display_name} SpdUp"] = base_time / t_ms if t_ms == t_ms and t_ms > 0 else 0.0

        for stride in [8, 16]:
            x_col = f"XAttn {stride} Layer Mean Sparsity (%)"
            d_col = f"DBA {stride} Layer Mean Sparsity (%)"
            gap_col = f"DBA-XAttn {stride} Layer Mean Sparsity Gap (%)"
            x_val = layer_results.get(x_col, float('nan'))
            d_val = layer_results.get(d_col, float('nan'))
            layer_results[gap_col] = (d_val - x_val) if x_val == x_val and d_val == d_val else float('nan')

        all_results.append(layer_results)

    df = pd.DataFrame(all_results)
    save_dir = "benchmark_whole_model_speedup_result"
    os.makedirs(save_dir, exist_ok=True)

    df.to_excel(os.path.join(save_dir, "whole_model_speedup.xlsx"), index=False)

    if len(layer_detail_rows) > 0:
        layer_df = pd.DataFrame(layer_detail_rows)
        layer_df = layer_df.sort_values(by=["Length", "Method", "Layer"]).reset_index(drop=True)
        layer_df.to_excel(os.path.join(save_dir, "whole_model_layer_sparsity_detail.xlsx"), index=False)

    reviewer_cols = [
        "Length",
        "XAttn 8 Layer Mean Sparsity (%)",
        "DBA 8 Layer Mean Sparsity (%)",
        "DBA-XAttn 8 Layer Mean Sparsity Gap (%)",
        "XAttn 16 Layer Mean Sparsity (%)",
        "DBA 16 Layer Mean Sparsity (%)",
        "DBA-XAttn 16 Layer Mean Sparsity Gap (%)",
    ]
    reviewer_cols = [c for c in reviewer_cols if c in df.columns]
    if len(reviewer_cols) > 1:
        df[reviewer_cols].to_excel(os.path.join(save_dir, "whole_model_layer_mean_sparsity_comparison.xlsx"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(df['Length']))
    methods_plot = [m[0] for m in methods_cfg if m[0] != 'Full Attention']
    width = 0.14

    for i, m in enumerate(methods_plot):
        spd_col = f"{m} SpdUp"
        t_col = f"{m} TTFT (ms)"
        offsets = x + (i - len(methods_plot)/2 + 0.5) * width
        bars = ax.bar(offsets, df[spd_col].fillna(0), width, label=m)

        for bar, spd, t in zip(bars, df[spd_col], df[t_col]):
            if pd.notna(spd) and spd > 0.01:
                label_str = f"{spd:.2f}x\n{t:.1f}ms"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        label_str, ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Speedup over Full Attention')
    ax.set_title('Whole Model TTFT Prefill Benchmark (Speedup & Time)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Length'])
    ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Full Attention (1.0x)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "whole_model_speedup_plot.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    sparse_display_names = [name for name, method, _ in methods_cfg if method in SPARSE_METHODS]
    if len(sparse_display_names) > 0:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for m in sparse_display_names:
            col = f"{m} Mask Sparsity (%)"
            if col in df.columns:
                ax2.plot(df['Length'], df[col], marker='o', linewidth=2, label=m)

        ax2.set_ylabel('Mask Sparsity (%)')
        ax2.set_xlabel('Sequence Length')
        ax2.set_title('Whole Model Mask Sparsity (Causal-Valid Block Space)')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        ax2.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "whole_model_mask_sparsity_plot.png"), dpi=300, bbox_inches='tight')
        plt.close(fig2)

    print(f"\nSaved whole model results to '{save_dir}'.")

if __name__ == "__main__":
    main()
