import os
import time
import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm

from DBA.src.dual_band_attention import DBA_forward
from DBA.src.Fullprefill import Full_prefill
from DBA.src.Minference import Minference_prefill
from DBA.src.Flexprefill import flex_prefill_attention
from DBA.src.Xattention import Xattention_prefill
from DBA.threshold.llama_threshold import llama_fuse_16, llama_fuse_8

import warnings
import logging
logging.getLogger("flashinfer.jit").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

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

def get_layer_hook(layer_idx, seq_len_str, target_len, threshold_8_val, threshold_16_val, results_list, pbar):
    def hook(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
        position_embeddings = kwargs.get("position_embeddings")

        bsz, q_len, _ = hidden_states.size()

        query_states = module.q_proj(hidden_states)
        key_states = module.k_proj(hidden_states)
        value_states = module.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, module.num_heads, module.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, module.num_key_value_heads, module.head_dim).transpose(1, 2)

        if position_embeddings is None:
            position_ids = kwargs.get("position_ids")
            cos, sin = module.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, module.num_key_value_groups)
        value_states = repeat_kv(value_states, module.num_key_value_groups)

        q = query_states.detach().clone().contiguous()
        k = key_states.detach().clone().contiguous()
        v = value_states.detach().clone().contiguous()

        num_warmups = 3
        num_iterations = 5
        gamma, tau = 0.95, 0.1

        def timed_run(func, *args_f, **kwargs_f):
            try:
                for _ in range(num_warmups): func(*args_f, **kwargs_f)
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(num_iterations): func(*args_f, **kwargs_f)
                end_event.record()
                torch.cuda.synchronize()
                return (start_event.elapsed_time(end_event) / 1000.0) / num_iterations
            except Exception as e:
                return float('inf')

        avg_full = timed_run(Full_prefill, q, k, v, causal=True)
        avg_minfer = timed_run(Minference_prefill, q, k, v, vertical_size=512, slash_size=2048)
        qt, kt, vt = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
        avg_flex = timed_run(flex_prefill_attention, qt, kt, vt, gamma, tau)
        del qt, kt, vt

        # Set both to threshold 8 to guarantee similar sparsity for speed testing.
        avg_xattn_8 = timed_run(Xattention_prefill, q, k, v, stride=8, threshold=threshold_8_val, use_triton=True, chunk_size=min(32768, target_len), keep_sink=True, keep_recent=True)
        avg_xattn_16 = timed_run(Xattention_prefill, q, k, v, stride=16, threshold=threshold_8_val, use_triton=True, chunk_size=min(32768, target_len), keep_sink=True, keep_recent=True)

        avg_dba_8 = timed_run(DBA_forward, q, k, v, sparse_block_size=128, patch_size=8, causal=True, cumulative_p_threshold=threshold_8_val, chunk_size=min(32768, target_len), keep_sink=True, keep_local=True)
        avg_dba_16 = timed_run(DBA_forward, q, k, v, sparse_block_size=128, patch_size=16, causal=True, cumulative_p_threshold=threshold_8_val, chunk_size=min(32768, target_len), keep_sink=True, keep_local=True)

        del q, k, v
        torch.cuda.empty_cache()

        def to_ms(x): return x * 1000 if x != float('inf') else float('nan')
        def to_spdup(x): return avg_full / x if x != float('inf') and x > 0 else 0.0

        results_list.append({
            "Layer": layer_idx,
            "Length": seq_len_str,
            "Full Attention Time (ms)": to_ms(avg_full),
            "MInference Time (ms)": to_ms(avg_minfer),
            "MInf SpdUp": to_spdup(avg_minfer),
            "FlexPrefill Time (ms)": to_ms(avg_flex),
            "Flex SpdUp": to_spdup(avg_flex),
            "XAttn 8 Time (ms)": to_ms(avg_xattn_8),
            "XAttn 8 SpdUp": to_spdup(avg_xattn_8),
            "XAttn 16 Time (ms)": to_ms(avg_xattn_16),
            "XAttn 16 SpdUp": to_spdup(avg_xattn_16),
            "DBA 8 Time (ms)": to_ms(avg_dba_8),
            "DBA 8 SpdUp": to_spdup(avg_dba_8),
            "DBA 16 Time (ms)": to_ms(avg_dba_16),
            "DBA 16 SpdUp": to_spdup(avg_dba_16)
        })

        pbar.update(1)
        return None # Do not interrupt forward pass!
    return hook

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "../../../../Models/Llama_3_1_8B_Instruct/"))
    seq_lens = [4, 8, 16, 32, 64, 128]

    print("Loading tokenizer and model for ALL LAYERS benchmarking...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Detected {num_layers} attention layers in the model.")

    threshold_8_all = torch.tensor(llama_fuse_8, device="cuda")
    threshold_16_all = torch.tensor(llama_fuse_16, device="cuda")

    all_results = []

    for seq_len in seq_lens:
        target_len = seq_len * 1024
        input_ids = generate_prompt(tokenizer, target_len)
        seq_len_str = f"{seq_len}K"

        print(f"\n--- Running benchmark array for sequence length: {seq_len_str} ---")
        pbar = tqdm(total=num_layers, desc=f"Hooking layers")
        handles = []

        # Attach hooks to ALL layers at once for this given sequence length
        for i in range(num_layers):
            t8_val = threshold_8_all[i % len(threshold_8_all)]
            t16_val = threshold_16_all[i % len(threshold_16_all)]

            h = model.model.layers[i].self_attn.register_forward_pre_hook(
                get_layer_hook(i, seq_len_str, target_len, t8_val, t16_val, all_results, pbar),
                with_kwargs=True
            )
            handles.append(h)

        # Trigger single forward pass (hooks will sequentially catch QKV, benchmark, then proceed cleanly)
        with torch.no_grad():
            try:
                model(input_ids)
            except Exception as e:
                print(f"Forward pass exception during {seq_len_str}: {e}")

        for h in handles:
            h.remove()

        pbar.close()

    # --------------- SAVE & PLOT RESULTS ---------------
    df = pd.DataFrame(all_results)
    save_dir = "benchmark_each_layer_speedup_result"
    os.makedirs(save_dir, exist_ok=True)

    methods = ['MInf', 'Flex', 'XAttn 8', 'XAttn 16', 'DBA 8', 'DBA 16']
    spdup_cols = ['MInf SpdUp', 'Flex SpdUp', 'XAttn 8 SpdUp', 'XAttn 16 SpdUp', 'DBA 8 SpdUp', 'DBA 16 SpdUp']
    time_cols = ['MInference Time (ms)', 'FlexPrefill Time (ms)', 'XAttn 8 Time (ms)', 'XAttn 16 Time (ms)', 'DBA 8 Time (ms)', 'DBA 16 Time (ms)']

    # 1. Output table and plot files separated for each layer
    print("Generating separated plots and tables for each layer...")
    for layer_idx in sorted(df['Layer'].unique()):
        layer_df = df[df['Layer'] == layer_idx]
        # Excel table without 'Layer' column
        out_excel = os.path.join(save_dir, f"benchmark_layer_{layer_idx}.xlsx")
        layer_df.drop(columns=['Layer']).to_excel(out_excel, index=False)

        # Plot for this layer
        fig, ax = plt.subplots(figsize=(16, 8))
        x = np.arange(len(layer_df['Length']))
        width = 0.14

        for i, (m, spd_col, t_col) in enumerate(zip(methods, spdup_cols, time_cols)):
            offsets = x + (i - len(methods)/2 + 0.5) * width
            bars = ax.bar(offsets, layer_df[spd_col].fillna(0), width, label=m)

            for bar, spd, t in zip(bars, layer_df[spd_col], layer_df[t_col]):
                if pd.notna(spd) and spd > 0.01:
                    label_str = f"{spd:.2f}x\n{t:.1f}ms"
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                            label_str, ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Speedup over Flash Attention')
        ax.set_title(f'Attention Prefill Benchmark (Speedup & Time) - Layer {layer_idx}')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_df['Length'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.0))
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Flash Attention (1.0x)')

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"benchmark_layer_{layer_idx}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 2. Combined Table
    df.to_excel(os.path.join(save_dir, "benchmark_each_layer_table_combined.xlsx"), index=False)
    print(f"\nSaved combined and separated layer-wise Excel results to folder '{save_dir}'.")

    # 3. Generate comprehensive plot containing all 32 layers
    print(f"Generating combined plot for all layers...")
    num_seq_lens = len(seq_lens)

    # We use a massive horizontal figure to nicely fit layers * 6 grouped bars
    fig, axes = plt.subplots(num_seq_lens, 1, figsize=(50, 8 * num_seq_lens))
    if num_seq_lens == 1:
        axes = [axes]

    for ax, seq_len in zip(axes, seq_lens):
        seq_str = f"{seq_len}K"
        subset = df[df['Length'] == seq_str].sort_values('Layer')

        x = np.arange(len(subset['Layer']))
        width = 0.14

        for i, (m, spd_col, t_col) in enumerate(zip(methods, spdup_cols, time_cols)):
            offsets = x + (i - len(methods)/2 + 0.5) * width
            bars = ax.bar(offsets, subset[spd_col].fillna(0), width, label=m)

            for bar, spd, t in zip(bars, subset[spd_col], subset[t_col]):
                if pd.notna(spd) and spd > 0.01:
                    label_str = f"{spd:.2f}x\n{t:.1f}"
                    # Add tiny offset on top of bar, writing text horizontally
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            label_str, ha='center', va='bottom', fontsize=8, rotation=0)

        ax.set_ylabel('Speedup over Flash Attention', fontsize=18)
        ax.set_title(f'Attention Prefill Benchmark - Sequence Length: {seq_str}', fontsize=24, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {l}" for l in subset['Layer']], fontsize=16)

        # Add baseline horizontal line
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Flash Attention (1.0x)')

        # Adjust legend out of plot
        ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1.0), fontsize=16)

        # Optional styling
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "benchmark_each_layer_plot_combined.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comprehensive speedup plot to: {plot_path}")

if __name__ == "__main__":
    main()
