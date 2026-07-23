import logging
import os
import time
import types
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

import DBA.src.dual_band_attention as dba_impl
from DBA.src.Fullprefill import Full_prefill
from DBA.src.dual_band_attention import DBA_forward
from DBA.threshold.llama_threshold import llama_fuse_8

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("flashinfer.jit").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

SEQ_LENS_K = [32, 64, 128]
NUM_WARMUPS = 2
NUM_ITERATIONS = 2
DBA_PATCH_SIZE = 16
SAVE_DIR = "benchmark_latency_breakdown_result"


class LatencyCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.linear_projection_ms = 0.0
        self.feed_forward_ms = 0.0
        self.attention_step_ms = 0.0
        self.dba_mask_generation_ms = 0.0
        self.dba_block_sparse_kernel_ms = 0.0

    def snapshot(self):
        return {
            "linear_projection_ms": self.linear_projection_ms,
            "feed_forward_ms": self.feed_forward_ms,
            "attention_step_ms": self.attention_step_ms,
            "dba_mask_generation_ms": self.dba_mask_generation_ms,
            "dba_block_sparse_kernel_ms": self.dba_block_sparse_kernel_ms,
        }


def cuda_timed_call(fn, *args, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    out = fn(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return out, float(start_event.elapsed_time(end_event))


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
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def generate_prompt(tokenizer, target_len):
    context = "A quick brown fox jumps over the lazy dog. What the dog doing?\n"
    needle = "This is a special needle that we need to find."
    if os.path.exists("needle.txt"):
        with open("needle.txt", "r") as f:
            needle = f.read()

    num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
    num_tokens_needle = len(tokenizer.encode(needle, add_special_tokens=False))

    prefix = (
        "This is a very long story book with knowledge of Flash Attention 4, "
        "which you need to remember for later question: <book> "
    )
    suffix = (
        "</book>\n Based on the content of the book, please briefly tell me about "
        "Flash Attention 4.\nAnswer:"
    )
    static_tokens = len(tokenizer.encode(prefix, add_special_tokens=True)) + len(
        tokenizer.encode(suffix, add_special_tokens=False)
    )

    available_context_tokens = target_len - static_tokens - num_tokens_needle
    num_repetitions = (
        0 if available_context_tokens < 0 else available_context_tokens // num_tokens_context
    )

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
        pad = torch.zeros((1, pad_len), dtype=torch.long, device="cuda")
        input_ids = torch.cat([input_ids, pad], dim=1)
    return input_ids


def dba_forward_with_breakdown(
    q,
    k,
    v,
    threshold_val,
    target_len,
    patch_size,
    collector,
):
    original_create_block_mask = dba_impl.create_block_mask
    original_block_sparse_kernel = dba_impl.block_sparse_flash_attn_func

    def timed_create_block_mask(*args, **kwargs):
        result, elapsed_ms = cuda_timed_call(original_create_block_mask, *args, **kwargs)
        collector.dba_mask_generation_ms += elapsed_ms
        return result

    def timed_block_sparse_kernel(*args, **kwargs):
        result, elapsed_ms = cuda_timed_call(original_block_sparse_kernel, *args, **kwargs)
        collector.dba_block_sparse_kernel_ms += elapsed_ms
        return result

    dba_impl.create_block_mask = timed_create_block_mask
    dba_impl.block_sparse_flash_attn_func = timed_block_sparse_kernel
    try:
        return DBA_forward(
            q,
            k,
            v,
            sparse_block_size=128,
            patch_size=patch_size,
            causal=True,
            cumulative_p_threshold=threshold_val,
            chunk_size=min(32768, target_len),
            keep_sink=True,
            keep_local=True,
        )
    finally:
        dba_impl.create_block_mask = original_create_block_mask
        dba_impl.block_sparse_flash_attn_func = original_block_sparse_kernel


def make_custom_forward(method_name, threshold_val, target_len, patch_size, collector):
    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states, elapsed_ms = cuda_timed_call(self.q_proj, hidden_states)
        collector.linear_projection_ms += elapsed_ms
        key_states, elapsed_ms = cuda_timed_call(self.k_proj, hidden_states)
        collector.linear_projection_ms += elapsed_ms
        value_states, elapsed_ms = cuda_timed_call(self.v_proj, hidden_states)
        collector.linear_projection_ms += elapsed_ms

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_ids is None:
            position_ids = torch.arange(
                kv_seq_len, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand(bsz, kv_seq_len)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        q = query_states.contiguous()
        k = key_states.contiguous()
        v = value_states.contiguous()
        del query_states, key_states, value_states, cos, sin

        if method_name == "Full":
            attn_output, elapsed_ms = cuda_timed_call(Full_prefill, q, k, v, True)
            collector.attention_step_ms += elapsed_ms
        elif method_name == "DBA":
            attn_output = dba_forward_with_breakdown(
                q,
                k,
                v,
                threshold_val=threshold_val,
                target_len=target_len,
                patch_size=patch_size,
                collector=collector,
            )
        else:
            raise ValueError(f"Unknown method {method_name}")

        del q, k, v

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output, elapsed_ms = cuda_timed_call(self.o_proj, attn_output)
        collector.linear_projection_ms += elapsed_ms

        return attn_output, None, None

    return custom_forward


def make_timed_mlp_forward(original_forward, collector):
    def timed_mlp_forward(self, *args, **kwargs):
        output, elapsed_ms = cuda_timed_call(original_forward, *args, **kwargs)
        collector.feed_forward_ms += elapsed_ms
        return output

    return timed_mlp_forward


def benchmark_one_method(
    model,
    input_ids,
    method_name,
    thresholds,
    target_len,
    num_layers,
    original_attn_forwards,
    original_mlp_forwards,
):
    collector = LatencyCollector()
    component_keys = [
        "linear_projection_ms",
        "feed_forward_ms",
        "attention_step_ms",
        "dba_mask_generation_ms",
        "dba_block_sparse_kernel_ms",
    ]
    component_sums = {k: 0.0 for k in component_keys}
    whole_forward_times_ms = []
    error_message = ""

    try:
        for i in range(num_layers):
            layer = model.model.layers[i]
            threshold_val = thresholds[i % len(thresholds)] if thresholds is not None else None

            new_attn_forward = make_custom_forward(
                method_name=method_name,
                threshold_val=threshold_val,
                target_len=target_len,
                patch_size=DBA_PATCH_SIZE,
                collector=collector,
            )
            layer.self_attn.forward = types.MethodType(new_attn_forward, layer.self_attn)

            timed_mlp_forward = make_timed_mlp_forward(original_mlp_forwards[i], collector)
            layer.mlp.forward = types.MethodType(timed_mlp_forward, layer.mlp)

        torch.cuda.empty_cache()

        with torch.no_grad():
            for _ in range(NUM_WARMUPS):
                collector.reset()
                out = model(input_ids, use_cache=False)
                del out

            torch.cuda.synchronize()

            for _ in range(NUM_ITERATIONS):
                collector.reset()
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                out = model(input_ids, use_cache=False)
                del out
                torch.cuda.synchronize()
                whole_ms = (time.perf_counter() - t_start) * 1000.0

                whole_forward_times_ms.append(whole_ms)
                snap = collector.snapshot()
                for key in component_keys:
                    component_sums[key] += snap[key]

    except Exception as exc:
        error_message = str(exc)

    finally:
        for i in range(num_layers):
            model.model.layers[i].self_attn.forward = original_attn_forwards[i]
            model.model.layers[i].mlp.forward = original_mlp_forwards[i]

    if not whole_forward_times_ms:
        return {
            "Whole Forward (ms)": float("nan"),
            "Linear Projection (ms)": float("nan"),
            "Feed-Forward (ms)": float("nan"),
            "Attention Step (ms)": float("nan"),
            "Mask Generation (ms)": float("nan"),
            "Block Sparse Kernel (ms)": float("nan"),
            "Unattributed (ms)": float("nan"),
            "Error": error_message,
        }

    num_measurements = len(whole_forward_times_ms)
    avg_whole = float(np.mean(whole_forward_times_ms))
    avg_components = {k: component_sums[k] / num_measurements for k in component_keys}

    if method_name == "Full":
        attributed = (
            avg_components["linear_projection_ms"]
            + avg_components["feed_forward_ms"]
            + avg_components["attention_step_ms"]
        )
    else:
        attributed = (
            avg_components["linear_projection_ms"]
            + avg_components["feed_forward_ms"]
            + avg_components["dba_mask_generation_ms"]
            + avg_components["dba_block_sparse_kernel_ms"]
        )

    return {
        "Whole Forward (ms)": avg_whole,
        "Linear Projection (ms)": avg_components["linear_projection_ms"],
        "Feed-Forward (ms)": avg_components["feed_forward_ms"],
        "Attention Step (ms)": avg_components["attention_step_ms"] if method_name == "Full" else 0.0,
        "Mask Generation (ms)": avg_components["dba_mask_generation_ms"] if method_name == "DBA" else 0.0,
        "Block Sparse Kernel (ms)": avg_components["dba_block_sparse_kernel_ms"] if method_name == "DBA" else 0.0,
        "Unattributed (ms)": avg_whole - attributed,
        "Error": error_message,
    }


def plot_breakdown(df, save_dir):
    lengths = sorted(df["Tokens"].unique())
    labels = [f"{k // 1024}K" for k in lengths]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    full_df = df[df["Method"] == "Full Attention"].set_index("Tokens").reindex(lengths)
    dba_df = df[df["Method"] == "DBA"].set_index("Tokens").reindex(lengths)

    full_components = [
        ("Linear Projection (ms)", "#4C78A8"),
        ("Attention Step (ms)", "#F58518"),
        ("Feed-Forward (ms)", "#54A24B"),
        ("Unattributed (ms)", "#B0B0B0"),
    ]
    dba_components = [
        ("Linear Projection (ms)", "#4C78A8"),
        ("Mask Generation (ms)", "#E45756"),
        ("Block Sparse Kernel (ms)", "#72B7B2"),
        ("Feed-Forward (ms)", "#54A24B"),
        ("Unattributed (ms)", "#B0B0B0"),
    ]

    bottom = np.zeros(len(lengths), dtype=np.float64)
    for col, color in full_components:
        values = full_df[col].fillna(0.0).to_numpy(dtype=np.float64)
        axes[0].bar(labels, values, bottom=bottom, label=col, color=color)
        bottom += values
    axes[0].set_title("Full Attention Breakdown")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].legend(fontsize=8)

    bottom = np.zeros(len(lengths), dtype=np.float64)
    for col, color in dba_components:
        values = dba_df[col].fillna(0.0).to_numpy(dtype=np.float64)
        axes[1].bar(labels, values, bottom=bottom, label=col, color=color)
        bottom += values
    axes[1].set_title("DBA Breakdown")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "latency_breakdown_stacked.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(
        os.path.join(current_dir, "../../../../Models/Llama_3_1_8B_Instruct/")
    )

    print("Loading tokenizer and model for whole-model latency breakdown profiling...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Detected {num_layers} attention layers in the model.")

    threshold_8_all = torch.tensor(llama_fuse_8, device="cuda")
    methods_cfg = [
        ("Full Attention", "Full", None),
        ("DBA", "DBA", threshold_8_all),
    ]

    original_attn_forwards = [model.model.layers[i].self_attn.forward for i in range(num_layers)]
    original_mlp_forwards = [model.model.layers[i].mlp.forward for i in range(num_layers)]

    all_rows = []

    for seq_len_k in SEQ_LENS_K:
        target_len = seq_len_k * 1024
        input_ids = generate_prompt(tokenizer, target_len)
        print(f"\n--- Profiling sequence length: {seq_len_k}K ({target_len} tokens) ---")

        for display_name, method_name, thresholds in methods_cfg:
            print(f"Profiling {display_name} ...")
            method_results = benchmark_one_method(
                model=model,
                input_ids=input_ids,
                method_name=method_name,
                thresholds=thresholds,
                target_len=target_len,
                num_layers=num_layers,
                original_attn_forwards=original_attn_forwards,
                original_mlp_forwards=original_mlp_forwards,
            )

            row = {
                "Length": f"{seq_len_k}K",
                "Tokens": target_len,
                "Method": display_name,
            }
            row.update(method_results)
            all_rows.append(row)

            if method_results["Error"]:
                print(f"  Failed: {method_results['Error']}")
            else:
                print(f"  Whole Forward: {method_results['Whole Forward (ms)']:.2f} ms")
                print(f"  Linear Projection: {method_results['Linear Projection (ms)']:.2f} ms")
                print(f"  Feed-Forward: {method_results['Feed-Forward (ms)']:.2f} ms")
                if method_name == "Full":
                    print(f"  Attention Step: {method_results['Attention Step (ms)']:.2f} ms")
                else:
                    print(f"  Mask Generation: {method_results['Mask Generation (ms)']:.2f} ms")
                    print(f"  Block Sparse Kernel: {method_results['Block Sparse Kernel (ms)']:.2f} ms")
                print(f"  Unattributed: {method_results['Unattributed (ms)']:.2f} ms")

    df = pd.DataFrame(all_rows).sort_values(by=["Tokens", "Method"]).reset_index(drop=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, "latency_breakdown.csv")
    df.to_csv(csv_path, index=False)
    plot_breakdown(df, SAVE_DIR)

    numeric_cols = [
        "Whole Forward (ms)",
        "Linear Projection (ms)",
        "Feed-Forward (ms)",
        "Attention Step (ms)",
        "Mask Generation (ms)",
        "Block Sparse Kernel (ms)",
        "Unattributed (ms)",
    ]
    rounded_df = df.copy()
    rounded_df[numeric_cols] = rounded_df[numeric_cols].round(3)

    print("\n=== Latency Breakdown Table (ms) ===")
    print(
        rounded_df[
            [
                "Length",
                "Method",
                "Whole Forward (ms)",
                "Linear Projection (ms)",
                "Feed-Forward (ms)",
                "Attention Step (ms)",
                "Mask Generation (ms)",
                "Block Sparse Kernel (ms)",
                "Unattributed (ms)",
                "Error",
            ]
        ].to_string(index=False)
    )

    full_df = df[df["Method"] == "Full Attention"].set_index("Tokens")
    dba_df = df[df["Method"] == "DBA"].set_index("Tokens")
    common_tokens = sorted(set(full_df.index).intersection(set(dba_df.index)))
    if common_tokens:
        speedup_rows = []
        for tokens in common_tokens:
            full_ms = full_df.loc[tokens, "Whole Forward (ms)"]
            dba_ms = dba_df.loc[tokens, "Whole Forward (ms)"]
            speedup = (
                full_ms / dba_ms
                if pd.notna(full_ms) and pd.notna(dba_ms) and dba_ms > 0
                else float("nan")
            )
            speedup_rows.append(
                {
                    "Length": f"{tokens // 1024}K",
                    "Full/DBA Speedup": speedup,
                }
            )
        speedup_df = pd.DataFrame(speedup_rows)
        print("\n=== Full vs DBA Whole-Forward Speedup ===")
        print(speedup_df.to_string(index=False))

    print(f"\nSaved latency breakdown results to '{SAVE_DIR}'.")


if __name__ == "__main__":
    main()