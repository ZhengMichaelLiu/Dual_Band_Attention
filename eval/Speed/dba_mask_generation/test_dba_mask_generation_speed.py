import os
import torch
import openpyxl
import matplotlib.pyplot as plt
from DBA.src.dual_band_attention import create_block_mask
from DBA.src.Xattention import xattn_estimate

def run_benchmark(func, args, num_runs=10):
    # Warmup
    for _ in range(3):
        func(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = func(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    mean = sum(times) / len(times)
    return mean

if __name__ == "__main__":
    seq_lens = [4096, 8192, 16384, 32768, 65536, 131072]
    strides = [8, 16, 32]
    batch_sizes = [1]
    block_size = 128
    num_runs = 10
    causal_options = [True, False]

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(curr_dir, "benchmark_mask_generation_speed_result")
    os.makedirs(out_dir, exist_ok=True)

    headers = [
        'Seq Length', 'Stride',
        'XAttn PT(ms)', 'XAttn Tri(ms)',
        'DBA(ms)', 'Spd (vs PT)', 'Spd (vs Tri)'
    ]

    def run_xattn(q, k, blk, strd, csl, tri):
        return xattn_estimate(q, k, block_size=blk, stride=strd, causal=csl, use_triton=tri)

    for bs in batch_sizes:
        plot_data = {
            True: {s: {'len': [], 'pt': [], 'tri': [], 'dba': []} for s in strides},
            False: {s: {'len': [], 'pt': [], 'tri': [], 'dba': []} for s in strides}
        }

        for causal in causal_options:
            causal_str = "Causal" if causal else "NonCausal"
            print(f"\n{'='*100}")
            print(f"BENCHMARK TABLE: Batch Size = {bs}, {causal_str}")

            header_fmt = "{:<12} | {:<8} | {:<15} | {:<15} | {:<12} | {:<15} | {:<15}"
            print(header_fmt.format(*headers))
            print("-" * 100)

            excel_data = []

            for seq_len in seq_lens:
                seq_len_str = f"{seq_len // 1024}K"

                Q = torch.randn(bs, 32, seq_len, 128, device='cuda', dtype=torch.float16)
                K = torch.randn(bs, 32, seq_len, 128, device='cuda', dtype=torch.float16)

                for stride in strides:
                    # 1. XAttn PT
                    xa_pt_m = run_benchmark(run_xattn, (Q, K, block_size, stride, causal, False), num_runs)

                    # 2. XAttn Tri
                    xa_tri_m = run_benchmark(run_xattn, (Q, K, block_size, stride, causal, True), num_runs)

                    # 3. DBA Combine (create_block_mask)
                    dba_m = run_benchmark(create_block_mask, (Q, K, block_size, stride, causal), num_runs)

                    # Store for plotting
                    plot_data[causal][stride]['len'].append(seq_len_str)
                    plot_data[causal][stride]['pt'].append(xa_pt_m)
                    plot_data[causal][stride]['tri'].append(xa_tri_m)
                    plot_data[causal][stride]['dba'].append(dba_m)

                    # Speedups
                    spd_vs_pt = xa_pt_m / dba_m if dba_m > 0 else 0
                    spd_vs_tri = xa_tri_m / dba_m if dba_m > 0 else 0

                    row = [
                        seq_len_str, stride,
                        round(xa_pt_m, 6),
                        round(xa_tri_m, 6),
                        round(dba_m, 6),
                        f"{spd_vs_pt:.2f}x", f"{spd_vs_tri:.2f}x"
                    ]
                    excel_data.append(row)

                    print(header_fmt.format(
                        seq_len_str, stride,
                        f"{xa_pt_m:.6f}",
                        f"{xa_tri_m:.6f}",
                        f"{dba_m:.6f}",
                        f"{spd_vs_pt:.2f}x", f"{spd_vs_tri:.2f}x"
                    ))

                del Q, K
                torch.cuda.empty_cache()

            # Write to Excel
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(headers)
            for row in excel_data:
                ws.append(row)

            filename = f"benchmark_mask_generation_BS{bs}_{causal_str}.xlsx"
            file_path = os.path.join(out_dir, filename)
            wb.save(file_path)
            print(f"-> Saved Excel: {file_path}")

        # Generating benchmark plots for the current batch size
        print(f"\nGenerating benchmark plots for Batch Size = {bs}...")
        fig, axes = plt.subplots(nrows=len(strides), ncols=2, figsize=(16, 4 * len(strides)), constrained_layout=True)

        col_names = {True: "Causal", False: "Non-Causal"}
        for c_idx, causal in enumerate([True, False]): # Left=Causal(True), Right=Non-Causal(False)
            for s_idx, stride in enumerate(strides):
                ax = axes[s_idx][c_idx]
                data = plot_data[causal][stride]
                x_ticks = range(len(data['len']))

                # Plot Lines
                ax.plot(x_ticks, data['pt'], label='XAttn PT', marker='o', linestyle='--', alpha=0.6)
                ax.plot(x_ticks, data['tri'], label='XAttn Tri', marker='s', linestyle='--', alpha=0.6)
                ax.plot(x_ticks, data['dba'], label='DBA', marker='^', linewidth=2)

                # Annotate Speedups (DBA vs Triton)
                for i, (comb, tri) in enumerate(zip(data['dba'], data['tri'])):
                    if comb > 0 and tri > 0:
                        spd = tri / comb
                        ax.annotate(f"{spd:.1f}x", (i, comb), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

                # Labels and Titles
                if s_idx == 0:
                    ax.set_title(f"{col_names[causal]} (BS={bs})", fontsize=14, fontweight='bold')

                ax.set_ylabel(f"Patch Size {stride}\nTime (ms)")
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(data['len'])
                ax.grid(True, alpha=0.3)

                if s_idx == len(strides) - 1:
                    ax.set_xlabel("Sequence Length")

                # Legend only on first plot
                if s_idx == 0 and c_idx == 1:
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.suptitle(f"DBA Mask Generation Benchmark (Batch Size = {bs})", fontsize=16)
        plot_filename = f"benchmark_results_BS{bs}.png"
        plot_path = os.path.join(out_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"-> Saved Plot: {plot_path}")
