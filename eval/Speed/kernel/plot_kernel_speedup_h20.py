import matplotlib.pyplot as plt
import numpy as np

# batch size = 1 causal = true
full_latency_causal = {"4K": 0.9957228, "8K": 2.78497, "16K": 10.5791776, "32K": 42.0766306, "64K": 171.4441348, "128K": 702.7431192}
dba_latency_causal = {
    "4K": {0.3: 0.960474901, 0.5: 0.793259906, 0.7: 0.5990859, 0.9: 0.4579302},
    "8K": {0.3: 2.5263915, 0.5: 1.927907101, 0.7: 1.305530001, 0.9: 0.7089106},
    "16K": {0.3: 8.6952711, 0.5: 6.3789054, 0.7: 4.0169085, 0.9: 1.636605},
    "32K": {0.3: 33.4970627, 0.5: 24.3871395, 0.7: 15.0110637, 0.9: 5.592033701},
    "64K": {0.3: 132.7641121, 0.5: 96.1454657, 0.7: 58.2314683, 0.9: 20.5809601},
    "128K": {0.3: 540.5659047, 0.5: 387.5452172, 0.7: 234.1955067, 0.9: 81.5657283}
}

# batch size = 1 causal = false
full_latency_non_causal = {"4K": 1.4054396, "8K": 5.1748994, "16K": 21.3578656, "32K": 84.6165954, "64K": 338.1116094, "128K": 1351.954715}
dba_latency_non_causal = {
    "4K": {0.3: 1.27657099, 0.5: 1.0086613, 0.7: 0.6980036, 0.9: 0.500671301},
    "8K": {0.3: 4.1667097, 0.5: 3.127967599, 0.7: 1.9814033, 0.9: 0.868728599},
    "16K": {0.3: 16.0229714, 0.5: 11.6914434, 0.7: 7.147967399, 0.9: 2.5181708},
    "32K": {0.3: 63.1188579, 0.5: 45.4289576, 0.7: 27.4122139, 0.9: 9.8202464},
    "64K": {0.3: 250.6992159, 0.5: 180.4147645, 0.7: 108.843054, 0.9: 38.0525332},
    "128K": {0.3: 1009.192452, 0.5: 728.5989234, 0.7: 441.3965499, 0.9: 152.406868}
}

# Set style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "serif"],
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def plot_combined_speedup():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    seq_lens = list(full_latency_causal.keys())
    sparsities = [0.3, 0.5, 0.7, 0.9]

    x = np.arange(len(seq_lens))

    # color palette and markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    markers = ['o', 's', '^', 'D']

    # Plot causal on Left (axes[0]), non-causal on Right (axes[1])
    for i, is_causal in enumerate([True, False]):
        ax = axes[i]

        # Draw grid behind the lines
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

        full_latency = full_latency_causal if is_causal else full_latency_non_causal
        dba_latency = dba_latency_causal if is_causal else dba_latency_non_causal
        title_str = 'Causal' if is_causal else 'Non-Causal'

        for idx, sparsity in enumerate(sparsities):
            # Calculate speedup: Baseline Latency / DBA Latency
            speedups = [full_latency[seq] / dba_latency[seq][sparsity] for seq in seq_lens]

            ax.plot(x, speedups, label=f'DBA (s={sparsity})',
                   color=colors[idx], marker=markers[idx], markersize=8,
                   linewidth=2.5, zorder=3)

        if is_causal:
            ax.set_ylabel('Speed Up', fontweight='bold')

        ax.set_xlabel('Sequence Length', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lens)

        # Add baseline
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='FlashAttn (1.0x)', zorder=4)

        # Add titles inside plot at top-left
        ax.text(0.04, 0.96, title_str, transform=ax.transAxes, fontweight='bold',
                fontsize=18, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        # Give a little bit of breathing room at the top
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.05)

        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_color('darkgreen')

    # Add shared Legend centrally, right above both plots
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
               ncol=5, fontsize=14, frameon=True, columnspacing=0.8, handletextpad=0.4)

    plt.tight_layout(w_pad=2.0)
    plot_filename = 'combined_kernel_speedup_h20.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

plot_combined_speedup()