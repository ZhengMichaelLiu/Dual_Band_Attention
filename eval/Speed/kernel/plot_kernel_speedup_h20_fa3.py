import matplotlib.pyplot as plt
import numpy as np

# batch size = 1 causal = true
full_latency_causal = {"4K": 1.1158779, "8K": 4.0422357, "16K": 15.793196, "32K": 62.3588398, "64K": 247.993487, "128K": 989.253952}
dba_latency_causal = {
    "4K": {0.3: 1.4640241, 0.5: 1.1280115, 0.7: 0.7395956, 0.9: 0.39033815},
    "8K": {0.3: 5.0129830, 0.5: 3.66645492, 0.7: 2.2570140, 0.9: 0.91449357},
    "16K": {0.3: 18.6643976, 0.5: 13.455218, 0.7: 8.124419, 0.9: 2.86007672},
    "32K": {0.3: 72.564858, 0.5: 52.1068342, 0.7: 31.363580, 0.9: 10.760899},
    "64K": {0.3: 284.571751, 0.5: 203.957882, 0.7: 122.618406, 0.9: 41.4623606},
    "128K": {0.3: 1131.99090, 0.5: 809.7818106, 0.7: 486.329128, 0.9: 163.495710}
}

# batch size = 1 causal = false
full_latency_non_causal = {"4K": 2.1906331, "8K": 8.1184811, "16K": 31.671566, "32K": 125.207883, "64K": 496.848991, "128K": 1978.5132}
dba_latency_non_causal = {
    "4K": {0.3: 2.46657319, 0.5: 1.8447011, 0.7: 1.122290, 0.9: 0.5029276},
    "8K": {0.3: 8.9518986, 0.5: 6.5734554, 0.7: 3.9960544, 0.9: 1.4161236},
    "16K": {0.3: 34.851188, 0.5: 25.148488, 0.7: 15.060139, 0.9: 4.9564979},
    "32K": {0.3: 139.078779, 0.5: 99.7355625, 0.7: 59.558219, 0.9: 20.009398},
    "64K": {0.3: 551.264386, 0.5: 394.649086, 0.7: 236.528368, 0.9: 79.883900},
    "128K": {0.3: 2196.712695, 0.5: 1571.744962, 0.7: 943.687186, 0.9: 315.50451405}
}

# Set style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "serif"],
    "axes.labelsize": 26,
    "axes.titlesize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
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
                fontsize=24, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        # Give a little bit of breathing room at the top
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.05)

        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_color('darkgreen')

    # # Add shared Legend centrally, right above both plots
    # lines, labels = axes[0].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
    #            ncol=3, fontsize=20, frameon=True, columnspacing=0.8, handletextpad=0.4)

    plt.tight_layout(w_pad=2.0)
    plot_filename = 'combined_kernel_speedup_h20.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

plot_combined_speedup()