import matplotlib.pyplot as plt
import numpy as np

# batch size = 1 causal = true
full_latency_causal = {"4K": 1.878384501, "8K": 6.674814969, "16K": 25.63945577, "32K": 99.63941947, "64K": 391.2945427, "128K": 1556.695834}
dba_latency_causal = {
    "4K": {0.3: 1.718856394, 0.5: 1.233434305, 0.7: 0.812049955, 0.9: 0.453345478},
    "8K": {0.3: 5.173068494, 0.5: 3.803761676, 0.7: 2.360003069, 0.9: 0.988889486},
    "16K": {0.3: 19.02027614, 0.5: 13.78438994, 0.7: 8.332122117, 0.9: 2.879679205},
    "32K": {0.3: 73.96154292, 0.5: 53.19644697, 0.7: 32.08392486, 0.9: 11.02869771},
    "64K": {0.3: 289.0441325, 0.5: 207.4819915, 0.7: 124.7459073, 0.9: 42.3651766},
    "128K": {0.3: 1146.12643, 0.5: 815.3384235, 0.7: 490.1348632, 0.9: 165.1170509}
}

# batch size = 1 causal = false
full_latency_non_causal = {"4K": 3.309026361, "8K": 12.51510531, "16K": 48.84336516, "32K": 194.3806522, "64K": 772.6374716, "128K": 3081.799727}
dba_latency_non_causal = {
    "4K": {0.3: 2.56880559, 0.5: 1.938344911, 0.7: 1.203730702, 0.9: 0.578802079},
    "8K": {0.3: 9.149881663, 0.5: 6.745134805, 0.7: 4.131102085, 0.9: 1.498572528},
    "16K": {0.3: 35.38559005, 0.5: 25.5547557, 0.7: 15.34366049, 0.9: 5.093603134},
    "32K": {0.3: 140.5871976, 0.5: 100.9085644, 0.7: 60.31991541, 0.9: 20.35273239},
    "64K": {0.3: 556.4797558, 0.5: 398.4568398, 0.7: 238.9361095, 0.9: 80.87235503},
    "128K": {0.3: 2221.365313, 0.5: 1586.306775, 0.7: 952.5526576, 0.9: 318.7832641}
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
    plot_filename = 'combined_kernel_speedup_a100.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

plot_combined_speedup()