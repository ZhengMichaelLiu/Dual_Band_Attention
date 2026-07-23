import matplotlib.pyplot as plt
import numpy as np

# batch size = 1
naive_mem_usage_data = {
    '64K': {0.5: 32, 0.7: 32, 0.9: 32},
    '128K': {0.5: 128, 0.7: 128, 0.9: 128},
    '256K': {0.5: 512, 0.7: 512, 0.9: 512},
    '512K': {0.5: 2048, 0.7: 2048, 0.9: 2048}
}

csr_mem_usage_data = {
    '64K': {0.5: 16.06, 0.7: 9.66, 0.9: 3.26},
    '128K': {0.5: 64.13, 0.7: 38.51, 0.9: 12.93},
    '256K': {0.5: 256.21, 0.7: 153.88, 0.9: 51.43},
    '512K': {0.5: 1024.54, 0.7: 614.97, 0.9: 205.29}
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

def plot_memory_comparison():
    fig, ax = plt.subplots(figsize=(10, 4.5))

    seq_lens = list(naive_mem_usage_data.keys())
    sparsities = [0.5, 0.7, 0.9]

    x = np.arange(len(seq_lens))

    # Naive memory usage is the same regardless of sparsity, so we just take the 0.5 value
    naive_vals = [naive_mem_usage_data[seq][0.5] for seq in seq_lens]

    # Plot Naive Mask
    ax.plot(x, naive_vals, label='Simple Mask', color='#d62728', marker='X',
            markersize=9, linewidth=2.5, linestyle='--', zorder=4)

    # Label Naive Mask values
    for i, val in enumerate(naive_vals):
        ax.annotate(f"{int(val)}", (x[i], val), textcoords="offset points",
                    xytext=(0, 8), ha='center', va='bottom', fontsize=18,
                    color='#d62728', fontweight='bold', zorder=5)

    # color palette for CSR
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # Draw grid behind the lines
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

    # Plot CSR Mask for each sparsity
    for idx, sparsity in enumerate(sparsities):
        csr_vals = [csr_mem_usage_data[seq][sparsity] for seq in seq_lens]
        ax.plot(x, csr_vals, label=f'CBM (s = {sparsity})',
                color=colors[idx], marker=markers[idx], markersize=8,
                linewidth=2.5, zorder=3)

        # Label CSR Mask values for sparsity 0.9
        if sparsity == 0.9:
            for i, val in enumerate(csr_vals):
                ax.annotate(f"{val:.2f}", (x[i], val), textcoords="offset points",
                            xytext=(0, 18), ha='center', va='top', fontsize=18,
                            color=colors[idx], fontweight='bold', zorder=5)

    ax.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)

    # Use log scale since memory grows quadratically
    ax.set_yscale('log', base=10)

    # Expand limits to prevent annotations from hitting the edges
    ax.set_ylim(top=6500)
    ax.set_xlim(left=-0.3, right=len(seq_lens) - 0.7)

    # Legend
    ax.legend(frameon=True, loc='upper left')

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('darkgreen')

    plt.tight_layout()
    plot_filename = 'mask_memory_compression.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

plot_memory_comparison()
