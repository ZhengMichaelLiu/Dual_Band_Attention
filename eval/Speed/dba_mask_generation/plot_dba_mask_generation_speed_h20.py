import matplotlib.pyplot as plt
import numpy as np

# batch size = 1	causal = true
speed_up_over_tri_causal = {
    '4K': {8: 5.94, 16: 4.60, 32: 3.93},
    '8K': {8: 5.24, 16: 4.96, 32: 4.35},
    '16K': {8: 1.51, 16: 1.67, 32: 1.90},
    '32K': {8: 1.41, 16: 1.52, 32: 1.64},
    '64K': {8: 1.24, 16: 1.26, 32: 1.26},
    '128K': {8: 1.00, 16: 1.01, 32: 0.85}
}

# batch size = 1	causal = false
speed_up_over_tri_non_causal = {
    '4K': {8: 8.82, 16: 7.99, 32: 4.65},
    '8K': {8: 4.62, 16: 4.36, 32: 4.24},
    '16K': {8: 1.23, 16: 1.35, 32: 1.22},
    '32K': {8: 1.22, 16: 1.40, 32: 1.47},
    '64K': {8: 1.21, 16: 1.55, 32: 1.52},
    '128K': {8: 1.03, 16: 1.65, 32: 1.66}
}

# Set style for academic papers (NeurIPS, ICML, ICLR style)
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

    seq_lens = list(speed_up_over_tri_causal.keys())
    strides = [8, 16, 32]

    x = np.arange(len(seq_lens))

    # Professional academic color palette and markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # Plot for causal (Left) and non-causal (Right)
    for i, is_causal in enumerate([True, False]):
        ax = axes[i]

        # Draw grid behind the lines
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

        data_dict = speed_up_over_tri_causal if is_causal else speed_up_over_tri_non_causal
        title_str = 'Causal' if is_causal else 'Non-Causal'

        # Plot for each stride
        for idx, stride in enumerate(strides):
            values = [data_dict[seq][stride] for seq in seq_lens]
            ax.plot(x, values, label=f'Patch Size {stride}',
                   color=colors[idx], marker=markers[idx], markersize=8,
                   linewidth=2.5, zorder=3)

        if is_causal:
            ax.set_ylabel('Speed Up', fontweight='bold')

        ax.set_xlabel('Sequence Length', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lens)

        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='XAtten (1.0x)', zorder=4)

        # Add titles inside plot at top-right
        ax.text(0.96, 0.96, title_str, transform=ax.transAxes, fontweight='bold',
                fontsize=24, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        # Give a little bit of breathing room at the top
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.05)

        # Make the plot edges thicker and green
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_color('darkgreen')

    plt.tight_layout(w_pad=2.0)
    plot_filename = 'combined_mask_speedup_h20.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

plot_combined_speedup()
