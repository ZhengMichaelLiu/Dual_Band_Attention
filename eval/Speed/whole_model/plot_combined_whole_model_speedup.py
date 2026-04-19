import matplotlib.pyplot as plt
import numpy as np

# Set style for academic papers
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

lengths = ["4K", "8K", "16K", "32K", "64K", "128K"]

# Speed up data extracted from the table
speedup_data_a100 = {
    "MInference": [0.23, 0.36, 0.50, 0.71, 1.05, 1.85],
    "FlexPrefill": [0.72, 0.77, 0.90, 1.05, 1.28, 1.71],
    "XAttn 8": [0.93, 1.00, 1.09, 1.22, 1.50, 1.96],
    "XAttn 16": [0.93, 1.00, 1.10, 1.24, 1.54, 2.05],
    "DBA 8": [0.97, 1.03, 1.10, 1.25, 1.52, 1.95],
    "DBA 16": [0.98, 1.03, 1.12, 1.28, 1.58, 2.07]
}

speedup_data_h20 = {
    "MInference": [0.42, 0.61, 0.83, 1.18, 1.77, 3.08],
    "FlexPrefill": [0.89, 0.96, 1.12, 1.38, 1.82, 2.59],
    "XAttn 8": [0.98, 1.06, 1.18, 1.38, 1.85, 2.50],
    "XAttn 16": [0.98, 1.06, 1.19, 1.40, 1.85, 2.53],
    "DBA 8": [1.02, 1.09, 1.22, 1.44, 1.88, 2.49],
    "DBA 16": [1.02, 1.10, 1.23, 1.47, 1.93, 2.58]
}

def plot_whole_model_speedup():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    x = np.arange(len(lengths))

    methods = ["MInference", "FlexPrefill", "XAttn 8", "XAttn 16", "DBA 8", "DBA 16"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'v', 'D', 'P']
    linestyles = ['-', '--', '-.', ':', '-', '--']

    for i, (device_name, speedup_data) in enumerate([("A100", speedup_data_a100), ("H20", speedup_data_h20)]):
        ax = axes[i]
        # Draw grid behind the lines
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

        # Plot each method
        for idx, method in enumerate(methods):
            ax.plot(x, speedup_data[method], label=method,
                    color=colors[idx], marker=markers[idx], markersize=9,
                    linewidth=2.5, linestyle=linestyles[idx], zorder=3)

        # Baseline Full Attention
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Full Attention (1.0x)', zorder=4)

        if i == 0:
            ax.set_ylabel('Speed Up', fontweight='bold')
        ax.set_xlabel('Sequence Length', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lengths)

        # We can add titles simply inside or as an actual title
        title_str = 'A100' if i == 0 else 'H20'
        ax.text(0.05, 0.92, title_str, transform=ax.transAxes, fontweight='bold',
                fontsize=18, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.1)

        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_color('darkgreen')

    # Legend
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02),
               ncol=4, fontsize=14, frameon=True, columnspacing=0.8, handletextpad=0.4)

    plt.tight_layout(w_pad=2.5)
    plot_filename = 'combined_whole_model_speedup.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

plot_whole_model_speedup()