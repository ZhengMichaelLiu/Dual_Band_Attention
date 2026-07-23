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
speedup_data = {
    "MInference": [0.40, 0.56, 0.73, 0.96, 1.34, 2.20],
    "FlexPrefill": [0.85, 0.88, 0.98, 1.13, 1.41, 1.82],
    "XAttn 8": [0.94, 0.96, 1.03, 1.13, 1.41, 1.77],
    "XAttn 16": [0.94, 0.98, 1.04, 1.15, 1.41, 1.79],
    "DBA 8": [0.97, 1.01, 1.07, 1.18, 1.42, 1.75],
    "DBA 16": [0.97, 1.01, 1.08, 1.20, 1.45, 1.82]
}

def plot_whole_model_speedup():
    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(lengths))

    methods = ["MInference", "FlexPrefill", "XAttn 8", "XAttn 16", "DBA 8", "DBA 16"]
    # methods = ["FlexPrefill", "XAttn 8", "XAttn 16", "DBA 8", "DBA 16"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'v', 'D', 'P']
    linestyles = ['-', '--', '-.', ':', '-', '--']

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

    ax.set_ylabel('Speed Up', fontweight='bold')
    ax.set_xlabel('Sequence Length', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.2)

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('darkgreen')

    # Legend
    ax.legend(frameon=True, loc='upper center', ncol=3, columnspacing=0.8, handletextpad=0.4)

    plt.tight_layout()
    plot_filename = 'whole_model_speedup_h20.pdf'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved plot to {plot_filename}")

plot_whole_model_speedup()
