import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Data (ms) — batch size = 1
lengths = ['32K', '64K', '128K']

full_attn_data = {
    'Linear Projection':        [337.4,   677.6,   1370.7],
    'Feed-Forward':              [1431.5,  2941.3,  6020.5],
    'Attention / Block Sparse':  [1366.7,  5289.0,  21694.0],
    'Mask Generation':           [0.0,     0.0,     0.0],
    'Unattributed':              [621.7,   1254.0,  2513.4],
}

dba_data = {
    'Linear Projection':        [339.7,   673.4,   1347.8],
    'Feed-Forward':              [1436.5,  2945.1,  5973.7],
    'Attention / Block Sparse':  [422.2,   1166.0,  4074.7],
    'Mask Generation':           [126.8,   417.6,   1419.5],
    'Unattributed':              [628.2,   1257.3,  2488.4],
}

# Set style for academic papers
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "serif"],
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

categories = ['Linear Projection', 'Feed-Forward', 'Attention / Block Sparse', 'Mask Generation', 'Unattributed']
colors = {
    'Linear Projection':        '#4C72B0',  # seaborn deep blue
    'Feed-Forward':              '#DD8452',  # seaborn deep orange
    'Attention / Block Sparse':  '#55A868',  # seaborn deep green
    'Mask Generation':           '#C44E52',  # seaborn deep red
    'Unattributed':              '#B4B4B4',  # neutral gray
}

n = len(lengths)
x = np.arange(n) * 0.75
bar_width = 0.2
bar_sep = 0.01
half = bar_width / 2 + bar_sep / 2

# Use one linear y-axis so every stacked segment has the same ms-to-height scale.
# Keep the exported page size identical to the existing manuscript figure.
fig, ax = plt.subplots(figsize=(457 / 72, 343 / 72))
fig.subplots_adjust(left=0.15, right=0.985, bottom=0.20, top=0.975)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

for i in range(n):
    for sign, data in [(-1, full_attn_data), (1, dba_data)]:
        bottom = 0.0
        for cat in categories:
            value = data[cat][i]
            ax.bar(x[i] + sign * half, value, bar_width, bottom=bottom,
                   color=colors[cat], edgecolor='black', linewidth=0.5, zorder=3)
            bottom += value
        ax.annotate(f'Total\n{bottom:.1f}', (x[i] + sign * half, bottom),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='black', zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.3))

ax.set_ylim(0, 35000)
ax.set_yticks(np.arange(0, 35001, 5000))
ax.ticklabel_format(axis='y', style='plain')
ax.tick_params(axis='y', labelsize=13)

# Label feed-forward and attention segments; keep small DBA segments legible
# with labels beside their bars instead of distorting segment heights.
for i in range(n):
    for sign, data in [(-1, full_attn_data), (1, dba_data)]:
        for cat in ['Feed-Forward', 'Attention / Block Sparse']:
            index = categories.index(cat)
            bottom = sum(data[categories[j]][i] for j in range(index))
            value = data[cat][i]
            center = bottom + value / 2
            if sign == 1 and cat == 'Attention / Block Sparse' and i < 2:
                y_offset = -250 if i == 1 else 0
                ax.annotate(f'{value:.1f}', (x[i] + half + bar_width / 2, center),
                            xytext=(x[i] + half + bar_width / 2 + 0.02,
                                    center + y_offset),
                            ha='left', va='center', fontsize=11,
                            fontweight='bold', color=colors[cat], zorder=6)
            else:
                ax.text(x[i] + sign * half, center, f'{value:.1f}',
                        ha='center', va='center',
                        fontsize=10 if value >= 10000 else 11,
                        fontweight='bold', color='white', zorder=6)

for i in [1, 2]:
    cat = 'Mask Generation'
    index = categories.index(cat)
    bottom = sum(dba_data[categories[j]][i] for j in range(index))
    value = dba_data[cat][i]
    center = bottom + value / 2
    if i == 1:
        ax.text(x[i] + half + bar_width / 2 + 0.02, center + 250,
                f'{value:.1f}', ha='left', va='center', fontsize=11,
                fontweight='bold', color=colors[cat], zorder=6)
    else:
        ax.text(x[i] + half, center, f'{value:.1f}',
                ha='center', va='center', fontsize=11,
                fontweight='bold', color='white', zorder=6)

positions = []
labels = []
for i in range(n):
    positions.extend([x[i] - half, x[i] + half])
    labels.extend(['FA2', 'DBA 16'])
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=12)
ax.tick_params(axis='x', pad=5)
for i, length in enumerate(lengths):
    ax.annotate(length, xy=(x[i], 0), xycoords=ax.get_xaxis_transform(),
                xytext=(0, -27), textcoords='offset points',
                ha='center', va='top', fontsize=15, fontweight='bold',
                annotation_clip=False)
ax.set_xlabel('Sequence Length', fontweight='bold', labelpad=30, fontsize=17)
ax.set_ylabel('Latency (ms)', fontweight='bold', fontsize=17, labelpad=4)
ax.set_xlim(-0.35, (n - 1) * 0.75 + 0.35)

handles = [Patch(facecolor=colors[cat],
                 label='Attention' if cat == 'Attention / Block Sparse' else cat)
           for cat in categories]
ax.legend(handles=handles, loc='upper left', ncol=2, fontsize=12,
          frameon=True, columnspacing=0.8, handletextpad=0.4)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
    spine.set_color('darkgreen')

plt.savefig('dba_latency_breakdown.pdf')
print('Saved plot to dba_latency_breakdown.pdf (linear y-axis: 0-35000 ms)')
