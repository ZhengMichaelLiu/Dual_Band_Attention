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

# Broken y-axis boundaries
y_break_low  = 17000   # top of bottom axis
y_break_high = 28000   # bottom of top axis

fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 5),
    gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.06})

for ax in [ax_top, ax_bottom]:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')

# Draw all bars on both axes (clipping handles visibility per range)
def draw_bars(ax):
    for i in range(n):
        bottom_fa = 0
        for cat in categories:
            val = full_attn_data[cat][i]
            ax.bar(x[i] - half, val, bar_width, bottom=bottom_fa,
                   color=colors[cat], edgecolor='black', linewidth=0.5, zorder=3)
            bottom_fa += val
        bottom_dba = 0
        for cat in categories:
            val = dba_data[cat][i]
            ax.bar(x[i] + half, val, bar_width, bottom=bottom_dba,
                   color=colors[cat], edgecolor='black', linewidth=0.5, zorder=3)
            bottom_dba += val

draw_bars(ax_top)
draw_bars(ax_bottom)

# Compute totals and set limits
fa_totals  = [sum(full_attn_data[cat][i] for cat in categories) for i in range(n)]
dba_totals = [sum(dba_data[cat][i]       for cat in categories) for i in range(n)]
max_total  = max(fa_totals + dba_totals)

ax_bottom.set_ylim(0, y_break_low)
ax_top.set_ylim(y_break_high, max_total * 1.12)

# Hide the spines at the break
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Diagonal break markers
d   = 0.5
bkw = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle='none',
           color='k', mec='k', mew=1.5, clip_on=False)
ax_top.plot(   [0, 1], [0, 0], transform=ax_top.transAxes,    **bkw)
ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **bkw)

# Helper: route text to the correct sub-axis based on y position
def ann(x_pos, y_pos, s, **kwargs):
    if y_pos >= y_break_high:
        ax_top.text(x_pos, y_pos, s, **kwargs)
    elif y_pos <= y_break_low:
        ax_bottom.text(x_pos, y_pos, s, **kwargs)

# --- Total annotations ---
for i in range(n):
    ann(x[i] - half, fa_totals[i],  f'Total\n{fa_totals[i]:.1f}',
        ha='center', va='bottom', fontsize=12, fontweight='bold', color='black', zorder=5)
    ann(x[i] + half, dba_totals[i], f'Total\n{dba_totals[i]:.1f}',
        ha='center', va='bottom', fontsize=12, fontweight='bold', color='black', zorder=5)

cat_idx = categories.index('Attention / Block Sparse')
ff_idx  = categories.index('Feed-Forward')
mg_idx  = categories.index('Mask Generation')

# --- Attention / Block Sparse annotations ---
# 128K FA: center is in break zone → place label in visible bottom portion
i_len = 2
val_fa_attn = full_attn_data['Attention / Block Sparse'][i_len]
bot_fa_attn = sum(full_attn_data[categories[j]][i_len] for j in range(cat_idx))
y_visible_center = (bot_fa_attn + y_break_low) / 2
ax_bottom.text(x[i_len] - half, y_visible_center, f'{val_fa_attn:.1f}',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)
# 128K DBA: inside bar in bottom axis
bot = sum(dba_data[categories[j]][i_len] for j in range(cat_idx))
val = dba_data['Attention / Block Sparse'][i_len]
ax_bottom.text(x[i_len] + half, bot + val / 2, f'{val:.1f}',
               ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)
# 64K FA: inside bar
i_len = 1
bot_fa = sum(full_attn_data[categories[j]][i_len] for j in range(cat_idx))
val_fa = full_attn_data['Attention / Block Sparse'][i_len]
ax_bottom.text(x[i_len] - half, bot_fa + val_fa / 2, f'{val_fa:.1f}',
               ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)
# 64K DBA: to the right with vertical offset
bot_dba = sum(dba_data[categories[j]][i_len] for j in range(cat_idx))
val_dba = dba_data['Attention / Block Sparse'][i_len]
ax_bottom.text(x[i_len] + half + bar_width / 2 + 0.02, bot_dba + val_dba / 2 - 200, f'{val_dba:.1f}',
               ha='left', va='center', fontsize=11, fontweight='bold',
               color=colors['Attention / Block Sparse'], zorder=6)

# --- Feed-Forward annotations (64K and 128K) ---
for i_len in [1, 2]:
    for sign, data in [(-1, full_attn_data), (1, dba_data)]:
        bot = sum(data[categories[j]][i_len] for j in range(ff_idx))
        val = data['Feed-Forward'][i_len]
        ax_bottom.text(x[i_len] + sign * half, bot + val / 2, f'{val:.1f}',
                       ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)

# --- Mask Generation annotations (DBA 16 only) ---
# 64K: to the right with vertical offset
i_len = 1
bot = sum(dba_data[categories[j]][i_len] for j in range(mg_idx))
val = dba_data['Mask Generation'][i_len]
ax_bottom.text(x[i_len] + half + bar_width / 2 + 0.02, bot + val / 2 + 200, f'{val:.1f}',
               ha='left', va='center', fontsize=11, fontweight='bold',
               color=colors['Mask Generation'], zorder=6)
# 128K: inside bar
i_len = 2
bot = sum(dba_data[categories[j]][i_len] for j in range(mg_idx))
val = dba_data['Mask Generation'][i_len]
ax_bottom.text(x[i_len] + half, bot + val / 2, f'{val:.1f}',
               ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)

# --- 32K specific annotations ---
i_len = 0
for cat, c_idx in [('Feed-Forward', ff_idx), ('Attention / Block Sparse', cat_idx)]:
    bot = sum(full_attn_data[categories[j]][i_len] for j in range(c_idx))
    val = full_attn_data[cat][i_len]
    ax_bottom.text(x[i_len] - half, bot + val / 2, f'{val:.1f}',
                   ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)
bot_ff = sum(dba_data[categories[j]][i_len] for j in range(ff_idx))
val_ff = dba_data['Feed-Forward'][i_len]
ax_bottom.text(x[i_len] + half, bot_ff + val_ff / 2, f'{val_ff:.1f}',
               ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)
bot_attn = sum(dba_data[categories[j]][i_len] for j in range(cat_idx))
val_attn = dba_data['Attention / Block Sparse'][i_len]
ax_bottom.text(x[i_len] + half + bar_width / 2 + 0.02, bot_attn + val_attn / 2, f'{val_attn:.1f}',
               ha='left', va='center', fontsize=11, fontweight='bold',
               color=colors['Attention / Block Sparse'], zorder=6)

# --- X-axis (shared, configured on bottom) ---
tick_pos = []
tick_lab = []
for i in range(n):
    tick_pos.extend([x[i] - half, x[i] + half])
    tick_lab.extend(['FA', 'DBA 16'])

ax_bottom.set_xticks(tick_pos)
ax_bottom.set_xticklabels(tick_lab, fontsize=12)
ax_bottom.tick_params(axis='x', pad=5)

for i, length in enumerate(lengths):
    ax_bottom.annotate(length, xy=(x[i], 0),
                       xycoords=ax_bottom.get_xaxis_transform(),
                       xytext=(0, -28), textcoords='offset points',
                       ha='center', va='top', fontsize=16, fontweight='bold',
                       annotation_clip=False)

ax_bottom.set_xlabel('Sequence Length', fontweight='bold', labelpad=28)
ax_bottom.set_xlim(-0.35, (n - 1) * 0.75 + 0.35)

# Shared y-label
fig.text(0.02, 0.5, 'Latency (ms)', va='center', ha='center',
         rotation='vertical', fontsize=18, fontweight='bold')

# Legend in bottom axis upper left
legend_handles = [Patch(facecolor=colors[cat], label=cat) for cat in categories]
legend_handles[categories.index('Mask Generation')] = Patch(
    facecolor=colors['Mask Generation'], label='Mask Generation')
legend_handles[categories.index('Attention / Block Sparse')] = Patch(
    facecolor=colors['Attention / Block Sparse'], label='Attention')
ax_top.legend(handles=legend_handles, loc='upper left', ncol=2,
                 frameon=True, columnspacing=0.8, handletextpad=0.4)

# Style spines
for ax in [ax_top, ax_bottom]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('darkgreen')
# Re-hide break spines (overridden by the loop above)
ax_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)

plt.savefig('dba_latency_breakdown.pdf', bbox_inches='tight')
print("Saved plot to dba_latency_breakdown.pdf")