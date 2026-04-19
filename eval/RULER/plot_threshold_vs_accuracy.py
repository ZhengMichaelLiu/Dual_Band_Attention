threshold_accuracy_speed = [
    {"threshold": 1.0, "accuracy": 74.08, "speed": 27241.539},
    {"threshold": 0.97, "accuracy": 73.25384615, "speed": 17045.334},
    {"threshold": 0.95, "accuracy": 70.98846154, "speed": 13846.516},
    {"threshold": 0.9, "accuracy": 65.23461538, "speed": 10420.441},
    {"threshold": 0.85, "accuracy": 62.25230769, "speed": 8653.714},
    {"threshold": 0.8, "accuracy": 58.51923077, "speed": 7554.117},
    {"threshold": 0.7, "accuracy": 53.17538462, "speed": 6262.703},
    {"threshold": "mtp", "mean_threshold": 0.91, "accuracy": 72.61, "speed": 12261.537},
]

import matplotlib.pyplot as plt

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

# Extract data for lines
line_data = [d for d in threshold_accuracy_speed if isinstance(d["threshold"], (int, float))]
line_data.sort(key=lambda x: x["threshold"])

thresholds = [d["threshold"] for d in line_data]
accuracies = [d["accuracy"] for d in line_data]
speeds = [d["speed"] for d in line_data]

# Extract data for MTP
mtp_data = [d for d in threshold_accuracy_speed if d["threshold"] == "mtp"][0]
mtp_x = mtp_data["mean_threshold"]
mtp_acc = mtp_data["accuracy"]
mtp_speed = mtp_data["speed"]

fig, ax1 = plt.subplots(figsize=(8, 6))

color1 = '#1f77b4' # Academic blue
ax1.set_xlabel('Threshold', fontweight='bold')
ax1.set_ylabel('Score', color=color1, fontweight='bold')
ax1.plot(thresholds, accuracies, marker='o', color=color1, label='Score', markersize=8, linewidth=2.5, zorder=4)
ax1.scatter(mtp_x, mtp_acc, color=color1, marker='*', s=350, label='MTP Score', zorder=5)
ax1.tick_params(axis='y', labelcolor=color1)

# Ensure grid goes behind lines
ax1.set_axisbelow(True)
ax1.grid(True, linestyle='--', alpha=0.7, color='gray')

# Annotate ax1 (Accuracy) values
for i, val in enumerate(accuracies):
    ax1.annotate(f"{val:.1f}", (thresholds[i], val), textcoords="offset points",
                 xytext=(0, 8), ha='center', va='bottom', fontsize=12,
                 color=color1, fontweight='bold', zorder=5)
ax1.annotate(f"{mtp_acc:.1f}", (mtp_x, mtp_acc), textcoords="offset points",
             xytext=(0, 12), ha='center', va='bottom', fontsize=12,
             color=color1, fontweight='bold', zorder=5)

ax2 = ax1.twinx()
color2 = '#d62728' # Academic red
ax2.set_ylabel('Speed (s)', color=color2, fontweight='bold')
ax2.plot(thresholds, speeds, marker='X', color=color2, linestyle='--', label='Speed', markersize=9, linewidth=2.5, zorder=3)
ax2.scatter(mtp_x, mtp_speed, color=color2, marker='*', s=350, label='MTP Speed', zorder=5)
ax2.tick_params(axis='y', labelcolor=color2)

# Expand limits proportionally to prevent annotations from hitting the edges and keep lines aligned
pad_ratio_bottom = 0.15
pad_ratio_top = 0.15

acc_range = max(accuracies) - min(accuracies)
ax1.set_ylim(bottom=min(accuracies) - acc_range * pad_ratio_bottom,
             top=max(accuracies) + acc_range * pad_ratio_top)

spd_range = max(speeds) - min(speeds)
ax2.set_ylim(bottom=min(speeds) - spd_range * pad_ratio_bottom,
             top=max(speeds) + spd_range * pad_ratio_top)

# Annotate ax2 (Speed) values
for i, val in enumerate(speeds):
    ax2.annotate(f"{int(val)}", (thresholds[i], val), textcoords="offset points",
                 xytext=(0, -18), ha='center', va='top', fontsize=12,
                 color=color2, fontweight='bold', zorder=5)
ax2.annotate(f"{int(mtp_speed)}", (mtp_x, mtp_speed), textcoords="offset points",
             xytext=(0, 18), ha='center', va='bottom', fontsize=12,
             color=color2, fontweight='bold', zorder=5)

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)

# Spines styling to match plot_mask_compression.py
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('darkgreen')

plt.tight_layout()
plot_filename = 'threshold_vs_accuracy_speed.pdf'
plt.savefig(plot_filename, bbox_inches='tight')
print(f"Saved plot to {plot_filename}")
plt.show()