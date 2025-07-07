import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
methods = [
    "HeteFoA", "Dynamic Composition", "Difficulty Based Width Initialization",
    "Runtime Width Adaptation", "Width Adaptation + Difficulty Based Width Initialization",
    "Difficulty Based Width Initialization + Dynamic Composition",
    "Skewed State Detection (V1)", "Runtime Width Adaptation + Skewed State Detection (V1)"
]

accuracy_means = [0.55, 0.60, 0.58, 0.63, 0.60, 0.61, 0.66, 0.60]
accuracy_errs  = [0.02, 0.03, 0.03, 0.02, 0.03, 0.02, 0.04, 0.03]

cost_means = [0.75, 0.78, 0.60, 0.88, 0.66, 0.67, 0.74, 0.76]
cost_errs  = [0.05, 0.06, 0.04, 0.03, 0.04, 0.03, 0.05, 0.02]

# Bar positions
x = np.arange(len(methods))
width = 0.35

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Twin axis for cost
ax2 = ax1.twinx()

# Plot bars
acc_bars = ax1.bar(x - width/2, accuracy_means, width, yerr=accuracy_errs, label='Accuracy', color='blue', capsize=5)
cost_bars = ax2.bar(x + width/2, cost_means, width, yerr=cost_errs, label='Cost', color='red', capsize=5)

# Horizontal reference lines (example values, you can adjust as needed)
for val in [0.55, 0.60, 0.65]:
    ax1.axhline(val, linestyle='--', color='blue', alpha=0.4)
for val in [0.65, 0.70, 0.75]:
    ax2.axhline(val, linestyle='--', color='red', alpha=0.4)

# Labels and title
ax1.set_ylabel('Accuracy', color='blue')
ax2.set_ylabel('Cost', color='red')
ax1.set_title('Environment: scibench')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha='right')

# Axis limits
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)

# Show the plot
plt.tight_layout()
plt.show()
