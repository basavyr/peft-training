import matplotlib.pyplot as plt

# Data: (rank, alpha, trainable%, trainable params)
data = [
    (4, 8, 0.2230),
    (8, 16, 0.4451),
    (16, 32, 0.8862),
    (32, 64, 1.7568),
    (8, 4, 0.4451),  # extra experiment
]

# Extract data
ranks = [d[0] for d in data]
alphas = [d[1] for d in data]
trainable_percent = [d[2] for d in data]

# Create plot
plt.figure(figsize=(8, 5))
plt.plot(ranks, trainable_percent, marker='s', linewidth=2, label="α = 2r")
# Highlight special case α=4
plt.scatter(8, 0.4451, color='red', s=80, label="α = 4 (special case)")

# Labels and styling
plt.title("Trainable Parameters (%) vs LoRA Rank", fontsize=14, fontweight='bold')
plt.xlabel("LoRA Rank (r)", fontsize=12)
plt.ylabel("Trainable Parameters (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()

# Save to file and close
plt.savefig("lora_trainable_params_percent.png", dpi=300)
plt.close()

