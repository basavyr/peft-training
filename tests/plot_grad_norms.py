import matplotlib.pyplot as plt

# Define data (from all experiments)
experiments = {
    (8, 4): [
        {'epoch': 0.31, 'grad_norm': 0.6093101501464844},
        {'epoch': 0.62, 'grad_norm': 0.9116100072860718},
        {'epoch': 0.94, 'grad_norm': 1.1311554908752441},
    ],
    (8, 8): [
        {'epoch': 0.31, 'grad_norm': 1.3949666023254395},
        {'epoch': 0.62, 'grad_norm': 1.6945232152938843},
        {'epoch': 0.94, 'grad_norm': 2.20668625831604},
    ],
    (8, 16): [
        {'epoch': 0.31, 'grad_norm': 2.7735562324523926},
        {'epoch': 0.62, 'grad_norm': 3.5233919620513916},
        {'epoch': 0.94, 'grad_norm': 4.484832763671875},
    ],
    (8, 32): [
        {'epoch': 0.31, 'grad_norm': 5.783611297607422},
        {'epoch': 0.62, 'grad_norm': 6.987475872039795},
        {'epoch': 0.94, 'grad_norm': 9.144457817077637},
    ]
}

# Create plot
plt.figure(figsize=(8, 5))

for (r, alpha), results in experiments.items():
    epochs = [entry['epoch'] for entry in results]
    grad_norms = [entry['grad_norm'] for entry in results]
    label = f"r={r}, α={alpha}"
    plt.plot(
        epochs,
        grad_norms,
        marker='o',
        linewidth=2,
        label=label
    )

# Style and labels
plt.title("Evolution of Gradient Norms During LoRA Training", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Gradient Norm", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="LoRA Configurations", fontsize=10)
plt.tight_layout()

# Save and display
plt.savefig("lora_grad_norms_all.png", dpi=300)
plt.close()
