import numpy as np
import matplotlib.pyplot as plt

# Equation: total_params = 2 * V * d_model + 28 * L * d_model^2
def total_params(d_model, L, V=32000):
    return 2 * V * d_model + 28 * L * (d_model ** 2)

# Constants
L_values = [6, 12, 24, 30, 48, 64, 80, 96]
d_model = np.linspace(0, 4096, 600)

# Plot setup
plt.figure(figsize=(9, 6))
for L in L_values:
    params = total_params(d_model, L)
    plt.plot(d_model, params / 1e9, label=f"L = {L}")

# Styling
#plt.title("Total Parameters vs d_model", fontsize=15, weight='bold')
plt.xlabel("d_model (embedding dimension)", fontsize=14)
plt.ylabel("Total Parameters (billions)", fontsize=14)
plt.legend(title="Layers (L)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Optional: limit the range for better visibility
plt.xlim(0, 4096)
plt.ylim(bottom=0)

plt.savefig("plot_params.pdf", dpi=300)
plt.savefig("plot_params.png", dpi=300)
# Show
plt.close()

