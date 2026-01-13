"""Analyze chemotaxis experiments - compare gradient following between clusters."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("data/divergent_genotype_v1")
EXPERIMENTS = {
    "c5": "chemotaxis_c5",
    "c17": "chemotaxis_c17",
}

def load_checkpoints(exp_dir):
    """Load all checkpoints and extract x-positions over time."""
    ckpt_dir = exp_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pkl"))

    steps = []
    mean_x = []
    std_x = []

    for ckpt_path in ckpts:
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        step = ckpt['step']
        positions = np.array(ckpt['state']['positions'])
        x_pos = positions[:, 1]  # x is column index
        steps.append(step)
        mean_x.append(x_pos.mean())
        std_x.append(x_pos.std())

    return np.array(steps), np.array(mean_x), np.array(std_x)

# Load and analyze
results = {}
for name, exp_name in EXPERIMENTS.items():
    exp_dir = OUTPUT_DIR / exp_name
    steps, mean_x, std_x = load_checkpoints(exp_dir)
    results[name] = {"steps": steps, "mean_x": mean_x, "std_x": std_x}
    print(f"{name}: initial x={mean_x[0]:.1f}, final x={mean_x[-1]:.1f}, delta={mean_x[-1]-mean_x[0]:.1f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Mean x over time
ax = axes[0]
for name, data in results.items():
    ax.plot(data["steps"], data["mean_x"], 'o-', label=f"Cluster {name}", markersize=4)
ax.axhline(128, color='gray', linestyle='--', alpha=0.5, label='Center')
ax.set_xlabel("Step")
ax.set_ylabel("Mean X Position")
ax.set_title("Chemotaxis: Movement Toward High Resources (right)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(100, 200)

# Final x distribution
ax = axes[1]
colors = {"c5": "tab:blue", "c17": "tab:orange"}
for name, exp_name in EXPERIMENTS.items():
    exp_dir = OUTPUT_DIR / exp_name
    with open(exp_dir / "checkpoints" / "ckpt_002000.pkl", 'rb') as f:
        ckpt = pickle.load(f)
    x_pos = np.array(ckpt['state']['positions'])[:, 1]
    ax.hist(x_pos, bins=30, alpha=0.5, label=f"Cluster {name}", color=colors[name])
ax.axvline(128, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("X Position")
ax.set_ylabel("Count")
ax.set_title("Final X Distribution (step 2000)")
ax.legend()

# Final positions scatter
ax = axes[2]
for name, exp_name in EXPERIMENTS.items():
    exp_dir = OUTPUT_DIR / exp_name
    with open(exp_dir / "checkpoints" / "ckpt_002000.pkl", 'rb') as f:
        ckpt = pickle.load(f)
    positions = np.array(ckpt['state']['positions'])
    ax.scatter(positions[:, 1], positions[:, 0], s=10, alpha=0.5,
               label=f"Cluster {name}", c=colors[name])
ax.axvline(128, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("X (gradient: 0→30)")
ax.set_ylabel("Y")
ax.set_title("Final Positions (step 2000)")
ax.legend()
ax.set_xlim(0, 256)
ax.set_ylim(0, 256)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chemotaxis_comparison.png", dpi=150)
print(f"\nSaved to {OUTPUT_DIR / 'chemotaxis_comparison.png'}")

# Summary stats
print("\n" + "="*50)
print("CHEMOTAXIS SUMMARY")
print("="*50)
print("Gradient: resources increase from left (0) to right (30)")
print("Start position: center (x=128)")
print()
for name, data in results.items():
    delta = data["mean_x"][-1] - data["mean_x"][0]
    print(f"Cluster {name}: moved {delta:+.1f} pixels toward {'high' if delta > 0 else 'low'} resources")
