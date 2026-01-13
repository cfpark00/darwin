"""Analyze diffusion experiments - compare MSD between clusters."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("data/divergent_genotype_v1")
EXPERIMENTS = {
    "c5": "diffusion_c5",
    "c17": "diffusion_c17",
}

def load_checkpoints(exp_dir):
    """Load all checkpoints and extract positions over time."""
    ckpt_dir = exp_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pkl"))

    steps = []
    positions_over_time = []

    for ckpt_path in ckpts:
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        step = ckpt['step']
        positions = np.array(ckpt['state']['positions'])
        steps.append(step)
        positions_over_time.append(positions)

    return np.array(steps), np.array(positions_over_time)

def compute_msd(positions_over_time):
    """Compute mean squared displacement from initial positions."""
    initial_pos = positions_over_time[0]
    msd = []
    for pos in positions_over_time:
        displacement = pos - initial_pos
        # Handle periodic boundary (256x256)
        displacement = np.where(displacement > 128, displacement - 256, displacement)
        displacement = np.where(displacement < -128, displacement + 256, displacement)
        squared_disp = (displacement ** 2).sum(axis=1)
        msd.append(squared_disp.mean())
    return np.array(msd)

# Load and analyze
results = {}
for name, exp_name in EXPERIMENTS.items():
    exp_dir = OUTPUT_DIR / exp_name
    steps, positions = load_checkpoints(exp_dir)
    msd = compute_msd(positions)
    results[name] = {"steps": steps, "msd": msd}
    print(f"{name}: {len(steps)} checkpoints, final MSD = {msd[-1]:.1f}")

# Compute diffusion coefficients (MSD = 4Dt for 2D)
print("\nDiffusion coefficients (D = MSD / 4t):")
for name, data in results.items():
    # Use linear fit on later points (exclude initial transient)
    t = data["steps"][2:]
    msd = data["msd"][2:]
    # D = slope / 4 for 2D diffusion
    slope = np.polyfit(t, msd, 1)[0]
    D = slope / 4
    print(f"  {name}: D = {D:.3f}")
    results[name]["D"] = D

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MSD over time
ax = axes[0]
for name, data in results.items():
    ax.plot(data["steps"], data["msd"], 'o-', label=f"Cluster {name} (D={data['D']:.2f})", markersize=4)
ax.set_xlabel("Step")
ax.set_ylabel("Mean Squared Displacement")
ax.set_title("MSD over Time")
ax.legend()
ax.grid(True, alpha=0.3)

# Final position distributions
ax = axes[1]
colors = {"c5": "tab:blue", "c17": "tab:orange"}
for name, exp_name in EXPERIMENTS.items():
    exp_dir = OUTPUT_DIR / exp_name
    with open(exp_dir / "checkpoints" / "ckpt_001000.pkl", 'rb') as f:
        ckpt = pickle.load(f)
    positions = np.array(ckpt['state']['positions'])
    ax.scatter(positions[:, 1], positions[:, 0], s=10, alpha=0.5,
               label=f"Cluster {name}", c=colors[name])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Final Positions (step 1000)")
ax.legend()
ax.set_xlim(0, 256)
ax.set_ylim(0, 256)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diffusion_comparison.png", dpi=150)
print(f"\nSaved to {OUTPUT_DIR / 'diffusion_comparison.png'}")
