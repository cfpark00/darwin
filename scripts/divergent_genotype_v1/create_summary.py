"""Create summary figure of all findings."""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("data/divergent_genotype_v1")

fig = plt.figure(figsize=(16, 10))

# 1. Diffusion comparison (top left)
ax1 = fig.add_subplot(2, 3, 1)
for name, exp_name in [("c5", "diffusion_c5"), ("c17", "diffusion_c17")]:
    exp_dir = OUTPUT_DIR / exp_name
    ckpts = sorted((exp_dir / "checkpoints").glob("ckpt_*.pkl"))
    steps, msd = [], []
    for ckpt_path in ckpts:
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        steps.append(ckpt['step'])
        pos = np.array(ckpt['state']['positions'])
        if len(msd) == 0:
            init_pos = pos
        disp = pos - init_pos
        disp = np.where(disp > 128, disp - 256, disp)
        disp = np.where(disp < -128, disp + 256, disp)
        msd.append((disp ** 2).sum(axis=1).mean())
    D = np.polyfit(steps[2:], msd[2:], 1)[0] / 4
    ax1.plot(steps, msd, 'o-', label=f"Cluster {name} (D={D:.2f})", markersize=4)
ax1.set_xlabel("Step")
ax1.set_ylabel("Mean Squared Displacement")
ax1.set_title("Diffusion: Cluster 17 moves 3.5x more")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Final positions diffusion (top middle)
ax2 = fig.add_subplot(2, 3, 2)
for name, exp_name, color in [("c5", "diffusion_c5", "tab:blue"), ("c17", "diffusion_c17", "tab:orange")]:
    with open(OUTPUT_DIR / exp_name / "checkpoints" / "ckpt_001000.pkl", 'rb') as f:
        ckpt = pickle.load(f)
    pos = np.array(ckpt['state']['positions'])
    ax2.scatter(pos[:, 1], pos[:, 0], s=10, alpha=0.5, label=f"Cluster {name}", c=color)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Diffusion Final Positions (step 1000)")
ax2.legend()
ax2.set_xlim(0, 256)
ax2.set_ylim(0, 256)

# 3. Action distribution (top right)
ax3 = fig.add_subplot(2, 3, 3)
actions = ["eat", "forward", "left", "right", "stay"]
ACTION_NAMES = ["eat", "forward", "left", "right", "stay", "reproduce"]
x = np.arange(len(actions))
width = 0.35

for i, (name, exp_name) in enumerate([("c5", "diffusion_c5"), ("c17", "diffusion_c17")]):
    log_path = OUTPUT_DIR / exp_name / "logs" / "base_log.jsonl"
    action_counts = {a: 0 for a in ACTION_NAMES}
    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            for a in ACTION_NAMES:
                action_counts[a] += entry['actions'].get(a, 0)
    total = sum(action_counts[a] for a in actions)
    pcts = [100 * action_counts[a] / total for a in actions]
    ax3.bar(x + (i - 0.5) * width, pcts, width, label=f'Cluster {name}')
ax3.set_ylabel('Fraction (%)')
ax3.set_title('Action Distribution')
ax3.set_xticks(x)
ax3.set_xticklabels(actions)
ax3.legend()

# 4. Key insight text (bottom left)
ax4 = fig.add_subplot(2, 3, 4)
ax4.axis('off')
text = """
KEY FINDING: Genetic Divergence → Behavioral Divergence

Cluster 5 "Circler":
• High left-turn rate (6.2%)
• Spins in place
• Low net displacement (D=0.23)

Cluster 17 "Linear Explorer":
• Almost no turns
• Goes mostly forward (21.8%)
• High displacement (D=0.81)

Diffusion ratio: 3.5x difference!
"""
ax4.text(0.1, 0.5, text, fontsize=11, family='monospace', verticalalignment='center')

# 5. Chemotaxis comparison (bottom middle)
ax5 = fig.add_subplot(2, 3, 5)
for name, exp_name, color in [("c5", "chemotaxis_c5", "tab:blue"), ("c17", "chemotaxis_c17", "tab:orange")]:
    with open(OUTPUT_DIR / exp_name / "checkpoints" / "ckpt_002000.pkl", 'rb') as f:
        ckpt = pickle.load(f)
    pos = np.array(ckpt['state']['positions'])
    ax5.scatter(pos[:, 1], pos[:, 0], s=10, alpha=0.5, label=f"Cluster {name}", c=color)
ax5.axvline(128, color='gray', linestyle='--', alpha=0.5)
ax5.set_xlabel("X (gradient: 0→30)")
ax5.set_ylabel("Y")
ax5.set_title("Chemotaxis Final Positions")
ax5.legend()
ax5.set_xlim(0, 256)
ax5.set_ylim(0, 256)

# 6. X-position histogram (bottom right)
ax6 = fig.add_subplot(2, 3, 6)
for name, exp_name, color in [("c5", "chemotaxis_c5", "tab:blue"), ("c17", "chemotaxis_c17", "tab:orange")]:
    with open(OUTPUT_DIR / exp_name / "checkpoints" / "ckpt_002000.pkl", 'rb') as f:
        ckpt = pickle.load(f)
    x_pos = np.array(ckpt['state']['positions'])[:, 1]
    ax6.hist(x_pos, bins=30, alpha=0.5, label=f"Cluster {name}", color=color)
ax6.axvline(128, color='gray', linestyle='--', alpha=0.5, label='Start')
ax6.set_xlabel("X Position")
ax6.set_ylabel("Count")
ax6.set_title("Chemotaxis: X Distribution")
ax6.legend()

plt.suptitle("Divergent Genotype Analysis: Cluster 5 vs Cluster 17", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "summary.png", dpi=150)
print(f"Saved to {OUTPUT_DIR / 'summary.png'}")
