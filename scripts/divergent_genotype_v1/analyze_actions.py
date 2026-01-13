"""Analyze action distributions from diffusion experiments."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("data/divergent_genotype_v1")
EXPERIMENTS = {
    "c5": "diffusion_c5",
    "c17": "diffusion_c17",
}

ACTION_NAMES = ["eat", "forward", "left", "right", "stay", "reproduce"]

def load_action_log(exp_dir):
    """Load action counts from base_log.jsonl."""
    log_path = exp_dir / "logs" / "base_log.jsonl"

    steps = []
    action_counts = {name: [] for name in ACTION_NAMES}

    with open(log_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            steps.append(entry['step'])
            for name in ACTION_NAMES:
                action_counts[name].append(entry['actions'].get(name, 0))

    return np.array(steps), {k: np.array(v) for k, v in action_counts.items()}

# Load and analyze
results = {}
for name, exp_name in EXPERIMENTS.items():
    exp_dir = OUTPUT_DIR / exp_name
    steps, actions = load_action_log(exp_dir)
    results[name] = {"steps": steps, "actions": actions}

# Compute action fractions (excluding reproduce since it's disabled)
print("Action distribution (fraction of total actions):\n")
print(f"{'Action':<12} | {'Cluster 5':>12} | {'Cluster 17':>12} | {'Diff':>10}")
print("-" * 52)

for action in ACTION_NAMES[:5]:  # Exclude reproduce
    frac_c5 = results["c5"]["actions"][action].sum()
    frac_c17 = results["c17"]["actions"][action].sum()
    total_c5 = sum(results["c5"]["actions"][a].sum() for a in ACTION_NAMES[:5])
    total_c17 = sum(results["c17"]["actions"][a].sum() for a in ACTION_NAMES[:5])

    pct_c5 = 100 * frac_c5 / total_c5
    pct_c17 = 100 * frac_c17 / total_c17
    diff = pct_c17 - pct_c5

    print(f"{action:<12} | {pct_c5:>11.1f}% | {pct_c17:>11.1f}% | {diff:>+9.1f}%")

# Movement fraction
move_actions = ["forward", "left", "right"]
move_c5 = sum(results["c5"]["actions"][a].sum() for a in move_actions)
move_c17 = sum(results["c17"]["actions"][a].sum() for a in move_actions)
total_c5 = sum(results["c5"]["actions"][a].sum() for a in ACTION_NAMES[:5])
total_c17 = sum(results["c17"]["actions"][a].sum() for a in ACTION_NAMES[:5])

print("-" * 52)
print(f"{'MOVEMENT':<12} | {100*move_c5/total_c5:>11.1f}% | {100*move_c17/total_c17:>11.1f}% | {100*(move_c17/total_c17 - move_c5/total_c5):>+9.1f}%")

# Plot action distributions over time
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, action in enumerate(ACTION_NAMES[:5]):
    ax = axes[i]
    for name in ["c5", "c17"]:
        data = results[name]
        total = sum(data["actions"][a] for a in ACTION_NAMES[:5])
        frac = data["actions"][action] / np.maximum(total, 1)
        ax.plot(data["steps"], frac * 100, label=f"Cluster {name}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction (%)")
    ax.set_title(f"'{action}' Action")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Summary bar chart
ax = axes[5]
actions_plot = ["eat", "forward", "left", "right", "stay"]
x = np.arange(len(actions_plot))
width = 0.35

pcts_c5 = []
pcts_c17 = []
for action in actions_plot:
    total_c5 = sum(results["c5"]["actions"][a].sum() for a in ACTION_NAMES[:5])
    total_c17 = sum(results["c17"]["actions"][a].sum() for a in ACTION_NAMES[:5])
    pcts_c5.append(100 * results["c5"]["actions"][action].sum() / total_c5)
    pcts_c17.append(100 * results["c17"]["actions"][action].sum() / total_c17)

ax.bar(x - width/2, pcts_c5, width, label='Cluster 5')
ax.bar(x + width/2, pcts_c17, width, label='Cluster 17')
ax.set_ylabel('Fraction (%)')
ax.set_title('Action Distribution Summary')
ax.set_xticks(x)
ax.set_xticklabels(actions_plot)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "action_comparison.png", dpi=150)
print(f"\nSaved to {OUTPUT_DIR / 'action_comparison.png'}")
