"""Experiment harness - CLI, logging, checkpointing, hardware management."""

import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import yaml

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation import Simulation, reset_timings, get_timings
from src.utils import make_key, init_directory
from src.physics import EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE, ATTACK


ACTION_NAMES = ["eat", "forward", "left", "right", "stay", "reproduce", "attack"]


def load_world_config(path: str = None) -> dict:
    """Load world configuration."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "world" / "default.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_run_config(path: str) -> dict:
    """Load run configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_action_counts(state: dict, actions: jax.Array) -> dict:
    """Count actions taken by alive agents."""
    alive = state["alive"]
    n = min(len(actions), len(alive))  # Handle size mismatch from offspring
    actions = actions[:n]
    alive_subset = alive[:n]
    counts = {}
    for i, name in enumerate(ACTION_NAMES):
        counts[name] = int(jnp.sum((actions == i) & alive_subset))
    return counts


def log_base(step: int, state: dict, stats: dict, actions: jax.Array,
             output_dir: Path, log_file) -> dict:
    """Base logging - every step. Returns log entry."""
    action_counts = compute_action_counts(state, actions)

    entry = {
        "step": step,
        "num_alive": stats["num_alive"],
        "total_agents": stats["total_agents"],
        "avg_energy": round(stats["avg_energy"], 2),
        "max_energy": round(stats["max_energy"], 2),
        "min_energy": round(stats["min_energy"], 2),
        "actions": action_counts,
    }

    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()

    return entry


def log_detailed(step: int, state: dict, stats: dict, output_dir: Path, world_config: dict,
                 history: dict):
    """Detailed logging - plot arena and time series."""
    size = world_config["world"]["size"]
    max_energy = world_config["energy"]["max"]

    # Collect energy snapshot for histogram
    alive = state["alive"]
    energies = np.array(state["energies"][alive])
    if "energy_snapshots" not in history:
        history["energy_snapshots"] = []
    history["energy_snapshots"].append((step, energies))
    # Keep only last 9
    if len(history["energy_snapshots"]) > 9:
        history["energy_snapshots"] = history["energy_snapshots"][-9:]

    # Energy histogram 3x3 panel
    if len(history["energy_snapshots"]) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.flatten()
        snapshots = history["energy_snapshots"]
        for i in range(9):
            ax = axes[i]
            if i < len(snapshots):
                snap_step, snap_energies = snapshots[i]
                if len(snap_energies) > 0:
                    ax.hist(snap_energies, bins=20, range=(0, max_energy), color="steelblue", edgecolor="black", alpha=0.7)
                ax.set_title(f"Step {snap_step}", fontsize=10)
                ax.set_xlim(0, max_energy)
            else:
                ax.set_visible(False)
            if i >= 6:  # Bottom row
                ax.set_xlabel("Energy")
            if i % 3 == 0:  # Left column
                ax.set_ylabel("Count")
        plt.suptitle("Energy Distribution (last 9 snapshots)", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "energy_histograms.png", dpi=100)
        plt.close()

    # Arena plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    extent = [0, size, 0, size]  # [left, right, bottom, top]
    resource_max = state["world"]["resource_max"]
    im = ax.imshow(state["world"]["resource"], cmap="copper", vmin=0, vmax=resource_max, extent=extent, origin='lower')
    ax.imshow(state["world"]["toxin"], cmap="Reds", alpha=0.5 * state["world"]["toxin"], extent=extent, origin='lower')
    alive = state["alive"]
    positions = state["positions"][alive]
    if len(positions) > 0:
        ax.scatter(positions[:, 1], positions[:, 0], c="cyan", s=2, alpha=0.7)
    ax.set_title(f"Step {step} | {stats['num_alive']} alive | avg E: {stats['avg_energy']:.1f}")
    plt.colorbar(im, ax=ax, label="Resource")
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / f"step_{step:06d}.png", dpi=100)
    plt.close()

    # Population plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(history["steps"], history["population"], "b-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Population")
    ax.set_title("Population over time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "population.png", dpi=100)
    plt.close()

    # Action ratio plot
    if len(history["steps"]) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        steps = np.array(history["steps"])
        action_history = np.array(history["actions"])  # (n_steps, 7)
        totals = action_history.sum(axis=1, keepdims=True)
        totals = np.maximum(totals, 1)  # avoid div by zero
        ratios = action_history / totals
        for i, name in enumerate(ACTION_NAMES):
            ax.plot(steps, ratios[:, i], label=name, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Action ratio")
        ax.set_title("Action distribution over time")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "action_ratio.png", dpi=100)
        plt.close()

    # Attack stats plot
    if len(history["attacks"]) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        steps = np.array(history["steps"])
        attacks = np.array(history["attacks"])
        kills = np.array(history["kills"])

        # Attack count over time
        ax1.plot(steps, attacks, "r-", linewidth=1, label="Attacks")
        ax1.plot(steps, kills, "k-", linewidth=1, label="Kills")
        ax1.set_ylabel("Count")
        ax1.set_title("Attack Activity")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Success ratio over time
        with np.errstate(divide='ignore', invalid='ignore'):
            success_ratio = np.where(attacks > 0, kills / attacks, 0.0)
        ax2.plot(steps, success_ratio, "g-", linewidth=1)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Success Ratio")
        ax2.set_title("Attack Success Rate (kills / attacks)")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "attack_stats.png", dpi=100)
        plt.close()

    # Toxin deaths plot
    if len(history["toxin_deaths"]) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        steps = np.array(history["steps"])
        toxin_deaths = np.array(history["toxin_deaths"])

        ax.plot(steps, toxin_deaths, "purple", linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Toxin Deaths")
        ax.set_title("Deaths from Toxin per Step")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "toxin_deaths.png", dpi=100)
        plt.close()

    # Y-density plot (agent distribution along y-axis)
    alive = state["alive"]
    positions = state["positions"][alive]
    if len(positions) > 0:
        y_positions = np.array(positions[:, 0])
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.hist(y_positions, bins=50, range=(0, size), color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Y position (0=bottom, max=top)")
        ax.set_ylabel("Agent count")
        ax.set_title(f"Agent Y-Distribution | Step {step} | {len(y_positions)} agents")
        ax.set_xlim(0, size)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "y_density.png", dpi=100)
        plt.close()


def save_checkpoint(step: int, state: dict, sim: Simulation, output_dir: Path):
    """Save full checkpoint."""
    checkpoint = {
        "step": step,
        "state": state,
    }
    with open(output_dir / "checkpoints" / f"ckpt_{step:06d}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def run_experiment(world_config: dict, run_config: dict, output_dir: Path, debug: bool = False) -> dict:
    """Run a full experiment."""
    reset_timings()
    seed = run_config["seed"]
    max_steps = run_config["max_steps"]
    detailed_interval = run_config["logging"]["detailed_interval"]
    checkpoint_interval = run_config["logging"]["checkpoint_interval"]

    # Save configs to output dir
    with open(output_dir / "world_config.yaml", "w") as f:
        yaml.dump(world_config, f)
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f)

    # Initialize
    sim = Simulation(world_config, run_config, debug=debug)
    key = make_key(seed)
    init_key, run_key = random.split(key)
    state = sim.reset(init_key)

    # History tracking
    history = {
        "steps": [],
        "population": [],
        "actions": [],  # list of [eat, forward, left, right, stay, reproduce, attack] counts
        "attacks": [],  # number of attack actions
        "kills": [],    # number of successful kills
        "toxin_deaths": [],  # number of deaths from toxin
    }

    # Open base log file
    log_file = open(output_dir / "logs" / "base_log.jsonl", "w")

    # Step 0: run actual step to get real actions (also serves as JIT warmup)
    run_key, step0_key = random.split(run_key)
    state = sim.step(state, step0_key)
    reset_timings()  # Clear timing from warmup

    # Log step 0 with real actions
    actions = state["actions"]
    stats = sim.get_stats(state)
    log_base(0, state, stats, actions, output_dir, log_file)

    # Update history
    history["steps"].append(0)
    history["population"].append(stats["num_alive"])
    action_counts = compute_action_counts(state, actions)
    history["actions"].append([action_counts[name] for name in ACTION_NAMES])
    history["attacks"].append(action_counts["attack"])
    history["kills"].append(state.get("num_kills", 0))
    history["toxin_deaths"].append(state.get("num_toxin_deaths", 0))

    log_detailed(0, state, stats, output_dir, world_config, history)
    save_checkpoint(0, state, sim, output_dir)

    # Main loop with tqdm
    pbar = tqdm(range(1, max_steps + 1), desc="Simulating", unit="step")
    all_dead = False

    for step in pbar:
        run_key, step_key = random.split(run_key)
        state = sim.step(state, step_key)
        # Note: buffer growth now happens automatically inside sim.step() when needed

        # Only do expensive stats/logging every N steps to avoid GPU sync overhead
        if step % 100 == 0:
            actions = state["actions"]
            stats = sim.get_stats(state)

            # Update history
            history["steps"].append(step)
            history["population"].append(stats["num_alive"])
            action_counts = compute_action_counts(state, actions)
            history["actions"].append([action_counts[name] for name in ACTION_NAMES])
            history["attacks"].append(action_counts["attack"])
            history["kills"].append(state.get("num_kills", 0))
            history["toxin_deaths"].append(state.get("num_toxin_deaths", 0))

            pbar.set_postfix({
                "alive": stats["num_alive"],
                "avg_E": f"{stats['avg_energy']:.1f}"
            })

            # Write timing to file
            timings = get_timings()
            if timings:
                total = sum(sum(t) for t in timings.values())
                timing_lines = [f"Step {step} timing (avg over {len(list(timings.values())[0])} steps)\n"]
                for name, times in sorted(timings.items(), key=lambda x: -sum(x[1])):
                    pct = 100 * sum(times) / total if total > 0 else 0
                    timing_lines.append(f"  {name}: {sum(times)/len(times)*1000:.1f}ms/step ({pct:.1f}%)\n")
                with open(output_dir / "logs" / "timing_live.txt", "w") as tf:
                    tf.writelines(timing_lines)

            # Log base stats (every 100 steps now)
            log_base(step, state, stats, actions, output_dir, log_file)

            # Check for extinction (only sync every 100 steps)
            if stats["num_alive"] == 0:
                all_dead = True
                tqdm.write(f"All agents died at step {step}")
                log_detailed(step, state, stats, output_dir, world_config, history)
                break

        # Detailed logging
        if step % detailed_interval == 0:
            if step % 100 != 0:  # Only compute if not already done above
                actions = state["actions"]
                stats = sim.get_stats(state)
            log_detailed(step, state, stats, output_dir, world_config, history)

        # Checkpoint
        if step % checkpoint_interval == 0:
            save_checkpoint(step, state, sim, output_dir)

    log_file.close()

    # Final checkpoint
    save_checkpoint(state["step"], state, sim, output_dir)

    # Save timing info (only if debug mode collected timing data)
    timings = get_timings()
    if timings:
        with open(output_dir / "logs" / "timings.json", "w") as f:
            timing_summary = {}
            for name, times in timings.items():
                timing_summary[name] = {
                    "total": sum(times),
                    "mean": sum(times) / len(times) if times else 0,
                    "count": len(times),
                }
            json.dump(timing_summary, f, indent=2)

        # Print timing summary
        print("\nTiming summary (seconds):")
        total_time = sum(sum(t) for t in timings.values())
        for name, times in sorted(timings.items(), key=lambda x: -sum(x[1])):
            pct = 100 * sum(times) / total_time if total_time > 0 else 0
            print(f"  {name}: {sum(times):.2f}s ({pct:.1f}%) - {sum(times)/len(times)*1000:.1f}ms/step")

    return state


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    """Main entry point."""
    # Load run config
    run_config = load_run_config(config_path)

    # Validate required fields
    if 'output_dir' not in run_config:
        raise ValueError("FATAL: 'output_dir' required in config")

    # Load world config (from run_config or default)
    world_config_path = run_config.get("world_config")
    world_config = load_world_config(world_config_path)

    # Setup output directory using the safe utility
    output_dir = init_directory(run_config["output_dir"], overwrite=overwrite)

    # Create standard subdirectories
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # Copy config to output directory for reproducibility
    shutil.copy(config_path, output_dir / "config.yaml")

    print(f"JAX devices: {jax.devices()}")
    print(f"Output dir: {output_dir}")
    print(f"World size: {world_config['world']['size']}")
    print(f"Initial agents: {world_config['world']['initial_agents']}")
    print(f"Max steps: {run_config['max_steps']}")
    print(f"Seed: {run_config['seed']}")
    print(f"Debug timing: {debug}")
    print()

    state = run_experiment(world_config, run_config, output_dir, debug=debug)

    stats = Simulation(world_config, run_config).get_stats(state)
    print(f"\nFinal: {stats['num_alive']} agents alive at step {stats['step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darwin ALife Simulation")
    parser.add_argument("config_path", type=str, help="Path to run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug timing (slower, breaks JIT fusion)")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
