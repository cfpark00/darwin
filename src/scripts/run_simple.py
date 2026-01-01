"""Experiment harness for simple environments (no toxin, no attack)."""

import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation_simple import SimulationSimple, reset_timings, get_timings
from src.utils import make_key, init_directory
from src.physics import EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE


ACTION_NAMES = ["eat", "forward", "left", "right", "stay", "reproduce"]


def load_world_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent.parent / "configs" / "world" / "default.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_run_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_action_counts(state: dict, actions: jax.Array) -> dict:
    alive = state["alive"]
    n = min(len(actions), len(alive))
    actions = actions[:n]
    alive_subset = alive[:n]
    counts = {}
    for i, name in enumerate(ACTION_NAMES):
        counts[name] = int(jnp.sum((actions == i) & alive_subset))
    return counts


def log_base(step: int, state: dict, stats: dict, actions: jax.Array,
             output_dir: Path, log_file) -> dict:
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
    size = world_config["world"]["size"]
    max_energy = world_config["energy"]["max"]

    # Energy histogram
    alive = state["alive"]
    energies = np.array(state["energies"][alive])
    if "energy_snapshots" not in history:
        history["energy_snapshots"] = []
    history["energy_snapshots"].append((step, energies))
    if len(history["energy_snapshots"]) > 9:
        history["energy_snapshots"] = history["energy_snapshots"][-9:]

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
            if i >= 6:
                ax.set_xlabel("Energy")
            if i % 3 == 0:
                ax.set_ylabel("Count")
        plt.suptitle("Energy Distribution (last 9 snapshots)", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "energy_histograms.png", dpi=100)
        plt.close()

    # Arena plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    extent = [0, size, 0, size]
    resource_max = state["world"]["resource_max"]
    im = ax.imshow(state["world"]["resource"], cmap="copper", vmin=0, vmax=resource_max, extent=extent, origin='lower')
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
        action_history = np.array(history["actions"])
        totals = action_history.sum(axis=1, keepdims=True)
        totals = np.maximum(totals, 1)
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

    # Y-density heatmap over time
    if "y_density" in history and len(history["y_density"]) > 1:
        y_density_array = np.array(history["y_density"]).T  # Shape: (y_bins, num_steps)
        steps_array = np.array(history["steps"])

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        # Normalize each column (timestep) to show density distribution
        col_sums = y_density_array.sum(axis=0, keepdims=True)
        col_sums = np.maximum(col_sums, 1)  # Avoid division by zero
        y_density_norm = y_density_array / col_sums

        im = ax.imshow(
            y_density_norm,
            aspect='auto',
            origin='lower',
            cmap='hot',
            extent=[steps_array[0], steps_array[-1], 0, size],
            interpolation='nearest'
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Y position")
        ax.set_title("Agent Y-Distribution Over Time (normalized density)")
        plt.colorbar(im, ax=ax, label="Density")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "y_density.png", dpi=150)
        plt.close()


def save_checkpoint(step: int, state: dict, sim: SimulationSimple, output_dir: Path):
    checkpoint = {
        "step": step,
        "state": state,
    }
    with open(output_dir / "checkpoints" / f"ckpt_{step:06d}.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def run_experiment(world_config: dict, run_config: dict, output_dir: Path, debug: bool = False) -> dict:
    reset_timings()
    seed = run_config["seed"]
    max_steps = run_config["max_steps"]
    detailed_interval = run_config["logging"]["detailed_interval"]
    checkpoint_interval = run_config["logging"]["checkpoint_interval"]

    with open(output_dir / "world_config.yaml", "w") as f:
        yaml.dump(world_config, f)
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f)

    sim = SimulationSimple(world_config, run_config, debug=debug)
    key = make_key(seed)
    init_key, run_key = random.split(key)
    state = sim.reset(init_key)

    history = {
        "steps": [],
        "population": [],
        "actions": [],
        "y_density": [],  # List of y-position histograms over time
    }
    y_bins = 64  # Number of bins for y-position histogram

    log_file = open(output_dir / "logs" / "base_log.jsonl", "w")

    # Step 0
    run_key, step0_key = random.split(run_key)
    state = sim.step(state, step0_key)
    reset_timings()

    actions = state["actions"]
    stats = sim.get_stats(state)
    log_base(0, state, stats, actions, output_dir, log_file)

    history["steps"].append(0)
    history["population"].append(stats["num_alive"])
    action_counts = compute_action_counts(state, actions)
    history["actions"].append([action_counts[name] for name in ACTION_NAMES])
    # Record y-density histogram
    alive = state["alive"]
    y_positions = np.array(state["positions"][alive][:, 0])
    y_hist, _ = np.histogram(y_positions, bins=y_bins, range=(0, world_config["world"]["size"]))
    history["y_density"].append(y_hist)

    log_detailed(0, state, stats, output_dir, world_config, history)
    save_checkpoint(0, state, sim, output_dir)

    pbar = tqdm(range(1, max_steps + 1), desc="Simulating", unit="step")
    all_dead = False

    for step in pbar:
        run_key, step_key = random.split(run_key)
        state = sim.step(state, step_key)

        if step % 100 == 0:
            actions = state["actions"]
            stats = sim.get_stats(state)

            history["steps"].append(step)
            history["population"].append(stats["num_alive"])
            action_counts = compute_action_counts(state, actions)
            history["actions"].append([action_counts[name] for name in ACTION_NAMES])
            # Record y-density histogram
            alive = state["alive"]
            y_positions = np.array(state["positions"][alive][:, 0])
            y_hist, _ = np.histogram(y_positions, bins=y_bins, range=(0, world_config["world"]["size"]))
            history["y_density"].append(y_hist)

            pbar.set_postfix({
                "alive": stats["num_alive"],
                "avg_E": f"{stats['avg_energy']:.1f}"
            })

            log_base(step, state, stats, actions, output_dir, log_file)

            if stats["num_alive"] == 0:
                all_dead = True
                tqdm.write(f"All agents died at step {step}")
                log_detailed(step, state, stats, output_dir, world_config, history)
                break

        if step % detailed_interval == 0:
            if step % 100 != 0:
                actions = state["actions"]
                stats = sim.get_stats(state)
            log_detailed(step, state, stats, output_dir, world_config, history)

        if step % checkpoint_interval == 0:
            save_checkpoint(step, state, sim, output_dir)

    log_file.close()
    save_checkpoint(state["step"], state, sim, output_dir)

    return state


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    run_config = load_run_config(config_path)

    if 'output_dir' not in run_config:
        raise ValueError("FATAL: 'output_dir' required in config")

    world_config_path = run_config.get("world_config")
    world_config = load_world_config(world_config_path)

    output_dir = init_directory(run_config["output_dir"], overwrite=overwrite)

    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, output_dir / "config.yaml")

    print(f"JAX devices: {jax.devices()}")
    print(f"Output dir: {output_dir}")
    print(f"World size: {world_config['world']['size']}")
    print(f"Arena type: {world_config.get('arena', {}).get('type', 'unknown')}")
    print(f"Initial agents: {world_config['world']['initial_agents']}")
    print(f"Max steps: {run_config['max_steps']}")
    print(f"Seed: {run_config['seed']}")
    print()

    state = run_experiment(world_config, run_config, output_dir, debug=debug)

    stats = SimulationSimple(world_config, run_config).get_stats(state)
    print(f"\nFinal: {stats['num_alive']} agents alive at step {stats['step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darwin Simple Simulation (no toxin/attack)")
    parser.add_argument("config_path", type=str, help="Path to run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug timing")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
