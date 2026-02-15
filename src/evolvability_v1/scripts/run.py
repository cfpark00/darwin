"""Main runner script for evolvability_v1 simulations.

Usage:
    uv run python src/evolvability_v1/scripts/run.py configs/evolvability_v1/run/default.yaml
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
import pickle

import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import jax
from jax import random

from src.utils import init_directory, make_key
from src.evolvability_v1 import (
    AgentConfig, PhysicsConfig, RunConfig,
    Simulation,
)
from src.evolvability_v1.environment import ENVIRONMENT_REGISTRY


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_environment_from_config(config: dict):
    """Create environment from config dict using the registry.

    Extracts size and sensor noise from config, merges with environment-specific
    params, and instantiates via ENVIRONMENT_REGISTRY.
    """
    env_config = config.get("environment", {})
    env_type = env_config.get("type", "gaussian")
    size = config["world"]["size"]

    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"Unknown environment type: {env_type}. "
            f"Available: {list(ENVIRONMENT_REGISTRY.keys())}"
        )

    # Build kwargs from environment config
    kwargs = {k: v for k, v in env_config.items() if k != "type"}

    # Always pass size
    kwargs["size"] = size

    # Map sensor noise from config
    sensors = config.get("sensors", {})
    if "food_noise_std" in sensors:
        kwargs.setdefault("food_noise_std", sensors["food_noise_std"])
    if "temperature_noise_std" in sensors:
        kwargs.setdefault("temp_noise_std", sensors["temperature_noise_std"])
    if "toxin_noise_std" in sensors:
        kwargs.setdefault("toxin_noise_std", sensors["toxin_noise_std"])

    env_class = ENVIRONMENT_REGISTRY[env_type]
    return env_class(**kwargs)


def create_configs_from_yaml(config: dict, env) -> tuple:
    """Create AgentConfig, PhysicsConfig, RunConfig from YAML config.

    Args:
        config: YAML config dict
        env: Environment instance (used to get I/O dimensions)
    """
    # Get I/O dimensions from environment - no magic numbers!
    input_dim = env.input_dim
    output_dim = env.output_dim

    agent_config = AgentConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.get("agent", {}).get("hidden_dim", 8),
        internal_noise_dim=config.get("agent", {}).get("internal_noise_dim", 4),
    )

    # Parse disabled_actions as tuple
    disabled_actions = config.get("disabled_actions", [])
    if disabled_actions:
        disabled_actions = tuple(disabled_actions)
    else:
        disabled_actions = ()

    physics_config = PhysicsConfig(
        world_size=config["world"]["size"],
        max_energy=config.get("energy", {}).get("max", 100.0),
        base_cost=config.get("energy", {}).get("base_cost", 0.5),
        base_cost_incremental=config.get("energy", {}).get("base_cost_incremental", 0.01),
        cost_eat=config.get("energy", {}).get("cost_eat", 0.0),
        cost_move=config.get("energy", {}).get("cost_move", 2.5),
        cost_stay=config.get("energy", {}).get("cost_stay", 0.0),
        cost_reproduce=config.get("energy", {}).get("cost_reproduce", 29.5),
        cost_attack=config.get("energy", {}).get("cost_attack", 5.0),
        offspring_energy=config.get("energy", {}).get("offspring", 25.0),
        eat_fraction=config.get("resource", {}).get("eat_fraction", 0.33),
        regen_timescale=config.get("resource", {}).get("regen_timescale", 100.0),
        mutation_std=config.get("agent", {}).get("mutation_std", 0.1),
        # Clamp options for controlled experiments
        energy_clamp=config.get("energy", {}).get("clamp", None),
        resource_clamp=config.get("resource", {}).get("clamp", None),
        disabled_actions=disabled_actions,
    )

    run_config = RunConfig(
        seed=config.get("seed", 0),
        max_steps=config.get("max_steps", 10000),
        initial_agents=config.get("initial_agents", 512),
        min_buffer_size=config.get("min_buffer_size", 1024),
        log_interval=config.get("logging", {}).get("interval", 10),
        checkpoint_interval=config.get("logging", {}).get("checkpoint_interval", 1000),
        detailed_interval=config.get("logging", {}).get("detailed_interval", 100),
        buffer_growth_threshold=config.get("buffer_growth_threshold", 0.5),
    )

    return agent_config, physics_config, run_config


def save_checkpoint(state, step: int, output_dir: Path):
    """Save simulation state to checkpoint."""
    checkpoint = {
        "step": step,
        "state": {
            "positions": np.array(state.positions),
            "orientations": np.array(state.orientations),
            "params": np.array(state.params),
            "brain_states": np.array(state.brain_states),
            "energies": np.array(state.energies),
            "alive": np.array(state.alive),
            "ages": np.array(state.ages),
            "uid": np.array(state.uid),
            "parent_uid": np.array(state.parent_uid),
            "next_uid": int(state.next_uid),
            "max_agents": state.max_agents,
        },
        "world": {k: np.array(v) for k, v in state.world.items() if not k.startswith("_")},
    }

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    path = checkpoint_dir / f"step_{step:06d}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)

    # Also save as latest
    latest_path = checkpoint_dir / "latest.pkl"
    with open(latest_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint from file.

    Args:
        checkpoint_path: Path to checkpoint file (.pkl)

    Returns:
        Checkpoint dict with 'step', 'state', 'world' keys
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def plot_arena(state, step: int, stats: dict, output_dir: Path, resource_max: float = 30.0):
    """Save arena screenshot showing resources, toxin, and agent positions."""
    fig_dir = output_dir / "figures" / "snapshots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    size = state.world["resource"].shape[0]
    has_toxin = "toxin" in state.world and np.any(state.world["toxin"])

    fig, ax = plt.subplots(figsize=(10, 10))

    # Resource field
    extent = [0, size, 0, size]
    im = ax.imshow(
        np.array(state.world["resource"]),
        cmap="copper", vmin=0, vmax=resource_max,
        extent=extent, origin='lower'
    )

    # Toxin overlay (red, semi-transparent)
    if has_toxin:
        toxin = np.array(state.world["toxin"])
        toxin_rgba = np.zeros((size, size, 4))
        toxin_rgba[toxin, 0] = 1.0  # Red
        toxin_rgba[toxin, 3] = 0.6  # Alpha
        ax.imshow(toxin_rgba, extent=extent, origin='lower')

    # Agent positions
    alive = np.array(state.alive)
    positions = np.array(state.positions)[alive]
    if len(positions) > 0:
        ax.scatter(positions[:, 1], positions[:, 0], c="cyan", s=2, alpha=0.7)

    title = f"Step {step} | {stats['num_alive']} alive | avg E: {stats.get('mean_energy', 0):.1f}"
    if has_toxin:
        title += " | toxin: red"
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Resource")
    plt.tight_layout()
    plt.savefig(fig_dir / f"step_{step:06d}.png", dpi=100)
    plt.close()


def plot_population(history: dict, output_dir: Path):
    """Plot population over time."""
    fig_dir = output_dir / "figures"
    steps = history["steps"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, history["num_alive"], "b-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Population")
    ax.set_title("Population over time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "population.png", dpi=100)
    plt.close()


def plot_action_ratio(history: dict, output_dir: Path, action_names: list):
    """Plot action distribution over time."""
    fig_dir = output_dir / "figures"
    steps = history["steps"]

    if len(steps) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for name in action_names:
        key = f"action_{name}"
        if key in history:
            ax.plot(steps, history[key], label=name, linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Action ratio")
    ax.set_title("Action distribution over time")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "action_ratio.png", dpi=100)
    plt.close()


def plot_mean_energy(history: dict, output_dir: Path):
    """Plot mean energy over time."""
    fig_dir = output_dir / "figures"
    steps = history["steps"]

    if "mean_energy" not in history or len(steps) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, history["mean_energy"], "g-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Energy")
    ax.set_title("Mean energy over time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "mean_energy.png", dpi=100)
    plt.close()


def plot_mean_age(history: dict, output_dir: Path):
    """Plot mean age over time."""
    fig_dir = output_dir / "figures"
    steps = history["steps"]

    if "mean_age" not in history or len(steps) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, history["mean_age"], "m-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Age")
    ax.set_title("Mean age over time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "mean_age.png", dpi=100)
    plt.close()


def plot_births_deaths(history: dict, output_dir: Path):
    """Plot births and deaths per interval."""
    fig_dir = output_dir / "figures"
    steps = history["steps"]

    if "births" not in history or len(steps) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, history["births"], "g-", linewidth=1, label="Births")
    ax.plot(steps, history["deaths"], "r-", linewidth=1, label="Deaths")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count (per interval)")
    ax.set_title("Births and deaths over time")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "births_deaths.png", dpi=100)
    plt.close()


def plot_buffer_util(history: dict, output_dir: Path):
    """Plot buffer utilization over time."""
    fig_dir = output_dir / "figures"
    steps = history["steps"]

    if "buffer_util" not in history or len(steps) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, history["buffer_util"], "orange", linewidth=1)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="Growth threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Buffer Utilization")
    ax.set_title("Buffer utilization (num_alive / max_agents)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "buffer_util.png", dpi=100)
    plt.close()


def plot_energy_heatmap(history: dict, output_dir: Path, max_energy: float):
    """Plot energy distribution as heatmap (step on x-axis, energy on y-axis)."""
    fig_dir = output_dir / "figures"

    if "energy_hists" not in history or len(history["energy_hists"]) < 2:
        return  # Need at least 2 points for heatmap

    steps = [h[0] for h in history["energy_hists"]]
    hists = np.array([h[1] for h in history["energy_hists"]])  # (n_snapshots, n_bins)

    # Normalize each row (snapshot) to sum to 1
    row_sums = hists.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)  # avoid div by zero
    hists_norm = hists / row_sums

    # Transpose so x=steps, y=energy bins
    hists_T = hists_norm.T  # (n_bins, n_snapshots)

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [steps[0], steps[-1], 0, max_energy]  # x=steps, y=energy
    im = ax.imshow(hists_T, aspect='auto', cmap='viridis', extent=extent, origin='lower')
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy distribution over time")
    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()
    plt.savefig(fig_dir / "energy_heatmap.png", dpi=100)
    plt.close()


def plot_age_heatmap(history: dict, output_dir: Path, max_age_bin: int = 500):
    """Plot age distribution as heatmap (step on x-axis, age on y-axis)."""
    fig_dir = output_dir / "figures"

    if "age_hists" not in history or len(history["age_hists"]) < 2:
        return  # Need at least 2 points for heatmap

    steps = [h[0] for h in history["age_hists"]]
    hists = np.array([h[1] for h in history["age_hists"]])

    # Normalize each row
    row_sums = hists.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    hists_norm = hists / row_sums

    # Transpose so x=steps, y=age bins
    hists_T = hists_norm.T

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [steps[0], steps[-1], 0, max_age_bin]  # x=steps, y=age
    im = ax.imshow(hists_T, aspect='auto', cmap='plasma', extent=extent, origin='lower')
    ax.set_xlabel("Step")
    ax.set_ylabel("Age")
    ax.set_title("Age distribution over time")
    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()
    plt.savefig(fig_dir / "age_heatmap.png", dpi=100)
    plt.close()


def plot_y_density_heatmap(history: dict, output_dir: Path, world_size: int):
    """Plot y-position distribution as heatmap (step on x-axis, y-position on y-axis)."""
    fig_dir = output_dir / "figures"

    if "y_hists" not in history or len(history["y_hists"]) < 2:
        return  # Need at least 2 points for heatmap

    steps = [h[0] for h in history["y_hists"]]
    hists = np.array([h[1] for h in history["y_hists"]])

    # Normalize each row
    row_sums = hists.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    hists_norm = hists / row_sums

    # Transpose so x=steps, y=position bins
    hists_T = hists_norm.T

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [steps[0], steps[-1], 0, world_size]  # x=steps, y=position
    im = ax.imshow(hists_T, aspect='auto', cmap='Blues', extent=extent, origin='lower')
    ax.set_xlabel("Step")
    ax.set_ylabel("Y position (0=bottom)")
    ax.set_title("Y-position distribution over time")
    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()
    plt.savefig(fig_dir / "y_density_heatmap.png", dpi=100)
    plt.close()


def plot_x_density_heatmap(history: dict, output_dir: Path, world_size: int):
    """Plot x-position distribution as heatmap (step on x-axis, x-position on y-axis)."""
    fig_dir = output_dir / "figures"

    if "x_hists" not in history or len(history["x_hists"]) < 2:
        return  # Need at least 2 points for heatmap

    steps = [h[0] for h in history["x_hists"]]
    hists = np.array([h[1] for h in history["x_hists"]])

    # Normalize each row
    row_sums = hists.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    hists_norm = hists / row_sums

    # Transpose so x=steps, y=position bins
    hists_T = hists_norm.T

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [steps[0], steps[-1], 0, world_size]  # x=steps, y=position
    im = ax.imshow(hists_T, aspect='auto', cmap='Oranges', extent=extent, origin='lower')
    ax.set_xlabel("Step")
    ax.set_ylabel("X position (0=left)")
    ax.set_title("X-position distribution over time")
    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()
    plt.savefig(fig_dir / "x_density_heatmap.png", dpi=100)
    plt.close()


def collect_histograms(state, step: int, history: dict, max_energy: float, world_size: int,
                       n_energy_bins: int = 50, n_age_bins: int = 50, n_pos_bins: int = 50,
                       max_age_bin: int = 500):
    """Collect histogram snapshots for heatmaps."""
    alive = np.array(state.alive)
    n_alive = alive.sum()

    if n_alive == 0:
        # Empty histograms
        history["energy_hists"].append((step, np.zeros(n_energy_bins)))
        history["age_hists"].append((step, np.zeros(n_age_bins)))
        history["y_hists"].append((step, np.zeros(n_pos_bins)))
        history["x_hists"].append((step, np.zeros(n_pos_bins)))
        return

    # Energy histogram
    energies = np.array(state.energies)[alive]
    energy_hist, _ = np.histogram(energies, bins=n_energy_bins, range=(0, max_energy))
    history["energy_hists"].append((step, energy_hist))

    # Age histogram
    ages = np.array(state.ages)[alive]
    age_hist, _ = np.histogram(ages, bins=n_age_bins, range=(0, max_age_bin))
    history["age_hists"].append((step, age_hist))

    # Position histograms
    positions = np.array(state.positions)[alive]
    y_hist, _ = np.histogram(positions[:, 0], bins=n_pos_bins, range=(0, world_size))
    x_hist, _ = np.histogram(positions[:, 1], bins=n_pos_bins, range=(0, world_size))
    history["y_hists"].append((step, y_hist))
    history["x_hists"].append((step, x_hist))


def plot_all_live(history: dict, output_dir: Path, env, config: dict):
    """Generate all live-updating plots at detailed_interval."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    world_size = config["world"]["size"]
    max_energy = config.get("energy", {}).get("max", 100.0)

    # Time series plots
    plot_population(history, output_dir)
    plot_action_ratio(history, output_dir, env.action_names)
    plot_mean_energy(history, output_dir)
    plot_mean_age(history, output_dir)
    plot_births_deaths(history, output_dir)
    plot_buffer_util(history, output_dir)

    # Heatmap plots
    plot_energy_heatmap(history, output_dir, max_energy)
    plot_age_heatmap(history, output_dir)
    plot_y_density_heatmap(history, output_dir, world_size)
    plot_x_density_heatmap(history, output_dir, world_size)


def plot_stats(history: dict, output_dir: Path, env):
    """Generate final plots from run history (legacy, calls plot_all_live)."""
    # Just call the live plotting function for final plots
    # This is kept for backwards compatibility
    pass


def main(config_path: str, overwrite: bool = False, debug: bool = False, from_checkpoint: str = None):
    """Run simulation.

    Args:
        config_path: Path to config YAML
        overwrite: Whether to overwrite output directory
        debug: Whether to limit to 100 steps
        from_checkpoint: Optional path to checkpoint to load agents from
    """
    # Load config
    config = load_config(config_path)

    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    # Debug mode: limit to 100 steps for quick testing
    if debug:
        config["max_steps"] = 100
        print("DEBUG MODE: limiting to 100 steps")

    # Initialize output directory
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)

    # Create subdirectories
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Copy config to output
    shutil.copy(config_path, output_dir / "config.yaml")

    # Create environment and configs
    env = create_environment_from_config(config)
    agent_config, physics_config, run_config = create_configs_from_yaml(config, env)

    print(f"Environment: {type(env).__name__}")
    print(f"  Input dim: {env.input_dim}, Output dim: {env.output_dim}")
    print(f"  Capabilities: has_toxin={env.has_toxin}, has_attack={env.has_attack}")
    print(f"  World size: {physics_config.world_size}")
    print(f"Agent config: hidden_dim={agent_config.hidden_dim}")
    print(f"Run config: max_steps={run_config.max_steps}, initial_agents={run_config.initial_agents}")

    # Create simulation
    sim = Simulation(env, agent_config, physics_config, run_config)

    # Initialize
    key = make_key(run_config.seed)
    init_key, run_key = random.split(key)
    state = sim.reset(init_key)

    # Load from checkpoint if specified
    if from_checkpoint:
        print(f"Loading agents from checkpoint: {from_checkpoint}")
        ckpt = load_checkpoint(from_checkpoint)
        ckpt_state = ckpt["state"]

        # Check for spawn_region in config (for diffusion experiments)
        spawn_region = config.get("spawn_region", None)
        if spawn_region is not None:
            spawn_region = tuple(spawn_region)  # [y_min, y_max, x_min, x_max]
            print(f"  Spawn region: y=[{spawn_region[0]}, {spawn_region[1]}), x=[{spawn_region[2]}, {spawn_region[3]})")

        # Transfer agent state from checkpoint
        # Note: We keep the new world, but load agent params
        n_ckpt = ckpt_state["alive"].sum()
        n_current = state.max_agents

        if n_ckpt > n_current:
            print(f"  WARNING: Checkpoint has {n_ckpt} agents but buffer is {n_current}")
            print(f"  Only loading first {n_current} alive agents")

        # Copy agent data (positions, params, etc.)
        import jax.numpy as jnp
        alive_mask = ckpt_state["alive"]
        alive_indices = np.where(alive_mask)[0][:n_current]

        # Reset to use checkpoint agents
        new_positions = np.zeros((n_current, 2), dtype=np.int32)
        new_orientations = np.zeros(n_current, dtype=np.int32)
        new_params = np.zeros((n_current, state.params.shape[1]), dtype=np.float32)
        new_brain_states = np.zeros((n_current, state.brain_states.shape[1]), dtype=np.float32)
        new_energies = np.zeros(n_current, dtype=np.float32)
        new_alive = np.zeros(n_current, dtype=bool)
        new_ages = np.zeros(n_current, dtype=np.int32)

        n_loaded = min(len(alive_indices), n_current)

        # Generate positions: either in spawn_region or original positions
        if spawn_region is not None:
            y_min, y_max, x_min, x_max = spawn_region
            region_h = y_max - y_min
            region_w = x_max - x_min
            total_cells = region_h * region_w

            # Limit agents to spawn region capacity
            if n_loaded > total_cells:
                print(f"  Limiting agents to spawn_region capacity: {total_cells}")
                n_loaded = total_cells
                alive_indices = alive_indices[:n_loaded]

            # Random positions within spawn region (unique)
            rng = np.random.RandomState(run_config.seed)
            flat_indices = rng.choice(total_cells, size=n_loaded, replace=False)
            rel_y = flat_indices // region_w
            rel_x = flat_indices % region_w

            for i, idx in enumerate(alive_indices[:n_loaded]):
                new_positions[i, 0] = y_min + rel_y[i]
                new_positions[i, 1] = x_min + rel_x[i]
                new_orientations[i] = rng.randint(0, 4)  # Random orientations
                new_params[i] = ckpt_state["params"][idx]
                new_brain_states[i] = np.zeros_like(ckpt_state["brain_states"][idx])  # Fresh state
                new_energies[i] = config.get("energy", {}).get("initial", 50.0)
                new_alive[i] = True
                new_ages[i] = 0
        else:
            # Preserve original positions from checkpoint
            for i, idx in enumerate(alive_indices[:n_loaded]):
                new_positions[i] = ckpt_state["positions"][idx]
                new_orientations[i] = ckpt_state["orientations"][idx]
                new_params[i] = ckpt_state["params"][idx]
                new_brain_states[i] = ckpt_state["brain_states"][idx]
                new_energies[i] = ckpt_state["energies"][idx]
                new_alive[i] = True
                new_ages[i] = 0  # Reset ages

        state = state._replace(
            positions=jnp.array(new_positions),
            orientations=jnp.array(new_orientations),
            params=jnp.array(new_params),
            brain_states=jnp.array(new_brain_states),
            energies=jnp.array(new_energies),
            alive=jnp.array(new_alive),
            ages=jnp.array(new_ages),
        )
        print(f"  Loaded {n_loaded} agents from checkpoint")

    print(f"Initialized with {int(state.alive.sum())} agents")

    # History for logging
    history = {
        "steps": [],
        "num_alive": [],
        "mean_energy": [],
        "mean_age": [],
        "births": [],
        "deaths": [],
        "buffer_util": [],
        # Histogram snapshots for heatmaps (collected at detailed_interval)
        "energy_hists": [],
        "age_hists": [],
        "y_hists": [],
        "x_hists": [],
    }
    for name in env.action_names:
        history[f"action_{name}"] = []

    # Track previous alive count for births/deaths
    prev_alive = run_config.initial_agents

    # Log file
    log_path = output_dir / "logs" / "run.jsonl"

    # Run simulation
    # Metrics are computed inside JIT, so get_stats() is now just device->host transfer.
    # We can afford to sync at log_interval without major performance penalty.

    with open(log_path, 'w') as log_file:
        pbar = tqdm(range(run_config.max_steps), desc="Simulating")

        for step in pbar:
            # Step (metrics computed inside JIT)
            run_key, step_key = random.split(run_key)
            state = sim.step(state, step_key)

            # Log at log_interval (sync is cheap now - metrics pre-computed in JIT)
            if step % run_config.log_interval == 0:
                stats = sim.get_stats(state)
                pbar.set_postfix(pop=stats["num_alive"])

                # Warn if unphysical events occurred
                if stats.get("repro_capped", 0) > 0:
                    print(f"\n  WARNING: {stats['repro_capped']} reproductions capped (unphysical)")

                # Log to file
                log_file.write(json.dumps(stats) + "\n")

                # Compute births/deaths from population change
                current_alive = stats["num_alive"]
                # births - deaths = current - prev
                # We can't separate them perfectly without tracking inside JIT,
                # but we can estimate: if pop increased, net births; if decreased, net deaths
                net_change = current_alive - prev_alive
                births = max(0, net_change)
                deaths = max(0, -net_change)

                # Update history
                history["steps"].append(step)
                history["num_alive"].append(stats["num_alive"])
                history["mean_energy"].append(stats.get("mean_energy", 0))
                history["mean_age"].append(stats.get("mean_age", 0))
                history["births"].append(births)
                history["deaths"].append(deaths)
                history["buffer_util"].append(stats["num_alive"] / stats["max_agents"])
                for name in env.action_names:
                    history[f"action_{name}"].append(stats.get(f"action_{name}", 0))

                prev_alive = current_alive

                # Check extinction
                if stats["num_alive"] == 0:
                    print(f"Population extinct at step {step}")
                    break

            # Checkpoint
            if step % run_config.checkpoint_interval == 0 and step > 0:
                save_checkpoint(state, step, output_dir)

            # Detailed logging at detailed_interval (arena + all live plots)
            if step % run_config.detailed_interval == 0:
                if step % run_config.log_interval != 0:
                    # Only compute stats if we didn't already
                    stats = sim.get_stats(state)
                resource_max = config.get("environment", {}).get("resource_max", 30.0)
                max_energy = config.get("energy", {}).get("max", 100.0)
                world_size = config["world"]["size"]

                # Arena screenshot
                plot_arena(state, step, stats, output_dir, resource_max)

                # Collect histogram snapshots for heatmaps
                collect_histograms(state, step, history, max_energy, world_size)

                # Generate all live-updating plots
                plot_all_live(history, output_dir, env, config)

    # Final checkpoint
    save_checkpoint(state, state.step, output_dir)

    # Final histogram collection and plots
    max_energy = config.get("energy", {}).get("max", 100.0)
    world_size = config["world"]["size"]
    final_stats = sim.get_stats(state)
    collect_histograms(state, state.step, history, max_energy, world_size)
    plot_all_live(history, output_dir, env, config)

    print(f"Done! Output saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evolvability_v1 simulation")
    parser.add_argument("config_path", type=str, help="Path to config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--from-checkpoint", type=str, default=None,
                        help="Path to checkpoint to load agents from")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug, args.from_checkpoint)
