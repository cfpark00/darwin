"""Transfer experiment - load evolved agents into a new environment."""

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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation import Simulation, reset_timings, get_timings
from src.utils import make_key, init_directory
from src.physics import EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE, ATTACK
from src.agent import compute_param_dim

# Import the run functions we can reuse
from src.scripts.run import (
    ACTION_NAMES, compute_action_counts, log_base, log_detailed, save_checkpoint
)


def load_world_config(path: str) -> dict:
    """Load world configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_run_config(path: str) -> dict:
    """Load run configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_agents_from_checkpoint(checkpoint_path: str, num_agents: int, key: jax.Array,
                                 expected_param_dim: int,
                                 position_filter: tuple = None) -> jax.Array:
    """Load random agents from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint pickle file
        num_agents: Number of agents to sample
        key: PRNG key for random selection
        expected_param_dim: Expected parameter dimension (MUST match checkpoint)
        position_filter: Optional (x_min, x_max, y_min, y_max) to filter by position

    Returns:
        (num_agents, param_dim) array of agent parameters

    Raises:
        ValueError: If checkpoint param_dim doesn't match expected_param_dim
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    state = checkpoint['state']
    alive = state['alive']
    params = state['params']
    positions = state['positions']  # (N, 2) array of [y, x]

    # FAIL FAST: Validate param dimensions match
    actual_param_dim = params.shape[1]
    if actual_param_dim != expected_param_dim:
        raise ValueError(
            f"FATAL: Checkpoint param_dim mismatch!\n"
            f"  Checkpoint has: {actual_param_dim}\n"
            f"  Expected (full agent): {expected_param_dim}\n"
            f"  This likely means you're trying to load a simple agent checkpoint (no toxin/attack)\n"
            f"  into a full simulation. Use transfer_simple_to_full.py instead."
        )

    # Get indices of alive agents
    alive_mask = alive

    # Apply position filter if specified
    if position_filter is not None:
        x_min, x_max, y_min, y_max = position_filter
        x_coords = positions[:, 1]
        y_coords = positions[:, 0]
        in_region = (x_coords >= x_min) & (x_coords < x_max) & (y_coords >= y_min) & (y_coords < y_max)
        alive_mask = alive & in_region
        print(f"Position filter: x=[{x_min}, {x_max}), y=[{y_min}, {y_max})")

    eligible_indices = jnp.where(alive_mask)[0]
    num_eligible = len(eligible_indices)

    if num_eligible < num_agents:
        raise ValueError(f"Only {num_eligible} eligible agents in checkpoint, need {num_agents}")

    # Randomly sample num_agents from eligible agents
    perm = random.permutation(key, num_eligible)
    selected_indices = eligible_indices[perm[:num_agents]]

    selected_params = params[selected_indices]
    print(f"Loaded {len(selected_params)} agents from {checkpoint_path}")
    print(f"  (checkpoint was at step {checkpoint['step']} with {num_eligible} eligible)")

    return selected_params


def run_transfer_experiment(world_config: dict, run_config: dict, output_dir: Path,
                            agent_params: jax.Array, debug: bool = False) -> dict:
    """Run experiment with pre-loaded agents."""
    reset_timings()
    seed = run_config["seed"]
    max_steps = run_config["max_steps"]
    detailed_interval = run_config["logging"]["detailed_interval"]
    checkpoint_interval = run_config["logging"]["checkpoint_interval"]

    # Get transfer config options
    transfer_config = run_config["transfer"]
    initial_energy = transfer_config["initial_energy"]
    spawn_region = transfer_config.get("spawn_region", None)
    if spawn_region is not None:
        spawn_region = tuple(spawn_region)  # Convert list to tuple

    # Save configs to output dir
    with open(output_dir / "world_config.yaml", "w") as f:
        yaml.dump(world_config, f)
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f)

    # Initialize with loaded agents
    sim = Simulation(world_config, run_config, debug=debug)
    key = make_key(seed)
    init_key, run_key = random.split(key)
    state = sim.reset_with_agents(init_key, agent_params, initial_energy=initial_energy,
                                   spawn_region=spawn_region)

    # History tracking
    history = {
        "steps": [],
        "population": [],
        "actions": [],
        "attacks": [],
        "kills": [],
        "toxin_deaths": [],
    }

    # Open base log file
    log_file = open(output_dir / "logs" / "base_log.jsonl", "w")

    # Step 0: run actual step to get real actions
    run_key, step0_key = random.split(run_key)
    state = sim.step(state, step0_key)
    reset_timings()

    # Log step 0
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

    # Main loop
    pbar = tqdm(range(1, max_steps + 1), desc="Simulating", unit="step")

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
            history["attacks"].append(action_counts["attack"])
            history["kills"].append(state.get("num_kills", 0))
            history["toxin_deaths"].append(state.get("num_toxin_deaths", 0))

            pbar.set_postfix({
                "alive": stats["num_alive"],
                "avg_E": f"{stats['avg_energy']:.1f}"
            })

            log_base(step, state, stats, actions, output_dir, log_file)

            if stats["num_alive"] == 0:
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
    """Main entry point."""
    run_config = load_run_config(config_path)

    if 'output_dir' not in run_config:
        raise ValueError("FATAL: 'output_dir' required in config")

    if 'transfer' not in run_config:
        raise ValueError("FATAL: 'transfer' section required for transfer experiment")

    transfer_config = run_config['transfer']
    if 'checkpoint_path' not in transfer_config:
        raise ValueError("FATAL: 'transfer.checkpoint_path' required")

    # Load world config
    world_config_path = run_config.get("world_config")
    if world_config_path is None:
        raise ValueError("FATAL: 'world_config' path required for transfer experiment")
    world_config = load_world_config(world_config_path)

    # Compute expected param dimension for validation
    hidden_dim = world_config["agent"]["hidden_dim"]
    internal_noise_dim = world_config["agent"]["internal_noise_dim"]
    expected_param_dim = compute_param_dim(hidden_dim, internal_noise_dim)

    # Setup output directory
    output_dir = init_directory(run_config["output_dir"], overwrite=overwrite)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy(config_path, output_dir / "config.yaml")

    # Load agents from checkpoint
    key = make_key(run_config["seed"])
    load_key, run_key = random.split(key)

    checkpoint_path = transfer_config['checkpoint_path']
    num_agents = transfer_config.get('num_agents', 100)
    position_filter = transfer_config.get('position_filter', None)
    if position_filter is not None:
        position_filter = tuple(position_filter)
    agent_params = load_agents_from_checkpoint(checkpoint_path, num_agents, load_key,
                                                expected_param_dim, position_filter)

    print(f"JAX devices: {jax.devices()}")
    print(f"Output dir: {output_dir}")
    print(f"World size: {world_config['world']['size']}")
    print(f"Arena type: {world_config.get('arena', {}).get('type', 'default')}")
    print(f"Transferred agents: {len(agent_params)}")
    print(f"Max steps: {run_config['max_steps']}")
    print(f"Seed: {run_config['seed']}")
    print()

    state = run_transfer_experiment(world_config, run_config, output_dir, agent_params, debug=debug)

    stats = Simulation(world_config, run_config).get_stats(state)
    print(f"\nFinal: {stats['num_alive']} agents alive at step {stats['step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darwin Transfer Experiment")
    parser.add_argument("config_path", type=str, help="Path to run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug timing")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
