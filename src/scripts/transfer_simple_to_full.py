"""Transfer simple agents to full environment with weight expansion.

Loads agents trained in simple environment (no toxin/attack) and transfers
them to full environment by expanding weights to handle toxin input and
attack output. New weights initialized to zero.
"""

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
from jax import random, vmap
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation import Simulation, reset_timings, get_timings
from src.utils import make_key, init_directory, expand_simple_to_full_params
from src.agent import compute_param_dim as full_param_dim
from src.agent_simple import compute_param_dim as simple_param_dim

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


def load_and_expand_agents(checkpoint_path: str, num_agents: int, key: jax.Array,
                           hidden_dim: int, internal_noise_dim: int,
                           position_filter: tuple = None) -> jax.Array:
    """Load simple agents from checkpoint and expand to full agent params.

    Args:
        checkpoint_path: Path to simple agent checkpoint
        num_agents: Number of agents to sample
        key: PRNG key for random selection
        hidden_dim: Hidden dimension
        internal_noise_dim: Internal noise dimension
        position_filter: Optional (x_min, x_max, y_min, y_max) to filter by position

    Returns:
        (num_agents, full_param_dim) array of expanded agent parameters
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    state = checkpoint['state']
    alive = state['alive']
    params = state['params']
    positions = state['positions']

    # Validate param dimensions
    expected_simple = simple_param_dim(hidden_dim, internal_noise_dim)
    actual_dim = params.shape[1]
    if actual_dim != expected_simple:
        raise ValueError(
            f"FATAL: Checkpoint has param_dim={actual_dim}, expected simple agent dim={expected_simple}. "
            f"Use transfer.py for full→full transfer or transfer_simple.py for simple→simple."
        )

    # Get eligible agents
    alive_mask = alive
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
        raise ValueError(f"FATAL: Only {num_eligible} eligible agents in checkpoint, need {num_agents}")

    # Randomly sample
    perm = random.permutation(key, num_eligible)
    selected_indices = eligible_indices[perm[:num_agents]]
    selected_params = params[selected_indices]

    print(f"Loaded {len(selected_params)} simple agents from {checkpoint_path}")
    print(f"  (checkpoint was at step {checkpoint['step']} with {num_eligible} eligible)")

    # Expand to full agent params
    print(f"Expanding params: {expected_simple} → {full_param_dim(hidden_dim, internal_noise_dim)}")
    expand_batch = vmap(lambda p: expand_simple_to_full_params(p, hidden_dim, internal_noise_dim))
    expanded_params = expand_batch(selected_params)
    print(f"  Expanded {len(expanded_params)} agents")

    return expanded_params


def run_transfer_experiment(world_config: dict, run_config: dict, output_dir: Path,
                            agent_params: jax.Array, debug: bool = False) -> dict:
    """Run experiment with pre-loaded (and expanded) agents."""
    reset_timings()
    seed = run_config["seed"]
    max_steps = run_config["max_steps"]
    detailed_interval = run_config["logging"]["detailed_interval"]
    checkpoint_interval = run_config["logging"]["checkpoint_interval"]

    transfer_config = run_config["transfer"]
    initial_energy = transfer_config["initial_energy"]
    spawn_region = transfer_config.get("spawn_region", None)
    if spawn_region is not None:
        spawn_region = tuple(spawn_region)

    # Save configs
    with open(output_dir / "world_config.yaml", "w") as f:
        yaml.dump(world_config, f)
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f)

    # Initialize with expanded agents
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
        raise ValueError("FATAL: 'transfer' section required")

    transfer_config = run_config['transfer']
    if 'checkpoint_path' not in transfer_config:
        raise ValueError("FATAL: 'transfer.checkpoint_path' required")

    # Load world config
    world_config_path = run_config.get("world_config")
    if world_config_path is None:
        # Default to default.yaml for full simulation
        world_config_path = Path(__file__).parent.parent.parent / "configs" / "world" / "default.yaml"
    world_config = load_world_config(world_config_path)

    # Get agent config for param dimensions
    hidden_dim = world_config["agent"]["hidden_dim"]
    internal_noise_dim = world_config["agent"]["internal_noise_dim"]

    # Setup output directory
    output_dir = init_directory(run_config["output_dir"], overwrite=overwrite)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, output_dir / "config.yaml")

    # Load and expand agents
    key = make_key(run_config["seed"])
    load_key, run_key = random.split(key)

    checkpoint_path = transfer_config['checkpoint_path']
    num_agents = transfer_config.get('num_agents', 512)
    position_filter = transfer_config.get('position_filter', None)
    if position_filter is not None:
        position_filter = tuple(position_filter)

    agent_params = load_and_expand_agents(
        checkpoint_path, num_agents, load_key,
        hidden_dim, internal_noise_dim, position_filter
    )

    print(f"\nJAX devices: {jax.devices()}")
    print(f"Output dir: {output_dir}")
    print(f"World size: {world_config['world']['size']}")
    print(f"Arena type: {world_config.get('arena', {}).get('type', 'gaussian')}")
    print(f"Transferred agents: {len(agent_params)} (expanded from simple→full)")
    print(f"Max steps: {run_config['max_steps']}")
    print(f"Seed: {run_config['seed']}")
    print()

    state = run_transfer_experiment(world_config, run_config, output_dir, agent_params, debug=debug)

    stats = Simulation(world_config, run_config).get_stats(state)
    print(f"\nFinal: {stats['num_alive']} agents alive at step {stats['step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darwin Transfer: Simple → Full")
    parser.add_argument("config_path", type=str, help="Path to run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug timing")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
