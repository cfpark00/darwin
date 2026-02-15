"""Transfer experiment filtering agents by genotype cluster.

Loads agents from a checkpoint, filters by cluster ID using pre-computed
cluster assignments, and runs them in a new environment.
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
from jax import random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.darwin_v0.simulation_simple import SimulationSimple, reset_timings, get_timings
from src.utils import make_key, init_directory
from src.darwin_v0.physics import EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE
from src.darwin_v0.agent_simple import compute_param_dim

from src.darwin_v0.scripts.run_simple import (
    ACTION_NAMES, compute_action_counts, log_base, log_detailed, save_checkpoint
)


def load_world_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_run_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_agents_by_cluster(checkpoint_path: str, cluster_assignments_path: str,
                            cluster_id: int, num_agents: int, key: jax.Array,
                            expected_param_dim: int) -> jax.Array:
    """Load agents from a specific genotype cluster.

    Args:
        checkpoint_path: Path to checkpoint pickle
        cluster_assignments_path: Path to cluster_assignments.npz
        cluster_id: Which cluster to load agents from
        num_agents: How many agents to load
        key: PRNG key for random selection
        expected_param_dim: Expected parameter dimension for validation

    Returns:
        Selected agent parameters (num_agents, param_dim)
    """
    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    state = checkpoint['state']
    alive = state['alive']
    params = state['params']

    # Validate param dimensions
    actual_param_dim = params.shape[1]
    if actual_param_dim != expected_param_dim:
        raise ValueError(
            f"FATAL: Checkpoint param_dim mismatch!\n"
            f"  Checkpoint has: {actual_param_dim}\n"
            f"  Expected: {expected_param_dim}"
        )

    # Load cluster assignments
    cluster_data = np.load(cluster_assignments_path)
    cluster_labels = cluster_data['cluster_labels']
    alive_indices = cluster_data['alive_indices']

    # Find agents in the specified cluster
    cluster_mask = cluster_labels == cluster_id
    cluster_alive_indices = alive_indices[cluster_mask]

    num_in_cluster = len(cluster_alive_indices)
    if num_in_cluster < num_agents:
        raise ValueError(f"Only {num_in_cluster} agents in cluster {cluster_id}, need {num_agents}")

    # Randomly select agents from cluster
    perm = random.permutation(key, num_in_cluster)
    selected_indices = cluster_alive_indices[perm[:num_agents]]

    selected_params = params[selected_indices]
    print(f"Loaded {len(selected_params)} agents from cluster {cluster_id}")
    print(f"  (cluster has {num_in_cluster} total agents)")
    print(f"  (checkpoint step: {checkpoint['step']})")

    return selected_params


def run_transfer_experiment(world_config: dict, world_config_path: str, run_config: dict,
                            output_dir: Path, agent_params: jax.Array, debug: bool = False) -> dict:
    """Run experiment with pre-loaded agents."""
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

    shutil.copy(world_config_path, output_dir / "world_config.yaml")

    sim = SimulationSimple(world_config, run_config, debug=debug)
    key = make_key(seed)
    init_key, run_key = random.split(key)
    state = sim.reset_with_agents(init_key, agent_params, initial_energy=initial_energy,
                                   spawn_region=spawn_region)

    history = {
        "steps": [],
        "population": [],
        "actions": [],
        "y_density": [],
        "x_density": [],
    }
    y_bins = 64
    x_bins = 64
    world_size = world_config["world"]["size"]

    log_file = open(output_dir / "logs" / "base_log.jsonl", "w")

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

    alive = state["alive"]
    y_positions = np.array(state["positions"][alive][:, 0])
    x_positions = np.array(state["positions"][alive][:, 1])
    y_hist, _ = np.histogram(y_positions, bins=y_bins, range=(0, world_size))
    x_hist, _ = np.histogram(x_positions, bins=x_bins, range=(0, world_size))
    history["y_density"].append(y_hist)
    history["x_density"].append(x_hist)

    log_detailed(0, state, stats, output_dir, world_config, history)
    save_checkpoint(0, state, sim, output_dir)

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

            alive = state["alive"]
            y_positions = np.array(state["positions"][alive][:, 0])
            x_positions = np.array(state["positions"][alive][:, 1])
            y_hist, _ = np.histogram(y_positions, bins=y_bins, range=(0, world_size))
            x_hist, _ = np.histogram(x_positions, bins=x_bins, range=(0, world_size))
            history["y_density"].append(y_hist)
            history["x_density"].append(x_hist)

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
    run_config = load_run_config(config_path)

    if 'output_dir' not in run_config:
        raise ValueError("FATAL: 'output_dir' required in config")

    if 'transfer' not in run_config:
        raise ValueError("FATAL: 'transfer' section required")

    transfer_config = run_config['transfer']
    required_fields = ['checkpoint_path', 'cluster_assignments_path', 'cluster_id']
    for field in required_fields:
        if field not in transfer_config:
            raise ValueError(f"FATAL: 'transfer.{field}' required")

    world_config_path = run_config.get("world_config")
    if world_config_path is None:
        raise ValueError("FATAL: 'world_config' path required")
    world_config = load_world_config(world_config_path)

    # Compute expected param dimension
    hidden_dim = world_config["agent"]["hidden_dim"]
    internal_noise_dim = world_config["agent"]["internal_noise_dim"]
    expected_param_dim = compute_param_dim(hidden_dim, internal_noise_dim)

    output_dir = init_directory(run_config["output_dir"], overwrite=overwrite)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, output_dir / "config.yaml")

    key = make_key(run_config["seed"])
    load_key, run_key = random.split(key)

    checkpoint_path = transfer_config['checkpoint_path']
    cluster_assignments_path = transfer_config['cluster_assignments_path']
    cluster_id = transfer_config['cluster_id']
    num_agents = transfer_config.get('num_agents', 256)

    agent_params = load_agents_by_cluster(
        checkpoint_path, cluster_assignments_path, cluster_id,
        num_agents, load_key, expected_param_dim
    )

    print(f"JAX devices: {jax.devices()}")
    print(f"Output dir: {output_dir}")
    print(f"World size: {world_config['world']['size']}")
    print(f"Arena type: {world_config.get('arena', {}).get('type', 'unknown')}")
    print(f"Cluster ID: {cluster_id}")
    print(f"Transferred agents: {len(agent_params)}")
    print(f"Max steps: {run_config['max_steps']}")
    print(f"Seed: {run_config['seed']}")
    print()

    state = run_transfer_experiment(world_config, world_config_path, run_config, output_dir, agent_params, debug=debug)

    stats = SimulationSimple(world_config, run_config).get_stats(state)
    print(f"\nFinal: {stats['num_alive']} agents alive at step {stats['step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darwin Transfer by Cluster")
    parser.add_argument("config_path", type=str, help="Path to run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug timing")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
