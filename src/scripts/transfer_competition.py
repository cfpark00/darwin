"""Competition experiment: two genotype clusters head-to-head.

Loads agents from two different clusters and runs them in the same arena.
Tracks population by lineage over time using genetic distance classification.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation_simple import SimulationSimple, reset_timings, get_timings
from src.utils import make_key, init_directory
from src.physics import EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE
from src.agent_simple import compute_param_dim

from src.scripts.run_simple import (
    ACTION_NAMES, compute_action_counts, log_base, log_detailed, save_checkpoint
)


def load_world_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_run_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_agents_from_cluster(checkpoint_path: str, cluster_assignments_path: str,
                              cluster_id: int, num_agents: int, key: jax.Array,
                              expected_param_dim: int) -> tuple:
    """Load agents from a specific genotype cluster.

    Returns:
        (selected_params, cluster_centroid) - params and mean param vector for lineage tracking
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    state = checkpoint['state']
    params = state['params']

    actual_param_dim = params.shape[1]
    if actual_param_dim != expected_param_dim:
        raise ValueError(
            f"FATAL: Checkpoint param_dim mismatch!\n"
            f"  Checkpoint has: {actual_param_dim}\n"
            f"  Expected: {expected_param_dim}"
        )

    cluster_data = np.load(cluster_assignments_path)
    cluster_labels = cluster_data['cluster_labels']
    alive_indices = cluster_data['alive_indices']

    cluster_mask = cluster_labels == cluster_id
    cluster_alive_indices = alive_indices[cluster_mask]

    num_in_cluster = len(cluster_alive_indices)
    if num_in_cluster < num_agents:
        raise ValueError(f"Only {num_in_cluster} agents in cluster {cluster_id}, need {num_agents}")

    # Get all params for this cluster to compute centroid
    all_cluster_params = params[cluster_alive_indices]
    cluster_centroid = np.mean(all_cluster_params, axis=0)

    # Randomly select subset
    perm = random.permutation(key, num_in_cluster)
    selected_indices = cluster_alive_indices[np.array(perm[:num_agents])]
    selected_params = params[selected_indices]

    return selected_params, cluster_centroid


def classify_by_lineage(params: np.ndarray, alive: np.ndarray,
                        centroid_a: np.ndarray, centroid_b: np.ndarray) -> tuple:
    """Classify alive agents by genetic distance to cluster centroids.

    Returns:
        (count_a, count_b) - number of agents closer to each centroid
    """
    alive_params = params[alive]
    if len(alive_params) == 0:
        return 0, 0

    # Compute squared distances to each centroid
    dist_a = np.sum((alive_params - centroid_a) ** 2, axis=1)
    dist_b = np.sum((alive_params - centroid_b) ** 2, axis=1)

    count_a = np.sum(dist_a < dist_b)
    count_b = np.sum(dist_b <= dist_a)

    return int(count_a), int(count_b)


def run_competition(world_config: dict, world_config_path: str, run_config: dict,
                    output_dir: Path, params_a: jax.Array, params_b: jax.Array,
                    centroid_a: np.ndarray, centroid_b: np.ndarray,
                    cluster_id_a: int, cluster_id_b: int,
                    debug: bool = False) -> dict:
    """Run competition experiment with agents from two clusters."""
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

    # Combine agents from both clusters
    combined_params = jnp.concatenate([params_a, params_b], axis=0)
    num_agents = len(combined_params)

    sim = SimulationSimple(world_config, run_config, debug=debug)
    key = make_key(seed)
    init_key, run_key = random.split(key)
    state = sim.reset_with_agents(init_key, combined_params, initial_energy=initial_energy,
                                   spawn_region=spawn_region)

    # Track lineage history
    lineage_history = {
        "steps": [],
        f"cluster_{cluster_id_a}": [],
        f"cluster_{cluster_id_b}": [],
        "total": [],
    }

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
    lineage_file = open(output_dir / "logs" / "lineage_log.jsonl", "w")

    run_key, step0_key = random.split(run_key)
    state = sim.step(state, step0_key)
    reset_timings()

    actions = state["actions"]
    stats = sim.get_stats(state)

    # Initial lineage classification
    count_a, count_b = classify_by_lineage(
        np.array(state["params"]), np.array(state["alive"]),
        centroid_a, centroid_b
    )
    lineage_history["steps"].append(0)
    lineage_history[f"cluster_{cluster_id_a}"].append(count_a)
    lineage_history[f"cluster_{cluster_id_b}"].append(count_b)
    lineage_history["total"].append(stats["num_alive"])
    lineage_file.write(json.dumps({
        "step": 0,
        f"cluster_{cluster_id_a}": count_a,
        f"cluster_{cluster_id_b}": count_b,
        "total": stats["num_alive"]
    }) + "\n")

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

    pbar = tqdm(range(1, max_steps + 1), desc="Competition", unit="step")

    for step in pbar:
        run_key, step_key = random.split(run_key)
        state = sim.step(state, step_key)

        if step % 100 == 0:
            actions = state["actions"]
            stats = sim.get_stats(state)

            # Classify lineage
            count_a, count_b = classify_by_lineage(
                np.array(state["params"]), np.array(state["alive"]),
                centroid_a, centroid_b
            )
            lineage_history["steps"].append(step)
            lineage_history[f"cluster_{cluster_id_a}"].append(count_a)
            lineage_history[f"cluster_{cluster_id_b}"].append(count_b)
            lineage_history["total"].append(stats["num_alive"])
            lineage_file.write(json.dumps({
                "step": step,
                f"cluster_{cluster_id_a}": count_a,
                f"cluster_{cluster_id_b}": count_b,
                "total": stats["num_alive"]
            }) + "\n")

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
                f"c{cluster_id_a}": count_a,
                f"c{cluster_id_b}": count_b,
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
    lineage_file.close()
    save_checkpoint(state["step"], state, sim, output_dir)

    # Save lineage history
    with open(output_dir / "results" / "lineage_history.json", "w") as f:
        json.dump(lineage_history, f, indent=2)

    # Plot lineage dynamics
    plot_lineage_dynamics(lineage_history, cluster_id_a, cluster_id_b, output_dir)

    return state, lineage_history


def plot_lineage_dynamics(lineage_history: dict, cluster_id_a: int, cluster_id_b: int, output_dir: Path):
    """Plot population dynamics by lineage."""
    steps = lineage_history["steps"]
    pop_a = lineage_history[f"cluster_{cluster_id_a}"]
    pop_b = lineage_history[f"cluster_{cluster_id_b}"]
    total = lineage_history["total"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Absolute populations
    ax1 = axes[0]
    ax1.plot(steps, pop_a, label=f"Cluster {cluster_id_a}", color="blue", linewidth=2)
    ax1.plot(steps, pop_b, label=f"Cluster {cluster_id_b}", color="red", linewidth=2)
    ax1.plot(steps, total, label="Total", color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Population")
    ax1.set_title("Population by Lineage (Absolute)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative fractions
    ax2 = axes[1]
    frac_a = [a / t if t > 0 else 0.5 for a, t in zip(pop_a, total)]
    frac_b = [b / t if t > 0 else 0.5 for b, t in zip(pop_b, total)]
    ax2.fill_between(steps, 0, frac_a, alpha=0.7, label=f"Cluster {cluster_id_a}", color="blue")
    ax2.fill_between(steps, frac_a, 1, alpha=0.7, label=f"Cluster {cluster_id_b}", color="red")
    ax2.axhline(y=0.5, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Fraction")
    ax2.set_title("Population by Lineage (Fraction)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "lineage_dynamics.png", dpi=150)
    plt.close()


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    run_config = load_run_config(config_path)

    if 'output_dir' not in run_config:
        raise ValueError("FATAL: 'output_dir' required in config")

    if 'transfer' not in run_config:
        raise ValueError("FATAL: 'transfer' section required")

    transfer_config = run_config['transfer']
    required_fields = ['checkpoint_path', 'cluster_assignments_path',
                       'cluster_id_a', 'cluster_id_b', 'num_agents_per_cluster']
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
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, output_dir / "config.yaml")

    key = make_key(run_config["seed"])
    key_a, key_b, run_key = random.split(key, 3)

    checkpoint_path = transfer_config['checkpoint_path']
    cluster_assignments_path = transfer_config['cluster_assignments_path']
    cluster_id_a = transfer_config['cluster_id_a']
    cluster_id_b = transfer_config['cluster_id_b']
    num_agents = transfer_config['num_agents_per_cluster']

    print(f"Loading {num_agents} agents from cluster {cluster_id_a}...")
    params_a, centroid_a = load_agents_from_cluster(
        checkpoint_path, cluster_assignments_path, cluster_id_a,
        num_agents, key_a, expected_param_dim
    )

    print(f"Loading {num_agents} agents from cluster {cluster_id_b}...")
    params_b, centroid_b = load_agents_from_cluster(
        checkpoint_path, cluster_assignments_path, cluster_id_b,
        num_agents, key_b, expected_param_dim
    )

    print(f"\nJAX devices: {jax.devices()}")
    print(f"Output dir: {output_dir}")
    print(f"World size: {world_config['world']['size']}")
    print(f"Arena type: {world_config.get('arena', {}).get('type', 'unknown')}")
    print(f"Cluster A: {cluster_id_a} ({len(params_a)} agents)")
    print(f"Cluster B: {cluster_id_b} ({len(params_b)} agents)")
    print(f"Total agents: {len(params_a) + len(params_b)}")
    print(f"Max steps: {run_config['max_steps']}")
    print(f"Seed: {run_config['seed']}")
    print()

    state, lineage_history = run_competition(
        world_config, world_config_path, run_config, output_dir,
        params_a, params_b, centroid_a, centroid_b,
        cluster_id_a, cluster_id_b, debug=debug
    )

    stats = SimulationSimple(world_config, run_config).get_stats(state)
    final_a = lineage_history[f"cluster_{cluster_id_a}"][-1]
    final_b = lineage_history[f"cluster_{cluster_id_b}"][-1]
    print(f"\nFinal: {stats['num_alive']} agents alive")
    print(f"  Cluster {cluster_id_a}: {final_a} ({100*final_a/max(stats['num_alive'],1):.1f}%)")
    print(f"  Cluster {cluster_id_b}: {final_b} ({100*final_b/max(stats['num_alive'],1):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darwin Competition Experiment")
    parser.add_argument("config_path", type=str, help="Path to run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug timing")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
