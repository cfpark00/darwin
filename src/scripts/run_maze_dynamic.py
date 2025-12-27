"""Dynamic maze experiment - goal moves when populated."""

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

from src.scripts.run import (
    ACTION_NAMES, compute_action_counts, log_base, log_detailed, save_checkpoint
)


# Passable cells in maze coords (i, j) where i=column, j=row from top
# Excludes start cell (0, 7)
PASSABLE_CELLS = [
    # Top rows - good goal candidates
    (7, 0), (6, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0),  # row 0 (top)
    (7, 1), (4, 1),  # row 1
    (7, 2), (5, 2), (4, 2), (2, 2), (1, 2), (0, 2),  # row 2
    (7, 3), (6, 3), (5, 3), (0, 3),  # row 3
    (7, 4), (5, 4), (4, 4), (3, 4), (2, 4), (0, 4),  # row 4
    (7, 5), (2, 5), (0, 5),  # row 5
    (7, 6), (6, 6), (5, 6), (4, 6), (2, 6), (0, 6),  # row 6
    (7, 7), (6, 7), (5, 7), (4, 7), (2, 7), (1, 7),  # row 7 (bottom), excluding (0,7) start
]


def cell_to_world_bounds(cell_i, cell_j, cell_size=32, maze_size=8):
    """Convert maze cell coords to world pixel bounds.

    Returns (x_min, x_max, y_min, y_max) in world coords.
    """
    x_min = cell_i * cell_size
    x_max = (cell_i + 1) * cell_size
    # Flip y: maze j=0 is top, world y=0 is also top in our convention
    # Actually in our maze.py: y_start = (maze_size - 1 - j) * cell_size
    y_min = (maze_size - 1 - cell_j) * cell_size
    y_max = (maze_size - cell_j) * cell_size
    return x_min, x_max, y_min, y_max


def count_agents_in_cell(positions, alive, cell_i, cell_j, cell_size=32, maze_size=8):
    """Count alive agents in a specific maze cell."""
    x_min, x_max, y_min, y_max = cell_to_world_bounds(cell_i, cell_j, cell_size, maze_size)

    x_coords = positions[:, 1]
    y_coords = positions[:, 0]

    in_cell = (
        (x_coords >= x_min) & (x_coords < x_max) &
        (y_coords >= y_min) & (y_coords < y_max) &
        alive
    )
    return int(jnp.sum(in_cell))


def set_goal_cell(world, goal_cell, resource_fertile, resource_normal, cell_size=32, maze_size=8):
    """Update world resources to set new goal cell."""
    # Reset all passable to normal
    resource = world["resource"]
    resource_base = world["resource_base"]

    # Set new goal
    x_min, x_max, y_min, y_max = cell_to_world_bounds(goal_cell[0], goal_cell[1], cell_size, maze_size)
    resource = resource.at[y_min:y_max, x_min:x_max].set(resource_fertile)
    resource_base = resource_base.at[y_min:y_max, x_min:x_max].set(resource_fertile)

    world["resource"] = resource
    world["resource_base"] = resource_base
    return world


def clear_goal_cell(world, goal_cell, resource_normal, cell_size=32, maze_size=8):
    """Reset goal cell to normal resources."""
    resource = world["resource"]
    resource_base = world["resource_base"]

    x_min, x_max, y_min, y_max = cell_to_world_bounds(goal_cell[0], goal_cell[1], cell_size, maze_size)
    resource = resource.at[y_min:y_max, x_min:x_max].set(resource_normal)
    resource_base = resource_base.at[y_min:y_max, x_min:x_max].set(resource_normal)

    world["resource"] = resource
    world["resource_base"] = resource_base
    return world


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_dynamic_maze(world_config: dict, run_config: dict, output_dir: Path, debug: bool = False):
    """Run dynamic maze experiment."""
    reset_timings()
    seed = run_config["seed"]
    max_steps = run_config["max_steps"]
    detailed_interval = run_config["logging"]["detailed_interval"]
    checkpoint_interval = run_config["logging"]["checkpoint_interval"]

    # Dynamic goal settings
    dg = run_config["dynamic_goal"]
    threshold = dg["threshold"]
    resource_fertile = dg["resource_fertile"]
    resource_normal = dg["resource_normal"]
    cell_size = world_config["arena"].get("cell_size", 32)

    # Save configs
    with open(output_dir / "world_config.yaml", "w") as f:
        yaml.dump(world_config, f)
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f)

    # Initialize
    sim = Simulation(world_config, run_config, debug=debug)
    key = make_key(seed)
    init_key, run_key = random.split(key)
    state = sim.reset(init_key)

    # Start with goal at top-right (7, 0)
    current_goal = (7, 0)
    goal_history = [(0, current_goal)]

    # History tracking
    history = {
        "steps": [],
        "population": [],
        "actions": [],
        "attacks": [],
        "kills": [],
        "toxin_deaths": [],
        "goal_population": [],  # Track agents in current goal
    }

    log_file = open(output_dir / "logs" / "base_log.jsonl", "w")
    goal_log = open(output_dir / "logs" / "goal_moves.jsonl", "w")

    # Step 0
    run_key, step0_key = random.split(run_key)
    state = sim.step(state, step0_key)
    reset_timings()

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

    goal_pop = count_agents_in_cell(state["positions"], state["alive"], current_goal[0], current_goal[1], cell_size)
    history["goal_population"].append(goal_pop)

    log_detailed(0, state, stats, output_dir, world_config, history)
    save_checkpoint(0, state, sim, output_dir)

    # Main loop
    pbar = tqdm(range(1, max_steps + 1), desc="Simulating", unit="step")
    goal_idx = PASSABLE_CELLS.index(current_goal)

    for step in pbar:
        run_key, step_key = random.split(run_key)
        state = sim.step(state, step_key)

        # Check goal population every step
        goal_pop = count_agents_in_cell(state["positions"], state["alive"], current_goal[0], current_goal[1], cell_size)

        if goal_pop > threshold:
            # Move goal to next cell in list
            old_goal = current_goal
            goal_idx = (goal_idx + 1) % len(PASSABLE_CELLS)
            current_goal = PASSABLE_CELLS[goal_idx]

            # Update world resources
            state["world"] = clear_goal_cell(state["world"], old_goal, resource_normal, cell_size)
            state["world"] = set_goal_cell(state["world"], current_goal, resource_fertile, resource_normal, cell_size)

            goal_history.append((step, current_goal))
            goal_log.write(json.dumps({"step": step, "old_goal": old_goal, "new_goal": current_goal, "population": goal_pop}) + "\n")
            goal_log.flush()
            tqdm.write(f"Step {step}: Goal moved from {old_goal} to {current_goal} (had {goal_pop} agents)")

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
            history["goal_population"].append(goal_pop)

            pbar.set_postfix({
                "alive": stats["num_alive"],
                "goal_pop": goal_pop,
                "goal": current_goal
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

            # Also plot goal population
            if len(history["goal_population"]) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                ax.plot(history["steps"], history["goal_population"], "g-", linewidth=1)
                ax.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold})")
                ax.set_xlabel("Step")
                ax.set_ylabel("Agents in Goal")
                ax.set_title(f"Goal Population (current goal: {current_goal})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "figures" / "goal_population.png", dpi=100)
                plt.close()

        if step % checkpoint_interval == 0:
            save_checkpoint(step, state, sim, output_dir)

    log_file.close()
    goal_log.close()
    save_checkpoint(state["step"], state, sim, output_dir)

    # Save goal history
    with open(output_dir / "logs" / "goal_history.json", "w") as f:
        json.dump(goal_history, f, indent=2)

    return state


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    run_config = load_config(config_path)

    if 'output_dir' not in run_config:
        raise ValueError("FATAL: 'output_dir' required in config")
    if 'dynamic_goal' not in run_config:
        raise ValueError("FATAL: 'dynamic_goal' section required for dynamic maze")

    world_config_path = run_config.get("world_config")
    if world_config_path is None:
        raise ValueError("FATAL: 'world_config' required for dynamic maze")
    world_config = load_config(world_config_path)

    output_dir = init_directory(run_config["output_dir"], overwrite=overwrite)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, output_dir / "config.yaml")

    print(f"JAX devices: {jax.devices()}")
    print(f"Output dir: {output_dir}")
    print(f"World size: {world_config['world']['size']}")
    print(f"Dynamic goal threshold: {run_config['dynamic_goal']['threshold']}")
    print(f"Max steps: {run_config['max_steps']}")
    print(f"Seed: {run_config['seed']}")
    print()

    state = run_dynamic_maze(world_config, run_config, output_dir, debug=debug)

    stats = Simulation(world_config, run_config).get_stats(state)
    print(f"\nFinal: {stats['num_alive']} agents alive at step {stats['step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darwin Dynamic Maze")
    parser.add_argument("config_path", type=str, help="Path to run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug timing")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
