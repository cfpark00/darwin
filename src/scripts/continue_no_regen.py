"""Continue simulation from checkpoint with dynamic resource regeneration.

Supports decaying regeneration where regen_timescale increases over time,
simulating resource depletion scenarios.
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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation import Simulation, reset_timings, get_timings
from src.utils import make_key, init_directory
from src.physics import EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE, ATTACK
from src.worlds.base import regenerate_resources

# Import the run functions we can reuse
from src.scripts.run import (
    ACTION_NAMES, compute_action_counts, log_base, log_detailed, save_checkpoint
)


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load full checkpoint including world state."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def run_continue_experiment(world_config: dict, run_config: dict, output_dir: Path,
                            initial_state: dict, start_step: int, debug: bool = False) -> dict:
    """Continue experiment from loaded state with dynamic regeneration.

    The simulation's built-in regen is disabled (very high timescale in config).
    We manually apply regeneration after each step with a dynamic timescale
    that increases over time.
    """
    reset_timings()
    seed = run_config["seed"]
    max_steps = run_config["max_steps"]
    detailed_interval = run_config["logging"]["detailed_interval"]
    checkpoint_interval = run_config["logging"]["checkpoint_interval"]

    # Dynamic regen config
    regen_config = run_config.get("dynamic_regen", {})
    regen_initial = regen_config.get("initial_timescale", 100.0)
    regen_growth_rate = regen_config.get("growth_rate", 0.10)  # 10% increase
    regen_growth_interval = regen_config.get("growth_interval", 100)  # every 100 steps

    # Save configs to output dir
    with open(output_dir / "world_config.yaml", "w") as f:
        yaml.dump(world_config, f)
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.dump(run_config, f)

    # Create simulation with disabled built-in regen (config has very high timescale)
    sim = Simulation(world_config, run_config, debug=debug)

    # Use the loaded state directly
    state = initial_state
    state["step"] = start_step

    # History tracking
    history = {
        "steps": [],
        "population": [],
        "actions": [],
        "attacks": [],
        "kills": [],
        "toxin_deaths": [],
        "regen_timescale": [],  # Track dynamic timescale
    }

    # Open base log file
    log_file = open(output_dir / "logs" / "base_log.jsonl", "w")

    # Initialize PRNG from seed + start_step for reproducibility
    key = make_key(seed + start_step)

    # Current regen timescale (will grow over time)
    current_regen_timescale = regen_initial

    # Log initial state
    actions = state.get("actions", jnp.zeros(state["max_agents"], dtype=jnp.int32))
    stats = sim.get_stats(state)
    log_base(start_step, state, stats, actions, output_dir, log_file)

    history["steps"].append(start_step)
    history["population"].append(stats["num_alive"])
    action_counts = compute_action_counts(state, actions)
    history["actions"].append([action_counts[name] for name in ACTION_NAMES])
    history["attacks"].append(action_counts["attack"])
    history["kills"].append(state.get("num_kills", 0))
    history["toxin_deaths"].append(state.get("num_toxin_deaths", 0))
    history["regen_timescale"].append(current_regen_timescale)

    log_detailed(start_step, state, stats, output_dir, world_config, history)
    save_checkpoint(start_step, state, sim, output_dir)

    # Main loop
    end_step = start_step + max_steps
    pbar = tqdm(range(start_step + 1, end_step + 1), desc="Simulating", unit="step")

    steps_since_start = 0

    for step in pbar:
        key, step_key = random.split(key)
        state = sim.step(state, step_key)
        steps_since_start += 1

        # Apply manual regeneration with current dynamic timescale
        world = state["world"]
        world["resource"] = regenerate_resources(
            world["resource"],
            world["resource_base"],
            current_regen_timescale
        )
        state["world"] = world

        # Update regen timescale every growth_interval steps
        if steps_since_start % regen_growth_interval == 0:
            current_regen_timescale *= (1.0 + regen_growth_rate)

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
            history["regen_timescale"].append(current_regen_timescale)

            pbar.set_postfix({
                "alive": stats["num_alive"],
                "avg_E": f"{stats['avg_energy']:.1f}",
                "regen_t": f"{current_regen_timescale:.0f}"
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

    # Save regen history
    with open(output_dir / "results" / "regen_history.json", "w") as f:
        json.dump({
            "steps": history["steps"],
            "regen_timescale": history["regen_timescale"],
        }, f)

    return state


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    """Main entry point."""
    run_config = load_config(config_path)

    if 'output_dir' not in run_config:
        raise ValueError("FATAL: 'output_dir' required in config")

    if 'continue' not in run_config:
        raise ValueError("FATAL: 'continue' section required")

    continue_config = run_config['continue']
    if 'checkpoint_path' not in continue_config:
        raise ValueError("FATAL: 'continue.checkpoint_path' required")

    # Load world config
    world_config_path = run_config.get("world_config")
    if world_config_path is None:
        raise ValueError("FATAL: 'world_config' path required")
    world_config = load_config(world_config_path)

    # Setup output directory
    output_dir = init_directory(run_config["output_dir"], overwrite=overwrite)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy(config_path, output_dir / "config.yaml")

    # Load checkpoint
    checkpoint_path = continue_config['checkpoint_path']
    checkpoint = load_checkpoint(checkpoint_path)
    start_step = checkpoint['step']
    initial_state = checkpoint['state']

    # Dynamic regen config
    regen_config = run_config.get("dynamic_regen", {})
    regen_initial = regen_config.get("initial_timescale", 100.0)
    regen_growth_rate = regen_config.get("growth_rate", 0.10)
    regen_growth_interval = regen_config.get("growth_interval", 100)

    print(f"JAX devices: {jax.devices()}")
    print(f"Output dir: {output_dir}")
    print(f"Continuing from step: {start_step}")
    print(f"Agents alive: {int(initial_state['alive'].sum())}")
    print(f"Dynamic regen: timescale starts at {regen_initial}, +{regen_growth_rate*100:.0f}% every {regen_growth_interval} steps")
    print(f"Additional steps: {run_config['max_steps']}")
    print()

    state = run_continue_experiment(world_config, run_config, output_dir,
                                     initial_state, start_step, debug=debug)

    stats = Simulation(world_config, run_config).get_stats(state)
    print(f"\nFinal: {stats['num_alive']} agents alive at step {stats['step']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue simulation with no resource regeneration")
    parser.add_argument("config_path", type=str, help="Path to run config YAML")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if exists")
    parser.add_argument("--debug", action="store_true", help="Enable debug timing")
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
