"""Simulation class for simple environments (no toxin sensing, no attack).

This is a simplified version for thermotaxis/pretrain experiments.
- Observations: 6 inputs (food, temp, 4 contact sensors) - no toxin
- Actions: 6 outputs (eat, forward, left, right, stay, reproduce) - no attack
"""

import time
import jax
import jax.numpy as jnp
from jax import random, vmap
from functools import partial

from src.world import create_world, regenerate_resources
from src.agent_simple import (
    init_params_flat, compute_param_dim, compute_state_dim,
    agent_forward_fully_flat, sample_action, mutate_params_flat
)
from src.physics import (
    compute_intended_position, compute_new_orientation,
    compute_energy_cost, resolve_move_conflicts,
    EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE
)
from src.worlds import thermotaxis, pretrain

# Global timing accumulator
_TIMINGS = {}
_DEBUG = False

def set_debug(enabled: bool):
    global _DEBUG
    _DEBUG = enabled

def reset_timings():
    global _TIMINGS
    _TIMINGS = {}

def get_timings():
    return _TIMINGS.copy()

def _record(name, elapsed):
    if not _DEBUG:
        return
    if name not in _TIMINGS:
        _TIMINGS[name] = []
    _TIMINGS[name].append(elapsed)


def _get_direction_deltas():
    return jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)


def _build_occupancy_grid(positions: jax.Array, alive: jax.Array, size: int) -> jax.Array:
    """Build occupancy grid: grid[y,x] = agent_idx+1 if occupied, 0 if empty."""
    n = len(alive)
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    agent_ids = jnp.where(alive, jnp.arange(n) + 1, 0)
    grid = grid.at[positions[:, 0], positions[:, 1]].max(agent_ids)
    return grid


def _compute_observation_single(
    world_resource: jax.Array,
    world_temperature: jax.Array,
    occupancy_grid: jax.Array,
    pos: jax.Array,
    ori: jax.Array,
    agent_idx: int,
    key: jax.Array,
    size: int,
    food_noise_std: float,
    temp_noise_std: float,
) -> jax.Array:
    """Compute observation for a single agent - O(1) lookups.

    Returns 6 values: food, temp, contact_front, contact_back, contact_left, contact_right
    No toxin sensing.
    """
    y, x = pos[0], pos[1]
    keys = random.split(key, 2)

    # Food and temperature with noise
    food = world_resource[y, x]
    temp = world_temperature[y, x]
    food_noisy = food + random.normal(keys[0]) * food_noise_std
    temp_noisy = temp + random.normal(keys[1]) * temp_noise_std

    # Contact sensors - O(1) grid lookups
    deltas = _get_direction_deltas()

    def check_contact(delta):
        ny, nx = y + delta[0], x + delta[1]
        in_bounds = (ny >= 0) & (ny < size) & (nx >= 0) & (nx < size)
        occupant = jnp.where(in_bounds, occupancy_grid[ny % size, nx % size], 0)
        has_other_agent = (occupant > 0) & (occupant != agent_idx + 1)
        return has_other_agent.astype(jnp.float32)

    contact_front = check_contact(deltas[ori])
    contact_back = check_contact(deltas[(ori + 2) % 4])
    contact_left = check_contact(deltas[(ori + 3) % 4])
    contact_right = check_contact(deltas[(ori + 1) % 4])

    return jnp.array([food_noisy, temp_noisy,
                      contact_front, contact_back, contact_left, contact_right])


def _init_params_batched(key: jax.Array, n: int, hidden_dim: int, internal_noise_dim: int) -> jax.Array:
    """Initialize batched flat agent parameters. Returns (n, param_dim) array."""
    keys = random.split(key, n)
    return vmap(lambda k: init_params_flat(k, hidden_dim, internal_noise_dim))(keys)


class SimulationSimple:
    """Simplified simulation engine - no toxin sensing, no attack action."""

    def __init__(self, world_config: dict, run_config: dict, debug: bool = False):
        """Initialize simulation with world and run configuration."""
        self.config = world_config
        self.run_config = run_config
        self.size = world_config["world"]["size"]
        self.hidden_dim = world_config["agent"]["hidden_dim"]
        self.internal_noise_dim = world_config["agent"]["internal_noise_dim"]
        self.min_buffer_size = run_config["min_buffer_size"]
        self.debug = debug

        set_debug(debug)

        # Pre-compute dimensions for flat genome and state
        self.param_dim = compute_param_dim(self.hidden_dim, self.internal_noise_dim)
        self.state_dim = compute_state_dim(self.hidden_dim)

        # Pre-compute config values for JIT
        self.food_noise_std = world_config["sensors"]["food_noise_std"]
        self.temp_noise_std = world_config["sensors"]["temperature_noise_std"]
        self.action_temperature = world_config["agent"]["action_temperature"]
        self.max_energy = world_config["energy"]["max"]
        self.mutation_std = world_config["agent"]["mutation_std"]
        self.offspring_energy = world_config["energy"]["offspring"]
        self.eat_fraction = world_config["resource"]["eat_fraction"]
        self.regen_timescale = world_config["resource"]["regen_timescale"]

        # Reproduction probability config (for thermotaxis)
        repro_config = world_config.get("reproduction", {})
        self.repro_temp_threshold = repro_config.get("temp_threshold", 0.5)
        self.repro_temp_max = repro_config.get("temp_max", 1.0)

        # Check arena type for dynamic world updates
        arena_type = world_config.get("arena", {}).get("type")
        self.is_thermotaxis = arena_type == "thermotaxis"
        self.is_pretrain = arena_type == "pretrain"

        # Energy costs - 6 actions only (no attack)
        self._energy_costs = jnp.array([
            world_config["energy"]["cost_eat"],
            world_config["energy"]["cost_move"],
            world_config["energy"]["cost_move"],
            world_config["energy"]["cost_move"],
            world_config["energy"]["cost_stay"],
            world_config["energy"]["cost_reproduce"],
        ], dtype=jnp.float32)

        # Build JIT-compiled step function
        self._step_jit = self._build_step_fn()

    def _build_step_fn(self):
        """Build a JIT-compiled step function with config captured in closure."""
        size = self.size
        hidden_dim = self.hidden_dim
        internal_noise_dim = self.internal_noise_dim
        food_noise_std = self.food_noise_std
        temp_noise_std = self.temp_noise_std
        action_temperature = self.action_temperature
        max_energy = self.max_energy
        mutation_std = self.mutation_std
        offspring_energy = self.offspring_energy
        eat_fraction = self.eat_fraction
        regen_timescale = self.regen_timescale
        energy_costs = self._energy_costs
        repro_temp_threshold = self.repro_temp_threshold
        repro_temp_max = self.repro_temp_max

        @partial(jax.jit, static_argnums=(10,))
        def step_jit(
            key,
            # World arrays
            world_resource, world_resource_base, world_temperature,
            # Agent arrays
            positions, orientations, params, agent_states, energies, alive,
            # Scalar (static)
            max_agents
        ):
            """Pure JIT-compiled step function - no toxin, no attack."""
            n = max_agents
            keys = random.split(key, 6)

            # === Phase 1: Observe and decide ===
            obs_keys = random.split(keys[0], n)
            action_keys = random.split(keys[1], n)
            noise = random.normal(keys[2], (n, internal_noise_dim))

            occupancy_grid = _build_occupancy_grid(positions, alive, size)

            # Sanity check
            pos_count = jnp.zeros((size, size), dtype=jnp.int32)
            pos_count = pos_count.at[positions[:, 0], positions[:, 1]].add(alive.astype(jnp.int32))
            has_collision = jnp.any(pos_count > 1)

            # Vectorized observation (no toxin)
            def compute_obs_for_agent(i, obs_key):
                return _compute_observation_single(
                    world_resource, world_temperature,
                    occupancy_grid, positions[i], orientations[i],
                    i, obs_key, size, food_noise_std, temp_noise_std
                )

            observations = vmap(compute_obs_for_agent)(jnp.arange(n), obs_keys)

            # Vectorized forward pass
            def forward_wrapper(flat_params_i, flat_state_i, obs_i, energy_i, noise_i):
                new_state, logits = agent_forward_fully_flat(
                    flat_params_i, flat_state_i, obs_i, energy_i / max_energy, noise_i,
                    hidden_dim, internal_noise_dim
                )
                return new_state, logits

            agent_states, all_logits = vmap(forward_wrapper)(
                params, agent_states, observations, energies, noise
            )

            # Sample actions (6 possible actions)
            actions = vmap(lambda k, l: sample_action(k, l, action_temperature))(action_keys, all_logits)
            actions = jnp.where(alive, actions, STAY)

            # === Phase 2: Energy costs ===
            metabolic_penalty = energies / max_energy
            action_energy_costs = energy_costs[actions] + metabolic_penalty
            energies = energies - action_energy_costs * alive.astype(jnp.float32)

            # No attack phase

            # === Phase 3: Movement ===
            deltas = _get_direction_deltas()
            intended_positions = vmap(
                lambda p, o, a: compute_intended_position(p, o, a, size)
            )(positions, orientations, actions)
            positions = resolve_move_conflicts(intended_positions, positions, actions, alive, size)
            orientations = vmap(compute_new_orientation)(orientations, actions)

            # No toxin death phase

            # === Phase 4: Reproduction ===
            occupancy_grid = _build_occupancy_grid(positions, alive, size)
            max_K = 64

            wants_repro = (actions == REPRODUCE) & alive

            repro_candidate_indices = jnp.where(wants_repro, jnp.arange(n), n)
            repro_candidate_sorted = jnp.sort(repro_candidate_indices)[:max_K]
            repro_candidate_valid = repro_candidate_sorted < n
            safe_repro_idx = jnp.minimum(repro_candidate_sorted, n - 1)

            def try_reproduce_single(agent_idx, k):
                ori = orientations[agent_idx]
                pos = positions[agent_idx]
                candidates = jnp.stack([
                    pos + deltas[ori],
                    pos + deltas[(ori + 2) % 4],
                    pos + deltas[(ori + 3) % 4],
                    pos + deltas[(ori + 1) % 4],
                ])

                def is_valid_cell(cell_pos):
                    in_bounds = (cell_pos[0] >= 0) & (cell_pos[0] < size) & \
                               (cell_pos[1] >= 0) & (cell_pos[1] < size)
                    occupied = jnp.where(in_bounds,
                                         occupancy_grid[cell_pos[0] % size, cell_pos[1] % size] > 0,
                                         True)
                    return in_bounds & ~occupied

                valid_mask = vmap(is_valid_cell)(candidates)
                any_valid = jnp.any(valid_mask)

                k1, k2 = random.split(k)
                probs = valid_mask.astype(jnp.float32)
                probs = probs / (probs.sum() + 1e-10)
                chosen_idx = random.categorical(k1, jnp.log(probs + 1e-10))
                offspring_pos = candidates[chosen_idx]
                offspring_ori = random.randint(k2, (), 0, 4)

                return any_valid, offspring_pos, offspring_ori

            repro_keys = random.split(keys[3], max_K)
            can_reproduce, offspring_positions, offspring_orientations = vmap(try_reproduce_single)(
                safe_repro_idx, repro_keys
            )
            repro_success = repro_candidate_valid & can_reproduce

            # Temperature-based reproduction probability
            parent_positions = positions[safe_repro_idx]
            parent_temps = world_temperature[parent_positions[:, 0], parent_positions[:, 1]]
            temp_prob = jnp.clip(
                1.0 - (parent_temps - repro_temp_threshold) / (repro_temp_max - repro_temp_threshold + 1e-8),
                0.0, 1.0
            )
            temp_key = random.fold_in(keys[3], 999)
            temp_rolls = random.uniform(temp_key, (max_K,))
            temp_success = temp_rolls < temp_prob
            repro_success = repro_success & temp_success

            # Resolve offspring position conflicts
            offspring_flat_pos = offspring_positions[:, 0] * size + offspring_positions[:, 1]
            claim_count = jnp.zeros(size * size, dtype=jnp.int32)
            claim_count = claim_count.at[offspring_flat_pos].add(repro_success.astype(jnp.int32))
            no_competition = claim_count[offspring_flat_pos] == 1
            repro_success = repro_success & no_competition

            # Find dead slots
            dead_mask = ~alive
            dead_indices = jnp.where(dead_mask, jnp.arange(n), n)
            dead_indices_sorted = jnp.sort(dead_indices)[:max_K]

            success_order = jnp.where(repro_success, jnp.arange(max_K), max_K)
            success_order_sorted = jnp.sort(success_order)

            safe_offspring_idx = jnp.minimum(success_order_sorted, max_K - 1)
            parent_agent_indices = safe_repro_idx[safe_offspring_idx]

            valid_arr = (success_order_sorted < max_K) & (dead_indices_sorted < n)

            safe_parent_idx = jnp.minimum(parent_agent_indices, n - 1)
            safe_dead_idx = jnp.minimum(dead_indices_sorted, n - 1)

            parent_params_gathered = params[safe_parent_idx]

            mutate_keys = random.split(keys[4], max_K)
            child_params = vmap(lambda k, p: mutate_params_flat(k, p, mutation_std))(
                mutate_keys, parent_params_gathered
            )

            new_offspring_pos = offspring_positions[safe_offspring_idx]
            new_offspring_ori = offspring_orientations[safe_offspring_idx]

            positions = positions.at[safe_dead_idx].set(
                jnp.where(valid_arr[:, None], new_offspring_pos, positions[safe_dead_idx])
            )
            orientations = orientations.at[safe_dead_idx].set(
                jnp.where(valid_arr, new_offspring_ori, orientations[safe_dead_idx])
            )
            energies = energies.at[safe_dead_idx].set(
                jnp.where(valid_arr, offspring_energy, energies[safe_dead_idx])
            )
            alive = alive.at[safe_dead_idx].set(
                jnp.where(valid_arr, True, alive[safe_dead_idx])
            )

            zeros_state = jnp.zeros((max_K, hidden_dim * 4))
            agent_states = agent_states.at[safe_dead_idx].set(
                jnp.where(valid_arr[:, None], zeros_state, agent_states[safe_dead_idx])
            )

            params = params.at[safe_dead_idx].set(
                jnp.where(valid_arr[:, None], child_params, params[safe_dead_idx])
            )

            # === Phase 5: Eating ===
            is_eating = (actions == EAT) & alive
            eat_y, eat_x = positions[:, 0], positions[:, 1]
            available = world_resource[eat_y, eat_x]
            eat_amounts = jnp.where(is_eating, available * eat_fraction, 0.0)
            energies = energies + eat_amounts
            world_resource = world_resource.at[eat_y, eat_x].add(-eat_amounts)

            # === Phase 6: Energy deaths ===
            alive = alive & (energies > 0)

            # === Phase 7: Resource regeneration ===
            world_resource = regenerate_resources(world_resource, world_resource_base, regen_timescale)

            num_alive = jnp.sum(alive)

            return (
                world_resource, positions, orientations, params,
                agent_states, energies, alive, actions, num_alive, has_collision
            )

        return step_jit

    def reset(self, key: jax.Array) -> dict:
        """Initialize/reset simulation state with pre-allocated buffers."""
        keys = random.split(key, 4)

        world = create_world(keys[0], self.config)

        num_agents = self.config["world"]["initial_agents"]
        max_agents = 1
        while max_agents < max(num_agents * 2, self.min_buffer_size):
            max_agents *= 2

        total_cells = self.size * self.size
        flat_indices = random.choice(keys[1], total_cells, shape=(num_agents,), replace=False)
        positions = jnp.zeros((max_agents, 2), dtype=jnp.int32)
        positions = positions.at[:num_agents, 0].set(flat_indices // self.size)
        positions = positions.at[:num_agents, 1].set(flat_indices % self.size)

        orientations = jnp.zeros(max_agents, dtype=jnp.int32)
        orientations = orientations.at[:num_agents].set(
            random.randint(keys[2], (num_agents,), 0, 4)
        )

        params = _init_params_batched(keys[3], max_agents, self.hidden_dim, self.internal_noise_dim)
        states = jnp.zeros((max_agents, self.state_dim))

        energies = jnp.zeros(max_agents)
        energies = energies.at[:num_agents].set(self.config["energy"]["initial"])
        alive = jnp.zeros(max_agents, dtype=bool)
        alive = alive.at[:num_agents].set(True)

        return {
            "world": world,
            "positions": positions,
            "orientations": orientations,
            "params": params,
            "states": states,
            "energies": energies,
            "alive": alive,
            "max_agents": max_agents,
            "step": 0,
        }

    def reset_with_agents(self, key: jax.Array, agent_params: jax.Array,
                          initial_energy: float = None,
                          spawn_region: tuple = None) -> dict:
        """Initialize simulation with pre-loaded agent parameters."""
        keys = random.split(key, 3)
        num_agents = len(agent_params)

        if initial_energy is None:
            initial_energy = self.config["energy"]["initial"]

        world = create_world(keys[0], self.config)

        max_agents = 1
        while max_agents < max(num_agents * 2, self.min_buffer_size):
            max_agents *= 2

        if spawn_region is not None:
            y_min, y_max, x_min, x_max = spawn_region
            region_h = y_max - y_min
            region_w = x_max - x_min
            total_cells = region_h * region_w
            flat_indices = random.choice(keys[1], total_cells, shape=(num_agents,), replace=False)
            rel_y = flat_indices // region_w
            rel_x = flat_indices % region_w
            positions = jnp.zeros((max_agents, 2), dtype=jnp.int32)
            positions = positions.at[:num_agents, 0].set(y_min + rel_y)
            positions = positions.at[:num_agents, 1].set(x_min + rel_x)
        else:
            total_cells = self.size * self.size
            flat_indices = random.choice(keys[1], total_cells, shape=(num_agents,), replace=False)
            positions = jnp.zeros((max_agents, 2), dtype=jnp.int32)
            positions = positions.at[:num_agents, 0].set(flat_indices // self.size)
            positions = positions.at[:num_agents, 1].set(flat_indices % self.size)

        orientations = jnp.zeros(max_agents, dtype=jnp.int32)
        orientations = orientations.at[:num_agents].set(
            random.randint(keys[2], (num_agents,), 0, 4)
        )

        params = _init_params_batched(keys[2], max_agents, self.hidden_dim, self.internal_noise_dim)
        params = params.at[:num_agents].set(agent_params)

        states = jnp.zeros((max_agents, self.state_dim))

        energies = jnp.zeros(max_agents)
        energies = energies.at[:num_agents].set(initial_energy)
        alive = jnp.zeros(max_agents, dtype=bool)
        alive = alive.at[:num_agents].set(True)

        return {
            "world": world,
            "positions": positions,
            "orientations": orientations,
            "params": params,
            "states": states,
            "energies": energies,
            "alive": alive,
            "max_agents": max_agents,
            "step": 0,
        }

    def _grow_buffers(self, state: dict, num_alive: int, key: jax.Array) -> dict:
        """Grow buffers to accommodate population."""
        max_agents = state["max_agents"]
        new_max = max_agents * 2
        grow_by = new_max - max_agents
        print(f"Growing buffers: {max_agents} -> {new_max} ({num_alive} alive, triggering JIT recompilation)")

        positions = jnp.concatenate([state["positions"], jnp.zeros((grow_by, 2), dtype=jnp.int32)])
        orientations = jnp.concatenate([state["orientations"], jnp.zeros(grow_by, dtype=jnp.int32)])
        energies = jnp.concatenate([state["energies"], jnp.zeros(grow_by)])
        alive = jnp.concatenate([state["alive"], jnp.zeros(grow_by, dtype=bool)])

        new_params = _init_params_batched(key, grow_by, self.hidden_dim, self.internal_noise_dim)
        params = jnp.concatenate([state["params"], new_params], axis=0)

        states = jnp.concatenate([state["states"], jnp.zeros((grow_by, self.state_dim))], axis=0)

        actions = state.get("actions")
        if actions is not None:
            actions = jnp.concatenate([actions, jnp.zeros(grow_by, dtype=actions.dtype)])

        result = {
            **state,
            "positions": positions,
            "orientations": orientations,
            "params": params,
            "states": states,
            "energies": energies,
            "alive": alive,
            "max_agents": new_max,
        }
        if actions is not None:
            result["actions"] = actions
        return result

    def step(self, state: dict, key: jax.Array) -> dict:
        """Execute one fully vectorized simulation step."""
        if self.debug:
            t0 = time.time()

        world = state["world"]
        max_agents = state["max_agents"]

        # Update world for dynamic environments
        if self.is_thermotaxis:
            world = thermotaxis.update_temperature(world, state["step"])
        elif self.is_pretrain:
            world = pretrain.update_food(world, state["step"])

        (
            new_resource, new_positions, new_orientations, new_params,
            new_states, new_energies, new_alive, actions, num_alive, has_collision
        ) = self._step_jit(
            key,
            world["resource"], world["resource_base"], world["temperature"],
            state["positions"], state["orientations"], state["params"],
            state["states"], state["energies"], state["alive"],
            max_agents
        )

        if bool(has_collision):
            raise RuntimeError(f"BUG: Multiple alive agents occupy the same cell at step {state['step']}")

        if self.debug:
            jax.block_until_ready(new_alive)
            _record("step_total", time.time() - t0)

        new_world = {**world, "resource": new_resource}

        new_state = {
            "world": new_world,
            "positions": new_positions,
            "orientations": new_orientations,
            "params": new_params,
            "states": new_states,
            "energies": new_energies,
            "alive": new_alive,
            "max_agents": max_agents,
            "step": state["step"] + 1,
            "actions": actions,
        }

        num_alive_int = int(num_alive)
        if num_alive_int > max_agents * 0.5:
            grow_key = random.fold_in(key, 999)
            new_state = self._grow_buffers(new_state, num_alive_int, grow_key)

        return new_state

    def get_stats(self, state: dict) -> dict:
        """Get summary statistics from state."""
        alive = state["alive"]
        num_alive = int(jnp.sum(alive))

        if num_alive > 0:
            avg_energy = float(jnp.mean(state["energies"][alive]))
            max_energy = float(jnp.max(state["energies"][alive]))
            min_energy = float(jnp.min(state["energies"][alive]))
        else:
            avg_energy = max_energy = min_energy = 0.0

        return {
            "step": state["step"],
            "num_alive": num_alive,
            "total_agents": state["max_agents"],
            "avg_energy": avg_energy,
            "max_energy": max_energy,
            "min_energy": min_energy,
        }
