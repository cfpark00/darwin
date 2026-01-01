"""Simulation class - fully vectorized physics engine."""

import time
import jax
import jax.numpy as jnp
from jax import random, vmap
from functools import partial

from src.world import create_world, regenerate_resources
from src.agent import (
    init_params_flat, compute_param_dim, compute_state_dim,
    agent_forward_fully_flat, sample_action, mutate_params_flat
)
from src.physics import (
    compute_intended_position, compute_new_orientation,
    compute_energy_cost, resolve_move_conflicts,
    EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE, ATTACK
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


def _make_circular_kernel(radius: int) -> jax.Array:
    """Create a circular kernel for toxin detection convolution."""
    size = 2 * radius + 1
    y, x = jnp.meshgrid(jnp.arange(size) - radius, jnp.arange(size) - radius, indexing='ij')
    dist = jnp.sqrt(y**2 + x**2)
    return (dist <= radius).astype(jnp.float32)


def _compute_toxin_nearby_field(world_toxin: jax.Array, kernel: jax.Array) -> jax.Array:
    """Convolve toxin field with circular kernel to get 'toxin nearby' field.

    Uses JAX's GPU-accelerated convolution. Kernel is pre-computed and passed in.
    """
    from jax.scipy.signal import convolve2d
    # Convolve and threshold - any toxin within radius
    convolved = convolve2d(world_toxin, kernel, mode='same')
    return convolved > 0


def _build_occupancy_grid(positions: jax.Array, alive: jax.Array, size: int) -> jax.Array:
    """Build occupancy grid: grid[y,x] = agent_idx+1 if occupied, 0 if empty.

    Uses .max() instead of .set() to handle duplicate positions correctly.
    Dead agents may share positions (e.g., all at [0,0] after buffer growth),
    and .set() with duplicates uses last-write-wins, which could overwrite
    an alive agent's cell with 0.
    """
    n = len(alive)
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    # Only place alive agents - use max() so alive agent IDs win over dead (0)
    agent_ids = jnp.where(alive, jnp.arange(n) + 1, 0)  # +1 so 0 means empty
    grid = grid.at[positions[:, 0], positions[:, 1]].max(agent_ids)
    return grid


def _compute_observation_single(
    world_resource: jax.Array,
    world_temperature: jax.Array,
    toxin_nearby_field: jax.Array,
    occupancy_grid: jax.Array,  # Pre-computed!
    pos: jax.Array,
    ori: jax.Array,
    agent_idx: int,
    key: jax.Array,
    size: int,
    food_noise_std: float,
    temp_noise_std: float,
    confusion_matrix: jax.Array,
) -> jax.Array:
    """Compute observation for a single agent - O(1) lookups."""
    y, x = pos[0], pos[1]
    keys = random.split(key, 3)

    # Food and temperature with noise
    food = world_resource[y, x]
    temp = world_temperature[y, x]
    food_noisy = food + random.normal(keys[0]) * food_noise_std
    temp_noisy = temp + random.normal(keys[1]) * temp_noise_std

    # Toxin detection - just a lookup
    toxin_nearby = toxin_nearby_field[y, x]

    # Apply confusion matrix
    report_prob = jnp.where(toxin_nearby, confusion_matrix[1, 1], confusion_matrix[0, 1])
    toxin_detected = random.bernoulli(keys[2], report_prob).astype(jnp.float32)

    # Contact sensors - O(1) grid lookups instead of O(n) scans
    deltas = _get_direction_deltas()

    def check_contact(delta):
        ny, nx = y + delta[0], x + delta[1]
        in_bounds = (ny >= 0) & (ny < size) & (nx >= 0) & (nx < size)
        # Grid lookup: >0 means occupied, check it's not self
        occupant = jnp.where(in_bounds, occupancy_grid[ny % size, nx % size], 0)
        has_other_agent = (occupant > 0) & (occupant != agent_idx + 1)
        return has_other_agent.astype(jnp.float32)

    contact_front = check_contact(deltas[ori])
    contact_back = check_contact(deltas[(ori + 2) % 4])
    contact_left = check_contact(deltas[(ori + 3) % 4])
    contact_right = check_contact(deltas[(ori + 1) % 4])

    return jnp.array([food_noisy, temp_noisy, toxin_detected,
                      contact_front, contact_back, contact_left, contact_right])


def _init_params_batched(key: jax.Array, n: int, hidden_dim: int, internal_noise_dim: int) -> jax.Array:
    """Initialize batched flat agent parameters. Returns (n, param_dim) array."""
    keys = random.split(key, n)
    return vmap(lambda k: init_params_flat(k, hidden_dim, internal_noise_dim))(keys)


class Simulation:
    """Fully vectorized ALife simulation engine."""

    def __init__(self, world_config: dict, run_config: dict, debug: bool = False):
        """Initialize simulation with world and run configuration."""
        self.config = world_config
        self.run_config = run_config
        self.size = world_config["world"]["size"]
        self.hidden_dim = world_config["agent"]["hidden_dim"]
        self.internal_noise_dim = world_config["agent"]["internal_noise_dim"]
        self.min_buffer_size = run_config["min_buffer_size"]
        self.debug = debug

        # Set global debug flag for timing
        set_debug(debug)

        # Pre-compute dimensions for flat genome and state
        self.param_dim = compute_param_dim(self.hidden_dim, self.internal_noise_dim)
        self.state_dim = compute_state_dim(self.hidden_dim)

        # Pre-compute config values for JIT
        self.food_noise_std = world_config["sensors"]["food_noise_std"]
        self.temp_noise_std = world_config["sensors"]["temperature_noise_std"]
        self.detection_radius = int(world_config["sensors"]["toxin_detection_radius"])
        self.confusion_matrix = jnp.array(world_config["sensors"]["toxin_confusion_matrix"])
        self.action_temperature = world_config["agent"]["action_temperature"]
        self.max_energy = world_config["energy"]["max"]
        self.base_cost = world_config["energy"]["base_cost"]
        self.base_cost_incremental = world_config["energy"]["base_cost_incremental"]
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

        # Pre-compute circular kernel for toxin detection (stays on GPU)
        self.toxin_kernel = _make_circular_kernel(self.detection_radius)

        # Pre-extract all config values needed for JIT
        # Action-specific costs (added on top of base_cost)
        self._action_costs = jnp.array([
            world_config["energy"]["cost_eat"],
            world_config["energy"]["cost_move"],
            world_config["energy"]["cost_move"],
            world_config["energy"]["cost_move"],
            world_config["energy"]["cost_stay"],
            world_config["energy"]["cost_reproduce"],
            world_config["energy"]["cost_attack"],
        ], dtype=jnp.float32)

        # Build JIT-compiled step function with all config captured in closure
        self._step_jit = self._build_step_fn()

    def _build_step_fn(self):
        """Build a JIT-compiled step function with config captured in closure."""
        # Capture all config values
        size = self.size
        hidden_dim = self.hidden_dim
        internal_noise_dim = self.internal_noise_dim
        food_noise_std = self.food_noise_std
        temp_noise_std = self.temp_noise_std
        confusion_matrix = self.confusion_matrix
        action_temperature = self.action_temperature
        max_energy = self.max_energy
        base_cost = self.base_cost
        base_cost_incremental = self.base_cost_incremental
        mutation_std = self.mutation_std
        offspring_energy = self.offspring_energy
        eat_fraction = self.eat_fraction
        regen_timescale = self.regen_timescale
        toxin_kernel = self.toxin_kernel
        action_costs = self._action_costs
        repro_temp_threshold = self.repro_temp_threshold
        repro_temp_max = self.repro_temp_max

        @partial(jax.jit, static_argnums=(11,))
        def step_jit(
            key,
            # World arrays
            world_resource, world_resource_base, world_temperature, world_toxin,
            # Agent arrays
            positions, orientations, params, agent_states, energies, alive,
            # Scalar (static - determines array shapes)
            max_agents
        ):
            """Pure JIT-compiled step function."""
            n = max_agents
            keys = random.split(key, 7)

            # === Phase 1: Observe and decide ===
            obs_keys = random.split(keys[0], n)
            action_keys = random.split(keys[1], n)
            noise = random.normal(keys[2], (n, internal_noise_dim))

            # Pre-compute grids
            toxin_nearby_field = _compute_toxin_nearby_field(world_toxin, toxin_kernel)
            occupancy_grid = _build_occupancy_grid(positions, alive, size)

            # Sanity check: detect if multiple alive agents share same cell (indicates bug)
            pos_count = jnp.zeros((size, size), dtype=jnp.int32)
            pos_count = pos_count.at[positions[:, 0], positions[:, 1]].add(alive.astype(jnp.int32))
            has_collision = jnp.any(pos_count > 1)

            # Vectorized observation
            def compute_obs_for_agent(i, obs_key):
                return _compute_observation_single(
                    world_resource, world_temperature, toxin_nearby_field,
                    occupancy_grid, positions[i], orientations[i],
                    i, obs_key, size, food_noise_std, temp_noise_std, confusion_matrix
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

            # Sample actions
            actions = vmap(lambda k, l: sample_action(k, l, action_temperature))(action_keys, all_logits)
            actions = jnp.where(alive, actions, STAY)

            # === Phase 2: Energy costs ===
            # Total cost = base_cost + base_cost_incremental * energy + action_cost
            # base_cost: minimum cost for any action
            # base_cost_incremental * energy: soft cap (higher energy = higher cost)
            # action_cost: action-specific additional cost
            energy_penalty = base_cost_incremental * energies
            total_action_costs = base_cost + energy_penalty + action_costs[actions]
            energies = energies - total_action_costs * alive.astype(jnp.float32)

            # === Phase 3: Attacks (fully parallel, no K cap) ===
            deltas = _get_direction_deltas()
            is_attacking = (actions == ATTACK) & alive

            # Compute attack target for ALL agents in parallel
            def get_attack_target(i):
                pos = positions[i]
                ori = orientations[i]
                target_pos = (pos + deltas[ori]) % size
                target_cell = occupancy_grid[target_pos[0], target_pos[1]]
                return jnp.where(target_cell > 0, target_cell - 1, -1)

            all_targets = vmap(get_attack_target)(jnp.arange(n))

            # Valid attacks: agent is attacking AND has a valid target
            valid_attack = is_attacking & (all_targets >= 0) & (all_targets < n)

            # Scatter-add to count kills (handles multiple attackers on same target)
            safe_targets = jnp.where(valid_attack, all_targets, 0)
            kill_votes = jnp.zeros(n, dtype=jnp.int32)
            kill_votes = kill_votes.at[safe_targets].add(valid_attack.astype(jnp.int32))
            killed_mask = kill_votes > 0
            num_kills = jnp.sum(killed_mask)

            alive = alive & ~killed_mask

            # Drop energy to ground for killed agents (unique positions, no race)
            kill_y = positions[:, 0]
            kill_x = positions[:, 1]
            energy_to_drop = jnp.where(killed_mask, energies, 0.0)
            world_resource = world_resource.at[kill_y, kill_x].add(energy_to_drop)

            # === Phase 4: Movement ===
            intended_positions = vmap(
                lambda p, o, a: compute_intended_position(p, o, a, size)
            )(positions, orientations, actions)
            positions = resolve_move_conflicts(intended_positions, positions, actions, alive, size)
            orientations = vmap(compute_new_orientation)(orientations, actions)

            # === Phase 5: Toxin deaths ===
            on_toxin = world_toxin[positions[:, 0], positions[:, 1]] > 0.5
            num_toxin_deaths = jnp.sum(alive & on_toxin)
            alive = alive & ~on_toxin

            # === Phase 6: Reproduction ===
            occupancy_grid = _build_occupancy_grid(positions, alive, size)
            max_K = 64  # Max reproducers per step

            wants_repro = (actions == REPRODUCE) & alive

            # Find first max_K agents that want to reproduce
            repro_candidate_indices = jnp.where(wants_repro, jnp.arange(n), n)
            repro_candidate_sorted = jnp.sort(repro_candidate_indices)[:max_K]
            repro_candidate_valid = repro_candidate_sorted < n
            safe_repro_idx = jnp.minimum(repro_candidate_sorted, n - 1)

            # Try reproduce for each candidate - check all 4 directions
            def try_reproduce_single(agent_idx, k):
                ori = orientations[agent_idx]
                pos = positions[agent_idx]
                # Try front, back, left, right
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
            # Get temperature at each reproducer's position
            parent_positions = positions[safe_repro_idx]
            parent_temps = world_temperature[parent_positions[:, 0], parent_positions[:, 1]]
            # Compute success probability: 1.0 below threshold, linear decay to 0 at max
            temp_prob = jnp.clip(
                1.0 - (parent_temps - repro_temp_threshold) / (repro_temp_max - repro_temp_threshold + 1e-8),
                0.0, 1.0
            )
            # Sample whether each reproduction succeeds based on temperature
            temp_key = random.fold_in(keys[3], 999)  # Derive a key for temperature sampling
            temp_rolls = random.uniform(temp_key, (max_K,))
            temp_success = temp_rolls < temp_prob
            repro_success = repro_success & temp_success

            # CRITICAL: Resolve offspring position conflicts
            # Rule: only place offspring if cell is empty AND no other offspring wants it
            # Convert positions to flat indices for conflict detection
            offspring_flat_pos = offspring_positions[:, 0] * size + offspring_positions[:, 1]

            # Count how many offspring want each cell
            claim_count = jnp.zeros(size * size, dtype=jnp.int32)
            claim_count = claim_count.at[offspring_flat_pos].add(repro_success.astype(jnp.int32))

            # Offspring succeeds only if it's the ONLY one wanting that cell
            # (cell being empty is already checked in try_reproduce_single via occupancy_grid)
            no_competition = claim_count[offspring_flat_pos] == 1
            repro_success = repro_success & no_competition

            # Find dead slots
            dead_mask = ~alive
            dead_indices = jnp.where(dead_mask, jnp.arange(n), n)
            dead_indices_sorted = jnp.sort(dead_indices)[:max_K]

            # Sort successful reproducers
            success_order = jnp.where(repro_success, jnp.arange(max_K), max_K)
            success_order_sorted = jnp.sort(success_order)

            safe_offspring_idx = jnp.minimum(success_order_sorted, max_K - 1)
            parent_agent_indices = safe_repro_idx[safe_offspring_idx]

            valid_arr = (success_order_sorted < max_K) & (dead_indices_sorted < n)

            safe_parent_idx = jnp.minimum(parent_agent_indices, n - 1)
            safe_dead_idx = jnp.minimum(dead_indices_sorted, n - 1)

            # Gather parent params
            parent_params_gathered = params[safe_parent_idx]

            # Mutate
            mutate_keys = random.split(keys[4], max_K)
            child_params = vmap(lambda k, p: mutate_params_flat(k, p, mutation_std))(
                mutate_keys, parent_params_gathered
            )

            # Gather offspring positions/orientations
            new_offspring_pos = offspring_positions[safe_offspring_idx]
            new_offspring_ori = offspring_orientations[safe_offspring_idx]

            # Scatter to dead slots
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

            # Reset child states
            zeros_state = jnp.zeros((max_K, hidden_dim * 4))
            agent_states = agent_states.at[safe_dead_idx].set(
                jnp.where(valid_arr[:, None], zeros_state, agent_states[safe_dead_idx])
            )

            # Scatter child params
            params = params.at[safe_dead_idx].set(
                jnp.where(valid_arr[:, None], child_params, params[safe_dead_idx])
            )

            # Note: Parent energy cost is already handled in Phase 2 via cost_reproduce

            # === Phase 7: Eating ===
            # No hard cap - metabolic penalty in Phase 2 creates soft pressure
            is_eating = (actions == EAT) & alive
            eat_y, eat_x = positions[:, 0], positions[:, 1]
            available = world_resource[eat_y, eat_x]
            eat_amounts = jnp.where(is_eating, available * eat_fraction, 0.0)
            energies = energies + eat_amounts
            world_resource = world_resource.at[eat_y, eat_x].add(-eat_amounts)

            # === Phase 8: Energy deaths ===
            alive = alive & (energies > 0)

            # === Phase 9: Resource regeneration ===
            world_resource = regenerate_resources(world_resource, world_resource_base, regen_timescale)

            # Compute num_alive for buffer management (cheap reduction)
            num_alive = jnp.sum(alive)

            return (
                world_resource, positions, orientations, params,
                agent_states, energies, alive, actions, num_alive, has_collision, num_kills, num_toxin_deaths
            )

        return step_jit

    def reset(self, key: jax.Array) -> dict:
        """Initialize/reset simulation state with pre-allocated buffers."""
        keys = random.split(key, 4)

        # Create world
        world = create_world(keys[0], self.config)

        # Initial and max agents - use power of 2 for JIT efficiency
        num_agents = self.config["world"]["initial_agents"]
        max_agents = 1
        while max_agents < max(num_agents * 2, self.min_buffer_size):
            max_agents *= 2

        # Initialize positions (only first num_agents are valid)
        total_cells = self.size * self.size
        flat_indices = random.choice(keys[1], total_cells, shape=(num_agents,), replace=False)
        positions = jnp.zeros((max_agents, 2), dtype=jnp.int32)
        positions = positions.at[:num_agents, 0].set(flat_indices // self.size)
        positions = positions.at[:num_agents, 1].set(flat_indices % self.size)

        # Random orientations
        orientations = jnp.zeros(max_agents, dtype=jnp.int32)
        orientations = orientations.at[:num_agents].set(
            random.randint(keys[2], (num_agents,), 0, 4)
        )

        # Initialize parameters for all slots
        params = _init_params_batched(keys[3], max_agents, self.hidden_dim, self.internal_noise_dim)

        # Initialize flat LSTM states: (max_agents, state_dim) where state_dim = 4 * hidden_dim
        states = jnp.zeros((max_agents, self.state_dim))

        # Energy and alive status
        energies = jnp.zeros(max_agents)
        energies = energies.at[:num_agents].set(self.config["energy"]["initial"])
        alive = jnp.zeros(max_agents, dtype=bool)
        alive = alive.at[:num_agents].set(True)

        # Kill agents spawned on toxin
        on_toxin = world["toxin"][positions[:, 0], positions[:, 1]] > 0.5
        alive = alive & ~on_toxin

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
        """Initialize simulation with pre-loaded agent parameters.

        Args:
            key: PRNG key
            agent_params: (num_agents, param_dim) array of neural network weights
            initial_energy: Energy to give each agent (default: config value)
            spawn_region: Optional (y_min, y_max, x_min, x_max) to restrict spawn area

        Returns:
            Initial state dict with loaded agents placed randomly in world
        """
        keys = random.split(key, 3)
        num_agents = len(agent_params)

        if initial_energy is None:
            initial_energy = self.config["energy"]["initial"]

        # Create world
        world = create_world(keys[0], self.config)

        # Buffer size - power of 2
        max_agents = 1
        while max_agents < max(num_agents * 2, self.min_buffer_size):
            max_agents *= 2

        # Random positions for loaded agents
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

        # Random orientations
        orientations = jnp.zeros(max_agents, dtype=jnp.int32)
        orientations = orientations.at[:num_agents].set(
            random.randint(keys[2], (num_agents,), 0, 4)
        )

        # Copy loaded params into buffer, fill rest with random
        params = _init_params_batched(keys[2], max_agents, self.hidden_dim, self.internal_noise_dim)
        params = params.at[:num_agents].set(agent_params)

        # Fresh LSTM states (don't carry over internal state)
        states = jnp.zeros((max_agents, self.state_dim))

        # Energy and alive status
        energies = jnp.zeros(max_agents)
        energies = energies.at[:num_agents].set(initial_energy)
        alive = jnp.zeros(max_agents, dtype=bool)
        alive = alive.at[:num_agents].set(True)

        # Kill agents spawned on toxin
        on_toxin = world["toxin"][positions[:, 0], positions[:, 1]] > 0.5
        alive = alive & ~on_toxin

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
        """Grow buffers to accommodate population. Called when >50% full.

        Doubles buffer size to ensure plenty of headroom.
        Uses power-of-2 sizes to minimize JIT recompilations.
        """
        max_agents = state["max_agents"]

        # Double the buffer size (power of 2)
        new_max = max_agents * 2

        grow_by = new_max - max_agents
        print(f"Growing buffers: {max_agents} -> {new_max} ({num_alive} alive, triggering JIT recompilation)")

        # Extend all arrays
        positions = jnp.concatenate([
            state["positions"],
            jnp.zeros((grow_by, 2), dtype=jnp.int32)
        ])
        orientations = jnp.concatenate([
            state["orientations"],
            jnp.zeros(grow_by, dtype=jnp.int32)
        ])
        energies = jnp.concatenate([
            state["energies"],
            jnp.zeros(grow_by)
        ])
        alive = jnp.concatenate([
            state["alive"],
            jnp.zeros(grow_by, dtype=bool)
        ])

        # Extend params (now a simple 2D array)
        new_params = _init_params_batched(key, grow_by, self.hidden_dim, self.internal_noise_dim)
        params = jnp.concatenate([state["params"], new_params], axis=0)

        # Extend flat states
        states = jnp.concatenate([state["states"], jnp.zeros((grow_by, self.state_dim))], axis=0)

        # Extend actions if present
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
        """Execute one fully vectorized simulation step using JIT-compiled kernel."""
        if self.debug:
            t0 = time.time()

        # Call the JIT-compiled step function
        world = state["world"]
        max_agents = state["max_agents"]

        # Update world for dynamic environments
        if self.is_thermotaxis:
            world = thermotaxis.update_temperature(world, state["step"])
        elif self.is_pretrain:
            world = pretrain.update_food(world, state["step"])

        (
            new_resource, new_positions, new_orientations, new_params,
            new_states, new_energies, new_alive, actions, num_alive, has_collision, num_kills, num_toxin_deaths
        ) = self._step_jit(
            key,
            world["resource"], world["resource_base"], world["temperature"], world["toxin"],
            state["positions"], state["orientations"], state["params"],
            state["states"], state["energies"], state["alive"],
            max_agents
        )

        # Sanity check: error if multiple agents in same cell (indicates simulation bug)
        if bool(has_collision):
            raise RuntimeError(f"BUG: Multiple alive agents occupy the same cell at step {state['step']}")

        if self.debug:
            jax.block_until_ready(new_alive)
            _record("step_total", time.time() - t0)

        # Build new state dict
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
            "num_kills": int(num_kills),
            "num_toxin_deaths": int(num_toxin_deaths),
        }

        # Preemptive buffer growth: grow when >50% full to avoid blocking reproduction
        # Reading num_alive is a cheap scalar sync; recompiles are rare (only on growth)
        num_alive_int = int(num_alive)
        if num_alive_int > max_agents * 0.5:
            grow_key = random.fold_in(key, 999)  # Deterministic subkey for growth
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
