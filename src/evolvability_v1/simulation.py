"""Core simulation engine for evolvability_v1.

Composes Agent + Environment with proper handshake validation.
Single implementation that handles all environment types.
"""

import time
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random, vmap

from src.evolvability_v1.types import (
    SimState, Metrics, AgentConfig, PhysicsConfig, RunConfig,
    EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE, ATTACK,
)
from src.evolvability_v1.agent import (
    compute_param_dim, compute_state_dim,
    init_params_batch, init_brain_states_batch,
    forward, sample_action, mutate_params,
)
from src.evolvability_v1.environment import Environment
from src.evolvability_v1.physics import (
    get_direction_deltas, compute_intended_position, compute_new_orientation,
    build_occupancy_grid, resolve_move_conflicts, regenerate_resources,
    get_action_costs,
)


class Simulation:
    """Main simulation engine.

    Composes an Environment (defines observations/actions) with agent physics.
    Validates that agent I/O matches environment at init time.
    """

    def __init__(
        self,
        environment: Environment,
        agent_config: AgentConfig,
        physics_config: PhysicsConfig,
        run_config: RunConfig,
    ):
        """Initialize simulation with environment and configs.

        Args:
            environment: Environment instance defining observations/actions
            agent_config: Agent neural network configuration
            physics_config: Physics/energy configuration
            run_config: Run parameters (steps, logging, etc.)

        Raises:
            ValueError: If agent I/O doesn't match environment
        """
        # === HANDSHAKE: Validate agent matches environment ===
        if agent_config.input_dim != environment.input_dim:
            raise ValueError(
                f"Agent/Environment I/O mismatch!\n"
                f"  Agent expects {agent_config.input_dim} inputs\n"
                f"  Environment provides {environment.input_dim} inputs"
            )
        if agent_config.output_dim != environment.output_dim:
            raise ValueError(
                f"Agent/Environment I/O mismatch!\n"
                f"  Agent outputs {agent_config.output_dim} actions\n"
                f"  Environment accepts {environment.output_dim} actions"
            )

        self.env = environment
        self.agent_config = agent_config
        self.physics = physics_config
        self.run_config = run_config

        # Pre-compute dimensions
        self.param_dim = compute_param_dim(agent_config)
        self.state_dim = compute_state_dim(agent_config)
        self.size = physics_config.world_size

        # Capability flags from environment
        self.has_toxin = environment.has_toxin
        self.has_attack = environment.has_attack

        # Build action costs array (include attack cost if has_attack)
        cost_dict = {
            "cost_eat": physics_config.cost_eat,
            "cost_move": physics_config.cost_move,
            "cost_stay": physics_config.cost_stay,
            "cost_reproduce": physics_config.cost_reproduce,
        }
        if self.has_attack:
            cost_dict["cost_attack"] = physics_config.cost_attack
        self.action_costs = get_action_costs(agent_config.output_dim, cost_dict)

        # Build JIT-compiled step function with capability flags
        self._step_jit = self._build_step_fn(
            has_toxin=self.has_toxin,
            has_attack=self.has_attack,
        )

    def _validate_world(self, world: dict):
        """Validate that world has required fields for environment capabilities."""
        required = ["resource", "resource_base", "temperature"]
        if self.has_toxin:
            required.append("toxin")

        for field in required:
            if field not in world:
                raise ValueError(
                    f"World missing required field '{field}'.\n"
                    f"Environment has_toxin={self.has_toxin}, has_attack={self.has_attack}\n"
                    f"World keys: {list(world.keys())}"
                )

    def reset(self, key: jax.Array) -> SimState:
        """Initialize simulation with random agents."""
        k1, k2 = random.split(key)

        n = self.run_config.initial_agents
        buffer_size = max(self.run_config.min_buffer_size, n * 2)

        # Create world and validate
        world = self.env.create_world(k1, self.size)
        self._validate_world(world)

        # Initialize agents
        k_params, k_pos, k_ori = random.split(k2, 3)

        params = jnp.zeros((buffer_size, self.param_dim))
        params = params.at[:n].set(init_params_batch(k_params, n, self.agent_config))

        brain_states = init_brain_states_batch(buffer_size, self.agent_config)

        # Random positions (avoiding collisions)
        positions = self._sample_unique_positions(k_pos, n, buffer_size)
        orientations = jnp.zeros(buffer_size, dtype=jnp.int32)
        orientations = orientations.at[:n].set(random.randint(k_ori, (n,), 0, 4))

        energies = jnp.zeros(buffer_size)
        energies = energies.at[:n].set(self.physics.max_energy)

        alive = jnp.zeros(buffer_size, dtype=bool)
        alive = alive.at[:n].set(True)

        ages = jnp.zeros(buffer_size, dtype=jnp.int32)

        # Lineage tracking
        uid = jnp.arange(buffer_size, dtype=jnp.int32)
        parent_uid = jnp.full(buffer_size, -1, dtype=jnp.int32)

        return SimState(
            world=world,
            positions=positions,
            orientations=orientations,
            params=params,
            brain_states=brain_states,
            energies=energies,
            alive=alive,
            ages=ages,
            uid=uid,
            parent_uid=parent_uid,
            next_uid=buffer_size,
            step=0,
            max_agents=buffer_size,
            actions=None,
        )

    def reset_with_agents(
        self,
        key: jax.Array,
        params: jax.Array,
        initial_energy: Optional[float] = None,
        spawn_region: Optional[tuple] = None,
    ) -> SimState:
        """Initialize simulation with provided agent parameters.

        Args:
            key: Random key
            params: Agent parameters (n_agents, param_dim)
            initial_energy: Starting energy (default: max_energy)
            spawn_region: Optional (y_min, y_max, x_min, x_max) for spawning

        Returns:
            Initial simulation state
        """
        k1, k2, k3 = random.split(key, 3)

        n = len(params)
        buffer_size = max(self.run_config.min_buffer_size, n * 2)

        # Validate param dimensions
        if params.shape[1] != self.param_dim:
            raise ValueError(
                f"Parameter dimension mismatch!\n"
                f"  Provided: {params.shape[1]}\n"
                f"  Expected: {self.param_dim}"
            )

        # Create world and validate
        world = self.env.create_world(k1, self.size)
        self._validate_world(world)

        # Initialize agent arrays
        all_params = jnp.zeros((buffer_size, self.param_dim))
        all_params = all_params.at[:n].set(params)

        brain_states = init_brain_states_batch(buffer_size, self.agent_config)

        # Spawn positions
        if spawn_region is not None:
            y_min, y_max, x_min, x_max = spawn_region
            positions = self._sample_unique_positions_in_region(
                k2, n, buffer_size, y_min, y_max, x_min, x_max
            )
        else:
            positions = self._sample_unique_positions(k2, n, buffer_size)

        orientations = jnp.zeros(buffer_size, dtype=jnp.int32)
        orientations = orientations.at[:n].set(random.randint(k3, (n,), 0, 4))

        energy = initial_energy if initial_energy is not None else self.physics.max_energy
        energies = jnp.zeros(buffer_size)
        energies = energies.at[:n].set(energy)

        alive = jnp.zeros(buffer_size, dtype=bool)
        alive = alive.at[:n].set(True)

        ages = jnp.zeros(buffer_size, dtype=jnp.int32)

        # Lineage tracking
        uid = jnp.arange(buffer_size, dtype=jnp.int32)
        parent_uid = jnp.full(buffer_size, -1, dtype=jnp.int32)

        return SimState(
            world=world,
            positions=positions,
            orientations=orientations,
            params=all_params,
            brain_states=brain_states,
            energies=energies,
            alive=alive,
            ages=ages,
            uid=uid,
            parent_uid=parent_uid,
            next_uid=buffer_size,
            step=0,
            max_agents=buffer_size,
            actions=None,
        )

    def _sample_unique_positions(
        self, key: jax.Array, n: int, buffer_size: int
    ) -> jax.Array:
        """Sample n unique positions, pad rest with zeros."""
        # Simple approach: sample more than needed, take unique
        positions = jnp.zeros((buffer_size, 2), dtype=jnp.int32)

        # Sample random positions
        k1, k2 = random.split(key)
        y = random.randint(k1, (n,), 0, self.size)
        x = random.randint(k2, (n,), 0, self.size)
        positions = positions.at[:n, 0].set(y)
        positions = positions.at[:n, 1].set(x)

        return positions

    def _sample_unique_positions_in_region(
        self, key: jax.Array, n: int, buffer_size: int,
        y_min: int, y_max: int, x_min: int, x_max: int,
    ) -> jax.Array:
        """Sample n positions within specified region."""
        positions = jnp.zeros((buffer_size, 2), dtype=jnp.int32)

        k1, k2 = random.split(key)
        y = random.randint(k1, (n,), y_min, y_max)
        x = random.randint(k2, (n,), x_min, x_max)
        positions = positions.at[:n, 0].set(y)
        positions = positions.at[:n, 1].set(x)

        return positions

    def step(self, state: SimState, key: jax.Array) -> SimState:
        """Execute one simulation step."""
        # Update world (for dynamic environments)
        world = self.env.update_world(state.world, state.step)

        # Get toxin field (or dummy if not applicable)
        if self.has_toxin:
            world_toxin = world["toxin"]
        else:
            # Dummy toxin field (all False) for environments without toxin
            world_toxin = jnp.zeros((self.size, self.size), dtype=bool)

        # Run JIT-compiled step
        results = self._step_jit(
            key,
            world["resource"],
            world["resource_base"],
            world["temperature"],
            world_toxin,
            state.positions,
            state.orientations,
            state.params,
            state.brain_states,
            state.energies,
            state.alive,
            state.ages,
            state.uid,
            state.parent_uid,
            state.next_uid,
            state.max_agents,
        )

        (
            new_resource, new_positions, new_orientations, new_params,
            new_brain_states, new_energies, new_alive, new_ages, actions,
            new_uid, new_parent_uid, new_next_uid,
            num_alive, mean_energy, mean_age, max_age, action_counts,
            repro_capped, num_toxin_deaths, num_attacks, num_kills
        ) = results

        # Update world with new resource
        new_world = {**world, "resource": new_resource}

        # Create metrics (stays on device until explicitly synced)
        metrics = Metrics(
            num_alive=num_alive,
            mean_energy=mean_energy,
            mean_age=mean_age,
            max_age=max_age,
            action_counts=action_counts,
            repro_capped=repro_capped,
            num_toxin_deaths=num_toxin_deaths,
            num_attacks=num_attacks,
            num_kills=num_kills,
        )

        new_state = SimState(
            world=new_world,
            positions=new_positions,
            orientations=new_orientations,
            params=new_params,
            brain_states=new_brain_states,
            energies=new_energies,
            alive=new_alive,
            ages=new_ages,
            uid=new_uid,
            parent_uid=new_parent_uid,
            next_uid=new_next_uid,
            step=state.step + 1,
            max_agents=state.max_agents,
            metrics=metrics,
            actions=actions,
        )

        # Buffer growth: grow when population > threshold * buffer_size
        # This is checked after step to ensure next step has room.
        # Reading num_alive is a cheap scalar sync; recompiles are rare (only on growth).
        num_alive_int = int(num_alive)
        if num_alive_int > state.max_agents * self.run_config.buffer_growth_threshold:
            new_size = state.max_agents * 2
            print(f"\nBuffer growth: {state.max_agents} -> {new_size} (pop={num_alive_int})")
            new_state = self.grow_buffer(new_state, new_size)

        return new_state

    def _build_step_fn(self, has_toxin: bool = False, has_attack: bool = False):
        """Build JIT-compiled step function with config in closure.

        Args:
            has_toxin: Whether to enable toxin death physics
            has_attack: Whether to enable attack physics
        """
        # Capture config values
        size = self.size
        hidden_dim = self.agent_config.hidden_dim
        internal_noise_dim = self.agent_config.internal_noise_dim
        input_dim = self.agent_config.input_dim
        output_dim = self.agent_config.output_dim

        max_energy = self.physics.max_energy
        base_cost = self.physics.base_cost
        base_cost_incremental = self.physics.base_cost_incremental
        mutation_std = self.physics.mutation_std
        offspring_energy = self.physics.offspring_energy
        eat_fraction = self.physics.eat_fraction
        regen_timescale = self.physics.regen_timescale
        action_costs = self.action_costs

        # Clamp options for controlled experiments
        energy_clamp = self.physics.energy_clamp
        resource_clamp = self.physics.resource_clamp
        disabled_actions = self.physics.disabled_actions

        # Build action mask from disabled_actions
        action_mask_base = self.env.get_action_mask()
        if disabled_actions:
            if action_mask_base is None:
                action_mask_base = jnp.ones(self.agent_config.output_dim, dtype=bool)
            for a in disabled_actions:
                action_mask_base = action_mask_base.at[a].set(False)

        # Get environment's observation function
        env = self.env

        # Capture capability flags in closure (they're static for this JIT)
        _has_toxin = has_toxin
        _has_attack = has_attack
        _energy_clamp = energy_clamp
        _resource_clamp = resource_clamp
        _action_mask = action_mask_base

        @partial(jax.jit, static_argnums=(15,))
        def step_jit(
            key,
            world_resource, world_resource_base, world_temperature, world_toxin,
            positions, orientations, params, brain_states, energies, alive, ages,
            uid, parent_uid, next_uid,
            max_agents,
        ):
            n = max_agents
            keys = random.split(key, 6)

            # === Phase 1: Observe and decide ===
            obs_keys = random.split(keys[0], n)
            action_keys = random.split(keys[1], n)
            noise = random.normal(keys[2], (n, internal_noise_dim))

            occupancy_grid = build_occupancy_grid(positions, alive, size)

            # Build world dict for observation
            world = {
                "resource": world_resource,
                "temperature": world_temperature,
                "toxin": world_toxin,
            }

            # Compute observations (vectorized)
            def compute_obs(i, obs_key):
                return env.compute_observation(
                    world, positions[i], orientations[i],
                    occupancy_grid, i, obs_key,
                )

            observations = vmap(compute_obs)(jnp.arange(n), obs_keys)

            # Forward pass through agent network
            def forward_agent(p, s, obs, e, noise_i):
                return forward(
                    p, s, obs, e / max_energy, noise_i,
                    input_dim, output_dim, hidden_dim, internal_noise_dim,
                )

            brain_states, all_logits = vmap(forward_agent)(
                params, brain_states, observations, energies, noise
            )

            # Sample actions (with optional disabled actions mask)
            actions = vmap(lambda k, l: sample_action(k, l, 1.0, _action_mask))(
                action_keys, all_logits
            )
            actions = jnp.where(alive, actions, STAY)

            # === Phase 2: Energy costs ===
            energy_penalty = base_cost_incremental * energies
            total_costs = base_cost + energy_penalty + action_costs[actions]
            energies = energies - total_costs * alive.astype(jnp.float32)

            # === Phase 2b: Attack (if enabled) ===
            num_attacks = jnp.array(0, dtype=jnp.int32)
            num_kills = jnp.array(0, dtype=jnp.int32)
            if _has_attack:
                # Find who is attacking
                is_attacking = (actions == ATTACK) & alive
                num_attacks = jnp.sum(is_attacking.astype(jnp.int32))

                # Compute target positions (cell in front of attacker)
                deltas = get_direction_deltas()
                attack_target_y = positions[:, 0] + deltas[orientations, 0]
                attack_target_x = positions[:, 1] + deltas[orientations, 1]

                # Clamp to world bounds
                attack_target_y = jnp.clip(attack_target_y, 0, size - 1)
                attack_target_x = jnp.clip(attack_target_x, 0, size - 1)

                # Check who is at target positions using occupancy grid
                target_agent_ids = occupancy_grid[attack_target_y, attack_target_x]
                # occupancy_grid has agent_idx+1 (0 = empty), so subtract 1
                target_agent_idx = target_agent_ids - 1

                # Valid attack: attacker is attacking AND target cell has an agent
                valid_attack = is_attacking & (target_agent_ids > 0)

                # Kill targets: mark them as dead
                # We need to scatter the kill flags to the target indices
                # target_agent_idx is -1 for empty cells, clip to valid range
                safe_target_idx = jnp.clip(target_agent_idx, 0, n - 1)

                # Create kill mask: for each agent, check if they're a valid target
                # Use scatter to mark killed agents
                kill_flags = jnp.zeros(n, dtype=bool)
                # Only update if valid_attack is true
                kill_flags = kill_flags.at[safe_target_idx].set(
                    kill_flags[safe_target_idx] | valid_attack
                )

                # Count kills before applying
                num_kills = jnp.sum(kill_flags.astype(jnp.int32))

                # Kill the targets
                alive = alive & ~kill_flags

                # Drop victim energy to ground (add to resource at their position)
                victim_energy = jnp.where(kill_flags, energies, 0.0)
                world_resource = world_resource.at[positions[:, 0], positions[:, 1]].add(victim_energy)

                # Zero out killed agents' energy
                energies = jnp.where(kill_flags, 0.0, energies)

            # === Phase 3: Movement ===
            intended_positions = vmap(
                lambda p, o, a: compute_intended_position(p, o, a, size)
            )(positions, orientations, actions)
            positions = resolve_move_conflicts(
                intended_positions, positions, actions, alive, size
            )
            orientations = vmap(compute_new_orientation)(orientations, actions)

            # === Phase 4: Reproduction ===
            occupancy_grid = build_occupancy_grid(positions, alive, size)
            # max_K = buffer size, so reproduction limit is physics-based (dead slots available)
            # not an arbitrary cap. Triggers recompilation if buffer grows.
            max_K = max_agents

            wants_repro = (actions == REPRODUCE) & alive

            repro_indices = jnp.where(wants_repro, jnp.arange(n), n)
            repro_sorted = jnp.sort(repro_indices)[:max_K]
            repro_valid = repro_sorted < n
            safe_repro_idx = jnp.minimum(repro_sorted, n - 1)

            deltas = get_direction_deltas()

            def try_reproduce(agent_idx, k):
                ori = orientations[agent_idx]
                pos = positions[agent_idx]
                candidates = jnp.stack([
                    pos + deltas[ori],
                    pos + deltas[(ori + 2) % 4],
                    pos + deltas[(ori + 3) % 4],
                    pos + deltas[(ori + 1) % 4],
                ])

                def is_valid(cell):
                    in_bounds = (
                        (cell[0] >= 0) & (cell[0] < size) &
                        (cell[1] >= 0) & (cell[1] < size)
                    )
                    occupied = jnp.where(
                        in_bounds,
                        occupancy_grid[cell[0] % size, cell[1] % size] > 0,
                        True
                    )
                    return in_bounds & ~occupied

                valid_mask = vmap(is_valid)(candidates)
                any_valid = jnp.any(valid_mask)

                k1, k2 = random.split(k)
                probs = valid_mask.astype(jnp.float32)
                probs = probs / (probs.sum() + 1e-10)
                chosen = random.categorical(k1, jnp.log(probs + 1e-10))
                offspring_pos = candidates[chosen]
                offspring_ori = random.randint(k2, (), 0, 4)

                return any_valid, offspring_pos, offspring_ori

            repro_keys = random.split(keys[3], max_K)
            can_repro, offspring_pos, offspring_ori = vmap(try_reproduce)(
                safe_repro_idx, repro_keys
            )
            repro_success = repro_valid & can_repro

            # Resolve offspring position conflicts
            flat_pos = offspring_pos[:, 0] * size + offspring_pos[:, 1]
            claim_count = jnp.zeros(size * size, dtype=jnp.int32)
            claim_count = claim_count.at[flat_pos].add(repro_success.astype(jnp.int32))
            no_conflict = claim_count[flat_pos] == 1
            repro_success = repro_success & no_conflict

            # Find dead slots
            dead_mask = ~alive
            dead_indices = jnp.where(dead_mask, jnp.arange(n), n)
            dead_sorted = jnp.sort(dead_indices)[:max_K]

            success_order = jnp.where(repro_success, jnp.arange(max_K), max_K)
            success_sorted = jnp.sort(success_order)

            safe_offspring_idx = jnp.minimum(success_sorted, max_K - 1)
            parent_indices = safe_repro_idx[safe_offspring_idx]

            valid = (success_sorted < max_K) & (dead_sorted < n)
            safe_parent = jnp.minimum(parent_indices, n - 1)
            safe_dead = jnp.minimum(dead_sorted, n - 1)

            # Track unphysical events: reproductions that failed due to lack of dead slots
            # These agents paid reproduction cost but got nothing (unphysical!)
            num_wanted_repro = jnp.sum(repro_success.astype(jnp.int32))
            num_actual_repro = jnp.sum(valid.astype(jnp.int32))
            repro_capped = num_wanted_repro - num_actual_repro

            # Mutate parent params for offspring
            parent_params = params[safe_parent]
            mutate_keys = random.split(keys[4], max_K)
            child_params = vmap(lambda k, p: mutate_params(k, p, mutation_std))(
                mutate_keys, parent_params
            )

            new_pos = offspring_pos[safe_offspring_idx]
            new_ori = offspring_ori[safe_offspring_idx]

            # Update arrays for offspring
            positions = positions.at[safe_dead].set(
                jnp.where(valid[:, None], new_pos, positions[safe_dead])
            )
            orientations = orientations.at[safe_dead].set(
                jnp.where(valid, new_ori, orientations[safe_dead])
            )
            energies = energies.at[safe_dead].set(
                jnp.where(valid, offspring_energy, energies[safe_dead])
            )
            ages = ages.at[safe_dead].set(
                jnp.where(valid, 0, ages[safe_dead])
            )
            alive = alive.at[safe_dead].set(
                jnp.where(valid, True, alive[safe_dead])
            )

            zeros_state = jnp.zeros((max_K, hidden_dim * 4))
            brain_states = brain_states.at[safe_dead].set(
                jnp.where(valid[:, None], zeros_state, brain_states[safe_dead])
            )
            params = params.at[safe_dead].set(
                jnp.where(valid[:, None], child_params, params[safe_dead])
            )

            # Lineage tracking
            parent_uids = uid[safe_parent]
            num_births = jnp.sum(valid.astype(jnp.int32))
            uid_offsets = jnp.cumsum(valid.astype(jnp.int32)) - 1
            new_uids = next_uid + uid_offsets

            uid = uid.at[safe_dead].set(
                jnp.where(valid, new_uids, uid[safe_dead])
            )
            parent_uid = parent_uid.at[safe_dead].set(
                jnp.where(valid, parent_uids, parent_uid[safe_dead])
            )
            next_uid = next_uid + num_births

            # === Phase 5: Eating ===
            is_eating = (actions == EAT) & alive
            eat_y, eat_x = positions[:, 0], positions[:, 1]
            available = world_resource[eat_y, eat_x]
            eat_amounts = jnp.where(is_eating, available * eat_fraction, 0.0)
            energies = energies + eat_amounts
            world_resource = world_resource.at[eat_y, eat_x].add(-eat_amounts)

            # Clamp energy to max
            energies = jnp.minimum(energies, max_energy)

            # === Phase 5b: Energy clamp (optional, for controlled experiments) ===
            if _energy_clamp is not None:
                energies = jnp.where(alive, _energy_clamp, energies)

            # === Phase 6: Energy deaths ===
            alive_before_energy_death = alive
            alive = alive & (energies > 0)

            # === Phase 6b: Toxin deaths (if enabled) ===
            num_toxin_deaths = jnp.array(0, dtype=jnp.int32)
            if _has_toxin:
                alive_before_toxin = alive
                on_toxin = world_toxin[positions[:, 0], positions[:, 1]]
                alive = alive & ~on_toxin
                num_toxin_deaths = jnp.sum((alive_before_toxin & ~alive).astype(jnp.int32))

            # === Phase 7: Age increment ===
            ages = ages + alive.astype(jnp.int32)

            # === Phase 8: Resource regeneration ===
            world_resource = regenerate_resources(
                world_resource, world_resource_base, regen_timescale
            )

            # === Phase 8b: Resource clamp (optional, for controlled experiments) ===
            if _resource_clamp is not None:
                world_resource = jnp.full_like(world_resource, _resource_clamp)

            # === Compute metrics using masked reductions (no fancy indexing!) ===
            num_alive = jnp.sum(alive)
            alive_float = alive.astype(jnp.float32)
            eps = 1e-8

            # Masked mean energy
            mean_energy = jnp.sum(energies * alive_float) / (num_alive + eps)

            # Masked mean/max age
            ages_float = ages.astype(jnp.float32)
            mean_age = jnp.sum(ages_float * alive_float) / (num_alive + eps)
            # For max, use where to mask dead agents to -inf equivalent
            max_age = jnp.max(jnp.where(alive, ages, 0))

            # Action distribution (masked)
            def count_action(action_id):
                is_action = (actions == action_id).astype(jnp.float32)
                return jnp.sum(is_action * alive_float) / (num_alive + eps)

            action_counts = jnp.array([count_action(i) for i in range(output_dim)])

            return (
                world_resource, positions, orientations, params,
                brain_states, energies, alive, ages, actions,
                uid, parent_uid, next_uid,
                num_alive, mean_energy, mean_age, max_age, action_counts,
                repro_capped, num_toxin_deaths, num_attacks, num_kills
            )

        return step_jit

    def get_stats(self, state: SimState) -> dict:
        """Convert pre-computed metrics to Python dict for logging.

        Metrics are computed inside JIT during step(), so this just
        does the device->host transfer. Call sparingly (every N steps).
        """
        stats = {"step": state.step, "max_agents": state.max_agents}

        if state.metrics is not None:
            m = state.metrics
            # Single sync point - all metrics transfer together
            stats.update({
                "num_alive": int(m.num_alive),
                "mean_energy": float(m.mean_energy),
                "mean_age": float(m.mean_age),
                "max_age": int(m.max_age),
                # Unphysical event tracking (should be 0 in well-configured sim)
                "repro_capped": int(m.repro_capped),
                # Combat/hazard metrics (only meaningful if env has these capabilities)
                "num_toxin_deaths": int(m.num_toxin_deaths),
                "num_attacks": int(m.num_attacks),
                "num_kills": int(m.num_kills),
            })

            # Action distribution
            action_counts = m.action_counts
            for i, name in enumerate(self.env.action_names):
                stats[f"action_{name}"] = float(action_counts[i])
        else:
            # Fallback for initial state before first step
            stats["num_alive"] = int(jnp.sum(state.alive))
            stats["repro_capped"] = 0
            stats["num_toxin_deaths"] = 0
            stats["num_attacks"] = 0
            stats["num_kills"] = 0

        return stats

    def get_metrics_jax(self, state: SimState) -> Metrics:
        """Return metrics as JAX arrays (no sync, stays on device).

        Use this inside tight loops when you don't need Python values yet.
        """
        return state.metrics

    def grow_buffer(self, state: SimState, new_size: int) -> SimState:
        """Grow agent buffer to new_size.

        Creates new larger arrays and copies existing data.
        This triggers JIT recompilation since max_agents is static.

        Args:
            state: Current simulation state
            new_size: New buffer size (must be > current max_agents)

        Returns:
            New SimState with larger buffer capacity
        """
        old_size = state.max_agents
        if new_size <= old_size:
            raise ValueError(f"new_size ({new_size}) must be > current max_agents ({old_size})")

        # Helper to grow an array
        def grow_array(arr, fill_value=0):
            new_shape = (new_size,) + arr.shape[1:]
            new_arr = jnp.full(new_shape, fill_value, dtype=arr.dtype)
            new_arr = new_arr.at[:old_size].set(arr)
            return new_arr

        # Grow all agent arrays
        new_positions = grow_array(state.positions)
        new_orientations = grow_array(state.orientations)
        new_params = grow_array(state.params, fill_value=0.0)
        new_brain_states = grow_array(state.brain_states, fill_value=0.0)
        new_energies = grow_array(state.energies, fill_value=0.0)
        new_alive = grow_array(state.alive, fill_value=False)
        new_ages = grow_array(state.ages)
        new_uid = jnp.arange(new_size, dtype=jnp.int32)
        new_uid = new_uid.at[:old_size].set(state.uid)
        new_parent_uid = grow_array(state.parent_uid, fill_value=-1)

        # Grow actions if present
        new_actions = None
        if state.actions is not None:
            new_actions = grow_array(state.actions)

        return SimState(
            world=state.world,
            positions=new_positions,
            orientations=new_orientations,
            params=new_params,
            brain_states=new_brain_states,
            energies=new_energies,
            alive=new_alive,
            ages=new_ages,
            uid=new_uid,
            parent_uid=new_parent_uid,
            next_uid=state.next_uid,
            step=state.step,
            max_agents=new_size,
            metrics=state.metrics,
            actions=new_actions,
        )
