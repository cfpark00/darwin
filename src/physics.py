"""Physics and dynamics - pure definitions."""

import jax
import jax.numpy as jnp
import numpy as np

# Action indices
EAT = 0
FORWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
REPRODUCE = 5
ATTACK = 6

# Direction deltas (dy, dx) for each orientation: 0=up, 1=right, 2=down, 3=left
_DIRECTION_DELTAS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int32)


def get_direction_deltas() -> jax.Array:
    """Get direction deltas as JAX array."""
    return jnp.array(_DIRECTION_DELTAS)


def compute_intended_position(pos: jax.Array, ori: jax.Array, action: jax.Array,
                              size: int) -> jax.Array:
    """Compute intended position after action (before conflict resolution)."""
    deltas = get_direction_deltas()
    front_pos = pos + deltas[ori]

    is_forward = action == FORWARD
    in_bounds = (front_pos[0] >= 0) & (front_pos[0] < size) & \
                (front_pos[1] >= 0) & (front_pos[1] < size)

    return jnp.where(is_forward & in_bounds, front_pos, pos)


def compute_new_orientation(ori: jax.Array, action: jax.Array) -> jax.Array:
    """Compute new orientation after action."""
    new_ori = jnp.where(action == LEFT, (ori + 3) % 4,
              jnp.where(action == RIGHT, (ori + 1) % 4, ori))
    return new_ori


def compute_energy_cost(action: jax.Array, world_config: dict) -> jax.Array:
    """Compute energy cost for an action."""
    costs = jnp.array([
        0.0,  # EAT - energy gain handled separately
        world_config["energy"]["cost_move"],      # FORWARD
        world_config["energy"]["cost_move"],      # LEFT
        world_config["energy"]["cost_move"],      # RIGHT
        world_config["energy"]["cost_stay"],      # STAY
        world_config["energy"]["cost_reproduce"], # REPRODUCE
        world_config["energy"]["cost_attack"],    # ATTACK
    ])
    return costs[action]


def resolve_move_conflicts(intended_positions: jax.Array, current_positions: jax.Array,
                           actions: jax.Array, alive: jax.Array, size: int) -> jax.Array:
    """Resolve move conflicts: only move if target is empty AND no one else wants it.

    Conservative rule: conflict → everyone loses (stays in place).
    This avoids cascading dependency issues with chains of movers.
    """
    n = len(alive)
    is_moving = (actions == FORWARD) & alive

    # Build grid counting how many movers want each cell
    move_count_grid = jnp.zeros((size, size), dtype=jnp.int32)
    mover_y = jnp.where(is_moving, intended_positions[:, 0], 0)
    mover_x = jnp.where(is_moving, intended_positions[:, 1], 0)
    move_count_grid = move_count_grid.at[mover_y, mover_x].add(is_moving.astype(jnp.int32))

    # Build grid of ALL occupied cells (anyone currently there, moving or not)
    # Use .max() instead of .set() to handle duplicate positions correctly.
    # Dead agents may share positions (e.g., all at [0,0] after buffer growth),
    # and .set() with duplicates uses last-write-wins, which could overwrite True with False.
    occupied_grid = jnp.zeros((size, size), dtype=bool)
    occupied_grid = occupied_grid.at[current_positions[:, 0], current_positions[:, 1]].max(alive)

    # Can only move if: target is empty AND no one else wants it
    def check_can_move(i):
        target = intended_positions[i]
        num_wanting = move_count_grid[target[0], target[1]]
        target_occupied = occupied_grid[target[0], target[1]]
        # Only move if: I'm the only one wanting it AND it's currently empty
        can_move = (num_wanting == 1) & ~target_occupied
        return jnp.where(can_move & is_moving[i], intended_positions[i], current_positions[i])

    resolved = jax.vmap(check_can_move)(jnp.arange(n))
    return resolved
