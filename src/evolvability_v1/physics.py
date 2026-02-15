"""Physics helpers for evolvability_v1.

Handles movement, collision detection, and resource mechanics.
Largely reused from darwin_v0 but cleaned up.
"""

import jax
import jax.numpy as jnp
from jax import random

from src.evolvability_v1.types import EAT, FORWARD, LEFT, RIGHT, STAY, REPRODUCE, ATTACK


def get_direction_deltas() -> jax.Array:
    """Direction deltas: 0=N(-y), 1=E(+x), 2=S(+y), 3=W(-x)."""
    return jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)


def compute_intended_position(
    position: jax.Array,
    orientation: int,
    action: int,
    size: int,
) -> jax.Array:
    """Compute where agent wants to move (may be blocked)."""
    deltas = get_direction_deltas()

    # Only FORWARD action causes movement
    is_forward = action == FORWARD
    delta = jnp.where(is_forward, deltas[orientation], jnp.array([0, 0]))
    new_pos = position + delta

    # Clamp to world bounds
    new_pos = jnp.clip(new_pos, 0, size - 1)

    return new_pos


def compute_new_orientation(orientation: int, action: int) -> int:
    """Update orientation based on turn action."""
    # LEFT = turn counter-clockwise, RIGHT = turn clockwise
    delta = jnp.where(action == LEFT, -1, jnp.where(action == RIGHT, 1, 0))
    return (orientation + delta) % 4


def build_occupancy_grid(
    positions: jax.Array,
    alive: jax.Array,
    size: int,
) -> jax.Array:
    """Build grid where grid[y,x] = agent_idx+1 if occupied, 0 if empty."""
    n = len(alive)
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    agent_ids = jnp.where(alive, jnp.arange(n) + 1, 0)
    grid = grid.at[positions[:, 0], positions[:, 1]].max(agent_ids)
    return grid


def resolve_move_conflicts(
    intended_positions: jax.Array,
    current_positions: jax.Array,
    actions: jax.Array,
    alive: jax.Array,
    size: int,
) -> jax.Array:
    """Resolve movement conflicts - if multiple agents want same cell, none move."""
    n = len(alive)

    # Only consider alive agents trying to move
    is_moving = (actions == FORWARD) & alive

    # Count claims per cell
    flat_intended = intended_positions[:, 0] * size + intended_positions[:, 1]
    claim_grid = jnp.zeros(size * size, dtype=jnp.int32)
    claim_grid = claim_grid.at[flat_intended].add(is_moving.astype(jnp.int32))

    # Agent can move only if they're the sole claimant
    claims_at_intended = claim_grid[flat_intended]
    can_move = is_moving & (claims_at_intended == 1)

    # Also check that target cell is currently empty
    current_occupancy = build_occupancy_grid(current_positions, alive, size)
    target_y, target_x = intended_positions[:, 0], intended_positions[:, 1]
    target_empty = current_occupancy[target_y, target_x] == 0

    can_move = can_move & target_empty

    # Apply movement
    new_positions = jnp.where(
        can_move[:, None],
        intended_positions,
        current_positions,
    )

    return new_positions


def regenerate_resources(
    resource: jax.Array,
    resource_base: jax.Array,
    timescale: float,
) -> jax.Array:
    """Regenerate resources toward base level."""
    if timescale <= 0:
        return resource
    alpha = 1.0 / timescale
    return resource + alpha * (resource_base - resource)


def get_action_costs(output_dim: int, config: dict) -> jax.Array:
    """Build action cost array from config."""
    costs = [
        config.get("cost_eat", 0.0),
        config.get("cost_move", 2.5),
        config.get("cost_move", 2.5),  # left
        config.get("cost_move", 2.5),  # right
        config.get("cost_stay", 0.0),
        config.get("cost_reproduce", 29.5),
    ]
    if output_dim >= 7:
        costs.append(config.get("cost_attack", 5.0))

    return jnp.array(costs[:output_dim], dtype=jnp.float32)
