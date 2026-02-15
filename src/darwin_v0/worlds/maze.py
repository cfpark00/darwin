"""Maze world - 8x8 cell maze where walls are toxic."""

import jax
import jax.numpy as jnp

from src.darwin_v0.worlds.base import generate_temperature_field


# Default 8x8 maze layout (1 = passable, 0 = wall)
# Start: bottom-left (0,0), Goal: top-right (7,7)
DEFAULT_MAZE = jnp.array([
    [1, 1, 1, 1, 1, 0, 1, 1],  # row 7 (top)
    [0, 0, 0, 0, 1, 0, 0, 1],  # row 6
    [1, 1, 1, 0, 1, 1, 0, 1],  # row 5
    [1, 0, 0, 0, 0, 1, 1, 1],  # row 4
    [1, 0, 1, 1, 1, 1, 0, 1],  # row 3
    [1, 0, 1, 0, 0, 0, 0, 1],  # row 2
    [1, 0, 1, 0, 1, 1, 1, 1],  # row 1
    [1, 1, 1, 0, 1, 1, 1, 1],  # row 0 (bottom)
], dtype=jnp.int8)


def create(key: jax.Array, config: dict) -> dict:
    """Create maze world with toxic walls.

    The maze is an 8x8 cell grid scaled up to the world size.
    Walls (0 in maze) become toxic zones that kill agents.
    Passable cells (1 in maze) have resources.
    The goal cell (top-right) has maximum resources.

    Arena config options (required):
        resource_start: Resource level in start cell
        resource_passable: Resource level in passable cells
        resource_goal: Resource level in goal cell

    Arena config options (optional):
        cell_size: Size of each maze cell in pixels (default: world_size // 8)
        resource_wall: Resource level in wall cells (default: 0.0)
        temperature_min, temperature_max: Temperature field bounds
        temperature_length_scale: Spatial smoothness of temperature
        maze: Custom 8x8 maze layout (2D list of 0s and 1s)

    Args:
        key: PRNG key
        config: World configuration dict

    Returns:
        World dict with: resource, resource_base, temperature, toxin
    """
    size = config["world"]["size"]
    arena = config.get("arena", {})

    # Get maze layout (use custom if provided, else default)
    if "maze" in arena:
        maze = jnp.array(arena["maze"], dtype=jnp.int8)
    else:
        maze = DEFAULT_MAZE

    maze_size = maze.shape[0]  # Assume square maze
    cell_size = arena.get("cell_size", size // maze_size)

    if "resource_start" not in arena:
        raise ValueError("FATAL: 'arena.resource_start' required for maze world")
    if "resource_passable" not in arena:
        raise ValueError("FATAL: 'arena.resource_passable' required for maze world")
    if "resource_goal" not in arena:
        raise ValueError("FATAL: 'arena.resource_goal' required for maze world")

    resource_start = arena["resource_start"]
    resource_passable = arena["resource_passable"]
    resource_goal = arena["resource_goal"]
    resource_wall = arena.get("resource_wall", 0.0)  # 0.0 is sensible for walls

    # Build resource and toxin grids by expanding maze cells
    resource = jnp.zeros((size, size), dtype=jnp.float32)
    toxin = jnp.zeros((size, size), dtype=jnp.float32)

    # Iterate over maze cells and fill corresponding regions
    # Note: maze[j, i] where j is row from top, i is column from left
    # We map: x (world) = i * cell_size, y (world) = (maze_size - 1 - j) * cell_size
    for j in range(maze_size):
        for i in range(maze_size):
            x_start = i * cell_size
            x_end = min((i + 1) * cell_size, size)
            # Flip y so row 0 of maze is at bottom of world
            y_start = (maze_size - 1 - j) * cell_size
            y_end = min((maze_size - j) * cell_size, size)

            if maze[j, i] == 0:  # Wall
                toxin = toxin.at[y_start:y_end, x_start:x_end].set(1.0)
                resource = resource.at[y_start:y_end, x_start:x_end].set(resource_wall)
            else:  # Passable
                resource = resource.at[y_start:y_end, x_start:x_end].set(resource_passable)

    # Set goal cell (top-right: i=7, j=0 in maze coords) to have max resources
    goal_i, goal_j = maze_size - 1, 0
    goal_x_start = goal_i * cell_size
    goal_x_end = min((goal_i + 1) * cell_size, size)
    goal_y_start = (maze_size - 1 - goal_j) * cell_size
    goal_y_end = min((maze_size - goal_j) * cell_size, size)
    resource = resource.at[goal_y_start:goal_y_end, goal_x_start:goal_x_end].set(resource_goal)

    # Set start cell (bottom-left: i=0, j=7 in maze coords) to have start resources
    start_i, start_j = 0, maze_size - 1
    start_x_start = start_i * cell_size
    start_x_end = min((start_i + 1) * cell_size, size)
    start_y_start = (maze_size - 1 - start_j) * cell_size
    start_y_end = min((maze_size - start_j) * cell_size, size)
    resource = resource.at[start_y_start:start_y_end, start_x_start:start_x_end].set(resource_start)

    # Temperature field
    temperature = generate_temperature_field(
        key, size,
        arena.get("temperature_min", 0.0),
        arena.get("temperature_max", 1.0),
        arena.get("temperature_length_scale", 400.0),
    )

    return {
        "resource": resource,
        "resource_base": resource.copy(),
        "temperature": temperature,
        "toxin": toxin,
        "resource_min": resource_wall,
        "resource_max": resource_goal,
    }
