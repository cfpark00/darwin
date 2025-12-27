"""World generation module.

Each world type is defined in its own file with a `create(key, config)` function.
The arena type is specified in config["arena"]["type"].
"""

import jax

from src.worlds import default, bridge, maze, thermotaxis, pretrain
from src.worlds.base import regenerate_resources

# Registry of world types -> creator functions
WORLD_TYPES = {
    "gaussian": default.create,
    "bridge": bridge.create,
    "maze": maze.create,
    "thermotaxis": thermotaxis.create,
    "pretrain": pretrain.create,
}


def create_world(key: jax.Array, config: dict) -> dict:
    """Create world based on config["arena"]["type"].

    Args:
        key: PRNG key
        config: World configuration dict

    Returns:
        World dict with: resource, resource_base, temperature, toxin

    Raises:
        ValueError: If arena type is not registered
    """
    arena_type = config.get("arena", {}).get("type", "gaussian")

    if arena_type not in WORLD_TYPES:
        available = ", ".join(WORLD_TYPES.keys())
        raise ValueError(f"Unknown arena type '{arena_type}'. Available: {available}")

    return WORLD_TYPES[arena_type](key, config)


def register_world(name: str, creator_fn):
    """Register a new world type.

    Args:
        name: Type name to use in config["arena"]["type"]
        creator_fn: Function(key, config) -> world dict
    """
    WORLD_TYPES[name] = creator_fn


# Re-export for convenience
__all__ = ["create_world", "regenerate_resources", "register_world", "WORLD_TYPES"]
