"""Bridge world - two fertile strips on edges connected by a narrow bridge."""

import jax
import jax.numpy as jnp
from jax import random

from src.darwin_v0.worlds.base import generate_temperature_field


def create(key: jax.Array, config: dict) -> dict:
    """Create world with two fertile strips connected by a bridge.

    Arena config options:
        strip_width: Width of fertile strips on left/right edges (default: 32)
        bridge_height: Height of connecting bridge (default: 16)
        resource_fertile: Resource level in fertile zones (default: 25.0)
        resource_bridge: Resource level on bridge (default: 10.0)
        resource_barren: Resource level elsewhere (default: 5.0)
        temperature_min, temperature_max: Temperature field bounds
        temperature_length_scale: Spatial smoothness of temperature

    Args:
        key: PRNG key
        config: World configuration dict

    Returns:
        World dict with: resource, resource_base, temperature, toxin
    """
    size = config["world"]["size"]
    arena = config.get("arena", {})

    strip_width = arena.get("strip_width", 32)
    bridge_height = arena.get("bridge_height", 16)
    resource_fertile = arena.get("resource_fertile", 25.0)
    resource_bridge = arena.get("resource_bridge", 10.0)
    resource_barren = arena.get("resource_barren", 5.0)

    # Start with barren
    resource = jnp.full((size, size), resource_barren, dtype=jnp.float32)

    # Left fertile strip (full height)
    resource = resource.at[:, :strip_width].set(resource_fertile)

    # Right fertile strip (full height)
    resource = resource.at[:, -strip_width:].set(resource_fertile)

    # Bridge in the middle (connects left and right)
    bridge_y_start = (size - bridge_height) // 2
    bridge_y_end = bridge_y_start + bridge_height
    resource = resource.at[bridge_y_start:bridge_y_end, strip_width:-strip_width].set(resource_bridge)

    # Temperature field
    temperature = generate_temperature_field(
        key, size,
        arena.get("temperature_min", 0.0),
        arena.get("temperature_max", 1.0),
        arena.get("temperature_length_scale", 400.0),
    )

    # No toxin for this arena
    toxin = jnp.zeros((size, size), dtype=jnp.float32)

    return {
        "resource": resource,
        "resource_base": resource.copy(),
        "temperature": temperature,
        "toxin": toxin,
        "resource_min": resource_barren,
        "resource_max": resource_fertile,
    }
