"""Linear gradient world - resource varies linearly across x-axis.

For chemotaxis experiments testing gradient-following behavior.
"""

import jax
import jax.numpy as jnp
from jax import random

from src.worlds.base import generate_gaussian_field


def create(key: jax.Array, config: dict) -> dict:
    """Create world with linear resource gradient.

    Arena config options:
        resource_left: Resource level at x=0 (default 0)
        resource_right: Resource level at x=size-1 (default 30)
        temperature_min, temperature_max: Temperature field bounds
        temperature_length_scale: Spatial smoothness of temperature

    Args:
        key: PRNG key
        config: World configuration dict

    Returns:
        World dict with linear gradient resources
    """
    size = config["world"]["size"]
    arena = config.get("arena", {})

    # Linear gradient along x-axis
    resource_left = arena.get("resource_left", 0.0)
    resource_right = arena.get("resource_right", 30.0)

    # Create gradient: columns have values interpolated from left to right
    x = jnp.linspace(resource_left, resource_right, size)
    resource = jnp.broadcast_to(x[None, :], (size, size)).astype(jnp.float32)

    # Static Gaussian temperature field
    temp_min = arena.get("temperature_min", 0.0)
    temp_max = arena.get("temperature_max", 1.0)
    temp_length_scale = arena.get("temperature_length_scale", 200.0)

    temp_field = generate_gaussian_field(key, size, temp_length_scale)
    temperature = temp_field * (temp_max - temp_min) + temp_min

    # No toxin
    toxin = jnp.zeros((size, size), dtype=jnp.float32)

    return {
        "resource": resource,
        "resource_base": resource.copy(),
        "temperature": temperature,
        "toxin": toxin,
        "resource_min": min(resource_left, resource_right),
        "resource_max": max(resource_left, resource_right),
    }
