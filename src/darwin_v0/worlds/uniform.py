"""Uniform world - constant resource level everywhere.

Simple controlled environment for behavioral experiments.
No spatial variation in resources, no toxin.
"""

import jax
import jax.numpy as jnp
from jax import random

from src.darwin_v0.worlds.base import generate_gaussian_field


def create(key: jax.Array, config: dict) -> dict:
    """Create uniform world with constant resources.

    Arena config options:
        resource_level: Constant resource level everywhere (required)
        temperature_min, temperature_max: Temperature field bounds
        temperature_length_scale: Spatial smoothness of temperature

    Args:
        key: PRNG key
        config: World configuration dict

    Returns:
        World dict with uniform resources
    """
    size = config["world"]["size"]
    arena = config.get("arena", {})

    # Uniform resource field
    resource_level = arena["resource_level"]  # Required, no default
    resource = jnp.full((size, size), resource_level, dtype=jnp.float32)

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
        "resource_min": 0.0,
        "resource_max": resource_level,
    }
