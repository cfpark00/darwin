"""Pretrain world - simple environment for basic survival training.

Uniform food that slowly decays over time (curriculum learning).
Static Gaussian temperature field.
No toxin.
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial

from src.worlds.base import generate_gaussian_field


def create(key: jax.Array, config: dict) -> dict:
    """Create pretrain world with uniform food.

    Arena config options:
        food_init: Initial food level everywhere (default: 30.0)
        food_final: Final food level after decay (default: 12.0)
        food_decay_steps: Steps to reach final food (default: 10000)
        temperature_min, temperature_max: Temperature field bounds
        temperature_length_scale: Spatial smoothness of temperature

    Args:
        key: PRNG key
        config: World configuration dict

    Returns:
        World dict with pretrain-specific params for food decay
    """
    size = config["world"]["size"]
    arena = config.get("arena", {})

    food_init = arena.get("food_init", 30.0)
    food_final = arena.get("food_final", 12.0)
    food_decay_steps = arena.get("food_decay_steps", 10000)

    # Uniform food field
    resource = jnp.full((size, size), food_init, dtype=jnp.float32)

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
        "resource_min": food_final,
        "resource_max": food_init,
        # Pretrain-specific params for food decay
        "arena_type": "pretrain",
        "food_init": food_init,
        "food_final": food_final,
        "food_decay_steps": food_decay_steps,
    }


@partial(jax.jit, static_argnums=(0,))
def compute_food_level(size: int, step: int, food_init: float,
                       food_final: float, food_decay_steps: int) -> jax.Array:
    """Compute current food level based on decay schedule.

    Linear decay from food_init to food_final over food_decay_steps.
    """
    progress = jnp.clip(step / food_decay_steps, 0.0, 1.0)
    current_level = food_init - (food_init - food_final) * progress
    return jnp.full((size, size), current_level, dtype=jnp.float32)


def update_food(world: dict, step: int) -> dict:
    """Update world food level based on decay schedule.

    Args:
        world: World dict with pretrain params
        step: Current simulation step

    Returns:
        Updated world dict with new resource_base
    """
    size = world["resource"].shape[0]
    new_base = compute_food_level(
        size, step,
        world["food_init"],
        world["food_final"],
        world["food_decay_steps"]
    )
    return {**world, "resource_base": new_base}
