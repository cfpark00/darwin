"""Default world - gaussian random fields."""

import jax
from jax import random

from src.darwin_v0.worlds.base import (
    generate_resource_field,
    generate_temperature_field,
    generate_toxin_field,
)


def create(key: jax.Array, config: dict) -> dict:
    """Create gaussian world with random resource/temperature/toxin fields.

    Arena config options:
        resource_min, resource_max, resource_mean: Resource field bounds
        resource_length_scale: Spatial smoothness of resource blobs
        temperature_min, temperature_max: Temperature field bounds
        temperature_length_scale: Spatial smoothness of temperature
        toxin_coverage: Fraction of world covered by toxin
        toxin_length_scale: Spatial smoothness of toxin blobs

    Args:
        key: PRNG key
        config: World configuration dict

    Returns:
        World dict with: resource, resource_base, temperature, toxin
    """
    size = config["world"]["size"]
    arena = config["arena"]
    keys = random.split(key, 3)

    resource = generate_resource_field(
        keys[0], size,
        arena["resource_min"],
        arena["resource_max"],
        arena["resource_mean"],
        arena["resource_length_scale"],
    )

    temperature = generate_temperature_field(
        keys[1], size,
        arena["temperature_min"],
        arena["temperature_max"],
        arena["temperature_length_scale"],
    )

    toxin = generate_toxin_field(
        keys[2], size,
        arena["toxin_coverage"],
        arena["toxin_length_scale"],
    )

    return {
        "resource": resource,
        "resource_base": resource.copy(),
        "temperature": temperature,
        "toxin": toxin,
        "resource_min": arena["resource_min"],
        "resource_max": arena["resource_max"],
    }
