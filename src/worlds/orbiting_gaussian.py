"""Orbiting Gaussian world - a high-resource blob orbits around the center.

Base uniform resource with a Gaussian blob that circles the arena center.
Tests whether agents can track a moving resource hotspot.
"""

import jax
import jax.numpy as jnp
from jax import random

from src.worlds.base import generate_gaussian_field


def create(key: jax.Array, config: dict) -> dict:
    """Create world with orbiting Gaussian resource blob.

    Arena config options:
        base_resource: Uniform background resource level (default 10)
        blob_max: Peak resource at blob center (default 30)
        blob_sigma: Gaussian blob width in pixels (default 48)
        orbit_radius: Distance from center to blob center (default 192)
        orbit_period: Steps for one full orbit (default 2000)
        temperature_min, temperature_max: Temperature field bounds
        temperature_length_scale: Spatial smoothness of temperature

    Returns:
        World dict with orbiting blob parameters stored for time evolution
    """
    size = config["world"]["size"]
    arena = config.get("arena", {})

    # Orbiting blob parameters
    base_resource = arena.get("base_resource", 10.0)
    blob_max = arena.get("blob_max", 30.0)
    blob_sigma = arena.get("blob_sigma", 48.0)
    orbit_radius = arena.get("orbit_radius", 192.0)
    orbit_period = arena.get("orbit_period", 2000.0)

    # Initial resource field (t=0, blob at angle=0, i.e. right of center)
    resource = _compute_resource_field(
        size, base_resource, blob_max, blob_sigma, orbit_radius, 0
    )

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
        "resource_min": base_resource,
        "resource_max": blob_max,
        # Store parameters for time evolution
        "_base_resource": base_resource,
        "_blob_max": blob_max,
        "_blob_sigma": blob_sigma,
        "_orbit_radius": orbit_radius,
        "_orbit_period": orbit_period,
    }


def _compute_resource_field(size: int, base_resource: float, blob_max: float,
                             blob_sigma: float, orbit_radius: float,
                             step: int) -> jax.Array:
    """Compute resource field at given timestep."""
    # Blob center position (orbiting around arena center)
    center = size / 2.0
    angle = 2.0 * jnp.pi * step / 2000.0  # Will be overridden by orbit_period in update
    blob_x = center + orbit_radius * jnp.cos(angle)
    blob_y = center + orbit_radius * jnp.sin(angle)

    # Create coordinate grids
    y_coords, x_coords = jnp.mgrid[0:size, 0:size]

    # Gaussian blob
    dist_sq = (x_coords - blob_x)**2 + (y_coords - blob_y)**2
    blob = (blob_max - base_resource) * jnp.exp(-dist_sq / (2 * blob_sigma**2))

    resource = base_resource + blob
    return resource.astype(jnp.float32)


def update_resource(world: dict, step: int, config: dict) -> dict:
    """Update resource field for current timestep.

    Called each simulation step to move the orbiting blob.
    """
    size = world["resource"].shape[0]
    base_resource = world["_base_resource"]
    blob_max = world["_blob_max"]
    blob_sigma = world["_blob_sigma"]
    orbit_radius = world["_orbit_radius"]
    orbit_period = world["_orbit_period"]

    # Compute blob position
    center = size / 2.0
    angle = 2.0 * jnp.pi * step / orbit_period
    blob_x = center + orbit_radius * jnp.cos(angle)
    blob_y = center + orbit_radius * jnp.sin(angle)

    # Create coordinate grids
    y_coords, x_coords = jnp.mgrid[0:size, 0:size]

    # Gaussian blob
    dist_sq = (x_coords - blob_x)**2 + (y_coords - blob_y)**2
    blob = (blob_max - base_resource) * jnp.exp(-dist_sq / (2 * blob_sigma**2))

    new_resource_base = (base_resource + blob).astype(jnp.float32)

    return {**world, "resource_base": new_resource_base}
