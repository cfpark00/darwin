"""Thermotaxis world - soil-like environment with dynamic temperature.

Based on Ramot et al. 2008 soil temperature model:
- Food: gradient (more at top) + Gaussian noise
- Temperature: sinusoidal at surface, decays + lags with depth
- No toxin
- Reproduction success depends on temperature
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial


def _generate_gaussian_field(key: jax.Array, shape: tuple, length_scale: float) -> jax.Array:
    """Generate smooth Gaussian random field using convolution."""
    # Generate white noise
    noise = random.normal(key, shape)

    # Create Gaussian kernel
    size = int(length_scale * 4)  # 4 sigma coverage
    if size % 2 == 0:
        size += 1
    x = jnp.arange(size) - size // 2
    kernel_1d = jnp.exp(-x**2 / (2 * length_scale**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable 2D convolution
    from jax.scipy.signal import convolve

    # Convolve along each axis
    smooth = convolve(noise, kernel_1d.reshape(-1, 1), mode='same')
    smooth = convolve(smooth, kernel_1d.reshape(1, -1), mode='same')

    # Normalize to [0, 1]
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    return smooth


def create(key: jax.Array, config: dict) -> dict:
    """Create thermotaxis world with noisy food gradient.

    Arena config options:
        food_max: Maximum food at top (default: 20.0)
        food_noise_amplitude: Noise variation around gradient (default: 8.0)
        food_length_scale: Spatial smoothness of noise (default: 30.0)
        temperature_period: Steps per temperature cycle (default: 1000)
        temperature_damping_depth: Depth for amplitude decay by e (default: 64)

    Args:
        key: PRNG key
        config: World configuration dict

    Returns:
        World dict with: resource, resource_base, temperature, toxin, arena_type,
                        and thermotaxis-specific params for temperature updates
    """
    size = config["world"]["size"]
    height = size
    width = size
    arena = config.get("arena", {})

    food_max = arena.get("food_max", 20.0)
    food_noise_amplitude = arena.get("food_noise_amplitude", 8.0)
    food_length_scale = arena.get("food_length_scale", 30.0)
    temperature_period = arena.get("temperature_period", 1000)
    temperature_damping_depth = arena.get("temperature_damping_depth", 64.0)

    # Food: gradient + noise
    y_coords = jnp.arange(height).reshape(-1, 1)  # (height, 1)
    food_gradient = food_max * y_coords / (height - 1)  # 0 at bottom, max at top
    food_gradient = jnp.broadcast_to(food_gradient, (height, width))

    food_noise = _generate_gaussian_field(key, (height, width), food_length_scale)
    resource = jnp.clip(
        food_gradient + food_noise_amplitude * (food_noise - 0.5),
        0.0, None
    )

    # Temperature at step 0
    temperature = compute_temperature(height, width, 0, temperature_period, temperature_damping_depth)

    # No toxin
    toxin = jnp.zeros((height, width), dtype=jnp.float32)

    return {
        "resource": resource,
        "resource_base": resource.copy(),
        "temperature": temperature,
        "toxin": toxin,
        "resource_min": 0.0,
        "resource_max": float(resource.max()),
        # Thermotaxis-specific params for dynamic updates
        "arena_type": "thermotaxis",
        "temperature_period": temperature_period,
        "temperature_damping_depth": temperature_damping_depth,
    }


@partial(jax.jit, static_argnums=(0, 1, 3, 4))
def compute_temperature(height: int, width: int, step: int,
                        period: int, damping_depth: float) -> jax.Array:
    """Compute temperature field for given simulation step.

    Modified soil temperature model (inspired by Ramot et al. 2008):
    - Mean temperature gradient: 0.5 at surface, 0.25 at bottom
    - Oscillation amplitude decays with depth
    - Phase lags with depth

    T(y,t) = mean(y) + 0.5 * exp(-z/zd) * sin(2πt/p - z/zd)
    where mean(y) = 0.25 + 0.25 * (y / max_y)

    Args:
        height: Grid height
        width: Grid width
        step: Current simulation step
        period: Temperature oscillation period
        damping_depth: Depth at which amplitude decays by e

    Returns:
        Temperature field (height, width) with values in [0, 1]
    """
    max_y = height - 1
    y_coords = jnp.arange(height).reshape(-1, 1)  # (height, 1)
    depth = max_y - y_coords  # depth from surface (0 at top, max at bottom)

    # Mean temperature: 0.5 at surface, 0.25 at bottom
    mean_temp = 0.25 + 0.25 * (y_coords / max_y)

    # Amplitude decay and phase lag both scale with depth/zd
    decay = jnp.exp(-depth / damping_depth)
    phase_lag = depth / damping_depth

    # Oscillation around the mean
    temp = mean_temp + 0.5 * decay * jnp.sin(2 * jnp.pi * step / period - phase_lag)

    # Broadcast to full grid (constant in x)
    return jnp.broadcast_to(temp, (height, width))


def update_temperature(world: dict, step: int) -> dict:
    """Update world temperature field for current step.

    Args:
        world: World dict with thermotaxis params
        step: Current simulation step

    Returns:
        Updated world dict with new temperature field
    """
    height, width = world["temperature"].shape
    period = world["temperature_period"]
    damping_depth = world["temperature_damping_depth"]

    new_temp = compute_temperature(height, width, step, period, damping_depth)
    return {**world, "temperature": new_temp}


def compute_reproduction_success(temperature: jax.Array,
                                  threshold: float = 0.5,
                                  max_temp: float = 1.0) -> jax.Array:
    """Compute reproduction success probability based on temperature.

    - 1.0 when temp <= threshold
    - Linear decay from 1.0 to 0.0 as temp goes from threshold to max_temp

    Args:
        temperature: Temperature values
        threshold: Temperature below which reproduction always succeeds
        max_temp: Temperature at which reproduction probability is 0

    Returns:
        Success probability in [0, 1]
    """
    return jnp.clip(1.0 - (temperature - threshold) / (max_temp - threshold), 0.0, 1.0)
