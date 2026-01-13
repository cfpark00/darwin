"""Temporal Gaussian world - time-varying resource field via Fourier phase rotation.

Large spatial structures evolve slowly, small features change faster (ω_k ∝ 1/|k|).
This creates a dynamic environment that agents must actively track.
"""

import jax
import jax.numpy as jnp
from jax import random

from src.worlds.base import generate_gaussian_field


def create(key: jax.Array, config: dict) -> dict:
    """Create world with time-varying Gaussian resource field.

    Arena config options:
        resource_min, resource_max: Resource field bounds
        resource_length_scale: Spatial smoothness
        base_omega: Base angular frequency for phase rotation (default 0.0002)
        temperature_min, temperature_max: Temperature field bounds
        temperature_length_scale: Spatial smoothness of temperature

    Returns:
        World dict with Fourier coefficients for time evolution
    """
    size = config["world"]["size"]
    arena = config.get("arena", {})

    resource_key, temp_key = random.split(key)

    # Resource field parameters
    resource_min = arena.get("resource_min", 0.0)
    resource_max = arena.get("resource_max", 30.0)
    resource_length_scale = arena.get("resource_length_scale", 600.0)
    base_omega = arena.get("base_omega", 0.0002)

    # Generate Fourier coefficients for resource field
    coefficients, freq_magnitude, omega = _generate_fourier_components(
        resource_key, size, resource_length_scale, base_omega
    )

    # Initial resource field (t=0)
    resource = _evolve_field(coefficients, omega, 0, resource_min, resource_max, size)

    # Static Gaussian temperature field
    temp_min = arena.get("temperature_min", 0.0)
    temp_max = arena.get("temperature_max", 1.0)
    temp_length_scale = arena.get("temperature_length_scale", 800.0)

    temp_field = generate_gaussian_field(temp_key, size, temp_length_scale)
    temperature = temp_field * (temp_max - temp_min) + temp_min

    # No toxin
    toxin = jnp.zeros((size, size), dtype=jnp.float32)

    return {
        "resource": resource,
        "resource_base": resource.copy(),  # Will be updated each step
        "temperature": temperature,
        "toxin": toxin,
        "resource_min": resource_min,
        "resource_max": resource_max,
        # Store Fourier components for time evolution
        "_fourier_coefficients": coefficients,
        "_fourier_omega": omega,
    }


def _generate_fourier_components(key: jax.Array, size: int, length_scale: float,
                                   base_omega: float):
    """Generate Fourier coefficients and frequencies for time evolution."""
    noise_key, phase_key = random.split(key)

    freqs = jnp.fft.fftfreq(size)
    fx, fy = jnp.meshgrid(freqs, freqs)
    freq_magnitude = jnp.sqrt(fx**2 + fy**2)

    # 1/f^beta spectrum with bandpass (same as base.py)
    beta = 4.0
    min_freq = 1.0 / length_scale
    max_freq = 10.0 / length_scale

    power_spectrum = jnp.where(
        freq_magnitude > 0,
        1.0 / (freq_magnitude ** beta + 1e-10),
        0.0
    )
    low_pass = jnp.exp(-0.5 * (freq_magnitude / max_freq) ** 4)
    high_pass = 1.0 - jnp.exp(-0.5 * (freq_magnitude / min_freq) ** 4)
    power_spectrum = power_spectrum * low_pass * high_pass

    # Random complex noise
    noise_real = random.normal(noise_key, (size, size))
    noise_imag = random.normal(phase_key, (size, size))
    noise_complex = noise_real + 1j * noise_imag

    # Fourier coefficients
    coefficients = noise_complex * jnp.sqrt(power_spectrum)

    # Angular frequency: ω_k ∝ 1/|k| (large structures evolve slowly)
    omega = jnp.where(
        freq_magnitude > 0,
        base_omega / (freq_magnitude + 0.01),
        0.0
    )
    # Cap maximum omega
    omega = jnp.clip(omega, 0, 2 * jnp.pi / 10)

    return coefficients, freq_magnitude, omega


def _evolve_field(coefficients: jax.Array, omega: jax.Array, t: int,
                   resource_min: float, resource_max: float, size: int) -> jax.Array:
    """Evolve field to time t by rotating Fourier phases."""
    phase_rotation = jnp.exp(1j * omega * t)
    coefficients_t = coefficients * phase_rotation
    field = jnp.real(jnp.fft.ifft2(coefficients_t))

    # Normalize to [0, 1] then scale to resource range
    field_min = field.min()
    field_max = field.max()
    field_norm = (field - field_min) / (field_max - field_min + 1e-8)
    resource = field_norm * (resource_max - resource_min) + resource_min

    return resource.astype(jnp.float32)


def update_resource(world: dict, step: int, config: dict) -> dict:
    """Update resource field for current timestep.

    Called each simulation step to evolve the resource target.
    """
    coefficients = world["_fourier_coefficients"]
    omega = world["_fourier_omega"]
    resource_min = world["resource_min"]
    resource_max = world["resource_max"]
    size = world["resource"].shape[0]

    # Compute new target field
    new_resource_base = _evolve_field(
        coefficients, omega, step, resource_min, resource_max, size
    )

    return {**world, "resource_base": new_resource_base}
