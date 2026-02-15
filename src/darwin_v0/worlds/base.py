"""Base utilities for world generation."""

import jax
import jax.numpy as jnp
from jax import random


def generate_gaussian_field(key: jax.Array, size: int, length_scale: float) -> jax.Array:
    """Generate a field with visible features using 1/f^beta power spectrum."""
    noise_key, phase_key = random.split(key)

    freqs = jnp.fft.fftfreq(size)
    fx, fy = jnp.meshgrid(freqs, freqs)
    freq_magnitude = jnp.sqrt(fx**2 + fy**2)

    # 1/f^beta spectrum with cutoffs for nice blob-like features
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

    noise_real = random.normal(noise_key, (size, size))
    noise_imag = random.normal(phase_key, (size, size))
    noise_complex = noise_real + 1j * noise_imag

    filtered = noise_complex * jnp.sqrt(power_spectrum)
    field = jnp.real(jnp.fft.ifft2(filtered))

    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field


def generate_resource_field(key: jax.Array, size: int, min_val: float, max_val: float,
                            mean_val: float, length_scale: float) -> jax.Array:
    """Generate resource distribution field."""
    field = generate_gaussian_field(key, size, length_scale)
    field = field * (max_val - min_val) + min_val
    current_mean = field.mean()
    field = field + (mean_val - current_mean)
    field = jnp.clip(field, min_val, max_val)
    return field


def generate_temperature_field(key: jax.Array, size: int, min_val: float,
                               max_val: float, length_scale: float) -> jax.Array:
    """Generate temperature distribution field."""
    field = generate_gaussian_field(key, size, length_scale)
    return field * (max_val - min_val) + min_val


def generate_toxin_field(key: jax.Array, size: int, coverage_fraction: float,
                         length_scale: float) -> jax.Array:
    """Generate binary toxin field with specified coverage fraction."""
    field = generate_gaussian_field(key, size, length_scale)
    threshold = jnp.percentile(field, (1.0 - coverage_fraction) * 100)
    return (field >= threshold).astype(jnp.float32)


def regenerate_resources(resource: jax.Array, resource_base: jax.Array,
                         timescale: float) -> jax.Array:
    """Regenerate resources toward base level."""
    rate = 1.0 - jnp.exp(-1.0 / timescale)
    return resource + (resource_base - resource) * rate
