"""Environment abstraction for evolvability_v1.

Environments define:
1. What observations agents receive (input_dim)
2. What actions are valid (output_dim)
3. How the world updates each step
4. How to compute observations from world state
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax import random

from src.evolvability_v1.types import AgentConfig, PhysicsConfig


class Environment(ABC):
    """Base class for all environments.

    Environments define:
    - Observations (input_dim, compute_observation)
    - Actions (output_dim, action_names)
    - World state (create_world, update_world)
    - Capabilities (has_toxin, has_attack) - what physics features are enabled

    The simulation reads capability flags and enables corresponding physics.
    """

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Number of observation values provided to agents."""
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Number of actions agents can take."""
        ...

    @property
    def action_names(self) -> list[str]:
        """Names of actions for logging."""
        return [f"action_{i}" for i in range(self.output_dim)]

    # -------------------------------------------------------------------------
    # Capability flags - declare what physics features this environment uses
    # -------------------------------------------------------------------------

    @property
    def has_toxin(self) -> bool:
        """Whether environment has toxin fields.

        If True:
        - create_world() MUST return a 'toxin' field (bool array)
        - Simulation will kill agents that step on toxin
        - Observation should include toxin sensing
        """
        return False

    @property
    def has_attack(self) -> bool:
        """Whether agents can attack each other.

        If True:
        - output_dim should include an attack action
        - Simulation will process attacks (kill agent in front, energy drops)
        """
        return False

    # -------------------------------------------------------------------------
    # World creation and updates
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create initial world state.

        Must return dict with at minimum:
        - 'resource': (size, size) float32 - food levels
        - 'resource_base': (size, size) float32 - base food for regeneration
        - 'temperature': (size, size) float32 - temperature field

        If has_toxin is True, must also include:
        - 'toxin': (size, size) bool - toxin locations
        """
        ...

    @abstractmethod
    def compute_observation(
        self,
        world: dict,
        position: jax.Array,
        orientation: int,
        occupancy_grid: jax.Array,
        agent_idx: int,
        key: jax.Array,
    ) -> jax.Array:
        """Compute observation for a single agent."""
        ...

    def update_world(self, world: dict, step: int) -> dict:
        """Update world state (override for dynamic environments)."""
        return world

    def get_action_mask(self) -> Optional[jax.Array]:
        """Return action mask (None = all actions enabled)."""
        return None

    def get_reproduction_modifier(
        self, world: dict, position: jax.Array
    ) -> float:
        """Return reproduction probability modifier (1.0 = always succeed)."""
        return 1.0


# -----------------------------------------------------------------------------
# Simple Environment (6 inputs, 6 outputs) - like darwin_v0 simple
# -----------------------------------------------------------------------------

class SimpleEnvironment(Environment):
    """Environment with food + temperature sensing, no toxin/attack.

    Observations: [food, temperature, contact_front, contact_back, contact_left, contact_right]
    Actions: [eat, forward, left, right, stay, reproduce]
    """

    def __init__(
        self,
        size: int = 256,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
    ):
        self.size = size
        self.food_noise_std = food_noise_std
        self.temp_noise_std = temp_noise_std

    @property
    def input_dim(self) -> int:
        return 6

    @property
    def output_dim(self) -> int:
        return 6

    @property
    def action_names(self) -> list[str]:
        return ["eat", "forward", "left", "right", "stay", "reproduce"]

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with uniform resources and temperature."""
        return {
            "resource": jnp.full((size, size), 15.0),
            "resource_base": jnp.full((size, size), 15.0),
            "temperature": jnp.full((size, size), 0.5),
        }

    def compute_observation(
        self,
        world: dict,
        position: jax.Array,
        orientation: int,
        occupancy_grid: jax.Array,
        agent_idx: int,
        key: jax.Array,
    ) -> jax.Array:
        """Compute 6-dim observation."""
        y, x = position[0], position[1]
        size = self.size
        k1, k2 = random.split(key)

        # Food and temperature with noise (clamped to physical range)
        food = world["resource"][y, x]
        temp = world["temperature"][y, x]
        food_noisy = jnp.maximum(0.0, food + random.normal(k1) * self.food_noise_std)
        temp_noisy = jnp.clip(temp + random.normal(k2) * self.temp_noise_std, 0.0, 1.0)

        # Contact sensors
        deltas = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)

        def check_contact(direction: int) -> float:
            delta = deltas[direction]
            ny, nx = y + delta[0], x + delta[1]
            in_bounds = (ny >= 0) & (ny < size) & (nx >= 0) & (nx < size)
            occupant = jnp.where(in_bounds, occupancy_grid[ny % size, nx % size], 0)
            has_other = (occupant > 0) & (occupant != agent_idx + 1)
            return has_other.astype(jnp.float32)

        contact_front = check_contact(orientation)
        contact_back = check_contact((orientation + 2) % 4)
        contact_left = check_contact((orientation + 3) % 4)
        contact_right = check_contact((orientation + 1) % 4)

        return jnp.array([
            food_noisy, temp_noisy,
            contact_front, contact_back, contact_left, contact_right
        ])


# -----------------------------------------------------------------------------
# Full Environment (7 inputs, 7 outputs) - like darwin_v0 full
# -----------------------------------------------------------------------------

class FullEnvironment(Environment):
    """Environment with toxin sensing and attack action.

    Observations: [food, temperature, toxin, contact_front, contact_back, contact_left, contact_right]
    Actions: [eat, forward, left, right, stay, reproduce, attack]

    Capabilities:
    - has_toxin=True: agents sense toxin and die when stepping on it
    - has_attack=True: agents can attack and kill the agent in front of them
    """

    def __init__(
        self,
        size: int = 512,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
        toxin_noise_std: float = 0.1,
        toxin_detection_radius: int = 5,
    ):
        self.size = size
        self.food_noise_std = food_noise_std
        self.temp_noise_std = temp_noise_std
        self.toxin_noise_std = toxin_noise_std
        self.toxin_detection_radius = toxin_detection_radius
        # Pre-compute circular kernel offsets for toxin detection
        self._toxin_kernel_offsets = self._make_circular_kernel_offsets(toxin_detection_radius)

    def _make_circular_kernel_offsets(self, radius: int) -> jax.Array:
        """Create array of (dy, dx) offsets within circular radius."""
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    offsets.append((dy, dx))
        return jnp.array(offsets, dtype=jnp.int32)

    @property
    def input_dim(self) -> int:
        return 7

    @property
    def output_dim(self) -> int:
        return 7

    @property
    def action_names(self) -> list[str]:
        return ["eat", "forward", "left", "right", "stay", "reproduce", "attack"]

    @property
    def has_toxin(self) -> bool:
        return True

    @property
    def has_attack(self) -> bool:
        return True

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with resources, temperature, and toxin."""
        k1, k2 = random.split(key)
        return {
            "resource": jnp.full((size, size), 15.0),
            "resource_base": jnp.full((size, size), 15.0),
            "temperature": jnp.full((size, size), 0.5),
            "toxin": jnp.zeros((size, size), dtype=bool),
        }

    def compute_observation(
        self,
        world: dict,
        position: jax.Array,
        orientation: int,
        occupancy_grid: jax.Array,
        agent_idx: int,
        key: jax.Array,
    ) -> jax.Array:
        """Compute 7-dim observation (includes toxin)."""
        y, x = position[0], position[1]
        size = self.size
        k1, k2, k3 = random.split(key, 3)

        # Food, temperature, toxin with noise
        food = world["resource"][y, x]
        temp = world["temperature"][y, x]
        toxin = self._compute_toxin_nearby(world["toxin"], y, x)

        # Clamp to physical ranges
        food_noisy = jnp.maximum(0.0, food + random.normal(k1) * self.food_noise_std)
        temp_noisy = jnp.clip(temp + random.normal(k2) * self.temp_noise_std, 0.0, 1.0)
        toxin_noisy = jnp.clip(toxin + random.normal(k3) * self.toxin_noise_std, 0.0, 1.0)

        # Contact sensors
        deltas = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)

        def check_contact(direction: int) -> float:
            delta = deltas[direction]
            ny, nx = y + delta[0], x + delta[1]
            in_bounds = (ny >= 0) & (ny < size) & (nx >= 0) & (nx < size)
            occupant = jnp.where(in_bounds, occupancy_grid[ny % size, nx % size], 0)
            has_other = (occupant > 0) & (occupant != agent_idx + 1)
            return has_other.astype(jnp.float32)

        contact_front = check_contact(orientation)
        contact_back = check_contact((orientation + 2) % 4)
        contact_left = check_contact((orientation + 3) % 4)
        contact_right = check_contact((orientation + 1) % 4)

        return jnp.array([
            food_noisy, temp_noisy, toxin_noisy,
            contact_front, contact_back, contact_left, contact_right
        ])

    def _compute_toxin_nearby(self, toxin_grid: jax.Array, y: int, x: int) -> float:
        """Compute toxin presence within circular detection radius.

        Returns 1.0 if any toxin is within radius, 0.0 otherwise.
        Matches darwin_v0 behavior.
        """
        size = self.size
        offsets = self._toxin_kernel_offsets

        def check_offset(offset):
            dy, dx = offset[0], offset[1]
            ny, nx = (y + dy) % size, (x + dx) % size
            return toxin_grid[ny, nx].astype(jnp.float32)

        # Check all positions within radius - any toxin triggers detection
        toxin_present = jax.vmap(check_offset)(offsets)
        return jnp.any(toxin_present > 0).astype(jnp.float32)


# -----------------------------------------------------------------------------
# Gaussian Environment - random resource/temperature fields
# -----------------------------------------------------------------------------

class GaussianEnvironment(SimpleEnvironment):
    """Environment with Gaussian random resource and temperature fields."""

    def __init__(
        self,
        size: int = 256,
        resource_mean: float = 15.0,
        resource_max: float = 30.0,
        resource_length_scale: float = 30.0,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
    ):
        super().__init__(size, food_noise_std, temp_noise_std)
        self.resource_mean = resource_mean
        self.resource_max = resource_max
        self.resource_length_scale = resource_length_scale

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with Gaussian random fields."""
        k1, k2 = random.split(key)

        # Generate Gaussian random field for resources
        resource = self._generate_gaussian_field(
            k1, size, self.resource_length_scale
        )
        # Scale to [0, max] with target mean
        resource = (resource + 1) / 2 * self.resource_max
        resource = jnp.clip(resource, 0, self.resource_max)

        # Temperature field
        temperature = self._generate_gaussian_field(k2, size, 50.0)
        temperature = (temperature + 1) / 2  # Scale to [0, 1]

        return {
            "resource": resource,
            "resource_base": resource.copy(),
            "temperature": temperature,
        }

    def _generate_gaussian_field(
        self, key: jax.Array, size: int, length_scale: float
    ) -> jax.Array:
        """Generate smooth random field using Gaussian kernel in Fourier space."""
        # Generate white noise
        noise = random.normal(key, (size, size))

        # Create frequency grid
        freq = jnp.fft.fftfreq(size)
        fx, fy = jnp.meshgrid(freq, freq)
        freq_sq = fx**2 + fy**2

        # Gaussian kernel in frequency space
        kernel = jnp.exp(-freq_sq * (2 * jnp.pi * length_scale)**2 / 2)

        # Apply kernel
        noise_fft = jnp.fft.fft2(noise)
        smooth_fft = noise_fft * kernel
        smooth = jnp.real(jnp.fft.ifft2(smooth_fft))

        # Normalize to [-1, 1]
        smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-8)
        return jnp.clip(smooth, -3, 3) / 3


# -----------------------------------------------------------------------------
# Uniform Environment - constant resources everywhere
# -----------------------------------------------------------------------------

class UniformEnvironment(SimpleEnvironment):
    """Environment with uniform resource level (for controlled experiments)."""

    def __init__(
        self,
        size: int = 256,
        resource_level: float = 15.0,
        temperature: float = 0.5,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
    ):
        super().__init__(size, food_noise_std, temp_noise_std)
        self.resource_level = resource_level
        self.temperature_value = temperature

    def create_world(self, key: jax.Array, size: int) -> dict:
        return {
            "resource": jnp.full((size, size), self.resource_level),
            "resource_base": jnp.full((size, size), self.resource_level),
            "temperature": jnp.full((size, size), self.temperature_value),
        }


# -----------------------------------------------------------------------------
# Cycling Environment - alternates between two configurations
# -----------------------------------------------------------------------------

class CyclingEnvironment(Environment):
    """Meta-environment that cycles between two environments.

    Useful for evolvability experiments where agents must adapt to
    alternating selective pressures.
    """

    def __init__(
        self,
        env_a: Environment,
        env_b: Environment,
        period: int = 1000,
    ):
        # Validate that environments are compatible
        if env_a.input_dim != env_b.input_dim:
            raise ValueError(
                f"Environment input_dim mismatch: {env_a.input_dim} vs {env_b.input_dim}"
            )
        if env_a.output_dim != env_b.output_dim:
            raise ValueError(
                f"Environment output_dim mismatch: {env_a.output_dim} vs {env_b.output_dim}"
            )

        self.env_a = env_a
        self.env_b = env_b
        self.period = period
        self._current_phase = 0

    @property
    def input_dim(self) -> int:
        return self.env_a.input_dim

    @property
    def output_dim(self) -> int:
        return self.env_a.output_dim

    @property
    def action_names(self) -> list[str]:
        return self.env_a.action_names

    def get_phase(self, step: int) -> int:
        """Return current phase (0 = env_a, 1 = env_b)."""
        return (step // self.period) % 2

    def get_current_env(self, step: int) -> Environment:
        """Return the currently active environment."""
        return self.env_a if self.get_phase(step) == 0 else self.env_b

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world from env_a (will be updated as phases change)."""
        k1, k2 = random.split(key)
        world_a = self.env_a.create_world(k1, size)
        world_b = self.env_b.create_world(k2, size)
        return {
            **world_a,
            "_world_a": world_a,
            "_world_b": world_b,
            "_phase": 0,
        }

    def update_world(self, world: dict, step: int) -> dict:
        """Switch between world configurations based on phase."""
        new_phase = self.get_phase(step)
        old_phase = world.get("_phase", 0)

        if new_phase != old_phase:
            # Phase changed - switch world configuration
            if new_phase == 0:
                source = world["_world_a"]
            else:
                source = world["_world_b"]

            world = {
                **world,
                "resource": source["resource"],
                "resource_base": source["resource_base"],
                "temperature": source["temperature"],
                "_phase": new_phase,
            }

        # Also call the current environment's update
        current_env = self.get_current_env(step)
        return current_env.update_world(world, step)

    def compute_observation(
        self,
        world: dict,
        position: jax.Array,
        orientation: int,
        occupancy_grid: jax.Array,
        agent_idx: int,
        key: jax.Array,
    ) -> jax.Array:
        """Delegate to current environment."""
        phase = world.get("_phase", 0)
        env = self.env_a if phase == 0 else self.env_b
        return env.compute_observation(
            world, position, orientation, occupancy_grid, agent_idx, key
        )


# -----------------------------------------------------------------------------
# Bridge Environment - two fertile strips connected by narrow bridge
# -----------------------------------------------------------------------------

class BridgeEnvironment(SimpleEnvironment):
    """Environment with two fertile strips on edges connected by a narrow bridge.

    Tests exploration and migration between resource patches.

    Geometry:
    - Left strip (full height): high resources
    - Right strip (full height): high resources
    - Narrow bridge in middle: medium resources
    - Rest: low resources
    """

    def __init__(
        self,
        size: int = 256,
        strip_width: int = 32,
        bridge_height: int = 16,
        resource_fertile: float = 25.0,
        resource_bridge: float = 10.0,
        resource_barren: float = 5.0,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
    ):
        super().__init__(size, food_noise_std, temp_noise_std)
        self.strip_width = strip_width
        self.bridge_height = bridge_height
        self.resource_fertile = resource_fertile
        self.resource_bridge = resource_bridge
        self.resource_barren = resource_barren

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with bridge geometry."""
        # Start with barren
        resource = jnp.full((size, size), self.resource_barren, dtype=jnp.float32)

        # Left fertile strip (full height)
        resource = resource.at[:, :self.strip_width].set(self.resource_fertile)

        # Right fertile strip (full height)
        resource = resource.at[:, -self.strip_width:].set(self.resource_fertile)

        # Bridge in the middle (connects left and right)
        bridge_y_start = (size - self.bridge_height) // 2
        bridge_y_end = bridge_y_start + self.bridge_height
        resource = resource.at[bridge_y_start:bridge_y_end, self.strip_width:-self.strip_width].set(
            self.resource_bridge
        )

        # Temperature field (use Gaussian random field)
        temperature = self._generate_temperature_field(key, size)

        return {
            "resource": resource,
            "resource_base": resource.copy(),
            "temperature": temperature,
        }

    def _generate_temperature_field(self, key: jax.Array, size: int) -> jax.Array:
        """Generate smooth random temperature field."""
        noise = random.normal(key, (size, size))
        freq = jnp.fft.fftfreq(size)
        fx, fy = jnp.meshgrid(freq, freq)
        freq_sq = fx**2 + fy**2
        # Smooth with large length scale
        kernel = jnp.exp(-freq_sq * (2 * jnp.pi * 50.0)**2 / 2)
        noise_fft = jnp.fft.fft2(noise)
        smooth = jnp.real(jnp.fft.ifft2(noise_fft * kernel))
        # Normalize to [0, 1]
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        return smooth.astype(jnp.float32)


# -----------------------------------------------------------------------------
# Temporal Gaussian Environment - time-varying resource via Fourier rotation
# -----------------------------------------------------------------------------

class TemporalGaussianEnvironment(SimpleEnvironment):
    """Environment with time-varying resource field.

    Large spatial structures evolve slowly, small features change faster.
    Uses Fourier phase rotation: ω_k ∝ 1/|k|.

    Tests whether agents can track changing resource distributions.
    """

    def __init__(
        self,
        size: int = 256,
        resource_min: float = 0.0,
        resource_max: float = 30.0,
        resource_length_scale: float = 60.0,
        base_omega: float = 0.001,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
    ):
        super().__init__(size, food_noise_std, temp_noise_std)
        self.resource_min = resource_min
        self.resource_max = resource_max
        self.resource_length_scale = resource_length_scale
        self.base_omega = base_omega

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with Fourier components for time evolution."""
        k1, k2 = random.split(key)

        # Generate Fourier components
        coefficients, omega = self._generate_fourier_components(k1, size)

        # Initial resource field (t=0)
        resource = self._evolve_field(coefficients, omega, 0, size)

        # Static temperature field
        temperature = self._generate_temperature_field(k2, size)

        return {
            "resource": resource,
            "resource_base": resource.copy(),
            "temperature": temperature,
            # Store Fourier components for time evolution
            "_fourier_coefficients": coefficients,
            "_fourier_omega": omega,
        }

    def _generate_fourier_components(self, key: jax.Array, size: int):
        """Generate Fourier coefficients and angular frequencies."""
        k1, k2 = random.split(key)

        freqs = jnp.fft.fftfreq(size)
        fx, fy = jnp.meshgrid(freqs, freqs)
        freq_magnitude = jnp.sqrt(fx**2 + fy**2)

        # 1/f spectrum with bandpass
        beta = 4.0
        min_freq = 1.0 / self.resource_length_scale
        max_freq = 10.0 / self.resource_length_scale

        power_spectrum = jnp.where(
            freq_magnitude > 0,
            1.0 / (freq_magnitude ** beta + 1e-10),
            0.0
        )
        low_pass = jnp.exp(-0.5 * (freq_magnitude / max_freq) ** 4)
        high_pass = 1.0 - jnp.exp(-0.5 * (freq_magnitude / min_freq) ** 4)
        power_spectrum = power_spectrum * low_pass * high_pass

        # Random complex noise
        noise_real = random.normal(k1, (size, size))
        noise_imag = random.normal(k2, (size, size))
        coefficients = (noise_real + 1j * noise_imag) * jnp.sqrt(power_spectrum)

        # Angular frequency: ω_k ∝ 1/|k| (large structures evolve slowly)
        omega = jnp.where(
            freq_magnitude > 0,
            self.base_omega / (freq_magnitude + 0.01),
            0.0
        )
        omega = jnp.clip(omega, 0, 2 * jnp.pi / 10)

        return coefficients, omega

    def _evolve_field(self, coefficients: jax.Array, omega: jax.Array,
                      step: int, size: int) -> jax.Array:
        """Evolve field to time t by rotating Fourier phases."""
        phase_rotation = jnp.exp(1j * omega * step)
        coefficients_t = coefficients * phase_rotation
        field = jnp.real(jnp.fft.ifft2(coefficients_t))

        # Normalize to resource range
        field_min = field.min()
        field_max = field.max()
        field_norm = (field - field_min) / (field_max - field_min + 1e-8)
        resource = field_norm * (self.resource_max - self.resource_min) + self.resource_min

        return resource.astype(jnp.float32)

    def _generate_temperature_field(self, key: jax.Array, size: int) -> jax.Array:
        """Generate smooth random temperature field."""
        noise = random.normal(key, (size, size))
        freq = jnp.fft.fftfreq(size)
        fx, fy = jnp.meshgrid(freq, freq)
        freq_sq = fx**2 + fy**2
        kernel = jnp.exp(-freq_sq * (2 * jnp.pi * 80.0)**2 / 2)
        noise_fft = jnp.fft.fft2(noise)
        smooth = jnp.real(jnp.fft.ifft2(noise_fft * kernel))
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        return smooth.astype(jnp.float32)

    def update_world(self, world: dict, step: int) -> dict:
        """Update resource field for current timestep."""
        coefficients = world["_fourier_coefficients"]
        omega = world["_fourier_omega"]
        size = world["resource"].shape[0]

        new_resource_base = self._evolve_field(coefficients, omega, step, size)

        return {**world, "resource_base": new_resource_base}


# -----------------------------------------------------------------------------
# Orbiting Gaussian Environment - resource blob orbits around center
# -----------------------------------------------------------------------------

class OrbitingGaussianEnvironment(SimpleEnvironment):
    """Environment with a resource blob that orbits the arena center.

    Tests whether agents can track a moving resource hotspot.
    """

    def __init__(
        self,
        size: int = 256,
        base_resource: float = 10.0,
        blob_max: float = 30.0,
        blob_sigma: float = 24.0,
        orbit_radius: float = 80.0,
        orbit_period: float = 1000.0,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
    ):
        super().__init__(size, food_noise_std, temp_noise_std)
        self.base_resource = base_resource
        self.blob_max = blob_max
        self.blob_sigma = blob_sigma
        self.orbit_radius = orbit_radius
        self.orbit_period = orbit_period

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with orbiting blob."""
        # Initial resource field (t=0, blob at angle=0)
        resource = self._compute_resource_field(size, 0)

        # Static temperature field
        temperature = self._generate_temperature_field(key, size)

        return {
            "resource": resource,
            "resource_base": resource.copy(),
            "temperature": temperature,
        }

    def _compute_resource_field(self, size: int, step: int) -> jax.Array:
        """Compute resource field at given timestep."""
        center = size / 2.0
        angle = 2.0 * jnp.pi * step / self.orbit_period
        blob_x = center + self.orbit_radius * jnp.cos(angle)
        blob_y = center + self.orbit_radius * jnp.sin(angle)

        # Create coordinate grids
        y_coords, x_coords = jnp.mgrid[0:size, 0:size]

        # Gaussian blob
        dist_sq = (x_coords - blob_x)**2 + (y_coords - blob_y)**2
        blob = (self.blob_max - self.base_resource) * jnp.exp(
            -dist_sq / (2 * self.blob_sigma**2)
        )

        resource = self.base_resource + blob
        return resource.astype(jnp.float32)

    def _generate_temperature_field(self, key: jax.Array, size: int) -> jax.Array:
        """Generate smooth random temperature field."""
        noise = random.normal(key, (size, size))
        freq = jnp.fft.fftfreq(size)
        fx, fy = jnp.meshgrid(freq, freq)
        freq_sq = fx**2 + fy**2
        kernel = jnp.exp(-freq_sq * (2 * jnp.pi * 50.0)**2 / 2)
        noise_fft = jnp.fft.fft2(noise)
        smooth = jnp.real(jnp.fft.ifft2(noise_fft * kernel))
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        return smooth.astype(jnp.float32)

    def update_world(self, world: dict, step: int) -> dict:
        """Update resource field for current timestep."""
        size = world["resource"].shape[0]
        new_resource_base = self._compute_resource_field(size, step)
        return {**world, "resource_base": new_resource_base}


# -----------------------------------------------------------------------------
# Toxin Pattern Environment - FullEnvironment with configurable toxin patterns
# -----------------------------------------------------------------------------

class ToxinPatternEnvironment(FullEnvironment):
    """Full environment with configurable toxin patterns.

    Supports several toxin patterns:
    - 'ring': Toxin ring around center
    - 'stripes': Vertical toxin stripes
    - 'maze': Simple maze-like toxin walls
    - 'random': Random toxin patches
    """

    def __init__(
        self,
        size: int = 256,
        pattern: str = "ring",
        toxin_density: float = 0.1,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
        toxin_noise_std: float = 0.1,
        toxin_detection_radius: int = 5,
    ):
        super().__init__(size, food_noise_std, temp_noise_std, toxin_noise_std, toxin_detection_radius)
        self.pattern = pattern
        self.toxin_density = toxin_density

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with toxin pattern."""
        k1, k2 = random.split(key)

        # Base resource and temperature
        resource = jnp.full((size, size), 15.0)
        temperature = self._generate_temperature_field(k1, size)

        # Create toxin pattern
        toxin = self._create_toxin_pattern(k2, size)

        return {
            "resource": resource,
            "resource_base": resource.copy(),
            "temperature": temperature,
            "toxin": toxin,
        }

    def _generate_temperature_field(self, key: jax.Array, size: int) -> jax.Array:
        """Generate smooth random temperature field."""
        noise = random.normal(key, (size, size))
        freq = jnp.fft.fftfreq(size)
        fx, fy = jnp.meshgrid(freq, freq)
        freq_sq = fx**2 + fy**2
        kernel = jnp.exp(-freq_sq * (2 * jnp.pi * 50.0)**2 / 2)
        noise_fft = jnp.fft.fft2(noise)
        smooth = jnp.real(jnp.fft.ifft2(noise_fft * kernel))
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        return smooth.astype(jnp.float32)

    def _create_toxin_pattern(self, key: jax.Array, size: int) -> jax.Array:
        """Create toxin pattern based on self.pattern."""
        if self.pattern == "ring":
            return self._create_ring_pattern(size)
        elif self.pattern == "stripes":
            return self._create_stripes_pattern(size)
        elif self.pattern == "maze":
            return self._create_maze_pattern(size)
        elif self.pattern == "random":
            return self._create_random_pattern(key, size)
        else:
            # Default: no toxin
            return jnp.zeros((size, size), dtype=bool)

    def _create_ring_pattern(self, size: int) -> jax.Array:
        """Create toxin ring around center."""
        center = size / 2.0
        y_coords, x_coords = jnp.mgrid[0:size, 0:size]
        dist = jnp.sqrt((x_coords - center)**2 + (y_coords - center)**2)

        # Ring between 40% and 50% of half-size
        inner_radius = size * 0.2
        outer_radius = size * 0.25
        toxin = (dist >= inner_radius) & (dist <= outer_radius)
        return toxin

    def _create_stripes_pattern(self, size: int) -> jax.Array:
        """Create vertical toxin stripes."""
        x_coords = jnp.arange(size)
        stripe_width = max(4, size // 16)
        stripe_spacing = max(16, size // 4)
        # Stripes at regular intervals
        toxin_1d = ((x_coords % stripe_spacing) < stripe_width)
        toxin = jnp.broadcast_to(toxin_1d, (size, size))
        return toxin

    def _create_maze_pattern(self, size: int) -> jax.Array:
        """Create simple maze-like toxin walls."""
        toxin = jnp.zeros((size, size), dtype=bool)

        # Horizontal walls with gaps
        wall_spacing = size // 4
        gap_width = size // 8
        wall_thickness = 2

        for i in range(1, 4):
            y = i * wall_spacing
            if y + wall_thickness < size:
                # Wall with gap
                gap_start = (i % 2) * (size // 2) + size // 4 - gap_width // 2
                gap_end = gap_start + gap_width
                wall = jnp.ones(size, dtype=bool)
                wall = wall.at[gap_start:gap_end].set(False)
                for dy in range(wall_thickness):
                    if y + dy < size:
                        toxin = toxin.at[y + dy, :].set(
                            toxin[y + dy, :] | wall
                        )

        return toxin

    def _create_random_pattern(self, key: jax.Array, size: int) -> jax.Array:
        """Create random toxin patches."""
        noise = random.uniform(key, (size, size))
        toxin = noise < self.toxin_density
        return toxin


# -----------------------------------------------------------------------------
# Pretrain Environment - uniform food with decay (curriculum learning)
# -----------------------------------------------------------------------------

class PretrainEnvironment(SimpleEnvironment):
    """Simple environment for basic survival training.

    Uniform food that decays over time (curriculum learning):
    - Starts high so initial population doesn't die immediately
    - Decays linearly to lower level so agents must learn to move and eat

    No toxin, no attack - just basic survival.
    """

    def __init__(
        self,
        size: int = 256,
        food_init: float = 30.0,
        food_final: float = 12.0,
        food_decay_steps: int = 10000,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
    ):
        super().__init__(size, food_noise_std, temp_noise_std)
        self.food_init = food_init
        self.food_final = food_final
        self.food_decay_steps = food_decay_steps

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with uniform food (will decay via update_world)."""
        # Uniform food field at initial level
        resource = jnp.full((size, size), self.food_init, dtype=jnp.float32)

        # Static Gaussian temperature field
        temperature = self._generate_temperature_field(key, size)

        return {
            "resource": resource,
            "resource_base": resource.copy(),
            "temperature": temperature,
            # Store decay params for update_world
            "_food_init": self.food_init,
            "_food_final": self.food_final,
            "_food_decay_steps": self.food_decay_steps,
        }

    def _generate_temperature_field(self, key: jax.Array, size: int) -> jax.Array:
        """Generate smooth random temperature field."""
        noise = random.normal(key, (size, size))
        freq = jnp.fft.fftfreq(size)
        fx, fy = jnp.meshgrid(freq, freq)
        freq_sq = fx**2 + fy**2
        # Large length scale for smooth temperature
        kernel = jnp.exp(-freq_sq * (2 * jnp.pi * 200.0)**2 / 2)
        noise_fft = jnp.fft.fft2(noise)
        smooth = jnp.real(jnp.fft.ifft2(noise_fft * kernel))
        # Normalize to [0, 1]
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        return smooth.astype(jnp.float32)

    def update_world(self, world: dict, step: int) -> dict:
        """Update food level based on decay schedule."""
        food_init = world["_food_init"]
        food_final = world["_food_final"]
        food_decay_steps = world["_food_decay_steps"]
        size = world["resource"].shape[0]

        # Linear decay from food_init to food_final
        progress = jnp.clip(step / food_decay_steps, 0.0, 1.0)
        current_level = food_init - (food_init - food_final) * progress
        new_base = jnp.full((size, size), current_level, dtype=jnp.float32)

        return {**world, "resource_base": new_base}


# -----------------------------------------------------------------------------
# Gaussian Toxin Environment - Gaussian fields for everything (like darwin_v0 default)
# -----------------------------------------------------------------------------

class GaussianToxinEnvironment(FullEnvironment):
    """Full environment with Gaussian random fields for resources, temperature, AND toxin.

    This matches darwin_v0's default world - smooth blob-shaped regions for everything.

    Observations: [food, temperature, toxin, contact_front, contact_back, contact_left, contact_right]
    Actions: [eat, forward, left, right, stay, reproduce, attack]
    """

    def __init__(
        self,
        size: int = 512,
        resource_mean: float = 15.0,
        resource_max: float = 30.0,
        resource_length_scale: float = 50.0,
        toxin_coverage: float = 0.05,
        toxin_length_scale: float = 30.0,
        food_noise_std: float = 0.1,
        temp_noise_std: float = 0.1,
        toxin_noise_std: float = 0.1,
        toxin_detection_radius: int = 5,
    ):
        super().__init__(size, food_noise_std, temp_noise_std, toxin_noise_std, toxin_detection_radius)
        self.resource_mean = resource_mean
        self.resource_max = resource_max
        self.resource_length_scale = resource_length_scale
        self.toxin_coverage = toxin_coverage
        self.toxin_length_scale = toxin_length_scale

    def create_world(self, key: jax.Array, size: int) -> dict:
        """Create world with Gaussian random fields for resources, temperature, and toxin."""
        k1, k2, k3 = random.split(key, 3)

        # Gaussian resource field
        resource = self._generate_gaussian_field(k1, size, self.resource_length_scale)
        resource = (resource + 1) / 2 * self.resource_max
        resource = jnp.clip(resource, 0, self.resource_max)

        # Gaussian temperature field
        temperature = self._generate_gaussian_field(k2, size, 50.0)
        temperature = (temperature + 1) / 2  # Scale to [0, 1]

        # Gaussian toxin blobs (threshold on Gaussian field)
        toxin_field = self._generate_gaussian_field(k3, size, self.toxin_length_scale)
        threshold = jnp.percentile(toxin_field, (1.0 - self.toxin_coverage) * 100)
        toxin = toxin_field >= threshold

        return {
            "resource": resource,
            "resource_base": resource.copy(),
            "temperature": temperature,
            "toxin": toxin,
        }

    def _generate_gaussian_field(
        self, key: jax.Array, size: int, length_scale: float
    ) -> jax.Array:
        """Generate smooth random field using Gaussian kernel in Fourier space."""
        noise = random.normal(key, (size, size))
        freq = jnp.fft.fftfreq(size)
        fx, fy = jnp.meshgrid(freq, freq)
        freq_sq = fx**2 + fy**2
        kernel = jnp.exp(-freq_sq * (2 * jnp.pi * length_scale)**2 / 2)
        noise_fft = jnp.fft.fft2(noise)
        smooth_fft = noise_fft * kernel
        smooth = jnp.real(jnp.fft.ifft2(smooth_fft))
        # Normalize to [-1, 1]
        smooth = (smooth - smooth.mean()) / (smooth.std() + 1e-8)
        return jnp.clip(smooth, -3, 3) / 3


# -----------------------------------------------------------------------------
# Factory function for creating environments from config
# -----------------------------------------------------------------------------

ENVIRONMENT_REGISTRY = {
    "simple": SimpleEnvironment,
    "full": FullEnvironment,
    "gaussian": GaussianEnvironment,
    "gaussian_toxin": GaussianToxinEnvironment,
    "uniform": UniformEnvironment,
    "pretrain": PretrainEnvironment,
    "bridge": BridgeEnvironment,
    "temporal": TemporalGaussianEnvironment,
    "orbiting": OrbitingGaussianEnvironment,
    "toxin_pattern": ToxinPatternEnvironment,
}


def create_environment(config: dict) -> Environment:
    """Create environment from configuration dict."""
    env_type = config.get("type", "simple")

    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment type: {env_type}")

    env_class = ENVIRONMENT_REGISTRY[env_type]

    # Extract relevant kwargs from config
    kwargs = {k: v for k, v in config.items() if k != "type"}

    return env_class(**kwargs)
