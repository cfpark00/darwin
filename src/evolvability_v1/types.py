"""Core types for evolvability_v1 simulation framework."""

from dataclasses import dataclass
from typing import NamedTuple, Optional
import jax.numpy as jnp
import jax


class Metrics(NamedTuple):
    """Metrics computed inside JIT - all JAX arrays, no Python conversion needed.

    Use masked reductions (sum with alive mask) instead of fancy indexing
    to keep static shapes and enable full JIT compilation.
    """
    num_alive: jax.Array      # scalar int32
    mean_energy: jax.Array    # scalar float32
    mean_age: jax.Array       # scalar float32
    max_age: jax.Array        # scalar int32
    action_counts: jax.Array  # (num_actions,) float32 - fraction of each action
    # Unphysical event tracking (should be 0 in ideal simulation)
    repro_capped: jax.Array   # scalar int32 - reproductions that failed due to lack of dead slots
    # Combat/hazard tracking (only populated if environment has these capabilities)
    num_toxin_deaths: jax.Array  # scalar int32 - deaths from toxin this step
    num_attacks: jax.Array       # scalar int32 - attack actions taken this step
    num_kills: jax.Array         # scalar int32 - successful kills from attacks this step


class SimState(NamedTuple):
    """Simulation state - all arrays have shape (max_agents, ...) or scalar."""
    # World
    world: dict  # Contains 'resource', 'temperature', etc.

    # Agent state
    positions: jax.Array      # (max_agents, 2) int32 - [y, x]
    orientations: jax.Array   # (max_agents,) int32 - 0=N, 1=E, 2=S, 3=W
    params: jax.Array         # (max_agents, param_dim) float32 - neural network weights
    brain_states: jax.Array   # (max_agents, state_dim) float32 - LSTM hidden states
    energies: jax.Array       # (max_agents,) float32
    alive: jax.Array          # (max_agents,) bool
    ages: jax.Array           # (max_agents,) int32

    # Lineage tracking
    uid: jax.Array            # (max_agents,) int32 - unique ID
    parent_uid: jax.Array     # (max_agents,) int32 - parent's UID (-1 for founders)
    next_uid: int             # Next UID to assign

    # Bookkeeping
    step: int
    max_agents: int

    # Computed inside JIT (no sync needed until logging)
    metrics: Optional[Metrics] = None

    # Optional (set during step)
    actions: Optional[jax.Array] = None  # (max_agents,) int32 - last actions taken


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent neural network."""
    input_dim: int            # Observation size
    output_dim: int           # Number of actions
    hidden_dim: int = 8       # LSTM hidden size
    internal_noise_dim: int = 4


@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for physics/energy costs."""
    world_size: int
    max_energy: float = 100.0
    base_cost: float = 0.5
    base_cost_incremental: float = 0.01  # Cost scales with energy
    cost_eat: float = 0.0
    cost_move: float = 2.5
    cost_stay: float = 0.0
    cost_reproduce: float = 29.5
    cost_attack: float = 5.0  # Only used if environment.has_attack
    offspring_energy: float = 25.0
    eat_fraction: float = 0.33
    regen_timescale: float = 100.0
    mutation_std: float = 0.1
    # Clamp options for controlled experiments
    energy_clamp: Optional[float] = None  # If set, clamp all energies to this value (immortal agents)
    resource_clamp: Optional[float] = None  # If set, keep resources at this fixed level
    disabled_actions: tuple = ()  # Tuple of action indices to disable (e.g., (5,) to disable reproduce)


@dataclass(frozen=True)
class RunConfig:
    """Configuration for running simulation."""
    seed: int = 0
    max_steps: int = 10000
    initial_agents: int = 512
    min_buffer_size: int = 1024
    log_interval: int = 10
    checkpoint_interval: int = 1000
    detailed_interval: int = 100
    # Buffer growth threshold: grow when population > threshold * buffer_size
    # WARNING: If population exceeds threshold and many agents reproduce,
    # some reproductions may fail due to lack of dead slots. These agents
    # still pay reproduction energy cost but get no offspring (unphysical).
    # Lower threshold = more headroom = fewer unphysical events.
    # At threshold T, up to (1-T) fraction of population can reproduce safely.
    # Default 0.5 means 50% of population can reproduce without hitting cap.
    buffer_growth_threshold: float = 0.5


# Action constants
EAT = 0
FORWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
REPRODUCE = 5
ATTACK = 6  # Only for environments that support it

ACTION_NAMES_SIMPLE = ["eat", "forward", "left", "right", "stay", "reproduce"]
ACTION_NAMES_FULL = ["eat", "forward", "left", "right", "stay", "reproduce", "attack"]
