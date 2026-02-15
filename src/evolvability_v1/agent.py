"""Unified agent with configurable I/O dimensions.

This replaces both agent.py and agent_simple.py from darwin_v0.
The only difference was input/output dimensions - now configurable.
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial

from src.evolvability_v1.types import AgentConfig


def compute_param_dim(config: AgentConfig) -> int:
    """Compute total number of parameters for flat genome representation."""
    h = config.hidden_dim
    n = config.internal_noise_dim
    i = config.input_dim
    o = config.output_dim

    # LSTM layer 1: input is [obs, energy, noise] = i + 1 + n
    # Gates: 4 * (input_size * hidden + hidden * hidden + hidden)
    lstm1_input = i + 1 + n
    lstm1 = 4 * (lstm1_input * h + h * h + h)

    # LSTM layer 2: input is hidden
    lstm2 = 4 * (h * h + h * h + h)

    # Output layer: hidden -> output_dim
    output = h * o + o

    return lstm1 + lstm2 + output


def compute_state_dim(config: AgentConfig) -> int:
    """Compute LSTM state dimension (h and c for 2 layers)."""
    return config.hidden_dim * 4  # h1, c1, h2, c2


def init_params(key: jax.Array, config: AgentConfig) -> jax.Array:
    """Initialize flat parameter vector with Xavier initialization."""
    param_dim = compute_param_dim(config)
    # Xavier initialization scaled for typical layer sizes
    std = 0.1
    return random.normal(key, (param_dim,)) * std


def init_brain_state(config: AgentConfig) -> jax.Array:
    """Initialize LSTM state (zeros)."""
    return jnp.zeros(compute_state_dim(config))


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def forward(
    params: jax.Array,
    brain_state: jax.Array,
    obs: jax.Array,
    energy_normalized: float,
    noise: jax.Array,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    internal_noise_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Forward pass through agent neural network.

    Args:
        params: Flat parameter vector
        brain_state: LSTM hidden states [h1, c1, h2, c2]
        obs: Observation vector (input_dim,)
        energy_normalized: Energy / max_energy
        noise: Internal noise vector (internal_noise_dim,)
        input_dim, output_dim, hidden_dim, internal_noise_dim: Static config

    Returns:
        (new_brain_state, action_logits)
    """
    h = hidden_dim
    n = internal_noise_dim
    i = input_dim
    o = output_dim

    # Unpack brain state
    h1 = brain_state[:h]
    c1 = brain_state[h:2*h]
    h2 = brain_state[2*h:3*h]
    c2 = brain_state[3*h:4*h]

    # Build input: [obs, energy, noise]
    x = jnp.concatenate([obs, jnp.array([energy_normalized]), noise])

    # Compute parameter slice indices
    lstm1_input = i + 1 + n
    lstm1_size = 4 * (lstm1_input * h + h * h + h)
    lstm2_size = 4 * (h * h + h * h + h)

    # Extract LSTM1 parameters
    idx = 0
    W_i1 = params[idx:idx + lstm1_input * h].reshape(lstm1_input, h)
    idx += lstm1_input * h
    W_h1 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    b1 = params[idx:idx + h]
    idx += h

    W_f1 = params[idx:idx + lstm1_input * h].reshape(lstm1_input, h)
    idx += lstm1_input * h
    W_hf1 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    bf1 = params[idx:idx + h]
    idx += h

    W_g1 = params[idx:idx + lstm1_input * h].reshape(lstm1_input, h)
    idx += lstm1_input * h
    W_hg1 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    bg1 = params[idx:idx + h]
    idx += h

    W_o1 = params[idx:idx + lstm1_input * h].reshape(lstm1_input, h)
    idx += lstm1_input * h
    W_ho1 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    bo1 = params[idx:idx + h]
    idx += h

    # LSTM1 forward
    i_gate1 = jax.nn.sigmoid(x @ W_i1 + h1 @ W_h1 + b1)
    f_gate1 = jax.nn.sigmoid(x @ W_f1 + h1 @ W_hf1 + bf1)
    g_gate1 = jnp.tanh(x @ W_g1 + h1 @ W_hg1 + bg1)
    o_gate1 = jax.nn.sigmoid(x @ W_o1 + h1 @ W_ho1 + bo1)
    c1_new = f_gate1 * c1 + i_gate1 * g_gate1
    h1_new = o_gate1 * jnp.tanh(c1_new)

    # Extract LSTM2 parameters
    W_i2 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    W_h2 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    b2 = params[idx:idx + h]
    idx += h

    W_f2 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    W_hf2 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    bf2 = params[idx:idx + h]
    idx += h

    W_g2 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    W_hg2 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    bg2 = params[idx:idx + h]
    idx += h

    W_o2 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    W_ho2 = params[idx:idx + h * h].reshape(h, h)
    idx += h * h
    bo2 = params[idx:idx + h]
    idx += h

    # LSTM2 forward
    i_gate2 = jax.nn.sigmoid(h1_new @ W_i2 + h2 @ W_h2 + b2)
    f_gate2 = jax.nn.sigmoid(h1_new @ W_f2 + h2 @ W_hf2 + bf2)
    g_gate2 = jnp.tanh(h1_new @ W_g2 + h2 @ W_hg2 + bg2)
    o_gate2 = jax.nn.sigmoid(h1_new @ W_o2 + h2 @ W_ho2 + bo2)
    c2_new = f_gate2 * c2 + i_gate2 * g_gate2
    h2_new = o_gate2 * jnp.tanh(c2_new)

    # Output layer
    W_out = params[idx:idx + h * o].reshape(h, o)
    idx += h * o
    b_out = params[idx:idx + o]

    logits = h2_new @ W_out + b_out

    # Pack new brain state
    new_brain_state = jnp.concatenate([h1_new, c1_new, h2_new, c2_new])

    return new_brain_state, logits


def sample_action(
    key: jax.Array,
    logits: jax.Array,
    temperature: float = 1.0,
    action_mask: jax.Array = None,
) -> int:
    """Sample action from logits with optional masking."""
    if action_mask is not None:
        # Set masked actions to -inf
        logits = jnp.where(action_mask, logits, -1e10)

    if temperature <= 0:
        return jnp.argmax(logits)

    probs = jax.nn.softmax(logits / temperature)
    return random.categorical(key, jnp.log(probs + 1e-10))


def mutate_params(key: jax.Array, params: jax.Array, std: float) -> jax.Array:
    """Mutate parameters with Gaussian noise."""
    noise = random.normal(key, params.shape) * std
    return params + noise


# Vectorized versions for batch operations
def init_params_batch(key: jax.Array, n: int, config: AgentConfig) -> jax.Array:
    """Initialize n agents' parameters."""
    keys = random.split(key, n)
    return jax.vmap(lambda k: init_params(k, config))(keys)


def init_brain_states_batch(n: int, config: AgentConfig) -> jax.Array:
    """Initialize n agents' brain states."""
    state_dim = compute_state_dim(config)
    return jnp.zeros((n, state_dim))
