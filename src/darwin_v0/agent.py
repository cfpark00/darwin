"""Agent architecture - pure definitions, no side effects."""

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple


class AgentState(NamedTuple):
    """LSTM hidden states (legacy, for gradual migration)."""
    h1: jax.Array
    c1: jax.Array
    h2: jax.Array
    c2: jax.Array


def compute_state_dim(hidden_dim: int) -> int:
    """Compute flat state dimension: h1 + c1 + h2 + c2."""
    return 4 * hidden_dim


def flatten_state(state: AgentState) -> jax.Array:
    """Flatten AgentState to 1D array."""
    return jnp.concatenate([state.h1, state.c1, state.h2, state.c2])


def unflatten_state(flat_state: jax.Array, hidden_dim: int) -> AgentState:
    """Unflatten 1D array to AgentState."""
    h1 = flat_state[0:hidden_dim]
    c1 = flat_state[hidden_dim:2*hidden_dim]
    h2 = flat_state[2*hidden_dim:3*hidden_dim]
    c2 = flat_state[3*hidden_dim:4*hidden_dim]
    return AgentState(h1=h1, c1=c1, h2=h2, c2=c2)


def flatten_state_to_components(flat_state: jax.Array, hidden_dim: int) -> tuple:
    """Unflatten to raw components (h1, c1, h2, c2) without NamedTuple overhead."""
    h1 = flat_state[0:hidden_dim]
    c1 = flat_state[hidden_dim:2*hidden_dim]
    h2 = flat_state[2*hidden_dim:3*hidden_dim]
    c2 = flat_state[3*hidden_dim:4*hidden_dim]
    return h1, c1, h2, c2


# Architecture constants
INPUT_DIM = 7  # food, temp, toxin, 4 contact sensors
OUTPUT_DIM = 7  # eat, forward, left, right, stay, reproduce, attack


def compute_param_dim(hidden_dim: int, internal_noise_dim: int) -> int:
    """Compute total number of parameters for flat genome."""
    input_dim_2 = hidden_dim + 1 + internal_noise_dim

    # LSTM1: 4 gates × (input_dim × hidden + hidden × hidden + hidden bias)
    lstm1_size = 4 * (INPUT_DIM * hidden_dim + hidden_dim * hidden_dim + hidden_dim)

    # LSTM2: 4 gates × (input_dim_2 × hidden + hidden × hidden + hidden bias)
    lstm2_size = 4 * (input_dim_2 * hidden_dim + hidden_dim * hidden_dim + hidden_dim)

    # Output: hidden × output + output bias
    output_size = hidden_dim * OUTPUT_DIM + OUTPUT_DIM

    return lstm1_size + lstm2_size + output_size


def init_params_flat(key: jax.Array, hidden_dim: int, internal_noise_dim: int) -> jax.Array:
    """Initialize flat parameter vector with Xavier-like initialization."""
    param_dim = compute_param_dim(hidden_dim, internal_noise_dim)
    # Use moderate scale - will be reshaped into different sized matrices
    scale = jnp.sqrt(2.0 / hidden_dim)
    return random.normal(key, (param_dim,)) * scale


def compute_param_slices(hidden_dim: int, internal_noise_dim: int) -> dict:
    """Pre-compute slice indices for unflattening. Call once at init."""
    input_dim_2 = hidden_dim + 1 + internal_noise_dim
    h = hidden_dim
    slices = {}
    idx = 0

    # LSTM1: 4 gates × (W_ix, W_hx, b_x)
    for gate in ['i', 'f', 'g', 'o']:
        slices[f'lstm1_W_i{gate}'] = (idx, idx + INPUT_DIM * h, (INPUT_DIM, h))
        idx += INPUT_DIM * h
        slices[f'lstm1_W_h{gate}'] = (idx, idx + h * h, (h, h))
        idx += h * h
        slices[f'lstm1_b_{gate}'] = (idx, idx + h, (h,))
        idx += h

    # LSTM2: 4 gates × (W_ix, W_hx, b_x)
    for gate in ['i', 'f', 'g', 'o']:
        slices[f'lstm2_W_i{gate}'] = (idx, idx + input_dim_2 * h, (input_dim_2, h))
        idx += input_dim_2 * h
        slices[f'lstm2_W_h{gate}'] = (idx, idx + h * h, (h, h))
        idx += h * h
        slices[f'lstm2_b_{gate}'] = (idx, idx + h, (h,))
        idx += h

    # Output
    slices['output_W'] = (idx, idx + h * OUTPUT_DIM, (h, OUTPUT_DIM))
    idx += h * OUTPUT_DIM
    slices['output_b'] = (idx, idx + OUTPUT_DIM, (OUTPUT_DIM,))

    return slices


def _lstm_step_direct(x: jax.Array, h: jax.Array, c: jax.Array,
                      W_ii: jax.Array, W_hi: jax.Array, b_i: jax.Array,
                      W_if: jax.Array, W_hf: jax.Array, b_f: jax.Array,
                      W_ig: jax.Array, W_hg: jax.Array, b_g: jax.Array,
                      W_io: jax.Array, W_ho: jax.Array, b_o: jax.Array) -> tuple:
    """Single LSTM step with direct weight arrays (no dict lookup)."""
    i = jax.nn.sigmoid(x @ W_ii + h @ W_hi + b_i)
    f = jax.nn.sigmoid(x @ W_if + h @ W_hf + b_f)
    g = jnp.tanh(x @ W_ig + h @ W_hg + b_g)
    o = jax.nn.sigmoid(x @ W_io + h @ W_ho + b_o)

    c_new = f * c + i * g
    h_new = o * jnp.tanh(c_new)
    return h_new, c_new


def agent_forward_fully_flat(flat_params: jax.Array, flat_state: jax.Array, obs: jax.Array,
                             energy: jax.Array, noise: jax.Array,
                             hidden_dim: int, internal_noise_dim: int) -> tuple:
    """Forward pass with both flat params and flat state.

    Uses direct array slicing without dict construction for maximum JIT efficiency.

    Args:
        flat_params: Flat parameter vector (param_dim,)
        flat_state: Flat state vector (4 * hidden_dim,) = [h1, c1, h2, c2]
        obs: Observation (7,)
        energy: Normalized energy level (scalar)
        noise: Random noise vector (internal_noise_dim,)
        hidden_dim: Hidden dimension
        internal_noise_dim: Noise dimension

    Returns:
        new_flat_state (4 * hidden_dim,), logits (7,)
    """
    h = hidden_dim
    input_dim_2 = h + 1 + internal_noise_dim

    # Pre-compute sizes for each LSTM gate block
    # LSTM1: each gate = INPUT_DIM*h + h*h + h
    gate1_size = INPUT_DIM * h + h * h + h
    lstm1_size = 4 * gate1_size

    # LSTM2: each gate = input_dim_2*h + h*h + h
    gate2_size = input_dim_2 * h + h * h + h
    lstm2_size = 4 * gate2_size

    # === Extract LSTM1 weights (inline slicing, no dict) ===
    idx = 0
    # Gate i
    W1_ii = flat_params[idx:idx + INPUT_DIM * h].reshape(INPUT_DIM, h); idx += INPUT_DIM * h
    W1_hi = flat_params[idx:idx + h * h].reshape(h, h); idx += h * h
    b1_i = flat_params[idx:idx + h]; idx += h
    # Gate f
    W1_if = flat_params[idx:idx + INPUT_DIM * h].reshape(INPUT_DIM, h); idx += INPUT_DIM * h
    W1_hf = flat_params[idx:idx + h * h].reshape(h, h); idx += h * h
    b1_f = flat_params[idx:idx + h]; idx += h
    # Gate g
    W1_ig = flat_params[idx:idx + INPUT_DIM * h].reshape(INPUT_DIM, h); idx += INPUT_DIM * h
    W1_hg = flat_params[idx:idx + h * h].reshape(h, h); idx += h * h
    b1_g = flat_params[idx:idx + h]; idx += h
    # Gate o
    W1_io = flat_params[idx:idx + INPUT_DIM * h].reshape(INPUT_DIM, h); idx += INPUT_DIM * h
    W1_ho = flat_params[idx:idx + h * h].reshape(h, h); idx += h * h
    b1_o = flat_params[idx:idx + h]; idx += h

    # === Extract LSTM2 weights ===
    # Gate i
    W2_ii = flat_params[idx:idx + input_dim_2 * h].reshape(input_dim_2, h); idx += input_dim_2 * h
    W2_hi = flat_params[idx:idx + h * h].reshape(h, h); idx += h * h
    b2_i = flat_params[idx:idx + h]; idx += h
    # Gate f
    W2_if = flat_params[idx:idx + input_dim_2 * h].reshape(input_dim_2, h); idx += input_dim_2 * h
    W2_hf = flat_params[idx:idx + h * h].reshape(h, h); idx += h * h
    b2_f = flat_params[idx:idx + h]; idx += h
    # Gate g
    W2_ig = flat_params[idx:idx + input_dim_2 * h].reshape(input_dim_2, h); idx += input_dim_2 * h
    W2_hg = flat_params[idx:idx + h * h].reshape(h, h); idx += h * h
    b2_g = flat_params[idx:idx + h]; idx += h
    # Gate o
    W2_io = flat_params[idx:idx + input_dim_2 * h].reshape(input_dim_2, h); idx += input_dim_2 * h
    W2_ho = flat_params[idx:idx + h * h].reshape(h, h); idx += h * h
    b2_o = flat_params[idx:idx + h]; idx += h

    # === Extract output weights ===
    output_W = flat_params[idx:idx + h * OUTPUT_DIM].reshape(h, OUTPUT_DIM); idx += h * OUTPUT_DIM
    output_b = flat_params[idx:idx + OUTPUT_DIM]

    # === Unflatten state ===
    h1 = flat_state[0:h]
    c1 = flat_state[h:2*h]
    h2 = flat_state[2*h:3*h]
    c2 = flat_state[3*h:4*h]

    # === Forward pass ===
    h1_new, c1_new = _lstm_step_direct(
        obs, h1, c1,
        W1_ii, W1_hi, b1_i, W1_if, W1_hf, b1_f,
        W1_ig, W1_hg, b1_g, W1_io, W1_ho, b1_o
    )

    internal = jnp.concatenate([h1_new, energy[None], noise])

    h2_new, c2_new = _lstm_step_direct(
        internal, h2, c2,
        W2_ii, W2_hi, b2_i, W2_if, W2_hf, b2_f,
        W2_ig, W2_hg, b2_g, W2_io, W2_ho, b2_o
    )

    logits = h2_new @ output_W + output_b

    # === Flatten new state ===
    new_flat_state = jnp.concatenate([h1_new, c1_new, h2_new, c2_new])
    return new_flat_state, logits


# === Legacy functions (kept for compatibility) ===

def unflatten_params(flat: jax.Array, hidden_dim: int, internal_noise_dim: int) -> dict:
    """Unflatten parameter vector into weight matrices. Legacy dict version."""
    input_dim_2 = hidden_dim + 1 + internal_noise_dim
    h = hidden_dim
    idx = 0

    lstm1 = {}
    for gate in ['i', 'f', 'g', 'o']:
        lstm1[f'W_i{gate}'] = flat[idx:idx + INPUT_DIM * h].reshape(INPUT_DIM, h)
        idx += INPUT_DIM * h
        lstm1[f'W_h{gate}'] = flat[idx:idx + h * h].reshape(h, h)
        idx += h * h
        lstm1[f'b_{gate}'] = flat[idx:idx + h]
        idx += h

    lstm2 = {}
    for gate in ['i', 'f', 'g', 'o']:
        lstm2[f'W_i{gate}'] = flat[idx:idx + input_dim_2 * h].reshape(input_dim_2, h)
        idx += input_dim_2 * h
        lstm2[f'W_h{gate}'] = flat[idx:idx + h * h].reshape(h, h)
        idx += h * h
        lstm2[f'b_{gate}'] = flat[idx:idx + h]
        idx += h

    output_W = flat[idx:idx + h * OUTPUT_DIM].reshape(h, OUTPUT_DIM)
    idx += h * OUTPUT_DIM
    output_b = flat[idx:idx + OUTPUT_DIM]

    return {'lstm1': lstm1, 'lstm2': lstm2, 'output_W': output_W, 'output_b': output_b}


def lstm_step(params: dict, h: jax.Array, c: jax.Array, x: jax.Array) -> tuple:
    """Single LSTM step. Legacy dict version."""
    i = jax.nn.sigmoid(x @ params['W_ii'] + h @ params['W_hi'] + params['b_i'])
    f = jax.nn.sigmoid(x @ params['W_if'] + h @ params['W_hf'] + params['b_f'])
    g = jnp.tanh(x @ params['W_ig'] + h @ params['W_hg'] + params['b_g'])
    o = jax.nn.sigmoid(x @ params['W_io'] + h @ params['W_ho'] + params['b_o'])

    c_new = f * c + i * g
    h_new = o * jnp.tanh(c_new)
    return h_new, c_new


def agent_forward_flat(flat_params: jax.Array, state: AgentState, obs: jax.Array,
                       energy: jax.Array, noise: jax.Array,
                       hidden_dim: int, internal_noise_dim: int) -> tuple:
    """Forward pass through agent network. Legacy version using AgentState."""
    params = unflatten_params(flat_params, hidden_dim, internal_noise_dim)

    h1_new, c1_new = lstm_step(params['lstm1'], state.h1, state.c1, obs)
    internal = jnp.concatenate([h1_new, energy[None], noise])
    h2_new, c2_new = lstm_step(params['lstm2'], state.h2, state.c2, internal)
    logits = h2_new @ params['output_W'] + params['output_b']

    new_state = AgentState(h1=h1_new, c1=c1_new, h2=h2_new, c2=c2_new)
    return new_state, logits


def sample_action(key: jax.Array, logits: jax.Array, temperature: float = 1.0) -> jax.Array:
    """Sample action from logits using softmax with temperature."""
    probs = jax.nn.softmax(logits / temperature)
    return random.categorical(key, jnp.log(probs + 1e-10))


def mutate_params_flat(key: jax.Array, flat_params: jax.Array, std: float) -> jax.Array:
    """Mutate flat parameter vector by adding Gaussian noise."""
    return flat_params + random.normal(key, flat_params.shape) * std


# === Legacy support (for gradual migration) ===

class LSTMParams(NamedTuple):
    """Parameters for a single LSTM layer."""
    W_ii: jax.Array
    W_hi: jax.Array
    b_i: jax.Array
    W_if: jax.Array
    W_hf: jax.Array
    b_f: jax.Array
    W_ig: jax.Array
    W_hg: jax.Array
    b_g: jax.Array
    W_io: jax.Array
    W_ho: jax.Array
    b_o: jax.Array


class AgentParams(NamedTuple):
    """Full agent network parameters."""
    lstm1: LSTMParams
    lstm2: LSTMParams
    output_W: jax.Array
    output_b: jax.Array
