#!/usr/bin/env python3
"""Utility functions for research projects."""

import os
import shutil
import sys
from pathlib import Path
from typing import Union
from dotenv import load_dotenv


def init_directory(directory: Union[str, Path], overwrite: bool = False) -> Path:
    """
    Initialize a directory with safety checks for overwriting.
    
    This is a generic tool for safely creating/overwriting directories. It uses the
    DATA_DIR environment variable to specify a safe prefix - only directories 
    under this prefix can be overwritten. This prevents accidental deletion of 
    important system directories.
    
    Args:
        directory: Path to directory (str or Path object)
        overwrite: Whether to overwrite existing directory
    
    Returns:
        Path object of the created directory
    
    Raises:
        SystemExit: If directory exists without overwrite, or safety checks fail
    """
    load_dotenv()
    
    directory = Path(directory)
    
    if directory.exists():
        if overwrite:
            # Get DATA_DIR from environment (loaded from .env)
            safe_prefix = os.environ.get('DATA_DIR')
            
            if not safe_prefix:
                print(f"Error: DATA_DIR not set in .env!")
                print(f"Cannot use --overwrite without DATA_DIR for safety.")
                print("Set DATA_DIR in .env file to specify where overwriting is allowed.")
                sys.exit(1)
            
            # Convert safe_prefix to absolute path for comparison
            safe_prefix = Path(safe_prefix).resolve()
            
            # Get absolute path of directory
            dir_absolute = directory.resolve()
            
            # Check if the absolute path starts with safe prefix
            if not str(dir_absolute).startswith(str(safe_prefix)):
                print(f"Error: Cannot overwrite {dir_absolute}")
                print(f"Directory must start with DATA_DIR: {safe_prefix}")
                print("This safety check prevents accidental deletion of important directories.")
                sys.exit(1)
            
            # Safe to remove
            print(f"Removing existing directory: {dir_absolute}")
            shutil.rmtree(dir_absolute)
            print("Directory removed successfully.")
        else:
            print(f"Error: Directory {directory} already exists!")
            print("Use --overwrite to remove it, or choose a different path.")
            sys.exit(1)
    
    # Create directory
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Created directory: {directory.resolve()}")
    return directory


# ============================================================================
# Other reusable utilities for the research
# ============================================================================
# Add stateless utility functions below that are expected to be used
# repetitively throughout the research project

import jax


def make_key(seed: int) -> jax.Array:
    """Create PRNG key."""
    return jax.random.PRNGKey(seed)


def expand_simple_to_full_params(simple_params: jax.Array, hidden_dim: int = 8,
                                  internal_noise_dim: int = 4) -> jax.Array:
    """Expand simple agent params (6 in, 6 out) to full agent params (7 in, 7 out).

    Simple agent: [food, temp, contact×4] → [eat, forward, left, right, stay, reproduce]
    Full agent: [food, temp, toxin, contact×4] → [eat, forward, left, right, stay, reproduce, attack]

    Toxin input weights initialized to 0 (agent ignores toxin initially).
    Attack output weights initialized to 0 (attack logit = 0, low probability).

    Args:
        simple_params: Flat parameter vector from simple agent (1238 for h=8, noise=4)
        hidden_dim: Hidden dimension (default 8)
        internal_noise_dim: Internal noise dimension (default 4)

    Returns:
        Flat parameter vector for full agent (1279 for h=8, noise=4)
    """
    import jax.numpy as jnp

    SIMPLE_INPUT = 6
    FULL_INPUT = 7
    SIMPLE_OUTPUT = 6
    FULL_OUTPUT = 7
    TOXIN_IDX = 2  # Where toxin is inserted in input

    h = hidden_dim
    input_dim_2 = h + 1 + internal_noise_dim  # LSTM2 input (same for both)

    # Expected sizes
    simple_lstm1_gate = SIMPLE_INPUT * h + h * h + h  # W_i + W_h + b
    full_lstm1_gate = FULL_INPUT * h + h * h + h
    lstm2_gate = input_dim_2 * h + h * h + h  # Same for both

    simple_expected = 4 * simple_lstm1_gate + 4 * lstm2_gate + h * SIMPLE_OUTPUT + SIMPLE_OUTPUT
    full_expected = 4 * full_lstm1_gate + 4 * lstm2_gate + h * FULL_OUTPUT + FULL_OUTPUT

    if simple_params.shape[0] != simple_expected:
        raise ValueError(f"FATAL: Expected simple params of size {simple_expected}, got {simple_params.shape[0]}")

    result_parts = []
    idx = 0

    # === LSTM1: expand W_i matrices (insert zero row for toxin) ===
    for gate in range(4):  # i, f, g, o gates
        # W_i: (SIMPLE_INPUT, h) → (FULL_INPUT, h)
        W_i_simple = simple_params[idx:idx + SIMPLE_INPUT * h].reshape(SIMPLE_INPUT, h)
        idx += SIMPLE_INPUT * h

        # Insert zero row at TOXIN_IDX
        W_i_full = jnp.concatenate([
            W_i_simple[:TOXIN_IDX],           # rows 0-1 (food, temp)
            jnp.zeros((1, h)),                 # row 2 (toxin) - zeros
            W_i_simple[TOXIN_IDX:]            # rows 2-5 → 3-6 (contacts)
        ], axis=0)
        result_parts.append(W_i_full.flatten())

        # W_h: (h, h) - unchanged
        W_h = simple_params[idx:idx + h * h]
        idx += h * h
        result_parts.append(W_h)

        # b: (h,) - unchanged
        b = simple_params[idx:idx + h]
        idx += h
        result_parts.append(b)

    # === LSTM2: unchanged (input is h1 + energy + noise, not observations) ===
    lstm2_size = 4 * lstm2_gate
    result_parts.append(simple_params[idx:idx + lstm2_size])
    idx += lstm2_size

    # === Output layer: expand for attack action ===
    # output_W: (h, SIMPLE_OUTPUT) → (h, FULL_OUTPUT)
    output_W_simple = simple_params[idx:idx + h * SIMPLE_OUTPUT].reshape(h, SIMPLE_OUTPUT)
    idx += h * SIMPLE_OUTPUT

    # Add zero column for attack
    output_W_full = jnp.concatenate([output_W_simple, jnp.zeros((h, 1))], axis=1)
    result_parts.append(output_W_full.flatten())

    # output_b: (SIMPLE_OUTPUT,) → (FULL_OUTPUT,)
    output_b_simple = simple_params[idx:idx + SIMPLE_OUTPUT]
    idx += SIMPLE_OUTPUT

    # Add zero for attack bias
    output_b_full = jnp.concatenate([output_b_simple, jnp.zeros((1,))])
    result_parts.append(output_b_full)

    # Concatenate all parts
    full_params = jnp.concatenate(result_parts)

    if full_params.shape[0] != full_expected:
        raise ValueError(f"FATAL: Expansion produced {full_params.shape[0]} params, expected {full_expected}")

    return full_params