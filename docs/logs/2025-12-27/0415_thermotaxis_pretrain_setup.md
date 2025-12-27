# Thermotaxis Environment and Pretrain Pipeline

## Summary
Created a thermotaxis environment simulating soil thermal dynamics for testing agent temperature-seeking behavior, plus a pretrain environment for curriculum learning.

## Thermotaxis Environment

### Design (based on Ramot et al. 2008)
- **Grid**: 512x512
- **Food**: Linear gradient (0 at y=0, 20 at y=max) + Gaussian noise (std=2.0)
  - Noise prevents agents from using food as a perfect y-coordinate proxy
- **Temperature**: Dynamic sinusoidal with depth-dependent decay and phase lag
  - Surface (top): oscillates 0-1 with period=2000 steps
  - Decay with depth: `exp(-depth/damping_depth)` where damping_depth=64
  - Phase lag: `depth/damping_depth` (deeper = delayed temperature response)
  - Formula: `T(z,t) = 0.5 + 0.5 * exp(-z/zd) * sin(2*pi*t/period - z/zd)`
- **Reproduction**: Temperature-dependent probability
  - 100% success at temp <= 0.5
  - Linear decay to 0% at temp = 1.0
- **No toxin or attack** (attack action does nothing)

### Files Created
- `src/worlds/thermotaxis.py` - Dynamic temperature computation with phase lag
- `configs/world/thermotaxis.yaml` - World configuration
- `configs/run/thermotaxis.yaml` - Run config with transfer from pretrain checkpoint

## Pretrain Environment

### Purpose
Simple "nursery" environment for agents to learn basic survival (eating, moving) before the harder thermotaxis challenge.

### Design
- **Grid**: 256x256
- **Food**: Curriculum learning
  - Starts at 30 (easy survival)
  - Decays to 12 over 10,000 steps (requires active foraging)
  - Minimum survivable density ~1.5 for reference
- **Temperature**: Static Gaussian field (no reproduction penalty)
- **512 initial agents**

### Files Created
- `src/worlds/pretrain.py` - Food decay curriculum
- `configs/world/pretrain.yaml` - World configuration
- `configs/run/pretrain.yaml` - Run config (10k steps)
- `scripts/run_pretrain.sh` - Execution script

## Transfer Pipeline
1. Run pretrain for 10k steps (agents learn to eat and move)
2. Load checkpoint into thermotaxis environment
3. Agents must now also learn temperature-seeking behavior

Modified `configs/run/thermotaxis.yaml` to include transfer section pointing to pretrain checkpoint.
Modified `scripts/run_thermotaxis.sh` to use `transfer.py` instead of `run.py`.

## Simulation Changes

### src/simulation.py
- Added temperature update in step() for thermotaxis arena type
- Added food base update in step() for pretrain arena type
- Added temperature-based reproduction probability calculation
- Fixed JAX random key handling (fold_in instead of array of keys)

### src/scripts/run.py
- Added y-density histogram plot for thermotaxis runs (agent distribution along y-axis)

## Issues Encountered
- Initial test showed all agents dying by step 400 with 512 initial agents
- Increased to 1024 initial agents for thermotaxis

## Key Insight
Food noise is critical - without it, agents can simply use food concentration to infer y-coordinate, bypassing the need to learn temperature sensing.
