# Thermotaxis Aging and Temperature Mechanics

## Summary
Implemented aging system and refined temperature mechanics for thermotaxis environment. Refactored energy costs across all world configs to use parametrized base cost system.

## Energy Cost Refactoring

### New Parametrization
All 7 world config files updated to use:
```yaml
energy:
  base_cost: 0.5                    # Fixed base metabolism
  base_cost_incremental: 0.01       # Multiplied by current energy (soft cap)
  cost_eat: 0.0                     # Action costs are ADDITIONAL to base
  cost_move: 2.5
  cost_stay: 0.0
  cost_reproduce: 29.5
```

Formula: `total_cost = base_cost + (base_cost_incremental * energy) + action_cost`

This replaces the previous system where each action had its own full cost including metabolism.

### Files Modified
- `configs/world/default.yaml`
- `configs/world/default_h24.yaml`
- `configs/world/bridge.yaml`
- `configs/world/maze.yaml`
- `configs/world/maze_dynamic.yaml`
- `configs/world/pretrain.yaml`
- `configs/world/thermotaxis.yaml`
- `src/simulation.py` - Updated cost calculation
- `src/simulation_simple.py` - Updated cost calculation

## Aging System (Thermotaxis Only)

### Age Tracking
Added `ages` array to simulation state in `simulation_simple.py`:
- Initialized to 0 for all agents
- Incremented by 1 each step for alive agents
- Reset to 0 for newborn offspring
- Tracked through reset, reset_with_agents, _grow_buffers, step functions

### Age-Dependent Metabolic Penalty
```yaml
base_cost_age_incremental: 0.001  # Multiplied by age (steps alive)
```

At age 1000 steps: +1.0 metabolic cost per action.

## Temperature-Dependent Metabolism

### Optimal Temperature
Temperature 0.5 is optimal for metabolism (minimal penalty):
```python
temp_penalty = base_cost_temperature_incremental * |temp - 0.5|
```

Config: `base_cost_temperature_incremental: 0.75`

At temp extremes (0 or 1): +0.375 metabolic cost per action.

## Temperature Model Fix

### Problem
Previous formula produced mean temperature of 0.5 everywhere, with oscillation only near surface.

### Corrected Model
```python
mean_temp = 0.25 + 0.25 * (y / max_y)  # 0.5 at surface, 0.25 at bottom
decay = exp(-depth / damping_depth)
phase_lag = depth / damping_depth
temp = mean_temp + 0.5 * decay * sin(2*pi*t/period - phase_lag)
```

Key properties:
- Surface (y=511): oscillates 0.0 to 1.0
- Bottom (y=0): constant 0.25
- Amplitude and phase both decay with depth
- Minimum possible temperature: 0.0 (surface at cold phase)
- Maximum possible temperature: 1.0 (surface at hot phase)

### Files Updated
- `src/worlds/thermotaxis.py` - compute_temperature function
- `scratch/thermotaxis/visualize_env.py` - get_temperature function

## Damping Depth
Increased temperature_damping_depth from 64 to 128 pixels:
- `configs/world/thermotaxis.yaml`
- `scratch/thermotaxis/visualize_env.py`

## Y-Density Visualization

Changed from histogram to heatmap:
- X-axis: time (steps)
- Y-axis: y-position
- Color: agent density

Files:
- `src/scripts/run_simple.py`
- `src/scripts/transfer_simple.py`

Now tracks `y_density` history throughout simulation and renders as imshow heatmap.

## Thermotaxis Visualization

Updated `scratch/thermotaxis/visualize_env.py`:
- 3x3 grid showing Food, Temp (peak/trough), Repro (peak/trough), Metabolic penalty (peak/trough)
- Added metabolic penalty heatmap (y vs time) with temp=0.5 contour
- Metabolic penalty formula: `0.75 * |temp - 0.5|`

## Run Configuration
- `configs/run/thermotaxis.yaml`: Changed max_steps from 100k to 50k

## Full Cost Formula (Thermotaxis)
```python
total_cost = base_cost
           + base_cost_incremental * energy
           + base_cost_age_incremental * age
           + base_cost_temperature_incremental * |temp - 0.5|
           + action_cost
```

Default values:
- base_cost: 0.5
- base_cost_incremental: 0.01
- base_cost_age_incremental: 0.001
- base_cost_temperature_incremental: 0.75

Example at age=500, energy=50, temp=0.8, moving:
- 0.5 + (0.01 * 50) + (0.001 * 500) + (0.75 * 0.3) + 2.5 = 0.5 + 0.5 + 0.5 + 0.225 + 2.5 = 4.225
