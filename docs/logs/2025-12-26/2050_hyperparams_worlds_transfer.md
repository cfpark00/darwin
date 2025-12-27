# 2025-12-26 Session Log: Hyperparameters, Modular Worlds, and Transfer Experiments

## Summary
Tuned simulation hyperparameters, added visualization features, created modular world system for different arena types, and set up infrastructure for transfer experiments (loading evolved agents into new environments).

## Tasks Completed

### 1. Bug Fix: Collision After Buffer Growth
- **Problem**: Multiple alive agents could occupy same cell after buffer growth
- **Root cause**: `_build_occupancy_grid` and `resolve_move_conflicts` used `.at[].set()` which with duplicate indices uses last-write-wins. Dead agents at position [0,0] after buffer growth could overwrite alive agents.
- **Fix**: Changed to `.at[].max()` so alive agent IDs always win over dead (0)

### 2. Hyperparameter Tuning
- `regen_timescale`: 50 → 100 (slower resource regeneration)
- `eat_fraction`: 0.5 → 0.33 (eat less per bite)
- Added `cost_eat`: 0.5 (eating now costs energy)
- `cost_attack`: 10 → 5
- `cost_stay`: 1.0 → 0.5
- `resource_max`: 50 → 30, `resource_mean`: 20 → 15

### 3. Metabolic Penalty (Soft Energy Cap)
- Replaced hard energy cap at 100 with continuous metabolic penalty
- Penalty = `energy / max_energy` added to all action costs
- At energy 100: +1 cost, at 50: +0.5, at 200: +2
- Creates soft pressure against hoarding energy without hard limit

### 4. Visualization Improvements
- **Energy histogram panel**: 3x3 grid showing last 9 energy distribution snapshots
- **Attack stats plot**: Attack count and kill count over time, plus success ratio
- **Arena plot fix**: Fixed thin white line at bottom of step images (explicit extent parameter)

### 5. Modular World System
Created `src/worlds/` directory with registry pattern:
- `src/worlds/base.py` - Shared utilities (Gaussian field generation, resource regeneration)
- `src/worlds/default.py` - Gaussian random world (original behavior)
- `src/worlds/bridge.py` - Two fertile strips connected by narrow bridge
- `src/worlds/__init__.py` - Registry dispatch (`WORLD_TYPES` dict)
- World creators return `resource_min`/`resource_max` in world dict

### 6. Transfer Experiment Infrastructure
- Created `src/scripts/transfer.py` - Load agents from checkpoint into new environment
- Created `configs/run/bridge.yaml` - Transfer experiment config
- Created `configs/world/bridge.yaml` - Bridge world config
- Created `scripts/transfer.sh` - Bash wrapper
- Added `Simulation.reset_with_agents()` method with `spawn_region` support

### 7. Config Cleanup
- Moved `toxin_detection_radius` from arena to sensors section (where it belongs)
- Added `min_buffer_size` as run config parameter (compute meta-param, not simulation param)
- Changed `Simulation.__init__` to take `run_config` dict (easier to add params later)
- Removed unused params from bridge config (no toxin in bridge world)

### 8. Performance Fix
- Adding `resource_min`/`resource_max` as static args to step_jit caused 5x slowdown
- Fixed by removing them entirely (regeneration naturally bounds resources)
- Reverted to single static arg (`max_agents`)

## Files Created
- `src/worlds/__init__.py`
- `src/worlds/base.py`
- `src/worlds/default.py`
- `src/worlds/bridge.py`
- `src/scripts/transfer.py`
- `configs/world/bridge.yaml`
- `configs/run/bridge.yaml`
- `scripts/transfer.sh`

## Files Modified
- `src/simulation.py` - Collision fix, metabolic penalty, reset_with_agents, run_config
- `src/physics.py` - Collision fix in resolve_move_conflicts
- `src/world.py` - Now dispatches to src/worlds/
- `src/scripts/run.py` - Energy histogram, attack stats, arena fix, run_config
- `configs/world/default.yaml` - Hyperparameters, toxin_detection_radius moved
- `configs/run/default.yaml` - Added min_buffer_size

## Current State
- Simulation runs at ~115 steps/s on GPU
- Modular world system ready for new arena types
- Transfer experiment infrastructure ready
- Next: Run transfer experiment to observe how evolved agents adapt to bridge environment
