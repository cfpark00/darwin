# 2025-12-23 Session Log: Simulation Migration and Bugfixes

## Summary
Migrated Darwin ALife simulation from `darwin_old` repo to new repo structure, analyzed the simulation architecture, and fixed critical bugs in movement and reproduction logic.

## Tasks Completed

### 1. Code Migration
- Migrated simulation code from `darwin_old/darwin_v0_0/` to new repo structure
- Created module structure in `src/`:
  - `src/agent.py` - Agent neural network (2-layer LSTM)
  - `src/physics.py` - Physics definitions (actions, movement)
  - `src/world.py` - World generation (resources, toxin, temperature)
  - `src/simulation.py` - Main simulation engine
  - `src/scripts/run.py` - Orchestration script (entry point)
- Created config structure in `configs/`:
  - `configs/run/default.yaml` - Run configuration
  - `configs/world/default.yaml` - World physics configuration
- Created `scripts/run.sh` bash wrapper
- Added JAX dependencies to `pyproject.toml`
- Set up `.env` with `DATA_DIR`

### 2. Architecture Analysis
- Documented the 9-phase simulation step loop
- Analyzed state storage structure (world, positions, params, energies, alive mask)
- Identified "un-Jaxxy" patterns limiting parallelization

### 3. Bug Fixes and Improvements

#### Attack Phase Optimization
- Removed artificial K=64 cap on attackers
- Changed from sparse sorting approach to fully parallel scatter-add
- Reduced code from 39 to 29 lines

#### Buffer Growth Fix
- Fixed bug where reproduction could be silently blocked when buffers full
- Changed from checking every 100 steps to checking every step
- `step_jit` now returns `num_alive` for cheap scalar sync
- Preemptive growth when >50% full (rare recompiles, no missed births)

#### Collision Detection
- Added sanity check at start of each step
- Detects if multiple alive agents occupy same cell (indicates bug)
- Cost: ~2-3μs per step (negligible)

#### CRITICAL: Reproduction Collision Bug
- **Found bug**: Multiple parents could independently claim same empty cell
- Parallel reproduction checks used stale occupancy snapshot
- Two offspring could be placed at same position
- **Fixed**: Added conflict detection with "everyone loses" rule

#### CRITICAL: Movement Cascade Bug
- **Found bug**: Movement didn't handle cascading blocks
- If A→B→C and C stays, B blocked but A didn't know
- Could result in A and B at same cell
- **Fixed**: Conservative rule - only move into currently empty cells

#### Unified Conflict Resolution
- Both movement and reproduction now use same rule:
  - Cell must be currently empty
  - No one else intends to go there
  - Conflict → everyone loses (stays/fails)

### 4. Minor Fixes
- Fixed step 0 logging to show real actions instead of dummy zeros
- Updated imports to use `from src.xxx import ...` pattern

## Files Modified/Created
- `src/agent.py` (new)
- `src/physics.py` (new, later modified for movement fix)
- `src/world.py` (new)
- `src/simulation.py` (new, multiple modifications for fixes)
- `src/utils.py` (added `make_key` function)
- `src/scripts/run.py` (new)
- `configs/run/default.yaml` (new)
- `configs/world/default.yaml` (new)
- `scripts/run.sh` (new)
- `pyproject.toml` (added JAX dependencies)
- `.env` (created with DATA_DIR)

## Current State
- Simulation should now run without collision errors
- Movement is more conservative (can't follow/swap with other agents)
- Reproduction conflicts result in failed reproduction (not duplicates)
- Ready for testing
