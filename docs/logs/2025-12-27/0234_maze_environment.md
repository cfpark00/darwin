# 2025-12-27 Session Log: Maze Environment & Dynamic Goals

## Summary
Added maze environment from chemo1 project, implemented toxin death tracking, created iterative maze solver selection experiments, and built dynamic maze where goal location shifts based on population.

## Tasks Completed

### 1. Maze World Implementation
- **Added**: `src/worlds/maze.py` - 8x8 cell maze where walls are toxic
- **Layout**: Start at bottom-left, goal at top-right
- **Required config fields**: `resource_start`, `resource_passable`, `resource_goal` (fail-fast)
- **Registered** in `src/worlds/__init__.py`

### 2. Toxin Death Tracking
- **Modified** `src/simulation.py`:
  - Count toxin deaths in Phase 5: `num_toxin_deaths = jnp.sum(alive & on_toxin)`
  - Return `num_toxin_deaths` in step output
- **Modified** `src/scripts/run.py` and `transfer.py`:
  - Track `toxin_deaths` in history
  - New plot: `figures/toxin_deaths.png`

### 3. Maze Experiment Configs
| Config | Purpose |
|--------|---------|
| `configs/world/maze.yaml` | Maze world (resource: start=5, passable=18, goal=50) |
| `configs/run/maze.yaml` | Transfer 100 agents from default run to maze start |
| `configs/run/maze_solvers.yaml` | Select agents from goal @ step 2500, restart |
| `configs/run/maze_solvers2.yaml` | Select from solvers @ step 1750, restart |

### 4. Dynamic Maze Environment
- **Created** `src/scripts/run_maze_dynamic.py` - Custom run script with moving goal
- **Logic**: When >100 agents in goal cell, move goal to next passable cell
- **Config** `configs/run/maze_dynamic.yaml` with `dynamic_goal.threshold: 100`
- **Output**: `logs/goal_moves.jsonl`, `logs/goal_history.json`, `figures/goal_population.png`

### 5. Hidden Dim Variant
- **Created** `configs/world/default_h24.yaml` - Same as default but `hidden_dim: 24`
- **Created** `configs/run/default_h24.yaml` and `scripts/run_h24.sh`
- Parameters: ~7,351 (vs ~1,055 for h=8)

### 6. Plotting Fix
- **Fixed** arena plot y-axis orientation in `src/scripts/run.py`
- Changed to `origin='lower'` so maze top appears at plot top
- Extent: `[0, size, 0, size]` (was `[0, size, size, 0]`)

## Files Created
- `src/worlds/maze.py`
- `configs/world/maze.yaml`
- `configs/world/maze_dynamic.yaml`
- `configs/world/default_h24.yaml`
- `configs/run/maze.yaml`
- `configs/run/maze_solvers.yaml`
- `configs/run/maze_solvers2.yaml`
- `configs/run/maze_dynamic.yaml`
- `configs/run/default_h24.yaml`
- `src/scripts/run_maze_dynamic.py`
- `scripts/transfer_maze.sh`
- `scripts/transfer_maze_solvers.sh`
- `scripts/transfer_maze_solvers2.sh`
- `scripts/run_maze_dynamic.sh`
- `scripts/run_h24.sh`

## Files Modified
- `src/worlds/__init__.py` - Registered maze world
- `src/simulation.py` - Added toxin death counting
- `src/scripts/run.py` - Toxin death history/plot, fixed plot orientation
- `src/scripts/transfer.py` - Toxin death history

## Energy Analysis (Maze)
- Passable (18): net +5.44/eat, supports ~66% movement speed
- Goal (50): net +16/eat, rapid reproduction
- Start (5): net +1.15/eat, pressure to leave

## Current State
- Maze environment fully functional
- Solver selection chain (maze → maze_solvers → maze_solvers2) ready
- Dynamic maze tested, goals move correctly
- Plot orientation fixed to match coordinate logs
