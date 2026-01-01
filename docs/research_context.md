# Darwin Research Context

## Goal
Simulating evolution to observe emergent multi-agent phenomena, including:
- Theory of mind
- Emergent strategies
- Complex agent behaviors

## System Overview

### Simulation
A 2D grid world where agents with neural network "brains" live, compete for resources, reproduce, and evolve through mutation.

### Agent Architecture
- **Brain**: 2-layer LSTM network (hidden_dim=8)
- **Full version** (`agent.py`):
  - Inputs (7): Food, temperature, toxin detection, 4 contact sensors
  - Outputs (7): eat, forward, left, right, stay, reproduce, attack
- **Simple version** (`agent_simple.py`): For pretrain/thermotaxis environments
  - Inputs (6): Food, temperature, 4 contact sensors (no toxin)
  - Outputs (6): eat, forward, left, right, stay, reproduce (no attack)
- **Evolution**: Offspring inherit parent's neural weights + Gaussian mutation (std=0.1)

### World Types
| Type | Description |
|------|-------------|
| gaussian | Random resource/toxin/temperature fields via Gaussian processes |
| bridge | Two fertile strips (edges) connected by narrow resource-poor bridge |
| maze | 8x8 cell maze with toxic walls, start at bottom-left, goal at top-right |
| pretrain | Food curriculum (30->12 over 10k steps), static temperature, no reproduction penalty |
| thermotaxis | Dynamic temperature simulating soil (sinusoidal at surface, decays+lags with depth), food gradient with noise |

### Actions
| Action | Cost | Effect |
|--------|------|--------|
| Eat | 0.5 | Consume 33% of resource at current cell |
| Forward | 3.0 | Move one cell in facing direction |
| Left/Right | 3.0 | Turn 90 degrees |
| Stay | 0.5 | Do nothing (baseline metabolism) |
| Reproduce | 30.0 | Create offspring (gets 25 energy) in adjacent empty cell |
| Attack | 5.0 | Kill agent in front, their energy drops to ground |

### Energy System
- **Initial**: 100
- **Base cost**: 0.5 per step (fixed metabolism)
- **Energy penalty**: `base_cost_incremental * energy` (default 0.01)
  - At energy 100: +1.0 cost to all actions
  - At energy 50: +0.5 cost
  - Creates pressure against hoarding without hard limit
- **Age penalty** (thermotaxis only): `base_cost_age_incremental * age` (default 0.001)
  - At age 1000: +1.0 cost to all actions
  - Simulates physiological aging
- **Temperature penalty** (thermotaxis only): `base_cost_temperature_incremental * |temp - 0.5|`
  - Optimal temperature: 0.5
  - At temp extremes (0 or 1): +0.375 cost
- **Total cost**: `base_cost + energy_penalty + age_penalty + temp_penalty + action_cost`
- **Death**: Energy <= 0

## Current State (2025-12-27)

### Completed
- Core simulation with collision-free movement and reproduction
- Modular world system (`src/worlds/`) for different arena types
- Transfer experiment infrastructure (load evolved agents into new environments)
- Visualization: population, action ratios, energy histograms, attack stats, toxin deaths
- Soft energy cap via metabolic penalty
- Position-based agent filtering for transfer experiments
- Maze environment with toxic walls and goal-reaching selection
- Dynamic maze with moving goal (relocates when population threshold exceeded)
- Hidden dim variant (hidden_dim=24 for ~7,351 params vs ~1,055 for h=8)
- Pretrain environment with food curriculum for basic survival training
- Thermotaxis environment with dynamic temperature (inspired by Ramot et al. 2008)
- Temperature-based reproduction probability system
- Y-density plot for thermotaxis runs
- Simple simulation infrastructure (`agent_simple.py`, `simulation_simple.py`) for pretrain/thermotaxis
- Parametrized energy cost system (base + incremental + action costs)
- Aging system for thermotaxis (metabolic cost scales with age)
- Temperature-dependent metabolism (optimal at temp=0.5)
- Corrected temperature model (mean gradient from 0.5 at surface to 0.25 at bottom)
- Y-density heatmap visualization (time vs y-position)
- Git repository initialized and pushed to github.com/cfpark00/darwin

### Infrastructure Ready
- **Default world**: 512x512 Gaussian, runs at ~115 steps/s
- **Bridge world**: 256x256, two fertile strips + connecting bridge
- **Maze world**: 256x256 (8x8 cells), toxic walls, resource gradient (start→passable→goal)
- **Dynamic maze**: Goal moves through passable cells when >100 agents occupy it
- **Pretrain world**: 256x256, food curriculum 30→12 over 10k steps
- **Thermotaxis world**: 512x512, dynamic temperature with phase lag, food gradient with noise
- **Transfer experiments**: Load checkpoint agents into new environment
- **Explorer selection**: Iterative selection of bridge-crossing agents (5 iterations configured)
- **Solver selection**: Iterative selection of maze-solving agents (2 iterations configured)
- **Thermotaxis pipeline**: Pretrain → thermotaxis transfer

### Current Experiment: Explorer Selection
Iteratively select agents that successfully cross the bridge to breed better explorers:
1. `bridge` - Transfer evolved agents from default world to bridge
2. `bridge_explorers` - Select agents from right side at step 2750
3. `bridge_explorers2` - Select from explorers at step 1750
4. `bridge_explorers3` - Select from explorers2 at step 1500
5. `bridge_explorers4` - Select from explorers3 at step 2000

Each iteration spawns 100 agents on left side, filters those who reach right fertile strip.

### Current Experiment: Maze Solver Selection
Iteratively select agents that navigate the maze to reach the goal:
1. `maze` - Transfer 100 agents from default world to maze start cell
2. `maze_solvers` - Select agents from goal cell at step 2500, restart from start
3. `maze_solvers2` - Select from solvers at step 1750, restart from start

Resource levels: start=5 (pressure to leave), passable=18 (sustainable), goal=50 (fertile).

### Dynamic Maze Experiment
Tests adaptability by moving the goal when agents congregate:
- Threshold: >100 agents in goal cell triggers relocation
- Goal cycles through all passable cells in the maze
- Tracks goal movement history and population dynamics

### Thermotaxis Experiment
Tests temperature-seeking behavior inspired by C. elegans thermotaxis (Ramot et al. 2008):

**Pretrain Phase** (10k steps):
- 256x256 grid, 512 agents
- Food starts at 30, decays to 12 (curriculum learning)
- Static Gaussian temperature, no reproduction penalty
- Goal: agents learn basic survival (eating, moving)

**Thermotaxis Phase** (50k steps):
- 512x512 grid, 512 agents transferred from pretrain
- Food: linear gradient (0 at bottom, 20 at top) + Gaussian noise (amplitude=8)
  - Noise prevents agents from using food as y-coordinate proxy
- Temperature: dynamic sinusoidal with depth-dependent decay and phase lag
  - Formula: `T(y,t) = mean(y) + 0.5 * exp(-z/zd) * sin(2*pi*t/period - z/zd)`
  - Mean temperature: 0.5 at surface (y=511), 0.25 at bottom (y=0)
  - Period: 1000 steps, damping depth: 128 pixels
  - Range: 0.0 (surface cold phase) to 1.0 (surface hot phase)
- Reproduction: 100% success at temp<=0.5, linear decay to 0% at temp=1.0
- Aging: metabolic cost increases linearly with age (0.001 per step alive)
- Temperature metabolism: optimal at temp=0.5, penalty for deviation
- No toxin, attack does nothing

**Research question**: Can agents learn to track optimal temperature zones that shift with the thermal wave while balancing food availability (top) vs thermal comfort (varies)?

## Key Parameters

### Default World
- Size: 512x512
- Initial agents: 512
- Resource: max=30, mean=15, regen_timescale=100
- Toxin coverage: 5%

### Bridge World
- Size: 256x256
- Fertile strips: width=32, resource=25
- Bridge: height=16, resource=12
- Barren: resource=5

### Maze World
- Size: 256x256 (8 cells × 32 pixels)
- Layout: 8x8 cell maze, walls are toxic (instant death)
- Start cell (bottom-left): resource=5 (net +1.15/eat)
- Passable cells: resource=18 (net +5.44/eat, ~66% movement sustainable)
- Goal cell (top-right): resource=50 (net +16/eat, rapid reproduction)

### Pretrain World
- Size: 256x256
- Initial agents: 512
- Food: uniform, starts at 30, decays to 12 over 10k steps
- Temperature: static Gaussian field
- Reproduction: no temperature penalty (threshold >= max temp)

### Thermotaxis World
- Size: 512x512
- Initial agents: 512 (transferred from pretrain)
- Food: gradient (0 at bottom, 20 at top) + Gaussian noise (amplitude=8, length_scale=30)
- Temperature: dynamic sinusoidal (period=1000, damping_depth=128)
  - Mean: 0.5 at surface, 0.25 at bottom
- Reproduction: temp-dependent (100% at <=0.5, 0% at 1.0)
- Aging: base_cost_age_incremental=0.001 (cost += age * 0.001)
- Temperature metabolism: base_cost_temperature_incremental=0.75 (cost += |temp-0.5| * 0.75)
- No toxin, attack disabled

## Open Questions
- What strategies will emerge in homogeneous vs structured environments?
- Will agents learn to cross the bridge to access isolated resources?
- Can evolved agents adapt when transferred to new environment types?
- Will cooperative or competitive behaviors emerge?
- Can agents learn to navigate the maze and avoid toxic walls?
- How does network capacity (hidden_dim) affect learning complex navigation?
- Will agents track moving goals in the dynamic maze environment?
- Can agents learn to track optimal temperature zones as thermal waves propagate through the environment?
