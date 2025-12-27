# Simple Simulation Infrastructure (No Toxin/Attack)

## Summary
Created separate "simple" simulation infrastructure for pretrain and thermotaxis environments that don't use toxin sensing or attack actions. This avoids modifying the core simulation files used by other experiments (default, bridge, maze).

## Problem
User requested removing toxin input and attack action for thermotaxis/pretrain experiments. Initial approach incorrectly modified the core `agent.py` and `simulation.py` files, which would have broken all other experiments (default, bridge, maze) that rely on these features.

## Solution: Separate Simple Files

### New Files Created
- `src/agent_simple.py` - Agent with INPUT_DIM=6, OUTPUT_DIM=6
  - Inputs: food, temp, 4 contact sensors (no toxin)
  - Outputs: eat, forward, left, right, stay, reproduce (no attack)
- `src/simulation_simple.py` - SimulationSimple class
  - No toxin sensing in observations
  - No attack phase
  - No toxin death phase
  - 6 energy costs (no attack cost)
- `src/scripts/run_simple.py` - Run script using SimulationSimple
- `src/scripts/transfer_simple.py` - Transfer script using SimulationSimple

### Modified Files
- `configs/world/pretrain.yaml` - Removed cost_attack and toxin sensor configs
- `configs/world/thermotaxis.yaml` - Removed cost_attack and toxin sensor configs
- `scripts/run_pretrain.sh` - Now calls `run_simple.py`
- `scripts/run_thermotaxis.sh` - Now calls `transfer_simple.py`

## Architecture Summary

**Simple environments (pretrain, thermotaxis):**
```
scripts/run_pretrain.sh → run_simple.py → simulation_simple.py → agent_simple.py
scripts/run_thermotaxis.sh → transfer_simple.py → simulation_simple.py → agent_simple.py
```

**Full environments (default, bridge, maze):**
```
scripts/run.sh → run.py → simulation.py → agent.py
scripts/transfer*.sh → transfer.py → simulation.py → agent.py
```

## Original Files Verified Intact
- `src/agent.py` - INPUT_DIM=7, OUTPUT_DIM=7 (includes toxin input, attack output)
- `src/physics.py` - ATTACK=6 constant present
- `src/simulation.py` - Phase 3: Attacks, toxin_detected, all toxin logic intact
- `configs/world/default.yaml` - cost_attack, toxin_detection_radius present
- `configs/world/bridge.yaml` - cost_attack, toxin_detection_radius present
- `configs/world/maze.yaml` - cost_attack, toxin_detection_radius present

## Git Setup
- Initialized git repository
- Added remote: git@github.com:cfpark00/darwin.git
- Pushed all commits to origin/main

## Other Fixes
- Removed temperature overlay from arena plot in `run_simple.py` (was causing confusing visualization)

## Commits
- `e380c2a` - Initial commit: Darwin evolution simulation
- `ea05adb` - Add thermotaxis scratch files (visualizations)
- `4c86586` - Add simulation_simple.py (no toxin sensing, no attack)
- `bfaa81e` - Add simple simulation infrastructure (no toxin/attack)
- `b3c9804` - Remove temperature overlay from run_simple.py arena plot
