# 2025-12-27 Session Log: Explorer Selection Experiments

## Summary
Extended transfer experiment infrastructure to support position-based agent filtering, enabling iterative selection of "explorer" agents that successfully cross the bridge.

## Tasks Completed

### 1. Position Filter for Transfer Experiments
- **Added**: `position_filter` parameter to `load_agents_from_checkpoint()` in `src/scripts/transfer.py`
- **Format**: `[x_min, x_max, y_min, y_max]` to select agents in a specific region
- **Error handling**: Now raises `ValueError` if fewer than requested agents match filter (fail-fast)

### 2. Explorer Selection Experiment Chain
Created iterative experiment configs to select for bridge-crossing ability:

| Experiment | Source Checkpoint | Filter | Seed |
|------------|-------------------|--------|------|
| bridge | run_default_archived @ 100000 | none | 0 |
| bridge_explorers | bridge @ 2750 | right fertile (x>=224) | 1 |
| bridge_explorers2 | bridge_explorers @ 1750 | right fertile | 2 |
| bridge_explorers3 | bridge_explorers2 @ 1500 | right fertile | 3 |
| bridge_explorers4 | bridge_explorers3 @ 2000 | right fertile | 4 |

**Hypothesis**: By iteratively selecting agents that cross the bridge early, we breed increasingly effective explorers.

### 3. Parameter Adjustments
- `resource_bridge`: 10 → 12 (slightly easier bridge crossing)
- All bridge experiments: max_steps = 8000, checkpoint_interval = 250

## Files Created
- `configs/run/bridge_explorers.yaml`
- `configs/run/bridge_explorers2.yaml`
- `configs/run/bridge_explorers3.yaml`
- `configs/run/bridge_explorers4.yaml`
- `scripts/transfer_explorers.sh`
- `scripts/transfer_explorers2.sh`
- `scripts/transfer_explorers3.sh`
- `scripts/transfer_explorers4.sh`

## Files Modified
- `src/scripts/transfer.py` - Added position_filter support, strict error on insufficient agents
- `configs/run/bridge.yaml` - seed 0, max_steps 8000, checkpoint_interval 250
- `configs/world/bridge.yaml` - resource_bridge 12

## Current State
- Explorer selection chain ready to run sequentially
- Each iteration samples 100 agents from right fertile strip (successful crossers)
- Early checkpoint times (1500-2750 steps) select for fast crossers
