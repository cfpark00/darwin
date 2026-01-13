# Maze from Default and Energy Max 500 Configs

## Summary
Created new experiment configurations for maze transfer from run_default checkpoint and a default run variant with higher energy cap.

## Files Created

### Maze Transfer from run_default
- `configs/run/maze_from_default.yaml` - Transfer 100 agents from `data/run_default/checkpoints/ckpt_050000.pkl` to maze
- `scripts/run_maze_from_default.sh` - Bash wrapper

### Default Run with Energy Max 500
- `configs/world/default_e500.yaml` - Same as default.yaml but `energy.max: 500`
- `configs/run/default_e500.yaml` - Run config using the e500 world
- `scripts/run_default_e500.sh` - Bash wrapper

## Files Updated
- `docs/structure.txt` - Added new configs and scripts

## Usage
```bash
# Transfer evolved agents from run_default to maze
bash scripts/run_maze_from_default.sh

# Run default simulation with energy max 500
bash scripts/run_default_e500.sh
```

## Notes
- Maze transfer uses checkpoint at step 50000 (not 100000 like the original transfer_maze.sh)
- Energy max 500 experiment may help observe different hoarding/reproduction strategies with higher energy cap
