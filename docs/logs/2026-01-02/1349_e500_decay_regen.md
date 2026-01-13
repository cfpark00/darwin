# E500 Decay Regeneration Experiment

## Summary
Created infrastructure to continue the e500 simulation with dynamically decaying resource regeneration. Resources regenerate slower over time, simulating environmental degradation.

## Files Created

### Script
- `src/scripts/continue_no_regen.py` - Continues simulation from checkpoint with dynamic regeneration
  - Disables built-in simulation regen (uses very high timescale in config)
  - Manually applies regeneration after each step with configurable dynamic timescale
  - Supports `dynamic_regen` config: `initial_timescale`, `growth_rate`, `growth_interval`
  - Tracks regen timescale history in `results/regen_history.json`

### Configs
- `configs/world/e500_decay_regen.yaml` - E500 world with built-in regen disabled (`regen_timescale: 1e9`)
- `configs/run/e500_decay_regen.yaml` - Run config with dynamic regen settings:
  - `initial_timescale: 100` (normal speed)
  - `growth_rate: 0.10` (+10% increase)
  - `growth_interval: 100` (every 100 steps)

### Bash Script
- `scripts/run_e500_decay_regen.sh` - Runner script

## Technical Details

The simulation's JIT-compiled step function captures `regen_timescale` at compile time, making dynamic changes impossible without recompilation. Solution:
1. Set built-in regen to effectively zero (`timescale = 1e9`)
2. Apply regeneration manually after each step using `regenerate_resources()` with the current dynamic timescale

After 50k steps with +10% every 100 steps:
- Timescale grows from 100 to ~13,780
- Resources recover ~138× slower than at start

## Usage
```bash
bash scripts/run_e500_decay_regen.sh
```

## Notes
- Continues from `data/run_default_e500/checkpoints/ckpt_050000.pkl`
- Full state preserved (agents + world resource distribution)
- Progress bar shows current `regen_t` value
