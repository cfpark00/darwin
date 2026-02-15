# Diffusion Experiments with Spawn Region

**Date:** 2026-01-18 05:01

## Summary

Added proper spawn_region support to evolvability_v1 for diffusion experiments, matching darwin_v0's methodology. Ran controlled diffusion experiments comparing agent movement under different resource levels.

## Tasks Completed

- Added `spawn_region` support to `run.py` checkpoint loading
  - When loading agents from checkpoint with `spawn_region` config, agents are placed in a specified rectangular region (e.g., 16x16 centered area)
  - Limits agents to spawn region capacity automatically
  - Fresh brain states and reset ages for spawned agents

- Updated `test_clamped_diffusion` configs:
  - Added `spawn_region: [120, 136, 120, 136]` (16x16 centered in 256x256 arena)
  - Set `initial_agents: 256` (fits in spawn region)
  - Added `initial: 30.0` energy for spawned agents

- Ran both diffusion experiments (resource_10 and resource_25):
  - 256 agents, energy clamped at 30, reproduction disabled
  - 5000 steps each

- Created diffusion analysis script (`scratch/diffusion_analysis_v1/analyze_diffusion.py`):
  - Computes Mean Squared Displacement (MSD) from checkpoint positions
  - Fits diffusion coefficient D via: MSD(t) = 4Dt
  - Generates comparison plots and summary

## Files Modified/Created

**Modified:**
- `src/evolvability_v1/scripts/run.py` - Added spawn_region support in checkpoint loading
- `configs/evolvability_v1/run/test_clamped_diffusion/resource_10.yaml` - Added spawn_region, adjusted initial_agents
- `configs/evolvability_v1/run/test_clamped_diffusion/resource_25.yaml` - Added spawn_region, adjusted initial_agents

**Created:**
- `scratch/diffusion_analysis_v1/analyze_diffusion.py` - Diffusion coefficient analysis
- `scratch/diffusion_analysis_v1/output/diffusion_comparison.png` - Comparison plot
- `scratch/diffusion_analysis_v1/output/diffusion_results.txt` - Results summary

## Key Results

| Condition | Resource | Diffusion Coef (D) | R² |
|-----------|----------|-------------------|-----|
| resource_10 | 10 | 0.0038 | 0.95 |
| resource_25 | 25 | 0.0135 | 0.91 |

**Finding:** Higher resources (25) leads to ~3.5x higher diffusion than low resources (10). This is opposite to the original hypothesis that lower resources would cause more exploration. Possible explanation: pretrained agents learned to eat when food is detected, requiring more movement when food is everywhere.

## Key Decisions

- Used darwin_v0's methodology: spawn_region to cluster agents in center, then measure spread over time
- MSD formula: MSD(t) = 4Dt for 2D diffusion
- Only track agents alive at both t=0 and current time for consistent measurement

## Next Steps

- Investigate why higher resources lead to more movement (analyze action distributions)
- Consider whether pretrained behavior biases results
- Could run with randomly initialized (untrained) agents for comparison
