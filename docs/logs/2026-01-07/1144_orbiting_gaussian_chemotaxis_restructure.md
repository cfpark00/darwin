# Orbiting Gaussian, Chemotaxis Restructure, Corner Expansion

## Summary
Created orbiting gaussian world type for chemotaxis training, restructured linear_chemotaxis experiments to use proper folder hierarchy, added corner expansion experiment, and created video generation tooling.

## Orbiting Gaussian World Type

### Concept
- Base uniform resource (10) with a Gaussian blob (max 30, sigma=48) that orbits around arena center
- Orbit radius: 192 pixels, period: 5000 steps
- Simpler than temporal_gaussian - single predictable moving target for agents to track

### Speed Analysis
- Initial period 2000 was too fast for agents to track while eating
- Blob speed at 2000: ~0.6 px/step
- Agent effective speed with move-eat pattern: ~0.5 px/step
- Increased period to 5000: blob speed ~0.24 px/step, agents can track comfortably

### Files Created
- `src/worlds/orbiting_gaussian.py` - World type with `update_resource()` for time evolution
- `configs/world/orbiting_gaussian_simple.yaml`
- `configs/run/from_pretrain_orbiting.yaml` (200k steps)
- `scripts/run_from_pretrain_orbiting.sh`

### Code Changes
- `src/worlds/__init__.py`: Registered orbiting_gaussian
- `src/simulation_simple.py`: Added `is_orbiting_gaussian` flag and update call

## Linear Chemotaxis Restructure

### Problem
Previous structure had `data/linear_chemotaxis_v1/` as single output dir instead of folder hosting multiple runs (like `data/diffusion_analysis_v1/`).

### New Structure
```
data/linear_chemotaxis_v1/
├── from_pretrain_simple/
├── from_temporal/
└── from_orbiting/
```

### Files Created
- `configs/world/linear_chemotaxis_v1/default.yaml` - Shared world config
- `configs/run/linear_chemotaxis_v1/from_pretrain_simple.yaml`
- `configs/run/linear_chemotaxis_v1/from_temporal.yaml`
- `configs/run/linear_chemotaxis_v1/from_orbiting.yaml`
- `scripts/linear_chemotaxis_v1/from_pretrain_simple.sh`
- `scripts/linear_chemotaxis_v1/from_temporal.sh`
- `scripts/linear_chemotaxis_v1/from_orbiting.sh`
- `scripts/linear_chemotaxis_v1/run_all.sh`

### Spawn Region Fix
- Discovered spawn_region format is `[y_min, y_max, x_min, x_max]` not `[x_min, x_max, y_min, y_max]`
- Changed to `[64, 192, 127, 129]` for vertical line at center x

### Checkpoint Updates
- from_temporal: ckpt_050001 → ckpt_200001 (after 200k run completed)
- from_orbiting: ckpt_032500 → ckpt_200001 (after 200k run completed)

## Corner Expansion Experiment

### Setup
- Uniform resource 20, no regeneration (timescale=1e6)
- Agents spawn in 16x16 box at corner [0, 16, 0, 16]
- Reproduction enabled
- 256 agents from run_from_pretrain_simple

### Files Created
- `configs/world/corner_expansion_simple.yaml`
- `configs/run/corner_expansion.yaml` (10k steps, checkpoint every 50)
- `scripts/run_corner_expansion.sh`

### Video Generation
- `scratch/corner_expansion_video/make_video.py` - Generates video from checkpoints
- Uses matplotlib for frames, ffmpeg for video encoding
- Output: `scratch/corner_expansion_video/corner_expansion.mp4`

## Analysis Scripts

### Linear Chemotaxis Comparison
- `scratch/linear_chemotaxis_v1_analysis/plot_avg_x.py`
- Compares avg x position over time for all three training sources
- Outputs comparison plot to `scratch/linear_chemotaxis_v1_analysis/output/comparison.png`

## Code Fixes

### X-Density Plotting
- Added x_density heatmap plotting to `src/scripts/run_simple.py`
- Previously only y_density was plotted despite x_density being tracked

## Documentation Updates
- Added to research_context.md: JAX memory fraction note for running two experiments in parallel
