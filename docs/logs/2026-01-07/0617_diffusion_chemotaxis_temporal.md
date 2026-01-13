# Diffusion v2, Chemotaxis, and Temporal Gaussian Fields

## Summary
Extended diffusion experiments, added chemotaxis experiments with linear gradients, and created time-varying resource fields for improved gradient-following learning.

## Diffusion Experiments

### v1 Analysis
- Ran analysis on existing diffusion_analysis_v1 data (24 conditions: 3 energy levels × 8 resource levels)
- Confirmed hypothesis: lower resources → higher diffusion (D drops ~3× from r5 to r40)
- All R² > 0.97 indicating excellent linear MSD fits

### v2 Setup (Fast Regen)
- Created diffusion_v2 suite with `regen_timescale: 10` (vs 100 in v1)
- Resources regenerate 10× faster, making environment more "forgiving"
- Initially tried resource.clamp but reverted to fast regen as more physical

### Code Changes
- Added `resource.clamp` option to `simulation_simple.py` (skips depletion when True)
- Fixed `transfer_simple.py` to copy world_config instead of dumping redundant run_config.yaml
- Updated all diffusion configs to point to `data/diffusion_analysis_v1/` and `data/diffusion_analysis_v2/`

## Single Agent Experiment
- Created single agent observation setup for detailed trajectory analysis
- Config: uniform arena, r=10, regen_timescale=100, energy clamped to 100, no reproduction
- Analysis script plots trajectory and MSD
- Showed D ≈ 0.09 for single agent in r=10 environment

## Theoretical Analysis
Calculated critical resource level for sustainable movement:
- Max distance without eating: ~29 cells
- Critical resource for infinite travel: R ≈ 17 (move-eat pattern sustainable above this)
- Below R≈3: better to sprint than eat
- At R=10: eating extends range from 29 → 120 cells (4× improvement)

## Linear Chemotaxis v1
- **Goal**: Test if agents follow resource gradient
- **Setup**: Linear gradient (0 at left, 30 at right), 256 agents centered, no reproduction, energy clamped to 30
- **New world type**: `src/worlds/linear_gradient.py`
- **Finding**: Agents did NOT learn chemotaxis - likely because static gradient allows fixed policies

## Temporal Gaussian World Type
- **Motivation**: Static gradients don't force gradient sensing; temporal dynamics required
- **Approach**: Fourier phase rotation - evolve Gaussian field modes over time
- **Key insight**: ω_k ∝ 1/|k| makes large structures evolve slowly, small features change faster
- **Implementation**:
  - `src/worlds/temporal_gaussian.py` - stores Fourier coefficients, evolves each step
  - `update_resource()` computes new `resource_base` via phase rotation + iFFT
  - Resources relax toward moving target via existing regeneration mechanic

### Demo
- Created `scratch/temporal_changing_field_v1/demo.py` with video output
- Parameters: 512×512, length_scale=600, base_omega=0.0002 (very slow evolution)
- Video shows smooth evolution of resource field

## Files Created

### New World Types
- `src/worlds/linear_gradient.py` - Linear resource gradient along x-axis
- `src/worlds/temporal_gaussian.py` - Time-varying Gaussian via Fourier phase rotation

### Configs
- `configs/world/diffusion_v2/*.yaml`, `configs/world/diffusion_v2_e15/*.yaml`, `configs/world/diffusion_v2_e30/*.yaml`
- `configs/run/diffusion_v2/*.yaml`, etc.
- `configs/world/single_agent.yaml`, `configs/run/single_agent.yaml`
- `configs/world/linear_chemotaxis_v1/default.yaml`, `configs/run/linear_chemotaxis_v1/default.yaml`
- `configs/world/temporal_gaussian_simple.yaml`, `configs/run/from_pretrain_temporal.yaml`

### Scripts
- `scripts/diffusion_v2/run_all.sh`, `scripts/diffusion_v2/run_remaining.sh`
- `scripts/run_single_agent.sh`
- `scripts/linear_chemotaxis_v1/run.sh`
- `scripts/run_from_pretrain_temporal.sh`

### Analysis Scripts (scratch/)
- `scratch/diffusion_analysis/analyze_diffusion_v2.py`
- `scratch/single_agent_analysis/plot_trajectory.py`
- `scratch/linear_chemotaxis_analysis_v1/plot_avg_x.py`
- `scratch/temporal_changing_field_v1/demo.py`

## Code Changes
- `src/simulation_simple.py`:
  - Added `resource.clamp` option
  - Added `is_temporal_gaussian` flag and update call
  - Added x_density tracking alongside y_density
- `src/worlds/__init__.py`: Registered linear_gradient and temporal_gaussian

## Next Steps
- Run temporal gaussian experiment to see if agents learn chemotaxis with dynamic fields
- May need to tune base_omega for appropriate timescale relative to agent movement speed
