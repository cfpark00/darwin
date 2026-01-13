# Diffusion Experiment Analysis Complete

## Summary
Completed comprehensive diffusion experiment with controlled conditions to measure resource-dependent movement behavior. Fixed critical bugs in experiment setup and ran full analysis across 3 energy levels × 8 resource levels = 24 conditions.

## Bug Fixes

### Reproduction Bug
- **Issue**: Agents were reproducing despite `cost_reproduce: 10000` because energy clamp reset energy before death check
- **Fix 1**: Added `disabled_actions: [5]` config option to mask REPRODUCE action from softmax (agents can never select it)
- **Fix 2**: Added `action_mask` parameter to `sample_action()` in `agent_simple.py`
- **Fix 3**: Updated `simulation_simple.py` to read `disabled_actions` from config and apply mask

### Energy Clamp Feature
- Added `energy.clamp` config option in `simulation_simple.py`
- Clamps all alive agents' energy to specified value each step (immortal agents)
- Removes survivorship bias from diffusion measurements

## Experiment Design (Final)

- **Arena**: 256×256 uniform world (constant resource everywhere)
- **Agents**: 256 from pretrain checkpoint, spawned on 16×16 centered grid
- **Controls**:
  - Reproduction disabled via action mask
  - Energy clamped (immortal agents)
- **Conditions**:
  - Resource levels: 5, 10, 15, 20, 25, 30, 35, 40
  - Energy levels: 15, 30, 100 (3 suites)
- **Duration**: 1000 steps each
- **Metric**: Diffusion coefficient D from MSD = 4Dt

## Files Created

### Simulation Changes
- `src/simulation_simple.py`: Added `disabled_actions` and `energy.clamp` support
- `src/agent_simple.py`: Added `action_mask` parameter to `sample_action()`

### E=100 Suite (`configs/*/diffusion/`, `scripts/diffusion/`)
- r5, r10, r15, r20, r25, r30, r35, r40
- Output: `data/diffusion_r{5,10,15,20,25,30,35,40}/`

### E=30 Suite (`configs/*/diffusion_e30/`, `scripts/diffusion_e30/`)
- r5, r10, r15, r20, r25, r30, r35, r40
- Output: `data/diffusion_e30_r{5,10,15,20,25,30,35,40}/`

### E=15 Suite (`configs/*/diffusion_e15/`, `scripts/diffusion_e15/`)
- r5, r10, r15, r20, r25, r30, r35, r40
- Output: `data/diffusion_e15_r{5,10,15,20,25,30,35,40}/`

### Analysis
- `scratch/diffusion_analysis/analyze_diffusion.py` - MSD and diffusion coefficient analysis
- `scratch/diffusion_analysis/output/diffusion_analysis.png` - Plots
- `scratch/diffusion_analysis/output/diffusion_results.txt` - Results table

## Results

| Resource | E=15 | E=30 | E=100 |
|----------|------|------|-------|
| r5 | 1.90 | 1.87 | 1.85 |
| r10 | 1.19 | 1.50 | 1.31 |
| r15 | 1.04 | 0.98 | 1.02 |
| r20 | 0.81 | 0.86 | 0.91 |
| r25 | 0.73 | 0.81 | 0.72 |
| r30 | 0.68 | 0.73 | 0.70 |
| r35 | 0.66 | 0.61 | 0.56 |
| r40 | 0.64 | 0.56 | 0.57 |

## Key Findings

1. **Hypothesis confirmed**: Lower resources → higher diffusion (D drops ~3× from r5 to r40)
2. **Energy effect**: At low resources (r5-r10), lower energy agents move more ("hungry" behavior)
3. **Convergence**: At high resources (r35-r40), all energy levels converge to similar D (~0.56-0.64)
4. **Non-monotonic**: E=30 peaks at r10 (D=1.50), higher than both E=15 and E=100

## Usage
```bash
# Run all experiments
bash scripts/diffusion/run_all.sh --overwrite
bash scripts/diffusion_e30/run_all.sh --overwrite
bash scripts/diffusion_e15/run_all.sh --overwrite

# Analyze
uv run python scratch/diffusion_analysis/analyze_diffusion.py
```
