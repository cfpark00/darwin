# Session Log: evolvability_v1 Performance and Physics Fixes

**Date**: 2026-01-18 00:29

## Summary

Major performance optimizations and physics correctness improvements to evolvability_v1, including fixing a 650x logging slowdown and removing artificial reproduction caps.

## Tasks Completed

### Performance Fixes
- **Fixed catastrophic logging slowdown** (650x): `get_stats()` was using fancy boolean indexing (`energies[alive]`) which created dynamic shapes, breaking JAX async dispatch
- **Implemented JIT-compiled metrics**: All metrics now computed inside JIT using masked reductions (`sum(energies * alive) / (sum(alive) + eps)`)
- **Added `Metrics` NamedTuple** to `types.py` for pre-computed metrics in state
- **Benchmarked results**:
  - Old per-step sync: 2.6 it/s
  - New per-step sync: 252.9 it/s (97x faster)
  - With log_interval=10: 1111 it/s (66% of pure step speed)

### Physics Correctness Fixes
- **Removed arbitrary max_K=64 reproduction cap**: Changed to `max_K = max_agents` (buffer size), so reproduction limit is now purely physics-based (dead slots, empty cells)
- **Added dynamic buffer growth inside `step()`**: Grows at configurable threshold (default 50%)
- **Added `buffer_growth_threshold` config parameter** with detailed comment explaining when things become unphysical
- **Track unphysical events**: New `repro_capped` metric counts reproductions that failed due to lack of dead slots
- **Fixed sensor noise clamping**: Sensor readings now clamped to physical ranges (food >= 0, temp/toxin in [0,1])

### Visualization
- **Added arena screenshots to evolvability_v1**: `plot_arena()` function saves step-wise environment snapshots at `detailed_interval`

### Documentation
- Created `/docs/computational_compromises.md` - shared compromises in both frameworks
- Created `/docs/tracks/evolvability_v1/computational_compromises.md` - evolvability_v1 specific choices
- Removed empty `/configs/evolvability_v1/world/` directory
- Fixed benchmark script path issue and added auto-generated results.md

## Files Modified/Created

### evolvability_v1 Core
- `src/evolvability_v1/types.py` - Added `Metrics` NamedTuple, `repro_capped` field, `buffer_growth_threshold` config
- `src/evolvability_v1/simulation.py` - JIT metrics, buffer growth, `grow_buffer()` method
- `src/evolvability_v1/environment.py` - Clamped sensor noise to physical ranges
- `src/evolvability_v1/scripts/run.py` - Updated logging, arena screenshots, warning for unphysical events
- `src/evolvability_v1/__init__.py` - Export `Metrics`

### Documentation
- `docs/computational_compromises.md` (new)
- `docs/tracks/evolvability_v1/computational_compromises.md` (new)

### Benchmarks
- `scratch/benchmark_v0_v1/benchmark_v0_vs_v1.py` - Fixed path, added auto results.md generation
- `scratch/benchmark_v0_v1/results.md` (new)

## Key Decisions

1. **Masked reductions over fancy indexing**: `sum(x * mask) / (sum(mask) + eps)` keeps static shapes for JIT
2. **Metrics in state**: Pre-compute inside JIT, sync only when logging
3. **Buffer growth at 50%**: At threshold T, (1-T) fraction can reproduce safely
4. **Track but don't crash on unphysical**: `repro_capped` warns but simulation continues

## Key Insight

JAX's performance model: fancy indexing (`arr[bool_mask]`) creates dynamic shapes that prevent async dispatch and force Python-level execution. Masked reductions maintain static shapes and enable full JIT fusion. The 650x slowdown was not from "sync overhead" per se, but from breaking JAX's execution model.

## Next Steps

- Consider adding `repro_capped` to history tracking for post-hoc analysis
- May want to add arena visualization as video output option
- darwin_v0 has same fancy indexing issue in get_stats() (just hidden by less frequent logging)
