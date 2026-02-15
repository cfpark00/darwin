# evolvability_v1 Track Progress

## Current State: Full darwin_v0 Feature Parity + Diffusion Experiments

The simulation framework has full darwin_v0 feature parity including toxin, attack, temporal environments, and controlled "neuro" experiments.

## What Works

- **Unified Agent**: Single implementation with configurable I/O (6, 7, or 8 dimensions depending on capabilities)
- **Environment Abstraction**: Rich environment registry with many types:
  - SimpleEnvironment, GaussianEnvironment, UniformEnvironment
  - FullEnvironment (with toxin/attack), GaussianToxinEnvironment
  - PretrainEnvironment (food decay curriculum)
  - BridgeEnvironment, TemporalGaussianEnvironment, OrbitingGaussianEnvironment
  - ToxinPatternEnvironment, CyclingEnvironment
- **Capability Flags**: `has_toxin`, `has_attack` on environments control I/O dimensions
- **Toxin Detection**: Configurable circular detection radius (default 5 cells, like darwin_v0)
- **Handshake Validation**: Agent/Environment I/O mismatch raises ValueError at init
- **Simulation Engine**: Full step() with reproduction, mutation, death, resource dynamics
- **Run Script**: Logging, checkpointing, arena screenshots, figure generation
- **Performance**: 1.7x faster than darwin_v0, proper JAX async dispatch
- **JIT Metrics**: All stats computed inside JIT using masked reductions (no fancy indexing)
- **Dynamic Buffer Growth**: Automatic buffer expansion at configurable threshold (default 50%)
- **Physics-based Reproduction**: No artificial cap (was 64), now limited only by dead slots and empty cells
- **Unphysical Event Tracking**: `repro_capped` metric warns when reproductions fail due to buffer limits
- **Controlled Experiments**: energy_clamp, resource_clamp, disabled_actions for "neuro" experiments
- **Checkpoint Loading**: `--from-checkpoint` with optional `spawn_region` for diffusion experiments

## What Doesn't Work Yet

- No lineage selection mechanism yet (needed for Barnett-style evolvability)
- Transfer learning scripts not yet fully implemented

## Recent Changes (2026-01-18)

### Controlled Experiment Features
- Added `energy_clamp`, `resource_clamp`, `disabled_actions` to PhysicsConfig
- Supports "immortal" agents with clamped energy, disabled reproduction
- Config options: `energy.clamp`, `resource.clamp`, `disabled_actions: [5]`

### Diffusion Experiments
- Added `spawn_region` support for checkpoint loading
- Agents loaded from checkpoint can be placed in a specified rectangular region
- Created test_clamped_diffusion experiments (resource_10, resource_25)
- Created diffusion analysis script (`scratch/diffusion_analysis_v1/`)

### Diffusion Results
| Condition | Resource | D (diffusion) | R² |
|-----------|----------|---------------|-----|
| resource_10 | 10 | 0.0038 | 0.95 |
| resource_25 | 25 | 0.0135 | 0.91 |

Higher resources → 3.5x more diffusion (opposite of expected). Agents move more when food is abundant.

### Performance Fixes (Earlier)
- Fixed 650x logging slowdown by moving metrics computation inside JIT
- Replaced fancy boolean indexing with masked reductions
- Added `Metrics` NamedTuple in types.py for pre-computed stats

### Physics Fixes (Earlier)
- Removed max_K=64 reproduction cap → now max_K=buffer_size
- Buffer growth inside step() at 50% threshold (configurable)
- Track `repro_capped` for unphysical reproduction failures
- Clamp sensor noise to physical ranges (food >= 0, temp/toxin in [0,1])

### Visualization
- Added arena screenshots (`plot_arena()`) at `detailed_interval`

## Benchmark Results (Updated)

| Test | Speed |
|------|-------|
| Pure step (no sync) | 1689 it/s |
| Sync every step (old get_stats) | 2.6 it/s |
| Sync every step (new get_stats) | 252.9 it/s |
| Sync every 10 steps | 1111 it/s |
| darwin_v0 comparison | 975 it/s |

**evolvability_v1 is 1.7x faster than darwin_v0** with proper logging.

## Configuration Options (New)

```yaml
buffer_growth_threshold: 0.5  # Grow buffer when pop > 50% capacity
```

At threshold T, up to (1-T) fraction of population can reproduce safely per step.

## Immediate Next Steps

1. Investigate why higher resources lead to more movement (analyze action distributions)
2. Consider running diffusion with untrained agents for comparison
3. Design first evolvability experiment (lineage selection, Barnett-style)
4. Implement transfer learning scripts

## Documentation

- `/docs/computational_compromises.md` - Shared physics approximations
- `/docs/tracks/evolvability_v1/computational_compromises.md` - v1-specific choices
