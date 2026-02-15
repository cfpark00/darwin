# evolvability_v1 Additional Computational Compromises

Approximations and design choices specific to evolvability_v1, beyond the shared compromises in `/docs/computational_compromises.md`.

## Metrics Computation

### Metrics Inside JIT (Masked Reductions)
- **What**: Compute `mean_energy = sum(energies * alive) / (sum(alive) + eps)` instead of `mean(energies[alive])`
- **Ideal**: Exact statistics over living agents only
- **Why**: Fancy boolean indexing (`arr[mask]`) creates dynamic shapes, breaking JIT and causing 100x+ slowdown
- **Impact**: Mathematically equivalent, but:
  - Dead agents contribute 0 to numerator (correct)
  - Epsilon (1e-8) causes tiny error when population near zero
  - `max_age` uses `where(alive, ages, 0)` which returns 0 for extinct population instead of undefined

### Pre-computed Action Counts
- **What**: Action distribution computed as `[count_action(i) for i in range(output_dim)]` inside JIT
- **Ideal**: Compute only when needed
- **Why**: Computing inside JIT avoids sync; list comprehension unrolled at trace time
- **Impact**: Always pays cost of computing all action stats even if not logged; 6-7 extra reductions per step.

## State Management

### NamedTuple State (Immutable)
- **What**: `SimState` is a NamedTuple, new state created each step via `_replace()`
- **Ideal**: Mutable state updated in-place
- **Why**: JAX functional paradigm; NamedTuples are pytree-compatible
- **Impact**: Memory allocation every step (though JAX may optimize); can't have circular references.

### Metrics in State
- **What**: `state.metrics` carries JAX arrays that persist until next step
- **Ideal**: Compute metrics on-demand
- **Why**: Pre-computation inside JIT avoids fancy indexing at query time
- **Impact**: State object larger; metrics from step N overwritten at step N+1 (no history in state).

### Optional Fields Default to None
- **What**: `actions: Optional[jax.Array] = None`, `metrics: Optional[Metrics] = None`
- **Ideal**: Always populated or separate types for pre/post-step state
- **Why**: Initial state before first step has no actions/metrics
- **Impact**: Need null checks; type hints less precise.

## Environment Abstraction

### Environment Update Outside JIT
- **What**: `env.update_world(world, step)` called in Python before JIT step
- **Ideal**: World updates inside JIT for full fusion
- **Why**: Allows Python-level environment logic (phase changes, config lookups)
- **Impact**: One Python call per step; can't fuse world update with agent computation. For static environments (Gaussian, Uniform), this is a no-op so no real cost.

### Observation via vmap over Python Method
- **What**: `vmap(lambda i, k: env.compute_observation(...))(arange(n), keys)`
- **Ideal**: Fully JAX observation function
- **Why**: Allows environment-specific observation logic without code duplication
- **Impact**: Method dispatch overhead; environment object captured in closure (traced once, then fast).

### Action Mask as Optional Array
- **What**: `env.get_action_mask()` returns `None` or array
- **Ideal**: Always return mask (all-True if no masking)
- **Why**: Avoid allocating mask when not needed
- **Impact**: Conditional in sample_action; slightly more complex logic.

## Handshake Validation

### I/O Dimension Check at Init
- **What**: `AgentConfig.input_dim == Environment.input_dim` validated once at construction
- **Ideal**: Static type checking at compile time
- **Why**: Python has no dependent types; runtime check catches mismatches early
- **Impact**: Error only at runtime, not caught by type checker; one-time cost.

## Reproduction

### Reproduction Success Not in Metrics
- **What**: We track action counts but not how many reproductions actually succeeded
- **Ideal**: Track births, deaths, failed reproductions separately
- **Why**: Would need additional counters inside JIT, more return values
- **Impact**: Can't distinguish "wanted to reproduce but couldn't" from "didn't try".

### Parent UID Tracking Overhead
- **What**: Every agent carries `uid` and `parent_uid` arrays, updated on reproduction
- **Ideal**: Optional lineage tracking
- **Why**: Useful for phylogenetic analysis; cost is ~2 int32 per agent
- **Impact**: Small memory overhead; cumsum for UID assignment adds ~1% compute.

## World Representation

### World as Dict (Not NamedTuple)
- **What**: `state.world = {"resource": ..., "temperature": ..., ...}`
- **Ideal**: Typed NamedTuple like SimState
- **Why**: Environments may add arbitrary fields (toxin, etc.); dict is flexible
- **Impact**: No type checking on world fields; key typos fail at runtime; dict overhead.

### Resource Base Stored Separately
- **What**: `world["resource"]` (current) and `world["resource_base"]` (target for regen)
- **Ideal**: Single array with regen computed from initial state
- **Why**: Allows dynamic base changes; regen formula needs both
- **Impact**: 2x memory for resource field; must keep in sync.

## JIT Compilation

### Static max_agents Argument
- **What**: `@partial(jax.jit, static_argnums=(14,))` for max_agents
- **Ideal**: Fully dynamic agent count
- **Why**: Loop bounds and array sizes need static values for JIT
- **Impact**: Recompilation if max_agents changes; can't resize mid-run.

### Config Values in Closure
- **What**: `_build_step_fn()` captures config values like `max_energy`, `base_cost` in closure
- **Ideal**: Pass config as argument
- **Why**: Avoids passing many static arguments; cleaner JIT signature
- **Impact**: Recompilation if config changes (which shouldn't happen mid-run anyway).

### Action Cost Array Pre-built
- **What**: `self.action_costs = jnp.array([cost_eat, cost_move, ...])` built once
- **Ideal**: Compute from config inside JIT
- **Why**: Avoid repeated array construction; enables `action_costs[actions]` indexing
- **Impact**: Assumes action indices stable (they are by design).

## Comparison with darwin_v0

| Aspect | darwin_v0 | evolvability_v1 | Trade-off |
|--------|-----------|-----------------|-----------|
| State type | dict | NamedTuple | Type safety vs flexibility |
| Metrics | Computed on-demand (slow) | Pre-computed in JIT (fast) | Memory vs speed |
| World updates | Python if/elif branches | Environment.update_world() | Coupling vs abstraction |
| Agent variants | Separate files (agent.py, agent_simple.py) | Single configurable agent | Code duplication vs complexity |
| Simulation variants | Separate files (simulation.py, simulation_simple.py) | Single simulation + environments | Code duplication vs abstraction |

## Performance Implications

These choices result in:
- **~1.7x faster** than darwin_v0 for equivalent simulation
- **~100x faster logging** (252 it/s vs 2.6 it/s per-step sync)
- **~5% overhead** for metrics computation inside JIT (acceptable)
- **Negligible overhead** from NamedTuple vs dict state
