# Computational Compromises

Approximations and non-ideal choices made for computational efficiency in both darwin_v0 and evolvability_v1.

## Memory & Allocation

### Fixed Agent Buffer
- **What**: Pre-allocate fixed-size arrays (e.g., 1024 agents max)
- **Ideal**: Dynamic population with no upper bound
- **Why**: JAX requires static shapes for JIT compilation. Dynamic arrays would require recompilation or Python-level loops.
- **Impact**: Population can't exceed buffer size; wastes memory when population is small.

### Dead Slot Reuse
- **What**: Dead agents leave "holes" in arrays, reused for offspring
- **Ideal**: Compact arrays with only living agents
- **Why**: Removing elements would change array shapes, breaking JIT
- **Impact**: Operations iterate over all slots including dead ones; `alive` mask filtering everywhere.

## Physics & Movement

### Discrete Grid World
- **What**: Agents occupy integer (y, x) cells on a grid
- **Ideal**: Continuous 2D positions with smooth movement
- **Why**: Grid enables O(1) collision lookup via occupancy array. Continuous would need spatial hashing or O(N²) pairwise checks.
- **Impact**: Movement is jerky; diagonal movement impossible; minimum distance = 1 cell.

### Synchronous Updates
- **What**: All agents observe → all decide → all act → all effects resolve
- **Ideal**: Asynchronous updates where agents act in random/priority order
- **Why**: Vectorized operations require all agents processed in parallel
- **Impact**: Creates artificial simultaneity; two agents can "swap" positions; no true first-mover advantage.

### Conflict Resolution by Random Tiebreak
- **What**: When multiple agents try to move to same cell, random winner
- **Ideal**: Physics-based collision (momentum, mass, pushing)
- **Why**: Deterministic priority would need sorting; physics sim too expensive
- **Impact**: Movement has random element; dense populations become chaotic.

### Instantaneous Actions
- **What**: Move/eat/reproduce complete in one step
- **Ideal**: Actions take time (animation, cooldowns)
- **Why**: Multi-step actions require state machines per agent, complex tracking
- **Impact**: No commitment to actions; can't interrupt or react mid-action.

## Reproduction

### Max Offspring Per Step (max_K=64)
- **What**: At most 64 reproduction events per step
- **Ideal**: All valid reproductions succeed
- **Why**: Need static loop bounds for JIT; prevents population explosions that exceed buffer
- **Impact**: In dense, fertile populations, some reproductions silently fail.

### Offspring Position Search
- **What**: Check 4 adjacent cells (front, back, left, right) for empty space
- **Ideal**: Search larger radius or optimal placement
- **Why**: Fixed 4 candidates keeps computation bounded
- **Impact**: Reproduction fails in crowded areas even if empty cells exist diagonally or 2 cells away.

### No Gestation/Development
- **What**: Offspring appear instantly with full capabilities
- **Ideal**: Gestation period, juvenile phase, maturation
- **Why**: Would require age-dependent state and capabilities
- **Impact**: No parental investment trade-offs; population dynamics unrealistic.

## Sensing & Perception

### Local-Only Sensing
- **What**: Agents sense only their current cell (food, temp) + 4 adjacent contacts
- **Ideal**: Vision cone, hearing radius, chemical gradients
- **Why**: Non-local sensing requires ray-casting or spatial queries
- **Impact**: Agents can't see food/threats at distance; no pursuit/evasion behavior.

### Binary Contact Sensors
- **What**: Contact = 1 if neighbor present, 0 otherwise
- **Ideal**: Continuous pressure/distance sensing, identity of neighbor
- **Why**: Richer sensing needs more input dimensions, more compute
- **Impact**: Can't distinguish friend from foe by touch; no force sensing.

### Noisy Sensors (Fixed Noise Model)
- **What**: Gaussian noise added to food/temp readings
- **Ideal**: Sensor noise dependent on conditions, signal strength
- **Why**: Simple additive noise is cheap
- **Impact**: Noise doesn't scale with signal; unrealistic at extremes.

## Neural Architecture

### Fixed Network Topology
- **What**: 2-layer LSTM with fixed hidden_dim, can't add/remove neurons
- **Ideal**: NEAT-style topology evolution (add nodes, connections)
- **Why**: Variable topology = variable param count = can't vectorize across agents
- **Impact**: Evolution limited to weight optimization; can't discover novel architectures.

### Flat Parameter Vector
- **What**: All weights flattened to 1D array, uniform mutation
- **Ideal**: Structured genotype with modularity, gene regulation
- **Why**: Flat arrays are simple to mutate and crossover
- **Impact**: No hierarchical organization; mutations affect all traits equally.

### No Crossover
- **What**: Offspring = mutated clone of single parent
- **Ideal**: Sexual reproduction with recombination
- **Why**: Crossover between neural networks is problematic (alignment issues)
- **Impact**: No genetic mixing; slower exploration of weight space.

## Resources & Environment

### Exponential Regeneration
- **What**: Resources regenerate as `r += (r_base - r) / timescale`
- **Ideal**: Ecological dynamics (growth, competition, seasons)
- **Why**: Simple first-order dynamics, no additional state
- **Impact**: Resources always return to baseline; no depletion, no ecosystem collapse.

### Static World Geometry
- **What**: World size fixed at initialization
- **Ideal**: Expandable world, dynamic boundaries
- **Why**: Resizing arrays breaks JIT; would need recompilation
- **Impact**: Can't simulate range expansion, habitat fragmentation.

### No Resource Flow
- **What**: Each cell independent, no diffusion between cells
- **Ideal**: Resources spread (water flow, nutrient diffusion)
- **Why**: Diffusion requires neighbor updates, iterative solving
- **Impact**: Sharp resource boundaries; no gradient smoothing.

## Time & Scheduling

### Fixed Timestep
- **What**: All steps equal duration, all agents same update rate
- **Ideal**: Variable timesteps, faster agents act more often
- **Why**: Vectorization requires uniform timing
- **Impact**: No metabolic rate differences; slow/fast agents meaningless.

### Deterministic Step Order
- **What**: Within a step: observe → decide → move → reproduce → eat → die → regen
- **Ideal**: Order could vary, be agent-specific
- **Why**: Fixed order enables clean vectorized phases
- **Impact**: Exploitation of phase order (e.g., eat after move always).

## Numerical

### Float32 Throughout
- **What**: All computations in 32-bit float
- **Ideal**: Float64 for sensitive calculations, mixed precision
- **Why**: GPU optimized for float32; half the memory
- **Impact**: Accumulation errors in long runs; energy accounting may drift.

### Epsilon Guards
- **What**: Division by `(n + 1e-8)` to avoid div-by-zero
- **Ideal**: Proper handling of edge cases
- **Why**: Branchless computation for JIT
- **Impact**: Tiny numerical errors when n≈0; metrics slightly off for extinct populations.
