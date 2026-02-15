# Darwin Research Context

## Goal
Simulating evolution to observe emergent multi-agent phenomena, including:
- Theory of mind
- Emergent strategies
- Complex agent behaviors

## System Overview

### Simulation
A 2D grid world where agents with neural network "brains" live, compete for resources, reproduce, and evolve through mutation.

### Agent Architecture
- **Brain**: 2-layer LSTM network (hidden_dim=8)
- **Full version** (`agent.py`):
  - Inputs (7): Food, temperature, toxin detection, 4 contact sensors
  - Outputs (7): eat, forward, left, right, stay, reproduce, attack
- **Simple version** (`agent_simple.py`): For pretrain/thermotaxis environments
  - Inputs (6): Food, temperature, 4 contact sensors (no toxin)
  - Outputs (6): eat, forward, left, right, stay, reproduce (no attack)
- **Evolution**: Offspring inherit parent's neural weights + Gaussian mutation (std=0.1)
- **Lineage Tracking**: Each agent has unique ID (`uid`) and parent reference (`parent_uid`, -1 for founders)
  - Enables ground truth genealogy reconstruction
  - Parallel-safe UID assignment via cumsum trick
  - Available in both `simulation.py` and `simulation_simple.py`

### World Types
| Type | Description |
|------|-------------|
| gaussian | Random resource/toxin/temperature fields via Gaussian processes |
| bridge | Two fertile strips (edges) connected by narrow resource-poor bridge |
| maze | 8x8 cell maze with toxic walls, start at bottom-left, goal at top-right |
| pretrain | Food curriculum (30->12 over 10k steps), static temperature, no reproduction penalty |
| thermotaxis | Dynamic temperature simulating soil (sinusoidal at surface, decays+lags with depth), food gradient with noise |
| uniform | Constant resource level everywhere (for controlled experiments like diffusion) |
| linear_gradient | Linear resource gradient along x-axis (for chemotaxis testing) |
| temporal_gaussian | Time-varying Gaussian via Fourier phase rotation (for dynamic chemotaxis training) |
| orbiting_gaussian | High-resource blob orbits arena center (simpler tracking target) |
| orbiting_gaussian_harsh | Harsh version with base_resource=3, blob_max=40 (forces blob dependence) |

### Actions
| Action | Cost | Effect |
|--------|------|--------|
| Eat | 0.5 | Consume 33% of resource at current cell |
| Forward | 3.0 | Move one cell in facing direction |
| Left/Right | 3.0 | Turn 90 degrees |
| Stay | 0.5 | Do nothing (baseline metabolism) |
| Reproduce | 30.0 | Create offspring (gets 25 energy) in adjacent empty cell |
| Attack | 5.0 | Kill agent in front, their energy drops to ground |

### Energy System
- **Initial**: 100
- **Base cost**: 0.5 per step (fixed metabolism)
- **Energy penalty**: `base_cost_incremental * energy` (default 0.01)
  - At energy 100: +1.0 cost to all actions
  - At energy 50: +0.5 cost
  - Creates pressure against hoarding without hard limit
- **Age penalty** (thermotaxis only): `base_cost_age_incremental * age` (default 0.001)
  - At age 1000: +1.0 cost to all actions
  - Simulates physiological aging
- **Temperature penalty** (thermotaxis only): `base_cost_temperature_incremental * |temp - 0.5|`
  - Optimal temperature: 0.5
  - At temp extremes (0 or 1): +0.375 cost
- **Total cost**: `base_cost + energy_penalty + age_penalty + temp_penalty + action_cost`
- **Death**: Energy <= 0

### Running Experiments
- **JAX memory**: Typically use `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` to run two experiments in parallel on a single GPU

### Experiment Groups
**CRITICAL: ALL experiments MUST be organized into groups within tracks. Never place config/script files directly in the track's run/ or scripts/ directory.**

A **group** is a collection of related experiments that share a common research question or experimental design. Groups are organized consistently across configs, scripts, and data:

```
configs/<track>/run/<group_name>/     # YAML config files for each experiment
configs/<track>/world/<group_name>/   # World configs (if needed)
scripts/<track>/<group_name>/         # Bash scripts to run experiments
data/<track>/<group_name>/            # Output data from all experiments in the group
```

**Examples:**
| Group | Purpose | Structure |
|-------|---------|-----------|
| `linear_chemotaxis_v1` | Compare chemotaxis from different training sources | 3 experiments: `from_pretrain_simple`, `from_temporal`, `from_orbiting` |
| `diffusion_v1` | Grid search over energy × resource levels | 24 experiments: `diffusion_e{15,30,100}_r{5,10,15,20,25,30,35,40}` |

**Benefits:**
- Related experiments stay together in version control and file system
- Easy to run all experiments in a group via `scripts/<group>/run_all.sh`
- Analysis scripts can iterate over `data/<group>/*/` to aggregate results

## Current State (2026-01-18)

### Track Structure
Codebase organized into track structure:

**darwin_v0** - Original simulation engine (legacy):
- `src/darwin_v0/` - Python source code
- `configs/darwin_v0/` - YAML configs (run/ and world/)
- `scripts/darwin_v0/` - Bash execution scripts
- `data/darwin_v0/` - Experiment outputs

**evolvability_v1** - Redesigned framework (active):
- `src/evolvability_v1/` - Unified engine with Environment/Agent abstraction
- `configs/evolvability_v1/` - YAML configs organized by groups
- `scripts/evolvability_v1/` - Bash execution scripts
- `data/evolvability_v1/` - Experiment outputs
- `docs/tracks/evolvability_v1/` - Track documentation

Cross-track utilities remain at `src/utils.py`.

### evolvability_v1 Framework
New simulation framework with full darwin_v0 feature parity:
- **Unified Agent**: Single implementation with configurable I/O dimensions (6, 7, or 8 depending on capabilities)
- **Environment Abstraction**: Rich registry with many environment types (see below)
- **Capability Flags**: `has_toxin`, `has_attack` on environments control I/O dimensions
- **1.7x faster** than darwin_v0 with equivalent settings
- **Clean composition**: Environment passed at init, no runtime world-type branching
- **JIT-compiled metrics**: All stats computed inside JIT using masked reductions
- **Physics-based reproduction**: No artificial cap, limited only by dead slots and empty cells
- **Dynamic buffer growth**: Automatic buffer expansion at configurable threshold (default 50%)
- **Controlled experiments**: `energy_clamp`, `resource_clamp`, `disabled_actions` for "neuro" experiments
- **Checkpoint loading**: `--from-checkpoint` with optional `spawn_region` for diffusion experiments

Current environments: SimpleEnvironment, GaussianEnvironment, UniformEnvironment, FullEnvironment, GaussianToxinEnvironment, PretrainEnvironment, BridgeEnvironment, TemporalGaussianEnvironment, OrbitingGaussianEnvironment, ToxinPatternEnvironment, CyclingEnvironment

### Recent Diffusion Experiment (2026-01-18)
Tested resource-dependent diffusion in evolvability_v1:
- 256 agents spawned in 16x16 centered region, energy clamped at 30, reproduction disabled
- Compared resource=10 (low) vs resource=25 (high)
- **Finding**: Higher resources → 3.5x higher diffusion (D=0.0135 vs D=0.0038)
- This is opposite to darwin_v0 finding where lower resources led to more movement
- Possible explanation: pretrained agents learned to eat when food is detected, requiring more movement when food is everywhere

### Completed (as of 2026-01-07)
- Core simulation with collision-free movement and reproduction
- Modular world system (`src/worlds/`) for different arena types
- Transfer experiment infrastructure (load evolved agents into new environments)
- Visualization: population, action ratios, energy histograms, attack stats, toxin deaths
- Soft energy cap via metabolic penalty
- Position-based agent filtering for transfer experiments
- Maze environment with toxic walls and goal-reaching selection
- Dynamic maze with moving goal (relocates when population threshold exceeded)
- Hidden dim variant (hidden_dim=24 for ~7,351 params vs ~1,055 for h=8)
- Pretrain environment with food curriculum for basic survival training
- Thermotaxis environment with dynamic temperature (inspired by Ramot et al. 2008)
- Temperature-based reproduction probability system
- Y-density plot for thermotaxis runs
- Simple simulation infrastructure (`agent_simple.py`, `simulation_simple.py`) for pretrain/thermotaxis
- Parametrized energy cost system (base + incremental + action costs)
- Aging system for thermotaxis (metabolic cost scales with age)
- Temperature-dependent metabolism (optimal at temp=0.5)
- Corrected temperature model (mean gradient from 0.5 at surface to 0.25 at bottom)
- Y-density heatmap visualization (time vs y-position)
- Git repository initialized and pushed to github.com/cfpark00/darwin
- Weight expansion for simple→full agent transfer (`expand_simple_to_full_params`)
- Fail-fast validation in transfer scripts (param dimension mismatch now errors immediately)
- Uniform world type for controlled experiments (constant resource level)
- Diffusion experiment infrastructure with action masking and energy clamp
- Action masking feature (`disabled_actions` config) to prevent agents from selecting certain actions
- Energy clamp feature (`energy.clamp` config) for immortal agents in controlled experiments
- Comprehensive diffusion experiment (3 energy levels × 8 resource levels = 24 conditions)
- Energy max 500 experiment (completed run with video)
- Dynamic regeneration infrastructure (`continue_no_regen.py`) for decaying resource scenarios
- Diffusion v2 suite (fast regen timescale=10 vs v1 timescale=100)
- Single agent observation experiment for detailed trajectory analysis
- Linear gradient world type for chemotaxis experiments
- Temporal gaussian world type with Fourier phase rotation for time-varying resource fields
- X-density tracking (alongside existing y-density) for spatial distribution analysis
- Orbiting gaussian world type for simplified chemotaxis training (single predictable blob)
- Corner expansion experiment infrastructure (agents spread from corner, no regen)
- Restructured linear_chemotaxis_v1 to compare multiple training sources
- Video generation tooling for checkpoint visualization and step-by-step replay (`scratch/*/make_video.py`, `resume_video.py`)
- Ground truth lineage tracking (uid, parent_uid) in both simulation.py and simulation_simple.py
- Competition experiment infrastructure (`transfer_competition.py`) for head-to-head cluster comparisons

### Infrastructure Ready
- **Default world**: 512x512 Gaussian, runs at ~115 steps/s
- **Bridge world**: 256x256, two fertile strips + connecting bridge
- **Maze world**: 256x256 (8x8 cells), toxic walls, resource gradient (start→passable→goal)
- **Dynamic maze**: Goal moves through passable cells when >100 agents occupy it
- **Pretrain world**: 256x256, food curriculum 30→12 over 10k steps
- **Thermotaxis world**: 512x512, dynamic temperature with phase lag, food gradient with noise
- **Transfer experiments**: Load checkpoint agents into new environment
- **Explorer selection**: Iterative selection of bridge-crossing agents (5 iterations configured)
- **Solver selection**: Iterative selection of maze-solving agents (2 iterations configured)
- **Thermotaxis pipeline**: Pretrain → thermotaxis transfer (simple→simple)
- **Pretrain to full pipeline**: Pretrain → default world transfer (simple→full with weight expansion)
- **Uniform world**: Constant resource level, no spatial variation (for controlled experiments)
- **Diffusion experiment**: 24 conditions (3 energy × 8 resource levels), action-masked, energy-clamped, 256 agents
- **E500 world**: Default Gaussian with energy max 500 (vs default 100)
- **E500 decay regen**: Continue E500 with dynamically increasing regen timescale (+10% every 100 steps)
- **Linear gradient world**: Resource gradient (0 at left, 30 at right) for chemotaxis testing
- **Temporal gaussian world**: Time-varying Gaussian via Fourier phase rotation (ω_k ∝ 1/|k| - large structures evolve slowly)
- **Single agent world**: Uniform arena (r=10), single immortal agent for trajectory analysis
- **Orbiting gaussian world**: Base resource 10, blob (max 30, sigma 48) orbits at radius 192, period 5000 steps
- **Corner expansion**: Uniform r=20, no regen, reproduction enabled, agents spawn in corner
- **Linear chemotaxis v1**: Compares chemotaxis from 3 training sources (pretrain_simple, temporal, orbiting)

### Current Experiment: Explorer Selection
Iteratively select agents that successfully cross the bridge to breed better explorers:
1. `bridge` - Transfer evolved agents from default world to bridge
2. `bridge_explorers` - Select agents from right side at step 2750
3. `bridge_explorers2` - Select from explorers at step 1750
4. `bridge_explorers3` - Select from explorers2 at step 1500
5. `bridge_explorers4` - Select from explorers3 at step 2000

Each iteration spawns 100 agents on left side, filters those who reach right fertile strip.

### Current Experiment: Maze Solver Selection
Iteratively select agents that navigate the maze to reach the goal:
1. `maze` - Transfer 100 agents from default world to maze start cell
2. `maze_solvers` - Select agents from goal cell at step 2500, restart from start
3. `maze_solvers2` - Select from solvers at step 1750, restart from start

Resource levels: start=5 (pressure to leave), passable=18 (sustainable), goal=50 (fertile).

### Dynamic Maze Experiment
Tests adaptability by moving the goal when agents congregate:
- Threshold: >100 agents in goal cell triggers relocation
- Goal cycles through all passable cells in the maze
- Tracks goal movement history and population dynamics

### Thermotaxis Experiment
Tests temperature-seeking behavior inspired by C. elegans thermotaxis (Ramot et al. 2008):

**Pretrain Phase** (10k steps):
- 256x256 grid, 512 agents
- Food starts at 30, decays to 12 (curriculum learning)
- Static Gaussian temperature, no reproduction penalty
- Goal: agents learn basic survival (eating, moving)

**Thermotaxis Phase** (50k steps):
- 512x512 grid, 512 agents transferred from pretrain
- Food: linear gradient (0 at bottom, 20 at top) + Gaussian noise (amplitude=8)
  - Noise prevents agents from using food as y-coordinate proxy
- Temperature: dynamic sinusoidal with depth-dependent decay and phase lag
  - Formula: `T(y,t) = mean(y) + 0.5 * exp(-z/zd) * sin(2*pi*t/period - z/zd)`
  - Mean temperature: 0.5 at surface (y=511), 0.25 at bottom (y=0)
  - Period: 1000 steps, damping depth: 128 pixels
  - Range: 0.0 (surface cold phase) to 1.0 (surface hot phase)
- Reproduction: 100% success at temp<=0.5, linear decay to 0% at temp=1.0
- Aging: metabolic cost increases linearly with age (0.001 per step alive)
- Temperature metabolism: optimal at temp=0.5, penalty for deviation
- No toxin, attack does nothing

**Research question**: Can agents learn to track optimal temperature zones that shift with the thermal wave while balancing food availability (top) vs thermal comfort (varies)?

### Linear Chemotaxis Experiment (Completed)
Tested whether pretrained agents can learn to follow a static resource gradient.

**Setup**:
- 256 agents from pretrain checkpoint, centered in arena
- Linear gradient: 0 at left edge, 30 at right edge
- Energy clamped to 30, no reproduction
- 5000 steps

**Finding**: Agents did NOT learn chemotaxis. Average x-position remained near center throughout.

**Conclusion**: Static gradients allow fixed policies - agents don't need to sense gradients if the optimal location never changes. Temporal dynamics are required to force gradient-following behavior.

### Temporal Gaussian Experiment (Completed - 200k steps)
Addresses the static gradient limitation by using time-varying resource fields.

**Approach**: Fourier phase rotation
- Generate Gaussian field in Fourier space
- Each mode k rotates at rate ω_k ∝ 1/|k|
- Large structures (low k) evolve slowly, small features (high k) change faster
- Resources relax toward moving target via existing regeneration mechanic

**Parameters**:
- 512×512 arena, length_scale=600, base_omega=0.0002
- Transfer 512 agents from pretrain, 200k steps

**Hypothesis**: Dynamic resource fields will force agents to sense and follow gradients, as the optimal location continuously shifts.

### Orbiting Gaussian Experiment (Completed - 200k steps)
Simpler alternative to temporal gaussian with a single predictable moving target.

**Setup**:
- Base uniform resource: 10
- High-resource blob: max 30, sigma 48 pixels
- Orbit: radius 192, period 5000 steps (blob speed ~0.24 px/step)
- Transfer 512 agents from pretrain, 200k steps

**Speed calculation**: Agents need to eat while tracking. With move-eat pattern, effective agent speed ~0.5 px/step. Blob at 0.24 px/step is trackable.

**Finding**: Agents did NOT learn to track the blob. Estimated lifespan ~45 steps (very short). See "Agent Lifespan and r-Selection Analysis" below.

### Harsh Orbiting Experiment (In Progress)
Tests whether reducing base resources forces blob-tracking behavior.

**Setup**:
- Base resource: 3 (was 10) - agents LOSE energy away from blob
- Blob max: 40 (was 30)
- Energy economics: -0.62/step away from blob, +8.42/step at blob center

**Config**: `configs/run/from_pretrain_orbiting_harsh.yaml`

**Results at 50k steps**:
- Lifespan: ~33 steps (vs 45 in original) - SHORTER, not longer
- Population: ~1500 (vs ~2500 in original)
- Distribution: Still Type III survivorship (most die young)

**Conclusion**: Harsh environment alone doesn't evolve blob-tracking. Agents just die faster.

### Linear Chemotaxis v1 (In Progress)
Compares chemotaxis ability of agents from different training environments.

**Setup**:
- 256 agents spawned as vertical line at center (x=127-128, y=64-192)
- Linear gradient: 0 at left, 30 at right
- Energy clamped to 30, reproduction disabled
- 5000 steps

**Training sources compared**:
1. `from_pretrain_simple`: Basic survival training only
2. `from_temporal`: 200k steps in temporal gaussian (complex dynamic field)
3. `from_orbiting`: 200k steps tracking orbiting blob (simple predictable target)

**Hypothesis**: Orbiting-trained agents may show better chemotaxis since they learned to track a coherent moving target, while temporal-trained agents experienced more chaotic dynamics.

### Corner Expansion Experiment
Tests population expansion dynamics from a localized starting point.

**Setup**:
- 256 agents from pretrain_simple, spawned in 16×16 corner box
- Uniform resource: 20, no regeneration (timescale 1e6)
- Reproduction enabled
- 10k steps

### Genotype Cluster Behavioral Divergence (Completed)
Investigated whether genetically distinct clusters of evolved agents exhibit different behaviors.

**Background**:
- After 50k steps of evolution in `run_from_pretrain_simple`, population diverged into distinct genotype clusters
- PCA on weight matrices: PC1 explains 53.6% of variance
- HDBSCAN identified 36 clusters in PCA space

**Target Clusters**:
| Cluster | N agents | PC1 | Description |
|---------|----------|-----|-------------|
| 5 | 1324 | +38.2 | Largest, right side of PC1 |
| 17 | 1040 | -34.7 | 2nd largest, left side of PC1 |

**Experiments**:
1. Diffusion: 256 agents, uniform resource r=20, energy clamped, 1000 steps
2. Chemotaxis: 256 agents, linear gradient (0→30), energy clamped, 2000 steps

**Key Finding - Behavioral Divergence Confirmed**:

| Cluster | Diffusion Coef (D) | Strategy |
|---------|-------------------|----------|
| 5 | 0.23 | "Circler" - 6.2% left turns, spins in place |
| 17 | 0.81 | "Explorer" - 21.8% forward, goes straight |

**Cluster 17 moves 3.5x more than Cluster 5** due to different turn behaviors, not movement frequency.

**Mechanism**: Both clusters eat similarly (~77%), but:
- Cluster 5: High left-turn rate → circular motion → low net displacement
- Cluster 17: Almost no turns, mostly forward → efficient linear exploration

**Chemotaxis**: Neither cluster followed the gradient, but cluster 17 spread wider (consistent with higher diffusion).

**Conclusion**: Genetic divergence leads to qualitative behavioral divergence. The two main lineages evolved distinct movement strategies.

**Infrastructure created**:
- `src/scripts/transfer_by_cluster.py` - Transfer script filtering agents by cluster ID
- `data/reference/genotype_clusters_v1/` - Static reference data for reproducibility
- `docs/concrete_questions/genotype_cluster_behavioral_divergence.md` - Research question documentation

### Genotype Cluster Competition V2 (Completed)
Parallel investigation to V1, exploring reproduction and competition dynamics instead of diffusion/chemotaxis.

**Research Question**: Same as V1 - Can we detect qualitative behavioral differences between genetically distinct clusters?

**Experiments**:

1. **Isolated Reproduction** (5000 steps each):
| Cluster | Initial | Final | Growth |
|---------|---------|-------|--------|
| 5 | 256 | 5,864 | 22.9× |
| 17 | 256 | 5,343 | 20.9× |

**Finding**: Similar reproduction rates in isolation (C5 slightly better)

2. **Head-to-Head Competition** (5000 steps, 256 from each cluster mixed):
| Metric | Cluster 5 | Cluster 17 |
|--------|-----------|------------|
| Initial | 256 (50%) | 256 (50%) |
| Step 100 | 435 (28%) | 1,134 (72%) |
| Step 500 | 348 (10%) | 3,183 (90%) |
| Final | **343 (8.4%)** | **3,726 (91.6%)** |

**Key Finding**: Cluster 17 dominates in direct competition despite similar isolated reproduction!

**Mechanism - Allopatric Speciation**:
- Visualized original positions in source checkpoint
- Cluster 5 centroid: y=96.8, x=328.1 (bottom-right region)
- Cluster 17 centroid: y=428.2, x=140.8 (top-left region)
- Distance: 381 pixels apart (in 512×512 arena)
- **Conclusion**: Clusters evolved in geographically separate regions, developing different competitive strategies

**V1 vs V2 Comparison**:
| Version | Approach | Finding |
|---------|----------|---------|
| V1 | Diffusion/Chemotaxis | C17 diffuses 3.5× more (explorer vs circler) |
| V2 | Reproduction/Competition | C17 dominates competition 91.6% vs 8.4% |

**Interpretation**: C17's "explorer" behavior translates to competitive advantage - more movement leads to finding resources first.

**Infrastructure created**:
- `src/scripts/transfer_competition.py` - Load agents from two clusters, track lineage by genetic distance
- `configs/run/divergent_genotype_v2/` and `configs/world/divergent_genotype_v2/`
- `scripts/divergent_genotype_v2/` - Bash scripts for all experiments
- `data/divergent_genotype_v2/RESULTS_SUMMARY.md` - Detailed findings

### Genotype Cluster Action Profiles V3 (Completed)
Measured action distributions for clusters 5 and 17 under different resource conditions.

**Setup**:
- 256 agents per experiment, energy clamped (immortal), reproduction disabled
- High resource: r=30, Low resource: r=5
- 2000 steps each

**Key Finding - Turning Chirality**:
| Cluster | Left turns | Right turns |
|---------|------------|-------------|
| 5 | 2.4-4.3% | **0%** |
| 17 | **0%** | 0.9-1.0% |

**Cluster 5 turns LEFT exclusively, Cluster 17 turns RIGHT exclusively!**

**Other findings**:
- C17 moves more than C5 at both resource levels (~3-5% higher)
- Both clusters double movement when resources are scarce (r=30 → r=5)

**Infrastructure**: `configs/run/divergent_genotype_v3/`, `scripts/divergent_genotype_v3/`, `data/divergent_genotype_v3/`

### Genotype Phylogenetic Analysis V4 (Completed)
Pure genotype analysis - hierarchical clustering on HDBSCAN cluster centroids to reconstruct evolutionary tree.

**Key Finding - Deep Evolutionary Split**:
C5 and C17 are in **different clades at ALL granularities** tested (2-10 superclusters).

At coarsest level (2 superclusters):
- **Clade A**: 13 clusters, 3,485 agents (contains C5)
- **Clade B**: 23 clusters, 3,764 agents (contains C17)

**Spatial Segregation**:
The two major clades occupy different regions of the arena - evidence of **allopatric speciation**.

**Diversification Hotspot**:
Identified concentration of phylogenetic split points near (300, 150):
- 2.2× higher population density
- 1.5× higher resource variance (heterogeneity)
This matches the concept of "diversification hotspot" in evolutionary biology.

**Outputs**:
- `data/divergent_genotype_v4/figures/phylogenetic_clusters.png` - Dendrogram with split ordering
- `data/divergent_genotype_v4/figures/spatial_2_clades.png` ... `spatial_8_clades.png` - Progressive clade splits
- `data/divergent_genotype_v4/figures/spatial_split_points.png` - All 35 split locations on arena

### Diffusion Experiment (Completed)
Controlled "neuroscience-style" experiment testing resource-dependent movement behavior.

**Hypothesis**: Lower resource availability → higher dispersion/diffusion as agents search for food.

**Setup**:
- 256 agents from run_from_pretrain_simple checkpoint, spawned on 16×16 centered grid
- 256×256 uniform arena (constant resources everywhere)
- Reproduction disabled via action mask (`disabled_actions: [5]`)
- Energy clamped to fixed value (immortal agents, no survivorship bias)
- 8 resource levels: 5, 10, 15, 20, 25, 30, 35, 40
- 3 energy levels: 15, 30, 100
- 1k steps each, checkpoints every 100 steps

**Results** (Diffusion coefficient D from MSD = 4Dt):

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

**Key Findings**:
1. **Hypothesis confirmed**: Lower resources → higher diffusion (D drops ~3× from r5 to r40)
2. **Energy effect**: At low resources, lower-energy agents move more ("hungry" behavior)
3. **Convergence**: At high resources (r35-r40), all energy levels converge to similar D (~0.56-0.64)
4. **Non-monotonic**: E=30 peaks at r10 (D=1.50), higher than both E=15 and E=100

## Key Parameters

### Default World
- Size: 512x512
- Initial agents: 512
- Resource: max=30, mean=15, regen_timescale=100
- Toxin coverage: 5%

### Bridge World
- Size: 256x256
- Fertile strips: width=32, resource=25
- Bridge: height=16, resource=12
- Barren: resource=5

### Maze World
- Size: 256x256 (8 cells × 32 pixels)
- Layout: 8x8 cell maze, walls are toxic (instant death)
- Start cell (bottom-left): resource=5 (net +1.15/eat)
- Passable cells: resource=18 (net +5.44/eat, ~66% movement sustainable)
- Goal cell (top-right): resource=50 (net +16/eat, rapid reproduction)

### Pretrain World
- Size: 256x256
- Initial agents: 512
- Food: uniform, starts at 30, decays to 12 over 10k steps
- Temperature: static Gaussian field
- Reproduction: no temperature penalty (threshold >= max temp)

### Thermotaxis World
- Size: 512x512
- Initial agents: 512 (transferred from pretrain)
- Food: gradient (0 at bottom, 20 at top) + Gaussian noise (amplitude=8, length_scale=30)
- Temperature: dynamic sinusoidal (period=1000, damping_depth=128)
  - Mean: 0.5 at surface, 0.25 at bottom
- Reproduction: temp-dependent (100% at <=0.5, 0% at 1.0)
- Aging: base_cost_age_incremental=0.001 (cost += age * 0.001)
- Temperature metabolism: base_cost_temperature_incremental=0.75 (cost += |temp-0.5| * 0.75)
- No toxin, attack disabled

### Energy Max 500 Experiment (Completed)
Testing whether higher energy cap leads to different behavioral strategies:
- **Hypothesis**: Higher energy storage (500 vs 100) may lead to more attack behaviors, as agents can afford the attack cost and benefit from stealing stored energy
- **Setup**: Default Gaussian world with `energy.max: 500`
- **Config**: `configs/run/default_e500.yaml`
- **Output**: `data/run_default_e500/` (video: `figures/simulation.mp4`)

### E500 Decay Regeneration Experiment
Continue E500 simulation with dynamically decaying resource regeneration:
- **Setup**: Continue from E500 step 50000, regen timescale starts at 100 and increases 10% every 100 steps
- **Hypothesis**: Gradually depleting resources will create selection pressure for more efficient foraging or competitive behaviors
- **Config**: `configs/run/e500_decay_regen.yaml`
- **Script**: `src/scripts/continue_no_regen.py` (disables built-in regen, applies manual dynamic regen)
- **Output**: `data/run_e500_decay_regen/`
- After 50k steps: timescale grows from 100 → ~13,780 (resources recover ~138× slower)

## Open Questions
- What strategies will emerge in homogeneous vs structured environments?
- Will agents learn to cross the bridge to access isolated resources?
- Can evolved agents adapt when transferred to new environment types?
- Will cooperative or competitive behaviors emerge?
- Does higher energy cap (500 vs 100) promote attack behaviors due to increased resource hoarding potential?
- Can agents learn to navigate the maze and avoid toxic walls?
- How does network capacity (hidden_dim) affect learning complex navigation?
- Will agents track moving goals in the dynamic maze environment?
- Can agents learn to track optimal temperature zones as thermal waves propagate through the environment?
- How do populations respond to gradually depleting resources? Will attack behaviors increase as competition intensifies?
- Can time-varying resource fields (temporal gaussian) teach agents to follow resource gradients?
- Does tracking a simple orbiting blob teach better chemotaxis than complex temporal fields?
- How do populations expand from a corner with limited resources?
- Can forced exploration (higher action_temperature, epsilon-greedy, minimum turn probability) prevent the turning capability from atrophying and enable gradient-following evolution?

## Answered Questions
- **Do evolved agents show resource-dependent diffusion?** YES - Diffusion coefficient D drops ~3× from r5 to r40. Agents move more when resources are scarce. Energy level also matters: "hungrier" agents (lower energy) tend to move more at low resource levels.
- **Can agents learn chemotaxis from static gradients?** NO - Static gradients allow fixed policies; agents don't need to sense gradients if optimal location never changes. Temporal dynamics required.
- **Do genetically distinct clusters have different behaviors?** YES - Clusters 5 and 17 evolved different movement strategies (circler vs explorer). V1 found 3.5× diffusion difference, V2 found 91.6% vs 8.4% competition dominance. V3 found turning chirality (C5=left only, C17=right only). Geographic separation (allopatric speciation) was the likely driver.
- **Does genetic phylogeny map onto spatial territory?** YES - Hierarchical clustering on cluster centroids (V4) shows C5 and C17 are in different clades at all granularities. The two major evolutionary lineages occupy different regions of the arena. A diversification hotspot was identified near (300, 150) with higher population density and resource heterogeneity.
- **Why don't long-lived agents evolve?** r-selection dominates when base resources are sufficient. With base_resource=10 in orbiting, agents gain ~1.1 energy/step even away from blob. No selection pressure for blob-tracking or longevity. Fast reproduction (more generations = more mutations) wins over longevity.
- **Can agents use LSTM memory for temporal gradient sensing?** NOT YET - While LSTMs theoretically enable temporal gradient sensing (compare current vs remembered food), agents have evolved to eliminate turning (0% left/right actions). Without turning to sample different directions, temporal sensing is useless. Turning existed in pretrain (1.2%) but was selected out in orbiting environments because turns cost energy with no immediate benefit.

## Agent Lifespan and r-Selection Analysis

### Lifespan Measurements
| Experiment | Est. Lifespan | Checkpoint Overlap |
|------------|---------------|-------------------|
| Orbiting (base=10) | 45 steps | 0% |
| Temporal gaussian | 130 steps | 0.1% |
| Harsh orbiting (base=3) | 33 steps | 0% |

### Type III Survivorship Curve
Age distributions show classic r-selection pattern (like fish/insects):
- Exponential decay: most agents die young
- Long tail: ~10% survive past 500 steps
- Median age ~20-35 steps across experiments
- Max ages reach 3000-6000 steps (rare survivors)

### Why r-Selection Dominates
1. **Resources everywhere**: Base resource sufficient for survival without tracking
2. **Reproduction nearly neutral**: Cost 29.5 energy, offspring gets 25
3. **Generation time matters**: More generations = more mutation = faster adaptation
4. **No longevity benefit**: Long life doesn't lead to more offspring

### The Turning Problem
| Stage | Left | Right | Total Turns |
|-------|------|-------|-------------|
| Pretrain 10k | 0.6% | 0.6% | 1.2% |
| Orbiting 35k | 0.0% | 0.1% | 0.1% |
| Harsh 50k | 0.0% | 0.0% | 0.0% |

**Critical finding**: Evolution eliminated turning because:
- Turns cost energy (2.5) with no immediate benefit
- Without turning, agents can't do gradient following
- LSTM memory becomes useless if agents never explore directions
- This is a local optimum trap - turning capability atrophied

## Theoretical Analysis
Movement constraints and resource thresholds:
- **Max distance without eating**: ~29 cells (starting from E=100, move cost=2.5, base cost=0.5)
- **Critical resource for infinite travel**: R ≈ 17 (move-eat pattern sustainable above this)
- **Eating crossover**: Below R≈3, better to sprint than stop to eat
- **At R=10**: Eating extends range from 29 → 120 cells (4× improvement)

## Future Directions (Explored)
- **Vision system**: Ray-traced vision with directional perception and occlusion was explored as a potential enhancement. See `docs/maybe_vision.md` for design discussion. Key considerations: ray tracing is computationally cheaper than pairwise operations (O(N×R×D) vs O(N²)), and would require new agent architecture (CNN+LSTM or larger LSTM).
- **MuJoCo MJX**: JAX-accelerated physics engine as alternative to grid-based simulation. Would provide built-in vision/sensors and richer physics, but requires full system redesign.
