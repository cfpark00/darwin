# Evolvability V1 Track Notes

## Research Focus

Computational simulation demonstrating the **evolution of evolvability** - inspired by Barnett et al. 2025 (biorxiv).

**Key question**: Can we design a simulation where the *capacity to generate adaptive variation* itself evolves through lineage-level selection?

---

## Lessons from darwin_v0

### What Worked Well

1. **JAX-accelerated physics**: Grid-based collision detection, vectorized operations, JIT compilation
2. **Flat genome representation**: Efficient for mutation, crossover, and parallel evolution
3. **World registry pattern**: Pluggable world types (gaussian, bridge, maze, etc.)
4. **Lineage tracking**: uid/parent_uid enables ground-truth genealogy reconstruction

### What Accumulated as Patches

| Patch | Purpose | Added For |
|-------|---------|-----------|
| `energy_clamp` | Immortal agents | Diffusion experiments |
| `resource_clamp` | Infinite resources | Controlled experiments |
| `disabled_actions` | Action masking | Disable reproduction |
| `age_penalty` | Age-based metabolism | Thermotaxis |
| `temp_penalty` | Temperature metabolism | Thermotaxis |
| Dynamic world branching | World updates in step() | Thermotaxis, orbiting blob |

**Pattern**: Each new experiment added config flags with silent defaults. Features were bolted-on rather than designed-in.

### Architectural Debt

1. **Two parallel codebases**: `simulation.py` vs `simulation_simple.py` (~1,200 lines duplicated)
   - Only real difference: 7 vs 6 I/O dimensions (toxin + attack)
   - Should be: Configurable I/O, single codebase

2. **Script duplication**: 9 scripts with 97%+ overlap (~1,600 lines duplicated)
   - `run.py`, `run_simple.py`, `transfer.py`, `transfer_simple.py`, etc.
   - Logging, checkpointing, visualization all repeated

3. **Python-level branching in JIT path**:
   ```python
   if self.is_thermotaxis:
       world = thermotaxis.update_temperature(world, step)
   elif self.is_pretrain:
       world = pretrain.update_food(world, step)
   ```
   - Prevents JIT optimization, requires code changes for new world types

4. **State dict as catch-all**: No schema, optional fields, type checking absent

### Key Research Findings

1. **Turning atrophied**: Evolution eliminated turning (0% left/right) because turns cost energy with no immediate benefit
2. **r-Selection dominance**: Short lifespans (~33-130 steps), high reproduction rate
3. **Allopatric speciation**: Geographic separation drove behavioral divergence
4. **Static gradients don't teach chemotaxis**: Temporal dynamics required

---

## What Evolvability Requires (from Barnett)

### The Experiment

Barnett et al. evolved *Pseudomonas fluorescens* bacteria under a regime that:
1. **Cycling environment**: Alternates between favoring cel+ (mat-forming) and cel- (dispersing)
2. **Lineage-level selection**: Lineages that fail to transition go extinct
3. **Bottleneck**: Single-colony transfer forces mutations to occur de novo
4. **Lineage reproduction**: Successful lineages can "reproduce" by replacing extinct ones

### The Outcome

1. **Global hypermutation** (meta-pop A, C): Mutations in DNA repair genes → 300× higher mutation rate everywhere
2. **Local hypermutation** (meta-pop B): A heptanucleotide repeat expanded → 10,000× higher mutation rate at one locus only
   - The locus controlled phenotype switching
   - More repeats = faster switching = better lineage survival
   - **The mutation rate at a specific locus became evolvable**

### Key Insight

> "Mutations providing immediate benefit (phenotype switch) also increase future mutation rate."

The evolved "contingency locus" is:
- Immediately adaptive (causes required phenotype)
- Self-amplifying (more repeats → higher mutation rate → more expansion)
- Reversible (can contract as well as expand)

---

## Proposed Abstraction for evolvability_v1

### Core Insight: Separate Concerns

darwin_v0 conflated:
- **Physics** (movement, eating, collision)
- **Selection** (energy-based death)
- **Genotype structure** (fixed flat params)
- **Phenotype** (implicit in behavior)

For evolvability, we need clean separation:

### 1. Agent (Unified, Configurable)

```python
@dataclass
class AgentConfig:
    input_dim: int      # Observation size (6 for simple, 7 for full)
    output_dim: int     # Action count
    hidden_dim: int     # LSTM hidden size
    internal_noise_dim: int
```

**Single implementation**, parameterized by config. No more agent.py vs agent_simple.py.

### 2. Genotype (Evolvable Structure)

```python
@dataclass
class Genotype:
    params: jax.Array           # Neural network weights
    mutation_rates: jax.Array   # Per-weight or per-region mutation rates (evolvable!)
    # Future: contingency_regions, crossover_points, etc.
```

**Key change**: Mutation structure is part of the heritable information, not a fixed global parameter.

### 3. Phenotype (Explicit Readout)

```python
class PhenotypeClassifier:
    def classify(self, agent_state, behavior_history) -> int:
        """Classify agent into discrete phenotype (0, 1, ...)"""
        ...
```

In Barnett: phenotype = cel+ or cel- (observable colony morphology).
In simulation: phenotype could be derived from:
- Action distribution (e.g., movement vs stationary)
- Behavioral pattern (e.g., explorer vs circler)
- Network output statistics

### 4. Selection Regime (Pluggable)

```python
class SelectionRegime(Protocol):
    def step(self, state, world) -> tuple[alive_mask, reproduction_mask]:
        """Determine survival and reproduction eligibility."""
        ...

class IndividualSelection(SelectionRegime):
    """Classic energy-based selection (darwin_v0 style)."""
    def step(self, state, world):
        alive = state.energies > 0
        can_reproduce = alive & (state.actions == REPRODUCE)
        return alive, can_reproduce

class LineageSelection(SelectionRegime):
    """Barnett-style lineage selection."""
    def __init__(self, target_phenotype_fn, transition_period: int):
        ...

    def step(self, state, world):
        # At transition boundaries:
        # 1. Check which lineages expressed target phenotype
        # 2. Kill lineages that failed
        # 3. Allow successful lineages to "reproduce" (replace extinct)
        ...
```

### 5. Environment (Cycling Support)

```python
class Environment(Protocol):
    def update(self, world: dict, step: int) -> dict:
        """Update world state."""
        ...

    def get_selection_context(self, step: int) -> dict:
        """Return context for selection (e.g., target phenotype)."""
        ...

class CyclingEnvironment(Environment):
    """Alternates between two selective contexts."""
    def __init__(self, period: int, context_a: dict, context_b: dict):
        self.period = period
        self.contexts = [context_a, context_b]

    def get_selection_context(self, step: int) -> dict:
        phase = (step // self.period) % 2
        return self.contexts[phase]
```

### 6. Simulation (Composed)

```python
class Simulation:
    def __init__(
        self,
        agent_config: AgentConfig,
        environment: Environment,
        selection: SelectionRegime,
        physics_config: dict,
    ):
        ...

    def step(self, state: SimState, key: jax.Array) -> SimState:
        # 1. Environment update
        world = self.environment.update(state.world, state.step)
        context = self.environment.get_selection_context(state.step)

        # 2. Observation and decision (physics)
        observations = self.observe(state, world)
        actions = self.decide(state, observations, key)

        # 3. Execute physics (movement, eating)
        state = self.physics.execute(state, actions, world)

        # 4. Apply selection regime
        alive, can_repro = self.selection.step(state, context)
        state = state._replace(alive=alive)

        # 5. Reproduction with evolvable mutation
        state = self.reproduce(state, can_repro, key)

        return state
```

---

## Key Differences from darwin_v0

| Aspect | darwin_v0 | evolvability_v1 |
|--------|-----------|-----------------|
| **Agent** | Two versions (full/simple) | One configurable |
| **Genotype** | Fixed flat params | Includes mutation structure |
| **Mutation** | Fixed global std=0.1 | Per-region rates (evolvable) |
| **Phenotype** | Implicit (behavior patterns) | Explicit classifier |
| **Selection** | Individual (energy death) | Pluggable (individual or lineage) |
| **Environment** | Static or dynamic (patched) | Unified with cycling support |
| **World updates** | Python if/elif | Composed at init time |

---

## Implementation Roadmap

### Phase 1: Core Framework
1. Unified Agent with configurable I/O
2. Genotype dataclass with mutation_rates field
3. SimState with typed schema (not dict catch-all)
4. Basic physics (reuse from darwin_v0)

### Phase 2: Selection Regimes
1. IndividualSelection (darwin_v0 style, for validation)
2. LineageSelection (Barnett style)
3. Bottleneck mechanism (single-founder restart)

### Phase 3: Evolvable Mutation
1. Per-weight mutation rates in genotype
2. Mutation of mutation rates (meta-mutation)
3. Tracking mutation rate evolution over time

### Phase 4: Cycling Environment
1. CyclingEnvironment with configurable period
2. Phenotype classifier (action-based or network-based)
3. Transition detection and lineage evaluation

### Phase 5: Experiments
1. Validate with individual selection (should match darwin_v0)
2. Run Barnett-style lineage selection
3. Measure evolution of mutation structure
4. Compare global vs local hypermutation outcomes

---

## Open Questions

1. **Phenotype definition**: What's the neural network analog of cel+/cel-?
   - Could be: action distribution, network output norm, behavioral pattern
   - Needs to be learnable (agents can mutate to express either phenotype)

2. **Transition mechanism**: How do we force de novo mutations?
   - Barnett used single-colony bottleneck
   - We could use: lineage restart from single agent, or require phenotype via fresh mutation

3. **Mutation granularity**: Per-weight? Per-layer? Per-region?
   - Barnett: per-locus (specific gene region)
   - Could group weights by layer, input/output, or arbitrary regions

4. **What counts as "lineage success"?**
   - Barnett: produced at least one colony of target phenotype
   - We could use: any descendant expressed target phenotype within N steps

---

## References

- Paper: `resources/barnett_2025/barnett_2025.pdf`
- Markdown: `resources/barnett_2025/md/barnett_2025.md`
- Link: https://www.biorxiv.org/content/10.1101/2024.05.01.592015v4
