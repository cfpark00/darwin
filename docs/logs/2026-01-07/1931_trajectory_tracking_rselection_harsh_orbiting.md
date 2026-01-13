# Trajectory Tracking, r-Selection, and Harsh Orbiting Experiment

## Summary
Investigated why linear_chemotaxis_v1 showed negative results. Discovered agents have very short lifespans (~45 steps), preventing trajectory tracking across checkpoints. Analyzed why long-lived agents haven't evolved (r-selection dominates). Created harsh orbiting experiment to force blob-tracking, but found agents evolved to eliminate turning entirely.

## Trajectory Tracking Investigation

### Initial Goal
Track individual agents across checkpoints to see if any follow the orbiting blob in the `run_from_pretrain_orbiting` experiment.

### Discovery: UID Tracking Missing in Old Checkpoints
- Old checkpoints (before recent code changes) don't have `uid` field
- Lineage tracking (uid, parent_uid, next_uid) was added recently
- `run_from_pretrain_orbiting` was re-run with current code to get uid tracking

### Agent Lifespan Analysis
Estimated lifespan from birth rate between consecutive checkpoints:

| Experiment | Est. Lifespan | Checkpoint Overlap |
|------------|---------------|-------------------|
| Orbiting | **45 steps** | 0% |
| Temporal | **130 steps** | 0.1% |

With 2500-step checkpoint intervals and 45-step lifespan, almost no agents survive between checkpoints. Can't track trajectories this way.

## Why Don't Long-Lived Agents Evolve?

### Energy Economics Analysis
Original orbiting config: `base_resource=10`, `blob_max=30`

Calculated per-step energy:
- Away from blob: **+1.09 energy/step** (agents GAIN energy)
- At blob center: higher gain

**Problem**: Agents can thrive on base resource alone. No selection pressure to track blob.

### r-Selection Dominates
- Fast reproduction beats longevity when resources are everywhere
- Reproduction is nearly neutral (cost 29.5, offspring gets 25)
- More generations = more mutations = faster adaptation
- Long life gives no advantage when food is everywhere

### Age Distribution
Created age pyramid plots showing Type III survivorship curve (like fish):
- Massive early mortality
- Few survivors to old age
- Distribution shape stable over time
- ~10% survive past 500 steps in temporal

## Harsh Orbiting Experiment

### Design
Created `configs/world/orbiting_gaussian_harsh.yaml`:
- `base_resource: 3` (was 10)
- `blob_max: 40` (was 30)

Energy economics with harsh settings:
- Away from blob: **-0.62 energy/step** (death)
- At blob center: **+8.42 energy/step** (thrive)

### Files Created
```
configs/world/orbiting_gaussian_harsh.yaml
configs/run/from_pretrain_orbiting_harsh.yaml
scripts/run_from_pretrain_orbiting_harsh.sh
scratch/trajectory_analysis/  (analysis scripts)
```

### Results at 50k Steps
| Metric | Harsh | Original |
|--------|-------|----------|
| Lifespan | 33 steps | 45 steps |
| Population | ~1500 | ~2500 |
| Median age | 21 | 35 |

Lifespan SHORTER in harsh environment, not longer. Selection pressure present but agents can't adapt.

## Critical Finding: Agents Don't Turn

### Action Distribution
| Stage | Left | Right | Total Turns |
|-------|------|-------|-------------|
| Pretrain 10k | 0.6% | 0.6% | **1.2%** |
| Orbiting 35k | 0.0% | 0.1% | 0.1% |
| Harsh 50k | 0.0% | 0.0% | **0.0%** |

### Implications
1. Pretrain agents COULD turn (1.2%)
2. Orbiting environments selected out turning
3. Without turning, temporal gradient sensing is impossible
4. LSTM memory is useless if agents never explore different directions

### Why Turns Were Selected Out
- Turning costs 2.5 energy
- Turning might move away from current food
- Evolution eliminated "wasteful" exploration
- But without exploration, can't learn gradient following

### The Chicken-and-Egg Problem
For LSTM-based chemotaxis:
1. Need to turn to sample different directions
2. Need to compare food before/after turn
3. Need to learn which turns were good

If step 1 is selected out, steps 2-3 never happen. Evolution painted itself into a corner.

## Conclusions

1. **Trajectory tracking requires more frequent checkpoints** or explicit logging (lifespan << checkpoint interval)
2. **r-selection dominates** when resources are abundant - no pressure for longevity
3. **Harsh environments alone don't evolve tracking** - agents just die faster
4. **Turning capability atrophied** under selection pressure - the key behavior for chemotaxis was eliminated
5. **LSTM memory is necessary but not sufficient** - agents also need exploratory behavior to use it

## Possible Future Directions
- Force minimum exploration (higher action_temperature, epsilon-greedy)
- Start from agents that still turn
- Reward turning directly (curiosity bonus)
- Different environment that rewards turning
