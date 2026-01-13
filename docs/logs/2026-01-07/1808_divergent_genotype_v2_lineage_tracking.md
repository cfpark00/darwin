# Divergent Genotype V2 & Lineage Tracking

## Summary
Created divergent_genotype_v2 experiment group (parallel to v1) exploring competition and reproduction dynamics. Found that Cluster 17 dominates Cluster 5 in head-to-head competition despite similar isolated reproduction. Added ground truth lineage tracking to both simulation files.

## Experiment Group: divergent_genotype_v2

### Research Question
Same as v1: Can we detect qualitative behavioral differences between genetically distinct clusters?

### Approach (Different from v1)
- **V1**: Diffusion and chemotaxis (movement patterns)
- **V2**: Reproduction rate and competition (fitness dynamics)

### Infrastructure
```
configs/run/divergent_genotype_v2/
configs/world/divergent_genotype_v2/
scripts/divergent_genotype_v2/
data/divergent_genotype_v2/
```

### New Script
- `src/scripts/transfer_competition.py` - Loads agents from TWO clusters, tracks lineage by genetic distance

### Experiments Run

#### 1. Isolated Reproduction (C5 vs C17 separately)
| Cluster | Initial | Final (5000 steps) | Growth |
|---------|---------|-------------------|--------|
| 5 | 256 | 5,864 | 22.9x |
| 17 | 256 | 5,343 | 20.9x |

**Finding**: Similar reproduction rates in isolation (C5 slightly better)

#### 2. Head-to-Head Competition
| Metric | Cluster 5 | Cluster 17 |
|--------|-----------|------------|
| Initial | 256 (50%) | 256 (50%) |
| Step 100 | 435 (28%) | 1,134 (72%) |
| Step 500 | 348 (10%) | 3,183 (90%) |
| Final | **343 (8.4%)** | **3,726 (91.6%)** |

**Finding**: Cluster 17 dominates in direct competition!

### Key Insight
The behavioral difference invisible in isolated tests becomes dramatic in competition:
- **Cluster 5**: Efficient in uncontested environments
- **Cluster 17**: Aggressive competitive displacement

### Spatial Analysis
Visualized original positions of clusters in source checkpoint:
- Cluster 5 centroid: y=96.8, x=328.1 (bottom-right)
- Cluster 17 centroid: y=428.2, x=140.8 (top-left)
- Distance: 381 pixels apart (in 512×512 arena)

**Conclusion**: Allopatric speciation - clusters evolved in different regions of the arena.

### Output Files
- `data/divergent_genotype_v2/RESULTS_SUMMARY.md`
- `data/divergent_genotype_v2/comparison_isolated_vs_competition.png`
- `data/divergent_genotype_v2/comparison_growth_factor.png`
- `data/divergent_genotype_v2/cluster_spatial_distribution.png`
- `data/divergent_genotype_v2/cluster_density_heatmap.png`
- `data/divergent_genotype_v2/competition/figures/lineage_dynamics.png`

## Lineage Tracking Implementation

### Motivation
Need ground truth lineage tracking (not genetic distance inference) for accurate genealogy.

### Design
- **Option 1**: Lineage ID (just tracks founding cluster) - simpler
- **Option 2**: UID + Parent UID (full genealogy tree) - chosen

### Implementation
Added to both `simulation.py` and `simulation_simple.py`:

| Field | Type | Description |
|-------|------|-------------|
| `uid` | int32 (max_agents,) | Unique ID per agent |
| `parent_uid` | int32 (max_agents,) | Parent's UID (-1 for founders) |
| `next_uid` | int32 scalar | Next available UID |

### Key Technical Details
- **Parallel-safe**: Uses `jnp.cumsum` on birth mask for sequential UID assignment
- **Buffer growth**: Extends uid/parent_uid arrays when buffers grow
- **dtype**: int32 (max ~2B UIDs, avoids JAX GPU warnings)

### Files Modified
- `src/simulation.py` - Full simulation (with toxin/attack)
- `src/simulation_simple.py` - Simple simulation (thermotaxis/pretrain)

### Verification
Tested with checkpoint at step 2500:
```
next_uid: 150,604
Total births: 150,092 (from 512 founders)
Founders alive: 0
Non-founders alive: 2,892
```

## Conclusions

1. **V2 found the real behavioral difference**: Not just movement patterns (v1), but competitive fitness
2. **Geographic speciation confirmed**: Clusters evolved in separate regions of arena
3. **Lineage tracking enables**: True genealogy reconstruction, accurate competition tracking
