# Divergent Genotype V3 & V4: Action Profiles and Phylogenetic Analysis

## Summary
Created two new experiment groups investigating genotype cluster behavioral divergence:
- **v3**: Action profiles under different resource conditions - discovered turning chirality difference
- **v4**: Phylogenetic tree reconstruction from genotypes - confirmed deep evolutionary split, spatial segregation

## Divergent Genotype V3: Action Profiles

### Approach
Measure action distributions for clusters 5 and 17 under high (r=30) and low (r=5) resource conditions.
- Uniform resource, energy clamped (immortal), reproduction disabled
- 2000 steps, 256 agents per experiment

### Key Finding: Turning Chirality
| Cluster | Left turns | Right turns |
|---------|------------|-------------|
| C5      | 2.4-4.3%   | **0%**      |
| C17     | **0%**     | 0.9-1.0%    |

**Cluster 5 turns LEFT exclusively, Cluster 17 turns RIGHT exclusively!**

### Other Findings
- C17 moves more than C5 at both resource levels (~3-5% higher)
- Both clusters adapt to scarcity: movement doubles when resources drop (r=30 → r=5)

### Files Created
- `configs/run/divergent_genotype_v3/` - 4 run configs
- `configs/world/divergent_genotype_v3/` - 2 world configs
- `scripts/divergent_genotype_v3/` - 5 bash scripts
- `data/divergent_genotype_v3/figures/action_profile_comparison.png`
- `scratch/v3_analysis/` - Analysis scripts

## Divergent Genotype V4: Phylogenetic Analysis

### Approach
Hierarchical clustering (Ward's method) on HDBSCAN cluster centroids to reconstruct evolutionary tree. No behavioral experiments - pure genotype analysis.

### Key Finding: Deep Evolutionary Split
C5 and C17 are in **different clades at ALL granularities** tested:

| Superclusters | C5 | C17 | Same? |
|---------------|-----|-----|-------|
| 2 | Clade A | Clade B | NO |
| 4 | 4 | 2 | NO |
| 8 | 8 | 4 | NO |

At coarsest level:
- **Clade A**: 13 clusters, 3485 agents (contains C5)
- **Clade B**: 23 clusters, 3764 agents (contains C17)

### Spatial Segregation
The two clades occupy different regions of the arena - evidence of **allopatric speciation**.

### Diversification Hotspot
Identified concentration of split points near (300, 150). Analysis showed:
- 2.2x higher population density
- 1.5x higher resource variance (heterogeneity)
This matches the concept of "diversification hotspot" in evolutionary biology.

### Files Created
- `data/divergent_genotype_v4/figures/phylogenetic_clusters.png` - Dendrogram with split ordering
- `data/divergent_genotype_v4/figures/spatial_2_clades.png` through `spatial_8_clades.png` - Progressive clade splits
- `data/divergent_genotype_v4/figures/spatial_split_points.png` - All 35 split locations
- `scratch/v4_analysis/` - Analysis scripts

## Documentation Updates
- Updated `docs/concrete_questions/genotype_cluster_behavioral_divergence.md`:
  - Added v3 and v4 to existing groups list
  - Changed template to suggest v5+ for new teams

## Conceptual Discussion
Covered evolutionary biology concepts:
- Phylogenetic tree validity (horizontal gene transfer, hybridization)
- Ward's distance (variance-based merge criterion)
- Diversification hotspots, adaptive radiation
- Coalescent theory and coalescent times
- Effective population size (Ne)

## Conclusion
Combined v3 (behavioral) and v4 (genetic) analyses show that:
1. Genetic divergence corresponds to behavioral divergence (turning chirality)
2. The phylogenetic tree structure maps onto spatial territory
3. The simulation exhibits allopatric speciation dynamics
