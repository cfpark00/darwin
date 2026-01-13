# Genotype Cluster Behavioral Analysis

## Summary
Performed PCA/HDBSCAN clustering on evolved agent genotypes, identified distinct clusters, and discovered significant behavioral differences between clusters. Created infrastructure for cluster-based transfer experiments.

## Genotype Analysis

### Approach
1. Loaded final checkpoint from `run_from_pretrain_simple` (step 50001, 8154 alive agents)
2. Extracted weight matrices (1238 parameters per agent)
3. Applied PCA (PC1 explains 53.6% variance) and HDBSCAN clustering
4. Identified 36 clusters with clear spatial separation in PCA space

### Key Clusters
| Cluster | N agents | PC1 | Description |
|---------|----------|-----|-------------|
| 5 | 1324 | +38.2 | Largest, right side of PC1 |
| 17 | 1040 | -34.7 | 2nd largest, left side of PC1 |

### Files Created
- `scratch/genotype_analysis_v1/load_genotypes.py` - Main analysis script
- `scratch/genotype_analysis_v1/plot_centers.py` - Cluster center visualization
- `scratch/genotype_analysis_v1/cluster_table.csv` - Cluster centroids
- `scratch/genotype_analysis_v1/cluster_assignments.npz` - Cluster labels

## Reference Data
Created static reference data for reproducibility across investigations:
- `data/reference/genotype_clusters_v1/cluster_assignments.npz`
- `data/reference/genotype_clusters_v1/cluster_table.csv`
- `data/reference/genotype_clusters_v1/README.md`

## Research Question Documentation
Created `docs/concrete_questions/genotype_cluster_behavioral_divergence.md` documenting:
- Research question and background
- Target clusters for comparison
- Proposed experiments (diffusion, chemotaxis, reproduction, actions)
- Success criteria

## Experiment Group: divergent_genotype_v1

### Infrastructure
```
configs/run/divergent_genotype_v1/
configs/world/divergent_genotype_v1/
scripts/divergent_genotype_v1/
data/divergent_genotype_v1/
```

### New Script
- `src/scripts/transfer_by_cluster.py` - Transfer script that filters agents by cluster ID

### Experiments Run
1. **Diffusion (cluster 5 vs 17)**: Uniform resource, energy clamped, reproduction disabled
2. **Chemotaxis (cluster 5 vs 17)**: Linear resource gradient, energy clamped

## Key Finding: Behavioral Divergence

### Diffusion Coefficients
| Cluster | D | Final MSD |
|---------|---|-----------|
| 5 | 0.23 | 731 |
| 17 | 0.81 | 2556 |

**Cluster 17 moves 3.5x more than Cluster 5!**

### Mechanism: Action Distribution
| Action | Cluster 5 | Cluster 17 |
|--------|-----------|------------|
| forward | 14.2% | 21.8% |
| left | 6.2% | 0.0% |

- **Cluster 5 "Circler"**: High left-turn rate → spins in place → low net displacement
- **Cluster 17 "Explorer"**: No turns, mostly forward → efficient straight-line movement

### Chemotaxis Results
- Neither cluster showed gradient-following behavior
- Cluster 17 spread wider (consistent with higher diffusion)
- Cluster 5 stayed more concentrated (consistent with circling)

## Documentation Updates
- Added "Experiment Groups" concept to `docs/research_context.md`
- Created `data/divergent_genotype_v1/FINDINGS.md` with full results

## Output Files
- `data/divergent_genotype_v1/diffusion_comparison.png`
- `data/divergent_genotype_v1/action_comparison.png`
- `data/divergent_genotype_v1/chemotaxis_comparison.png`
- `data/divergent_genotype_v1/summary.png`

## Conclusion
Successfully demonstrated that genetic divergence leads to qualitative behavioral divergence. The two main lineages evolved distinct movement strategies: "circling" vs "linear exploration".
